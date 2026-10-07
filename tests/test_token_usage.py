"""
tests/test_token_usage.py

Tests for per-call LLM token usage logging.

No API calls: the LangChain callback is fed hand-built LLMResult objects
shaped like real OpenRouter responses.
"""

import json
import logging
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import token_usage
from token_usage import (
    format_usage_summary,
    get_usage_totals,
    record_usage,
    reset_usage_totals,
)
from token_usage_callback import TokenUsageCallbackHandler


@pytest.fixture(autouse=True)
def isolated_usage(tmp_path, monkeypatch):
    """Fresh totals and a temp JSONL file for every test."""
    usage_file = tmp_path / "llm_usage.jsonl"
    monkeypatch.setenv("LLM_USAGE_LOG_FILE", str(usage_file))
    reset_usage_totals()
    yield usage_file
    reset_usage_totals()


def _read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _openrouter_result(input_tokens, output_tokens, cost=None, model="anthropic/claude-haiku-5.5"):
    """An LLMResult shaped like a ChatOpenAI response from OpenRouter."""
    token_usage_block = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cost is not None:
        token_usage_block["cost"] = cost
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"token_usage": token_usage_block, "model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestRecordUsage:

    def test_accumulates_totals_per_role(self):
        record_usage("quiz", "m", 100, 20, cost_usd=0.001)
        record_usage("quiz", "m", 50, 10, cost_usd=0.002)
        record_usage("coach", "m", 7, 3)

        totals = get_usage_totals()
        assert totals["quiz"].calls == 2
        assert totals["quiz"].input_tokens == 150
        assert totals["quiz"].output_tokens == 30
        assert totals["quiz"].cost_usd == pytest.approx(0.003)
        assert totals["coach"].calls == 1

    def test_writes_one_jsonl_record_per_call(self, isolated_usage):
        record_usage("planner", "anthropic/claude-sonnet-5.5", 120, 400,
                     latency_s=1.234, cost_usd=0.0042, session_id="abc", node="curriculum_planner")
        record_usage("grader", "openai/gpt-6-luna", 80, 30)

        records = _read_records(isolated_usage)
        assert len(records) == 2
        first = records[0]
        assert first["role"] == "planner"
        assert first["model"] == "anthropic/claude-sonnet-5.5"
        assert first["input_tokens"] == 120
        assert first["output_tokens"] == 400
        assert first["total_tokens"] == 520
        assert first["cost_usd"] == 0.0042
        assert first["latency_s"] == 1.234
        assert first["session_id"] == "abc"
        assert first["node"] == "curriculum_planner"
        assert first["status"] == "ok"

    def test_missing_token_counts_are_recorded_as_zero(self, isolated_usage):
        record_usage("coach", "m", None, None)
        assert _read_records(isolated_usage)[0]["total_tokens"] == 0

    def test_empty_file_setting_disables_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USAGE_LOG_FILE", "")
        monkeypatch.chdir(tmp_path)
        record_usage("coach", "m", 1, 1)
        assert list(tmp_path.iterdir()) == []
        assert get_usage_totals()["coach"].calls == 1

    def test_logs_one_line_with_token_counts(self, caplog):
        caplog.set_level(logging.INFO, logger="llm_usage")
        token_usage.logger.addHandler(caplog.handler)
        try:
            record_usage("explainer", "anthropic/claude-sonnet-5.5", 1834, 212, cost_usd=0.0058)
        finally:
            token_usage.logger.removeHandler(caplog.handler)
        line = caplog.records[-1].getMessage()
        assert "role=explainer" in line
        assert "input=1834" in line
        assert "output=212" in line
        assert "total=2046" in line
        assert "cost=$0.0058" in line

    def test_never_raises_on_unwritable_file(self, tmp_path, monkeypatch):
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")
        monkeypatch.setenv("LLM_USAGE_LOG_FILE", str(blocker / "usage.jsonl"))
        record_usage("coach", "m", 1, 1)   # must not raise

    def test_summary_has_row_per_role_and_total(self):
        record_usage("quiz", "m", 100, 20)
        record_usage("grader", "m", 50, 10)
        summary = format_usage_summary()
        assert "quiz" in summary
        assert "grader" in summary
        total_line = summary.splitlines()[-1]
        assert total_line.startswith("TOTAL")
        assert "150" in total_line and "30" in total_line and "180" in total_line

    def test_summary_empty_when_no_calls(self):
        assert format_usage_summary() == ""


class TestTokenUsageCallbackHandler:

    def test_records_tokens_cost_and_session(self, isolated_usage):
        handler = TokenUsageCallbackHandler("quiz", "anthropic/claude-haiku-5.5")
        run_id = uuid4()
        handler.on_chat_model_start(
            {}, [[]], run_id=run_id,
            metadata={"thread_id": "sess-1", "langgraph_node": "quiz_generator"},
        )
        handler.on_llm_end(_openrouter_result(321, 45, cost=0.0001), run_id=run_id)

        record = _read_records(isolated_usage)[0]
        assert record["role"] == "quiz"
        assert record["input_tokens"] == 321
        assert record["output_tokens"] == 45
        assert record["cost_usd"] == 0.0001
        assert record["session_id"] == "sess-1"
        assert record["node"] == "quiz_generator"
        assert record["latency_s"] is not None

    def test_falls_back_to_llm_output_usage(self, isolated_usage):
        handler = TokenUsageCallbackHandler("coach", "m")
        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="x"))]],
            llm_output={"token_usage": {"prompt_tokens": 11, "completion_tokens": 4}},
        )
        handler.on_llm_end(result, run_id=uuid4())
        record = _read_records(isolated_usage)[0]
        assert record["input_tokens"] == 11
        assert record["output_tokens"] == 4

    def test_records_failed_calls(self, isolated_usage):
        handler = TokenUsageCallbackHandler("grader", "openai/gpt-6-luna")
        run_id = uuid4()
        handler.on_chat_model_start({}, [[]], run_id=run_id)
        handler.on_llm_error(TimeoutError("timed out"), run_id=run_id)
        record = _read_records(isolated_usage)[0]
        assert record["status"] == "error"
        assert "TimeoutError" in record["error"]
        assert record["total_tokens"] == 0

    def test_get_chat_model_attaches_handler(self):
        from llm import get_chat_model
        model = get_chat_model("explainer")
        handlers = [h for h in model.callbacks if isinstance(h, TokenUsageCallbackHandler)]
        assert len(handlers) == 1
        assert handlers[0].role == "explainer"
