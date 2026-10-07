"""
src/token_usage_callback.py

LangChain callback that records token usage for every chat model call.

Attached to each model in llm.get_chat_model(), so every agent call
(planner, explainer tool-loop iterations, quiz, grader, coach, eval judge)
is logged without touching agent code.

Kept separate from token_usage.py so the CrewAI side can record usage
without importing LangChain.
"""

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from token_usage import record_usage


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Records input/output tokens, cost and latency per LLM call."""

    def __init__(self, role: str, model: str):
        self.role = role
        self.model = model
        # run_id -> (start time, LangGraph metadata) for in-flight calls
        self._runs: dict[UUID, tuple[float, dict]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Inside the graph, LangGraph's config propagates to nested calls, so
        # metadata carries thread_id (= session_id) and the node name.
        self._runs[run_id] = (time.perf_counter(), metadata or {})

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start, metadata = self._runs.pop(run_id, (None, {}))
        latency = time.perf_counter() - start if start is not None else None

        input_tokens = output_tokens = None
        cost = None
        model = self.model

        # Chat models put usage on the message: usage_metadata is LangChain's
        # standard shape; response_metadata["token_usage"] is OpenRouter's raw
        # usage block, which also carries the billed cost.
        generations = response.generations[0] if response.generations else []
        message = getattr(generations[0], "message", None) if generations else None
        if message is not None:
            usage = getattr(message, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            response_metadata = getattr(message, "response_metadata", {}) or {}
            cost = (response_metadata.get("token_usage") or {}).get("cost")
            model = response_metadata.get("model_name") or model

        # Fallback for providers that only report usage in llm_output
        if input_tokens is None and response.llm_output:
            token_usage = response.llm_output.get("token_usage") or {}
            input_tokens = token_usage.get("prompt_tokens")
            output_tokens = token_usage.get("completion_tokens")
            cost = cost if cost is not None else token_usage.get("cost")

        record_usage(
            self.role,
            model,
            input_tokens,
            output_tokens,
            latency_s=latency,
            cost_usd=cost,
            session_id=metadata.get("thread_id"),
            node=metadata.get("langgraph_node"),
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        start, metadata = self._runs.pop(run_id, (None, {}))
        latency = time.perf_counter() - start if start is not None else None
        record_usage(
            self.role,
            self.model,
            0,
            0,
            latency_s=latency,
            session_id=metadata.get("thread_id"),
            node=metadata.get("langgraph_node"),
            status="error",
            error=f"{type(error).__name__}: {error}"[:300],
        )
