"""
src/token_usage.py

Token usage tracking for every LLM call in the system.

Each call is recorded three ways:
  1. A log line on the "llm_usage" logger, e.g.
       [LLM usage] role=explainer model=anthropic/claude-sonnet-5.5
                   input=1834 output=212 total=2046 cost=$0.0058 latency=2.41s
  2. A JSON line appended to data/llm_usage.jsonl (one record per call),
     for analysis across sessions and processes
  3. In-process running totals per role, printed by main.py at session end

Where calls are captured:
  - LangGraph agents + DeepEval judge: a LangChain callback attached in
    llm.get_chat_model() (see token_usage_callback.py)
  - CrewAI Study Buddy: a CrewAI event-bus listener in study_buddy.py

This module has no LangChain or CrewAI imports, so both sides can use it.

Configuration (.env):
  LLM_USAGE_LOG_FILE   JSONL path, default data/llm_usage.jsonl ("" disables)
  LLM_USAGE_LOG_LEVEL  Console log level, default INFO (WARNING silences)
"""

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Logger setup
#
# A dedicated logger with its own handler and propagate=False, so usage lines
# show up even though the app mostly uses print(), and aren't duplicated by
# handlers other libraries install on the root logger.
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("llm_usage")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[LLM usage] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(os.getenv("LLM_USAGE_LOG_LEVEL", "INFO").upper())
    logger.propagate = False


@dataclass
class RoleTotals:
    """Running totals for one agent role within this process."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


_totals: dict[str, RoleTotals] = {}
_lock = threading.Lock()   # A2A services run calls from worker threads


def _usage_file() -> Path | None:
    path = os.getenv("LLM_USAGE_LOG_FILE", "data/llm_usage.jsonl")
    return Path(path) if path else None


def record_usage(
    role: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    latency_s: float | None = None,
    cost_usd: float | None = None,
    session_id: str | None = None,
    node: str | None = None,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """
    Record one LLM call. Never raises: tracking must not break an agent.

    Args:
        role:          Agent role from llm.ROLE_MODELS, e.g. "explainer".
        model:         Model id that served the call.
        input_tokens:  Prompt tokens (None if the provider didn't report it).
        output_tokens: Completion tokens (None if not reported).
        latency_s:     Wall-clock duration of the call.
        cost_usd:      Cost reported by OpenRouter, when available.
        session_id:    LangGraph thread_id, when the call ran inside the graph.
        node:          LangGraph node name, when available.
        status:        "ok" or "error".
        error:         Error message for failed calls.
    """
    try:
        in_tok = int(input_tokens or 0)
        out_tok = int(output_tokens or 0)

        with _lock:
            totals = _totals.setdefault(role, RoleTotals())
            totals.calls += 1
            totals.input_tokens += in_tok
            totals.output_tokens += out_tok
            totals.cost_usd += float(cost_usd or 0.0)

        parts = [
            f"role={role}",
            f"model={model}",
            f"input={in_tok}",
            f"output={out_tok}",
            f"total={in_tok + out_tok}",
        ]
        if cost_usd is not None:
            parts.append(f"cost=${cost_usd:.4f}")
        if latency_s is not None:
            parts.append(f"latency={latency_s:.2f}s")
        if session_id:
            parts.append(f"session={session_id}")
        if status != "ok":
            parts.append(f"status={status} error={error}")
        logger.log(logging.WARNING if status != "ok" else logging.INFO, " ".join(parts))

        path = _usage_file()
        if path:
            record = {
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "role":          role,
                "model":         model,
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "total_tokens":  in_tok + out_tok,
                "cost_usd":      cost_usd,
                "latency_s":     round(latency_s, 3) if latency_s is not None else None,
                "session_id":    session_id,
                "node":          node,
                "status":        status,
                "error":         error,
                "pid":           os.getpid(),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with _lock, path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:   # tracking must never break the agent
        logger.warning(f"failed to record usage for role={role}: {e}")


def get_usage_totals() -> dict[str, RoleTotals]:
    """Snapshot of per-role totals for this process."""
    with _lock:
        return {role: RoleTotals(**vars(t)) for role, t in _totals.items()}


def reset_usage_totals() -> None:
    """Clear in-process totals (the JSONL file is left untouched)."""
    with _lock:
        _totals.clear()


def format_usage_summary() -> str:
    """Per-role table of calls and tokens for this process, '' if no calls."""
    totals = get_usage_totals()
    if not totals:
        return ""

    lines = [f"{'Role':<12} {'Calls':>5} {'Input':>9} {'Output':>9} {'Total':>9} {'Cost':>9}"]
    grand = RoleTotals()
    for role, t in totals.items():
        lines.append(
            f"{role:<12} {t.calls:>5} {t.input_tokens:>9,} {t.output_tokens:>9,} "
            f"{t.input_tokens + t.output_tokens:>9,} {'$' + format(t.cost_usd, '.4f'):>9}"
        )
        grand.calls += t.calls
        grand.input_tokens += t.input_tokens
        grand.output_tokens += t.output_tokens
        grand.cost_usd += t.cost_usd
    lines.append(
        f"{'TOTAL':<12} {grand.calls:>5} {grand.input_tokens:>9,} {grand.output_tokens:>9,} "
        f"{grand.input_tokens + grand.output_tokens:>9,} {'$' + format(grand.cost_usd, '.4f'):>9}"
    )
    return "\n".join(lines)
