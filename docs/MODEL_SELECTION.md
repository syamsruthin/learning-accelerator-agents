# Model Selection Guide

> **Note:** This project now uses OpenRouter instead of local Ollama for all LLM calls.
> Per-agent model defaults live in `src/llm.py` (`ROLE_MODELS`). Ollama-specific guidance below is kept for reference.

All agents in this system use Ollama for local inference.
This guide helps you choose the right model for your hardware.

---

## Why model choice matters for agents

Agents call tools by generating structured JSON. A model that
hallucinates tool names or produces malformed JSON will fail
silently, the tool call won't execute, the agent will loop,
and you'll hit the `max_iterations` limit without understanding why.

**Minimum viable tier for reliable tool calling: 7B parameters.**
Sub-7B models work for simple chat but produce too many JSON errors
for production agentic use.

---

## Recommendations by VRAM

| VRAM | Model | Pull command | Best for |
|---|---|---|---|
| 8 GB | `qwen2.5:7b` | `ollama pull qwen2.5:7b` | General purpose, good tool calling |
| 8 GB | `qwen3:8b` | `ollama pull qwen3:8b` | Better reasoning, same VRAM |
| 24 GB | `qwen2.5-coder:32b` | `ollama pull qwen2.5-coder:32b` | Best tool calling at this tier |
| 24 GB | `qwen3:32b` | `ollama pull qwen3:32b` | Best overall at this tier |
| CPU only | `qwen2.5:7b` (Q4_K_M) | `ollama pull qwen2.5:7b` | Works, 5-10× slower |

**Check your VRAM on Mac:** Apple menu → About This Mac → chip info.
Apple Silicon unified memory is shared between CPU and GPU.
16 GB unified ≈ 8 GB available for the model.

---

## Agent-specific settings

These settings are used throughout the codebase:

```python
# Structured output (Curriculum Planner, Quiz Generator grader)
ChatOllama(temperature=0.1, format="json")

# Tool-calling (Explainer)
ChatOllama(temperature=0.3)

# Creative/coaching (Progress Coach, Quiz Generator questions)
ChatOllama(temperature=0.4, format="json")
```

Low temperature = more consistent JSON. Higher temperature = more
varied explanations and coaching messages. Never use temperature > 0.5
for any agent that produces structured output.

---

## Switching models

Change `OLLAMA_MODEL` in `.env`. No code changes needed.

```bash
# .env
OLLAMA_MODEL=qwen2.5-coder:32b
```

Then pull the model if you haven't:
```bash
ollama pull qwen2.5-coder:32b
```

---

## Eval test scores by model

Thresholds in `tests/test_eval.py` are calibrated for 7B models (0.6).
Larger models typically score higher:

| Model | Faithfulness | Relevancy | Question Quality |
|---|---|---|---|
| `qwen2.5:7b` | 0.70-0.82 | 0.75-0.90 | 0.65-0.75 |
| `qwen2.5-coder:32b` | 0.82-0.92 | 0.85-0.95 | 0.78-0.88 |

If eval tests consistently fail with a 7B model, try the 32B tier
before lowering thresholds.
