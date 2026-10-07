# Production-Grade Multi-Agent AI System

Build a complete multi-agent AI system. The agents and services run on your
machine; LLM inference goes through [OpenRouter](https://openrouter.ai), with a
model chosen per agent role (see `src/llm.py`).

> This fork replaces the original local Ollama inference with OpenRouter.

This project is the companion code for the freeCodeCamp handbook:
**How to Build a Local Multi-Agent AI System with LangGraph, MCP, and A2A**
(link to be added once published)

---

## What this builds

A **Learning Accelerator**: a four-agent system that plans a study curriculum,
explains topics from your own notes, quizzes you, and adapts based on results.
The use case is the teaching vehicle. The architecture is the real subject.

```
Goal: "Learn Python closures and decorators"
  │
  ▼
Curriculum Planner  →  structured study roadmap
  │
  ▼ (you approve the plan)
Explainer           →  reads your notes via MCP, explains each topic
  │
  ▼
Quiz Generator      →  tests understanding, grades answers with LLM-as-judge
  │
  ▼
Progress Coach      →  adapts roadmap, calls CrewAI Study Buddy via A2A
  │
  └── loops back to Explainer for next topic
```

The same architecture pattern runs in production for sales enablement,
compliance training, customer support onboarding, and engineering ramp-up.

---

## Architecture

| Layer | Technology | What it does |
|---|---|---|
| Orchestration | LangGraph 1.1.0 | Stateful agent graph with checkpointing |
| Tool integration | MCP (mcp 1.26.0) | Standardised agent-to-tool protocol |
| Agent coordination | A2A (a2a-sdk 0.3.25) | Cross-framework agent-to-agent protocol |
| Inference | OpenRouter (via `langchain-openai`) | One API key, a model per agent role |
| Cross-framework | CrewAI 1.13.0 | Study Buddy agent (called via A2A) |
| Observability | Langfuse 4.0.1 | Full trace of every agent and LLM call |
| Evaluation | DeepEval 3.9.1 | LLM-as-judge quality metrics |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a full deep-dive.

---

## Requirements

- Python 3.11+
- An [OpenRouter API key](https://openrouter.ai/keys)
- Docker Desktop (for Langfuse observability, optional)

---

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/sandeepmb/freecodecamp-multi-agent-ai-system
cd freecodecamp-multi-agent-ai-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY (per-agent models are optional overrides)

# 3. Run
python main.py
```

For the full system with A2A services and observability, see [Running all services](#running-all-services).

---

## Project structure

```
freecodecamp-multi-agent-ai-system/
├── src/
│   ├── agents/                 # LangGraph agent nodes
│   │   ├── curriculum_planner.py
│   │   ├── explainer.py
│   │   ├── quiz_generator.py
│   │   ├── progress_coach.py
│   │   └── human_approval.py
│   ├── graph/
│   │   ├── state.py            # Shared AgentState TypedDict
│   │   └── workflow.py         # LangGraph graph definition
│   ├── mcp_servers/            # MCP tool servers
│   │   ├── filesystem_server.py
│   │   └── memory_server.py
│   ├── a2a_services/           # A2A protocol services and client
│   │   ├── quiz_service.py     # Quiz Generator as A2A service
│   │   └── a2a_client.py       # Client for calling A2A services
│   ├── crewai_agent/
│   │   └── study_buddy.py      # CrewAI agent served via A2A
│   └── observability/
│       └── langfuse_setup.py   # Langfuse callback handler
├── tests/
│   ├── conftest.py             # Shared fixtures and markers
│   ├── test_state.py           # 24 tests
│   ├── test_curriculum_planner.py  # 11 tests
│   ├── test_mcp_servers.py     # 36 tests
│   ├── test_explainer.py       # 14 tests
│   ├── test_quiz_and_coach.py  # 17 tests
│   ├── test_checkpointing.py   # 20 tests
│   ├── test_observability.py   # 16 tests
│   ├── test_a2a.py             # 19 tests
│   ├── test_crewai_interop.py  # 25 tests
│   └── test_eval.py            # 12 eval tests (real OpenRouter calls)
├── study_materials/
│   └── sample_notes/           # Markdown files the agents read
├── docs/
│   ├── ARCHITECTURE.md
│   └── MODEL_SELECTION.md
├── data/                       # SQLite checkpoint DB (created at runtime)
├── main.py                     # Entry point
├── docker-compose.yml          # Langfuse self-hosted stack
├── Makefile                    # One-command startup
├── requirements.txt
└── .env.example
```

---

## Running all services

The full system has three processes. Open three terminal tabs.

**Tab 1: Quiz Generator A2A service:**
```bash
source .venv/bin/activate
python src/a2a_services/quiz_service.py
# Serves at http://localhost:9001
```

**Tab 2: CrewAI Study Buddy A2A service:**
```bash
source .venv/bin/activate
python src/crewai_agent/study_buddy.py
# Serves at http://localhost:9002
```

**Tab 3: Main LangGraph application:**
```bash
source .venv/bin/activate
python main.py
```

Or use the Makefile:
```bash
make services   # starts both A2A services in background
make run        # runs the main application
```

---

## Session resume

Every session is checkpointed to `data/checkpoints.db` after each agent node.
To resume a stopped session:

```bash
python main.py --resume <session-id>
```

The session ID is printed at the start of every run.

---

## Observability

Start Langfuse locally:
```bash
docker compose up -d
# Open http://localhost:3000
```

Add your API keys to `.env` (from the Langfuse project settings), then run
as normal. Every agent call, LLM completion, and tool call appears in the
trace UI automatically.

---

## Testing

```bash
# Fast unit tests, run during development (~3 seconds)
# 182 tests across 9 test files
pytest tests/ -m "not eval" -v

# Quality evaluation tests, run before releases (~90 seconds, real OpenRouter calls)
# 12 LLM-as-judge tests
pytest tests/test_eval.py -v -s -m eval
```

---

## Adding your own study materials

Replace or add Markdown files in `study_materials/sample_notes/`.
The Explainer agent reads every `.md` file in that directory automatically
via the MCP filesystem server. No configuration changes needed.

---

## Configuration reference

See `.env.example` for all available settings.

Key toggles:
| Variable | Default | Effect |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | Key for all LLM calls |
| `OPENROUTER_MODEL_<ROLE>` | see `src/llm.py` | Per-agent model override (PLANNER, EXPLAINER, QUIZ, GRADER, COACH, STUDY_BUDDY, EVAL_JUDGE) |
| `USE_A2A_QUIZ` | `true` | Route quiz tasks to A2A service |
| `USE_STUDY_BUDDY` | `true` | Call CrewAI Study Buddy for low scores |
| `CHECKPOINT_DB` | `data/checkpoints.db` | SQLite path for checkpoints |
| `LLM_USAGE_LOG_FILE` | `data/llm_usage.jsonl` | Per-call token usage log (JSON lines); empty disables |
| `LLM_USAGE_LOG_LEVEL` | `INFO` | `INFO` logs every LLM call's tokens; `WARNING` only failures |

---

## Article

This code is the companion to the freeCodeCamp handbook:
**How to Build Production-Grade Multi-Agent AI Systems with LangGraph, MCP, A2A, and Ollama**

The handbook explains every architectural decision, walks through each
agent step by step, and covers the production patterns that make this
system different from a basic tutorial.

---

## Author

**Sandeep Bharadwaj Mannapur**
Lead Data and AI/ML Engineer with 15+ years in financial services and enterprise SaaS.

[GitHub](https://github.com/sandeepmb) · [freeCodeCamp](https://www.freecodecamp.org/news/author/sandeep-mannapur/)
