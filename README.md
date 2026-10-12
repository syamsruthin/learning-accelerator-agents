# Learning Accelerator Agents

A multi-agent AI study coach built with **LangGraph**, **MCP**, **A2A**, and
**CrewAI**. Give it a learning goal and it plans a curriculum, explains each
topic from your own notes, quizzes you, grades your answers, and adapts the
plan based on how you did.

All LLM calls go through [OpenRouter](https://openrouter.ai), with a model
chosen per agent role. Sessions are checkpointed so you can stop and resume,
every call can be traced in Langfuse, and every LLM call's token usage is
logged.

> Forked from [sandeepmb/freecodecamp-multi-agent-ai-system](https://github.com/sandeepmb/freecodecamp-multi-agent-ai-system),
> the companion code for the freeCodeCamp handbook on multi-agent systems.
> This fork replaces local Ollama inference with OpenRouter, adds
> per-role model selection and token usage tracking, and includes a
> production-readiness plan for on-prem deployment.

---

## How it works

```
Goal: "Learn Python closures and decorators"
  │
  ▼
Curriculum Planner  →  builds a structured study roadmap
  │
  ▼ (human-in-the-loop: you approve or reject the plan)
Explainer           →  reads your notes via MCP tools, explains each topic
  │
  ▼
Quiz Generator      →  asks questions, grades answers with an LLM-as-judge
  │                     (runs locally or as an A2A service)
  ▼
Progress Coach      →  adapts the roadmap; on low scores, asks the CrewAI
  │                     Study Buddy (over A2A) for a fresh explanation
  └── loops back to the Explainer for the next topic
```

| Layer | Technology | Role |
|---|---|---|
| Orchestration | LangGraph 1.1 | Stateful agent graph, interrupts, SQLite checkpointing |
| Tool integration | MCP 1.26 | Filesystem (study notes) and memory servers |
| Agent-to-agent | A2A SDK 0.3 | Quiz Generator and Study Buddy as network services |
| Cross-framework | CrewAI 1.13 | Study Buddy agent, called from LangGraph via A2A |
| Inference | OpenRouter via `langchain-openai` | One API key, a model per agent role |
| Observability | Langfuse 4 | Traces of every agent, LLM, and tool call |
| Evaluation | DeepEval 3.9 | LLM-as-judge quality tests |
| UI | Terminal (`main.py`) and Streamlit | Same graph, two front ends |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

---

## Models per agent

Each agent role resolves its model in [src/llm.py](src/llm.py). Any role can
be overridden in `.env` without code changes.

| Role | Default model | Override |
|---|---|---|
| Curriculum Planner | `anthropic/claude-sonnet-5.5` | `OPENROUTER_MODEL_PLANNER` |
| Explainer | `anthropic/claude-sonnet-5.5` | `OPENROUTER_MODEL_EXPLAINER` |
| Quiz Generator | `anthropic/claude-haiku-5.5` | `OPENROUTER_MODEL_QUIZ` |
| Answer Grader | `openai/gpt-6-luna` | `OPENROUTER_MODEL_GRADER` |
| Progress Coach | `anthropic/claude-haiku-5.5` | `OPENROUTER_MODEL_COACH` |
| CrewAI Study Buddy | `google/gemini-3.8-flash` | `OPENROUTER_MODEL_STUDY_BUDDY` |
| Eval judge (tests) | `openai/gpt-6-sol` | `OPENROUTER_MODEL_EVAL_JUDGE` |

The grader, Study Buddy, and eval judge deliberately use a different model
family from the agents they check or complement. See
[docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md).

---

## Quick start

Requirements: Python 3.11+, an [OpenRouter API key](https://openrouter.ai/keys),
and optionally Docker (for Langfuse).

```bash
git clone https://github.com/syamsruthin/learning-accelerator-agents.git
cd learning-accelerator-agents
make setup                     # creates .venv, installs deps, copies .env.example to .env
```

Set `OPENROUTER_API_KEY` in `.env`, then:

```bash
make services                  # start the Quiz and Study Buddy A2A services
make run                       # terminal session
make streamlit                 # or the web UI at http://localhost:8501
```

To run without the A2A services, set `USE_A2A_QUIZ=false` and
`USE_STUDY_BUDDY=false` in `.env` and just run `python main.py`.

### Useful commands

| Command | What it does |
|---|---|
| `make run-goal GOAL='Learn async Python'` | Start with a specific goal |
| `make resume SESSION=<id>` | Resume a checkpointed session (ID is printed at start) |
| `make stop` | Stop the background A2A services |
| `make langfuse` | Start self-hosted Langfuse at http://localhost:3000 |
| `make test` | Fast unit tests (no network) |
| `make eval` | LLM-as-judge evaluation tests (real OpenRouter calls) |
| `make clean` | Remove checkpoints and caches |

---

## Using your own notes

Drop Markdown files into `study_materials/sample_notes/` (or point
`NOTES_PATH` elsewhere). The Explainer reads them through the MCP filesystem
server; no other configuration is needed.

---

## Observability and token usage

- **Langfuse:** run `make langfuse`, create a project at http://localhost:3000,
  and put its keys in `.env`. Leave the keys empty to run without tracing.
- **Token usage:** every LLM call logs its role, model, and input/output tokens
  to the console and appends a JSON line to `data/llm_usage.jsonl`. A per-session
  summary is printed at the end of each run. Configure with
  `LLM_USAGE_LOG_FILE` and `LLM_USAGE_LOG_LEVEL`.

---

## Configuration

All settings live in [.env.example](.env.example).

| Variable | Default | Effect |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | Key for all LLM calls |
| `OPENROUTER_MODEL_<ROLE>` | see table above | Per-role model override |
| `NOTES_PATH` | `study_materials/sample_notes` | Folder of Markdown notes |
| `CHECKPOINT_DB` | `data/checkpoints.db` | SQLite checkpoint path |
| `USE_A2A_QUIZ` | `true` | Use the Quiz A2A service instead of local quiz generation |
| `USE_STUDY_BUDDY` | `true` | Call the CrewAI Study Buddy on low scores |
| `QUIZ_SERVICE_URL` / `STUDY_BUDDY_URL` | `localhost:9001` / `:9002` | A2A service endpoints |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | empty / `localhost:3000` | Tracing |
| `LLM_USAGE_LOG_FILE` | `data/llm_usage.jsonl` | Token usage log; empty disables |
| `LLM_USAGE_LOG_LEVEL` | `INFO` | `INFO` logs every call, `WARNING` only failures |

---

## Testing

```bash
pytest tests/ -m "not eval" -v          # 199 unit tests, no network, a few seconds
pytest tests/test_eval.py -m eval -v -s # 12 LLM-as-judge evals, real OpenRouter calls
```

---

## Project structure

```
learning-accelerator-agents/
├── main.py                     # Terminal entry point
├── streamlit_app.py            # Streamlit web UI (same graph)
├── src/
│   ├── llm.py                  # OpenRouter chat-model factory, model per role
│   ├── token_usage.py          # Token usage tracking and summaries
│   ├── token_usage_callback.py # LangChain callback that records usage
│   ├── agents/                 # Planner, Explainer, Quiz, Coach, human approval
│   ├── graph/                  # AgentState and LangGraph workflow
│   ├── mcp_servers/            # Filesystem and memory MCP servers
│   ├── a2a_services/           # Quiz A2A service and A2A client
│   ├── crewai_agent/           # CrewAI Study Buddy served over A2A
│   └── observability/          # Langfuse setup
├── tests/                      # Unit tests and DeepEval evals
├── study_materials/sample_notes/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_SELECTION.md
│   └── PRODUCTION_READINESS_PLAN.md
├── docker-compose.yml          # Self-hosted Langfuse
├── Makefile
└── requirements.txt
```

---

## Roadmap

This is currently a single-user system.
[docs/PRODUCTION_READINESS_PLAN.md](docs/PRODUCTION_READINESS_PLAN.md) covers
the gap analysis and phased plan for a multi-user, on-prem deployment: an
API and job queue, Postgres checkpointing, SSO, secured A2A services, an LLM
gateway, and CI/CD.

---

## Credits and license

Original project and handbook by
[Sandeep Bharadwaj Mannapur](https://github.com/sandeepmb). OpenRouter
migration, token tracking, and production planning by
[Syamsruthin Kaliki](https://github.com/syamsruthin).

MIT licensed. See [LICENSE](LICENSE).
