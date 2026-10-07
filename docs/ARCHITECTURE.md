# Architecture Reference

> **Note:** This project now uses OpenRouter instead of local Ollama for all LLM calls.
> Per-agent model defaults live in `src/llm.py` (`ROLE_MODELS`). Ollama-specific guidance below is kept for reference.

This document explains the architectural decisions behind the Learning
Accelerator multi-agent system.

---

## System overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph System                          │
│                                                             │
│  curriculum_planner → human_approval → explainer            │
│                              ↑              ↓              │
│                       (rejected)      quiz_generator        │
│                                            ↓               │
│                                      progress_coach         │
│                                       ↓        ↓           │
│                                (next topic)  (done→END)     │
└─────────────────────────────────────────────────────────────┘
         │ MCP (tools)              │ A2A (agents)
         ▼                          ▼
┌──────────────────┐    ┌─────────────────────┐  ┌──────────────────────┐
│  Filesystem MCP  │    │  Quiz Generator A2A  │  │ CrewAI Study Buddy   │
│  Memory MCP      │    │  (port 9001)         │  │ A2A (port 9002)      │
└──────────────────┘    └─────────────────────┘  └──────────────────────┘
         │                                                │
         ▼                                                ▼
┌──────────────────────────────────────────────────────────────┐
│                      Ollama (localhost:11434)                 │
│              qwen2.5:7b  |  qwen2.5-coder:32b                │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                 Langfuse (localhost:3000)                     │
│         Full traces · Token counts · Latency per agent       │
└──────────────────────────────────────────────────────────────┘
```

---

## Why LangGraph

LangGraph models the agent workflow as a directed graph where:
- **Nodes** are Python functions (agents or utilities)
- **Edges** define routing, static or conditional
- **State** is a typed `dict` shared across all nodes
- **Checkpoints** are saved to SQLite after every node execution

The alternative (a simple loop) loses all progress on crash and cannot
support human-in-the-loop approval without invasive changes. LangGraph
makes resilience and human oversight first-class primitives.

### Graph state

All agents read from and write to a single `AgentState` dict:

```python
class AgentState(TypedDict):
    messages:             Annotated[list[BaseMessage], add_messages]
    goal:                 str
    roadmap:              StudyRoadmap | None
    approved:             bool
    current_topic_index:  int
    quiz_results:         list[QuizResult]
    weak_areas:           list[str]
    session_id:           str
    study_materials_path: str
    error:                str | None
```

Nodes return **partial updates**, only the keys they changed.
LangGraph merges the update into the full state.

### Checkpointing

We call `sqlite3.connect(db_path, check_same_thread=False)` directly and pass
the connection to `SqliteSaver(conn)` (see `src/graph/workflow.py`).
The `check_same_thread=False` flag is required because LangGraph runs node
functions and checkpoint writes on different threads internally.
The connection stays open for the process lifetime, this is intentional.

```python
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)
```

### Human-in-the-loop

`interrupt()` inside `human_approval_node` pauses the graph and
persists a checkpoint. `main.py` collects user input and resumes
with `graph.invoke(Command(resume=user_input), config=config)`.
The same mechanism works for web UIs, the only difference is how
input is collected.

---

## Why MCP

Before MCP, every agent-tool integration was a custom adapter. MCP
standardises the interface so a tool server built once works with
any MCP-compatible client.

### MCP primitives used

| Primitive | Example | When used |
|---|---|---|
| **Tool** | `read_study_file(filename)` | Agent needs to do something |
| **Resource** | `notes://index` | Agent needs to read structured data |

### Process boundary

The MCP servers run as Python imports in development (simpler, testable)
but are designed to run as separate processes in production. The tool
functions are plain Python, switching from import to subprocess changes
one line.

### Security

`read_study_file()` resolves the requested path and verifies it falls
within `NOTES_BASE` before reading. Path traversal attempts return
an error string rather than raising, so the LLM can see what went wrong.

---

## Why A2A

MCP connects agents to tools. A2A connects agents to other agents.

The Quiz Generator is exposed as an A2A service so any agent from any
framework, LangGraph, CrewAI, Google ADK, a custom Python script.
can request quiz generation and grading without importing our code.

### A2A request flow

```
Progress Coach (LangGraph)
  1. GET /.well-known/agent-card.json  → discover capabilities
  2. POST /tasks/send                  → submit task (JSON-RPC 2.0)
  3. Parse result from artifacts[0].parts[0].text
```

### Circuit breaker pattern

Every A2A call first checks `is_quiz_service_available()`. If the
service is down, `try_a2a_quiz_delegation()` returns `None` and the
system falls back to local quiz generation. A2A is always optional,
never load-bearing.

### SDK version

Using `a2a-sdk==0.3.25` (spec v0.3.0). The v1.0 spec shipped March 2026.
The migration when the v1.0 SDK ships to PyPI is primarily enum casing
(`"submitted"` → `"TASK_STATE_SUBMITTED"`) and Agent Card URL restructuring.

---

## Why CrewAI for the Study Buddy

The Study Buddy is built with CrewAI to prove the architecture claim:
two agents from completely different frameworks can collaborate via A2A
without either knowing about the other.

CrewAI's role/goal/backstory abstraction maps naturally to a tutoring
agent. The `TopicAnalyserTool` gives the agent structured access to
topic context. The whole crew is rebuilt per-request to avoid state
leakage between sessions.

`crew.kickoff()` is synchronous, so it runs in `asyncio.to_thread()`
inside the A2A executor to avoid blocking the event loop.

---

## Observability

Langfuse integrates via LangChain's callback system. One handler
attached to `graph.invoke()` captures every agent node, LLM call,
and tool call automatically:

```python
config = {
    "configurable": {"thread_id": session_id},
    "callbacks": [langfuse_handler],
}
graph.invoke(state, config=config)
```

`flush_langfuse()` at process exit ensures all async traces are sent
before the process terminates.

---

## Testing strategy

| Tier | Command | Speed | Dependencies |
|---|---|---|---|
| Unit | `pytest tests/ -m "not eval"` | ~3s | None |
| Eval | `pytest tests/test_eval.py -m eval` | ~90s | Ollama |

Unit tests mock all LLM calls and test parsing, routing, and tool
logic deterministically. Eval tests run the actual agents and use
LLM-as-judge (via DeepEval + local Ollama) to score output quality.

### Key test patterns

- **Agent node tests**: mock `ChatOllama`, test state transformation
- **Routing tests**: call routing functions directly with crafted state
- **MCP tests**: call tool functions directly (no subprocess needed)
- **A2A tests**: mock `httpx` for all network calls
- **Eval tests**: run real agents, score with `FaithfulnessMetric` / `GEval`

---

## Data flow: one complete session

```
1. User: "Learn Python closures"
   → initial_state() creates AgentState

2. curriculum_planner_node(state)
   → LLM call: goal → StudyRoadmap JSON
   → state["roadmap"] = StudyRoadmap(5 topics)

3. interrupt() in human_approval_node
   → graph pauses, checkpoint saved
   → main.py shows roadmap, collects "yes"
   → graph.invoke(Command(resume="yes"))
   → state["approved"] = True

4. explainer_node(state)  [repeats per topic]
   → MCP: list_study_files() → ["closures.md", ...]
   → MCP: search_notes("closures") → matches
   → MCP: read_study_file("closures.md") → content
   → LLM: produces grounded explanation
   → MCP: memory_set(session_id, "explained_topics", ...)

5. quiz_generator_node(state)
   → LLM: generates 3 questions
   → input(): collects user answers
   → LLM × 3: grades each answer
   → state["quiz_results"].append(QuizResult)

6. progress_coach_node(state)
   → LLM: coaching message
   → MCP: memory_set(session_id, "progress_topic_0", ...)
   → if score < 0.5: A2A → CrewAI Study Buddy
   → state["current_topic_index"] += 1

7. route_after_coach(state)
   → index < len(topics): → "explainer" (loop)
   → index >= len(topics): → END
```

---

## Production considerations

See `docs/MODEL_SELECTION.md` for model recommendations by hardware.

For deploying beyond local development:
- Replace `InMemoryTaskStore` with a persistent store (PostgreSQL)
- Replace `MemoryServer._store` dict with Redis
- Add authentication to A2A services (`AgentAuthentication`)
- Run MCP servers as separate processes via stdio transport
- Use LiteLLM proxy for rate limiting and model fallbacks
- Add structured logging to replace print statements
