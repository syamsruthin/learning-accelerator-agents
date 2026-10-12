# Production Readiness Plan — On-Prem Deployment

**Date:** 2026-10-09
**Baseline:** commit `5425ad7` (OpenRouter migration + token usage tracking)
**Target:** a multi-user, highly available, observable, secure deployment of the Learning Accelerator inside an on-prem data center.

---

## 0. Where we are today (gap analysis)

The current system is a well-structured **single-user, single-process demo**. Every item below is something that breaks, leaks, or is invisible once more than one person uses it on shared infrastructure.

| # | Area | Current state | Why it blocks production | Where |
|---|---|---|---|---|
| G1 | Serving model | CLI (`main.py`) and Streamlit; quiz answers collected with `input()` inside a graph node | A graph node cannot block on stdin in a server. No API exists for any other client. | `src/agents/quiz_generator.py:230`, `main.py` |
| G2 | Checkpointing | `SqliteSaver` on a local file, connection opened at **import time** (`graph = build_graph()`) | SQLite cannot be shared by multiple pods/replicas; import-time side effects break workers and tests | `src/graph/workflow.py` |
| G3 | Session memory | MCP memory server is a module-level `_store: dict` | Lost on restart, not shared across replicas, unbounded growth | `src/mcp_servers/memory_server.py:55` |
| G4 | A2A services | `InMemoryTaskStore`, bound to `0.0.0.0`, no auth, no TLS, started with `&` and stopped with `pkill` | Any host on the network can call them and spend LLM budget; tasks lost on restart | `src/a2a_services/quiz_service.py`, `src/crewai_agent/study_buddy.py` |
| G5 | Identity | No users. Session id = first 8 chars of a UUID; Langfuse `user_id="local"` | Guessable/colliding ids; anyone with an id can resume anyone's session | `main.py`, `src/observability/langfuse_setup.py` |
| G6 | Secrets | `.env` file; Langfuse compose has hard-coded `NEXTAUTH_SECRET`, `SALT`, all-zero `ENCRYPTION_KEY`, `postgres/postgres` | Unacceptable in any shared environment | `docker-compose.yml`, `.env` |
| G7 | Observability stack | Langfuse v3 compose ships only Postgres | Langfuse v3 also needs ClickHouse, Redis and S3-compatible storage — the current compose is incomplete for a real install | `docker-compose.yml` |
| G8 | Logging / metrics | `print()` everywhere; token usage in a process-global dict + local JSONL file | No structured logs, no correlation ids, no metrics, usage totals mix all users together | all agents, `src/token_usage.py:63` |
| G9 | LLM egress | Direct HTTPS from app to `openrouter.ai`; notes content is sent to third-party providers | On-prem networks usually forbid direct egress; data-residency/classification review required (e.g. `clinical_trials_basics.md`) | `src/llm.py` |
| G10 | Resilience | Broad `except Exception` → silent fallbacks; 120 s timeout × 2 retries; Explainer up to 8 sequential calls; **roadmap rejection loop is uncapped** | Requests can run for many minutes; failures are masked as "success"; a user can loop the planner forever and burn budget | `src/agents/*`, `src/graph/workflow.py` |
| G11 | Notes / content | Notes are a fixed folder on local disk; search is substring grep | No per-user upload, no shared storage across replicas, no scaling | `src/mcp_servers/filesystem_server.py` |
| G12 | Packaging | No Dockerfile; `sys.path.insert` hacks; `requirements.txt` mixes runtime + test + eval deps (deepeval, pytest); no lock hashes | Not reproducible, image bloat, larger attack surface | `requirements.txt`, `main.py`, `pyproject.toml` |
| G13 | CI/CD | None | No gate on lint, types, tests, vulnerabilities, or eval regressions | — |
| G14 | Docs drift | `pyproject.toml` still says eval tests "require Ollama" | Minor, but signals config drift | `pyproject.toml` |

---

## 1. Decisions to make before starting

These change the shape of the work. Recommended option first.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | **LLM inference location** | (a) On-prem LLM gateway (LiteLLM Proxy) forwarding to OpenRouter through an egress proxy; (b) fully self-hosted open-weight models on GPUs (vLLM); (c) hybrid | **(a) now, designed so (b)/(c) is a config change.** Put a LiteLLM Proxy inside the DC; the app only ever talks to it with OpenAI-compatible calls. Routing, keys, budgets, rate limits, logging and later vLLM backends live there. Requires a security/data-classification sign-off for content leaving the DC. If the data classification forbids that, (b) becomes mandatory and adds a GPU work stream (Phase 4b). |
| D2 | **Runtime platform** | (a) Kubernetes (RKE2, OpenShift, or Tanzu — whatever the DC already runs); (b) Docker Compose / Podman on VMs with systemd | **(a)** if a cluster exists; (b) only for a pilot (<50 users) — the plan below notes the compose-equivalent where it matters. |
| D3 | **Frontend** | (a) Keep Streamlit, refactored to call the API; (b) proper SPA (React/Next.js) | **(a) for v1**, (b) later. Streamlit's sticky-session/websocket model is acceptable behind the ingress with session affinity. |
| D4 | **Identity provider** | Corporate OIDC (Keycloak, ADFS, Azure AD/Entra on-prem bridge, Okta) | Use whatever the org SSO is; app speaks **OIDC** only. |
| D5 | **Scale targets / SLOs** | Concurrent users, sessions/day, p95 latency per step, availability | Need numbers. Placeholder: 200 concurrent sessions, 99.5% availability (business hours), p95 first-token < 5 s with streaming. |
| D6 | **Data retention & classification** | How long to keep sessions, answers, traces, usage logs | Placeholder: sessions 1 yr, traces 30 d, usage 13 mo; right-to-delete per user. |
| D7 | **Keep CrewAI + A2A as separate services?** | (a) Keep as separate deployables; (b) fold into the main worker | **(a)** — it's the architecture's point, but they must be secured (Phase 3). Cost: two more services to run. |

---

## 2. Target architecture

```
                   ┌──────────────── On-prem DC ──────────────────────────────────────────────┐
 Users ──HTTPS──▶  │ L4/L7 LB ─▶ Ingress (TLS, WAF) ─▶ Streamlit UI ─▶ API (FastAPI)          │
 (SSO)             │                                     │               │  ▲ SSE stream    │
                   │                                     └── OIDC ──▶ IdP│  │               │
                   │                                                     ▼  │               │
                   │                                Job queue (Redis Streams / Arq / Celery)│
                   │                                                     ▼                  │
                   │                         Graph workers (LangGraph, async) ──▶ A2A: Quiz svc│
                   │                              │       │        │          ──▶ A2A: Study  │
                   │                              │       │        │               Buddy     │
                   │                   PostgreSQL (HA)   Redis   MinIO (notes, artifacts)   │
                   │                   - checkpoints     - queue                             │
                   │                   - app data        - rate limits                       │
                   │                   - session memory  - cache                             │
                   │                              │                                          │
                   │                              ▼                                          │
                   │                    LiteLLM Proxy (keys, budgets, routing, retries)      │
                   │                              │                     └──▶ vLLM (optional) │
                   │                    Egress proxy (allowlist) ─────────────▶ OpenRouter   │
                   │                                                                         │
                   │  Platform: Vault · cert-manager/internal CA · Harbor · Prometheus/Grafana│
                   │            Loki · Tempo/OTel · Alertmanager · Langfuse (PG+CH+Redis+S3)  │
                   │            Velero / pgBackRest backups                                   │
                   └─────────────────────────────────────────────────────────────────────────┘
```

Key principle: **the LangGraph code stays the same**; only I/O, storage backends, and process boundaries change.

---

## 3. Phased plan

Each step lists *what*, *where*, and the *done-when* criterion. Phases 1–3 are application work and can largely run in parallel with Phase 4 (platform).

### Phase 1 — Application hardening (make it servable)

**1.1 Packaging & project layout**
- [ ] Convert to an installable package: move `src/*` under `src/learning_accelerator/`, define it in `pyproject.toml` (`[project]`, `[project.scripts]`), remove every `sys.path.insert` (`main.py`, `quiz_service.py`, `tests/conftest.py`).
- [ ] Split dependencies: runtime / `a2a-quiz` / `study-buddy` (crewai, litellm) / `dev` (pytest, ruff, mypy) / `eval` (deepeval). The API image must not contain deepeval or pytest.
- [ ] Generate a lock file with hashes (`uv lock` or `pip-compile --generate-hashes`) and install with `--require-hashes`.
- [ ] Fix stale `pyproject.toml` Ollama references (G14).
- *Done when:* `pip install .` works in a clean venv with no path hacks; tests pass.

**1.2 Configuration**
- [ ] Replace scattered `os.getenv` with one `pydantic-settings` `Settings` class (validated at startup, fails fast on missing required values, typed).
- [ ] Remove `load_dotenv()` from production code paths (keep for local dev only via a `DEV_MODE` flag).
- [ ] All secrets read from env vars / mounted files injected by Vault — never from a file in the image.
- *Done when:* app refuses to start with a clear error listing missing/invalid config.

**1.3 Remove blocking I/O from graph nodes (G1)**
- [ ] Rewrite `quiz_generator_node` to use `interrupt()` per question (or one interrupt for the whole question set), exactly as `human_approval_node` already does. Delete the `input()` path.
- [ ] Remove all `print()` user-facing output from nodes; nodes return state only. Presentation moves to the client (CLI/Streamlit/API).
- [ ] Keep `main.py` as a thin CLI client that drives interrupts — useful for ops smoke tests.
- *Done when:* a full session can be driven entirely through `graph.invoke(Command(resume=...))` with no stdin.

**1.4 Graph construction & checkpointing (G2)**
- [ ] Remove the module-level `graph = build_graph()`; build the graph in an app factory / worker startup with an injected checkpointer.
- [ ] Switch to `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`) with a pooled connection (psycopg pool). Keep `SqliteSaver` only for local dev/tests.
- [ ] Move to the async graph API (`ainvoke` / `astream`) and async LLM calls end-to-end.
- [ ] Use full UUIDv4 (or ULID) for session/thread ids — never truncated (G5).
- *Done when:* two worker replicas can resume each other's sessions.

**1.5 Persistent session memory (G3)**
- [ ] Back the memory MCP tools with Postgres (table `session_memory(session_id, key, value jsonb, updated_at)`) or Redis with TTL. Keep the same function signatures.
- [ ] Enforce that `session_id` comes from the authenticated request context, **not from LLM tool arguments** (today the LLM passes `session_id` into `tool_memory_get/set` — a prompt-injected model could read another session's memory). Bind it via closure/`RunnableConfig`.
- *Done when:* memory survives restarts and a tool call cannot address another session.

**1.6 Application data model**
- [ ] Postgres schema with **Alembic** migrations: `users`, `study_sessions` (owner, goal, status, created/updated), `quiz_results`, `llm_usage` (per user/session/role/model, tokens, cost, latency), `notes` metadata, `audit_log`.
- [ ] Replace JSONL usage logging (`src/token_usage.py`) and the process-global `_totals` with per-request accounting written to `llm_usage` + Prometheus counters (G8).
- *Done when:* migrations run idempotently on deploy; usage is attributable per user/session.

**1.7 Resilience & limits (G10)**
- [ ] Cap planner rejections (e.g. 3 replans, then end with a clear message) in `route_after_approval`.
- [ ] Per-call timeouts tuned per role (e.g. 30 s coach/quiz, 60 s planner/explainer) and a **total per-step deadline**.
- [ ] Replace broad `except Exception` + silent fallbacks with typed errors; fallbacks are fine but must be **logged, metered and flagged in state** (`degraded=True`) so the UI can show it.
- [ ] Retries with jitter only on retryable errors (429, 5xx, timeouts) — tenacity is already a dependency; LiteLLM proxy also retries, so avoid retry multiplication (set app retries to 0–1 when the proxy retries).
- [ ] Circuit breaker around A2A calls (fall back to local quiz generation, which already exists).
- [ ] Input validation & size limits: goal length, answer length, max topics, max notes per user, max file size.
- [ ] Idempotency: resume requests carry an idempotency key so double-submits don't advance the graph twice.

**1.8 Notes / content storage (G11)**
- [ ] Per-user note upload API → MinIO (S3-compatible) bucket, metadata in Postgres.
- [ ] Filesystem MCP tools read from object storage scoped to the current user (keep the existing path-traversal defense, add ownership check).
- [ ] File type allowlist (md/txt/pdf), size limit, antivirus scan (ClamAV) on upload.
- [ ] (Later) Replace substring search with embeddings + pgvector for retrieval quality.

**1.9 LLM safety**
- [ ] Treat note content and quiz answers as untrusted: delimit them in prompts, instruct models not to follow embedded instructions, never let tool args select identities (see 1.5).
- [ ] Output validation: Pydantic schemas for every JSON-mode response (planner, quiz, grader, coach); reject/repair invalid output.
- [ ] Grader manipulation: clamp scores to [0,1], detect answers that attempt to instruct the grader.
- [ ] Optional: PII redaction before egress (Presidio) if D1 sign-off requires it.

### Phase 2 — Service layer (API, workers, UI)

**2.1 Backend API (FastAPI — already a dependency)**
- [ ] Endpoints (versioned `/api/v1`):
  - `POST /sessions` (goal) → session id, starts planning job
  - `GET /sessions/{id}` → state, pending interrupt payload
  - `POST /sessions/{id}/resume` → interrupt answer (approve/reject roadmap, quiz answers)
  - `GET /sessions/{id}/events` → **SSE** stream of progress & streamed tokens
  - `GET /sessions` (mine), `DELETE /sessions/{id}`
  - `POST/GET/DELETE /notes`
  - `GET /me/usage`
  - `GET /healthz` (liveness), `GET /readyz` (DB, Redis, LLM gateway reachability), `GET /metrics`
- [ ] OpenAPI schema published; request/response models in Pydantic.
- [ ] Authorization check on every session/notes endpoint: `session.owner_id == current_user.id` (or admin role).
- [ ] Gunicorn + Uvicorn workers, graceful shutdown (drain in-flight requests, `SIGTERM` handling).

**2.2 Async workers**
- [ ] Long-running graph steps (planner, explainer loop) run in background workers fed by a queue (Arq/Redis Streams or Celery); API returns immediately and clients follow via SSE.
- [ ] Per-user concurrency limit (e.g. 1 active step per session, N sessions per user).
- [ ] Dead-letter handling + retry policy for failed jobs; step-level state is already checkpointed so a crashed worker can resume.

**2.3 A2A services (G4)**
- [ ] Containerize each service; bind to pod network only (no public ingress).
- [ ] Authentication: mTLS between services (service mesh like Linkerd/Istio, or cert-manager-issued client certs) **and/or** OAuth2 client-credentials JWT validated by the A2A server; publish the auth scheme in the Agent Card `securitySchemes`.
- [ ] Replace `InMemoryTaskStore` with a persistent task store (a2a-sdk `DatabaseTaskStore` on Postgres) or accept stateless semantics and document it.
- [ ] Propagate user/session id and trace context (W3C `traceparent`) on every A2A call.
- [ ] Health endpoints and resource limits per service.

**2.4 Frontend**
- [ ] Refactor `streamlit_app.py` to be a pure API client (no direct `build_graph()` import, no direct DB access).
- [ ] OIDC login (via the ingress `oauth2-proxy` or Streamlit's auth), pass the access token to the API.
- [ ] Show degraded-mode banners, streaming explanations, session list/resume UI.
- [ ] Session affinity on ingress for Streamlit websockets.

### Phase 3 — Security & compliance

**3.1 Identity & access**
- [ ] OIDC with the corporate IdP (D4); API validates JWT (issuer, audience, expiry, signature via JWKS).
- [ ] Roles: `learner`, `admin`, `ops-readonly`. Admin endpoints for usage reports and user data deletion.
- [ ] Service-to-service identities (Phase 2.3).

**3.2 Secrets**
- [ ] HashiCorp Vault (or the DC's existing secret manager) + External Secrets Operator / Vault Agent injector.
- [ ] Secrets: OpenRouter key (held **only by LiteLLM proxy**), LiteLLM virtual keys per service, DB creds (dynamic DB secrets if possible), Langfuse keys, MinIO creds, OIDC client secret.
- [ ] Rotate everything that ever lived in a `.env` on a laptop, including the current OpenRouter key.
- [ ] Regenerate Langfuse `NEXTAUTH_SECRET`, `SALT`, `ENCRYPTION_KEY` (32-byte random) (G6).
- [ ] Pre-commit + CI secret scanning (gitleaks).

**3.3 Network**
- [ ] Default-deny `NetworkPolicy`; explicit allow: ingress→UI→API; API/workers→Postgres/Redis/MinIO/LiteLLM/A2A; LiteLLM→egress proxy only.
- [ ] Egress proxy allowlist: `openrouter.ai` only (plus whatever package mirrors CI needs).
- [ ] TLS everywhere: ingress certs from internal CA via cert-manager; in-cluster mTLS via mesh or per-service certs; Postgres `sslmode=verify-full`.
- [ ] WAF/rate limiting at ingress (per IP and per user).

**3.4 Data protection**
- [ ] Encryption at rest: Postgres volumes (LUKS/storage-level), MinIO SSE, backups encrypted.
- [ ] Data classification review of what is sent to OpenRouter and which upstream providers it may route to; configure OpenRouter provider routing / data-policy settings (e.g. disallow providers that train on/retain data) accordingly — **sign-off required (D1)**.
- [ ] Retention jobs per D6 (sessions, traces, usage, uploaded notes); user data export & hard-delete flow.
- [ ] Audit log of admin actions and data deletions.

**3.5 Supply chain & hardening**
- [ ] Container images: distroless or `python:3.x-slim`, non-root user, read-only root FS, no shell in runtime image where feasible, pinned base digest.
- [ ] Vulnerability scanning: `pip-audit` + Trivy/Grype on images; fail CI on high/critical.
- [ ] SBOM generation (Syft) and image signing (Cosign); admission policy (Kyverno/Gatekeeper) enforces signed images from internal registry only, no `:latest`, resource limits present, non-root.
- [ ] SAST (Bandit/Semgrep) in CI.
- [ ] Threat model (STRIDE) of the agent system incl. prompt injection, tool abuse, A2A spoofing, cost-exhaustion; pen test before go-live.

### Phase 4 — On-prem platform

**4.1 Foundations (coordinate with infra team)**
- [ ] Environments: `dev`, `staging` (prod-like, smaller), `prod`. Separate namespaces or clusters; separate secrets and LLM budgets.
- [ ] Kubernetes cluster (D2) with ≥3 control-plane nodes and ≥3 workers across failure domains (racks/hypervisor hosts); CIS-hardened nodes; NTP; internal DNS records.
- [ ] Storage class with replicated block storage (Ceph/Longhorn/SAN CSI) for Postgres/ClickHouse/MinIO.
- [ ] Load balancer (MetalLB / F5 / HAProxy pair with keepalived) + Ingress controller (ingress-nginx/Traefik/OpenShift router).
- [ ] cert-manager with internal CA issuer.
- [ ] Internal artifact mirrors for air-gapped or restricted networks: **Harbor** (container images, proxy cache for Docker Hub/GHCR), PyPI mirror (Nexus/Artifactory/devpi), Helm chart repo.

**4.2 Data services**
- [ ] **PostgreSQL HA** via CloudNativePG (or Patroni on VMs): 1 primary + 2 replicas, synchronous replication for checkpoints, PgBouncer pooling, pgBackRest/Barman WAL archiving to MinIO/backup target. Separate databases (or clusters) for app and Langfuse.
- [ ] **Redis** (Sentinel or operator) for queue, rate limits, cache; persistence enabled for the queue.
- [ ] **MinIO** distributed mode (≥4 nodes) for notes, Langfuse events/media, backups — or the DC's existing S3-compatible store.

**4.3 LLM gateway (D1)**
- [ ] Deploy **LiteLLM Proxy** (HA, 2+ replicas, Postgres-backed) with:
  - model aliases matching our roles (`planner`, `explainer`, `quiz`, `grader`, `coach`, `study_buddy`, `eval_judge`) → OpenRouter model ids; change `src/llm.py` to call the proxy with alias names so model swaps are a proxy config change
  - virtual keys per service/environment, monthly budgets, RPM/TPM limits
  - fallbacks (e.g. explainer → secondary model on provider outage)
  - outbound via egress proxy (`HTTPS_PROXY`)
  - spend logs → Prometheus
- [ ] Point `OPENROUTER_BASE_URL` (rename to `LLM_GATEWAY_URL`) at the proxy; CrewAI/LiteLLM client in Study Buddy points there too.

**4.4 (Optional, if D1 = self-hosted) GPU inference**
- [ ] GPU nodes + NVIDIA GPU Operator; vLLM serving chosen open-weight models; capacity sizing from token metrics gathered in staging.
- [ ] Re-run the DeepEval suite against self-hosted models; JSON-mode/tool-calling compatibility check for each role; adjust per-role model mapping in the gateway.

**4.5 Observability**
- [ ] **Logging:** structured JSON logs (structlog), request/session/user/trace ids on every line, no secrets or full prompts at INFO; ship via Fluent Bit/Promtail → Loki (or the org's ELK/Splunk).
- [ ] **Metrics:** Prometheus + Grafana. App metrics: request rate/latency/errors per endpoint, graph step duration per node, LLM calls/tokens/cost/latency/errors per role & model, fallback/degraded count, queue depth & age, active sessions, A2A call success. Platform metrics via kube-prometheus-stack, postgres/redis/minio exporters.
- [ ] **Tracing:** OpenTelemetry SDK in API, workers, A2A services → OTel Collector → Tempo/Jaeger; propagate context across queue and A2A.
- [ ] **LLM tracing:** Langfuse v3 deployed properly via its Helm chart (web + worker, Postgres, ClickHouse, Redis, S3/MinIO) — fixes G7. Set real `user_id`/`session_id` per trace; sampling rate configurable; retention per D6.
- [ ] **Dashboards:** service overview, LLM cost & tokens per role/user, quality (grader score distributions, eval trend), infrastructure.
- [ ] **Alerts (Alertmanager → on-call channel):** API 5xx rate, p95 latency breach, queue age, worker crash loop, LLM gateway error rate / provider outage, budget at 80%/100%, Postgres replication lag / disk, backup failure, cert expiry < 14 d.

**4.6 Deployment packaging**
- [ ] Dockerfiles: `api`, `worker`, `ui`, `a2a-quiz`, `a2a-study-buddy` (multi-stage, shared base).
- [ ] Helm chart (or Kustomize) for the app: Deployments, Services, HPA (CPU + queue-depth via KEDA for workers), PodDisruptionBudgets, probes, resource requests/limits, anti-affinity, NetworkPolicies, ServiceAccounts, migration `Job` (pre-upgrade hook).
- [ ] GitOps (Argo CD/Flux) syncing per-environment values from a config repo.
- [ ] Compose-on-VM fallback (if D2 = b): `docker-compose.prod.yml` with the same images, systemd units, HAProxy in front, and an equivalent backup/monitoring setup.

### Phase 5 — CI/CD & quality gates

- [ ] CI runner inside the DC (GitLab Runner / GitHub Actions self-hosted / Jenkins) with access to internal mirrors.
- [ ] Pipeline on every PR: `ruff` lint+format, `mypy`/pyright, unit tests with coverage threshold, `pip-audit`, Bandit/Semgrep, gitleaks, Docker build, Trivy scan, SBOM.
- [ ] Integration tests with Testcontainers (Postgres, Redis, MinIO) + a **mock LLM gateway** (deterministic fixtures) so CI doesn't spend tokens.
- [ ] Contract tests for A2A Agent Cards and request/response formats.
- [ ] **Eval gate:** nightly + pre-release DeepEval run against staging through the gateway with its own budget; fail release if faithfulness/relevancy/grading-agreement drop below thresholds; track scores over time in Langfuse datasets.
- [ ] Release flow: main → build signed images → deploy to staging automatically → smoke tests + eval gate → manual approval → prod (canary or blue/green via Argo Rollouts) → automatic rollback on SLO burn.
- [ ] Semantic versioning, changelog, DB migrations must be backward compatible for one release (expand/contract).

### Phase 6 — Testing for production

- [ ] **Load test** (k6/Locust) with the mock LLM to find app bottlenecks, then a small budgeted run against real models to measure end-to-end latency and provider rate limits; size HPA & DB pool from results (D5).
- [ ] **Soak test** (8–24 h) for memory leaks and connection-pool exhaustion.
- [ ] **Failure injection:** kill workers mid-step (must resume from checkpoint), Postgres failover, Redis restart, LLM gateway 429/5xx/timeouts, A2A service down (must fall back), egress proxy down.
- [ ] **Security tests:** authz tests (user A cannot read/resume user B's sessions or notes), prompt-injection test suite on notes and answers, path-traversal & upload fuzzing, DAST (OWASP ZAP) against staging, external pen test.
- [ ] **Backup restore drill:** restore Postgres + MinIO into a fresh namespace and resume a session.

### Phase 7 — Operations readiness

- [ ] SLOs & error budgets (D5) defined in Grafana/Sloth.
- [ ] Runbooks: LLM provider outage / budget exhausted, queue backlog, Postgres failover, restoring from backup, rotating secrets, rotating LLM keys, cert renewal failure, user data deletion request, model swap.
- [ ] DR: documented RPO/RTO (e.g. RPO 15 min via WAL archiving, RTO 4 h); off-site/second-DC backup copy; annual DR test.
- [ ] Capacity plan & cost model: tokens per session by role (we already log this) × expected sessions → monthly LLM spend; set gateway budgets accordingly.
- [ ] On-call rotation, escalation paths, status communication.
- [ ] Patch cadence: base images & dependencies monthly (Renovate/Dependabot against internal mirrors), K8s/OS per infra policy.
- [ ] User-facing: acceptable-use notice, disclosure that content is processed by external LLM providers (if D1 = a), support channel.

### Phase 8 — Go-live

- [ ] Production readiness review checklist sign-off (security, infra, data protection, ops).
- [ ] Pilot with a small user group (feature flag / allowlist), monitor cost/quality/latency for 1–2 weeks.
- [ ] Gradual rollout; hypercare period with daily review of dashboards, cost, eval scores, and user feedback.
- [ ] Post-launch backlog: SPA frontend, pgVector retrieval, self-hosted models, multi-DC.

---

## 4. Suggested sequencing

| Sprint (2 wk) | Application track | Platform track |
|---|---|---|
| 1 | 1.1 packaging, 1.2 config, 1.3 remove `input()` | D1–D7 decisions, 4.1 cluster/registry/mirrors |
| 2 | 1.4 Postgres checkpointer + async, 1.5 memory, 1.6 schema | 4.2 Postgres/Redis/MinIO, 3.2 Vault |
| 3 | 2.1 API + 3.1 auth, 2.2 workers | 4.3 LiteLLM proxy + egress, 4.6 Dockerfiles/Helm |
| 4 | 2.3 A2A security, 2.4 Streamlit refactor, 1.7 resilience | 4.5 observability + Langfuse v3, 5 CI/CD |
| 5 | 1.8 notes storage, 1.9 LLM safety | 3.3 network policies, 3.5 supply chain |
| 6 | Phase 6 testing, fixes | Phase 7 ops readiness, DR drill |
| 7 | Phase 8 pilot | hypercare |

Roughly 12–14 weeks for 2–3 engineers plus infra support, assuming a Kubernetes cluster already exists. Add GPU procurement and model validation time if D1 = self-hosted.

---

## 5. Definition of done (production ready)

- Multiple users authenticate via SSO and cannot see each other's data (verified by tests).
- Any single pod, node, or Postgres primary can fail without losing a session.
- Every LLM call is attributable to a user/session/role with cost, and budgets are enforced at the gateway.
- No secrets in git, images, or `.env` files; all rotated.
- Logs, metrics, traces, and LLM traces exist for every request; alerts page on-call.
- CI blocks merges on tests, vulnerabilities, and eval-quality regressions.
- Backups are restored successfully in a drill; runbooks exist for the top failure modes.
- Security review and data-classification sign-off are complete.
