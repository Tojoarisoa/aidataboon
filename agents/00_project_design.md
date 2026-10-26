# AIDataBoon Agents — LLMOps & Multi-Agent Infrastructure
## Architecture & Design Document

**Domain:** agents.aidataboon.com (subdomain of aidataboon.com)
**Version:** 1.0
**Builds on:** `data_platform/` (Bronze/Silver/Gold lakehouse, RAG API, observability stack)

---

## 1. Overview

The data platform (`data_platform`) shipped a RAG API over a curated knowledge base
(Hacker News, arXiv, GitHub trending). This module turns that RAG API into the
retrieval tool for a small LLMOps platform: a **prompt registry** with
versioning and A/B rollout, a **RAG evaluation harness** with a golden
question set and CI regression gating, a **LangGraph-style multi-agent**
(planner → retriever → critic → answerer) with episodic memory, and an
**observability/guardrails layer** (OTel GenAI tracing, Prometheus agent
metrics, prompt-injection / PII / moderation / cost guardrails). All of it is
fronted by a FastAPI **agent API** that mounts the prompt registry's admin
routes and reuses the existing Postgres + pgvector instance from
`data_platform/01_local_stack.yml`.

Cost target: **+$0/mo** over the data platform stack when `LLM_PROVIDER=none`
(deterministic extractive fallback) or `LLM_PROVIDER=ollama` (local model);
optional pay-per-token providers (OpenAI `gpt-4o-mini`, Anthropic
`claude-haiku-4-5`) are supported via env vars but not required for the demo.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  agents.aidataboon.com  (Cloudflare Worker / Tunnel → agent-api:8100)    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────────┐
│  06_agent_api.py  (FastAPI, port 8100)                                    │
│                                                                            │
│   POST /agents/chat              -> Agent.chat(query, user_id)           │
│   GET  /agents/runs/{run_id}      -> AgentMemory.get_run                  │
│   POST /agents/runs/{id}/feedback -> AgentMemory.set_feedback             │
│   GET  /health  /metrics                                                  │
│   mount /prompts -> 02_prompt_registry.py admin sub-app                  │
│                                                                            │
│   Wires: prompt_provider (PromptRegistry.get_active),                    │
│          cost_estimator (GuardrailPipeline.cost.estimate_cost),           │
│          AgentTracer + AgentMetricsRegistry + GuardrailPipeline           │
└───────┬───────────────────┬───────────────────┬──────────────────────────┘
        │                   │                   │
┌───────▼────────┐  ┌───────▼─────────┐  ┌──────▼────────────────────────┐
│ 02_prompt_      │  │ 04_agent_       │  │ 05_agent_observability.py     │
│ registry.py     │  │ orchestration.py│  │  - OTel GenAI tracer            │
│  - versions     │  │  - LangGraph     │  │  - Prometheus agent_* metrics  │
│  - A/B buckets  │  │    graph: plan-  │  │  - PromptInjectionGuard        │
│  - promote/     │  │    ner/retriever/│  │  - PIIRedactionGuard           │
│    rollback     │  │    critic/       │  │  - OutputModerationGuard       │
│                 │  │    answerer      │  │  - CostTracker                 │
│                 │  │  - AgentMemory   │  │  - Grafana dashboard generator │
│                 │  │    (episodic,    │  │                                 │
│                 │  │    pgvector)     │  │                                 │
└────────┬────────┘  └────────┬─────────┘  └────────────────────────────────┘
         │                    │ tool: RAG search over HTTP
         │                    ▼
         │           ┌─────────────────────────┐
         │           │ data_platform/     │
         └──────────►│ 05_rag_service.py        │
                      │ (port 8000, pgvector)    │
                      └─────────────────────────┘

03_eval_harness.py — standalone CLI + CI gate:
  golden_qa.jsonl -> runs Agent.chat() per question -> RAGAS-style metrics
  (faithfulness, answer_relevancy, context_precision/recall via heuristic
  fallback or real `ragas` if installed) -> writes eval_report.json,
  fails (`exit 1`) if mean score < threshold.
```

---

## 3. Architecture Decision Records

### ADR-1: Prompt Registry — Postgres table + in-memory fallback, A/B by hash bucket
Prompts are versioned rows in a `prompt_versions` table (same Postgres as the
RAG API's pgvector). Each prompt `name` (e.g. `agent.planner.system`) can have
multiple versions; `is_active` + `traffic_pct` define a canary rollout.
`get_active(name, user_id)` deterministically buckets `user_id` via
`sha256(f"{name}:{user_id}") % 100` so the same user always sees the same
variant. Falls back to an in-memory dict (seeded from `DEFAULT_PROMPTS`) when
`psycopg`/Postgres is unavailable — keeps local dev and CI dependency-free.

### ADR-2: Observability — OTel GenAI semantic conventions + Prometheus + 4 guardrails
Reuses the data platform's `setup_tracing()`/`MetricsRegistry` pattern. Spans use the
OTel GenAI semconv attribute names (`gen_ai.system`, `gen_ai.request.model`,
`gen_ai.usage.*`, `gen_ai.agent.name`, `gen_ai.operation.name`) as plain
string constants (the `opentelemetry.semconv._incubating` module is private
and unstable, so we don't import from it). New `agent_*` Prometheus metrics
(run duration, token/cost counters, guardrail blocks, eval scores) are
exposed on the agent API's own `/metrics` and added as a 7-panel Grafana
dashboard JSON dropped into the data platform's existing Grafana dashboard volume, no
compose changes needed. Four guardrails run on every request:
prompt-injection pattern matching, PII detection + redaction, output
moderation (banned terms + max length), and a cost/budget tracker.

### ADR-3: Agent Orchestration — LangGraph graph with deterministic fallback
`AgentOrchestrator` builds a 4-node graph (planner → retriever → critic →
answerer, with a critic→retriever loop up to `max_iterations`) using
`langgraph.graph.StateGraph` when installed, and a hand-rolled
`SimpleAgentGraph` (same node functions, plain Python loop) otherwise — so
the agent runs identically with or without the `langgraph` package. The
retriever node calls the data platform RAG API over HTTP (`httpx`); planner/critic/
answerer call `LLMClient`, which itself supports `openai`, `anthropic`,
`ollama`, or `none` (deterministic extractive answer from retrieved docs) via
the same `_AVAILABLE` flag pattern as the rest of the codebase.

### ADR-4: Episodic memory — pgvector table of past runs, embedded for recall
`AgentMemory` persists every run (query, full step trace, answer, citations,
token usage, cost, latency, user feedback) to a `agent_runs` /
`agent_messages` table pair, embedding the query with the same
`sentence-transformers`-or-hash-fallback embedder as `05_rag_service.py`
(`EMBEDDING_DIM=384`) so `recall_similar(query)` can surface prior runs on
similar questions as few-shot context. Falls back to an in-memory list of
dicts when Postgres is unavailable.

### ADR-5: Eval harness — golden Q&A set + RAGAS-style metrics, CI-gated
`03_eval_harness.py` ships a small `golden_qa.jsonl` (10-15 Q&A pairs with
expected-keyword ground truth, drawn from the AIDataBoon knowledge base). It
runs each question through `Agent.chat()`, scores
faithfulness/answer-relevancy/context-precision/context-recall using `ragas`
if installed, else a keyword/embedding-overlap heuristic, aggregates into
`eval_report.json`, and exits non-zero if the mean composite score drops
below `EVAL_MIN_SCORE` (default 0.6; CI uses 0.30, a baseline for the
heuristic fallbacks) — wired into `.github/workflows/agents-cicd.yml` as a
regression gate on every PR.

### ADR-6: user_id threading for per-user A/B without per-request reconnects
`Agent`/`AgentOrchestrator` are constructed once (singletons in the FastAPI
app's lifespan) to avoid reconnecting to Postgres per request. `user_id`
flows through `AgentState` (TypedDict) into `prompt_provider(name, user_id)`
and `AgentMemory.start_run(query, user_id=...)`, so prompt A/B bucketing and
episodic memory stay per-user without recreating any client objects.

### ADR-7: Cloudflare — separate Terraform root for `agents.aidataboon.com`
`07_cloudflare_agents.tf` is a standalone Terraform configuration (separate
state from `data_platform/09_cloudflare_infra.tf`) that adds a
`agents` DNS record + Cloudflare Tunnel/Worker route for the agent API,
following the same "$0-5/mo" pattern (Cloudflare Tunnel to a local/VM
container, no managed compute spend).

---

## 4. Cost Summary

| Component                  | Local dev | Production (min cost)            |
|-----------------------------|-----------|-----------------------------------|
| Prompt registry              | $0 (Postgres reuse) | $0 (same Postgres) |
| Agent orchestration (LLM_PROVIDER=none/ollama) | $0 | $0 |
| Agent orchestration (OpenAI gpt-4o-mini) | ~$0.15/$0.60 per 1M tok | usage-based, capped by CostTracker |
| Episodic memory (pgvector)   | $0 (Postgres reuse) | $0 |
| Observability (OTel/Prometheus/Grafana) | $0 (data-platform stack) | $0 |
| Cloudflare DNS/Tunnel for agents.aidataboon.com | $0 | $0 |
| **Total incremental**        | **$0**    | **$0 - usage-based LLM tokens only** |

---

## 5. Module Map

| File | Purpose |
|------|---------|
| `00_project_design.md` | This document |
| `01_agents_stack.yml` | Docker Compose `include:` of data-platform stack + `agent-api` service |
| `02_prompt_registry.py` | Prompt versioning, A/B rollout, admin FastAPI sub-app |
| `03_eval_harness.py` | Golden Q&A + RAGAS-style eval, CI gate |
| `04_agent_orchestration.py` | LangGraph multi-agent + episodic memory |
| `05_agent_observability.py` | OTel tracing, Prometheus metrics, guardrails, Grafana dashboard |
| `06_agent_api.py` | FastAPI agent API (integration layer) |
| `Dockerfile.agents` / `requirements-agents.txt` / `.dockerignore` | Container image for `agent-api` |
| `07_cloudflare_agents.tf` | DNS/Tunnel for `agents.aidataboon.com` |
| `../.github/workflows/agents-cicd.yml` | GitHub Actions: lint, typecheck, eval gate, image build (Render deploys after checks pass) |
