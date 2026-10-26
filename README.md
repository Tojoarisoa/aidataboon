# AI Data Boon

Today's top Hacker News stories, new arXiv papers and trending GitHub repositories on one dashboard, with plain-English search over all of them and short answers that cite their sources.

Live at [aidataboon.com](https://aidataboon.com).

## What it does

- **Daily dashboard.** A scheduled pipeline ingests Hacker News, arXiv and GitHub trending every day; the page opens on the latest batch from each source.
- **Search.** Hybrid retrieval (pgvector HNSW plus Postgres full-text search, fused with Reciprocal Rank Fusion) over everything ingested. Results come back newest first, and time phrases such as "this week" or "last 3 days" filter by publish date.
- **Cited answers.** Claude writes a two-sentence answer from the top results, citing them as [1], [2], and so on; the page links each citation to its result. Without an LLM configured, search still works and the answer is left out.

## Repository layout

| Folder | What it is |
|---|---|
| [`data_platform/`](data_platform/) | Ingestion (HN, arXiv, GitHub trending), a Bronze/Silver/Gold lakehouse (DuckDB, Delta Lake, Cloudflare R2), the RAG search API (FastAPI, pgvector), an ML platform (MLflow, Feast), observability (OpenTelemetry, Prometheus, Grafana) and infrastructure as code (Terraform for Cloudflare, plus an optional AWS variant). |
| [`agents/`](agents/) | LLMOps and multi-agent infrastructure on top of the search API: a prompt registry with A/B rollout, a CI-gated evaluation harness, a multi-agent pipeline with episodic memory, guardrails and tracing. |
| [`frontend/`](frontend/) | The single-page dashboard and search UI served at aidataboon.com. |
| [`.github/workflows/`](.github/workflows/) | Daily ingestion, the agents CI/CD pipeline and the search API deploy. |

Each of `data_platform/` and `agents/` has a `00_project_design.md` with its architecture and design decisions.

## Run it locally

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.14 and Docker.

```bash
uv sync

# Platform: the full local stack (Postgres/pgvector, MinIO, Airflow, Kafka, the RAG API, Prometheus, Grafana, ...)
cp data_platform/.env.example data_platform/.env   # optional settings
docker compose -f data_platform/01_local_stack.yml up -d

# Agents: includes the platform stack and adds the agent API on :8100
cp agents/.env.example agents/.env                 # optional settings
docker compose -f agents/01_agents_stack.yml up -d

# Evaluation gate (needs the RAG API on :8000, for example from the platform stack)
uv run python agents/03_eval_harness.py
```

Every module also runs on its own with built-in fallbacks (synthetic data, in-memory stores, no LLM), for example `uv run python data_platform/05_rag_service.py`.

## CI/CD

| Workflow | When it runs | What it does | Secrets |
|---|---|---|---|
| `ingest.yml` | Daily at 05:17 UTC, or by hand | Ingests HN, arXiv and GitHub into Bronze, builds Silver and Gold, then redeploys the search API so it indexes the new data | `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` (an R2 token with Object Read & Write on the `aidataboon-*` buckets), `RENDER_DEPLOY_HOOK_API` |
| `agents-cicd.yml` | Changes to `agents/` or the search service | Lint, type check, evaluation gate, image build, then deploy of the agent API | `RENDER_DEPLOY_HOOK_AGENTS` |
| `api-deploy.yml` | Changes to the search API image | Builds the image, smoke-tests it, then deploys it | `RENDER_DEPLOY_HOOK_API` |

## Deployment

- **Search API and agent API:** Render web services defined by `data_platform/render.yaml` and `agents/render.yaml`. Render's own auto-deploy is off; the workflows above deploy through each service's deploy hook. The search API's answers use `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` set on the service.
- **Database:** Neon Postgres with pgvector, set as `DATABASE_URL` on each service.
- **Frontend:** Cloudflare Pages, deployed with `npx wrangler pages deploy frontend --project-name aidataboon --branch main`.
- **Edge and DNS:** Cloudflare Terraform in `data_platform/09_cloudflare_infra.tf` and `agents/07_cloudflare_agents.tf`, each with its own state. Terraform state files are git-ignored.
