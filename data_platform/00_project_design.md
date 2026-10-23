# AIDataBoon — AI-Powered Data Infrastructure Platform
## Architecture & Design Document

**Domain:** aidataboon.com  
**Version:** 1.0  
**Last Updated:** 2026-06-09

---

## 1. Project Overview

AIDataBoon is a production-grade AI data infrastructure platform that demonstrates the full data engineering lifecycle: from raw API ingestion through a medallion lakehouse architecture to natural language search powered by retrieval-augmented generation (RAG). It continuously harvests signal from three high-value technical communities — Hacker News, arXiv research papers, and GitHub trending repositories — and makes that knowledge instantly searchable via plain English queries like "show me papers about LLM fine-tuning from this week" or "find trending Rust projects for systems programming."

The platform solves a real problem faced by engineers and researchers: the fragmentation of technical knowledge across multiple sources with no unified discovery layer. A data engineer monitoring the ML/AI space today must manually check HN, browse arXiv, scan GitHub trending, and reconcile duplicates across sources. AIDataBoon eliminates that friction by ingesting all three sources on a defined schedule, deduplicating and enriching the data through a medallion lakehouse (Bronze → Silver → Gold), extracting semantic embeddings, and serving them through a RAG API that lets users ask natural language questions with cited, ranked results.

As a portfolio project, AIDataBoon demonstrates mastery of the full modern data stack: Apache Airflow for orchestration, DuckDB + Delta Lake for lakehouse transforms, Apache Kafka + Flink for real-time streaming, pgvector for vector similarity search, sentence-transformers for local embedding generation, MLflow for experiment tracking, Feast for feature serving, FastAPI for the RAG API layer, and Prometheus/Grafana for observability — all deployable locally via Docker Compose and to production for under $15/month. The live demo runs at [aidataboon.com](https://aidataboon.com).

---

## 2. System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                           AIDataBoon System Architecture                               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                                        │
│                                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────────┐   │
│  │  Hacker News      │  │  arXiv Papers    │  │  GitHub Trending                 │   │
│  │  Algolia API      │  │  Atom/RSS API    │  │  Daily HTML scrape               │   │
│  │  points > 50      │  │  cat:cs.AI       │  │  github.com/trending?l=*         │   │
│  │  ~200 stories/day │  │  100 papers/day  │  │  ~25 repos/language/day          │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────┬───────────────────┘   │
└───────────┼──────────────────────┼───────────────────────────┼───────────────────────┘
            │                      │                           │
            ▼                      ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  INGESTION LAYER (Apache Airflow 2.9 — CeleryExecutor)                               │
│                                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                         │
│  │  hn_ingest DAG │  │ arxiv_ingest   │  │ github_ingest  │                         │
│  │  Schedule: @H  │  │ DAG            │  │ DAG            │                         │
│  │                │  │ Schedule: @D   │  │ Schedule: @D   │                         │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                         │
│          │                   │                   │                                   │
│          └───────────────────┴───────────────────┘                                   │
│                              │                                                       │
│                              ▼                                                       │
│                   ┌──────────────────────┐                                           │
│                   │  Raw JSON → Kafka    │   ◄── Also writes to Bronze directly      │
│                   │  topic: raw.events   │                                           │
│                   └──────────┬───────────┘                                           │
└──────────────────────────────┼──────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STREAMING LAYER (Apache Kafka + Flink)                                               │
│                                                                                      │
│  Kafka Topics (Confluent 7.6, single-broker dev / multi-broker prod):                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                       │
│  │ raw.hn.stories  │  │ raw.arxiv.papers│  │ raw.github.repos│                       │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                       │
│           │                   │                    │                                 │
│           └───────────────────┴────────────────────┘                                 │
│                               │                                                      │
│                    ┌──────────▼───────────┐                                          │
│                    │  Flink Stream Jobs   │  ← dedup, schema validation              │
│                    │  (Table API / SQL)   │  ← windowed aggregations                 │
│                    └──────────┬───────────┘                                          │
└───────────────────────────────┼──────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  MEDALLION LAKEHOUSE (DuckDB + Delta Lake via delta-rs + MinIO / Cloudflare R2)       │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │  BRONZE LAYER  (raw, append-only, schema-on-read)                               │ │
│  │  s3://aidataboon-bronze/hn/YYYY/MM/DD/                                            │ │
│  │  s3://aidataboon-bronze/arxiv/YYYY/MM/DD/                                         │ │
│  │  s3://aidataboon-bronze/github/YYYY/MM/DD/                                        │ │
│  │  Format: Delta Lake (Parquet + _delta_log/)    Retention: 90 days               │ │
│  └───────────────────────────────────┬─────────────────────────────────────────────┘ │
│                                      │  DuckDB SQL transforms                        │
│                                      ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │  SILVER LAYER  (cleaned, typed, deduplicated, unified schema)                   │ │
│  │  s3://aidataboon-silver/content/                                                  │ │
│  │  Unified schema: id, source, title, url, body, tags[], score, ts_published,     │ │
│  │                  ts_ingested, author, language, dedup_hash                      │ │
│  │  Format: Delta Lake    Partitioned by: source, date                             │ │
│  └───────────────────────────────────┬─────────────────────────────────────────────┘ │
│                                      │  DuckDB SQL aggregations                      │
│                                      ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │  GOLD LAYER  (business-ready, embedding-ready, feature-ready)                   │ │
│  │  s3://aidataboon-gold/trending_topics/   (daily aggregates, top-N by source)      │ │
│  │  s3://aidataboon-gold/content_features/  (enriched + embedding vectors)           │ │
│  │  s3://aidataboon-gold/author_profiles/   (author activity over time)              │ │
│  │  Format: Delta Lake + materialized DuckDB views                                 │ │
│  └───────────────────────────────────┬─────────────────────────────────────────────┘ │
└───────────────────────────────────────┼──────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
┌─────────────────────┐    ┌────────────────────────┐   ┌────────────────────────────┐
│  FEATURE STORE       │    │  VECTOR INDEX           │   │  ML PLATFORM               │
│  (Feast)             │    │  (pgvector + HNSW)      │   │  (MLflow)                  │
│                      │    │                         │   │                            │
│  feast-server:6566   │    │  PostgreSQL 16 + pgvec  │   │  mlflow:5000               │
│  Features:           │    │  Table: embeddings      │   │  Artifacts: MinIO          │
│  - trending_score    │    │  Dim: 384 (MiniLM-L6)   │   │  Models: sentence-trf      │
│  - freshness_score   │    │  Index: HNSW             │   │  Experiments: per-source   │
│  - source_credibility│    │  Distance: cosine        │   │                            │
│  - tech_category     │    │  ~500K vectors at scale  │   │                            │
└─────────┬───────────┘    └───────────┬─────────────┘   └────────────────────────────┘
          │                            │
          └──────────────┬─────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  RAG API LAYER (FastAPI + LangChain)                                                 │
│                                                                                      │
│  rag-api:8000                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  POST /search                                                                │   │
│  │    1. Embed query with all-MiniLM-L6-v2 (local, free)                       │   │
│  │    2. ANN search in pgvector (top-K=20)                                      │   │
│  │    3. Re-rank with cross-encoder (ms-marco-MiniLM-L-6-v2)                   │   │
│  │    4. Fetch feature scores from Feast                                        │   │
│  │    5. Compose context + generate answer (Ollama/local or OpenAI fallback)   │   │
│  │    6. Return: answer + cited sources + relevance scores                      │   │
│  │                                                                              │   │
│  │  GET  /trending          — top items by source, last 24h                    │   │
│  │  GET  /topics            — clustered topic summary                           │   │
│  │  GET  /health            — readiness + liveness                              │   │
│  │  GET  /metrics           — Prometheus exposition format                      │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
┌─────────────────────┐       ┌────────────────────────┐   ┌──────────────────────────┐
│  FRONTEND            │       │  OBSERVABILITY          │   │  CLOUDFLARE EDGE          │
│  Cloudflare Pages    │       │                         │   │                          │
│                      │       │  Prometheus:9090        │   │  aidataboon.com          │
│  aidataboon.com      │       │  Grafana:3000           │   │    → CF Pages            │
│  Next.js / static    │       │  Jaeger:16686           │   │  api.aidataboon.com      │
│  Search UI           │       │  OTel Collector:4317    │   │    → CF Worker           │
│  Trending dashboard  │       │    gRPC + HTTP          │   │    → Render backend      │
│                      │       │                         │   │  dashboard.aidataboon    │
│  CF R2 for assets    │       │  Dashboards:            │   │    → Grafana tunnel      │
└─────────────────────┘       │  - Pipeline health      │   └──────────────────────────┘
                               │  - RAG latency p50/p99  │
                               │  - Embedding throughput │
                               │  - Source freshness     │
                               │  - Vector index growth  │
                               └────────────────────────┘
```

---

## 3. Architecture Decision Records (ADRs)

### ADR-001: DuckDB Over Apache Spark for Medallion Transforms

**Status:** Accepted  
**Date:** 2026-06-09

**Context:**  
The medallion lakehouse requires SQL-based transforms to move data from Bronze → Silver → Gold. Options considered: Apache Spark (PySpark), DuckDB, Polars, and DuckDB over Delta Lake via delta-rs.

**Decision:**  
Use DuckDB as the primary query engine for all lakehouse transforms, reading and writing Delta Lake tables via the `delta-rs` Python library.

**Rationale:**
- **Cost:** DuckDB is free and runs in-process. A Spark cluster on EMR or Databricks costs $0.10–$2.00/DBU even for small workloads. For AIDataBoon's data volumes (10K–100K rows/day), DuckDB handles transforms in seconds on a single node.
- **Simplicity:** No cluster management, no JVM, no YARN/YARN configuration. Install `duckdb` and `deltalake` Python packages. That is the entire setup.
- **Performance:** DuckDB's vectorized columnar engine matches or exceeds Spark for datasets under ~100GB on a single machine. It reads Parquet natively from MinIO/S3 with `httpfs` extension.
- **Delta Lake compatibility:** DuckDB can read Delta tables via `delta_scan()`. Writes use the `deltalake` Python library (delta-rs, Rust-backed) which supports ACID transactions, schema enforcement, and time travel.

**Consequences:**
- Positive: Zero cluster cost, sub-second query startup, simple Python integration.
- Negative: Cannot scale horizontally. If data volumes exceed ~500GB, migration to Spark or Daft would be required.
- Mitigation: The Airflow DAG structure and Delta Lake table contracts are Spark-compatible. Migration path exists if needed.

---

### ADR-002: pgvector Over Pinecone / Qdrant for Vector Search

**Status:** Accepted  
**Date:** 2026-06-09

**Context:**  
The RAG API requires ANN (approximate nearest neighbor) search over ~100K–500K embedding vectors of dimensionality 384. Options evaluated: Pinecone (managed SaaS), Qdrant (self-hosted or managed), Weaviate, Milvus, and pgvector (PostgreSQL extension).

**Decision:**  
Use pgvector with HNSW indexing on PostgreSQL 16, hosted on Neon.tech free tier in production.

**Rationale:**
- **Cost:** Pinecone starter plan is $70/month for 1M vectors. Qdrant Cloud free tier limits to 1GB RAM. Neon.tech free tier is genuinely free with 0.5 GB storage (sufficient for 500K × 384-dim float32 vectors ≈ 768 MB — marginal, use float16 or PQ compression for free tier). Self-hosted pgvector on Render costs $0/month on the free instance class (512MB RAM, sleeps when idle).
- **Operational simplicity:** PostgreSQL is already in the stack for Airflow metadata, MLflow backend, and application data. pgvector reuses the same connection pool, same backup strategy, same operational knowledge. Adding Pinecone adds a new vendor, a new API key, a new failure domain.
- **HNSW quality:** pgvector 0.7+ ships HNSW indexing with `m=16, ef_construction=64` defaults. For 384-dim vectors, recall@10 exceeds 0.97 at <10ms latency on modest hardware.
- **No vendor lock-in:** The embedding table is plain PostgreSQL. Migration to another vector DB is a simple `pg_dump` + reimport.

**Consequences:**
- Positive: $0 incremental infrastructure cost, unified operational surface.
- Negative: PostgreSQL is not purpose-built for vector search; at 10M+ vectors, purpose-built ANN engines (Qdrant, Milvus) outperform pgvector. 
- Mitigation: AIDataBoon's scale (100K–500K vectors) is well within pgvector's sweet spot. Documented upgrade path to Qdrant if needed.

---

### ADR-003: Cloudflare R2 Over AWS S3 for Object Storage

**Status:** Accepted  
**Date:** 2026-06-09

**Context:**  
Delta Lake tables, MLflow artifacts, and static frontend assets require durable object storage. Primary candidates: AWS S3, Google Cloud Storage, MinIO (self-hosted), and Cloudflare R2.

**Decision:**  
Use Cloudflare R2 in production (with MinIO locally for development parity).

**Rationale:**
- **Egress cost:** AWS S3 charges $0.09/GB for data egress. Cloudflare R2 charges $0.00/GB for egress. For a data platform that reads Delta Lake files for every transform run, egress costs on S3 accumulate rapidly. R2 eliminates this entirely.
- **Domain integration:** aidataboon.com is already registered and managed in Cloudflare. R2 buckets can be exposed via custom domains (e.g., `data.aidataboon.com`) using Cloudflare's native routing — no CORS headaches, no IAM configuration for CDN.
- **S3 compatibility:** R2 is fully S3-compatible. DuckDB's `httpfs` extension, the `deltalake` Python library (delta-rs), and MLflow all support S3-compatible endpoints via `endpoint_url`. Switching from MinIO (local) to R2 (prod) requires only changing three environment variables.
- **Cost:** R2 free tier includes 10 GB storage, 1M Class A operations/month, 10M Class B operations/month. AIDataBoon's production usage fits within free tier for the first 6–12 months.

**Consequences:**
- Positive: Zero egress fees, seamless Cloudflare domain integration, S3-compatible.
- Negative: No native Lifecycle rules as mature as S3 (workaround: Airflow DAG to delete old Bronze partitions). R2 is not available in all regions (EU compliance may require S3).

---

### ADR-004: Apache Airflow Over Prefect / Dagster for Orchestration

**Status:** Accepted  
**Date:** 2026-06-09

**Context:**  
The ingestion and transform pipelines require a scheduler/orchestrator. Candidates: Apache Airflow, Prefect 2.x, Dagster, and simple cron + Python scripts.

**Decision:**  
Use Apache Airflow 2.9 with CeleryExecutor.

**Rationale:**
- **Industry prevalence:** Airflow is the most widely deployed orchestration platform in enterprise data engineering. Demonstrating Airflow proficiency has higher signal value in a portfolio than Prefect, which is less common in large organizations.
- **Cost:** Airflow is fully open-source. Prefect Cloud free tier is limited to 3 deployments. Dagster Cloud free tier limits runs. Self-hosted Airflow has no usage limits.
- **Ecosystem:** Airflow has providers for every data source and destination in this stack: `apache-airflow-providers-http`, `apache-airflow-providers-amazon` (for S3/R2), `apache-airflow-providers-postgres`, `apache-airflow-providers-apache-kafka`.
- **CeleryExecutor:** Enables distributed task execution across multiple workers. Provides the basis for demonstrating horizontal scaling patterns even in the local Docker Compose setup.

**Consequences:**
- Positive: Maximum portfolio signal, no usage limits, broad provider ecosystem.
- Negative: Airflow has heavier resource requirements than Prefect (requires Redis + Postgres for Celery). In the Docker Compose setup, this adds ~500MB RAM overhead.

---

### ADR-005: sentence-transformers (all-MiniLM-L6-v2) Over OpenAI Embeddings

**Status:** Accepted  
**Date:** 2026-06-09

**Context:**  
Generating embeddings for 100K+ documents at ingestion time and for every search query requires a reliable, cost-effective embedding model. Candidates: OpenAI text-embedding-3-small ($0.02/1M tokens), Cohere Embed ($0.10/1M tokens), and local models via sentence-transformers.

**Decision:**  
Use `sentence-transformers` library with the `all-MiniLM-L6-v2` model for all embedding generation.

**Rationale:**
- **Cost:** OpenAI embeddings at $0.02/1M tokens sounds cheap, but at 100K documents × 512 tokens average = 51.2M tokens = ~$1.02 per full re-embed. Re-embedding on model upgrades or schema changes adds up. Local inference is $0.00.
- **Latency:** `all-MiniLM-L6-v2` runs inference in ~2ms/document on CPU (batch size 32), ~0.3ms/document on GPU. OpenAI API adds 50–200ms network round-trip per batch request.
- **Privacy:** Documents ingested from HN/arXiv/GitHub may contain sensitive commentary. Keeping embeddings local avoids sending data to third-party APIs.
- **Quality:** `all-MiniLM-L6-v2` (384-dim) achieves 0.7897 on MTEB STS Benchmark — sufficient for technical document search. The model size is 22MB and fits in RAM.
- **Portability:** No API key required. The Docker image bundles the model weights. The system works offline.

**Consequences:**
- Positive: Zero embedding cost, offline capability, no API key dependency.
- Negative: 384-dim embeddings have lower expressiveness than OpenAI's 1536-dim text-embedding-3-small. For highly semantic queries, quality may be marginally lower.
- Mitigation: Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) compensates for bi-encoder limitations. Optional OpenAI fallback configurable via `EMBEDDING_PROVIDER=openai` environment variable.

---

### ADR-006: delta-rs Python Library Over PySpark for Delta Lake Writes

**Status:** Accepted  
**Date:** 2026-06-09

**Context:**  
Writing ACID-compliant Delta Lake tables from Python without Apache Spark. Candidates: PySpark + Delta Lake JAR, delta-rs (Rust implementation with Python bindings), and plain Parquet without ACID guarantees.

**Decision:**  
Use the `deltalake` Python package (backed by delta-rs, written in Rust) for all Delta Lake read/write operations.

**Rationale:**
- **No JVM overhead:** PySpark requires a JVM, which adds 1–2 GB RAM overhead and 10–30 second startup time for each transform task. delta-rs is a native Rust library with Python bindings — startup is instantaneous, memory overhead is minimal.
- **Feature completeness:** delta-rs 0.17+ supports ACID writes, schema evolution, time travel (version queries), Z-ordering, and OPTIMIZE compaction. All features needed for Bronze/Silver/Gold transforms are available.
- **DuckDB integration:** DuckDB 0.10+ includes a `delta` extension that reads Delta tables via delta-rs internals. DuckDB + delta-rs share the same underlying Rust library, ensuring compatibility.
- **Cost:** delta-rs is Apache 2.0 licensed, open source, no licensing fees.

**Consequences:**
- Positive: No JVM, fast startup, DuckDB-compatible, full ACID support.
- Negative: delta-rs does not support all Spark Delta Lake features (e.g., Change Data Feed in older versions, some Delta Sharing features). Spark is required for very large-scale merges (>100GB).

---

## 4. Cost Analysis

### Local Development

| Component | Tool | Cost/Month |
|-----------|------|-----------|
| All services | Docker Compose (local laptop) | $0.00 |
| Object storage | MinIO (containerized) | $0.00 |
| Database | PostgreSQL (containerized) | $0.00 |
| Kafka | Confluent containers | $0.00 |
| Embeddings | sentence-transformers CPU | $0.00 |
| **Total** | | **$0.00** |

### Production — Minimal (~$0/month, no credit card required)

| Component | Tool | Cost/Month |
|-----------|------|-----------|
| Compute (API + workers) | Render free web services (sleep on idle) | $0.00 |
| Object storage | Cloudflare R2 (10GB free) | $0.00 |
| Vector + relational DB | Neon.tech free tier (PostgreSQL + pgvector) | $0.00 |
| Kafka (managed) | Upstash Kafka free tier (10K msg/day) | $0.00 |
| Frontend hosting | Cloudflare Pages (free) | $0.00 |
| DNS + CDN | Cloudflare free plan | $0.00 |
| ML tracking | MLflow on Render (shared with API) | $0.00 |
| Monitoring | Grafana Cloud free (10K metrics/month) | $0.00 |
| Airflow | Render free web service (sleeps on idle) | $0.00 |
| **Total** | | **~$0.00** |

### Production — Full AWS Alternative

| Component | Tool | Cost/Month |
|-----------|------|-----------|
| Compute (ECS Fargate) | 2 vCPU × 4GB tasks × 2 | $35.00 |
| Object storage | AWS S3 (50GB + egress 20GB) | $8.00 |
| Relational DB | AWS RDS PostgreSQL t3.micro | $15.00 |
| Kafka (managed) | AWS MSK smallest broker | $45.00 |
| Frontend | AWS CloudFront + S3 | $3.00 |
| Monitoring | AWS CloudWatch | $10.00 |
| Airflow | MWAA (managed) — minimum | $45.00 |
| **Total** | | **~$161.00** |

**Cost savings by choosing the minimal stack: ~$149/month (~92% reduction)**

---

## 5. Technology Stack

| Component | Technology | Version | Rationale | Monthly Cost |
|-----------|-----------|---------|-----------|-------------|
| **Orchestration** | Apache Airflow | 2.9 | Industry standard, CeleryExecutor, rich providers | $0 (self-hosted) |
| **Object Storage (local)** | MinIO | RELEASE.2024-01-01 | S3-compatible, zero config, Docker-native | $0 |
| **Object Storage (prod)** | Cloudflare R2 | — | Zero egress fees, CF domain integration | $0 free tier |
| **Table Format** | Delta Lake (delta-rs) | 0.17+ | ACID, time travel, DuckDB-compatible, no JVM | $0 (OSS) |
| **Query Engine** | DuckDB | 0.10+ | In-process, vectorized, reads Delta/Parquet/S3 | $0 (OSS) |
| **Relational DB** | PostgreSQL 16 | 16.x | Airflow meta, MLflow backend, pgvector host | $0 free tier |
| **Vector Search** | pgvector (HNSW) | 0.7+ | In-Postgres, HNSW, no extra vendor | $0 free tier |
| **Streaming Broker** | Apache Kafka | 7.6 (Confluent) | Durable log, exactly-once semantics | $0 (self-hosted) |
| **Stream Processing** | Apache Flink | 1.18 | Stateful stream processing, SQL interface | $0 (OSS) |
| **Schema Registry** | Confluent Schema Registry | 7.6 | Avro/JSON Schema enforcement on Kafka topics | $0 (OSS) |
| **Embedding Model** | sentence-transformers | 2.x | all-MiniLM-L6-v2, 384-dim, 22MB, CPU-capable | $0 (local) |
| **Cross-Encoder** | sentence-transformers | 2.x | ms-marco-MiniLM-L-6-v2, re-ranking | $0 (local) |
| **LLM (answer gen)** | Ollama (local) / OpenAI | — | Local-first, OpenAI fallback configurable | $0 / pay-per-use |
| **ML Tracking** | MLflow | 2.x | Experiment tracking, model registry, artifact store | $0 (OSS) |
| **Feature Store** | Feast | 0.36+ | Online + offline feature serving, DuckDB offline store | $0 (OSS) |
| **API Framework** | FastAPI | 0.110+ | Async, OpenAPI docs, Pydantic validation, fast | $0 (OSS) |
| **Metrics** | Prometheus | 2.x | Pull-based metrics, standard exporter ecosystem | $0 (OSS) |
| **Dashboards** | Grafana | 10.x | Prometheus + Loki datasources, alerting | $0 (OSS) |
| **Tracing** | OpenTelemetry + Jaeger | OTEL 0.96 | Distributed traces, W3C trace context | $0 (OSS) |
| **IaC** | Terraform | 1.8+ | Cloudflare DNS, Render resources, Neon DB | $0 (OSS) |
| **CDN / Edge** | Cloudflare Pages + Workers | — | Global CDN, domain management | $0 free plan |
| **CI/CD** | GitHub Actions | — | Free for public repos, matrix builds, OIDC deploy | $0 public repo |
| **Container Runtime** | Docker + Docker Compose | 27.x + 2.x | Local dev, reproducible environments | $0 (OSS) |
| **Language** | Python | 3.11 | Universal data ecosystem compatibility | $0 |

---

## 6. Data Sources

### 6.1 Hacker News — Algolia Search API

**Endpoint:**
```
https://hn.algolia.com/api/v1/search?tags=story&numericFilters=points>50&hitsPerPage=100
```

**Schedule:** Hourly (`@hourly` Airflow DAG)  
**Volume:** ~50–200 stories/hour during peak

**Key fields extracted:**
```json
{
  "objectID": "39823456",
  "title": "Show HN: I built a distributed ...",
  "url": "https://github.com/...",
  "author": "throwaway_dev",
  "points": 247,
  "num_comments": 89,
  "created_at": "2026-06-09T14:23:01.000Z",
  "created_at_i": 1749477781,
  "_tags": ["story", "ask_hn", "front_page"]
}
```

**Additional endpoints:**
- Comments: `https://hn.algolia.com/api/v1/items/{id}` (fetch top comments for context)
- By date: `https://hn.algolia.com/api/v1/search_by_date?tags=story&numericFilters=created_at_i>1749477781`

**Rate limits:** No authentication required. Algolia asks for reasonable use (<1 req/sec). Airflow task uses 500ms delay between pages.

---

### 6.2 arXiv — Atom API

**Endpoint:**
```
http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL&max_results=100&sortBy=submittedDate&sortOrder=descending
```

**Schedule:** Daily at 06:00 UTC (`0 6 * * *` Airflow DAG)  
**Volume:** ~100–500 new papers/day in cs.AI + cs.LG + cs.CL

**Categories ingested:**
- `cs.AI` — Artificial Intelligence
- `cs.LG` — Machine Learning
- `cs.CL` — Computation and Language (NLP)
- `cs.DB` — Databases (for data infrastructure papers)
- `cs.DC` — Distributed Computing

**Key fields extracted from Atom XML:**
```python
{
    "arxiv_id": "2406.12345",
    "title": "Scaling Laws for Neural Language Models",
    "abstract": "We study empirical scaling laws...",
    "authors": ["Jared Kaplan", "Sam McCandlish"],
    "categories": ["cs.LG", "cs.AI"],
    "submitted": "2026-06-09T00:00:00Z",
    "updated": "2026-06-09T12:00:00Z",
    "pdf_url": "https://arxiv.org/pdf/2406.12345",
    "doi": "10.48550/arXiv.2406.12345"
}
```

**Rate limits:** arXiv requests ≤3 requests/second per their Terms of Use. Airflow task uses exponential backoff.

---

### 6.3 GitHub Trending — Daily Scrape

**URL pattern:**
```
https://github.com/trending?since=daily
https://github.com/trending?since=daily&spoken_language_code=en
https://github.com/trending/{language}?since=daily   # per language
```

**Schedule:** Daily at 08:00 UTC  
**Languages scraped:** Python, Rust, Go, TypeScript, JavaScript, Java, C++, Julia

**Extraction approach:** HTTP GET + BeautifulSoup HTML parsing (GitHub has no official trending API).

```python
{
    "repo_name": "owner/repository",
    "description": "A blazingly fast ...",
    "language": "Rust",
    "stars_total": 12847,
    "stars_today": 423,
    "forks": 891,
    "url": "https://github.com/owner/repository",
    "topics": ["cli", "performance", "systems"],
    "contributors": ["user1", "user2", "user3"],
    "trending_date": "2026-06-09",
    "trending_language": "rust"
}
```

**Rate limits:** GitHub rate-limits unauthenticated requests to 60/hour. The Airflow DAG uses `GITHUB_TOKEN` (optional) for 5000/hour. Scraping runs once daily — well within limits.

---

## 7. Domain Setup — aidataboon.com

### 7.1 DNS Architecture

All DNS is managed via Cloudflare. The Terraform configuration in `10_iac/cloudflare/main.tf` provisions the following records:

```
aidataboon.com          CNAME  → Cloudflare Pages (static frontend)
api.aidataboon.com      CNAME  → Cloudflare Worker (routes to Render)
dashboard.aidataboon.com CNAME → Cloudflare Tunnel → Grafana:3000
data.aidataboon.com     CNAME  → Cloudflare R2 public bucket
```

### 7.2 Cloudflare Terraform Configuration

```hcl
# 10_iac/cloudflare/main.tf

terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

variable "cloudflare_zone_id"    { type = string }
variable "cloudflare_api_token"  { type = string }

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# Root domain → Cloudflare Pages
resource "cloudflare_record" "root" {
  zone_id = var.cloudflare_zone_id
  name    = "@"
  value   = "aidataboon.pages.dev"
  type    = "CNAME"
  proxied = true
}

# API subdomain → Cloudflare Worker → Render
resource "cloudflare_record" "api" {
  zone_id = var.cloudflare_zone_id
  name    = "api"
  value   = "aidataboon-api.onrender.com"
  type    = "CNAME"
  proxied = true
}

# Grafana dashboard → Cloudflare Tunnel
resource "cloudflare_record" "dashboard" {
  zone_id = var.cloudflare_zone_id
  name    = "dashboard"
  value   = "${cloudflare_tunnel.grafana.cname}"
  type    = "CNAME"
  proxied = true
}
```

### 7.3 Cloudflare Worker (API Proxy)

The Worker at `api.aidataboon.com` handles:
- Rate limiting (10 req/s per IP)
- Authentication header injection (API key from CF secret → Render)
- CORS headers for browser requests from `aidataboon.com`
- Request routing: `/v1/*` → Render RAG API, `/health` → direct response

### 7.4 Cloudflare Pages (Frontend)

The Next.js frontend is deployed to Cloudflare Pages via GitHub Actions:
```yaml
# .github/workflows/deploy-frontend.yml
- uses: cloudflare/pages-action@v1
  with:
    apiToken: ${{ secrets.CF_API_TOKEN }}
    accountId: ${{ secrets.CF_ACCOUNT_ID }}
    projectName: aidataboon
    directory: frontend/out
    gitHubToken: ${{ secrets.GITHUB_TOKEN }}
```

---

## 8. Getting Started

### Prerequisites

```bash
# Required
docker >= 27.0
docker compose >= 2.24  # v2 plugin (not docker-compose v1)
python >= 3.11
git >= 2.40

# Optional (for production deploy)
terraform >= 1.8
render-cli (optional, for Render Blueprint sync)
cloudflare wrangler >= 3.0
```

### Clone and Configure

```bash
git clone https://github.com/yourusername/aidataboon.git
cd aidataboon/data_platform

# Copy environment template
cp .env.example .env

# Edit secrets (MinIO, Postgres passwords, optional API keys)
nano .env
```

### Environment Variables

```bash
# .env — minimum required for local development
POSTGRES_PASSWORD=aidataboon_secret_2024
MINIO_ROOT_PASSWORD=minioadmin123_secure
AIRFLOW_FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
AIRFLOW_SECRET_KEY=$(openssl rand -hex 32)

# Optional: GitHub token for higher trending scrape rate limit
GITHUB_TOKEN=ghp_your_token_here

# Optional: OpenAI fallback for answer generation
OPENAI_API_KEY=sk-...
```

### Start Local Stack

```bash
# Start all services (first run downloads ~8GB of images)
docker compose -f 01_local_stack.yml up -d

# Wait for all services to be healthy (~3-5 minutes)
docker compose -f 01_local_stack.yml ps

# Watch health status
watch -n 5 'docker compose -f 01_local_stack.yml ps --format "table {{.Name}}\t{{.Status}}"'
```

### Service URLs (after startup)

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| MLflow UI | http://localhost:5000 | — |
| Jaeger UI | http://localhost:16686 | — |
| Prometheus | http://localhost:9090 | — |
| RAG API docs | http://localhost:8000/docs | — |
| Feast server | http://localhost:6566 | — |
| Kafka UI | — | Use Airflow or kafka-ui |
| Schema Registry | http://localhost:8081 | — |

### First Pipeline Run

```bash
# Create MinIO buckets (one-time setup)
docker compose -f 01_local_stack.yml exec minio \
  mc alias set local http://localhost:9000 minioadmin minioadmin123 && \
  mc mb local/aidataboon-bronze local/aidataboon-silver local/aidataboon-gold local/mlflow-artifacts

# Initialize pgvector extension and embedding table
docker compose -f 01_local_stack.yml exec postgres \
  psql -U aidataboon -d aidataboon -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Trigger HN ingestion manually
docker compose -f 01_local_stack.yml exec airflow-scheduler \
  airflow dags trigger hn_ingest_hourly

# Monitor pipeline
open http://localhost:8080  # Airflow UI
```

### Validate RAG API

```bash
# Health check
curl http://localhost:8000/health

# Natural language search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "LLM fine-tuning techniques 2024", "limit": 5}'

# Trending items
curl http://localhost:8000/trending?source=hackernews&hours=24
```

### Shut Down

```bash
# Stop all services (preserves volumes)
docker compose -f 01_local_stack.yml down

# Stop and remove all data (full reset)
docker compose -f 01_local_stack.yml down -v --remove-orphans
```

---

## 9. Project Structure

```
data_platform/
├── 00_project_design.md          ← This document
├── 01_local_stack.yml            ← Docker Compose (all 18 services)
├── .env.example                  ← Environment template
├── 02_ingestion/
│   ├── dags/
│   │   ├── hn_ingest_hourly.py
│   │   ├── arxiv_ingest_daily.py
│   │   └── github_trending_daily.py
│   └── operators/
│       ├── hn_operator.py
│       └── arxiv_operator.py
├── 03_lakehouse/
│   ├── dags/
│   │   ├── bronze_to_silver.py
│   │   └── silver_to_gold.py
│   └── transforms/
│       ├── silver_transform.sql
│       └── gold_aggregations.sql
├── 04_streaming/
│   ├── producers/
│   │   └── hn_kafka_producer.py
│   └── flink_jobs/
│       └── dedup_and_score.py
├── 05_rag_api/
│   ├── Dockerfile.rag
│   ├── main.py
│   ├── embeddings.py
│   ├── retrieval.py
│   └── requirements.txt
├── 06_ml_platform/
│   ├── feast/
│   │   └── feature_store.yaml
│   └── mlflow/
│       └── train_embeddings.py
├── 07_observability/
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   └── datasources/
│   │       └── datasources.yml
│   └── otel/
│       └── otel-collector-config.yml
├── 08_ci_cd/
│   └── .github/workflows/
│       ├── ci.yml
│       └── deploy.yml
├── 09_frontend/
│   └── (Next.js app)
└── 10_iac/
    └── cloudflare/
        └── main.tf
```

---

## 10. Key Metrics and SLOs

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| HN ingestion lag | < 5 minutes | > 30 minutes |
| arXiv ingestion lag | < 1 hour | > 4 hours |
| Bronze→Silver transform time | < 2 minutes | > 10 minutes |
| Silver→Gold transform time | < 5 minutes | > 20 minutes |
| Embedding batch throughput | > 500 docs/minute | < 100 docs/minute |
| RAG API p50 latency | < 200ms | > 500ms |
| RAG API p99 latency | < 2000ms | > 5000ms |
| Vector index freshness | < 30 minutes behind Gold | > 2 hours |
| Kafka consumer lag | < 1000 messages | > 10000 messages |
| Pipeline success rate | > 99% | < 95% |

---

*AIDataBoon — AI Data Infrastructure Portfolio Project*  
*Domain: aidataboon.com | Stack: Airflow + DuckDB + Delta Lake + Kafka + pgvector + FastAPI*
