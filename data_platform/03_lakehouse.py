"""
AIDataBoon — Medallion Lakehouse (Bronze → Silver → Gold)

Stack: DuckDB (query engine) + delta-rs (Delta Lake Python bindings)
Storage: MinIO locally / Cloudflare R2 in production

  Bronze  s3://aidataboon-bronze/{hn,arxiv,github-trending}/year=/month=/day=/...
          → raw NDJSON landed by 02_data_ingestion.py, as-is
          → s3://aidataboon-bronze-delta/{table}/        (Delta, partitioned by _source)

  Silver  s3://aidataboon-silver/content/
          → unified schema across all three sources, deduplicated, quality-gated

  Gold    s3://aidataboon-gold/{knowledge_base,feature_table,trending_topics,
                               daily_summary,author_profiles}/
          → also mirrored to ./data/gold/*.parquet for Feast + MLflow (06_ml_platform.py)

Run standalone (no MinIO required — falls back to synthetic Bronze + local Parquet):
    uv run python data_platform/03_lakehouse.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aidataboon.lakehouse")

# ---------------------------------------------------------------------------
# Optional delta-rs — gracefully degrade to plain Parquet if not installed.
# In production, `pip install deltalake` unlocks MERGE, time travel, vacuum,
# schema evolution and ACID multi-writer guarantees on top of S3/R2/MinIO.
# ---------------------------------------------------------------------------
try:
    from deltalake import DeltaTable, write_deltalake
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False
    log.warning("deltalake not installed — falling back to Parquet (no MERGE/time-travel)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LAKEHOUSE ENGINE (DuckDB + S3/MinIO/R2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LakehouseConfig:
    """S3-compatible storage config. Same shape works for MinIO, AWS S3, and
    Cloudflare R2 — only the endpoint/credentials change between environments.

    Defaults target the local MinIO stack; set S3_ENDPOINT_URL / S3_ACCESS_KEY /
    S3_SECRET_KEY (same names used by 02_data_ingestion.py and 05_rag_service.py)
    to point this at production Cloudflare R2 instead."""
    s3_endpoint: str = field(default_factory=lambda: os.environ.get("S3_ENDPOINT_URL", "http://minio:9000"))
    s3_access_key: str = field(default_factory=lambda: os.environ.get("S3_ACCESS_KEY", "minioadmin"))
    s3_secret_key: str = field(default_factory=lambda: os.environ.get("S3_SECRET_KEY", "minioadmin123"))
    s3_region: str = field(default_factory=lambda: os.environ.get("S3_REGION", "us-east-1"))
    s3_use_ssl: bool = field(default_factory=lambda: os.environ.get("S3_ENDPOINT_URL", "http://minio:9000").startswith("https://"))

    bronze_bucket: str = field(default_factory=lambda: os.environ.get("S3_BRONZE_BUCKET", "aidataboon-bronze"))
    bronze_delta_bucket: str = "aidataboon-bronze-delta"
    silver_bucket: str = field(default_factory=lambda: os.environ.get("S3_SILVER_BUCKET", "aidataboon-silver"))
    gold_bucket: str = field(default_factory=lambda: os.environ.get("S3_GOLD_BUCKET", "aidataboon-gold"))

    # Local mirror of Gold tables — read by 06_ml_platform.py / Feast offline store
    local_gold_dir: str = "data/gold"


class LakehouseEngine:
    """Thin DuckDB wrapper that wires up S3/MinIO access and Delta I/O."""

    def __init__(self, config: LakehouseConfig | None = None) -> None:
        self.config = config or LakehouseConfig()
        self.con = duckdb.connect(":memory:")
        self.s3_available = self._configure_s3()

    def _configure_s3(self) -> bool:
        """Register an S3 secret so `s3://` paths work transparently in SQL.
        Returns False (and disables all `s3://` reads) if the `httpfs`
        extension can't be installed — e.g. offline / air-gapped sandboxes."""
        try:
            self.con.execute("INSTALL httpfs")
            self.con.execute("LOAD httpfs")
            self.con.execute(
                f"""
                CREATE OR REPLACE SECRET aidataboon_s3 (
                    TYPE S3,
                    KEY_ID '{self.config.s3_access_key}',
                    SECRET '{self.config.s3_secret_key}',
                    ENDPOINT '{self.config.s3_endpoint.split('://')[-1]}',
                    REGION '{self.config.s3_region}',
                    URL_STYLE 'path',
                    USE_SSL {str(self.config.s3_use_ssl).lower()}
                )
                """
            )
            log.info("DuckDB S3 secret configured for endpoint %s", self.config.s3_endpoint)
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            log.warning("S3 configuration skipped (%s) — local-only mode", exc)
            return False

    def list_bronze_keys(self, prefix: str) -> list[str]:
        """List object keys under `prefix` in the Bronze bucket via boto3
        (S3 ListObjectsV2 works fine on R2; it's only DuckDB's httpfs glob
        that probes the literal prefix with a GET and 404s)."""
        if not self.s3_available:
            return []
        try:
            import boto3
            from botocore.config import Config as BotoConfig

            client = boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint,
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key,
                region_name=self.config.s3_region,
                # R2 doesn't support virtual-hosted-style addressing and
                # returns a misleading NoSuchKey for ListObjectsV2 otherwise.
                config=BotoConfig(s3={"addressing_style": "path"}),
            )
            paginator = client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self.config.bronze_bucket, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith(".ndjson"))
            return keys
        except Exception as exc:
            log.warning("Bronze listing failed for %s (%s)", prefix, exc)
            return []

    def _storage_options(self) -> dict[str, str]:
        """delta-rs / object_store storage options — same creds as DuckDB secret."""
        return {
            "AWS_ENDPOINT_URL": self.config.s3_endpoint,
            "AWS_ACCESS_KEY_ID": self.config.s3_access_key,
            "AWS_SECRET_ACCESS_KEY": self.config.s3_secret_key,
            "AWS_REGION": self.config.s3_region,
            "AWS_ALLOW_HTTP": "true" if not self.config.s3_use_ssl else "false",
            "AWS_S3_ALLOW_UNSAFE_RENAME": "true",  # required for MinIO (no copy-if-not-exists)
        }

    def sql(self, query: str, **relations: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        """Run `query`, registering each kwarg as a queryable view first.

        DuckDB's `FROM <python_var>` replacement scan only inspects the
        *immediate* calling frame — which here would be this wrapper's
        frame (locals: `self`, `query`), not the caller's. Passing
        relations explicitly as kwargs sidesteps that entirely.

        The result is materialized eagerly (Arrow) before returning.
        Connection-wide registrations are last-write-wins on a *lazy*
        relation — without eager materialization, an earlier call's
        relation would silently re-resolve its alias against whatever a
        later call registered under the same name by the time it's
        finally executed (e.g. when union'd together)."""
        for alias, rel in relations.items():
            self.con.register(alias, rel)
        table = self.con.sql(query).to_arrow_table()
        for alias in relations:
            self.con.unregister(alias)
        return self.con.from_arrow(table)

    def read_ndjson(self, s3_glob: str | list[str]) -> duckdb.DuckDBPyRelation | None:
        """Read newline-delimited JSON from S3/MinIO, tolerating schema drift
        across files via `union_by_name`. Returns None if nothing matches
        (or if S3 is unreachable, e.g. local sandbox with no httpfs).

        `s3_glob` may be a glob pattern or an explicit list of `s3://` URIs —
        prefer an explicit list on R2, since DuckDB's httpfs glob support
        probes the literal pre-wildcard prefix with a GET and 404s there
        instead of listing."""
        if isinstance(s3_glob, list):
            if not s3_glob:
                return None
            path_arg = "[" + ", ".join(f"'{p}'" for p in s3_glob) + "]"
            log_path = s3_glob[0]
        else:
            path_arg = f"'{s3_glob}'"
            log_path = s3_glob
        if log_path.startswith("s3://") and not self.s3_available:
            log.warning("S3 unavailable — skipping %s", log_path)
            return None
        try:
            return self.con.sql(
                f"""
                SELECT *
                FROM read_json_auto(
                    {path_arg},
                    format = 'newline_delimited',
                    union_by_name = true,
                    ignore_errors = true
                )
                """
            )
        except duckdb.Error as exc:
            log.warning("No Bronze data at %s (%s)", log_path, exc)
            return None

    def write_delta(
        self,
        relation: duckdb.DuckDBPyRelation,
        path: str,
        mode: str = "append",
        partition_by: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write a relation as a Delta table (or Parquet if delta-rs absent)."""
        # `.to_arrow_table()` (not `.arrow()`) — relations built over a
        # `con.register()`'d view come back as a RecordBatchReader from
        # `.arrow()`, which has no `.num_rows` and can only be consumed once.
        arrow_table = relation.to_arrow_table()
        rows = arrow_table.num_rows

        if DELTA_AVAILABLE:
            write_deltalake(
                path,
                arrow_table,
                mode=mode,
                partition_by=partition_by,
                storage_options=self._storage_options(),
            )
        else:
            local_path = Path(path.replace("s3://", "data/").rstrip("/"))
            local_path.mkdir(parents=True, exist_ok=True)
            out_file = local_path / f"part-{int(time.time() * 1000)}.parquet"
            self.con.sql(f"COPY (SELECT * FROM relation) TO '{out_file}' (FORMAT PARQUET)")
            path = str(local_path)

        log.info("Wrote %d rows → %s (mode=%s, delta=%s)", rows, path, mode, DELTA_AVAILABLE)
        return {"path": path, "rows": rows, "mode": mode, "delta": DELTA_AVAILABLE}

    def read_delta(self, path: str) -> duckdb.DuckDBPyRelation:
        """Read a Delta table (or Parquet directory fallback) into DuckDB."""
        if DELTA_AVAILABLE:
            try:
                dt = DeltaTable(path, storage_options=self._storage_options())
                self.con.register("_delta_scan", dt.to_pyarrow_dataset())
                return self.con.sql("SELECT * FROM _delta_scan")
            except Exception as exc:
                log.warning("Delta read failed for %s (%s) — trying Parquet", path, exc)

        local_path = path.replace("s3://", "data/").rstrip("/")
        return self.con.sql(f"SELECT * FROM read_parquet('{local_path}/*.parquet', union_by_name=true)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BRONZE LAYER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BronzeLoadResult:
    source: str
    rows: int
    path: str
    partition_date: date


class BronzeProcessor:
    """Lands raw, schema-on-read JSON into a Delta-backed Bronze table,
    tagging every record with lineage metadata."""

    def __init__(self, engine: LakehouseEngine, config: LakehouseConfig | None = None) -> None:
        self.engine = engine
        self.config = config or engine.config

    def ingest_raw(self, source: str, partition_date: date) -> duckdb.DuckDBPyRelation | None:
        """Read all NDJSON batches for `source` on `partition_date`."""
        prefix = (
            f"{source}/"
            f"year={partition_date:%Y}/month={partition_date:%m}/day={partition_date:%d}/"
        )
        # DuckDB's httpfs glob support probes the literal pre-wildcard prefix
        # with a GET and 404s on R2 instead of listing, so list the exact
        # keys via boto3 and hand DuckDB an explicit file list.
        keys = self.engine.list_bronze_keys(prefix)
        if not keys:
            log.warning("No Bronze data at s3://%s/%s*.ndjson", self.config.bronze_bucket, prefix)
            return None
        uris = [f"s3://{self.config.bronze_bucket}/{key}" for key in keys]
        log.info("Bronze ingest: %s ← %d file(s) under %s", source, len(uris), prefix)
        return self.engine.read_ndjson(uris)

    @staticmethod
    def validate_structure(relation: duckdb.DuckDBPyRelation, required_fields: list[str]) -> bool:
        """Cheap structural check: required fields must exist (Great Expectations
        does the deeper value-level checks in 02_data_ingestion.py)."""
        cols = set(relation.columns)
        missing = [f for f in required_fields if f not in cols]
        if missing:
            log.error("Bronze validation failed — missing fields: %s", missing)
            return False
        return True

    def add_metadata(
        self, relation: duckdb.DuckDBPyRelation, source: str, partition_date: date
    ) -> duckdb.DuckDBPyRelation:
        """Stamp lineage columns used for audit + incremental processing."""
        return self.engine.sql(
            f"""
            SELECT
                *,
                '{source}'                     AS _bronze_source,
                DATE '{partition_date.isoformat()}' AS _partition_date,
                now()                           AS _bronze_loaded_at
            FROM relation
            """,
            relation=relation,
        )

    def write_bronze_delta(self, relation: duckdb.DuckDBPyRelation, table_name: str = "raw_events") -> str:
        path = f"s3://{self.config.bronze_delta_bucket}/{table_name}/"
        try:
            result = self.engine.write_delta(relation, path, mode="append", partition_by=["_bronze_source"])
            return result["path"]
        except Exception as exc:
            log.warning("Bronze Delta write failed for %s (%s) — skipping", table_name, exc)
            return path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SILVER LAYER (unified schema, deduplicated, quality-gated)
# ─────────────────────────────────────────────────────────────────────────────

# Unified Silver schema — every source maps onto these columns.
SILVER_COLUMNS = [
    "id", "source", "title", "content", "abstract", "url", "author",
    "tags", "score", "num_comments", "published_at", "ingested_at", "dedup_hash",
]


def _has(columns: list[str], name: str) -> bool:
    return name in columns


class SilverProcessor:
    """Per-source transforms into the unified Silver schema, plus
    incremental dedup and a completeness quality gate."""

    def __init__(self, engine: LakehouseEngine, config: LakehouseConfig | None = None) -> None:
        self.engine = engine
        self.config = config or engine.config

    # -- Source-specific transforms ------------------------------------------------

    def transform_hn(self, bronze_rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        cols = bronze_rel.columns
        # Bronze HN records may come from either the Algolia search API
        # ("objectID", "story_text", "num_comments") or the HN Firebase item
        # API ("id", "text", "children") — tolerate both.
        id_expr = "objectID" if _has(cols, "objectID") else "id"
        story_text = "story_text" if _has(cols, "story_text") else ("text" if _has(cols, "text") else "NULL")
        url_expr = "url" if _has(cols, "url") else "NULL"
        num_comments_expr = (
            "num_comments" if _has(cols, "num_comments")
            else ("len(children)" if _has(cols, "children") else "0")
        )
        return self.engine.sql(
            f"""
            SELECT
                'hn_' || CAST({id_expr} AS VARCHAR)                         AS id,
                'hackernews'                                                AS source,
                title                                                       AS title,
                COALESCE({story_text}, '')                                  AS content,
                COALESCE({story_text}, title)                               AS abstract,
                COALESCE({url_expr}, 'https://news.ycombinator.com/item?id=' || CAST({id_expr} AS VARCHAR)) AS url,
                COALESCE(author, 'unknown')                                 AS author,
                []::VARCHAR[]                                               AS tags,
                COALESCE(points, 0)                                         AS score,
                COALESCE({num_comments_expr}, 0)                            AS num_comments,
                TRY_CAST(created_at AS TIMESTAMP)                           AS published_at,
                TRY_CAST(_ingested_at AS TIMESTAMP)                         AS ingested_at,
                md5(lower(title) || COALESCE({url_expr}, ''))               AS dedup_hash
            FROM bronze_rel
            WHERE title IS NOT NULL AND {id_expr} IS NOT NULL
            """,
            bronze_rel=bronze_rel,
        )

    def transform_arxiv(self, bronze_rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        return self.engine.sql(
            """
            SELECT
                'arxiv_' || arxiv_id                                        AS id,
                'arxiv'                                                     AS source,
                title                                                       AS title,
                abstract                                                    AS content,
                left(abstract, 500)                                         AS abstract,
                pdf_url                                                     AS url,
                COALESCE(list_extract(authors, 1), 'unknown')               AS author,
                categories                                                  AS tags,
                0                                                           AS score,
                0                                                           AS num_comments,
                TRY_CAST(published AS TIMESTAMP)                            AS published_at,
                TRY_CAST(_ingested_at AS TIMESTAMP)                         AS ingested_at,
                md5(arxiv_id)                                               AS dedup_hash
            FROM bronze_rel
            WHERE title IS NOT NULL AND arxiv_id IS NOT NULL
            """,
            bronze_rel=bronze_rel,
        )

    def transform_github(self, bronze_rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        return self.engine.sql(
            """
            SELECT
                'gh_' || CAST(repo_id AS VARCHAR)                           AS id,
                'github'                                                    AS source,
                full_name                                                   AS title,
                COALESCE(description, '')                                   AS content,
                COALESCE(description, '')                                   AS abstract,
                url                                                         AS url,
                COALESCE(owner_login, 'unknown')                            AS author,
                list_append(
                    COALESCE(topics, []::VARCHAR[]),
                    COALESCE(language, 'unknown')
                )                                                           AS tags,
                COALESCE(stars, 0)                                          AS score,
                COALESCE(forks, 0)                                          AS num_comments,
                TRY_CAST(created_at AS TIMESTAMP)                           AS published_at,
                TRY_CAST(_ingested_at AS TIMESTAMP)                         AS ingested_at,
                md5(full_name)                                              AS dedup_hash
            FROM bronze_rel
            WHERE full_name IS NOT NULL AND repo_id IS NOT NULL
            """,
            bronze_rel=bronze_rel,
        )

    # -- Union, dedup, quality -------------------------------------------------------

    def union_sources(self, *relations: duckdb.DuckDBPyRelation | None) -> duckdb.DuckDBPyRelation:
        valid = [r for r in relations if r is not None]
        if not valid:
            raise ValueError("union_sources: no Silver relations to union")
        rel = valid[0]
        for other in valid[1:]:
            rel = rel.union(other)
        # Repeated ingestion runs re-fetch overlapping top items, so Bronze
        # accumulates many duplicate dedup_hash rows across batches — keep
        # only the most-recently-ingested copy of each.
        return self.engine.sql(
            """
            SELECT * FROM rel
            QUALIFY ROW_NUMBER() OVER (PARTITION BY dedup_hash ORDER BY ingested_at DESC) = 1
            """,
            rel=rel,
        )

    def quality_gate(self, relation: duckdb.DuckDBPyRelation) -> tuple[bool, dict[str, Any]]:
        """Completeness + uniqueness gate. Mirrors the DataQualityChecker in
        02_data_ingestion.py but applied post-transform on the unified schema."""
        total = relation.count("*").fetchone()[0]
        if total == 0:
            return False, {"total_rows": 0, "reason": "empty batch"}

        non_null_title = relation.filter("title IS NOT NULL AND title != ''").count("*").fetchone()[0]
        non_null_content = relation.filter("content IS NOT NULL AND content != ''").count("*").fetchone()[0]
        distinct_hashes = relation.aggregate("count(distinct dedup_hash)").fetchone()[0]

        title_completeness = non_null_title / total
        content_completeness = non_null_content / total
        uniqueness = distinct_hashes / total

        report = {
            "total_rows": total,
            "title_completeness": round(title_completeness, 4),
            "content_completeness": round(content_completeness, 4),
            "uniqueness": round(uniqueness, 4),
        }
        passed = title_completeness >= 0.99 and uniqueness >= 0.95
        if not passed:
            log.warning("Silver quality gate FAILED: %s", report)
        else:
            log.info("Silver quality gate passed: %s", report)
        return passed, report

    def deduplicate_incremental(
        self, new_relation: duckdb.DuckDBPyRelation, table_path: str
    ) -> dict[str, Any]:
        """Upsert new rows into the Silver Delta table, keyed on dedup_hash.
        Uses a Delta MERGE when delta-rs is available; otherwise an
        anti-join + append against the existing Parquet snapshot."""
        new_arrow = new_relation.to_arrow_table()
        incoming = new_arrow.num_rows

        if DELTA_AVAILABLE:
            try:
                dt = DeltaTable(table_path, storage_options=self.engine._storage_options())
                (
                    dt.merge(
                        source=new_arrow,
                        predicate="target.dedup_hash = source.dedup_hash",
                        source_alias="source",
                        target_alias="target",
                    )
                    .when_not_matched_insert_all()
                    .execute()
                )
                return {"incoming": incoming, "strategy": "delta_merge", "path": table_path}
            except Exception as exc:
                log.warning("Delta MERGE unavailable (%s) — full append", exc)

        # Fallback: anti-join against existing snapshot, then append-only write.
        local_path = Path(table_path.replace("s3://", "data/").rstrip("/"))
        if local_path.exists() and any(local_path.glob("*.parquet")):
            existing = self.engine.con.sql(f"SELECT dedup_hash FROM read_parquet('{local_path}/*.parquet')")
            self.engine.con.register("_new", new_arrow)
            self.engine.con.register("_existing", existing)
            deduped = self.engine.con.sql(
                "SELECT _new.* FROM _new ANTI JOIN _existing USING (dedup_hash)"
            )
            new_rows = deduped.to_arrow_table().num_rows
        else:
            deduped = self.engine.con.sql("SELECT * FROM new_arrow")
            new_rows = incoming

        try:
            self.engine.write_delta(deduped, table_path, mode="append")
        except Exception as exc:
            log.warning("Silver Delta write failed for %s (%s) — skipping", table_path, exc)
        return {"incoming": incoming, "new_rows": new_rows, "strategy": "anti_join_append", "path": table_path}

    def write_silver(self, relation: duckdb.DuckDBPyRelation) -> str:
        path = f"s3://{self.config.silver_bucket}/content/"
        return self.deduplicate_incremental(relation, path)["path"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — GOLD LAYER (aggregated, embedding-ready, feature-ready)
# ─────────────────────────────────────────────────────────────────────────────

class GoldProcessor:
    """Builds the four Gold tables consumed downstream:
    - knowledge_base   → 05_rag_service.py embedding/index source
    - feature_table    → 06_ml_platform.py engagement features
    - trending_topics  → RAG API /trending and /topics endpoints
    - author_profiles  → analytics / leaderboards
    """

    def __init__(self, engine: LakehouseEngine, config: LakehouseConfig | None = None) -> None:
        self.engine = engine
        self.config = config or engine.config

    def build_daily_summary(self, silver_rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        return self.engine.sql(
            """
            SELECT
                source,
                CAST(published_at AS DATE)     AS published_date,
                count(*)                       AS item_count,
                avg(score)                      AS avg_score,
                max(score)                      AS max_score,
                sum(num_comments)               AS total_comments
            FROM silver_rel
            WHERE published_at IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 2 DESC, 1
            """,
            silver_rel=silver_rel,
        )

    def build_trending_topics(self, silver_rel: duckdb.DuckDBPyRelation, window_days: int = 7) -> duckdb.DuckDBPyRelation:
        """Topics ranked by velocity = recent_count / (prior_count + 1),
        a simple growth-rate signal over a sliding window."""
        return self.engine.sql(
            f"""
            WITH exploded AS (
                SELECT unnest(tags) AS topic, score, published_at, source
                FROM silver_rel
                WHERE published_at >= now() - INTERVAL '{window_days} days'
                  AND len(tags) > 0
            ),
            recent AS (
                SELECT topic, count(*) AS cnt_recent, avg(score) AS avg_score_recent
                FROM exploded
                WHERE published_at >= now() - INTERVAL '1 day'
                GROUP BY topic
            ),
            prior AS (
                SELECT topic, count(*) AS cnt_prior
                FROM exploded
                WHERE published_at < now() - INTERVAL '1 day'
                GROUP BY topic
            )
            SELECT
                r.topic,
                r.cnt_recent,
                COALESCE(p.cnt_prior, 0)                                   AS cnt_prior,
                CAST(r.cnt_recent AS DOUBLE) / (COALESCE(p.cnt_prior, 0) + 1) AS velocity,
                round(r.avg_score_recent, 2)                               AS avg_score_recent
            FROM recent r
            LEFT JOIN prior p USING (topic)
            ORDER BY velocity DESC, cnt_recent DESC
            """,
            silver_rel=silver_rel,
        )

    def build_knowledge_base(self, silver_rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        """Denormalized table feeding the RAG vector index — one row per
        chunkable document, with everything 05_rag_service.py needs."""
        return self.engine.sql(
            """
            SELECT
                id, source, title, content, abstract, url, author,
                tags, score, num_comments, published_at, ingested_at
            FROM silver_rel
            WHERE content IS NOT NULL AND content != ''
            """,
            silver_rel=silver_rel,
        )

    def build_feature_table(self, silver_rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        """ML-ready features for 06_ml_platform.py (topic classifier +
        drift monitor). `days_since_published` and `num_comments` here are
        read directly by FeatureEngineer.compute_engagement_features()."""
        return self.engine.sql(
            """
            SELECT
                id,
                source,
                len(title)                                          AS title_length,
                score,
                num_comments,
                greatest(date_diff('day', published_at, now()), 0)  AS days_since_published,
                extract(hour FROM published_at)                     AS hour_of_day,
                extract(dow  FROM published_at)                     AS day_of_week,
                (content LIKE '%```%'
                    OR content LIKE '%def %'
                    OR content LIKE '%function %')                  AS has_code
            FROM silver_rel
            WHERE published_at IS NOT NULL
            """,
            silver_rel=silver_rel,
        )

    def build_author_profiles(self, silver_rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        return self.engine.sql(
            """
            SELECT
                author,
                source,
                count(*)            AS item_count,
                avg(score)           AS avg_score,
                max(published_at)    AS last_active
            FROM silver_rel
            WHERE author IS NOT NULL AND author != 'unknown'
            GROUP BY author, source
            ORDER BY item_count DESC
            """,
            silver_rel=silver_rel,
        )

    def write_gold(
        self,
        relation: duckdb.DuckDBPyRelation,
        name: str,
        partition_by: list[str] | None = None,
        mirror_local: bool = False,
    ) -> str:
        path = f"s3://{self.config.gold_bucket}/{name}/"
        try:
            result = self.engine.write_delta(relation, path, mode="overwrite", partition_by=partition_by)
            written_path = result["path"]
        except Exception as exc:
            log.warning("Gold S3 write failed for %s (%s) — local only", name, exc)
            written_path = path

        if self.engine.s3_available:
            # Plain Parquet alongside the (possibly failed) Delta write — R2's
            # ListObjectsV2 + start-after combination used by delta-rs to
            # detect existing tables returns NoSuchKey on an empty prefix, so
            # the Delta write above can fail even though plain S3 PUTs work
            # fine. 05_rag_service.py reads this glob directly as a fallback.
            try:
                self.engine.con.register("_gold_rel", relation)
                self.engine.con.sql(
                    f"COPY (SELECT * FROM _gold_rel) TO '{path}data.parquet' (FORMAT PARQUET)"
                )
                self.engine.con.unregister("_gold_rel")
                log.info("Wrote Gold table '%s' parquet → %sdata.parquet", name, path)
            except Exception as exc:
                log.warning("Gold S3 parquet write failed for %s (%s)", name, exc)

        if mirror_local:
            local_dir = Path(self.config.local_gold_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / f"{name}.parquet"
            self.engine.con.sql(f"COPY (SELECT * FROM relation) TO '{local_file}' (FORMAT PARQUET)")
            log.info("Mirrored Gold table '%s' → %s", name, local_file)

        return written_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DELTA LAKE OPERATIONS (time travel, compaction, vacuum)
# ─────────────────────────────────────────────────────────────────────────────

class DeltaLakeManager:
    """Operational toolkit for Delta tables — the commands an AI Data Infra
    engineer runs to keep a lakehouse healthy and auditable."""

    def __init__(self, engine: LakehouseEngine) -> None:
        self.engine = engine

    def get_table_history(self, path: str) -> list[dict[str, Any]]:
        if not DELTA_AVAILABLE:
            log.warning("get_table_history: deltalake not installed")
            return []
        dt = DeltaTable(path, storage_options=self.engine._storage_options())
        return dt.history()

    def time_travel(self, path: str, version: int | None = None, timestamp: str | None = None) -> duckdb.DuckDBPyRelation:
        """Read a Delta table as of a past version or ISO timestamp."""
        if not DELTA_AVAILABLE:
            raise RuntimeError("time_travel requires deltalake")
        dt = DeltaTable(path, storage_options=self.engine._storage_options())
        if version is not None:
            dt.load_as_version(version)
        elif timestamp is not None:
            dt.load_with_datetime(timestamp)
        self.engine.con.register("_delta_tt", dt.to_pyarrow_dataset())
        return self.engine.con.sql("SELECT * FROM _delta_tt")

    def compact_small_files(self, path: str) -> dict[str, Any]:
        """OPTIMIZE: bin-pack small Parquet files into larger ones — the
        single highest-leverage Delta maintenance operation for query cost."""
        if not DELTA_AVAILABLE:
            return {"skipped": True, "reason": "deltalake not installed"}
        dt = DeltaTable(path, storage_options=self.engine._storage_options())
        metrics = dt.optimize.compact()
        log.info("Compacted %s: %s", path, metrics)
        return metrics

    def vacuum(self, path: str, retention_hours: int = 168, dry_run: bool = True) -> list[str]:
        """Remove tombstoned files older than `retention_hours` (default 7
        days). ALWAYS dry-run first in production — vacuum is irreversible
        and breaks time travel before the retention cutoff."""
        if not DELTA_AVAILABLE:
            return []
        dt = DeltaTable(path, storage_options=self.engine._storage_options())
        files = dt.vacuum(retention_hours=retention_hours, dry_run=dry_run)
        log.info("Vacuum(%s, dry_run=%s) → %d files", path, dry_run, len(files))
        return files

    def demonstrate_schema_evolution(self, path: str) -> None:
        """Show that adding a new nullable column doesn't break existing
        readers — the standard Delta/Iceberg schema-evolution guarantee."""
        if not DELTA_AVAILABLE:
            log.warning("schema evolution demo requires deltalake")
            return
        import pyarrow as pa

        dt = DeltaTable(path, storage_options=self.engine._storage_options())
        sample = dt.to_pyarrow_dataset().head(1)
        new_col = pa.array([None], type=pa.string())
        evolved = sample.append_column("sentiment_label", new_col)
        write_deltalake(
            path, evolved, mode="append",
            schema_mode="merge",  # merge new column into table schema
            storage_options=self.engine._storage_options(),
        )
        log.info("Schema evolved: added nullable 'sentiment_label' column to %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    run_date: date
    bronze_rows: dict[str, int] = field(default_factory=dict)
    silver_rows: int = 0
    silver_quality: dict[str, Any] = field(default_factory=dict)
    gold_tables: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


def _sample_bronze_relations(engine: LakehouseEngine) -> tuple[duckdb.DuckDBPyRelation, duckdb.DuckDBPyRelation, duckdb.DuckDBPyRelation]:
    """Synthetic Bronze data so the pipeline runs end-to-end without MinIO.
    Shapes match exactly what 02_data_ingestion.py lands in production."""
    now = datetime.now(timezone.utc)
    iso = now.isoformat()

    hn = engine.con.sql(
        f"""
        SELECT * FROM (VALUES
            (1001, 'Show HN: pgvector hits 1M QPS with HNSW tuning', 'https://example.com/pgvector', 'alice', 412, 88, NULL, TIMESTAMP '{(now - timedelta(hours=2)).isoformat()}', TIMESTAMP '{iso}'),
            (1002, 'Why we moved our lakehouse from Spark to DuckDB',  'https://example.com/duckdb',   'bob',   301, 64, NULL, TIMESTAMP '{(now - timedelta(hours=5)).isoformat()}', TIMESTAMP '{iso}'),
            (1003, 'Ask HN: best practices for Delta Lake on R2?',     NULL,                            'carol', 150, 42, 'Curious about MERGE performance on object storage.', TIMESTAMP '{(now - timedelta(hours=20)).isoformat()}', TIMESTAMP '{iso}')
        ) AS t(objectID, title, url, author, points, num_comments, story_text, created_at, _ingested_at)
        """
    )

    arxiv = engine.con.sql(
        f"""
        SELECT * FROM (VALUES
            ('2406.00123', 'Efficient Fine-Tuning of LLMs with QLoRA at Scale', 'We present a study of QLoRA fine-tuning across model families...', ['Ada Lin','Bo Chen'], ['cs.AI','cs.LG'], 'https://arxiv.org/pdf/2406.00123', TIMESTAMP '{(now - timedelta(days=1)).isoformat()}', TIMESTAMP '{iso}'),
            ('2406.00456', 'Graph RAG: Structured Retrieval over Knowledge Graphs', 'This paper introduces a Graph RAG architecture combining...', ['Dee Wu'], ['cs.CL','cs.AI'], 'https://arxiv.org/pdf/2406.00456', TIMESTAMP '{(now - timedelta(days=3)).isoformat()}', TIMESTAMP '{iso}')
        ) AS t(arxiv_id, title, abstract, authors, categories, pdf_url, published, _ingested_at)
        """
    )

    github = engine.con.sql(
        f"""
        SELECT * FROM (VALUES
            (501, 'aidataboon/lakehouse-rs', 'Rust toolkit for medallion lakehouses on object storage', 'Rust',   1840, 92, 'rustlang-org', 'https://github.com/aidataboon/lakehouse-rs', ['delta-lake','rust','data-engineering'], TIMESTAMP '{(now - timedelta(days=10)).isoformat()}', TIMESTAMP '{iso}'),
            (502, 'vectorstack/pgvector-bench', 'Benchmarks for HNSW vs IVFFlat on pgvector 0.7',         'Python', 980,  35, 'vectorstack',  'https://github.com/vectorstack/pgvector-bench', ['pgvector','vector-search','benchmark'], TIMESTAMP '{(now - timedelta(days=2)).isoformat()}', TIMESTAMP '{iso}')
        ) AS t(repo_id, full_name, description, language, stars, forks, owner_login, url, topics, created_at, _ingested_at)
        """
    )
    return hn, arxiv, github


def run_full_pipeline(run_date: date | None = None) -> PipelineResult:
    """Bronze → Silver → Gold for `run_date` (defaults to today, UTC).

    Tries real MinIO/S3 Bronze data first; if none is found (e.g. the
    Airflow DAGs in 02_data_ingestion.py haven't run yet), falls back to
    synthetic sample data so the lakehouse logic can still be exercised."""
    run_date = run_date or datetime.now(timezone.utc).date()
    started = time.monotonic()
    log.info("=== AIDataBoon Lakehouse Pipeline — %s ===", run_date.isoformat())

    engine = LakehouseEngine()
    bronze = BronzeProcessor(engine)
    silver = SilverProcessor(engine)
    gold = GoldProcessor(engine)
    result = PipelineResult(run_date=run_date)

    # ---- Bronze ---------------------------------------------------------------
    hn_raw = bronze.ingest_raw("hn", run_date)
    arxiv_raw = bronze.ingest_raw("arxiv", run_date)
    github_raw = bronze.ingest_raw("github-trending", run_date)

    if hn_raw is None and arxiv_raw is None and github_raw is None:
        log.info("No Bronze data found in MinIO — using synthetic sample data")
        hn_raw, arxiv_raw, github_raw = _sample_bronze_relations(engine)

    for name, rel in (("hn", hn_raw), ("arxiv", arxiv_raw), ("github", github_raw)):
        if rel is not None:
            result.bronze_rows[name] = rel.count("*").fetchone()[0]
            bronze.write_bronze_delta(bronze.add_metadata(rel, name, run_date), table_name="raw_events")

    # ---- Silver -----------------------------------------------------------------
    silver_parts = []
    if hn_raw is not None:
        log.info("hn_raw columns: %s", hn_raw.columns)
        try:
            silver_parts.append(silver.transform_hn(hn_raw))
        except duckdb.Error as exc:
            log.warning("transform_hn failed (%s) — skipping hn for this run", exc)
    if arxiv_raw is not None:
        log.info("arxiv_raw columns: %s", arxiv_raw.columns)
        try:
            silver_parts.append(silver.transform_arxiv(arxiv_raw))
        except duckdb.Error as exc:
            log.warning("transform_arxiv failed (%s) — skipping arxiv for this run", exc)
    if github_raw is not None:
        log.info("github_raw columns: %s", github_raw.columns)
        try:
            silver_parts.append(silver.transform_github(github_raw))
        except duckdb.Error as exc:
            log.warning("transform_github failed (%s) — skipping github for this run", exc)

    silver_rel = silver.union_sources(*silver_parts)
    passed, report = silver.quality_gate(silver_rel)
    result.silver_quality = report
    result.silver_rows = report["total_rows"]

    if not passed:
        log.error("Silver quality gate failed — aborting before Gold. Report: %s", report)
        result.duration_seconds = time.monotonic() - started
        return result

    silver.write_silver(silver_rel)

    # ---- Gold ---------------------------------------------------------------------
    knowledge_base = gold.build_knowledge_base(silver_rel)
    feature_table = gold.build_feature_table(silver_rel)
    trending_topics = gold.build_trending_topics(silver_rel)
    daily_summary = gold.build_daily_summary(silver_rel)
    author_profiles = gold.build_author_profiles(silver_rel)

    for name, rel in (
        ("knowledge_base", knowledge_base),
        ("feature_table", feature_table),
        ("trending_topics", trending_topics),
        ("daily_summary", daily_summary),
        ("author_profiles", author_profiles),
    ):
        result.gold_tables[name] = rel.count("*").fetchone()[0]
        gold.write_gold(rel, name, mirror_local=name in ("knowledge_base", "feature_table"))

    result.duration_seconds = time.monotonic() - started
    log.info(
        "Pipeline complete in %.2fs — bronze=%s silver=%d gold=%s",
        result.duration_seconds, result.bronze_rows, result.silver_rows, result.gold_tables,
    )
    return result


if __name__ == "__main__":
    res = run_full_pipeline()
    print()
    print("=== AIDataBoon Lakehouse — Run Summary ===")
    print(f"  Run date        : {res.run_date}")
    print(f"  Bronze rows     : {res.bronze_rows}")
    print(f"  Silver rows     : {res.silver_rows}")
    print(f"  Silver quality  : {res.silver_quality}")
    print(f"  Gold tables     : {res.gold_tables}")
    print(f"  Duration        : {res.duration_seconds:.2f}s")
    print()
    print("  Gold mirrors written to ./data/gold/*.parquet for 06_ml_platform.py")
