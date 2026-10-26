"""
AIDataBoon — Data Ingestion Pipeline

Three Airflow DAGs:
1. hn_batch_ingestion     — Hacker News top stories (runs every 30 min)
2. arxiv_batch_ingestion  — arXiv AI/ML papers (runs daily at 6am)
3. github_trending_ingest — GitHub trending repos (runs daily at 8am)

Storage: MinIO/S3-compatible (s3://aidataboon-bronze/)
Quality: Great Expectations inline checks
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import requests

# ── Structured logging setup ──────────────────────────────────────────────────
log = logging.getLogger("aidataboon.ingestion")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSourceConfig:
    """Per-source ingestion configuration."""
    api_url: str
    bucket: str
    prefix: str
    batch_size: int = 100
    max_retries: int = 3
    retry_backoff_base: float = 2.0       # seconds; exponential backoff multiplier
    request_timeout: int = 30             # seconds per HTTP call
    dead_letter_prefix: str = "dlq"      # prefix for failed-batch storage


@dataclass
class StorageConfig:
    """S3-compatible object storage config (MinIO / AWS S3 / Cloudflare R2)."""
    endpoint_url: str                     # http://minio:9000 locally; None for AWS
    access_key: str
    secret_key: str
    bucket_name: str
    region_name: str = "us-east-1"
    use_ssl: bool = False                 # True for R2 / AWS in production


@dataclass
class QualityConfig:
    """Inline data-quality thresholds."""
    min_completeness: float = 0.95        # fraction of non-null values required
    max_nulls_pct: float = 0.05           # allowed null fraction per required column
    required_columns: list[str] = field(default_factory=list)
    max_age_hours: int = 24               # freshness gate


# Default configs ──────────────────────────────────────────────────────────────

HN_SOURCE_CONFIG = DataSourceConfig(
    # search_by_date (not search) — plain "search" ranks by relevance/points
    # and surfaces ancient all-time classics, which fail the freshness gate.
    api_url="https://hn.algolia.com/api/v1/search_by_date",
    bucket="aidataboon-bronze",
    prefix="hn",
    batch_size=100,
    max_retries=3,
)

ARXIV_SOURCE_CONFIG = DataSourceConfig(
    api_url="https://export.arxiv.org/api/query",
    bucket="aidataboon-bronze",
    prefix="arxiv",
    batch_size=200,
    max_retries=3,
    request_timeout=15,
)

GITHUB_SOURCE_CONFIG = DataSourceConfig(
    api_url="https://api.github.com/search/repositories",
    bucket="aidataboon-bronze",
    prefix="github-trending",
    batch_size=100,
    max_retries=3,
)

# Reads R2/S3 credentials from the environment (same names used by
# 05_rag_service.py / render.yaml) so this script can target either the
# local MinIO stack (defaults below) or production Cloudflare R2.
STORAGE_CONFIG = StorageConfig(
    endpoint_url=os.environ.get("S3_ENDPOINT_URL", "http://minio:9000"),
    access_key=os.environ.get("S3_ACCESS_KEY", "minioadmin"),
    secret_key=os.environ.get("S3_SECRET_KEY", "minioadmin"),
    bucket_name=os.environ.get("S3_BRONZE_BUCKET", "aidataboon-bronze"),
    region_name=os.environ.get("S3_REGION", "us-east-1"),
    use_ssl=os.environ.get("S3_ENDPOINT_URL", "http://minio:9000").startswith("https://"),
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — STORAGE CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class S3StorageClient:
    """
    Thin wrapper around boto3 for Hive-partitioned JSON storage.
    Compatible with MinIO (local), AWS S3, and Cloudflare R2 (endpoint_url).
    """

    def __init__(self, config: StorageConfig) -> None:
        import boto3  # deferred import; not available in all environments
        from botocore.config import Config as BotoConfig

        self._bucket = config.bucket_name
        session = boto3.Session(
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region_name,
        )
        kwargs: dict[str, Any] = {}
        if config.endpoint_url:
            kwargs["endpoint_url"] = config.endpoint_url
        # Cloudflare R2 doesn't support virtual-hosted-style addressing
        # (bucket.endpoint) — boto3 defaults to it for non-AWS endpoints and
        # R2 responds with a misleading NoSuchKey error. Force path-style
        # (endpoint/bucket), matching the DuckDB URL_STYLE 'path' secret in
        # 03_lakehouse.py / 05_rag_service.py.
        kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})
        self._s3 = session.client("s3", **kwargs)
        log.info("S3StorageClient initialised", extra={"bucket": self._bucket})

    # ── Hive-style partition key builder ──────────────────────────────────────

    @staticmethod
    def _partition_path(prefix: str, partition_keys: dict[str, str]) -> str:
        """Build s3_key from Hive-style partition keys.

        Example:
            prefix="hn", keys={"year":"2026","month":"06","day":"09","hour":"14"}
            → "hn/year=2026/month=06/day=09/hour=14"
        """
        parts = "/".join(f"{k}={v}" for k, v in partition_keys.items())
        return f"{prefix}/{parts}"

    @staticmethod
    def _now_partition_keys() -> dict[str, str]:
        now = datetime.now(tz=timezone.utc)
        return {
            "year": now.strftime("%Y"),
            "month": now.strftime("%m"),
            "day": now.strftime("%d"),
            "hour": now.strftime("%H"),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def write_json_batch(
        self,
        prefix: str,
        data: list[dict],
        partition_keys: dict[str, str] | None = None,
        filename: str | None = None,
    ) -> str:
        """Serialise *data* as newline-delimited JSON and write to S3.

        Returns the full S3 URI written (s3://bucket/key).
        """
        if partition_keys is None:
            partition_keys = self._now_partition_keys()

        folder = self._partition_path(prefix, partition_keys)
        fname = filename or f"batch_{uuid.uuid4().hex[:8]}.ndjson"
        key = f"{folder}/{fname}"

        payload = "\n".join(json.dumps(record, default=str) for record in data)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=payload.encode("utf-8"),
            ContentType="application/x-ndjson",
        )
        s3_uri = f"s3://{self._bucket}/{key}"
        log.info("Wrote batch to S3", extra={"uri": s3_uri, "rows": len(data)})
        return s3_uri

    def read_json_batch(self, prefix: str) -> list[dict]:
        """Read all NDJSON objects under *prefix* and return merged record list."""
        objects = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        records: list[dict] = []
        for obj in objects.get("Contents", []):
            body = self._s3.get_object(Bucket=self._bucket, Key=obj["Key"])["Body"].read()
            for line in body.decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def list_partitions(self, prefix: str) -> list[str]:
        """Return all unique Hive partition paths under *prefix*."""
        paginator = self._s3.get_paginator("list_objects_v2")
        partitions: set[str] = set()
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                partitions.add(cp["Prefix"])
        return sorted(partitions)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    passed: bool
    score: float               # 0.0 – 1.0
    violations: list[str]
    row_count: int
    checked_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError(
                f"Quality gate failed (score={self.score:.2f}): "
                + "; ".join(self.violations)
            )


class DataQualityChecker:
    """
    Inline Great-Expectations-style quality checks backed by pure Python /
    DuckDB — no GE library dependency for portability in serverless envs.
    """

    def __init__(self, config: QualityConfig) -> None:
        self._cfg = config

    def check_schema(
        self, records: list[dict], expected_schema: dict[str, type]
    ) -> list[str]:
        """Validate that every expected field exists and has correct Python type."""
        violations: list[str] = []
        for col, expected_type in expected_schema.items():
            missing = sum(1 for r in records if col not in r)
            if missing:
                violations.append(f"schema: column '{col}' missing in {missing}/{len(records)} rows")
                continue
            wrong_type = sum(
                1 for r in records
                if r.get(col) is not None and not isinstance(r[col], expected_type)
            )
            if wrong_type:
                violations.append(
                    f"schema: column '{col}' wrong type in {wrong_type}/{len(records)} rows "
                    f"(expected {expected_type.__name__})"
                )
        return violations

    def check_completeness(
        self, records: list[dict], column: str, min_pct: float
    ) -> list[str]:
        """Fraction of non-null values must be ≥ min_pct."""
        violations: list[str] = []
        if not records:
            violations.append(f"completeness: empty record set for column '{column}'")
            return violations
        non_null = sum(1 for r in records if r.get(column) is not None)
        actual_pct = non_null / len(records)
        if actual_pct < min_pct:
            violations.append(
                f"completeness: '{column}' is {actual_pct:.1%} complete "
                f"(required ≥ {min_pct:.1%})"
            )
        return violations

    def check_uniqueness(
        self, records: list[dict], key_columns: list[str]
    ) -> list[str]:
        """Composite key formed by key_columns must be unique across all rows."""
        violations: list[str] = []
        seen: set[tuple] = set()
        duplicates = 0
        for r in records:
            key = tuple(r.get(c) for c in key_columns)
            if key in seen:
                duplicates += 1
            seen.add(key)
        if duplicates:
            violations.append(
                f"uniqueness: {duplicates} duplicate(s) on key {key_columns}"
            )
        return violations

    def check_freshness(
        self,
        records: list[dict],
        ts_column: str,
        max_age_hours: int,
    ) -> list[str]:
        """Most recent record must be within max_age_hours of now."""
        violations: list[str] = []
        timestamps: list[datetime] = []
        for r in records:
            raw = r.get(ts_column)
            if raw is None:
                continue
            if isinstance(raw, datetime):
                timestamps.append(raw)
            else:
                try:
                    timestamps.append(datetime.fromisoformat(str(raw)))
                except (ValueError, TypeError):
                    continue
        if not timestamps:
            violations.append(f"freshness: no parseable timestamps in '{ts_column}'")
            return violations
        latest = max(timestamps)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        age = datetime.now(tz=timezone.utc) - latest
        if age > timedelta(hours=max_age_hours):
            violations.append(
                f"freshness: latest '{ts_column}' is {age} old "
                f"(max allowed {max_age_hours}h)"
            )
        return violations

    def run_all(
        self,
        records: list[dict],
        expected_schema: dict[str, type],
        key_columns: list[str],
        ts_column: str,
    ) -> QualityResult:
        """Run the full quality suite and return a scored QualityResult."""
        violations: list[str] = []
        checks_run = 4  # schema, completeness, uniqueness, freshness

        violations += self.check_schema(records, expected_schema)
        for col in self._cfg.required_columns:
            violations += self.check_completeness(
                records, col, 1.0 - self._cfg.max_nulls_pct
            )
        violations += self.check_uniqueness(records, key_columns)
        violations += self.check_freshness(records, ts_column, self._cfg.max_age_hours)

        # Score = fraction of checks that produced no violation
        check_types_failed = {v.split(":")[0] for v in violations}
        score = 1.0 - (len(check_types_failed) / checks_run)
        return QualityResult(
            passed=len(violations) == 0,
            score=max(0.0, score),
            violations=violations,
            row_count=len(records),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HN INGESTION DAG
# ─────────────────────────────────────────────────────────────────────────────

# Guard: only import Airflow when the module is loaded in an Airflow environment.
# The plain-Python fallback at module bottom (Section 8) runs without Airflow.
try:
    from airflow.decorators import dag, task
    from airflow.utils.dates import days_ago
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    # Minimal no-op decorators so the rest of the file is importable standalone.
    def dag(*dargs, **dkwargs):  # type: ignore[misc]
        def wrapper(fn):
            return fn
        return wrapper

    def task(*targs, **tkwargs):  # type: ignore[misc]
        def wrapper(fn):
            return fn
        return wrapper


_HN_DEFAULT_ARGS = {
    "owner": "aidataboon",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(minutes=15),
    "email_on_failure": True,
    "email": ["alerts@aidataboon.ai"],
}

_HN_EXPECTED_SCHEMA: dict[str, type] = {
    "objectID": str,
    "title": str,
    "points": int,
    "num_comments": int,
}
_HN_QUALITY_CONFIG = QualityConfig(
    min_completeness=0.95,
    max_nulls_pct=0.05,
    required_columns=["objectID", "title", "points"],
)


@dag(
    dag_id="hn_batch_ingestion",
    schedule="*/30 * * * *",
    start_date=days_ago(1) if _AIRFLOW_AVAILABLE else datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=_HN_DEFAULT_ARGS,
    tags=["aidataboon", "bronze", "hackernews"],
    doc_md=__doc__,
)
def hn_batch_ingestion() -> None:
    """Ingest Hacker News top stories into Bronze every 30 minutes."""

    @task(task_id="fetch_top_stories")
    def fetch_top_stories() -> list[dict]:
        """Pull up to 100 high-signal HN stories via the Algolia search API."""
        params = {
            "tags": "story",
            "numericFilters": "points>50",
            "hitsPerPage": HN_SOURCE_CONFIG.batch_size,
        }
        url = f"{HN_SOURCE_CONFIG.api_url}?{urlencode(params)}"
        log.info("Fetching HN top stories", extra={"url": url})

        for attempt in range(1, HN_SOURCE_CONFIG.max_retries + 1):
            try:
                resp = requests.get(url, timeout=HN_SOURCE_CONFIG.request_timeout)
                resp.raise_for_status()
                hits: list[dict] = resp.json().get("hits", [])
                log.info("Fetched HN stories", extra={"count": len(hits)})
                return hits
            except requests.RequestException as exc:
                wait = HN_SOURCE_CONFIG.retry_backoff_base ** attempt
                log.warning("HN fetch failed, retrying", extra={"attempt": attempt, "error": str(exc)})
                if attempt == HN_SOURCE_CONFIG.max_retries:
                    raise RuntimeError(f"HN fetch exhausted {attempt} retries") from exc
                time.sleep(wait)
        return []  # unreachable but satisfies type checker

    @task(task_id="fetch_story_details")
    def fetch_story_details(story_ids: list[str]) -> list[dict]:
        """Enrich each story with full detail from the HN items endpoint.

        Best-effort: uses a short per-request timeout and bails out after a
        few consecutive failures (the items endpoint is either reachable or
        it isn't — no point burning the full per-item timeout 100 times)."""
        base = "https://hn.algolia.com/api/v1/items"
        enriched: list[dict] = []
        consecutive_failures = 0
        for story_id in story_ids[:HN_SOURCE_CONFIG.batch_size]:
            try:
                resp = requests.get(f"{base}/{story_id}", timeout=5)
                resp.raise_for_status()
                enriched.append(resp.json())
                consecutive_failures = 0
            except requests.RequestException as exc:
                log.warning("Could not fetch HN story detail", extra={"id": story_id, "error": str(exc)})
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    log.warning("HN items endpoint unreachable — aborting enrichment early")
                    break
        return enriched

    @task(task_id="validate_quality")
    def validate_quality(stories: list[dict]) -> dict:
        """Run inline quality checks; surface violations without hard-failing the DAG."""
        checker = DataQualityChecker(config=_HN_QUALITY_CONFIG)
        result = checker.run_all(
            records=stories,
            expected_schema=_HN_EXPECTED_SCHEMA,
            key_columns=["objectID"],
            ts_column="created_at",
        )
        log.info(
            "HN quality check",
            extra={"passed": result.passed, "score": result.score, "violations": result.violations},
        )
        # Serialise dataclass so it flows through XCom (must be JSON-serialisable)
        return {
            "passed": result.passed,
            "score": result.score,
            "violations": result.violations,
            "row_count": result.row_count,
        }

    @task(task_id="write_to_bronze")
    def write_to_bronze(stories: list[dict], quality: dict) -> str:
        """Persist the batch to MinIO Bronze, tagging with quality metadata."""
        if not stories:
            raise ValueError("write_to_bronze received empty stories list")

        # Attach ingestion metadata to every record
        batch_id = uuid.uuid4().hex
        ingested_at = datetime.now(tz=timezone.utc).isoformat()
        for story in stories:
            story["_ingested_at"] = ingested_at
            story["_batch_id"] = batch_id
            story["_source"] = "hackernews"
            story["_quality_score"] = quality["score"]

        client = S3StorageClient(STORAGE_CONFIG)
        now = datetime.now(tz=timezone.utc)
        uri = client.write_json_batch(
            prefix=HN_SOURCE_CONFIG.prefix,
            data=stories,
            partition_keys={
                "year": now.strftime("%Y"),
                "month": now.strftime("%m"),
                "day": now.strftime("%d"),
                "hour": now.strftime("%H"),
            },
            filename=f"hn_{batch_id}.ndjson",
        )
        log.info("HN Bronze write complete", extra={"uri": uri, "rows": len(stories)})

        # Dead-letter the batch if quality gate failed
        if not quality["passed"]:
            dlq_uri = client.write_json_batch(
                prefix=f"{HN_SOURCE_CONFIG.dead_letter_prefix}/hn",
                data=stories,
                filename=f"dlq_{batch_id}.ndjson",
            )
            log.warning("HN batch sent to DLQ", extra={"dlq_uri": dlq_uri})

        return uri

    @task(task_id="emit_quality_metrics")
    def emit_quality_metrics(quality: dict) -> None:
        """Push quality metrics to Prometheus Pushgateway."""
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway  # type: ignore

            registry = CollectorRegistry()
            g_score = Gauge(
                "aidataboon_quality_score",
                "Data quality score 0-1",
                ["source"],
                registry=registry,
            )
            g_rows = Gauge(
                "aidataboon_ingested_rows",
                "Rows ingested per batch",
                ["source"],
                registry=registry,
            )
            g_score.labels(source="hackernews").set(quality["score"])
            g_rows.labels(source="hackernews").set(quality["row_count"])
            push_to_gateway("pushgateway:9091", job="aidataboon_ingestion", registry=registry)
            log.info("Quality metrics pushed to Prometheus")
        except ImportError:
            log.debug("prometheus_client not installed; skipping metrics push")
        except Exception as exc:  # network errors must not kill the DAG
            log.warning("Prometheus push failed", extra={"error": str(exc)})

    # ── Wire tasks ────────────────────────────────────────────────────────────
    raw_stories = fetch_top_stories()
    story_ids = [s.get("objectID", "") for s in raw_stories] if isinstance(raw_stories, list) else []
    # In real Airflow the downstream tasks receive XCom values automatically
    detailed = fetch_story_details(story_ids)
    if not detailed and raw_stories:
        # /items/{id} enrichment is best-effort — the /search response already
        # carries title, url, points, author, etc., so fall back to it rather
        # than dropping the whole batch if the items endpoint is unreachable.
        log.warning("HN item enrichment returned nothing — using /search results as-is")
        detailed = raw_stories
    quality_result = validate_quality(detailed)
    bronze_uri = write_to_bronze(detailed, quality_result)
    emit_quality_metrics(quality_result)
    # Silence "unused variable" lints — these create the task-dependency graph
    _ = bronze_uri


try:
    hn_dag_instance = hn_batch_ingestion()
except Exception as exc:
    log.warning("HN ingestion failed, skipping (%s)", exc)
    hn_dag_instance = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — arXiv INGESTION DAG
# ─────────────────────────────────────────────────────────────────────────────

_ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_ARXIV_DEFAULT_ARGS = {**_HN_DEFAULT_ARGS, "execution_timeout": timedelta(minutes=30)}
_ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]


@dag(
    dag_id="arxiv_batch_ingestion",
    schedule="0 6 * * *",
    start_date=days_ago(1) if _AIRFLOW_AVAILABLE else datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=_ARXIV_DEFAULT_ARGS,
    tags=["aidataboon", "bronze", "arxiv"],
)
def arxiv_batch_ingestion() -> None:
    """Ingest recent arXiv AI/ML papers into Bronze every day at 6am UTC."""

    @task(task_id="fetch_recent_papers")
    def fetch_recent_papers(categories: list[str] = _ARXIV_CATEGORIES) -> dict[str, str]:
        """Download Atom/XML feed per category from the arXiv API."""
        raw_per_category: dict[str, str] = {}
        for cat in categories:
            params = {
                "search_query": f"cat:{cat}",
                "max_results": ARXIV_SOURCE_CONFIG.batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            url = f"{ARXIV_SOURCE_CONFIG.api_url}?{urlencode(params)}"
            log.info("Fetching arXiv category", extra={"category": cat, "url": url})

            for attempt in range(1, ARXIV_SOURCE_CONFIG.max_retries + 1):
                try:
                    resp = requests.get(url, timeout=60)  # arXiv can be slow
                    resp.raise_for_status()
                    raw_per_category[cat] = resp.text
                    # arXiv rate-limit: 3 second delay between requests
                    time.sleep(3)
                    break
                except requests.RequestException as exc:
                    wait = ARXIV_SOURCE_CONFIG.retry_backoff_base ** attempt
                    log.warning("arXiv fetch failed", extra={"cat": cat, "attempt": attempt, "error": str(exc)})
                    if attempt == ARXIV_SOURCE_CONFIG.max_retries:
                        log.error("Skipping category after max retries", extra={"cat": cat})
                    else:
                        time.sleep(wait)
        return raw_per_category

    @task(task_id="parse_arxiv_xml")
    def parse_arxiv_xml(raw_by_category: dict[str, str]) -> list[dict]:
        """Parse Atom XML for each category and return normalised paper records."""
        papers: list[dict] = []
        for category, xml_text in raw_by_category.items():
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as exc:
                log.error("XML parse error", extra={"category": category, "error": str(exc)})
                continue

            for entry in root.findall("atom:entry", _ARXIV_NAMESPACES):
                def _text(tag: str) -> str | None:
                    el = entry.find(tag, _ARXIV_NAMESPACES)
                    return el.text.strip() if el is not None and el.text else None

                # arxiv_id lives in the <id> element as a URL
                raw_id = _text("atom:id") or ""
                arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

                # Authors: list of <author><name> nodes
                authors = [
                    a.text.strip()
                    for a in entry.findall("atom:author/atom:name", _ARXIV_NAMESPACES)
                    if a.text
                ]

                # Categories: primary from <arxiv:primary_category> + all <category> terms
                primary_cat_el = entry.find("arxiv:primary_category", _ARXIV_NAMESPACES)
                primary_cat = primary_cat_el.attrib.get("term", "") if primary_cat_el is not None else ""
                categories = list({
                    c.attrib.get("term", "")
                    for c in entry.findall("atom:category", _ARXIV_NAMESPACES)
                    if c.attrib.get("term")
                })

                # PDF link
                pdf_url = None
                for link in entry.findall("atom:link", _ARXIV_NAMESPACES):
                    if link.attrib.get("type") == "application/pdf":
                        pdf_url = link.attrib.get("href")

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": (_text("atom:title") or "").replace("\n", " "),
                    "abstract": (_text("atom:summary") or "").replace("\n", " "),
                    "authors": authors,
                    "primary_category": primary_cat,
                    "categories": categories,
                    "published": _text("atom:published"),
                    "updated": _text("atom:updated"),
                    "pdf_url": pdf_url,
                    "entry_url": raw_id,
                    "_fetched_category": category,
                })

        log.info("Parsed arXiv XML", extra={"total_papers": len(papers)})
        return papers

    @task(task_id="deduplicate_papers")
    def deduplicate(papers: list[dict]) -> list[dict]:
        """Remove duplicate arxiv_ids; keep first occurrence (most recently fetched)."""
        seen: set[str] = set()
        unique: list[dict] = []
        for paper in papers:
            aid = paper.get("arxiv_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                unique.append(paper)
        removed = len(papers) - len(unique)
        if removed:
            log.info("Deduplicated arXiv papers", extra={"removed": removed, "kept": len(unique)})
        return unique

    @task(task_id="write_to_bronze")
    def write_to_bronze(papers: list[dict]) -> str:
        """Write deduplicated papers to Bronze as Hive-partitioned NDJSON."""
        if not papers:
            raise ValueError("arxiv write_to_bronze: no papers to write")
        batch_id = uuid.uuid4().hex
        ingested_at = datetime.now(tz=timezone.utc).isoformat()
        for p in papers:
            p["_ingested_at"] = ingested_at
            p["_batch_id"] = batch_id
            p["_source"] = "arxiv"

        client = S3StorageClient(STORAGE_CONFIG)
        now = datetime.now(tz=timezone.utc)
        uri = client.write_json_batch(
            prefix=ARXIV_SOURCE_CONFIG.prefix,
            data=papers,
            partition_keys={
                "year": now.strftime("%Y"),
                "month": now.strftime("%m"),
                "day": now.strftime("%d"),
                "hour": "00",  # daily ingestion — fix hour to 00 for idempotency
            },
            filename=f"arxiv_{batch_id}.ndjson",
        )
        log.info("arXiv Bronze write complete", extra={"uri": uri, "rows": len(papers)})
        return uri

    # ── Wire tasks ────────────────────────────────────────────────────────────
    raw_xml = fetch_recent_papers()
    parsed = parse_arxiv_xml(raw_xml)
    deduped = deduplicate(parsed)
    write_to_bronze(deduped)


try:
    arxiv_dag_instance = arxiv_batch_ingestion()
except Exception as exc:
    log.warning("arXiv ingestion failed, skipping (%s)", exc)
    arxiv_dag_instance = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GITHUB TRENDING DAG
# ─────────────────────────────────────────────────────────────────────────────

_GITHUB_DEFAULT_ARGS = {**_HN_DEFAULT_ARGS, "execution_timeout": timedelta(minutes=20)}
_GH_SEARCH_URL = "https://api.github.com/search/repositories"


@dag(
    dag_id="github_trending_ingest",
    schedule="0 8 * * *",
    start_date=days_ago(1) if _AIRFLOW_AVAILABLE else datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=_GITHUB_DEFAULT_ARGS,
    tags=["aidataboon", "bronze", "github"],
)
def github_trending_ingest() -> None:
    """Ingest GitHub trending AI/ML repos into Bronze every day at 8am UTC."""

    @task(task_id="fetch_trending_repos")
    def fetch_trending_repos() -> list[dict]:
        """
        GitHub Search API: repos recently pushed with highest star velocity.
        Uses date window of last 7 days for 'trending' signal.
        """
        import os

        since_date = (datetime.now(tz=timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        params = {
            "q": f"pushed:>{since_date} topic:machine-learning language:python stars:>100",
            "sort": "stars",
            "order": "desc",
            "per_page": GITHUB_SOURCE_CONFIG.batch_size,
        }
        url = f"{_GH_SEARCH_URL}?{urlencode(params)}"

        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        gh_token = os.getenv("GITHUB_TOKEN")
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
        else:
            log.warning("GITHUB_TOKEN not set; unauthenticated requests are rate-limited to 10/min")

        log.info("Fetching GitHub trending repos", extra={"url": url})
        for attempt in range(1, GITHUB_SOURCE_CONFIG.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=GITHUB_SOURCE_CONFIG.request_timeout,
                )
                resp.raise_for_status()
                items: list[dict] = resp.json().get("items", [])
                log.info("Fetched GitHub repos", extra={"count": len(items)})
                return items
            except requests.RequestException as exc:
                wait = GITHUB_SOURCE_CONFIG.retry_backoff_base ** attempt
                log.warning("GitHub fetch failed", extra={"attempt": attempt, "error": str(exc)})
                if attempt == GITHUB_SOURCE_CONFIG.max_retries:
                    raise RuntimeError(f"GitHub fetch exhausted {attempt} retries") from exc
                time.sleep(wait)
        return []

    @task(task_id="extract_repo_fields")
    def extract_repo_fields(raw_repos: list[dict]) -> list[dict]:
        """Normalise GitHub API response to a flat record per repo."""
        normalised: list[dict] = []
        for repo in raw_repos:
            owner_info = repo.get("owner") or {}
            normalised.append({
                "repo_id": repo.get("id"),
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "watchers": repo.get("watchers_count", 0),
                "open_issues": repo.get("open_issues_count", 0),
                "url": repo.get("html_url"),
                "api_url": repo.get("url"),
                "topics": repo.get("topics", []),
                "owner_login": owner_info.get("login"),
                "owner_type": owner_info.get("type"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "pushed_at": repo.get("pushed_at"),
                "license": (repo.get("license") or {}).get("spdx_id"),
                "default_branch": repo.get("default_branch"),
                "archived": repo.get("archived", False),
            })
        return normalised

    @task(task_id="validate_quality")
    def validate_quality(repos: list[dict]) -> dict:
        """Inline quality check on GitHub records."""
        cfg = QualityConfig(
            min_completeness=0.90,
            max_nulls_pct=0.10,
            required_columns=["repo_id", "full_name", "url"],
        )
        checker = DataQualityChecker(config=cfg)
        result = checker.run_all(
            records=repos,
            expected_schema={"repo_id": int, "full_name": str, "stars": int},
            key_columns=["repo_id"],
            ts_column="pushed_at",
        )
        log.info("GitHub quality result", extra={"passed": result.passed, "score": result.score})
        return {
            "passed": result.passed,
            "score": result.score,
            "violations": result.violations,
            "row_count": result.row_count,
        }

    @task(task_id="write_to_bronze")
    def write_to_bronze(repos: list[dict], quality: dict) -> str:
        """Persist normalised GitHub records to Bronze."""
        if not repos:
            raise ValueError("github write_to_bronze: no repos to write")
        batch_id = uuid.uuid4().hex
        ingested_at = datetime.now(tz=timezone.utc).isoformat()
        for r in repos:
            r["_ingested_at"] = ingested_at
            r["_batch_id"] = batch_id
            r["_source"] = "github_trending"
            r["_quality_score"] = quality["score"]

        client = S3StorageClient(STORAGE_CONFIG)
        now = datetime.now(tz=timezone.utc)
        uri = client.write_json_batch(
            prefix=GITHUB_SOURCE_CONFIG.prefix,
            data=repos,
            partition_keys={
                "year": now.strftime("%Y"),
                "month": now.strftime("%m"),
                "day": now.strftime("%d"),
                "hour": "08",  # daily at 8am; fix hour for idempotency
            },
            filename=f"github_{batch_id}.ndjson",
        )
        log.info("GitHub Bronze write complete", extra={"uri": uri, "rows": len(repos)})

        if not quality["passed"]:
            dlq_uri = client.write_json_batch(
                prefix=f"{GITHUB_SOURCE_CONFIG.dead_letter_prefix}/github",
                data=repos,
                filename=f"dlq_{batch_id}.ndjson",
            )
            log.warning("GitHub batch sent to DLQ", extra={"dlq_uri": dlq_uri})
        return uri

    # ── Wire tasks ────────────────────────────────────────────────────────────
    raw = fetch_trending_repos()
    normalised = extract_repo_fields(raw)
    quality_result = validate_quality(normalised)
    write_to_bronze(normalised, quality_result)


try:
    github_dag_instance = github_trending_ingest()
except Exception as exc:
    log.warning("GitHub trending ingestion failed, skipping (%s)", exc)
    github_dag_instance = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CDC CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PostgresDBConfig:
    """Source PostgreSQL connection config for Debezium CDC."""
    hostname: str
    port: int
    user: str
    password: str
    dbname: str
    server_name: str          # Debezium logical server name (unique per deployment)
    table_include_list: list[str] = field(default_factory=list)
    slot_name: str = "debezium_slot"
    publication_name: str = "dbz_publication"
    schema_registry_url: str = "http://schema-registry:8081"
    kafka_connect_url: str = "http://kafka-connect:8083"


def generate_debezium_connector_config(source_db_config: PostgresDBConfig) -> dict:
    """
    Build the Kafka Connect JSON body for a Debezium PostgreSQL source connector.

    Includes:
    - Logical replication via pgoutput plugin
    - Avro serialisation with Schema Registry
    - Two SMTs: TopicRouting (per-table topics) + ExtractNewRecordState (unwrap envelope)
    - Initial snapshot then ongoing CDC streaming

    Usage:
        config = generate_debezium_connector_config(db_cfg)
        post_connector_to_kafka_connect(config, db_cfg.kafka_connect_url)
    """
    cfg = source_db_config
    connector_name = f"aidataboon-pg-cdc-{cfg.server_name}"

    connector_config: dict[str, str] = {
        # ── Core connector ────────────────────────────────────────────────────
        "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
        "tasks.max": "1",   # PostgreSQL CDC is single-threaded per slot

        # ── Connection ────────────────────────────────────────────────────────
        "database.hostname": cfg.hostname,
        "database.port": str(cfg.port),
        "database.user": cfg.user,
        "database.password": cfg.password,
        "database.dbname": cfg.dbname,
        "database.server.name": cfg.server_name,

        # ── Replication slot & publication ────────────────────────────────────
        "plugin.name": "pgoutput",          # built-in from Postgres 10+
        "slot.name": cfg.slot_name,
        "publication.name": cfg.publication_name,
        "publication.autocreate.mode": "filtered",  # only tables in include list

        # ── Table filter ─────────────────────────────────────────────────────
        "table.include.list": ",".join(cfg.table_include_list) if cfg.table_include_list else "",

        # ── Snapshot ─────────────────────────────────────────────────────────
        "snapshot.mode": "initial",         # snapshot once, then stream WAL
        "snapshot.isolation.mode": "repeatable_read",

        # ── Topic naming ─────────────────────────────────────────────────────
        # Default: <server_name>.<schema>.<table>  → one topic per table

        # ── Serialisation (Avro + Schema Registry) ────────────────────────────
        "key.converter": "io.confluent.kafka.serializers.KafkaAvroSerializer",
        "key.converter.schema.registry.url": cfg.schema_registry_url,
        "value.converter": "io.confluent.kafka.serializers.KafkaAvroSerializer",
        "value.converter.schema.registry.url": cfg.schema_registry_url,

        # ── Single Message Transforms ─────────────────────────────────────────
        # 1. Route each table to its own topic (overrides default naming)
        "transforms": "route,unwrap",

        "transforms.route.type": (
            "org.apache.kafka.connect.transforms.ReplaceField$Value"
        ),
        # 2. ExtractNewRecordState: unwrap Debezium envelope → plain after-image record
        "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
        "transforms.unwrap.drop.tombstones": "false",   # preserve DELETEs as tombstones
        "transforms.unwrap.add.fields": "op,ts_ms,source.ts_ms",  # keep audit fields
        "transforms.unwrap.delete.handling.mode": "rewrite",  # emit __deleted=true

        # ── Heartbeat (detects idle WAL slot lag) ────────────────────────────
        "heartbeat.interval.ms": "30000",
        "heartbeat.action.query": (
            f"INSERT INTO public.debezium_heartbeat (source) VALUES ('{cfg.server_name}') "
            "ON CONFLICT DO NOTHING"
        ),

        # ── Error handling ────────────────────────────────────────────────────
        "errors.tolerance": "none",            # fail fast on malformed events
        "errors.retry.delay.max.ms": "60000",
        "errors.retry.timeout": "300000",
    }

    return {
        "name": connector_name,
        "config": connector_config,
    }


def post_connector_to_kafka_connect(
    connector_payload: dict,
    kafka_connect_url: str,
    update_if_exists: bool = True,
) -> dict:
    """
    POST (or PUT) a Kafka Connect connector config via the REST API.

    Args:
        connector_payload: Dict from generate_debezium_connector_config().
        kafka_connect_url: Base URL of Kafka Connect REST API, e.g. http://kafka-connect:8083
        update_if_exists: If True, use PUT /connectors/{name}/config when connector
                          already exists (idempotent upsert).

    Returns:
        The parsed JSON response from Kafka Connect.

    Raises:
        requests.HTTPError: On non-2xx response (includes 409 Conflict).
    """
    name = connector_payload["name"]
    base = kafka_connect_url.rstrip("/")

    # Check if connector already exists
    check_resp = requests.get(f"{base}/connectors/{name}", timeout=10)
    already_exists = check_resp.status_code == 200

    if already_exists and update_if_exists:
        # PUT /connectors/{name}/config — update only the config, keep status/offsets
        config_only = connector_payload["config"]
        resp = requests.put(
            f"{base}/connectors/{name}/config",
            json=config_only,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
    else:
        # POST /connectors — create new connector
        resp = requests.post(
            f"{base}/connectors",
            json=connector_payload,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )

    resp.raise_for_status()
    log.info(
        "Kafka Connect connector configured",
        extra={"name": name, "status_code": resp.status_code, "action": "update" if already_exists else "create"},
    )
    return resp.json()


def setup_postgres_for_cdc(db_config: PostgresDBConfig) -> list[str]:
    """
    Return the SQL statements that must be run on the source Postgres instance
    to enable logical replication (requires superuser / replication privilege).

    These are idempotent where possible (IF NOT EXISTS / CREATE IF NOT EXISTS).
    """
    tables_sql = ", ".join(db_config.table_include_list) if db_config.table_include_list else "ALL TABLES"
    publication_for = f"FOR TABLE {tables_sql}" if db_config.table_include_list else "FOR ALL TABLES"

    return [
        # 1. Enable logical replication level in postgresql.conf
        "ALTER SYSTEM SET wal_level = logical;",
        "SELECT pg_reload_conf();",

        # 2. Create replication slot (idempotent guard)
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_replication_slots WHERE slot_name = '{db_config.slot_name}'
            ) THEN
                PERFORM pg_create_logical_replication_slot('{db_config.slot_name}', 'pgoutput');
            END IF;
        END;
        $$;
        """,

        # 3. Create publication
        f"CREATE PUBLICATION {db_config.publication_name} {publication_for};",

        # 4. Grant replication permission to the CDC user
        f"GRANT REPLICATION ON DATABASE {db_config.dbname} TO {db_config.user};",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {db_config.user};",

        # 5. Heartbeat table (used by Debezium to track WAL lag on idle DBs)
        """
        CREATE TABLE IF NOT EXISTS public.debezium_heartbeat (
            source TEXT PRIMARY KEY,
            ts     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — STANDALONE DEMO (no Airflow required)
# ─────────────────────────────────────────────────────────────────────────────

def _demo_hn_fetch() -> list[dict]:
    """Fetch HN stories directly (no Airflow) for quick local testing."""
    params = {
        "tags": "story",
        "numericFilters": "points>50",
        "hitsPerPage": 10,  # small batch for demo
    }
    url = f"{HN_SOURCE_CONFIG.api_url}?{urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("hits", [])


def _demo_arxiv_fetch() -> list[dict]:
    """Fetch one arXiv category directly for quick local testing."""
    params = {"search_query": "cat:cs.AI", "max_results": 5, "sortBy": "submittedDate"}
    url = f"{ARXIV_SOURCE_CONFIG.api_url}?{urlencode(params)}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers: list[dict] = []
    for entry in root.findall("atom:entry", _ARXIV_NAMESPACES):
        def _t(tag: str) -> str | None:
            el = entry.find(tag, _ARXIV_NAMESPACES)
            return el.text.strip() if el is not None and el.text else None

        raw_id = _t("atom:id") or ""
        papers.append({
            "arxiv_id": raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id,
            "title": (_t("atom:title") or "").replace("\n", " "),
            "abstract": (_t("atom:summary") or "")[:200] + "...",
            "published": _t("atom:published"),
        })
    return papers


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    print("=" * 64)
    print("AIDataBoon — Data Ingestion Demo (standalone, no Airflow)")
    print("=" * 64)

    # ── HN ────────────────────────────────────────────────────────────────────
    print("\n[1/3] Hacker News — top stories")
    try:
        stories = _demo_hn_fetch()
        checker = DataQualityChecker(config=_HN_QUALITY_CONFIG)
        result = checker.run_all(
            records=stories,
            expected_schema=_HN_EXPECTED_SCHEMA,
            key_columns=["objectID"],
            ts_column="created_at",
        )
        print(f"  Fetched  : {len(stories)} stories")
        print(f"  Quality  : {'PASS' if result.passed else 'FAIL'} (score={result.score:.2f})")
        if result.violations:
            for v in result.violations:
                print(f"  Violation: {v}")
        for s in stories[:3]:
            print(f"  [{s.get('points',0):>4}pts] {s.get('title','')[:72]}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # ── arXiv ─────────────────────────────────────────────────────────────────
    print("\n[2/3] arXiv — recent cs.AI papers")
    try:
        papers = _demo_arxiv_fetch()
        print(f"  Fetched  : {len(papers)} papers")
        for p in papers[:3]:
            print(f"  [{p.get('arxiv_id','')}] {p.get('title','')[:72]}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # ── Debezium CDC config ───────────────────────────────────────────────────
    print("\n[3/3] Debezium CDC connector config (for reference)")
    db_cfg = PostgresDBConfig(
        hostname="postgres",
        port=5432,
        user="replicator",
        password="<secret>",
        dbname="aidataboon",
        server_name="aidataboon-prod",
        table_include_list=["public.users", "public.events"],
        kafka_connect_url="http://kafka-connect:8083",
    )
    connector_cfg = generate_debezium_connector_config(db_cfg)
    print(f"  Connector name : {connector_cfg['name']}")
    print(f"  Slot           : {connector_cfg['config']['slot.name']}")
    print(f"  Publication    : {connector_cfg['config']['publication.name']}")
    print(f"  Snapshot mode  : {connector_cfg['config']['snapshot.mode']}")
    print(f"  Tables         : {connector_cfg['config']['table.include.list']}")

    print("\n  SQL statements to run on Postgres for CDC:")
    for stmt in setup_postgres_for_cdc(db_cfg):
        preview = " ".join(stmt.split())[:80]
        print(f"    {preview}…" if len(preview) >= 80 else f"    {preview}")

    print("\nDemo complete.")


if __name__ == "__main__":
    # Running this file has already ingested HN, arXiv and GitHub into Bronze:
    # without Airflow, the DAG functions above run when the module loads. The
    # standalone demo only re-fetches a few items and prints them, so it runs
    # only on request (python 02_data_ingestion.py --demo).
    if "--demo" in sys.argv[1:]:
        main()
