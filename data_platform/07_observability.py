"""
AIDataBoon Observability: Tracing, Metrics, SLOs, Alerts, Health Checks

Stack:
  OpenTelemetry SDK : tracing, exported via OTLP/gRPC to otel-collector:4317
                       (falls back to console export if the OTLP exporter
                       package isn't installed)
  prometheus_client : custom `aidataboon_*` application metrics, exposed in
                       Prometheus exposition format (hand-rolled fallback if
                       prometheus_client isn't installed)
  Prometheus/Grafana: this module is the single source of truth for the
                       SLO table in 00_project_design.md Section 11; it
                       renders the Prometheus scrape config + alerting
                       rules, the OTel Collector pipeline config, and the
                       Grafana datasource/dashboard provisioning + the four
                       dashboards (pipeline, RAG latency, vectors,
                       infrastructure)
  HealthChecker     : Postgres/Redis/Kafka/MinIO/MLflow liveness probes
                       over raw sockets + HTTP, so they work without
                       psycopg/redis-py/kafka-python/boto3 installed

`generate_config_files()` writes the rendered configs under
07_observability/, resolving the bind-mount paths 01_local_stack.yml
expects:
  07_observability/prometheus/prometheus.yml
  07_observability/prometheus/rules/slo_alerts.yml
  07_observability/grafana/provisioning/datasources/datasources.yml
  07_observability/grafana/provisioning/dashboards/dashboards.yml
  07_observability/grafana/dashboards/*.json
  07_observability/otel/otel-collector-config.yml

Run:
  uv run python data_platform/07_observability.py
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aidataboon.observability")

# ---------------------------------------------------------------------------
# Optional heavy dependencies: each degrades to a local fallback so this
# module is always importable and `__main__` always runs end-to-end.
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False
    logger.warning("opentelemetry-sdk not installed: tracing disabled")

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _OTEL_OTLP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_OTLP_AVAILABLE = False
    if _OTEL_AVAILABLE:
        logger.warning("OTLP exporter not installed: tracing falls back to console export")

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed: /metrics falls back to hand-rolled exposition text")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw else default


@dataclass
class ObservabilityConfig:
    """Connection settings for the local stack's observability + stateful
    services. Defaults match the service names / ports in 01_local_stack.yml;
    env var names match what rag-api and feast-server already use
    (PGVECTOR_*, REDIS_*, S3_ENDPOINT_URL, OTEL_*) so the same .env works."""

    otel_service_name: str = field(default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "aidataboon-observability"))
    otel_exporter_endpoint: str = field(default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))

    postgres_host: str = field(default_factory=lambda: os.environ.get("PGVECTOR_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: _env_int("PGVECTOR_PORT", 5432))

    redis_host: str = field(default_factory=lambda: os.environ.get("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))

    kafka_bootstrap: str = field(default_factory=lambda: os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))

    minio_endpoint: str = field(default_factory=lambda: os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"))
    mlflow_url: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    prometheus_url: str = field(default_factory=lambda: os.environ.get("PROMETHEUS_URL", "http://localhost:9090"))
    grafana_url: str = field(default_factory=lambda: os.environ.get("GRAFANA_URL", "http://localhost:3000"))
    jaeger_url: str = field(default_factory=lambda: os.environ.get("JAEGER_QUERY_URL", "http://localhost:16686"))

    config_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "07_observability")

    health_check_timeout: float = 1.5

    @property
    def kafka_host(self) -> str:
        return self.kafka_bootstrap.split(":")[0]

    @property
    def kafka_port(self) -> int:
        parts = self.kafka_bootstrap.split(":")
        return int(parts[1]) if len(parts) > 1 else 9092


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: OPENTELEMETRY TRACING
# ─────────────────────────────────────────────────────────────────────────────

def setup_tracing(config: ObservabilityConfig | None = None) -> "trace.Tracer | None":
    """Configure a global TracerProvider exporting spans via OTLP/gRPC to the
    OTel Collector (which forwards to Jaeger, see otel-collector-config.yml).
    Falls back to console export if the OTLP exporter package or collector
    is unavailable, so spans remain visible during local development.

    Returns None (and logs a warning) if opentelemetry-sdk itself is not
    installed: callers should treat tracing as a no-op in that case.
    """
    if not _OTEL_AVAILABLE:
        return None

    cfg = config or ObservabilityConfig()
    resource = Resource.create({"service.name": cfg.otel_service_name})
    provider = TracerProvider(resource=resource)

    backend = "console-fallback"
    if _OTEL_OTLP_AVAILABLE:
        try:
            exporter = OTLPSpanExporter(endpoint=cfg.otel_exporter_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            backend = f"otlp@{cfg.otel_exporter_endpoint}"
        except Exception as exc:  # pragma: no cover - defensive, e.g. bad endpoint config
            logger.warning("OTLP exporter init failed (%s): falling back to console export", exc)
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    logger.info("Tracing configured: backend=%s service=%s", backend, cfg.otel_service_name)
    return trace.get_tracer(cfg.otel_service_name)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PROMETHEUS METRICS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricSpec:
    name: str
    description: str
    kind: str  # "counter" | "gauge" | "histogram"
    labelnames: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()


# AIDataBoon's custom application metrics. Every PromQL `expr` in SLOS and in
# the generated Grafana dashboards (Section 5) refers to one of these; plus
# `rag_search_requests_total` / `rag_documents_indexed` / `rag_health` from
# 05_rag_service.py and Prometheus's built-in `up`.
METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "aidataboon_ingestion_lag_seconds",
        "Seconds since the last successful ingestion batch, per source",
        "gauge", labelnames=("source",),
    ),
    MetricSpec(
        "aidataboon_transform_duration_seconds",
        "Medallion bronze/silver/gold transform duration",
        "histogram", labelnames=("layer",),
        buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600),
    ),
    MetricSpec(
        "aidataboon_embedding_throughput_docs_per_minute",
        "Embedding batch throughput",
        "gauge",
    ),
    MetricSpec(
        "aidataboon_rag_request_duration_seconds",
        "RAG API request latency",
        "histogram", labelnames=("endpoint",),
        buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    ),
    MetricSpec(
        "aidataboon_vector_index_freshness_seconds",
        "Seconds the vector index lags behind the Gold layer",
        "gauge",
    ),
    MetricSpec(
        "aidataboon_kafka_consumer_lag_messages",
        "Kafka consumer group lag, in messages",
        "gauge", labelnames=("topic", "group"),
    ),
    MetricSpec(
        "aidataboon_pipeline_runs_total",
        "Pipeline run outcomes",
        "counter", labelnames=("pipeline", "status"),
    ),
)


class MetricsRegistry:
    """AIDataBoon's custom Prometheus metrics (METRIC_SPECS / SLOS).

    Backed by real `prometheus_client` collectors when available; otherwise
    an in-memory store rendered as exposition-format text; the same
    fallback 05_rag_service.py uses for its `/metrics` endpoint. Either way,
    `.render()` returns `(body, content_type)` ready for an HTTP response.
    """

    def __init__(self, specs: tuple[MetricSpec, ...] = METRIC_SPECS) -> None:
        self.specs: dict[str, MetricSpec] = {s.name: s for s in specs}
        self.backend = "prometheus_client" if _PROMETHEUS_AVAILABLE else "text-fallback"
        self._collectors: dict[str, Any] = {}
        self._state: dict[str, dict[tuple[str, ...], dict[str, float]]] = {s.name: {} for s in specs}

        if _PROMETHEUS_AVAILABLE:
            self.registry: Any = CollectorRegistry()
            for s in specs:
                if s.kind == "counter":
                    self._collectors[s.name] = Counter(s.name, s.description, s.labelnames, registry=self.registry)
                elif s.kind == "gauge":
                    self._collectors[s.name] = Gauge(s.name, s.description, s.labelnames, registry=self.registry)
                else:
                    self._collectors[s.name] = Histogram(
                        s.name, s.description, s.labelnames,
                        buckets=s.buckets, registry=self.registry,
                    )
        else:
            self.registry = None

    # -- low-level ----------------------------------------------------------
    @staticmethod
    def _key(spec: MetricSpec, labels: dict[str, str]) -> tuple[str, ...]:
        return tuple(str(labels.get(name, "")) for name in spec.labelnames)

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        spec = self.specs[name]
        if self.backend == "prometheus_client":
            collector = self._collectors[name]
            (collector.labels(**labels) if spec.labelnames else collector).set(value)
        else:
            self._state[name][self._key(spec, labels)] = {"value": float(value)}

    def inc_counter(self, name: str, amount: float = 1.0, **labels: str) -> None:
        spec = self.specs[name]
        if self.backend == "prometheus_client":
            collector = self._collectors[name]
            (collector.labels(**labels) if spec.labelnames else collector).inc(amount)
        else:
            bucket = self._state[name].setdefault(self._key(spec, labels), {"value": 0.0})
            bucket["value"] += amount

    def observe(self, name: str, value: float, **labels: str) -> None:
        spec = self.specs[name]
        if self.backend == "prometheus_client":
            collector = self._collectors[name]
            (collector.labels(**labels) if spec.labelnames else collector).observe(value)
        else:
            bucket = self._state[name].setdefault(self._key(spec, labels), {"sum": 0.0, "count": 0.0})
            bucket["sum"] += value
            bucket["count"] += 1

    # -- AIDataBoon-specific convenience wrappers (mirror the SLO table) ------
    def record_ingestion_lag(self, source: str, seconds: float) -> None:
        self.set_gauge("aidataboon_ingestion_lag_seconds", seconds, source=source)

    def record_transform_duration(self, layer: str, seconds: float) -> None:
        self.observe("aidataboon_transform_duration_seconds", seconds, layer=layer)

    def record_embedding_throughput(self, docs_per_minute: float) -> None:
        self.set_gauge("aidataboon_embedding_throughput_docs_per_minute", docs_per_minute)

    def record_rag_request(self, endpoint: str, seconds: float) -> None:
        self.observe("aidataboon_rag_request_duration_seconds", seconds, endpoint=endpoint)

    def record_vector_index_freshness(self, seconds: float) -> None:
        self.set_gauge("aidataboon_vector_index_freshness_seconds", seconds)

    def record_kafka_consumer_lag(self, topic: str, group: str, lag: int) -> None:
        self.set_gauge("aidataboon_kafka_consumer_lag_messages", lag, topic=topic, group=group)

    def record_pipeline_run(self, pipeline: str, success: bool) -> None:
        self.inc_counter("aidataboon_pipeline_runs_total", pipeline=pipeline, status="success" if success else "failure")

    # -- exposition -----------------------------------------------------------
    def render(self) -> tuple[bytes, str]:
        """Render Prometheus exposition format for a `/metrics` endpoint."""
        if self.backend == "prometheus_client":
            return generate_latest(self.registry), CONTENT_TYPE_LATEST

        lines: list[str] = []
        for name, spec in self.specs.items():
            lines.append(f"# HELP {name} {spec.description}")
            lines.append(f"# TYPE {name} {spec.kind}")
            for label_values, values in self._state[name].items():
                label_str = ""
                if spec.labelnames:
                    pairs = ",".join(f'{k}="{v}"' for k, v in zip(spec.labelnames, label_values))
                    label_str = "{" + pairs + "}"
                if spec.kind == "histogram":
                    lines.append(f"{name}_sum{label_str} {values['sum']}")
                    lines.append(f"{name}_count{label_str} {int(values['count'])}")
                else:
                    lines.append(f"{name}{label_str} {values['value']}")
        return ("\n".join(lines) + "\n").encode("utf-8"), "text/plain; version=0.0.4"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SLOs AND ERROR BUDGETS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SLO:
    """One row of 00_project_design.md Section 11's SLO table, with a PromQL
    `expr` that evaluates to the metric's current value."""

    name: str
    description: str
    expr: str
    target: float
    alert_threshold: float
    unit: str
    comparison: str  # "lte" (lower is better) | "gte" (higher is better)


SLOS: tuple[SLO, ...] = (
    SLO(
        "hn_ingestion_lag", "Hacker News ingestion lag",
        'aidataboon_ingestion_lag_seconds{source="hackernews"}',
        target=300, alert_threshold=1800, unit="seconds", comparison="lte",
    ),
    SLO(
        "arxiv_ingestion_lag", "arXiv ingestion lag",
        'aidataboon_ingestion_lag_seconds{source="arxiv"}',
        target=3600, alert_threshold=14400, unit="seconds", comparison="lte",
    ),
    SLO(
        "bronze_to_silver_duration", "Bronze→Silver transform duration (p95)",
        'histogram_quantile(0.95, sum(rate(aidataboon_transform_duration_seconds_bucket{layer="silver"}[15m])) by (le))',
        target=120, alert_threshold=600, unit="seconds", comparison="lte",
    ),
    SLO(
        "silver_to_gold_duration", "Silver→Gold transform duration (p95)",
        'histogram_quantile(0.95, sum(rate(aidataboon_transform_duration_seconds_bucket{layer="gold"}[15m])) by (le))',
        target=300, alert_threshold=1200, unit="seconds", comparison="lte",
    ),
    SLO(
        "embedding_throughput", "Embedding batch throughput",
        "aidataboon_embedding_throughput_docs_per_minute",
        target=500, alert_threshold=100, unit="docs/minute", comparison="gte",
    ),
    SLO(
        "rag_p50_latency", "RAG /search p50 latency",
        'histogram_quantile(0.50, sum(rate(aidataboon_rag_request_duration_seconds_bucket{endpoint="/search"}[5m])) by (le))',
        target=0.2, alert_threshold=0.5, unit="seconds", comparison="lte",
    ),
    SLO(
        "rag_p99_latency", "RAG /search p99 latency",
        'histogram_quantile(0.99, sum(rate(aidataboon_rag_request_duration_seconds_bucket{endpoint="/search"}[5m])) by (le))',
        target=2.0, alert_threshold=5.0, unit="seconds", comparison="lte",
    ),
    SLO(
        "vector_index_freshness", "Vector index freshness vs. Gold",
        "aidataboon_vector_index_freshness_seconds",
        target=1800, alert_threshold=7200, unit="seconds", comparison="lte",
    ),
    SLO(
        "kafka_consumer_lag", "Kafka consumer lag",
        "max(aidataboon_kafka_consumer_lag_messages)",
        target=1000, alert_threshold=10000, unit="messages", comparison="lte",
    ),
    SLO(
        "pipeline_success_rate", "Pipeline success rate (1h)",
        'sum(rate(aidataboon_pipeline_runs_total{status="success"}[1h])) / sum(rate(aidataboon_pipeline_runs_total[1h]))',
        target=0.99, alert_threshold=0.95, unit="ratio", comparison="gte",
    ),
)

_COMPARISON_OPERATORS = {"lte": "<=", "gte": ">="}
_BREACH_OPERATORS = {"lte": ">", "gte": "<"}


@dataclass
class ErrorBudgetStatus:
    slo: SLO
    current_value: float
    within_target: bool
    budget_remaining: float  # 1.0 = at/better than target, 0.0 = at/past alert_threshold
    severity: str  # "ok" | "warning" | "breached"


def evaluate_error_budget(slo: SLO, current_value: float) -> ErrorBudgetStatus:
    """Locate `current_value` between `slo.target` (full budget) and
    `slo.alert_threshold` (budget exhausted / breached)."""
    span = abs(slo.alert_threshold - slo.target)
    if slo.comparison == "lte":
        within_target = current_value <= slo.target
        breached = current_value >= slo.alert_threshold
        remaining = (slo.alert_threshold - current_value) / span if span else 1.0
    else:
        within_target = current_value >= slo.target
        breached = current_value <= slo.alert_threshold
        remaining = (current_value - slo.alert_threshold) / span if span else 1.0

    severity = "breached" if breached else ("ok" if within_target else "warning")
    return ErrorBudgetStatus(
        slo=slo,
        current_value=current_value,
        within_target=within_target,
        budget_remaining=max(0.0, min(1.0, remaining)),
        severity=severity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CONFIG FILE GENERATORS (Prometheus / Grafana / OTel Collector)
# ─────────────────────────────────────────────────────────────────────────────

_GENERATED_HEADER = "# Generated by 07_observability.py: do not edit by hand.\n"


def generate_prometheus_config(config: ObservabilityConfig | None = None) -> str:
    """Render prometheus.yml: scrape jobs for Prometheus itself, the RAG API
    (`/metrics`), and the OTel Collector's Prometheus exporter (pipeline
    metrics on :8889) and self-telemetry (:8888, enabled by the OTel
    Collector config below). Loads alerting rules from
    07_observability/prometheus/rules/*.yml: Prometheus evaluates and
    displays these in its Alerts UI even with no Alertmanager deployed;
    add an `alerting:` block here once one is."""
    doc = {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s",
            "external_labels": {"project": "aidataboon", "environment": "local"},
        },
        "rule_files": ["/etc/prometheus/rules/*.yml"],
        "scrape_configs": [
            {"job_name": "prometheus", "static_configs": [{"targets": ["localhost:9090"]}]},
            {"job_name": "rag-api", "metrics_path": "/metrics", "static_configs": [{"targets": ["rag-api:8000"]}]},
            {"job_name": "otel-collector", "static_configs": [{"targets": ["otel-collector:8889"]}]},
            {"job_name": "otel-collector-self", "static_configs": [{"targets": ["otel-collector:8888"]}]},
        ],
    }
    return _GENERATED_HEADER + yaml.dump(doc, sort_keys=False, default_flow_style=False)


def generate_alert_rules(slos: tuple[SLO, ...] = SLOS) -> str:
    """Render Prometheus alerting rules (slo_alerts.yml) from SLOS."""
    rules = []
    for slo in slos:
        breach_op = _BREACH_OPERATORS[slo.comparison]
        alert_name = "AIDataBoon" + "".join(part.capitalize() for part in slo.name.split("_")) + "Breach"
        rules.append({
            "alert": alert_name,
            "expr": f"{slo.expr} {breach_op} {slo.alert_threshold}",
            "for": "10m",
            "labels": {"severity": "warning", "slo": slo.name},
            "annotations": {
                "summary": f"{slo.description} is breaching its SLO",
                "description": (
                    f"{slo.description} = {{{{ $value }}}} {slo.unit} "
                    f"(target {_COMPARISON_OPERATORS[slo.comparison]} {slo.target} {slo.unit}, "
                    f"alert when {breach_op} {slo.alert_threshold} {slo.unit})"
                ),
            },
        })
    doc = {"groups": [{"name": "aidataboon_slos", "interval": "1m", "rules": rules}]}
    return _GENERATED_HEADER + yaml.dump(doc, sort_keys=False, default_flow_style=False)


def generate_otel_collector_config() -> str:
    """Render otel-collector-config.yml: OTLP receivers (gRPC :4317, HTTP
    :4318) → batch processor → Prometheus exporter (:8889, scraped by
    Prometheus) for metrics, OTLP export to Jaeger for traces. Self-telemetry
    is exposed on :8888 for the `otel-collector-self` Prometheus job."""
    doc = {
        "receivers": {
            "otlp": {
                "protocols": {
                    "grpc": {"endpoint": "0.0.0.0:4317"},
                    "http": {"endpoint": "0.0.0.0:4318"},
                },
            },
        },
        "processors": {
            "batch": {"timeout": "5s", "send_batch_size": 1024},
        },
        "exporters": {
            "prometheus": {
                "endpoint": "0.0.0.0:8889",
                "namespace": "aidataboon",
                "const_labels": {"project": "aidataboon"},
            },
            "otlp/jaeger": {
                "endpoint": "jaeger:4317",
                "tls": {"insecure": True},
            },
            "debug": {"verbosity": "basic"},
        },
        "extensions": {
            "health_check": {"endpoint": "0.0.0.0:13133"},
        },
        "service": {
            "extensions": ["health_check"],
            "telemetry": {"metrics": {"address": "0.0.0.0:8888"}},
            "pipelines": {
                "traces": {"receivers": ["otlp"], "processors": ["batch"], "exporters": ["otlp/jaeger", "debug"]},
                "metrics": {"receivers": ["otlp"], "processors": ["batch"], "exporters": ["prometheus"]},
                "logs": {"receivers": ["otlp"], "processors": ["batch"], "exporters": ["debug"]},
            },
        },
    }
    return _GENERATED_HEADER + yaml.dump(doc, sort_keys=False, default_flow_style=False)


def generate_grafana_datasources() -> str:
    """Render the Grafana provisioning datasources.yml: Prometheus (default)
    + Jaeger, both reached over the aidataboon-net Docker network."""
    doc = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": "http://prometheus:9090",
                "isDefault": True,
                "editable": False,
            },
            {
                "name": "Jaeger",
                "type": "jaeger",
                "access": "proxy",
                "url": "http://jaeger:16686",
                "editable": False,
            },
        ],
    }
    return _GENERATED_HEADER + yaml.dump(doc, sort_keys=False, default_flow_style=False)


def generate_grafana_dashboard_provider() -> str:
    """Render the Grafana provisioning dashboards.yml, pointing at the
    dashboard JSON files written to grafana/dashboards/."""
    doc = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "aidataboon",
                "orgId": 1,
                "folder": "AIDataBoon",
                "type": "file",
                "disableDeletion": False,
                "allowUiUpdates": True,
                "updateIntervalSeconds": 30,
                "options": {"path": "/var/lib/grafana/dashboards"},
            },
        ],
    }
    return _GENERATED_HEADER + yaml.dump(doc, sort_keys=False, default_flow_style=False)


def _panel(
    panel_id: int,
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    unit: str = "short",
    panel_type: str = "timeseries",
    legend: str = "",
) -> dict[str, Any]:
    return {
        "id": panel_id,
        "title": title,
        "type": panel_type,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [{"expr": expr, "legendFormat": legend, "refId": "A"}],
    }


def _dashboard(uid: str, title: str, panels: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "uid": uid,
        "title": title,
        "tags": ["aidataboon"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "editable": True,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "panels": panels,
    }


def _dashboard_pipeline_health() -> dict[str, Any]:
    return _dashboard("aidataboon-pipeline-health", "AIDataBoon / Pipeline Health", [
        _panel(1, "Ingestion lag by source", "aidataboon_ingestion_lag_seconds", 0, 0, unit="s", legend="{{source}}"),
        _panel(
            2, "Transform duration p95 by layer",
            "histogram_quantile(0.95, sum(rate(aidataboon_transform_duration_seconds_bucket[15m])) by (le, layer))",
            12, 0, unit="s", legend="{{layer}}",
        ),
        _panel(
            3, "Pipeline success rate (1h)",
            'sum(rate(aidataboon_pipeline_runs_total{status="success"}[1h])) / sum(rate(aidataboon_pipeline_runs_total[1h]))',
            0, 8, unit="percentunit",
        ),
        _panel(
            4, "Pipeline runs by status (rate)",
            "sum by (pipeline, status) (rate(aidataboon_pipeline_runs_total[15m]))",
            12, 8, legend="{{pipeline}} / {{status}}",
        ),
    ])


def _dashboard_rag_latency() -> dict[str, Any]:
    return _dashboard("aidataboon-rag-latency", "AIDataBoon / RAG API Latency", [
        _panel(
            1, "Search latency p50",
            'histogram_quantile(0.50, sum(rate(aidataboon_rag_request_duration_seconds_bucket{endpoint="/search"}[5m])) by (le))',
            0, 0, unit="s",
        ),
        _panel(
            2, "Search latency p99",
            'histogram_quantile(0.99, sum(rate(aidataboon_rag_request_duration_seconds_bucket{endpoint="/search"}[5m])) by (le))',
            12, 0, unit="s",
        ),
        _panel(
            3, "Search error rate",
            'sum(rate(rag_search_requests_total{status="error"}[5m])) / sum(rate(rag_search_requests_total[5m]))',
            0, 8, unit="percentunit",
        ),
        _panel(
            4, "Search throughput (req/s)",
            "sum(rate(rag_search_requests_total[5m]))",
            12, 8,
        ),
    ])


def _dashboard_vector_index() -> dict[str, Any]:
    return _dashboard("aidataboon-vector-index", "AIDataBoon / Embeddings & Vector Index", [
        _panel(1, "Documents indexed", "rag_documents_indexed", 0, 0, panel_type="stat"),
        _panel(2, "Embedding throughput (docs/min)", "aidataboon_embedding_throughput_docs_per_minute", 12, 0),
        _panel(3, "RAG pipeline health", "rag_health", 0, 8, panel_type="stat"),
        _panel(
            4, "Search requests by status (rate)",
            "sum by (status) (rate(rag_search_requests_total[5m]))",
            12, 8, legend="{{status}}",
        ),
    ])


def _dashboard_infrastructure() -> dict[str, Any]:
    return _dashboard("aidataboon-infrastructure", "AIDataBoon / Infrastructure", [
        _panel(1, "Scrape target health (up)", "up", 0, 0, panel_type="stat", legend="{{job}}"),
        _panel(
            2, "Kafka consumer lag by topic",
            "aidataboon_kafka_consumer_lag_messages",
            12, 0, legend="{{topic}} / {{group}}",
        ),
        _panel(
            3, "Pipeline runs (success vs failure, 1h)",
            "sum by (status) (increase(aidataboon_pipeline_runs_total[1h]))",
            0, 8, legend="{{status}}",
        ),
        _panel(4, "Vector index freshness", "aidataboon_vector_index_freshness_seconds", 12, 8, unit="s"),
    ])


def generate_grafana_dashboards() -> dict[str, dict[str, Any]]:
    """Render the four dashboards called out in 00_project_design.md:
    pipeline health, RAG latency, vector index/embeddings, and
    infrastructure."""
    return {
        "pipeline_health.json": _dashboard_pipeline_health(),
        "rag_latency.json": _dashboard_rag_latency(),
        "vector_index.json": _dashboard_vector_index(),
        "infrastructure.json": _dashboard_infrastructure(),
    }


def generate_config_files(
    config: ObservabilityConfig | None = None,
    base_dir: Path | None = None,
) -> list[Path]:
    """Write the Prometheus/Grafana/OTel config files that 01_local_stack.yml
    bind-mounts read-only under 07_observability/. Safe to re-run: each file
    is freshly rendered from SLOS / METRIC_SPECS, so editing those and
    re-running this keeps dashboards, alerts, and scrape configs in sync."""
    cfg = config or ObservabilityConfig()
    base = base_dir or cfg.config_dir

    files: dict[Path, str] = {
        base / "prometheus" / "prometheus.yml": generate_prometheus_config(cfg),
        base / "prometheus" / "rules" / "slo_alerts.yml": generate_alert_rules(),
        base / "grafana" / "provisioning" / "datasources" / "datasources.yml": generate_grafana_datasources(),
        base / "grafana" / "provisioning" / "dashboards" / "dashboards.yml": generate_grafana_dashboard_provider(),
        base / "otel" / "otel-collector-config.yml": generate_otel_collector_config(),
    }
    for filename, dashboard in generate_grafana_dashboards().items():
        files[base / "grafana" / "dashboards" / filename] = json.dumps(dashboard, indent=2) + "\n"

    written: list[Path] = []
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(path)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: HEALTH CHECKS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServiceHealth:
    name: str
    status: str  # "ok" | "down"
    latency_ms: float | None
    detail: str = ""


class HealthChecker:
    """Liveness probes for the local stack's stateful services.

    Uses raw sockets and HTTP rather than service-specific clients
    (psycopg, redis-py, kafka-python, boto3/minio) so it works even when
    those optional dependencies aren't installed: the same constraint
    05_rag_service.py and 03_lakehouse.py work under."""

    def __init__(self, config: ObservabilityConfig | None = None) -> None:
        self.config = config or ObservabilityConfig()

    def _tcp_check(self, name: str, host: str, port: int, detail: str = "") -> ServiceHealth:
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=self.config.health_check_timeout):
                pass
            latency_ms = (time.perf_counter() - t0) * 1000
            return ServiceHealth(name, "ok", latency_ms, detail or f"tcp connect {host}:{port}")
        except OSError as exc:
            return ServiceHealth(name, "down", None, str(exc))

    def _http_check(self, name: str, url: str) -> ServiceHealth:
        t0 = time.perf_counter()
        try:
            resp = httpx.get(url, timeout=self.config.health_check_timeout)
            latency_ms = (time.perf_counter() - t0) * 1000
            status = "ok" if resp.status_code < 400 else "down"
            return ServiceHealth(name, status, latency_ms, f"GET {url} -> HTTP {resp.status_code}")
        except httpx.HTTPError as exc:
            return ServiceHealth(name, "down", None, f"GET {url} -> {exc}")

    def check_postgres(self) -> ServiceHealth:
        return self._tcp_check("postgres", self.config.postgres_host, self.config.postgres_port)

    def check_redis(self) -> ServiceHealth:
        """Speak the Redis (RESP) protocol directly: send PING, expect +PONG."""
        host, port = self.config.redis_host, self.config.redis_port
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=self.config.health_check_timeout) as sock:
                sock.sendall(b"PING\r\n")
                reply = sock.recv(64)
            latency_ms = (time.perf_counter() - t0) * 1000
            ok = reply.startswith(b"+PONG")
            return ServiceHealth("redis", "ok" if ok else "down", latency_ms, reply.decode(errors="replace").strip())
        except OSError as exc:
            return ServiceHealth("redis", "down", None, str(exc))

    def check_kafka(self) -> ServiceHealth:
        return self._tcp_check(
            "kafka", self.config.kafka_host, self.config.kafka_port,
            detail="tcp connect only: broker API check needs a Kafka client library",
        )

    def check_minio(self) -> ServiceHealth:
        return self._http_check("minio", f"{self.config.minio_endpoint}/minio/health/live")

    def check_mlflow(self) -> ServiceHealth:
        return self._http_check("mlflow", f"{self.config.mlflow_url}/health")

    def run_all(self) -> dict[str, Any]:
        checks = [
            self.check_postgres(),
            self.check_redis(),
            self.check_kafka(),
            self.check_minio(),
            self.check_mlflow(),
        ]
        overall = "ok" if all(c.status == "ok" for c in checks) else "degraded"
        return {"status": overall, "services": {c.name: asdict(c) for c in checks}}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config = ObservabilityConfig()

    print("=== AIDataBoon Observability: Demo ===")

    # 1. Generate Prometheus/Grafana/OTel config files -----------------------
    written = generate_config_files(config)
    print(f"\nGenerated {len(written)} config file(s) under {config.config_dir}/:")
    for path in written:
        print(f"  - {path.relative_to(config.config_dir.parent)}")

    # 2. Tracing demo ----------------------------------------------------------
    tracer = setup_tracing(config)
    if tracer is not None:
        with tracer.start_as_current_span("rag.search") as span:
            span.set_attribute("query", "hybrid search demo")
            with tracer.start_as_current_span("vector_store.hybrid_search"):
                time.sleep(0.01)
            with tracer.start_as_current_span("reranker.rerank"):
                time.sleep(0.005)
        backend = f"otlp@{config.otel_exporter_endpoint}" if _OTEL_OTLP_AVAILABLE else "console-fallback"
        print(f"\nTracing backend: {backend} (3 spans emitted for a demo 'rag.search' trace)")
    else:
        print("\nTracing: skipped (opentelemetry-sdk not installed)")

    # 3. Metrics demo ------------------------------------------------------------
    metrics = MetricsRegistry()
    metrics.record_ingestion_lag("hackernews", 145.0)
    metrics.record_ingestion_lag("arxiv", 1800.0)
    metrics.record_transform_duration("silver", 38.2)
    metrics.record_transform_duration("gold", 95.4)
    metrics.record_embedding_throughput(620.0)
    metrics.record_rag_request("/search", 0.18)
    metrics.record_rag_request("/search", 0.42)
    metrics.record_vector_index_freshness(540.0)
    metrics.record_kafka_consumer_lag("hn-stories", "rag-indexer", 230)
    metrics.record_pipeline_run("bronze_to_silver", success=True)
    metrics.record_pipeline_run("bronze_to_silver", success=True)
    metrics.record_pipeline_run("silver_to_gold", success=False)

    body, content_type = metrics.render()
    print(f"\nMetrics backend: {metrics.backend}")
    print(f"/metrics ({content_type}): first lines:")
    for line in body.decode().splitlines()[:14]:
        print(f"  {line}")

    # 4. SLO / error-budget demo ---------------------------------------------
    print("\nSLO error-budget snapshot (synthetic current values):")
    sample_values = {
        "hn_ingestion_lag": 145.0,
        "arxiv_ingestion_lag": 1800.0,
        "bronze_to_silver_duration": 38.2,
        "silver_to_gold_duration": 1500.0,
        "embedding_throughput": 620.0,
        "rag_p50_latency": 0.18,
        "rag_p99_latency": 1.4,
        "vector_index_freshness": 540.0,
        "kafka_consumer_lag": 230.0,
        "pipeline_success_rate": 0.97,
    }
    for slo in SLOS:
        status = evaluate_error_budget(slo, sample_values[slo.name])
        op = _COMPARISON_OPERATORS[slo.comparison]
        print(
            f"  {slo.name:28s} value={status.current_value:<8g} "
            f"target {op} {slo.target:<6g} {slo.unit:<10s} "
            f"budget_remaining={status.budget_remaining:5.0%}  [{status.severity}]"
        )

    # 5. Health checks -----------------------------------------------------------
    print("\nHealth checks (local stack: expect 'down' if it isn't running):")
    health = HealthChecker(config).run_all()
    print(f"  overall: {health['status']}")
    for name, info in health["services"].items():
        latency = f"{info['latency_ms']:.1f}ms" if info["latency_ms"] is not None else "-"
        print(f"  - {name:10s} {info['status']:5s} {latency:>8s}  {info['detail']}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
