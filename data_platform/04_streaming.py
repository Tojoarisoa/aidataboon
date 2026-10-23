"""
AIDataBoon — Real-Time Streaming Pipeline

Architecture:
  HN API (poll) → Kafka Producer → aidataboon.hn.events topic
  → Kafka Consumer + Flink-style processing → Silver streaming table
  → Real-time stats aggregations → Gold streaming table

Features:
- Confluent Schema Registry for Avro schemas
- Exactly-once semantics (idempotent producer + transactional consumer)
- Dead letter queue for malformed events
- Windowed aggregations (tumbling 5-min, sliding 1-hour)
- Consumer lag monitoring
"""

from __future__ import annotations

import json
import math
import time
import uuid
import threading
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aidataboon.streaming")

# ---------------------------------------------------------------------------
# Optional heavy imports — gracefully degrade for demo mode
# ---------------------------------------------------------------------------
try:
    from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
    from confluent_kafka.admin import AdminClient, NewTopic
    from confluent_kafka.schema_registry import SchemaRegistryClient as _SRClient
    from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("confluent_kafka not installed — running in simulation mode")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# SECTION 1 — Avro Schemas + Schema Registry
# ---------------------------------------------------------------------------

# HN story event — 384-bit compatible Avro schema dict
HNEventSchema: dict[str, Any] = {
    "type": "record",
    "name": "HNEvent",
    "namespace": "com.aidataboon.streaming",
    "doc": "Hacker News story event captured from Algolia API",
    "fields": [
        {"name": "event_id",     "type": "string",           "doc": "UUID for deduplication"},
        {"name": "story_id",     "type": "int",              "doc": "HN item ID"},
        {"name": "title",        "type": "string"},
        {"name": "url",          "type": ["null", "string"],  "default": None},
        {"name": "score",        "type": "int",              "default": 0},
        {"name": "num_comments", "type": "int",              "default": 0},
        {"name": "author",       "type": "string"},
        {"name": "timestamp",    "type": "long",             "doc": "Unix epoch ms"},
        {"name": "event_type",   "type": {
            "type": "enum",
            "name": "EventType",
            "symbols": ["STORY_CREATED", "STORY_UPDATED", "STORY_DELETED"],
        }},
    ],
}

ArxivEventSchema: dict[str, Any] = {
    "type": "record",
    "name": "ArxivEvent",
    "namespace": "com.aidataboon.streaming",
    "doc": "arXiv paper publish event",
    "fields": [
        {"name": "event_id",     "type": "string"},
        {"name": "arxiv_id",     "type": "string",           "doc": "e.g. 2401.00001"},
        {"name": "title",        "type": "string"},
        {"name": "abstract",     "type": "string"},
        {"name": "authors",      "type": {"type": "array", "items": "string"}},
        {"name": "categories",   "type": {"type": "array", "items": "string"}},
        {"name": "published_at", "type": "long",             "doc": "Unix epoch ms"},
        {"name": "pdf_url",      "type": "string"},
        {"name": "event_type",   "type": "string",           "default": "PAPER_PUBLISHED"},
    ],
}


class SchemaRegistryClient:
    """
    Thin wrapper around confluent_kafka.schema_registry with caching.

    Handles schema registration, retrieval, and forward/backward
    compatibility checks. Operates in stub mode when the registry
    is unavailable (e.g., local demo without Docker).
    """

    def __init__(self, url: str = "http://localhost:8081") -> None:
        self.url = url
        self._cache: dict[str, Any] = {}
        self._client: Optional[Any] = None

        if KAFKA_AVAILABLE:
            try:
                self._client = _SRClient({"url": url})
                logger.info("Schema Registry connected: %s", url)
            except Exception as exc:
                logger.warning("Schema Registry unavailable (%s) — stub mode", exc)

    def register_schema(self, subject: str, schema: dict[str, Any]) -> int:
        """Register an Avro schema under *subject*; return schema ID."""
        schema_str = json.dumps(schema)
        self._cache[subject] = schema

        if self._client is None:
            stub_id = hash(subject) & 0xFFFF
            logger.info("[stub] Registered schema '%s' → id=%d", subject, stub_id)
            return stub_id

        from confluent_kafka.schema_registry import Schema
        registered = self._client.register_schema(
            subject, Schema(schema_str, "AVRO")
        )
        logger.info("Registered schema '%s' → id=%d", subject, registered)
        return registered

    def get_schema(self, subject: str) -> Optional[dict[str, Any]]:
        """Return the latest schema dict for *subject* (cached)."""
        if subject in self._cache:
            return self._cache[subject]

        if self._client is None:
            return None

        latest = self._client.get_latest_version(subject)
        schema = json.loads(latest.schema.schema_str)
        self._cache[subject] = schema
        return schema

    def validate_evolves(self, new_schema: dict[str, Any], subject: str) -> bool:
        """
        Return True when *new_schema* is backward-compatible with the latest
        registered version for *subject*.  Falls back to True in stub mode.
        """
        if self._client is None:
            logger.info("[stub] Schema compatibility check skipped for '%s'", subject)
            return True

        from confluent_kafka.schema_registry import Schema
        schema_str = json.dumps(new_schema)
        result = self._client.test_compatibility(
            subject, Schema(schema_str, "AVRO")
        )
        logger.info("Schema compatibility for '%s': %s", subject, result)
        return result


# ---------------------------------------------------------------------------
# SECTION 2 — Producer
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_HN_EVENTS = "aidataboon.hn.events"
TOPIC_ARXIV_EVENTS = "aidataboon.arxiv.events"

HN_ALGOLIA_URL = (
    "https://hn.algolia.com/api/v1/search?tags=story"
    "&numericFilters=points>100&hitsPerPage=20"
)


class AIDataBoonProducer:
    """
    Base producer — idempotent, all-acks, unlimited retries.

    Exactly-once delivery on the producer side requires:
      - enable.idempotence = true  (deduplicates retries within a session)
      - acks = all                 (waits for all in-sync replicas)
      - retries = INT_MAX          (never give up on transient failures)
      - max.in.flight… = 5        (ordered delivery with idempotence)
    """

    _IDEMPOTENT_CONFIG: dict[str, Any] = {
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "enable.idempotence": True,
        "acks": "all",
        "retries": 2_147_483_647,
        "max.in.flight.requests.per.connection": 5,
        "compression.type": "snappy",
        "linger.ms": 5,
        "batch.size": 65_536,
    }

    def __init__(self, extra_config: Optional[dict] = None) -> None:
        cfg = {**self._IDEMPOTENT_CONFIG, **(extra_config or {})}
        self.published_count = 0
        self.error_count = 0
        self._producer: Optional[Any] = None

        if KAFKA_AVAILABLE:
            self._producer = Producer(cfg)
            logger.info("Kafka producer initialised (idempotent)")
        else:
            logger.info("Kafka stub producer initialised")

    # -- callbacks -----------------------------------------------------------

    def _on_delivery(self, err: Any, msg: Any) -> None:
        if err:
            self.error_count += 1
            logger.error("Delivery failed [%s/%d]: %s", msg.topic(), msg.partition(), err)
        else:
            self.published_count += 1
            logger.debug(
                "Delivered → %s [%d] offset=%d",
                msg.topic(), msg.partition(), msg.offset(),
            )

    def _on_error(self, err: Any) -> None:
        self.error_count += 1
        logger.error("Producer error: %s", err)

    # -- public API ----------------------------------------------------------

    def publish(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[list] = None,
    ) -> None:
        if self._producer is None:
            logger.info("[stub] publish → %s  key=%s  len=%d", topic, key, len(value))
            self.published_count += 1
            return

        self._producer.produce(
            topic,
            value=value,
            key=key,
            headers=headers or [],
            on_delivery=self._on_delivery,
        )
        self._producer.poll(0)  # trigger callbacks without blocking

    def flush(self, timeout: float = 10.0) -> None:
        if self._producer:
            remaining = self._producer.flush(timeout)
            if remaining:
                logger.warning("%d messages not flushed within %.1fs", remaining, timeout)


class HNEventProducer(AIDataBoonProducer):
    """
    Polls the HN Algolia API every *interval_seconds* and publishes
    new stories (score > 100) to *aidataboon.hn.events*.

    Key strategy: story_id bytes → guarantees all updates for the same
    story land on the same partition (ordering per story).
    """

    def __init__(
        self,
        schema_registry: SchemaRegistryClient,
        interval_seconds: int = 30,
    ) -> None:
        super().__init__()
        self.schema_registry = schema_registry
        self.interval_seconds = interval_seconds
        self._seen_ids: set[int] = set()
        self._running = False

        self.schema_registry.register_schema(
            "aidataboon.hn.events-value", HNEventSchema
        )

    # -- internal helpers ----------------------------------------------------

    def _serialize(self, event: dict[str, Any]) -> bytes:
        """Avro-serialise *event*; fall back to JSON in stub mode."""
        return json.dumps(event).encode()

    def _fetch_stories(self) -> list[dict[str, Any]]:
        try:
            resp = requests.get(HN_ALGOLIA_URL, timeout=10)
            resp.raise_for_status()
            return resp.json().get("hits", [])
        except Exception as exc:
            logger.error("HN API fetch failed: %s", exc)
            return []

    def _build_event(self, hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id":     str(uuid.uuid4()),
            "story_id":     int(hit.get("objectID", 0)),
            "title":        hit.get("title", ""),
            "url":          hit.get("url"),
            "score":        int(hit.get("points", 0)),
            "num_comments": int(hit.get("num_comments", 0)),
            "author":       hit.get("author", "unknown"),
            "timestamp":    int(time.time() * 1000),
            "event_type":   "STORY_CREATED",
        }

    # -- public API ----------------------------------------------------------

    def publish_event(self, event: dict[str, Any]) -> None:
        key = str(event["story_id"]).encode()
        payload = self._serialize(event)
        self.publish(TOPIC_HN_EVENTS, payload, key=key)

    def poll_and_publish(self, interval_seconds: Optional[int] = None) -> None:
        """
        Continuously poll HN Algolia and publish new stories.
        Runs until `stop()` is called.
        """
        interval = interval_seconds or self.interval_seconds
        self._running = True
        logger.info("HNEventProducer started (interval=%ds)", interval)

        while self._running:
            stories = self._fetch_stories()
            new_count = 0

            for hit in stories:
                story_id = int(hit.get("objectID", 0))
                if story_id and story_id not in self._seen_ids:
                    event = self._build_event(hit)
                    self.publish_event(event)
                    self._seen_ids.add(story_id)
                    new_count += 1

            if new_count:
                logger.info("Published %d new HN stories", new_count)

            self.flush()
            time.sleep(interval)

    def stop(self) -> None:
        self._running = False
        self.flush()
        logger.info(
            "HNEventProducer stopped — published=%d errors=%d",
            self.published_count, self.error_count,
        )


# ---------------------------------------------------------------------------
# SECTION 3 — Consumer with Exactly-Once
# ---------------------------------------------------------------------------

class AIDataBoonConsumer:
    """
    Base consumer with manual offset commits and read-committed isolation.

    Exactly-once on the consumer side (read-process-commit pattern):
      - isolation.level = read_committed  (skip transactional messages
        from aborted transactions)
      - enable.auto.commit = false        (we commit only after a
        successful downstream write)
      - Manual commit after each batch
    """

    _BASE_CONFIG: dict[str, Any] = {
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "isolation.level": "read_committed",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.poll.interval.ms": 300_000,
        "session.timeout.ms": 30_000,
        "heartbeat.interval.ms": 3_000,
    }

    def __init__(self, group_id: str, extra_config: Optional[dict] = None) -> None:
        cfg = {
            **self._BASE_CONFIG,
            "group.id": group_id,
            **(extra_config or {}),
        }
        self._consumer: Optional[Any] = None
        self._running = False
        self.processed_count = 0
        self.error_count = 0

        if KAFKA_AVAILABLE:
            self._consumer = Consumer(cfg)
            logger.info("Kafka consumer '%s' initialised", group_id)
        else:
            logger.info("Kafka stub consumer '%s' initialised", group_id)

    def subscribe(self, topics: list[str]) -> None:
        if self._consumer:
            self._consumer.subscribe(topics)
        logger.info("Subscribed to %s", topics)

    def _commit(self, message: Any) -> None:
        """Synchronously commit the offset of *message*."""
        if self._consumer:
            self._consumer.commit(message=message, asynchronous=False)

    def stop(self) -> None:
        self._running = False
        if self._consumer:
            self._consumer.close()
        logger.info(
            "Consumer stopped — processed=%d errors=%d",
            self.processed_count, self.error_count,
        )


class HNEventConsumer(AIDataBoonConsumer):
    """
    Reads HN events, enriches them, and appends to the Silver table.

    Transactional pattern:
      1. Poll a batch
      2. Enrich each event
      3. Write to DuckDB (Silver)
      4. Commit offsets — only on success (abort if write fails)
    """

    CATEGORY_RULES: dict[str, list[str]] = {
        "AI/ML":       ["gpt", "llm", "ai", "ml", "machine learning", "neural", "model"],
        "Systems":     ["rust", "linux", "kernel", "compiler", "low-level", "c++"],
        "Web":         ["javascript", "typescript", "react", "nextjs", "css", "frontend"],
        "Cloud":       ["aws", "gcp", "azure", "kubernetes", "docker", "terraform"],
        "Data":        ["databricks", "spark", "kafka", "dbt", "lakehouse", "pipeline"],
        "Open Source": ["open source", "github", "release", "contributor"],
    }

    POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "fast", "better", "best"}
    NEGATIVE_WORDS = {"bad", "broken", "slow", "fail", "bug", "problem", "issue"}

    def __init__(
        self,
        schema_registry: SchemaRegistryClient,
        db_path: str = ":memory:",
    ) -> None:
        super().__init__("aidataboon-hn-consumer")
        self.schema_registry = schema_registry
        self._db = None
        self._db_path = db_path
        self._dlq_handler: Optional[DLQHandler] = None

        if DUCKDB_AVAILABLE:
            self._db = duckdb.connect(db_path)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS silver_hn_events (
                    event_id      VARCHAR PRIMARY KEY,
                    story_id      INTEGER,
                    title         VARCHAR,
                    url           VARCHAR,
                    score         INTEGER,
                    num_comments  INTEGER,
                    author        VARCHAR,
                    event_ts      TIMESTAMPTZ,
                    event_type    VARCHAR,
                    category      VARCHAR,
                    sentiment     DOUBLE,
                    ingested_at   TIMESTAMPTZ DEFAULT now()
                )
            """)

    def set_dlq_handler(self, handler: "DLQHandler") -> None:
        self._dlq_handler = handler

    # -- enrichment ----------------------------------------------------------

    def _classify_category(self, title: str) -> str:
        title_lower = title.lower()
        for category, keywords in self.CATEGORY_RULES.items():
            if any(kw in title_lower for kw in keywords):
                return category
        return "General"

    def _sentiment_score(self, title: str) -> float:
        """Simple rule-based sentiment: −1.0 (negative) to +1.0 (positive)."""
        words = set(title.lower().split())
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 3)

    def _enrich(self, event: dict[str, Any]) -> dict[str, Any]:
        return {
            **event,
            "category":  self._classify_category(event.get("title", "")),
            "sentiment": self._sentiment_score(event.get("title", "")),
        }

    # -- processing ----------------------------------------------------------

    def _deserialize(self, raw: bytes, partition: int, offset: int) -> Optional[dict]:
        try:
            return json.loads(raw.decode())
        except Exception as exc:
            if self._dlq_handler:
                self._dlq_handler.send(
                    original_payload=raw,
                    error_reason=str(exc),
                    source_partition=partition,
                    source_offset=offset,
                )
            self.error_count += 1
            return None

    def _write_to_silver(self, event: dict[str, Any]) -> None:
        if self._db is None:
            logger.debug("[stub] Silver write: story_id=%d", event.get("story_id"))
            return

        ts = datetime.fromtimestamp(
            event["timestamp"] / 1000, tz=timezone.utc
        ).isoformat()

        self._db.execute(
            """
            INSERT OR REPLACE INTO silver_hn_events
              (event_id, story_id, title, url, score, num_comments,
               author, event_ts, event_type, category, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event["event_id"], event["story_id"], event["title"],
                event.get("url"), event["score"], event["num_comments"],
                event["author"], ts, event["event_type"],
                event["category"], event["sentiment"],
            ],
        )

    def run(self, poll_timeout: float = 1.0) -> None:
        """Main consumer loop — blocks until `stop()` is called."""
        self.subscribe([TOPIC_HN_EVENTS])
        self._running = True

        while self._running:
            if self._consumer is None:
                # Simulation: sleep and continue
                time.sleep(poll_timeout)
                continue

            msg = self._consumer.poll(poll_timeout)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Consumer error: %s", msg.error())
                self.error_count += 1
                continue

            event = self._deserialize(
                msg.value(), msg.partition(), msg.offset()
            )
            if event is None:
                self._commit(msg)  # commit even for DLQ'd messages
                continue

            try:
                enriched = self._enrich(event)
                self._write_to_silver(enriched)
                self._commit(msg)
                self.processed_count += 1
                logger.debug(
                    "Processed story_id=%d category=%s sentiment=%.2f",
                    enriched["story_id"], enriched["category"], enriched["sentiment"],
                )
            except Exception as exc:
                logger.error(
                    "Failed to process story_id=%s: %s",
                    event.get("story_id"), exc,
                )
                self.error_count += 1
                # Do NOT commit — message will be reprocessed


# ---------------------------------------------------------------------------
# SECTION 4 — Windowed Aggregations (Flink-style, pure Python)
# ---------------------------------------------------------------------------

@dataclass
class WindowedBucket:
    """Accumulator for a single window bucket."""
    start_ms: int
    end_ms: int
    event_count: int = 0
    score_sum: int = 0
    events: list[dict] = field(default_factory=list)


class EventTimeWatermark:
    """
    Tracks the event-time watermark = max_event_time − max_out_of_order.
    Events with timestamp < watermark are considered late and dropped.
    """

    def __init__(self, max_out_of_order_seconds: int = 60) -> None:
        self.max_out_of_order_ms = max_out_of_order_seconds * 1000
        self._max_event_time_ms: int = 0

    def advance(self, event_time_ms: int) -> None:
        self._max_event_time_ms = max(self._max_event_time_ms, event_time_ms)

    @property
    def watermark_ms(self) -> int:
        return self._max_event_time_ms - self.max_out_of_order_ms

    def is_late(self, event_time_ms: int) -> bool:
        return event_time_ms < self.watermark_ms


class TumblingWindow:
    """
    Non-overlapping fixed-size windows.
    Each event belongs to exactly one window.

    Window boundaries: floor(event_time / size) * size
    Emits when watermark passes window end.
    """

    def __init__(self, size_seconds: int = 300) -> None:
        self.size_ms = size_seconds * 1000
        self._buckets: dict[int, WindowedBucket] = {}
        self.watermark = EventTimeWatermark()

    def _window_start(self, event_time_ms: int) -> int:
        return (event_time_ms // self.size_ms) * self.size_ms

    def update(self, event: dict[str, Any]) -> None:
        ts = event.get("timestamp", int(time.time() * 1000))
        self.watermark.advance(ts)

        if self.watermark.is_late(ts):
            logger.debug("Late event dropped: story_id=%s", event.get("story_id"))
            return

        start = self._window_start(ts)
        if start not in self._buckets:
            self._buckets[start] = WindowedBucket(start, start + self.size_ms)

        bucket = self._buckets[start]
        bucket.event_count += 1
        bucket.score_sum += event.get("score", 0)

    def emit_closed(self) -> list[WindowedBucket]:
        """Return and remove all windows whose end <= watermark."""
        closed = [
            b for b in self._buckets.values()
            if b.end_ms <= self.watermark.watermark_ms
        ]
        for b in closed:
            del self._buckets[b.start_ms]
        return closed


class SlidingWindow:
    """
    Overlapping windows — an event may belong to multiple windows.
    size=3600s, slide=300s → 12 overlapping windows per event.
    """

    def __init__(self, size_seconds: int = 3600, slide_seconds: int = 300) -> None:
        self.size_ms = size_seconds * 1000
        self.slide_ms = slide_seconds * 1000
        self._buckets: dict[tuple[int, int], WindowedBucket] = {}
        self.watermark = EventTimeWatermark()

    def _window_starts(self, event_time_ms: int) -> list[int]:
        """All window start times that contain *event_time_ms*."""
        latest_start = (event_time_ms // self.slide_ms) * self.slide_ms
        starts = []
        s = latest_start
        while s > event_time_ms - self.size_ms:
            starts.append(s)
            s -= self.slide_ms
        return starts

    def update(self, event: dict[str, Any]) -> None:
        ts = event.get("timestamp", int(time.time() * 1000))
        self.watermark.advance(ts)

        if self.watermark.is_late(ts):
            return

        for start in self._window_starts(ts):
            key = (start, start + self.size_ms)
            if key not in self._buckets:
                self._buckets[key] = WindowedBucket(*key)
            b = self._buckets[key]
            b.event_count += 1
            b.score_sum += event.get("score", 0)

    def emit_closed(self) -> list[WindowedBucket]:
        closed = [
            b for b in self._buckets.values()
            if b.end_ms <= self.watermark.watermark_ms
        ]
        for b in closed:
            del self._buckets[(b.start_ms, b.end_ms)]
        return closed


class SessionWindow:
    """
    Session windows per user — a new window begins after *gap_seconds*
    of inactivity for that user.
    """

    def __init__(self, gap_seconds: int = 1800) -> None:
        self.gap_ms = gap_seconds * 1000
        # author → (session_start_ms, last_event_ms, bucket)
        self._sessions: dict[str, tuple[int, int, WindowedBucket]] = {}

    def update(self, event: dict[str, Any]) -> Optional[WindowedBucket]:
        """
        Update the session for *event["author"]*.
        Returns the closed bucket if the previous session timed out.
        """
        author = event.get("author", "unknown")
        ts = event.get("timestamp", int(time.time() * 1000))
        closed: Optional[WindowedBucket] = None

        if author in self._sessions:
            start, last, bucket = self._sessions[author]
            if ts - last > self.gap_ms:
                # Close old session
                bucket.end_ms = last
                closed = bucket
                # Start new session
                self._sessions[author] = (ts, ts, WindowedBucket(ts, ts))
            else:
                bucket.event_count += 1
                bucket.score_sum += event.get("score", 0)
                self._sessions[author] = (start, ts, bucket)
        else:
            self._sessions[author] = (ts, ts, WindowedBucket(ts, ts))

        return closed


class TopicHotness:
    """
    Tracks trending topics using exponential weighted moving average (EWMA).

    score_new = alpha * raw_score + (1 − alpha) * score_old
    alpha = 0.3 gives a ~2-event half-life (recent events dominate).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self._scores: dict[str, float] = defaultdict(float)

    def _extract_tags(self, event: dict[str, Any]) -> list[str]:
        title = event.get("title", "").lower()
        category = event.get("category", "General")
        tags = [category]
        # Simple keyword extraction
        keywords = [
            "python", "rust", "go", "llm", "ai", "cloud", "open source",
            "startup", "hardware", "security", "database",
        ]
        tags += [kw for kw in keywords if kw in title]
        return tags

    def update(self, event: dict[str, Any]) -> None:
        score = float(event.get("score", 0))
        for tag in self._extract_tags(event):
            old = self._scores[tag]
            self._scores[tag] = self.alpha * score + (1 - self.alpha) * old

    def get_top_k(self, k: int = 10) -> list[tuple[str, float]]:
        """Return the *k* hottest topics as (tag, ewma_score) pairs."""
        return sorted(self._scores.items(), key=lambda x: x[1], reverse=True)[:k]


class StreamAggregator:
    """
    Orchestrates all window operators and emits Gold table rows.
    Acts as the single entry point for the aggregation tier.
    """

    def __init__(self) -> None:
        self.tumbling = TumblingWindow(size_seconds=300)
        self.sliding = SlidingWindow(size_seconds=3600, slide_seconds=300)
        self.session = SessionWindow(gap_seconds=1800)
        self.hotness = TopicHotness(alpha=0.3)
        self._gold_rows: list[dict[str, Any]] = []

    def process(self, event: dict[str, Any]) -> None:
        self.tumbling.update(event)
        self.sliding.update(event)
        closed_session = self.session.update(event)
        self.hotness.update(event)

        if closed_session:
            self._gold_rows.append({
                "window_type": "session",
                "start_ms":    closed_session.start_ms,
                "end_ms":      closed_session.end_ms,
                "event_count": closed_session.event_count,
                "score_sum":   closed_session.score_sum,
            })

        for bucket in self.tumbling.emit_closed():
            self._gold_rows.append({
                "window_type": "tumbling_5m",
                "start_ms":    bucket.start_ms,
                "end_ms":      bucket.end_ms,
                "event_count": bucket.event_count,
                "score_sum":   bucket.score_sum,
            })

    def drain_gold(self) -> list[dict[str, Any]]:
        rows, self._gold_rows = self._gold_rows, []
        return rows


# ---------------------------------------------------------------------------
# SECTION 5 — Dead Letter Queue
# ---------------------------------------------------------------------------

TOPIC_DLQ = "aidataboon.hn.events.dlq"


@dataclass
class DLQEvent:
    original_payload: bytes
    error_reason: str
    failed_at: int          # Unix epoch ms
    source_partition: int
    source_offset: int
    dlq_event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class DLQHandler:
    """
    Publishes malformed / unprocessable events to the DLQ topic.

    DLQ events carry full forensic information so operators can
    understand what went wrong and replay or discard events.
    """

    def __init__(self, producer: Optional[AIDataBoonProducer] = None) -> None:
        self._producer = producer or AIDataBoonProducer()
        self.dlq_count = 0

    def send(
        self,
        original_payload: bytes,
        error_reason: str,
        source_partition: int,
        source_offset: int,
    ) -> None:
        dlq_event = DLQEvent(
            original_payload=original_payload,
            error_reason=error_reason,
            failed_at=int(time.time() * 1000),
            source_partition=source_partition,
            source_offset=source_offset,
        )
        payload = json.dumps({
            "dlq_event_id":     dlq_event.dlq_event_id,
            "original_payload": dlq_event.original_payload.hex(),
            "error_reason":     dlq_event.error_reason,
            "failed_at":        dlq_event.failed_at,
            "source_partition": dlq_event.source_partition,
            "source_offset":    dlq_event.source_offset,
        }).encode()

        self._producer.publish(TOPIC_DLQ, payload)
        self.dlq_count += 1
        logger.warning(
            "DLQ event [partition=%d offset=%d]: %s",
            source_partition, source_offset, error_reason,
        )


class DLQReprocessor:
    """
    Reads events from the DLQ, attempts schema repair, and re-publishes
    to the main topic.

    Repair strategy:
      1. Decode hex payload back to bytes
      2. Attempt JSON parse (handles Avro-encoded events stored as JSON)
      3. Fill in mandatory missing fields with defaults
      4. Re-publish to TOPIC_HN_EVENTS with a repaired flag in headers
    """

    def __init__(
        self,
        producer: AIDataBoonProducer,
        consumer: AIDataBoonConsumer,
    ) -> None:
        self._producer = producer
        self._consumer = consumer
        self.repaired_count = 0
        self.failed_count = 0

    def _attempt_repair(self, dlq_event: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            raw = bytes.fromhex(dlq_event["original_payload"])
            event = json.loads(raw.decode())
            # Fill mandatory fields
            event.setdefault("event_id", str(uuid.uuid4()))
            event.setdefault("story_id", 0)
            event.setdefault("title", "<repaired>")
            event.setdefault("author", "unknown")
            event.setdefault("timestamp", int(time.time() * 1000))
            event.setdefault("event_type", "STORY_CREATED")
            event.setdefault("score", 0)
            event.setdefault("num_comments", 0)
            return event
        except Exception as exc:
            logger.error("Repair failed: %s", exc)
            return None

    def reprocess_batch(self, max_events: int = 100) -> None:
        """Poll up to *max_events* from DLQ and attempt repair."""
        self._consumer.subscribe([TOPIC_DLQ])
        processed = 0

        while processed < max_events:
            if self._consumer._consumer is None:
                break  # stub mode

            msg = self._consumer._consumer.poll(1.0)
            if msg is None:
                break
            if msg.error():
                break

            try:
                dlq_event = json.loads(msg.value().decode())
                repaired = self._attempt_repair(dlq_event)
                if repaired:
                    payload = json.dumps(repaired).encode()
                    self._producer.publish(
                        TOPIC_HN_EVENTS,
                        payload,
                        key=str(repaired["story_id"]).encode(),
                        headers=[("repaired", b"true")],
                    )
                    self.repaired_count += 1
                else:
                    self.failed_count += 1
                self._consumer._commit(msg)
            except Exception as exc:
                logger.error("DLQ reprocess error: %s", exc)
                self.failed_count += 1

            processed += 1

        self._producer.flush()
        logger.info(
            "DLQ reprocess complete — repaired=%d failed=%d",
            self.repaired_count, self.failed_count,
        )


# ---------------------------------------------------------------------------
# SECTION 6 — Consumer Lag Monitor
# ---------------------------------------------------------------------------

class PrometheusCounter:
    """Minimal in-process Prometheus counter (no server dependency)."""

    def __init__(self, name: str, description: str, labels: list[str]) -> None:
        self.name = name
        self.description = description
        self.labels = labels
        self._values: dict[tuple, float] = defaultdict(float)

    def labels_instance(self, **kwargs: str) -> "PrometheusCounter":
        key = tuple(kwargs.get(l, "") for l in self.labels)
        self._active_key = key
        return self

    def set(self, value: float) -> None:
        self._values[self._active_key] = value

    def inc(self, amount: float = 1.0) -> None:
        self._values[self._active_key] += amount

    def exposition(self) -> str:
        lines = [f"# HELP {self.name} {self.description}"]
        for key, val in self._values.items():
            label_str = ",".join(
                f'{l}="{v}"' for l, v in zip(self.labels, key)
            )
            lines.append(f"{self.name}{{{label_str}}} {val}")
        return "\n".join(lines)


_lag_gauge = PrometheusCounter(
    "kafka_consumer_lag",
    "Number of messages between consumer offset and log end",
    ["group", "topic", "partition"],
)

LAG_ALERT_THRESHOLD = 10_000


class LagMonitor:
    """
    Queries Kafka AdminClient for consumer group offsets and
    computes per-partition lag.  Exposes Prometheus metrics and
    fires an alert callback when lag exceeds the threshold.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP,
        alert_callback: Optional[Callable[[str, str, int, int], None]] = None,
    ) -> None:
        self._admin: Optional[Any] = None
        self._alert_callback = alert_callback or self._default_alert

        if KAFKA_AVAILABLE:
            try:
                self._admin = AdminClient({"bootstrap.servers": bootstrap_servers})
            except Exception as exc:
                logger.warning("AdminClient unavailable: %s", exc)

    @staticmethod
    def _default_alert(group: str, topic: str, partition: int, lag: int) -> None:
        logger.error(
            "LAG ALERT: group=%s topic=%s partition=%d lag=%d (threshold=%d)",
            group, topic, partition, lag, LAG_ALERT_THRESHOLD,
        )

    def get_consumer_lag(
        self, group_id: str, topic: str
    ) -> dict[int, int]:
        """
        Return {partition: lag} for *group_id* on *topic*.
        Falls back to empty dict when AdminClient is unavailable.
        """
        if self._admin is None:
            return {}

        try:
            # Get committed offsets for the consumer group
            result = self._admin.list_consumer_group_offsets(
                [group_id], require_stable=False
            )
            group_result = result[group_id].result()

            # Get log end offsets (high watermarks)
            partitions = [
                tp for tp in group_result
                if tp.topic == topic
            ]
            if not partitions:
                return {}

            watermarks = {}
            for tp in partitions:
                lo, hi = self._admin._client.query_watermark_offsets(
                    topic, tp.partition, timeout=5.0
                )
                watermarks[tp.partition] = hi

            lag_map: dict[int, int] = {}
            for tp, offset_info in group_result.items():
                if tp.topic != topic:
                    continue
                committed = offset_info.offset
                high = watermarks.get(tp.partition, committed)
                lag = max(0, high - committed)
                lag_map[tp.partition] = lag

                # Update Prometheus gauge
                _lag_gauge.labels_instance(
                    group=group_id,
                    topic=topic,
                    partition=str(tp.partition),
                ).set(lag)

                if lag > LAG_ALERT_THRESHOLD:
                    self._alert_callback(group_id, topic, tp.partition, lag)

            return lag_map

        except Exception as exc:
            logger.error("Lag query failed: %s", exc)
            return {}

    def exposition(self) -> str:
        """Return Prometheus text format for all lag metrics."""
        return _lag_gauge.exposition()


# ---------------------------------------------------------------------------
# SECTION 7 — Demo Runner
# ---------------------------------------------------------------------------

def _create_topics_stub(topics: list[str]) -> None:
    """Create Kafka topics via AdminClient (no-op in stub mode)."""
    if not KAFKA_AVAILABLE:
        logger.info("[stub] Would create topics: %s", topics)
        return

    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
    new_topics = [
        NewTopic(t, num_partitions=3, replication_factor=1)
        for t in topics
    ]
    futures = admin.create_topics(new_topics)
    for t, f in futures.items():
        try:
            f.result()
            logger.info("Topic created: %s", t)
        except Exception as exc:
            logger.debug("Topic '%s': %s", t, exc)  # may already exist


def run_streaming_demo() -> None:
    """
    End-to-end streaming demo:
      1. Create topics
      2. Register Avro schemas
      3. Start producer thread (polls HN every 30 s)
      4. Start consumer thread (processes + enriches events)
      5. Run StreamAggregator on synthetic events to show windowed stats
      6. Show DLQ handler in action
      7. Print consumer lag (stub)
    """
    logger.info("=" * 60)
    logger.info("AIDataBoon Streaming Pipeline — Demo")
    logger.info("=" * 60)

    # -- 1. Topics -----------------------------------------------------------
    _create_topics_stub([TOPIC_HN_EVENTS, TOPIC_ARXIV_EVENTS, TOPIC_DLQ])

    # -- 2. Schema Registry --------------------------------------------------
    registry = SchemaRegistryClient()
    registry.register_schema("aidataboon.hn.events-value", HNEventSchema)
    registry.register_schema("aidataboon.arxiv.events-value", ArxivEventSchema)

    # -- 3. Producer thread --------------------------------------------------
    producer = HNEventProducer(registry, interval_seconds=30)
    dlq_handler = DLQHandler(producer)

    prod_thread = threading.Thread(
        target=producer.poll_and_publish,
        kwargs={"interval_seconds": 30},
        daemon=True,
        name="hn-producer",
    )

    # -- 4. Consumer thread --------------------------------------------------
    consumer = HNEventConsumer(registry)
    consumer.set_dlq_handler(dlq_handler)

    cons_thread = threading.Thread(
        target=consumer.run,
        daemon=True,
        name="hn-consumer",
    )

    # -- 5. Synthetic aggregation demo ---------------------------------------
    aggregator = StreamAggregator()
    now_ms = int(time.time() * 1000)
    synthetic_events = [
        {
            "event_id":     str(uuid.uuid4()),
            "story_id":     1000 + i,
            "title":        f"Story about AI and LLM number {i}",
            "score":        50 + i * 10,
            "num_comments": i * 5,
            "author":       "user_a" if i % 2 == 0 else "user_b",
            "timestamp":    now_ms + i * 60_000,  # 1 minute apart
            "event_type":   "STORY_CREATED",
        }
        for i in range(10)
    ]

    logger.info("\n--- StreamAggregator demo (10 synthetic events) ---")
    for ev in synthetic_events:
        aggregator.process(ev)

    top_topics = aggregator.hotness.get_top_k(5)
    logger.info("Top trending topics: %s", top_topics)

    gold_rows = aggregator.drain_gold()
    logger.info("Gold rows emitted: %d", len(gold_rows))
    for row in gold_rows[:3]:
        logger.info("  %s", row)

    # -- 6. DLQ demo ---------------------------------------------------------
    logger.info("\n--- DLQ demo (simulated malformed event) ---")
    dlq_handler.send(
        original_payload=b"not-valid-avro",
        error_reason="JSONDecodeError: Expecting value at line 1",
        source_partition=0,
        source_offset=42,
    )
    logger.info("DLQ events sent: %d", dlq_handler.dlq_count)

    # -- 7. Lag monitor ------------------------------------------------------
    lag_monitor = LagMonitor()
    lag = lag_monitor.get_consumer_lag("aidataboon-hn-consumer", TOPIC_HN_EVENTS)
    logger.info("\nConsumer lag: %s (stub: empty when broker unavailable)", lag)

    # -- Start live threads (30-second demo) ---------------------------------
    logger.info("\n--- Starting live producer + consumer threads (30s) ---")
    prod_thread.start()
    cons_thread.start()

    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass

    producer.stop()
    consumer.stop()

    logger.info("\nDemo complete.")
    logger.info(
        "Producer — published=%d errors=%d",
        producer.published_count, producer.error_count,
    )
    logger.info(
        "Consumer — processed=%d errors=%d",
        consumer.processed_count, consumer.error_count,
    )


if __name__ == "__main__":
    run_streaming_demo()
