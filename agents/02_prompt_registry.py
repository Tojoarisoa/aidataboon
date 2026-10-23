"""
AIDataBoon Agents: Prompt Registry

Versioned prompt storage with deterministic A/B traffic-splitting and a
small FastAPI admin sub-app (mounted at /prompts by 06_agent_api.py).

Implements ADR-1 (00_project_design.md): prompts live in a `prompt_versions`
Postgres table when available, falling back to an in-memory dict seeded from
DEFAULT_PROMPTS so local dev / CI work without a database.

Standalone demo:
    uv run python agents/02_prompt_registry.py
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aidataboon.prompts")

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Json

    _PSYCOPG_AVAILABLE = True
except ImportError:
    _PSYCOPG_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - pydantic/fastapi import edge cases
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PromptRegistryConfig:
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://aidataboon:aidataboon@localhost:5432/aidataboon"
        )
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

PROMPT_REGISTRY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    template    TEXT NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT FALSE,
    traffic_pct INTEGER NOT NULL DEFAULT 100,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (name, version)
);
CREATE INDEX IF NOT EXISTS idx_prompt_versions_name ON prompt_versions (name);
"""


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class PromptVersion:
    name: str
    version: int
    template: str
    is_active: bool = False
    traffic_pct: int = 100
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables (str.format style)."""
        return self.template.format(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "is_active": self.is_active,
            "traffic_pct": self.traffic_pct,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


def _row_to_prompt_version(row: dict[str, Any]) -> PromptVersion:
    return PromptVersion(
        name=row["name"],
        version=row["version"],
        template=row["template"],
        is_active=row["is_active"],
        traffic_pct=row["traffic_pct"],
        created_at=row["created_at"],
        metadata=row.get("metadata") or {},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """Versioned prompt store with Postgres backend + in-memory fallback."""

    def __init__(self, config: PromptRegistryConfig | None = None) -> None:
        self.config = config or PromptRegistryConfig()
        self._conn: Any = None
        self._fallback: dict[str, list[PromptVersion]] = {}
        self._connect_postgres()

    def _connect_postgres(self) -> None:
        if not _PSYCOPG_AVAILABLE:
            log.warning("psycopg not installed — using in-memory prompt registry")
            return
        try:
            conn = psycopg.connect(self.config.database_url, autocommit=True, row_factory=dict_row)
            conn.execute(PROMPT_REGISTRY_SCHEMA_SQL)
            self._conn = conn
            log.info("PromptRegistry connected to Postgres")
        except Exception as exc:
            log.warning("Postgres unavailable (%s) — using in-memory prompt registry", exc)
            self._conn = None

    @property
    def backend(self) -> str:
        return "postgres" if self._conn is not None else "memory"

    # -- in-memory fallback helpers --------------------------------------

    def _load_fallback(self, name: str) -> list[PromptVersion]:
        return self._fallback.get(name, [])

    def _save_fallback(self, version: PromptVersion) -> None:
        versions = self._fallback.setdefault(version.name, [])
        versions = [v for v in versions if v.version != version.version]
        versions.append(version)
        self._fallback[version.name] = sorted(versions, key=lambda v: v.version)

    # -- queries ------------------------------------------------------------

    def list_versions(self, name: str) -> list[PromptVersion]:
        if self._conn is not None:
            rows = self._conn.execute(
                "SELECT * FROM prompt_versions WHERE name = %s ORDER BY version", (name,)
            ).fetchall()
            return [_row_to_prompt_version(r) for r in rows]
        return list(self._load_fallback(name))

    def get_version(self, name: str, version: int) -> PromptVersion | None:
        if self._conn is not None:
            row = self._conn.execute(
                "SELECT * FROM prompt_versions WHERE name = %s AND version = %s", (name, version)
            ).fetchone()
            return _row_to_prompt_version(row) if row else None
        for v in self._load_fallback(name):
            if v.version == version:
                return v
        return None

    def register(self, name: str, template: str, metadata: dict[str, Any] | None = None) -> PromptVersion:
        """Register a new (inactive) version of a prompt, auto-incrementing version."""
        existing = self.list_versions(name)
        next_version = (max((v.version for v in existing), default=0)) + 1
        is_active = not existing  # first version of a prompt is active by default
        pv = PromptVersion(
            name=name,
            version=next_version,
            template=template,
            is_active=is_active,
            traffic_pct=100 if is_active else 0,
            metadata=metadata or {},
        )
        if self._conn is not None:
            self._conn.execute(
                """
                INSERT INTO prompt_versions (name, version, template, is_active, traffic_pct, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (pv.name, pv.version, pv.template, pv.is_active, pv.traffic_pct, Json(pv.metadata)),
            )
        else:
            self._save_fallback(pv)
        log.info("Registered prompt %s v%d (active=%s)", name, next_version, is_active)
        return pv

    def get_active(self, name: str, user_id: str | None = None) -> str:
        """Return the active template for `name`, A/B-bucketing by user_id.

        Active versions with traffic_pct < 100 split traffic deterministically:
        sha256(f"{name}:{user_id}") % 100 < traffic_pct routes to that version,
        otherwise falls through to the next active version (or the last one).
        """
        versions = [v for v in self.list_versions(name) if v.is_active]
        if not versions:
            raise KeyError(f"no active prompt versions for {name!r}")
        if len(versions) == 1:
            return versions[0].template

        bucket_key = f"{name}:{user_id or 'anonymous'}"
        bucket = int(hashlib.sha256(bucket_key.encode()).hexdigest(), 16) % 100

        cumulative = 0
        for v in versions:
            cumulative += v.traffic_pct
            if bucket < cumulative:
                return v.template
        return versions[-1].template

    def promote(self, name: str, version: int, traffic_pct: int = 100) -> PromptVersion:
        """Mark `version` active with `traffic_pct`, capping other active versions."""
        target = self.get_version(name, version)
        if target is None:
            raise KeyError(f"{name!r} v{version} not found")

        others = [v for v in self.list_versions(name) if v.version != version and v.is_active]
        remaining = max(0, 100 - traffic_pct)
        for other in others:
            other_pct = remaining // len(others) if others else 0
            self._set_active(name, other.version, other.is_active and other_pct > 0, other_pct)

        self._set_active(name, version, True, traffic_pct)
        log.info("Promoted %s v%d to %d%% traffic", name, version, traffic_pct)
        result = self.get_version(name, version)
        assert result is not None
        return result

    def rollback(self, name: str) -> PromptVersion:
        """Revert to the highest-versioned previously-active version, deactivating the current one."""
        versions = self.list_versions(name)
        active = [v for v in versions if v.is_active]
        if len(active) < 2:
            inactive = sorted((v for v in versions if not v.is_active), key=lambda v: -v.version)
            if not inactive:
                raise ValueError(f"no previous version of {name!r} to roll back to")
            return self.promote(name, inactive[0].version, 100)

        current = max(active, key=lambda v: v.version)
        previous = max((v for v in active if v.version != current.version), key=lambda v: v.version)
        self._set_active(name, current.version, False, 0)
        return self.promote(name, previous.version, 100)

    def _set_active(self, name: str, version: int, is_active: bool, traffic_pct: int) -> None:
        if self._conn is not None:
            self._conn.execute(
                "UPDATE prompt_versions SET is_active = %s, traffic_pct = %s WHERE name = %s AND version = %s",
                (is_active, traffic_pct, name, version),
            )
            return
        for v in self._fallback.get(name, []):
            if v.version == version:
                v.is_active = is_active
                v.traffic_pct = traffic_pct

    def health(self) -> dict[str, Any]:
        names: set[str] = set()
        if self._conn is not None:
            rows = self._conn.execute("SELECT DISTINCT name FROM prompt_versions").fetchall()
            names = {r["name"] for r in rows}
        else:
            names = set(self._fallback.keys())
        return {"backend": self.backend, "prompts": sorted(names)}


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: dict[str, str] = {
    "agent.planner.system": (
        "You are the planning step of a research agent for the AIDataBoon "
        "knowledge base (Hacker News, arXiv, GitHub trending). Given the "
        "user's question, output 1-3 short search queries (one per line) "
        "that would retrieve the most relevant documents."
    ),
    "agent.critic.system": (
        "You are the critic step of a research agent. Given the user's "
        "question and the documents retrieved so far, decide if they are "
        "sufficient to answer the question. If not, output ONE additional "
        "search query that would fill the gap. If sufficient, reply with "
        "exactly: SUFFICIENT"
    ),
    "agent.answerer.system": (
        "You are the answering step of a research agent for the AIDataBoon "
        "knowledge base. Using ONLY the provided documents, write a concise, "
        "well-cited answer to the user's question. Cite sources by their "
        "[n] index. If the documents don't answer the question, say so."
    ),
}


def seed_default_prompts(registry: PromptRegistry) -> None:
    """Register DEFAULT_PROMPTS as v1 (active) for any prompt not yet present."""
    for name, template in DEFAULT_PROMPTS.items():
        if not registry.list_versions(name):
            registry.register(name, template, metadata={"seed": True})


# ---------------------------------------------------------------------------
# FastAPI admin sub-app
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class RegisterRequest(BaseModel):
        name: str
        template: str
        metadata: dict[str, Any] = Field(default_factory=dict)

    class PromoteRequest(BaseModel):
        version: int
        traffic_pct: int = 100


def create_app(registry: PromptRegistry | None = None) -> Any:
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("fastapi/pydantic not installed — cannot create prompt registry app")

    reg = registry or PromptRegistry()
    seed_default_prompts(reg)

    app = FastAPI(title="AIDataBoon Prompt Registry", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return reg.health()

    @app.get("/{name:path}/versions")
    def list_versions(name: str) -> list[dict[str, Any]]:
        return [v.to_dict() for v in reg.list_versions(name)]

    @app.get("/{name:path}/active")
    def active(name: str, user_id: str | None = None) -> dict[str, Any]:
        try:
            return {"name": name, "user_id": user_id, "template": reg.get_active(name, user_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/register")
    def register(req: RegisterRequest) -> dict[str, Any]:
        return reg.register(req.name, req.template, req.metadata).to_dict()

    @app.post("/{name:path}/promote")
    def promote(name: str, req: PromoteRequest) -> dict[str, Any]:
        try:
            return reg.promote(name, req.version, req.traffic_pct).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/{name:path}/rollback")
    def rollback(name: str) -> dict[str, Any]:
        try:
            return reg.rollback(name).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


if _FASTAPI_AVAILABLE:
    app = create_app()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    registry = PromptRegistry()
    log.info("Backend: %s", registry.backend)
    seed_default_prompts(registry)

    for name in DEFAULT_PROMPTS:
        active = registry.get_active(name)
        log.info("%s v1 active: %s...", name, active[:60])

    # Register a v2 candidate and canary it at 30% traffic.
    pv2 = registry.register(
        "agent.answerer.system",
        DEFAULT_PROMPTS["agent.answerer.system"] + " Keep answers under 150 words.",
        metadata={"experiment": "concise-answers"},
    )
    registry.promote("agent.answerer.system", pv2.version, traffic_pct=30)

    buckets = {"v1": 0, "v2": 0}
    for i in range(200):
        template = registry.get_active("agent.answerer.system", user_id=f"user-{i}")
        buckets["v2" if "150 words" in template else "v1"] += 1
    log.info("A/B split over 200 users: %s", buckets)

    registry.rollback("agent.answerer.system")
    log.info("After rollback, active v1: %s", "150 words" not in registry.get_active("agent.answerer.system"))

    log.info("Health: %s", registry.health())


if __name__ == "__main__":
    main()
