"""
AIDataBoon Agents: Multi-Agent Orchestration & Episodic Memory

Implements ADR-3 (LangGraph graph with deterministic fallback) and ADR-4
(pgvector episodic memory) and ADR-6 (user_id threading for per-user A/B
without per-request reconnects) from 00_project_design.md.

Graph: planner -> retriever -> critic -> [retriever | answerer] -> answerer
  - planner:   turns the user's question into 1-3 search queries
  - retriever: calls the data platform RAG API (data_platform/05_rag_service.py)
  - critic:    decides whether retrieved docs are sufficient, or asks for one
               more search query (looped up to max_iterations)
  - answerer:  writes a cited answer from the retrieved documents

LLM_PROVIDER=none (default) runs a fully deterministic, dependency-free path
suitable for local dev and CI: planner emits the raw query, critic always
judges "sufficient", and the answerer extractively summarizes retrieved docs.

Standalone demo:
    uv run python agents/04_agent_orchestration.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aidataboon.agent")

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Json

    _PSYCOPG_AVAILABLE = True
except ImportError:
    _PSYCOPG_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from langgraph.graph import END, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Embeddings (hash-fallback, mirrors data_platform/05_rag_service.py,
# with tokenization tuned per the scire-tutor repository's 02_curriculum_index.py
# _tokenize: see its docstring for why plain .split() hashing is weak)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no topical signal but appear in nearly every chunk,
# so hashing them in lets two chunks look similar because they share
# "the"/"how"/"does" rather than any real topic overlap.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being of to in on for and or but as
    at by from with about into through during before after this that
    these those it its what which who how why when where does do did
    can could will would should you your yours i me my we us our they
    them their he she his her not no yes if so than then there here
    have has had having also just very
    """.split()
)


def _normalize_token(token: str) -> str:
    """Crude suffix-stripping so plural/verb variants of a word hash to the
    same bucket. Applied identically to indexed content and queries, so an
    imprecise stem is harmless as long as it's consistent."""
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """[a-z0-9]+ (not .split()) so punctuation/banner runs don't become
    single giant tokens that pollute the hash sum, plus stopword filtering."""
    return [_normalize_token(tok) for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


class EmbeddingModel:
    DIM = EMBEDDING_DIM

    def __init__(self, model_name: str | None = None) -> None:
        self._model: Any = None
        if _ST_AVAILABLE:
            name = model_name or "all-MiniLM-L6-v2"
            try:
                self._model = SentenceTransformer(name)
            except Exception as exc:
                log.warning("Failed to load sentence-transformers model %s (%s): using hash embeddings", name, exc)

    @property
    def backend(self) -> str:
        return "sentence-transformers" if self._model is not None else "hash-fallback"

    def encode_one(self, text: str) -> np.ndarray:
        if self._model is not None:
            vec = self._model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vec, dtype=np.float32)
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "none"))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    api_key: str | None = field(default_factory=lambda: os.environ.get("LLM_API_KEY"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))

    def __post_init__(self) -> None:
        if not self.model:
            self.model = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-haiku-4-5-20251001",
                "ollama": "llama3.2",
                "none": "none",
            }.get(self.provider, "none")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


class LLMClient:
    """Thin client over OpenAI / Anthropic / Ollama REST APIs, or a no-op."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def complete(self, system: str, user: str, max_tokens: int = 512, timeout: float = 30.0) -> LLMResponse:
        if self.provider == "openai":
            return self._complete_openai(system, user, max_tokens, timeout)
        if self.provider == "anthropic":
            return self._complete_anthropic(system, user, max_tokens, timeout)
        if self.provider == "ollama":
            return self._complete_ollama(system, user, max_tokens, timeout)
        return LLMResponse(text="", input_tokens=0, output_tokens=0, finish_reason="no_llm")

    def _complete_openai(self, system: str, user: str, max_tokens: int, timeout: float) -> LLMResponse:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            text=choice["message"]["content"],
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def _complete_anthropic(self, system: str, user: str, max_tokens: int, timeout: float) -> LLMResponse:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.config.api_key or "",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return LLMResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason") or "stop",
        )

    def _complete_ollama(self, system: str, user: str, max_tokens: int, timeout: float) -> LLMResponse:
        resp = httpx.post(
            f"{self.config.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        # Ollama doesn't report token counts uniformly; estimate ~4 chars/token.
        return LLMResponse(
            text=text,
            input_tokens=int(len(system + user) / 4),
            output_tokens=int(len(text) / 4),
            finish_reason="stop",
        )

    def health(self) -> dict[str, Any]:
        if self.provider in ("openai", "anthropic"):
            configured = bool(self.config.api_key)
        else:
            configured = True
        return {"provider": self.provider, "model": self.model, "configured": configured}


# ---------------------------------------------------------------------------
# Agent state & graph nodes
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    query: str
    user_id: str | None
    search_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    critique: str | None
    answer: str | None
    citations: list[str]
    iteration: int
    usage: list[dict[str, Any]]
    trace: list[dict[str, Any]]


def initial_state(query: str, user_id: str | None = None) -> AgentState:
    return AgentState(
        query=query,
        user_id=user_id,
        search_queries=[],
        retrieved_docs=[],
        critique=None,
        answer=None,
        citations=[],
        iteration=0,
        usage=[],
        trace=[],
    )


@dataclass
class AgentResult:
    answer: str
    citations: list[str]
    sources: list[dict[str, Any]]
    iterations: int
    usage: list[dict[str, Any]]
    trace: list[dict[str, Any]]

    @property
    def total_input_tokens(self) -> int:
        return sum(u["input_tokens"] for u in self.usage)

    @property
    def total_output_tokens(self) -> int:
        return sum(u["output_tokens"] for u in self.usage)


def _record_usage(state: AgentState, node: str, response: LLMResponse, llm: LLMClient) -> None:
    state["usage"].append(
        {
            "node": node,
            "provider": llm.provider,
            "model": llm.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
    )


class AgentOrchestrator:
    """Builds and runs the planner -> retriever -> critic -> answerer graph."""

    DEFAULT_MAX_ITERATIONS = 2

    def __init__(
        self,
        llm: LLMClient | None = None,
        rag_api_url: str | None = None,
        prompt_provider: Callable[[str, str | None], str] | None = None,
        max_iterations: int | None = None,
        request_timeout: float = 10.0,
    ) -> None:
        self.llm = llm or LLMClient()
        self.rag_api_url = rag_api_url or os.environ.get("RAG_API_URL", "http://localhost:8000")
        self.prompt_provider = prompt_provider
        self.max_iterations = max_iterations or self.DEFAULT_MAX_ITERATIONS
        self.request_timeout = request_timeout
        self._graph = self._build_graph()

    def _get_prompt(self, name: str, user_id: str | None = None) -> str:
        if self.prompt_provider is not None:
            try:
                return self.prompt_provider(name, user_id)
            except Exception as exc:
                log.warning("prompt_provider failed for %s (%s): using DEFAULT_PROMPTS", name, exc)
        return DEFAULT_PROMPTS[name]

    # -- nodes ------------------------------------------------------------

    def _planner_node(self, state: AgentState) -> AgentState:
        if self.llm.provider == "none":
            state["search_queries"] = [state["query"]]
            state["trace"].append({"step": "planner", "role": "system", "content": state["query"]})
            return state

        system = self._get_prompt("agent.planner.system", state.get("user_id"))
        response = self.llm.complete(system, state["query"], max_tokens=128)
        _record_usage(state, "planner", response, self.llm)
        queries = [line.strip("- ").strip() for line in response.text.splitlines() if line.strip()]
        state["search_queries"] = queries or [state["query"]]
        state["trace"].append({"step": "planner", "role": "assistant", "content": response.text})
        return state

    def _retriever_node(self, state: AgentState) -> AgentState:
        seen_ids: set[str] = {d.get("id", "") for d in state["retrieved_docs"]}
        new_docs: list[dict[str, Any]] = []
        for query in state["search_queries"]:
            try:
                resp = httpx.post(
                    f"{self.rag_api_url}/search",
                    json={"query": query, "limit": 5},
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                results = resp.json().get("sources", [])
            except Exception as exc:
                log.warning("RAG API search failed for %r (%s)", query, exc)
                results = []
            for doc in results:
                doc_id = str(doc.get("id", ""))
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    new_docs.append(doc)

        state["retrieved_docs"].extend(new_docs)
        state["trace"].append(
            {
                "step": "retriever",
                "role": "tool",
                "content": f"queries={state['search_queries']!r} -> {len(new_docs)} new docs (total {len(state['retrieved_docs'])})",
            }
        )
        return state

    def _critic_node(self, state: AgentState) -> AgentState:
        state["iteration"] += 1
        if self.llm.provider == "none" or state["iteration"] >= self.max_iterations:
            state["critique"] = None
            state["trace"].append({"step": "critic", "role": "system", "content": "SUFFICIENT"})
            return state

        system = self._get_prompt("agent.critic.system", state.get("user_id"))
        docs_text = "\n".join(f"[{i + 1}] {d.get('title', '')}" for i, d in enumerate(state["retrieved_docs"]))
        user = f"Question: {state['query']}\n\nRetrieved documents:\n{docs_text}"
        response = self.llm.complete(system, user, max_tokens=64)
        _record_usage(state, "critic", response, self.llm)

        text = response.text.strip()
        if text.upper().startswith("SUFFICIENT"):
            state["critique"] = None
        else:
            state["critique"] = text
            state["search_queries"] = [text]
        state["trace"].append({"step": "critic", "role": "assistant", "content": text})
        return state

    def _answerer_node(self, state: AgentState) -> AgentState:
        docs = state["retrieved_docs"]
        if not docs:
            state["answer"] = "No matching documents were found in the knowledge base."
            state["citations"] = []
            state["trace"].append({"step": "answerer", "role": "assistant", "content": state["answer"]})
            return state

        if self.llm.provider == "none":
            top = docs[:3]
            lines = [f"[{i + 1}] {d.get('title', 'untitled')}" for i, d in enumerate(top)]
            state["answer"] = "Based on the knowledge base:\n" + "\n".join(lines)
            state["citations"] = [str(i + 1) for i in range(len(top))]
            state["trace"].append({"step": "answerer", "role": "assistant", "content": state["answer"]})
            return state

        system = self._get_prompt("agent.answerer.system", state.get("user_id"))
        docs_text = "\n".join(
            f"[{i + 1}] {d.get('title', '')}: {d.get('body', d.get('summary', ''))[:300]}"
            for i, d in enumerate(docs)
        )
        user = f"Question: {state['query']}\n\nDocuments:\n{docs_text}"
        response = self.llm.complete(system, user, max_tokens=512)
        _record_usage(state, "answerer", response, self.llm)

        state["answer"] = response.text
        state["citations"] = sorted(set(re_findall_citations(response.text)))
        state["trace"].append({"step": "answerer", "role": "assistant", "content": response.text})
        return state

    # -- graph construction -------------------------------------------------

    def _should_continue(self, state: AgentState) -> str:
        if state["critique"] and state["iteration"] < self.max_iterations:
            return "retriever"
        return "answerer"

    def _build_graph(self) -> Any:
        if _LANGGRAPH_AVAILABLE:
            graph: Any = StateGraph(AgentState)
            graph.add_node("planner", self._planner_node)
            graph.add_node("retriever", self._retriever_node)
            graph.add_node("critic", self._critic_node)
            graph.add_node("answerer", self._answerer_node)
            graph.set_entry_point("planner")
            graph.add_edge("planner", "retriever")
            graph.add_edge("retriever", "critic")
            graph.add_conditional_edges("critic", self._should_continue, {"retriever": "retriever", "answerer": "answerer"})
            graph.add_edge("answerer", END)
            return graph.compile()
        return SimpleAgentGraph(self)

    def run(self, query: str, user_id: str | None = None) -> AgentResult:
        state = self._graph.invoke(initial_state(query, user_id))
        return AgentResult(
            answer=state["answer"] or "",
            citations=state["citations"],
            sources=state["retrieved_docs"],
            iterations=state["iteration"],
            usage=state["usage"],
            trace=state["trace"],
        )


def re_findall_citations(text: str) -> list[str]:
    import re

    return re.findall(r"\[(\d+)\]", text)


class SimpleAgentGraph:
    """Dependency-free fallback for AgentOrchestrator when langgraph is absent."""

    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self.orchestrator = orchestrator

    def invoke(self, state: AgentState) -> AgentState:
        o = self.orchestrator
        state = o._planner_node(state)
        state = o._retriever_node(state)
        while True:
            state = o._critic_node(state)
            if o._should_continue(state) == "retriever":
                state = o._retriever_node(state)
            else:
                break
        return o._answerer_node(state)


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


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------

AGENT_MEMORY_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS agent_runs (
    id            UUID PRIMARY KEY,
    query         TEXT NOT NULL,
    user_id       TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    answer        TEXT,
    citations     JSONB NOT NULL DEFAULT '[]'::jsonb,
    usage         JSONB NOT NULL DEFAULT '[]'::jsonb,
    cost_usd      DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    feedback      INTEGER,
    error         TEXT,
    embedding     VECTOR({EMBEDDING_DIM}),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS agent_messages (
    id            BIGSERIAL PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    step          TEXT NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag_api_url: str = field(default_factory=lambda: os.environ.get("RAG_API_URL", "http://localhost:8000"))
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://aidataboon:aidataboon@localhost:5432/aidataboon"
        )
    )
    max_iterations: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_ITERATIONS", "2")))
    request_timeout: float = 10.0
    embedding_model: str | None = None


class AgentMemory:
    """Persists agent runs + step traces, with pgvector recall of similar runs."""

    def __init__(self, database_url: str, embedding_model: str | None = None) -> None:
        self.embedder = EmbeddingModel(embedding_model)
        self._conn: Any = None
        self._runs: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._connect_postgres(database_url)

    def _connect_postgres(self, database_url: str) -> None:
        if not _PSYCOPG_AVAILABLE:
            log.warning("psycopg not installed: using in-memory agent memory")
            return
        try:
            conn = psycopg.connect(database_url, autocommit=True, row_factory=dict_row)
            conn.execute(AGENT_MEMORY_SCHEMA_SQL)
            self._conn = conn
            log.info("AgentMemory connected to Postgres")
        except Exception as exc:
            log.warning("Postgres unavailable (%s): using in-memory agent memory", exc)
            self._conn = None

    @property
    def backend(self) -> str:
        return "postgres" if self._conn is not None else "memory"

    def start_run(self, query: str, user_id: str | None = None) -> str:
        run_id = str(uuid.uuid4())
        embedding = self.embedder.encode_one(query)
        if self._conn is not None:
            vector_literal = "[" + ",".join(f"{x:.6f}" for x in embedding.tolist()) + "]"
            self._conn.execute(
                "INSERT INTO agent_runs (id, query, user_id, status, embedding) VALUES (%s, %s, %s, 'running', %s)",
                (run_id, query, user_id, vector_literal),
            )
        else:
            self._runs[run_id] = {
                "id": run_id,
                "query": query,
                "user_id": user_id,
                "status": "running",
                "answer": None,
                "citations": [],
                "usage": [],
                "cost_usd": 0.0,
                "latency_ms": None,
                "feedback": None,
                "error": None,
                "embedding": embedding,
                "created_at": time.time(),
            }
            self._messages[run_id] = []
        return run_id

    def record_message(self, run_id: str, step: str, role: str, content: str) -> None:
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO agent_messages (run_id, step, role, content) VALUES (%s, %s, %s, %s)",
                (run_id, step, role, content),
            )
        else:
            self._messages.setdefault(run_id, []).append({"step": step, "role": role, "content": content})

    def complete_run(
        self, run_id: str, result: AgentResult, latency_ms: int, cost_usd: float = 0.0, status: str = "completed"
    ) -> None:
        if self._conn is not None:
            self._conn.execute(
                """
                UPDATE agent_runs
                SET status = %s, answer = %s, citations = %s, usage = %s, cost_usd = %s, latency_ms = %s
                WHERE id = %s
                """,
                (status, result.answer, Json(list(result.citations)), Json(result.usage), cost_usd, latency_ms, run_id),
            )
        else:
            mem_run = self._runs.get(run_id)
            if mem_run is not None:
                mem_run.update(
                    status=status,
                    answer=result.answer,
                    citations=list(result.citations),
                    usage=result.usage,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                )

    def fail_run(self, run_id: str, error: str, latency_ms: int) -> None:
        if self._conn is not None:
            self._conn.execute(
                "UPDATE agent_runs SET status = 'failed', error = %s, latency_ms = %s WHERE id = %s",
                (error, latency_ms, run_id),
            )
        else:
            mem_run = self._runs.get(run_id)
            if mem_run is not None:
                mem_run.update(status="failed", error=error, latency_ms=latency_ms)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if self._conn is not None:
            row = self._conn.execute(
                "SELECT id, query, user_id, status, answer, citations, usage, cost_usd, latency_ms, feedback, error, created_at FROM agent_runs WHERE id = %s",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            row = dict(row)
            row["id"] = str(row["id"])
            messages = self._conn.execute(
                "SELECT step, role, content FROM agent_messages WHERE run_id = %s ORDER BY id", (run_id,)
            ).fetchall()
            row["trace"] = [dict(m) for m in messages]
            return row

        mem_run = self._runs.get(run_id)
        if mem_run is None:
            return None
        result = {k: v for k, v in mem_run.items() if k != "embedding"}
        result["trace"] = list(self._messages.get(run_id, []))
        return result

    def set_feedback(self, run_id: str, rating: int) -> bool:
        if rating not in (-1, 0, 1):
            raise ValueError("rating must be -1, 0, or 1")
        if self._conn is not None:
            cur = self._conn.execute("UPDATE agent_runs SET feedback = %s WHERE id = %s", (rating, run_id))
            return cur.rowcount > 0
        mem_run = self._runs.get(run_id)
        if mem_run is None:
            return False
        mem_run["feedback"] = rating
        return True

    def recall_similar(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        embedding = self.embedder.encode_one(query)
        if self._conn is not None:
            vector_literal = "[" + ",".join(f"{x:.6f}" for x in embedding.tolist()) + "]"
            rows = self._conn.execute(
                """
                SELECT id, query, answer, embedding <-> %s AS distance
                FROM agent_runs
                WHERE status = 'completed'
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (vector_literal, vector_literal, limit),
            ).fetchall()
            return [{"id": str(r["id"]), "query": r["query"], "answer": r["answer"], "distance": r["distance"]} for r in rows]

        scored = []
        for mem_run in self._runs.values():
            if mem_run["status"] != "completed":
                continue
            distance = float(np.linalg.norm(mem_run["embedding"] - embedding))
            scored.append({"id": mem_run["id"], "query": mem_run["query"], "answer": mem_run["answer"], "distance": distance})
        scored.sort(key=lambda r: r["distance"])
        return scored[:limit]

    def health(self) -> dict[str, Any]:
        if self._conn is not None:
            n_runs = self._conn.execute("SELECT count(*) AS n FROM agent_runs").fetchone()["n"]
        else:
            n_runs = len(self._runs)
        return {"backend": self.backend, "embedding_model": self.embedder.backend, "runs": n_runs}


# ---------------------------------------------------------------------------
# Agent (top-level facade)
# ---------------------------------------------------------------------------


class Agent:
    """Top-level facade: orchestrator + episodic memory + cost tracking."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        prompt_provider: Callable[[str, str | None], str] | None = None,
        cost_estimator: Callable[[str, str, int, int], float] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm = LLMClient(self.config.llm)
        self.orchestrator = AgentOrchestrator(
            llm=self.llm,
            rag_api_url=self.config.rag_api_url,
            prompt_provider=prompt_provider,
            max_iterations=self.config.max_iterations,
            request_timeout=self.config.request_timeout,
        )
        self.memory = AgentMemory(self.config.database_url, embedding_model=self.config.embedding_model)
        self.cost_estimator = cost_estimator

    def chat(self, query: str, user_id: str | None = None) -> dict[str, Any]:
        run_id = self.memory.start_run(query, user_id=user_id)
        t0 = time.perf_counter()
        try:
            result = self.orchestrator.run(query, user_id=user_id)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            self.memory.fail_run(run_id, str(exc), latency_ms)
            raise

        for entry in result.trace:
            self.memory.record_message(run_id, entry["step"], entry["role"], entry["content"])

        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost_usd = 0.0
        if self.cost_estimator is not None:
            try:
                cost_usd = self.cost_estimator(self.llm.provider, self.llm.model, result.total_input_tokens, result.total_output_tokens)
            except Exception as exc:
                log.warning("cost_estimator failed (%s): recording cost as 0.0", exc)

        self.memory.complete_run(run_id, result, latency_ms, cost_usd=cost_usd, status="completed")
        return {
            "run_id": run_id,
            "answer": result.answer,
            "citations": result.citations,
            "sources": result.sources,
            "iterations": result.iterations,
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "cost_usd": cost_usd,
            "usage": result.usage,
            "trace": result.trace,
            "latency_ms": latency_ms,
        }

    def health(self) -> dict[str, Any]:
        return {
            "graph_backend": "langgraph" if _LANGGRAPH_AVAILABLE else "simple",
            "llm": self.llm.health(),
            "memory": self.memory.health(),
            "rag_api_url": self.config.rag_api_url,
            "max_iterations": self.config.max_iterations,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    agent = Agent()
    log.info("Health: %s", agent.health())

    result = agent.chat("What are the latest trends in LLM fine-tuning?", user_id="demo-user")
    log.info("Answer: %s", result["answer"])
    log.info("Citations: %s", result["citations"])
    log.info("Iterations: %d", result["iterations"])
    log.info("Tokens: in=%d out=%d (cost=$%.6f)", result["input_tokens"], result["output_tokens"], result["cost_usd"])
    for entry in result["trace"]:
        log.info("  [%s/%s] %s", entry["step"], entry["role"], entry["content"][:100])

    feedback_ok = agent.memory.set_feedback(result["run_id"], 1)
    log.info("Feedback recorded: %s", feedback_ok)

    similar = agent.memory.recall_similar("LLM fine-tuning trends")
    log.info("Recall similar: %d hits", len(similar))


if __name__ == "__main__":
    main()
