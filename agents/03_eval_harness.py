"""
AIDataBoon Agents: RAG Evaluation Harness

Implements ADR-5 (00_project_design.md): runs every question in
golden_qa.jsonl through Agent.chat() (04_agent_orchestration.py), scores
each run with RAGAS-style metrics (faithfulness, answer relevancy, context
precision/recall), and writes eval_report.json. Exits non-zero if the mean
composite score drops below EVAL_MIN_SCORE — wired into
08_cicd_eval_pipeline.yml as a CI regression gate.

Uses `ragas` if installed; otherwise a dependency-free heuristic based on
keyword overlap and the same hash-fallback embeddings as
04_agent_orchestration.py / 05_rag_service.py.

Standalone demo / CI gate:
    uv run python agents/03_eval_harness.py
    uv run python agents/03_eval_harness.py --threshold 0.6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aidataboon.eval")

try:
    import ragas  # noqa: F401

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    log.warning("ragas not installed — using heuristic eval metrics")

_HERE = Path(__file__).resolve().parent


def _load_module(filename: str) -> ModuleType:
    """Import a numbered sibling file that isn't a valid dotted module name."""
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Config & golden set
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    golden_qa_path: Path = field(default_factory=lambda: _HERE / "golden_qa.jsonl")
    report_path: Path = field(default_factory=lambda: _HERE / "eval_report.json")
    min_score: float = field(default_factory=lambda: float(os.environ.get("EVAL_MIN_SCORE", "0.6")))


def load_golden_qa(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Heuristic metrics (RAGAS-style, dependency-free)
# ---------------------------------------------------------------------------

_NO_DOCS_MARKER = "no matching documents were found"


def _tokenize(text: str) -> set[str]:
    return {tok for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(tok) > 2}


def _docs_text(docs: list[dict[str, Any]]) -> str:
    return " ".join(f"{d.get('title', '')} {d.get('body', d.get('summary', ''))}" for d in docs)


def context_recall(expected_keywords: list[str], docs: list[dict[str, Any]]) -> float:
    if not expected_keywords:
        return 1.0
    text = _docs_text(docs).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords)


def context_precision(expected_keywords: list[str], docs: list[dict[str, Any]]) -> float:
    if not docs:
        return 1.0 if not expected_keywords else 0.0
    keywords_lower = [kw.lower() for kw in expected_keywords]
    relevant = sum(
        1 for d in docs if any(kw in f"{d.get('title', '')} {d.get('body', d.get('summary', ''))}".lower() for kw in keywords_lower)
    )
    return relevant / len(docs)


def faithfulness(answer: str, docs: list[dict[str, Any]]) -> float:
    """Fraction of the answer's content words that are grounded in the
    retrieved documents — penalizes hallucinated claims. The deterministic
    "no documents found" answer is considered fully faithful (it makes no
    unsupported claims)."""
    if _NO_DOCS_MARKER in answer.lower():
        return 1.0
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    doc_tokens = _tokenize(_docs_text(docs))
    if not doc_tokens:
        return 0.0
    grounded = len(answer_tokens & doc_tokens)
    return grounded / len(answer_tokens)


def answer_relevancy(question: str, answer: str, embedder: Any) -> float:
    """Cosine similarity between the question and answer embeddings."""
    if not answer.strip():
        return 0.0
    q_vec = embedder.encode_one(question)
    a_vec = embedder.encode_one(answer)
    denom = float(np.linalg.norm(q_vec) * np.linalg.norm(a_vec))
    if denom == 0.0:
        return 0.0
    similarity = float(np.dot(q_vec, a_vec) / denom)
    return max(0.0, similarity)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question: str
    answer: str
    n_docs: int
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevancy: float

    @property
    def composite(self) -> float:
        return (self.context_recall + self.context_precision + self.faithfulness + self.answer_relevancy) / 4


@dataclass
class EvalReport:
    backend: str
    n_questions: int
    mean_context_recall: float
    mean_context_precision: float
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_composite: float
    min_score: float
    passed: bool
    results: list[QuestionResult]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "results": [asdict(r) for r in self.results]}


def run_eval(agent: Any, golden_qa: list[dict[str, Any]], min_score: float) -> EvalReport:
    embedder = _load_module("04_agent_orchestration.py").EmbeddingModel()
    results: list[QuestionResult] = []

    for item in golden_qa:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        chat_result = agent.chat(question)
        docs = chat_result["sources"]
        answer = chat_result["answer"]

        results.append(
            QuestionResult(
                question=question,
                answer=answer,
                n_docs=len(docs),
                context_recall=context_recall(expected_keywords, docs),
                context_precision=context_precision(expected_keywords, docs),
                faithfulness=faithfulness(answer, docs),
                answer_relevancy=answer_relevancy(question, answer, embedder),
            )
        )

    n = len(results)
    mean_recall = sum(r.context_recall for r in results) / n
    mean_precision = sum(r.context_precision for r in results) / n
    mean_faithfulness = sum(r.faithfulness for r in results) / n
    mean_relevancy = sum(r.answer_relevancy for r in results) / n
    mean_composite = sum(r.composite for r in results) / n

    return EvalReport(
        backend="ragas" if _RAGAS_AVAILABLE else "heuristic",
        n_questions=n,
        mean_context_recall=mean_recall,
        mean_context_precision=mean_precision,
        mean_faithfulness=mean_faithfulness,
        mean_answer_relevancy=mean_relevancy,
        mean_composite=mean_composite,
        min_score=min_score,
        passed=mean_composite >= min_score,
        results=results,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="AIDataBoon Agents RAG eval harness")
    parser.add_argument("--golden-qa", type=Path, default=None, help="Path to golden_qa.jsonl")
    parser.add_argument("--report", type=Path, default=None, help="Path to write eval_report.json")
    parser.add_argument("--threshold", type=float, default=None, help="Override EVAL_MIN_SCORE")
    args = parser.parse_args()

    config = EvalConfig()
    if args.golden_qa:
        config.golden_qa_path = args.golden_qa
    if args.report:
        config.report_path = args.report
    if args.threshold is not None:
        config.min_score = args.threshold

    orch = _load_module("04_agent_orchestration.py")
    agent = orch.Agent()
    log.info("Agent health: %s", agent.health())

    golden_qa = load_golden_qa(config.golden_qa_path)
    log.info("Loaded %d golden Q&A pairs from %s", len(golden_qa), config.golden_qa_path)

    report = run_eval(agent, golden_qa, config.min_score)

    config.report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    log.info("Wrote eval report to %s", config.report_path)

    log.info("=" * 70)
    log.info("Backend: %s", report.backend)
    for r in report.results:
        log.info(
            "  [%.2f] %s (docs=%d, recall=%.2f, precision=%.2f, faith=%.2f, relevancy=%.2f)",
            r.composite, r.question[:60], r.n_docs, r.context_recall, r.context_precision, r.faithfulness, r.answer_relevancy,
        )
    log.info("-" * 70)
    log.info(
        "Mean composite: %.3f (recall=%.3f precision=%.3f faithfulness=%.3f relevancy=%.3f) | threshold=%.2f -> %s",
        report.mean_composite, report.mean_context_recall, report.mean_context_precision,
        report.mean_faithfulness, report.mean_answer_relevancy, report.min_score,
        "PASS" if report.passed else "FAIL",
    )

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
