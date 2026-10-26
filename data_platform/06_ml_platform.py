"""
AIDataBoon ML Platform: Feature Store + Experiment Tracking + Model Serving

Stack:
  Feast: offline/online feature store
  DuckDB: feature transformations (reads from Gold layer)
  MLflow 2.x: experiment tracking + model registry
  scikit-learn: topic classifier (lightweight, no GPU needed)
  FastAPI wrapper: model serving endpoint
  PSI: production drift detection
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section 1: Feature Store (Feast)
# ---------------------------------------------------------------------------

FEATURE_STORE_YAML = textwrap.dedent("""\
    project: aidataboon
    provider: local
    registry: data/registry.db
    offline_store:
        type: duckdb
    online_store:
        type: redis
        connection_string: redis://localhost:6379
    entity_key_serialization_version: 2
""")

FEATURES_PY = textwrap.dedent("""\
    from datetime import timedelta
    from feast import Entity, FeatureView, Field, FileSource
    from feast.types import Float32, Int64, String, Bool

    content_source = Entity(
        name="content_id",
        join_keys=["id"],
        description="Unique identifier for a knowledge-base content item",
    )

    content_source_file = FileSource(
        path="data/gold/knowledge_base.parquet",
        timestamp_field="event_timestamp",
    )

    content_stats_fv = FeatureView(
        name="content_stats",
        entities=[content_source],
        ttl=timedelta(days=30),
        schema=[
            Field(name="title_length", dtype=Int64),
            Field(name="word_count", dtype=Int64),
            Field(name="has_code_snippet", dtype=Bool),
            Field(name="topic_count", dtype=Int64),
            Field(name="source", dtype=String),
        ],
        source=content_source_file,
        online=True,
        description="Content-level features derived from the Gold knowledge_base table",
    )

    engagement_source_file = FileSource(
        path="data/gold/feature_table.parquet",
        timestamp_field="event_timestamp",
    )

    engagement_stats_fv = FeatureView(
        name="engagement_stats",
        entities=[content_source],
        ttl=timedelta(days=7),
        schema=[
            Field(name="score_normalized", dtype=Float32),
            Field(name="velocity_score", dtype=Float32),
            Field(name="comment_ratio", dtype=Float32),
        ],
        source=engagement_source_file,
        online=True,
        description="Engagement features from the Gold feature_table",
    )

    embeddings_source_file = FileSource(
        path="data/gold/embeddings.parquet",
        timestamp_field="event_timestamp",
    )

    text_embeddings_fv = FeatureView(
        name="text_embeddings",
        entities=[content_source],
        ttl=timedelta(days=30),
        schema=[
            Field(name="embedding_l2_norm", dtype=Float32),
        ],
        source=embeddings_source_file,
        online=True,
        description="L2-norm summary of text embeddings for drift detection",
    )
""")


@dataclass
class FeatureStoreConfig:
    repo_path: str = "feast_repo"
    offline_store_type: str = "duckdb"
    online_store_type: str = "redis"
    online_store_connection: str = "redis://localhost:6379"
    registry_path: str = "data/registry.db"

    def write_repo_files(self) -> None:
        """Materialise the embedded Feast repo files to disk."""
        repo = Path(self.repo_path)
        repo.mkdir(parents=True, exist_ok=True)
        (repo / "feature_store.yaml").write_text(FEATURE_STORE_YAML)
        (repo / "features.py").write_text(FEATURES_PY)
        logger.info("Feast repo files written to %s", repo)


class FeatureEngineer:
    """Compute and materialise features using DuckDB against the Gold layer."""

    def __init__(self, config: FeatureStoreConfig | None = None) -> None:
        self.config = config or FeatureStoreConfig()
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(":memory:")

    # ------------------------------------------------------------------
    def compute_content_features(self, knowledge_base_path: str) -> pd.DataFrame:
        """
        Derive content-level features directly from the Gold knowledge_base
        Parquet file using DuckDB SQL.
        """
        query = f"""
            SELECT
                id,
                len(title)                                          AS title_length,
                len(string_split(content, ' '))                    AS word_count,
                (content LIKE '%```%')                             AS has_code_snippet,
                COALESCE(array_length(tags), 0)                    AS topic_count,
                source,
                NOW()                                              AS event_timestamp
            FROM read_parquet('{knowledge_base_path}')
        """
        df = self._conn.execute(query).fetchdf()
        logger.info("Computed content features: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    def compute_engagement_features(self, gold_path: str) -> pd.DataFrame:
        """
        Derive engagement features from the Gold feature_table.
        Normalises score and computes a velocity proxy.
        """
        query = f"""
            WITH base AS (
                SELECT
                    id,
                    score,
                    num_comments,
                    days_since_published,
                    MIN(score) OVER () AS min_score,
                    MAX(score) OVER () AS max_score
                FROM read_parquet('{gold_path}')
            )
            SELECT
                id,
                CASE
                    WHEN (max_score - min_score) = 0 THEN 0.0
                    ELSE CAST((score - min_score) AS DOUBLE) / (max_score - min_score)
                END                                                  AS score_normalized,
                CASE
                    WHEN days_since_published = 0 THEN score
                    ELSE CAST(score AS DOUBLE) / days_since_published
                END                                                  AS velocity_score,
                CASE
                    WHEN score = 0 THEN 0.0
                    ELSE CAST(num_comments AS DOUBLE) / score
                END                                                  AS comment_ratio,
                NOW()                                                AS event_timestamp
            FROM base
        """
        df = self._conn.execute(query).fetchdf()
        logger.info("Computed engagement features: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    def materialize_to_feast(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Push computed features into the Feast online store."""
        try:
            from feast import FeatureStore  # type: ignore[import]

            store = FeatureStore(repo_path=self.config.repo_path)
            store.materialize(start_date=start_date, end_date=end_date)
            logger.info(
                "Feast materialisation complete: %s → %s",
                start_date.isoformat(),
                end_date.isoformat(),
            )
        except ImportError:
            logger.warning("feast not installed: skipping online-store materialisation")


# ---------------------------------------------------------------------------
# Section 2: Training Pipeline
# ---------------------------------------------------------------------------

TOPIC_LABELS: list[str] = [
    "ai/ml",
    "systems",
    "web/frontend",
    "security",
    "data/databases",
    "devops",
    "open-source",
    "startup/business",
    "science/research",
    "other",
]

# Keyword heuristic used to derive a training label from tag lists
_LABEL_KEYWORDS: dict[str, list[str]] = {
    "ai/ml": ["machine learning", "deep learning", "neural", "llm", "gpt", "ai", "ml"],
    "systems": ["kernel", "os", "cpu", "memory", "performance", "rust", "c++", "systems"],
    "web/frontend": ["javascript", "css", "html", "react", "vue", "frontend", "browser"],
    "security": ["security", "vulnerability", "exploit", "cve", "cryptography", "privacy"],
    "data/databases": ["database", "sql", "nosql", "postgres", "redis", "kafka", "data"],
    "devops": ["kubernetes", "docker", "ci/cd", "devops", "cloud", "terraform", "aws"],
    "open-source": ["open source", "open-source", "oss", "github", "foss"],
    "startup/business": ["startup", "founder", "vc", "fundraising", "yc", "business"],
    "science/research": ["research", "paper", "science", "physics", "biology", "math"],
}


def _derive_label(tags: list[str] | None) -> str:
    if not tags:
        return "other"
    joined = " ".join(t.lower() for t in tags)
    for label, keywords in _LABEL_KEYWORDS.items():
        if any(kw in joined for kw in keywords):
            return label
    return "other"


class TopicClassifier:
    """Lightweight TF-IDF + Logistic Regression topic classifier."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5_000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        strip_accents="unicode",
                        analyzer="word",
                        token_pattern=r"\w{2,}",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1_000,
                        class_weight="balanced",
                        solver="lbfgs",  # multinomial for multi-class targets (the default)
                        random_state=42,
                    ),
                ),
            ]
        )
        self.label_encoder = LabelEncoder()

    @property
    def classes_(self) -> list[str]:
        return list(self.pipeline.classes_) if hasattr(self.pipeline, "classes_") else TOPIC_LABELS


class TrainingPipeline:
    """End-to-end training orchestration."""

    def __init__(self, classifier: TopicClassifier | None = None) -> None:
        self.classifier = classifier or TopicClassifier()

    # ------------------------------------------------------------------
    def load_training_data(
        self, gold_path: str
    ) -> tuple[list[str], list[str]]:
        """
        Load title + abstract from the Gold knowledge_base Parquet file and
        derive a topic label from the tags column.
        """
        conn = duckdb.connect(":memory:")
        df = conn.execute(
            f"SELECT title, abstract, tags FROM read_parquet('{gold_path}')"
        ).fetchdf()
        conn.close()

        texts: list[str] = []
        labels: list[str] = []
        for _, row in df.iterrows():
            text = f"{row['title']} {row.get('abstract', '') or ''}"
            texts.append(text.strip())
            tags = row["tags"] if isinstance(row["tags"], list) else []
            labels.append(_derive_label(tags))

        logger.info(
            "Loaded %d training samples; label distribution: %s",
            len(texts),
            pd.Series(labels).value_counts().to_dict(),
        )
        return texts, labels

    # ------------------------------------------------------------------
    def split_data(
        self,
        X: list[str],
        y: list[str],
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        """Stratified 80 / 10 / 10 train / val / test split."""
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            stratify=y_train_val,
            random_state=random_state,
        )
        logger.info(
            "Split sizes: train %d, val %d, test %d",
            len(X_train), len(X_val), len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    def train(self, X_train: list[str], y_train: list[str]) -> Pipeline:
        """Fit the classifier pipeline and return it."""
        self.classifier.pipeline.fit(X_train, y_train)
        logger.info("Training complete")
        return self.classifier.pipeline

    # ------------------------------------------------------------------
    def evaluate(
        self,
        pipeline: Pipeline,
        X_test: list[str],
        y_test: list[str],
    ) -> dict[str, Any]:
        """Return classification metrics for the given split."""
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = {
            "accuracy": report["accuracy"],
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "classification_report": report,
        }
        logger.info(
            "Evaluation: accuracy %.4f, f1_macro %.4f",
            metrics["accuracy"], metrics["f1_macro"],
        )
        return metrics

    # ------------------------------------------------------------------
    def cross_validate_5fold(
        self,
        X: list[str],
        y: list[str],
    ) -> dict[str, float]:
        """Return mean ± std for accuracy and F1 over 5 stratified folds."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        result = cross_validate(
            self.classifier.pipeline,
            X, y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
            n_jobs=-1,
        )
        summary: dict[str, float] = {}
        for key, values in result.items():
            if key.startswith("test_"):
                metric = key[len("test_"):]
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
        logger.info("5-fold CV results: %s", summary)
        return summary


# ---------------------------------------------------------------------------
# Section 3: MLflow Experiment Tracking
# ---------------------------------------------------------------------------

class MLflowTracker:
    """Thin wrapper around MLflow for experiment tracking and model registry."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "aidataboon-topic-classifier",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    # ------------------------------------------------------------------
    def log_training_run(
        self,
        params: dict[str, Any],
        metrics: dict[str, Any],
        pipeline: Pipeline,
        X_val: list[str],
        y_val: list[str],
    ) -> str:
        """Log one complete training run and return the run_id."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        with mlflow.start_run() as run:
            # --- parameters -----------------------------------------------
            mlflow.log_params(
                {
                    "model_type": params.get("model_type", "LogisticRegression"),
                    "tfidf_max_features": params.get("tfidf_max_features", 5_000),
                    "C": params.get("C", 1.0),
                    "ngram_range": str(params.get("ngram_range", "(1,2)")),
                    "train_size": params.get("train_size", 0),
                }
            )

            # --- scalar metrics -------------------------------------------
            mlflow.log_metrics(
                {
                    "accuracy": float(metrics.get("accuracy", 0)),
                    "f1_macro": float(metrics.get("f1_macro", 0)),
                    "f1_weighted": float(metrics.get("f1_weighted", 0)),
                    "val_accuracy": float(
                        metrics.get("val_accuracy", metrics.get("accuracy", 0))
                    ),
                }
            )

            # --- confusion matrix artifact --------------------------------
            y_pred_val = pipeline.predict(X_val)
            cm = confusion_matrix(y_val, y_pred_val, labels=pipeline.classes_)
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
            disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
            ax.set_title("Validation Confusion Matrix")
            fig.tight_layout()
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # --- feature importances (top 20 terms per class) -------------
            tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
            clf: LogisticRegression = pipeline.named_steps["clf"]
            vocab = tfidf.get_feature_names_out()
            importances: dict[str, list[str]] = {}
            for i, cls in enumerate(clf.classes_):
                top_idx = np.argsort(clf.coef_[i])[-20:][::-1]
                importances[str(cls)] = [str(vocab[j]) for j in top_idx]
            importances_path = "/tmp/feature_importances.json"
            with open(importances_path, "w") as fh:
                json.dump(importances, fh, indent=2)
            mlflow.log_artifact(importances_path, artifact_path="feature_analysis")

            # --- model ---------------------------------------------------
            input_example = pd.DataFrame({"text": X_val[:5]})
            signature = mlflow.models.infer_signature(
                input_example, pipeline.predict(X_val[:5])
            )
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
                registered_model_name=None,  # register separately
            )

            run_id = run.info.run_id
            logger.info("MLflow run logged: %s", run_id)
            return run_id

    # ------------------------------------------------------------------
    def register_model(
        self,
        run_id: str,
        model_name: str = "aidataboon-topic-classifier",
    ) -> str:
        """Register the model artifact from a run and return the version number."""
        model_uri = f"runs:/{run_id}/model"
        try:
            self.client.create_registered_model(
                name=model_name,
                description="AIDataBoon topic classifier: TF-IDF + LogisticRegression",
            )
        except mlflow.exceptions.MlflowException:
            # Model already exists
            pass
        mv = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )
        logger.info("Registered model %s version %s", model_name, mv.version)
        return mv.version

    # ------------------------------------------------------------------
    def promote_to_production(
        self,
        model_name: str,
        version: str,
    ) -> None:
        """Transition a model version to the Production stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Model %s v%s promoted to Production", model_name, version)

    # ------------------------------------------------------------------
    def load_production_model(self, model_name: str) -> Pipeline:
        """Load the currently active Production model from the registry."""
        model_uri = f"models:/{model_name}/Production"
        pipeline: Pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded production model from %s", model_uri)
        return pipeline


# ---------------------------------------------------------------------------
# Section 4: Model Serving (FastAPI)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False


if _FASTAPI_AVAILABLE:

    class ClassifyRequest(BaseModel):
        text: str

    class ClassifyResponse(BaseModel):
        label: str
        confidence: float
        top_classes: list[dict[str, float]]


    class ModelServingApp:
        """FastAPI sub-application that serves the AIDataBoon topic classifier."""

        MODEL_NAME = "aidataboon-topic-classifier"

        def __init__(
            self,
            tracking_uri: str = "http://localhost:5000",
        ) -> None:
            self.tracking_uri = tracking_uri
            self.app = self._build_app()

        # --------------------------------------------------------------
        def _build_app(self) -> FastAPI:
            try:
                from prometheus_client import Histogram as PromHistogram
                REQUEST_LATENCY = PromHistogram(
                    "model_request_latency_seconds",
                    "Model serving request latency",
                    ["model_name"],
                    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
                )
            except ImportError:
                REQUEST_LATENCY = None  # type: ignore[assignment]

            @asynccontextmanager
            async def lifespan(app: FastAPI):  # type: ignore[misc]
                tracker = MLflowTracker(tracking_uri=self.tracking_uri)
                try:
                    app.state.model = tracker.load_production_model(self.MODEL_NAME)
                    app.state.model_version = "production"
                    logger.info("Production model loaded on startup")
                except Exception as exc:
                    logger.warning("Could not load production model: %s", exc)
                    app.state.model = None
                    app.state.model_version = "unavailable"
                yield
                app.state.model = None

            app = FastAPI(
                title="AIDataBoon Model Serving",
                version="1.0.0",
                lifespan=lifespan,
            )

            @app.post("/classify", response_model=ClassifyResponse)
            async def classify(request: ClassifyRequest) -> ClassifyResponse:
                if app.state.model is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")

                import time
                t0 = time.perf_counter()
                pipeline: Pipeline = app.state.model
                proba = pipeline.predict_proba([request.text])[0]
                classes = pipeline.classes_
                top3_idx = np.argsort(proba)[-3:][::-1]
                top_classes = [
                    {"label": str(classes[i]), "confidence": float(proba[i])}
                    for i in top3_idx
                ]
                latency = time.perf_counter() - t0
                if REQUEST_LATENCY is not None:
                    REQUEST_LATENCY.labels(model_name=self.MODEL_NAME).observe(latency)
                return ClassifyResponse(
                    label=str(classes[top3_idx[0]]),
                    confidence=float(proba[top3_idx[0]]),
                    top_classes=top_classes,
                )

            @app.post("/classify/batch", response_model=list[ClassifyResponse])
            async def classify_batch(
                requests: list[ClassifyRequest],
            ) -> list[ClassifyResponse]:
                if app.state.model is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")

                async def _single(req: ClassifyRequest) -> ClassifyResponse:
                    return await classify(req)

                results = await asyncio.gather(*[_single(r) for r in requests])
                return list(results)

            @app.get("/model/info")
            async def model_info() -> dict[str, Any]:
                client = MlflowClient(tracking_uri=self.tracking_uri)
                try:
                    versions = client.get_latest_versions(
                        self.MODEL_NAME, stages=["Production"]
                    )
                    if versions:
                        v = versions[0]
                        run = client.get_run(v.run_id)
                        return {
                            "model_name": self.MODEL_NAME,
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id,
                            "metrics": run.data.metrics,
                            "params": run.data.params,
                            "status": app.state.model_version,
                        }
                except Exception as exc:
                    logger.warning("Could not fetch model info: %s", exc)
                return {"model_name": self.MODEL_NAME, "status": app.state.model_version}

            return app


# ---------------------------------------------------------------------------
# Section 5: Drift Detection
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    timestamp: datetime
    feature_psi: dict[str, float]
    feature_ks: dict[str, tuple[float, float]]
    overall_drift: str  # "none" | "moderate" | "significant"
    alerts: list[str] = field(default_factory=list)


class DriftDetector:
    """PSI and KS-based production drift detection."""

    @staticmethod
    def compute_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """
        Population Stability Index.

        PSI < 0.10  → no drift
        PSI 0.10–0.20 → moderate drift
        PSI > 0.20  → significant drift
        """
        eps = 1e-8
        breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
        breakpoints = np.unique(breakpoints)

        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
        actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return psi

    @staticmethod
    def compute_ks_test(
        expected: np.ndarray,
        actual: np.ndarray,
    ) -> tuple[float, float]:
        """Return (KS statistic, p-value) via scipy two-sample KS test."""
        result = stats.ks_2samp(expected, actual)
        return float(result.statistic), float(result.pvalue)


class DataDriftMonitor:
    """Monitor feature distributions between a reference window and new batches."""

    MONITORED_FEATURES: ClassVar[list[str]] = ["title_length", "score", "topic_count"]
    REFERENCE_PATH = "/tmp/aidataboon_drift_reference.json"
    PSI_ALERT_THRESHOLD = 0.2

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
    ) -> None:
        self.detector = DriftDetector()
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._reference: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def update_reference(self, df: pd.DataFrame) -> None:
        """Persist reference feature distributions to disk."""
        ref: dict[str, list[float]] = {}
        for feat in self.MONITORED_FEATURES:
            if feat in df.columns:
                ref[feat] = df[feat].dropna().tolist()
        with open(self.REFERENCE_PATH, "w") as fh:
            json.dump(ref, fh)
        self._reference = {k: np.array(v) for k, v in ref.items()}
        logger.info("Drift reference updated: %d features, %d samples", len(ref), len(df))

    # ------------------------------------------------------------------
    def _load_reference(self) -> None:
        if self._reference:
            return
        try:
            with open(self.REFERENCE_PATH) as fh:
                raw = json.load(fh)
            self._reference = {k: np.array(v) for k, v in raw.items()}
        except FileNotFoundError:
            logger.warning("No drift reference file found at %s", self.REFERENCE_PATH)

    # ------------------------------------------------------------------
    def check_batch(self, df: pd.DataFrame) -> DriftReport:
        """Compare current batch against the reference distribution."""
        self._load_reference()

        feature_psi: dict[str, float] = {}
        feature_ks: dict[str, tuple[float, float]] = {}
        alerts: list[str] = []

        for feat in self.MONITORED_FEATURES:
            if feat not in df.columns or feat not in self._reference:
                continue
            actual = df[feat].dropna().to_numpy(dtype=float)
            expected = self._reference[feat]
            if len(actual) < 10 or len(expected) < 10:
                continue

            psi = self.detector.compute_psi(expected, actual)
            ks_stat, ks_p = self.detector.compute_ks_test(expected, actual)
            feature_psi[feat] = psi
            feature_ks[feat] = (ks_stat, ks_p)

            if psi > self.PSI_ALERT_THRESHOLD:
                alerts.append(f"ALERT: PSI drift on '{feat}' = {psi:.4f} (> {self.PSI_ALERT_THRESHOLD})")
                logger.warning("Drift detected on feature '%s': PSI=%.4f", feat, psi)

        # Prometheus counter increment
        try:
            from prometheus_client import Counter
            DRIFT_ALERT_COUNTER = Counter(
                "aidataboon_drift_alerts_total",
                "Number of PSI drift alerts fired",
                ["feature"],
            )
            for alert_feat in [f for f, p in feature_psi.items() if p > self.PSI_ALERT_THRESHOLD]:
                DRIFT_ALERT_COUNTER.labels(feature=alert_feat).inc()
        except Exception as exc:
            logger.debug("Prometheus drift alert counter not updated (%s)", exc)

        # Log PSI scores to MLflow
        self._log_drift_to_mlflow(feature_psi)

        max_psi = max(feature_psi.values(), default=0.0)
        if max_psi < 0.1:
            overall = "none"
        elif max_psi < 0.2:
            overall = "moderate"
        else:
            overall = "significant"

        return DriftReport(
            timestamp=datetime.now(UTC),
            feature_psi=feature_psi,
            feature_ks=feature_ks,
            overall_drift=overall,
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    def _log_drift_to_mlflow(self, feature_psi: dict[str, float]) -> None:
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_name="drift-check"):
                for feat, psi in feature_psi.items():
                    mlflow.log_metric(f"psi_{feat}", psi)
        except Exception as exc:
            logger.debug("Could not log drift to MLflow: %s", exc)


# ---------------------------------------------------------------------------
# Section 6: Main Runner
# ---------------------------------------------------------------------------

def run_ml_pipeline(
    knowledge_base_path: str = "data/gold/knowledge_base.parquet",
    feature_table_path: str = "data/gold/feature_table.parquet",
    mlflow_uri: str = "http://localhost:5000",
) -> None:
    """
    End-to-end ML pipeline runner:
      1. Feature engineering
      2. Model training
      3. MLflow experiment logging + model registration
      4. Drift detection on a synthetic sample
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.info("=== AIDataBoon ML Platform: Pipeline Start ===")

    # -- Feature engineering -----------------------------------------------
    feat_config = FeatureStoreConfig()
    feat_config.write_repo_files()

    logger.info("Step 1: Feature engineering (synthetic data)")
    # Synthetic DataFrame so the pipeline runs without real Gold data
    np.random.seed(42)
    n = 500
    synthetic_kb = pd.DataFrame(
        {
            "id": range(n),
            "title": [f"Post title {i}" for i in range(n)],
            "abstract": [f"Abstract text about topic {i % 10}" for i in range(n)],
            "content": ["Normal text " * 20 + ("```python\ncode\n```" if i % 3 == 0 else "") for i in range(n)],
            "tags": [[TOPIC_LABELS[i % len(TOPIC_LABELS)]] for i in range(n)],
            "source": np.random.choice(["hackernews", "reddit", "lobsters"], n),
        }
    )
    synthetic_feat = pd.DataFrame(
        {
            "id": range(n),
            "score": np.random.randint(1, 500, n),
            "num_comments": np.random.randint(0, 100, n),
            "days_since_published": np.random.randint(1, 365, n),
        }
    )

    # -- Training -------------------------------------------------------------
    logger.info("Step 2: Training TopicClassifier")
    trainer = TrainingPipeline()
    texts = [f"{r['title']} {r['abstract']}" for _, r in synthetic_kb.iterrows()]
    labels = [_derive_label(r["tags"]) for _, r in synthetic_kb.iterrows()]

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(texts, labels)
    pipeline = trainer.train(X_train, y_train)
    test_metrics = trainer.evaluate(pipeline, X_test, y_test)
    val_metrics = trainer.evaluate(pipeline, X_val, y_val)

    # -- MLflow ---------------------------------------------------------------
    logger.info("Step 3: MLflow experiment tracking")
    try:
        tracker = MLflowTracker(tracking_uri=mlflow_uri)
        run_id = tracker.log_training_run(
            params={
                "model_type": "LogisticRegression",
                "tfidf_max_features": 5_000,
                "C": 1.0,
                "ngram_range": "(1,2)",
                "train_size": len(X_train),
            },
            metrics={**test_metrics, "val_accuracy": val_metrics["accuracy"]},
            pipeline=pipeline,
            X_val=X_val,
            y_val=y_val,
        )
        version = tracker.register_model(run_id)
        tracker.promote_to_production("aidataboon-topic-classifier", version)
    except Exception as exc:
        logger.warning("MLflow step skipped (server unavailable): %s", exc)

    # -- Drift detection ------------------------------------------------------
    logger.info("Step 4: Drift detection on sample data")
    monitor = DataDriftMonitor(mlflow_tracking_uri=mlflow_uri)
    ref_df = synthetic_feat.rename(columns={"score": "score"})
    ref_df["title_length"] = synthetic_kb["title"].str.len()
    ref_df["topic_count"] = synthetic_kb["tags"].apply(len)
    monitor.update_reference(ref_df)

    # Simulate slight drift
    drifted_df = ref_df.copy()
    drifted_df["score"] = drifted_df["score"] * 2 + np.random.randint(0, 50, n)
    report = monitor.check_batch(drifted_df)
    logger.info(
        "Drift report: overall %s, PSI scores %s, alerts %s",
        report.overall_drift,
        {k: f"{v:.4f}" for k, v in report.feature_psi.items()},
        report.alerts,
    )

    logger.info("=== AIDataBoon ML Platform: Pipeline Complete ===")


if __name__ == "__main__":
    run_ml_pipeline()
