"""
MLOps pipeline orchestration.

This module implements the core abstractions for building reproducible ML
workflows: a composable pipeline executor, a versioned model registry, and
an experiment tracker.  Together they form the backbone of an MLOps system
that separates *what* to run from *how* to run it — a key principle in
production ML engineering.

Design Pattern — Chain of Responsibility:
    The pipeline passes a shared context dict through each step.  Each step
    can read from the context and write new keys back.  This is similar to
    middleware chains in web frameworks, and makes it easy to add, remove,
    or reorder processing stages without touching other steps.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Step
# ---------------------------------------------------------------------------

class PipelineStep:
    """A single, self-contained pipeline step.

    Each step wraps a callable, records execution metadata (duration, status,
    errors), and converts itself to a serialisable dict for audit logging.

    Why track duration per step?
        In production, step-level timing lets you pinpoint bottlenecks.
        A slow "feature engineering" step (say 80 % of wall time) tells you
        where to focus optimisation, while a slow "train" step might indicate
        your model complexity has outgrown the hardware.
    """

    def __init__(self, name: str, func: Optional[Callable] = None) -> None:
        self.name = name
        self._func = func
        self.status: str = "pending"
        self.duration: float = 0.0
        self.output: Any = None
        self.error: Optional[str] = None

    def run(self, **kwargs) -> Any:
        """Execute the step's callable with the current pipeline context.

        Timing uses ``time.time()`` (wall-clock) rather than
        ``time.perf_counter()`` because we care about real elapsed time
        including I/O waits — not just CPU time.
        """
        start = time.time()
        try:
            self.output = self._func(**kwargs) if self._func else kwargs
            self.status = "success"
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error("Step '%s' failed: %s", self.name, e)
        self.duration = round(time.time() - start, 4)
        return self.output

    def to_dict(self) -> Dict[str, Any]:
        """Serialise step metadata — useful for pipeline run reports."""
        return {
            "name": self.name,
            "status": self.status,
            "duration": self.duration,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# ML Pipeline
# ---------------------------------------------------------------------------

class MLPipeline:
    """ML pipeline with ordered step orchestration.

    Usage::

        pipe = MLPipeline("training-v2")
        pipe.add_step("ingest", ingest_fn)
        pipe.add_step("preprocess", preprocess_fn)
        pipe.add_step("train", train_fn)
        result = pipe.run(dataset_path="/data/train.csv")

    The ``run()`` method passes a **shared context dict** through each step.
    Steps that return a dict have their keys merged into the context, so
    downstream steps automatically see upstream outputs.  This is the
    *implicit wiring* pattern — simple for linear pipelines but should be
    replaced with explicit dependency declarations for complex DAGs.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._steps: List[PipelineStep] = []
        self._metrics: Dict[str, Any] = {}

    # -- Builder pattern: allows method chaining (pipe.add_step(...).add_step(...))

    def add_step(self, name: str, func: Optional[Callable] = None) -> "MLPipeline":
        """Append a step.  Returns ``self`` for fluent chaining."""
        self._steps.append(PipelineStep(name, func))
        return self

    def run(self, **initial_kwargs) -> Dict[str, Any]:
        """Execute all steps sequentially and collect metrics.

        Pipeline status is ``"success"`` only when *every* step succeeds.
        This fail-fast philosophy prevents silent data corruption — a failed
        validation step should stop the whole pipeline, not let bad data
        sneak into training.
        """
        context: Dict[str, Any] = dict(initial_kwargs)

        for step in self._steps:
            logger.info("Running step: %s", step.name)
            step.run(**context)
            if step.status == "success" and isinstance(step.output, dict):
                context.update(step.output)

        total_duration = sum(s.duration for s in self._steps)
        all_passed = all(s.status == "success" for s in self._steps)

        self._metrics = {
            "pipeline": self.name,
            "status": "success" if all_passed else "failed",
            "total_duration": round(total_duration, 4),
            "steps": [s.to_dict() for s in self._steps],
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(
            "Pipeline '%s' complete: %s (%.2fs)",
            self.name,
            self._metrics["status"],
            total_duration,
        )
        return self._metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Return the metrics dict from the last ``run()`` call."""
        return self._metrics

    def save_metrics(self, path: Optional[Path] = None) -> Path:
        """Persist run metrics as JSON for downstream analysis.

        Returns the resolved path so callers can log or display it.
        """
        path = path or Path("metrics/run.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)
        logger.info("Metrics saved to %s", path)
        return path


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Simple file-based model registry.

    In production, tools like MLflow, Weights & Biases, or cloud-native
    registries (SageMaker, Vertex AI) fill this role.  The key abstraction
    is the same: a **versioned catalog** that links a model artifact to its
    training metrics and metadata so you can always answer *"which model is
    in production and how was it trained?"*.

    Key concepts:
        - **Model name**: logical identifier (e.g., ``"churn-classifier"``)
        - **Version**: semantic or date-based tag (e.g., ``"1.2"`` or ``"2026-03-25"``)
        - **Metrics**: the evaluation scores at registration time
        - **Metadata**: any extra info (git SHA, dataset hash, author, etc.)
    """

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self.registry_dir = registry_dir or Path("artifacts/models")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        version: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a model version with its evaluation metrics.

        The entry is stored both in-memory and on disk.  The dual-write
        ensures the registry survives process restarts (disk) while
        remaining fast to query within a session (memory).
        """
        entry: Dict[str, Any] = {
            "name": name,
            "version": version,
            "metrics": metrics,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
        }
        key = f"{name}/{version}"
        self._models[key] = entry

        # Persist to disk — one JSON file per model version
        path = self.registry_dir / f"{name}_v{version}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)
        logger.info("Registered model: %s v%s", name, version)
        return entry

    def list_models(self) -> List[Dict[str, Any]]:
        """Return all registered model entries."""
        return list(self._models.values())

    def get_model(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        """Look up a specific model version.  Returns ``None`` if not found."""
        return self._models.get(f"{name}/{version}")


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Track hyperparameter experiments and compare results.

    Experiment tracking solves the "which run was best?" problem that
    plagues ad-hoc ML development.  By logging *params* (what you tried)
    alongside *metrics* (what happened), you build a searchable history
    that makes ablation studies, hyperparameter sweeps, and regression
    checks trivial.

    This implementation uses flat JSON files — one per experiment.  For
    larger teams, consider MLflow Tracking or Weights & Biases which add
    UI, collaboration, and artifact linking.
    """

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir or Path("metrics/experiments")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: List[Dict[str, Any]] = []

    def log(
        self,
        name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a single experiment run.

        Each experiment gets a sequential ID and is written to disk
        immediately — no buffering.  In production you might batch writes
        or use a database, but for portfolio work, simplicity wins.
        """
        exp: Dict[str, Any] = {
            "name": name,
            "params": params,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "id": len(self._experiments),
        }
        self._experiments.append(exp)

        path = self.storage_dir / f"exp_{exp['id']:04d}.json"
        with open(path, "w") as f:
            json.dump(exp, f, indent=2)
        return exp

    def get_best(
        self,
        metric: str,
        mode: str = "max",
    ) -> Optional[Dict[str, Any]]:
        """Find the experiment with the best value for *metric*.

        Args:
            metric: Name of the metric to optimise.
            mode: ``"max"`` for metrics where higher is better (accuracy,
                  F1) or ``"min"`` for loss/error metrics.

        Why separate *mode*?
            Some metrics improve when they go up (accuracy) and some when
            they go down (MSE, loss).  Making this explicit avoids a
            common bug where you accidentally select the *worst* model.
        """
        if not self._experiments:
            return None

        sentinel = float("-inf") if mode == "max" else float("inf")

        def key_fn(e: Dict[str, Any]) -> float:
            return e["metrics"].get(metric, sentinel)

        if mode == "max":
            return max(self._experiments, key=key_fn)
        return min(self._experiments, key=key_fn)

    def compare(self) -> pd.DataFrame:
        """Build a comparison DataFrame across all logged experiments.

        Flattens params into ``param_<name>`` columns so they sit beside
        metric columns — handy for quick visual comparison or exporting
        to a spreadsheet.
        """
        if not self._experiments:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for exp in self._experiments:
            row: Dict[str, Any] = {"id": exp["id"], "name": exp["name"]}
            row.update({f"param_{k}": v for k, v in exp["params"].items()})
            row.update(exp["metrics"])
            rows.append(row)
        return pd.DataFrame(rows)
