"""MLOps pipeline orchestration."""
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)


class PipelineStep:
    """A single pipeline step."""

    def __init__(self, name: str, func: callable = None) -> None:
        self.name = name
        self._func = func
        self.status = "pending"
        self.duration: float = 0.0
        self.output: Any = None
        self.error: Optional[str] = None

    def run(self, **kwargs) -> Any:
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

    def to_dict(self) -> Dict:
        return {"name": self.name, "status": self.status, "duration": self.duration, "error": self.error}


class MLPipeline:
    """ML pipeline with step orchestration."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._steps: List[PipelineStep] = []
        self._metrics: Dict[str, Any] = {}

    def add_step(self, name: str, func: callable = None) -> "MLPipeline":
        self._steps.append(PipelineStep(name, func))
        return self

    def run(self, **initial_kwargs) -> Dict[str, Any]:
        context = dict(initial_kwargs)
        for step in self._steps:
            logger.info("Running step: %s", step.name)
            step.run(**context)
            if step.status == "success" and step.output is not None:
                if isinstance(step.output, dict):
                    context.update(step.output)
        total_duration = sum(s.duration for s in self._steps)
        self._metrics = {
            "pipeline": self.name, "status": "success" if all(s.status == "success" for s in self._steps) else "failed",
            "total_duration": round(total_duration, 4),
            "steps": [s.to_dict() for s in self._steps],
            "timestamp": datetime.now().isoformat(),
        }
        logger.info("Pipeline '%s' complete: %s (%.2fs)", self.name, self._metrics["status"], total_duration)
        return self._metrics

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def save_metrics(self, path: Path = None) -> None:
        path = path or Path("metrics/run.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)
        logger.info("Metrics saved to %s", path)


class ModelRegistry:
    """Simple model registry."""

    def __init__(self, registry_dir: Path = None) -> None:
        self.registry_dir = registry_dir or Path("artifacts/models")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Dict] = {}

    def register(self, name: str, version: str, metrics: Dict, metadata: Dict = None) -> Dict:
        entry = {"name": name, "version": version, "metrics": metrics,
                "metadata": metadata or {}, "registered_at": datetime.now().isoformat()}
        self._models[f"{name}/{version}"] = entry
        path = self.registry_dir / f"{name}_v{version}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)
        logger.info("Registered model: %s v%s", name, version)
        return entry

    def list_models(self) -> List[Dict]:
        return list(self._models.values())

    def get_model(self, name: str, version: str) -> Optional[Dict]:
        return self._models.get(f"{name}/{version}")


class ExperimentTracker:
    """Track experiments and their metrics."""

    def __init__(self, storage_dir: Path = None) -> None:
        self.storage_dir = storage_dir or Path("metrics/experiments")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: List[Dict] = []

    def log(self, name: str, params: Dict, metrics: Dict) -> Dict:
        exp = {"name": name, "params": params, "metrics": metrics,
               "timestamp": datetime.now().isoformat(), "id": len(self._experiments)}
        self._experiments.append(exp)
        path = self.storage_dir / f"exp_{exp['id']:04d}.json"
        with open(path, "w") as f:
            json.dump(exp, f, indent=2)
        return exp

    def get_best(self, metric: str, mode: str = "max") -> Optional[Dict]:
        if not self._experiments: return None
        key = lambda e: e["metrics"].get(metric, float("-inf") if mode == "max" else float("inf"))
        return max(self._experiments, key=key) if mode == "max" else min(self._experiments, key=key)

    def compare(self) -> pd.DataFrame:
        import pandas as pd
        if not self._experiments: return pd.DataFrame()
        rows = []
        for exp in self._experiments:
            row = {"id": exp["id"], "name": exp["name"]}
            row.update({f"param_{k}": v for k, v in exp["params"].items()})
            row.update(exp["metrics"])
            rows.append(row)
        return pd.DataFrame(rows)
