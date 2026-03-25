"""
FastAPI application for the MLOps Pipeline.

This API exposes pipeline operations, model registry queries, experiment
tracking, and data validation over HTTP.  In production MLOps, the API
layer is the integration point — CI/CD systems trigger pipeline runs,
monitoring dashboards poll for metrics, and deployment services query
the model registry.

Architecture note:
    This is a *thin* API layer.  Business logic lives in the ``src/``
    modules (pipeline, monitoring, validation).  The API only handles
    HTTP concerns: request parsing, response formatting, error codes.
    This separation makes the core logic testable without HTTP overhead.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline import MLPipeline, ModelRegistry, ExperimentTracker

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MLOps Pipeline API",
    version="1.0.0",
    description=(
        "REST API for ML pipeline orchestration, model registry, "
        "experiment tracking, and data validation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Shared state (in-memory; production would use a database) ---------------
_registry = ModelRegistry()
_tracker = ExperimentTracker()
_run_history: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ModelRegistration(BaseModel):
    """Request body for registering a new model version."""
    name: str = Field(..., description="Logical model name, e.g. 'churn-classifier'")
    version: str = Field(..., description="Version tag, e.g. '1.2' or '2026-03-25'")
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics at registration time")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra info (git SHA, author, etc.)")


class ExperimentLog(BaseModel):
    """Request body for logging an experiment."""
    name: str = Field(..., description="Experiment name or sweep identifier")
    params: Dict[str, Any] = Field(..., description="Hyperparameters used in this run")
    metrics: Dict[str, float] = Field(..., description="Resulting metrics")


class PipelineRunRequest(BaseModel):
    """Request body for triggering a demo pipeline run."""
    name: str = Field(default="api-triggered", description="Pipeline name")
    n_rows: int = Field(default=1000, ge=1, le=1_000_000, description="Simulated dataset size")


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health():
    """Liveness probe — returns 200 if the service is up."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/pipeline/status", tags=["pipeline"])
async def pipeline_status():
    """Return a summary of pipeline run history."""
    return {
        "status": "idle",
        "total_runs": len(_run_history),
        "last_run": _run_history[-1] if _run_history else None,
    }


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

@app.post("/pipeline/run", tags=["pipeline"])
async def run_pipeline(request: PipelineRunRequest):
    """Trigger a demo pipeline run with simulated steps.

    In a real system, this would launch an async job (Celery, Airflow DAG,
    Kubernetes Job) rather than blocking the HTTP request.  Here we run
    synchronously for simplicity.
    """
    pipe = MLPipeline(request.name)
    pipe.add_step("ingest", lambda **kw: {"n_rows": request.n_rows})
    pipe.add_step("validate", lambda n_rows=0, **kw: {"valid": True, "n_rows": n_rows})
    pipe.add_step("split", lambda n_rows=0, **kw: {
        "train": int(n_rows * 0.8),
        "test": n_rows - int(n_rows * 0.8),
    })
    pipe.add_step("train", lambda **kw: {"model": "trained", "accuracy": 0.92})

    metrics = pipe.run()
    _run_history.append(metrics)
    return metrics


@app.get("/pipeline/history", tags=["pipeline"])
async def pipeline_history(
    limit: int = Query(default=10, ge=1, le=100, description="Max runs to return"),
):
    """Return recent pipeline run results."""
    return {"runs": _run_history[-limit:], "total": len(_run_history)}


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

@app.get("/models", tags=["registry"])
async def list_models():
    """List all registered model versions."""
    return {"models": _registry.list_models()}


@app.post("/models/register", tags=["registry"])
async def register_model(body: ModelRegistration):
    """Register a new model version with its evaluation metrics."""
    entry = _registry.register(
        name=body.name,
        version=body.version,
        metrics=body.metrics,
        metadata=body.metadata,
    )
    return {"registered": entry}


@app.get("/models/{name}/{version}", tags=["registry"])
async def get_model(name: str, version: str):
    """Look up a specific model version."""
    model = _registry.get_model(name, version)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' v{version} not found.")
    return model


# ---------------------------------------------------------------------------
# Experiment Tracking
# ---------------------------------------------------------------------------

@app.post("/experiments/log", tags=["experiments"])
async def log_experiment(body: ExperimentLog):
    """Log an experiment run with its parameters and results."""
    exp = _tracker.log(name=body.name, params=body.params, metrics=body.metrics)
    return {"logged": exp}


@app.get("/experiments", tags=["experiments"])
async def list_experiments():
    """Return all logged experiments."""
    return {"experiments": _tracker._experiments}


@app.get("/experiments/best", tags=["experiments"])
async def best_experiment(
    metric: str = Query(..., description="Metric name to optimise"),
    mode: str = Query(default="max", pattern="^(max|min)$", description="'max' or 'min'"),
):
    """Find the experiment with the best value for a given metric."""
    best = _tracker.get_best(metric=metric, mode=mode)
    if best is None:
        raise HTTPException(status_code=404, detail="No experiments logged yet.")
    return {"best": best}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
