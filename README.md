<div align="center">

# ⚙️ MLOps Pipeline

**Production-grade ML pipeline orchestration** with experiment tracking, model registry, data validation, drift detection, and performance monitoring

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-50%2B%20passed-success?style=flat-square)](#testing)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

An **MLOps infrastructure toolkit** that implements the core abstractions for building, tracking, validating, and monitoring ML workflows.  Designed as an educational reference that maps each component to its production equivalent (MLflow, Airflow, Great Expectations, etc.).

### What This Project Demonstrates

| Concept | Implementation | Production Equivalent |
|---------|---------------|----------------------|
| Pipeline orchestration | `MLPipeline` + `PipelineStep` | Airflow, Kubeflow, Dagster |
| Model versioning | `ModelRegistry` | MLflow, W&B, SageMaker |
| Experiment tracking | `ExperimentTracker` | MLflow Tracking, W&B |
| Data validation | `DataValidator` + `DataSchema` | Great Expectations, TFDV |
| Data drift detection | `DataDriftDetector` | Evidently AI, NannyML |
| Performance monitoring | `PerformanceMonitor` | Prometheus + Grafana |
| REST API | FastAPI (9 endpoints) | Cloud ML APIs |
| Interactive dashboard | Streamlit (5 pages) | Internal tools |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  Ingest  │ → │ Validate │ → │  Split   │ → │  Train   │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │              │                              │           │
│       ▼              ▼                              ▼           │
│  ┌──────────┐   ┌──────────┐                  ┌──────────┐    │
│  │  Drift   │   │  Schema  │                  │  Model   │    │
│  │ Detector │   │  Report  │                  │ Registry │    │
│  └──────────┘   └──────────┘                  └──────────┘    │
│       │                                            │           │
│       ▼                                            ▼           │
│  ┌──────────┐                              ┌──────────────┐   │
│  │  Perf    │                              │  Experiment  │   │
│  │ Monitor  │                              │   Tracker    │   │
│  └──────────┘                              └──────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│           REST API (FastAPI)  │  Dashboard (Streamlit)          │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- 🔗 **Pipeline Orchestration** — Composable step-based pipeline with context passing, per-step timing, and fail-fast error handling
- 📦 **Model Registry** — Versioned model storage with metrics, metadata, and dual-write persistence (memory + disk)
- 🧪 **Experiment Tracker** — Log, compare, and find best experiments across any metric with DataFrame export
- ✅ **Data Validation** — Schema-based checks for dtypes, nullability, ranges, and categorical values with structured reports
- 🚨 **Data Drift Detection** — Z-score mean shift detection with configurable thresholds and first-call calibration
- 📉 **Performance Monitor** — Time-series metric tracking with relative-change degradation alerts
- 🚀 **REST API** — 9 endpoints for pipeline execution, model registry, and experiment tracking
- 📊 **5-Page Dashboard** — Overview, pipeline runner, experiment comparison, interactive monitoring, and educational content

## Project Structure

```
15-mlops-pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py           # Centralised settings and directory setup
│   ├── pipeline.py         # MLPipeline, ModelRegistry, ExperimentTracker
│   ├── monitoring.py       # DataDriftDetector, PerformanceMonitor
│   ├── validation.py       # DataValidator, DataSchema, ColumnSchema
│   └── api/
│       ├── __init__.py
│       └── main.py         # FastAPI app with 9 endpoints
├── tests/
│   ├── test_pipeline.py    # Pipeline, registry, tracker tests
│   ├── test_monitoring.py  # Drift and performance tests
│   ├── test_validation.py  # Schema validation tests
│   └── test_api.py         # API endpoint tests
├── streamlit_app/
│   ├── app.py              # Multi-page app entry point
│   └── pages/
│       ├── 1_📊_Overview.py    # Architecture and component cards
│       ├── 2_🚀_Pipeline.py    # Interactive pipeline runner
│       ├── 3_📈_Experiments.py  # Experiment comparison table
│       ├── 4_🔍_Monitoring.py   # Drift and performance simulators
│       └── 5_📖_Learn.py       # Educational concept deep-dives
├── docs/
│   ├── ARCHITECTURE.md     # System design decisions
│   ├── CONCEPTS.md         # MLOps concepts guide
│   ├── CONTRIBUTING.md     # Contribution guidelines
│   └── DEVELOPMENT.md      # Development setup guide
├── artifacts/models/       # Registered model JSON files
├── metrics/experiments/    # Experiment log JSON files
├── examples/
│   ├── quickstart.py       # Getting started script
│   └── demo.py             # Full demo of all features
└── requirements.txt        # Python dependencies
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/mohamed-elkholy95/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start the API server
uvicorn src.api.main:app --reload --port 8015

# Launch the dashboard
streamlit run streamlit_app/app.py
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/pipeline/status` | Pipeline run summary |
| `POST` | `/pipeline/run` | Trigger a pipeline run |
| `GET` | `/pipeline/history` | Recent run results |
| `GET` | `/models` | List registered models |
| `POST` | `/models/register` | Register a model version |
| `GET` | `/models/{name}/{version}` | Look up a model |
| `POST` | `/experiments/log` | Log an experiment |
| `GET` | `/experiments/best?metric=accuracy` | Find best experiment |

### Example: Register a Model

```bash
curl -X POST http://localhost:8015/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "churn-classifier",
    "version": "1.0",
    "metrics": {"accuracy": 0.93, "f1": 0.91},
    "metadata": {"git_sha": "abc123"}
  }'
```

### Example: Log an Experiment

```bash
curl -X POST http://localhost:8015/experiments/log \
  -H "Content-Type: application/json" \
  -d '{
    "name": "lr-sweep",
    "params": {"lr": 0.001, "epochs": 50},
    "metrics": {"accuracy": 0.94, "loss": 0.18}
  }'
```

## Usage Examples

### Pipeline with Validation

```python
from src.pipeline import MLPipeline
from src.validation import DataValidator, DataSchema, ColumnSchema

# Define data schema
schema = DataSchema(columns=[
    ColumnSchema("age", dtype="float64", nullable=False, min_value=0, max_value=150),
    ColumnSchema("income", dtype="float64", nullable=False, min_value=0),
])

# Build pipeline with validation step
pipe = MLPipeline("training-v1")
pipe.add_step("ingest", load_data)
pipe.add_step("validate", lambda df, **kw: validate_data(df, schema))
pipe.add_step("train", train_model)
result = pipe.run()
```

### Drift Detection

```python
from src.monitoring import DataDriftDetector
import numpy as np

# Calibrate with training data
detector = DataDriftDetector(X_train, threshold=0.1)

# Check incoming batch
result = detector.detect(X_new_batch)
if result["drift_detected"]:
    print(f"⚠️ Drift detected! Score: {result['score']:.4f}")
```

### Experiment Tracking

```python
from src.pipeline import ExperimentTracker

tracker = ExperimentTracker()
tracker.log("bert-v1", {"lr": 0.001, "epochs": 10}, {"accuracy": 0.88})
tracker.log("bert-v2", {"lr": 0.0005, "epochs": 20}, {"accuracy": 0.94})

best = tracker.get_best("accuracy")
print(f"Best: {best['params']} → {best['metrics']}")

# Compare all experiments
df = tracker.compare()
print(df.sort_values("accuracy", ascending=False))
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific module tests
python -m pytest tests/test_validation.py -v
python -m pytest tests/test_monitoring.py -v
```

## Key Concepts

This project implements ideas from these foundational MLOps resources:

- **[Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)** — Why ML systems accumulate debt faster than traditional software
- **[Google MLOps Maturity Model](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)** — Levels 0–2 of ML automation
- **[ML Test Score](https://research.google/pubs/pub46555/)** — A rubric for production ML readiness

See [`docs/CONCEPTS.md`](docs/CONCEPTS.md) for detailed explanations of each concept.

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
