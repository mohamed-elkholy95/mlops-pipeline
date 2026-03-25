# MLOps Concepts Guide

A reference for the core ideas behind this project. Each section maps a concept
to the code that implements it.

---

## 1. Pipeline Orchestration

**What:** Breaking an ML workflow into discrete, composable steps that execute
in sequence (or as a DAG) with shared context.

**Why:** Monolithic training scripts are hard to test, debug, and modify.
Pipelines give you:
- **Isolation** — each step has clear inputs/outputs
- **Reusability** — swap a preprocessing step without touching training
- **Observability** — per-step timing, logging, and status tracking
- **Reproducibility** — the same pipeline definition produces the same flow

**In this project:** `MLPipeline` + `PipelineStep` in `src/pipeline.py`

**Production tools:** Apache Airflow, Kubeflow Pipelines, Prefect, Dagster,
AWS Step Functions, Vertex AI Pipelines

---

## 2. Model Registry

**What:** A versioned catalog that links model artifacts to their training
metadata, evaluation metrics, and lineage information.

**Why:** Without a registry, answering "which model is deployed and how was it
trained?" requires archaeology. Registries provide:
- **Versioning** — roll back to any prior model
- **Auditability** — trace predictions back to training data and code
- **Promotion workflows** — stage → canary → production

**In this project:** `ModelRegistry` in `src/pipeline.py`

**Production tools:** MLflow Model Registry, Weights & Biases, SageMaker Model
Registry, Vertex AI Model Registry

---

## 3. Experiment Tracking

**What:** Logging hyperparameters, metrics, and artifacts for every training run
so you can compare, reproduce, and iterate.

**Why:** ML development is inherently experimental. Without tracking:
- You lose the parameter settings that produced your best model
- You can't compare runs systematically
- Ablation studies require manual spreadsheets

**In this project:** `ExperimentTracker` in `src/pipeline.py`

**Key operations:**
- `log(name, params, metrics)` → record a run
- `get_best(metric, mode)` → find the winner
- `compare()` → tabular comparison across runs

---

## 4. Data Validation

**What:** Programmatic checks that verify data conforms to expected schemas,
distributions, and business rules *before* it enters the pipeline.

**Why:** Google's research found that data quality issues cause more ML failures
than model bugs. Validation catches:
- Missing or extra columns (schema changes)
- Null values in non-nullable fields
- Out-of-range values (negative ages, future dates)
- Unexpected categorical values

**In this project:** `DataValidator`, `DataSchema`, `ColumnSchema` in
`src/validation.py`

**Production tools:** Great Expectations, TensorFlow Data Validation (TFDV),
Pandera, Deequ (AWS)

---

## 5. Data Drift Detection

**What:** Monitoring whether the statistical distribution of incoming data has
shifted compared to the training data.

**Why:** ML models are functions of their training distribution. When real-world
data shifts (new user demographics, seasonal changes, upstream bugs), model
accuracy degrades — often silently.

**Detection approaches:**
| Method | What It Detects | Complexity |
|--------|----------------|------------|
| Mean shift (this project) | Location changes | Low |
| KS test | Any distribution change | Medium |
| PSI (Population Stability Index) | Binned distribution shift | Medium |
| KL divergence | Information-theoretic distance | High |
| MMD (Maximum Mean Discrepancy) | Kernel-based distribution distance | High |

**In this project:** `DataDriftDetector` in `src/monitoring.py`

---

## 6. Performance Monitoring

**What:** Tracking model metrics (accuracy, latency, error rates) over time and
alerting when they degrade beyond acceptable thresholds.

**Why:** Models don't crash — they degrade. A model returning 85 % accuracy
instead of 92 % still returns HTTP 200. Without monitoring, nobody notices until
revenue drops.

**Degradation signals:**
- **Sudden drops** → upstream data pipeline broke
- **Gradual decline** → concept drift (the world changed)
- **Periodic dips** → seasonal patterns not captured by the model

**In this project:** `PerformanceMonitor` in `src/monitoring.py`

---

## 7. The MLOps Maturity Model

Teams typically progress through these levels:

| Level | Characteristics |
|-------|----------------|
| **0 — Manual** | Jupyter notebooks, manual deployment, no versioning |
| **1 — Pipeline** | Automated training, basic experiment tracking |
| **2 — CI/CD for ML** | Automated testing, model validation gates, registry |
| **3 — Full MLOps** | Continuous training, monitoring, drift detection, auto-retraining |

This project implements concepts from **Levels 1–3** in a lightweight way.

---

## 8. Key Principles

### Reproducibility
Every run should be reproducible given the same code, data, and parameters.
This means: version your data, pin your dependencies, seed your random state,
and log everything.

### Fail Fast
Catch problems early. Data validation should happen *before* a 6-hour training
job, not after. Schema checks are cheap; GPU hours are not.

### Separation of Concerns
- **Pipeline logic** ≠ **ML code** ≠ **API code** ≠ **monitoring code**
- Each module has a single responsibility
- The API layer doesn't contain business logic
- Monitoring doesn't know about HTTP

### Observability
If you can't see it, you can't fix it. Log step durations, track metrics,
alert on drift. The three pillars: **logs**, **metrics**, **traces**.

---

## Further Reading

- [Google: Reliable ML in the Wild](https://research.google/pubs/pub46555/)
  — The "ML Test Score" paper on testing production ML systems
- [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
  — The classic NeurIPS paper on ML system complexity
- [MLOps: Continuous Delivery for ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
  — Google Cloud's MLOps maturity framework
- [Evidently AI Blog](https://www.evidentlyai.com/blog)
  — Practical guides on monitoring and drift detection
