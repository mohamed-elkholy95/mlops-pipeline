"""
Learn — Educational deep-dives into MLOps concepts.

This page provides interactive explorations of the core ideas behind
production ML systems, connecting theory to the project's implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("📖 MLOps Concepts")
st.markdown(
    "Deep-dive into the ideas behind this project. "
    "Each section connects theory to implementation."
)

# ---------------------------------------------------------------------------
# 1. Pipeline Architecture
# ---------------------------------------------------------------------------

with st.expander("🔗 Pipeline Architecture", expanded=True):
    st.markdown("""
    ### Why Pipelines?

    Monolithic training scripts are the Jupyter notebooks of production:
    fine for exploration, terrible for reliability.  Pipelines decompose
    workflows into **discrete steps** with:

    - **Clear contracts** — each step declares what it needs and produces
    - **Independent testing** — test data ingestion without training
    - **Parallel execution** — independent steps can run concurrently (DAGs)
    - **Fault isolation** — a failed step doesn't corrupt other outputs

    ### This Project's Approach

    ```
    ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌─────────┐
    │  Ingest  │ →  │ Validate  │ →  │  Split  │ →  │  Train  │
    └──────────┘    └───────────┘    └─────────┘    └─────────┘
         ↓               ↓               ↓              ↓
       context ──────→ context ──────→ context ────→ context
    ```

    Steps share a **context dict**.  Each step reads what it needs and writes
    new keys.  This is the *implicit wiring* pattern — simple for linear
    pipelines, but consider explicit dependency graphs (Airflow, Dagster)
    for complex workflows.

    ### Key Classes

    | Class | Responsibility |
    |-------|---------------|
    | `PipelineStep` | Wraps a callable, tracks timing and status |
    | `MLPipeline` | Orchestrates steps, collects metrics |

    **Code:** `src/pipeline.py`
    """)

# ---------------------------------------------------------------------------
# 2. Model Registry
# ---------------------------------------------------------------------------

with st.expander("📦 Model Registry"):
    st.markdown("""
    ### The "Which Model Is This?" Problem

    Without a registry, production models are mystery boxes.  Someone
    trained it, someone deployed it, but nobody can tell you:
    - What data was it trained on?
    - What hyperparameters were used?
    - How did it perform on the test set?
    - Which git commit produced it?

    ### Registry as Version Control for Models

    Just as git versions code, a model registry versions *trained artifacts*:

    ```
    churn-classifier/
    ├── v1.0  →  accuracy=0.89, trained=2026-01-15
    ├── v1.1  →  accuracy=0.91, trained=2026-02-20
    └── v2.0  →  accuracy=0.94, trained=2026-03-10  ← production
    ```

    Each entry stores:
    - **Metrics** at registration time (not re-evaluated on new data)
    - **Metadata** like git SHA, dataset hash, author
    - **Timestamp** for auditing

    ### Production Registries

    Tools like MLflow, Weights & Biases, and cloud registries add:
    - Stage management (staging → canary → production)
    - Approval workflows before promotion
    - Automatic A/B testing between versions
    """)

# ---------------------------------------------------------------------------
# 3. Data Drift
# ---------------------------------------------------------------------------

with st.expander("🚨 Data Drift"):
    st.markdown("""
    ### When the World Changes

    ML models assume the future looks like the past.  When it doesn't,
    predictions degrade.  Common drift causes:

    | Cause | Example |
    |-------|---------|
    | User behaviour change | COVID shifted e-commerce patterns |
    | Upstream data change | A partner API changes field formats |
    | Seasonal patterns | Holiday spending vs. January |
    | Population shift | Expanding to a new market segment |

    ### Detection Methods

    **This project uses mean-shift detection:**

    ```
    drift_score = mean(|μ_incoming - μ_reference| / σ_reference)
    ```

    Simple, interpretable, but limited — it only catches location shifts.
    For production, consider:

    - **KS Test** — non-parametric, catches any distribution change
    - **PSI** — standard in banking/finance for scorecard monitoring
    - **KL Divergence** — information-theoretic, but sensitive to binning

    ### Response Strategies

    1. **Alert** — notify the ML team for investigation
    2. **Auto-retrain** — trigger retraining on fresh data
    3. **Fallback** — switch to a simpler, more robust model
    4. **Quarantine** — stop serving predictions until investigated
    """)

# ---------------------------------------------------------------------------
# 4. Experiment Tracking
# ---------------------------------------------------------------------------

with st.expander("🧪 Experiment Tracking"):
    st.markdown("""
    ### The Spreadsheet Graveyard

    Every ML team has a spreadsheet of "runs I tried."  It starts organized,
    then devolves into:
    - "lr=0.01 (maybe 0.001?)"
    - "best model (copy)" → "best model (copy 2)"
    - "TODO: which dataset was this?"

    Experiment tracking automates this with structured logging:

    ```python
    tracker.log(
        name="bert-finetune-v3",
        params={"lr": 0.001, "epochs": 10, "dropout": 0.3},
        metrics={"accuracy": 0.94, "f1": 0.92, "loss": 0.18},
    )
    ```

    ### What to Track

    | Category | Examples |
    |----------|---------|
    | Hyperparameters | Learning rate, batch size, architecture |
    | Data | Dataset version, split ratio, augmentation |
    | Environment | GPU type, library versions, random seed |
    | Results | All metrics, not just the primary one |

    ### Comparing Experiments

    The `compare()` method flattens all experiments into a DataFrame,
    making it trivial to sort, filter, and find the winning configuration.
    """)

# ---------------------------------------------------------------------------
# 5. Data Validation
# ---------------------------------------------------------------------------

with st.expander("✅ Data Validation"):
    st.markdown("""
    ### The Cheapest Bug Fix

    Data validation is the highest-ROI practice in MLOps.  A 5-line schema
    check can prevent a 6-hour training job from producing garbage:

    ```python
    schema = DataSchema(columns=[
        ColumnSchema("age", nullable=False, min_value=0, max_value=150),
        ColumnSchema("income", nullable=False, min_value=0),
        ColumnSchema("segment", allowed_values={"A", "B", "C"}),
    ])
    report = DataValidator(schema).validate(df)
    if not report.passed:
        raise ValueError(report.summary)
    ```

    ### What to Validate

    - **Schema** — correct columns, types, no surprises
    - **Completeness** — acceptable null rates
    - **Range** — values within physical/business bounds
    - **Consistency** — cross-column rules (e.g., end_date > start_date)
    - **Freshness** — data isn't stale

    ### The Google Insight

    Google's paper on ML data management found that data issues — not
    model bugs — cause the majority of production ML failures.  Validation
    is your first line of defence.
    """)

# ---------------------------------------------------------------------------
# Further Reading
# ---------------------------------------------------------------------------

st.header("📚 Further Reading")
st.markdown("""
- **[Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)** — The classic NeurIPS paper
- **[Google MLOps Maturity Model](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)** — Levels 0–2 explained
- **[ML Test Score](https://research.google/pubs/pub46555/)** — Rubric for production ML readiness
- **[Evidently AI Blog](https://www.evidentlyai.com/blog)** — Practical monitoring guides
- **[Made With ML](https://madewithml.com/)** — End-to-end MLOps course
""")
