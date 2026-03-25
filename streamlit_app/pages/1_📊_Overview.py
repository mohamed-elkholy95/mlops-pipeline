"""
Overview — System architecture, component cards, and project metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("⚙️ MLOps Pipeline")
st.markdown(
    "A production-grade ML pipeline toolkit with experiment tracking, "
    "model registry, data validation, and monitoring."
)

# ---------------------------------------------------------------------------
# Architecture Diagram
# ---------------------------------------------------------------------------

st.header("🏗️ System Architecture")
st.code("""
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
│  │  Data    │   │  Schema  │                  │  Model   │    │
│  │  Drift   │   │  Report  │                  │ Registry │    │
│  │ Detector │   │          │                  │          │    │
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
""", language=None)

# ---------------------------------------------------------------------------
# Component Cards
# ---------------------------------------------------------------------------

st.header("🧩 Components")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔗 Pipeline")
    st.markdown("""
    **Step-based orchestration** with shared context passing.

    - Composable `PipelineStep` objects
    - Per-step timing and error tracking
    - Fail-fast: one failure stops the pipeline
    - JSON metrics export for audit trails
    """)
    with st.expander("Key Classes"):
        st.markdown("""
        - `PipelineStep` — wraps a callable with status tracking
        - `MLPipeline` — orchestrates steps, collects metrics
        """)

with col2:
    st.subheader("📦 Registry & Tracking")
    st.markdown("""
    **Model versioning** and **experiment comparison**.

    - Register models with metrics + metadata
    - Version-based lookup (`name/version`)
    - Log experiments with params and results
    - Find best run by any metric (max/min)
    """)
    with st.expander("Key Classes"):
        st.markdown("""
        - `ModelRegistry` — versioned model catalog on disk
        - `ExperimentTracker` — log, compare, find best
        """)

with col3:
    st.subheader("🚨 Monitoring")
    st.markdown("""
    **Drift detection** and **degradation alerts**.

    - Statistical drift scoring vs. reference data
    - Configurable thresholds per use case
    - Time-series performance tracking
    - Relative-change degradation detection
    """)
    with st.expander("Key Classes"):
        st.markdown("""
        - `DataDriftDetector` — Z-score mean shift detection
        - `PerformanceMonitor` — metric trend analysis
        """)

# Second row
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("✅ Validation")
    st.markdown("""
    **Schema-based data quality** checks.

    - Column existence and dtype verification
    - Null detection for non-nullable fields
    - Numeric range enforcement (min/max)
    - Categorical value whitelisting
    """)

with col5:
    st.subheader("🚀 REST API")
    st.markdown("""
    **FastAPI endpoints** for integration.

    - `POST /pipeline/run` — trigger pipeline
    - `POST /models/register` — register model
    - `POST /experiments/log` — log experiment
    - `GET /experiments/best` — find winner
    """)

with col6:
    st.subheader("📊 Dashboard")
    st.markdown("""
    **5-page Streamlit app** for exploration.

    - Overview (this page)
    - Pipeline runner with live results
    - Experiment comparison table
    - Interactive monitoring simulator
    - Educational concept deep-dives
    """)

# ---------------------------------------------------------------------------
# Quick Start
# ---------------------------------------------------------------------------

st.header("🚀 Quick Start")
st.code("""
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start the API
uvicorn src.api.main:app --reload --port 8015

# Launch dashboard
streamlit run streamlit_app/app.py
""", language="bash")

# ---------------------------------------------------------------------------
# Project Metrics
# ---------------------------------------------------------------------------

st.header("📈 Project Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Core Modules", "4", help="pipeline, monitoring, validation, api")
m2.metric("API Endpoints", "9", help="health, status, run, history, models, register, get, experiments, best")
m3.metric("Test Cases", "50+", help="Across pipeline, monitoring, validation, and API")
m4.metric("Dashboard Pages", "5", help="Overview, Pipeline, Experiments, Monitoring, Learn")
