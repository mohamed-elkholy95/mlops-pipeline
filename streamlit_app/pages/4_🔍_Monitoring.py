"""
Monitoring Dashboard — Data Drift & Performance Tracking.

This page provides interactive visualizations for the two core monitoring
capabilities: data drift detection and performance degradation analysis.
Users can experiment with different distributions, thresholds, and
degradation patterns to build intuition for production monitoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.monitoring import DataDriftDetector, PerformanceMonitor

st.title("🔍 Monitoring Dashboard")
st.markdown(
    "Explore data drift detection and performance monitoring — "
    "two essential capabilities for keeping ML models reliable in production."
)

# ---------------------------------------------------------------------------
# Section 1: Data Drift Detection
# ---------------------------------------------------------------------------

st.header("📊 Data Drift Detection")
st.markdown("""
**What is data drift?** When the statistical distribution of incoming data
differs from what the model was trained on, predictions become unreliable.
This demo lets you simulate drift and see how the detector responds.
""")

col1, col2 = st.columns(2)
with col1:
    n_features = st.slider("Number of features", 1, 10, 3, key="drift_features")
    n_samples = st.slider("Samples per batch", 50, 500, 100, key="drift_samples")
    threshold = st.slider("Drift threshold", 0.01, 1.0, 0.1, step=0.01, key="drift_threshold")

with col2:
    shift_magnitude = st.slider(
        "Mean shift magnitude", 0.0, 5.0, 0.0, step=0.1,
        help="How much to shift the incoming data's mean relative to the reference.",
    )
    noise_scale = st.slider(
        "Noise scale", 0.5, 3.0, 1.0, step=0.1,
        help="Standard deviation multiplier for the incoming data.",
    )

# Generate reference and incoming data
np.random.seed(42)
reference = np.random.randn(n_samples, n_features)
incoming = np.random.randn(n_samples, n_features) * noise_scale + shift_magnitude

# Detect drift
detector = DataDriftDetector(reference, threshold=threshold)
result = detector.detect(incoming)

# Display result
if result["drift_detected"]:
    st.error(f"🚨 **Drift Detected!** Score: {result['score']:.4f} (threshold: {threshold})")
else:
    st.success(f"✅ **No Drift.** Score: {result['score']:.4f} (threshold: {threshold})")

# Distribution comparison chart
fig = go.Figure()
for i in range(min(n_features, 3)):
    fig.add_trace(go.Histogram(
        x=reference[:, i], name=f"Reference (feat {i})",
        opacity=0.5, nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=incoming[:, i], name=f"Incoming (feat {i})",
        opacity=0.5, nbinsx=30,
    ))
fig.update_layout(
    title="Feature Distribution Comparison (first 3 features)",
    barmode="overlay",
    xaxis_title="Value",
    yaxis_title="Count",
    template="plotly_dark",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# Per-feature drift breakdown
st.subheader("Per-Feature Drift Scores")
ref_mean = np.mean(reference, axis=0)
ref_std = np.std(reference, axis=0)
inc_mean = np.mean(incoming, axis=0)
per_feature = np.abs(inc_mean - ref_mean) / (ref_std + 1e-8)

drift_df = pd.DataFrame({
    "Feature": [f"Feature {i}" for i in range(n_features)],
    "Drift Score": per_feature,
    "Reference Mean": ref_mean,
    "Incoming Mean": inc_mean,
})
fig2 = px.bar(
    drift_df, x="Feature", y="Drift Score",
    color="Drift Score",
    color_continuous_scale="RdYlGn_r",
    title="Per-Feature Drift Magnitude",
    template="plotly_dark",
)
fig2.add_hline(y=threshold, line_dash="dash", line_color="red",
               annotation_text=f"Threshold ({threshold})")
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: Performance Monitoring
# ---------------------------------------------------------------------------

st.header("📉 Performance Monitoring")
st.markdown("""
**Why monitor performance?** Models don't crash — they degrade silently.
A model returning 85% accuracy instead of 92% still returns HTTP 200.
Without monitoring, nobody notices until revenue drops.
""")

col3, col4 = st.columns(2)
with col3:
    pattern = st.selectbox("Degradation pattern", [
        "Stable", "Gradual decline", "Sudden drop", "Seasonal fluctuation",
    ])
    n_points = st.slider("Number of observations", 10, 100, 30, key="perf_points")

with col4:
    degrade_window = st.slider("Detection window", 3, 20, 5, key="perf_window")
    degrade_threshold = st.slider(
        "Degradation threshold (%)", -20.0, 0.0, -5.0, step=0.5,
        help="Relative drop that triggers a degradation alert.",
    )

# Generate synthetic performance data
np.random.seed(123)
base = 0.92
noise = np.random.randn(n_points) * 0.005

if pattern == "Stable":
    values = base + noise
elif pattern == "Gradual decline":
    trend = np.linspace(0, -0.08, n_points)
    values = base + trend + noise
elif pattern == "Sudden drop":
    values = base + noise
    drop_idx = n_points // 2
    values[drop_idx:] -= 0.07
elif pattern == "Seasonal fluctuation":
    season = 0.03 * np.sin(np.linspace(0, 4 * np.pi, n_points))
    values = base + season + noise

# Track in PerformanceMonitor
monitor = PerformanceMonitor()
for v in values:
    monitor.log("accuracy", float(v))

is_degrading = monitor.is_degrading(
    "accuracy",
    window=degrade_window,
    threshold=degrade_threshold / 100,
)

if is_degrading:
    st.error("🚨 **Performance Degradation Detected!**")
else:
    st.success("✅ **Performance Stable.**")

# Time series chart
perf_df = pd.DataFrame({
    "Observation": range(n_points),
    "Accuracy": values,
})
fig3 = px.line(
    perf_df, x="Observation", y="Accuracy",
    title=f"Model Accuracy Over Time ({pattern})",
    template="plotly_dark",
    markers=True,
)
fig3.add_hline(
    y=base * (1 + degrade_threshold / 100),
    line_dash="dash", line_color="red",
    annotation_text=f"Alert threshold ({degrade_threshold}%)",
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")
scol1, scol2, scol3, scol4 = st.columns(4)
scol1.metric("Mean", f"{np.mean(values):.4f}")
scol2.metric("Std Dev", f"{np.std(values):.4f}")
scol3.metric("Min", f"{np.min(values):.4f}")
scol4.metric("Max", f"{np.max(values):.4f}")
