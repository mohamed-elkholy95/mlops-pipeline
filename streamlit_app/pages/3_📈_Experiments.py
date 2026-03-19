import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from src.pipeline import ExperimentTracker
st.title("📈 Experiments")
tracker = ExperimentTracker()
tracker.log("lr_tuning", {"lr": 0.01}, {"accuracy": 0.85, "f1": 0.83})
tracker.log("lr_tuning", {"lr": 0.001}, {"accuracy": 0.92, "f1": 0.90})
df = tracker.compare()
st.dataframe(df)
best = tracker.get_best("accuracy")
if best: st.success(f"Best: {best['params']} → accuracy={best['metrics']['accuracy']}")
