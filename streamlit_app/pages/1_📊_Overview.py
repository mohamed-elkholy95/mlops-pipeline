import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
st.title("⚙️ MLOps Pipeline")
st.markdown("ML pipeline orchestration with experiment tracking and model registry.")
col1, col2 = st.columns(2)
with col1: st.subheader("Components"); st.markdown("- Pipeline orchestration\n- Model registry\n- Experiment tracking")
with col2: st.subheader("Monitoring"); st.markdown("- Data drift detection\n- Performance degradation alerts")
