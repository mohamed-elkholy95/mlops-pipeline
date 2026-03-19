import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from src.pipeline import MLPipeline
st.title("🚀 Run Pipeline")
p = MLPipeline("demo")
p.add_step("ingest", lambda **kw: {"n_rows": 1000})
p.add_step("validate", lambda n_rows=0, **kw: {"valid": True})
p.add_step("split", lambda **kw: {"train": 800, "test": 200})
p.add_step("train", lambda **kw: {"model": "trained"})
if st.button("Run Pipeline", type="primary"):
    with st.spinner("Running..."):
        metrics = p.run()
    st.json(metrics)
