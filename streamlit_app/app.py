"""
MLOps Pipeline — Streamlit Dashboard Entry Point.

Configures the multi-page app with dark theme styling and navigation
across five pages: Overview, Pipeline Runner, Experiments, Monitoring,
and an educational Learn page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="MLOps Pipeline",
    layout="wide",
    page_icon="⚙️",
)

# Dark theme styling
st.markdown(
    '<style>'
    '[data-testid="stSidebar"]{background-color:#262730}'
    '.stApp{background-color:#0e1117;color:#fff}'
    'h1,h2,h3{color:#1f77b4}'
    '</style>',
    unsafe_allow_html=True,
)

pg = st.navigation([
    st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_🚀_Pipeline.py", title="Pipeline", icon="🚀"),
    st.Page("pages/3_📈_Experiments.py", title="Experiments", icon="📈"),
    st.Page("pages/4_🔍_Monitoring.py", title="Monitoring", icon="🔍"),
    st.Page("pages/5_📖_Learn.py", title="Learn", icon="📖"),
])
pg.run()
