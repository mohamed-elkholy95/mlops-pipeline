<div align="center">

# ⚙️ MLOps Pipeline

**ML pipeline orchestration** with experiment tracking, model registry, and data drift monitoring

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-16%20passed-success?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

An **MLOps infrastructure toolkit** providing pipeline step orchestration, model versioning/registry, experiment tracking with comparison, data drift detection, and performance degradation monitoring. Designed to wrap around any ML workflow.

## Features

- 🔗 **Pipeline Orchestration** — Composable step-based pipeline with error handling and timing
- 📦 **Model Registry** — Versioned model storage with metadata and registration timestamps
- 🧪 **Experiment Tracker** — Log, compare, and find best experiments (max/min any metric)
- 🚨 **Data Drift Detection** — Statistical drift scoring with configurable thresholds
- 📉 **Performance Monitor** — Track metrics over time with degradation alerts
- 📊 **3-Page Dashboard** — Overview, pipeline runner, and experiment comparison
- 🚀 **REST API** — Pipeline status and model listing endpoints

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
