"""
Configuration — centralised settings for the MLOps pipeline.

Why a config module?
    Scattering magic numbers and paths across source files makes a
    project fragile and hard to reconfigure.  A single config module
    provides one place to look up (and override) defaults.

    In production, consider environment variables (``os.getenv``) or
    config files (YAML/TOML) with tools like Hydra, Dynaconf, or
    Pydantic Settings.  For this project, module-level constants are
    sufficient and keep the dependency footprint small.

Directory structure:
    BASE_DIR/
    ├── artifacts/       → trained model files and registry JSONs
    │   └── models/      → versioned model entries
    ├── metrics/         → pipeline run metrics and experiment logs
    │   └── experiments/ → per-experiment JSON files
    └── logs/            → application log files
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Configure root logger with a structured format that includes timestamps,
# severity, and the originating module — essential for debugging pipeline
# runs where multiple steps log concurrently.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# BASE_DIR resolves to the project root (one level above src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Artifact directories — created eagerly so downstream code doesn't need
# to check existence before writing.
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METRICS_DIR = BASE_DIR / "metrics"
LOG_DIR = BASE_DIR / "logs"

for _dir in [ARTIFACTS_DIR, METRICS_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Global random seed for NumPy, scikit-learn, and any other library that
# respects seeding.  Set this before any randomised operation to ensure
# reproducible results across runs.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_HOST = "0.0.0.0"
API_PORT = 8015
