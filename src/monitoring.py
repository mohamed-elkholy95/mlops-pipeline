"""WORK IN PROGRESS — Core structure and imports."""

"""Model monitoring."""
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data distribution drift."""

    def __init__(self, reference_data: np.ndarray = None, threshold: float = 0.1) -> None:
        self._ref_mean = np.mean(reference_data, axis=0) if reference_data is not None else None
        self._ref_std = np.std(reference_data, axis=0) if reference_data is not None else None
