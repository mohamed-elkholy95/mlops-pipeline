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
        self.threshold = threshold

    def detect(self, data: np.ndarray) -> Dict[str, Any]:
        if self._ref_mean is None:
            self._ref_mean = np.mean(data, axis=0)
            self._ref_std = np.std(data, axis=0)
            return {"drift_detected": False, "score": 0.0}
        current_mean = np.mean(data, axis=0)
        score = float(np.mean(np.abs(current_mean - self._ref_mean) / (self._ref_std + 1e-8)))
        return {"drift_detected": score > self.threshold, "score": round(score, 4), "threshold": self.threshold}


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self) -> None:
        self._history: List[Dict] = []

    def log(self, metric_name: str, value: float, timestamp: str = None) -> None:
        self._history.append({"metric": metric_name, "value": value,
            "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S")})

    def get_trend(self, metric_name: str, n_last: int = 10) -> List[Dict]:
        entries = [h for h in self._history if h["metric"] == metric_name]
        return entries[-n_last:]

    def is_degrading(self, metric_name: str, window: int = 5, threshold: float = -0.05) -> bool:
        trend = self.get_trend(metric_name, window + 1)
        if len(trend) < 2: return False
        values = [h["value"] for h in trend]
        delta = (values[-1] - values[0]) / max(abs(values[0]), 1e-8)
        return delta < threshold
