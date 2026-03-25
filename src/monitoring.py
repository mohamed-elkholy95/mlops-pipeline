"""
Model monitoring — drift detection and performance tracking.

Once a model is deployed, the world doesn't stand still.  User behaviour
changes, data distributions shift, and upstream pipelines break.  This
module implements two complementary monitoring strategies:

1. **Data Drift Detection** — catches when *input* distributions change.
   Uses a simple but effective approach: compare the mean of incoming
   features against a stored reference distribution.  More sophisticated
   methods (KL divergence, KS tests, PSI) exist, but the Z-score
   approach here is intuitive and works well for tabular data.

2. **Performance Monitoring** — catches when *output* quality degrades.
   Tracks any named metric over time and detects downward trends.
   In production, this often hooks into alerting systems (PagerDuty,
   Slack webhooks) so the ML team finds out *before* users complain.

Together, these answer the two critical monitoring questions:
    - "Has the data changed?"  →  DataDriftDetector
    - "Is the model still good?"  →  PerformanceMonitor
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Drift Detection
# ---------------------------------------------------------------------------

class DataDriftDetector:
    """Detect distribution drift between reference and incoming data.

    How it works:
        1. During *calibration*, compute the mean and std of each feature
           from a reference dataset (typically the training data).
        2. During *inference*, compute the mean of each feature in the
           incoming batch and measure how far it has shifted in units of
           the reference standard deviation — essentially a Z-score.
        3. Average the per-feature drift scores into a single scalar.
        4. If the scalar exceeds ``threshold``, flag drift.

    Why this matters:
        A model trained on data from distribution P will produce unreliable
        predictions when fed data from a different distribution Q.  Drift
        detection is the early warning system — it tells you the model's
        assumptions may no longer hold.

    Limitations:
        - Only detects mean shift; won't catch variance or shape changes.
        - Assumes features are independent (no correlation drift).
        - For categorical features, prefer Population Stability Index (PSI).
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        threshold: float = 0.1,
    ) -> None:
        # Store reference statistics — not the raw data — to keep memory
        # usage constant regardless of training set size.
        self._ref_mean: Optional[np.ndarray] = (
            np.mean(reference_data, axis=0) if reference_data is not None else None
        )
        self._ref_std: Optional[np.ndarray] = (
            np.std(reference_data, axis=0) if reference_data is not None else None
        )
        self.threshold = threshold

    def detect(self, data: np.ndarray) -> Dict[str, Any]:
        """Score incoming data against the reference distribution.

        If no reference was provided at init time, the first call to
        ``detect()`` calibrates the detector — a common pattern for
        online learning systems where the reference emerges at runtime.

        Returns:
            dict with ``drift_detected`` (bool), ``score`` (float), and
            ``threshold`` (float) for logging/alerting.
        """
        if self._ref_mean is None:
            # First-call calibration — treat this batch as the reference
            self._ref_mean = np.mean(data, axis=0)
            self._ref_std = np.std(data, axis=0)
            return {"drift_detected": False, "score": 0.0}

        current_mean = np.mean(data, axis=0)

        # Per-feature drift: how many reference-stds has the mean shifted?
        # The 1e-8 epsilon prevents division by zero for constant features.
        per_feature_drift = np.abs(current_mean - self._ref_mean) / (self._ref_std + 1e-8)

        # Aggregate into a single drift score (mean across features)
        score = float(np.mean(per_feature_drift))

        return {
            "drift_detected": score > self.threshold,
            "score": round(score, 4),
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Performance Monitoring
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """Track model performance metrics over time and detect degradation.

    In production ML systems, models degrade silently.  Unlike traditional
    software where errors crash the process, a degraded model just returns
    *worse* predictions.  Nobody gets an exception; the business just
    slowly loses money.

    This monitor maintains a time-series of metric values and checks for
    downward trends using a simple relative-change heuristic.  Production
    systems often use more advanced techniques:
        - CUSUM (cumulative sum control charts)
        - Exponentially weighted moving averages (EWMA)
        - Statistical process control (SPC) with control limits

    Usage::

        mon = PerformanceMonitor()
        mon.log("accuracy", 0.92)
        mon.log("accuracy", 0.91)
        mon.log("accuracy", 0.88)
        if mon.is_degrading("accuracy"):
            alert_on_call_engineer()
    """

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []

    def log(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a metric observation.

        If no timestamp is provided, the current wall-clock time is used.
        In production, you'd typically pass the prediction-batch timestamp
        so the time-series reflects *when the predictions were made*, not
        when the log call happened.
        """
        self._history.append({
            "metric": metric_name,
            "value": value,
            "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_trend(self, metric_name: str, n_last: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent ``n_last`` entries for a given metric."""
        entries = [h for h in self._history if h["metric"] == metric_name]
        return entries[-n_last:]

    def is_degrading(
        self,
        metric_name: str,
        window: int = 5,
        threshold: float = -0.05,
    ) -> bool:
        """Check whether a metric is trending downward.

        Computes the relative change between the oldest and newest values
        in the window.  A relative drop below ``threshold`` (default −5 %)
        triggers a degradation signal.

        Args:
            metric_name: Which metric to inspect.
            window: How many recent observations to consider.
            threshold: Relative change that counts as degradation.
                       Negative because we're looking for *drops*.

        Why relative change?
            Absolute thresholds don't generalise across metrics.  A 0.01
            drop in accuracy (99 % → 98 %) might be critical, while a
            0.01 drop in a loss term (2.50 → 2.49) is meaningless.
            Relative change normalises by the baseline value.
        """
        trend = self.get_trend(metric_name, window + 1)
        if len(trend) < 2:
            return False

        values = [h["value"] for h in trend]
        baseline = abs(values[0]) if abs(values[0]) > 1e-8 else 1e-8
        relative_change = (values[-1] - values[0]) / baseline
        return relative_change < threshold
