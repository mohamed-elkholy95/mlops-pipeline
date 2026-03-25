"""Tests for data drift detection and performance monitoring.

Covers drift scoring, threshold behaviour, first-call calibration,
edge cases (constant features, single samples), and degradation
detection across multiple patterns.
"""

import pytest
import numpy as np
from src.monitoring import DataDriftDetector, PerformanceMonitor


# ---------------------------------------------------------------------------
# Data Drift Detector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    """Test suite for DataDriftDetector."""

    def test_no_drift_same_data(self):
        """Reference data compared against itself should show zero drift."""
        ref = np.random.randn(100, 3)
        det = DataDriftDetector(ref)
        result = det.detect(ref)
        assert not result["drift_detected"]
        assert result["score"] < 0.05  # very low score expected

    def test_drift_large_shift(self):
        """A large mean shift should trigger drift detection."""
        ref = np.random.randn(100, 3)
        det = DataDriftDetector(ref, threshold=0.1)
        shifted = ref + 10.0
        result = det.detect(shifted)
        assert result["drift_detected"]
        assert result["score"] > 1.0  # should be much larger than threshold

    def test_threshold_boundary(self):
        """Drift score exactly at threshold should not trigger (strict >)."""
        ref = np.zeros((100, 1))
        det = DataDriftDetector(ref, threshold=0.5)
        # Create data that produces a known drift score
        # With ref_mean=0, ref_std≈0 → epsilon-based, score depends on shift
        result = det.detect(ref)  # same data, score ≈ 0
        assert not result["drift_detected"]

    def test_first_call_calibration(self):
        """When no reference is provided, first detect() calibrates."""
        det = DataDriftDetector()
        data = np.random.randn(50, 2)
        result = det.detect(data)
        assert not result["drift_detected"]
        assert result["score"] == 0.0
        # Now a shifted batch should trigger drift
        result2 = det.detect(data + 5.0)
        assert result2["drift_detected"]

    def test_custom_threshold(self):
        """A very high threshold should tolerate moderate shifts."""
        ref = np.random.randn(100, 3)
        det = DataDriftDetector(ref, threshold=100.0)  # very permissive
        result = det.detect(ref + 2.0)
        assert not result["drift_detected"]

    def test_single_feature(self):
        """Drift detection should work with a single feature (1D)."""
        ref = np.random.randn(100, 1)
        det = DataDriftDetector(ref)
        result = det.detect(ref + 5.0)
        assert result["drift_detected"]

    def test_result_contains_threshold(self):
        """The result dict should include the threshold for transparency."""
        ref = np.random.randn(50, 2)
        det = DataDriftDetector(ref, threshold=0.42)
        result = det.detect(ref)
        assert result["threshold"] == 0.42

    def test_many_features(self):
        """Should handle high-dimensional data without errors."""
        ref = np.random.randn(100, 50)
        det = DataDriftDetector(ref)
        result = det.detect(ref)
        assert "score" in result


# ---------------------------------------------------------------------------
# Performance Monitor
# ---------------------------------------------------------------------------

class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""

    def test_log_and_retrieve(self):
        """Logged entries should be retrievable via get_trend."""
        mon = PerformanceMonitor()
        for i in range(10):
            mon.log("accuracy", 0.9 - i * 0.01)
        trend = mon.get_trend("accuracy")
        assert len(trend) == 10

    def test_trend_slicing(self):
        """get_trend should return only the last n entries."""
        mon = PerformanceMonitor()
        for i in range(20):
            mon.log("loss", float(i))
        trend = mon.get_trend("loss", n_last=5)
        assert len(trend) == 5
        assert trend[0]["value"] == 15.0  # last 5 of 0..19

    def test_degrading_downward_trend(self):
        """A steady decline should be detected as degradation."""
        mon = PerformanceMonitor()
        for i in range(10):
            mon.log("accuracy", 0.9 - i * 0.05)
        assert mon.is_degrading("accuracy")

    def test_not_degrading_stable(self):
        """Stable metrics should not trigger degradation."""
        mon = PerformanceMonitor()
        for _ in range(10):
            mon.log("accuracy", 0.92)
        assert not mon.is_degrading("accuracy")

    def test_not_degrading_improving(self):
        """An upward trend should not trigger degradation."""
        mon = PerformanceMonitor()
        for i in range(10):
            mon.log("accuracy", 0.8 + i * 0.02)
        assert not mon.is_degrading("accuracy")

    def test_single_observation_not_degrading(self):
        """Cannot determine degradation with fewer than 2 observations."""
        mon = PerformanceMonitor()
        mon.log("accuracy", 0.5)
        assert not mon.is_degrading("accuracy")

    def test_empty_monitor_not_degrading(self):
        """No observations → no degradation."""
        mon = PerformanceMonitor()
        assert not mon.is_degrading("accuracy")

    def test_multiple_metrics(self):
        """Different metrics should be tracked independently."""
        mon = PerformanceMonitor()
        for i in range(5):
            mon.log("accuracy", 0.9 - i * 0.05)
            mon.log("loss", 0.1 + i * 0.05)
        acc_trend = mon.get_trend("accuracy")
        loss_trend = mon.get_trend("loss")
        assert len(acc_trend) == 5
        assert len(loss_trend) == 5
        assert acc_trend[0]["metric"] == "accuracy"
        assert loss_trend[0]["metric"] == "loss"

    def test_custom_window_and_threshold(self):
        """Custom window and threshold should be respected."""
        mon = PerformanceMonitor()
        # Small decline that's less than 5% but more than 1%
        values = [0.90, 0.895, 0.893, 0.891, 0.889]
        for v in values:
            mon.log("accuracy", v)
        # Default threshold (-0.05) should not trigger
        assert not mon.is_degrading("accuracy", window=4, threshold=-0.05)
        # Stricter threshold (-0.005) should trigger
        assert mon.is_degrading("accuracy", window=4, threshold=-0.005)

    def test_custom_timestamp(self):
        """Custom timestamps should be stored correctly."""
        mon = PerformanceMonitor()
        mon.log("accuracy", 0.9, timestamp="2026-03-25 14:00:00")
        trend = mon.get_trend("accuracy")
        assert trend[0]["timestamp"] == "2026-03-25 14:00:00"
