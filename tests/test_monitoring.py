import pytest
import numpy as np
from src.monitoring import DataDriftDetector, PerformanceMonitor

class TestDriftDetector:
    def test_no_drift(self):
        ref = np.random.randn(100, 3)
        det = DataDriftDetector(ref)
        result = det.detect(ref)
        assert not result["drift_detected"]

    def test_drift(self):
        ref = np.random.randn(100, 3)
        det = DataDriftDetector(ref)
        shifted = ref + 10.0
        result = det.detect(shifted)
        assert result["drift_detected"]

class TestPerformanceMonitor:
    def test_log_and_trend(self):
        mon = PerformanceMonitor()
        for i in range(10): mon.log("accuracy", 0.9 - i * 0.01)
        trend = mon.get_trend("accuracy")
        assert len(trend) == 10

    def test_degrading(self):
        mon = PerformanceMonitor()
        for i in range(10): mon.log("accuracy", 0.9 - i * 0.05)
        assert mon.is_degrading("accuracy")
