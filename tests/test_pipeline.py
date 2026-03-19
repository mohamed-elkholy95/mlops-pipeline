import pytest
from src.pipeline import PipelineStep, MLPipeline, ModelRegistry, ExperimentTracker
from pathlib import Path

class TestPipelineStep:
    def test_success(self):
        step = PipelineStep("test", lambda x: x * 2)
        assert step.run(x=5) == 10
        assert step.status == "success"

    def test_failure(self):
        step = PipelineStep("fail", lambda: 1/0)
        step.run()
        assert step.status == "failed"
        assert step.error is not None

    def test_to_dict(self):
        step = PipelineStep("test", lambda: None)
        step.run()
        assert "status" in step.to_dict()

class TestMLPipeline:
    def test_run(self):
        p = MLPipeline("test")
        p.add_step("double", lambda x=0, **kw: {"result": kw.get("x", 1) * 2})
        p.add_step("add", lambda result=0, **kw: {"final": result + 10})
        metrics = p.run(x=5)
        assert metrics["status"] == "success"

    def test_save_metrics(self, tmp_path):
        p = MLPipeline("test")
        p.add_step("step", lambda: None)
        p.run()
        p.save_metrics(Path(tmp_path) / "metrics.json")
        assert (tmp_path / "metrics.json").exists()

class TestModelRegistry:
    def test_register(self):
        reg = ModelRegistry()
        entry = reg.register("test_model", "1.0", {"accuracy": 0.9})
        assert entry["version"] == "1.0"

    def test_list(self):
        reg = ModelRegistry()
        reg.register("a", "1.0", {}); reg.register("b", "1.0", {})
        assert len(reg.list_models()) == 2

class TestExperimentTracker:
    def test_log(self):
        tracker = ExperimentTracker()
        exp = tracker.log("test", {"lr": 0.01}, {"accuracy": 0.9})
        assert exp["name"] == "test"

    def test_compare(self):
        tracker = ExperimentTracker()
        tracker.log("exp1", {"lr": 0.01}, {"accuracy": 0.85})
        tracker.log("exp2", {"lr": 0.001}, {"accuracy": 0.92})
        df = tracker.compare()
        assert len(df) == 2
