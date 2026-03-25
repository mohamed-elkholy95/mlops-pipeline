"""Tests for pipeline orchestration, model registry, and experiment tracking.

Covers the core pipeline execution flow, error handling, metrics persistence,
model registration lifecycle, and experiment comparison/ranking.
"""

import json
import pytest
from pathlib import Path
from src.pipeline import PipelineStep, MLPipeline, ModelRegistry, ExperimentTracker


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------

class TestPipelineStep:
    def test_success_with_return_value(self):
        """A step that returns a value should capture it in output."""
        step = PipelineStep("double", lambda x: x * 2)
        result = step.run(x=5)
        assert result == 10
        assert step.status == "success"
        assert step.output == 10
        assert step.duration >= 0

    def test_failure_captures_error(self):
        """A step that raises should set status='failed' and record the error."""
        step = PipelineStep("boom", lambda: 1 / 0)
        step.run()
        assert step.status == "failed"
        assert step.error is not None
        assert "division" in step.error.lower()

    def test_no_func_returns_kwargs(self):
        """A step with no function should pass kwargs through as output."""
        step = PipelineStep("passthrough")
        result = step.run(a=1, b=2)
        assert result == {"a": 1, "b": 2}
        assert step.status == "success"

    def test_to_dict_structure(self):
        """to_dict should contain name, status, duration, and error."""
        step = PipelineStep("test", lambda: "ok")
        step.run()
        d = step.to_dict()
        assert set(d.keys()) == {"name", "status", "duration", "error"}
        assert d["name"] == "test"
        assert d["status"] == "success"
        assert d["error"] is None

    def test_initial_state(self):
        """A fresh step should be pending with zero duration."""
        step = PipelineStep("fresh")
        assert step.status == "pending"
        assert step.duration == 0.0
        assert step.output is None


# ---------------------------------------------------------------------------
# MLPipeline
# ---------------------------------------------------------------------------

class TestMLPipeline:
    def test_linear_pipeline(self):
        """Steps should execute in order with context passing."""
        p = MLPipeline("linear")
        p.add_step("init", lambda **kw: {"x": 10})
        p.add_step("double", lambda x=0, **kw: {"result": x * 2})
        metrics = p.run()
        assert metrics["status"] == "success"
        assert len(metrics["steps"]) == 2

    def test_failed_step_marks_pipeline_failed(self):
        """If any step fails, the overall pipeline status should be 'failed'."""
        p = MLPipeline("fail-test")
        p.add_step("ok", lambda **kw: {"a": 1})
        p.add_step("bad", lambda **kw: (_ for _ in ()).throw(ValueError("oops")))
        p.add_step("after", lambda **kw: {"b": 2})
        metrics = p.run()
        assert metrics["status"] == "failed"

    def test_chaining(self):
        """add_step should return self for fluent chaining."""
        p = MLPipeline("chain")
        result = p.add_step("a").add_step("b").add_step("c")
        assert result is p
        assert len(p._steps) == 3

    def test_empty_pipeline_succeeds(self):
        """A pipeline with no steps should succeed (vacuously true)."""
        p = MLPipeline("empty")
        metrics = p.run()
        assert metrics["status"] == "success"
        assert metrics["total_duration"] == 0.0
        assert metrics["steps"] == []

    def test_metrics_include_timestamp(self):
        """Pipeline metrics should include a timestamp."""
        p = MLPipeline("ts")
        p.add_step("noop", lambda **kw: None)
        metrics = p.run()
        assert "timestamp" in metrics

    def test_save_metrics(self, tmp_path):
        """save_metrics should write valid JSON to disk."""
        p = MLPipeline("save-test")
        p.add_step("step", lambda **kw: {"val": 42})
        p.run()
        out = tmp_path / "run_metrics.json"
        p.save_metrics(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["pipeline"] == "save-test"
        assert data["status"] == "success"

    def test_save_metrics_returns_path(self, tmp_path):
        """save_metrics should return the path it wrote to."""
        p = MLPipeline("ret")
        p.add_step("s", lambda **kw: None)
        p.run()
        result = p.save_metrics(tmp_path / "m.json")
        assert result == tmp_path / "m.json"

    def test_get_metrics_before_run(self):
        """get_metrics before run should return empty dict."""
        p = MLPipeline("pre-run")
        assert p.get_metrics() == {}

    def test_context_accumulation(self):
        """Each step's dict output should merge into the shared context."""
        p = MLPipeline("ctx")
        p.add_step("a", lambda **kw: {"x": 1})
        p.add_step("b", lambda x=0, **kw: {"y": x + 10})
        p.add_step("c", lambda x=0, y=0, **kw: {"z": x + y})
        p.run()
        # If context accumulated correctly, step c should have seen x=1, y=11


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_retrieve(self):
        """Registered models should be retrievable by name/version."""
        reg = ModelRegistry()
        reg.register("clf", "1.0", {"accuracy": 0.9})
        model = reg.get_model("clf", "1.0")
        assert model is not None
        assert model["metrics"]["accuracy"] == 0.9

    def test_register_with_metadata(self):
        """Metadata should be stored alongside metrics."""
        reg = ModelRegistry()
        entry = reg.register("clf", "2.0", {"f1": 0.88}, {"git_sha": "abc123"})
        assert entry["metadata"]["git_sha"] == "abc123"

    def test_list_models(self):
        """list_models should return all registered entries."""
        reg = ModelRegistry()
        reg.register("a", "1.0", {})
        reg.register("b", "1.0", {})
        assert len(reg.list_models()) == 2

    def test_get_nonexistent_returns_none(self):
        """Looking up an unregistered model should return None."""
        reg = ModelRegistry()
        assert reg.get_model("ghost", "0.0") is None

    def test_overwrite_version(self):
        """Re-registering the same name/version should overwrite."""
        reg = ModelRegistry()
        reg.register("m", "1.0", {"accuracy": 0.8})
        reg.register("m", "1.0", {"accuracy": 0.95})
        model = reg.get_model("m", "1.0")
        assert model["metrics"]["accuracy"] == 0.95
        # list should still show 1 entry (overwritten, not duplicated)
        assert len(reg.list_models()) == 1

    def test_registered_at_timestamp(self):
        """Registered entries should include a timestamp."""
        reg = ModelRegistry()
        entry = reg.register("t", "1.0", {})
        assert "registered_at" in entry

    def test_persists_to_disk(self, tmp_path):
        """Registry should write JSON files to the registry directory."""
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register("disk", "1.0", {"accuracy": 0.9})
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["name"] == "disk"


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    def test_log_returns_experiment(self):
        """log() should return the experiment dict with an ID."""
        tracker = ExperimentTracker()
        exp = tracker.log("test", {"lr": 0.01}, {"accuracy": 0.9})
        assert exp["name"] == "test"
        assert exp["id"] == 0
        assert "timestamp" in exp

    def test_sequential_ids(self):
        """Experiments should get sequential IDs starting from 0."""
        tracker = ExperimentTracker()
        e0 = tracker.log("a", {}, {})
        e1 = tracker.log("b", {}, {})
        assert e0["id"] == 0
        assert e1["id"] == 1

    def test_compare_dataframe(self):
        """compare() should return a DataFrame with params and metrics."""
        tracker = ExperimentTracker()
        tracker.log("exp1", {"lr": 0.01}, {"accuracy": 0.85})
        tracker.log("exp2", {"lr": 0.001}, {"accuracy": 0.92})
        df = tracker.compare()
        assert len(df) == 2
        assert "param_lr" in df.columns
        assert "accuracy" in df.columns

    def test_compare_empty(self):
        """compare() with no experiments should return empty DataFrame."""
        tracker = ExperimentTracker()
        df = tracker.compare()
        assert len(df) == 0

    def test_get_best_max(self):
        """get_best with mode='max' should return highest-scoring run."""
        tracker = ExperimentTracker()
        tracker.log("low", {}, {"accuracy": 0.7})
        tracker.log("high", {}, {"accuracy": 0.99})
        tracker.log("mid", {}, {"accuracy": 0.85})
        best = tracker.get_best("accuracy", mode="max")
        assert best["name"] == "high"

    def test_get_best_min(self):
        """get_best with mode='min' should return lowest-scoring run."""
        tracker = ExperimentTracker()
        tracker.log("high_loss", {}, {"loss": 2.5})
        tracker.log("low_loss", {}, {"loss": 0.1})
        best = tracker.get_best("loss", mode="min")
        assert best["name"] == "low_loss"

    def test_get_best_empty(self):
        """get_best with no experiments should return None."""
        tracker = ExperimentTracker()
        assert tracker.get_best("accuracy") is None

    def test_persists_to_disk(self, tmp_path):
        """Experiments should be written to disk as JSON files."""
        tracker = ExperimentTracker(storage_dir=tmp_path)
        tracker.log("disk-test", {"lr": 0.01}, {"accuracy": 0.9})
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["name"] == "disk-test"
