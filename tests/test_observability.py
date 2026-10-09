"""Verify that a real local MLflow run and root trace can be read back."""

import mlflow

from observability import ask_with_tracking, configure_tracking, read_operations
from sample_data import SAMPLE_GRAPH
from tests.test_workflow import FakeResponder, FakeStore
from workflow import build_workflow


def test_question_persists_run_metrics_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "integration-test")
    experiment_id = configure_tracking()
    session_id = "a" * 32
    result = ask_with_tracking(build_workflow(FakeStore(), FakeResponder()),
        "Who acted in Inception?", "my-graph",
        [entity.model_dump() for entity in SAMPLE_GRAPH.entities], session_id)
    runs = read_operations(experiment_id, session_id)
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.fact_count"] == 1
    assert runs.iloc[0]["metrics.error"] == 0
    assert runs.iloc[0]["tags.result"] == "answered"
    assert mlflow.get_trace(result["trace_id"], flush=True) is not None
    assert read_operations(experiment_id, "b" * 32).empty
