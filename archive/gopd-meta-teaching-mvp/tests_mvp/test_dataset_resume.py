"""Real Parquet identity checks around simulated controller execution.

All data and launcher records live under pytest temporary directories. The
backend is explicitly fake: these tests do not load a model or start GPU work.
"""

import copy
import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow", reason="dataset resume tests require real Parquet preparation")
pq = pytest.importorskip("pyarrow.parquet", reason="dataset resume tests require real Parquet preparation")

from meta_teaching.common import read_json
from meta_teaching.config import DEFAULTS, validate_config
from meta_teaching.data import generate_demo_jsonl, prepare, sha256_file, validate_manifest
from meta_teaching.runner import ExperimentRunner
from test_runner import FakeBackend


@pytest.fixture
def prepared_experiment(tmp_path):
    sources = generate_demo_jsonl(tmp_path / "raw", train_count=16, meta_val_count=4, test_count=4)
    destination = tmp_path / "prepared"
    prepare(sources["train"], sources["meta_val"], sources["test"], destination, demo_only=True)
    manifest = destination / "manifest.json"
    validate_manifest(manifest)
    config = copy.deepcopy(DEFAULTS)
    config["models"] = {"student": str(tmp_path / "student"), "teacher": str(tmp_path / "teacher")}
    config["data"]["manifest"] = str(manifest)
    config["experiment"].update(
        output_dir=str(tmp_path / "run"), rounds=1, repeats=1,
        validation_reserve_seconds=0.1, max_total_gpu_hours=100,
    )
    validate_config(config)
    return config, sources, destination


def replace_prepared_data(sources, destination):
    """Overwrite the same paths with a different, internally valid dataset."""
    path = sources["meta_val"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["question"] = "Calculate 912 + 87. Put the final answer in \\boxed{}."
    rows[0]["answer"] = "999"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    prepare(sources["train"], sources["meta_val"], sources["test"], destination, demo_only=True)
    validate_manifest(destination / "manifest.json")


def initial_eval_spec(runner):
    return runner._spec("initial_eval", "eval", "initial", None, 0, runner.config["experiment"]["seed"])


def test_resume_rejects_same_path_new_valid_manifest(prepared_experiment):
    config, sources, destination = prepared_experiment
    backend = FakeBackend(config)
    runner = ExperimentRunner(config, backend=backend)
    runner._phase(initial_eval_spec(runner))
    previous_hash = sha256_file(destination / "meta_val.parquet")

    replace_prepared_data(sources, destination)

    assert sha256_file(destination / "meta_val.parquet") != previous_hash
    with pytest.raises(ValueError, match="Dataset/manifest contents changed"):
        ExperimentRunner(config, backend=backend, resume=True)
    assert len(backend.executions) == 1


def test_resume_rejects_new_parquet_with_unchanged_manifest(prepared_experiment):
    config, _, destination = prepared_experiment
    backend = FakeBackend(config)
    ExperimentRunner(config, backend=backend)
    manifest_bytes = (destination / "manifest.json").read_bytes()
    parquet = destination / "test.parquet"
    previous_hash = sha256_file(parquet)
    rows = pq.read_table(parquet).to_pylist()
    rows[0]["reward_model"]["ground_truth"] = "999"
    pq.write_table(pa.Table.from_pylist(rows), parquet)

    assert (destination / "manifest.json").read_bytes() == manifest_bytes
    assert sha256_file(parquet) != previous_hash
    with pytest.raises(ValueError, match="Dataset/manifest contents changed"):
        ExperimentRunner(config, backend=backend, resume=True)
    assert not backend.executions


def test_cached_phase_rejects_data_changed_after_controller_construction(prepared_experiment):
    config, sources, destination = prepared_experiment
    backend = FakeBackend(config)
    runner = ExperimentRunner(config, backend=backend)
    runner._phase(initial_eval_spec(runner))
    replace_prepared_data(sources, destination)

    with pytest.raises(ValueError, match="Dataset/manifest changed during"):
        runner._phase(initial_eval_spec(runner))
    assert len(backend.executions) == 1


def test_data_changed_during_phase_marks_record_failed_and_retains_cost(prepared_experiment):
    config, sources, destination = prepared_experiment

    class DataChangingBackend(FakeBackend):
        def execute(self, spec):
            record = super().execute(spec)
            replace_prepared_data(sources, destination)
            return record

    backend = DataChangingBackend(config)
    runner = ExperimentRunner(config, backend=backend)
    with pytest.raises(ValueError, match="Dataset/manifest changed during"):
        runner.run()

    record = read_json(Path(config["experiment"]["output_dir"]) / "phases/initial_eval/launcher_record.json")
    assert record["status"] == "failed"
    assert "Dataset/manifest changed during" in record["error"]
    assert runner.state["status"] == "failed"
    assert runner.state["phase_results"]["initial_eval"]["status"] == "failed"
    assert runner.state["initial_score"] is None
    assert runner.policy.total_updates == 0
    events = [event for event in runner.state["cost_events"].values() if event["phase_id"] == "initial_eval"]
    assert len(events) == 1
    assert events[0]["allocated_gpu_seconds"] == record["allocated_gpu_seconds"] > 0


def test_unchanged_dataset_completed_resume_reuses_results(prepared_experiment):
    config, _, _ = prepared_experiment
    backend = FakeBackend(config)
    runner = ExperimentRunner(config, backend=backend)
    first = runner.run()
    executions = len(backend.executions)
    snapshot = copy.deepcopy(runner.state["dataset_snapshot"])

    resumed = ExperimentRunner(config, backend=backend, resume=True)
    second = resumed.run()

    assert first["status"] == second["status"] == "completed"
    assert len(backend.executions) == executions
    assert resumed.state["dataset_snapshot"] == snapshot
    assert second["final_test"] == first["final_test"]
    assert resumed.policy.state_dict() == runner.policy.state_dict()


def test_completed_resume_checks_again_before_returning_cached_result(prepared_experiment):
    config, sources, destination = prepared_experiment
    backend = FakeBackend(config)
    runner = ExperimentRunner(config, backend=backend)
    runner.run()
    executions = len(backend.executions)
    resumed = ExperimentRunner(config, backend=backend, resume=True)
    replace_prepared_data(sources, destination)

    with pytest.raises(ValueError, match="Dataset/manifest changed during"):
        resumed.run()
    assert len(backend.executions) == executions
