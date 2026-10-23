"""Acceptance inputs must be immutable and comparisons must preserve trajectories."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from harbor_support.catalog import TASKS
from harbor_support.export import Settings
from harbor_support.files import content_hash, file_hash, inventory, write_json
from harbor_support.reference import compare_verdicts, load_references
from harbor_support.replay_agent import ReferenceReplayAgent
from harbor_support.verifier import EvaluationError


@pytest.fixture
def native():
    protocol = Settings(num_envs=2).protocol(TASKS[0])
    verdict = {"preset": protocol["preset"], "scene": protocol["scene"], "seed": 0,
               "num_envs": 2, "score": 0.5, "success": 0.5,
               "per_traj": [{"env": 0, "score": 0.0, "success": False},
                            {"env": 1, "score": 1.0, "success": True}]}
    return protocol, verdict


def test_same_aggregate_cannot_hide_trajectory_disagreement(native):
    protocol, expected = native
    actual = copy.deepcopy(expected)
    actual["per_traj"][0].update(score=1.0, success=True)
    actual["per_traj"][1].update(score=0.0, success=False)
    with pytest.raises(ValueError, match="trajectory 0"):
        compare_verdicts(actual, expected, protocol, 0)


@pytest.mark.parametrize("fault", ["seed", "trajectory_order", "timeout", "error"])
def test_invalid_baselines_are_not_parity_evidence(native, fault):
    protocol, verdict = native
    bad = copy.deepcopy(verdict)
    if fault == "seed": bad["seed"] = 1
    if fault == "trajectory_order": bad["per_traj"][0]["env"] = 1
    if fault == "timeout": bad["timed_out"] = True
    if fault == "error": bad["error"] = "fixture solution error"
    with pytest.raises((ValueError, EvaluationError)):
        compare_verdicts(bad, verdict, protocol, 0)


def test_reference_tolerance_is_explicit_and_bounded(native):
    protocol, expected = native
    actual = copy.deepcopy(expected)
    actual["per_traj"][0]["score"] = 0.001
    actual["score"] = 0.5005
    with pytest.raises(ValueError, match="differs"):
        compare_verdicts(actual, expected, protocol, 0)
    result = compare_verdicts(actual, expected, protocol, 0, score_tolerance=0.001)
    assert result["max_trajectory_score_delta"] == 0.001
    for tolerance in (-1, float("nan"), float("inf"), True, 0.1):
        with pytest.raises(ValueError, match="tolerance"):
            compare_verdicts(actual, expected, protocol, 0, score_tolerance=tolerance)


@pytest.fixture
def manifest(tmp_path, native):
    _, verdict = native
    solution = tmp_path / "reference with spaces"
    solution.mkdir()
    (solution / "solve.py").write_text("def solve(env):\n    pass\n")
    (solution / ".data").mkdir()
    (solution / ".data/params").write_text("42\n")
    baseline = tmp_path / "native.json"
    write_json(baseline, verdict)
    record = {"path": solution.name, "sha256": content_hash(inventory(solution)),
              "native": {"0": {"path": baseline.name, "sha256": file_hash(baseline)}}}
    path = tmp_path / "manifest.json"
    write_json(path, {"schema_version": 1, "tasks": {
        TASKS[0].id: {"checkpoints": [record], "final": record}}})
    return path


@pytest.mark.parametrize("fault", ["solution", "native", "symlink", "empty", "unknown_task"])
def test_changed_or_invalid_reference_inputs_rejected(manifest, fault):
    if fault == "solution":
        (manifest.parent / "reference with spaces/.data/params").write_text("changed")
    elif fault == "native":
        (manifest.parent / "native.json").write_text("{}")
    elif fault == "symlink":
        (manifest.parent / "reference with spaces/link").symlink_to(manifest)
    else:
        value = json.loads(manifest.read_text())
        if fault == "empty": value["tasks"] = {}
        else: value["tasks"]["unknown"] = value["tasks"].pop(TASKS[0].id)
        write_json(manifest, value)
    with pytest.raises(ValueError):
        load_references(manifest)


def test_replay_agent_uses_real_harbor_factory(manifest, tmp_path):
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig

    config = AgentConfig(import_path="harbor_support.replay_agent:ReferenceReplayAgent",
                         kwargs={"manifest": str(manifest)})
    AgentFactory.run_preflight(config)
    agent_class = AgentFactory.get_agent_class_from_config(config)
    assert agent_class is ReferenceReplayAgent
    agent = agent_class(logs_dir=tmp_path / "logs", **config.kwargs)
    record = agent.references["tasks"][TASKS[0].id]["final"]
    assert Path(record["path"]).is_absolute()
    assert Path(record["native"]["0"]["path"]).is_absolute()
