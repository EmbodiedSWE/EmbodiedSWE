"""Harbor adapter contract and failure tests; no simulator/GPU is needed."""

import asyncio
import copy
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from envbuild.condition import load
from envbuild.manifest import tree_hash
from harbor_support.catalog import TASKS, select_tasks, validate_registry
from harbor_support.export import DEFAULT_CONDITION, Settings, export_dataset, world_hash
from harbor_support.files import content_hash, inventory, write_json
from harbor_support.submit import snapshot
from harbor_support.verifier import EvaluationError, validate_verdict, verify
from harbor_support.validate import validate_dataset


@pytest.fixture
def config():
    return {**Settings(rollout_timeout_sec=2, termination_grace_sec=0.2).protocol(TASKS[0]),
            "worker_user": None, "cameras": False, "seeds": [0, 1]}


@pytest.fixture
def worlds(tmp_path):
    root = tmp_path / "worlds"
    for task in TASKS:
        exp = root / task.id
        stage = exp / "stages/01_task"
        package = stage / "bench/robobench"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"# fixture for {task.preset}\n")
        (stage / "describe.md").write_text(f"Goal for {task.preset}\n")
        write_json(exp / "resolved.json", {
            "boot_seed": 0, "set_states": True, "repo_rev": "fixture",
            "stages": [{"dir": "01_task", "preset": task.preset, "boot_checked": True,
                        "control_mode_frozen": True, "tree_sha256": tree_hash(stage / "bench")}],
        })
    return root


def export(worlds, output, tasks=TASKS, **kwargs):
    return export_dataset(worlds=worlds, output=output, tasks=tasks,
                          images={"physx": "test/runtime@sha256:" + "a" * 64,
                                  "newton": "test/newton@sha256:" + "b" * 64},
                          condition=load(DEFAULT_CONDITION), settings=kwargs.pop("settings", Settings()),
                          **kwargs)


def test_catalog_resolves_to_28_real_presets_and_graders():
    assert len(TASKS) == len({t.id for t in TASKS}) == 28
    assert sum(t.backend == "newton" for t in TASKS) == 4
    assert "assembly.pc_motherboard_gpu_ram" not in {t.id for t in TASKS}
    validate_registry()
    with pytest.raises(ValueError):
        select_tasks(["unknown"])
    with pytest.raises(ValueError):
        select_tasks([TASKS[0].id] * 2)


def test_all_tasks_load_in_actual_harbor_and_export_repeatably(worlds, tmp_path):
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    one, two = export(worlds, tmp_path / "one"), export(worlds, tmp_path / "two")
    assert inventory(one) == inventory(two)
    assert validate_dataset(one) == 28
    assert json.loads((one / "dataset.json").read_text())["task_count"] == 28
    for spec in TASKS:
        task_dir = one / spec.id
        task = Task(task_dir)
        assert Task.is_valid_dir(task_dir)
        assert task.config.agent.user == "agent"
        assert task.config.environment.gpus == 1
        assert task.config.verifier.environment_mode.value == "separate"
        definition = resolve_verifier_environment_definition(task.config, task.paths)
        assert definition.bundled_tests
        assert definition.directory == task_dir / "tests"
        assert spec.preset in task.instruction
        assert task.config.artifacts[0].source == "/workspace/solution"
        assert task.config.artifacts[1].source == "/submissions"
        env_files = inventory(task_dir / "environment")
        assert not any("graders" in name or "test.sh" in name for name in env_files)
        assert "VLM_JUDGE_API_KEY" not in (task_dir / "environment/Dockerfile").read_text()
        assert bool(task.config.verifier.env) == (spec.backend == "newton")
        assert (task_dir / "tests/graders/__init__.py").is_file()
        receipt = json.loads((task_dir / "export.json").read_text())
        actual = inventory(task_dir)
        del actual["export.json"]
        assert receipt["files"] == actual
        assert receipt["sha256"] == content_hash(actual)


@pytest.mark.parametrize("fault", ["unbooted", "changed", "wrong_preset", "missing", "path", "leak", "condition", "symlink"])
def test_export_rejects_invalid_world_without_partial_dataset(worlds, tmp_path, fault):
    exp = worlds / TASKS[0].id
    receipt_path = exp / "resolved.json"
    receipt = json.loads(receipt_path.read_text())
    stage = exp / "stages/01_task"
    if fault == "unbooted": receipt["stages"][0]["boot_checked"] = False
    if fault == "changed": (stage / "bench/robobench/__init__.py").write_text("changed")
    if fault == "wrong_preset": receipt["stages"][0]["preset"] = TASKS[1].preset
    if fault == "missing": (stage / "describe.md").unlink()
    if fault == "path": receipt["stages"][0]["dir"] = "../../outside"
    if fault == "condition": receipt["set_states"] = False
    if fault == "leak":
        path = stage / "bench/robobench/suites/assembly/grader"
        path.mkdir(parents=True)
        (path / "secret.py").write_text("# privileged")
        receipt["stages"][0]["tree_sha256"] = tree_hash(stage / "bench")
    if fault == "symlink": (stage / "bench/link").symlink_to(stage / "describe.md")
    write_json(receipt_path, receipt)
    with pytest.raises((ValueError, OSError)):
        export(worlds, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset*"))


def test_missing_task_cannot_silently_shrink_benchmark(worlds, tmp_path):
    shutil.rmtree(worlds / TASKS[-1].id)
    with pytest.raises(FileNotFoundError):
        export(worlds, tmp_path / "dataset")


def test_private_downloaded_assets_are_readable_to_runtime_worker(worlds, tmp_path):
    exp = worlds / TASKS[0].id
    bench = exp / "stages/01_task/bench"
    asset = bench / "robobench/assets/private/model.usd"
    asset.parent.mkdir(parents=True, mode=0o700)
    asset.write_bytes(b"downloaded asset")
    asset.chmod(0o600)
    receipt = json.loads((exp / "resolved.json").read_text())
    receipt["stages"][0]["tree_sha256"] = tree_hash(bench)
    write_json(exp / "resolved.json", receipt)
    previous = os.umask(0o077)
    try:
        target = export(worlds, tmp_path / "dataset", tasks=TASKS[:1])
    finally:
        os.umask(previous)
    assert target.stat().st_mode & 0o005 == 0o005
    assert (target / "dataset.json").stat().st_mode & 0o004
    for context in ("environment", "tests"):
        copied = target / TASKS[0].id / context / "bench/robobench/assets/private/model.usd"
        assert copied.stat().st_mode & 0o004
        assert copied.parent.stat().st_mode & 0o005 == 0o005
    assert validate_dataset(target) == 1


def test_modified_export_fails_validation(worlds, tmp_path):
    target = export(worlds, tmp_path / "dataset", tasks=TASKS[:1])
    (target / TASKS[0].id / "tests/grade.py").write_text("# modified grader")
    with pytest.raises(ValueError, match="content mismatch"):
        validate_dataset(target)


def test_never_overwrite_export(worlds, tmp_path):
    target = export(worlds, tmp_path / "dataset", tasks=TASKS[:1])
    before = inventory(target)
    with pytest.raises(FileExistsError):
        export(worlds, target, tasks=TASKS[:1])
    assert inventory(target) == before


@pytest.mark.parametrize("kwargs", [
    {"agent_timeout_sec": float("nan")}, {"rollout_timeout_sec": 0}, {"num_envs": 0},
    {"max_submissions": -1}, {"seeds": ()}, {"seeds": (0, 0)}, {"seeds": (-1,)},
])
def test_reject_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()


def test_world_hash_streaming_is_backward_compatible(tmp_path):
    (tmp_path / "large.bin").write_bytes(os.urandom(3 * 1024 * 1024))
    assert world_hash(tmp_path) == tree_hash(tmp_path)


def _parallel_submit(paths):
    solution, submissions = map(Path, paths)
    return snapshot(solution, submissions, note="concurrent", limit=64).name


def test_concurrent_submissions_are_atomic_and_unique(tmp_path):
    solution, submissions = tmp_path / "solution", tmp_path / "submissions"
    solution.mkdir()
    (solution / "solve.py").write_text("def solve(env): pass\n")
    (solution / "weights.bin").write_bytes(os.urandom(1024 * 1024))
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn")) as pool:
        names = list(pool.map(_parallel_submit, [(str(solution), str(submissions))] * 32))
    assert len(set(names)) == 32
    assert set(names) == {f"{i:06d}" for i in range(1, 33)}
    for name in names:
        receipt = json.loads((submissions / name / "submission.json").read_text())
        assert receipt["files"] == inventory(submissions / name / "solution")
    assert not list(submissions.glob(".pending-*"))
    with pytest.raises(ValueError, match="limit"):
        snapshot(solution, submissions, note="over limit", limit=32)


@pytest.fixture
def driver(tmp_path):
    # Real subprocess fixture: exercises args, timeouts, exits, serialization, and
    # seed aggregation at the same boundary as the Isaac driver.
    path = tmp_path / "driver.py"
    path.write_text('''import argparse,json,time,sys
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('preset','scene','solution','graders','out'): p.add_argument('--'+k,required=True)
for k in ('seed','num-envs'): p.add_argument('--'+k,type=int,required=True)
p.add_argument('--cameras',action='store_true'); a=p.parse_args()
data=json.loads(Path(a.solution).read_text())
if data.get('mutate_self'): Path(a.solution).write_text('{"score":0}')
if data.get('mutate_path'): Path(data['mutate_path']).write_text('{"score":0}')
if data.get('phase'): Path(a.out,'phase.json').write_text(json.dumps({'phase':data['phase']}))
if data.get('crash'): sys.exit(7)
if data.get('sleep'): time.sleep(30)
if data.get('no_verdict'): sys.exit(0)
score=data.get(str(a.seed),data.get('score',0))
per=[{'score':score,'success':score==1} for _ in range(a.num_envs)]
v={'preset':a.preset,'scene':a.scene,'seed':a.seed,'num_envs':a.num_envs,
   'score':score,'success':float(score==1),'per_traj':per}
Path(a.out,'verdict.json').write_text(json.dumps(v))
''')
    return path


def evaluate(tmp_path, config, driver):
    return verify(config, submissions=tmp_path / "submissions", final=tmp_path / "solution",
                  out=tmp_path / "out", driver=driver, graders=tmp_path / "graders")


def solution(tmp_path, data):
    path = tmp_path / "solution"
    path.mkdir(exist_ok=True)
    (path / "solve.py").write_text(json.dumps(data))
    return path


def test_best_checkpoint_averages_seeds_before_selection(tmp_path, config, driver):
    sol = solution(tmp_path, {"0": 1, "1": 0})
    snapshot(sol, tmp_path / "submissions", note="first", limit=64)
    solution(tmp_path, {"score": 0.6})
    snapshot(sol, tmp_path / "submissions", note="best", limit=64)
    solution(tmp_path, {"0": 0, "1": 1})
    report = evaluate(tmp_path, config, driver)
    assert report["selected_submission"] == "000002"
    assert report["reward"] == {"score": 0.6, "success": 0.0}
    assert len(list((tmp_path / "out/rollouts").glob("*/seed_*/verdict.json"))) == 6


def test_duplicate_final_reuses_checkpoint(tmp_path, config, driver):
    sol = solution(tmp_path, {"score": 1})
    snapshot(sol, tmp_path / "submissions", note="done", limit=64)
    report = evaluate(tmp_path, config, driver)
    assert report["reward"] == {"score": 1, "success": 1}
    assert report["submissions"][-1]["duplicate_of"] == "000001"
    assert not (tmp_path / "out/rollouts/final").exists()


def test_equal_decimal_means_select_earliest_despite_float_rounding(tmp_path, config, driver):
    config["seeds"] = [0, 1, 2]
    sol = solution(tmp_path, {"0": 0.0, "1": 0.1, "2": 0.5})
    snapshot(sol, tmp_path / "submissions", note="earliest equal mean", limit=4)
    solution(tmp_path, {"0": 0.0, "1": 0.2, "2": 0.4})
    report = evaluate(tmp_path, config, driver)
    assert report["selected_submission"] == "000001"
    assert report["reward"]["score"] == 0.2


def test_each_seed_gets_fresh_writable_solution_copy(tmp_path, config, driver):
    original = solution(tmp_path, {"score": 1, "mutate_self": True})
    report = evaluate(tmp_path, config, driver)
    assert report["reward"]["score"] == 1
    assert json.loads((original / "solve.py").read_text())["score"] == 1
    frozen = tmp_path / "out/inputs/final/solve.py"
    assert frozen.stat().st_mode & 0o222 == 0
    assert json.loads(frozen.read_text())["score"] == 1


def test_all_candidates_frozen_before_first_replay(tmp_path, config, driver):
    sol = solution(tmp_path, {"score": 0.1, "mutate_path": str(tmp_path / "solution/solve.py")})
    snapshot(sol, tmp_path / "submissions", note="writes original final", limit=4)
    solution(tmp_path, {"score": 0.8})
    report = evaluate(tmp_path, config, driver)
    assert report["selected_submission"] == "final"
    assert report["reward"]["score"] == 0.8
    assert json.loads((sol / "solve.py").read_text())["score"] == 0


@pytest.mark.parametrize("difference", ["executable", "empty_directory"])
def test_submission_identity_includes_layout_and_execute_bit(tmp_path, config, driver, difference):
    sol = solution(tmp_path, {"score": 0.4})
    snapshot(sol, tmp_path / "submissions", note="baseline", limit=4)
    if difference == "executable":
        (sol / "solve.py").chmod(0o755)
    else:
        (sol / "scratch").mkdir()
    report = evaluate(tmp_path, config, driver)
    assert [r["status"] for r in report["submissions"]] == ["graded", "graded"]


def test_missing_solution_is_zero(tmp_path, config, driver):
    assert evaluate(tmp_path, config, driver)["reward"] == {"score": 0, "success": 0}


def test_tampered_checkpoint_cannot_earn_reward(tmp_path, config, driver):
    sol = solution(tmp_path, {"score": 1})
    checkpoint = snapshot(sol, tmp_path / "submissions", note="first", limit=64)
    (checkpoint / "solution/solve.py").write_text('{"score":0.9}')
    shutil.rmtree(sol)
    report = evaluate(tmp_path, config, driver)
    assert report["reward"]["score"] == 0
    assert report["submissions"][0]["status"] == "invalid_submission"


def test_malformed_receipt_does_not_discard_valid_final(tmp_path, config, driver):
    sol = solution(tmp_path, {"score": 1})
    checkpoint = snapshot(sol, tmp_path / "submissions", note="bad receipt", limit=4)
    (checkpoint / "submission.json").write_text("[]")
    solution(tmp_path, {"score": 0.6})
    report = evaluate(tmp_path, config, driver)
    assert report["submissions"][0]["status"] == "invalid_submission"
    assert report["reward"]["score"] == 0.6


def test_storage_failure_never_becomes_model_zero(tmp_path, config, driver, monkeypatch):
    import errno
    from harbor_support import verifier
    solution(tmp_path, {"score": 1})

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(verifier.shutil, "copytree", full_disk)
    with pytest.raises(OSError, match="space"):
        evaluate(tmp_path, config, driver)
    assert not (tmp_path / "out/reward.json").exists()


@pytest.mark.parametrize("data", [{"crash": True}, {"no_verdict": True}, {"sleep": True}, {"score": float("nan")}, {"score": 1.1}, {"score": True}])
def test_infrastructure_failure_never_emits_reward(tmp_path, config, driver, data):
    solution(tmp_path, data)
    with pytest.raises(EvaluationError):
        evaluate(tmp_path, config, driver)
    assert not (tmp_path / "out/reward.json").exists()


def test_stale_reward_removed_on_rerun(tmp_path, config, driver):
    solution(tmp_path, {"score": 1})
    evaluate(tmp_path, config, driver)
    with pytest.raises(EvaluationError):
        evaluate(tmp_path, config, driver)
    assert not (tmp_path / "out/reward.json").exists()


def test_hung_final_solution_does_not_discard_good_checkpoint(tmp_path, config, driver):
    sol = solution(tmp_path, {"score": 0.8})
    snapshot(sol, tmp_path / "submissions", note="working", limit=64)
    solution(tmp_path, {"sleep": True, "phase": "solving"})
    config["seeds"] = [0]
    report = evaluate(tmp_path, config, driver)
    assert report["reward"]["score"] == 0.8
    assert report["submissions"][-1]["rollouts"][0]["timed_out"]


@pytest.mark.parametrize("exit_behavior", ["crash", "no_verdict"])
def test_exiting_solution_is_zero_not_infrastructure_failure(tmp_path, config, driver, exit_behavior):
    solution(tmp_path, {exit_behavior: True, "phase": "solving"})
    report = evaluate(tmp_path, config, driver)
    assert report["reward"] == {"score": 0, "success": 0}
    assert "exited" in report["submissions"][0]["rollouts"][0]["solution_error"]


def test_snapshot_rejects_symlinks_and_changes_during_copy(tmp_path, monkeypatch):
    sol = solution(tmp_path, {"score": 0.5})
    (sol / "link").symlink_to(tmp_path / "external")
    with pytest.raises(ValueError, match="Symlink"):
        snapshot(sol, tmp_path / "submissions", note="bad", limit=64)
    (sol / "link").unlink()
    original_copy = shutil.copytree

    def changing_copy(src, dest):
        result = original_copy(src, dest)
        (sol / "solve.py").write_text("changed while copying")
        return result

    monkeypatch.setattr(shutil, "copytree", changing_copy)
    with pytest.raises(ValueError, match="changed"):
        snapshot(sol, tmp_path / "submissions", note="racing", limit=64)
    assert not [p for p in (tmp_path / "submissions").iterdir() if p.is_dir()]


def test_mutable_images_rejected_by_default(worlds, tmp_path):
    with pytest.raises(ValueError, match="sha256"):
        export_dataset(worlds=worlds, output=tmp_path / "dataset", tasks=TASKS[:1],
                       images={"physx": "runtime:latest"}, condition=load(DEFAULT_CONDITION),
                       settings=Settings())


def test_newton_batch_size_rejected_before_export(worlds, tmp_path):
    with pytest.raises(ValueError, match="num_envs=1"):
        export(worlds, tmp_path / "dataset", settings=Settings(num_envs=2))


def test_deformable_missing_credential_is_infrastructure_failure(tmp_path, config, driver, monkeypatch):
    monkeypatch.delenv("VLM_JUDGE_API_KEY", raising=False)
    config["cameras"] = True
    solution(tmp_path, {"score": 0.5})
    with pytest.raises(EvaluationError, match="VLM_JUDGE_API_KEY"):
        evaluate(tmp_path, config, driver)


def test_empty_delivery_does_not_require_native_grader_services(tmp_path, config, driver, monkeypatch):
    monkeypatch.delenv("VLM_JUDGE_API_KEY", raising=False)
    config["cameras"] = True
    report = evaluate(tmp_path, config, driver)
    assert report["reward"] == {"score": 0.0, "success": 0.0}
    assert report["submissions"][0]["status"] == "missing_solution"
    assert not (tmp_path / "out/rollouts").exists()


def test_vlm_failure_and_inconsistent_verdict_rejected(config):
    good = {"preset": config["preset"], "scene": config["scene"], "seed": 0,
            "num_envs": 1, "score": 0.5, "success": 0,
            "per_traj": [{"score": 0.5, "success": False}]}
    assert validate_verdict(good, config, 0)["score"] == 0.5
    bad = copy.deepcopy(good)
    bad["per_traj"][0]["final"] = {"judge": {"error": "API unavailable"}}
    with pytest.raises(EvaluationError, match="judge"):
        validate_verdict(bad, config, 0)
    bad = copy.deepcopy(good)
    bad["score"] = 1
    with pytest.raises(EvaluationError, match="disagrees"):
        validate_verdict(bad, config, 0)


def judge_verdict(config, decision=False):
    answer = {"plausible": decision, "reason": "fixture decision"}
    gate = {"decision": decision, "value": float(decision), "parsed": answer,
            "http_status": 200, "attempts": 1, "model": "fixture-model",
            "logid": "fixture-log", "prompt": "fixture prompt", "image": "traj_000/final.png",
            "answer_text": json.dumps(answer), "raw_response": {"output": ["fixture"]}}
    return {"preset": config["preset"], "scene": config["scene"], "seed": 0,
            "num_envs": 1, "score": float(decision), "success": float(decision),
            "per_traj": [{"env": 0, "score": float(decision), "success": decision,
                          "final": {"plausible": gate}}]}


@pytest.mark.parametrize("decision", [True, False])
def test_deformable_complete_judge_audit_accepted(config, decision):
    config["cameras"] = True
    assert validate_verdict(judge_verdict(config, decision), config, 0)["success"] == float(decision)


@pytest.mark.parametrize("fault", ["missing", "empty", "partial", "inconsistent", "non_boolean",
                                   "wrong_value", "contradictory_success", "no_http", "no_response",
                                   "no_image", "empty_reason"])
def test_incomplete_or_contradictory_judge_audits_rejected(config, fault):
    config["cameras"] = True
    verdict = judge_verdict(config)
    trajectory = verdict["per_traj"][0]
    gate = trajectory["final"]["plausible"]
    if fault == "missing": trajectory.pop("final")
    if fault == "empty": trajectory["final"] = {}
    if fault == "partial": trajectory["final"]["plausible"] = {"value": 0}
    if fault == "inconsistent": gate["parsed"]["plausible"] = True
    if fault == "non_boolean": gate["decision"] = 0
    if fault == "wrong_value": gate["value"] = 1.0
    if fault == "contradictory_success": trajectory["success"] = True
    if fault == "no_http": gate.pop("http_status")
    if fault == "no_response": gate["raw_response"] = {}
    if fault == "no_image": gate.pop("image")
    if fault == "empty_reason": gate["parsed"]["reason"] = " "
    with pytest.raises(EvaluationError, match="judge"):
        validate_verdict(verdict, config, 0)


def test_compose_overrides_harbor_legacy_gpu_request(worlds, tmp_path, monkeypatch):
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    if shutil.which("docker") is None:
        pytest.skip("Docker Compose CLI required (daemon is not required)")
    if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode:
        pytest.skip("Docker Compose CLI is unavailable")
    task = export(worlds, tmp_path / "dataset", tasks=TASKS[:1]) / TASKS[0].id
    config = Task(task).config
    monkeypatch.setenv("ROBOBENCH_GPU", "2")

    async def check(context, env_config):
        environment = DockerEnvironment(context, "robobench", "robobench-contract",
                                        TrialPaths(tmp_path / "trial"), env_config)
        try:
            environment._resources_compose_path = await environment._write_resources_compose_file()
            result = await environment._run_docker_compose_command(["config", "--format", "json"])
            main = json.loads(result.stdout)["services"]["main"]
            assert main["devices"][0]["source"] == "nvidia.com/gpu=2"
            assert not main.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices")
        finally:
            environment._cleanup_resources_compose_file()

    asyncio.run(check(task / "environment", config.environment))
    asyncio.run(check(task / "tests", config.verifier.environment))
