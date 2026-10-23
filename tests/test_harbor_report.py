"""Publication metrics use actual Harbor models and reject incomplete/error jobs."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from harbor_support.report import summarize
from harbor_support.files import write_json
from test_harbor_support import worlds, export, TASKS


@pytest.fixture
def completed_job(worlds, tmp_path):
    from dirhash import dirhash
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.task.id import LocalTaskId
    from harbor.models.task.task import Task
    from harbor.models.trial.config import TrialConfig, TaskConfig
    from harbor.models.trial.result import AgentInfo, ModelInfo, TrialResult
    from harbor.models.verifier.result import VerifierResult

    dataset = export(worlds, tmp_path / "dataset", tasks=TASKS[:2])
    job = tmp_path / "job"
    now = datetime.now(timezone.utc)
    for index, task in enumerate(TASKS[:2]):
        trial_path = job / f"trial_{index}"
        verifier = trial_path / "verifier"
        verifier.mkdir(parents=True)
        reward = {"score": [0.2, 0.8][index], "success": 0.0}
        protocol = json.loads((dataset / task.id / "tests/protocol.json").read_text())
        write_json(verifier / "results.json", {"protocol": protocol, "reward": reward,
                                              "selected_submission": "final"})
        write_json(verifier / "reward.json", reward)
        result = TrialResult(task_name=Task(dataset / task.id).name, trial_name=trial_path.name,
                             trial_uri=str(trial_path), task_id=LocalTaskId(path=dataset / task.id),
                             task_checksum=dirhash(dataset / task.id, "sha256"),
                             config=TrialConfig(task=TaskConfig(path=dataset / task.id)),
                             agent_info=AgentInfo(name="test", version="1", model_info=ModelInfo(name="fixture")),
                             verifier_result=VerifierResult(rewards=reward),
                             verifier_environment_mode="separate", started_at=now, finished_at=now)
        (trial_path / "result.json").write_text(result.model_dump_json())
    result = JobResult(id=uuid4(), started_at=now, finished_at=now, n_total_trials=2, stats=JobStats())
    (job / "result.json").write_text(result.model_dump_json())
    return dataset, job


def test_strict_summary_keeps_score_and_success_separate(completed_job):
    dataset, job = completed_job
    report = summarize(dataset, [job])
    assert report["task_count"] == 2
    assert report["mean"] == {"score": 0.5, "success": 0.0}


@pytest.mark.parametrize("fault", ["missing_trial", "unfinished_job", "failed_trial", "missing_reward",
                                   "wrong_export", "different_agent", "reward_mismatch", "verifier_error",
                                   "wrong_namespace", "bare_task_name"])
def test_invalid_jobs_cannot_become_benchmark_zeroes(completed_job, fault):
    dataset, job = completed_job
    path = job / "trial_0/result.json"
    trial = json.loads(path.read_text())
    if fault == "missing_trial":
        path.unlink()
    elif fault == "unfinished_job":
        value = json.loads((job / "result.json").read_text())
        value["finished_at"] = None
        write_json(job / "result.json", value)
    else:
        if fault == "failed_trial":
            trial["exception_info"] = {"exception_type": "VerifierTimeoutError", "exception_message": "timeout",
                                       "exception_traceback": "test", "occurred_at": trial["finished_at"]}
        if fault == "missing_reward": trial["verifier_result"] = None
        if fault == "wrong_export": trial["task_checksum"] = "wrong"
        if fault == "wrong_namespace": trial["task_name"] = trial["task_name"].replace("robobench/", "other/")
        if fault == "bare_task_name": trial["task_name"] = trial["task_name"].split("/")[-1]
        if fault == "different_agent": trial["agent_info"]["version"] = "different"
        if fault == "reward_mismatch": trial["verifier_result"]["rewards"]["score"] = 0.9
        if fault == "verifier_error": write_json(job / "trial_0/verifier/error.json", {"error": "judge offline"})
        write_json(path, trial)
    with pytest.raises(ValueError):
        summarize(dataset, [job])


def test_missing_repetitions_and_duplicate_jobs_rejected(completed_job):
    dataset, job = completed_job
    with pytest.raises(ValueError, match="expected 2"):
        summarize(dataset, [job], attempts=2)
    with pytest.raises(ValueError, match="Duplicate trial"):
        summarize(dataset, [job, job])


def test_agent_budget_exhaustion_with_valid_verification_remains_eligible(completed_job):
    dataset, job = completed_job
    path = job / "trial_0/result.json"
    trial = json.loads(path.read_text())
    trial["exception_info"] = {"exception_type": "AgentTimeoutError", "exception_message": "agent budget",
                               "exception_traceback": "test", "occurred_at": trial["finished_at"]}
    write_json(path, trial)
    assert summarize(dataset, [job])["mean"]["score"] == 0.5


@pytest.fixture
def completed_boot_job(completed_job):
    dataset, job = completed_job
    for path in job.glob("*/result.json"):
        trial = json.loads(path.read_text())
        reward = {"score": 0.0, "success": 0.0}
        trial["verifier_result"]["rewards"] = reward
        trial["agent_result"] = {"metadata": {"gpu_build_reset": "passed", "delivery": "empty"}}
        trial["agent_info"] = {"name": "robobench-container-boot", "version": "1"}
        trial["config"]["agent"]["import_path"] = "harbor_support.boot_agent:ContainerBootAgent"
        write_json(path, trial)
        verifier = path.parent / "verifier"
        report = json.loads((verifier / "results.json").read_text())
        report["reward"] = reward
        report["submissions"] = [{"submission": "final", "status": "missing_solution", "rollouts": []}]
        write_json(verifier / "results.json", report)
        write_json(verifier / "reward.json", reward)
        (path.parent / "agent").mkdir()
        (path.parent / "agent/gpu-boot.log").write_text("HARBOR_GPU_BOOT_OK " + report["protocol"]["preset"] + "\n")
    return dataset, job


def test_completed_container_boot_job_accepted(completed_boot_job):
    from scripts.harbor_container_smoke import check_job
    assert check_job(*completed_boot_job)["status"] == "passed"


@pytest.mark.parametrize("fault", [None, "missing_partition", "duplicate_partition", "second_boot_missing"])
def test_partitioned_container_boots_require_complete_evidence(completed_boot_job, fault):
    from scripts.harbor_container_smoke import check_jobs
    dataset, first = completed_boot_job
    second = first.parent / "second-job"
    second.mkdir()
    (first / "trial_1").rename(second / "trial_1")
    record = json.loads((first / "result.json").read_text())
    record["n_total_trials"] = 1
    write_json(first / "result.json", record)
    record["id"] = str(uuid4())
    write_json(second / "result.json", record)
    jobs = [first, second]
    if fault == "missing_partition": jobs.pop()
    if fault == "duplicate_partition": jobs.append(second)
    if fault == "second_boot_missing":
        (second / "trial_1/agent/gpu-boot.log").write_text("GPU_BOOT_START only\n")
    if fault:
        with pytest.raises(ValueError):
            check_jobs(dataset, jobs)
    else:
        result = check_jobs(dataset, jobs)
        assert result["status"] == "passed"
        assert result["summary"]["task_count"] == 2


@pytest.mark.parametrize("fault", ["agent_timeout", "metadata", "wrong_preset", "graded_delivery", "wrong_agent"])
def test_empty_reward_is_not_sufficient_for_container_boot_acceptance(completed_boot_job, fault):
    from scripts.harbor_container_smoke import check_job
    dataset, job = completed_boot_job
    path = job / "trial_0/result.json"
    trial = json.loads(path.read_text())
    if fault == "agent_timeout":
        trial["exception_info"] = {"exception_type": "AgentTimeoutError", "exception_message": "timeout",
                                   "exception_traceback": "test", "occurred_at": trial["finished_at"]}
    if fault == "metadata": trial["agent_result"] = None
    if fault == "wrong_agent":
        trial["agent_info"]["name"] = "different-agent"
        trial["config"]["agent"]["import_path"] = "different:Agent"
    write_json(path, trial)
    if fault == "wrong_preset":
        (path.parent / "agent/gpu-boot.log").write_text("HARBOR_GPU_BOOT_OK wrong-preset\n")
    if fault == "graded_delivery":
        report_path = path.parent / "verifier/results.json"
        report = json.loads(report_path.read_text())
        report["submissions"][0]["status"] = "graded"
        write_json(report_path, report)
    with pytest.raises(ValueError):
        check_job(dataset, job)
