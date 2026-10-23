"""Strict publication summary: never turn missing/failed evaluations into zero."""

import json
from pathlib import Path

from .validate import validate_dataset
from .verifier import unit_number


def summarize(dataset: Path, jobs: list[Path], attempts: int = 1) -> dict:
    from dirhash import dirhash
    from harbor.models.job.result import JobResult
    from harbor.models.task.task import Task
    from harbor.models.trial.result import TrialResult

    if type(attempts) is not int or attempts < 1:
        raise ValueError("Expected a positive number of attempts per task")
    validate_dataset(dataset)
    manifest = json.loads((dataset / "dataset.json").read_text())
    tasks = {record["task"]: record for record in manifest["tasks"]}
    # TrialResult uses the declared Harbor name, including its organization.
    # Dataset directories use our short catalog IDs. Never strip arbitrary prefixes.
    task_names = {Task(dataset / name).name: name for name in tasks}
    if len(task_names) != len(tasks):
        raise ValueError("Dataset declares duplicate Harbor task names")
    trials = {name: [] for name in tasks}
    checksums, identities, trial_ids = {}, set(), set()
    agent = None
    if not jobs:
        raise ValueError("Provide at least one Harbor job directory")
    for job in jobs:
        job_result = JobResult.model_validate_json((job / "result.json").read_text())
        results = sorted(job.glob("*/result.json"))
        if not results or job_result.finished_at is None or job_result.n_total_trials != len(results):
            raise ValueError(f"Incomplete Harbor job or missing trial results: {job}")
        for path in results:
            trial = TrialResult.model_validate_json(path.read_text())
            if trial.task_name not in task_names:
                raise ValueError(f"Unexpected task {trial.task_name} in {path}")
            name = task_names[trial.task_name]
            if trial.id in trial_ids:
                raise ValueError(f"Duplicate trial ID in {path}")
            trial_ids.add(trial.id)
            if trial.finished_at is None or trial.step_results:
                raise ValueError(f"Incomplete or unsupported multi-step trial: {path}")
            # An exhausted agent budget still permits valid checkpoint grading.
            if trial.exception_info and trial.exception_info.exception_type != "AgentTimeoutError":
                raise ValueError(f"Trial failed ({trial.exception_info.exception_type}): {path}")
            if trial.verifier_result is None or trial.verifier_result.rewards is None:
                raise ValueError(f"Trial has no verified reward: {path}")
            if trial.verifier_environment_mode is None or trial.verifier_environment_mode.value != "separate":
                raise ValueError(f"Trial did not use separate verification: {path}")
            if name not in checksums:
                # This is the exact checksum used by Harbor 0.24.0 TrialResult.
                checksums[name] = dirhash(dataset / name, "sha256")
            if trial.task_checksum != checksums[name]:
                raise ValueError(f"Trial ran a different task export: {path}")
            agent = trial.agent_info.model_dump(mode="json")
            identity = {"agent": agent, "config": trial.config.agent.model_dump(mode="json"),
                        "timeout_multiplier": trial.config.timeout_multiplier,
                        "agent_timeout_multiplier": trial.config.agent_timeout_multiplier,
                        "verifier_timeout_multiplier": trial.config.verifier_timeout_multiplier}
            identities.add(json.dumps(identity, sort_keys=True))
            if len(identities) != 1:
                raise ValueError("Jobs mix agent/model versions, configurations, or time budgets")
            verifier = path.parent / "verifier"
            if (verifier / "error.json").exists():
                raise ValueError(f"Verifier reported an error: {path}")
            report = json.loads((verifier / "results.json").read_text())
            protocol = json.loads((dataset / name / "tests/protocol.json").read_text())
            if report.get("protocol") != protocol:
                raise ValueError(f"Verifier protocol differs from the export: {path}")
            reward = report.get("reward")
            if not isinstance(reward, dict) or set(reward) != {"score", "success"}:
                raise ValueError(f"Missing score/success metrics: {path}")
            reward = {key: unit_number(value, key) for key, value in reward.items()}
            on_disk = json.loads((verifier / "reward.json").read_text())
            if reward != on_disk or reward != trial.verifier_result.rewards:
                raise ValueError(f"Harbor and verifier rewards disagree: {path}")
            trials[name].append({"trial_id": str(trial.id), "trial_name": trial.trial_name,
                                 "selected_submission": report["selected_submission"], **reward})
    for name, values in trials.items():
        if len(values) != attempts:
            raise ValueError(f"{name}: expected {attempts} trials, found {len(values)}")
    per_task = {name: {key: sum(row[key] for row in values) / attempts
                      for key in ("score", "success")} for name, values in trials.items()}
    return {"schema_version": 1, "task_count": len(tasks), "attempts_per_task": attempts,
            "agent": agent, "dataset_tasks": manifest["tasks"], "per_task": per_task,
            "mean": {key: sum(row[key] for row in per_task.values()) / len(tasks)
                     for key in ("score", "success")}, "trials": trials}
