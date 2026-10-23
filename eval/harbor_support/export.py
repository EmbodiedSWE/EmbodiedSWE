"""Transactional export of boot-validated worlds to self-contained Harbor tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from envbuild.condition import Condition, check_facts
from envbuild.prompts import render_task_dir

from .catalog import TaskSpec
from .files import content_hash, inventory, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEMPLATES = HERE / "templates"
DEFAULT_CONDITION = HERE / "condition.yaml"


@dataclass(frozen=True)
class Settings:
    agent_timeout_sec: float = 14400
    rollout_timeout_sec: float = 1800
    termination_grace_sec: float = 30
    max_submissions: int = 64
    seeds: tuple[int, ...] = (0,)
    num_envs: int = 1
    cpus: int = 8
    memory_mb: int = 32768
    allow_mutable_images: bool = False

    def validate(self) -> None:
        for name in ("agent_timeout_sec", "rollout_timeout_sec", "termination_grace_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_submissions", "num_envs", "cpus", "memory_mb"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be nonempty and unique")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("Seeds must be unsigned 32-bit integers")

    def protocol(self, task: TaskSpec) -> dict:
        if task.backend == "newton" and self.num_envs != 1:
            raise ValueError("Newton tasks require num_envs=1; use multiple seeds for replication")
        return {"schema_version": 1, "selection": "best_mean_score_then_success_then_earliest",
                "preset": task.preset, "scene": task.scene, "seeds": list(self.seeds),
                "num_envs": self.num_envs, "cameras": task.backend == "newton",
                "max_submissions": self.max_submissions,
                "rollout_timeout_sec": self.rollout_timeout_sec,
                "termination_grace_sec": self.termination_grace_sec, "worker_user": "agent"}


def world_hash(root: Path) -> str:
    """Compatible with envbuild.manifest.tree_hash, without loading assets into RAM."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_image(image: str, allow_mutable: bool) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image):
        raise ValueError(f"Invalid image reference: {image!r}")
    if not allow_mutable and not re.search(r"@sha256:[0-9a-f]{64}$", image):
        raise ValueError("Use an image pinned by @sha256 digest; --allow-mutable-images is for development")


def load_world(root: Path, task: TaskSpec, condition: Condition) -> tuple[Path, dict, dict]:
    exp = root / task.id
    receipt = json.loads((exp / "resolved.json").read_text())
    if type(receipt.get("boot_seed")) is not int or not 0 <= receipt["boot_seed"] < 2**32:
        raise ValueError(f"{exp}: missing or invalid boot seed")
    stages = receipt.get("stages", [])
    if len(stages) != 1 or stages[0].get("preset") != task.preset:
        raise ValueError(f"{exp}: expected exactly one stage for {task.preset}")
    record = stages[0]
    dirname = record.get("dir", "")
    if not dirname or Path(dirname).name != dirname or dirname in (".", ".."):
        raise ValueError(f"{exp}: unsafe stage path")
    stage = exp / "stages" / dirname
    if stage.is_symlink() or not stage.resolve().is_relative_to(exp.resolve()):
        raise ValueError(f"{exp}: stage escapes the world directory")
    if record.get("boot_checked") is not True:
        raise ValueError(f"{exp}: world has not passed boot validation")
    files = inventory(stage / "bench")
    banned = [name for name in files if any(part in {"grader", "smokes", "__pycache__", ".git"}
              for part in Path(name).parts) or name.endswith((".pyc", ".pyo"))]
    if banned:
        raise ValueError(f"{exp}: privileged or stale files in agent world: {banned[:3]}")
    if world_hash(stage / "bench") != record.get("tree_sha256"):
        raise ValueError(f"{exp}: world changed since boot validation; rebuild it")
    if not (stage / "describe.md").read_text().strip():
        raise ValueError(f"{exp}: empty task description")
    facts = {"set_states": receipt.get("set_states"),
             "control_mode_frozen": record.get("control_mode_frozen")}
    expected = {"set_states": condition.enabled("set_states"),
                "control_mode_frozen": not condition.enabled("control_mode")}
    if facts != expected:
        raise ValueError(f"{exp}: world features {facts} do not match condition {expected}")
    check_facts(condition, facts)
    return stage, receipt, facts


def _dockerfile(image: str, verifier: bool) -> str:
    common = f'''FROM {image}
USER root
RUN test -x /opt/venv/bin/python && /opt/venv/bin/python -c "import isaaclab, torch, numpy, huggingface_hub, filelock"
RUN id -u agent >/dev/null 2>&1 || useradd --create-home --shell /bin/bash agent
ENV PATH="/opt/venv/bin:${{PATH}}" PYTHONPATH="/bench:/task/tools" PYTHONDONTWRITEBYTECODE=1
ENV OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y HF_HUB_OFFLINE=1 COSIGEN_ASSET_DIR=/bench
ENV XDG_CACHE_HOME=/ovcache NVIDIA_DRIVER_CAPABILITIES=all
RUN /opt/venv/bin/python -c "import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_paths()['purelib']) / 'isaacsim' / 'kit'; [(p / n).mkdir(parents=True, exist_ok=True) for n in ('cache', 'data', 'logs')]" && for d in /opt/venv/lib/python*/site-packages/isaacsim/kit/cache /opt/venv/lib/python*/site-packages/isaacsim/kit/data /opt/venv/lib/python*/site-packages/isaacsim/kit/logs; do chmod -R a+rwX "$d"; done
COPY bench/ /bench/
RUN chmod -R a+rX,a-w /bench && mkdir -p /bench/.cache/robobench_bundles /ovcache /workspace/solution /submissions && chmod -R 777 /bench/.cache /ovcache && chown -R agent:agent /workspace /submissions
WORKDIR /workspace
ENTRYPOINT []
CMD ["sleep", "infinity"]
'''
    if verifier:
        return common + '''COPY grade.py verifier.py files.py protocol.json test.sh /tests/
COPY graders/ /tests/graders/
RUN chmod -R a+rX,a-w /tests && chmod 755 /tests/test.sh
'''
    return common + '''COPY task/ /task/
COPY submit.py files.py /opt/robobench-harbor/
RUN chmod -R a+rX,a-w /task /opt/robobench-harbor && chmod 755 /opt/robobench-harbor/submit.py && ln -s /opt/robobench-harbor/submit.py /usr/local/bin/submit
'''


def _task_toml(task: TaskSpec, settings: Settings) -> str:
    timeout = (settings.max_submissions + 1) * len(settings.seeds) * (
        settings.rollout_timeout_sec + settings.termination_grace_sec) + 300
    text = f'''schema_version = "1.4"
artifacts = [{{ source = "/workspace/solution", destination = "solution" }}, {{ source = "/submissions", destination = "submissions" }}]

[task]
name = "robobench/{task.id}"

[metadata]
benchmark = "RoboBench"
suite = "{task.suite}"
preset = "{task.preset}"
backend = "{task.backend}"
protocol = "best-submission"
harbor_version = "0.24.0"

[agent]
timeout_sec = {float(settings.agent_timeout_sec)}
user = "agent"

[verifier]
timeout_sec = {float(timeout)}
environment_mode = "separate"
user = "root"
'''
    if task.backend == "newton":
        # Empty deliveries can be scored without starting the native grader.
        # Its credential is checked only when an actual rollout is required.
        text += 'env = { VLM_JUDGE_API_KEY = "${VLM_JUDGE_API_KEY:-}" }\n'
    for section in ("environment", "verifier.environment"):
        text += f'''
[{section}]
build_timeout_sec = 3600.0
cpus = {settings.cpus}
memory_mb = {settings.memory_mb}
gpus = 1
workdir = "/workspace"
'''
    return text


def export_dataset(*, worlds: Path, output: Path, tasks: tuple[TaskSpec, ...],
                   images: dict[str, str], condition: Condition, settings: Settings) -> Path:
    settings.validate()
    if not tasks or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Select a nonempty set of distinct tasks")
    for task in tasks:
        if task.backend not in images:
            raise ValueError(f"Provide a {task.backend} runtime image")
        validate_image(images[task.backend], settings.allow_mutable_images)
        settings.protocol(task)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.export.lock"
    # Lock prevents concurrent writers from publishing different versions to one destination.
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temp = None
    try:
        loaded = [(task, *load_world(worlds, task, condition)) for task in tasks]
        temp = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        records = []
        for task, stage, receipt, facts in loaded:
            target = temp / task.id
            environment, tests = target / "environment", target / "tests"
            environment.mkdir(parents=True)
            tests.mkdir()
            image = images[task.backend]
            # Copy source worlds, never link to mutable experiment files.
            for context in (environment, tests):
                shutil.copytree(stage / "bench", context / "bench",
                                ignore=shutil.ignore_patterns(".cache"))
                (context / "Dockerfile").write_text(_dockerfile(image, context == tests))
                shutil.copy2(TEMPLATES / "docker-compose.yaml", context / "docker-compose.yaml")
            protocol = settings.protocol(task)
            task_dir = environment / "task"
            render_task_dir(task_dir, scene=task.scene, preset=task.preset,
                            describe_text=(stage / "describe.md").read_text(), facts=facts,
                            condition=condition, seed=receipt["boot_seed"])
            # Harbor owns the lifecycle; the generic runner contract would incorrectly
            # promise live grading and a different submission directory layout.
            instruction = (TEMPLATES / "contract.md").read_text()
            instruction += f"\nAgent wall-clock budget: {settings.agent_timeout_sec:g} seconds.\n"
            instruction += "\n" + (task_dir / "task.md").read_text()
            for path in sorted((task_dir / "rules").glob("*.md")):
                instruction += "\n" + path.read_text()
            if (task_dir / "tool_router.md").exists():
                instruction += "\n" + (task_dir / "tool_router.md").read_text()
            (task_dir / "instructions.md").write_text(instruction)
            (target / "instruction.md").write_text(instruction)
            write_json(task_dir / "protocol.json", protocol)
            write_json(tests / "protocol.json", protocol)
            for name in ("submit.py", "files.py"):
                shutil.copy2(HERE / name, environment / name)
            for name in ("verifier.py", "files.py"):
                shutil.copy2(HERE / name, tests / name)
            shutil.copy2(REPO / "eval/grader/grade.py", tests / "grade.py")
            shutil.copy2(TEMPLATES / "test.sh", tests / "test.sh")
            shutil.copytree(REPO / "robobench/suites" / task.suite / "grader", tests / "graders",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (target / "task.toml").write_text(_task_toml(task, settings))
            # Reject source changes while copying, before making any dataset visible.
            if world_hash(stage / "bench") != receipt["stages"][0]["tree_sha256"]:
                raise ValueError(f"World changed during export: {task.id}")
            files = inventory(target)
            record = {"task": task.id, "preset": task.preset, "image": image,
                      "source_world": receipt, "files": files, "sha256": content_hash(files)}
            write_json(target / "export.json", record)
            records.append({key: record[key] for key in ("task", "preset", "image", "sha256")})
        write_json(temp / "dataset.json", {"schema_version": 1, "harbor_version": "0.24.0",
                   "task_count": len(records), "tasks": records,
                   "condition": condition.as_record(), "selection": "best-submission"})
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        # mkdtemp starts private, and downloaded assets may be mode 0600.
        # Published contexts must be readable by builders and non-root workers.
        # Read/write modes are normalized; execute flags (in receipts) survive.
        for path in [temp, *temp.rglob("*")]:
            path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)
        temp.rename(output)
        temp = None
        return output
    finally:
        if temp is not None:
            shutil.rmtree(temp)
        os.close(lock)
        lock_path.unlink()
