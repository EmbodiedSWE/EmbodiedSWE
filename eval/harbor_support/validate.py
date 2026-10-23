"""Validate a portable export's receipt and the pinned Harbor task contract."""

import json
from importlib.metadata import version
from pathlib import Path

from .catalog import TASKS
from .files import content_hash, inventory


def validate_dataset(root: Path) -> int:
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    if version("harbor") != "0.24.0":
        raise ValueError("This adapter is validated against harbor==0.24.0; install requirements.txt")
    dataset = json.loads((root / "dataset.json").read_text())
    records = dataset["tasks"]
    if not records or dataset["task_count"] != len(records):
        raise ValueError("Dataset task count does not match the receipt")
    catalog = {task.id: task for task in TASKS}
    names = [record["task"] for record in records]
    if len(names) != len(set(names)) or any(name not in catalog for name in names):
        raise ValueError("Dataset contains duplicate or unknown tasks")
    directories = {p.name for p in root.iterdir() if p.is_dir()}
    if directories != set(names):
        raise ValueError("Dataset directories do not match the recorded task roster")
    for record in records:
        path = root / record["task"]
        receipt = json.loads((path / "export.json").read_text())
        actual = inventory(path)
        del actual["export.json"]
        if actual != receipt["files"] or content_hash(actual) != receipt["sha256"]:
            raise ValueError(f"Export content mismatch: {path.name}")
        for field in ("task", "preset", "sha256", "image"):
            if record[field] != receipt[field]:
                raise ValueError(f"Dataset and task receipts disagree on {field}: {path.name}")
        if receipt["preset"] != catalog[path.name].preset:
            raise ValueError(f"Preset differs from the declared catalog: {path.name}")
        task = Task(path)
        definition = resolve_verifier_environment_definition(task.config, task.paths)
        if definition is None or not definition.bundled_tests or definition.directory != path / "tests":
            raise ValueError(f"Expected a dedicated verifier build context: {path.name}")
    return len(records)
