"""Trusted reference inputs and per-trajectory parity checks for acceptance runs.

These checks consume independently retained native grader results. They do not
establish the correctness of a supplied reference solution or its provenance.
"""

import json
import math
from pathlib import Path

from .catalog import TASKS
from .files import content_hash, file_hash, inventory
from .verifier import validate_verdict


def load_references(path: Path) -> dict:
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or not isinstance(data.get("tasks"), dict) or not data["tasks"]):
        raise ValueError("Expected reference schema_version=1 with a nonempty tasks object")
    catalog = {task.id for task in TASKS}
    for name, case in data["tasks"].items():
        if name not in catalog or not isinstance(case, dict):
            raise ValueError(f"Unknown reference task: {name}")
        if not isinstance(case.get("checkpoints"), list) or not case["checkpoints"]:
            raise ValueError(f"{name}: provide at least one checkpoint and a final solution")
        for record in [*case["checkpoints"], case.get("final")]:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError(f"{name}: invalid reference solution")
            solution = path.parent / record["path"]
            files = inventory(solution)
            if "solve.py" not in files or files["solve.py"]["type"] != "file":
                raise ValueError(f"{name}: reference lacks solve.py")
            if content_hash(files) != record.get("sha256"):
                raise ValueError(f"{name}: reference solution checksum mismatch")
            record["path"] = str(solution.absolute())
            if not isinstance(record.get("native"), dict) or not record["native"]:
                raise ValueError(f"{name}: provide native verdicts keyed by seed")
            for seed, baseline in record["native"].items():
                if not seed.isdecimal() or str(int(seed)) != seed or not 0 <= int(seed) < 2**32:
                    raise ValueError(f"{name}: invalid native seed {seed}")
                if not isinstance(baseline, dict) or not isinstance(baseline.get("path"), str):
                    raise ValueError(f"{name}: invalid native verdict record")
                verdict = path.parent / baseline["path"]
                if verdict.is_symlink() or not verdict.is_file() or file_hash(verdict) != baseline.get("sha256"):
                    raise ValueError(f"{name}: native verdict checksum mismatch")
                baseline["path"] = str(verdict.absolute())
    return data


def compare_verdicts(actual: dict, expected: dict, protocol: dict, seed: int,
                     *, score_tolerance: float = 0.0) -> dict:
    if (type(score_tolerance) not in (int, float) or not math.isfinite(score_tolerance)
            or not 0 <= score_tolerance <= 0.01):
        raise ValueError("Score tolerance must be finite and in [0, 0.01]")
    got = validate_verdict(actual, protocol, seed)
    want = validate_verdict(expected, protocol, seed)
    for label, verdict in (("Harbor", actual), ("native", expected)):
        if verdict.get("error") or verdict.get("timed_out"):
            raise ValueError(f"{label} reference rollout failed or timed out")
    deltas = []
    for index, (a, b) in enumerate(zip(actual["per_traj"], expected["per_traj"], strict=True)):
        if a.get("env") != index or b.get("env") != index:
            raise ValueError("Missing or reordered trajectory identities")
        delta = abs(a["score"] - b["score"])
        # Decimal scores are rounded by the native grader; tolerate only float subtraction noise.
        if a["success"] != b["success"] or delta > score_tolerance + 1e-12:
            raise ValueError(f"Native/Harbor trajectory {index} differs: {a['score']}/{a['success']} "
                             f"versus {b['score']}/{b['success']}")
        if protocol["cameras"]:
            ga, gb = a["final"]["plausible"], b["final"]["plausible"]
            if any(ga[key] != gb[key] for key in ("decision", "model", "prompt")):
                raise ValueError(f"Native/Harbor visual judgment differs for trajectory {index}")
        deltas.append(delta)
    return {"seed": seed, "harbor": got, "native": want,
            "max_trajectory_score_delta": max(deltas), "score_tolerance": score_tolerance}
