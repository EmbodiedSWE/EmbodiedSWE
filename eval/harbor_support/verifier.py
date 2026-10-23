#!/opt/venv/bin/python
"""Grade submitted checkpoints and final solution in fresh simulator processes.

Only complete, validated results produce a Harbor reward. A simulator crash, a
missing verdict, or an unavailable VLM judge is an evaluation error, not a zero.
"""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

if __package__:
    from .files import content_hash, inventory, write_json
else:
    from files import content_hash, inventory, write_json


class EvaluationError(RuntimeError):
    pass


def unit_number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise EvaluationError(f"Invalid {name}: expected a finite number in [0, 1]")
    return float(value)


def validate_verdict(verdict: dict, config: dict, seed: int) -> dict[str, float]:
    if not isinstance(verdict, dict):
        raise EvaluationError("Verdict must be a JSON object")
    for key, expected in (("preset", config["preset"]), ("scene", config["scene"]),
                          ("seed", seed), ("num_envs", config["num_envs"])):
        if verdict.get(key) != expected:
            raise EvaluationError(f"Verdict {key} mismatch: expected {expected!r}")
    per = verdict.get("per_traj")
    if not isinstance(per, list) or len(per) != config["num_envs"]:
        raise EvaluationError("Missing or incomplete per-trajectory verdicts")
    for item in per:
        if not isinstance(item, dict) or type(item.get("success")) is not bool:
            raise EvaluationError("Invalid per-trajectory success")
        unit_number(item.get("score"), "trajectory score")
        final = item.get("final", {})
        if not isinstance(final, dict):
            raise EvaluationError("Invalid final-frame audit")
        for gate in final.values():
            if not isinstance(gate, dict):
                raise EvaluationError("Invalid final-frame gate audit")
            if gate.get("error"):
                raise EvaluationError("Final-frame judge failed; inspect rollout verdict.json")
        if config["cameras"]:
            # Every currently exported Newton task has one mandatory gate.
            # An absent audit is an incomplete evaluation, even for a zero score.
            gate = final.get("plausible")
            if not isinstance(gate, dict):
                raise EvaluationError("Missing final-frame judge audit")
            decision = gate.get("decision")
            parsed = gate.get("parsed")
            if (type(decision) is not bool or not isinstance(parsed, dict)
                    or type(parsed.get("plausible")) is not bool
                    or parsed["plausible"] != decision
                    or not isinstance(parsed.get("reason"), str)
                    or not parsed["reason"].strip()
                    or type(gate.get("http_status")) is not int or gate["http_status"] != 200
                    or type(gate.get("attempts")) is not int or not 1 <= gate["attempts"] <= 4
                    or not isinstance(gate.get("raw_response"), dict)
                    or not gate["raw_response"]
                    or any(not isinstance(gate.get(key), str) or not gate[key].strip()
                           for key in ("model", "logid", "prompt", "image", "answer_text"))):
                raise EvaluationError("Incomplete or inconsistent final-frame judge audit")
            if unit_number(gate.get("value"), "judge gate value") != float(decision):
                raise EvaluationError("Final-frame judge value disagrees with its decision")
            if item["success"] and not decision:
                raise EvaluationError("Trajectory success contradicts the final-frame judge")
    score = round(sum(item["score"] for item in per) / len(per), 4)
    success = round(sum(item["success"] for item in per) / len(per), 4)
    if unit_number(verdict.get("score"), "score") != score:
        raise EvaluationError("Score disagrees with trajectory verdicts")
    if unit_number(verdict.get("success"), "success") != success:
        raise EvaluationError("Success disagrees with trajectory verdicts")
    return {"score": score, "success": success}


def _kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def run_rollout(solution: Path, out: Path, config: dict, seed: int, *,
                driver: Path, graders: Path, python: str = sys.executable) -> dict:
    if config["cameras"] and not os.environ.get("VLM_JUDGE_API_KEY"):
        raise EvaluationError("VLM_JUDGE_API_KEY is required for deformable grading")
    out.mkdir(parents=True, exist_ok=False)
    # Each replay receives a fresh writable working copy. Solves often create
    # scratch files; those must not alter the checkpoint or another seed.
    working = out / "solution"
    shutil.copytree(solution, working)
    command = [python, str(driver), "--preset", config["preset"], "--scene", config["scene"],
               "--seed", str(seed), "--num-envs", str(config["num_envs"]),
               "--solution", str(working / "solve.py"), "--graders", str(graders),
               "--out", str(out)]
    if config["cameras"]:
        command.append("--cameras")
    timed_out = False
    worker = config.get("worker_user")
    credentials = {}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if worker:
        import pwd
        account = pwd.getpwnam(worker)
        if os.geteuid() != 0 or account.pw_uid == 0:
            raise EvaluationError("A root verifier and a non-root rollout worker are required")
        os.chown(out, account.pw_uid, account.pw_gid)
        for path in [working, *working.rglob("*")]:
            os.chown(path, account.pw_uid, account.pw_gid)
        credentials = {"user": account.pw_uid, "group": account.pw_gid, "extra_groups": []}
        env["HOME"] = account.pw_dir
    for path in [working, *working.rglob("*")]:
        path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)
    with (out / "process.log").open("wb") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=working,
                                env=env, **credentials)
        try:
            try:
                code = proc.wait(timeout=config["rollout_timeout_sec"])
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=config["termination_grace_sec"])
                except subprocess.TimeoutExpired:
                    _kill_group(proc.pid, signal.SIGKILL)
                    code = proc.wait()
        finally:
            # A solve can spawn helpers. No worker processes may bleed into the next rollout.
            _kill_group(proc.pid, signal.SIGKILL)
            proc.wait()
    write_json(out / "process.json", {"exit_code": code, "timed_out": timed_out})
    if code != 0 or not (out / "verdict.json").exists():
        try:
            phase = json.loads((out / "phase.json").read_text())["phase"]
        except (OSError, ValueError, KeyError):
            phase = "unknown"
        if phase in ("loading_solution", "solving") and (timed_out or code >= 0):
            reason = "exceeded its rollout budget" if timed_out else f"exited ({code})"
            return {"score": 0.0, "success": 0.0, "timed_out": timed_out,
                    "solution_error": f"Solution {reason} without a verdict"}
        if timed_out:
            raise EvaluationError(f"Grading process timed out in {phase}: {out}")
    if code != 0:
        raise EvaluationError(f"Grading process exited {code}: {out / 'process.log'}")
    try:
        verdict = json.loads((out / "verdict.json").read_text())
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Missing or invalid verdict: {out}") from exc
    reward = validate_verdict(verdict, config, seed)
    return {**reward, "timed_out": timed_out, "solution_error": verdict.get("error")}


def candidates(submissions: Path, final: Path, limit: int) -> list[tuple[str, Path]]:
    found = []
    if submissions.exists():
        if submissions.is_symlink() or not submissions.is_dir():
            raise EvaluationError("Submissions must be a directory")
        for entry in sorted(submissions.iterdir(), key=lambda p: p.name):
            if entry.name.isdecimal():
                if len(entry.name) != 6 or entry.is_symlink() or not entry.is_dir():
                    raise EvaluationError(f"Invalid submission directory: {entry.name}")
                found.append((entry.name, entry / "solution"))
    if len(found) > limit:
        raise EvaluationError(f"Received more than {limit} submitted checkpoints")
    return [*found, ("final", final)]


def verify(config: dict, *, submissions: Path, final: Path, out: Path,
           driver: Path, graders: Path, python: str = sys.executable) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    # Never leave a previous reward behind on a failed rerun.
    for filename in ("reward.json", "reward.txt"):
        (out / filename).unlink(missing_ok=True)
    if any((out / name).exists() for name in ("results.json", "rollouts", "inputs")):
        raise EvaluationError("Verifier output already contains a run; use a fresh output directory")
    results, seen, prepared = [], {}, []
    # Freeze every input before executing any submitted code. The production
    # verifier owns these copies as root; workers only receive disposable copies.
    for name, solution in candidates(submissions, final, config["max_submissions"]):
        entry = {"submission": name, "score": 0.0, "success": 0.0, "rollouts": []}
        if not (solution / "solve.py").is_file():
            entry["status"] = "missing_solution"
        else:
            try:
                files = inventory(solution)
                digest = content_hash(files)
                if name != "final":
                    receipt = json.loads((solution.parent / "submission.json").read_text())
                    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
                        raise ValueError("Invalid submission receipt")
                    if receipt.get("files") != files or receipt.get("sha256") != digest:
                        raise ValueError("Snapshot contents differ from their submission receipt")
            except (OSError, ValueError) as exc:
                entry.update(status="invalid_submission", error=str(exc))
                prepared.append((entry, None))
                continue
            frozen = out / "inputs" / name
            # Storage failures during freezing are infrastructure errors, not
            # bad submissions: do not silently convert a full disk to a zero.
            shutil.copytree(solution, frozen)
            if inventory(frozen) != files or inventory(solution) != files:
                raise EvaluationError("Solution changed while preparing verification")
            for path in [frozen, *frozen.rglob("*")]:
                path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
            entry["sha256"] = digest
            prepared.append((entry, frozen))
            continue
        prepared.append((entry, None))
    for entry, solution in prepared:
        name = entry["submission"]
        if solution is not None:
            digest = entry["sha256"]
            if digest in seen:
                previous = seen[digest]
                entry.update(score=previous["score"], success=previous["success"],
                             status="duplicate", duplicate_of=previous["submission"])
            else:
                for seed in config["seeds"]:
                    result = run_rollout(solution, out / "rollouts" / name / f"seed_{seed}",
                                         config, seed, driver=driver, graders=graders, python=python)
                    entry["rollouts"].append({"seed": seed, **result})
                entry.update(status="graded", **{
                    # Native per-seed metrics are rounded decimals. Preserve
                    # exact ties rather than letting float summation pick a
                    # different checkpoint by a rounding bit.
                    key: float(sum(Decimal(str(r[key])) for r in entry["rollouts"])
                               / len(entry["rollouts"]))
                    for key in ("score", "success")
                })
                seen[digest] = entry
        results.append(entry)
        write_json(out / "progress.json", {"submissions": results})
    # Select one deliverable after averaging its seeds; never cherry-pick a different
    # checkpoint per seed or combine score from one with success from another.
    best = max(results, key=lambda r: (r["score"], r["success"]))
    report = {"protocol": config, "selected_submission": best["submission"],
              "submissions": results, "reward": {key: best[key] for key in ("score", "success")}}
    write_json(out / "results.json", report)
    temp = out / ".reward.json.tmp"
    write_json(temp, report["reward"])
    temp.replace(out / "reward.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/tests/protocol.json"))
    parser.add_argument("--submissions", type=Path, default=Path("/submissions"))
    parser.add_argument("--final", type=Path, default=Path("/workspace/solution"))
    parser.add_argument("--out", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--driver", type=Path, default=Path("/tests/grade.py"))
    parser.add_argument("--graders", type=Path, default=Path("/tests/graders"))
    args = parser.parse_args()
    try:
        verify(json.loads(args.config.read_text()), submissions=args.submissions,
               final=args.final, out=args.out, driver=args.driver, graders=args.graders)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        for filename in ("reward.json", "reward.txt"):
            (args.out / filename).unlink(missing_ok=True)
        write_json(args.out / "error.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
