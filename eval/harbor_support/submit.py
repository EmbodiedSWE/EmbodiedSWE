#!/opt/venv/bin/python
"""Atomically snapshot solution/; serialize concurrent submits on Linux with flock."""

import argparse
import fcntl
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .files import content_hash, inventory, write_json
else:
    from files import content_hash, inventory, write_json


def snapshot(solution: Path, submissions: Path, *, note: str, limit: int) -> Path:
    if not (solution / "solve.py").is_file():
        raise ValueError(f"No solve.py in {solution}")
    submissions.mkdir(parents=True, exist_ok=True)
    with (submissions / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = [p for p in submissions.iterdir() if p.name.isdecimal()]
        if len(existing) >= limit:
            raise ValueError(f"Submission limit ({limit}) reached; final solution/ is still graded")
        number = max((int(p.name) for p in existing), default=0) + 1
        target = submissions / f"{number:06d}"
        before = inventory(solution)
        temp = Path(tempfile.mkdtemp(prefix=".pending-", dir=submissions))
        try:
            shutil.copytree(solution, temp / "solution")
            copied = inventory(temp / "solution")
            if copied != before or inventory(solution) != before:
                raise ValueError("Solution changed during submit; stop writers and submit again")
            write_json(temp / "submission.json", {
                "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "note": note, "files": copied, "sha256": content_hash(copied),
            })
            temp.rename(target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("note", nargs="*", help="Short description of this checkpoint")
    args = parser.parse_args()
    config = json.loads(Path("/task/protocol.json").read_text())
    try:
        path = snapshot(Path("/workspace/solution"), Path("/submissions"),
                        note=" ".join(args.note), limit=config["max_submissions"])
    except (OSError, ValueError) as exc:
        parser.exit(1, f"submit: {exc}\n")
    print(f"Submitted {path.name}. Continue working; final solution/ is also graded.")


if __name__ == "__main__":
    main()
