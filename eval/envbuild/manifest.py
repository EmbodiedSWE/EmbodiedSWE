"""Provenance receipt: the experiment folder explains and can reproduce itself."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def write_receipt(exp_dir: Path, payload: dict) -> None:
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    payload = {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_rev": rev,
        "repo_dirty_files": len(dirty.splitlines()),
        **payload,
    }
    (exp_dir / "resolved.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [f"{k}: {v}" for k, v in payload.items() if k != "stages"]
    for s in payload.get("stages", []):
        lines.append(f"stage {s['dir']}: preset={s['preset']} tree_sha256={s['tree_sha256'][:16]}…")
    (exp_dir / "MANIFEST").write_text("\n".join(lines) + "\n")
