"""Deterministic, streaming content receipts for exported trees and submissions."""

import hashlib
import json
import stat
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a real directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        name = path.relative_to(root).as_posix()
        if stat.S_ISDIR(mode):
            result[name] = {"type": "directory"}
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Symlink or special file is not supported: {path}")
        result[name] = {"type": "file", "sha256": file_hash(path), "executable": bool(mode & 0o111)}
    return result


def content_hash(files: dict[str, dict]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
