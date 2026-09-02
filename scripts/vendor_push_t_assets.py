#!/usr/bin/env python3
"""Verify and extract the two upstream RoboDojo ``push_T`` USDZ packages.

Download the files listed in
``robobench/suites/puzzle/assets/push_t/SOURCE.md`` into one directory, then run:

    python3 scripts/vendor_push_t_assets.py --source-dir /path/to/downloads

The script is stdlib-only and preserves the source ``main.usdc`` bytes exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


ASSETS = {
    "t": {
        "sha256": "98bae9ae3dddf67e58004695930c09f3d36f715dfa2bf5d65d1ed2b7179fa493",
        "output": ("block", "source.usdc"),
    },
    "target_t": {
        "sha256": "4823fa3cfaf1a41cd26dd29deca0f5549da20f1211284d5d944618701a2115bf",
        "output": ("target_pad", "source.usdc"),
    },
}
GITHUB_FILE_LIMIT = 100 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_one(source_dir: Path, output_dir: Path, name: str, spec: dict) -> None:
    archive = source_dir / f"{name}.usdz"
    if not archive.is_file():
        raise FileNotFoundError(f"missing raw asset: {archive}")
    actual = sha256(archive)
    if actual != spec["sha256"]:
        raise ValueError(
            f"checksum mismatch for {archive.name}: expected {spec['sha256']}, got {actual}"
        )
    with zipfile.ZipFile(archive) as package:
        files = [item for item in package.infolist() if not item.is_dir()]
        if [item.filename for item in files] != ["main.usdc"]:
            raise ValueError(
                f"unexpected members in {archive.name}: {[item.filename for item in files]}"
            )
        member = files[0]
        if member.file_size >= GITHUB_FILE_LIMIT:
            raise ValueError(f"{archive.name}:{member.filename} exceeds GitHub's file limit")
        relative_dir, filename = spec["output"]
        destination = output_dir / relative_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with package.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"[vendor] {name}/{filename} ({member.file_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="directory containing t.usdz and target_t.usdz",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = repo / "robobench" / "suites" / "puzzle" / "assets" / "push_t"
    for name, spec in ASSETS.items():
        vendor_one(args.source_dir.resolve(), output, name, spec)
    print(f"[vendor] complete: {output}")


if __name__ == "__main__":
    main()
