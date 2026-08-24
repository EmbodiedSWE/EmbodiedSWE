#!/usr/bin/env python3
"""Vendor the upstream RoboDojo ``fill_egg_holder`` assets into CoSiGen.

Raw USDZ downloads deliberately stay outside the repository.  Download the three files
listed in ``robobench/suites/packing/assets/egg_carton/SOURCE.md`` into one directory, then:

    python3 scripts/vendor_egg_carton_assets.py --source-dir /path/to/downloads

The basket package is larger than GitHub's 100 MiB per-file limit.  Extracting the USDZ
keeps every tracked file below that limit while preserving all relative texture/sublayer
paths.  The script is intentionally stdlib-only so asset provenance can be reproduced on
a machine without Isaac Sim.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


ASSETS = {
    "egg_holder": {
        "sha256": "417f7b067e13f5371c84164daf2d597dd29b7d96f7af8c3952758695e400f830",
        "members": (
            "main.usdc",
            "SubUSDs/material.usd",
            "SubUSDs/link_0_0.usd",
        ),
    },
    "egg": {
        "sha256": "867733884c64ee06b16374c7228e328824a16538d780f2ed2f61b889a8fc872d",
        "members": (
            "main.usdc",
            "SubUSDs/textures/9a800fe9e9709dcf0a7378a161ba5ba0.png",
        ),
    },
    "egg_basket": {
        "sha256": "76e5e07252e70f39b6b07798369a5f29a470ed1d87caec323a5300f43e112b77",
        "members": (
            "main.usdc",
            "SubUSDs/textures/extracted_image_1.jpg_roughness.png",
            "SubUSDs/textures/extracted_image_2.jpg",
            "SubUSDs/textures/extracted_image_1.jpg_metallic.png",
            "SubUSDs/textures/extracted_image_0.jpg",
            "SubUSDs/textures/extracted_image_3.jpg",
        ),
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

    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    expected = set(spec["members"])
    with zipfile.ZipFile(archive) as package:
        packaged = {item.filename for item in package.infolist() if not item.is_dir()}
        if packaged != expected:
            raise ValueError(
                f"unexpected members in {archive.name}: expected {sorted(expected)}, "
                f"got {sorted(packaged)}"
            )
        for member in spec["members"]:
            info = package.getinfo(member)
            if info.file_size >= GITHUB_FILE_LIMIT:
                raise ValueError(f"{archive.name}:{member} exceeds GitHub's 100 MiB limit")
            dst = target / member
            dst.parent.mkdir(parents=True, exist_ok=True)
            with package.open(info) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            print(f"[vendor] {name}/{member} ({info.file_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="directory containing egg_holder.usdz, egg.usdz, and egg_basket.usdz",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = repo / "robobench" / "suites" / "packing" / "assets" / "egg_carton"
    for name, spec in ASSETS.items():
        vendor_one(args.source_dir.resolve(), output, name, spec)
    print(f"[vendor] complete: {output}")


if __name__ == "__main__":
    main()
