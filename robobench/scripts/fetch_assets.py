"""Fetch the large binary assets (USD, textures, policies, reference videos) from Hugging Face.

The git repo keeps only code and small text layers (`*.usda`, `*.mdl`, provenance json/md). Every
`*.usd`/`*.usdc`, texture, policy and reference video under `robobench/**/assets/` (and the deformable
`videos/`) lives in the dataset repo `CoSiGen/robobench-assets`, mirrored at the
SAME relative path, so after a fetch the tree is byte-identical to a checkout that vendored them.

`robobench/assets_manifest.json` is the contract: path -> {sha256, bytes} for every remote file.
Fetching downloads what is missing or mismatched and verifies the result against the manifest.

    python -m robobench.scripts.fetch_assets            # download missing/mismatched, then verify
    python -m robobench.scripts.fetch_assets --check    # verify only (exit 1 on any mismatch)

Maintainers (after adding/changing binary assets in the working tree):

    python -m robobench.scripts.fetch_assets --update-manifest   # rescan the tree, rewrite the manifest
    python -m robobench.scripts.fetch_assets --upload            # push the manifest's files to HF

Both `--upload` and private-repo fetches need `hf auth login` (or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import robobench

REPO_ID = "CoSiGen/robobench-assets"
REPO_TYPE = "dataset"
PKG_DIR = Path(robobench.__file__).resolve().parent
ROOT = PKG_DIR.parent  # repo root for editable installs; site-packages for wheels — paths start with robobench/
MANIFEST = PKG_DIR / "assets_manifest.json"

# What counts as a remote asset: extension decides — one of these under any `assets/` (or `videos/`) dir.
BINARY_EXT = {".usd", ".usdc", ".png", ".jpg", ".jpeg", ".mp4", ".pt", ".exr"}
ASSET_DIRS = ("assets", "videos")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_id(path: Path) -> str:
    """git's blob id: sha1 over "blob <size>\\0" + content (what the hub reports for non-LFS files)."""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_remote_asset(rel: str) -> bool:
    p = Path(rel)
    return p.suffix.lower() in BINARY_EXT and any(d in p.parts[:-1] for d in ASSET_DIRS)


def load_manifest() -> dict[str, dict]:
    with open(MANIFEST) as f:
        return json.load(f)


def verify(manifest: dict[str, dict], fast: bool = False) -> list[str]:
    """Return the manifest paths that are missing or differ (size, or sha256 unless `fast`)."""
    bad = []
    for rel, meta in manifest.items():
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size != meta["bytes"]:
            bad.append(rel)
        elif not fast and _sha256(p) != meta["sha256"]:
            bad.append(rel)
    return bad


def fetch(revision: str | None) -> int:
    from huggingface_hub import snapshot_download

    manifest = load_manifest()
    todo = verify(manifest, fast=True)
    if not todo:
        print(f"all {len(manifest)} assets present (size check); verifying hashes ...")
    else:
        mb = sum(manifest[r]["bytes"] for r in todo) / 2**20
        print(f"fetching {len(todo)} / {len(manifest)} assets ({mb:.0f} MB) from {REPO_ID} into {ROOT}")
        snapshot_download(REPO_ID, repo_type=REPO_TYPE, revision=revision, local_dir=str(ROOT),
                          allow_patterns=todo)
    bad = verify(manifest)
    if bad:
        print(f"FAILED: {len(bad)} assets still missing or mismatched, e.g. {bad[:5]}", file=sys.stderr)
        return 1
    print(f"OK: {len(manifest)} assets match the manifest")
    return 0


def check() -> int:
    manifest = load_manifest()
    bad = verify(manifest)
    for rel in bad:
        print("MISMATCH " + rel)
    n = len(manifest)
    print(f"FAILED: {len(bad)} mismatched of {n}" if bad else f"OK: {n}/{n} assets match the manifest")
    return 1 if bad else 0


def update_manifest() -> int:
    manifest = {}
    for dirpath, dirnames, filenames in os.walk(PKG_DIR):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            if _is_remote_asset(rel):
                p = ROOT / rel
                manifest[rel] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(manifest)} files, "
          f"{sum(m['bytes'] for m in manifest.values()) / 2**20:.0f} MB")
    return 0


def upload(batch: int) -> int:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True)
    manifest = load_manifest()
    # Skip files already on the hub with identical content: LFS files expose their sha256, small
    # (non-LFS) files expose a git blob id, so compare whichever the hub gives us.
    info = {f.path: f for f in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True)
            if hasattr(f, "size")}
    remote = set(info)
    todo = []
    for rel, meta in sorted(manifest.items()):
        f = info.get(rel)
        if f is not None:
            lfs_sha = getattr(getattr(f, "lfs", None), "sha256", None)
            same = lfs_sha == meta["sha256"] if lfs_sha else f.blob_id == _git_blob_id(ROOT / rel)
            if same:
                continue
        todo.append(rel)
    print(f"{len(manifest) - len(todo)} already on the hub, uploading {len(todo)} in batches of {batch}")
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(ROOT / rel)) for rel in chunk]
        api.create_commit(REPO_ID, repo_type=REPO_TYPE, operations=ops,
                          commit_message=f"assets batch {i // batch + 1}: {chunk[0]} .. {chunk[-1]}")
        print(f"  committed {i + len(chunk)} / {len(todo)}", flush=True)
    stale = sorted(remote - set(manifest))
    if stale:
        print(f"note: {len(stale)} files on the hub are not in the manifest (left in place): {stale[:5]}")
    print("upload done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="verify the local tree against the manifest")
    g.add_argument("--update-manifest", action="store_true", help="(maintainer) rescan the tree")
    g.add_argument("--upload", action="store_true", help="(maintainer) push manifest files to the hub")
    ap.add_argument("--revision", default=None, help="hub revision to fetch (default: main)")
    ap.add_argument("--batch", type=int, default=100, help="files per upload commit")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.update_manifest:
        return update_manifest()
    if a.upload:
        return upload(a.batch)
    return fetch(a.revision)


if __name__ == "__main__":
    sys.exit(main())
