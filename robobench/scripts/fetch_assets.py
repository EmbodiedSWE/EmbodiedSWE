"""Fetch the large binary assets (USD, textures, policies, reference videos) from Hugging Face.

The git repo keeps code and task text layers (`*.usda`, `*.mdl`, provenance json/md). Every
`*.usd`/`*.usdc`, texture, policy and reference video under `robobench/**/assets/` (and the deformable
`videos/`) lives in the dataset repo `CoSiGen/robobench-assets`, mirrored at the SAME relative path, so
after a fetch the tree is byte-identical to a checkout that vendored them. Prepared backdrop groups
also keep their text layers on the hub so each room is a complete, independently downloadable tree.

`robobench/assets_manifest.json` is the contract:
  {"revision": hub_commit, "files": {path: {sha256, bytes}},
   "bundles": {asset_root: {path, sha256, bytes, files}}}
Every remote file is listed individually (incremental fetches download just what changed). Each asset
root (e.g. `robobench/suites/cutting/assets`) is ALSO published as one tar bundle, so a fresh checkout
costs ~8 downloads instead of ~800 (the hub rate-limits API requests per 5 minutes).

    python -m robobench.scripts.fetch_assets            # download missing/mismatched, then verify
    python -m robobench.scripts.fetch_assets --check    # verify only (exit 1 on any mismatch)

Maintainers (after adding/changing binary assets in the working tree):

    python -m robobench.scripts.fetch_assets --update-manifest   # rescan the tree, rebuild bundles + manifest
    python -m robobench.scripts.fetch_assets --upload            # push changed files + bundles to the hub

Both `--upload` and private-repo fetches need `hf auth login` (or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import tempfile
from filelock import FileLock
from pathlib import Path

import robobench

REPO_ID = "CoSiGen/robobench-assets"
REPO_TYPE = "dataset"
PKG_DIR = Path(robobench.__file__).resolve().parent
ROOT = Path(os.environ.get("COSIGEN_ASSET_DIR", str(PKG_DIR.parent))).expanduser().resolve()
MANIFEST = PKG_DIR / "assets_manifest.json"
BUNDLE_DIR = ROOT / ".cache" / "robobench_bundles"  # scratch for --update-manifest / --upload / fetch

# What counts as a remote asset: extension decides — one of these under any `assets/` (or `videos/`) dir.
BINARY_EXT = {".usd", ".usdc", ".png", ".jpg", ".jpeg", ".mp4", ".pt", ".exr", ".hdr", ".tif", ".tiff", ".dds", ".ktx"}
ASSET_DIRS = ("assets", "videos")
# Use the bundle for an asset root when at least this many of its files are missing/stale.
BUNDLE_MIN_FILES = 16
RATE_LIMIT_WAIT_S = 60
RATE_LIMIT_TRIES = 8


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
    if rel.startswith("robobench/backdrops/assets/"):
        return not any(part.startswith(".") for part in p.parts) and p.suffix != ".py"
    return p.suffix.lower() in BINARY_EXT and any(d in p.parts[:-1] for d in ASSET_DIRS)


def _asset_root(rel: str) -> str:
    """`robobench/suites/cutting/assets/banana/piece_0.usd` -> `robobench/suites/cutting/assets`."""
    parts = Path(rel).parts
    i = next(k for k, part in enumerate(parts[:-1]) if part in ASSET_DIRS)
    return "/".join(parts[: i + (2 if rel.startswith("robobench/backdrops/assets/") else 1)])


def _bundle_name(root: str) -> str:
    return "bundles/" + root.replace("/", "__") + ".tar"


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        m = json.load(f)
    if "files" not in m:  # first (flat) manifest format
        m = {"files": m, "bundles": {}}
    m.setdefault("bundles", {})
    for rel in m["files"]:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "robobench":
            raise ValueError(f"Unsafe asset path in manifest: {rel}")
    return m


def verify(files: dict[str, dict], fast: bool = False) -> list[str]:
    """Return the manifest paths that are missing or differ (size, or sha256 unless `fast`)."""
    bad = []
    for rel, meta in files.items():
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size != meta["bytes"]:
            bad.append(rel)
        elif not fast and _sha256(p) != meta["sha256"]:
            bad.append(rel)
    return bad


def _is_rate_limit(exc: BaseException) -> bool:
    return "429" in str(exc) or "Too Many Requests" in str(exc) or "rate limit" in str(exc).lower()


def _with_rate_limit_retry(fn, what: str):
    for attempt in range(RATE_LIMIT_TRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — hub errors come in several classes
            if not _is_rate_limit(exc) or attempt == RATE_LIMIT_TRIES - 1:
                raise
            print(f"hub rate limit hit while {what} (1000 API requests / 5 min); "
                  f"waiting {RATE_LIMIT_WAIT_S}s (retry {attempt + 1}/{RATE_LIMIT_TRIES - 1}) ...", flush=True)
            time.sleep(RATE_LIMIT_WAIT_S)


def _atomic_copy(src, rel: str, meta: dict) -> None:
    """Verify a temporary sibling before publishing it; interrupted writes never look ready."""
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dst.name + ".", suffix=".partial", dir=dst.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        temporary = Path(tmp)
        if temporary.stat().st_size != meta["bytes"] or _sha256(temporary) != meta["sha256"]:
            raise ValueError(f"Downloaded asset checksum mismatch: {rel}")
        os.replace(tmp, dst)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _extract_bundle(tar_path: Path, files: dict[str, dict], wanted: set[str]) -> int:
    n = 0
    with tarfile.open(tar_path, "r:") as tar:
        for member in tar:
            if member.name not in wanted or not member.isfile():
                continue
            src = tar.extractfile(member)
            assert src is not None
            with src:
                _atomic_copy(src, member.name, files[member.name])
            n += 1
    return n


_VERIFIED = set()


def _missing(files):
    bad = []
    for rel, meta in files.items():
        p = ROOT / rel
        if not p.is_file():
            bad.append(rel)
            continue
        st = p.stat()
        key = (str(p), meta["sha256"], st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if key not in _VERIFIED:
            if st.st_size != meta["bytes"] or _sha256(p) != meta["sha256"]:
                bad.append(rel)
            else:
                _VERIFIED.add(key)
    return bad


def fetch(revision: str | None = None, prefixes=None) -> int:
    """Fetch selected groups at the manifest revision; existing verified files work offline."""
    from huggingface_hub import hf_hub_download

    man = load_manifest()
    revision = revision or man.get("revision")
    files = {p: m for p, m in man["files"].items()
             if prefixes is None or any(p == prefix or p.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)}
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with FileLock(str(BUNDLE_DIR / "fetch.lock")):
        todo = _missing(files)
        if not todo:
            return 0
        if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}:
            raise FileNotFoundError(f"Offline: {len(todo)} required assets missing or mismatched; first: {todo[0]}. "
                                    "Run python -m robobench.scripts.fetch_assets with network access first.")
        print(f"Fetching {len(todo)} assets from {REPO_ID}@{revision or 'main'}", flush=True)
        by_root = {}
        for rel in todo:
            by_root.setdefault(_asset_root(rel), []).append(rel)
        try:
            for root, rels in sorted(by_root.items()):
                bundle = man["bundles"].get(root)
                # Use whole-root bundles only when the selected group includes the entire root.
                entire_root = all(p in files for p in man["files"] if _asset_root(p) == root)
                if bundle is None or len(rels) < BUNDLE_MIN_FILES or not entire_root:
                    continue
                local = Path(_with_rate_limit_retry(
                    lambda: hf_hub_download(REPO_ID, bundle["path"], repo_type=REPO_TYPE, revision=revision),
                    f"downloading {bundle['path']}"))
                if _sha256(local) != bundle["sha256"]:
                    raise ValueError(f"Bundle checksum mismatch: {bundle['path']}")
                _extract_bundle(local, files, set(rels))
            for rel in _missing(files):
                local = _with_rate_limit_retry(
                    lambda: hf_hub_download(REPO_ID, rel, repo_type=REPO_TYPE, revision=revision),
                    f"downloading {rel}")
                with open(local, "rb") as src:
                    _atomic_copy(src, rel, files[rel])
        except Exception as exc:
            raise RuntimeError(f"Could not fetch assets from {REPO_ID}@{revision or 'main'}: {exc}. "
                               "Check network access; private datasets require HF_TOKEN or hf auth login.") from exc
        bad = _missing(files)
        if bad:
            raise RuntimeError(f"Assets still missing or mismatched: {bad[:5]}")
    print(f"Verified {len(files)} required assets", flush=True)
    return 0


def check() -> int:
    files = load_manifest()["files"]
    bad = verify(files)
    for rel in bad:
        print("MISMATCH " + rel)
    n = len(files)
    print(f"FAILED: {len(bad)} mismatched of {n}" if bad else f"OK: {n}/{n} assets match the manifest")
    return 1 if bad else 0


def _build_bundle(root: str, rels: list[str]) -> Path:
    """Deterministic tar (sorted members, zeroed mtime/owner) so an unchanged root re-hashes identically."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    out = BUNDLE_DIR / Path(_bundle_name(root)).name
    with tarfile.open(out, "w:", format=tarfile.PAX_FORMAT) as tar:
        for rel in sorted(rels):
            p = ROOT / rel
            info = tar.gettarinfo(str(p), arcname=rel)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(p, "rb") as f:
                tar.addfile(info, f)
    return out


def update_manifest() -> int:
    previous = load_manifest()
    files: dict[str, dict] = dict(previous["files"])
    for dirpath, dirnames, filenames in os.walk(ROOT / "robobench"):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            if _is_remote_asset(rel):
                p = ROOT / rel
                files[rel] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    by_root: dict[str, list[str]] = {}
    for rel in files:
        by_root.setdefault(_asset_root(rel), []).append(rel)
    bundles = {}
    for root, rels in sorted(by_root.items()):
        if all(files[r] == previous["files"].get(r) for r in rels) and root in previous["bundles"]:
            bundles[root] = previous["bundles"][root]
            continue
        absent = [r for r in rels if not (ROOT / r).is_file()]
        if absent:
            raise FileNotFoundError(f"Cannot rebuild {root}; fetch its existing assets first: {absent[:3]}")
        out = _build_bundle(root, rels)
        bundles[root] = {"path": _bundle_name(root), "sha256": _sha256(out), "bytes": out.stat().st_size,
                         "files": len(rels)}
        print(f"  bundle {bundles[root]['path']}: {len(rels)} files, {out.stat().st_size / 2**20:.0f} MB")
    with open(MANIFEST, "w") as f:
        json.dump({**previous, "files": dict(sorted(files.items())), "bundles": bundles}, f, indent=1)
        f.write("\n")
    print(f"wrote {MANIFEST}: {len(files)} files, "
          f"{sum(m['bytes'] for m in files.values()) / 2**20:.0f} MB, {len(bundles)} bundles "
          f"(tars staged in {BUNDLE_DIR.relative_to(ROOT)} for --upload)")
    return 0


def upload(batch: int) -> int:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.repo_info(REPO_ID, repo_type=REPO_TYPE)  # publish into the existing dataset without changing visibility
    man = load_manifest()
    files, bundles = man["files"], man["bundles"]
    # Skip files already on the hub with identical content: LFS files expose their sha256, small
    # (non-LFS) files expose a git blob id, so compare whichever the hub gives us.
    info = {f.path: f for f in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True)
            if hasattr(f, "size")}

    def _same(rel_in_repo: str, local: Path, sha256: str) -> bool:
        f = info.get(rel_in_repo)
        if f is None:
            return False
        lfs_sha = getattr(getattr(f, "lfs", None), "sha256", None)
        return lfs_sha == sha256 if lfs_sha else local.is_file() and f.blob_id == _git_blob_id(local)

    todo = [rel for rel, meta in sorted(files.items()) if not _same(rel, ROOT / rel, meta["sha256"])]
    print(f"{len(files) - len(todo)} files already on the hub, uploading {len(todo)} in batches of {batch}")
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(ROOT / rel)) for rel in chunk]
        api.create_commit(REPO_ID, repo_type=REPO_TYPE, operations=ops,
                          commit_message=f"assets batch {i // batch + 1}: {chunk[0]} .. {chunk[-1]}")
        print(f"  committed {i + len(chunk)} / {len(todo)}", flush=True)
    # Preserve matching remote bundles even when their original archive is not staged locally.
    # Older tar writers can produce different archive bytes for identical asset contents.
    ops = []
    for root, b in sorted(bundles.items()):
        local = BUNDLE_DIR / Path(b["path"]).name
        if _same(b["path"], local, b["sha256"]):
            continue
        if not local.is_file() or _sha256(local) != b["sha256"]:
            local = _build_bundle(root, [r for r in files if _asset_root(r) == root])
            if _sha256(local) != b["sha256"]:
                print(f"ERROR: rebuilt {b['path']} does not match the manifest — run --update-manifest first",
                      file=sys.stderr)
                return 1
        if _same(b["path"], local, b["sha256"]):
            continue
        ops.append(CommitOperationAdd(path_in_repo=b["path"], path_or_fileobj=str(local)))
    if ops:
        api.create_commit(REPO_ID, repo_type=REPO_TYPE, operations=ops,
                          commit_message=f"bundles: {', '.join(Path(o.path_in_repo).name for o in ops)}")
    print(f"{len(bundles) - len(ops)} bundles already on the hub, uploaded {len(ops)}")
    stale = sorted(set(info) - set(files) - {b["path"] for b in bundles.values()} - {".gitattributes"})
    if stale:
        print(f"note: {len(stale)} files on the hub are not in the manifest (left in place): {stale[:5]}")
    man["revision"] = api.repo_info(REPO_ID, repo_type=REPO_TYPE).sha
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    print(f"upload done; manifest pinned to {man['revision']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="verify the local tree against the manifest")
    g.add_argument("--update-manifest", action="store_true", help="(maintainer) rescan the tree, rebuild bundles")
    g.add_argument("--upload", action="store_true", help="(maintainer) push manifest files + bundles to the hub")
    ap.add_argument("--revision", default=None, help="hub revision to fetch (default: manifest revision, otherwise main)")
    ap.add_argument("--prefix", action="append", help="fetch only this manifest path prefix (repeatable)")
    ap.add_argument("--batch", type=int, default=100, help="files per upload commit")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.update_manifest:
        return update_manifest()
    if a.upload:
        return upload(a.batch)
    return fetch(a.revision, prefixes=a.prefix)


if __name__ == "__main__":
    sys.exit(main())
