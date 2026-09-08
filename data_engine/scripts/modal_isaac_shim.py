#!/usr/bin/env python3
"""modal_isaac_shim — an `--isaac-py` whose GPU lives on Modal.

The orchestrator invokes its Isaac interpreter as

    <isaac_py> data_engine/scripts/<generate|render>.py <gen_root> <flags...>

For tasks whose simulator stack exists only on Modal (Newton cloth/liquid), point
`--isaac-py` at THIS script: it syncs the campaign's inputs to the `cosigen-dgen-io`
volume, calls the deployed `cosigen-dgen::dgen_exec` function with the argv
rewritten to the remote layout, pulls the produced outputs back into the local
campaign, and exits with the REMOTE return code — a failed remote batch fails the
local invocation exactly like a local Isaac would (rc swallowing once let a repair
agent "fix" a healthy solve over a shim bug).

    python3 data_engine/scripts/modal_isaac_shim.py <script.py> <gen_root> ...

Prereqs: `modal deploy data_engine/scripts/modal_dgen_app.py`, a Modal profile with
access to the `cosigen-newton` env volume (MODAL_PROFILE selects it).

What syncs, and why it is small:
  up    the campaign control plane (gen.yaml + scenes/, follows the asset
        symlinks) every call — sessions edit scenes between calls;
        for render: the episode inputs (traj.npz + meta.json) not yet uploaded
        (immutable once written, so content never re-uploads)
  down  generate: the new data/<batch>/ directory
        render: each requested episode's imgs/ (and the episode metas render
        may have touched)
Concurrent invocations are safe: batches own disjoint dirs, episode inputs are
immutable, and the conductor refreshes derived metas locally after each wave.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

IO_VOLUME = "cosigen-dgen-io"
APP, FUNC = "cosigen-dgen", "dgen_exec"
REMOTE_IO = "/dgio"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(map(str, cmd)), text=True, **kw)


def volume_put(local: Path, remote: str) -> None:
    p = sh(["modal", "volume", "put", IO_VOLUME, local, remote, "--force"],
           capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"volume put {local} -> {remote} failed:\n{p.stdout}{p.stderr}")


def volume_get(remote: str, local_parent: Path) -> bool:
    """Download `remote` under `local_parent`. modal volume get creates the LEAF
    itself, so the destination is always the PARENT dir (measured 2026-08-28:
    passing the leaf path makes it nest one level deeper)."""
    local_parent.mkdir(parents=True, exist_ok=True)
    p = sh(["modal", "volume", "get", IO_VOLUME, remote, str(local_parent), "--force"],
           capture_output=True)
    return p.returncode == 0


def upload_control_plane(gen_root: Path, key: str) -> None:
    """gen.yaml + scenes/ (symlinks resolved: remote has no repo to point into).
    Staged into one tar-shaped temp tree and pushed in a single put."""
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "cp"
        (stage / "data").mkdir(parents=True)
        sh(["cp", gen_root / "gen.yaml", stage / "gen.yaml"], check=True)
        sh(["rsync", "-aL", "--exclude", "__pycache__", "--exclude", ".agent",
            f"{gen_root}/scenes/", f"{stage}/scenes/"], check=True)
        volume_put(stage, f"/{key}")


def upload_episodes(gen_root: Path, key: str, episodes: list[Path],
                    manifest: Path) -> None:
    """Episode inputs for render: traj.npz + meta.json, uploaded once each
    (immutable after generation writes them). A local manifest remembers what is
    already up, so repeated shards cost nothing."""
    done = set(json.loads(manifest.read_text())) if manifest.is_file() else set()
    todo = []
    for ep in episodes:
        rel = str(ep.resolve().relative_to(gen_root.resolve()))
        if rel not in done:
            todo.append((ep, rel))
    if not todo:
        return
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "eps"
        for ep, rel in todo:
            dst = stage / rel
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("traj.npz", "meta.json"):
                if (ep / name).is_file():
                    sh(["cp", ep / name, dst / name], check=True)
        volume_put(stage, f"/{key}")
    done |= {rel for _, rel in todo}
    manifest.write_text(json.dumps(sorted(done)))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("script")
    args, passthrough = ap.parse_known_args()
    script = Path(args.script).name
    if script not in ("generate.py", "render.py"):
        raise SystemExit(f"modal_isaac_shim forwards generate.py/render.py, "
                         f"not {script!r}")
    if not passthrough:
        raise SystemExit("expected: <script.py> <gen_root> [flags...]")
    gen_root = Path(passthrough[0]).resolve()
    key = f"{gen_root.parents[2].name}__{gen_root.name}"   # <run>__<gen_name>

    # rewrite every local campaign path in argv to the remote layout
    remote_root = f"{REMOTE_IO}/{key}"
    argv = [remote_root]
    episodes: list[Path] = []
    it = iter(passthrough[1:])
    for tok in it:
        if tok == "--episodes":
            argv.append(tok)
            for ep in it:
                if ep.startswith("--"):
                    tok = ep
                    break
                episodes.append(Path(ep))
                rel = Path(ep).resolve().relative_to(gen_root)
                argv.append(f"{remote_root}/{rel}")
            else:
                break
            argv.append(tok)
            continue
        argv.append(tok.replace(str(gen_root), remote_root))

    upload_control_plane(gen_root, key)
    if episodes:
        upload_episodes(gen_root, key, episodes,
                        gen_root / ".modal_shim_uploaded.json")

    import modal

    fn = modal.Function.from_name(APP, FUNC)
    print(f"[shim] dgen_exec {script} {' '.join(map(shlex.quote, argv))}", flush=True)
    out = fn.remote(script, argv)
    rc = int(out.get("returncode", 1))
    if rc != 0:
        print(f"[shim] REMOTE FAILED rc={rc}; last output:\n{out.get('tail', '')}",
              flush=True)

    # pull the produced outputs back into the local campaign
    if script == "generate.py":
        batch = None
        for i, tok in enumerate(passthrough):
            if tok == "--batch" and i + 1 < len(passthrough):
                batch = passthrough[i + 1]
        if batch:
            got = volume_get(f"/{key}/data/{batch}", gen_root / "data")
            print(f"[shim] pulled data/{batch}: {'ok' if got else 'NOTHING'}",
                  flush=True)
        else:
            # unnamed batch (timestamped remotely): pull the whole data/ delta
            got = volume_get(f"/{key}/data", gen_root)
            print(f"[shim] pulled data/ (unnamed batch): {'ok' if got else 'NOTHING'}",
                  flush=True)
    else:
        for ep in episodes:
            rel = ep.resolve().relative_to(gen_root)
            volume_get(f"/{key}/{rel}/imgs", gen_root / rel)
            volume_get(f"/{key}/{rel}/meta.json", gen_root / rel)
    return rc


if __name__ == "__main__":
    sys.exit(main())
