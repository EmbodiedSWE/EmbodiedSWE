"""CoSiGen pod-side configuration: feature flags, function-gating sets, sibling
hot-reload machinery, and the answer-leakage guard. Imported first by every other
cosigen_* module (see cosigen_loop, the aggregator).\n\nNOTE: like the rest of the
pod-side stack this imports only stdlib — safe to import before Isaac boots."""
from __future__ import annotations

import os
import subprocess

_RELOAD_HDFS_DIR = os.environ.get("CAPX_RELOAD_DIR",
                                  "hdfs://haruna/tmp/zeyu.shen/cosigen_harness")


def _import_sibling(module: str):
    """Import a module that lives next to this file, fetching it from the reload dir if
    the pod predates it. A running render server only re-fetches the file list it was
    booted with, so a NEW harness module would otherwise require a pod restart."""
    import importlib

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["hdfs", "dfs", "-get", "-f",
                        f"{_RELOAD_HDFS_DIR}/{module}.py", here],
                       capture_output=True, check=False)
        importlib.invalidate_caches()
        return importlib.import_module(module)


def _refresh_sibling_on_reload(module: str) -> None:
    """cosigen_loop is re-executed on every /reload, but a sibling module imported via
    _import_sibling stays cached in sys.modules with whatever code it had — a running
    server only re-fetches the file list it booted with. Refreshing here gives siblings
    the same hot-reload behavior as this file (stale cosigen_opt on a live pod, 2026-07-25)."""
    import importlib
    import sys as _sys

    if module in _sys.modules:
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["hdfs", "dfs", "-get", "-f",
                        f"{_RELOAD_HDFS_DIR}/{module}.py", here],
                       capture_output=True, check=False)
        importlib.reload(_sys.modules[module])


_refresh_sibling_on_reload("cosigen_opt")




# Ablation baselines: CAPX_DISABLE_FEATURES="checkpoint,opt" removes the checkpoint-tree
# and/or optimize harness from the agent-visible prompt AND namespace (server-side
# turn bookkeeping keeps using the tree internally; the agent just can't drive it).
# (The retired RL sub-policy engine lives as inert history in eval/legacy/.)
_FEATURES_OFF = set(filter(None, os.environ.get("CAPX_DISABLE_FEATURES", "").split(",")))

# ANSWER-LEAKAGE GUARD (user directive; violated once on 2026-07-24 at great cost):
# the ikea reference solution must NEVER be readable by an agent under evaluation.
# The tarball excludes it, but pods extracted before the fix still carry a stale copy —
# sweep it on every import (module boot AND every /reload) so pods self-clean.
def _purge_reference_solutions() -> None:
    import glob as _glob
    for pat in ("/home/tiger/CoSiGen/**/solve_ikea*", "/home/tiger/CoSiGen/**/*solve_ikea*"):
        for path in _glob.glob(pat, recursive=True):
            try:
                os.remove(path)
                print(f"[leakage-guard] removed {path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[leakage-guard] FAILED to remove {path}: {exc!r}", flush=True)


_purge_reference_solutions()
_CKPT_FNS = {"checkpoint", "goto", "list_checkpoints", "get_checkpoint_code",
             "get_checkpoint_scene", "get_checkpoint_log", "render_checkpoint"}
# the parameter-search surface (ablation arm "opt"); `optimize` itself is a driver
# tool — this only gates the in-namespace redirect stub
_OPT_FNS = {"optimize"}

