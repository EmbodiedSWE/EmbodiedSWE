"""Boot the REGISTERED preset from the extracted tree — and harvest describe().

One headless sim launch proves the code closure AND the asset closure (a
missing USD only surfaces at env-build time), validates the actual baked
preset the agent will face, and captures a sectioned markdown description
(scene / robot, from the live objects) for the prompt.
Principle: no bundle ships unbooted.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_PY = REPO / ".venv" / "bin" / "python"

_D0, _D1 = "<<DESCRIBE>>", "<<END_DESCRIBE>>"
_MISSING = "<<MISSING_ASSETS>>"


class MissingAssets(Exception):
    """The stage composed against USD files that are not in the extracted tree."""

    def __init__(self, paths: list[str]):
        super().__init__("unresolved USD references:\n  " + "\n  ".join(paths))
        self.paths = paths


def boot_preset(tree: Path, preset: str, seed: int = 0) -> str:
    """Build + reset the registered env from `tree`; return a sectioned
    markdown description (scene / robot) from the live objects.

    A build failure is inspected before it is reported: USD only warns when a
    reference does not resolve, and the real error arrives later as something
    unrecognisable (a leg with no rigid bodies, say), so we ask the composed
    stage which of its references are missing and raise MissingAssets with them.
    """
    code = f"""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os
import robobench
assert robobench.__file__.startswith('{tree}'), 'wrong robobench: ' + robobench.__file__
robobench.discover()
from robobench.core.registries import ENVS


def unresolved():
    \"\"\"Asset paths this stage references and does not have.\"\"\"
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []
    bad = set()
    for prim in stage.TraverseAll():
        for spec in prim.GetPrimStack():
            arcs = (list(spec.referenceList.GetAddedOrExplicitItems())
                    + list(spec.payloadList.GetAddedOrExplicitItems()))
            for arc in arcs:
                if not arc.assetPath:
                    continue
                target = spec.layer.ComputeAbsolutePath(arc.assetPath)
                if not os.path.exists(target):
                    bad.add(target)
    return sorted(bad)


try:
    env = ENVS.get('{preset}')().build(num_envs=1, seed={seed})
    env.reset(seed={seed})
except BaseException as exc:
    import json, traceback
    traceback.print_exc()
    try:
        print('{_MISSING}' + json.dumps(unresolved()), flush=True)
    except BaseException as scan_exc:
        print('could not scan the stage for missing assets: %r' % (scan_exc,), flush=True)
    print('BOOT_FAILED: %s: %s' % (type(exc).__name__, exc), flush=True)
    os._exit(1)
print('{_D0}')
print('## Scene'); print()
print(env.scene.describe()); print()
print('## Robot'); print()
print(env.robot.describe())
print('{_D1}')
print('BOOT_OK', flush=True)
os._exit(0)
"""
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    print(f"[validate] booting registered preset '{preset}' from the extracted tree ...")
    # A FILE under /tmp, not `python -c`: isaacsim 6 (Newton)'s AppLauncher parses sys.argv for
    # Kit args, and a bare "-c" argv[0] is misread as a Kit arg and segfaults Kit at startup
    # (measured on Modal L4, 2026-08-15). Writing it under /tmp (not the tree) keeps the tree
    # byte-identical to its hash, and sys.path[0] becomes /tmp (no robobench there), so
    # PYTHONPATH=tree still resolves the extracted robobench — same isolation as before.
    boot_file = Path(tempfile.mkdtemp()) / "_boot_validate.py"
    boot_file.write_text(code)
    # Inherit the parent env (the GPU driver mount's LD_LIBRARY_PATH, which Kit needs on Modal),
    # then override PYTHONPATH to the extracted tree so its robobench — not the repo's — is used.
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(tree), "HOME": str(Path.home()),
        "PYTHONDONTWRITEBYTECODE": "1",  # keep the validated tree byte-identical to its hash
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        # Isaac's breakpad crash-handler segfaults in a restricted container (cannot fork to
        # write a dump), killing Kit at startup; disabling it lets Kit boot.
        "OMNI_KIT_CRASH_REPORTER": "0", "CARB_CRASHREPORTER_ENABLED": "0",
        "OMNI_KIT_ALLOW_ROOT": "1",
    })
    r = subprocess.run(
        [str(py), str(boot_file)],
        cwd=str(tree),  # cwd MUST NOT contain the real robobench (sys.path shadows PYTHONPATH)
        env=env,
        # 2400, not 600: a first boot on a cold kit cache COOKS the scene's SDF collision
        # meshes (the pc slot channels take the longest), and 600 s killed exactly those
        # builds (pc_gpu / pc_gpu_ram, 2026-08-14) while every warm-cache boot passes in
        # ~60 s. The timeout still exists so a genuinely hung boot cannot stall a build
        # pipeline forever.
        capture_output=True, text=True, timeout=2400,
    )
    if "BOOT_OK" not in r.stdout:
        missing = []
        for line in r.stdout.splitlines():
            if line.startswith(_MISSING):
                missing = json.loads(line[len(_MISSING):])
        # Only a reference the extractor could actually supply counts as a missing asset. A
        # composed stage also lists references it can never resolve locally — every scene using
        # a stock ground plane names a remote Nucleus URL — and reporting those as "missing"
        # made MissingAssets the verdict for ANY boot failure in such a scene, so the real error
        # was replaced by an unsuppliable URL and never printed at all.
        remote = [p for p in missing if not str(p).startswith(str(tree))]
        missing = [p for p in missing if str(p).startswith(str(tree))]
        if remote and not missing:
            print(f"[validate] references that do not resolve locally and are not the tree's to "
                  f"supply (not the failure unless the error below says so): {', '.join(remote)}")
        # A scene that checks its own assets raises FileNotFoundError before the stage is
        # composed, so `unresolved()` never sees it — the pc scenes reach their directory
        # through a cfg field rather than a literal `assets / "name"`, which is what the
        # extractor's heuristic reads. Any path inside the tree that the failure names and
        # that is not on disk is a missing asset too, whichever way it was reported.
        named = re.findall(rf"{re.escape(str(tree))}[^\s'\"]+", r.stdout + r.stderr)
        missing += [p for p in dict.fromkeys(named) if not Path(p).exists()]
        if missing:
            raise MissingAssets(missing)
        errs = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "rror" in l)[-2000:]
        raise SystemExit(f"BOOT CHECK FAILED for '{preset}' (missing asset? bad preset?):\n{errs}")
    describe = r.stdout.split(_D0, 1)[1].split(_D1, 1)[0].strip()
    print(f"[validate] BOOT_OK — env built + reset; describe() harvested ({len(describe)} chars)")
    return describe
