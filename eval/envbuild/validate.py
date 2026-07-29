"""Boot the REGISTERED preset from the extracted tree — and harvest describe().

One headless sim launch proves the code closure AND the asset closure (a
missing USD only surfaces at env-build time), validates the actual baked
preset the agent will face, and captures a sectioned markdown description
(scene / robot, from the live objects) for the prompt.
Principle: no bundle ships unbooted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_PY = REPO / ".venv" / "bin" / "python"

_D0, _D1 = "<<DESCRIBE>>", "<<END_DESCRIBE>>"


def boot_preset(tree: Path, preset: str) -> str:
    """Build + reset the registered env from `tree`; return a sectioned
    markdown description (scene / robot) from the live objects."""
    code = f"""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import robobench
assert robobench.__file__.startswith('{tree}'), 'wrong robobench: ' + robobench.__file__
robobench.discover()
from robobench.core.registries import ENVS
env = ENVS.get('{preset}')().build(num_envs=1)
env.reset()
print('{_D0}')
print('## Scene'); print()
print(env.scene.describe()); print()
print('## Robot'); print()
print(env.robot.describe())
print('{_D1}')
print('BOOT_OK', flush=True)
import os
os._exit(0)
"""
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    print(f"[validate] booting registered preset '{preset}' from the extracted tree ...")
    r = subprocess.run(
        [str(py), "-c", code],
        cwd=str(tree),  # cwd MUST NOT contain the real robobench (sys.path shadows PYTHONPATH)
        env={
            "PYTHONPATH": str(tree), "HOME": str(Path.home()), "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",  # keep the validated tree byte-identical to its hash
            "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y",
        },
        capture_output=True, text=True, timeout=600,
    )
    if "BOOT_OK" not in r.stdout:
        errs = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "rror" in l)[-2000:]
        raise SystemExit(f"BOOT CHECK FAILED for '{preset}' (missing asset? bad preset?):\n{errs}")
    describe = r.stdout.split(_D0, 1)[1].split(_D1, 1)[0].strip()
    print(f"[validate] BOOT_OK — env built + reset; describe() harvested ({len(describe)} chars)")
    return describe
