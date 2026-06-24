"""Re-download the vendored Unitree G1 USD tree (provenance + reproducibility).

The G1 assets under `g1/` are vendored into the repo so robobench stays relocatable (no dependence
on a live Nucleus / the Isaac Lab tree). This script regenerates that folder from the public Isaac
Sim asset bucket, so the copy is reproducible and auditable.

Source: `G1_29DOF_CFG.spawn.usd_path` = `{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd`, where
`ISAAC_NUCLEUS_DIR` resolves (Isaac Sim 5.1) to the S3 cloud root below. `g1.usd` is a thin wrapper
whose `configuration/*.usd` layers + meshes are followed recursively (relative asset paths only).

We KEEP the default `ThreeFinger` hand variant (what `G1_29DOF_CFG` / the FixedBaseUpperBodyIK task
use) and PRUNE the unused `Inspire` hand variant (~75 MB) — the G1 never selects it (the Inspire G1
is a different USD upstream). The 3 `OmniPBR.mdl` 404s are core Omniverse material shaders that
resolve from the runtime MDL search path, not downloadable assets — expected, harmless.

Run with a plain `usd-core` Python (no AppLauncher needed):
    python robobench/robots/assets/fetch_g1.py
"""

from __future__ import annotations

import os
import posixpath
import urllib.request

# Isaac Sim 5.1 cloud asset root (= ISAAC_NUCLEUS_DIR), pinned for reproducibility.
ROOT_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/"
OUT = os.path.join(os.path.dirname(__file__), "g1")
START = "g1.usd"
PRUNE = ("configuration/inspire_hand", "g1_left_hand_inspire", "g1_right_hand_inspire")  # unused variant
USD_EXT = (".usd", ".usda", ".usdc", ".usdz")


def _deps(local: str) -> list[str]:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(local)
    if layer is None:
        return []
    out = set(layer.GetCompositionAssetDependencies())

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.AttributeSpec) and spec.default is not None:
            v = spec.default
            if isinstance(v, Sdf.AssetPath):
                out.add(v.path)
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "AssetPathArray":
                out.update(e.path for e in v if isinstance(e, Sdf.AssetPath))

    layer.Traverse(Sdf.Path("/"), visit)
    return [d for d in out if d]


def main() -> None:
    seen: set[str] = set()
    escaped: set[str] = set()

    def walk(rel: str) -> None:
        if rel in seen or any(p in rel for p in PRUNE):
            return
        seen.add(rel)
        local = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        try:
            with urllib.request.urlopen(ROOT_URL + rel, timeout=120) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  MISS {rel}: {e}")
            return
        with open(local, "wb") as f:
            f.write(data)
        if not rel.lower().endswith(USD_EXT):
            return
        base = posixpath.dirname(rel)
        for dep in _deps(local):
            if dep.startswith("/") or "://" in dep:
                escaped.add(dep)
                continue
            child = posixpath.normpath(posixpath.join(base, dep))
            if child.startswith(".."):
                escaped.add(dep)
                continue
            walk(child)

    walk(START)
    print(f"\nVendored {len(seen)} files into {OUT}")
    if escaped:
        print("absolute/escaping refs (left to the runtime resolver):")
        for e in sorted(escaped):
            print("  ", e)


if __name__ == "__main__":
    main()
