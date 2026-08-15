"""Download the standalone gripper assets into this folder, so everything loads locally.

`assets/gripper/` holds the grippers used by `XArm7RobotCfg.gripper`:

  - `panda_hand.usd`  the Franka hand (authored here; built from the panda USD by reference)

This script fetches any downloaded entries in GRIPPERS from the public Isaac Sim 5.1 asset
bucket (one-time; the repo never touches the network at runtime). Currently empty — the Robotiq
Hand-E was vendored, found blocked on the bulb task by its ~42 mm usable aperture, and removed.
Run with a plain `usd-core` Python:

    python robobench/robots/assets/gripper/fetch_grippers.py
"""

from __future__ import annotations

import os
import posixpath
import urllib.request

# Isaac Sim 5.1 cloud asset root (= ISAAC_NUCLEUS_DIR), pinned for reproducibility.
ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/"
HERE = os.path.dirname(__file__)
USD_EXT = (".usd", ".usda", ".usdc", ".usdz")

#: (source folder under ROOT, start USD inside it, local folder under assets/gripper/)
GRIPPERS = ()


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


def vendor(src: str, start: str, dst: str) -> None:
    out_dir = os.path.join(HERE, dst)
    seen: set[str] = set()
    escaped: set[str] = set()

    def walk(rel: str) -> None:
        if rel in seen:
            return
        seen.add(rel)
        local = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(local) or out_dir, exist_ok=True)
        try:
            with urllib.request.urlopen(ROOT + src + rel, timeout=120) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  MISS {src}{rel}: {e}")
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

    walk(start)
    print(f"{dst}: vendored {len(seen)} files into {out_dir}")
    if escaped:
        print("  absolute/escaping refs (left to the runtime resolver):")
        for e in sorted(escaped):
            print("   ", e)


def main() -> None:
    for src, start, dst in GRIPPERS:
        vendor(src, start, dst)


if __name__ == "__main__":
    main()
