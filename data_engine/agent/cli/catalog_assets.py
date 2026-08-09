"""catalog_assets — list every spawnable asset with its measured facts, as JSON.

Inside a diversify container this is on PATH:

    catalog_assets                  every asset under the suites' assets/ trees
    catalog_assets --suite assembly only that suite
    catalog_assets --grep table     name/path substring filter

Fresh on every run (a few seconds, bare pxr — no Isaac boot), printed to stdout,
stored nowhere. Per asset:

    name, path        use the path in sim_utils.UsdFileCfg(usd_path=...)
    size_m            composed-stage bounding box in METERS (x, y, z) — place new
                      objects clear of the task using this, never the raw-file numbers
                      (the spawner replaces the root prim's authored offset)
    scale             the factor UsdFileCfg needs so the asset comes out in meters
                      (1.0 for meter-authored assets; 0.01 for cm-authored ones)
    physics           how it may be spawned:
                        rigid        colliders + rigid body -> a free object (RigidObjectCfg)
                        articulation joints inside -> ArticulationCfg
                        static       colliders only -> fixture/decoration (AssetBaseCfg)
                        visual       NO colliders -> set dressing only; everything
                                     passes through it, never a graspable "distractor"
    has_mass          True = mass/inertia authored in the file; do NOT override it
                      with mass_props (PhysX would recompute inertia from colliders)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXTS = {".usd", ".usda", ".usdc"}


def measure(path: Path) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError("stage failed to open")
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    size = [round((box.GetMax()[i] - box.GetMin()[i]) * mpu, 4) for i in range(3)] if not box.IsEmpty() else None

    rigid = articulated = collides = has_mass = False
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):  # colliders often live INSIDE instances
        rigid = rigid or prim.HasAPI(UsdPhysics.RigidBodyAPI)
        collides = collides or prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mass = has_mass or prim.HasAPI(UsdPhysics.MassAPI)
        articulated = articulated or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    physics = ("articulation" if articulated else "rigid" if rigid
               else "static" if collides else "visual")
    return {"size_m": size, "scale": mpu if mpu != 1.0 else 1.0, "physics": physics,
            "has_mass": has_mass}


def main() -> None:
    try:
        import pxr  # noqa: F401  (usd-core — bare USD, no Isaac boot)
    except ImportError:
        sys.exit("catalog_assets needs the standalone USD wheel: pip install usd-core")
    ap = argparse.ArgumentParser(description="list spawnable assets with measured facts (JSON)")
    ap.add_argument("--suite", default=None, help="only this suite's assets (e.g. assembly)")
    ap.add_argument("--grep", default=None, help="substring filter on name/path")
    args = ap.parse_args()

    roots = sorted((REPO / "robobench" / "suites").glob(f"{args.suite or '*'}/assets"))
    assets = []
    for root in roots:
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() not in EXTS or not f.is_file():
                continue
            rel = str(f.relative_to(REPO))
            if args.grep and args.grep.lower() not in rel.lower():
                continue
            entry = {"name": f.stem, "suite": root.parent.name, "path": rel}
            try:
                entry.update(measure(f))
            except Exception as e:  # unreadable/component-only files stay listed, flagged
                entry["error"] = f"{type(e).__name__}: {e}"
            assets.append(entry)
    json.dump({"asset_count": len(assets), "assets": assets}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
