"""Give the vendored `clear_organic_objects` produce ONE clean convex hull each.

Why (measured on the G1 sort cell, 2026-09-01). Every vendored object ships with a
`convexDecomposition` collider on its raw scan mesh, and the meshes differ by two orders of
magnitude: the lemons are ~700-point meshes (a few clean hulls) while the oranges are 9.9k
points plus a separate STEM collider, the pumpkins 65-94k, the onion 33k. A decomposition of a
high-poly near-sphere is dozens of thin sliver hulls, and a robot finger pressing into a seam
between two of them is a deep-penetration event that PhysX resolves by launching the object:
on the G1 the 63 mm orange went 9.5 m and then to a numerical blow-up, a 45 mm orange 167 m,
the pumpkin 11 m, the onion and avocado tens of metres -- while a lemon, same hand, same
grasp, never did. The near-convex produce loses nothing to a single hull, so that is what it
gets; the orange stems (sliver colliders the skin hull already covers) are switched off.

Runs on the dev pod (needs `pxr`, which lives inside kit):
    python scripts/fix_clear_organic_colliders.py --headless [--root <assets dir>]
Edits the USDs in place and prints what it changed. Idempotent.
"""

from __future__ import annotations

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="robobench/suites/packing/assets/clear_organic_objects")
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
app = AppLauncher(args).app

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

# near-convex objects: one hull is the faithful collider. The lemons (and `lime_g`, the lemon
# mesh in its own file) are in the list too: their ~700-point meshes decompose into 256 hulls
# (PhysX's cap), i.e. slivers, and a lemon pressed straight down onto the table by the descending
# hand tunnelled and was launched 20-190 m -- bit-identical across runs and unchanged by capping
# the arm and hand effort, so a collider event, not a force one. The lime keeps its 4k-point
# decomposition (scenery only now).
HULL = ("orange1", "orange2", "pomegranate", "pumpkinlarge",
        "pumpkinsmall", "avocado", "red_onion", "crabbypenholder")
# `lemon_g` and `lime_g` -- the sort tier's OWN copies of the lemon mesh (see
# make_lime_from_lemon.py) -- keep a DECOMPOSITION but a COARSE one. A single hull is wrong for
# this shape: the smooth hull rocks up and stands on its nipple within seconds (measured item
# height 51 -> 76 mm before the first grasp) and the palm grasp then meets a 66-84 mm span. The
# source mesh's own decomposition is 256 slivers, which tunnelled through the table when the
# descending hand pressed the fruit and launched it hundreds of metres (bit-identical across
# runs, unchanged by capping arm and hand effort). Eight hulls keep the flat spots that hold the
# lemon lying and have no slivers.
#
# WHY COPIES AND NOT lemon1/lemon2 THEMSELVES. `assets/clear_organic_objects/` is shared: the
# `fruits_on_plate` scene (and `locomanip.fruit_delivery` on top of it) read their produce from
# here. Re-collidering lemon1 in place took the verified `packing.fruits_on_plate.g1.joint`
# solution from score 100 to 0 -- its cage lifted the coarse lemon 166 mm and then shed it on
# the way to the staging pose. Collider choice is part of a GRASP's calibration, so the tier
# that needs a different one gets its own asset.
COARSE = {"lemon_g": 8, "lime_g": 8}
# sliver colliders fully inside another collider's hull
DISABLE_SUBSTR = ("Stem",)


def main() -> None:
    for key in HULL:
        files = glob.glob(os.path.join(args.root, key, "*.usd*"))
        if not files:
            print(f"[fix] {key}: no usd found", flush=True)
            continue
        stage = Usd.Stage.Open(files[0])
        changed = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            name = prim.GetName()
            if any(s in name for s in DISABLE_SUBSTR):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                changed.append(f"{name}: collision OFF")
                continue
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
                before = attr.Get()
                attr.Set("convexHull")
                changed.append(f"{name}: {before} -> convexHull")
        stage.GetRootLayer().Save()
        print(f"[fix] {key}: " + "; ".join(changed), flush=True)
    for key, n_hulls in COARSE.items():
        files = glob.glob(os.path.join(args.root, key, "*.usd*"))
        if not files:
            print(f"[fix] {key}: no usd found", flush=True)
            continue
        stage = Usd.Stage.Open(files[0])
        changed = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                continue
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("convexDecomposition")
            api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            api.CreateMaxConvexHullsAttr().Set(n_hulls)
            api.CreateHullVertexLimitAttr().Set(32)
            api.CreateMinThicknessAttr().Set(0.005)
            changed.append(f"{prim.GetName()}: convexDecomposition, maxConvexHulls={n_hulls}, "
                           f"hullVertexLimit=32, minThickness=5mm")
        stage.GetRootLayer().Save()
        print(f"[fix] {key}: " + "; ".join(changed), flush=True)
    print("FIX_DONE", flush=True)


if __name__ == "__main__":
    main()
    app.close()
