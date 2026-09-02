"""Derive the sort tier's OWN produce copies from the vendored `lemon1`:

    lemon_g  — the lemon mesh, untinted
    lime_g   — the same mesh tinted lime-green (a second graspable produce class)

WHY COPIES AT ALL — two independent reasons, both measured.

  * SHARED DIRECTORY. `assets/clear_organic_objects/` is read by the `fruits_on_plate` scene as
    well (and by `locomanip.fruit_delivery` on top of it). The G1 sort tier's palm grasp needs a
    coarse convex decomposition of the lemon (see fix_clear_organic_colliders.py); applying that
    to `lemon1` in place took the verified `packing.fruits_on_plate.g1.joint` solution from score
    100 to 0. Collider choice belongs to a grasp's calibration, so the tier that needs its own
    gets its own file.
  * SHARED COOKED COLLISION DATA. Spawning several RigidObjects from the SAME usd was measured to
    be unsafe here: whenever one instance was lifted, another instance of that usd 8-10 cm away
    was launched hundreds of metres (four runs, always at a sibling's lift, never for the two
    picked first). Separate files give PhysX separate cooked data.

The tint is baked into the copy's OmniPBR shader (`diffuse_tint`) rather than bound at spawn, so
each asset is self-contained.

Runs on the dev pod (needs `pxr`, which lives inside kit), from the repo root:
    python scripts/make_lime_from_lemon.py --headless
then re-run `scripts/fix_clear_organic_colliders.py` to give both copies their coarse colliders.
Add `lemon_g`/`lime_g` entries to extents.json by copying lemon1's bbox. Idempotent.
"""

from __future__ import annotations

import argparse
import os
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="robobench/suites/packing/assets/clear_organic_objects")
parser.add_argument("--tint", default="0.55,0.95,0.35")
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
app = AppLauncher(args).app

from pxr import Gf, Sdf, Usd, UsdShade  # noqa: E402

# key -> tint, or None to copy the mesh unchanged
COPIES: dict[str, str | None] = {"lemon_g": None, "lime_g": args.tint}


def main() -> None:
    src = os.path.join(args.root, "lemon1")
    for key, tint_s in COPIES.items():
        dst = os.path.join(args.root, key)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        os.rename(os.path.join(dst, "lemon1.usd"), os.path.join(dst, f"{key}.usd"))
        if tint_s is None:
            print(f"[{key}] copied from lemon1 (untinted) -> {dst}", flush=True)
            continue
        stage = Usd.Stage.Open(os.path.join(dst, f"{key}.usd"))
        tint = Gf.Vec3f(*(float(v) for v in tint_s.split(",")))
        n = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Shader):
                sh = UsdShade.Shader(prim)
                src_asset = sh.GetSourceAsset("mdl")
                if src_asset and "OmniPBR" in str(src_asset):
                    sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(tint)
                    n += 1
        stage.GetRootLayer().Save()
        print(f"[{key}] tinted {n} OmniPBR shader(s) -> {dst}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
