"""Derive the `lime_g` asset (a lime-green lemon) from the vendored `lemon1`.

Why a separate asset and not a material override at spawn. The sort tier needs a second
graspable produce class, and the lemon mesh is the one object the G1 hand grasps reliably
(see the tier's solve). Spawning extra RigidObjects from the SAME lemon USD was measured to be
unsafe: every time one instance was lifted, another instance of the same USD 8-10 cm away was
launched hundreds of metres (four runs, always at a sibling's lift, never for the two that were
picked first). Giving the green lemons their own USD file gives PhysX its own cooked collision
data for them. The tint is baked into that file's OmniPBR shader (`diffuse_tint`) rather than
bound at spawn, so the asset is self-contained.

Runs on the dev pod (needs `pxr`, which lives inside kit), from the repo root:
    python scripts/make_lime_from_lemon.py --headless
Produces robobench/suites/packing/assets/clear_organic_objects/lime_g/ (mesh, materials and
textures copied from lemon1; `lime_g.usd` tinted). Add a `lime_g` entry to extents.json by
copying lemon1's bbox. Idempotent.
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


def main() -> None:
    src = os.path.join(args.root, "lemon1")
    dst = os.path.join(args.root, "lime_g")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    os.rename(os.path.join(dst, "lemon1.usd"), os.path.join(dst, "lime_g.usd"))
    stage = Usd.Stage.Open(os.path.join(dst, "lime_g.usd"))
    tint = Gf.Vec3f(*(float(v) for v in args.tint.split(",")))
    n = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            sh = UsdShade.Shader(prim)
            src_asset = sh.GetSourceAsset("mdl")
            if src_asset and "OmniPBR" in str(src_asset):
                sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(tint)
                n += 1
    stage.GetRootLayer().Save()
    print(f"[lime_g] tinted {n} OmniPBR shader(s) -> {dst}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
