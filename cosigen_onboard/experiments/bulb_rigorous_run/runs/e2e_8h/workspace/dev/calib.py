"""Calibrate bulb/socket geometry: USD bboxes in the bulb's local frame."""
import os
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)

from pxr import Usd, UsdGeom, Gf

stage = env.stage
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])

bulb = env.scene.bulbs[0]
bp = bulb.data.root_pos_w[0].cpu().numpy()
bq = bulb.data.root_quat_w[0].cpu().numpy()  # wxyz
print("bulb root", bp.tolist(), bq.tolist())

rot = Gf.Rotation(Gf.Quatd(float(bq[0]), float(bq[1]), float(bq[2]), float(bq[3])))
xf = Gf.Matrix4d().SetRotate(rot) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*[float(v) for v in bp]))
inv = xf.GetInverse()

def show(prim, indent=0):
    path = str(prim.GetPath())
    tname = prim.GetTypeName()
    extra = ""
    if prim.IsA(UsdGeom.Imageable) and tname in ("Mesh", "Cylinder", "Cube", "Sphere", "Capsule"):
        wb = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = wb.GetMin(), wb.GetMax()
        lo_l = inv.Transform(lo); hi_l = inv.Transform(hi)
        # local-frame aabb corners (only valid if axis-aligned, still useful)
        extra = f" world[{[round(v,4) for v in lo]}..{[round(v,4) for v in hi]}]"
        if "Bulb" in path:
            extra += f" bulbframe[{[round(v,4) for v in lo_l]}..{[round(v,4) for v in hi_l]}]"
        purp = UsdGeom.Imageable(prim).ComputePurpose()
        extra += f" purpose={purp}"
    print("  "*indent + f"{tname} {path}{extra}")

for root in ["/World/envs/env_0/Bulb_0", "/World/envs/env_0/Socket_0"]:
    p = stage.GetPrimAtPath(root)
    for prim in Usd.PrimRange(p):
        show(prim)

# physx collider info for the bulb
mats = bulb.root_physx_view.get_material_properties()
print("bulb material props", mats.tolist())
sock = env.scene.sockets[0]
print("sock material props", sock.root_physx_view.get_material_properties().tolist())

print("DONE-CALIB")
os._exit(0)
