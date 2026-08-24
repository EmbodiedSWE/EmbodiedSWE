"""Calibrate a generated object mesh into a sim-ready USD.

Takes an objects-stage mesh (real_to_sim/objects/data/objects/<name>/mesh.glb,
unitless, visual-only) plus ONE caliper measurement, and bakes:

    <name>.usd            rigid body: visual mesh + separate collision + mass
    <name>_visual.usd     the converted visual mesh (referenced by the above)

    python calibrate_object.py <name> --diameter 0.08 [--mass 0.35]
    python calibrate_object.py <name> --height 0.26   [--mass 0.35]

Conventions baked in:
  - origin at the object's BOTTOM CENTER -> placing at (x, y, 0) sits it on
    the calibrated work surface (desk top = z0 in this example's scenes)
  - up axis: the mesh's longest bbox extent, rotated to +z (--up overrides)
  - collision: a fitted CYLINDER (purpose=guide, invisible in renders) —
    deliberately simpler than the visuals; edit/replace per object as needed
  - mass on the root (default 0.35 kg — REPLACE with the weighed value)

Runs inside the repo's Isaac venv (self-bootstraps) — the GLB->USD conversion
uses omni.kit.asset_converter, so first run boots a headless kit app (~1 min).
"""

import argparse
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[3]
OBJECTS_DATA = _HERE.parents[2] / "objects" / "data" / "objects"

_ISAAC_PY = REPO / ".venv" / "bin" / "python"
# asset_converter's native bindings need libxml2.so.2 (dropped by new distros);
# the background stage's shim provides it. Must be in LD_LIBRARY_PATH before
# process start, so the bootstrap re-exec also fixes the env.
_SHIM = _HERE.parents[2] / "background" / ".setup_shim"
_LDP = os.environ.get("LD_LIBRARY_PATH", "")
_needs_env = (_SHIM / "libxml2.so.2").exists() and str(_SHIM) not in _LDP
if _needs_env or (_ISAAC_PY.exists() and pathlib.Path(sys.executable).resolve() != _ISAAC_PY.resolve()):
    os.environ["LD_LIBRARY_PATH"] = f"{_SHIM}:{_LDP}" if _needs_env else _LDP
    os.execv(str(_ISAAC_PY), [str(_ISAAC_PY), *sys.argv])

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("name", help="object name under objects/data/objects/")
ap.add_argument("--diameter", type=float, help="measured horizontal diameter/width (m)")
ap.add_argument("--height", type=float, help="measured height (m) — alternative pin")
ap.add_argument("--mass", type=float, default=0.35, help="measured mass (kg); default 0.35")
ap.add_argument("--up", choices=["auto", "x", "y", "z"], default="auto",
                help="mesh axis that should become +z (auto = longest extent)")
ap.add_argument("--flip", action="store_true", help="flip the up axis (lid ended up at the bottom)")
ap.add_argument("--glb", help="override input glb (default: <name>/mesh.glb)")
ap.add_argument("--collision", choices=["stack", "cylinder"], default="stack",
                help="collider: 'stack' = lathe profile from mesh cross-sections "
                     "(graspable where the object narrows); 'cylinder' = one bbox cylinder")
ap.add_argument("--friction", type=float, default=1.0,
                help="collider friction (static=dynamic); PhysX default 0.5 is too slippery to grasp")
args = ap.parse_args()
if not args.diameter and not args.height:
    ap.error("need --diameter or --height (calipers!)")

obj_dir = pathlib.Path(args.glb).parent if args.glb else OBJECTS_DATA / args.name
glb = pathlib.Path(args.glb) if args.glb else obj_dir / "mesh.glb"
assert glb.exists(), f"missing {glb}"
visual_usd = obj_dir / f"{args.name}_visual.usd"
out_usd = obj_dir / f"{args.name}.usd"

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
ok = False
try:
    import asyncio

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter as converter
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    # ---- 1. GLB -> USD (visual) --------------------------------------------
    async def convert():
        task = converter.get_instance().create_converter_task(str(glb), str(visual_usd), None, None)
        return await task.wait_until_finished()

    assert asyncio.get_event_loop().run_until_complete(convert()), "asset_converter failed"
    print(f"converted: {visual_usd}")

    # ---- 2. measure the converted mesh -------------------------------------
    vstage = Usd.Stage.Open(str(visual_usd))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = cache.ComputeWorldBound(vstage.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
    ext = hi - lo
    up_idx = {"x": 0, "y": 1, "z": 2}.get(args.up, int(np.argmax(ext)))
    diam_raw = float(max(np.delete(ext, up_idx)))
    height_raw = float(ext[up_idx])
    scale = (args.diameter / diam_raw) if args.diameter else (args.height / height_raw)
    print(f"raw bbox: {ext.round(4)} (up={'xyz'[up_idx]}); scale {scale:.5f} m/unit")

    # rotation taking mesh up axis -> +z (about the axis perpendicular to both)
    rot = {0: Gf.Rotation(Gf.Vec3d(0, 1, 0), 90), 1: Gf.Rotation(Gf.Vec3d(1, 0, 0), -90),
           2: Gf.Rotation(Gf.Vec3d(1, 0, 0), 0)}[up_idx]
    if args.flip:
        rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), 180) * rot
    M = Gf.Matrix4d(1.0)
    M.SetRotate(rot)
    M = M * Gf.Matrix4d(1.0).SetScale(scale)  # world = scale * rot * local
    corners = [M.Transform(Gf.Vec3d(*[(lo, hi)[b][k] for k, b in enumerate((i >> np.arange(3)) & 1)]))
               for i in range(8)]
    c = np.array([[v[0], v[1], v[2]] for v in corners])
    clo, chi = c.min(axis=0), c.max(axis=0)
    center_xy = (clo[:2] + chi[:2]) / 2
    shift = Gf.Vec3d(-center_xy[0], -center_xy[1], -clo[2])  # bottom center -> origin
    h_final = float(chi[2] - clo[2])
    r_final = float(max(chi[0] - clo[0], chi[1] - clo[1])) / 2
    print(f"final: diameter {2*r_final:.4f} m, height {h_final:.4f} m, mass {args.mass} kg")

    # ---- 3. author the rigid-body wrapper -----------------------------------
    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{args.name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(args.mass)

    vis = UsdGeom.Xform.Define(stage, f"/{args.name}/visual")
    vis.GetPrim().GetReferences().AddReference(f"./{visual_usd.name}")
    M_total = M * Gf.Matrix4d(1.0).SetTranslate(shift)  # rot+scale, then shift
    UsdGeom.Xformable(vis).AddTransformOp().Set(M_total)

    from pxr import UsdShade

    pmat = UsdShade.Material.Define(stage, f"/{args.name}/physics_material")
    pmat_api = UsdPhysics.MaterialAPI.Apply(pmat.GetPrim())
    pmat_api.CreateStaticFrictionAttr().Set(args.friction)
    pmat_api.CreateDynamicFrictionAttr().Set(args.friction)
    pmat_api.CreateRestitutionAttr().Set(0.0)

    def add_col_cylinder(path, radius, height, z_center):
        col = UsdGeom.Cylinder.Define(stage, path)
        col.GetAxisAttr().Set("Z")
        col.GetRadiusAttr().Set(radius)
        col.GetHeightAttr().Set(height)
        col.GetExtentAttr().Set([(-radius, -radius, -height / 2), (radius, radius, height / 2)])
        UsdGeom.XformCommonAPI(col).SetTranslate(Gf.Vec3d(0, 0, z_center))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        col.GetPurposeAttr().Set(UsdGeom.Tokens.guide)  # physics-only, never rendered
        UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(
            pmat, materialPurpose="physics")

    if args.collision == "cylinder":
        add_col_cylinder(f"/{args.name}/collision", r_final, h_final, h_final / 2)
    else:
        # lathe-profile stack: radius per height band from the mesh's own
        # cross-sections — a gripper can pinch wherever the object narrows
        vcache = UsdGeom.XformCache()
        pts_all = []
        for prim in vstage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                L2W = np.array(vcache.GetLocalToWorldTransform(prim))
                pts = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ L2W
                pts_all.append(pts[:, :3])
        p = np.concatenate(pts_all)
        p = np.array([[*M.Transform(Gf.Vec3d(*q))] for q in p]) + np.array(shift)
        n_slices = max(4, min(16, int(np.ceil(h_final / 0.02))))
        edges = np.linspace(0.0, h_final, n_slices + 1)
        print("collision stack:")
        for k in range(n_slices):
            band = p[(p[:, 2] >= edges[k]) & (p[:, 2] <= edges[k + 1])]
            if len(band) < 8:
                r_k = r_final if k == 0 else prev_r  # noqa: F821 — sparse band: reuse
            else:
                r_k = float(np.percentile(np.linalg.norm(band[:, :2], axis=1), 99))
            prev_r = r_k
            zc = (edges[k] + edges[k + 1]) / 2
            add_col_cylinder(f"/{args.name}/collision_{k:02d}", r_k, edges[k + 1] - edges[k], zc)
            print(f"  z {edges[k]:.3f}-{edges[k+1]:.3f}: r {r_k:.4f}")

    stage.Save()
    print(f"wrote {out_usd}")

    # ---- 4. verify -----------------------------------------------------------
    check = Usd.Stage.Open(str(out_usd))
    b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        check.GetPrimAtPath(f"/{args.name}/visual")).ComputeAlignedRange()
    print(f"verify visual bbox: min {np.array(b.GetMin()).round(4)} max {np.array(b.GetMax()).round(4)}")
    ok = True
except Exception:
    import traceback

    traceback.print_exc()
finally:
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    app.close()
