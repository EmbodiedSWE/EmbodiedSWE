"""Author the pc_ram case's MISTAKE-STATE colliders (app-free; pxr only).

Adds to `pc_case_ram_assembly_mb.usd` — the leaf layer only the RAM tasks load, so the shared
base case (pc_gpu / pc_motherboard / pc_all) is untouched — a `mistake_fixture` scope of
invisible (guide-purpose) cuboid colliders where a wrong-place insertion lands today:

  nontarget_slot_{1,3}   the two visual DIMM slots WITHOUT channels, as the closed 6.5 x 131 x 4.5 mm
                         plastic bodies they are — a stick pressed there now rests ON the plastic
                         instead of sinking through it to the plate (measured visual bounds,
                         collision atlas 2026-08-24)
  board_plate_ext        the whole ATX board face (footprint from the seven screw fixtures + 8 mm
                         margin; overlaps the DIMM strip's existing plate), top flush at z 0, so
                         nothing falls 29 mm through the motherboard visual to the table
  tray_floor             the case's side panel it lies on (z -0.0289 .. -0.0259), inside the shell
                         walls, so anything dropped in the case rests on the tray

None of these sit on a nominal trajectory (the solves' pick/carry/press corridors are clear of
all four volumes), so verified solves should re-pass at their historical numbers — re-run the
pc_ram smoke + a franka solve to confirm, #45-style. Idempotent: re-running rewrites the scope.

    env_newton/bin/python robobench/scripts/author_pc_ram_mistake_colliders.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

USD = Path(__file__).resolve().parents[1] / "suites" / "assembly" / "assets" / "pc" / "pc_case_ram_assembly_mb.usd"
CASE = "/pc_case_ram_assembly/case"
SCOPE = f"{CASE}/mistake_fixture"

# (name, (xmin, ymin, zmin), (xmax, ymax, zmax)) in the case frame, metres
BOXES = (
    ("nontarget_slot_1", (-0.1361, -0.1333, 0.0), (-0.1297, -0.0023, 0.0045)),
    ("nontarget_slot_3", (-0.1167, -0.1333, 0.0), (-0.1102, -0.0023, 0.0045)),
    ("board_plate_ext", (-0.1730, -0.1530, -0.0030), (0.0700, 0.1430, 0.0)),  # the ATX footprint
    # (screw pattern x [-0.165, +0.062] y [-0.145, +0.135] + 8 mm); overlaps the DIMM strip plate
    ("tray_floor", (-0.3411, -0.1902, -0.0289), (0.1210, 0.2587, -0.0259)),
)


def cuboid(stage: Usd.Stage, path: str, mn, mx) -> None:
    """A collision-only cube: guide purpose (invisible to render), CollisionAPI, no rigid body of
    its own (it rides the parent case body like the dimm/shell fixtures)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    c = [(a + b) / 2 for a, b in zip(mn, mx)]
    s = [b - a for a, b in zip(mn, mx)]
    xf.AddTranslateOp().Set(Gf.Vec3d(*c))
    xf.AddScaleOp().Set(Gf.Vec3f(*s))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stage = Usd.Stage.Open(str(USD))
    assert stage.GetPrimAtPath(CASE).IsValid(), CASE
    if stage.GetPrimAtPath(SCOPE).IsValid():
        stage.RemovePrim(SCOPE)
    UsdGeom.Scope.Define(stage, SCOPE)
    for name, mn, mx in BOXES:
        cuboid(stage, f"{SCOPE}/{name}", mn, mx)
        print(f"  {name:18s} min {mn} max {mx}")
    if args.dry_run:
        print("dry run — not saved")
        return
    stage.GetRootLayer().Save()
    print(f"saved {USD}")


if __name__ == "__main__":
    main()
