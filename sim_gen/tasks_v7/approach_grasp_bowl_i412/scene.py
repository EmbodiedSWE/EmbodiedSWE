"""ChimneyCatchScene — cover the live floor slot with the bowl, pull the chimney gate
to drop the hidden ball into it, then deliver the loaded bowl to the dock
(sim_gen task `approach_grasp_bowl_i412`).

Derived from pick_place/approach_grasp_bowl, but STRATEGICALLY different: the seed is a
single-object prehensile plan — approach ONE bowl in tabletop clutter, close the parallel
jaw on its rim, lift it and carry it along marked waypoints; every load-bearing step is
one grasp affordance plus free-space transport, and the bowl itself is the whole goal.
Here the bowl is a TOOL-LIKE CATCH VESSEL in a three-stage, physically ORDER-FORCED plan
with an irreversible hidden-failure branch. A raised deck conceals a sealed under-deck
plenum; two identical drop-chimneys stand over open slots cut straight through the deck.
One chimney (random per episode) holds the orange ball on its gate blade, ~11 cm below
the rim — far beyond finger reach, so the seed's verb (reach in and grasp the goal
object) is physically impossible. The only path to the ball is to work the machine in
the right ORDER: (1) slide the bowl along the deck until it covers the live slot,
directly under the chute mouth (the bowl passes UNDER the chute — it can never be
carried in from above); (2) push the yellow gate knob outboard so the blade slides out
from under the ball and it free-falls down the chute into the bowl; (3) slide the loaded
bowl onto the red dock disc and let it settle. Triggering the gate BEFORE the slot is
covered drops the ball through the slot into the sealed plenum — irrecoverable by
construction (latched `lost`; success impossible, score capped). Nothing is ever lifted;
the bowl is never grasped-and-carried through free space by the sanctioned plan — it is
slid, and gravity through the machine does the only transfer of the goal object.

success() (all live, judged on physical poses):
  - the ball rests INSIDE the upright bowl (xy within the inner wall, z in the resting
    band above the bowl floor and below the rim),
  - the bowl is DOCKED: xy within `dock_tol` of the dock disc centre, resting on the
    deck (z band), upright,
  - bowl and ball are settled (slow), and the ball was never lost to the plenum.
score() = latched progress anchored in the demonstrated solution: 0.25 COVERED (bowl
covering the live slot, upright, on the deck, slow) + 0.35 CAUGHT (ball inside the bowl
and slow), capped at 0.60; a lost ball caps the score at 0.25 no matter what else
happened; exactly 1.0 iff success(). Doing nothing ~0; docking the empty bowl ~0.

Assets are fully procedural (no external files):
  - deck: KINEMATIC raised platform (0.72 x 0.60 m, top at z=0.10) built from five top
    plates that leave two 75 mm square slots at (+/-0.22, 0.10), perimeter seal walls
    down to the ground (the plenum under the deck is SEALED), and a low curb around the
    top edge.
  - towers (x2, fixed): a square chute (inner 55 mm) suspended 70 mm above the deck so
    the bowl passes underneath, with a blade table behind it (+y), guide ribs, an end
    stop, and a support gantry standing clear of the bowl's path.
  - blades (x2, DYNAMIC): flat gate blades with yellow knobs, resting closed through the
    chute; pushed outboard (+y) to open. A knob-vs-wall hard stop bounds inboard travel.
  - bowl (DYNAMIC, blue): flanged base disc (r 85 mm) + wall ring (inner r 65 mm, walls
    50 mm) — origin and CoM at the base centre, so it slides without tipping.
  - ball (DYNAMIC, orange): 40 mm sphere, spawned on the LIVE tower's blade inside the
    chute.
  - dock (KINEMATIC): a red visual-only disc flush with the deck (no collision — no sill
    to catch the sliding bowl) plus a tiny hidden collider inside the plenum.

Per-episode randomization (verified by readback in smoke): live side (+x/-x — which
chimney holds the ball), dock disc xy, bowl spawn xy + free yaw, ball jitter on the
blade. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         yaw: float = 0.0) -> None:
    """A colored box prim (optional yaw about local z); collides iff `contact_offset`
    is not None."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        sxf.AddRotateZOp().Set(math.degrees(yaw))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _cylinder(stage, path: str, radius: float, height: float, center, color,
              contact_offset: float | None) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_dynamic(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit MassAPI mass (custom
    spawners apply NO cfg schemas — author everything here), damping, no sleeping while
    we judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)


def _spawn_deck(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC deck at `prim_path`. Origin = ground-level centre. Five top
    plates (tops flush at deck_h) leave two square slots at (+/-slot_x, slot_y); four
    seal walls close the under-deck plenum down to the ground; a low curb rings the top
    so nothing slides off the edge."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)

    hx, hy, h, pt = cfg.half_x, cfg.half_y, cfg.deck_h, cfg.plate_t
    a2 = cfg.slot_a / 2
    sx, sy = cfg.slot_x, cfg.slot_y
    zc = h - pt / 2
    fy1 = sy - a2          # front plate ends at the slot's lower-y edge
    by0 = sy + a2          # back plate starts at the slot's upper-y edge
    plates = (
        ("plate_front", (2 * hx, fy1 + hy, pt), (0.0, (fy1 - hy) / 2, zc)),
        ("plate_back", (2 * hx, hy - by0, pt), (0.0, (by0 + hy) / 2, zc)),
        ("plate_left", (hx - sx - a2, cfg.slot_a, pt), ((-hx - sx - a2) / 2, sy, zc)),
        ("plate_mid", (2 * (sx - a2), cfg.slot_a, pt), (0.0, sy, zc)),
        ("plate_right", (hx - sx - a2, cfg.slot_a, pt), ((hx + sx + a2) / 2, sy, zc)),
    )
    for name, size, center in plates:
        _box(stage, f"{prim_path}/{name}", size, center, cfg.deck_color, cfg.contact_offset)
    wt, wh = cfg.seal_t, h - pt
    for name, cx, cy, szx, szy in (
        ("seal_nx", -(hx - wt / 2), 0.0, wt, 2 * hy),
        ("seal_px", hx - wt / 2, 0.0, wt, 2 * hy),
        ("seal_ny", 0.0, -(hy - wt / 2), 2 * hx, wt),
        ("seal_py", 0.0, hy - wt / 2, 2 * hx, wt),
    ):
        _box(stage, f"{prim_path}/{name}", (szx, szy, wh), (cx, cy, wh / 2),
             cfg.seal_color, cfg.contact_offset)
    ct, ch = cfg.curb_t, cfg.curb_h
    for name, cx, cy, szx, szy in (
        ("curb_nx", -(hx - ct / 2), 0.0, ct, 2 * hy),
        ("curb_px", hx - ct / 2, 0.0, ct, 2 * hy),
        ("curb_ny", 0.0, -(hy - ct / 2), 2 * hx, ct),
        ("curb_py", 0.0, hy - ct / 2, 2 * hx, ct),
    ):
        _box(stage, f"{prim_path}/{name}", (szx, szy, ch), (cx, cy, h + ch / 2),
             cfg.seal_color, cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.deck_mu, cfg.deck_mu - 0.05)
    for name, _, _ in plates:
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{name}"), mat)
    return root


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one KINEMATIC drop-chimney at `prim_path`. Origin = ground level under the
    chute centre; local xy is the chute frame. Square chute (x-walls full height from
    chute_z0; y-walls stop short of the bottom so the blade passes beneath them), two
    interior ledges that carry the blade's inboard end, a blade table on +y with guide
    ribs and an end stop, and a two-post gantry (visual bracing) standing clear of the
    bowl's swept circle."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)

    a, wt = cfg.chute_ihalf, cfg.chute_wt
    z0, z1 = cfg.chute_z0, cfg.chute_z1
    wall_x_h = z1 - z0
    wall_y_h = z1 - (z0 + cfg.blade_t + cfg.blade_gap)
    wall_y_z0 = z0 + cfg.blade_t + cfg.blade_gap
    # x-walls (normal to x): full height, y-span covers the corners.
    for name, sxs in (("wall_nx", -1.0), ("wall_px", 1.0)):
        _box(stage, f"{prim_path}/{name}", (wt, 2 * (a + wt), wall_x_h),
             (sxs * (a + wt / 2), 0.0, z0 + wall_x_h / 2), cfg.tower_color, cfg.contact_offset)
    # y-walls (normal to y): raised bottoms — the blade slides under them.
    for name, sys_ in (("wall_ny", -1.0), ("wall_py", 1.0)):
        _box(stage, f"{prim_path}/{name}", (2 * a, wt, wall_y_h),
             (0.0, sys_ * (a + wt / 2), wall_y_z0 + wall_y_h / 2),
             cfg.tower_color, cfg.contact_offset)
    # interior ledges on the x-wall inner faces: carry the blade's inboard end.
    for name, sxs in (("ledge_nx", -1.0), ("ledge_px", 1.0)):
        _box(stage, f"{prim_path}/{name}", (cfg.ledge_w, 2 * a, cfg.table_t),
             (sxs * (a - cfg.ledge_w / 2), 0.0, z0 - cfg.table_t / 2),
             cfg.tower_color, cfg.contact_offset)
    # blade table on +y (top flush with chute_z0) + leg down to the deck.
    ty0, ty1 = a + wt, cfg.table_y1
    _box(stage, f"{prim_path}/table", (cfg.table_w, ty1 - ty0, cfg.table_t),
         (0.0, (ty0 + ty1) / 2, z0 - cfg.table_t / 2), cfg.tower_color, cfg.contact_offset)
    _box(stage, f"{prim_path}/leg", (cfg.table_w - 0.02, 0.020, z0 - cfg.table_t - cfg.deck_h),
         (0.0, cfg.leg_y, (z0 - cfg.table_t + cfg.deck_h) / 2),
         cfg.tower_color, cfg.contact_offset)
    # guide ribs flanking the blade on the table.
    rx = cfg.blade_w / 2 + cfg.rib_gap + cfg.rib_w / 2
    for name, sxs in (("rib_nx", -1.0), ("rib_px", 1.0)):
        _box(stage, f"{prim_path}/{name}", (cfg.rib_w, ty1 - ty0, cfg.rib_h),
             (sxs * rx, (ty0 + ty1) / 2, z0 + cfg.rib_h / 2),
             cfg.tower_color, cfg.contact_offset)
    # end stop bounding the withdrawal stroke.
    _box(stage, f"{prim_path}/stop", (cfg.blade_w + 0.020, cfg.stop_t, cfg.stop_h),
         (0.0, cfg.stop_y + cfg.stop_t / 2, z0 + cfg.stop_h / 2),
         cfg.tower_color, cfg.contact_offset)
    # gantry: two posts + slanted-looking braces up to the chute top (visual bracing —
    # the tower is one kinematic body, so structural connection is cosmetic).
    for name, sxs in (("post_nx", -1.0), ("post_px", 1.0)):
        _box(stage, f"{prim_path}/{name}", (cfg.post_a, cfg.post_a, z1 - cfg.deck_h),
             (sxs * cfg.post_x, cfg.post_y, (z1 + cfg.deck_h) / 2),
             cfg.tower_color, cfg.contact_offset)
        _box(stage, f"{prim_path}/beam{name[-3:]}", (0.030, cfg.post_y - a, 0.030),
             (sxs * (a + wt / 2), (cfg.post_y + a) / 2, z1 - 0.015),
             cfg.tower_color, cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.table_mu, cfg.table_mu - 0.03)
    for n in ("table", "ledge_nx", "ledge_px", "rib_nx", "rib_px"):
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{n}"), mat)
    return root


def _spawn_blade(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC gate blade at `prim_path`. Origin = plate centre. Flat plate
    (long axis = y, the withdrawal direction) + yellow knob near the outboard end. The
    knob doubles as the inboard hard stop (it hits the chute back wall)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.15, ang_damp=0.50)

    _box(stage, f"{prim_path}/plate", (cfg.blade_w, cfg.blade_len, cfg.blade_t),
         (0.0, 0.0, 0.0), cfg.blade_color, cfg.contact_offset)
    _box(stage, f"{prim_path}/knob", (cfg.knob_a, cfg.knob_a, cfg.knob_h),
         (0.0, cfg.knob_y, cfg.blade_t / 2 + cfg.knob_h / 2),
         (0.95, 0.85, 0.10), cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.30, 0.25)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/plate"), mat)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC bowl at `prim_path`. Origin = centre of the base disc's
    BOTTOM face; MassAPI mass keeps the CoM at the origin, so the sliding bowl is
    bottom-heavy by construction. Flanged base disc + 10-segment wall ring."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.20, ang_damp=0.60)

    _cylinder(stage, f"{prim_path}/base", cfg.base_r, cfg.base_t,
              (0.0, 0.0, cfg.base_t / 2), cfg.body_color, cfg.contact_offset)
    n_seg = 10
    outer_r = cfg.inner_r + cfg.wall_t
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * outer_r * math.tan(math.pi / n_seg) + 0.002
    zc = cfg.base_t + cfg.wall_h / 2
    for k in range(n_seg):
        ang = 2 * math.pi * k / n_seg
        _box(stage, f"{prim_path}/wall_{k:02d}", (cfg.wall_t, seg_len, cfg.wall_h),
             (r_mid * math.cos(ang), r_mid * math.sin(ang), zc),
             cfg.body_color, cfg.contact_offset, yaw=ang)
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.25, 0.20)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/base"), mat)
    return root


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC dock marker at `prim_path`. Origin = deck-top level at the
    disc centre. The red disc is VISUAL ONLY (no collision — no sill for the sliding
    bowl to catch on); a tiny hidden collider floats inside the sealed plenum so the
    body owns at least one collision shape."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)
    _cylinder(stage, f"{prim_path}/disc", cfg.disc_r, 0.002,
              (0.0, 0.0, 0.001), (0.90, 0.12, 0.10), None)  # visual only
    _box(stage, f"{prim_path}/hidden", (0.010, 0.010, 0.010),
         (0.0, 0.0, -0.035), (0.3, 0.3, 0.3), cfg.contact_offset)
    return root


def _deck_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "deck" not in _SPAWNER_CACHE:

        @configclass
        class DeckSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_deck)
            half_x: float = 0.36
            half_y: float = 0.30
            deck_h: float = 0.10
            plate_t: float = 0.012
            seal_t: float = 0.012
            curb_t: float = 0.012
            curb_h: float = 0.020
            slot_a: float = 0.075
            slot_x: float = 0.22
            slot_y: float = 0.10
            deck_mu: float = 0.30
            deck_color: tuple = (0.60, 0.60, 0.62)
            seal_color: tuple = (0.42, 0.42, 0.46)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["deck"] = DeckSpawnerCfg

    return _SPAWNER_CACHE["deck"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


def _tower_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tower" not in _SPAWNER_CACHE:

        @configclass
        class TowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tower)
            deck_h: float = 0.10
            chute_ihalf: float = 0.0275
            chute_wt: float = 0.010
            chute_z0: float = 0.170
            chute_z1: float = 0.330
            blade_t: float = 0.008
            blade_gap: float = 0.002
            blade_w: float = 0.051
            ledge_w: float = 0.004
            table_t: float = 0.006
            table_w: float = 0.090
            table_y1: float = 0.178
            leg_y: float = 0.130
            rib_w: float = 0.008
            rib_h: float = 0.014
            rib_gap: float = 0.002
            stop_y: float = 0.155
            stop_t: float = 0.010
            stop_h: float = 0.020
            post_x: float = 0.055
            post_y: float = 0.115
            post_a: float = 0.020
            table_mu: float = 0.20
            tower_color: tuple = (0.30, 0.34, 0.42)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tower"] = TowerSpawnerCfg

    return _SPAWNER_CACHE["tower"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


def _blade_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "blade" not in _SPAWNER_CACHE:

        @configclass
        class BladeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_blade)
            blade_w: float = 0.051
            blade_len: float = 0.110
            blade_t: float = 0.008
            knob_a: float = 0.018
            knob_h: float = 0.030
            knob_y: float = 0.042
            mass: float = 0.060
            blade_color: tuple = (0.75, 0.62, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["blade"] = BladeSpawnerCfg

    return _SPAWNER_CACHE["blade"](**kw)


def _bowl_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            base_r: float = 0.085
            base_t: float = 0.008
            inner_r: float = 0.065
            wall_t: float = 0.010
            wall_h: float = 0.050
            mass: float = 0.35
            body_color: tuple = (0.15, 0.35, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](**kw)


def _dock_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dock" not in _SPAWNER_CACHE:

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            disc_r: float = 0.070
            contact_offset: float = 0.002

        _SPAWNER_CACHE["dock"] = DockSpawnerCfg

    return _SPAWNER_CACHE["dock"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChimneyCatchSceneCfg(BaseCfg):
    """Config for `ChimneyCatchScene`. Honesty knobs asserted in `__post_init__`: the
    catch is GUARANTEED whenever the bowl is within `cover_tol` of the live slot (chute
    guidance + wall-press exit puts the ball inside the inner wall with margin); the
    bowl passes UNDER the chute with clearance; the blade stroke opens the full aperture
    before the end stop; the ball on the blade is beyond finger reach; the lost gate
    fires only inside the sealed plenum; the dock, bowl spawn, and slots are mutually
    separated so no credit is granted at spawn."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    cover_tol: float = tunable(0.028)   # bowl-centre xy distance to the live slot centre (m)
    cover_vmax: float = tunable(0.12)   # bowl speed gate for the covered latch (m/s)
    caught_vmax: float = tunable(0.30)  # ball speed gate for the caught latch (m/s)
    dock_tol: float = tunable(0.030)    # bowl-centre xy distance to the dock centre (m)
    bowl_z_lo: float = tunable(0.095)   # bowl origin resting band on the deck (m)
    bowl_z_hi: float = tunable(0.112)
    upright_min: float = tunable(0.95)  # min world-z component of the bowl's body z-axis
    settle_vel: float = tunable(0.06)   # max bowl/ball |lin vel| at success (m/s)
    lost_z: float = tunable(0.055)      # ball centre below this = sealed in the plenum (latch)
    in_bowl_slack: float = tunable(0.004)  # in-bowl radius = inner_r - ball_r + slack
    # (the bowl floor is flat, so the generic rest is BALL AGAINST THE INNER WALL at
    #  exactly inner_r - ball_r; slack is outward. Rim-perch is rejected by the z band,
    #  an outside-leaning ball sits at >= outer_r + ball_r = 0.095 - far past the gate.)
    ball_rel_z_lo: float = tunable(0.014)  # ball centre minus bowl origin: resting band
    ball_rel_z_hi: float = tunable(0.056)  # (rest is 0.028; rim-perch ~0.070 -> rejected)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    dock_x: float = tunable(0.20)       # dock centre x in +/- this (m)
    dock_y_lo: float = tunable(-0.16)   # dock centre y range (m)
    dock_y_hi: float = tunable(-0.09)
    bowl_x: float = tunable(0.06)       # bowl spawn x in +/- this (m)
    bowl_y_lo: float = tunable(-0.02)   # bowl spawn y range (m)
    bowl_y_hi: float = tunable(0.06)
    ball_jitter: float = tunable(0.004)  # ball xy jitter on the blade (m)

    # --- info: structure ----------------------------------------------------------------------
    half_x: float = info(0.36)          # deck half-length (x)
    half_y: float = info(0.30)          # deck half-width (y)
    deck_h: float = info(0.10)          # deck top height
    plate_t: float = info(0.012)        # deck top-plate thickness
    seal_t: float = info(0.012)         # plenum seal-wall thickness
    curb_t: float = info(0.012)
    curb_h: float = info(0.020)
    slot_a: float = info(0.075)         # square slot full width
    slot_x: float = info(0.22)          # slot / tower centres at (+/-slot_x, slot_y)
    slot_y: float = info(0.10)
    chute_ihalf: float = info(0.0275)   # chute inner half-width (inner width 55 mm)
    chute_wt: float = info(0.010)
    chute_z0: float = info(0.170)       # chute mouth (bottom of the x-walls / blade top plane)
    chute_z1: float = info(0.330)       # chute rim
    ledge_w: float = info(0.004)        # interior blade-support ledge protrusion
    table_t: float = info(0.006)
    table_w: float = info(0.090)
    table_y1: float = info(0.178)       # table outboard end (chute frame, m)
    stop_y: float = info(0.155)         # end-stop inner face (chute frame, m)
    rib_w: float = info(0.008)
    rib_h: float = info(0.014)
    rib_gap: float = info(0.002)
    post_x: float = info(0.055)         # gantry post centres (chute frame)
    post_y: float = info(0.115)
    post_a: float = info(0.020)
    blade_len: float = info(0.110)      # blade long axis (y = withdrawal direction)
    blade_w: float = info(0.051)
    blade_t: float = info(0.008)
    blade_gap: float = info(0.002)      # clearance between blade top and y-wall bottoms
    blade_close_y: float = info(0.020)  # blade-centre y in the chute frame when closed
    knob_a: float = info(0.018)
    knob_h: float = info(0.030)
    knob_y: float = info(0.042)         # knob centre in the blade frame
    blade_mass: float = info(0.060)
    base_r: float = info(0.085)         # bowl flanged base radius
    base_t: float = info(0.008)
    inner_r: float = info(0.065)        # bowl inner wall radius
    bowl_wall_t: float = info(0.010)
    bowl_wall_h: float = info(0.050)
    bowl_mass: float = info(0.35)
    ball_r: float = info(0.020)
    ball_mass: float = info(0.030)
    finger_reach: float = info(0.055)   # Franka finger length (embodiment honesty)
    dock_disc_r: float = info(0.070)
    deck_mu: float = info(0.30)
    table_mu: float = info(0.20)
    deck_color: tuple = info((0.60, 0.60, 0.62))
    seal_color: tuple = info((0.42, 0.42, 0.46))
    tower_color: tuple = info((0.30, 0.34, 0.42))
    blade_color: tuple = info((0.75, 0.62, 0.12))
    bowl_color: tuple = info((0.15, 0.35, 0.85))
    ball_color: tuple = info((0.95, 0.50, 0.10))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    bowl_top: float = field(default=None, init=False)     # deck + base_t + wall_h
    blade_z: float = field(default=None, init=False)      # blade-centre rest height
    ball_blade_z: float = field(default=None, init=False)  # ball-centre rest on the blade
    blade_open_y: float = field(default=None, init=False)  # blade-centre y = aperture open
    in_bowl_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bowl_top = self.deck_h + self.base_t + self.bowl_wall_h            # 0.158
        self.blade_z = self.chute_z0 + self.blade_t / 2                         # 0.174
        self.ball_blade_z = self.chute_z0 + self.blade_t + self.ball_r          # 0.198
        # Aperture fully open once the blade's inboard tip passes +chute_ihalf.
        self.blade_open_y = self.chute_ihalf + self.blade_len / 2 + 0.004       # 0.0865
        self.in_bowl_r = self.inner_r - self.ball_r + self.in_bowl_slack        # 0.049

        a, r = self.chute_ihalf, self.ball_r
        # Slot passes the ball freely; the chute admits it; the ledge aperture passes it.
        assert self.slot_a > 2 * r + 0.010, "slot must pass the falling ball"
        assert 2 * a > 2 * r + 0.008, "chute must admit the ball"
        assert 2 * (a - self.ledge_w) > 2 * r + 0.005, "ledge aperture must pass the ball"
        # Blade support: the plate overlaps each ledge, clears the x-walls, and slides
        # beneath the raised y-walls with play.
        assert self.blade_w / 2 - (a - self.ledge_w) >= 0.0015, "blade must rest on the ledges"
        assert self.blade_w / 2 + 0.0015 < a + self.chute_wt / 2, "blade must clear the x-walls"
        assert self.blade_gap >= 0.0015, "blade needs play under the y-walls"
        # Underpass: the bowl slides beneath the chute mouth AND the thinner table.
        assert self.chute_z0 - self.bowl_top >= 0.010, "chute mouth must clear the bowl rim"
        assert (self.chute_z0 - self.table_t) - self.bowl_top >= 0.005, (
            "table underside must clear the bowl rim")
        # CATCH GUARANTEE: worst-case exit offset (pressed into a chute corner) plus the
        # cover tolerance still lands the ball's centre inside the inner wall w/ margin.
        assert self.cover_tol + math.sqrt(2.0) * (a - r) <= self.in_bowl_r, (
            "covered bowl must catch every guided drop")
        # Falling ball clears the bowl rim ring on the way in.
        assert self.cover_tol + math.sqrt(2.0) * (a - r) + r <= self.inner_r - 0.004, (
            "drop path must clear the bowl rim")
        # Stroke: the end stop leaves room to open the full aperture, with margin.
        avail = self.stop_y - (self.blade_close_y + self.blade_len / 2)
        need = (a - (self.blade_close_y - self.blade_len / 2)) + 0.008
        assert avail >= need, "withdrawal stroke must fully open the aperture"
        # The blade edge (not the knob) meets the end stop; the knob meets the back wall
        # before the blade tip can leave the front slit region inboard.
        assert self.knob_y + self.knob_a / 2 <= self.blade_len / 2 - 0.003, (
            "blade edge must reach the end stop before the knob")
        assert self.blade_close_y + self.knob_y - self.knob_a / 2 > a + self.chute_wt + 0.002, (
            "knob must sit outboard of the back wall (it is the inboard hard stop)")
        # Ball depth: beyond finger reach from the rim (the seed's grasp is impossible).
        depth = self.chute_z1 - (self.chute_z0 + self.blade_t + 2 * r)
        assert depth >= self.finger_reach + 0.045, "ball must be far beyond finger reach"
        # Lost gate: strictly inside the sealed plenum; unreachable above the deck.
        assert r + 0.005 < self.lost_z < self.deck_h - self.plate_t - 0.005
        assert self.deck_h + r > self.lost_z + 0.030, "an on-deck ball must never read lost"
        # Dock zone: far from both slots, inside the curb, and clear of the bowl spawn.
        assert self.slot_y - self.dock_y_hi > self.cover_tol + self.dock_tol + 0.020, (
            "dock and slots must be mutually exclusive credit zones")
        assert self.dock_x + self.dock_tol + self.base_r < self.half_x - self.curb_t - 0.004
        assert -self.dock_y_lo + self.dock_tol + self.base_r < self.half_y - self.curb_t - 0.004
        assert self.bowl_y_lo - self.dock_y_hi > self.dock_tol + 0.020, (
            "bowl must never spawn already docked")
        assert self.slot_x - self.bowl_x > self.cover_tol + 0.020, (
            "bowl must never spawn already covering a slot")
        assert self.bowl_x + self.base_r < self.half_x - self.curb_t - 0.004
        assert self.bowl_y_hi + self.base_r < self.slot_y + self.post_y - self.post_a, (
            "bowl spawn must stay clear of the gantry posts")
        # Covering bowl clears the gantry posts and the table leg.
        assert math.hypot(self.post_x, self.post_y) >= self.base_r + self.post_a / 2 + 0.008, (
            "gantry posts must clear the covering bowl")
        # Deck fixtures fit: table end inside the curb; slots inside the deck footprint.
        assert self.slot_y + self.table_y1 < self.half_y - self.curb_t - 0.006
        assert self.slot_x + self.slot_a / 2 < self.half_x - self.seal_t - 0.020
        # Resting bands bracket the physical rest states only.
        assert self.bowl_z_lo < self.deck_h < self.bowl_z_hi
        rest_rel = self.base_t + self.ball_r  # 0.028
        assert self.ball_rel_z_lo < rest_rel < self.ball_rel_z_hi
        assert self.ball_rel_z_hi < self.base_t + self.bowl_wall_h + self.ball_r - 0.010, (
            "a rim-perched ball must fail the in-bowl z band")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chimney_catch")
class ChimneyCatchScene(BaseScene):
    cfg: ChimneyCatchSceneCfg

    def __init__(self, cfg: ChimneyCatchSceneCfg | None = None) -> None:
        super().__init__(cfg or ChimneyCatchSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        tower_kw = dict(
            deck_h=c.deck_h, chute_ihalf=c.chute_ihalf, chute_wt=c.chute_wt,
            chute_z0=c.chute_z0, chute_z1=c.chute_z1, blade_t=c.blade_t,
            blade_gap=c.blade_gap, blade_w=c.blade_w, ledge_w=c.ledge_w,
            table_t=c.table_t, table_w=c.table_w, table_y1=c.table_y1,
            rib_w=c.rib_w, rib_h=c.rib_h, rib_gap=c.rib_gap, stop_y=c.stop_y,
            post_x=c.post_x, post_y=c.post_y, post_a=c.post_a, table_mu=c.table_mu,
            tower_color=c.tower_color, contact_offset=c.contact_offset,
        )
        blade_kw = dict(
            blade_w=c.blade_w, blade_len=c.blade_len, blade_t=c.blade_t,
            knob_a=c.knob_a, knob_h=c.knob_h, knob_y=c.knob_y, mass=c.blade_mass,
            blade_color=c.blade_color, contact_offset=c.contact_offset,
        )
        ball_spawn = sim_utils.SphereCfg(
            radius=c.ball_r,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.06, angular_damping=0.22,
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,  # kills the GPU sphere-creep artifact
                sleep_threshold=0.0, stabilization_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
        )
        blade_pos = (c.slot_x, c.slot_y + c.blade_close_y, c.blade_z)
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=_deck_spawner_cfg(
                    half_x=c.half_x, half_y=c.half_y, deck_h=c.deck_h, plate_t=c.plate_t,
                    seal_t=c.seal_t, curb_t=c.curb_t, curb_h=c.curb_h, slot_a=c.slot_a,
                    slot_x=c.slot_x, slot_y=c.slot_y, deck_mu=c.deck_mu,
                    deck_color=c.deck_color, seal_color=c.seal_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "tower_p": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TowerP",
                spawn=_tower_spawner_cfg(leg_y=0.130, **tower_kw),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.slot_x, c.slot_y, 0.0)),
            ),
            "tower_n": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TowerN",
                spawn=_tower_spawner_cfg(leg_y=0.130, **tower_kw),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-c.slot_x, c.slot_y, 0.0)),
            ),
            "blade_p": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BladeP",
                spawn=_blade_spawner_cfg(**blade_kw),
                init_state=RigidObjectCfg.InitialStateCfg(pos=blade_pos),
            ),
            "blade_n": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BladeN",
                spawn=_blade_spawner_cfg(**blade_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-blade_pos[0], blade_pos[1], blade_pos[2])),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=_bowl_spawner_cfg(
                    base_r=c.base_r, base_t=c.base_t, inner_r=c.inner_r,
                    wall_t=c.bowl_wall_t, wall_h=c.bowl_wall_h, mass=c.bowl_mass,
                    body_color=c.bowl_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.02, c.deck_h + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=ball_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, c.slot_y, c.ball_blade_z + 0.002)),
            ),
            "dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock",
                spawn=_dock_spawner_cfg(disc_r=c.dock_disc_r, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.125, c.deck_h)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.deck: RigidObject = env.iscene["deck"]
        self.towers: list[RigidObject] = [env.iscene["tower_p"], env.iscene["tower_n"]]
        self.blades: dict[str, RigidObject] = {
            "p": env.iscene["blade_p"], "n": env.iscene["blade_n"]}
        self.bowl: RigidObject = env.iscene["bowl"]
        self.ball: RigidObject = env.iscene["ball"]
        self.dock: RigidObject = env.iscene["dock"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.side = torch.ones(n, device=dev)     # +1: ball on the +x tower; -1: -x
        self.dock_xy = torch.zeros(n, 2, device=dev)
        self.covered = torch.zeros(n, device=dev)
        self.caught = torch.zeros(n, device=dev)
        self.lost = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: coin-flip the live tower; park BOTH blades closed; set the
        ball on the live blade (small xy jitter); teleport the dock disc to a random
        spot in the front zone; drop the bowl at a random spawn pose (free yaw); zero
        the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self.side[env_ids] = side

        # --- blades: both closed (they are dynamic; re-park them every episode) ---
        for key, sx in (("p", 1.0), ("n", -1.0)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx * c.slot_x
            st[:, 1] = c.slot_y + c.blade_close_y
            st[:, 2] = c.blade_z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.blades[key].write_root_state_to_sim(st, env_ids)

        # --- ball: on the LIVE blade inside the chute, tiny jitter ---
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = side * c.slot_x + jit[:, 0]
        st[:, 1] = c.slot_y + jit[:, 1]
        st[:, 2] = c.ball_blade_z + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        # --- dock: random spot in the front zone ---
        dx = (torch.rand(m, device=dev) * 2 - 1) * c.dock_x
        dy = c.dock_y_lo + torch.rand(m, device=dev) * (c.dock_y_hi - c.dock_y_lo)
        self.dock_xy[env_ids, 0] = dx
        self.dock_xy[env_ids, 1] = dy
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = dx
        st[:, 1] = dy
        st[:, 2] = c.deck_h
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.dock.write_root_state_to_sim(st, env_ids)

        # --- bowl: random spawn pose, free yaw ---
        bx = (torch.rand(m, device=dev) * 2 - 1) * c.bowl_x
        by = c.bowl_y_lo + torch.rand(m, device=dev) * (c.bowl_y_hi - c.bowl_y_lo)
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = bx
        st[:, 1] = by
        st[:, 2] = c.deck_h + 0.002
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.covered[env_ids] = 0.0
        self.caught[env_ids] = 0.0
        self.lost[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "blade_p": self.blades["p"].data.root_state_w[env_ids].clone(),
            "blade_n": self.blades["n"].data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "dock": self.dock.data.root_state_w[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "dock_xy": self.dock_xy[env_ids].clone(),
            "covered": self.covered[env_ids].clone(),
            "caught": self.caught[env_ids].clone(),
            "lost": self.lost[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.blades["p"].write_root_state_to_sim(state["blade_p"], env_ids)
        self.blades["n"].write_root_state_to_sim(state["blade_n"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.dock.write_root_state_to_sim(state["dock"], env_ids)
        self.side[env_ids] = state["side"]
        self.dock_xy[env_ids] = state["dock_xy"]
        self.covered[env_ids] = state["covered"]
        self.caught[env_ids] = state["caught"]
        self.lost[env_ids] = state["lost"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A raised gray DECK ({2 * c.half_x * 100:.0f} x {2 * c.half_y * 100:.0f} cm, top "
            f"{c.deck_h * 100:.0f} cm up, curbed all round) hides a SEALED cavity underneath. "
            f"Two identical slate DROP-CHIMNEYS stand over open floor slots cut straight "
            f"through the deck; each chimney is a square chute suspended "
            f"{(c.chute_z0 - c.deck_h) * 100:.0f} mm above the deck, closed at the bottom by a "
            f"sliding GATE BLADE with a YELLOW KNOB on the table behind it. ONE chimney "
            f"(varies by episode) holds a {2 * c.ball_r * 1000:.0f} mm ORANGE BALL on its "
            f"blade, visible down the open top but far too deep to reach. A BLUE BOWL sits on "
            f"the deck, and a RED DISC painted on the deck marks the dock.\n"
            f"Goal: get the ball resting inside the bowl and the loaded bowl parked on the red "
            f"dock disc. The bowl slides UNDER the chimney — first slide it until it covers "
            f"the live floor slot directly below the chute, then push the yellow knob away "
            f"from the chimney so the blade slides out and the ball drops down the chute into "
            f"the bowl, and finally slide the loaded bowl onto the red disc and let it rest. "
            f"ORDER MATTERS: opening the gate while the slot is uncovered drops the ball "
            f"through the floor into the sealed cavity — it can NEVER be recovered."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the blue bowl under the chimney that holds the orange ball so it covers "
            "the floor slot, push that chimney's yellow knob to drop the ball into the bowl, "
            "then slide the loaded bowl onto the red dock disc. Never open the gate before "
            "the bowl is in place — the ball would fall through the floor and be lost."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _bowl_pos(self) -> torch.Tensor:
        return self.bowl.data.root_pos_w - self.env_origins

    def _ball_pos(self) -> torch.Tensor:
        return self.ball.data.root_pos_w - self.env_origins

    def _bowl_upright(self) -> torch.Tensor:
        q = self.bowl.data.root_quat_w  # (N,4) wxyz
        z_w = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return z_w > self.cfg.upright_min

    def _live_slot_xy(self) -> torch.Tensor:
        c = self.cfg
        return torch.stack(
            [self.side * c.slot_x, torch.full_like(self.side, c.slot_y)], dim=-1)

    def _in_bowl(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the bowl interior (xy within the inner wall,
        z in the resting band above the bowl floor and below the rim)."""
        c = self.cfg
        bp, wp = self._ball_pos(), self._bowl_pos()
        d_xy = (bp[:, :2] - wp[:, :2]).norm(dim=-1)
        rel_z = bp[:, 2] - wp[:, 2]
        return (d_xy < c.in_bowl_r) & (rel_z > c.ball_rel_z_lo) & (rel_z < c.ball_rel_z_hi)

    def _on_deck_upright(self) -> torch.Tensor:
        c = self.cfg
        wp = self._bowl_pos()
        return (wp[:, 2] > c.bowl_z_lo) & (wp[:, 2] < c.bowl_z_hi) & self._bowl_upright()

    def _docked(self) -> torch.Tensor:
        wp = self._bowl_pos()
        d = (wp[:, :2] - self.dock_xy).norm(dim=-1)
        return (d < self.cfg.dock_tol) & self._on_deck_upright()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the lost gate (ball ever inside the sealed plenum), covered (bowl over
        the live slot, upright, on the deck, slow), and caught (ball inside the bowl and
        slow) every physics substep, so transient progress keeps its credit and the
        irreversible failure keeps its penalty."""
        c = self.cfg
        bp = self._ball_pos()
        self.lost = self.lost | (bp[:, 2] < c.lost_z)
        wp = self._bowl_pos()
        d_cov = (wp[:, :2] - self._live_slot_xy()).norm(dim=-1)
        slow_bowl = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.cover_vmax
        cov = (d_cov < c.cover_tol) & self._on_deck_upright() & slow_bowl
        self.covered = torch.maximum(self.covered, cov.float())
        slow_ball = self.ball.data.root_lin_vel_w.norm(dim=-1) < c.caught_vmax
        self.caught = torch.maximum(self.caught, (self._in_bowl() & slow_ball).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: ball resting inside the upright bowl NOW, bowl docked on the red
        disc, both settled, and the ball never lost to the plenum."""
        c = self.cfg
        settled = ((self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_vel)
                   & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_vel))
        return self._in_bowl() & self._docked() & settled & ~self.lost

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 * covered + 0.35 * caught, capped at 0.60; a lost
        ball caps everything at 0.25; exactly 1.0 iff success(). Doing nothing ~0;
        docking the empty bowl (the seed's carry-the-bowl strategy) ~0; triggering the
        gate first strands the ball in the plenum and caps the episode at 0.25."""
        base = (0.25 * self.covered + 0.35 * self.caught).clamp(0.0, 0.60)
        base = torch.where(self.lost, base.clamp(max=0.25), base)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="chimney_catch", robot="null"))
