"""DepositVaultScene — bank the cash brick through the vault's rotary transfer hatch.

Derived from rlbench/put_money_in_safe ("put the money away in the safe on the top
shelf": a dollar stack and a safe whose door stands open; the robot grasps the stack,
carries it through the door opening and sets it down on a shelf inside — ONE
grasp-transport-place onto a static receptacle). Here there is NO door and NO
reachable shelf goal: the vault interior is sealed on every face, and the only way in
is an AIRLOCK — a vertical-axis revolving TURRET (one open pocket, one sealing back
slab) driven by a crank on the roof, over a fixed half-disc transfer shelf:

  1. crank the turret to its receive stop so its pocket faces the front deposit window
     (at rest the pocket faces the sealed rear and the turret's back slab blocks
     the window — a brick pushed at the window is physically refused, shown in
     smoke with a real force);
  2. push the CASH BRICK through the window into the open pocket; it comes to rest
     on the transfer shelf INSIDE the pocket — still not banked (the shelf level is
     above the bin and the window is open: smoke shows this state is rejected);
  3. crank the turret back to its seal stop: the pocket walls sweep the brick
     around the axis across the shelf; the shelf only covers the FRONT half-disc,
     so past the diameter edge the brick loses support and DROPS into the bin
     below, and the back slab arrives back in front of the window — resealed.

The goal state is a PHYSICAL outcome pair: brick settled inside the lower bin
chamber (below the shelf) AND the hatch resealed (turret back on its seal stop).
Execution order is forced by the geometry: sealed hatch refuses the brick; a loaded
pocket only delivers by rotation; the seal can only return after the sweep.

Assets are fully procedural (compound spawners; per-child density on the turret so
the crank inertia is real; root MassAPI on the heavy vault — memory: custom spawners
apply no cfg schemas, so mass/collision are authored in the funcs):
  - vault: heavy DYNAMIC box (footprint 0.30 x 0.30 m, 0.42 m tall, 15 mm walls,
    80 kg, origin at footprint centre on the ground). Front (+x) wall carries the
    DEPOSIT WINDOW (100 mm wide x 60 mm tall, sill at z 0.23). Inside: a bin floor,
    and the TRANSFER SHELF at z 0.215..0.23 covering only the front half (x > 0).
    Roof at z 0.403..0.418 with a 60 mm axle hole (fully covered by the turret disc
    below it). Dynamic, not kinematic: a joint anchored to a kinematic body0 stays
    world-fixed when the body is teleported at reset.
  - turret: DYNAMIC compound on a spawn-authored vertical RevoluteJoint through the
    vault centre (limits: seal stop at 0 deg, receive stop at 170 deg — strictly
    inside the PhysX +/-180 wrap boundary — 0.5 deg slack). Top disc (r 120 mm) under the roof; POCKET = two radial walls at
    turret-local bearings 190 +/- 50 deg (= 360 - receive_deg, so the pocket faces
    the window EXACTLY at the 170 deg receive stop and its walls sit clear of the
    +/-30 mm insertion corridor; radial span 65..125 mm, z 0.237..0.385,
    7 mm above the shelf so a swept brick cannot wedge under); SEAL SLAB = a
    140 mm chord plate on the turret-local +x side (covers the window whenever the
    turret is within ~13 deg of its seal stop; its corner sweep leaves only 8 mm
    to the walls — under the brick's 30 mm minimum dimension, so nothing passes
    beside it); axle through the roof hole; CRANK bar + red knob on top (the crank
    points at the window when sealed, away from it at receive — the visual state
    cue).
  - brick: the cash brick, a banded green block 120 x 60 x 30 mm, ~0.19 kg.
  - stand: small KINEMATIC teller stand (160 x 120 x 100 mm) where the brick
    starts, at grasp height clear of the ground.

Per-episode randomization (readback-verifiable): vault yaw +/-20 deg + xy jitter,
initial turret angle U(0, 6) deg on its seal stop, stand bearing U(-50, 50) deg at
0.55 m off the vault front + own yaw, brick yaw + xy jitter on the stand.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.20 * open_frac — latched max turret angle / 160 deg (crank progress)
  0.25 * loaded    — brick at shelf level within the pocket radius while the
                     turret is at receive (latched, 3-step persistence)
  0.25 * dropped   — brick inside the bin volume (latched, 3-step persistence)
  1.0 iff success() — brick settled in the bin AND turret back on its seal stop
                     (<= 8 deg), everything still and finite. Non-success capped
                     at 0.70; null policy ~0 (initial angle credits <= 0.008).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- USD authoring helpers --------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None, density: float | None = None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable,
             density: float | None = None):
    """One z-axis cylinder child."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The vault: heavy DYNAMIC compound. Local frame: origin at the footprint
    centre on the ground; +x = front (deposit window side); the turret axis is
    the local vertical through (0, 0)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.vault_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body, steel = c.vault_color, c.steel_color
    ho, hi = c.half_out, c.half_in          # 0.15 / 0.135
    wt = ho - hi                            # wall 0.015
    zr0, zr1 = c.roof_z0, c.roof_z1        # 0.403 / 0.418
    wy, wz0, wz1 = c.win_half_y, c.win_z0, c.win_z1  # 0.05 / 0.23 / 0.29
    # floor
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.0075),
             size=(2 * ho, 2 * ho, 0.015), color=body, collide=collide)
    # side walls (full height under the roof)
    for name, yc in (("wall_yp", ho - wt / 2), ("wall_yn", -(ho - wt / 2))):
        _add_box(stage, f"{prim_path}/{name}", center=(0.0, yc, (0.015 + zr0) / 2),
                 size=(2 * ho, wt, zr0 - 0.015), color=body, collide=collide)
    # rear wall
    _add_box(stage, f"{prim_path}/wall_rear", center=(-(ho - wt / 2), 0.0, (0.015 + zr0) / 2),
             size=(wt, 2 * hi, zr0 - 0.015), color=body, collide=collide)
    # front wall: below window / above window / window cheeks
    xc = ho - wt / 2
    _add_box(stage, f"{prim_path}/front_lo", center=(xc, 0.0, (0.015 + wz0) / 2),
             size=(wt, 2 * hi, wz0 - 0.015), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/front_hi", center=(xc, 0.0, (wz1 + zr0) / 2),
             size=(wt, 2 * hi, zr0 - wz1), color=body, collide=collide)
    for name, yc in (("front_yp", (wy + hi) / 2), ("front_yn", -(wy + hi) / 2)):
        _add_box(stage, f"{prim_path}/{name}", center=(xc, yc, (wz0 + wz1) / 2),
                 size=(wt, hi - wy, wz1 - wz0), color=body, collide=collide)
    # transfer shelf: FRONT half-disc region only (x > 0)
    _add_box(stage, f"{prim_path}/shelf", center=(hi / 2, 0.0, (c.shelf_z0 + c.shelf_z1) / 2),
             size=(hi, 2 * hi, c.shelf_z1 - c.shelf_z0), color=steel, collide=collide)
    # roof with a square axle hole (half width c.hole_half)
    hh = c.hole_half
    _add_box(stage, f"{prim_path}/roof_f", center=((hh + ho) / 2, 0.0, (zr0 + zr1) / 2),
             size=(ho - hh, 2 * ho, zr1 - zr0), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_b", center=(-(hh + ho) / 2, 0.0, (zr0 + zr1) / 2),
             size=(ho - hh, 2 * ho, zr1 - zr0), color=body, collide=collide)
    for name, yc in (("roof_yp", (hh + ho) / 2), ("roof_yn", -(hh + ho) / 2)):
        _add_box(stage, f"{prim_path}/{name}", center=(0.0, yc, (zr0 + zr1) / 2),
                 size=(2 * hh, ho - hh, zr1 - zr0), color=body, collide=collide)
    return root


def _spawn_turret(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The transfer turret: DYNAMIC compound, body origin ON the rotation axis at
    ground level (turret-local -x = pocket opening direction, +x = seal slab /
    crank direction), plus the spawn-authored vertical REVOLUTE joint to the
    sibling vault (joints must be authored at spawn). Masses via per-child
    DENSITY (true CoM + crank inertia)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(float(c.turret_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    steel = c.steel_color
    # top disc (covers the roof's axle hole from below)
    _add_cyl(stage, f"{prim_path}/disc", center=(0.0, 0.0, (c.disc_z0 + c.disc_z1) / 2),
             radius=c.disc_r, height=c.disc_z1 - c.disc_z0, color=steel,
             collide=collide, density=c.steel_density)
    # pocket walls at turret-local bearings pocket_center +/- pocket_half_deg
    # (pocket_center = 360 - receive_deg, so the pocket faces the window EXACTLY
    # at the receive stop and the wall tips stay clear of the insertion corridor)
    zc = (c.wall_z0 + c.wall_z1) / 2
    rlen = c.pock_r1 - c.pock_r0
    rmid = (c.pock_r0 + c.pock_r1) / 2
    for name, bear in (("pwall_a", c.pocket_center_deg - c.pocket_half_deg),
                       ("pwall_b", c.pocket_center_deg + c.pocket_half_deg)):
        b = math.radians(bear)
        _add_box(stage, f"{prim_path}/{name}",
                 center=(rmid * math.cos(b), rmid * math.sin(b), zc),
                 size=(rlen, c.wall_t, c.wall_z1 - c.wall_z0),
                 color=c.pocket_color, collide=collide,
                 orient=(math.cos(b / 2), 0.0, 0.0, math.sin(b / 2)),
                 density=c.steel_density)
    # seal slab: chord plate on the +x side (blocks the window at the seal stop)
    _add_box(stage, f"{prim_path}/slab", center=(c.slab_x, 0.0, zc),
             size=(c.wall_t, c.slab_w, c.wall_z1 - c.wall_z0),
             color=c.slab_color, collide=collide, density=c.steel_density)
    # axle up through the roof hole + crank bar + knob (points +x = seal side)
    _add_cyl(stage, f"{prim_path}/axle", center=(0.0, 0.0, (c.disc_z0 + c.axle_z1) / 2),
             radius=c.axle_r, height=c.axle_z1 - c.disc_z0, color=steel,
             collide=collide, density=c.steel_density)
    _add_box(stage, f"{prim_path}/crank", center=(c.crank_r / 2, 0.0, c.axle_z1 + 0.007),
             size=(c.crank_r + 0.02, 0.024, 0.014), color=steel, collide=collide,
             density=c.steel_density)
    _add_cyl(stage, f"{prim_path}/knob", center=(c.crank_r, 0.0, c.axle_z1 + 0.030),
             radius=c.knob_r, height=0.032, color=c.knob_color, collide=collide,
             density=c.steel_density)

    # revolute joint to the sibling vault, vertical axis through the vault centre
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # turret<->vault contact stays ON (USD default for a joint pair is filtered);
    # clearances are authored (>= 8 mm everywhere at every turret angle)
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.30))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.30))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # joint angle = turret rotation rel vault about z; 0 = SEAL stop (slab in
    # front of the window), +receive_deg = RECEIVE stop (pocket at the window)
    j.CreateLowerLimitAttr(-0.5)
    j.CreateUpperLimitAttr(float(cfg.receive_deg) + 0.5)
    return root


def _spawn_brick(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The cash brick: one green box (120 x 60 x 30 mm) with a paper band stripe."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0),
             size=(c.brick_l, c.brick_w, c.brick_h), color=c.brick_color,
             collide=collide, density=c.brick_density)
    # visual band (thin overlay box, same body — decoration, collider too (harmless))
    _add_box(stage, f"{prim_path}/band",
             center=(0.0, 0.0, 0.0),
             size=(c.brick_l * 0.25, c.brick_w + 0.0004, c.brick_h + 0.0004),
             color=(0.92, 0.90, 0.82), collide=collide, density=c.brick_density)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The teller stand: KINEMATIC block the brick starts on (grasp height).
    No joints attach to it, so kinematic is safe."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(cfg.contact_offset)
    _add_box(stage, f"{prim_path}/block", center=(0.0, 0.0, cfg.stand_h / 2),
             size=(0.16, 0.12, cfg.stand_h), color=cfg.stand_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            vault_mass: float = 80.0
            half_out: float = 0.15
            half_in: float = 0.135
            roof_z0: float = 0.403
            roof_z1: float = 0.418
            win_half_y: float = 0.05
            win_z0: float = 0.23
            win_z1: float = 0.29
            shelf_z0: float = 0.215
            shelf_z1: float = 0.23
            hole_half: float = 0.03
            vault_color: tuple = (0.30, 0.33, 0.38)
            steel_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.002

        @configclass
        class TurretSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_turret)
            disc_r: float = 0.12
            disc_z0: float = 0.38
            disc_z1: float = 0.395
            wall_z0: float = 0.237
            wall_z1: float = 0.385
            wall_t: float = 0.012
            pock_r0: float = 0.065
            pock_r1: float = 0.125
            pocket_center_deg: float = 190.0
            pocket_half_deg: float = 50.0
            slab_x: float = 0.10
            slab_w: float = 0.14
            axle_r: float = 0.018
            axle_z1: float = 0.44
            crank_r: float = 0.09
            knob_r: float = 0.014
            # STRICTLY inside (-180, 180): a stop AT 180 sits on the PhysX
            # revolute angle-wrap boundary — pressed against it the joint reading
            # wraps to -180, the limit constraint sees a huge violation and
            # slingshots the turret a full turn (observed on the forge)
            receive_deg: float = 170.0
            turret_ang_damping: float = 0.8
            steel_density: float = 2000.0
            steel_color: tuple = (0.55, 0.57, 0.60)
            pocket_color: tuple = (0.75, 0.62, 0.20)
            slab_color: tuple = (0.20, 0.22, 0.60)
            knob_color: tuple = (0.85, 0.15, 0.15)
            contact_offset: float = 0.002

        @configclass
        class BrickSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_brick)
            brick_l: float = 0.12
            brick_w: float = 0.06
            brick_h: float = 0.03
            brick_density: float = 900.0
            brick_color: tuple = (0.16, 0.45, 0.18)
            contact_offset: float = 0.002

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            stand_h: float = 0.10
            stand_color: tuple = (0.70, 0.62, 0.45)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg
        _SPAWNER_CACHE["turret"] = TurretSpawnerCfg
        _SPAWNER_CACHE["brick"] = BrickSpawnerCfg
        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class DepositVaultSceneCfg(BaseCfg):
    """Config for `DepositVaultScene`. The airlock geometry is honest by
    construction — every clause is asserted numerically in __post_init__: the
    sealed slab covers the whole window at every initial angle; no gap beside the
    turret admits the brick; the loaded brick fits wholly inside the pocket
    sweep; the bin volume lies strictly below the shelf."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    sealed_max_deg: float = tunable(8.0)   # turret angle at/below this counts as resealed
    bin_top_z: float = tunable(0.18)       # brick CoM below this (vault local) = in the bin
    bin_half_xy: float = tunable(0.12)     # brick CoM within this of the vault axis (x and y)
    load_r_max: float = tunable(0.085)     # loaded: brick CoM radius about the axis below this
    load_z0: float = tunable(0.225)        # loaded: brick CoM z band (on the shelf)
    load_z1: float = tunable(0.33)
    load_min_deg: float = tunable(150.0)   # loaded only counts while the turret is at receive
    settle_speed: float = tunable(0.05)    # max brick/vault |lin vel| when judging (m/s)
    turret_settle_avel: float = tunable(0.30)  # max turret |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    vault_yaw_deg: float = tunable(20.0)   # vault yaw about nominal (+/- deg)
    vault_jitter: float = tunable(0.05)    # vault xy jitter (+/- m)
    theta0_range: tuple = tunable((0.0, 6.0))  # initial turret angle U(range) deg (on the seal)
    stand_bear_range: tuple = tunable((-50.0, 50.0))  # stand bearing about the vault front (deg)
    stand_r: float = tunable(0.55)         # stand distance from the vault centre (m)
    brick_jitter: float = tunable(0.015)   # brick xy jitter on the stand (+/- m)

    # --- info: layout (all vault-local; origin at footprint centre on the ground) ---------------
    vault_pos: tuple = info((0.35, 0.0))   # vault origin on the ground (nominal)
    vault_mass: float = info(80.0)
    half_out: float = info(0.15)
    half_in: float = info(0.135)
    roof_z0: float = info(0.403)
    roof_z1: float = info(0.418)
    win_half_y: float = info(0.05)         # deposit window half width
    win_z0: float = info(0.23)             # window sill (= shelf top: brick slides through)
    win_z1: float = info(0.29)             # window head
    shelf_z0: float = info(0.215)
    shelf_z1: float = info(0.23)
    hole_half: float = info(0.03)          # roof axle-hole half width
    # turret (local -x = pocket, +x = seal slab and crank)
    disc_r: float = info(0.12)
    disc_z0: float = info(0.38)
    disc_z1: float = info(0.395)
    wall_z0: float = info(0.237)           # pocket/slab lower edge (7 mm above the shelf)
    wall_z1: float = info(0.385)
    wall_t: float = info(0.012)
    pock_r0: float = info(0.065)
    pock_r1: float = info(0.125)
    pocket_center_deg: float = info(190.0) # pocket centre bearing (turret-local)
    pocket_half_deg: float = info(50.0)
    slab_x: float = info(0.10)             # seal slab centre plane
    slab_w: float = info(0.14)             # seal slab chord width
    axle_r: float = info(0.018)
    axle_z1: float = info(0.44)
    crank_r: float = info(0.09)            # knob orbit radius
    knob_r: float = info(0.014)
    receive_deg: float = info(170.0)       # receive stop (joint upper limit; wrap-safe < 180)
    turret_ang_damping: float = info(0.8)
    steel_density: float = info(2000.0)
    # brick / stand
    brick_l: float = info(0.12)
    brick_w: float = info(0.06)
    brick_h: float = info(0.03)
    brick_density: float = info(900.0)
    stand_h: float = info(0.10)
    insert_com_x: float = info(0.060)      # insertion target: brick CoM depth (vault-local x)
    # rubric weights (0.20 + 0.25 + 0.25 = 0.70 = the non-success cap)
    w_open: float = info(0.20)
    w_load: float = info(0.25)
    w_drop: float = info(0.25)
    open_norm_deg: float = info(160.0)     # open_frac = max theta / this, clamped to 1
    # colors / misc
    vault_color: tuple = info((0.30, 0.33, 0.38))
    steel_color: tuple = info((0.55, 0.57, 0.60))
    pocket_color: tuple = info((0.75, 0.62, 0.20))
    slab_color: tuple = info((0.20, 0.22, 0.60))
    knob_color: tuple = info((0.85, 0.15, 0.15))
    brick_color: tuple = info((0.16, 0.45, 0.18))
    stand_color: tuple = info((0.70, 0.62, 0.45))
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        """Audit the airlock geometry (all lengths in metres, angles in degrees)."""
        brick_min = min(self.brick_l, self.brick_w, self.brick_h)  # 0.03
        # window passes the brick lying flat with clearance
        assert 2 * self.win_half_y >= self.brick_w + 0.03
        assert (self.win_z1 - self.win_z0) >= self.brick_h + 0.02
        # window sill is the shelf top: the brick slides through onto the shelf
        assert abs(self.win_z0 - self.shelf_z1) < 1e-6
        # SEAL: at every initial angle the slab's chord covers the window's full
        # angular span (seen from the axis) with margin
        slab_half_ang = math.degrees(math.atan2(self.slab_w / 2, self.slab_x + self.wall_t / 2))
        win_half_ang = math.degrees(math.atan2(self.win_half_y, self.half_in))
        assert slab_half_ang >= win_half_ang + self.theta0_range[1] + 5.0, \
            (slab_half_ang, win_half_ang)
        # nothing passes BESIDE the slab: wall-to-slab-corner gap under the brick
        slab_sweep = math.hypot(self.slab_x + self.wall_t / 2, self.slab_w / 2)
        assert self.half_in - slab_sweep < brick_min - 0.005, (slab_sweep,)
        # ... but the sweep still clears the vault walls (with both contact offsets)
        assert self.half_in - slab_sweep >= 0.006
        # nothing passes UNDER the turret walls (wall bottom to shelf top gap)
        assert 0.0 < self.wall_z0 - self.shelf_z1 < brick_min - 0.01
        # nothing passes ABOVE: window head is below the slab top
        assert self.win_z1 <= self.wall_z1 - 0.05
        # roof hole is fully covered by the disc below (and the disc-roof gap is
        # far under the brick's minimum dimension)
        assert self.disc_r >= self.hole_half * math.sqrt(2.0) + 0.02
        assert 0.004 < self.roof_z0 - self.disc_z1 < brick_min - 0.01
        # disc clears the walls at every angle
        assert self.half_in - self.disc_r >= 0.010
        # pocket walls clear the window's angular span at receive (insertion is
        # free even with the receive stop's (180 - receive_deg) misalignment)
        assert self.pocket_half_deg >= win_half_ang + (180.0 - self.receive_deg) + 10.0
        # the receive stop stays clear of the PhysX +/-180 wrap boundary
        assert self.receive_deg <= 175.0
        assert self.receive_deg - 4.0 > self.load_min_deg
        # POCKET FACES THE WINDOW AT RECEIVE: with the pocket centred at
        # turret-local 360 - receive_deg, at theta = receive_deg the pocket's
        # centre bearing lands exactly on the window normal (vault +x)
        assert abs(self.pocket_center_deg + self.receive_deg - 360.0) < 1.0
        # INSERTION CORRIDOR: at receive the pocket walls sit at vault bearings
        # +/- pocket_half_deg (symmetric).  The wall inner-tip corner nearest
        # the corridor has |y| = r0 sin(ph) - (t/2) cos(ph); it must clear the
        # brick half-width + both contact offsets + margin, and since the wall
        # edge's |y| grows with x the whole wall then stays outside the corridor
        ph = math.radians(self.pocket_half_deg)
        tip_y = self.pock_r0 * math.sin(ph) - (self.wall_t / 2) * math.cos(ph)
        assert tip_y >= self.brick_w / 2 + 2 * self.contact_offset + 0.007, (tip_y,)
        # the loaded brick's tail clears the vault wall during the sweep, the
        # pocket walls' radial span overlaps the brick body (sweep engagement),
        # and `loaded` reads it
        assert self.insert_com_x + 0.008 + self.brick_l / 2 <= self.half_in - 0.006
        assert self.pock_r0 <= self.insert_com_x + self.brick_l / 2 - 0.04
        assert self.insert_com_x + 0.01 < self.pock_r1
        assert self.insert_com_x + 0.010 < self.load_r_max
        # pocket walls sweep clear of the vault walls
        assert self.half_in - self.pock_r1 >= 0.008
        # the bin volume lies strictly below the shelf (containment below the
        # aperture: judge z < sill - margin, never xy-margins at the doorway)
        assert self.bin_top_z <= self.shelf_z0 - 0.03
        # a brick resting anywhere IN the bin reads in-bin: tallest rest pose CoM
        assert 0.015 + self.brick_l / 2 < self.bin_top_z
        # bin xy bound covers the whole interior a resting brick can occupy
        assert self.bin_half_xy >= self.half_in - 0.02
        # knob is graspable by a Franka parallel jaw, standing proud of the roof
        assert 2 * self.knob_r < 0.06
        assert self.axle_z1 + 0.014 > self.roof_z1 + 0.02
        # crank bar (bottom face at axle_z1) sweeps clear above the roof
        assert self.axle_z1 > self.roof_z1 + 0.02
        # stand at every bearing stays clear of the vault footprint
        assert self.stand_r - 0.10 > math.hypot(self.half_out, self.half_out) + 0.10


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("deposit_vault")
class DepositVaultScene(BaseScene):
    cfg: DepositVaultSceneCfg

    def __init__(self, cfg: DepositVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or DepositVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        vault_spawn = cls["vault"](
            vault_mass=c.vault_mass, half_out=c.half_out, half_in=c.half_in,
            roof_z0=c.roof_z0, roof_z1=c.roof_z1, win_half_y=c.win_half_y,
            win_z0=c.win_z0, win_z1=c.win_z1, shelf_z0=c.shelf_z0,
            shelf_z1=c.shelf_z1, hole_half=c.hole_half, vault_color=c.vault_color,
            steel_color=c.steel_color, contact_offset=c.contact_offset)
        turret_spawn = cls["turret"](
            disc_r=c.disc_r, disc_z0=c.disc_z0, disc_z1=c.disc_z1,
            wall_z0=c.wall_z0, wall_z1=c.wall_z1, wall_t=c.wall_t,
            pock_r0=c.pock_r0, pock_r1=c.pock_r1,
            pocket_center_deg=c.pocket_center_deg, pocket_half_deg=c.pocket_half_deg,
            slab_x=c.slab_x, slab_w=c.slab_w, axle_r=c.axle_r, axle_z1=c.axle_z1,
            crank_r=c.crank_r, knob_r=c.knob_r, receive_deg=c.receive_deg,
            turret_ang_damping=c.turret_ang_damping, steel_density=c.steel_density,
            steel_color=c.steel_color, pocket_color=c.pocket_color,
            slab_color=c.slab_color, knob_color=c.knob_color,
            contact_offset=c.contact_offset)
        brick_spawn = cls["brick"](
            brick_l=c.brick_l, brick_w=c.brick_w, brick_h=c.brick_h,
            brick_density=c.brick_density, brick_color=c.brick_color,
            contact_offset=c.contact_offset)
        stand_spawn = cls["stand"](stand_h=c.stand_h, stand_color=c.stand_color,
                                   contact_offset=c.contact_offset)

        # template poses: the turret MUST spawn consistent with its authored joint
        # frames (vault at nominal pose, turret at joint angle 0 = sealed)
        px, py = c.vault_pos
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=vault_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "turret": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Turret",
                spawn=turret_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "brick": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brick",
                spawn=brick_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.stand_r, 0.0, c.stand_h + c.brick_h / 2 + 0.002)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + c.stand_r, 0.0, 0.0)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive bodies via set_external_force_and_torque;
                # without this flag wrenches are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.vault: RigidObject = env.iscene["vault"]
        self.turret: RigidObject = env.iscene["turret"]
        self.brick: RigidObject = env.iscene["brick"]
        self.stand: RigidObject = env.iscene["stand"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.theta0 = torch.zeros(n, device=dev)  # initial turret angle (deg)
        # latches (partial credit survives transients; success is judged live)
        self._open_max = torch.zeros(n, device=dev)          # max angle reached (deg)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._load_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._drop_cnt = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the vault (yaw + xy jitter), hang the turret on its
        pivot at a small random seal-stop angle (pose consistent with the joint
        frames — both origins sit ON the axis, so any angle is a pure pose write),
        place the stand on a random bearing off the vault front, lay the brick on
        it with jitter + free yaw, clear the latches. The whole linkage is written
        together (teleporting one body of a jointed pair gets depenetrated back by
        the other)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_deg)
        q_v = _qz(yaw)
        vp = torch.zeros(m, 3, device=dev)
        vp[:, 0] = c.vault_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        vp[:, 1] = c.vault_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = vp + origin
        st[:, 3:7] = q_v
        self.vault.write_root_state_to_sim(st, env_ids)

        # turret: joint angle theta0 ~ U(range) on the seal stop
        lo, hi = c.theta0_range
        th0 = lo + torch.rand(m, device=dev) * (hi - lo)
        self.theta0[env_ids] = th0
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = vp + origin
        st[:, 3:7] = _qmul(q_v, _qz(torch.deg2rad(th0)))
        self.turret.write_root_state_to_sim(st, env_ids)

        # stand: bearing off the vault front (+x), own free yaw
        b0, b1 = c.stand_bear_range
        bear = yaw + torch.deg2rad(b0 + torch.rand(m, device=dev) * (b1 - b0))
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0] = vp[:, 0] + c.stand_r * torch.cos(bear)
        sp[:, 1] = vp[:, 1] + c.stand_r * torch.sin(bear)
        q_s = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 3:7] = q_s
        self.stand.write_root_state_to_sim(st, env_ids)

        # brick: lying flat on the stand top, xy jitter + free yaw
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0:2] = sp[:, 0:2] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.brick_jitter
        bp[:, 2] = c.stand_h + c.brick_h / 2 + 0.002
        q_b = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_b
        self.brick.write_root_state_to_sim(st, env_ids)

        self._open_max[env_ids] = 0.0
        self._loaded[env_ids] = False
        self._load_cnt[env_ids] = 0
        self._dropped[env_ids] = False
        self._drop_cnt[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "turret": self.turret.data.root_state_w[env_ids].clone(),
            "brick": self.brick.data.root_state_w[env_ids].clone(),
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "theta0": self.theta0[env_ids].clone(),
            "open_max": self._open_max[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "load_cnt": self._load_cnt[env_ids].clone(),
            "dropped": self._dropped[env_ids].clone(),
            "drop_cnt": self._drop_cnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.turret.write_root_state_to_sim(state["turret"], env_ids)
        self.brick.write_root_state_to_sim(state["brick"], env_ids)
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.theta0[env_ids] = state["theta0"]
        self._open_max[env_ids] = state["open_max"]
        self._loaded[env_ids] = state["loaded"]
        self._load_cnt[env_ids] = state["load_cnt"]
        self._dropped[env_ids] = state["dropped"]
        self._drop_cnt[env_ids] = state["drop_cnt"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A dark steel-grey DEPOSIT VAULT (a 300 x 300 mm box, 420 mm tall) "
            "stands on the ground. Its interior is sealed on every face; the only "
            "opening is a letterbox DEPOSIT WINDOW on the front wall (100 mm wide, "
            "60 mm tall, sill 230 mm up). Behind the window, on a vertical axle "
            "through the vault centre, turns a TRANSFER TURRET: a revolving hatch "
            "with ONE open pocket (two amber radial walls, 100 deg apart) and, "
            "opposite the pocket, a BLUE SEAL PLATE. At rest the turret sits on its "
            "SEAL STOP: the blue plate stands directly behind the window (you can "
            "see it through the slot) and the pocket faces the sealed rear — "
            "nothing can be pushed in. The turret is driven by the CRANK on the "
            "roof: a steel bar with a RED KNOB that orbits the axle on a 90 mm "
            "radius. The crank points AT the window when sealed and AWAY from it "
            "at the receive stop; the turret turns freely between its two stops "
            "(0 and 170 deg) and stays where it is left. Under the turret, at "
            "window-sill height, a steel TRANSFER SHELF spans only the FRONT half "
            "of the interior; behind its straight edge the vault drops away into "
            "the BIN below. On a small tan stand in front of the vault lies the "
            "CASH BRICK: a green banded block, 120 x 60 x 30 mm.\n"
            "Goal: bank the brick — get it into the bin (the chamber below the "
            "shelf) and leave the hatch resealed. The only working order: turn the "
            "crank to the receive stop (just under half a turn) so the pocket opens to the "
            "window; push the brick through the window all the way into the "
            "pocket (long side leading; it rests on the shelf between the amber "
            "walls); then crank back to the seal stop — the pocket walls carry the "
            "brick around past the shelf edge, it falls into the bin, and the blue "
            "plate returns behind the window. A brick left in the pocket, on the "
            "shelf, in the window or anywhere outside the bin does not count, and "
            "the vault is not banked until the turret is back on its seal stop "
            f"(within {c.sealed_max_deg:.0f} deg) with everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Crank the red-knobbed roof handle to its far stop so the turret pocket "
            "opens to the vault's front window, push the green cash brick through "
            "the window fully into the pocket, then crank back to the seal stop so "
            "the turret carries the brick over the shelf edge and drops it into "
            "the bin below. The task fails unless the brick ends inside the lower "
            "bin chamber and the turret is back on its seal stop."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.vault.data.root_quat_w,
                                  pos_w - self.vault.data.root_pos_w)

    def turret_angle_deg(self) -> torch.Tensor:
        """(N,) float: turret joint angle in degrees (0 = seal stop, +receive_deg
        = receive stop). No joint-state API exists on a plain spawn-authored USD
        joint; this is the pivot readout, mapped to (-90, 270) so the 180 deg
        receive stop never wraps."""
        qv = self.vault.data.root_quat_w
        qt = self.turret.data.root_quat_w
        qv_inv = qv * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qv.device)
        rel = _qmul(qv_inv, qt)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 3], rel[:, 0]))
        ang = torch.remainder(ang + 90.0, 360.0) - 90.0
        return ang

    def brick_in_bin(self) -> torch.Tensor:
        """(N,) bool, geometric: brick CoM inside the bin volume — within the
        interior footprint AND below the shelf with margin (containment below the
        aperture: judged on z, so a brick on the shelf/in the window never
        counts)."""
        c = self.cfg
        p = self._vault_local(self.brick.data.root_pos_w)
        return (p[:, 0].abs() < c.bin_half_xy) & (p[:, 1].abs() < c.bin_half_xy) \
            & (p[:, 2] > 0.005) & (p[:, 2] < c.bin_top_z)

    def brick_loaded(self) -> torch.Tensor:
        """(N,) bool: brick at shelf level within the pocket radius while the
        turret is at receive — the mid-plan state the load latch persists on."""
        c = self.cfg
        p = self._vault_local(self.brick.data.root_pos_w)
        r = p[:, :2].norm(dim=-1)
        return (r < c.load_r_max) & (p[:, 2] > c.load_z0) & (p[:, 2] < c.load_z1) \
            & (self.turret_angle_deg() > c.load_min_deg)

    def sealed(self) -> torch.Tensor:
        """(N,) bool: turret back on its seal stop."""
        return self.turret_angle_deg() <= self.cfg.sealed_max_deg

    def settled(self) -> torch.Tensor:
        """(N,) bool: brick, turret swing and vault all still."""
        c = self.cfg
        return (self.brick.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.turret.data.root_ang_vel_w.norm(dim=-1) < c.turret_settle_avel) \
            & (self.vault.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.vault, self.turret, self.brick)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        fin = self._finite()
        ang = self.turret_angle_deg().clamp(min=0.0)
        self._open_max = torch.where(fin, torch.maximum(self._open_max, ang),
                                     self._open_max)
        ld = self.brick_loaded() & fin
        self._load_cnt = torch.where(ld, self._load_cnt + 1,
                                     torch.zeros_like(self._load_cnt))
        self._loaded |= self._load_cnt >= 3
        dr = self.brick_in_bin() & fin
        self._drop_cnt = torch.where(dr, self._drop_cnt + 1,
                                     torch.zeros_like(self._drop_cnt))
        self._dropped |= self._drop_cnt >= 3

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the brick is BANKED — settled inside the bin volume with the
        turret back on its seal stop, everything still and finite. Both clauses
        are live physical outcomes: the bin is only reachable through the turret
        cycle (every other path is refused by authored geometry, shown in
        smoke)."""
        return self.brick_in_bin() & self.sealed() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*open_frac + 0.25*loaded + 0.25*dropped (all
        latched; ~0 for the null policy — the turret spawns within 6 deg of its
        seal stop and the brick on its stand), capped at 0.70 — and exactly 1.0
        iff success() holds live."""
        c = self.cfg
        open_frac = (self._open_max / c.open_norm_deg).clamp(0.0, 1.0)
        base = (c.w_open * open_frac
                + c.w_load * self._loaded.float()
                + c.w_drop * self._dropped.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="deposit_vault", robot="null"))
