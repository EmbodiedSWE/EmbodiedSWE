"""SeesawVaultScene — deposit the cash brick through a counterweight-tolled roof lid.

Derived from rlbench/put_money_in_safe ("put the money away in the safe on the top
shelf": a dollar stack and a safe whose door already stands open; the demonstrated
strategy is ONE grasp-transport-place onto a shelf inside a static, always-open
receptacle). Here that plan is impossible and the required plan is a THIRD-HAND
puzzle: the vault has no door and no open face — the only way in is a roof DEPOSIT
MOUTH covered by a heavy LID on a gravity-biased SEE-SAW lever, and a single
gripper can never hold the lid open and insert the money at the same time:

  1. PARK the brass COUNTERWEIGHT on the see-saw's pedal pan (overhanging the
     vault front): its standing weight — not the hand — back-drives the lever and
     holds the lid open on its 60 deg stop. The cash brick itself is too light to
     open the lid (2x torque margin, proven by a real probe in smoke), so the
     dedicated counterweight is the only tool that works.
  2. DROP the CASH BRICK through the uncovered rear half of the mouth; it falls
     into the vault chamber and settles on the vault floor (pure contact
     dynamics; a closed lid physically refuses the brick — its hover-gap is far
     under the brick's minimum dimension and the closed stop is a joint limit).
  3. UNPARK the counterweight (lift it off the pan): the lid falls shut under its
     own gravity bias and the mouth is sealed again.

The goal state is a PHYSICAL outcome pair: brick settled INSIDE the vault chamber
(below the roof, judged on z — containment below the aperture) AND the lid back on
its closed stop. Execution order is forced by gravity: the closed lid refuses the
brick, only the parked counterweight holds it open, and the lid can only close
again once the counterweight is off the pan.

Assets are fully procedural (compound spawners; per-child density so the lever's
torque budget is real; root MassAPI on the heavy vault — memory: custom spawners
apply no cfg schemas, so mass/collision are authored in the funcs):
  - vault: heavy DYNAMIC box (footprint 0.34 x 0.34 m, roof top at 0.275 m,
    15 mm walls, 60 kg, origin at footprint centre on the ground; +x = front).
    The roof carries the deposit MOUTH (x -0.115..0.075, y +/-0.075) and two
    small hinge posts. Dynamic, not kinematic: a joint anchored to a kinematic
    body0 stays world-fixed when the body is teleported at reset.
  - lever (rotor): DYNAMIC compound on a spawn-authored RevoluteJoint (axis y,
    through (0.17, 0, 0.296) vault-local — DIRECTLY OVER THE FRONT WALL, so the
    whole pedal side swings in free air and never sweeps the vault's own roof;
    limits -0.5 deg = CLOSED stop, +60 deg = OPEN stop; positive angle lifts
    the lid and sinks the pedal). LID plate (295 x 170 x 10 mm) reaches back
    from the axis and hovers 13 mm over the roof covering the whole mouth at
    closed; pedal bar + fenced PAN (radius 0.06..0.16 from the axis) hang in
    free air in front. Torque budget (asserted in __post_init__): lid bias
    ~0.60 N.m closed vs pedal ~0.21 N.m — the 0.24 kg brick on the pan
    (<= 0.40 N.m even at the outer fence) CANNOT open the lid; the 1.28 kg
    weight (~1.38 N.m) opens it with >= 1.9x margin and holds it on the open
    stop. Anything set ON the lid presses it MORE closed (the lid is the
    closing arm), so no load on the roof side is a loophole.
  - brick: the cash brick, banded green block 120 x 60 x 30 mm, ~0.24 kg.
  - weight: the brass counterweight, a 48 mm dia x 70 mm cylinder with a 20 mm
    grasp knob, ~1.28 kg (well inside Franka payload).
  - two KINEMATIC stands (brick teller stand / weight pedestal) at mirrored
    random bearings in front of the vault; which side holds which is sampled.

Per-episode randomization (readback-verifiable): vault yaw +/-20 deg + xy jitter,
stand bearings U(15, 35) deg magnitude with the brick/weight SIDES swapped by a
coin flip (torch.rand comparison — first-randint degeneracy memory), per-object
xy jitter and free yaw for the brick.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.25 * open_frac    — latched max lid angle / 55 deg (the toll was paid)
  0.35 * dropped_in   — brick inside the vault chamber (latched, 3-step streak)
  0.15 * closed_after — lid back under closed_max_deg AFTER dropped_in latched
                        (latched, 3-step streak)
  1.0 iff success()   — brick settled inside AND lid on its closed stop
                        (<= 5 deg), everything still and finite. Non-success
                        capped at 0.75; null policy ~0 (lid rests on its stop).

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
             density: float | None = None):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable,
             axis: str = "Z", density: float | None = None):
    """One cylinder child (axis Z or Y)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    if axis == "Z":
        ext = [Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)]
    else:  # Y
        ext = [Gf.Vec3f(-radius, -height / 2, -radius), Gf.Vec3f(radius, height / 2, radius)]
    cyl.CreateExtentAttr(ext)
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
    centre on the ground; +x = front (pedal side); the mouth is in the roof."""
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
    body = c.vault_color
    hx, hy = c.half_x, c.half_y
    wt = c.wall_t
    zr0, zr1 = c.roof_z0, c.roof_z1
    m_x0, m_x1, m_hy = c.mouth_x0, c.mouth_x1, c.mouth_half_y
    # floor
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.0075),
             size=(2 * hx, 2 * hy, 0.015), color=body, collide=collide)
    # side walls (y = +/-)
    for name, yc in (("wall_yp", hy - wt / 2), ("wall_yn", -(hy - wt / 2))):
        _add_box(stage, f"{prim_path}/{name}", center=(0.0, yc, (0.015 + zr0) / 2),
                 size=(2 * hx, wt, zr0 - 0.015), color=body, collide=collide)
    # front / rear walls
    for name, xc in (("wall_front", hx - wt / 2), ("wall_rear", -(hx - wt / 2))):
        _add_box(stage, f"{prim_path}/{name}", center=(xc, 0.0, (0.015 + zr0) / 2),
                 size=(wt, 2 * (hy - wt), zr0 - 0.015), color=body, collide=collide)
    # roof around the mouth: front strip / back strip / two side strips
    _add_box(stage, f"{prim_path}/roof_front", center=((m_x1 + hx) / 2, 0.0, (zr0 + zr1) / 2),
             size=(hx - m_x1, 2 * hy, zr1 - zr0), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_back", center=((m_x0 - hx) / 2, 0.0, (zr0 + zr1) / 2),
             size=(hx + m_x0, 2 * hy, zr1 - zr0), color=body, collide=collide)
    for name, yc in (("roof_yp", (m_hy + hy) / 2), ("roof_yn", -(m_hy + hy) / 2)):
        _add_box(stage, f"{prim_path}/{name}",
                 center=((m_x0 + m_x1) / 2, yc, (zr0 + zr1) / 2),
                 size=(m_x1 - m_x0, hy - m_hy, zr1 - zr0), color=body, collide=collide)
    # hinge posts (static side of the bearing; the joint does the constraint),
    # seated on the roof front strip just behind the axis
    for name, yc in (("post_yp", c.post_y), ("post_yn", -c.post_y)):
        _add_box(stage, f"{prim_path}/{name}",
                 center=(c.axis_x - 0.008, yc, (zr1 + c.axis_z + 0.012) / 2),
                 size=(0.024, 0.020, c.axis_z + 0.012 - zr1),
                 color=c.steel_color, collide=collide)
    return root


def _spawn_lever(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The see-saw lever (rotor): DYNAMIC compound in the VAULT's local frame
    (same root frame as the vault), plus the spawn-authored RevoluteJoint to the
    sibling vault (axis y through (axis_x, 0, axis_z); positive angle lifts the
    lid, sinks the pedal). Masses via per-child DENSITY (true CoM: the torque
    budget IS the mechanism)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.lever_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    ax, az = c.axis_x, c.axis_z
    # hub on the axis
    _add_cyl(stage, f"{prim_path}/hub", center=(ax, 0.0, az), radius=0.012,
             height=2 * c.hub_half_len, color=c.steel_color, collide=collide,
             axis="Y", density=c.frame_density)
    # LID plate (dense: the closed gravity bias)
    _add_box(stage, f"{prim_path}/lid",
             center=(c.lid_cx, 0.0, az - 0.003),
             size=(c.lid_len, 2 * c.lid_half_y, c.lid_t), color=c.lid_color,
             collide=collide, density=c.lid_density)
    # pedal bar
    _add_box(stage, f"{prim_path}/bar",
             center=((c.bar_x0 + c.bar_x1) / 2, 0.0, az),
             size=(c.bar_x1 - c.bar_x0, 0.03, 0.010), color=c.steel_color,
             collide=collide, density=c.frame_density)
    # PAN: floor + 4 fences (open top)
    px0, px1 = c.pan_x0, c.pan_x1
    pxc = (px0 + px1) / 2
    _add_box(stage, f"{prim_path}/pan_floor", center=(pxc, 0.0, az - 0.020),
             size=(px1 - px0, 2 * c.pan_half_y, 0.008), color=c.pan_color,
             collide=collide, density=c.frame_density)
    for name, xc in (("pan_fin", px0 + 0.004), ("pan_fout", px1 - 0.004)):
        _add_box(stage, f"{prim_path}/{name}", center=(xc, 0.0, az - 0.001),
                 size=(0.008, 2 * c.pan_half_y, c.fence_h), color=c.pan_color,
                 collide=collide, density=c.frame_density)
    for name, yc in (("pan_syp", c.pan_half_y - 0.004), ("pan_syn", -(c.pan_half_y - 0.004))):
        _add_box(stage, f"{prim_path}/{name}", center=(pxc, yc, az - 0.001),
                 size=(px1 - px0, 0.008, c.fence_h), color=c.pan_color,
                 collide=collide, density=c.frame_density)

    # revolute joint to the sibling vault, axis y through (axis_x, 0, axis_z)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # lever<->vault contact stays ON (USD default for a joint pair is filtered);
    # clearances are authored (lid hovers 13 mm over the roof at closed; the
    # axis sits over the front wall so the pedal side swings only free air —
    # asserted in __post_init__)
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(ax), 0.0, float(az)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(float(ax), 0.0, float(az)))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # joint angle: 0 = CLOSED stop (lid over the mouth), +open_deg = OPEN stop
    j.CreateLowerLimitAttr(-0.5)
    j.CreateUpperLimitAttr(float(c.open_deg) + 0.5)
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
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(c.brick_l, c.brick_w, c.brick_h), color=c.brick_color,
             collide=collide, density=c.brick_density)
    _add_box(stage, f"{prim_path}/band", center=(0.0, 0.0, 0.0),
             size=(c.brick_l * 0.25, c.brick_w + 0.0004, c.brick_h + 0.0004),
             color=(0.92, 0.90, 0.82), collide=collide, density=c.brick_density)
    return root


def _spawn_weight(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The brass counterweight: 44 mm dia x 60 mm body + 20 mm grasp knob,
    ~0.9 kg via density. Origin at the BASE centre (sits at its resting z)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.2)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_cyl(stage, f"{prim_path}/body", center=(0.0, 0.0, c.w_body_h / 2),
             radius=c.w_body_r, height=c.w_body_h, color=c.weight_color,
             collide=collide, density=c.brass_density)
    _add_cyl(stage, f"{prim_path}/knob", center=(0.0, 0.0, c.w_body_h + c.w_knob_h / 2),
             radius=c.w_knob_r, height=c.w_knob_h, color=c.weight_color,
             collide=collide, density=c.brass_density)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A kinematic pedestal (brick teller stand / weight pedestal)."""
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
            vault_mass: float = 60.0
            half_x: float = 0.17
            half_y: float = 0.17
            wall_t: float = 0.015
            roof_z0: float = 0.26
            roof_z1: float = 0.275
            mouth_x0: float = -0.115
            mouth_x1: float = 0.075
            mouth_half_y: float = 0.075
            axis_x: float = 0.17
            axis_z: float = 0.296
            post_y: float = 0.105
            vault_color: tuple = (0.30, 0.33, 0.38)
            steel_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.002

        @configclass
        class LeverSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lever)
            axis_x: float = 0.17
            axis_z: float = 0.296
            hub_half_len: float = 0.085
            lid_cx: float = 0.0225
            lid_len: float = 0.295
            lid_half_y: float = 0.085
            lid_t: float = 0.010
            lid_density: float = 1120.0
            bar_x0: float = 0.176
            bar_x1: float = 0.23
            pan_x0: float = 0.23
            pan_x1: float = 0.33
            pan_half_y: float = 0.055
            fence_h: float = 0.030
            frame_density: float = 1000.0
            open_deg: float = 60.0
            lever_ang_damping: float = 0.6
            steel_color: tuple = (0.55, 0.57, 0.60)
            lid_color: tuple = (0.20, 0.22, 0.60)
            pan_color: tuple = (0.75, 0.62, 0.20)
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
        class WeightSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_weight)
            w_body_r: float = 0.024
            w_body_h: float = 0.070
            w_knob_r: float = 0.010
            w_knob_h: float = 0.030
            brass_density: float = 9400.0
            weight_color: tuple = (0.78, 0.62, 0.22)
            contact_offset: float = 0.002

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            stand_h: float = 0.10
            stand_color: tuple = (0.70, 0.62, 0.45)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg
        _SPAWNER_CACHE["lever"] = LeverSpawnerCfg
        _SPAWNER_CACHE["brick"] = BrickSpawnerCfg
        _SPAWNER_CACHE["weight"] = WeightSpawnerCfg
        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class SeesawVaultSceneCfg(BaseCfg):
    """Config for `SeesawVaultScene`. The see-saw toll is honest by construction —
    the torque budget and every geometric clause are asserted numerically in
    __post_init__: the closed lid covers the whole mouth and its hover-gap passes
    nothing; the brick cannot pay the toll, the counterweight can; the open lid
    uncovers a drop zone that passes the brick flat; the pan swings clear of the
    vault at every angle."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_max_deg: float = tunable(5.0)   # lid angle at/below this counts as closed
    open_min_deg: float = tunable(48.0)    # lid angle at/above this counts as held open
    in_half_x: float = tunable(0.150)      # brick CoM inside |x| (vault local)
    in_half_y: float = tunable(0.150)      # brick CoM inside |y|
    in_top_z: float = tunable(0.20)        # brick CoM below this = inside the chamber
    settle_speed: float = tunable(0.05)    # max |lin vel| (brick/weight/vault) when judging
    lever_settle_avel: float = tunable(0.30)  # max lever |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    vault_yaw_deg: float = tunable(20.0)   # vault yaw about nominal (+/- deg)
    vault_jitter: float = tunable(0.04)    # vault xy jitter (+/- m)
    bear_range: tuple = tunable((15.0, 35.0))  # stand bearing magnitude off the vault front (deg)
    stand_r: float = tunable(0.62)         # stand distance from the vault centre (m)
    obj_jitter: float = tunable(0.015)     # brick/weight xy jitter on their stands (+/- m)

    # --- info: vault (all vault-local; origin at footprint centre on the ground; +x front) ------
    vault_pos: tuple = info((0.0, 0.0))
    vault_mass: float = info(60.0)
    half_x: float = info(0.17)
    half_y: float = info(0.17)
    wall_t: float = info(0.015)
    roof_z0: float = info(0.26)
    roof_z1: float = info(0.275)
    mouth_x0: float = info(-0.115)         # mouth x span (back .. front edge)
    mouth_x1: float = info(0.075)
    mouth_half_y: float = info(0.075)
    axis_x: float = info(0.17)             # hinge axis (y-parallel) — over the front wall
    axis_z: float = info(0.296)            # hinge axis height
    post_y: float = info(0.105)
    # lever
    hub_half_len: float = info(0.085)
    lid_cx: float = info(0.0225)           # lid plate centre x (covers the mouth)
    lid_len: float = info(0.295)
    lid_half_y: float = info(0.085)
    lid_t: float = info(0.010)
    lid_density: float = info(1120.0)
    bar_x0: float = info(0.176)
    bar_x1: float = info(0.23)
    pan_x0: float = info(0.23)             # pan span (fully in front of the wall)
    pan_x1: float = info(0.33)
    pan_half_y: float = info(0.055)
    fence_h: float = info(0.030)
    frame_density: float = info(1000.0)
    open_deg: float = info(60.0)           # open stop (joint upper limit)
    lever_ang_damping: float = info(0.6)
    # brick / weight / stands
    brick_l: float = info(0.12)
    brick_w: float = info(0.06)
    brick_h: float = info(0.03)
    brick_density: float = info(900.0)
    w_body_r: float = info(0.024)
    w_body_h: float = info(0.070)
    w_knob_r: float = info(0.010)
    w_knob_h: float = info(0.030)
    brass_density: float = info(9400.0)
    stand_h: float = info(0.10)
    # solve/probe targets
    drop_x: float = info(-0.070)           # drop-zone centre (vault-local x; long axis on y)
    drop_hover_z: float = info(0.42)       # hover height for the release
    # rubric weights (0.25 + 0.35 + 0.15 = 0.75 = the non-success cap)
    w_open: float = info(0.25)
    w_in: float = info(0.35)
    w_close: float = info(0.15)
    open_norm_deg: float = info(55.0)      # open_frac = max angle / this, clamped to 1
    # colors / misc
    vault_color: tuple = info((0.30, 0.33, 0.38))
    steel_color: tuple = info((0.55, 0.57, 0.60))
    lid_color: tuple = info((0.20, 0.22, 0.60))
    pan_color: tuple = info((0.75, 0.62, 0.20))
    brick_color: tuple = info((0.16, 0.45, 0.18))
    weight_color: tuple = info((0.78, 0.62, 0.22))
    stand_color: tuple = info((0.70, 0.62, 0.45))
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        """Audit the see-saw toll (lengths m, angles deg, torques N.m)."""
        g = 9.81
        brick_min = min(self.brick_l, self.brick_w, self.brick_h)   # 0.03
        # --- closed lid seals the mouth ---
        lid_x0 = self.lid_cx - self.lid_len / 2
        lid_x1 = self.lid_cx + self.lid_len / 2
        assert lid_x0 <= self.mouth_x0 - 0.008 and lid_x1 >= self.mouth_x1 + 0.008
        assert self.lid_half_y >= self.mouth_half_y + 0.008
        # hover gap under the closed lid passes nothing (lid bottom vs roof top)
        lid_bot = self.axis_z - 0.003 - self.lid_t / 2
        gap = lid_bot - self.roof_z1
        assert 0.004 < gap < brick_min - 0.015, (gap,)
        # lid clears the hinge posts sideways
        assert self.lid_half_y <= self.post_y - 0.010 - 0.008
        assert self.hub_half_len <= self.post_y - 0.010 - 0.008
        # --- the pedal side sweeps only FREE AIR (the i-9deg jam regression) ---
        # axis directly over the front wall: everything at x > axis stays outside
        # the footprint at every angle, and the swung bar/hub clear the wall-top
        # corner that sits (axis_z - roof_z1) straight below the axis
        assert abs(self.axis_x - self.half_x) < 1e-6
        assert self.bar_x0 >= self.half_x + 0.005 and self.pan_x0 >= self.bar_x0
        d_corner = self.axis_z - self.roof_z1
        assert d_corner * math.cos(math.radians(self.open_deg)) > 0.005 + 0.004
        assert d_corner > 0.012 + 0.004  # hub radius + margin
        # --- open lid uncovers a drop zone that passes the brick FLAT ---
        # lid reach behind the axis, projected at the open stop
        reach = self.axis_x - lid_x0
        proj = self.axis_x - reach * math.cos(math.radians(self.open_deg))
        clear_len = proj - self.mouth_x0          # uncovered mouth length (back part)
        assert clear_len >= self.brick_w + 0.030, (clear_len,)
        assert 2 * self.mouth_half_y >= self.brick_l + 0.024
        # drop target sits centred in the clear zone with margin
        assert self.mouth_x0 + self.brick_w / 2 + 0.010 < self.drop_x < proj - self.brick_w / 2 - 0.010
        # hover is above the roof and under the raised lid tip's swept cylinder is avoided:
        # the drop column (x < proj) is free air up to any height (lid occupies x >= proj)
        assert self.drop_hover_z > self.roof_z1 + 0.10
        # --- torque budget (per-child density -> component masses) ---
        m_lid = self.lid_len * 2 * self.lid_half_y * self.lid_t * self.lid_density
        r_lid = abs(self.lid_cx - self.axis_x)
        tau_lid = m_lid * g * r_lid
        m_bar = (self.bar_x1 - self.bar_x0) * 0.03 * 0.010 * self.frame_density
        r_bar = (self.bar_x0 + self.bar_x1) / 2 - self.axis_x
        m_floor = (self.pan_x1 - self.pan_x0) * 2 * self.pan_half_y * 0.008 * self.frame_density
        r_pan = (self.pan_x0 + self.pan_x1) / 2 - self.axis_x
        m_fence = (2 * 0.008 * 2 * self.pan_half_y * self.fence_h
                   + 2 * (self.pan_x1 - self.pan_x0) * 0.008 * self.fence_h) * self.frame_density
        tau_pedal = m_bar * g * r_bar + (m_floor + m_fence) * g * r_pan
        tau_bias = tau_lid - tau_pedal            # net CLOSED bias, pedal empty
        assert tau_bias > 0.50, (tau_bias,)
        # the BRICK cannot pay the toll even at the pan's outer fence
        m_brick = self.brick_l * self.brick_w * self.brick_h * self.brick_density * 1.31
        # (x1.31: the overlapping band box also carries density — full-size in y/z,
        # 0.25 l long, ~+26% mass; the readback in solve asserts the real sum)
        tau_brick = m_brick * g * (self.pan_x1 - self.axis_x)
        assert tau_brick < tau_bias * 0.80, (tau_brick, tau_bias)
        # the WEIGHT opens with margin (at the pan centre radius)
        m_w = (math.pi * self.w_body_r ** 2 * self.w_body_h
               + math.pi * self.w_knob_r ** 2 * self.w_knob_h) * self.brass_density
        tau_w = m_w * g * r_pan
        assert tau_w > tau_bias * 1.9, (tau_w, tau_bias)
        # ... and still holds on the open stop with the weight against the outer
        # fence (worst case: same cos factor on both sides, radius grows -> holds)
        assert m_w * g * (self.pan_x0 + 0.03 - self.axis_x) > tau_bias * 1.2
        # --- pan swings clear of the vault at every angle ---
        r_pan_in = self.pan_x0 - self.axis_x
        th = math.radians(self.open_deg)
        assert self.axis_x + r_pan_in * math.cos(th) > self.half_x + 0.004
        # pan bottom stays above the ground at the open stop
        pan_low_z = self.axis_z - (self.pan_x1 - self.axis_x) * math.sin(th) - 0.05
        assert pan_low_z > 0.02, (pan_low_z,)
        # weight fits the pan cage loosely (fence inner faces sit 8 mm in per side)
        assert (self.pan_x1 - self.pan_x0) - 0.016 > 2 * self.w_body_r + 0.020
        assert 2 * self.pan_half_y - 0.016 > 2 * self.w_body_r + 0.020
        # --- containment threshold: below the aperture, above any rest pose ---
        assert self.in_top_z <= self.roof_z0 - 0.05
        assert 0.015 + self.brick_l / 2 < self.in_top_z
        assert self.in_half_x <= self.half_x - self.wall_t - 0.004
        assert self.in_half_y <= self.half_y - self.wall_t - 0.004
        # --- stands clear the vault and the pan's swing ---
        b_lo = math.radians(self.bear_range[0])
        reach_pan = math.hypot(self.pan_x1, self.pan_half_y + 0.01)
        assert self.stand_r * math.cos(math.radians(self.bear_range[1])) - 0.08 \
            > max(self.half_x, reach_pan) + 0.005
        assert self.stand_r * math.sin(b_lo) - 0.08 > self.pan_half_y + 0.004
        # open/closed thresholds inside the travel
        assert self.open_min_deg < self.open_deg - 8.0
        assert self.closed_max_deg < self.open_min_deg / 4


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("seesaw_vault")
class SeesawVaultScene(BaseScene):
    cfg: SeesawVaultSceneCfg

    def __init__(self, cfg: SeesawVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or SeesawVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        vault_spawn = cls["vault"](
            vault_mass=c.vault_mass, half_x=c.half_x, half_y=c.half_y, wall_t=c.wall_t,
            roof_z0=c.roof_z0, roof_z1=c.roof_z1, mouth_x0=c.mouth_x0,
            mouth_x1=c.mouth_x1, mouth_half_y=c.mouth_half_y, axis_x=c.axis_x,
            axis_z=c.axis_z, post_y=c.post_y, vault_color=c.vault_color,
            steel_color=c.steel_color, contact_offset=c.contact_offset)
        lever_spawn = cls["lever"](
            axis_x=c.axis_x, axis_z=c.axis_z, hub_half_len=c.hub_half_len,
            lid_cx=c.lid_cx, lid_len=c.lid_len, lid_half_y=c.lid_half_y,
            lid_t=c.lid_t, lid_density=c.lid_density, bar_x0=c.bar_x0,
            bar_x1=c.bar_x1, pan_x0=c.pan_x0, pan_x1=c.pan_x1,
            pan_half_y=c.pan_half_y, fence_h=c.fence_h, frame_density=c.frame_density,
            open_deg=c.open_deg, lever_ang_damping=c.lever_ang_damping,
            steel_color=c.steel_color, lid_color=c.lid_color, pan_color=c.pan_color,
            contact_offset=c.contact_offset)
        brick_spawn = cls["brick"](
            brick_l=c.brick_l, brick_w=c.brick_w, brick_h=c.brick_h,
            brick_density=c.brick_density, brick_color=c.brick_color,
            contact_offset=c.contact_offset)
        weight_spawn = cls["weight"](
            w_body_r=c.w_body_r, w_body_h=c.w_body_h, w_knob_r=c.w_knob_r,
            w_knob_h=c.w_knob_h, brass_density=c.brass_density,
            weight_color=c.weight_color, contact_offset=c.contact_offset)
        stand_spawn = cls["stand"](stand_h=c.stand_h, stand_color=c.stand_color,
                                   contact_offset=c.contact_offset)

        # template poses: the lever MUST spawn consistent with its authored joint
        # frames (vault at nominal pose, lever at joint angle 0 = closed)
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
            "lever": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lever",
                spawn=lever_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "brick": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brick",
                spawn=brick_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.stand_r * 0.9, 0.20, c.stand_h + c.brick_h / 2 + 0.002)),
            ),
            "weight": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Weight",
                spawn=weight_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.stand_r * 0.9, -0.20, c.stand_h + 0.002)),
            ),
            "stand_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/StandA",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + c.stand_r * 0.9, 0.20, 0.0)),
            ),
            "stand_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/StandB",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + c.stand_r * 0.9, -0.20, 0.0)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # smoke drives bodies via set_external_force_and_torque; without
                # this flag wrenches are under-applied across TGS iterations
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
        self.lever: RigidObject = env.iscene["lever"]
        self.brick: RigidObject = env.iscene["brick"]
        self.weight: RigidObject = env.iscene["weight"]
        self.stand_a: RigidObject = env.iscene["stand_a"]
        self.stand_b: RigidObject = env.iscene["stand_b"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.brick_on_a = torch.ones(n, dtype=torch.bool, device=dev)  # side sample
        # latches (partial credit survives transients; success is judged live)
        self._open_max = torch.zeros(n, device=dev)          # max lid angle reached (deg)
        self._in = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._closed_after = torch.zeros(n, dtype=torch.bool, device=dev)
        self._close_cnt = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the vault (yaw + xy jitter) and the lever WITH it
        at joint angle 0 (both roots share one frame — the whole linkage is
        written together; teleporting one body of a jointed pair gets
        depenetrated back by the other). Stands at mirrored random bearings off
        the vault front; the brick/weight SIDES are sampled by a coin flip
        (torch.rand comparison); brick with free yaw + jitter, weight upright
        with jitter. Latches cleared."""
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
        # lever: same root frame, joint angle 0 (rests on its closed stop)
        self.lever.write_root_state_to_sim(st.clone(), env_ids)

        # stands: mirrored bearings off the vault front (+x), magnitudes sampled
        b0, b1 = (math.radians(v) for v in c.bear_range)
        mag_a = b0 + torch.rand(m, device=dev) * (b1 - b0)
        mag_b = b0 + torch.rand(m, device=dev) * (b1 - b0)
        bear_a = yaw + mag_a
        bear_b = yaw - mag_b
        pa = torch.zeros(m, 3, device=dev)
        pa[:, 0] = vp[:, 0] + c.stand_r * torch.cos(bear_a)
        pa[:, 1] = vp[:, 1] + c.stand_r * torch.sin(bear_a)
        pb = torch.zeros(m, 3, device=dev)
        pb[:, 0] = vp[:, 0] + c.stand_r * torch.cos(bear_b)
        pb[:, 1] = vp[:, 1] + c.stand_r * torch.sin(bear_b)
        for stand, p in ((self.stand_a, pa), (self.stand_b, pb)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = p + origin
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            stand.write_root_state_to_sim(st, env_ids)

        # which stand holds the brick (coin flip; the weight takes the other)
        side = torch.rand(m, device=dev) > 0.5
        self.brick_on_a[env_ids] = side
        p_brick = torch.where(side.unsqueeze(1), pa, pb)
        p_weight = torch.where(side.unsqueeze(1), pb, pa)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = p_brick[:, 0:2] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.obj_jitter
        st[:, 2] = c.stand_h + c.brick_h / 2 + 0.002
        st[:, 0:3] += origin
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.brick.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = p_weight[:, 0:2] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.obj_jitter
        st[:, 2] = c.stand_h + 0.002
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.weight.write_root_state_to_sim(st, env_ids)

        self._open_max[env_ids] = 0.0
        self._in[env_ids] = False
        self._in_cnt[env_ids] = 0
        self._closed_after[env_ids] = False
        self._close_cnt[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "lever": self.lever.data.root_state_w[env_ids].clone(),
            "brick": self.brick.data.root_state_w[env_ids].clone(),
            "weight": self.weight.data.root_state_w[env_ids].clone(),
            "stand_a": self.stand_a.data.root_state_w[env_ids].clone(),
            "stand_b": self.stand_b.data.root_state_w[env_ids].clone(),
            "brick_on_a": self.brick_on_a[env_ids].clone(),
            "open_max": self._open_max[env_ids].clone(),
            "in": self._in[env_ids].clone(),
            "in_cnt": self._in_cnt[env_ids].clone(),
            "closed_after": self._closed_after[env_ids].clone(),
            "close_cnt": self._close_cnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.lever.write_root_state_to_sim(state["lever"], env_ids)
        self.brick.write_root_state_to_sim(state["brick"], env_ids)
        self.weight.write_root_state_to_sim(state["weight"], env_ids)
        self.stand_a.write_root_state_to_sim(state["stand_a"], env_ids)
        self.stand_b.write_root_state_to_sim(state["stand_b"], env_ids)
        self.brick_on_a[env_ids] = state["brick_on_a"]
        self._open_max[env_ids] = state["open_max"]
        self._in[env_ids] = state["in"]
        self._in_cnt[env_ids] = state["in_cnt"]
        self._closed_after[env_ids] = state["closed_after"]
        self._close_cnt[env_ids] = state["close_cnt"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A dark steel-grey VAULT (a 340 x 340 mm box, roof 275 mm up) stands "
            "on the ground; it has no door — every wall is solid. The only way in "
            "is the DEPOSIT MOUTH: a 190 x 150 mm rectangular opening in the roof, "
            "covered at rest by a heavy BLUE LID. The lid is one arm of a SEE-SAW "
            "lever hinged on two posts near the roof's front edge; its other arm "
            "reaches forward over the front wall and ends in an open-top AMBER "
            "PEDAL PAN hanging in free air in front of the vault. The lid is the "
            "heavy side: at rest it presses down on its closed stop over the "
            "mouth (a blocked slot — pushing anything against the closed lid is "
            "refused), and the empty pan rides high. On two small tan pedestals "
            "in front of the vault sit the CASH BRICK (a green banded block, "
            "120 x 60 x 30 mm, light) and the brass COUNTERWEIGHT (a gold-toned "
            "cylinder, 48 mm across, 100 mm tall with a narrow grasp knob, heavy "
            "— about five times the brick's mass). Which pedestal holds which "
            "varies; tell them apart by shape and colour.\n"
            "Goal: bank the brick inside the vault and leave the lid closed. The "
            "brick is far too light to tip the see-saw, and one gripper cannot "
            "hold the lid open and carry the brick at the same time — so the "
            "only working order is: (1) set the COUNTERWEIGHT into the pedal "
            "pan; its weight tips the see-saw and swings the lid up to its 60 "
            "degree stop, uncovering the REAR part of the mouth; (2) drop or "
            "lower the BRICK through that uncovered rear opening (long side "
            "across the mouth) so it falls to the vault floor; (3) lift the "
            "counterweight back OFF the pan — the lid falls shut on its own. "
            "Success needs the brick settled inside the chamber below the roof "
            f"AND the lid back on its closed stop (within {c.closed_max_deg:.0f} "
            "degrees), everything at rest. A brick left on the lid, on the roof, "
            "in the pan or anywhere outside the chamber does not count; while "
            "the counterweight still sits in the pan the lid stands open, so the "
            "task is not done until it is removed."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Set the brass counterweight into the amber pedal pan so the see-saw "
            "lifts the blue lid off the vault's roof mouth, drop the green cash "
            "brick through the uncovered opening into the vault, then lift the "
            "counterweight off the pan so the lid falls shut. The task fails "
            "unless the brick ends inside the vault with the lid fully closed."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.vault.data.root_quat_w,
                                  pos_w - self.vault.data.root_pos_w)

    def lid_angle_deg(self) -> torch.Tensor:
        """(N,) float: lever hinge angle in degrees (0 = closed stop, +open_deg =
        open stop). No joint-state API exists on a plain spawn-authored USD
        joint; this is the hinge readout about the y axis."""
        qv = self.vault.data.root_quat_w
        ql = self.lever.data.root_quat_w
        qv_inv = qv * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qv.device)
        rel = _qmul(qv_inv, ql)
        return torch.rad2deg(2.0 * torch.atan2(rel[:, 2], rel[:, 0]))

    def brick_inside(self) -> torch.Tensor:
        """(N,) bool, geometric: brick CoM inside the vault chamber — within the
        interior footprint AND below the roof with margin (containment below the
        aperture: judged on z, so a brick on the lid/roof/pan never counts)."""
        c = self.cfg
        p = self._vault_local(self.brick.data.root_pos_w)
        return (p[:, 0].abs() < c.in_half_x) & (p[:, 1].abs() < c.in_half_y) \
            & (p[:, 2] > 0.004) & (p[:, 2] < c.in_top_z)

    def lid_closed(self) -> torch.Tensor:
        """(N,) bool: lid back on its closed stop."""
        return self.lid_angle_deg() <= self.cfg.closed_max_deg

    def lid_open(self) -> torch.Tensor:
        """(N,) bool: lid held at/above the open threshold."""
        return self.lid_angle_deg() >= self.cfg.open_min_deg

    def settled(self) -> torch.Tensor:
        """(N,) bool: brick, weight, lever swing and vault all still."""
        c = self.cfg
        return (self.brick.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.weight.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.lever.data.root_ang_vel_w.norm(dim=-1) < c.lever_settle_avel) \
            & (self.vault.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.vault, self.lever, self.brick, self.weight)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        fin = self._finite()
        ang = self.lid_angle_deg().clamp(min=0.0)
        self._open_max = torch.where(fin, torch.maximum(self._open_max, ang),
                                     self._open_max)
        inside = self.brick_inside() & fin
        self._in_cnt = torch.where(inside, self._in_cnt + 1,
                                   torch.zeros_like(self._in_cnt))
        self._in |= self._in_cnt >= 3
        cl = self._in & self.lid_closed() & fin
        self._close_cnt = torch.where(cl, self._close_cnt + 1,
                                      torch.zeros_like(self._close_cnt))
        self._closed_after |= self._close_cnt >= 3

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the brick is BANKED — settled inside the vault chamber with
        the lid back on its closed stop, everything still and finite. Both
        clauses are live physical outcomes: the chamber is only reachable under
        a lid the counterweight holds open, and the lid can only be closed with
        the counterweight OFF the pan (gravity — shown load-bearing in smoke)."""
        return self.brick_inside() & self.lid_closed() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*open_frac + 0.35*in + 0.15*closed_after
        (all latched; ~0 for the null policy — the lid rests on its closed stop
        and the brick on its pedestal), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        open_frac = (self._open_max / c.open_norm_deg).clamp(0.0, 1.0)
        base = (c.w_open * open_frac
                + c.w_in * self._in.float()
                + c.w_close * self._closed_after.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="seesaw_vault", robot="null"))
