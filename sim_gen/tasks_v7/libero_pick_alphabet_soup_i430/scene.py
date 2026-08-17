"""press_latch_vault — a two-point simultaneity interlock guards the soup vault.

Derived from the LIBERO seed `libero_pick_alphabet_soup` (pick the alphabet-soup can
among distractors, put it in an open basket). The goal object survives — a RED soup
can must end up in a container — but the PLAN is strategically different: nothing is
grasped-and-dropped into open storage. The container is a fully ROOFED vault whose
only doorway is sealed by a spring-LOADED sliding gate that wants to open but is
held shut by TWO independent spring pins. Each pin blocks the gate within millimetres,
so the gate moves only while BOTH pins are pressed down at the same time — a
momentary, two-point simultaneity interlock. The pin caps sit 17 cm apart, so one
finger (or one object) cannot do it: the robot must SELECT the long press bar (a
too-short decoy rod is also present), lay it across both caps, and push down ~25 N.
The instant both pins clear, the gate drives itself fully open and the released pins
pop up between the gate's catch strips — the doorway can never be fully re-sealed.
Then the can is PUSHED along a porch through the doorway (never lifted over walls —
the roof forbids top entry by construction).

Strategic differences vs the seed and prior read tasks:
- seed: grasp can -> carry -> drop in open basket. Here: tool selection + bridged
  simultaneous two-point press + self-opening one-way gate + push-through delivery.
- vs the ballast-gate vault (i426): that gate is opened by PARKING WEIGHT on a tray
  (a persistent, hands-free state) and must be RE-SEALED for success. This gate
  ignores parked weight by design (spring preload > any loose part's weight, and
  clearing both pins takes more force than ALL loose parts weigh — asserted below);
  it opens only during an active, simultaneous, bridged press, and the ratchet makes
  opening IRREVERSIBLE. No counterweight, no re-seal, opposite temporal structure.

Geometry (vault body frame; origin = base center on the ground; +x = out the door):
- Roofed vault 26 x 26 x 19.2 cm; interior floor (plinth top) z=0.068; doorway in
  the +x wall, y| <= 0.052, z 0.068..0.180 — the ONLY aperture.
- ORANGE gate plate (x 0.132..0.142, half-width 0.110, z 0.065..0.195) on a
  prismatic Y slide, travel 0..0.168; a linear drive (k=8, target 0.30 — beyond the
  open stop) pushes it OPEN always. Two catch strips hang below the plate
  (z 0.045..0.065) at y -0.100 and +0.070 (closed).
- Two pins on prismatic Z slides (q in [-0.024, 0], spring k=140 toward target
  +0.05 => 7.0 N preload up, 10.4 N at full press). Each pin: a collar at the plate
  x-station (z 0.043..0.063 up) that stands in a strip's path, an outboard stem, and
  a 3x3 cm WHITE press cap (top z 0.130) at x-station 0.162, stations y = -/+0.085.
  Collar clears the strips only below q <= -0.020.
- BLUE press bar 26 x 3 x 2 cm (0.60 kg) and YELLOW decoy rod 12 x 3 x 2 cm on the
  ground. RED soup can (r 0.030, h 0.100, 0.30 kg) standing on the porch (deck
  x 0.144..0.344, |y| <= 0.050, top z 0.072); GREEN decoy can on the ground.
- Transit surfaces step DOWN 2 mm each: porch 0.072 -> door sill 0.070 -> floor 0.068.

success(): RED can settled with its center inside the vault interior window
(|x| <= 0.085, |y| <= 0.085, z in [0.080, 0.150]) — any orientation. score():
latched and order-gated: 0.20 once both pins have been pressed clear SIMULTANEOUSLY,
0.50 once the gate has driven fully open AFTER that, 0.80 once the can enters the
window after that; exactly 1.0 iff success(). Null policy: ~0 (gate sealed, can on
the porch outside).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene + env — stays app-free.
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

# ----- geometry constants (vault body frame, metres) -------------------------------------------
VAULT_H = 0.130          # outer half-extent (x and y)
WALL_T = 0.015
PLINTH_H = 0.068         # interior floor top = doorway sill base
ROOF_Z0 = 0.180          # roof underside = doorway top
ROOF_T = 0.012
DOOR_HW = 0.052          # doorway half-width (y)
SILL_T = 0.002           # proud sill strip on top of the front wall band

# gate plate (prismatic +y slide on the outer face of the front wall)
GATE_X0, GATE_X1 = 0.132, 0.142
GATE_HW = 0.110          # plate half-width (y); closed = centered on the doorway
GATE_ZB, GATE_ZT = 0.065, 0.195
GATE_TRAVEL = 0.168      # q in [0, GATE_TRAVEL]; 0 = closed
GATE_K, GATE_C = 8.0, 10.0
GATE_TARGET = 0.30       # beyond the open stop -> the drive holds the gate open
GATE_MASS = 0.30

# catch strips (hang from the plate bottom, same x-extent as the plate)
STRIP_W = 0.010          # y width
STRIP_ZB, STRIP_ZT = 0.045, 0.065
STRIP_Y = (-0.100, 0.070)  # centers at the CLOSED pose

# pins (prismatic z slides; q in [-PIN_TRAVEL, 0], 0 = up)
PIN_Y = (-0.085, 0.085)  # stations
PIN_TRAVEL = 0.024
PIN_K, PIN_C = 140.0, 8.0
PIN_TARGET = 0.05        # above the up stop -> 7.0 N preload, 10.4 N at full press
PIN_MASS = 0.20
COLLAR_HY = 0.006        # collar y half-width
COLLAR_ZB, COLLAR_ZT = 0.043, 0.063   # at q = 0 (up); top stays under the plate bottom
ARM_ZB, ARM_ZT = 0.033, 0.043
STEM_X0, STEM_X1 = 0.152, 0.164
CAP_C = 0.162            # press-cap x center
CAP_HALF = 0.015         # 3 cm square cap
CAP_ZB, CAP_ZT = 0.118, 0.130
PIN_CLEAR_Q = -0.020     # collar top at this q is 2 mm under STRIP_ZB (strips clear)

# porch + cargo
PORCH_X0, PORCH_X1 = 0.144, 0.344
PORCH_HW = 0.050
PORCH_TOP = 0.072
CAN_R, CAN_H = 0.030, 0.100
CAN_MASS = 0.30
BAR_L, BAR_W, BAR_H, BAR_MASS = 0.260, 0.030, 0.020, 0.60
ROD_L, ROD_MASS = 0.120, 0.25

_SPAWNER_CACHE: dict[str, Any] = {}


# ----- small quaternion helpers (wxyz) ---------------------------------------------------------
def _qz(yaw: torch.Tensor) -> torch.Tensor:
    h = yaw * 0.5
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(h)
    q[:, 3] = torch.sin(h)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


# ----- USD authoring helpers -------------------------------------------------------------------
def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(0.0015)
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rb_setup(root, mass: float, com, inertia) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The roofed vault + porch as ONE heavy DYNAMIC compound body (never kinematic:
    the gate/pin joint anchors must follow the reset teleport). Origin = base center
    on the ground; +x = out through the doorway."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    from pxr import Gf
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(cfg.mass_props.mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.02, 0.0, 0.07))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(0.35, 0.35, 0.45))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    grip = _friction_material(stage, f"{prim_path}/grip_mat", cfg.mu_body_s, cfg.mu_body_d)
    slide = _friction_material(stage, f"{prim_path}/slide_mat", cfg.mu_slide_s, cfg.mu_slide_d)
    gray = (0.45, 0.48, 0.50)
    dark = (0.30, 0.32, 0.35)
    wall_h = ROOF_Z0 - PLINTH_H
    wall_cz = PLINTH_H + wall_h / 2
    # plinth (its top = interior floor), transit-slick on top region
    _box(stage, f"{prim_path}/plinth", (2 * VAULT_H, 2 * VAULT_H, PLINTH_H),
         (0.0, 0.0, PLINTH_H / 2), dark, material=slide)
    # back wall
    _box(stage, f"{prim_path}/back", (WALL_T, 2 * VAULT_H, wall_h),
         (-(VAULT_H - WALL_T / 2), 0.0, wall_cz), gray, material=grip)
    # side walls
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (2 * VAULT_H, WALL_T, wall_h),
             (0.0, sy * (VAULT_H - WALL_T / 2), wall_cz), gray, material=grip)
    # front wall: two segments flanking the doorway (|y| in [DOOR_HW, VAULT_H])
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/front_{tag}",
             (WALL_T, VAULT_H - DOOR_HW, wall_h),
             (VAULT_H - WALL_T / 2, sy * (DOOR_HW + (VAULT_H - DOOR_HW) / 2), wall_cz),
             gray, material=grip)
    # proud door sill strip (top 0.070 — transit steps DOWN toward the interior)
    _box(stage, f"{prim_path}/sill", (WALL_T, 2 * DOOR_HW, SILL_T),
         (VAULT_H - WALL_T / 2, 0.0, PLINTH_H + SILL_T / 2), dark, material=slide)
    # FULL roof — top entry denied by construction
    _box(stage, f"{prim_path}/roof", (2 * VAULT_H, 2 * VAULT_H, ROOF_T),
         (0.0, 0.0, ROOF_Z0 + ROOF_T / 2), gray, material=grip)
    # porch deck on legs (deck top 0.072; clear of the gate/strip running gap)
    deck_l = PORCH_X1 - PORCH_X0
    _box(stage, f"{prim_path}/porch_deck", (deck_l, 2 * PORCH_HW, 0.012),
         ((PORCH_X0 + PORCH_X1) / 2, 0.0, PORCH_TOP - 0.006), dark, material=slide)
    for tag, sx in (("i", PORCH_X0 + 0.015), ("o", PORCH_X1 - 0.015)):
        _box(stage, f"{prim_path}/porch_leg_{tag}", (0.03, 2 * PORCH_HW, PORCH_TOP - 0.012),
             (sx, 0.0, (PORCH_TOP - 0.012) / 2), dark, material=grip)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The ORANGE gate plate + two catch strips as one body on a PrismaticJoint
    (axis Y) into the sibling vault, with a linear DriveAPI whose target sits BEYOND
    the open stop: the gate always wants to open and holds itself open. Origin =
    vault origin (geometry authored in vault coordinates at the CLOSED pose)."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rb_setup(root, GATE_MASS, ((GATE_X0 + GATE_X1) / 2, 0.0, 0.128), (2.0e-3, 2.0e-3, 2.0e-3))

    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_gate_s, cfg.mu_gate_d)
    orange = (0.95, 0.45, 0.08)
    dark_o = (0.70, 0.30, 0.05)
    _box(stage, f"{prim_path}/plate", (GATE_X1 - GATE_X0, 2 * GATE_HW, GATE_ZT - GATE_ZB),
         ((GATE_X0 + GATE_X1) / 2, 0.0, (GATE_ZB + GATE_ZT) / 2), orange, material=slick)
    for i, sy in enumerate(STRIP_Y):
        _box(stage, f"{prim_path}/strip_{i}", (GATE_X1 - GATE_X0, STRIP_W, STRIP_ZT - STRIP_ZB),
             ((GATE_X0 + GATE_X1) / 2, sy, (STRIP_ZB + STRIP_ZT) / 2), dark_o, material=slick)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(GATE_TRAVEL))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(GATE_K))
    drv.CreateDampingAttr(float(GATE_C))
    drv.CreateTargetPositionAttr(float(GATE_TARGET))
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One spring pin: blocking collar (at the plate x-station) + low arm + outboard
    stem + WHITE press cap, on a PrismaticJoint (axis Z) into the sibling vault with
    an upward spring (target above the up stop). Origin = vault origin."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    sy = float(cfg.station_y)
    _rb_setup(root, PIN_MASS, (0.150, sy, 0.075), (4.0e-4, 4.0e-4, 4.0e-4))

    slick = _friction_material(stage, f"{prim_path}/slick_mat", 0.20, 0.15)
    grip = _friction_material(stage, f"{prim_path}/grip_mat", 0.80, 0.70)
    white = (0.92, 0.92, 0.95)
    steel = (0.55, 0.58, 0.62)
    # blocking collar in the strip channel
    _box(stage, f"{prim_path}/collar", (GATE_X1 - GATE_X0, 2 * COLLAR_HY, COLLAR_ZT - COLLAR_ZB),
         ((GATE_X0 + GATE_X1) / 2, sy, (COLLAR_ZB + COLLAR_ZT) / 2), steel, material=slick)
    # low arm out to the stem (below the strip sweep)
    _box(stage, f"{prim_path}/arm", (STEM_X1 - GATE_X0, 2 * COLLAR_HY, ARM_ZT - ARM_ZB),
         ((GATE_X0 + STEM_X1) / 2, sy, (ARM_ZB + ARM_ZT) / 2), steel, material=slick)
    # outboard stem up to the cap
    _box(stage, f"{prim_path}/stem", (STEM_X1 - STEM_X0, 0.012, CAP_ZB - ARM_ZB),
         ((STEM_X0 + STEM_X1) / 2, sy, (ARM_ZB + CAP_ZB) / 2), steel, material=slick)
    # press cap (grippy so the bar does not skate off)
    _box(stage, f"{prim_path}/cap", (2 * CAP_HALF, 2 * CAP_HALF, CAP_ZT - CAP_ZB),
         (CAP_C, sy, (CAP_ZB + CAP_ZT) / 2), white, material=grip)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-PIN_TRAVEL))
    j.CreateUpperLimitAttr(0.0)
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(PIN_K))
    drv.CreateDampingAttr(float(PIN_C))
    drv.CreateTargetPositionAttr(float(PIN_TARGET))
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            mu_body_s: float = 0.45
            mu_body_d: float = 0.40
            mu_slide_s: float = 0.30
            mu_slide_d: float = 0.25

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            mu_gate_s: float = 0.20
            mu_gate_d: float = 0.15

        @configclass
        class PinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            station_y: float = 0.0

        _SPAWNER_CACHE.update(vault=VaultSpawnerCfg, gate=GateSpawnerCfg, pin=PinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PressLatchVaultSceneCfg(BaseCfg):
    """Config for `PressLatchVaultScene`. `__post_init__` audits the interlock
    honestly: each pin independently blocks the gate within millimetres; the
    released pins pop up cleanly between the strips at full travel (no snap-lock
    race); parked weight — even EVERY loose object stacked on the caps — cannot
    clear both pins (spring budget beats total loose weight with margin); the
    decoy rod genuinely cannot bridge both caps while the bar can; the push window
    on the can (slide before tip) is real; and the doorway genuinely passes the can
    while the success window rejects a can resting in the doorway."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    in_x_max: float = tunable(0.085)     # success window (vault frame)
    in_y_max: float = tunable(0.085)
    in_z_win: tuple = tunable((0.080, 0.150))
    gate_open_q: float = tunable(0.150)  # "fully open": q >= this (stop = 0.168)
    gate_shut_q: float = tunable(0.020)  # "still sealed": q <= this
    pin_clear_q: float = tunable(-0.021)  # unlock latch: BOTH pins at q <= this
    settle_lin: float = tunable(0.05)
    settle_ang: float = tunable(1.0)

    # --- tunable: randomization ----------------------------------------------------------------
    vault_jitter: float = tunable(0.05)
    vault_yaw_max: float = tunable(180.0)  # deg, free heading
    can_x_win: tuple = tunable((0.260, 0.310))  # red can porch station (vault frame)
    can_y_max: float = tunable(0.016)
    tool_bearing: tuple = tunable((40.0, 75.0))  # bar/rod arcs off vault +x (deg)
    tool_radius: tuple = tunable((0.42, 0.55))
    decoy_bearing: tuple = tunable((125.0, 170.0))  # green can arc
    decoy_radius: tuple = tunable((0.40, 0.55))

    # --- info: structure -----------------------------------------------------------------------
    can_r: float = info(CAN_R)
    can_h: float = info(CAN_H)
    can_mass: float = info(CAN_MASS)
    bar_mass: float = info(BAR_MASS)
    rod_mass: float = info(ROD_MASS)
    gate_mass: float = info(GATE_MASS)
    pin_mass: float = info(PIN_MASS)
    vault_mass: float = info(30.0)
    pin_k: float = info(PIN_K)
    pin_target: float = info(PIN_TARGET)
    gate_k: float = info(GATE_K)
    gate_target: float = info(GATE_TARGET)
    mu_can_s: float = info(0.30)
    mu_can_d: float = info(0.25)
    mu_tool_s: float = info(0.80)
    mu_tool_d: float = info(0.70)
    mu_slide_s: float = info(0.30)  # porch / sill / floor transit surfaces
    mu_slide_d: float = info(0.25)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    def __post_init__(self) -> None:
        g = 9.81
        # -- the closed plate seals the doorway; the open plate clears it ----------------------
        assert GATE_HW >= DOOR_HW + 0.020, "plate must cover the doorway width"
        assert GATE_ZB <= PLINTH_H - 0.003 and GATE_ZT >= ROOF_Z0 + 0.010, \
            "plate must cover the doorway height"
        assert GATE_TRAVEL - GATE_HW >= DOOR_HW + 0.004, "open plate must clear the doorway"
        assert GATE_X0 >= VAULT_H + 0.002, "plate must run clear of the wall/roof faces"
        assert PORCH_X0 >= GATE_X1 + 0.002, "porch deck must clear the plate/strip running gap"
        # -- each pin independently blocks the gate within millimetres -------------------------
        for sy_strip, sy_pin in zip(STRIP_Y, PIN_Y):
            play = (sy_pin - COLLAR_HY) - (sy_strip + STRIP_W / 2)
            assert 0.002 <= play <= 0.006, f"strip->collar play {play:.3f} m out of band"
        overlap = min(STRIP_ZT, COLLAR_ZT) - max(STRIP_ZB, COLLAR_ZB)
        assert overlap >= 0.015, "up collar must solidly overlap the strip channel"
        assert COLLAR_ZT - PIN_TRAVEL <= STRIP_ZB - 0.003, "full press must clear the strips"
        assert COLLAR_ZT + self.pin_clear_q <= STRIP_ZB - 0.001, \
            "the unlock-latch threshold must itself imply cleared strips"
        # -- no snap-lock race: the trailing strip stops short of the far pin ------------------
        trail_face = STRIP_Y[0] + STRIP_W / 2 + GATE_TRAVEL
        assert trail_face <= PIN_Y[1] - COLLAR_HY - 0.004, \
            "released far pin must pop up cleanly between the strips at full travel"
        # -- pins never touch the plate body, only the strips ----------------------------------
        assert COLLAR_ZT <= GATE_ZB - 0.001, "up collar must clear the plate bottom"
        assert ARM_ZT <= STRIP_ZB - 0.002, "pin arm must duck the strip sweep"
        assert ARM_ZB - PIN_TRAVEL >= 0.005, "pressed pin arm must clear the ground"
        # -- parked weight cannot unlock (the anti-cheat force budget) -------------------------
        preload = PIN_K * PIN_TARGET
        heaviest = max(self.bar_mass, self.rod_mass, self.can_mass) * g
        assert preload >= 1.15 * heaviest, \
            "one pin's preload must ignore the heaviest loose object parked on it"
        clear_force = 2 * PIN_K * (PIN_TARGET - self.pin_clear_q)  # both pins to the latch line
        total_loose = (self.bar_mass + self.rod_mass + 2 * self.can_mass) * g
        assert total_loose <= 0.85 * clear_force, \
            "ALL loose objects stacked on the caps must still not clear both pins"
        # -- the bar bridges both caps; the rod cannot -----------------------------------------
        assert BAR_L >= (PIN_Y[1] - PIN_Y[0]) + 2 * CAP_HALF + 0.020, "bar must bridge both caps"
        assert ROD_L <= (PIN_Y[1] - PIN_Y[0]) - 2 * CAP_HALF - 0.010, \
            "decoy rod must be unable to touch both caps at once"
        assert CAP_C - CAP_HALF >= GATE_X1 + 0.004, \
            "the press station must be outboard of the sliding plate"
        assert CAP_ZB - PIN_TRAVEL >= STRIP_ZT + 0.020, \
            "a fully pressed bar must ride above the strip sweep"
        # -- gate drive: opens when free, holds open, but cannot bulldoze the pins -------------
        assert GATE_K * (GATE_TARGET - GATE_TRAVEL) >= 1.0, "drive must hold the gate open"
        assert GATE_K * GATE_TARGET <= 4.0, \
            "drive push on a blocked collar stays far below the GPU-contact creep band"
        # -- push window on the can: slides before it tips -------------------------------------
        mu = (self.mu_can_d + self.mu_slide_d) / 2  # PhysX pair combine: average
        f_slide = mu * self.can_mass * g
        f_tip = self.can_mass * g * CAN_R / (CAN_H / 2)
        assert f_tip >= 1.8 * f_slide, "CoM push must slide the can well before tipping it"
        # -- doorway passes the can; porch retains and aims it ---------------------------------
        assert 2 * DOOR_HW >= 2 * CAN_R + 0.030, "doorway width must pass the can"
        assert ROOF_Z0 - (PLINTH_H + SILL_T) >= CAN_H + 0.008, "doorway must pass a standing can"
        assert PORCH_HW - CAN_R >= self.can_y_max + 0.002, "porch must retain the spawned can"
        assert self.can_x_win[1] + CAN_R <= PORCH_X1 - 0.004, "can spawn stays on the deck"
        # -- success window: strictly interior; a can resting in the doorway is OUT ------------
        assert self.in_x_max <= VAULT_H - WALL_T - CAN_R, "window must mean FULLY inside"
        assert VAULT_H - WALL_T / 2 >= self.in_x_max + 0.020, \
            "a can parked in the doorway must be outside the window"
        assert self.in_z_win[0] < PLINTH_H + CAN_R < self.in_z_win[1], "lying rest in window"
        assert self.in_z_win[0] < PLINTH_H + CAN_H / 2 < self.in_z_win[1], "standing rest in window"
        assert self.in_z_win[1] < ROOF_Z0, "window top stays under the roof"
        # -- null policy scores ~0: the can spawns well outside the window ---------------------
        assert self.can_x_win[0] - CAN_R > self.in_x_max + 0.050, "porch spawn far outside"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("press_latch_vault")
class PressLatchVaultScene(BaseScene):
    cfg: PressLatchVaultSceneCfg

    def __init__(self, cfg: PressLatchVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or PressLatchVaultSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0, linear_damping=0.05, angular_damping=0.1)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)
        can_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_can_s, dynamic_friction=c.mu_can_d, restitution=0.0)
        tool_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_tool_s, dynamic_friction=c.mu_tool_d, restitution=0.0)

        def _tool(color, size, mass):
            return sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                physics_material=tool_mat, rigid_props=rigid, collision_props=coll,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass))

        def _can(color):
            return sim_utils.CylinderCfg(
                radius=c.can_r, height=c.can_h, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                physics_material=can_mat, rigid_props=rigid, collision_props=coll,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass))

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # NOTE: the vault MUST spawn before the gate and pins (joints target it).
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=spawners["vault"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.vault_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_body_s=0.45, mu_body_d=0.40,
                    mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=spawners["gate"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_gate_s=0.20, mu_gate_d=0.15),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003)),
            ),
            "pin_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PinA",
                spawn=spawners["pin"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    station_y=PIN_Y[0]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003)),
            ),
            "pin_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PinB",
                spawn=spawners["pin"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    station_y=PIN_Y[1]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003)),
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar",
                spawn=_tool((0.10, 0.25, 0.90), (BAR_W, BAR_L, BAR_H), c.bar_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.35, BAR_H / 2 + 0.003)),
            ),
            "rod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod",
                spawn=_tool((0.95, 0.85, 0.10), (BAR_W, ROD_L, BAR_H), c.rod_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.35, BAR_H / 2 + 0.003)),
            ),
            "can_red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CanRed",
                spawn=_can((0.85, 0.10, 0.10)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.28, 0.0, PORCH_TOP + CAN_H / 2 + 0.003)),
            ),
            "can_green": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CanGreen",
                spawn=_can((0.10, 0.65, 0.15)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.45, 0.35, CAN_H / 2 + 0.003)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive everything with per-step external wrenches
                "enable_external_forces_every_iteration": True,
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
        self.vault: RigidObject = env.iscene["vault"]
        self.gate: RigidObject = env.iscene["gate"]
        self.pin_a: RigidObject = env.iscene["pin_a"]
        self.pin_b: RigidObject = env.iscene["pin_b"]
        self.bar: RigidObject = env.iscene["bar"]
        self.rod: RigidObject = env.iscene["rod"]
        self.can_red: RigidObject = env.iscene["can_red"]
        self.can_green: RigidObject = env.iscene["can_green"]
        self.env_origins = env.iscene.env_origins
        n = self.vault.data.root_pos_w.shape[0]
        dev = env.device
        # order-gated progress latches (updated every substep in post_step)
        self.l_unlock = torch.zeros(n, dtype=torch.bool, device=dev)
        self.l_open = torch.zeros(n, dtype=torch.bool, device=dev)
        self.l_enter = torch.zeros(n, dtype=torch.bool, device=dev)

    _BODIES = ("vault", "gate", "pin_a", "pin_b", "bar", "rod", "can_red", "can_green")

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the vault (heavy DYNAMIC teleport — every
        joint anchor follows), write the gate CLOSED and both pins UP consistently in
        the vault's new frame, stand the RED can on the porch, and scatter the bar,
        rod and GREEN can on jittered arcs around the vault. Latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        vxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.vault_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_max)
        q_vault = _qz(yaw)
        z0 = torch.full((m, 1), 0.003, device=dev)
        vault_pos = torch.cat([vxy, z0], dim=-1)
        write(self.vault, vault_pos, q_vault)
        # jointed parts: origins coincide with the vault origin at q = 0 (closed / up)
        for body in (self.gate, self.pin_a, self.pin_b):
            write(body, vault_pos, q_vault)

        # RED can standing on the porch (vault frame)
        cx = c.can_x_win[0] + torch.rand(m, device=dev) * (c.can_x_win[1] - c.can_x_win[0])
        cy = (torch.rand(m, device=dev) * 2 - 1) * c.can_y_max
        cz = torch.full((m,), PORCH_TOP + CAN_H / 2 + 0.003, device=dev)
        write(self.can_red, vault_pos + _qapply(q_vault, torch.stack([cx, cy, cz], -1)),
              _qmul(q_vault, _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)))

        # bar + rod on mirrored tool arcs; GREEN can on a rear arc — all free yaw
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
        specs = (
            (self.bar, side, c.tool_bearing, c.tool_radius, BAR_H / 2),
            (self.rod, -side, c.tool_bearing, c.tool_radius, BAR_H / 2),
            (self.can_green, side, c.decoy_bearing, c.decoy_radius, CAN_H / 2),
        )
        for body, sgn, (b0d, b1d), (r0, r1), hz in specs:
            b0, b1 = math.radians(b0d), math.radians(b1d)
            bear = (b0 + torch.rand(m, device=dev) * (b1 - b0)) * sgn
            rad = r0 + torch.rand(m, device=dev) * (r1 - r0)
            local = torch.stack([rad * torch.cos(bear), rad * torch.sin(bear),
                                 torch.full((m,), hz + 0.003, device=dev)], dim=-1)
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(body, vault_pos + _qapply(q_vault, local), _qmul(q_vault, _qz(byaw)))

        self.l_unlock[env_ids] = False
        self.l_open[env_ids] = False
        self.l_enter[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the ORDERED milestones at sim rate: the unlock event (both pins
        pressed clear in the SAME substep) is transient — score() alone would miss it."""
        c = self.cfg
        both_clear = (self.pin_q(self.pin_a) <= c.pin_clear_q) \
            & (self.pin_q(self.pin_b) <= c.pin_clear_q)
        self.l_unlock |= both_clear
        self.l_open |= self.l_unlock & (self.gate_q() >= c.gate_open_q)
        self.l_enter |= self.l_open & self.can_in_vault()

    # ----- state (full, restorable — includes the latches) ------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        st = {nm: getattr(self, nm).data.root_state_w[env_ids].clone() for nm in self._BODIES}
        st["latches"] = torch.stack(
            [self.l_unlock[env_ids], self.l_open[env_ids], self.l_enter[env_ids]], dim=-1).clone()
        return st

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in self._BODIES:
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        lat = state["latches"]
        self.l_unlock[env_ids] = lat[:, 0]
        self.l_open[env_ids] = lat[:, 1]
        self.l_enter[env_ids] = lat[:, 2]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A roofed gray vault (26 x 26 cm, 19 cm tall) stands on the floor. Its FULL "
            "roof means nothing can be dropped in from above; the only way in is a "
            "doorway (10.4 cm wide, sill at 7 cm, top at 18 cm) in the front (+x) wall. "
            "An ORANGE gate plate seals the doorway. The plate slides SIDEWAYS (toward "
            "+y) and is spring-DRIVEN to open — it wants to slide open on its own — but "
            "two steel catch strips under it are blocked by two spring PINS, and each "
            "pin alone stops the gate within a few millimetres. The pins carry square "
            "WHITE press caps (3 x 3 cm, tops 13 cm high) on stands 16 cm out from the "
            "vault center, at y = -8.5 and +8.5 cm — 17 cm apart. A pin unlocks only "
            "while pushed DOWN about 2 cm, and the gate moves only while BOTH pins are "
            "down AT THE SAME TIME. Each pin resists ~7-10 N, so lay the BLUE press bar "
            "(26 x 3 x 2 cm) across BOTH caps and push down on its middle (~25 N): the "
            "gate then drives itself fully open, and the released pins pop up between "
            "the strips so the doorway can never be fully re-sealed. The YELLOW rod "
            "(12 cm) is too short to bridge the caps; parking objects on the caps "
            "cannot press them out — every loose object stacked up is still too light. "
            "A RED soup can (6 cm diameter, 10 cm tall) stands on the porch deck in "
            "front of the doorway (deck top 7.2 cm; the deck, door sill and interior "
            "floor step down 2 mm each, so a pushed can never catches a lip). A GREEN "
            "decoy can stands on the floor elsewhere. The vault's position and heading "
            "and every object's pose change each episode — read the scene by looking.\n"
            "Goal: get the RED can to rest fully INSIDE the vault (any orientation). "
            "Bridge both white caps with the blue bar and press down until the orange "
            "gate slides fully open; move the bar out of the porch lane; then push the "
            "red can along the porch, over the sill, through the doorway, until it "
            "rests inside. Leave the green can and the rod alone."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the blue bar across both white pin caps and press down until the "
            "orange gate slides open, set the bar aside, then push the red soup can "
            "along the porch through the doorway so it rests inside the vault."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> vault frame (origin = base center on the ground)."""
        return _qapply(_qinv(self.vault.data.root_quat_w),
                       pos_w - self.vault.data.root_pos_w)

    def gate_q(self) -> torch.Tensor:
        """(N,) gate slide coordinate: 0 = closed, GATE_TRAVEL = fully open."""
        return self.vault_local(self.gate.data.root_pos_w)[:, 1]

    def pin_q(self, pin) -> torch.Tensor:
        """(N,) pin slide coordinate: 0 = up (blocking), -PIN_TRAVEL = fully pressed."""
        return self.vault_local(pin.data.root_pos_w)[:, 2]

    def can_in_vault(self) -> torch.Tensor:
        """(N,) bool: RED can center inside the interior window (any orientation)."""
        c = self.cfg
        loc = self.vault_local(self.can_red.data.root_pos_w)
        return (loc[:, 0].abs() <= c.in_x_max) & (loc[:, 1].abs() <= c.in_y_max) \
            & (loc[:, 2] >= c.in_z_win[0]) & (loc[:, 2] <= c.in_z_win[1])

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the RED can rests settled fully inside the roofed vault, any
        orientation. Reaching the interior is physically gated by the interlock: the
        roof is full, the doorway is the only aperture, and it opens only during a
        simultaneous two-point pin press."""
        return self.can_in_vault() & self.settled(self.can_red)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched, ORDER-GATED milestones — 0.20 once both pins
        have ever been pressed clear in the same substep, 0.50 once the gate has driven
        fully open AFTER that, 0.80 once the red can enters the interior window after
        that; exactly 1.0 iff success(). Non-decreasing along the intended solution
        (press -> self-open -> push through -> settle); the null policy scores ~0."""
        base = torch.maximum(
            0.20 * self.l_unlock.float(),
            torch.maximum(0.50 * self.l_open.float(), 0.80 * self.l_enter.float()))
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="press_latch_vault", robot="null", env_spacing=3.0))
