"""BallastGateScene — weight the guillotine gate open with the RED juice carton, carry
the WHITE milk through the doorway into the crib, then remove the ballast so the spring
re-seals the vault (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i426`).

Derived from libero_90/living_room_scene2 "pick up the milk and put it in the basket",
but STRATEGICALLY inverted at every load-bearing step. The seed is a top-drop: grasp the
free-standing milk carton, carry it OVER the open basket, release, gravity finishes; the
other cartons are inert clutter. Here the goal container (a green-walled CRIB) sits
inside a fully ROOFED vault whose only doorway is sealed by a spring-preloaded ORANGE
guillotine gate that must SINK to open. The gate cannot be usefully held: a single-arm
robot needs both the doorway open AND a hand free to carry the milk through it. The gate
carries a walled ballast TRAY on its top edge, offset to the side of the doorway — the
solver must first fetch the seed's decoy, the RED juice carton, and lay it on the tray;
its weight overwhelms the spring and the gate sinks fully open HANDS-FREE. Then the milk
is carried horizontally through the held-open doorway (over the sunken gate's top edge,
under the roof) and laid in the crib. Finally the ballast must be REMOVED again: success
requires the gate re-sealed shut, which the spring only does once the tray is empty. The
decoy is thus REQUIRED EQUIPMENT (a counterweight key), the container is roofed against
the seed's entire delivery plan, and the task ends with an un-doing step (unload the
ballast) that the seed has no counterpart for. Ordering is physically forced both ways:
a closed gate blocks the milk's entry, and a loaded gate can never close.

Strategy vs what was read while building: sibling i177 (`dump_hopper`) cages the milk
untouchable and has the solver carry the CONTAINER and continuously HOLD a lever through
a gravity discharge — here the milk is directly manipulated throughout, the container
never moves, and no mechanism is ever held: the weight of a placed object does the
holding, hands-free. Sibling i275 (`drop_chute`) has a passive gravity-closed flap that
YIELDS to cargo pushed against it — here the gate is weight-LOCKED shut (2.4 N spring
preload; cargo pressed against the closed gate does NOT open it, a smoke probe), the
insertion is a carried lay-in through a doorway held open by ballast, and the task
additionally requires actively re-closing the door by unloading the ballast. pen_holder
(exemplar) is many-object tip-up insertion into a carriable open cup with no mechanism.

success(): milk resting inside the crib, gate fully re-closed (spring on its upper
stop), milk and juice both settled, and the juice nowhere in / at / on the vault.
score(): stateless monotone — 0.25 gate sunk fully open (ballast working), 0.65 milk
inside the crib, max-combined; 1.0 iff success(). The order is forced by geometry (the
roof and the sealed doorway are the only boundary), so stateless credit cannot be
farmed out of order. Null policy scores ~0 (gate spawns closed, cartons far outside).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - vault (dynamic, 30 kg — heavy DYNAMIC fixture, never kinematic, so the gate
    joint's anchor follows the reset teleport): solid base slab (doorway sill), back
    and side walls, two front pillars flanking the doorway, a FULL roof, and the green
    crib (pad + 4 low walls) on the interior floor.
  - gate (dynamic, 120 g): ORANGE panel sliding on a PrismaticJoint (axis Z, limits
    [-GATE_TRAVEL, 0]) hung just OUTSIDE the front wall, plus a walled ballast tray
    cantilevered off its top corner, beside the doorway (open sky above it — the roof
    does not cover the tray). A linear DriveAPI spring (stiffness against a target
    ABOVE the upper stop) preloads the gate SHUT; enough dead weight on the tray
    overwhelms it and the gate sinks to the bottom stop.
  - milk carton (target): 60 x 60 x 160 mm white box, 350 g.
  - juice carton (ballast key): same box, RED, 450 g — heavy enough to fully sink the
    gate (asserted with margin); the empty spring re-closes with margin.
Per-episode randomization (readback-verified in smoke): vault xy jitter + FREE yaw,
milk and juice standing on mirrored jittered arcs (side, bearing, radius, free yaw).
Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
VAULT_HX = 0.200  # vault outer half-extent, x (+x = out through the doorway)
VAULT_HY = 0.200
WALL_T = 0.012
BASE_H = 0.260  # solid base slab: z 0 .. 0.260 (its top is the doorway sill / interior floor)
ROOF_Z0 = 0.560  # roof underside (walls span z 0.260 .. 0.560)
ROOF_T = 0.012  # roof: 0.560 .. 0.572, FULL footprint — top entry denied
DOOR_HW = 0.120  # doorway half-width (gap between the front pillars)
GATE_PLANE_X = 0.210  # gate panel center plane, OUTSIDE the front wall (4 mm running gap)
GATE_T = 0.012
GATE_W = 0.270  # covers the 240 mm doorway with margin
GATE_H = 0.320  # closed: spans z 0.245 .. 0.565 — seals the whole doorway
GATE_Z0 = 0.405  # closed panel center height (upper joint stop)
GATE_TRAVEL = 0.225  # downward stroke (open: spans z 0.020 .. 0.340)
GATE_MASS = 0.12
SPRING_K = 8.0  # N/m linear drive stiffness
SPRING_C = 10.0  # N*s/m linear drive damping
SPRING_TARGET = 0.300  # drive target ABOVE the upper stop -> 2.4 N preload holds the gate SHUT
TRAY_CX = 0.070  # ballast tray plate center, gate frame (world x 0.280 at closed)
TRAY_CY = 0.235  # offset to +y: BESIDE the doorway corridor, under open sky
TRAY_CZ = 0.126  # plate center z, gate frame (plate top at +0.130)
TRAY_LX = 0.140
TRAY_LY = 0.200
TRAY_T = 0.008
TRAY_WALL_T = 0.012
TRAY_WALL_H = 0.035  # retains the LYING ballast carton
CRIB_CX = 0.030  # crib center, vault frame
CRIB_HX = 0.140  # crib outer half-extents
CRIB_HY = 0.110
CRIB_WALL_T = 0.012
CRIB_WALL_H = 0.056  # rim top = 0.316
CRIB_PAD_T = 0.006  # pad top = 0.266
MILK_W = 0.060  # carton square cross-section
MILK_H = 0.160  # carton long dimension


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


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


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
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
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the roofed vault + interior crib as ONE heavy DYNAMIC compound body
    (never kinematic: the gate joint's anchor must follow the reset teleport — a
    kinematic body0's anchor stays world-fixed at the spawn pose). Origin = base
    center on the ground; +x = out through the doorway."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # ZERO sleep/stabilization: a sleeping vault would freeze the gate-joint anchor.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    grip = _friction_material(stage, f"{prim_path}/grip_mat", cfg.mu_body_s, cfg.mu_body_d)
    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_slide_s, cfg.mu_slide_d)
    gray = (0.45, 0.48, 0.50)
    dark = (0.30, 0.32, 0.35)
    green = (0.10, 0.62, 0.20)
    wall_h = ROOF_Z0 - BASE_H  # 0.300, walls from z 0.260 to 0.560
    wall_cz = BASE_H + wall_h / 2  # 0.410
    # solid base slab (its top = doorway sill = interior floor), grippy
    _box(stage, f"{prim_path}/base", (2 * VAULT_HX, 2 * VAULT_HY, BASE_H),
         (0.0, 0.0, BASE_H / 2), dark, 0.0015, material=grip)
    # back wall (inner face x = -0.188)
    _box(stage, f"{prim_path}/back", (WALL_T, 2 * VAULT_HY, wall_h),
         (-(VAULT_HX - WALL_T / 2), 0.0, wall_cz), gray, 0.0015, material=grip)
    # side walls (inner faces |y| = 0.188)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (2 * VAULT_HX, WALL_T, wall_h),
             (0.0, sy * (VAULT_HY - WALL_T / 2), wall_cz), gray, 0.0015, material=grip)
    # front pillars flanking the doorway (|y| in [DOOR_HW, VAULT_HY]); their OUTER
    # face (x = 0.200) is the gate's running plane — slick
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/pillar_{tag}",
             (WALL_T, VAULT_HY - DOOR_HW, wall_h),
             (VAULT_HX - WALL_T / 2, sy * (DOOR_HW + (VAULT_HY - DOOR_HW) / 2), wall_cz),
             gray, 0.0015, material=slick)
    # FULL roof — top entry is denied by construction (does NOT cover the tray outside)
    _box(stage, f"{prim_path}/roof", (2 * VAULT_HX, 2 * VAULT_HY, ROOF_T),
         (0.0, 0.0, ROOF_Z0 + ROOF_T / 2), gray, 0.0015, material=grip)
    # green crib on the interior floor: pad + 4 low walls
    _box(stage, f"{prim_path}/crib_pad",
         (2 * (CRIB_HX - CRIB_WALL_T), 2 * (CRIB_HY - CRIB_WALL_T), CRIB_PAD_T),
         (CRIB_CX, 0.0, BASE_H + CRIB_PAD_T / 2), green, 0.0015, material=grip)
    for tag, sx in (("f", 1.0), ("b", -1.0)):
        _box(stage, f"{prim_path}/crib_wx{tag}", (CRIB_WALL_T, 2 * CRIB_HY, CRIB_WALL_H),
             (CRIB_CX + sx * (CRIB_HX - CRIB_WALL_T / 2), 0.0, BASE_H + CRIB_WALL_H / 2),
             green, 0.0015, material=grip)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/crib_wy{tag}", (2 * CRIB_HX, CRIB_WALL_T, CRIB_WALL_H),
             (CRIB_CX, sy * (CRIB_HY - CRIB_WALL_T / 2), BASE_H + CRIB_WALL_H / 2),
             green, 0.0015, material=grip)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the spring-preloaded guillotine gate: an ORANGE panel + walled ballast
    tray as one light body on a PrismaticJoint (axis Z) into the sibling vault, with
    a linear DriveAPI spring whose target sits ABOVE the upper stop (preloads the
    gate SHUT with 2.4 N; dead weight on the tray overwhelms it). Origin = panel
    center at the CLOSED pose."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    # Authored CoM at the body origin + diagonal inertia: the prismatic joint locks
    # all rotation, so only the vertical force balance is load-bearing — keep it
    # exactly = authored mass * g against the authored spring.
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2.0e-3, 2.0e-3, 2.0e-3))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_gate_s, cfg.mu_gate_d)
    grip = _friction_material(stage, f"{prim_path}/grip_mat", cfg.mu_tray_s, cfg.mu_tray_d)
    orange = (0.95, 0.45, 0.08)
    # sliding panel (slick: pressed cargo slides, friction drag can't open the gate)
    _box(stage, f"{prim_path}/panel", (GATE_T, GATE_W, GATE_H),
         (0.0, 0.0, 0.0), orange, 0.0015, material=slick)
    # ballast tray: plate + 4 low walls, cantilevered off the top corner (grippy)
    _box(stage, f"{prim_path}/tray_plate", (TRAY_LX, TRAY_LY, TRAY_T),
         (TRAY_CX, TRAY_CY, TRAY_CZ), orange, 0.0015, material=grip)
    wz = TRAY_CZ + TRAY_T / 2 + TRAY_WALL_H / 2
    for tag, sx in (("f", 1.0), ("b", -1.0)):
        _box(stage, f"{prim_path}/tray_wx{tag}", (TRAY_WALL_T, TRAY_LY, TRAY_WALL_H),
             (TRAY_CX + sx * (TRAY_LX - TRAY_WALL_T) / 2, TRAY_CY, wz),
             orange, 0.0015, material=grip)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/tray_wy{tag}", (TRAY_LX, TRAY_WALL_T, TRAY_WALL_H),
             (TRAY_CX, TRAY_CY + sy * (TRAY_LY - TRAY_WALL_T) / 2, wz),
             orange, 0.0015, material=grip)

    # prismatic joint to the sibling vault, axis Z, anchored at the closed pose.
    # q = 0 closed (upper stop), q = -GATE_TRAVEL fully open (bottom stop).
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(GATE_PLANE_X), 0.0, float(GATE_Z0)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-GATE_TRAVEL))
    j.CreateUpperLimitAttr(0.0)
    # the preload spring: implicit linear drive toward a target ABOVE the upper stop
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(SPRING_K))
    drv.CreateDampingAttr(float(SPRING_C))
    drv.CreateTargetPositionAttr(float(SPRING_TARGET))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            mu_body_s: float = 0.45
            mu_body_d: float = 0.40
            mu_slide_s: float = 0.15
            mu_slide_d: float = 0.12

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            mu_gate_s: float = 0.15
            mu_gate_d: float = 0.12
            mu_tray_s: float = 0.45
            mu_tray_d: float = 0.40

        _SPAWNER_CACHE.update(vault=VaultSpawnerCfg, gate=GateSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastGateSceneCfg(BaseCfg):
    """Config for `BallastGateScene`. The honesty knobs are asserted in
    `__post_init__`: the closed gate genuinely seals the doorway (and cargo pressed
    against it cannot drag it open), the ballast carton genuinely overwhelms the
    spring while the empty spring genuinely re-closes (both with margin), the open
    corridor genuinely passes a carried carton over the sunken gate's top edge and
    the crib rim, the crib window accepts every legitimate rest pose and rejects
    rim-perches, and both cartons spawn beyond the tray and the vault so the null
    policy scores ~0."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    gate_closed_q: float = tunable(-0.006)  # closed: q >= this (upper stop = 0)
    gate_open_q: float = tunable(-0.200)  # fully open: q <= this (bottom stop = -0.225)
    crib_x_max: float = tunable(0.105)  # |x - CRIB_CX|, vault frame
    crib_y_max: float = tunable(0.075)
    crib_z_win: tuple = tunable((0.268, 0.362))  # lying AND standing rests; carry is above
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)
    decoy_xy_max: float = tunable(0.260)  # juice violation box (vault frame)
    decoy_z_win: tuple = tunable((0.200, 0.800))  # covers interior, doorway, roof-top

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    vault_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the vault (m)
    vault_yaw_max: float = tunable(180.0)  # uniform +/- vault yaw (deg; FREE heading)
    arc_bearing: tuple = tunable((30.0, 70.0))  # carton bearing band off vault +x (deg)
    arc_radius: tuple = tunable((0.60, 0.72))  # ... radius band around the vault (m)

    # --- info: structure -----------------------------------------------------------------------
    milk_w: float = info(MILK_W)
    milk_h: float = info(MILK_H)
    milk_mass: float = info(0.35)
    juice_mass: float = info(0.45)  # the ballast key: sinks the gate with margin
    gate_mass: float = info(GATE_MASS)
    vault_mass: float = info(30.0)  # heavy dynamic fixture (joint anchor follows teleports)
    spring_k: float = info(SPRING_K)
    spring_c: float = info(SPRING_C)
    spring_target: float = info(SPRING_TARGET)
    mu_carton_s: float = info(0.30)
    mu_carton_d: float = info(0.25)
    mu_gate_s: float = info(0.15)  # panel + pillar faces (pressed cargo slides)
    mu_gate_d: float = info(0.12)
    mu_body_s: float = info(0.45)  # vault interior + crib + tray floor (cargo parks)
    mu_body_d: float = info(0.40)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    def __post_init__(self) -> None:
        g = 9.81
        # -- the closed gate seals the vault's only opening ------------------------------------
        assert GATE_W / 2 >= DOOR_HW + 0.010, "gate panel must cover the doorway width"
        assert GATE_Z0 - GATE_H / 2 <= BASE_H - 0.010, "closed panel must reach below the sill"
        assert GATE_Z0 + GATE_H / 2 >= ROOF_Z0 + 0.004, "closed panel must reach past the roof line"
        gap = (GATE_PLANE_X - GATE_T / 2) - VAULT_HX
        assert 0.003 <= gap <= 0.020, f"panel-wall running gap {gap:.3f} m out of band"
        # -- the open corridor passes a carried lying carton -----------------------------------
        open_top = GATE_Z0 - GATE_TRAVEL + GATE_H / 2  # sunken gate's top edge (the curb)
        assert ROOF_Z0 - open_top >= MILK_W + 0.100, "open corridor must pass a carried carton"
        assert GATE_Z0 - GATE_TRAVEL - GATE_H / 2 >= 0.015, "open panel must clear the ground"
        rim = BASE_H + CRIB_WALL_H
        assert open_top >= rim + 0.015, "the doorway curb, not the crib rim, binds the carry height"
        # -- spring balance: closed holds, ballast sinks, empty re-closes (all with margin) ----
        w_gate = GATE_MASS * g
        f0 = SPRING_K * SPRING_TARGET  # preload at the upper stop
        f1 = SPRING_K * (SPRING_TARGET + GATE_TRAVEL)  # spring at the bottom stop
        assert f0 >= 1.8 * w_gate, f"preload {f0:.2f} N must hold the {w_gate:.2f} N gate shut"
        assert (GATE_MASS + self.juice_mass) * g >= 1.25 * f1, \
            "the ballast carton must overwhelm the spring at FULL stroke"
        assert (GATE_MASS + self.milk_mass) * g >= 1.05 * f1, \
            "(even the lighter milk would sink the gate — the key is weight, not identity)"
        assert f1 >= 2.0 * w_gate, "the empty spring must re-close the gate from full open"
        # -- cargo pressed against the CLOSED gate cannot drag it open (weight-locked door) ----
        mu_drag = (self.mu_gate_d + self.mu_carton_d) / 2  # PhysX default combine: average
        assert f0 - w_gate >= 2.0 * mu_drag * 2.0, \
            "a 2 N horizontal press must not drag the closed gate down"
        # -- the under-weight / over-weight probe pair is non-vacuous --------------------------
        assert 0.8 + w_gate <= f0 - 0.2, "a 0.8 N load must NOT open the gate"
        assert 3.0 + w_gate >= f0 + 0.5, "a 3.0 N load must visibly sink the gate"
        # -- ballast tray: fits the LYING carton, retains it, sits under open sky --------------
        inner_x = TRAY_LX - 2 * TRAY_WALL_T
        inner_y = TRAY_LY - 2 * TRAY_WALL_T
        assert inner_y >= MILK_H + 0.012 and inner_x >= MILK_W + 0.030, \
            "tray must fit the ballast carton lying along y"
        assert TRAY_WALL_H >= 0.025, "tray walls must retain the lying ballast"
        tray_x0 = GATE_PLANE_X + TRAY_CX - TRAY_LX / 2
        assert tray_x0 >= VAULT_HX + 0.008, \
            "tray (and its drop approach) must clear the vault face and the roof edge"
        assert TRAY_CY - TRAY_LY / 2 >= DOOR_HW + 0.010, \
            "tray must sit clear of the doorway carry corridor"
        # -- crib windows: accept every legitimate rest, reject rim-perches --------------------
        pad_top = BASE_H + CRIB_PAD_T
        assert self.crib_z_win[0] < pad_top + MILK_W / 2 < self.crib_z_win[1], \
            "a carton LYING on the crib pad must be in the z window"
        assert self.crib_z_win[0] < pad_top + MILK_H / 2 < self.crib_z_win[1], \
            "a carton STANDING on the crib pad must be in the z window"
        in_hx = CRIB_HX - CRIB_WALL_T  # crib inner half-extents
        in_hy = CRIB_HY - CRIB_WALL_T
        assert 2 * in_hx >= MILK_H + 0.030, "crib must fit the milk lying along x"
        assert self.crib_x_max >= in_hx - MILK_W / 2 + 0.005, \
            "x window must cover a lying carton pushed against the crib wall"
        assert self.crib_y_max >= in_hy - MILK_W / 2 + 0.005, \
            "y window must cover a carton pushed against the y wall"
        assert CRIB_HX - CRIB_WALL_T / 2 >= self.crib_x_max + 0.020, \
            "a carton perched on the x rim walls must be OUTSIDE the x window"
        assert CRIB_HY - CRIB_WALL_T / 2 >= self.crib_y_max + 0.020, \
            "a carton perched on the y rim walls must be OUTSIDE the y window"
        assert CRIB_CX + CRIB_HX <= VAULT_HX - WALL_T - 0.005, \
            "crib must sit clear of the vault front wall"
        # -- juice violation box covers the vault, the doorway, and the roof-top ---------------
        assert self.decoy_xy_max >= VAULT_HX + 0.050
        assert self.decoy_z_win[0] <= BASE_H - 0.020
        assert self.decoy_z_win[1] >= ROOF_Z0 + ROOF_T + MILK_H / 2 + 0.040, \
            "a carton standing ON the roof must be in the violation box"
        # -- null policy scores 0: cartons spawn beyond the tray and the vault -----------------
        tray_reach = math.hypot(GATE_PLANE_X + TRAY_CX + TRAY_LX / 2,
                                TRAY_CY + TRAY_LY / 2)
        assert self.arc_radius[0] >= tray_reach + math.hypot(MILK_W, MILK_H) / 2 + 0.020, \
            "spawn arc must start beyond the tray's outer corner"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_gate")
class BallastGateScene(BaseScene):
    cfg: BallastGateSceneCfg

    def __init__(self, cfg: BallastGateSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastGateSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        carton_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0, linear_damping=0.05, angular_damping=0.1)
        carton_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)
        carton_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_carton_s, dynamic_friction=c.mu_carton_d, restitution=0.0)

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
            # NOTE: the vault MUST spawn before the gate (the joint targets it).
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=spawners["vault"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.vault_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_body_s=c.mu_body_s, mu_body_d=c.mu_body_d,
                    mu_slide_s=c.mu_gate_s, mu_slide_d=c.mu_gate_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=spawners["gate"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_gate_s=c.mu_gate_s, mu_gate_d=c.mu_gate_d,
                    mu_tray_s=c.mu_body_s, mu_tray_d=c.mu_body_d),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(GATE_PLANE_X, 0.0, GATE_Z0 + 0.003)),
            ),
            "milk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Milk",
                spawn=sim_utils.CuboidCfg(
                    size=(c.milk_w, c.milk_w, c.milk_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.95, 0.97)),
                    physics_material=carton_mat, rigid_props=carton_rigid,
                    collision_props=carton_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.milk_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.65, 0.35, c.milk_h / 2 + 0.003)),
            ),
            "juice": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Juice",
                spawn=sim_utils.CuboidCfg(
                    size=(c.milk_w, c.milk_w, c.milk_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.12, 0.12)),
                    physics_material=carton_mat, rigid_props=carton_rigid,
                    collision_props=carton_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.juice_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.65, -0.35, c.milk_h / 2 + 0.003)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # External-wrench carry plant: without this the body-frame carry force
                # is under-applied across TGS iterations and the transport sags.
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
        self.milk: RigidObject = env.iscene["milk"]
        self.juice: RigidObject = env.iscene["juice"]
        self.env_origins = env.iscene.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the vault (heavy DYNAMIC teleport — the
        gate joint anchor follows), write the gate CONSISTENTLY closed on its slide
        in the vault's new frame, and stand the milk and juice cartons on mirrored
        jittered arcs around the vault (side, bearing, radius, free yaw)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- vault: xy jitter + free yaw (write the WHOLE linkage together) ---
        vxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.vault_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_max)
        q_vault = _qz(yaw)
        z0 = torch.full((m, 1), 0.003, device=dev)
        vault_pos = torch.cat([vxy, z0], dim=-1)
        write(self.vault, vault_pos, q_vault)

        # --- gate: CLOSED on its slide, in the vault's new frame ---
        gate_local = torch.tensor([GATE_PLANE_X, 0.0, GATE_Z0], device=dev).expand(m, 3)
        write(self.gate, vault_pos + _qapply(q_vault, gate_local), q_vault)

        # --- milk + juice: STANDING on mirrored jittered arcs around the vault ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
        b0, b1 = (math.radians(v) for v in c.arc_bearing)
        r0, r1 = c.arc_radius
        for body, sgn in ((self.milk, side), (self.juice, -side)):
            bear = (b0 + torch.rand(m, device=dev) * (b1 - b0)) * sgn
            rad = r0 + torch.rand(m, device=dev) * (r1 - r0)
            local = torch.stack([rad * torch.cos(bear), rad * torch.sin(bear),
                                 torch.full((m,), MILK_H / 2 + 0.003, device=dev)], dim=-1)
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(body, vault_pos + _qapply(q_vault, local), _qmul(q_vault, _qz(byaw)))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
                for nm in ("vault", "gate", "milk", "juice")}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("vault", "gate", "milk", "juice"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A roofed gray vault ({2 * VAULT_HX * 100:.0f} x {2 * VAULT_HY * 100:.0f} cm, "
            f"{(ROOF_Z0 + ROOF_T) * 100:.0f} cm tall) stands on the floor with a GREEN "
            f"crib inside it. The top is a FULL roof — nothing can be dropped in from "
            f"above. The only doorway ({2 * DOOR_HW * 100:.0f} cm wide, in the front "
            f"wall above a {BASE_H * 100:.0f} cm sill) is sealed by an ORANGE guillotine "
            f"gate that slides straight DOWN to open. A spring holds the gate SHUT: "
            f"pushing cargo against it will not open it, and it cannot be usefully held "
            f"down while also carrying something through. The gate carries a small "
            f"walled TRAY on its top edge, beside the doorway, under open sky. Lay "
            f"something heavy on the tray and the gate sinks fully open and STAYS open "
            f"hands-free; empty the tray and the spring re-seals the doorway. A WHITE "
            f"milk carton ({c.milk_w * 100:.0f} x {c.milk_w * 100:.0f} x "
            f"{c.milk_h * 100:.0f} cm) and a heavier RED juice carton stand on the "
            f"floor on opposite sides of the vault. The vault's position and heading "
            f"and both cartons' poses change every episode — read the scene by "
            f"looking.\n"
            f"Goal: get the WHITE milk carton to rest inside the GREEN crib and leave "
            f"the vault SEALED. Lay the RED juice carton on the gate's tray as a "
            f"counterweight — the gate sinks open; carry the milk in through the "
            f"doorway (over the sunken gate's top edge) and lay it in the crib; then "
            f"take the juice off the tray so the spring closes the gate again. Finish "
            f"with the milk in the crib, the gate fully shut, the juice well away from "
            f"the vault, and everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the red juice carton on the orange gate's tray so its weight sinks "
            "the gate open, carry the white milk carton through the doorway and lay "
            "it in the green crib, then take the juice off the tray so the spring "
            "seals the gate shut behind it."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> vault frame (origin = base center on the ground)."""
        return _qapply(_qinv(self.vault.data.root_quat_w),
                       pos_w - self.vault.data.root_pos_w)

    def gate_q(self) -> torch.Tensor:
        """(N,) gate slide coordinate: 0 = closed (upper stop), -GATE_TRAVEL = open."""
        loc = self.vault_local(self.gate.data.root_pos_w)
        return loc[:, 2] - GATE_Z0

    def gate_closed(self) -> torch.Tensor:
        """(N,) bool: the gate is pressed shut on its upper stop."""
        return self.gate_q() >= self.cfg.gate_closed_q

    def gate_open_deep(self) -> torch.Tensor:
        """(N,) bool: the gate is sunk essentially to the bottom stop."""
        return self.gate_q() <= self.cfg.gate_open_q

    def milk_in_crib(self) -> torch.Tensor:
        """(N,) bool: milk center inside the green crib (vault frame)."""
        c = self.cfg
        loc = self.vault_local(self.milk.data.root_pos_w)
        return ((loc[:, 0] - CRIB_CX).abs() <= c.crib_x_max) \
            & (loc[:, 1].abs() <= c.crib_y_max) \
            & (loc[:, 2] >= c.crib_z_win[0]) & (loc[:, 2] <= c.crib_z_win[1])

    def decoy_out(self) -> torch.Tensor:
        """(N,) bool: the juice carton is nowhere in / at / on the vault (its legal
        mid-run perch, the ballast tray, sits OUTSIDE this box — but a loaded tray
        holds the gate open, which success() separately rejects)."""
        c = self.cfg
        loc = self.vault_local(self.juice.data.root_pos_w)
        inside = (loc[:, 0].abs() <= c.decoy_xy_max) & (loc[:, 1].abs() <= c.decoy_xy_max) \
            & (loc[:, 2] >= c.decoy_z_win[0]) & (loc[:, 2] <= c.decoy_z_win[1])
        return ~inside

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: milk resting inside the crib, gate re-sealed on its upper stop
        (only possible with the ballast tray empty), milk and juice settled, juice
        nowhere in / at / on the vault."""
        return self.milk_in_crib() & self.gate_closed() & self.settled(self.milk) \
            & self.settled(self.juice) & self.decoy_out()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: max of 0.25 (gate sunk fully open — the ballast is
        doing its job) and 0.65 (milk inside the crib — only reachable through the
        ballast-opened doorway: the roof and the sealed gate are the only boundary);
        1.0 iff success(). Stateless and monotone along the intended solution
        (ballast on -> gate sinks -> carry through -> unload ballast -> gate seals);
        the null policy scores ~0 (gate spawns closed, cartons far outside)."""
        opened = self.gate_open_deep().float()
        delivered = self.milk_in_crib().float()
        base = torch.maximum(0.25 * opened, 0.65 * delivered)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="ballast_gate", robot="null", env_spacing=3.0))
