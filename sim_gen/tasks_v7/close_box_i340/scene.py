"""SlamShutCourierScene — close the box by SLAMMING it shut: shove the courier box
down a roofed alley so it slides free, hits the end bumper, and the arrest itself
flips the over-center lid closed over the cargo.

Derived from rlbench/close_box ("close the box": an articulated box USD whose open lid
the robot pushes shut about its hinge, judged by the lid joint angle). The hinge is
KEPT here — but the seed's closing action (push the lid panel) is made geometrically
impossible almost everywhere: the box sits inside a slick alley under a LOW ROOF whose
underside intersects the lid's closing sweep (the sweep apex pokes ~10 mm above the
roof plane), so any attempt to rotate the lid shut in place jams on the roof ~110 deg
from closed. The lid is gravity-BISTABLE (its open rest, -120 deg, is past vertical),
so it never falls shut on its own. The only place the sweep clears is a high-canopy
arrest chamber at the far end, whose lid airspace the low roof stops guarding only
within ~15 mm of the end bumper. Closure must therefore be MANUFACTURED BY INERTIA:
accelerate the box to ~1 m/s, let it coast down the alley, and let the dead-stop
arrest at the bumper convert the box's momentum into lid rotation — the lid's angular
momentum about the suddenly-arrested hinge carries it over the apex and gravity slams
it home over the cargo cube, which the box walls retain through the crash.

Assets are fully procedural:
  - fixture: KINEMATIC alley — slick floor slab, two side walls, a LOW roof over the
    launch/coast run, a HIGH canopy over the arrest chamber, a full-height end bumper,
    and a small one-way SILL behind the box spawn (the box can only go forward).
  - box: DYNAMIC compound courier box (160 x 160 mm footprint, 100 mm walls, 0.6 kg,
    low CoM) with a spawn-authored revolute LID (155 x 150 x 8 mm, 0.08 kg) hinged at
    the rear top edge, joint limits [-open_deg, 0] (0 = closed flat on the walls).
  - cargo: DYNAMIC 30 mm gold cube inside the box.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.20 * entered — box centre ever deep in the alley (past `enter_x`)       (latched)
  0.20 * arrived — box front ever within `arrive_tol` of the bumper        (latched)
  0.25 * shut    — lid ever within `shut_deg` of closed (unreachable by
                   pushing the lid in place: the roof jam stops ~110 deg
                   from closed, far short of `shut_deg`)                    (latched)
  1.0 iff success() — box DELIVERED (front within `deliver_tol` of the bumper, in
                   the alley, upright), lid CLOSED (within `closed_deg` of the
                   limit), cargo cube inside the box below the lid plane, everything
                   settled and finite. Non-success capped at 0.70.

Honesty geometry (asserted in `__post_init__`):
  - the lid's open rest is over-center (gravity holds it open; holds it shut once
    closed) and its leaning tail rides clear UNDER the low roof during the coast;
  - the closing sweep apex rises ABOVE the low roof plane (in-place closure jams),
    the high canopy clears the apex (the flip can complete at the bumper), and the
    roof-jam angle is far short of the `shut` latch;
  - the low roof ends so the closure window opens only within ~15 mm of full
    delivery; the closed lid covers the cavity and clears the bumper;
  - the arrest speed needed to carry the lid over the apex (rod-arrest model) is
    reachable with margin from a launch inside the accessible entry;
  - spawn bands keep the box clear of the sill and walls under all jitters.

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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


# ----- custom compound spawners -----------------------------------------------------------------
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
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


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC alley at `prim_path`. Local frame: origin at the launch
    reference on the ground, alley running toward local +x, slab top at z=slab_t."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    x_mid = (c.slab_x0 + c.slab_x1) / 2
    slab_len = c.slab_x1 - c.slab_x0
    wall_cy = c.wall_in + c.wall_t / 2
    width = 2 * (c.wall_in + c.wall_t)
    x_hi1 = c.x_bump + c.bump_t
    kids = [
        # slick floor slab the box rides on
        _box(stage, f"{prim_path}/slab", center=(x_mid, 0.0, c.slab_t / 2),
             size=(slab_len, width, c.slab_t), color=c.floor_color, contact_offset=co),
        # side walls
        _box(stage, f"{prim_path}/wall_p", center=(x_mid, wall_cy, c.wall_h / 2),
             size=(slab_len, c.wall_t, c.wall_h), color=c.wall_color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_n", center=(x_mid, -wall_cy, c.wall_h / 2),
             size=(slab_len, c.wall_t, c.wall_h), color=c.wall_color, contact_offset=co),
        # LOW roof over the launch/coast run (blocks the lid's closing sweep)
        _box(stage, f"{prim_path}/roof_lo",
             center=((c.roof_lo_x0 + c.roof_lo_x1) / 2, 0.0, c.roof_lo_z + c.roof_t / 2),
             size=(c.roof_lo_x1 - c.roof_lo_x0, width, c.roof_t),
             color=c.roof_color, contact_offset=co),
        # HIGH canopy over the arrest chamber (clears the flip, blocks the hand)
        _box(stage, f"{prim_path}/roof_hi",
             center=((c.roof_lo_x1 + x_hi1) / 2, 0.0, c.roof_hi_z + c.roof_t / 2),
             size=(x_hi1 - c.roof_lo_x1, width, c.roof_t),
             color=c.canopy_color, contact_offset=co),
        # end bumper (full height: also stops the box from pitching over on impact)
        _box(stage, f"{prim_path}/bumper",
             center=(c.x_bump + c.bump_t / 2, 0.0, c.wall_h / 2),
             size=(c.bump_t, width, c.wall_h), color=c.bump_color, contact_offset=co),
        # one-way sill behind the spawn: the box can never be backed out of the alley
        _box(stage, f"{prim_path}/sill",
             center=((c.sill_x0 + c.sill_x1) / 2, 0.0, (c.slab_t + c.sill_top) / 2),
             size=(c.sill_x1 - c.sill_x0, 2 * c.wall_in, c.sill_top - c.slab_t),
             color=c.bump_color, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_box(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC courier box at `prim_path`. Local frame: origin at the
    footprint centre on the box's bottom face; open mouth up, front toward +x."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.8e-3, 1.8e-3, 2.6e-3))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.02)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    wall_cz = (c.floor_t + c.height) / 2
    wall_hz = c.height - c.floor_t
    kids = [
        _box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(2 * c.hx, 2 * c.hy, c.floor_t), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_front",
             center=(c.hx - c.wall_t / 2, 0.0, wall_cz),
             size=(c.wall_t, 2 * c.hy, wall_hz), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_rear",
             center=(-(c.hx - c.wall_t / 2), 0.0, wall_cz),
             size=(c.wall_t, 2 * c.hy, wall_hz), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_left",
             center=(0.0, c.hy - c.wall_t / 2, wall_cz),
             size=(2 * (c.hx - c.wall_t), c.wall_t, wall_hz),
             color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_right",
             center=(0.0, -(c.hy - c.wall_t / 2), wall_cz),
             size=(2 * (c.hx - c.wall_t), c.wall_t, wall_hz),
             color=c.color, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC lid at `prim_path`: root origin ON the hinge line, plate
    extending toward local +x (so joint angle 0 = closed flat over the box mouth).
    A revolute joint (axis Y) to the sibling Box is authored in-spawn."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(float(cfg.length) / 2, 0.0, float(cfg.t) / 2))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.5e-4, 1.6e-4, 3.1e-4))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.02)
    px.CreateAngularDampingAttr(0.03)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    plate = _box(stage, f"{prim_path}/plate",
                 center=(float(cfg.length) / 2, 0.0, float(cfg.t) / 2),
                 size=(cfg.length, cfg.width, cfg.t), color=cfg.color,
                 contact_offset=cfg.contact_offset)
    _bind_material(plate, mat)
    # revolute hinge to the sibling Box (pair collision FILTERED by the joint;
    # the closed/open rests are the joint limits themselves)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Box"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in cfg.hinge_in_box]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.open_deg))
    j.CreateUpperLimitAttr(0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            slab_x0: float = -0.16
            slab_x1: float = 0.66
            slab_t: float = 0.004
            wall_in: float = 0.088
            wall_t: float = 0.012
            wall_h: float = 0.30
            roof_lo_x0: float = -0.11
            roof_lo_x1: float = 0.369
            roof_lo_z: float = 0.2485
            roof_hi_z: float = 0.270
            roof_t: float = 0.008
            x_bump: float = 0.60
            bump_t: float = 0.012
            sill_x0: float = -0.022
            sill_x1: float = -0.008
            sill_top: float = 0.018
            mu_static: float = 0.05
            mu_dynamic: float = 0.04
            floor_color: tuple = (0.55, 0.57, 0.60)
            wall_color: tuple = (0.45, 0.47, 0.50)
            roof_color: tuple = (0.35, 0.37, 0.42)
            canopy_color: tuple = (0.60, 0.55, 0.30)
            bump_color: tuple = (0.70, 0.25, 0.15)
            contact_offset: float = 0.002

        @configclass
        class BoxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_box)
            hx: float = 0.08
            hy: float = 0.08
            height: float = 0.10
            wall_t: float = 0.008
            floor_t: float = 0.008
            mass: float = 0.6
            com_z: float = 0.03
            mu_static: float = 0.05
            mu_dynamic: float = 0.04
            color: tuple = (0.15, 0.45, 0.50)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            length: float = 0.155
            width: float = 0.150
            t: float = 0.008
            mass: float = 0.08
            open_deg: float = 120.0
            hinge_in_box: tuple = (-0.08, 0.0, 0.10)
            mu_static: float = 0.05
            mu_dynamic: float = 0.04
            color: tuple = (0.85, 0.30, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["fixture"] = FixtureSpawnerCfg
        _SPAWNER_CACHE["box"] = BoxSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SlamShutCourierSceneCfg(BaseCfg):
    """Config for `SlamShutCourierScene`. The low roof jams any in-place lid closure,
    the high canopy admits the arrest flip, the sill makes the alley one-way, and the
    arrest speed is reachable with margin. All asserted in `__post_init__`."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    enter_x: float = tunable(0.20)          # box centre past this fixture-x -> `entered`
    arrive_tol: float = tunable(0.040)      # box front within this of the bumper -> `arrived`
    shut_deg: float = tunable(45.0)         # lid ever within this of closed -> `shut`
    deliver_tol: float = tunable(0.025)     # box front within this of the bumper at judging
    closed_deg: float = tunable(10.0)       # lid within this of the closed limit
    alley_y_tol: float = tunable(0.030)     # |box y| in the fixture frame when judged
    upright_deg: float = tunable(10.0)      # box up-axis cone
    cargo_pad: float = tunable(0.010)       # cavity bound slack for the cargo test
    cargo_z_lo: float = tunable(0.010)      # cargo centre band in the BOX frame
    cargo_z_hi: float = tunable(0.088)      # (below the closed-lid plane at 0.10)
    settle_speed: float = tunable(0.05)     # max |lin vel| (box, lid, cargo) when judging

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    fix_yaw_deg: float = tunable(20.0)      # alley heading jitter (+/- deg)
    fix_jitter: float = tunable(0.030)      # alley xy jitter (+/- m)
    box_x_jitter: float = tunable(0.012)    # box spawn along-alley jitter (+/- m)
    box_y_jitter: float = tunable(0.002)    # box spawn cross-alley jitter (+/- m)
    cargo_jitter: float = tunable(0.020)    # cargo box-frame xy jitter (+/- m)

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    fix_pos: tuple = info((0.30, 0.0))      # fixture origin (alley entry reference)
    fix_yaw_nom_deg: float = info(0.0)      # alley runs away from the robot
    box_spawn_x: float = info(0.090)        # box centre, fixture frame
    # --- info: fixture structure (must match FixtureSpawnerCfg) ----------------------------------
    slab_x0: float = info(-0.16)
    slab_x1: float = info(0.66)
    slab_t: float = info(0.004)
    wall_in: float = info(0.088)
    wall_t: float = info(0.012)
    wall_h: float = info(0.30)
    roof_lo_x0: float = info(-0.11)
    roof_lo_x1: float = info(0.369)
    roof_lo_z: float = info(0.2485)
    roof_hi_z: float = info(0.270)
    roof_t: float = info(0.008)
    x_bump: float = info(0.60)
    bump_t: float = info(0.012)
    sill_x0: float = info(-0.022)
    sill_x1: float = info(-0.008)
    sill_top: float = info(0.018)
    mu_static: float = info(0.05)           # alley + box + lid material (slick run)
    mu_dynamic: float = info(0.04)
    # --- info: box + lid -------------------------------------------------------------------------
    box_hx: float = info(0.08)
    box_hy: float = info(0.08)
    box_h: float = info(0.10)
    box_wall_t: float = info(0.008)
    box_floor_t: float = info(0.008)
    box_mass: float = info(0.6)
    lid_L: float = info(0.155)
    lid_w: float = info(0.150)
    lid_t: float = info(0.008)
    lid_mass: float = info(0.08)
    open_deg: float = info(120.0)
    # --- info: cargo -----------------------------------------------------------------------------
    cargo: float = info(0.030)
    cargo_mass: float = info(0.05)
    cargo_color: tuple = info((0.85, 0.70, 0.15))
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.40)
    # --- info: solve support ---------------------------------------------------------------------
    release_x: float = info(0.26)           # box-centre fixture-x at launch force cutoff
    v_margin: float = info(2.2)             # energy margin on the arrest-flip condition
    # rubric weights (0.20 + 0.20 + 0.25 = 0.65; non-success cap 0.70)
    w_enter: float = info(0.20)
    w_arrive: float = info(0.20)
    w_shut: float = info(0.25)

    # ----- derived (computed, not tuned) ---------------------------------------------------------
    @property
    def z_hinge(self) -> float:
        """Hinge height above the fixture ground plane (box rides the slab)."""
        return self.slab_t + self.box_h

    @property
    def hinge_b(self) -> float:
        """Hinge fixture-x when the box front face touches the bumper."""
        return self.x_bump - 2 * self.box_hx

    @property
    def apex_z(self) -> float:
        """Lid tip height at the top of the closing sweep (lid vertical)."""
        return self.z_hinge + self.lid_L

    @property
    def tail_z(self) -> float:
        """Lid tip height at the open rest (leaning back over-center)."""
        return self.z_hinge + self.lid_L * math.sin(math.radians(self.open_deg))

    @property
    def alpha_cross(self) -> float:
        """Sweep angle-from-closed (rad) where the lid tip crosses the low-roof plane
        on the OPEN side of the apex."""
        return math.pi - math.asin((self.roof_lo_z - self.z_hinge) / self.lid_L)

    @property
    def x_close_min(self) -> float:
        """Min hinge fixture-x from which the closing sweep clears the low roof."""
        return self.roof_lo_x1 - self.lid_L * math.cos(self.alpha_cross)

    @property
    def v_req(self) -> float:
        """Arrest speed that just carries the lid over the apex (rod-arrest model:
        omega = 3 v sin(open) / (2 L), KE_rot >= dPE to the apex)."""
        s = math.sin(math.radians(self.open_deg))
        return math.sqrt((4.0 * 9.81 * self.lid_L / 3.0) * (1.0 - s) / (s * s))

    @property
    def x_delivered(self) -> float:
        """Min box-centre fixture-x that counts as delivered."""
        return self.x_bump - self.box_hx - self.deliver_tol

    @property
    def x_arrived(self) -> float:
        """Min box-centre fixture-x that latches `arrived`."""
        return self.x_bump - self.box_hx - self.arrive_tol

    def __post_init__(self) -> None:
        th = math.radians(self.open_deg)
        # over-center bistable lid: open rest past vertical, gravity holds both rests
        assert 95.0 < self.open_deg < 150.0, "lid open rest must be over-center"
        # the leaning open lid rides clear UNDER the low roof during the coast...
        assert self.tail_z < self.roof_lo_z - 0.008, "open lid would scrape the low roof"
        # ...but the closing sweep apex rises ABOVE the low roof plane: in-place
        # closure under the low roof always jams on the roof underside
        assert self.apex_z > self.roof_lo_z + 0.008, "low roof would admit an in-place closure"
        # the high canopy clears the apex: the flip can complete in the chamber
        assert self.roof_hi_z > self.apex_z + 0.008, "canopy would block the arrest flip"
        # the closure window opens only within ~15-25 mm of full delivery
        window = self.hinge_b - self.x_close_min
        assert 0.005 < window < 0.025, f"closure window {window * 1000:.1f}mm out of band"
        # during the flip at the bumper the tip passes the low-roof end below it
        z_at_end = self.z_hinge + self.lid_L * math.sin(
            math.acos((self.roof_lo_x1 - self.hinge_b) / self.lid_L))
        assert z_at_end < self.roof_lo_z - 0.004, "flip would clip the low-roof end face"
        # the roof jam stops the lid FAR short of the `shut` latch
        jam_deg = math.degrees(self.alpha_cross)
        assert jam_deg > self.shut_deg + 30.0, "`shut` latch reachable by the roof jam"
        # the closed lid covers the cavity, rests on the walls' plane, clears the bumper
        tip = -self.box_hx + self.lid_L
        assert tip >= self.box_hx - self.box_wall_t + 0.002, "closed lid would not cover the mouth"
        assert tip <= self.box_hx - 0.004, "closed lid would touch the bumper at delivery"
        assert self.lid_w >= 2 * (self.box_hx - self.box_wall_t) + 0.004, "lid too narrow"
        assert self.lid_w <= 2 * self.wall_in - 0.010, "lid would jam between the alley walls"
        # box fits the alley with clearance beyond the spawn band
        gap = self.wall_in - self.box_hy
        assert gap >= self.box_y_jitter + 0.005, "box could spawn against an alley wall"
        # spawn clear of the sill; sill still tall enough to be one-way
        rear_min = self.box_spawn_x - self.box_hx - self.box_x_jitter
        assert rear_min >= self.sill_x1 + 0.005, "box could spawn on the sill"
        assert self.sill_top > self.slab_t + 0.008, "sill too low to block back-out"
        # open lid tail stays under the low roof at spawn (no reachable lid surface)
        hinge_min = self.box_spawn_x - self.box_hx - self.box_x_jitter
        tail_x = hinge_min + self.lid_L * math.cos(th)
        assert tail_x > self.roof_lo_x0 + 0.010, "open lid tail would poke out of the roof"
        # launch feasibility: v_req reachable with margin from an in-reach release
        d_coast = self.x_bump - self.box_hx - self.release_x
        assert d_coast > 0.12, "no room to coast"
        v_launch = math.sqrt(self.v_margin * self.v_req ** 2
                             + 2 * self.mu_dynamic * 9.81 * d_coast)
        assert v_launch < 1.5, "required launch speed implausibly high"
        assert self.release_x < self.roof_lo_x1 - 0.05, "release point past the low roof"
        # cargo: fits the cavity under all jitters, below the closed-lid plane
        in_hx = self.box_hx - self.box_wall_t
        assert self.cargo_jitter + self.cargo / 2 < in_hx - 0.005, "cargo could spawn in a wall"
        # a cargo ON the closed lid rests at box-z >= box_h + lid_t + cargo/2: the
        # accepted band must top out below the lid plane
        assert self.cargo_z_hi <= self.box_h - 0.005, \
            "a cargo ON the lid must fail the contents test"
        assert self.box_floor_t + self.cargo < self.box_h - 0.010, "no headroom for the cargo"
        # rubric thresholds ordered along the alley
        assert self.box_spawn_x + self.box_x_jitter + 0.05 < self.enter_x < self.x_arrived \
            < self.x_delivered < self.x_bump - self.box_hx, "latch thresholds out of order"
        assert self.deliver_tol < self.arrive_tol, "deliver band must be tighter than arrive"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("slam_shut_courier")
class SlamShutCourierScene(BaseScene):
    cfg: SlamShutCourierSceneCfg

    def __init__(self, cfg: SlamShutCourierSceneCfg | None = None) -> None:
        super().__init__(cfg or SlamShutCourierSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        fixture_spawn = cls["fixture"](
            slab_x0=c.slab_x0, slab_x1=c.slab_x1, slab_t=c.slab_t, wall_in=c.wall_in,
            wall_t=c.wall_t, wall_h=c.wall_h, roof_lo_x0=c.roof_lo_x0,
            roof_lo_x1=c.roof_lo_x1, roof_lo_z=c.roof_lo_z, roof_hi_z=c.roof_hi_z,
            roof_t=c.roof_t, x_bump=c.x_bump, bump_t=c.bump_t, sill_x0=c.sill_x0,
            sill_x1=c.sill_x1, sill_top=c.sill_top, mu_static=c.mu_static,
            mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)
        box_spawn = cls["box"](
            hx=c.box_hx, hy=c.box_hy, height=c.box_h, wall_t=c.box_wall_t,
            floor_t=c.box_floor_t, mass=c.box_mass, mu_static=c.mu_static,
            mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)
        lid_spawn = cls["lid"](
            length=c.lid_L, width=c.lid_w, t=c.lid_t, mass=c.lid_mass,
            open_deg=c.open_deg, hinge_in_box=(-c.box_hx, 0.0, c.box_h),
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=fixture_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0], c.fix_pos[1], 0.0)),
            ),
            "box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box",
                spawn=box_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + c.box_spawn_x, c.fix_pos[1], c.slab_t)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + c.box_spawn_x - c.box_hx, c.fix_pos[1],
                         c.slab_t + c.box_h),
                    rot=(math.cos(math.radians(c.open_deg) / 2), 0.0,
                         -math.sin(math.radians(c.open_deg) / 2), 0.0)),
            ),
            "cargo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cargo, c.cargo, c.cargo),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cargo_color),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + c.box_spawn_x, c.fix_pos[1],
                         c.slab_t + c.box_floor_t + c.cargo / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.fixture: RigidObject = env.iscene["fixture"]
        self.box: RigidObject = env.iscene["box"]
        self.lid: RigidObject = env.iscene["lid"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arrived = torch.zeros(n, dtype=torch.bool, device=dev)
        self._shut = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: alley heading + xy jitter; box at the launch pad (along-
        alley jitter), lid written CONSISTENTLY at the open joint limit; cargo cube on
        the box floor (box-frame jitter); latches cleared. Box and lid are one linkage
        — both root states are written back to back with no stepping in between."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for _ in range(4):  # burn post-seed draws (early Philox draws are seed-correlated)
            torch.rand(2 * m, device=dev)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, pos, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, env_ids)

        # --- fixture: kinematic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.fix_yaw_nom_deg) + rnd(math.radians(c.fix_yaw_deg))
        q_fix = _qz(yaw)
        fpos = torch.zeros(m, 3, device=dev)
        fpos[:, 0] = c.fix_pos[0] + rnd(c.fix_jitter)
        fpos[:, 1] = c.fix_pos[1] + rnd(c.fix_jitter)
        write(self.fixture, fpos, q_fix)

        # --- box + lid (one linkage, consistent poses, no stepping between) ---
        bloc = torch.zeros(m, 3, device=dev)
        bloc[:, 0] = c.box_spawn_x + rnd(c.box_x_jitter)
        bloc[:, 1] = rnd(c.box_y_jitter)
        bloc[:, 2] = c.slab_t
        bpos = fpos + _qapply(q_fix, bloc)
        q_box = q_fix.clone()
        write(self.box, bpos, q_box)
        hinge = torch.tensor([-c.box_hx, 0.0, c.box_h], device=dev).expand(m, 3)
        lpos = bpos + _qapply(q_box, hinge)
        q_lid = _qmul(q_box, _qy(torch.full((m,), -math.radians(c.open_deg), device=dev)))
        write(self.lid, lpos, q_lid)

        # --- cargo: on the box floor, box-frame xy jitter ---
        kloc = torch.zeros(m, 3, device=dev)
        kloc[:, 0] = rnd(c.cargo_jitter)
        kloc[:, 1] = rnd(c.cargo_jitter)
        kloc[:, 2] = c.box_floor_t + c.cargo / 2 + 0.002
        write(self.cargo, bpos + _qapply(q_box, kloc), _qmul(q_box, _qz(rnd(math.pi))))

        self._entered[env_ids] = False
        self._arrived[env_ids] = False
        self._shut[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fixture": self.fixture.data.root_state_w[env_ids].clone(),
            "box": self.box.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
            "arrived": self._arrived[env_ids].clone(),
            "shut": self._shut[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fixture.write_root_state_to_sim(state["fixture"], env_ids)
        self.box.write_root_state_to_sim(state["box"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        self._entered[env_ids] = state["entered"]
        self._arrived[env_ids] = state["arrived"]
        self._shut[env_ids] = state["shut"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slick metal ALLEY stands on the floor: a low grey duct "
            f"({2 * c.wall_in * 1000:.0f} mm wide inside) whose near end is open and "
            f"whose far end is a red BUMPER wall, with a low roof over the near run and "
            f"a taller yellow canopy over the far chamber. Just inside the entry, a teal "
            f"COURIER BOX ({2 * c.box_hx * 1000:.0f} mm square, {c.box_h * 1000:.0f} mm "
            f"walls) sits on the slick floor holding a {c.cargo * 1000:.0f} mm GOLD cargo "
            f"cube. The box's orange LID is hinged at its rear top edge and rests leaned "
            f"back OVER-CENTER ({c.open_deg:.0f} deg open) — gravity holds it open, and "
            f"the low roof sits inside the lid's closing arc, so the lid physically "
            f"CANNOT be swung shut while the box is under it (it jams on the roof); a "
            f"sill behind the box means the box can only move deeper in. The alley's "
            f"position and heading, the box's spot on the launch pad and the cargo's "
            f"spot in the box all vary per episode.\n"
            f"Goal: CLOSE the box, delivered. Shove the box down the alley hard enough "
            f"that it slides free and SLAMS into the red bumper — the sudden stop flips "
            f"the lid up over its balance point and gravity drops it shut over the "
            f"cargo. To count, the box must end at the bumper (front within "
            f"{c.deliver_tol * 1000:.0f} mm), upright in the alley, with the lid within "
            f"{c.closed_deg:.0f} deg of fully closed and the gold cube still inside "
            f"below the lid. A box parked short, a lid still open or only partly "
            f"swung, or cargo outside the box does not count; a gentle push to the "
            f"bumper leaves the lid open. Nothing may still be moving when judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Close the courier box by slamming it: shove the teal box hard down the "
            "alley so it slides into the red bumper and the impact flips its orange "
            "lid shut over the gold cargo cube. It must end at the bumper, lid fully "
            "closed, cube inside."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _fix_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) fixture frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.fixture.data.root_quat_w),
                       pos_w - self.fixture.data.root_pos_w)

    def box_local(self) -> torch.Tensor:
        """(N,3) box centre in the fixture frame."""
        return self._fix_local(self.box.data.root_pos_w)

    def lid_angle(self) -> torch.Tensor:
        """(N,) lid hinge angle in RADIANS: 0 = closed flat, -open_deg = open rest."""
        q_rel = _qmul(_qinv(self.box.data.root_quat_w), self.lid.data.root_quat_w)
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        d = _qapply(q_rel, ex)
        return torch.atan2(-d[:, 2], d[:, 0])

    def upright(self) -> torch.Tensor:
        """(N,) bool: box up-axis within the upright cone."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        u = _qapply(self.box.data.root_quat_w, ez)
        return u[:, 2] > math.cos(math.radians(self.cfg.upright_deg))

    def in_alley(self) -> torch.Tensor:
        """(N,) bool: box centred in the alley band, on the slab run."""
        c = self.cfg
        p = self.box_local()
        return (p[:, 1].abs() < c.alley_y_tol) & (p[:, 0] > c.sill_x1) \
            & (p[:, 0] < c.x_bump) & (p[:, 2] < c.slab_t + 0.02)

    def delivered(self) -> torch.Tensor:
        """(N,) bool: box front pressed to within `deliver_tol` of the bumper."""
        c = self.cfg
        return self.in_alley() & self.upright() & (self.box_local()[:, 0] >= c.x_delivered)

    def closed(self) -> torch.Tensor:
        """(N,) bool: lid within `closed_deg` of the closed limit."""
        return self.lid_angle() > -math.radians(self.cfg.closed_deg)

    def cargo_in(self) -> torch.Tensor:
        """(N,) bool: cargo cube inside the box cavity BELOW the lid plane."""
        c = self.cfg
        loc = _qapply(_qinv(self.box.data.root_quat_w),
                      self.cargo.data.root_pos_w - self.box.data.root_pos_w)
        in_h = c.box_hx - c.box_wall_t - c.cargo / 2 + c.cargo_pad
        return (loc[:, 0].abs() < in_h) & (loc[:, 1].abs() < in_h) \
            & (loc[:, 2] > c.cargo_z_lo) & (loc[:, 2] < c.cargo_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: box, lid and cargo |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.box, self.lid, self.cargo)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.box, self.lid, self.cargo)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        px = self.box_local()[:, 0]
        ok = self.in_alley() & self.upright() & fin
        self._entered |= ok & (px > c.enter_x)
        self._arrived |= ok & (px >= c.x_arrived)
        self._shut |= fin & (self.lid_angle() > -math.radians(c.shut_deg))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: box delivered to the bumper, lid closed, cargo inside below the
        lid, everything settled and finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.delivered() & self.closed() & self.cargo_in() & self.settled() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*entered + 0.20*arrived + 0.25*shut (all latched;
        ~0 for doing nothing — the spawned box is well short of `enter_x` and the open
        lid rests 120 deg from closed), capped at 0.70 — and exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_enter * self._entered.float() + c.w_arrive * self._arrived.float()
                + c.w_shut * self._shut.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="slam_shut_courier", robot="null"))
