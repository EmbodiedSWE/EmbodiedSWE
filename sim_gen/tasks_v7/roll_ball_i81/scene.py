"""ArchQuarryScene — the target ball is IMPRISONED under a two-boulder compression
arch inside a V-trough quarry cell; expel one (ungraspable) boulder uphill over the
crest ramp to release the arch, then lift the freed target ball out of the open-top
cell and set it into a hexagonal nest ring on the open floor. Derived from
maniskill/roll_ball but the target cannot even MOVE until a non-target body has been
sacrificially driven out of the scene.

Seed (maniskill/roll_ball): push/roll one free ball across an open table into a flat
goal region — a single unobstructed conveyance; the only physics is rolling on a
plane. Here the conveyance is the trivial last step and everything before it is a
mechanism the seed has no concept of:

- The 50 mm RED target ball rests in the apex groove of a V-TROUGH (two 20-degree
  plates meeting in a line across a walled cell). Two 95 mm, 1.1 kg WHITE BOULDERS
  are wedged above it, each pressing wall-and-ball: the boulders cannot move outward
  (walls), cannot roll fore/aft (the V restores them onto the red ball), and their
  combined ~23 N weight pins the red ball into the groove. Pushing or yanking the red
  ball — the seed's entire skill — moves it nothing (smoke proves an 8 N drag and an
  8 N vertical yank both fail, and that the same yank trivially lifts a FREE red
  ball).
- The boulders are DELIBERATELY ungraspable: 95 mm diameter exceeds the 80 mm
  parallel-jaw opening. The only way to clear the arch is to ROLL one boulder up the
  +u plate (the crest ramp, ~3.9 N sustained uphill push) and over the 75 mm crest,
  off the fixture. The fixture is bound with a slick (mu ~0.12 < tan 20deg) material,
  so an abandoned boulder ALWAYS slides back into the trough and re-locks the arch —
  partial pushes leave nothing behind (restorative interlock, not a ratchet).
- With one boulder expelled, the survivor topples off the red ball to the plate floor
  and the red ball becomes liftable (its 50 mm fits the jaw); it exits through the
  OPEN TOP and must be set INSIDE a low hexagonal NEST RING on the open floor at a
  randomized bearing — inside the rim on the ground, not perched on it, with no
  boulder near the ring.

So the solver needs a different PLAN (clear a mechanism by expelling a NON-target
body uphill against a restoring slope, then extract-and-place the target) and
different CODE STRUCTURE (uphill force servo on a rolling body + vertical contact
extraction + ring placement instead of one flat-ground push). The order is physically
forced: boulder out FIRST — no force the arm can exert on the red ball opens the arch.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - quarry: KINEMATIC compound — two tilted V-plates (back slope + crest ramp), two
    side walls, a tall back wall. Origin at the apex-line centre on the floor; local
    +u points up the crest ramp; slick physics material bound at the root
    (friction_combine_mode="min").
  - nest: KINEMATIC hexagonal ring (6 boxes, 12 mm tall, ~66 mm inner span) on the
    open floor.
  - red ball: DYNAMIC 50 mm sphere, 0.15 kg (graspable).
  - boulders a/b: DYNAMIC 95 mm spheres, 1.1 kg (ungraspable: > 80 mm jaw).

Per-episode randomization (readback-verifiable): quarry xy + yaw, nest bearing sector
+ distance + yaw, red/boulder spawn jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.30 * expelled  — any boulder ever fully out of the cell region (latched; the
                     slick restoring ramp means only a completed crest crossing — or
                     a probe teleport — ever gets one out)
  0.20 * freed     — red ball ever risen clear of the trough / out of the cell
                     (latched; physically impossible before expulsion)
  0.15 * approach  — red-ball approach to the nest centre, gated on freed (running
                     max; ~0 for doing nothing)
  1.0 iff success() — red ball settled ON THE GROUND inside the nest ring, no
                     boulder near the ring. Non-success cap 0.65.

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


# ----- pure-torch quaternion helpers (shared with solve.py) -------------------------------------
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


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way (critical for ROLLING bodies, whose R_now churns every step) and lock
    it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


# ----- custom compound spawners ------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             quat=None) -> None:
    """Author one box collider (optionally rotated: quat = (w, x, y, z))."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _spawn_quarry(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the quarry cell: KINEMATIC compound. Origin at the apex-line centre on
    the floor; the V-trough plates rise at `alpha` toward -u (back slope, capped by a
    tall back wall) and +u (the crest ramp, ending in free air at crest height); side
    walls flank the whole run. A slick physics material is bound at the root
    (min-combine) so the ramp is always restoring (mu < tan alpha)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    a = float(c.alpha)
    sa, ca, ta = math.sin(a), math.cos(a), math.tan(a)
    plate_w = c.in_w + 2 * c.t
    crest_u = c.crest_h / ta
    # crest ramp: top surface z = u * tan(alpha) for u in [0, crest_u]
    n = (-sa, 0.0, ca)  # top-face normal
    m = (crest_u / 2, 0.0, c.crest_h / 2)  # top-face midpoint
    h = a / 2
    _add_box(stage, f"{prim_path}/ramp",
             center=(m[0] - n[0] * c.plate_t / 2, 0.0, m[2] - n[2] * c.plate_t / 2),
             size=(crest_u / ca, plate_w, c.plate_t), color=c.color,
             collide=collide, quat=(math.cos(h), 0.0, -math.sin(h), 0.0))
    # back slope: top surface z = -u * tan(alpha) for u in [-back_u, 0]
    n = (sa, 0.0, ca)
    m = (-c.back_u / 2, 0.0, c.back_u * ta / 2)
    _add_box(stage, f"{prim_path}/back_slope",
             center=(m[0] - n[0] * c.plate_t / 2, 0.0, m[2] - n[2] * c.plate_t / 2),
             size=(c.back_u / ca, plate_w, c.plate_t), color=c.color,
             collide=collide, quat=(math.cos(h), 0.0, math.sin(h), 0.0))
    # side walls: full run from behind the back wall to just past the crest
    u_lo, u_hi = -(c.back_u + c.t), crest_u + 0.004
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((u_lo + u_hi) / 2, sgn * (c.in_w / 2 + c.t / 2), c.wall_h / 2),
                 size=(u_hi - u_lo, c.t, c.wall_h), color=c.wall_color,
                 collide=collide)
    # back wall: tall, seals the -u end above the back slope's top edge
    _add_box(stage, f"{prim_path}/wall_back",
             center=(-(c.back_u + c.t / 2), 0.0, c.back_wall_h / 2),
             size=(c.t, plate_w, c.back_wall_h), color=c.wall_color, collide=collide)
    # slick material at the root (inherits to children; min-combine wins vs balls)
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    sim_utils.spawn_rigid_body_material(
        f"{prim_path}/physmat",
        sim_utils.RigidBodyMaterialCfg(
            static_friction=float(c.mu_static), dynamic_friction=float(c.mu_dynamic),
            restitution=0.0, friction_combine_mode="min",
            restitution_combine_mode="min"))
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _spawn_nest(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the nest ring: KINEMATIC hexagonal rim of 6 boxes on the ground."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    apo = c.inner_apothem + c.rim_t / 2  # centre-line apothem
    side = 2 * apo * math.tan(math.pi / 6) + 0.004
    for k in range(6):
        th = k * math.pi / 3
        h = th / 2
        _add_box(stage, f"{prim_path}/rim_{k}",
                 center=(apo * math.cos(th), apo * math.sin(th), c.rim_h / 2),
                 size=(c.rim_t, side, c.rim_h), color=c.color, collide=collide,
                 quat=(math.cos(h), 0.0, 0.0, math.sin(h)))
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a ball: DYNAMIC sphere. Damping so it rolls to rest; sleep thresholds
    zeroed (force-driven and judged for stillness); velocity iterations 4 (kills the
    GPU capsule/sphere phantom-creep artifact)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.50)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    r = float(cfg.radius)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg.contact_offset)(sph.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "quarry" not in _SPAWNER_CACHE:

        @configclass
        class QuarrySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_quarry)
            alpha: float = math.radians(20.0)
            in_w: float = 0.225
            t: float = 0.015
            wall_h: float = 0.095
            back_wall_h: float = 0.13
            back_u: float = 0.15
            crest_h: float = 0.075
            plate_t: float = 0.012
            mu_static: float = 0.15
            mu_dynamic: float = 0.12
            color: tuple = (0.45, 0.42, 0.38)
            wall_color: tuple = (0.30, 0.33, 0.38)
            contact_offset: float = 0.0015

        @configclass
        class NestSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_nest)
            inner_apothem: float = 0.033
            rim_t: float = 0.012
            rim_h: float = 0.012
            color: tuple = (0.12, 0.35, 0.16)
            contact_offset: float = 0.0015

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.025
            color: tuple = (0.85, 0.10, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(quarry=QuarrySpawnerCfg, nest=NestSpawnerCfg,
                              ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class ArchQuarrySceneCfg(BaseCfg):
    """Config for `ArchQuarryScene`. The interlocks are metric and asserted below:
    the boulders reach across the trough (arch closes), a wedged boulder hangs CLEAR
    of the plates (its weight goes through the red ball, not the floor), the boulders
    exceed the 80 mm jaw while the red ball fits it, and the fixture's friction is
    below tan(alpha) so the crest ramp is always restoring."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.04)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(1.2)  # max |ang vel| when judging (rad/s)
    nest_tol: float = tunable(0.020)  # red-in-nest xy tolerance (m)
    nest_z_lo: float = tunable(0.016)  # red-in-nest centre height band (on ground,
    nest_z_hi: float = tunable(0.031)  # not perched on the rim)
    nest_excl: float = tunable(0.10)  # no boulder centre within this xy of the nest
    appr_d0: float = tunable(0.50)  # approach ramp: p = 1 - d/appr_d0
    freed_z: float = tunable(0.12)  # red local z above which it counts as extracted

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    quarry_xy_max: float = tunable(0.03)  # quarry xy jitter (+/- m)
    quarry_yaw_max_deg: float = tunable(20.0)  # quarry yaw (+/- deg)
    nest_d_min: float = tunable(0.30)  # nest distance from the quarry origin (m)
    nest_d_max: float = tunable(0.38)
    nest_th_min_deg: float = tunable(95.0)  # nest bearing sector (quarry local, off
    nest_th_max_deg: float = tunable(150.0)  # the ramp axis; sign is a coin flip)
    ball_jitter: float = tunable(0.008)  # red/boulder spawn jitter (+/- m)

    # --- info: layout (single Franka base at the origin) -----------------------------------------
    quarry_x: float = info(0.42)  # quarry origin distance from the base

    # --- info: quarry structure ------------------------------------------------------------------
    alpha_deg: float = info(20.0)  # V-plate / crest-ramp slope
    in_w: float = info(0.225)  # inner trough width (v)
    t: float = info(0.015)  # wall thickness
    wall_h: float = info(0.095)  # side-wall height
    back_wall_h: float = info(0.13)  # back-wall height (no exit at -u)
    back_u: float = info(0.15)  # back slope horizontal run
    crest_h: float = info(0.075)  # crest height — the ramp ends in free air here
    plate_t: float = info(0.012)
    mu_static: float = info(0.15)  # fixture material (min-combine): mu < tan(alpha)
    mu_dynamic: float = info(0.12)
    quarry_color: tuple = info((0.45, 0.42, 0.38))
    wall_color: tuple = info((0.30, 0.33, 0.38))

    # --- info: nest ring -------------------------------------------------------------------------
    nest_apothem: float = info(0.033)  # inner apothem (~66 mm opening vs 50 mm ball)
    rim_t: float = info(0.012)
    rim_h: float = info(0.012)
    nest_color: tuple = info((0.12, 0.35, 0.16))

    # --- info: balls -----------------------------------------------------------------------------
    red_r: float = info(0.025)  # 50 mm dia — fits the 80 mm jaw
    red_mass: float = info(0.15)
    red_color: tuple = info((0.85, 0.10, 0.10))
    boulder_r: float = info(0.0475)  # 95 mm dia — EXCEEDS the 80 mm jaw
    boulder_mass: float = info(1.1)
    boulder_color: tuple = info((0.92, 0.92, 0.90))

    contact_offset: float = info(0.0015)
    # rubric weights (0.30 + 0.20 + 0.15 = 0.65 = the non-success cap)
    w_expel: float = info(0.30)
    w_free: float = info(0.20)
    w_app: float = info(0.15)

    def __post_init__(self) -> None:
        a = math.radians(self.alpha_deg)
        rw, rr = self.boulder_r, self.red_r
        # arch closes: a wall-contact boulder reaches the centred red ball
        dv = self.in_w / 2 - rw  # boulder centre |v| at wall contact
        assert (rw + rr) ** 2 > dv * dv, "arch does not close: boulders miss the red ball"
        dz = math.sqrt((rw + rr) ** 2 - dv * dv)
        red_z = rr / math.cos(a)  # red centre height resting in the V groove
        bz = red_z + dz  # wedged boulder centre height
        # wedged boulder hangs CLEAR of the plates: its weight pins the red ball
        plate_clear = bz * math.cos(a) - rw  # distance to each plate plane at u=0
        assert plate_clear > 0.004, f"wedged boulder rests on the plates ({plate_clear:.4f} m)"
        # jaw feasibility: red graspable, boulders not
        assert 2 * rr < 0.080 < 2 * rw, "grasp asymmetry broken (jaw 80 mm)"
        # restoring ramp: fixture friction below the slope angle
        assert self.mu_static < math.tan(a), "ramp is not restoring (mu >= tan alpha)"
        # crest push feasible for a fingertip
        assert self.boulder_mass * 9.81 * math.tan(a) < 5.0, "crest push force infeasible"
        # boulder crowns proud of the side walls (pushable from the open top)
        assert bz + rw > self.wall_h, "wedged boulder crown below the wall top"
        # nest admits the red ball but not a boulder
        assert 2 * self.nest_apothem > 2 * rr + 0.010, "nest opening too tight for the red ball"
        assert 2 * self.nest_apothem < 2 * rw, "a boulder would fit inside the nest"


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("arch_quarry")
class ArchQuarryScene(BaseScene):
    cfg: ArchQuarrySceneCfg

    def __init__(self, cfg: ArchQuarrySceneCfg | None = None) -> None:
        super().__init__(cfg or ArchQuarrySceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        quarry_spawn = spawners["quarry"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            alpha=math.radians(c.alpha_deg), in_w=c.in_w, t=c.t, wall_h=c.wall_h,
            back_wall_h=c.back_wall_h, back_u=c.back_u, crest_h=c.crest_h,
            plate_t=c.plate_t, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            color=c.quarry_color, wall_color=c.wall_color,
            contact_offset=c.contact_offset,
        )
        nest_spawn = spawners["nest"](
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            inner_apothem=c.nest_apothem, rim_t=c.rim_t, rim_h=c.rim_h,
            color=c.nest_color, contact_offset=c.contact_offset,
        )

        def ball_spawn(radius, mass, color):
            return spawners["ball"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                radius=radius, color=color, contact_offset=c.contact_offset,
            )

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
            "quarry": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Quarry",
                spawn=quarry_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.quarry_x, 0.0, 0.0)),
            ),
            "nest": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Nest",
                spawn=nest_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.35, 0.0)),
            ),
            "red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedBall",
                spawn=ball_spawn(c.red_r, c.red_mass, c.red_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.quarry_x, 0.0, 0.03)),
            ),
            "boulder_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BoulderA",
                spawn=ball_spawn(c.boulder_r, c.boulder_mass, c.boulder_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.quarry_x, 0.06, 0.12)),
            ),
            "boulder_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BoulderB",
                spawn=ball_spawn(c.boulder_r, c.boulder_mass, c.boulder_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.quarry_x, -0.06, 0.12)),
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
        self.quarry: RigidObject = env.iscene["quarry"]
        self.nest: RigidObject = env.iscene["nest"]
        self.red: RigidObject = env.iscene["red"]
        self.boulder_a: RigidObject = env.iscene["boulder_a"]
        self.boulder_b: RigidObject = env.iscene["boulder_b"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._expelled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._freed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._app_max = torch.zeros(n, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: quarry re-posed (xy jitter + yaw), nest re-posed on a random
        bearing sector, red ball dropped into the apex groove, boulders dropped from
        just above their wedge seats (they fall onto the red ball and wall and close
        the arch during settling), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- quarry (kinematic): xy jitter + yaw ---
        dxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.quarry_xy_max
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.quarry_yaw_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.quarry_x + dxy[:, 0]
        st[:, 1] = dxy[:, 1]
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.quarry.write_root_state_to_sim(st, env_ids)
        q_pos, q_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- nest (kinematic): random bearing sector off the ramp axis + own yaw ---
        d = c.nest_d_min + torch.rand(m, device=dev) * (c.nest_d_max - c.nest_d_min)
        th = math.radians(c.nest_th_min_deg) + torch.rand(m, device=dev) \
            * math.radians(c.nest_th_max_deg - c.nest_th_min_deg)
        sign = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        th = th * sign
        loc = torch.stack([d * torch.cos(th), d * torch.sin(th),
                           torch.zeros(m, device=dev)], dim=-1)
        nyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = q_pos + _qapply(q_quat, loc)
        st[:, 3], st[:, 6] = torch.cos(nyaw / 2), torch.sin(nyaw / 2)
        self.nest.write_root_state_to_sim(st, env_ids)

        # --- red ball: apex groove (small v jitter), just above its seat ---
        jit = (torch.rand(m, 3, device=dev) * 2 - 1) * c.ball_jitter
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 1] = jit[:, 1]
        loc[:, 2] = c.red_r / math.cos(math.radians(c.alpha_deg)) + 0.004
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = q_pos + _qapply(q_quat, loc)
        st[:, 3] = 1.0
        self.red.write_root_state_to_sim(st, env_ids)

        # --- boulders: dropped from just above their wedge seats, either side ---
        for body, sgn in ((self.boulder_a, 1.0), (self.boulder_b, -1.0)):
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = jit[:, 0]
            loc[:, 1] = sgn * 0.060 + jit[:, 1] * 0.5
            loc[:, 2] = 0.12
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = q_pos + _qapply(q_quat, loc)
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._expelled[env_ids] = False
        self._freed[env_ids] = False
        self._app_max[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "quarry": self.quarry.data.root_state_w[env_ids].clone(),
            "nest": self.nest.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "boulder_a": self.boulder_a.data.root_state_w[env_ids].clone(),
            "boulder_b": self.boulder_b.data.root_state_w[env_ids].clone(),
            "expelled": self._expelled[env_ids].clone(),
            "freed": self._freed[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.quarry.write_root_state_to_sim(state["quarry"], env_ids)
        self.nest.write_root_state_to_sim(state["nest"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.boulder_a.write_root_state_to_sim(state["boulder_a"], env_ids)
        self.boulder_b.write_root_state_to_sim(state["boulder_b"], env_ids)
        self._expelled[env_ids] = state["expelled"]
        self._freed[env_ids] = state["freed"]
        self._app_max[env_ids] = state["app_max"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands an open-top QUARRY CELL: two {c.alpha_deg:.0f}-degree "
            f"plates meet in a V-groove across a walled trough ({c.in_w * 100:.1f} cm "
            f"between the side walls, walls {c.wall_h * 100:.1f} cm tall). Toward one "
            f"end the plate keeps rising as a CREST RAMP and ends in free air "
            f"{c.crest_h * 100:.1f} cm up — beyond the crest is open floor; the other "
            f"end is sealed by a taller back wall. In the groove sits the "
            f"{2 * c.red_r * 1000:.0f} mm RED TARGET BALL, and WEDGED ABOVE IT are two "
            f"{2 * c.boulder_r * 1000:.0f} mm, {c.boulder_mass:.1f} kg WHITE BOULDERS, "
            f"each pressing wall-and-ball: the walls stop them moving outward, the V "
            f"restores them onto the red ball fore/aft, and their combined weight "
            f"(~{2 * c.boulder_mass * 9.81:.0f} N) PINS the red ball in the groove. "
            f"No push or pull on the red ball moves it while the arch stands. The "
            f"boulders are TOO BIG for the {0.080 * 1000:.0f} mm gripper jaw — they "
            f"cannot be picked out; the only way to break the arch is to ROLL ONE "
            f"BOULDER UP THE CREST RAMP (a sustained ~4 N push; its crown stands "
            f"proud of the walls and the open top admits the hand) and over the "
            f"crest, off the fixture. The fixture is slick: a boulder abandoned on "
            f"the ramp ALWAYS rolls back down and re-locks the arch — only a "
            f"completed crossing counts. Once a boulder is out, the survivor topples "
            f"aside and the red ball comes free: lift it out through the open top "
            f"(it fits the jaw) and set it INSIDE the GREEN HEXAGONAL NEST RING lying "
            f"on the open floor (inner span ~{2 * c.nest_apothem * 100:.1f} cm, rim "
            f"{c.rim_h * 1000:.0f} mm) — resting on the ground within the rim, not "
            f"perched on it.\n"
            f"Goal: the RED ball settled on the ground INSIDE the nest ring, with no "
            f"boulder near the ring (none within {c.nest_excl * 100:.0f} cm of its "
            f"centre). Expelling a boulder first is REQUIRED — the arch cannot be "
            f"opened any other way. Dragging or yanking the red ball while the arch "
            f"stands, part-pushing a boulder up the ramp, leaving the red ball beside "
            f"or on top of the ring, or parking a boulder at the ring — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Roll one white boulder up the crest ramp and over the edge so the arch "
            "pinning the red ball releases, then lift the red ball out of the trough "
            "and set it down inside the green hexagonal nest ring on the floor."
        )

    # ----- readings / rubric ---------------------------------------------------------------------
    def _quarry_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM in the quarry frame (u up the ramp, v across, z up)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.quarry.data.root_pos_w
        return quat_apply_inverse(self.quarry.data.root_quat_w, rel)

    def _nest_xy_d(self, body: RigidObject) -> torch.Tensor:
        """(N,) horizontal distance from the body CoM to the nest centre."""
        rel = body.data.root_pos_w - self.nest.data.root_pos_w
        return rel[:, :2].norm(dim=-1)

    def _in_cell(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM inside the quarry cell region (generous margins)."""
        c = self.cfg
        loc = self._quarry_local(body)
        crest_u = c.crest_h / math.tan(math.radians(c.alpha_deg))
        return ((loc[:, 0] > -(c.back_u + 0.03)) & (loc[:, 0] < crest_u + 0.055)
                & (loc[:, 1].abs() < c.in_w / 2 + c.t + 0.02) & (loc[:, 2] < 0.30))

    def _still(self, body: RigidObject) -> torch.Tensor:
        c = self.cfg
        return ((body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _red_in_nest(self) -> torch.Tensor:
        """(N,) bool: red ball on the ground inside the nest rim (not perched)."""
        c = self.cfg
        z = self.red.data.root_pos_w[:, 2] - self.nest.data.root_pos_w[:, 2]
        return ((self._nest_xy_d(self.red) < c.nest_tol)
                & (z > c.nest_z_lo) & (z < c.nest_z_hi))

    def _update_latches(self) -> None:
        c = self.cfg
        self._expelled |= ~self._in_cell(self.boulder_a) | ~self._in_cell(self.boulder_b)
        red_loc = self._quarry_local(self.red)
        self._freed |= (red_loc[:, 2] > c.freed_z) | ~self._in_cell(self.red)
        app = (1.0 - self._nest_xy_d(self.red) / c.appr_d0).clamp(0.0, 1.0) \
            * self._freed.float()
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)

    # ----- step-coupled bookkeeping (every substep) ----------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant here (the fixture is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: red ball settled on the ground inside the nest ring, no boulder
        near the ring. Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        clear = ((self._nest_xy_d(self.boulder_a) > c.nest_excl)
                 & (self._nest_xy_d(self.boulder_b) > c.nest_excl))
        finite = (torch.isfinite(self.red.data.root_state_w).all(dim=-1)
                  & torch.isfinite(self.boulder_a.data.root_state_w).all(dim=-1)
                  & torch.isfinite(self.boulder_b.data.root_state_w).all(dim=-1))
        return self._red_in_nest() & self._still(self.red) & clear & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*expelled + 0.20*freed + 0.15*approach (gated on
        freed) — all latched, ~0 for doing nothing, capped 0.65 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_expel * self._expelled.float() + c.w_free * self._freed.float()
                + c.w_app * self._app_max).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="arch_quarry", robot="null"))
