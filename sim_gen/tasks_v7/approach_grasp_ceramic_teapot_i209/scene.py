"""TeaBallTransferScene — unseal the capped tea canister, move the steel infuser
ball into the OTHER (open) canister, and seat the stopper lid in THAT canister's
funnel mouth. Derived from pick_place/approach_grasp_ceramic_teapot but the entire
plan is replaced: the episode is judged on a re-SEALED container, not a held object.

Seed (pick_place/approach_grasp_ceramic_teapot): a Franka approaches a ceramic
teapot among table clutter and closes its jaw around it; success is a
gripper-object distance relation held for a few frames, then a small joint lift —
the episode ends HOLDING the object, and the rubric reads the gripper. Here no
gripper relation is ever read and holding anything is worth nothing:

- The payload (a 36 mm steel infuser ball) starts SEALED inside one of two squat
  octagonal tea canisters, under a loose stopper lid. While the lid is seated the
  ball is physically caged (smoke shoves it with a bounded lateral force and it
  cannot leave) — so the order unseal -> extract is geometry-forced.
- The goal is a state SWAP with a seal: ball resting INSIDE the other canister AND
  the stopper seated in that canister's mouth — plug wedged centred in the funnel
  collar, level, at bearing depth. Success is the settled END STATE: a lid balanced
  askew on the rim, the ball left anywhere else, or the ORIGINAL canister re-capped
  all fail.
- The precision interaction is the stopper seat: a 124 mm plug lowered into a
  58->76 mm funnel collar (+/-14 mm capture, then contact dynamics self-centre the
  wedge). The seed has no insertion at all.
- Which canister starts capped (and holds the ball) flips per episode, and both
  canisters are re-posed with xy jitter + free yaw — a memorised fixed sequence
  targets the wrong canister half the time. The capped canister is identified
  VISUALLY (it is the one wearing the lid); colours are fixed per canister
  (terracotta / slate) so language can refer to either.

So a solver needs a different PLAN (open a container, transfer contents, re-seal
the OTHER container — three ordered sub-goals ending hands-free) and different CODE
STRUCTURE (an unseal/extract/deposit/seat pipeline with a wedge-seat placement
controller, instead of an approach-grasp-hold servo).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - canister x2 (KINEMATIC fixtures; terracotta and slate): base disc, 8 wall
    boxes (octagon, inner inradius 65 mm, sill 48 mm high), 8 outward-tilted
    collar boxes forming a funnel mouth (inner inradius 58 mm at the sill ->
    76 mm at the rim, rim at 68 mm). The interior is deeper than the ball is
    tall: a resting ball sits fully below the aperture.
  - lid (DYNAMIC, 0.15 kg, cream): plug disc (r 62 mm, 14 mm), cap disc
    (r 82 mm, 8 mm), square knob (20 x 20 x 26 mm — a clean parallel-jaw pinch).
    Seated = plug bottom edge wedged on the 8 collar flats where the funnel
    inradius equals the plug radius (analytic seat plane 52.5 mm).
  - ball (DYNAMIC, 60 g, steel-grey sphere r 18 mm — an easy jaw grasp).

Per-episode randomization (readback-verifiable): per-canister xy jitter + free
yaw, Bernoulli role swap (which canister is capped/loaded), ball xy jitter inside
the capped canister.

Rubric (0..1; partial progress latched so credit never evaporates):
  0.15 * opened      — the lid ever clearly off the source mouth (latched)
  0.25 * transferred — the ball ever resting inside the DESTINATION canister
                       (latched)
  1.0 iff success()  — ball inside the destination AND the lid seated in the
                       destination's mouth, everything settled. Non-success cap
                       0.40; null policy scores 0 (the lid starts seated, so no
                       latch fires by itself).

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

FRANKA_JAW_SPAN = 0.080  # Franka parallel-jaw max opening (m)


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _quat_mul_f(q1, q2):
    """(w,x,y,z) float quaternion product."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None) -> None:
    """Author one box collider (optionally rotated: orient = wxyz quaternion)."""
    from pxr import Gf, UsdGeom

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


def _add_cyl(stage, path: str, *, center, radius: float, height: float, color,
             collide: Callable) -> None:
    """Author one z-axis cylinder collider (PhysX convex-hull approximation)."""
    from pxr import Gf, UsdGeom

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


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _spawn_canister(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one canister: ONE KINEMATIC compound. Origin at the base centre on
    the ground. Base disc -> 8 octagon wall boxes -> 8 outward-tilted funnel-collar
    boxes (the stopper seat)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg

    _add_cyl(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_t / 2),
             radius=c.base_r, height=c.base_t, color=c.color, collide=collide)

    sill_z = c.base_t + c.wall_h
    wall_rc = c.wall_in_r + c.wall_t / 2
    wall_len = 2.0 * wall_rc * math.tan(math.pi / 8) + 0.006  # overlap corners
    for k in range(8):
        phi = (k + 0.5) * math.pi / 4
        q = (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(wall_rc * math.cos(phi), wall_rc * math.sin(phi),
                         c.base_t + c.wall_h / 2),
                 size=(c.wall_t, wall_len, c.wall_h), color=c.color,
                 collide=collide, orient=q)

    # funnel collar: 8 plates tilted outward by beta from vertical; inner surface
    # runs (r=cone_r0, z=sill) -> (r=cone_r1, z=sill+cone_h), extended cone_ext
    # below the sill so wall and collar are continuous.
    beta = math.atan2(c.cone_r1 - c.cone_r0, c.cone_h)
    sb, cb = math.sin(beta), math.cos(beta)
    slope_len = math.hypot(c.cone_r1 - c.cone_r0, c.cone_h) + 2 * c.cone_ext
    u_c = slope_len / 2 - c.cone_ext  # box centre along the slope from the sill pt
    r_top = c.cone_r0 + (u_c + slope_len / 2) * sb
    cone_len = 2.0 * r_top * math.tan(math.pi / 8) + 0.010
    for k in range(8):
        phi = (k + 0.5) * math.pi / 4
        r_mid = c.cone_r0 + u_c * sb + (c.cone_t / 2) * cb
        z_mid = sill_z + u_c * cb - (c.cone_t / 2) * sb
        qz = (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))
        qy = (math.cos(beta / 2), 0.0, math.sin(beta / 2), 0.0)
        _add_box(stage, f"{prim_path}/collar_{k}",
                 center=(r_mid * math.cos(phi), r_mid * math.sin(phi), z_mid),
                 size=(c.cone_t, cone_len, slope_len), color=c.collar_color,
                 collide=collide, orient=_quat_mul_f(qz, qy))
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the stopper lid: DYNAMIC — plug disc (origin at its centre, so the
    explicit MassAPI mass keeps the CoM there) + cap disc + square knob."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    c = cfg
    _add_cyl(stage, f"{prim_path}/plug", center=(0.0, 0.0, 0.0),
             radius=c.plug_r, height=c.plug_t, color=c.color, collide=collide)
    _add_cyl(stage, f"{prim_path}/cap",
             center=(0.0, 0.0, c.plug_t / 2 + c.cap_t / 2),
             radius=c.cap_r, height=c.cap_t, color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/knob",
             center=(0.0, 0.0, c.plug_t / 2 + c.cap_t + c.knob_h / 2),
             size=(c.knob_w, c.knob_w, c.knob_h), color=c.knob_color,
             collide=collide)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the infuser ball: DYNAMIC steel-grey sphere."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    r = float(cfg.radius)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg)(sph.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "canister" not in _SPAWNER_CACHE:

        @configclass
        class CanisterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_canister)
            base_r: float = 0.078
            base_t: float = 0.008
            wall_in_r: float = 0.065
            wall_t: float = 0.008
            wall_h: float = 0.040
            cone_r0: float = 0.058
            cone_r1: float = 0.074
            cone_h: float = 0.018
            cone_t: float = 0.008
            cone_ext: float = 0.003
            color: tuple = (0.72, 0.45, 0.30)
            collar_color: tuple = (0.80, 0.55, 0.40)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            plug_r: float = 0.062
            plug_t: float = 0.014
            cap_r: float = 0.082
            cap_t: float = 0.008
            knob_w: float = 0.020
            knob_h: float = 0.026
            color: tuple = (0.92, 0.88, 0.78)
            knob_color: tuple = (0.25, 0.20, 0.16)
            contact_offset: float = 0.002

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.018
            color: tuple = (0.62, 0.64, 0.68)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(canister=CanisterSpawnerCfg, lid=LidSpawnerCfg,
                              ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TeaBallTransferSceneCfg(BaseCfg):
    """Config for `TeaBallTransferScene`. The interlocks are metric: while the lid
    is seated, the ball (top 44 mm) is below the seated plug bottom (52.5 mm) with
    walls all round — physically caged; the funnel collar (58 -> 76 mm inradius)
    gives the 62 mm plug a +/-14 mm self-centring capture; the seated plug clears a
    resting ball's crown by ~8.5 mm, so the loaded canister can be sealed."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(1.0)  # max |ang vel| when judging (rad/s)
    seat_xy_tol: float = tunable(0.012)  # seated lid axis offset from pot axis (m)
    seat_z_tol: float = tunable(0.008)  # seated lid origin height tolerance (m)
    seat_tilt_max_deg: float = tunable(8.0)  # seated lid tilt from level (deg)
    in_xy_tol: float = tunable(0.055)  # ball-in-canister axis offset bound (m):
    # covers any interior floor rest incl. octagon corners (circumradius 70.3 mm
    # - ball_r = 52 mm) yet stays inside the wall shell (inner face at 65 mm)
    in_z_lo: float = tunable(0.012)  # ball-in-canister CoM height band (m)
    in_z_hi: float = tunable(0.044)  # ... below the sill: containment below aperture
    open_away_xy: float = tunable(0.16)  # lid clearly off the source mouth (m)
    open_away_z: float = tunable(0.05)  # ... or lifted this far above its seat (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pot_jitter: float = tunable(0.04)  # per-canister uniform +/- xy jitter (m)
    pot_yaw_deg: float = tunable(180.0)  # per-canister free yaw (+/- deg)
    swap_roles: bool = tunable(True)  # Bernoulli: which canister starts capped
    ball_jitter: float = tunable(0.012)  # ball xy jitter inside the capped pot (m)

    # --- info: layout (one Franka base at the origin, facing +x) ---------------------------------
    pot_a_pos: tuple = info((0.45, 0.19))  # terracotta canister nominal centre
    pot_b_pos: tuple = info((0.45, -0.19))  # slate canister nominal centre

    # --- info: canister structure (local frame: origin at base centre on the ground) -------------
    base_r: float = info(0.078)
    base_t: float = info(0.008)
    wall_in_r: float = info(0.065)  # octagon flat inradius (inner wall face)
    wall_t: float = info(0.008)
    wall_h: float = info(0.040)  # interior sill height above the floor plate
    cone_r0: float = info(0.058)  # funnel inner inradius at the sill
    cone_r1: float = info(0.074)  # ... at nominal rim (boxes extend slightly past)
    cone_h: float = info(0.018)
    cone_t: float = info(0.008)
    cone_ext: float = info(0.003)  # collar-box overshoot beyond each cone end
    pot_a_color: tuple = info((0.72, 0.45, 0.30))  # terracotta
    pot_b_color: tuple = info((0.38, 0.45, 0.58))  # slate

    # --- info: lid ---------------------------------------------------------------------------------
    plug_r: float = info(0.062)
    plug_t: float = info(0.014)
    cap_r: float = info(0.082)
    cap_t: float = info(0.008)
    knob_w: float = info(0.020)  # < jaw span: a clean pinch (asserted)
    knob_h: float = info(0.026)
    lid_mass: float = info(0.15)

    # --- info: ball --------------------------------------------------------------------------------
    ball_r: float = info(0.018)  # 36 mm dia < jaw span: an easy grasp (asserted)
    ball_mass: float = info(0.06)

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 = 0.40 = the non-success cap)
    w_open: float = info(0.15)
    w_transfer: float = info(0.25)

    def __post_init__(self) -> None:
        assert 2 * self.ball_r < FRANKA_JAW_SPAN - 0.02, \
            "ball must be comfortably jaw-graspable"
        assert self.knob_w < FRANKA_JAW_SPAN - 0.03, \
            "knob must be a comfortable parallel-jaw pinch"
        assert self.cone_r0 < self.plug_r < self.cone_r1, \
            "plug must wedge inside the funnel collar (the stopper seat)"
        assert self.cone_r0 > self.ball_r + 0.010, \
            "ball must pass the narrowest aperture with clearance"
        # derived geometry (same formulas as the spawner)
        self.sill_z: float = self.base_t + self.wall_h  # 0.048
        beta = math.atan2(self.cone_r1 - self.cone_r0, self.cone_h)
        slope_len = math.hypot(self.cone_r1 - self.cone_r0, self.cone_h) \
            + 2 * self.cone_ext
        self.rim_z: float = self.sill_z + (slope_len - self.cone_ext) \
            * math.cos(beta)  # collar top height ~0.068
        self.mouth_r_top: float = self.cone_r0 + (slope_len - self.cone_ext) \
            * math.sin(beta)  # top opening inradius ~0.076
        # plug bottom-edge circle wedges where the funnel inradius = plug_r
        self.seat_plane_z: float = self.sill_z + self.cone_h \
            * (self.plug_r - self.cone_r0) / (self.cone_r1 - self.cone_r0)  # 0.0525
        self.lid_seat_z: float = self.seat_plane_z + self.plug_t / 2  # 0.0595
        self.ball_rest_z: float = self.base_t + self.ball_r  # 0.026
        assert self.seat_plane_z > self.base_t + 2 * self.ball_r + 0.006, \
            "seated plug must clear a resting ball's crown"
        assert self.mouth_r_top - self.plug_r > 0.010, \
            "funnel capture range must exceed 10 mm"
        assert self.in_z_hi < self.sill_z, "containment band must sit below the sill"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tea_ball_transfer")
class TeaBallTransferScene(BaseScene):
    cfg: TeaBallTransferSceneCfg

    def __init__(self, cfg: TeaBallTransferSceneCfg | None = None) -> None:
        super().__init__(cfg or TeaBallTransferSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()

        def pot_spawn(color, collar):
            return spawners["canister"](
                mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                base_r=c.base_r, base_t=c.base_t, wall_in_r=c.wall_in_r,
                wall_t=c.wall_t, wall_h=c.wall_h, cone_r0=c.cone_r0,
                cone_r1=c.cone_r1, cone_h=c.cone_h, cone_t=c.cone_t,
                cone_ext=c.cone_ext, color=color, collar_color=collar,
                contact_offset=c.contact_offset,
            )

        lid_spawn = spawners["lid"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            plug_r=c.plug_r, plug_t=c.plug_t, cap_r=c.cap_r, cap_t=c.cap_t,
            knob_w=c.knob_w, knob_h=c.knob_h, contact_offset=c.contact_offset,
        )
        ball_spawn = spawners["ball"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.ball_r, contact_offset=c.contact_offset,
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
            "pot_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PotA",
                spawn=pot_spawn(c.pot_a_color, (0.80, 0.55, 0.40)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_a_pos[0], c.pot_a_pos[1], 0.0)),
            ),
            "pot_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PotB",
                spawn=pot_spawn(c.pot_b_color, (0.48, 0.55, 0.68)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_b_pos[0], c.pot_b_pos[1], 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_a_pos[0], c.pot_a_pos[1], 0.0595 + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=ball_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_a_pos[0], c.pot_a_pos[1], 0.026 + 0.002)),
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
        self.pot_a: RigidObject = env.iscene["pot_a"]
        self.pot_b: RigidObject = env.iscene["pot_b"]
        self.lid: RigidObject = env.iscene["lid"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # src_is_a[e]: pot_a is the capped/loaded canister in episode e
        self.src_is_a = torch.ones(n, dtype=torch.bool, device=dev)
        # latches: partial progress survives transient achievements
        self._opened_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._transferred_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: both canisters re-posed (xy jitter + free yaw), roles
        Bernoulli-swapped, ball placed inside the source with xy jitter, lid written
        at its analytic seat on the source (it settles seated in a few steps),
        latches cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- canisters: nominal slot + jitter + free yaw (kinematic writes) ---
        poses = []
        for nominal, body in ((c.pot_a_pos, self.pot_a), (c.pot_b_pos, self.pot_b)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = nominal[0]
            st[:, 1] = nominal[1]
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pot_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pot_yaw_deg)
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)
            poses.append((st[:, 0:3].clone(), st[:, 3:7].clone()))

        # --- roles: Bernoulli via torch.rand comparison ---
        if c.swap_roles:
            self.src_is_a[env_ids] = torch.rand(m, device=dev) < 0.5
        else:
            self.src_is_a[env_ids] = True
        src = self.src_is_a[env_ids].unsqueeze(1)
        s_pos = torch.where(src, poses[0][0], poses[1][0])
        s_quat = torch.where(src, poses[0][1], poses[1][1])

        # --- ball: inside the source, xy jitter, resting height ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
        loc[:, 2] = c.ball_rest_z + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        # --- lid: at its analytic seat on the source (settles seated) ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 2] = c.lid_seat_z + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        st[:, 3:7] = s_quat
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._opened_ever[env_ids] = False
        self._transferred_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pot_a": self.pot_a.data.root_state_w[env_ids].clone(),
            "pot_b": self.pot_b.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "src_is_a": self.src_is_a[env_ids].clone(),
            "opened_ever": self._opened_ever[env_ids].clone(),
            "transferred_ever": self._transferred_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pot_a.write_root_state_to_sim(state["pot_a"], env_ids)
        self.pot_b.write_root_state_to_sim(state["pot_b"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.src_is_a[env_ids] = state["src_is_a"]
        self._opened_ever[env_ids] = state["opened_ever"]
        self._transferred_ever[env_ids] = state["transferred_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two squat octagonal ceramic tea canisters stand on the floor, about "
            f"{2 * c.base_r * 100:.0f} cm wide and {c.rim_z * 100:.1f} cm tall: one "
            f"TERRACOTTA (orange-brown), one SLATE (blue-grey). Each has an open "
            f"funnel-shaped mouth. ONE of them — it varies between episodes, spot "
            f"it visually — is CAPPED by a loose cream stopper lid (a wide cap "
            f"disc with a small dark square knob on top, {c.knob_w * 1000:.0f} mm "
            f"across — pinch the knob with the jaws to lift it) and holds a steel "
            f"infuser BALL ({2 * c.ball_r * 1000:.0f} mm) inside; the other "
            f"canister stands OPEN and empty.\n"
            f"Goal: the steel ball must end up resting INSIDE the open (initially "
            f"empty) canister, AND the stopper lid must end up SEATED in that same "
            f"canister's mouth — plug dropped into the funnel so it self-centres "
            f"and sits level at its bearing depth — with everything at rest and "
            f"nothing held. Recommended order (the geometry forces the first "
            f"step): lift the lid off the capped canister by its knob and set it "
            f"aside; take the ball out of the now-open canister and drop it into "
            f"the other one; then place the lid into THAT canister's funnel mouth "
            f"and let it seat. The funnel gives about +/-{(c.mouth_r_top - c.plug_r) * 1000:.0f} mm "
            f"of centring capture — release the lid roughly centred and level and "
            f"it wedges itself in. A lid balanced askew on the rim (tilted or "
            f"proud of its bearing depth), the ball left in the original canister "
            f"or anywhere else, or the ORIGINAL canister re-capped instead — all "
            f"failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the stopper lid off the capped tea canister, move the steel "
            "infuser ball from inside it into the other, open canister, then seat "
            "the lid level in the open canister's funnel mouth. The ball must end "
            "up inside the newly capped canister; re-capping the original "
            "canister or leaving the lid askew on the rim fails."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _pot_state(self, source: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,3), quat (N,4)) of the source (or destination) canister."""
        pick_a = self.src_is_a if source else ~self.src_is_a
        pos = torch.where(pick_a.unsqueeze(1), self.pot_a.data.root_pos_w,
                          self.pot_b.data.root_pos_w)
        quat = torch.where(pick_a.unsqueeze(1), self.pot_a.data.root_quat_w,
                           self.pot_b.data.root_quat_w)
        return pos, quat

    def _local(self, body: RigidObject, source: bool) -> torch.Tensor:
        """(N, 3) body origin in the source/destination canister frame."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, quat = self._pot_state(source)
        return quat_apply_inverse(quat, body.data.root_pos_w - pos)

    def _lid_level(self) -> torch.Tensor:
        """(N,) bool: lid z-axis within seat_tilt_max_deg of world-up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.lid.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(
            math.radians(self.cfg.seat_tilt_max_deg))

    def _lid_seated_on(self, source: bool, require_still: bool = True) -> torch.Tensor:
        """(N,) bool: lid wedged at its bearing seat in the given canister's mouth —
        axis centred, at seat depth, level (and still unless probing under load)."""
        c = self.cfg
        loc = self._local(self.lid, source)
        ok = ((loc[:, :2].norm(dim=-1) < c.seat_xy_tol)
              & ((loc[:, 2] - c.lid_seat_z).abs() < c.seat_z_tol)
              & self._lid_level())
        if not require_still:
            return ok
        still = self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return ok & still

    def _ball_in(self, source: bool) -> torch.Tensor:
        """(N,) bool: ball CoM inside the given canister's interior — near the
        axis, in the resting band BELOW the sill (containment below the aperture:
        a slow transit through the mouth or a rest on the collar never counts)."""
        c = self.cfg
        loc = self._local(self.ball, source)
        return ((loc[:, :2].norm(dim=-1) < c.in_xy_tol)
                & (loc[:, 2] > c.in_z_lo) & (loc[:, 2] < c.in_z_hi))

    def _lid_open(self) -> torch.Tensor:
        """(N,) bool: lid CLEARLY off the source mouth — far to the side, or
        lifted well above its seat (robust: transient settle wobble never fires)."""
        c = self.cfg
        loc = self._local(self.lid, True)
        return ((loc[:, :2].norm(dim=-1) > c.open_away_xy)
                | (loc[:, 2] > c.lid_seat_z + c.open_away_z))

    def _ball_still(self) -> torch.Tensor:
        c = self.cfg
        return ((self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.ball.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _update_latches(self) -> None:
        self._opened_ever |= self._lid_open()
        slow = self.ball.data.root_lin_vel_w.norm(dim=-1) < 0.3
        self._transferred_ever |= self._ball_in(False) & slow

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the canisters are kinematic and jointless) — just latch rubric
        progress every substep so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: ball resting inside the DESTINATION canister AND the lid
        seated in the destination's mouth, everything settled. Physical end state
        only — there is one lid, so sealing the destination necessarily left the
        source open, and the ball can only have entered through real contact
        transport (it starts physically caged under the seated lid)."""
        self._update_latches()
        return self._ball_in(False) & self._ball_still() \
            & self._lid_seated_on(False, require_still=True)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*opened + 0.25*transferred — both latched, 0
        for the null policy (the lid starts seated; nothing fires by itself),
        non-success cap 0.40 — and exactly 1.0 iff success() holds."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._opened_ever.float()
                + c.w_transfer * self._transferred_ever.float()).clamp(max=0.40)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="tea_ball_transfer", robot="null"))
