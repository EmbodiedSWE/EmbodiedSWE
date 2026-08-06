"""RollInGarageScene — lift the blocking post out of the doorway, then roll the ketchup
bottle through the garage's only opening and over the internal ramp crest.

Derived from libero_90/living_room_scene1 "pick up the ketchup and put it in the basket"
(grasp the standing ketchup bottle, carry it OVER the open-topped basket, release,
bbox containment check), but the receptacle geometry INVERTS the seed's plan: the
container is a ROOFED SIDE-ENTRY GARAGE whose only opening is a floor-level doorway.

- The roof covers the whole interior, so the seed's move — carry the bottle above the
  receptacle and let go — leaves it resting ON the roof and scores nothing (smoke #6).
- The doorway is BLOCKED by an orange post (taller than the roof, jaw-sized cross
  section). In every randomized layout the side gaps next to the post are narrower than
  the bottle in ANY orientation (max gap ~50 mm < 55 mm bottle diameter), so the post
  must be moved out of the doorway before the bottle can pass (smoke #9 constructs the
  blocked end state).
- Just inside the doorway a 12 deg RAMP rises to a ~13 mm CREST with an overhanging
  drop-off face: the bottle must be pushed/rolled along the floor, through the doorway,
  UP the ramp and over the crest; it then drops into the landing bay, where the crest's
  back face retains it (rolling back out would mean climbing an overhang). A bottle
  left anywhere before the crest — in the throat, on the ramp slope — is a rejected
  near miss (smoke #7, #8).
- The target lies ON ITS SIDE (it is a roller, not a stander) and a YELLOW mustard
  bottle of the same shape must stay out (smoke #10, #11).

So a solver needs a different plan (declutter the aperture, then a ground-level
push-roll insertion through a side doorway with a climb-and-drop commitment point) and
different code structure (obstacle relocation + floor-plane rolling control), not
different parameters on grasp-carry-drop.

Assets are fully procedural (pen_holder-pattern compound spawners; child colliders of
one body never self-collide):
  - garage: KINEMATIC compound — two side walls, back wall, full roof (underside
    120 mm), two 30 deg funnel wings flanking the doorway, and the internal ramp (a
    pitched box; crest ~12.6 mm at 75 mm inside the doorway plane). Origin at the
    DOORWAY CENTRE on the floor; local +x points INTO the garage (axis "u").
  - ketchup / mustard: DYNAMIC compounds — body cylinder (55 mm dia) + slightly
    thinner cap cylinder (visual identity; only the body ever touches the floor, so
    the bottle rolls straight). Angular damping 0.35 so the roller settles.
  - post: DYNAMIC orange box 55 x 130 x 150 mm standing in the doorway throat.

Per-episode randomization (readback-verifiable): garage lateral offset + yaw, Bernoulli
LEFT/RIGHT slot swap of the two bottles + per-bottle xy jitter + per-bottle axis yaw
jitter, post lateral jitter in the doorway.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * cleared     — post ever observed OUT of the doorway-blocking zone (latched)
  0.15 * approach    — ketchup approach to the doorway centre, gated on cleared
                       (latched running max; ~0 for doing nothing)
  0.25 * entered     — ketchup CoM ever inside the garage past the doorway plane
                       (throat/ramp, under the roof) (latched)
  0.35 * landed      — ketchup CoM ever in the landing bay past the crest (latched)
  1.0 iff success()  — ketchup in the landing bay, resting on the floor, at rest,
                       mustard NOT inside the garage. Non-success cap 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             quat=None) -> None:
    """Author one box collider. `quat` (w, x, y, z) is an optional local orientation
    applied between translate and scale (T * R * S) — used for the funnel wings (yaw)
    and the internal ramp (pitch)."""
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


def _spawn_garage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the garage: KINEMATIC compound. Origin at the doorway centre on the
    floor; interior spans u in [0, in_d], |v| <= in_w/2, roof underside at roof_z.
    Two 30 deg funnel wings flank the doorway; the pitched ramp box sits inside with
    its crest crest_u into the garage."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    wall_h = c.roof_z + c.roof_t  # walls reach the roof's top face
    u_len = c.in_d + c.t  # walls run from the doorway plane to the back wall's outside
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(u_len / 2, sgn * (c.in_w / 2 + c.t / 2), wall_h / 2),
                 size=(u_len, c.t, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(c.in_d + c.t / 2, 0.0, wall_h / 2),
             size=(c.t, c.in_w + 2 * c.t, wall_h), color=c.color, collide=collide)
    # roof: covers the whole interior plus a 5 mm doorway lip
    roof_u0, roof_u1 = -0.005, c.in_d + c.t
    _add_box(stage, f"{prim_path}/roof",
             center=((roof_u0 + roof_u1) / 2, 0.0, c.roof_z + c.roof_t / 2),
             size=(roof_u1 - roof_u0, c.in_w + 2 * c.t, c.roof_t),
             color=c.roof_color, collide=collide)
    # funnel wings: rooted at the doorway corners, flaring outward by wing_ang
    a = math.radians(c.wing_ang_deg)
    for sgn, nm in ((1.0, "wing_l"), (-1.0, "wing_r")):
        cx = -(c.wing_len / 2) * math.cos(a)
        cy = sgn * (c.in_w / 2 + c.t / 2 + (c.wing_len / 2) * math.sin(a))
        half = -sgn * a / 2  # yaw about z by -sgn*a
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(cx, cy, wall_h / 2),
                 size=(c.wing_len, c.t, wall_h), color=c.color, collide=collide,
                 quat=(math.cos(half), 0.0, 0.0, math.sin(half)))
    # internal ramp: pitched box, +u end lifted by ramp_pitch (rotation about +y by
    # -pitch maps +x -> (cos, 0, +sin))
    half = -math.radians(c.ramp_pitch_deg) / 2
    _add_box(stage, f"{prim_path}/ramp",
             center=(c.ramp_center_u, 0.0, c.ramp_center_z),
             size=(c.ramp_len, c.in_w - 0.004, c.ramp_t), color=c.ramp_color,
             collide=collide, quat=(math.cos(half), 0.0, math.sin(half), 0.0))
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a squeeze bottle: DYNAMIC body cylinder + slightly thinner cap cylinder
    along local +z. Origin at the body cylinder's centre. The cap never touches the
    floor when the bottle lies on its side, so it rolls straight on the body."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # Lying bottles are rollers: heavy angular damping so they settle instead of
    # circling the landing bay for many seconds. Sleep thresholds zeroed — they are
    # force-driven and judged for stillness.
    pxrb.CreateLinearDampingAttr(0.06)
    pxrb.CreateAngularDampingAttr(0.35)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    for nm, r, h, z0, col in (
            ("body", c.body_r, c.body_h, 0.0, c.color),
            ("cap", c.cap_r, c.cap_h, c.body_h / 2 + c.cap_h / 2, c.cap_color)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0))
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*col)])
        collide(cyl.GetPrim())
    return root


def _spawn_post(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the doorway post: DYNAMIC standing box. Origin at its base centre on the
    floor."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/post",
             center=(0.0, 0.0, c.height / 2),
             size=(c.depth, c.width, c.height), color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the three compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "garage" not in _SPAWNER_CACHE:

        @configclass
        class GarageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_garage)
            in_w: float = 0.20
            in_d: float = 0.24
            roof_z: float = 0.12
            roof_t: float = 0.012
            t: float = 0.012
            wing_len: float = 0.11
            wing_ang_deg: float = 30.0
            ramp_len: float = 0.070
            ramp_t: float = 0.010
            ramp_pitch_deg: float = 12.0
            ramp_center_u: float = 0.0418
            ramp_center_z: float = 0.0004
            color: tuple = (0.30, 0.38, 0.55)
            roof_color: tuple = (0.22, 0.28, 0.42)
            ramp_color: tuple = (0.55, 0.58, 0.62)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.0275
            body_h: float = 0.105
            cap_r: float = 0.0255
            cap_h: float = 0.028
            color: tuple = (0.5, 0.5, 0.5)
            cap_color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.002

        @configclass
        class PostSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_post)
            depth: float = 0.055
            width: float = 0.13
            height: float = 0.15
            color: tuple = (0.90, 0.45, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(garage=GarageSpawnerCfg, bottle=BottleSpawnerCfg,
                              post=PostSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RollInGarageSceneCfg(BaseCfg):
    """Config for `RollInGarageScene`. The interlocks are metric: the roof covers the
    whole interior (top-down insertion is impossible everywhere past the doorway
    plane), and the post's doorway side gaps (max (in_w - post_w)/2 + post_v_jit =
    50 mm) are narrower than the bottle body (55 mm), so the doorway is impassable in
    every randomized layout until the post is moved."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.04)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    approach_d0: float = tunable(0.40)  # approach ramp: p = 1 - d/approach_d0
    land_u_margin: float = tunable(0.015)  # landing bay starts crest_u + this

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    garage_dy_max: float = tunable(0.05)  # garage lateral offset (+/- m)
    garage_yaw_max_deg: float = tunable(12.0)  # garage yaw (+/- deg)
    slot_jitter: float = tunable(0.03)  # per-bottle spawn xy jitter (+/- m)
    axis_yaw_jitter_deg: float = tunable(20.0)  # per-bottle lying-axis yaw jitter (+/-)
    swap_slots: bool = tunable(True)  # Bernoulli ketchup/mustard slot swap
    post_v_jit: float = tunable(0.015)  # post lateral jitter in the doorway (+/- m)

    # --- info: layout (single Franka base at the origin; radii 0.30-0.60 m) ---------------------
    garage_x: float = info(0.52)  # doorway plane distance from the base
    slot_a: tuple = info((0.32, 0.20))  # bottle spawn slot A (left)
    slot_b: tuple = info((0.32, -0.20))  # bottle spawn slot B (right)
    post_u: float = info(-0.050)  # post base centre, garage-local u (in the throat)
    park_spot: tuple = info((0.20, 0.40))  # a free patch (describe() suggests it)

    # --- info: garage structure ------------------------------------------------------------------
    in_w: float = info(0.20)  # interior width (v)
    in_d: float = info(0.24)  # interior depth (u), doorway plane to back wall
    roof_z: float = info(0.12)  # roof underside height
    roof_t: float = info(0.012)
    t: float = info(0.012)  # wall thickness
    wing_len: float = info(0.11)
    wing_ang_deg: float = info(30.0)
    ramp_len: float = info(0.070)
    ramp_t: float = info(0.010)
    ramp_pitch_deg: float = info(12.0)
    crest_u: float = info(0.075)  # crest (top trailing edge) distance past the doorway
    garage_color: tuple = info((0.30, 0.38, 0.55))  # slate blue
    roof_color: tuple = info((0.22, 0.28, 0.42))
    ramp_color: tuple = info((0.55, 0.58, 0.62))  # light gray

    # --- info: bottles ---------------------------------------------------------------------------
    body_r: float = info(0.0275)  # 55 mm body dia
    body_h: float = info(0.105)
    cap_r: float = info(0.0255)  # thinner cap: only the body touches the floor
    cap_h: float = info(0.028)
    ketchup_mass: float = info(0.30)
    mustard_mass: float = info(0.28)
    ketchup_color: tuple = info((0.72, 0.07, 0.05))  # red
    mustard_color: tuple = info((0.83, 0.68, 0.08))  # yellow
    cap_color: tuple = info((0.95, 0.95, 0.92))  # white

    # --- info: post ------------------------------------------------------------------------------
    post_depth: float = info(0.055)  # u extent: the parallel-jaw grasp span
    post_width: float = info(0.13)  # v extent: leaves <= 50 mm side gaps
    post_height: float = info(0.15)  # taller than the roof top (132 mm)
    post_mass: float = info(0.45)
    post_color: tuple = info((0.90, 0.45, 0.10))  # orange

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.15 + 0.25 + 0.35 = 0.85 = the non-success cap)
    w_clear: float = info(0.10)
    w_app: float = info(0.15)
    w_enter: float = info(0.25)
    w_land: float = info(0.35)

    # Derived (filled in __post_init__).
    crest_h: float = field(default=None, init=False)  # crest top height
    ramp_center_u: float = field(default=None, init=False)
    ramp_center_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        th = math.radians(self.ramp_pitch_deg)
        s, co = math.sin(th), math.cos(th)
        # place the pitched ramp box so its top leading edge sits 2 mm below the floor
        # (no entry lip) and its top trailing edge (the crest) lands at crest_u
        self.ramp_center_z = -0.002 + (self.ramp_len / 2) * s - (self.ramp_t / 2) * co
        self.ramp_center_u = self.crest_u - (self.ramp_len / 2) * co + (self.ramp_t / 2) * s
        self.crest_h = self.ramp_center_z + (self.ramp_len / 2) * s + (self.ramp_t / 2) * co


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("roll_in_garage")
class RollInGarageScene(BaseScene):
    cfg: RollInGarageSceneCfg

    def __init__(self, cfg: RollInGarageSceneCfg | None = None) -> None:
        super().__init__(cfg or RollInGarageSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        garage_spawn = spawners["garage"](
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_w=c.in_w, in_d=c.in_d, roof_z=c.roof_z, roof_t=c.roof_t, t=c.t,
            wing_len=c.wing_len, wing_ang_deg=c.wing_ang_deg, ramp_len=c.ramp_len,
            ramp_t=c.ramp_t, ramp_pitch_deg=c.ramp_pitch_deg,
            ramp_center_u=c.ramp_center_u, ramp_center_z=c.ramp_center_z,
            color=c.garage_color, roof_color=c.roof_color, ramp_color=c.ramp_color,
            contact_offset=c.contact_offset,
        )

        def bottle_spawn(mass, color):
            return spawners["bottle"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                body_r=c.body_r, body_h=c.body_h, cap_r=c.cap_r, cap_h=c.cap_h,
                color=color, cap_color=c.cap_color, contact_offset=c.contact_offset,
            )

        post_spawn = spawners["post"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.post_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            depth=c.post_depth, width=c.post_width, height=c.post_height,
            color=c.post_color, contact_offset=c.contact_offset,
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
            "garage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Garage",
                spawn=garage_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.garage_x, 0.0, 0.0)),
            ),
            "ketchup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ketchup",
                spawn=bottle_spawn(c.ketchup_mass, c.ketchup_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.body_r + 0.002)),
            ),
            "mustard": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mustard",
                spawn=bottle_spawn(c.mustard_mass, c.mustard_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.body_r + 0.002)),
            ),
            "post": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post",
                spawn=post_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.garage_x + c.post_u, 0.0, 0.002)),
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
        self.garage: RigidObject = env.iscene["garage"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.mustard: RigidObject = env.iscene["mustard"]
        self.post: RigidObject = env.iscene["post"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._cleared = torch.zeros(n, dtype=torch.bool, device=dev)  # post ever out
        self._app_max = torch.zeros(n, device=dev)  # doorway approach, running max
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)  # ever past doorway
        self._landed = torch.zeros(n, dtype=torch.bool, device=dev)  # ever past crest

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: garage re-posed with a random lateral offset + yaw, post
        standing in the doorway throat (lateral jitter), bottles randomly ASSIGNED to
        the two spawn slots (+ xy and axis-yaw jitter) lying on their sides, latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- garage (kinematic): lateral offset + yaw ---
        dy = (torch.rand(m, device=dev) * 2 - 1) * c.garage_dy_max
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.garage_yaw_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = c.garage_x, dy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.garage.write_root_state_to_sim(st, env_ids)
        g_pos, g_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- post: standing in the doorway throat, expressed in the garage frame ---
        from isaaclab.utils.math import quat_apply

        vj = (torch.rand(m, device=dev) * 2 - 1) * c.post_v_jit
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0], loc[:, 1], loc[:, 2] = c.post_u, vj, 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = g_pos + quat_apply(g_quat, loc)
        st[:, 3:7] = g_quat
        self.post.write_root_state_to_sim(st, env_ids)

        # --- bottles: Bernoulli slot swap + jitter, lying on their sides ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        k_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        m_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.ketchup, k_xy), (self.mustard, m_xy)):
            # lying pose: body +z -> world +y (qx(-90 deg)), then a random yaw
            psi = (torch.rand(m, device=dev) * 2 - 1) \
                * math.radians(c.axis_yaw_jitter_deg)
            hx = torch.tensor(-math.pi / 4, device=dev)
            qx = torch.stack([torch.cos(hx).expand(m), torch.sin(hx).expand(m),
                              torch.zeros(m, device=dev), torch.zeros(m, device=dev)],
                             dim=1)
            qz = torch.stack([torch.cos(psi / 2), torch.zeros(m, device=dev),
                              torch.zeros(m, device=dev), torch.sin(psi / 2)], dim=1)
            from isaaclab.utils.math import quat_mul

            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.body_r + 0.002
            st[:, 3:7] = quat_mul(qz, qx)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._cleared[env_ids] = False
        self._app_max[env_ids] = 0.0
        self._entered[env_ids] = False
        self._landed[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "garage": self.garage.data.root_state_w[env_ids].clone(),
            "ketchup": self.ketchup.data.root_state_w[env_ids].clone(),
            "mustard": self.mustard.data.root_state_w[env_ids].clone(),
            "post": self.post.data.root_state_w[env_ids].clone(),
            "cleared": self._cleared[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
            "landed": self._landed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.garage.write_root_state_to_sim(state["garage"], env_ids)
        self.ketchup.write_root_state_to_sim(state["ketchup"], env_ids)
        self.mustard.write_root_state_to_sim(state["mustard"], env_ids)
        self.post.write_root_state_to_sim(state["post"], env_ids)
        self._cleared[env_ids] = state["cleared"]
        self._app_max[env_ids] = state["app_max"]
        self._entered[env_ids] = state["entered"]
        self._landed[env_ids] = state["landed"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a slate-blue GARAGE: a covered box (interior "
            f"{c.in_w * 100:.0f} cm wide x {c.in_d * 100:.0f} cm deep, roof underside "
            f"{c.roof_z * 100:.0f} cm) whose ONLY opening is a floor-level DOORWAY on "
            f"the side facing the robot, flanked by two flared guide wings. The roof "
            f"covers the entire interior, so nothing can be dropped in from above — "
            f"objects released over the garage just rest on the roof. Just inside the "
            f"doorway a light-gray RAMP rises at {c.ramp_pitch_deg:.0f} deg to a "
            f"{c.crest_h * 1000:.0f} mm CREST {c.crest_u * 100:.1f} cm past the doorway "
            f"plane, then drops off into the landing bay behind it; the drop-off face "
            f"overhangs, so whatever rolls in past the crest stays in. An ORANGE POST "
            f"({c.post_depth * 100:.1f} x {c.post_width * 100:.0f} cm footprint, "
            f"{c.post_height * 100:.0f} cm tall — taller than the roof) stands in the "
            f"doorway throat; the gaps beside it are narrower than the bottles, so the "
            f"doorway is impassable until the post is moved. In front of the garage "
            f"lie TWO squeeze bottles ON THEIR SIDES (positions swap between episodes "
            f"— identify by COLOR): a RED ketchup bottle and a YELLOW mustard bottle, "
            f"both {2 * c.body_r * 100:.1f} cm dia x ~{(c.body_h + c.cap_h) * 100:.1f} "
            f"cm long with white caps.\n"
            f"Goal: the RED ketchup bottle must end up INSIDE the garage's landing bay "
            f"— through the doorway and PAST the ramp crest — resting on the floor at "
            f"rest; the YELLOW mustard bottle must remain OUTSIDE the garage. First "
            f"move the orange post anywhere that leaves the doorway clear (e.g. the "
            f"open floor to the side), then push the red bottle along the floor so it "
            f"rolls through the doorway, up the ramp and over the crest; the guide "
            f"wings funnel it. Placing the bottle on the roof, leaving it in the "
            f"doorway throat or on the ramp slope short of the crest, garaging the "
            f"yellow bottle (alone or additionally), or leaving the doorway blocked "
            f"with the red bottle outside — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the orange post out of the garage doorway, then push the red "
            "ketchup bottle through the doorway so it rolls up over the internal ramp "
            "crest and rests inside the covered garage. Keep the yellow mustard "
            "bottle outside."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _garage_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the garage frame (u into the garage, v across
        the doorway, z up)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.garage.data.root_pos_w
        return quat_apply_inverse(self.garage.data.root_quat_w, rel)

    def _post_blocking(self) -> torch.Tensor:
        """(N,) bool: post CoM inside the doorway-blocking zone (throat + doorway
        plane), standing or toppled."""
        loc = self._garage_local(self.post)
        return ((loc[:, 0] >= -0.11) & (loc[:, 0] <= 0.03)
                & (loc[:, 1].abs() <= 0.135) & (loc[:, 2] <= 0.20))

    def _in_garage(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM past the doorway plane, between the walls, under the
        roof (throat/ramp OR landing bay)."""
        c = self.cfg
        loc = self._garage_local(body)
        return ((loc[:, 0] >= 0.005) & (loc[:, 0] <= c.in_d - 0.002)
                & (loc[:, 1].abs() <= c.in_w / 2 - 0.003) & (loc[:, 2] <= c.roof_z - 0.01))

    def _in_landing_bay(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM past the crest, resting height, inside the bay."""
        c = self.cfg
        loc = self._garage_local(body)
        return ((loc[:, 0] >= c.crest_u + c.land_u_margin)
                & (loc[:, 0] <= c.in_d - 0.005)
                & (loc[:, 1].abs() <= c.in_w / 2 - 0.005)
                & (loc[:, 2] > 0.005) & (loc[:, 2] <= 0.08))

    def _update_latches(self) -> None:
        c = self.cfg
        self._cleared |= ~self._post_blocking()
        # doorway approach, gated on the doorway being cleared first
        mouth = self.garage.data.root_pos_w.clone()
        mouth[:, 2] += c.body_r
        d = (self.ketchup.data.root_pos_w - mouth).norm(dim=-1)
        app = (1.0 - d / c.approach_d0).clamp(0.0, 1.0) * self._cleared.float()
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        self._entered |= self._in_garage(self.ketchup)
        self._landed |= self._in_landing_bay(self.ketchup)

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant here (the garage is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: ketchup in the landing bay (past the crest, resting on the
        floor), at rest, mustard NOT inside the garage. Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        k_still = ((self.ketchup.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                   & (self.ketchup.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        return (self._in_landing_bay(self.ketchup) & k_still
                & ~self._in_garage(self.mustard))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*cleared + 0.15*doorway-approach (gated on
        cleared) + 0.25*entered + 0.35*landed — all latched, ~0 for doing nothing,
        capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_clear * self._cleared.float() + c.w_app * self._app_max
                + c.w_enter * self._entered.float()
                + c.w_land * self._landed.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="roll_in_garage", robot="null"))
