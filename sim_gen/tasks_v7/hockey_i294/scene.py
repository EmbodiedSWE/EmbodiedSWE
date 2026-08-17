"""CageCaptureScene — capture the white ball WHERE IT LIES: slide the heavy roofed
cage across the floor, mouth-first, until it swallows the ball, then bar the mouth
with the yellow chock. The ball itself must never be displaced. Derived from
rlbench/hockey with the seed's plan INVERTED.

Seed (rlbench/hockey): grasp a hockey stick and STRIKE the ball across open floor
into a fixed, open-mouthed goal — the ball is the thing you propel, the goal just
waits. Here every element of that plan is inverted:

- THE BALL MUST NOT MOVE. Success requires the white ball to sit within
  `anchor_tol` (4.5 cm) of the exact spot where it settled at reset, and every
  partial-credit latch is gated on that same anchor. Propelling the ball anywhere
  — the seed's entire strategy — forfeits the task (the smoke battery fires the
  ball into the cage mouth and watches the rubric reject the outcome).
- THE GOAL IS THE THING YOU MOVE. The "goal" is a free, floorless, roofed CAGE
  (three walls + roof, open front mouth, open bottom) standing elsewhere on the
  floor. At 3.2 kg it exceeds the arm's payload — it cannot be lifted, only SLID
  on the floor. The only way the ball can end up inside is for the cage to travel
  mouth-first over it: the roof means nothing can be dropped in from above, the
  walls mean nothing enters from the sides or back, and the anchor clause means
  the ball cannot be brought to the cage.
- THE FINAL ACT IS A PLACEMENT, NOT A STRIKE: once the ball is deep inside, a
  yellow CHOCK BAR must be laid flat on the floor across the mouth (centered,
  aligned with the mouth) to enclose the ball. A black decoy ball of the same
  size must remain OUTSIDE the cage.

A solver therefore needs a different PLAN (steer a heavy receptacle around a
stationary target without touching it, then bar the door) and different CODE
STRUCTURE (cage-frame servo on the BALL's relative position + a precision stop +
a place-from-above, instead of grasp-tool + swing).

Assets are fully procedural (compound spawners; no external files):
  - cage: DYNAMIC compound — side walls, back wall, roof; NO floor, NO front
    wall. Origin at the interior floor centre, mouth plane at local +x = hx.
    Mass 3.2 kg with authored CoM + diagonal inertia (custom spawners ignore cfg
    mass_props) and a slick authored material so the slide force stays in the
    arm's comfortable range.
  - chock: DYNAMIC yellow square bar (40 x 40 x 210 mm) — jaw-sized, stable flat.
  - ball / decoy: DYNAMIC spheres (60 mm; white / black), solver velocity
    iterations 4 (GPU sphere-creep fix — the anchor clause needs a truly still
    ball), restitution 0.

Per-episode randomization (readback-verifiable): white ball xy in a band, decoy
on a Bernoulli LEFT/RIGHT flank of the ball, cage start pose (xy + yaw roughly
facing the ball), chock parked on the flank opposite the decoy.

Rubric (0..1; latched so transient achievements keep credit; every latch is
gated on the ball anchor — credit is only ever earned while the ball sits where
it lay):
  0.15 * approach — mouth centre ever within `approach_dist` of the anchored ball
  0.20 * partial  — anchored ball ever past the mouth plane (cage-frame readback)
  0.25 * capture  — anchored ball ever fully inside (past the capture margin)
  0.10 * chocked  — chock ever seated across the mouth while the ball is captured
  1.0 iff success() — ball fully inside AND at its anchor, chock seated across
      the mouth, decoy outside, cage upright and seated on the floor, everything
      at rest. Non-success cap 0.70.

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> Any:
    """Author one axis-aligned box collider (idempotent per unique path)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _make_material(stage, path: str, mu: float, restitution: float = 0.0):
    """Author a physics material prim (friction both static+dynamic = mu)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu))
    api.CreateDynamicFrictionAttr(float(mu))
    api.CreateRestitutionAttr(float(restitution))
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics")


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


def _dynamic_body(root, *, mass: float, lin_damp: float, ang_damp: float,
                  com=None, inertia=None) -> None:
    """Author a dynamic rigid body: explicit mass (root-level mass_props on custom
    spawner cfgs is silently ignored), optional explicit CoM + diagonal inertia
    (compound roots keep the CoM at the ORIGIN unless authored), damping, zeroed
    sleep thresholds, solver velocity iterations 4 (GPU sphere-creep fix)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    if com is not None:
        massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_cage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the capture cage: DYNAMIC compound — two side walls, back wall,
    roof; NO floor, NO front wall (the mouth, at local +x). Origin at the
    interior floor centre. Slick authored material on the wall bottoms (the whole
    body) keeps the slide force in the arm's range; authored CoM sits low so
    pushes cannot tip it."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _dynamic_body(root, mass=c.mass, lin_damp=0.2, ang_damp=0.8,
                  com=(-0.010, 0.0, 0.035),
                  inertia=(0.014, 0.020, 0.027))
    collide = _make_collide(c)
    mat = _make_material(stage, f"{prim_path}/phys_mat", c.mu, 0.0)
    hx, hy, t, wh = c.hx, c.hy, c.wall_t, c.wall_h
    prims = []
    for sgn, nm in ((1.0, "side_l"), (-1.0, "side_r")):
        prims.append(_add_box(stage, f"{prim_path}/{nm}",
                              center=(-t / 2, sgn * (hy + t / 2), wh / 2),
                              size=(2 * hx + t, t, wh), color=c.color,
                              collide=collide))
    prims.append(_add_box(stage, f"{prim_path}/back",
                          center=(-hx - t / 2, 0.0, wh / 2),
                          size=(t, 2 * (hy + t), wh), color=c.color,
                          collide=collide))
    prims.append(_add_box(stage, f"{prim_path}/roof",
                          center=(-t / 2, 0.0, wh + t / 2),
                          size=(2 * hx + t, 2 * (hy + t), t),
                          color=c.roof_color, collide=collide))
    for p in prims:
        _bind_material(p, mat)
    return root


def _spawn_chock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the chock: DYNAMIC yellow square bar, long axis along local y."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _dynamic_body(root, mass=c.mass, lin_damp=0.2, ang_damp=0.8)
    collide = _make_collide(c)
    mat = _make_material(stage, f"{prim_path}/phys_mat", c.mu, 0.0)
    p = _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0),
                 size=(c.side, 2 * c.half_len, c.side), color=c.color,
                 collide=collide)
    _bind_material(p, mat)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a ball: DYNAMIC sphere with angular damping so it rolls to rest,
    restitution 0 (no bounce), velocity iterations 4 (anti-creep)."""
    from pxr import Gf, UsdGeom

    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, mass=cfg.mass, lin_damp=0.05, ang_damp=0.30)
    mat = _make_material(stage, f"{prim_path}/phys_mat", 0.6, 0.0)
    r = float(cfg.radius)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg)(sph.GetPrim())
    _bind_material(sph.GetPrim(), mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cage" not in _SPAWNER_CACHE:

        @configclass
        class CageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cage)
            hx: float = 0.105
            hy: float = 0.080
            wall_t: float = 0.012
            wall_h: float = 0.110
            mass: float = 3.2
            mu: float = 0.35
            color: tuple = (0.30, 0.40, 0.62)
            roof_color: tuple = (0.18, 0.24, 0.40)
            contact_offset: float = 0.002

        @configclass
        class ChockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chock)
            side: float = 0.040
            half_len: float = 0.105
            mass: float = 0.18
            mu: float = 0.6
            color: tuple = (0.93, 0.80, 0.12)
            contact_offset: float = 0.002

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.030
            mass: float = 0.06
            color: tuple = (0.95, 0.95, 0.95)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cage=CageSpawnerCfg, chock=ChockSpawnerCfg,
                              ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CageCaptureSceneCfg(BaseCfg):
    """Config for `CageCaptureScene`. The interlocks are metric: the cage's roof
    and walls admit nothing except through the 160 mm x 110 mm mouth, the anchor
    clause (4.5 cm) forbids moving the 60 mm ball to the cage, and the 3.2 kg
    cage exceeds the arm's payload so it can only be slid."""

    # --- tunable: rubric thresholds ---------------------------------------------------------
    anchor_tol: float = tunable(0.045)     # ball must stay within this of its reset spot (m)
    approach_dist: float = tunable(0.09)   # mouth centre within this of the ball -> approach
    capture_margin: float = tunable(0.040) # ball centre this far behind the mouth plane
    chock_x_tol: float = tunable(0.020)    # chock seat window along the cage axis
    chock_y_tol: float = tunable(0.040)    # chock seat window across the mouth
    chock_align_deg: float = tunable(25.0) # chock long axis vs cage mouth line
    settle_speed: float = tunable(0.04)    # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)    # max |ang vel| when judging (rad/s)
    upright_deg: float = tunable(8.0)      # cage roof-up cone
    seated_z_tol: float = tunable(0.010)   # cage root height above the floor when seated

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    ball_x_min: float = tunable(0.55)      # white ball spawn band
    ball_x_max: float = tunable(0.66)
    ball_y_max: float = tunable(0.15)      # +/- band
    decoy_dy_min: float = tunable(0.28)    # decoy flank offset from the white ball
    decoy_dy_jit: float = tunable(0.06)
    decoy_dx_max: float = tunable(0.06)    # +/- along x
    cage_x_min: float = tunable(0.20)      # cage start band
    cage_x_max: float = tunable(0.28)
    cage_y_max: float = tunable(0.12)      # +/- band
    cage_yaw_jit_deg: float = tunable(25.0)  # yaw jitter about the facing-the-ball heading
    chock_x_min: float = tunable(0.34)     # chock park band (flank opposite the decoy)
    chock_x_max: float = tunable(0.42)
    chock_y_min: float = tunable(0.30)
    chock_y_jit: float = tunable(0.06)
    chock_yaw_jit: float = tunable(0.35)   # rad, about long-axis-along-x

    # --- info: cage structure -----------------------------------------------------------------
    hx: float = info(0.105)                # interior half-depth; mouth plane at local +hx
    hy: float = info(0.080)                # interior half-width (160 mm mouth)
    wall_t: float = info(0.012)
    wall_h: float = info(0.110)            # roof underside (mouth height)
    cage_mass: float = info(3.2)           # > the arm's 3 kg payload: slide, don't lift
    cage_mu: float = info(0.35)            # authored slick material (slide force ~13 N)

    # --- info: chock + balls ------------------------------------------------------------------
    chock_side: float = info(0.040)        # square bar cross-section (jaw-sized)
    chock_half_len: float = info(0.105)    # 210 mm long: spans the 160+2x12 mm mouth
    chock_mass: float = info(0.18)
    chock_seat_dx: float = info(0.026)     # nominal chock centre: hx + this (just outside)
    ball_r: float = info(0.030)            # 60 mm dia
    ball_mass: float = info(0.06)

    # --- info: colors + misc -------------------------------------------------------------------
    cage_color: tuple = info((0.30, 0.40, 0.62))
    roof_color: tuple = info((0.18, 0.24, 0.40))
    chock_color: tuple = info((0.93, 0.80, 0.12))
    white_color: tuple = info((0.95, 0.95, 0.95))
    black_color: tuple = info((0.06, 0.06, 0.06))
    ground_mu: float = info(0.5)
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.20 + 0.25 + 0.10 = 0.70 = the non-success cap)
    w_appr: float = info(0.15)
    w_part: float = info(0.20)
    w_capt: float = info(0.25)
    w_chk: float = info(0.10)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cage_capture")
class CageCaptureScene(BaseScene):
    cfg: CageCaptureSceneCfg

    def __init__(self, cfg: CageCaptureSceneCfg | None = None) -> None:
        super().__init__(cfg or CageCaptureSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        cage_spawn = sp["cage"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            hx=c.hx, hy=c.hy, wall_t=c.wall_t, wall_h=c.wall_h, mass=c.cage_mass,
            mu=c.cage_mu, color=c.cage_color, roof_color=c.roof_color,
            contact_offset=c.contact_offset,
        )
        chock_spawn = sp["chock"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            side=c.chock_side, half_len=c.chock_half_len, mass=c.chock_mass,
            color=c.chock_color, contact_offset=c.contact_offset,
        )
        white_spawn = sp["ball"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.ball_r, mass=c.ball_mass, color=c.white_color,
            contact_offset=c.contact_offset,
        )
        black_spawn = sp["ball"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.ball_r, mass=c.ball_mass, color=c.black_color,
            contact_offset=c.contact_offset,
        )
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "cage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cage",
                spawn=cage_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.0, 0.002)),
            ),
            "chock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chock",
                spawn=chock_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.38, -0.32, c.chock_side / 2 + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BallWhite",
                spawn=white_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.60, 0.0, c.ball_r + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BallBlack",
                spawn=black_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.60, 0.30, c.ball_r + 0.002)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cage: RigidObject = env.iscene["cage"]
        self.chock: RigidObject = env.iscene["chock"]
        self.ball: RigidObject = env.iscene["ball"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # the ANCHOR: where the white ball was put down at reset (world xy). Every
        # credit latch and success() itself is gated on the ball still being there.
        self._anchor = torch.zeros(n, 2, device=dev)
        # latches: partial progress survives transient achievements
        self._appr_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._part_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._capt_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._chk_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: white ball in its band (the ANCHOR is recorded here),
        decoy on a Bernoulli flank of the ball, cage in its start band with yaw
        roughly facing the ball, chock parked on the flank opposite the decoy."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(m, device=dev)  # burn: the first post-seed draw is degenerate

        # --- white ball: xy band; the spot it is put down at IS the anchor ---
        wx = c.ball_x_min + torch.rand(m, device=dev) * (c.ball_x_max - c.ball_x_min)
        wy = (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_max
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.ball_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)
        self._anchor[env_ids, 0] = wx + origin[:, 0]
        self._anchor[env_ids, 1] = wy + origin[:, 1]

        # --- decoy: Bernoulli flank of the white ball (torch.rand comparison) ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        dx = (torch.rand(m, device=dev) * 2 - 1) * c.decoy_dx_max
        dy = side * (c.decoy_dy_min + torch.rand(m, device=dev) * c.decoy_dy_jit)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = wx + dx, wy + dy, c.ball_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.decoy.write_root_state_to_sim(st, env_ids)

        # --- cage: start band, yaw = facing-the-ball heading + jitter ---
        cx = c.cage_x_min + torch.rand(m, device=dev) * (c.cage_x_max - c.cage_x_min)
        cy = (torch.rand(m, device=dev) * 2 - 1) * c.cage_y_max
        head = torch.atan2(wy - cy, wx - cx)
        yaw = head + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cage_yaw_jit_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = cx, cy, 0.002
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.cage.write_root_state_to_sim(st, env_ids)

        # --- chock: parked flat on the flank OPPOSITE the decoy, long axis ~ x ---
        kx = c.chock_x_min + torch.rand(m, device=dev) * (c.chock_x_max - c.chock_x_min)
        ky = -side * (c.chock_y_min + torch.rand(m, device=dev) * c.chock_y_jit)
        kpsi = math.pi / 2 + (torch.rand(m, device=dev) * 2 - 1) * c.chock_yaw_jit
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = kx, ky, c.chock_side / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(kpsi / 2), torch.sin(kpsi / 2)
        st[:, 0:3] += origin
        self.chock.write_root_state_to_sim(st, env_ids)

        self._appr_ever[env_ids] = False
        self._part_ever[env_ids] = False
        self._capt_ever[env_ids] = False
        self._chk_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cage": self.cage.data.root_state_w[env_ids].clone(),
            "chock": self.chock.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "anchor": self._anchor[env_ids].clone(),
            "appr": self._appr_ever[env_ids].clone(),
            "part": self._part_ever[env_ids].clone(),
            "capt": self._capt_ever[env_ids].clone(),
            "chk": self._chk_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cage.write_root_state_to_sim(state["cage"], env_ids)
        self.chock.write_root_state_to_sim(state["chock"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._anchor[env_ids] = state["anchor"]
        self._appr_ever[env_ids] = state["appr"]
        self._part_ever[env_ids] = state["part"]
        self._capt_ever[env_ids] = state["capt"]
        self._chk_ever[env_ids] = state["chk"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the open floor lies a WHITE BALL ({2 * c.ball_r * 100:.0f} cm) — the "
            f"target — and, {c.decoy_dy_min * 100:.0f}+ cm to its left or right, an "
            f"identical BLACK BALL (a decoy; side and positions change between "
            f"episodes). Nearer the robot stands a slate-blue CAPTURE CAGE: a box "
            f"with two side walls, a back wall and a dark roof, but NO floor and NO "
            f"front wall — its open front MOUTH is {2 * c.hy * 100:.0f} cm wide and "
            f"{c.wall_h * 100:.0f} cm tall. The cage weighs {c.cage_mass:.1f} kg — "
            f"more than the arm can lift — but it slides on the floor when pushed. "
            f"Off to one side lies a YELLOW CHOCK BAR ({2 * c.chock_half_len * 100:.0f} "
            f"cm long, {c.chock_side * 100:.0f} cm square), light and easy to grasp.\n"
            f"Goal: CAPTURE THE WHITE BALL WHERE IT LIES. The white ball must NOT be "
            f"moved: if it ends up more than {c.anchor_tol * 100:.1f} cm from the spot "
            f"where it started, the task is failed — so it cannot be pushed, carried "
            f"or knocked into the cage. Instead, slide the cage across the floor, "
            f"open mouth first, steering so the mouth passes around the ball without "
            f"touching it, until the ball sits DEEP inside (at least "
            f"{c.capture_margin * 100:.0f} cm behind the mouth plane, i.e. roughly "
            f"under the middle of the roof). Then pick up the yellow chock bar and "
            f"lay it flat on the floor squarely across the mouth opening (centered "
            f"on the mouth, parallel to it, just outside the walls) to close the "
            f"cage. Finish with everything at rest: ball inside at its original "
            f"spot, chock barring the mouth, cage upright and seated on the floor, "
            f"and the BLACK ball still OUTSIDE the cage. Capturing the black ball, "
            f"displacing the white ball, leaving the mouth unbarred, or parking the "
            f"chock anywhere but across the mouth — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the heavy blue cage across the floor, open mouth first, so it "
            "swallows the white ball where it lies — the white ball must not move "
            "from its spot, and the black ball must stay outside. Then lay the "
            "yellow chock bar flat across the cage mouth to close it."
        )

    # ----- readings -------------------------------------------------------------------------------
    def _local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the CAGE's body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.cage.data.root_pos_w
        return quat_apply_inverse(self.cage.data.root_quat_w, rel)

    def anchor_ok(self) -> torch.Tensor:
        """(N,) bool: white ball within anchor_tol of where it was put down."""
        d = self.ball.data.root_pos_w[:, 0:2] - self._anchor
        return d.norm(dim=-1) < self.cfg.anchor_tol

    def cage_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.cage.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_deg))

    def cage_seated(self) -> torch.Tensor:
        """(N,) bool: cage upright with its wall bottoms on the floor."""
        z = (self.cage.data.root_pos_w - self.env_origins)[:, 2]
        return self.cage_upright() & (z.abs() < self.cfg.seated_z_tol)

    def mouth_center_w(self) -> torch.Tensor:
        """(N, 3) the mouth centre in world coordinates."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = torch.tensor([c.hx, 0.0, c.ball_r], device=self.env.device).expand(
            self.env.num_envs, 3)
        return self.cage.data.root_pos_w + quat_apply(self.cage.data.root_quat_w, loc)

    def _approach(self) -> torch.Tensor:
        """(N,) bool: mouth centre horizontally within approach_dist of the ball."""
        d = (self.mouth_center_w() - self.ball.data.root_pos_w)[:, 0:2]
        return (d.norm(dim=-1) < self.cfg.approach_dist) & self.cage_upright()

    def _partial(self) -> torch.Tensor:
        """(N,) bool: ball past the mouth plane, between the walls, under the roof."""
        c = self.cfg
        loc = self._local(self.ball)
        return ((loc[:, 0] < c.hx - 0.005) & (loc[:, 0] > -c.hx)
                & (loc[:, 1].abs() < c.hy) & (loc[:, 2] > 0.005)
                & (loc[:, 2] < c.wall_h - 0.005) & self.cage_upright())

    def _captured(self) -> torch.Tensor:
        """(N,) bool: ball FULLY inside — at least capture_margin behind the mouth
        plane, clear of the back wall, between the walls, under the roof."""
        c = self.cfg
        loc = self._local(self.ball)
        return ((loc[:, 0] < c.hx - c.capture_margin) & (loc[:, 0] > -c.hx + 0.012)
                & (loc[:, 1].abs() < c.hy - 0.004) & (loc[:, 2] > 0.005)
                & (loc[:, 2] < c.wall_h - 0.005) & self.cage_upright())

    def _decoy_inside(self) -> torch.Tensor:
        """(N,) bool (loose): decoy within the cage footprint under the roof."""
        c = self.cfg
        loc = self._local(self.decoy)
        return ((loc[:, 0] < c.hx + 0.010) & (loc[:, 0] > -c.hx - 0.020)
                & (loc[:, 1].abs() < c.hy + 0.012) & (loc[:, 2] < c.wall_h))

    def _chock_seated(self) -> torch.Tensor:
        """(N,) bool: chock flat on the floor squarely across the mouth — cage-frame
        window on its centre, long axis aligned with the mouth line."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = self._local(self.chock)
        x_ok = (loc[:, 0] - (c.hx + c.chock_seat_dx)).abs() < c.chock_x_tol
        y_ok = loc[:, 1].abs() < c.chock_y_tol
        z_ok = (loc[:, 2] > 0.012) & (loc[:, 2] < 0.030)
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        bar_axis = quat_apply(self.chock.data.root_quat_w, ey)
        mouth_axis = quat_apply(self.cage.data.root_quat_w, ey)
        align = (bar_axis * mouth_axis).sum(dim=-1).abs() >= math.cos(
            math.radians(c.chock_align_deg))
        return x_ok & y_ok & z_ok & align & self.cage_upright()

    def _update_latches(self) -> None:
        """Latch progress — every latch gated on the ball anchor at latch time, so
        credit is only ever earned while the ball sits where it lay."""
        a = self.anchor_ok()
        self._appr_ever |= self._approach() & a
        self._part_ever |= self._partial() & a
        self._capt_ever |= self._captured() & a
        self._chk_ever |= self._chock_seated() & self._captured() & a

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def _still(self) -> torch.Tensor:
        c = self.cfg
        return ((self.cage.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.cage.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)
                & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.ball.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)
                & (self.chock.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed))

    def success(self) -> torch.Tensor:
        """(N,) bool: white ball fully inside AND at its anchor, chock seated
        across the mouth, decoy outside, cage upright + seated, all at rest.
        Physical outcomes only."""
        self._update_latches()
        return (self._captured() & self.anchor_ok() & self._chock_seated()
                & ~self._decoy_inside() & self.cage_seated() & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*approach + 0.20*partial + 0.25*capture +
        0.10*chocked — all latched and anchor-gated, ~0 for doing nothing, capped
        0.70 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_appr * self._appr_ever.float()
                + c.w_part * self._part_ever.float()
                + c.w_capt * self._capt_ever.float()
                + c.w_chk * self._chk_ever.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="cage_capture", robot="null"))
