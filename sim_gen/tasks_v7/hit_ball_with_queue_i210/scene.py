"""BallCorralScene — trap the white ball under a mouth-down cage, then ESCORT the
caged ball across a tilted plateau until the cage sits centred on the green target
disc. Derived from rlbench/hit_ball_with_queue, with the seed's whole theory of the
task inverted: the payload is never propelled and never travels a single centimetre
uncontained.

Seed (rlbench/hit_ball_with_queue): grasp a cue stick and STRIKE a free ball across
an open surface into a pocket — tool-mediated impulse transfer; aim + momentum decide
the outcome; the ball flies free the whole way. Here free ball travel IS the failure
mode:

- The work surface is an elevated PLATEAU tilted 1.8-2.6 deg with OPEN drop-off
  edges. A free ball can rest ONLY in one of two shallow recessed cradles; released
  anywhere else it rolls downhill and off the edge to the ground, where it is
  UNRECOVERABLE (85 mm diameter > the 80 mm Franka jaw span, asserted below, and the
  plateau face is a vertical 0.24 m cliff no ball can be rolled up).
- Striking the ball (the seed's move) therefore scores nothing: any impulse pops it
  over its cradle lip and gravity takes it off the downhill edge (smoke's
  seed-strategy check executes exactly this).
- The goal marker (a flat green disc, upslope) is bare tilted surface: a bare ball
  CANNOT REST there, so no amount of careful bare pushing can ever succeed either.
- The only winning plan is CAPTURE-THEN-ESCORT: lower the open-mouth blue CAGE
  straight down over the white ball until its rim sits FLUSH on the surface (a real
  contact seating — the rim must land around the ball without clipping it), then
  SLIDE the caged assembly quasi-statically across the plateau, the ball rolling
  captive inside, until the cage is centred on the green disc. Lifting the cage
  mid-escort frees the ball on open slope — instant loss — so the transport itself
  must stay a sustained, rim-down pushing contact.
- Identity matters: an identical-size ORANGE decoy ball waits in the other cradle;
  caging and delivering the decoy scores nothing.

So a solver needs a different PLAN from the seed (bring the container to the payload,
seat it, then drag the closed container — no aiming, no impulse, no free flight) and
different CODE STRUCTURE (a vertical-seating placement plus a closed-loop lateral
slide, instead of a grasp-stick + swing/strike controller).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - plateau: ONE KINEMATIC compound, origin at the CENTRE of its top face, local +x
    pointing DOWNHILL once the root is pitched at reset. A 0.70 x 0.60 m deck on a
    plinth (top 0.24 m above ground), with two 62 mm square, 4 mm deep dark recessed
    CRADLES (the only ball rest points).
  - marker: a flat green disc (kinematic, NO collision — purely visual), re-posed
    per episode at the sampled goal point; the judged zone centre is its centre.
  - cage: DYNAMIC blue octagonal open-bottom cage (inner span 120 mm across flats,
    walls 105 mm tall, two roof slats, and a 12 mm carry BAR raised on posts — a
    clean parallel-jaw pinch). Its origin (and authored CoM) sits at rim level, so
    the seated cage is extremely stable and low pushes cannot tip it.
  - balls: white TARGET and orange DECOY, both 85 mm (> 80 mm jaw span: never
    graspable, only pushable — and pushable only into loss, except via the cage).

Per-episode randomization (readback-verifiable): plateau yaw + xy offset + TILT
angle, white/decoy cradle Bernoulli swap, ball in-cradle jitter, cage parking jitter
+ free yaw, goal-zone centre sampled over the upslope half.

Rubric (0..1; partial progress latched so credit never evaporates):
  0.30 * caged      — the WHITE ball ever enclosed under the flush-seated cage
                      (rim at surface level, cage upright, ball inside) (latched)
  0.35 * delivered  — ever caged AND cage centred within `zone_tol` of the goal
                      point (latched; only an enclosed arrival earns this)
  1.0 iff success() — caged, centred on the goal, everything settled.
                      Non-success cap 0.65; null policy ~0.

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

FRANKA_JAW_SPAN = 0.080  # Franka parallel-jaw max opening (m) — the ungraspability bound


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             orient=None) -> None:
    """Author one box (collider unless collide is None; orient = wxyz quaternion)."""
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
    if collide is not None:
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


def _armor(root, *, lin_damp: float, ang_damp: float) -> None:
    """The dynamic-body physics armor: depenetration cap, damping, no sleeping
    (judged for stillness), velocity iterations 4 (kills sphere-on-box creep)."""
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(lin_damp)
    pxrb.CreateAngularDampingAttr(ang_damp)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_plateau(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the plateau: ONE KINEMATIC compound. Origin at the CENTRE of the TOP
    face; +x is the downhill direction once the root is pitched. Top = a deck layer
    (thickness = recess depth) with two square holes over a dark bulk slab whose top
    face forms the recess floors; plinth column to the ground below."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    hx, hy, d = c.slab_hx, c.slab_hy, c.recess_depth
    rx, ry, hw = c.recess_x, c.recess_dy, c.recess_hw

    def plate(tag: str, x0: float, x1: float, y0: float, y1: float) -> None:
        _add_box(stage, f"{prim_path}/deck_{tag}",
                 center=((x0 + x1) / 2, (y0 + y1) / 2, -d / 2),
                 size=(x1 - x0, y1 - y0, d), color=c.deck_color, collide=collide)

    # deck layer: 3 full-width y-strips + 2 x-plates per recess row
    plate("s_lo", -hx, hx, -hy, -ry - hw)
    plate("s_mid", -hx, hx, -ry + hw, ry - hw)
    plate("s_hi", -hx, hx, ry + hw, hy)
    for sgn, nm in ((-1.0, "a"), (1.0, "b")):
        y0, y1 = sgn * ry - hw, sgn * ry + hw
        plate(f"r{nm}_w", -hx, rx - hw, y0, y1)
        plate(f"r{nm}_e", rx + hw, hx, y0, y1)

    # bulk slab (its top face = the recess floors) + plinth column
    _add_box(stage, f"{prim_path}/bulk",
             center=(0.0, 0.0, -d - c.bulk_t / 2),
             size=(2 * hx, 2 * hy, c.bulk_t), color=c.recess_color, collide=collide)
    plinth_h = c.top_height - d - c.bulk_t
    _add_box(stage, f"{prim_path}/plinth",
             center=(0.0, 0.0, -d - c.bulk_t - plinth_h / 2),
             size=(0.30, 0.30, plinth_h), color=c.plinth_color, collide=collide)
    return root


def _spawn_marker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the goal marker: a KINEMATIC flat green disc with NO collider — purely
    visual (the sliding cage passes over it without a step)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    r, h = float(cfg.radius), float(cfg.height)
    disc.CreateRadiusAttr(r)
    disc.CreateHeightAttr(h)
    disc.CreateAxisAttr("Z")
    disc.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    return root


def _spawn_cage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cage: DYNAMIC — 8 octagon wall segments (open bottom = the mouth),
    two roof slats, and a raised carry bar on posts. Origin at the RIM PLANE centre,
    so the MassAPI mass puts the CoM at floor level: the seated cage is extremely
    stable and pushes near the rim produce no tipping moment."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    _armor(root, lin_damp=0.10, ang_damp=0.50)
    collide = _make_collide(cfg)
    c = cfg
    a, t, h = c.inner_a, c.wall_t, c.wall_h
    r_c = a + t / 2
    seg = 2 * r_c * math.tan(math.pi / 8) + 0.004  # slight overlap closes corners
    for i in range(8):
        phi = i * math.pi / 4
        q = (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(r_c * math.cos(phi), r_c * math.sin(phi), h / 2),
                 size=(t, seg, h), color=c.color, collide=collide, orient=q)
    slat_l = 2 * (a + t) / math.cos(math.pi / 8) * 0.99
    _add_box(stage, f"{prim_path}/slat_x", center=(0.0, 0.0, h + 0.003),
             size=(slat_l, 0.044, 0.006), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/slat_y", center=(0.0, 0.0, h + 0.003),
             size=(0.044, slat_l, 0.006), color=c.color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/post_{'l' if sgn > 0 else 'r'}",
                 center=(0.0, sgn * 0.030, h + 0.006 + 0.006),
                 size=(0.012, 0.012, 0.012), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, h + 0.006 + 0.012 + 0.006),
             size=(0.012, 0.072, 0.012), color=c.bar_color, collide=collide)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a ball: DYNAMIC sphere. Damping is set so a lost ball still rolls
    briskly off the tilted plateau (terminal v ~ g*sin(tilt)/c_ang ~ 0.5 m/s) but a
    captive ball parks in the cage corner instead of wall-rolling for tens of
    seconds; no sleeping (judged for stillness)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    _armor(root, lin_damp=0.06, ang_damp=0.45)
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

    if "plateau" not in _SPAWNER_CACHE:

        @configclass
        class PlateauSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plateau)
            slab_hx: float = 0.35
            slab_hy: float = 0.30
            recess_depth: float = 0.004
            bulk_t: float = 0.046
            top_height: float = 0.24
            recess_x: float = 0.08
            recess_dy: float = 0.13
            recess_hw: float = 0.031
            deck_color: tuple = (0.62, 0.64, 0.68)
            recess_color: tuple = (0.22, 0.23, 0.27)
            plinth_color: tuple = (0.35, 0.36, 0.40)
            contact_offset: float = 0.002

        @configclass
        class MarkerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_marker)
            radius: float = 0.075
            height: float = 0.002
            color: tuple = (0.12, 0.72, 0.25)

        @configclass
        class CageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cage)
            inner_a: float = 0.060
            wall_t: float = 0.006
            wall_h: float = 0.105
            color: tuple = (0.15, 0.35, 0.85)
            bar_color: tuple = (0.10, 0.22, 0.55)
            contact_offset: float = 0.002

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.0425
            color: tuple = (0.95, 0.95, 0.95)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(plateau=PlateauSpawnerCfg, marker=MarkerSpawnerCfg,
                              cage=CageSpawnerCfg, ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallCorralSceneCfg(BaseCfg):
    """Config for `BallCorralScene`. The interlocks are metric: both balls (85 mm)
    are wider than the Franka jaw span (80 mm, asserted) so they can never be grasped
    or carried; the tilted deck lets a free ball rest ONLY inside a cradle (4 mm lip
    vs 1.8-2.6 deg slope), so a ball pushed loose anywhere else rolls off the open
    downhill edge and is unrecoverable on the ground; the goal zone is bare slope, so
    ONLY the caged assembly can ever be at rest there; the cage mouth (120 mm across
    flats) clears the ball by 17.5 mm laterally — a real but comfortable seating
    tolerance."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.80)  # max |ang vel| when judging (rad/s)
    flush_z_tol: float = tunable(0.006)  # cage rim-plane height above deck for "flush" (m)
    flush_up_cos: float = tunable(0.9945)  # min cage-up . plateau-up (~6 deg)
    inside_xy_tol: float = tunable(0.035)  # ball centre to cage axis for "inside" (m):
    # honest by construction — a ball physically inside can be at most
    # inner_a - ball_r = 17.5 mm off-axis; one outside touching the wall is >= 109 mm
    zone_tol: float = tunable(0.045)  # cage centre to goal-point distance for "delivered"

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_max_deg: float = tunable(25.0)  # whole-plateau yaw (+/- deg)
    xy_jitter: float = tunable(0.05)  # whole-plateau xy offset (+/- m)
    tilt_min_deg: float = tunable(1.8)  # plateau tilt range (downhill = local +x)
    tilt_max_deg: float = tunable(2.6)
    swap_cradles: bool = tunable(True)  # Bernoulli white/decoy cradle swap
    ball_jitter: float = tunable(0.008)  # in-cradle ball xy jitter (+/- m)
    cage_jitter: float = tunable(0.020)  # cage parking xy jitter (+/- m)
    cage_yaw_deg: float = tunable(180.0)  # cage free yaw (+/- deg)
    zone_x: tuple = tunable((-0.23, -0.14))  # goal-point sampling box (plateau frame,
    zone_y: tuple = tunable((-0.12, 0.12))  # upslope half — bare slope, no rest point)

    # --- info: layout (plateau frame: origin at top-face centre, +x downhill) --------------------
    slab_hx: float = info(0.35)
    slab_hy: float = info(0.30)
    top_height: float = info(0.24)  # deck top above the ground (the cliff height)
    recess_depth: float = info(0.004)
    recess_x: float = info(0.08)  # both cradles at this x (downhill half)
    recess_dy: float = info(0.13)  # cradles at y = -/+ this
    recess_hw: float = info(0.031)  # cradle square half-width
    parking: tuple = info((-0.05, -0.20))  # cage parking point

    # --- info: cage ------------------------------------------------------------------------------
    inner_a: float = info(0.060)  # octagon inner half-span (across flats)
    wall_t: float = info(0.006)
    wall_h: float = info(0.105)
    cage_mass: float = info(0.35)
    cage_color: tuple = info((0.15, 0.35, 0.85))

    # --- info: balls -----------------------------------------------------------------------------
    ball_r: float = info(0.0425)  # 85 mm dia > 80 mm jaw span: never graspable
    ball_mass: float = info(0.25)
    white_color: tuple = info((0.95, 0.95, 0.95))
    decoy_color: tuple = info((0.95, 0.55, 0.10))

    marker_r: float = info(0.075)
    contact_offset: float = info(0.002)
    # rubric weights (0.30 + 0.35 = 0.65 = the non-success cap)
    w_cage: float = info(0.30)
    w_deliver: float = info(0.35)

    def __post_init__(self) -> None:
        assert 2 * self.ball_r > FRANKA_JAW_SPAN, \
            "balls must be wider than the Franka jaw span (ungraspable by design)"
        assert self.inner_a > self.ball_r + 0.012, \
            "cage mouth must clear the ball with a real seating tolerance"
        assert self.recess_hw * math.sqrt(2.0) < self.inner_a, \
            "the flush rim must land clear of the cradle hole it straddles"
        cage_out_r = (self.inner_a + self.wall_t) / math.cos(math.pi / 8)
        assert self.zone_x[0] - self.zone_tol - cage_out_r >= -self.slab_hx, \
            "goal zone (+ tolerance + cage footprint) must stay on the slab"
        # a ball resting in a cradle: centre height above the DECK plane
        self.ball_rest_z: float = self.ball_r - self.recess_depth


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ball_corral")
class BallCorralScene(BaseScene):
    cfg: BallCorralSceneCfg

    def __init__(self, cfg: BallCorralSceneCfg | None = None) -> None:
        super().__init__(cfg or BallCorralSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        plateau_spawn = sp["plateau"](
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            slab_hx=c.slab_hx, slab_hy=c.slab_hy, recess_depth=c.recess_depth,
            top_height=c.top_height, recess_x=c.recess_x, recess_dy=c.recess_dy,
            recess_hw=c.recess_hw, contact_offset=c.contact_offset,
        )
        marker_spawn = sp["marker"](
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            radius=c.marker_r,
        )
        cage_spawn = sp["cage"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.cage_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            inner_a=c.inner_a, wall_t=c.wall_t, wall_h=c.wall_h,
            color=c.cage_color, contact_offset=c.contact_offset,
        )

        def ball_spawn(color):
            return sp["ball"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                radius=c.ball_r, color=color, contact_offset=c.contact_offset,
            )

        z0 = c.top_height
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
            "plateau": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plateau",
                spawn=plateau_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0)),
            ),
            "marker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marker",
                spawn=marker_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.20, 0.0, z0 + 0.001)),
            ),
            "cage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cage",
                spawn=cage_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.parking[0], c.parking[1], z0 + 0.003)),
            ),
            "white": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WhiteBall",
                spawn=ball_spawn(c.white_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.recess_x, -c.recess_dy, z0 + c.ball_rest_z + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DecoyBall",
                spawn=ball_spawn(c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.recess_x, c.recess_dy, z0 + c.ball_rest_z + 0.002)),
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
        self.plateau: RigidObject = env.iscene["plateau"]
        self.marker: RigidObject = env.iscene["marker"]
        self.cage: RigidObject = env.iscene["cage"]
        self.white: RigidObject = env.iscene["white"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._zone = torch.zeros(n, 2, device=dev)  # goal point, plateau frame
        # latches: partial progress survives transient regressions (rubric requirement)
        self._caged_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plateau re-posed (yaw + xy + TILT), goal point sampled and
        the marker disc moved there, white/decoy Bernoulli cradle swap with in-cradle
        jitter, cage re-parked with jitter + free yaw, latches cleared."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- plateau (kinematic): yaw * pitch(tilt), xy offset ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max_deg)
        tilt = math.radians(c.tilt_min_deg) + torch.rand(m, device=dev) * (
            math.radians(c.tilt_max_deg) - math.radians(c.tilt_min_deg))
        qz = torch.zeros(m, 4, device=dev)
        qz[:, 0], qz[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        qy = torch.zeros(m, 4, device=dev)
        qy[:, 0], qy[:, 2] = torch.cos(tilt / 2), torch.sin(tilt / 2)
        q_root = quat_mul(qz, qy)  # pitch about local y: +x tips downhill
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        st[:, 2] = c.top_height
        st[:, 3:7] = q_root
        st[:, 0:3] += origin
        self.plateau.write_root_state_to_sim(st, env_ids)
        p_pos, p_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        def place(body, loc: torch.Tensor, quat: torch.Tensor) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = p_pos + quat_apply(p_quat, loc)
            s[:, 3:7] = quat
            body.write_root_state_to_sim(s, env_ids)

        # --- goal point + marker disc ---
        zone = torch.zeros(m, 2, device=dev)
        zone[:, 0] = c.zone_x[0] + torch.rand(m, device=dev) * (c.zone_x[1] - c.zone_x[0])
        zone[:, 1] = c.zone_y[0] + torch.rand(m, device=dev) * (c.zone_y[1] - c.zone_y[0])
        self._zone[env_ids] = zone
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0:2] = zone
        # disc TOP sits 0.2 mm above the deck plane: visible, but no step for the
        # sliding cage rim even in the worst case (the disc has no collider anyway)
        loc[:, 2] = 0.0002 - 0.001
        place(self.marker, loc, p_quat)

        # --- balls: Bernoulli cradle swap + in-cradle jitter ---
        if c.swap_cradles:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        sgn_white = torch.where(swap, 1.0, -1.0)
        ident = torch.zeros(m, 4, device=dev)
        ident[:, 0] = 1.0
        for body, sgn in ((self.white, sgn_white), (self.decoy, -sgn_white)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.recess_x
            loc[:, 1] = sgn * c.recess_dy
            loc[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
            loc[:, 2] = c.ball_rest_z + 0.002
            place(body, loc, ident)

        # --- cage: parking + jitter + free yaw about the plateau normal ---
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cage_yaw_deg)
        qc = torch.zeros(m, 4, device=dev)
        qc[:, 0], qc[:, 3] = torch.cos(cyaw / 2), torch.sin(cyaw / 2)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.parking[0]
        loc[:, 1] = c.parking[1]
        loc[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.cage_jitter
        loc[:, 2] = 0.003
        place(self.cage, loc, quat_mul(p_quat, qc))

        # --- clear latches ---
        self._caged_ever[env_ids] = False
        self._delivered_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plateau": self.plateau.data.root_state_w[env_ids].clone(),
            "marker": self.marker.data.root_state_w[env_ids].clone(),
            "cage": self.cage.data.root_state_w[env_ids].clone(),
            "white": self.white.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "zone": self._zone[env_ids].clone(),
            "caged_ever": self._caged_ever[env_ids].clone(),
            "delivered_ever": self._delivered_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plateau.write_root_state_to_sim(state["plateau"], env_ids)
        self.marker.write_root_state_to_sim(state["marker"], env_ids)
        self.cage.write_root_state_to_sim(state["cage"], env_ids)
        self.white.write_root_state_to_sim(state["white"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._zone[env_ids] = state["zone"]
        self._caged_ever[env_ids] = state["caged_ever"]
        self._delivered_ever[env_ids] = state["delivered_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An elevated gray PLATEAU (a {2 * c.slab_hx * 100:.0f} x "
            f"{2 * c.slab_hy * 100:.0f} cm deck standing {c.top_height * 100:.0f} cm "
            f"above the floor on a plinth) is TILTED by about 2 degrees; its edges "
            f"are open drop-offs. Two shallow DARK SQUARE CRADLES "
            f"({2 * c.recess_hw * 1000:.0f} mm wide, {c.recess_depth * 1000:.0f} mm "
            f"deep) are recessed into the deck on its downhill half; a WHITE ball "
            f"sits in one and an ORANGE ball in the other (which sits where changes "
            f"between episodes). Both balls are {2 * c.ball_r * 1000:.0f} mm across "
            f"— too wide for the gripper jaws: they can only ever be pushed, never "
            f"grasped or carried. Because the deck is tilted, the cradles are the "
            f"ONLY places a free ball can rest: a ball that leaves its cradle rolls "
            f"downhill, off the edge, and is LOST (it cannot be grasped and the "
            f"plateau face cannot be rolled up). On the upslope half a flat GREEN "
            f"DISC ({2 * c.marker_r * 100:.0f} cm across, purely a surface marking) "
            f"shows the goal point; bare tilted deck — no ball can rest on it "
            f"uncontained. A BLUE open-bottomed CAGE (octagonal, "
            f"{2 * c.inner_a * 1000:.0f} mm across inside, {c.wall_h * 1000:.0f} mm "
            f"tall walls, barred roof) stands mouth-down elsewhere on the deck, with "
            f"a raised {12:.0f} mm carry BAR across its top — pinch the bar with the "
            f"jaws to lift it.\n"
            f"Goal: the WHITE ball must end up enclosed under the cage with the "
            f"cage's rim flat on the deck and the cage centred on the green disc "
            f"(within about {c.zone_tol * 100:.1f} cm), everything at rest. The only "
            f"workable plan: lift the cage by its bar, centre it above the white "
            f"ball, lower it STRAIGHT down until the rim sits flush on the deck "
            f"around the ball (the mouth clears the ball by under 2 cm — do not "
            f"clip the ball on the way down or it will be knocked loose and lost), "
            f"release, then SLIDE the caged assembly along the deck (push low on its "
            f"wall) until it is centred on the green disc. NEVER strike or shove a "
            f"bare ball — any free-rolling ball leaves the plateau and the task is "
            f"failed. Never lift the cage once the ball is inside anywhere but over "
            f"a cradle: the freed ball rolls away. Caging or delivering the ORANGE "
            f"ball counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the blue cage by its bar, lower it straight down over the white "
            "ball so its rim sits flat on the deck, then slide the caged ball along "
            "the tilted deck until the cage is centred on the green disc, and leave "
            "it at rest there. Never strike the balls or lift the loaded cage: a "
            "ball rolling free falls off the plateau and the task is failed."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body position in the plateau frame (+x downhill, z = deck normal,
        origin at the deck-top centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.plateau.data.root_pos_w
        return quat_apply_inverse(self.plateau.data.root_quat_w, rel)

    def _cage_flush(self) -> torch.Tensor:
        """(N,) bool: cage rim plane at deck level and cage upright (plateau frame)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up_w = quat_apply(self.cage.data.root_quat_w, ez)
        up_p = quat_apply_inverse(self.plateau.data.root_quat_w, up_w)
        loc = self._local(self.cage)
        return (loc[:, 2].abs() < c.flush_z_tol) & (up_p[:, 2] > c.flush_up_cos)

    def _inside(self, ball: RigidObject) -> torch.Tensor:
        """(N,) bool: ball centre within `inside_xy_tol` of the cage axis AND at
        deck-resting height (plateau frame) — with a flush rim this is honest by
        construction (see cfg docstring)."""
        c = self.cfg
        b, g = self._local(ball), self._local(self.cage)
        near = (b[:, 0:2] - g[:, 0:2]).norm(dim=-1) < c.inside_xy_tol
        on_deck = (b[:, 2] > c.ball_r - 0.013) & (b[:, 2] < c.ball_r + 0.033)
        return near & on_deck

    def caged(self) -> torch.Tensor:
        """(N,) bool: the WHITE ball enclosed under the flush-seated cage."""
        return self._cage_flush() & self._inside(self.white)

    def in_zone(self) -> torch.Tensor:
        """(N,) bool: cage centre within `zone_tol` of the goal point."""
        g = self._local(self.cage)
        return (g[:, 0:2] - self._zone).norm(dim=-1) < self.cfg.zone_tol

    def _still(self) -> torch.Tensor:
        c = self.cfg
        return ((self.cage.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.white.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.white.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _update_latches(self) -> None:
        caged = self.caged()
        self._caged_ever |= caged
        self._delivered_ever |= caged & self.in_zone()

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the plateau is kinematic and jointless) — just latch rubric
        progress every step so credit never evaporates under correct behavior."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: white ball caged, cage centred on the goal point, everything
        settled. Physical outcome only — a bare ball can never rest in the zone (the
        zone is bare tilted deck) and the balls can never be grasped, so only a real
        capture-then-escort can reach this state."""
        self._update_latches()
        return self.caged() & self.in_zone() & self._still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*caged + 0.35*delivered — both latched, ~0 for
        doing nothing, non-success cap 0.65 — and exactly 1.0 iff success() holds."""
        c = self.cfg
        self._update_latches()
        base = (c.w_cage * self._caged_ever.float()
                + c.w_deliver * self._delivered_ever.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="ball_corral", robot="null"))
