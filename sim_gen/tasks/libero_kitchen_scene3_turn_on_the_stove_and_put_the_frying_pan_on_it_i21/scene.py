"""PrimerStoveScene — pump-prime the stove with repeated full plunger strokes, then set
the griddle on the burner (seed: libero_90 kitchen_scene3 "turn on the stove and put the
frying pan on it", strategically rebuilt).

The seed's plan is a ONE-SHOT TOGGLE plus a placement: rotate the stove knob past a
threshold once, then put the frying pan on the burner. Here the stove has NO knob at
all — it is a pressure-primed camp stove whose burner lights only after the fuel line
has been pressurized by N full strokes of a spring-return pump plunger (N in {2, 3, 4},
sampled per episode). A "stroke" is counted with FULL HYSTERESIS: the red pump cap must
be pressed all the way down (>= 80% of its 35 mm travel) AND then released all the way
back up (<= 20%) before it counts. That makes every one-shot or degenerate actuation
measured-insufficient by construction:

  * the seed's single toggle (one press-and-release) leaves the counter at 1 < N;
  * press-and-HOLD never counts a stroke at all (no release edge);
  * partial jiggling that never reaches full depth, or never fully releases, counts 0.

The solver therefore needs a different plan: a CYCLIC, repeated actuation loop with
feedback (pump ... watch the stroke lamps / burner ... pump again), followed by the
placement. Execution order is NOT required: priming before or after placing the griddle
both succeed — success is the settled conjunction.

Mechanism (all procedural primitives; the prismatic spring-plate pattern proven in this
batch's weighbridge and in robobench's microwave button): a kinematic stove slab carries
a kinematic burner disc at one end and a kinematic pump boss at the other; the dynamic
pump cap rides a Z prismatic joint on the boss (pair collision disabled — the joint
limits are the mechanical stops; SYMMETRIC limits +/- travel, the GPU sign-convention
hedge), returned by a post_step spring `f_z = k*(home - z) - c*v_z` with gravity
disabled on the cap so home is the exact rest pose. Four indicator lamps on the slab
show stroke progress (dark-red dot = a stroke still needed this episode, amber = done,
dark-gray = unused slot — the lamp row also DISPLAYS how many strokes this episode
needs), and a flame ring on the burner glows orange once primed. The priming latch is
permanent for the episode (a primed stove stays lit).

The graded object is a black cast-iron GRIDDLE: a flat disc with a steel center-knob
handle (the knob-post pinch + carry pattern proven in this batch — CoM hangs directly
under the pinch). A silver moka pot (the seed's own distractor) rides along as a
distractor; it belongs nowhere.

Judged on PHYSICAL outcome only: live cap depth drives the stroke counter; the griddle
must physically REST centered on the burner disc (xy within `on_burner_r` of the burner
axis, bottom at the burner top, upright, settled) while primed, SUSTAINED for
`hold_steps` consecutive substeps. score() in [0, 1]: stroke progress up to 0.35,
0.45 latched once primed, 0.12 latched once the griddle is lifted, 0.25 latched once
the griddle has rested on the burner (primed or not — the seed-strategy cap), 0.85
latched once primed-AND-on-burner have coexisted, 1.0 iff success().

Layout is designed for a single Franka at base (-0.50, 0, 0), chosen in solve.py: the
stove slab is fixed ~0.55 m in front of the base (burner end at bearing ~ -11 deg, pump
cap at ~ +12 deg), the griddle spawns on the floor on a sampled polar spot at bearing
-45 +/- 12 deg, radius 0.42-0.50 m, the moka pot mirrored on the + side. The
stove/boss/cap mechanism is deliberately FIXED at its authored pose: on this PhysX
stack a per-episode teleport of a jointed pair is unreliable (the joint frame stays
anchored at the authored pose), so the mechanism never moves and the free bodies plus
the required stroke count randomize instead.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawner: the griddle (disc + center knob post) ---------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _set_op(xformable, op_type, add_fn, value):
    """Set an xform op, creating it only if absent. Define() returns the EXISTING
    prim on re-authoring, but AddXxxOp() raises on a duplicate op ('xformOp:...
    already exists in xformOpOrder', the deterministic engine-relaunch boot
    failure of 2026-08-04) -- so every authoring path must reuse the op when the
    prim already carries one."""
    op = next((o for o in xformable.GetOrderedXformOps()
               if o.GetOpType() == op_type), None)
    (op if op is not None else add_fn()).Set(value)


def _spawn_griddle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a flat cast-iron disc with a square steel knob post standing at its
    center — the pinch affordance (CoM directly below the pinch). Idempotent authoring;
    clone() replicates per env."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        _set_op(xf, UsdGeom.XformOp.TypeTranslate, xf.AddTranslateOp,
                Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        _set_op(xf, UsdGeom.XformOp.TypeOrient, xf.AddOrientOp,
                Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(0.5)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    # disc: root frame origin at the DISC CENTER (rest z = support top + disc_t/2)
    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(cfg.disc_r)
    disc.CreateHeightAttr(cfg.disc_t)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.disc_r, -cfg.disc_r, -cfg.disc_t / 2),
                           Gf.Vec3f(cfg.disc_r, cfg.disc_r, cfg.disc_t / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    collide(disc.GetPrim())

    # knob post: square section, standing on the disc top
    knob = UsdGeom.Cube.Define(stage, f"{prim_path}/knob")
    knob.CreateSizeAttr(1.0)
    kxf = UsdGeom.Xformable(knob.GetPrim())
    _set_op(kxf, UsdGeom.XformOp.TypeTranslate, kxf.AddTranslateOp,
            Gf.Vec3d(0.0, 0.0, cfg.disc_t / 2 + cfg.knob_h / 2 - 0.002))
    _set_op(kxf, UsdGeom.XformOp.TypeScale, kxf.AddScaleOp,
            Gf.Vec3f(cfg.knob_t, cfg.knob_t, cfg.knob_h + 0.004))
    knob.CreateDisplayColorAttr([Gf.Vec3f(*cfg.knob_color)])
    collide(knob.GetPrim())
    return root


def _griddle_spawner_cfg(*, disc_r: float, disc_t: float, knob_t: float, knob_h: float,
                         mass: float, color: tuple, knob_color: tuple,
                         contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "griddle" not in _SPAWNER_CACHE:

        @configclass
        class GriddleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_griddle)
            disc_r: float = 0.070
            disc_t: float = 0.014
            knob_t: float = 0.024
            knob_h: float = 0.048
            color: tuple = (0.06, 0.06, 0.07)
            knob_color: tuple = (0.62, 0.64, 0.68)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["griddle"] = GriddleSpawnerCfg

    return _SPAWNER_CACHE["griddle"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        disc_r=disc_r, disc_t=disc_t, knob_t=knob_t, knob_h=knob_h,
        color=color, knob_color=knob_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PrimerStoveSceneCfg(BaseCfg):
    """Config for `PrimerStoveScene`. Spring constants sized for 120 Hz stability
    (m=0.08 kg, k=80 N/m — the weighbridge/microwave-button precedent)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    on_burner_r: float = tunable(0.045)  # griddle center within this of the burner axis (m)
    on_burner_z_tol: float = tunable(0.012)  # |griddle rest z - nominal rest z| below this (m)
    tilt_max_deg: float = tunable(10.0)  # griddle upright gate
    settle_speed: float = tunable(0.05)  # max |v| when judging the griddle (m/s)
    hold_steps: int = tunable(90)  # consecutive substeps success must be sustained
    lift_h: float = tunable(0.060)  # griddle bottom above this height latches "lifted"
    down_frac: float = tunable(0.80)  # stroke DOWN edge: depth >= down_frac * travel
    up_frac: float = tunable(0.20)  # stroke UP edge: depth <= up_frac * travel

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    n_strokes_lo: int = tunable(2)  # required full strokes sampled in [lo, hi]
    n_strokes_hi: int = tunable(4)
    pan_bearing_deg: tuple = tunable((-57.0, -33.0))  # griddle spawn: polar from the anchor
    pan_radius: tuple = tunable((0.42, 0.50))
    moka_bearing_deg: tuple = tunable((33.0, 57.0))  # moka pot spawn: mirrored polar band
    moka_radius: tuple = tunable((0.42, 0.50))
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per free body at reset

    # --- tunable: placement (laid out for a Franka base at stage_anchor) ---------------------
    stage_anchor: tuple = tunable((-0.50, 0.0, 0.0))  # the intended robot base pose
    stove_pos: tuple = tunable((0.05, 0.0))  # stove slab center on the ground
    burner_off_y: float = tunable(-0.10)  # burner axis offset along the slab (body frame)
    boss_off_y: float = tunable(0.115)  # pump boss offset along the slab (body frame)

    # --- info: structure ----------------------------------------------------------------------
    body_size: tuple = info((0.18, 0.34, 0.05))  # kinematic stove slab (x, y, z)
    body_color: tuple = info((0.13, 0.28, 0.16))  # dark green camp-stove enamel
    burner_r: float = info(0.065)  # kinematic burner disc
    burner_h: float = info(0.014)
    burner_color: tuple = info((0.16, 0.16, 0.18))  # charcoal
    boss_size: tuple = info((0.07, 0.07, 0.05))  # kinematic pump boss under the cap
    boss_color: tuple = info((0.35, 0.36, 0.40))
    cap_size: tuple = info((0.056, 0.056, 0.024))  # the dynamic pump cap (bright red)
    cap_color: tuple = info((0.85, 0.12, 0.10))
    cap_mass: float = info(0.08)
    travel: float = info(0.035)  # cap vertical travel, home -> bottomed (m)
    spring_k: float = info(80.0)  # spring return (N/m); full-depth force = 2.8 N
    spring_c: float = info(4.0)  # damping (N*s/m), ~0.7 critical for the free cap
    pan_disc_r: float = info(0.070)  # the griddle
    pan_disc_t: float = info(0.014)
    pan_knob_t: float = info(0.024)  # square knob post (the proven 24 mm pinch)
    pan_knob_h: float = info(0.048)
    pan_mass: float = info(0.30)
    pan_color: tuple = info((0.06, 0.06, 0.07))
    pan_knob_color: tuple = info((0.62, 0.64, 0.68))
    moka_r: float = info(0.030)  # the seed's distractor, procedural stand-in
    moka_h: float = info(0.105)
    moka_mass: float = info(0.35)
    moka_color: tuple = info((0.72, 0.73, 0.76))
    lamp_r: float = info(0.008)  # stroke-progress lamps on the slab top
    lamp_x: float = info(-0.065)  # lamp row, slab body frame (front edge, robot side)
    lamp_y0: float = info(0.040)
    lamp_dy: float = info(0.030)
    lamp_needed: tuple = info((0.45, 0.06, 0.05))  # dark red: stroke still needed
    lamp_done: tuple = info((1.0, 0.72, 0.10))  # amber: stroke done
    lamp_unused: tuple = info((0.12, 0.12, 0.13))  # dark gray: slot unused this episode
    flame_dim: tuple = info((0.22, 0.23, 0.27))  # burner flame ring, unlit
    flame_lit: tuple = info((1.0, 0.42, 0.08))  # ... and primed (glows orange)
    contact_offset: float = info(0.004)

    # Derived (filled in __post_init__).
    cap_home_lz: float = field(default=None, init=False)  # cap center z above ground, home
    down_depth: float = field(default=None, init=False)  # stroke DOWN edge depth (m)
    up_depth: float = field(default=None, init=False)  # stroke UP edge depth (m)
    pan_rest_lz: float = field(default=None, init=False)  # griddle center z resting on burner

    def __post_init__(self) -> None:
        self.cap_home_lz = round(
            self.body_size[2] + self.boss_size[2] + self.travel + self.cap_size[2] / 2, 4)
        self.down_depth = round(self.down_frac * self.travel, 4)
        self.up_depth = round(self.up_frac * self.travel, 4)
        self.pan_rest_lz = round(self.body_size[2] + self.burner_h + self.pan_disc_t / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("primer_stove")
class PrimerStoveScene(BaseScene):
    cfg: PrimerStoveSceneCfg

    def __init__(self, cfg: PrimerStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or PrimerStoveSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic stove pieces (slab + burner + pump boss), the
        dynamic pump cap at home, and the free bodies at nominal spots (reset() re-places
        the free bodies and re-samples the stroke requirement)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sx, sy = c.stove_pos
        ax, ay, _az = c.stage_anchor

        def polar(bearing_deg: float, radius: float) -> tuple:
            a = math.radians(bearing_deg)
            return (ax + radius * math.cos(a), ay + radius * math.sin(a))

        pan_xy = polar(sum(c.pan_bearing_deg) / 2, sum(c.pan_radius) / 2)
        moka_xy = polar(sum(c.moka_bearing_deg) / 2, sum(c.moka_radius) / 2)

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
            "body": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Body",
                spawn=sim_utils.CuboidCfg(
                    size=c.body_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.body_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, c.body_size[2] / 2)),
            ),
            "burner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Burner",
                spawn=sim_utils.CylinderCfg(
                    radius=c.burner_r,
                    height=c.burner_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.burner_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy + c.burner_off_y, c.body_size[2] + c.burner_h / 2)),
            ),
            "boss": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Boss",
                spawn=sim_utils.CuboidCfg(
                    size=c.boss_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.boss_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy + c.boss_off_y, c.body_size[2] + c.boss_size[2] / 2)),
            ),
            # Gravity is DISABLED on the cap: the post_step spring about `home` then makes
            # home the exact rest pose (the preload is the disabled weight).
            "cap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cap",
                spawn=sim_utils.CuboidCfg(
                    size=c.cap_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True, max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cap_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cap_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy + c.boss_off_y, c.cap_home_lz)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Griddle",
                spawn=_griddle_spawner_cfg(
                    disc_r=c.pan_disc_r, disc_t=c.pan_disc_t, knob_t=c.pan_knob_t,
                    knob_h=c.pan_knob_h, mass=c.pan_mass, color=c.pan_color,
                    knob_color=c.pan_knob_color, contact_offset=0.002),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(pan_xy[0], pan_xy[1], c.pan_disc_t / 2 + 0.003)),
            ),
            "moka": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Moka",
                spawn=sim_utils.CylinderCfg(
                    radius=c.moka_r,
                    height=c.moka_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.moka_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.moka_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(moka_xy[0], moka_xy[1], c.moka_h / 2 + 0.003)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, author the cap slide joint, build lamps + flame ring, allocate
        the counters/latches."""
        super().bind(env)
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.body: RigidObject = env.iscene["body"]
        self.burner: RigidObject = env.iscene["burner"]
        self.boss: RigidObject = env.iscene["boss"]
        self.cap: RigidObject = env.iscene["cap"]
        self.pan: RigidObject = env.iscene["pan"]
        self.moka: RigidObject = env.iscene["moka"]
        self.env_origins = env.iscene.env_origins
        # Cap home z is a constant of the flat world (the mechanism never moves).
        self._home_z = self.env_origins[:, 2] + c.cap_home_lz
        # Stroke counter state + latches.
        self._n_req = torch.full((n,), c.n_strokes_lo, dtype=torch.long, device=dev)
        self._strokes = torch.zeros(n, dtype=torch.long, device=dev)
        self._down = torch.zeros(n, dtype=torch.bool, device=dev)
        self._primed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pan_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pan_placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._both = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)
        self._author_joints()
        self._build_indicators()

    def _author_joints(self) -> None:
        """Per env: the cap's Z prismatic slide on the pump boss — pair collision disabled
        (the joint limit is the mechanical stop), SYMMETRIC limits +/- travel."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/cap_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Boss"])
            j.CreateBody1Rel().SetTargets([f"{base}/Cap"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(
                0.0, 0.0, c.boss_size[2] / 2 + c.travel + c.cap_size[2] / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.travel)
            j.CreateUpperLimitAttr(c.travel)
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
            if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac Sim 5.1 schema
                lim.CreateContactDistanceAttr(0.001)

    def _build_indicators(self) -> None:
        """Visual children (no collision): 4 stroke lamps on the slab top near the pump
        (the lamp row also displays this episode's required stroke count) and a flame
        ring on the burner that glows orange once primed."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        self._lamps: list[list] = []
        self._flames: list = []
        for i in range(self.env.num_envs):
            row = []
            for k in range(c.n_strokes_hi):
                lamp = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{i}/Body/lamp_{k}")
                lamp.CreateRadiusAttr(c.lamp_r)
                lxf = UsdGeom.Xformable(lamp.GetPrim())
                _set_op(lxf, UsdGeom.XformOp.TypeTranslate, lxf.AddTranslateOp,
                        Gf.Vec3d(c.lamp_x, c.lamp_y0 + k * c.lamp_dy,
                                 c.body_size[2] / 2 + c.lamp_r / 2))
                lamp.CreateDisplayColorAttr([Gf.Vec3f(*c.lamp_unused)])
                row.append(lamp)
            self._lamps.append(row)
            ring = UsdGeom.Cylinder.Define(stage, f"/World/envs/env_{i}/Burner/flame")
            ring.CreateRadiusAttr(c.burner_r - 0.016)
            ring.CreateHeightAttr(0.004)
            r = c.burner_r - 0.016
            ring.CreateExtentAttr([Gf.Vec3f(-r, -r, -0.002), Gf.Vec3f(r, r, 0.002)])
            rxf = UsdGeom.Xformable(ring.GetPrim())
            _set_op(rxf, UsdGeom.XformOp.TypeTranslate, rxf.AddTranslateOp,
                    Gf.Vec3d(0.0, 0.0, c.burner_h / 2 + 0.002))
            ring.CreateDisplayColorAttr([Gf.Vec3f(*c.flame_dim)])
            self._flames.append(ring)
        self._lamp_state = [(-1, False)] * self.env.num_envs  # (shown strokes, shown primed)

    def _refresh_indicators(self, force: bool = False) -> None:
        from pxr import Gf

        c = self.cfg
        for e in range(self.env.num_envs):
            got = int(self._strokes[e].clamp(max=c.n_strokes_hi))
            lit = bool(self._primed[e])
            if not force and self._lamp_state[e] == (got, lit):
                continue
            self._lamp_state[e] = (got, lit)
            need = int(self._n_req[e])
            for k, lamp in enumerate(self._lamps[e]):
                if k < min(got, need):
                    col = c.lamp_done
                elif k < need:
                    col = c.lamp_needed
                else:
                    col = c.lamp_unused
                lamp.GetDisplayColorAttr().Set([Gf.Vec3f(*col)])
            self._flames[e].GetDisplayColorAttr().Set(
                [Gf.Vec3f(*(c.flame_lit if lit else c.flame_dim))])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the stove mechanism re-pinned at its FIXED authored pose (a
        jointed pair must never teleport on this stack), free bodies re-sampled on their
        polar bands, the stroke requirement re-sampled, counters/latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        sx, sy = c.stove_pos

        def pin(obj, lx: float, ly: float, lz: float) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = lx
            st[:, 1] = ly
            st[:, 2] = lz
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        pin(self.body, sx, sy, c.body_size[2] / 2)
        pin(self.burner, sx, sy + c.burner_off_y, c.body_size[2] + c.burner_h / 2)
        pin(self.boss, sx, sy + c.boss_off_y, c.body_size[2] + c.boss_size[2] / 2)
        pin(self.cap, sx, sy + c.boss_off_y, c.cap_home_lz)

        def scatter(obj, bearing: tuple, radius: tuple, lz: float) -> None:
            ax, ay, _az = c.stage_anchor
            b0, b1 = (math.radians(v) for v in bearing)
            ang = b0 + torch.rand(m, device=dev) * (b1 - b0)
            rad = radius[0] + torch.rand(m, device=dev) * (radius[1] - radius[0])
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = ax + rad * torch.cos(ang)
            st[:, 1] = ay + rad * torch.sin(ang)
            st[:, 2] = lz
            yhalf = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.reset_yaw_deg) / 2
            st[:, 3] = torch.cos(yhalf)
            st[:, 6] = torch.sin(yhalf)
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        scatter(self.pan, c.pan_bearing_deg, c.pan_radius, c.pan_disc_t / 2 + 0.003)
        scatter(self.moka, c.moka_bearing_deg, c.moka_radius, c.moka_h / 2 + 0.003)

        self._n_req[env_ids] = torch.randint(
            c.n_strokes_lo, c.n_strokes_hi + 1, (m,), device=dev)
        self._strokes[env_ids] = 0
        self._down[env_ids] = False
        self._primed[env_ids] = False
        self._pan_lifted[env_ids] = False
        self._pan_placed[env_ids] = False
        self._both[env_ids] = False
        self._hold[env_ids] = 0
        # Zero the cap's external-force buffer: a stale spring force from the previous
        # episode would kick the gravity-free cap for one substep before post_step runs.
        n = self.env.num_envs
        self.cap.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))
        self._refresh_indicators(force=True)

    # ----- geometry queries ---------------------------------------------------------------------
    def pump_depth(self) -> torch.Tensor:
        """(N,) cap depression below home (m), 0 = fully up."""
        return (self._home_z - self.cap.data.root_pos_w[:, 2]).clamp(min=0.0)

    def pan_on_burner_geom(self) -> torch.Tensor:
        """(N,) bool, geometric: griddle centered on the burner axis, resting at the
        burner top, upright (settledness judged separately)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pp = self.pan.data.root_pos_w
        bp = self.burner.data.root_pos_w
        near = (pp[:, :2] - bp[:, :2]).norm(dim=-1) < c.on_burner_r
        rest_z = self.env_origins[:, 2] + c.pan_rest_lz
        z_ok = (pp[:, 2] - rest_z).abs() < c.on_burner_z_tol
        ez = torch.tensor([0.0, 0.0, 1.0], device=pp.device).expand(pp.shape[0], 3)
        up = quat_apply(self.pan.data.root_quat_w, ez)[:, 2] > math.cos(
            math.radians(c.tilt_max_deg))
        return near & z_ok & up

    def pan_on_burner_now(self) -> torch.Tensor:
        """(N,) bool, instantaneous: on the burner AND settled."""
        still = self.pan.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return self.pan_on_burner_geom() & still

    def primed(self) -> torch.Tensor:
        """(N,) bool: the burner is lit (permanent episode latch)."""
        return self._primed.clone()

    def strokes(self) -> torch.Tensor:
        """(N,) long: completed full pump strokes so far."""
        return self._strokes.clone()

    def n_required(self) -> torch.Tensor:
        """(N,) long: full strokes this episode needs to prime the stove."""
        return self._n_req.clone()

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        # Spring return about home (gravity is disabled on the cap, so home is the exact
        # rest pose): f_z = k*(home - z) - c*v_z.
        d = self._home_z - self.cap.data.root_pos_w[:, 2]
        v_z = self.cap.data.root_lin_vel_w[:, 2]
        f_z = c.spring_k * d - c.spring_c * v_z
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        self.cap.set_external_force_and_torque(
            f_z.view(n, 1, 1) * ez.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

        # Full-hysteresis stroke counter: DOWN edge at >= down_depth, then the stroke
        # counts only on the UP edge at <= up_depth. Press-and-hold or partial jiggling
        # never counts.
        depth = self.pump_depth()
        down_now = depth >= c.down_depth
        up_now = depth <= c.up_depth
        counted = self._down & up_now
        self._strokes = self._strokes + counted.long()
        self._down = (self._down | down_now) & ~up_now
        self._primed = self._primed | (self._strokes >= self._n_req)

        # Griddle latches + the sustained-success counter.
        pan_bottom = (self.pan.data.root_pos_w[:, 2] - self.env_origins[:, 2]
                      - c.pan_disc_t / 2)
        self._pan_lifted |= pan_bottom > c.lift_h
        geom = self.pan_on_burner_geom()
        self._pan_placed |= geom
        self._both |= self._primed & geom
        now = self._primed & self.pan_on_burner_now()
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

        self._refresh_indicators()

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {k: getattr(self, k).data.root_state_w[env_ids].clone()
                  for k in ("body", "burner", "boss", "cap", "pan", "moka")}
        latches = {k: getattr(self, k)[env_ids].clone()
                   for k in ("_n_req", "_strokes", "_down", "_primed", "_pan_lifted",
                             "_pan_placed", "_both", "_hold")}
        return {"bodies": bodies, "latches": latches}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, v in state["bodies"].items():
            getattr(self, k).write_root_state_to_sim(v, env_ids)
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v
        self._refresh_indicators(force=True)

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-green camp stove slab ({c.body_size[0] * 100:.0f} x "
            f"{c.body_size[1] * 100:.0f} cm, {c.body_size[2] * 100:.0f} cm tall) is fixed "
            f"on the floor. At one end of the slab sits the charcoal-gray BURNER disc "
            f"({c.burner_r * 200:.0f} cm across) with a pale ring on top; at the other end "
            f"a gray pump boss carries a bright-red square PUMP CAP that can be pressed "
            f"straight down {c.travel * 1000:.0f} mm and springs back up when released. "
            f"A row of small indicator lamps on the slab top beside the pump shows the "
            f"priming state: each DARK-RED dot is one full pump stroke still needed this "
            f"episode ({c.n_strokes_lo}-{c.n_strokes_hi} depending on the episode), a dot "
            f"turns AMBER when its stroke is done, and dark-gray dots are unused. On the "
            f"floor nearby lie a black cast-iron GRIDDLE (a {c.pan_disc_r * 200:.0f} cm "
            f"disc with a steel center knob handle — grip the knob to carry it) and a "
            f"silver moka pot, which is a distractor and belongs nowhere.\n"
            f"Goal: light the burner, and rest the griddle centered on it. The stove has "
            f"NO knob: it lights only when the fuel line is pressure-primed by FULL pump "
            f"strokes — press the red cap all the way down (at least "
            f"{c.down_depth * 1000:.0f} mm of its {c.travel * 1000:.0f} mm travel) and let "
            f"it return all the way up; repeat until every dark-red lamp has turned amber "
            f"and the ring on the burner glows ORANGE (the stove then stays lit). Pressing "
            f"and holding the cap does nothing (the stroke only counts on the full "
            f"release), and shallow partial pumps never count. Then (or before — no "
            f"execution order is required) place the griddle flat on the burner disc, its "
            f"center within {c.on_burner_r * 100:.1f} cm of the burner center, upright and "
            f"settled, and leave it there. Success = burner primed AND the griddle "
            f"resting centered on it, sustained with everything at rest."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: primed AND the griddle resting centered on the burner, sustained
        `hold_steps` consecutive substeps."""
        return self._hold >= self.cfg.hold_steps

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1], latched partial credit that never evaporates:
        strokes/N * 0.35 (the counter is monotone), 0.12 once the griddle was lifted,
        0.25 once the griddle rested on the burner (primed or not — the seed-strategy
        cap), 0.45 once primed, 0.85 once primed and on-burner coexisted, 1.0 iff
        success()."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        frac = torch.minimum(self._strokes, self._n_req).float() / self._n_req.float()
        s = 0.35 * frac
        s = torch.where(self._pan_lifted, torch.maximum(s, torch.full_like(s, 0.12)), s)
        s = torch.where(self._pan_placed, torch.maximum(s, torch.full_like(s, 0.25)), s)
        s = torch.where(self._primed, torch.maximum(s, torch.full_like(s, 0.45)), s)
        s = torch.where(self._both, torch.maximum(s, torch.full_like(s, 0.85)), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="primer_stove", robot="null", env_spacing=3))
