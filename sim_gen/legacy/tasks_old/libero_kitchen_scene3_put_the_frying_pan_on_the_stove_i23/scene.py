"""SeesawStoveScene — level a tilting cooktop: pan on the commanded burner, moka pot as
counterweight at the correct lever arm (seed: libero_90 scene3 put-the-frying-pan-on-the-
stove, strategically inverted).

The seed is a single pick-and-place: grasp the frying pan, set it on a rigid stove, done —
the distractor moka pot is never touched and the support surface is inert. Here the stove
top is NOT rigid: it is a see-saw cooktop, a long board on a central sprung hinge (weak
centering spring, +/-12 deg travel limits). The frying pan on a burner tips the board onto
its limit stop — the seed's own plan ends with a visibly capsized cooktop and never scores
past the placement stage. To succeed the solver must ALSO use the moka pot: park it on the
counterweight rail on the opposite half of the board at the lever arm that cancels the
pan's torque (arm = burner_x * m_pan / m_pot against a weak spring, so a ~+/-4 cm band),
bringing the cooktop level within `level_tol_deg`. Which of the three burners is
commanded is sampled per episode and marked by an orange indicator post standing beside
the board, so the required counterweight arm changes episode to episode — static
equilibrium reasoning (continuous lever-arm placement), not a memorized drop point.

Strategy axes claimed: static-equilibrium / torque balance with a CONTINUOUS placement
variable, plus commanded-goal selection (3 burners). Deliberately distinct from sibling
tasks: no structure is built and nothing is stacked (i11 builds towers), no weight is
CHOSEN among distractors (i6 selects the heavy block — here both objects are essential
and the variable is WHERE, not WHICH), no confinement or sliding (i4/i9), no pouring
(i5/i2), no knobs (i7/i21's stove controls are untouched — our stove has no controls).

Mechanism (all procedural primitives; the jointed-mechanism crib proven in
push_button_i6 / microwave_meal): a kinematic pedestal, a dynamic board joined to it by a
Y revolute joint (pair collision irrelevant — 25 mm air gap; SYMMETRIC +/-12 deg limits,
the GPU sign-convention hedge), gravity disabled on the board (its CoM is pinned at the
hinge anyway, so disabling kills any collider-approximation CoM bias) and a post_step
angular spring about the hinge axis, applied as a BODY-LOCAL +y torque (the hinge axis is
exactly board-local y): tau = -k*theta - c*omega. Physics of the task is honest: the pan
and pot rest on the board by contact, their weights load the hinge through real contact
forces, and the judged tilt is the settled outcome. Propping the board level from below
is impossible by construction: the board floats 165 mm above the ground and the tallest
loose object (the pot, 110 mm) cannot reach it anywhere reachable (needs x >= 0.9 m).

Judged on PHYSICAL outcome only, pan/pot poses expressed in the BOARD'S BODY FRAME (a
tilted board judges its riders consistently): success = pan seated flat on the COMMANDED
burner (centre within `xy_tol`, resting on the surface, upright w.r.t. the board) AND the
pot anywhere on the counterweight rail AND the board level within `level_tol_deg` AND
everything settled. score() in [0,1]: 0.15 latched once the pan has ridden the board
anywhere, 0.40 latched once the pan has seated on the commanded burner, 0.55 latched once
both are placed (pan on burner + pot on rail, any order), 0.55..0.90 live scaling as the
board approaches level while both remain placed, 1.0 iff success. Balancing the WRONG
burner (level board, pan on an uncommanded burner) stays at 0.15 — the goal is commanded,
not merely "level".

Per-episode randomization: commanded burner index (1 of 3, indicator teleported to
match), stove pedestal xy jitter + yaw (board follows rigidly — the jointed-pair reset
teleport), pan and pot spawn xy jitter + free yaw on the far side of the workspace.

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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, child colliders + visual-only decoration, authored with raw pxr
# APIs; `isaaclab.sim.utils.clone` provides the per-env replicate machinery (the pen_holder
# pattern).

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_body(root, mass: float, com=(0.0, 0.0, 0.0), disable_gravity: bool = False):
    """RigidBodyAPI + explicit MassAPI (mass AND centre of mass — the balance arithmetic
    must not depend on collider-volume approximations) + the depenetration cap."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if disable_gravity:
        px.CreateDisableGravityAttr(True)


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _xform_root(stage, prim_path: str, translation, orientation):
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return xform.GetPrim()


def _spawn_board(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The see-saw cooktop board: slab collider + two symmetric end curbs (colliders, so
    nothing slides off at the tilt limit) + VISUAL-ONLY decoration (3 burner discs on the
    +x half, a pale counterweight-rail stripe on the -x half — no CollisionAPI, so they
    add neither contact bumps nor inertia). Explicit mass + CoM at the hinge."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation)
    _apply_body(root, cfg.mass_props.mass, com=(0.0, 0.0, 0.0), disable_gravity=True)

    bx, by, bt = cfg.size
    slab = UsdGeom.Cube.Define(stage, f"{prim_path}/slab")
    slab.CreateSizeAttr(1.0)
    UsdGeom.Xformable(slab.GetPrim()).AddScaleOp().Set(Gf.Vec3f(bx, by, bt))
    slab.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(slab.GetPrim(), cfg.contact_offset)

    for k, sx in enumerate((-1.0, 1.0)):
        curb = UsdGeom.Cube.Define(stage, f"{prim_path}/curb_{k}")
        curb.CreateSizeAttr(1.0)
        cx = UsdGeom.Xformable(curb.GetPrim())
        cx.AddTranslateOp().Set(Gf.Vec3d(sx * cfg.curb_x, 0.0, bt / 2 + cfg.curb_h / 2))
        cx.AddScaleOp().Set(Gf.Vec3f(cfg.curb_t, by, cfg.curb_h))
        curb.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        _collide(curb.GetPrim(), cfg.contact_offset)

    for k, xb in enumerate(cfg.burner_xs):  # visual only — the burner rings
        disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/burner_{k}")
        disc.CreateRadiusAttr(cfg.burner_r)
        disc.CreateHeightAttr(0.004)
        disc.CreateExtentAttr([Gf.Vec3f(-cfg.burner_r, -cfg.burner_r, -0.002),
                               Gf.Vec3f(cfg.burner_r, cfg.burner_r, 0.002)])
        UsdGeom.Xformable(disc.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(xb), 0.0, bt / 2 + 0.002))
        disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.burner_color)])

    rail = UsdGeom.Cube.Define(stage, f"{prim_path}/rail")  # visual only — the rail lane
    rail.CreateSizeAttr(1.0)
    rx = UsdGeom.Xformable(rail.GetPrim())
    rx.AddTranslateOp().Set(Gf.Vec3d(cfg.rail_mid, 0.0, bt / 2 + 0.001))
    rx.AddScaleOp().Set(Gf.Vec3f(cfg.rail_len, 0.10, 0.002))
    rail.CreateDisplayColorAttr([Gf.Vec3f(*cfg.rail_color)])
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The frying pan: a flat disc collider plus a handle box collider extending along
    local +x (handle is 1 mm shy of the disc's bottom face, so the pan rests on the disc
    only). Explicit CoM at the DISC CENTRE — the handle is declared lightweight, keeping
    the counterweight arithmetic exact (arm = burner_x * m_pan / m_pot)."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation)
    _apply_body(root, cfg.mass_props.mass, com=(0.0, 0.0, 0.0))

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(cfg.disc_r)
    disc.CreateHeightAttr(cfg.disc_t)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.disc_r, -cfg.disc_r, -cfg.disc_t / 2),
                           Gf.Vec3f(cfg.disc_r, cfg.disc_r, cfg.disc_t / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(disc.GetPrim(), cfg.contact_offset)

    hl, hw, ht = cfg.handle_size
    handle = UsdGeom.Cube.Define(stage, f"{prim_path}/handle")
    handle.CreateSizeAttr(1.0)
    hx = UsdGeom.Xformable(handle.GetPrim())
    hx.AddTranslateOp().Set(Gf.Vec3d(cfg.disc_r + hl / 2 - 0.005, 0.0, 0.001))
    hx.AddScaleOp().Set(Gf.Vec3f(hl, hw, ht))
    handle.CreateDisplayColorAttr([Gf.Vec3f(*cfg.handle_color)])
    _collide(handle.GetPrim(), cfg.contact_offset)

    rim = UsdGeom.Cylinder.Define(stage, f"{prim_path}/rim")  # visual only — pan lip
    rim.CreateRadiusAttr(cfg.disc_r - 0.006)
    rim.CreateHeightAttr(0.004)
    rim.CreateExtentAttr([Gf.Vec3f(-cfg.disc_r, -cfg.disc_r, -0.002),
                          Gf.Vec3f(cfg.disc_r, cfg.disc_r, 0.002)])
    UsdGeom.Xformable(rim.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, cfg.disc_t / 2 + 0.002))
    rim.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.30, 0.32)])
    return root


def _board_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "board" not in _SPAWNER_CACHE:

        @configclass
        class BoardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_board)
            size: tuple = (0.72, 0.26, 0.02)
            curb_x: float = 0.35
            curb_t: float = 0.02
            curb_h: float = 0.03
            burner_xs: tuple = (0.11, 0.165, 0.22)
            burner_r: float = 0.055
            rail_mid: float = -0.19
            rail_len: float = 0.26
            color: tuple = (0.42, 0.40, 0.38)
            burner_color: tuple = (0.10, 0.10, 0.12)
            rail_color: tuple = (0.78, 0.74, 0.62)
            contact_offset: float = 0.004

        _SPAWNER_CACHE["board"] = BoardSpawnerCfg

    return _SPAWNER_CACHE["board"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.board_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        size=c.board_size, curb_x=c.curb_x, curb_t=c.curb_t, curb_h=c.curb_h,
        burner_xs=c.burner_xs, burner_r=c.burner_r,
        rail_mid=(c.rail_x[0] + c.rail_x[1]) / 2, rail_len=c.rail_x[1] - c.rail_x[0],
        color=c.board_color, burner_color=c.burner_color, rail_color=c.rail_color,
        contact_offset=c.contact_offset,
    )


def _pan_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pan" not in _SPAWNER_CACHE:

        @configclass
        class PanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            disc_r: float = 0.075
            disc_t: float = 0.016
            handle_size: tuple = (0.11, 0.022, 0.014)
            color: tuple = (0.10, 0.10, 0.11)
            handle_color: tuple = (0.16, 0.15, 0.14)
            contact_offset: float = 0.004

        _SPAWNER_CACHE["pan"] = PanSpawnerCfg

    return _SPAWNER_CACHE["pan"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.pan_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        disc_r=c.pan_r, disc_t=c.pan_t, handle_size=c.handle_size,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SeesawStoveSceneCfg(BaseCfg):
    """Config for `SeesawStoveScene`. Spring sized for 120 Hz stability (loaded inertia
    ~0.06 kg m^2, k = 2 N m/rad -> ~0.9 Hz, ~130 substeps/period; damping ~0.7 critical).
    Balance band: pot-arm error tolerance = k * level_tol / (m_pot * g) ~= +/-41 mm."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    xy_tol: float = tunable(0.030)  # pan centre within this of the commanded burner (board xy)
    level_tol_deg: float = tunable(3.5)  # board counts as level within this tilt
    pan_upright_max_deg: float = tunable(15.0)  # pan axis within this of the board normal
    settle_speed: float = tunable(0.05)  # max |lin vel| (pan, pot) when judging (m/s)
    board_settle_w: float = tunable(0.08)  # max board |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = tunable(0.05)  # uniform +/- xy jitter (pan, pot spawn) at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per loose body at reset
    ped_jitter: float = tunable(0.03)  # stove pedestal xy jitter (board follows rigidly)
    ped_yaw_deg: float = tunable(20.0)  # stove pedestal yaw range (deg)

    # --- tunable: placement ------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    ped_pos: tuple = tunable((0.30, 0.0))  # stove pedestal centre
    pan_center: tuple = tunable((-0.14, 0.16))  # pan spawn centre (far side of the workspace)
    pot_center: tuple = tunable((-0.14, -0.16))  # pot spawn centre

    # --- info: structure ----------------------------------------------------------------------
    ped_size: tuple = info((0.12, 0.24, 0.14))  # kinematic pedestal (narrow in x: tilt clearance)
    board_size: tuple = info((0.72, 0.26, 0.02))  # the see-saw cooktop board
    board_gap: float = info(0.025)  # air gap board-bottom to pedestal-top (25 mm >> offsets)
    tilt_limit_deg: float = info(12.0)  # revolute joint limits, symmetric
    curb_x: float = info(0.35)  # end-curb centre |x|
    curb_t: float = info(0.02)
    curb_h: float = info(0.03)  # curb rises above the slab top — keeps riders on at the stop
    burner_xs: tuple = info((0.11, 0.165, 0.22))  # burner centres, board frame +x half
    burner_r: float = info(0.055)
    rail_x: tuple = info((-0.32, -0.06))  # visual rail stripe span (board frame x)
    rail_zone: tuple = info((-0.335, -0.04))  # judged on-rail x band (board frame)
    rail_half_w: float = info(0.09)  # judged on-rail |y| band
    board_mass: float = info(0.60)
    spring_k: float = info(2.0)  # centering spring (N m / rad)
    spring_c: float = info(0.5)  # hinge damping (N m s / rad), ~0.7 critical loaded
    pan_r: float = info(0.075)  # pan disc radius
    pan_t: float = info(0.016)  # pan disc thickness
    handle_size: tuple = info((0.11, 0.022, 0.014))
    pan_mass: float = info(0.36)
    pot_r: float = info(0.032)  # moka pot (counterweight) radius
    pot_h: float = info(0.11)
    pot_mass: float = info(0.30)
    ind_size: tuple = info((0.028, 0.028, 0.15))  # commanded-burner indicator post
    ind_y_off: float = info(0.19)  # indicator stands this far off the board centreline (-y)
    board_color: tuple = info((0.42, 0.40, 0.38))
    burner_color: tuple = info((0.10, 0.10, 0.12))
    rail_color: tuple = info((0.78, 0.74, 0.62))
    ped_color: tuple = info((0.26, 0.26, 0.30))
    pot_color: tuple = info((0.62, 0.65, 0.72))
    ind_color: tuple = info((0.95, 0.35, 0.10))
    contact_offset: float = info(0.004)
    gravity: float = info(9.81)

    # Derived (filled in __post_init__).
    board_lz: float = field(default=None, init=False)  # board centre height above surface
    mass_ratio: float = field(default=None, init=False)  # m_pan / m_pot = arm / burner_x
    arm_band: float = field(default=None, init=False)  # pot-arm error that stays level (m)

    def __post_init__(self) -> None:
        self.board_lz = round(self.ped_size[2] + self.board_gap + self.board_size[2] / 2, 4)
        self.mass_ratio = round(self.pan_mass / self.pot_mass, 4)
        self.arm_band = round(
            self.spring_k * math.radians(self.level_tol_deg) / (self.pot_mass * self.gravity), 4)

    def exact_arm(self, burner_idx: int) -> float:
        """Board-frame |x| on the rail that exactly cancels a centred pan on burner
        `burner_idx` (spring residual -> 0)."""
        return self.burner_xs[burner_idx] * self.mass_ratio


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("seesaw_stove")
class SeesawStoveScene(BaseScene):
    cfg: SeesawStoveSceneCfg

    def __init__(self, cfg: SeesawStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or SeesawStoveSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic pedestal, the hinged board (level), the commanded-burner
        indicator post, and the pan/pot at nominal spawn slots (reset() re-places all)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=c.ped_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ped_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_pos[0], c.ped_pos[1], z0 + c.ped_size[2] / 2)),
            ),
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=_board_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_pos[0], c.ped_pos[1], z0 + c.board_lz)),
            ),
            "indicator": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Indicator",
                spawn=sim_utils.CuboidCfg(
                    size=c.ind_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ind_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_pos[0] + c.burner_xs[1], c.ped_pos[1] - c.ind_y_off,
                         z0 + c.ind_size[2] / 2)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_pan_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pan_center[0], c.pan_center[1], z0 + c.pan_t / 2 + 0.003)),
            ),
            "pot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pot",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pot_r, height=c.pot_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pot_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pot_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_center[0], c.pot_center[1], z0 + c.pot_h / 2 + 0.003)),
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
        """Grab handles, author the hinge joints, allocate latches + the commanded burner."""
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.board: RigidObject = env.iscene["board"]
        self.indicator: RigidObject = env.iscene["indicator"]
        self.pan: RigidObject = env.iscene["pan"]
        self.pot: RigidObject = env.iscene["pot"]
        self.env_origins = env.iscene.env_origins
        self._active = torch.ones(n, dtype=torch.long, device=dev)  # commanded burner idx
        self._pan_on_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pan_burner_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._both_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the board's Y revolute hinge on the pedestal — SYMMETRIC limits
        +/- tilt_limit (degrees; the GPU sign-convention hedge). The spring/damping live
        in post_step as a body-local torque, not in a drive (no unit ambiguity)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        z01 = c.board_lz - c.ped_size[2] / 2  # board centre above pedestal centre
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/board_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Board"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, z01))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.tilt_limit_deg)
            j.CreateUpperLimitAttr(c.tilt_limit_deg)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the commanded burner; stove pedestal at ped_pos + jitter
        with a random yaw and the board riding LEVEL above it (common planar transform —
        the joint sees an unchanged relative pose); indicator post beside the commanded
        burner; pan and pot scattered flat on the far side; latches + force buffers
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        self._active[env_ids] = torch.randint(0, len(c.burner_xs), (m,), device=dev)

        # --- pedestal + board: common planar transform ---
        pxy = torch.zeros(m, 2, device=dev)
        pxy[:, 0] = c.ped_pos[0]
        pxy[:, 1] = c.ped_pos[1]
        pxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.ped_jitter
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ped_yaw_deg) / 2
        qw, qz = torch.cos(half), torch.sin(half)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = c.surface_z + c.ped_size[2] / 2
        st[:, 3], st[:, 6] = qw, qz
        st[:, 0:3] += origin
        self.pedestal.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = c.surface_z + c.board_lz
        st[:, 3], st[:, 6] = qw, qz
        st[:, 0:3] += origin
        self.board.write_root_state_to_sim(st, env_ids)

        # --- indicator: board-frame (burner_x, -ind_y_off) rotated by the pedestal yaw ---
        yaw = 2 * torch.atan2(qz, qw)
        bx = torch.tensor(c.burner_xs, device=dev)[self._active[env_ids]]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = pxy[:, 0] + bx * cy - (-c.ind_y_off) * sy
        st[:, 1] = pxy[:, 1] + bx * sy + (-c.ind_y_off) * cy
        st[:, 2] = c.surface_z + c.ind_size[2] / 2
        st[:, 3], st[:, 6] = qw, qz
        st[:, 0:3] += origin
        self.indicator.write_root_state_to_sim(st, env_ids)

        # --- pan + pot: spawn slots + jitter + free yaw, flat on the surface ---
        for body, center, hz in ((self.pan, c.pan_center, c.pan_t / 2 + 0.003),
                                 (self.pot, c.pot_center, c.pot_h / 2 + 0.003)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = center[0]
            st[:, 1] = center[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = c.surface_z + hz
            bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.reset_yaw_deg) / 2
            st[:, 3] = torch.cos(bhalf)
            st[:, 6] = torch.sin(bhalf)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._pan_on_ever[env_ids] = False
        self._pan_burner_ever[env_ids] = False
        self._both_ever[env_ids] = False
        # Zero the board's external-torque buffer (a stale spring torque from the previous
        # episode would kick the gravity-free board for one substep before post_step runs).
        n = self.env.num_envs
        self.board.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))

    # ----- geometry queries ---------------------------------------------------------------------
    def tilt(self) -> torch.Tensor:
        """(N,) signed board tilt (rad): + = burner side (+x) down. Rotation about the
        hinge sends board-local +x to world z-component -sin(theta), pedestal yaw does not
        touch the z component."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(self.env.num_envs, 3)
        x_w = quat_apply(self.board.data.root_quat_w, ex)
        return torch.asin((-x_w[:, 2]).clamp(-1.0, 1.0))

    def _to_board(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> board body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.board.data.root_quat_w,
                                  pos_w - self.board.data.root_pos_w)

    def pan_on_board(self) -> torch.Tensor:
        """(N,) bool: pan resting anywhere on the board (board frame), upright w.r.t. it."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        p = self._to_board(self.pan.data.root_pos_w)
        on = (p[:, 0].abs() < c.board_size[0] / 2) & (p[:, 1].abs() < c.board_size[1] / 2) \
            & (p[:, 2] > c.board_size[2] / 2) & (p[:, 2] < c.board_size[2] / 2 + 0.04)
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        pan_up_w = quat_apply(self.pan.data.root_quat_w, ez)
        board_up_w = quat_apply(self.board.data.root_quat_w, ez)
        upright = (pan_up_w * board_up_w).sum(-1).clamp(-1.0, 1.0) \
            >= math.cos(math.radians(c.pan_upright_max_deg))
        return on & upright

    def pan_on_burner(self) -> torch.Tensor:
        """(N,) bool: pan seated on the COMMANDED burner — centre within `xy_tol` of the
        commanded burner centre in the board frame, resting on the surface, upright."""
        c = self.cfg
        p = self._to_board(self.pan.data.root_pos_w)
        bx = torch.tensor(c.burner_xs, device=self.env.device)[self._active]
        d = torch.stack([p[:, 0] - bx, p[:, 1]], dim=-1).norm(dim=-1)
        return self.pan_on_board() & (d < c.xy_tol)

    def pot_on_rail(self) -> torch.Tensor:
        """(N,) bool: pot on the counterweight rail (board frame; any orientation — a
        counterweight balances however it lies)."""
        c = self.cfg
        p = self._to_board(self.pot.data.root_pos_w)
        return (p[:, 0] > c.rail_zone[0]) & (p[:, 0] < c.rail_zone[1]) \
            & (p[:, 1].abs() < c.rail_half_w) \
            & (p[:, 2] > c.board_size[2] / 2 + 0.01) & (p[:, 2] < c.board_size[2] / 2 + 0.10)

    def level(self) -> torch.Tensor:
        """(N,) bool: board within `level_tol_deg` of horizontal."""
        return self.tilt().abs() <= math.radians(self.cfg.level_tol_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: pan + pot slow, board rotation slow."""
        c = self.cfg
        return (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.pot.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.board.data.root_ang_vel_w.norm(dim=-1) < c.board_settle_w)

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self) -> None:
        """The hinge spring: tau = -k*theta - c*omega_hinge about board-local +y (the hinge
        axis IS local y — body-local application, the frame-exact form), plus latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        theta = self.tilt()
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        axis_w = quat_apply(self.board.data.root_quat_w, ey)
        omega_h = (self.board.data.root_ang_vel_w * axis_w).sum(-1)
        tau = -c.spring_k * theta - c.spring_c * omega_h
        torques = torch.zeros(n, 1, 3, device=dev)
        torques[:, 0, 1] = tau  # body-local +y = the hinge axis, exactly
        self.board.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torques)

        self._pan_on_ever |= self.pan_on_board()
        self._pan_burner_ever |= self.pan_on_burner()
        self._both_ever |= self._pan_burner_ever & self.pot_on_rail()

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {k: getattr(self, k).data.root_state_w[env_ids].clone()
                       for k in ("pedestal", "board", "indicator", "pan", "pot")},
            "active": self._active[env_ids].clone(),
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_pan_on_ever", "_pan_burner_ever", "_both_ever")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, v in state["bodies"].items():
            getattr(self, k).write_root_state_to_sim(v, env_ids)
        self._active[env_ids] = state["active"]
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A stove pedestal carries a long SEE-SAW cooktop board "
            f"({c.board_size[0] * 100:.0f} x {c.board_size[1] * 100:.0f} cm) on a central "
            f"hinge with a weak centering spring (tilts up to {c.tilt_limit_deg:.0f} deg "
            f"each way). One half of the board has three dark burner rings; the other half "
            f"is a pale counterweight rail. An orange indicator post standing beside the "
            f"board marks the COMMANDED burner for this episode. A black frying pan "
            f"(~{c.pan_mass * 1000:.0f} g) and a steel moka pot (~{c.pot_mass * 1000:.0f} g) "
            f"lie on the far side of the workspace.\n"
            f"Goal: put the frying pan flat on the commanded burner AND leave the cooktop "
            f"LEVEL (within {c.level_tol_deg:.1f} deg). The pan alone tips the board onto "
            f"its stop, so the moka pot must be parked on the counterweight rail at the "
            f"lever arm that cancels the pan's torque (farther out for a farther burner). "
            f"Balancing around the wrong burner does not count, and nothing may still be "
            f"moving when judged."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, current physical state: pan seated on the commanded burner + pot on
        the rail + board level + everything settled."""
        return self.pan_on_burner() & self.pot_on_rail() & self.level() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0 nothing; 0.15 latched pan-ever-on-board; 0.40 latched
        pan-ever-on-commanded-burner; 0.55 latched both-ever-placed; 0.55..0.90 live as the
        board approaches level while both remain placed; 1.0 iff success."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        s = torch.zeros(n, device=dev)
        s = torch.where(self._pan_on_ever, torch.full_like(s, 0.15), s)
        s = torch.where(self._pan_burner_ever, torch.full_like(s, 0.40), s)
        s = torch.where(self._both_ever, torch.full_like(s, 0.55), s)
        lim = math.radians(c.tilt_limit_deg)
        tol = math.radians(c.level_tol_deg)
        frac = ((lim - self.tilt().abs()) / (lim - tol)).clamp(0.0, 1.0)
        live = torch.where(self.pan_on_burner() & self.pot_on_rail(),
                           0.55 + 0.35 * frac, torch.zeros(n, device=dev))
        s = torch.maximum(s, live)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="seesaw_stove", robot="null", env_spacing=3))
