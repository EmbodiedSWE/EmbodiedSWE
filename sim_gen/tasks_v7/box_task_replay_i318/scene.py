"""WeaveCloseCrateScene — pack both loose items into the open crate, then CLOSE its
two-flap lid in the forced WEAVE order: short red tuck flap first, long blue main
flap second, so the main flap rests ON TOP of the seated tuck flap.

Derived from the `box_task/box_task_replay` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is "pick loose products and place them into an already-open,
ready cardboard box" — the container is passive scenery, the box state never changes,
and the episode ends with the box still gaping. Here packing the items is only the
FIRST HALF of the job: the crate has a real articulated two-flap lid (spawn-authored
revolute joints), and the goal state is the CLOSED, woven lid — an ordered
ARTICULATION-CLOSURE plan grafted onto the packing plan. The order is geometrically
enforced for the final state: the short tuck flap rests bistably on its 0-degree
joint stop, so the long main flap, whose closing sweep approaches from above, lands
ON it and cannot depress or undercut it (the tuck's stop blocks inward rotation).
Closing the main flap FIRST instead parks it on its own small dip stop; a tuck flap
dropped onto it afterwards rides the main plate, cocked ~13 degrees open — outside
the closed tolerance and with the weave INVERTED. Recovering requires reopening.

Mechanics: the crate is a heavy DYNAMIC compound (floor + 4 walls, custom spawner —
schemas authored in the func) so both flap joints are dynamic-dynamic and the whole
linkage teleports at reset. Each flap is a plate whose body origin sits ON its hinge
line (local +x inboard when closed, local y = hinge axis); its spawner authors the
RevoluteJoint to the sibling crate (axis Y, joint-pair collision filtered — the
stops, not rim contact, define the flap rests). Joint travel 170 degrees (< 175:
wrap-safe): open rest at the -170 stop (flap hangs outboard, ~10 degrees below
horizontal, gravity-pressed), closed rest at the upper stop (tuck: 0; main: +2.5
dip). Gravity is bistable about vertical, so a flap stays where it is put.
`post_step` owns the wrench slots: `tuck_tau` / `main_tau` are torques about the
flap's BODY-Y axis (positive = closing), fingertip-scale caps; hinge rates are
finite-differenced (root_ang_vel_w is phantom under external wrenches).

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py run):
  - `packed_can` / `packed_candle` (latch, 30-substep streak): item inside the
    upright crate (crate-body-frame bounds, centre below the rim band), settled.
  - `tuck_set` (latch): tuck flap closed within tolerance, slow, with BOTH items
    currently contained and the crate settled.
  - `lid_set` (latch): full closure — both flaps in the closed band, weave delta
    positive (main plate mid ABOVE tuck plate mid at the overlap probe), slow.
  - success(): CURRENT state — crate upright/settled, both items contained and
    settled, both flaps in the closed band, weave delta > weave_min, rates slow.
  - score() = 1.0 iff success(), else 0.2 * (packed_can + packed_candle + tuck_set
    + lid_set). ~0 for the null policy; disturbing the lid after success falls
    back to 0.80, never 0.

Per-episode randomization (verified by READBACK in smoke.py): crate xy + free yaw
(whole linkage teleported), each item's polar offset in the +-body-y cones (the
open flaps stick outboard along +-body-x), which item spawns on which side, item yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

_SPAWNER_CACHE: dict[str, Any] = {}


# ----- USD authoring helpers ---------------------------------------------------------------------
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


def _rigid_body(root, *, mass, com, inertia, lin_damp, ang_damp):
    """RigidBodyAPI + MassAPI (mass, CoM AND diagonal inertia — PhysX's shape-derived
    inertia is unauditable) + Physx solver/damping attrs (custom spawners apply no
    cfg schemas, so everything is authored here)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    ma = UsdPhysics.MassAPI.Apply(root)
    ma.CreateMassAttr(float(mass))
    ma.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _make_collide(contact_offset: float, mat=None) -> Callable:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if mat is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")

    return collide


def _friction_mat(stage, path: str, mu: tuple):
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(float(mu[0]))
    mapi.CreateDynamicFrictionAttr(float(mu[1]))
    mapi.CreateRestitutionAttr(0.0)
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pxm.CreateRestitutionCombineModeAttr().Set("min")
    return mat


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _flap_rot0(sx: float, theta_j_deg: float) -> tuple:
    """Init-state quat for a flap: q_mount(sx) * q_y(theta_j). Mount = identity on the
    -x hinge (inboard = +x), 180-deg yaw on the +x hinge (inboard = -x), so 'closing'
    is +theta_j about the body-local y axis on BOTH flaps."""
    th = math.radians(theta_j_deg) / 2
    w2, y2 = math.cos(th), math.sin(th)
    if sx < 0:
        return (w2, 0.0, y2, 0.0)
    # (0,0,0,1) * (w2, 0, y2, 0) = (0, y2, 0, w2)
    return (0.0, y2, 0.0, w2)


# ----- compound spawn funcs ----------------------------------------------------------------------
def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The crate: heavy DYNAMIC compound (dynamic so the flap joints are
    dynamic-dynamic and the whole linkage teleports at reset). Local frame: origin at
    the OUTER-box centre, mouth at +z. Children: floor plate + 4 walls."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(0.0, 0.0, -0.010), inertia=c.inertia,
                lin_damp=0.5, ang_damp=0.5)
    mat = _friction_mat(stage, f"{prim_path}/mat", c.mu)
    collide = _make_collide(c.contact_offset, mat)
    hx, hy, hz = c.outer[0] / 2, c.outer[1] / 2, c.outer[2] / 2
    t = c.wall_t
    wall_h = c.outer[2] - t
    wall_cz = -hz + t + wall_h / 2
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, -hz + t / 2),
             size=(c.outer[0], c.outer[1], t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_yp", center=(0.0, hy - t / 2, wall_cz),
             size=(c.outer[0], t, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_yn", center=(0.0, -hy + t / 2, wall_cz),
             size=(c.outer[0], t, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_xp", center=(hx - t / 2, 0.0, wall_cz),
             size=(t, c.outer[1] - 2 * t, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_xn", center=(-hx + t / 2, 0.0, wall_cz),
             size=(t, c.outer[1] - 2 * t, wall_h), color=c.color, collide=collide)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A lid flap: body origin ON the hinge line (local +x inboard when closed,
    local y along the hinge axis). One plate child starting 3 mm inboard of the
    hinge line (clears the wall-top corner through the swing). Spawn-authors the
    RevoluteJoint to the sibling crate: axis Y, limits [lo_deg, hi_deg] in JOINT
    convention (negative = open/outboard, positive = closed-and-beyond);
    joint-pair collision filtered — the stops define the rests."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    plate_len = c.reach - 0.003
    plate_c = (c.reach + 0.003) / 2
    _rigid_body(root, mass=c.mass, com=(plate_c, 0.0, 0.0), inertia=c.inertia,
                lin_damp=0.05, ang_damp=0.3)
    mat = _friction_mat(stage, f"{prim_path}/mat", c.mu)
    collide = _make_collide(c.contact_offset, mat)
    _add_box(stage, f"{prim_path}/plate", center=(plate_c, 0.0, 0.0),
             size=(plate_len, c.width, c.thick), color=c.color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    sx = float(c.side)
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Crate"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(sx * float(c.hinge_x), 0.0, float(c.hinge_z)))
    if sx < 0:
        j.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    else:
        j.CreateLocalRot0Attr(Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLowerLimitAttr(float(c.lo_deg))
    j.CreateUpperLimitAttr(float(c.hi_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            outer: tuple = (0.196, 0.150, 0.130)
            wall_t: float = 0.008
            mass: float = 6.0
            inertia: tuple = (0.0197, 0.0277, 0.0305)
            mu: tuple = (0.90, 0.80)
            color: tuple = (0.58, 0.42, 0.22)
            contact_offset: float = 0.002

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            side: float = 1.0
            reach: float = 0.062
            width: float = 0.130
            thick: float = 0.008
            hinge_x: float = 0.098
            hinge_z: float = 0.071
            lo_deg: float = -170.0
            hi_deg: float = 0.0
            mass: float = 0.06
            inertia: tuple = (8.5e-5, 2.0e-5, 1.04e-4)
            mu: tuple = (0.60, 0.50)
            color: tuple = (0.85, 0.20, 0.15)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["crate"] = CrateSpawnerCfg
        _SPAWNER_CACHE["flap"] = FlapSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class WeaveCloseCrateSceneCfg(BaseCfg):
    """Config for `WeaveCloseCrateScene`. Geometry is derived once in `__post_init__`
    so the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    closed_tol: tuple = tunable((math.radians(-4.0), math.radians(7.0)))  # OPEN-angle
    # band that counts as closed: tuck rests at 0, main rests at +3.0 on the tuck
    # (asin(tuck_thick/main_reach)) or at -2.5 on its dip stop; a tuck cocked on a
    # closed main sits at ~+13 — outside the band
    weave_min: float = tunable(0.006)  # main plate mid must sit ABOVE the tuck plate
    # mid by this at the overlap probe (correct weave +0.013, inverted weave -0.012)
    contain_x: float = tunable(0.075)  # |item x| in crate frame (interior half-x 0.090;
    # max physical inside offset 0.062 < this; outside the shell >= 0.126)
    contain_y: float = tunable(0.052)  # |item y| in crate frame (interior half-y 0.067)
    contain_z: tuple = tunable((-0.056, 0.040))  # item-centre band: above the floor
    # (local -0.057), below the rim (local +0.065) by >= 2.5 cm — containment is
    # judged BELOW the aperture (an item on the rim/wall top reads >= +0.065)
    upright_deg: float = tunable(10.0)  # crate mouth-up cone
    rest_z_tol: float = tunable(0.015)  # crate centre near its rest height
    settle_v: float = tunable(0.10)  # max |lin vel| when judging (above phantom band)
    settle_w: float = tunable(0.60)  # max crate |ang vel| when judging (rad/s)
    settle_rate: float = tunable(0.60)  # max |FD flap rate| when judging (rad/s)
    latch_streak: int = tunable(30)  # substeps a latch condition must hold (0.25 s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    crate_pos: tuple = tunable((0.0, 0.0))  # nominal crate centre
    pos_jitter: float = tunable(0.05)  # +- xy jitter of the crate at reset
    yaw_deg: float = tunable(180.0)  # +- crate yaw at reset (free)
    item_r: tuple = tunable((0.19, 0.25))  # item polar radius from the crate centre
    item_cone_deg: float = tunable(35.0)  # items sampled within +-this of the
    # +-body-y directions (the open flaps stick outboard along +-body-x)

    # --- tunable: plant (difficulty dials) -------------------------------------------------------
    tuck_tau_max: float = tunable(0.10)  # N*m cap for the tuck hinge (~3 N fingertip
    # at the flap mid; lift-from-stop needs 0.018) — SHARED with smoke probes
    main_tau_max: float = tunable(0.55)  # N*m cap for the main hinge (lift-from-stop
    # needs 0.23) — SHARED with smoke's wrong-order and protrusion probes
    crate_mass: float = tunable(6.0)
    tuck_mass: float = tunable(0.06)
    main_mass: float = tunable(0.30)
    can_mass: float = tunable(0.15)
    candle_mass: float = tunable(0.12)
    crate_mu: tuple = tunable((0.90, 0.80))
    flap_mu: tuple = tunable((0.60, 0.50))
    item_mu: tuple = tunable((0.60, 0.50))

    # --- info: structure (crate body frame; origin = outer centre, mouth at +z) ------------------
    outer: tuple = info((0.196, 0.150, 0.130))
    wall_t: float = info(0.008)
    hinge_x: float = info(0.098)  # |x| of both hinge lines (tuck on +x, main on -x)
    tuck_hinge_z: float = info(0.071)  # rim 0.065 + thick/2 + 2 mm clearance
    main_hinge_z: float = info(0.077)
    tuck_reach: float = info(0.062)  # tuck covers crate x [0.036, 0.095]
    main_reach: float = info(0.155)  # main covers crate x [-0.095, 0.057]
    flap_width: float = info(0.130)
    tuck_thick: float = info(0.008)
    main_thick: float = info(0.020)
    joint_lo_deg: float = info(-170.0)  # open stop (travel 170 < 175: wrap-safe)
    tuck_hi_deg: float = info(0.0)  # tuck closed stop: flat over the mouth
    main_hi_deg: float = info(2.5)  # main dip stop (a molded lip): cannot enter the box
    can_r: float = info(0.028)
    can_h: float = info(0.095)  # standing top at local +0.038 — 2.7 cm below the rim
    candle_s: float = info(0.048)  # square section
    candle_h: float = info(0.085)
    drop_dx: tuple = info((-0.050, 0.050))  # in-crate drop x (can, candle): wall
    # clearance >= 12 mm, inter-item gap 48 mm (no seam wedging)
    drop_clear: float = info(0.020)  # release height of the item bottom above the rim
    probe_x: float = info(0.046)  # crate-frame x of the weave overlap probe
    base_pos: tuple = info((0.0, -0.40))  # documented Franka base xy (TASK.md)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    int_hx: float = field(default=None, init=False)
    int_hy: float = field(default=None, init=False)
    floor_z_local: float = field(default=None, init=False)
    rim_z_local: float = field(default=None, init=False)
    rest_z: float = field(default=None, init=False)  # crate centre resting on the ground
    d_tuck: float = field(default=None, init=False)  # probe lever from the tuck hinge
    d_main: float = field(default=None, init=False)  # probe lever from the main hinge
    tuck_mgd: float = field(default=None, init=False)  # gravity torque scale (N*m)
    main_mgd: float = field(default=None, init=False)
    tuck_inertia_h: float = field(default=None, init=False)  # inertia about the hinge
    main_inertia_h: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        lx, ly, h = self.outer
        self.int_hx = lx / 2 - self.wall_t
        self.int_hy = ly / 2 - self.wall_t
        self.floor_z_local = -h / 2 + self.wall_t
        self.rim_z_local = h / 2
        self.rest_z = h / 2
        self.d_tuck = self.hinge_x - self.probe_x  # 0.052
        self.d_main = self.hinge_x + self.probe_x  # 0.144
        g = 9.81
        com_t = (self.tuck_reach + 0.003) / 2  # 0.0325
        com_m = (self.main_reach + 0.003) / 2  # 0.079
        self.tuck_mgd = self.tuck_mass * g * com_t  # 0.019
        self.main_mgd = self.main_mass * g * com_m  # 0.233
        self.tuck_inertia_h = 2.0e-5 + self.tuck_mass * com_t**2  # 8.3e-5
        self.main_inertia_h = 6.1e-4 + self.main_mass * com_m**2  # 2.5e-3


# ----- scene -------------------------------------------------------------------------------------
class WeaveCloseCrateScene(BaseScene):
    cfg: WeaveCloseCrateSceneCfg

    def __init__(self, cfg: WeaveCloseCrateSceneCfg | None = None) -> None:
        super().__init__(cfg or WeaveCloseCrateSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the crate compound (FIRST — the flaps' spawn-authored
        joints target their sibling), the two flaps (spawned near open rest), and
        the two items (built-in shape cfgs so their cfg schemas apply)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

        def item_props(mass: float) -> dict:
            return dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.10, angular_damping=0.50, **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.item_mu[0], dynamic_friction=c.item_mu[1],
                    restitution=0.0),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.85, dynamic_friction=0.75, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=sp["crate"](outer=c.outer, wall_t=c.wall_t, mass=c.crate_mass,
                                  mu=c.crate_mu, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.rest_z + 0.002)),
            ),
            "tuck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tuck",
                spawn=sp["flap"](side=1.0, reach=c.tuck_reach, width=c.flap_width,
                                 thick=c.tuck_thick, hinge_x=c.hinge_x,
                                 hinge_z=c.tuck_hinge_z, lo_deg=c.joint_lo_deg,
                                 hi_deg=c.tuck_hi_deg, mass=c.tuck_mass,
                                 inertia=(8.5e-5, 2.0e-5, 1.04e-4), mu=c.flap_mu,
                                 color=(0.85, 0.20, 0.15),
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hinge_x, 0.0, c.rest_z + 0.002 + c.tuck_hinge_z),
                    rot=_flap_rot0(1.0, -165.0)),
            ),
            "main": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Main",
                spawn=sp["flap"](side=-1.0, reach=c.main_reach, width=c.flap_width,
                                 thick=c.main_thick, hinge_x=c.hinge_x,
                                 hinge_z=c.main_hinge_z, lo_deg=c.joint_lo_deg,
                                 hi_deg=c.main_hi_deg, mass=c.main_mass,
                                 inertia=(4.3e-4, 6.1e-4, 1.02e-3), mu=c.flap_mu,
                                 color=(0.15, 0.35, 0.85),
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.hinge_x, 0.0, c.rest_z + 0.002 + c.main_hinge_z),
                    rot=_flap_rot0(-1.0, -165.0)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.90, 0.45, 0.08)),
                    **item_props(c.can_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.22, c.can_h / 2 + 0.002)),
            ),
            "candle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Candle",
                spawn=sim_utils.CuboidCfg(
                    size=(c.candle_s, c.candle_s, c.candle_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.15, 0.65, 0.20)),
                    **item_props(c.candle_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.22, c.candle_h / 2 + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the solve drives the flaps with per-substep wrenches; without this
                # an applied wrench is integrated only on the first solver iteration
                # and the hinge plant stalls/rings
                "enable_external_forces_every_iteration": True,
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
        n, dev = env.num_envs, env.device
        self.crate: RigidObject = env.iscene["crate"]
        self.tuck: RigidObject = env.iscene["tuck"]
        self.main: RigidObject = env.iscene["main"]
        self.can: RigidObject = env.iscene["can"]
        self.candle: RigidObject = env.iscene["candle"]
        self.env_origins = env.iscene.env_origins
        # Mount quats per flap column (col 0 = tuck/+x hinge, col 1 = main/-x hinge).
        self._q_mount = torch.zeros(2, 4, device=dev)
        self._q_mount[0] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        self._q_mount[1] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        # External drive input (solve/smoke write; post_step consumes + owns the
        # wrench slots — never call set_external_force_and_torque directly).
        self.tuck_tau = torch.zeros(n, device=dev)  # +ve = closing (body-y torque)
        self.main_tau = torch.zeros(n, device=dev)
        # FD hinge-rate state (root_ang_vel_w is phantom under external wrenches).
        self._ang_prev = torch.zeros(n, 2, device=dev)
        self.flap_rate = torch.zeros(n, 2, device=dev)  # d(open angle)/dt
        # Rubric latches + streak counters.
        self.packed_can = torch.zeros(n, dtype=torch.bool, device=dev)
        self.packed_candle = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tuck_set = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lid_set = torch.zeros(n, dtype=torch.bool, device=dev)
        self._streaks = torch.zeros(n, 4, dtype=torch.long, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample ONE crate transform (xy jitter + free yaw) and write
        the WHOLE linkage with it (crate + both flaps at the open rest); items
        standing upright at sampled polar offsets in the +-body-y cones (the open
        flaps stick outboard along +-body-x), sides swapped at random; latches,
        streaks, FD state and drive buffers zeroed."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(8, device=dev)  # burn post-seed draws (first-draw degeneracy)

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0] = torch.cos(yaw / 2)
        q_yaw[:, 3] = torch.sin(yaw / 2)
        cxy = torch.empty(m, 2, device=dev)
        cxy[:, 0] = c.crate_pos[0]
        cxy[:, 1] = c.crate_pos[1]
        cxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        crate_p = torch.cat([cxy, torch.full((m, 1), c.rest_z + 0.002, device=dev)],
                            dim=1) + origin

        def write(body, local_pos, q_local) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = crate_p + quat_apply(q_yaw, local_pos)
            st[:, 3:7] = quat_mul(q_yaw, q_local)
            body.write_root_state_to_sim(st, env_ids)

        write(self.crate, torch.zeros(m, 3, device=dev),
              torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4).clone())

        # Flaps at joint -165 deg (settle onto the -170 open stop).
        th0 = math.radians(-165.0)
        q_h = torch.tensor([math.cos(th0 / 2), 0.0, math.sin(th0 / 2), 0.0],
                           device=dev).expand(m, 4)
        for col, (body, sx, hz) in enumerate(
                ((self.tuck, 1.0, c.tuck_hinge_z), (self.main, -1.0, c.main_hinge_z))):
            anchor = torch.zeros(m, 3, device=dev)
            anchor[:, 0] = sx * c.hinge_x
            anchor[:, 2] = hz
            write(body, anchor, quat_mul(self._q_mount[col].expand(m, 4), q_h))
            self._ang_prev[env_ids, col] = -th0  # open angle = -joint angle
        self.flap_rate[env_ids] = 0.0

        # Items: polar offsets in the +-body-y cones; side assignment random.
        swap = torch.rand(m, device=dev) < 0.5
        cone = math.radians(c.item_cone_deg)
        for i, (body, h) in enumerate(((self.can, c.can_h), (self.candle, c.candle_h))):
            side = torch.where(swap ^ (i == 1),
                               torch.full((m,), math.pi / 2, device=dev),
                               torch.full((m,), -math.pi / 2, device=dev))
            phi = yaw + side + (torch.rand(m, device=dev) * 2 - 1) * cone
            r = c.item_r[0] + torch.rand(m, device=dev) * (c.item_r[1] - c.item_r[0])
            iyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = cxy[:, 0] + r * torch.cos(phi)
            st[:, 1] = cxy[:, 1] + r * torch.sin(phi)
            st[:, 2] = h / 2 + 0.002
            st[:, 0:3] += origin
            st[:, 3] = torch.cos(iyaw / 2)
            st[:, 6] = torch.sin(iyaw / 2)
            body.write_root_state_to_sim(st, env_ids)

        for t in (self.packed_can, self.packed_candle, self.tuck_set, self.lid_set):
            t[env_ids] = False
        self._streaks[env_ids] = 0
        self.tuck_tau[env_ids] = 0.0
        self.main_tau[env_ids] = 0.0

    # ----- readings ------------------------------------------------------------------------------
    def crate_pos(self) -> torch.Tensor:
        """(N, 3) crate centre (env-origin corrected)."""
        return self.crate.data.root_pos_w - self.env_origins

    def up_z(self) -> torch.Tensor:
        """(N,) world-z component of the crate's body +z (mouth normal)."""
        q = self.crate.data.root_quat_w
        return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)

    def _flap_open(self, body, col: int) -> torch.Tensor:
        """(N,) OPEN angle of a flap (0 = closed flat over the mouth, +170 = open
        rest outboard; negative = dipped past flat). open = -joint angle;
        q_h = conj(q_crate * q_mount) * q_flap, joint = 2*atan2(q_h[y], q_h[w])."""
        from isaaclab.utils.math import quat_conjugate, quat_mul

        n = self.env.num_envs
        q_ref = quat_mul(self.crate.data.root_quat_w, self._q_mount[col].expand(n, 4))
        q_h = quat_mul(quat_conjugate(q_ref), body.data.root_quat_w)
        ang = 2.0 * torch.atan2(q_h[:, 2], q_h[:, 0])
        return -torch.atan2(torch.sin(ang), torch.cos(ang))

    def tuck_open(self) -> torch.Tensor:
        return self._flap_open(self.tuck, 0)

    def main_open(self) -> torch.Tensor:
        return self._flap_open(self.main, 1)

    def weave_delta(self) -> torch.Tensor:
        """(N,) crate-frame z of the MAIN plate mid-surface minus the TUCK plate
        mid-surface, both evaluated over the overlap probe x = probe_x (flap-local
        points at levers d_main / d_tuck). Positive = main ON TOP of tuck (correct
        weave, +0.013); negative = tuck riding on main (inverted weave, -0.012)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        def probe_z(body, lever: float) -> torch.Tensor:
            p_local = torch.zeros(n, 3, device=dev)
            p_local[:, 0] = lever
            p_w = body.data.root_pos_w + quat_apply(body.data.root_quat_w, p_local)
            return quat_apply_inverse(self.crate.data.root_quat_w,
                                      p_w - self.crate.data.root_pos_w)[:, 2]

        return probe_z(self.main, c.d_main) - probe_z(self.tuck, c.d_tuck)

    def items_local(self) -> torch.Tensor:
        """(N, 2, 3) item centres in the CRATE BODY frame (can, candle)."""
        from isaaclab.utils.math import quat_apply_inverse

        pos = torch.stack([self.can.data.root_pos_w, self.candle.data.root_pos_w], dim=1)
        n = pos.shape[0]
        hq = self.crate.data.root_quat_w[:, None, :].expand(n, 2, 4).reshape(n * 2, 4)
        hp = self.crate.data.root_pos_w[:, None, :]
        return quat_apply_inverse(hq, (pos - hp).reshape(n * 2, 3)).reshape(n, 2, 3)

    def crate_upright(self) -> torch.Tensor:
        c = self.cfg
        up = self.up_z() >= math.cos(math.radians(c.upright_deg))
        at_rest = (self.crate_pos()[:, 2] - c.rest_z).abs() < c.rest_z_tol
        return up & at_rest

    def crate_settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.crate.data.root_lin_vel_w.norm(dim=-1) < c.settle_v)
                & (self.crate.data.root_ang_vel_w.norm(dim=-1) < c.settle_w))

    def items_settled(self) -> torch.Tensor:
        """(N, 2) bool."""
        v = torch.stack([self.can.data.root_lin_vel_w.norm(dim=-1),
                         self.candle.data.root_lin_vel_w.norm(dim=-1)], dim=1)
        return v < self.cfg.settle_v

    def flaps_slow(self) -> torch.Tensor:
        return self.flap_rate.abs().max(dim=1).values < self.cfg.settle_rate

    def contained(self) -> torch.Tensor:
        """(N, 2) bool, geometric: item centre inside the UPRIGHT crate's interior
        in the crate body frame, below the rim band. An item on a wall top, on the
        closed lid, leaning outside, or on the ground beside the crate fails the
        bounds or the upright gate."""
        c = self.cfg
        loc = self.items_local()
        inside = ((loc[:, :, 0].abs() < c.contain_x)
                  & (loc[:, :, 1].abs() < c.contain_y)
                  & (loc[:, :, 2] > c.contain_z[0]) & (loc[:, :, 2] < c.contain_z[1]))
        return inside & self.crate_upright().unsqueeze(-1)

    def _in_closed_band(self, ang: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return (ang > c.closed_tol[0]) & (ang < c.closed_tol[1])

    def _finite(self) -> torch.Tensor:
        return (torch.isfinite(self.crate.data.root_pos_w).all(dim=-1)
                & torch.isfinite(self.can.data.root_pos_w).all(dim=-1)
                & torch.isfinite(self.candle.data.root_pos_w).all(dim=-1)
                & torch.isfinite(self.tuck_open()) & torch.isfinite(self.main_open()))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, CURRENT physical state: crate upright and settled, BOTH items
        contained and settled, BOTH flaps in the closed band with the weave the
        right way round (main above tuck at the overlap), flap rates slow, finite."""
        return (self.crate_upright() & self.crate_settled()
                & self.contained().all(dim=1) & self.items_settled().all(dim=1)
                & self._in_closed_band(self.tuck_open())
                & self._in_closed_band(self.main_open())
                & (self.weave_delta() > self.cfg.weave_min)
                & self.flaps_slow() & self._finite())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); otherwise 0.2 * (packed_can
        + packed_candle + tuck_set + lid_set). ~0 for the null policy; a lid
        disturbance after success falls back to 0.80 (latched credit does not
        evaporate)."""
        partial = 0.2 * (self.packed_can.float() + self.packed_candle.float()
                         + self.tuck_set.float() + self.lid_set.float())
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Flap plant: `tuck_tau` / `main_tau` about each flap's BODY-Y axis
        (positive = closing), clamped to the fingertip-scale caps. Then FD hinge
        rates and the streak-gated rubric latches (a fly-through state earns
        nothing; NaN comparisons are False, so a garbage frame earns nothing)."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        for body, tau, cap in ((self.tuck, self.tuck_tau, c.tuck_tau_max),
                               (self.main, self.main_tau, c.main_tau_max)):
            t = torch.zeros(n, 1, 3, device=dev)
            t[:, 0, 1] = tau.clamp(-cap, cap)
            body.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), t)

        dt = self.env.dt
        ang = torch.stack([self.tuck_open(), self.main_open()], dim=1)
        d = torch.atan2(torch.sin(ang - self._ang_prev), torch.cos(ang - self._ang_prev))
        self.flap_rate = d / dt
        self._ang_prev = ang

        crate_ok = self.crate_upright() & self.crate_settled()
        pack_now = self.contained() & crate_ok.unsqueeze(-1) & self.items_settled()
        both_in = self.contained().all(dim=1)
        tuck_closed = (self._in_closed_band(ang[:, 0])
                       & (self.flap_rate[:, 0].abs() < c.settle_rate))
        tuck_now = tuck_closed & both_in & crate_ok
        lid_now = (tuck_closed & self._in_closed_band(ang[:, 1])
                   & (self.flap_rate[:, 1].abs() < c.settle_rate)
                   & (self.weave_delta() > c.weave_min) & both_in & crate_ok)

        conds = torch.stack([pack_now[:, 0], pack_now[:, 1], tuck_now, lid_now], dim=1)
        self._streaks = torch.where(conds, self._streaks + 1,
                                    torch.zeros_like(self._streaks))
        hit = self._streaks >= c.latch_streak
        self.packed_can = self.packed_can | hit[:, 0]
        self.packed_candle = self.packed_candle | hit[:, 1]
        self.tuck_set = self.tuck_set | hit[:, 2]
        self.lid_set = self.lid_set | hit[:, 3]

    # ----- state (full, restorable) --------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"crate": self.crate, "tuck": self.tuck, "main": self.main,
                "can": self.can, "candle": self.candle}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("packed_can", "packed_candle", "tuck_set", "lid_set",
                               "_streaks", "_ang_prev", "flap_rate",
                               "tuck_tau", "main_tau")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        lx, ly, h = (v * 100 for v in c.outer)
        return (
            f"A brown wooden crate ({lx:.0f} x {ly:.0f} cm footprint, {h:.0f} cm tall, "
            f"open mouth up) stands on the floor; its position and heading vary per "
            f"episode. Its lid is TWO hinged flaps, both flung open so they hang "
            f"outboard off opposite short ends: a short RED tuck flap "
            f"({c.tuck_reach * 100:.1f} cm) and a long BLUE main flap "
            f"({c.main_reach * 100:.1f} cm) that together cover the mouth with a small "
            f"overlap. On the floor near the crate, one to each long side, stand two "
            f"loose items: an ORANGE can (cylinder, {2 * c.can_r * 100:.1f} cm across, "
            f"{c.can_h * 100:.1f} cm tall) and a GREEN candle (square block, "
            f"{c.candle_s * 100:.1f} cm wide, {c.candle_h * 100:.1f} cm tall). Which "
            f"item is on which side varies.\n"
            f"Goal: put BOTH items inside the crate, then close the lid in the WEAVE "
            f"order — swing the short RED flap shut first (it lies flat on its stop), "
            f"then swing the long BLUE flap shut ON TOP of it, so the blue flap's tip "
            f"rests on the red flap. Each flap is bistable: swung past vertical it "
            f"falls to the other rest, so flip each one with a firm push and let it "
            f"settle. Closing the BLUE flap first does not work: it parks on its own "
            f"stop and the red flap then rides on top of it, cocked open — the weave "
            f"is inverted and the lid is not closed; you would have to reopen the "
            f"blue flap. Items on the rim, on the closed lid, leaning on the crate, "
            f"or on the floor do not count; only items resting inside, below the "
            f"rim, count. The crate must end upright and settled, both items settled "
            f"inside, both flaps closed with blue woven over red."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Put the orange can and the green candle inside the open crate, then "
            "close its lid in order: swing the short red flap shut first, then the "
            "long blue flap shut on top of it, so the blue flap rests on the red one."
        )


# Guarded registration: the forge may import this module under two names.
if "weave_close_crate" not in SCENES.list():
    SCENES.register("weave_close_crate", WeaveCloseCrateScene)
if "simgen.weave_close_crate" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="weave_close_crate", robot="null"))
