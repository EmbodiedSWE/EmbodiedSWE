"""SpringBayScene — press the blue can against a spring plunger and seat it captive
behind the dispenser's front lip (sim_gen task
`living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i38`).

Derived from libero_90/living_room_scene2 "pick up the alphabet soup and put it in the
basket", but STRATEGICALLY different: the seed is one gravity drop — pick the can off
the table and release it inside an open basket; success is a bounding-box containment
readout, and the container accepts the can from any direction with no mechanism in the
way. Here nothing can be dropped into place AT ALL: the goal cavity is SHORTER than
the can. A dispenser bay on the floor holds a spring-loaded plunger (a real prismatic
joint with a linear drive) facing a fixed front lip; the free gap between the lip's
inner face and the plunger's rest face is 18 mm less than the can is long. A can
lowered casually into the bay ends up PROPPED — back end on the channel floor, front
end resting on top of the lip — and stays there scoring nothing. To finish, the solver
must load the mechanism: press the can axially into the plunger pad, compressing the
spring until the gap opens enough for the front end to slip off the lip edge and fall
flat, then let go — the spring shoves the can forward and pins it against the back of
the lip. The judged end state is a PRELOADED mechanism: the rubric reads the plunger's
compression (readback of the joint's actual travel, >= 10 mm; the spring sits at
~18 mm when genuinely seated, while gravity alone on the propped can yields only
~4 mm), plus the can lying flat, aligned, and settled inside the channel.

Strategy vs the corpus (tasks read: pull_cube_i20 beam-scale, close_box_i26 trap
crate, living_room_scene3_i33 flap pantry, pen_holder exemplar, plus the one-line
survey of every other tasks_v7 card): no existing task involves an elastic element —
every corpus mechanism is kinematic (hinges, slides, bayonets), gravity-driven
(counterweights, drop gates, drains) or pure contact (bridges, wedges, piles). Here
the load-bearing interaction is COMPRESS-THEN-SEAT against a preloaded spring: the
solver must push through a force that grows with progress, hold it while a second
degree of freedom (the front end dropping behind the lip) resolves, and then release
deliberately so the stored energy finishes the seating. The i33 flap is the nearest
neighbour (one moving part on the container) and differs in plan: its flap is a
passive one-way gate that gravity closes after a push-through; nothing is ever loaded
against it and its end state is unstressed. The seed's own end state (can loose inside
an open container) is constructed in smoke and REJECTED here.

success(): the BLUE can lies flat in the channel (bay-frame windows on x / y / z),
axis aligned with the channel within `align_max_deg`, plunger compression >=
`comp_min` (the spring genuinely loaded through can contact), can and plunger settled,
and the red decoy can NOT anywhere on/in the bay. score(): stateless partial credit
0.10 (blue can within `near_r` of the bay) + 0.15 (blue can inside the loose channel
volume) + 0.30 (in-channel AND spring engaged >= `comp_engage`), capped at 0.55;
1.0 iff success(). The null policy scores ~0 (both cans spawn well outside `near_r`).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - bay (dynamic, 25 kg — a heavy DYNAMIC fixture, never kinematic, so the prismatic
    joint's anchor follows the reset teleport): base flange + plinth whose flat top is
    the channel floor (0.100 m up), two side walls, a back housing wall, and the
    ORANGE front lip (45 mm above the channel floor — higher than the lying can's
    axis, so a seated can cannot ride up and over: it is genuinely captive).
  - plunger (dynamic, 0.25 kg): a dark pad riding a PrismaticJoint into the bay
    (axis = channel x, travel -32..+1 mm) with a linear drive (stiffness 400 N/m,
    damping 25 — the spring; seat preload ~7 N). Joint-pair collision stays FILTERED
    (the USD default): the pad may sweep through bay geometry, its travel is bounded
    by the joint limits alone.
  - blue can (target): r 33 mm x 105 mm, 350 g — the alphabet-soup stand-in;
    graspable across the diameter (66 < ~80 mm jaw).
  - red can (decoy): identical cylinder, red — the seed's distractor clutter reduced
    to one identified foil; it must stay OFF the bay.

Per-episode randomization (readback-verified in smoke): bay xy jitter + FREE yaw
(the press axis points anywhere), each can on its own jittered arc slot (angle +
radius + free yaw) around the bay. Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
R_CAN = 0.033  # can radius (66 mm dia < ~80 mm parallel jaw)
L_CAN = 0.105  # can length
Z_F = 0.100  # channel floor top above ground (bay frame z)
W_CH = 0.096  # channel inner width (y): > can dia + 24 mm fingers, < can length (no
#               crosswise jam — a can can NEVER lie across the channel)
SEAT_INTERF = 0.018  # rest gap = L_CAN - this: the cavity is SHORTER than the can
LIP_X0 = 0.070  # lip INNER face (bay frame x; +x = plunger -> lip = "forward")
LIP_T = 0.020  # lip thickness -> outer face at 0.090
H_LIP = 0.045  # lip height above the channel floor (> R_CAN: retention above center)
PLG_FACE_X = LIP_X0 - (L_CAN - SEAT_INTERF)  # plunger face REST x = -0.017
PLG_T = 0.020  # pad thickness -> pad center rest x = -0.027
PLG_H = 0.058  # pad height (covers the lying can's full cross-section)
PLG_TRAVEL = 0.032  # joint lower limit (compression range)
WALL_H = 0.050  # side wall height above the channel floor
BACK_X0 = -0.069  # back housing wall inner face (pad back at full travel: -0.069)


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
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
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


def _spawn_bay(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dispenser bay as ONE heavy DYNAMIC compound body (never kinematic:
    the prismatic joint's anchor must follow the reset teleport — a kinematic body0's
    anchor stays world-fixed at the spawn pose). Origin = bay center on the ground;
    +x = channel axis toward the lip."""
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
    # ZERO sleep/stabilization: a sleeping bay would freeze the joint anchor.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_slide_s, cfg.mu_slide_d)
    gray = (0.45, 0.48, 0.50)
    orange = (0.90, 0.45, 0.10)
    # base flange (stability + visual footprint) and plinth (channel floor top = Z_F)
    _box(stage, f"{prim_path}/base", (0.26, 0.20, 0.020), (0.0, 0.0, 0.010), gray,
         0.0015, material=body)
    _box(stage, f"{prim_path}/plinth", (0.179, 0.136, 0.080), (0.0005, 0.0, 0.060),
         gray, 0.0015, material=body)
    # side walls (inner faces |y| = W_CH/2)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (0.179, 0.020, WALL_H),
             (0.0005, sy * (W_CH / 2 + 0.010), Z_F + WALL_H / 2), gray, 0.0015,
             material=body)
    # ORANGE front lip: inner face x = LIP_X0, top = Z_F + H_LIP (above can center).
    # Slick top: the propped can's front end must slide backward off it under the press.
    _box(stage, f"{prim_path}/lip", (LIP_T, W_CH, H_LIP),
         (LIP_X0 + LIP_T / 2, 0.0, Z_F + H_LIP / 2), orange, 0.0015, material=slick)
    # back housing wall (behind the plunger's full travel; visual, pad never hits it)
    _box(stage, f"{prim_path}/back", (0.020, W_CH, 0.059),
         (BACK_X0 - 0.010, 0.0, Z_F + 0.0295), gray, 0.0015, material=body)
    return root


def _spawn_plunger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the spring plunger: one dark pad + the PrismaticJoint into the sibling
    bay with a linear drive (the spring). Origin = pad center."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    slick = _friction_material(stage, f"{prim_path}/pad_mat",
                               cfg.mu_slide_s, cfg.mu_slide_d)
    _box(stage, f"{prim_path}/pad", (PLG_T, W_CH - 0.004, PLG_H), (0.0, 0.0, 0.0),
         (0.15, 0.15, 0.18), 0.0015, material=slick)

    # prismatic spring joint to the sibling bay, axis = channel x
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/spring")
    j.CreateBody0Rel().SetTargets([f"{base}/Bay"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # joint-pair collision stays FILTERED (the USD default): the pad's travel is
    # bounded by the joint limits alone and it may sweep through bay geometry.
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(PLG_FACE_X - PLG_T / 2), 0.0, float(Z_F + 0.030)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-PLG_TRAVEL))
    j.CreateUpperLimitAttr(0.001)
    # the SPRING: linear drive toward joint position 0 (rest)
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(cfg.spring_k))
    drv.CreateDampingAttr(float(cfg.spring_c))
    drv.CreateTargetPositionAttr(0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bay" not in _SPAWNER_CACHE:

        @configclass
        class BaySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bay)
            mu_body_s: float = 0.35
            mu_body_d: float = 0.30
            mu_slide_s: float = 0.10
            mu_slide_d: float = 0.08

        @configclass
        class PlungerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plunger)
            mu_slide_s: float = 0.10
            mu_slide_d: float = 0.08
            spring_k: float = 400.0
            spring_c: float = 25.0

        _SPAWNER_CACHE.update(bay=BaySpawnerCfg, plunger=PlungerSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SpringBaySceneCfg(BaseCfg):
    """Config for `SpringBayScene`. The honesty knobs are asserted in `__post_init__`:
    the rest gap is SHORTER than the can (nothing seats without compressing the
    spring), the compression threshold sits far above what gravity alone produces on
    a propped can and far below the genuine seated preload, and the lip stands above
    the lying can's axis (a seated can is captive)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_x_win: tuple = tunable((0.000, 0.036))  # can center x (bay frame; seated ~0.0175)
    seat_y_max: float = tunable(0.026)  # |y| of the can center
    seat_z_win: tuple = tunable((0.026, 0.041))  # can center z - Z_F (lying: 0.033)
    align_max_deg: float = tunable(20.0)  # can axis vs channel axis
    comp_min: float = tunable(0.010)  # plunger compression for success (seated ~0.018)
    comp_engage: float = tunable(0.008)  # score: "spring engaged" partial-credit gate
    settle_lin: float = tunable(0.05)  # can settle gate (m/s)
    settle_ang: float = tunable(1.0)  # can settle gate (rad/s)
    plg_settle_lin: float = tunable(0.03)  # plunger settle gate (m/s)
    near_r: float = tunable(0.25)  # score: "brought near the bay" xy radius
    chan_x_win: tuple = tunable((-0.05, 0.095))  # loose channel volume (score credit)
    chan_y_max: float = tunable(0.048)
    chan_z_win: tuple = tunable((-0.005, 0.075))  # ... z - Z_F
    decoy_x_max: float = tunable(0.14)  # decoy-clear violation region (bay frame):
    decoy_y_max: float = tunable(0.11)  # |x|,|y| within these AND
    decoy_z_min: float = tunable(0.06)  # z above this = decoy on/in the bay -> fail

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    bay_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the bay at reset (m)
    bay_yaw_max: float = tunable(180.0)  # uniform +/- bay yaw (deg; FREE heading)
    can_slots: tuple = tunable((120.0, 240.0))  # world-frame arc slots: (blue, red) (deg)
    slot_jitter: float = tunable(25.0)  # uniform +/- per-can arc-angle jitter (deg)
    can_radius: float = tunable(0.32)  # arc radius around the bay (m)
    radius_jitter: float = tunable(0.03)

    # --- info: structure -----------------------------------------------------------------------
    can_r: float = info(R_CAN)
    can_l: float = info(L_CAN)
    can_mass: float = info(0.35)
    bay_mass: float = info(25.0)  # heavy dynamic fixture (joint anchor follows teleports)
    plg_mass: float = info(0.25)
    spring_k: float = info(400.0)  # N/m -> seated preload ~7 N, insertion peak <13 N
    spring_c: float = info(25.0)  # ~critical for the 0.25 kg pad
    mu_can_s: float = info(0.30)
    mu_can_d: float = info(0.25)
    mu_body_s: float = info(0.35)
    mu_body_d: float = info(0.30)
    mu_slide_s: float = info(0.10)  # lip top + plunger face (the can must slide on both)
    mu_slide_d: float = info(0.08)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    # Derived (filled in __post_init__).
    seat_comp: float = field(default=None, init=False)  # nominal seated compression
    prop_deg: float = field(default=None, init=False)  # propped-can tilt angle

    def __post_init__(self) -> None:
        self.seat_comp = SEAT_INTERF
        self.prop_deg = math.degrees(math.asin(H_LIP / L_CAN))
        # -- embodiment: the can fits a parallel jaw; fingers fit beside a channeled can --
        assert 2 * R_CAN < 0.08, "can must fit an ~80 mm parallel jaw"
        assert W_CH > 2 * R_CAN + 0.024, "channel must leave >=12 mm finger slots per side"
        # -- no crosswise jam: a can can never lie across the channel --
        assert W_CH < L_CAN - 0.005, "channel width must be less than the can length"
        # -- captive when seated: the lip stands above the lying can's axis --
        assert H_LIP > R_CAN + 0.008, "lip must rise above the seated can's axis"
        # -- nothing seats without compression; travel opens the gap past the can --
        assert SEAT_INTERF >= 0.012, "rest gap must be meaningfully shorter than the can"
        assert PLG_TRAVEL >= SEAT_INTERF + 0.012, \
            "plunger travel must open the gap past the can length with margin"
        # -- the compression threshold separates gravity-propped from genuinely seated --
        grav_comp = self.can_mass * 9.81 * (H_LIP / L_CAN) / self.spring_k
        assert grav_comp < 0.7 * self.comp_min, \
            f"gravity alone on a propped can ({grav_comp * 1000:.1f} mm) must sit well " \
            f"below comp_min"
        assert self.comp_min <= SEAT_INTERF - 0.006, \
            "comp_min must sit well below the genuine seated compression"
        assert self.comp_engage <= self.comp_min
        # -- seat preload within an arm's comfortable push --
        preload = self.spring_k * SEAT_INTERF
        assert 4.0 <= preload <= 15.0, f"seat preload {preload:.1f} N out of band"
        # -- null policy scores 0: cans spawn beyond the near-credit radius --
        assert self.can_radius - self.radius_jitter > self.near_r + 0.03, \
            "can spawn arc must start beyond the near-credit radius"
        # -- rubric windows consistent with geometry --
        assert self.seat_x_win[0] < LIP_X0 - L_CAN / 2 < self.seat_x_win[1], \
            "seated can center must sit inside the x window"
        assert self.seat_z_win[0] < R_CAN < self.seat_z_win[1]


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("spring_bay")
class SpringBayScene(BaseScene):
    cfg: SpringBaySceneCfg

    def __init__(self, cfg: SpringBaySceneCfg | None = None) -> None:
        super().__init__(cfg or SpringBaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        can_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_can_s, dynamic_friction=c.mu_can_d, restitution=0.0)
        can_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0, linear_damping=0.05, angular_damping=0.1)
        can_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        out: dict[str, Any] = {
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
            # NOTE: the bay MUST spawn before the plunger (the joint targets it).
            "bay": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bay",
                spawn=spawners["bay"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bay_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_body_s=c.mu_body_s, mu_body_d=c.mu_body_d,
                    mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "plunger": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plunger",
                spawn=spawners["plunger"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plg_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d,
                    spring_k=c.spring_k, spring_c=c.spring_c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(PLG_FACE_X - PLG_T / 2, 0.0, Z_F + 0.030)),
            ),
        }
        for name, color, x0 in (("can_blue", (0.15, 0.35, 0.85), 0.55),
                                ("can_red", (0.85, 0.12, 0.12), 0.75)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + name.split("_")[1],
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_l, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=can_mat, rigid_props=can_rigid,
                    collision_props=can_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x0, 0.60, c.can_l / 2 + 0.003)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
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
        self.bay: RigidObject = env.iscene["bay"]
        self.plunger: RigidObject = env.iscene["plunger"]
        self.blue: RigidObject = env.iscene["can_blue"]
        self.red: RigidObject = env.iscene["can_red"]
        self.env_origins = env.iscene.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the bay (heavy DYNAMIC teleport — the
        joint anchor follows), write the plunger CONSISTENTLY at its rest pose in the
        bay's new frame, and place each can upright on its own jittered arc slot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- bay: xy jitter + free yaw ---
        bxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.bay_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bay_yaw_max)
        q_bay = _qz(yaw)
        zeros = torch.zeros(m, device=dev)
        write(self.bay, torch.cat([bxy, zeros.unsqueeze(-1)], dim=-1), q_bay)

        # --- plunger: rest pose IN THE BAY'S NEW FRAME (consistent joint state) ---
        rest_local = torch.tensor([PLG_FACE_X - PLG_T / 2, 0.0, Z_F + 0.030],
                                  device=dev).expand(m, 3)
        plg_pos = torch.cat([bxy, zeros.unsqueeze(-1)], dim=-1) + _qapply(q_bay, rest_local)
        write(self.plunger, plg_pos, q_bay)

        # --- cans: upright on jittered world-frame arc slots around the bay ---
        for body, slot in ((self.blue, c.can_slots[0]), (self.red, c.can_slots[1])):
            ang = math.radians(slot) \
                + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.slot_jitter)
            rad = c.can_radius + (torch.rand(m, device=dev) * 2 - 1) * c.radius_jitter
            pos = torch.stack([bxy[:, 0] + rad * torch.cos(ang),
                               bxy[:, 1] + rad * torch.sin(ang),
                               torch.full((m,), c.can_l / 2 + 0.003, device=dev)], dim=-1)
            cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(body, pos, _qz(cyaw))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
                for nm in ("bay", "plunger", "blue", "red")}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("bay", "plunger", "blue", "red"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray dispenser bay stands on the floor: a knee-high block "
            f"({Z_F * 100:.0f} cm plinth) with an open-topped channel along its top — "
            f"two side walls {W_CH * 100:.1f} cm apart, an ORANGE retaining lip "
            f"{H_LIP * 100:.1f} cm tall closing the front end, and a dark spring-loaded "
            f"plunger pad facing the lip from the back end. The plunger rides a real "
            f"spring: it can be pushed back about {PLG_TRAVEL * 100:.0f} cm and shoves "
            f"back when released. At rest the free gap between the lip and the pad is "
            f"about {(L_CAN - SEAT_INTERF) * 100:.1f} cm — SHORTER than the cans are "
            f"long. Two identical food cans ({2 * R_CAN * 100:.1f} cm across, "
            f"{L_CAN * 100:.1f} cm tall, light enough to lift, narrow enough for a "
            f"parallel jaw) stand upright on the floor around the bay: one BLUE, one "
            f"RED. The bay's position and heading and both can positions change every "
            f"episode — read the scene by looking.\n"
            f"Goal: load the BLUE can into the bay so the spring holds it captive. "
            f"Because the gap is shorter than the can, it cannot simply be dropped in: "
            f"laid in casually it ends up propped at an angle with its front end "
            f"resting on top of the orange lip, and that counts for nothing. Lay the "
            f"blue can into the channel lengthwise and press it axially into the "
            f"plunger pad, compressing the spring at least ~1 cm, until the gap opens "
            f"enough for the raised front end to slip down behind the lip; then let "
            f"go. The spring must end up pressing the can flat against the back of "
            f"the lip — lying level in the channel, aligned with it, wedged by at "
            f"least {c.comp_min * 100:.0f} mm of spring compression, and at rest. "
            f"Finish with the RED can untouched or at least entirely OFF the bay: any "
            f"part of it resting on or in the bay fails the task. No particular order "
            f"is required."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Load the blue can into the dispenser bay: lay it in the channel, press "
            "it back against the spring plunger until its front end drops behind the "
            "orange lip, and release so the spring pins it flat and captive. Keep the "
            "red can off the bay."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def bay_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> bay frame."""
        return _qapply(_qinv(self.bay.data.root_quat_w), pos_w - self.bay.data.root_pos_w)

    def compression(self) -> torch.Tensor:
        """(N,) plunger spring compression in meters (readback of the joint's actual
        travel: rest-face x minus current-face x in the bay frame)."""
        loc = self.bay_local(self.plunger.data.root_pos_w)
        return ((PLG_FACE_X - PLG_T / 2) - loc[:, 0]).clamp(min=0.0)

    def can_axis_local(self, body) -> torch.Tensor:
        """(N,3) the can's cylinder axis (its local +z) expressed in the bay frame."""
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        axis_w = _qapply(body.data.root_quat_w, ez)
        return _qapply(_qinv(self.bay.data.root_quat_w), axis_w)

    def in_channel_loose(self, body) -> torch.Tensor:
        """(N,) bool: body center inside the loose channel volume (score credit)."""
        c = self.cfg
        loc = self.bay_local(body.data.root_pos_w)
        return (loc[:, 0] >= c.chan_x_win[0]) & (loc[:, 0] <= c.chan_x_win[1]) \
            & (loc[:, 1].abs() <= c.chan_y_max) \
            & (loc[:, 2] - Z_F >= c.chan_z_win[0]) & (loc[:, 2] - Z_F <= c.chan_z_win[1])

    def seated_geom(self) -> torch.Tensor:
        """(N,) bool: the blue can lies flat, centered and aligned in the channel."""
        c = self.cfg
        loc = self.bay_local(self.blue.data.root_pos_w)
        ax = self.can_axis_local(self.blue)
        return (loc[:, 0] >= c.seat_x_win[0]) & (loc[:, 0] <= c.seat_x_win[1]) \
            & (loc[:, 1].abs() <= c.seat_y_max) \
            & (loc[:, 2] - Z_F >= c.seat_z_win[0]) & (loc[:, 2] - Z_F <= c.seat_z_win[1]) \
            & (ax[:, 0].abs() >= math.cos(math.radians(c.align_max_deg)))

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def plunger_settled(self) -> torch.Tensor:
        return self.plunger.data.root_lin_vel_w.norm(dim=-1) < self.cfg.plg_settle_lin

    def decoy_clear(self) -> torch.Tensor:
        """(N,) bool: the red can is nowhere on/in the bay."""
        c = self.cfg
        loc = self.bay_local(self.red.data.root_pos_w)
        on_bay = (loc[:, 0].abs() <= c.decoy_x_max) & (loc[:, 1].abs() <= c.decoy_y_max) \
            & (loc[:, 2] >= c.decoy_z_min)
        return ~on_bay

    def near_bay(self, body) -> torch.Tensor:
        d = (body.data.root_pos_w[:, :2] - self.bay.data.root_pos_w[:, :2]).norm(dim=-1)
        return d < self.cfg.near_r

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: blue can seated flat + aligned in the channel, spring genuinely
        compressed >= comp_min through can contact, can and plunger at rest, red decoy
        entirely off the bay."""
        return self.seated_geom() & (self.compression() >= self.cfg.comp_min) \
            & self.settled(self.blue) & self.plunger_settled() & self.decoy_clear()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 blue can near the bay + 0.15 inside the loose
        channel volume + 0.30 in-channel with the spring engaged, capped at 0.55;
        1.0 iff success(). Stateless and monotone along the intended solution
        (approach -> lay in -> press -> seated); the null policy scores ~0 (cans
        spawn beyond the near radius)."""
        c = self.cfg
        near = self.near_bay(self.blue).float()
        chan = self.in_channel_loose(self.blue).float()
        engaged = (self.compression() >= c.comp_engage).float() * chan
        base = (0.10 * near + 0.15 * chan + 0.30 * engaged).clamp(max=0.55)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="spring_bay", robot="null", env_spacing=3.0))
