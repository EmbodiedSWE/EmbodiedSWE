"""GantryStampScene — operate a 2-axis gantry plotter to stamp the red pads
(sim_gen task `draw_svg_i388`).

Derived from maniskill/draw_svg — "drag a red marker cube along a prescribed SVG
path": the arm holds a free object and its own continuous contact-maintained
trajectory IS the product. STRATEGICALLY different: here the marking tool is
CAPTIVE in a machine and the robot can only reach the targets THROUGH the
machine's kinematics. A Cartesian gantry stands over a board: a BRIDGE beam
slides along X on side rails, a CARRIAGE slides along Y on the bridge, and a
spring-returned STYLUS plunges along Z through the carriage. Five printed pads
lie on the board at randomized positions — three RED (targets) and two GRAY
(decoys). The goal is to stamp every red pad: position the stylus tip over the
pad by sliding the two axes, then press the stylus head down against its return
spring until the tip dwells on the pad face; one full press on a gray pad is a
PERMANENT foul and the episode can never succeed. There is no path to trace, no
free object to hold: the solver must READ the pad layout, DECOMPOSE each target
into bridge/carriage coordinates (the tip rides a fixed forward offset ahead of
the carriage), settle each axis, and execute discrete spring-loaded press
strokes — an operate-the-mechanism plan with a foul constraint, not a
trajectory-following plan.

Success(): all three red pads stamp-LATCHED (each latch requires the tip
sustained on the pad face: xy within `stamp_r` of the pad centre AND tip bottom
within `stamp_dz` of the pad top for `press_steps` consecutive sim steps), NO
decoy ever stamped (foul latch), the stylus back up at its retracted rest
(spring-returned, tip above `retract_z`), and the machine settled. score():
latched 0.25 per red pad stamped (monotone, travels through get/set_state),
1.0 iff success(). Null policy = 0.0 (the spring holds the stylus 19 mm above
the pads; nothing ever reaches stamp depth).

Assets are fully procedural (compound spawners; joints authored at spawn):
  - frame (heavy DYNAMIC 40 kg fixture — never kinematic, so every joint anchor
    follows the reset teleport): dark board slab + two side rails.
  - bridge: steel-blue beam spanning the board on a PrismaticJoint (axis X) into
    the frame; a green MAST post on top is the push/grasp handle.
  - carriage: dark-blue block under the bridge on a PrismaticJoint (axis Y) into
    the bridge; a yellow KNOB post rises behind it (parallel-jaw graspable).
  - stylus: silver shaft + orange press HEAD on a PrismaticJoint (axis Z) into
    the carriage, with a linear DriveAPI spring (stiffness `spring_k`, target
    above the upper stop) that parks it retracted; pressing the head ~6 N
    plunges the tip onto the board plane. The tip rides `tip_off` ahead (+x,
    frame frame) of the carriage centre.
  - pads: five thin KINEMATIC plates (55 mm square): three red, two gray.

Per-episode randomization (readback-verified in smoke): frame xy jitter + yaw,
the 5 pads over a random 5-of-6 slot permutation with per-pad jitter, and the
machine's initial configuration (bridge x, carriage y) — so no memorized
coordinate sequence works: the layout AND the starting configuration change.

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

if TYPE_CHECKING:
    from robobench.core import BaseEnv


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
BOARD_X = 0.56  # board slab footprint
BOARD_Y = 0.46
BOARD_T = 0.030  # slab thickness -> board top at z = BOARD_T
RAIL_Y = 0.27  # side-rail centreline (frame local |y|)
RAIL_W = 0.04
RAIL_H = 0.20  # rail top at BOARD_T + RAIL_H = 0.230 (5 mm below the bridge belly)
BRIDGE_Z = 0.26  # bridge beam centre height
BRIDGE_LEN = 0.60  # beam length (along y)
BRIDGE_SQ = 0.05  # beam cross-section
MAST_S = 0.022  # bridge handle post square
MAST_H = 0.11  # ... height (top at 0.395)
BR_LO = -0.20  # bridge joint travel (frame local x, relative to spawn anchor)
BR_HI = 0.08
CAR_X = 0.09  # carriage block
CAR_Y = 0.08
CAR_Z_SZ = 0.08
CAR_Z = 0.19  # carriage centre height (top 0.230, bridge belly 0.235)
CAR_LIM = 0.11  # carriage joint travel (bridge local +/- y)
KNOB_S = 0.018  # carriage handle post square (parallel-jaw pinch)
KNOB_H = 0.10  # ... (top at 0.330), at carriage local (-KNOB_OFF, 0)
KNOB_OFF = 0.060
TIP_OFF = 0.060  # stylus axis rides this far AHEAD (+x) of the carriage centre
SHAFT_R = 0.008  # stylus shaft
SHAFT_H = 0.26  # stylus body origin = shaft centre -> tip bottom = origin - SHAFT_H/2
HEAD_R = 0.028  # orange press head (disc on top of the shaft)
HEAD_H = 0.014
TIP_DROP = SHAFT_H / 2  # tip bottom below the stylus body origin
TIP_UP_Z = 0.056  # RETRACTED tip-bottom height (frame local z); spawn stylus origin at
STYLUS_Z0 = TIP_UP_Z + TIP_DROP  # ... 0.186
STROKE = 0.027  # plunge stroke: joint limits [-STROKE, +0.001]
PAD_S = 0.055  # pad plate square
PAD_T = 0.006
PAD_Z = BOARD_T + PAD_T / 2 + 0.001  # pad centre: 1 mm float above the board (kinematic)
SLOT_X = (-0.10, 0.0, 0.10)  # 3 x 2 slot grid the 5 pads occupy (5-of-6 permutation)
SLOT_Y = (-0.065, 0.065)
PAD_JIT = 0.012  # per-pad slot jitter
N_RED = 3
N_GRAY = 2
PAD_NAMES = ("pad_red0", "pad_red1", "pad_red2", "pad_gray0", "pad_gray1")


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    """Author one colliding z-cylinder child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                *, iters: int = 16, vel_iters: int = 4, damp: float = 0.0,
                ang_damp: float | None = None):
    """Author a dynamic compound-body root with zeroed sleep (a sleeping fixture
    would freeze its joint anchors)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(1.0)
    pxrb.CreateSolverPositionIterationCountAttr(iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    if damp:
        pxrb.CreateLinearDampingAttr(float(damp))
        pxrb.CreateAngularDampingAttr(float(ang_damp if ang_damp is not None else damp))
    return root


def _prismatic(stage, path: str, body0: str, body1: str, axis: str,
               pos0, pos1, lo: float, hi: float):
    from pxr import Gf, UsdPhysics

    j = UsdPhysics.PrismaticJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in pos1]))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(lo))
    j.CreateUpperLimitAttr(float(hi))
    return j


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The plotter base: dark board slab + two side rails. One heavy DYNAMIC
    compound body (joint anchors follow reset teleports). Origin = board centre
    on the ground."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation,
                       cfg.mass_props.mass, damp=0.5)
    mat = _friction_material(stage, f"{prim_path}/frame_mat", 0.6, 0.55)
    _box(stage, f"{prim_path}/board", (BOARD_X, BOARD_Y, BOARD_T),
         (0.0, 0.0, BOARD_T / 2), (0.16, 0.17, 0.19), 0.002, material=mat)
    for tag, sy in (("n", 1.0), ("s", -1.0)):
        _box(stage, f"{prim_path}/rail_{tag}", (BOARD_X, RAIL_W, RAIL_H),
             (0.0, sy * RAIL_Y, BOARD_T + RAIL_H / 2), (0.45, 0.47, 0.50),
             0.002, material=mat)
    return root


def _spawn_bridge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The X-axis: steel-blue beam spanning the board + green mast handle, on a
    PrismaticJoint (axis X) into the sibling frame. Origin = beam centre."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation,
                       cfg.mass_props.mass, damp=cfg.damp)
    mat = _friction_material(stage, f"{prim_path}/bridge_mat", 0.3, 0.25)
    _box(stage, f"{prim_path}/beam", (BRIDGE_SQ, BRIDGE_LEN, BRIDGE_SQ),
         (0.0, 0.0, 0.0), (0.25, 0.38, 0.60), 0.002, material=mat)
    _box(stage, f"{prim_path}/mast", (MAST_S, MAST_S, MAST_H),
         (0.0, 0.0, BRIDGE_SQ / 2 + MAST_H / 2), (0.12, 0.62, 0.22),
         0.002, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    _prismatic(stage, f"{prim_path}/slide_x", f"{base}/Frame", prim_path, "X",
               (0.0, 0.0, BRIDGE_Z), (0.0, 0.0, 0.0), BR_LO, BR_HI)
    return root


def _spawn_carriage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The Y-axis: dark-blue block under the bridge + yellow knob handle, on a
    PrismaticJoint (axis Y) into the sibling bridge. Origin = block centre."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation,
                       cfg.mass_props.mass, damp=cfg.damp)
    mat = _friction_material(stage, f"{prim_path}/car_mat", 0.3, 0.25)
    _box(stage, f"{prim_path}/block", (CAR_X, CAR_Y, CAR_Z_SZ),
         (0.0, 0.0, 0.0), (0.13, 0.16, 0.34), 0.002, material=mat)
    _box(stage, f"{prim_path}/knob", (KNOB_S, KNOB_S, KNOB_H),
         (-KNOB_OFF, 0.0, CAR_Z_SZ / 2 + KNOB_H / 2), (0.92, 0.80, 0.10),
         0.002, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    _prismatic(stage, f"{prim_path}/slide_y", f"{base}/Bridge", prim_path, "Y",
               (0.0, 0.0, CAR_Z - BRIDGE_Z), (0.0, 0.0, 0.0), -CAR_LIM, CAR_LIM)
    return root


def _spawn_stylus(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The Z-axis: silver shaft + orange press head on a PrismaticJoint (axis Z)
    into the sibling carriage, with a linear DriveAPI return spring that parks
    it retracted. Origin = shaft centre (tip bottom = origin - TIP_DROP)."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation,
                       cfg.mass_props.mass, damp=0.2)
    mat = _friction_material(stage, f"{prim_path}/sty_mat", 0.8, 0.75)
    _cyl(stage, f"{prim_path}/shaft", SHAFT_R, SHAFT_H, (0.0, 0.0, 0.0),
         (0.75, 0.76, 0.78), 0.002, material=mat)
    _cyl(stage, f"{prim_path}/head", HEAD_R, HEAD_H,
         (0.0, 0.0, SHAFT_H / 2 + HEAD_H / 2), (0.90, 0.45, 0.08),
         0.002, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = _prismatic(stage, f"{prim_path}/plunge", f"{base}/Carriage", prim_path, "Z",
                   (TIP_OFF, 0.0, STYLUS_Z0 - CAR_Z), (0.0, 0.0, 0.0),
                   -STROKE, 0.001)
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(cfg.spring_k))
    drv.CreateDampingAttr(float(cfg.spring_c))
    drv.CreateTargetPositionAttr(float(cfg.spring_target))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)

        @configclass
        class BridgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bridge)
            damp: float = 3.0

        @configclass
        class CarriageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carriage)
            damp: float = 3.0

        @configclass
        class StylusSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stylus)
            spring_k: float = 90.0
            spring_c: float = 3.0
            spring_target: float = 0.008

        _SPAWNER_CACHE.update(frame=FrameSpawnerCfg, bridge=BridgeSpawnerCfg,
                              carriage=CarriageSpawnerCfg, stylus=StylusSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GantryStampSceneCfg(BaseCfg):
    """Config for `GantryStampScene`. Honesty knobs are asserted in
    `__post_init__`: the retracted tip clears the pads by a wide margin (null
    policy stamps nothing), stamp depth is physically reachable only through a
    real press stroke onto real pad contact, adjacent pads can never be confused
    within the stamp tolerance, and the sliders' travel covers every slot."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    stamp_r: float = tunable(0.018)  # tip xy within this of the pad centre
    stamp_dz: float = tunable(0.0025)  # tip bottom within this of the pad TOP face
    press_steps: int = tunable(8)  # consecutive sim steps at stamp depth to latch
    retract_z: float = tunable(0.050)  # success: tip bottom back above this (frame local)
    settle_lin: float = tunable(0.08)  # machine settle gate (m/s; above the GPU
    # phantom-velocity band so a genuinely parked machine judges settled)
    settle_ang: float = tunable(1.0)  # (rad/s)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    frame_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the frame at reset
    frame_yaw_deg: float = tunable(20.0)  # uniform +/- frame yaw (deg)
    br0_range: tuple = tunable((-0.17, 0.05))  # initial bridge coordinate range
    car0_range: tuple = tunable((-0.095, 0.095))  # initial carriage coordinate range

    # --- info: structure -----------------------------------------------------------------------
    frame_mass: float = info(40.0)  # heavy dynamic fixture (anchors follow teleports)
    bridge_mass: float = info(0.80)
    carriage_mass: float = info(0.25)
    stylus_mass: float = info(0.04)
    pad_mass: float = info(0.20)  # kinematic; mass is bookkeeping only
    spring_k: float = info(90.0)  # stylus return-spring stiffness (N/m)
    spring_c: float = info(3.0)  # ... damping
    spring_target: float = info(0.008)  # drive target ABOVE the upper stop -> preload
    press_force_ref: float = info(6.0)  # reference press force the solution uses (N)

    # Derived (filled in __post_init__).
    pad_top: float = field(default=None, init=False)  # pad TOP face height (frame local)

    def __post_init__(self) -> None:
        self.pad_top = PAD_Z + PAD_T / 2
        g = 9.81
        w_sty = self.stylus_mass * g
        # -- spring parks the stylus retracted: preload at the upper stop beats weight --
        preload = self.spring_k * (self.spring_target - 0.001)
        assert preload > 1.3 * w_sty, \
            f"spring preload {preload:.2f} N must hold the stylus up ({w_sty:.2f} N)"
        # -- and a comfortable arm press overcomes it through the full stroke --
        f_full = self.spring_k * (self.spring_target + STROKE) + w_sty
        assert f_full < 0.8 * self.press_force_ref + 2.0, \
            f"full-stroke press force {f_full:.2f} N must stay comfortably below 8 N"
        assert f_full > 1.5 * w_sty, "the spring must be a real spring, not decoration"
        # -- retracted tip clears the pads (null policy can never stamp) --
        clear = TIP_UP_Z - self.pad_top
        assert clear > 0.015, f"traverse clearance {clear * 1000:.1f} mm too small"
        assert TIP_UP_Z - 0.004 > self.retract_z > self.pad_top + self.stamp_dz + 0.005, \
            "retract gate must sit between stamp depth and the retracted rest"
        # -- stamp depth is reachable ONLY by a real press: stroke bottoms past the pad --
        assert TIP_UP_Z - STROKE < self.pad_top - 0.004, \
            "plunge stroke must carry the tip onto the pad face with >= 4 mm margin"
        assert TIP_UP_Z - STROKE > BOARD_T - 0.0015, \
            "joint stop must sit at/below board contact, not above it"
        # -- adjacent pads can never be confused within the stamp tolerance --
        min_slot = min(SLOT_X[1] - SLOT_X[0], SLOT_Y[1] - SLOT_Y[0])
        assert min_slot - 2 * PAD_JIT > PAD_S + 0.015, "pads must never overlap"
        assert min_slot - 2 * PAD_JIT > 2 * self.stamp_r + PAD_S / 2, \
            "a press inside one pad's tolerance must be far outside every other pad's"
        # -- slider travel covers every slot (tip = bridge + TIP_OFF; carriage = pad y) --
        need_lo = SLOT_X[0] - PAD_JIT - TIP_OFF
        need_hi = SLOT_X[-1] + PAD_JIT - TIP_OFF
        assert BR_LO < need_lo - 0.005 and BR_HI > need_hi + 0.005, \
            f"bridge travel [{BR_LO},{BR_HI}] must cover [{need_lo:.3f},{need_hi:.3f}]"
        assert CAR_LIM > SLOT_Y[-1] + PAD_JIT + 0.005, "carriage travel must cover the slots"
        assert self.br0_range[0] > BR_LO + 0.02 and self.br0_range[1] < BR_HI - 0.02
        assert abs(self.car0_range[0]) < CAR_LIM - 0.01
        # -- embodiment: handles are jaw-sized, nothing above comfortable reach height --
        assert MAST_S < 0.05 and KNOB_S < 0.05 and 2 * HEAD_R < 0.07
        assert BRIDGE_Z + BRIDGE_SQ / 2 + MAST_H < 0.42, "handles stay below 0.42 m"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gantry_stamp")
class GantryStampScene(BaseScene):
    cfg: GantryStampSceneCfg

    def __init__(self, cfg: GantryStampSceneCfg | None = None) -> None:
        super().__init__(cfg or GantryStampSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        mass = sim_utils.MassPropertiesCfg
        rigid = sim_utils.RigidBodyPropertiesCfg

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # NOTE: spawn order matters — each joint targets its already-spawned parent.
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=sp["frame"](mass_props=mass(mass=c.frame_mass), rigid_props=rigid()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "bridge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bridge",
                spawn=sp["bridge"](mass_props=mass(mass=c.bridge_mass), rigid_props=rigid()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BRIDGE_Z)),
            ),
            "carriage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carriage",
                spawn=sp["carriage"](mass_props=mass(mass=c.carriage_mass),
                                     rigid_props=rigid()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CAR_Z)),
            ),
            "stylus": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stylus",
                spawn=sp["stylus"](mass_props=mass(mass=c.stylus_mass), rigid_props=rigid(),
                                   spring_k=c.spring_k, spring_c=c.spring_c,
                                   spring_target=c.spring_target),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(TIP_OFF, 0.0, STYLUS_Z0)),
            ),
        }
        for i, name in enumerate(PAD_NAMES):
            red = i < N_RED
            color = (0.85, 0.10, 0.10) if red else (0.62, 0.62, 0.64)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad_" + name.split("_")[1],
                spawn=sim_utils.CuboidCfg(
                    size=(PAD_S, PAD_S, PAD_T),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=mass(mass=c.pad_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(SLOT_X[i % 3], SLOT_Y[i // 3], PAD_Z)),
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
        for nm in ("frame", "bridge", "carriage", "stylus") + PAD_NAMES:
            setattr(self, nm, env.iscene[nm])
        self.pads = [getattr(self, nm) for nm in PAD_NAMES]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # per-pad sustained-press counters + stamp latches (targets AND decoys)
        self.press_ctr = torch.zeros(n, len(PAD_NAMES), dtype=torch.long, device=dev)
        self.stamp_latch = torch.zeros(n, len(PAD_NAMES), dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + yaw the frame (heavy DYNAMIC teleport — every
        joint anchor follows), write the WHOLE machine chain coherently in the
        frame's new pose (random initial bridge/carriage coordinates, stylus
        retracted), scatter the 5 pads over a random 5-of-6 slot permutation
        with per-pad jitter, and clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(3, device=dev)  # burn post-seed draws (degenerate-first-draw trap)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- frame: xy jitter + yaw ---
        fxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.frame_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.frame_yaw_deg)
        q = _qz(yaw)
        z0 = torch.zeros(m, 1, device=dev)
        fpos = torch.cat([fxy, z0], dim=-1)
        write(self.frame, fpos, q)

        # --- machine chain, coherently in the frame's new pose ---
        br0 = c.br0_range[0] + torch.rand(m, device=dev) * (c.br0_range[1] - c.br0_range[0])
        car0 = c.car0_range[0] + torch.rand(m, device=dev) * (c.car0_range[1] - c.car0_range[0])
        zeros = torch.zeros(m, device=dev)
        write(self.bridge, fpos + _qapply(q, torch.stack(
            [br0, zeros, torch.full((m,), BRIDGE_Z, device=dev)], dim=-1)), q)
        write(self.carriage, fpos + _qapply(q, torch.stack(
            [br0, car0, torch.full((m,), CAR_Z, device=dev)], dim=-1)), q)
        write(self.stylus, fpos + _qapply(q, torch.stack(
            [br0 + TIP_OFF, car0, torch.full((m,), STYLUS_Z0, device=dev)], dim=-1)), q)

        # --- pads: random 5-of-6 slot permutation + per-pad jitter (frame frame) ---
        slots = torch.tensor([(sx, sy) for sy in SLOT_Y for sx in SLOT_X], device=dev)
        perm = torch.rand(m, slots.shape[0], device=dev).argsort(dim=1)  # (m, 6)
        for i, name in enumerate(PAD_NAMES):
            sl = slots[perm[:, i]]  # (m, 2)
            pxy = sl + (torch.rand(m, 2, device=dev) * 2 - 1) * PAD_JIT
            local = torch.cat([pxy, torch.full((m, 1), PAD_Z, device=dev)], dim=-1)
            write(getattr(self, name), fpos + _qapply(q, local), q)

        # --- clear the latches ---
        self.press_ctr[env_ids] = 0
        self.stamp_latch[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    _BODIES = ("frame", "bridge", "carriage", "stylus") + PAD_NAMES

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
               for nm in self._BODIES}
        out["press_ctr"] = self.press_ctr[env_ids].clone()
        out["stamp_latch"] = self.stamp_latch[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in self._BODIES:
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        self.press_ctr[env_ids] = state["press_ctr"]
        self.stamp_latch[env_ids] = state["stamp_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A Cartesian gantry plotter stands on the floor: a dark board "
            f"({BOARD_X * 100:.0f} x {BOARD_Y * 100:.0f} cm) between two gray side "
            f"rails, and a steel-blue BRIDGE beam spanning the board "
            f"{BRIDGE_Z * 100:.0f} cm up. The bridge slides only along the board's "
            f"LONG axis (call it X); a short green MAST on the bridge is its handle. "
            f"A dark-blue CARRIAGE hangs under the bridge and slides only along the "
            f"bridge (Y); the yellow KNOB post on its back is its handle. Through the "
            f"carriage's front runs a vertical silver STYLUS with an orange press "
            f"HEAD on top: a return spring holds its tip about "
            f"{(TIP_UP_Z - c.pad_top) * 1000:.0f} mm above the board, and pressing "
            f"the head down (about {c.press_force_ref:.0f} N) plunges the tip onto "
            f"the board. The stylus tip sits a fixed {TIP_OFF * 1000:.0f} mm ahead "
            f"of the carriage centre, on the bridge's +X side. On the board lie five "
            f"printed pads ({PAD_S * 1000:.0f} mm squares): THREE RED and TWO GRAY. "
            f"They are marks on the board — they cannot be moved. The machine's "
            f"position and heading, the pad layout, and the bridge/carriage starting "
            f"position change every episode: read the scene by looking.\n"
            f"Goal: STAMP ALL THREE RED PADS with the stylus. For each red pad, "
            f"slide the bridge and the carriage (push or pull their handles — the "
            f"pads themselves cannot be touched usefully) until the stylus tip is "
            f"directly over the pad (within {c.stamp_r * 1000:.0f} mm of its "
            f"centre), then press the stylus head straight down and hold briefly "
            f"until the tip dwells on the pad face; then let the spring lift it "
            f"back up. A press only counts at full depth — a partial press, or a "
            f"press next to a pad, does nothing. NEVER press the stylus down over a "
            f"GRAY pad: one full-depth press on a gray pad is a permanent foul and "
            f"the task can no longer be completed (traversing above gray pads with "
            f"the tip up is safe). Finish with the stylus released (tip back up at "
            f"its spring rest) and the machine at rest. Order does not matter."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Operate the gantry plotter: slide the bridge and carriage by their "
            "handles to place the stylus tip over each RED pad, and press the "
            "stylus head down until the tip stamps the pad — all three red pads. "
            "Never press the stylus down over a gray pad; one press on gray fails "
            "the task. Finish with the stylus released and the machine at rest."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def frame_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> frame frame."""
        return _qapply(_qinv(self.frame.data.root_quat_w),
                       pos_w - self.frame.data.root_pos_w)

    def machine_coords(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(N,) bridge coordinate (frame-local x), (N,) carriage coordinate
        (frame-local y), (N,) tip-bottom height (frame-local z)."""
        b = self.frame_local(self.bridge.data.root_pos_w)
        s = self.frame_local(self.stylus.data.root_pos_w)
        return b[:, 0], s[:, 1], s[:, 2] - TIP_DROP

    def pad_layout(self) -> torch.Tensor:
        """(N, 5, 2) pad centres in the frame frame (red 0..2, gray 3..4)."""
        return torch.stack([self.frame_local(p.data.root_pos_w)[:, :2]
                            for p in self.pads], dim=1)

    def tip_on_pad(self) -> torch.Tensor:
        """(N, 5) bool, geometric: stylus tip at stamp depth on each pad — xy
        within `stamp_r` of the pad centre AND tip bottom within `stamp_dz` of
        the pad top face (world-frame; both bodies share the frame's yaw)."""
        c = self.cfg
        tip_xy = self.stylus.data.root_pos_w[:, :2]
        tip_z = self.stylus.data.root_pos_w[:, 2] - TIP_DROP
        pad_xy = torch.stack([p.data.root_pos_w[:, :2] for p in self.pads], dim=1)
        pad_top = torch.stack([p.data.root_pos_w[:, 2] for p in self.pads], dim=1) \
            + PAD_T / 2
        near = (tip_xy.unsqueeze(1) - pad_xy).norm(dim=-1) < c.stamp_r
        deep = tip_z.unsqueeze(1) <= pad_top + c.stamp_dz
        return near & deep

    def stylus_retracted(self) -> torch.Tensor:
        """(N,) bool: tip bottom back above `retract_z` (spring rest)."""
        _b, _c, tz = self.machine_coords()
        return tz >= self.cfg.retract_z

    def settled(self) -> torch.Tensor:
        """(N,) bool: bridge, carriage and stylus below the settle gates."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.bridge, self.carriage, self.stylus):
            ok &= (b.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)
        return ok

    def fouled(self) -> torch.Tensor:
        """(N,) bool: any GRAY pad ever stamped (permanent)."""
        return self.stamp_latch[:, N_RED:].any(dim=1)

    def stamped_red(self) -> torch.Tensor:
        """(N,) count of red pads stamp-latched."""
        return self.stamp_latch[:, :N_RED].sum(dim=1)

    # ----- step-coupled latches ---------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Sustained-press latching at sim rate: a pad latches after
        `press_steps` CONSECUTIVE steps with the tip at stamp depth on it
        (targets and decoys alike — decoy latches are the foul)."""
        on = self.tip_on_pad()
        self.press_ctr = torch.where(on, self.press_ctr + 1,
                                     torch.zeros_like(self.press_ctr))
        self.stamp_latch |= self.press_ctr >= self.cfg.press_steps

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three red pads stamped, no gray pad ever stamped,
        stylus back at its retracted rest, machine settled."""
        return (self.stamped_red() == N_RED) & ~self.fouled() \
            & self.stylus_retracted() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched 0.25 per red pad stamped (monotone,
        travels through get/set_state); 1.0 iff success(). A foul permanently
        caps the score at 0.75 because success can never hold. Null policy 0.0
        (the spring holds the tip far above stamp depth)."""
        base = 0.25 * self.stamped_red().float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="gantry_stamp", robot="null", env_spacing=3.0))
