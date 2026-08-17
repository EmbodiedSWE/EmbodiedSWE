"""HoodPropScene — prop a falling chest lid half-open with a peg standing in a socket
(sim_gen task `lift_peg_upright_i116`).

Derived from maniskill/lift_peg_upright, but STRATEGICALLY different: the seed is one
terminal reorientation — grasp a peg lying flat, pitch it 90 degrees, stand it free on
the table; success is a pose readout (height + axis alignment) on the manipulated peg
itself, and the episode ends the instant the peg stands. Here an upright peg is never
the goal — it is a LOAD-BEARING TOOL inside a three-stage, mechanically ordered
mechanism, and the judged outcome is the angle of a body the peg holds up. A chest has
a heavy lid riding a captive contact hinge (a capsule axle rocking in capped pockets —
no joint). The hinge is slick: anywhere below vertical the lid falls shut; pushed past
vertical it rests against a back-stop (the temporary hold a single one-gripper arm
needs). The sill socket that receives the peg is COVERED by the closed lid, so the
order is forced by geometry: (1) swing the lid past vertical onto the back-stop,
(2) stand the LONG red peg upright in the sill socket, (3) pull the lid off the stop
and lower it until it rests ON the peg — propped half-open in a band of angles
[`band_min`, `band_max`] that is gravitationally unstable without support (constructed
either side: the back-stop rest angle is ~105 deg > band_max, and an unsupported lid
posed inside the band falls shut — smoke proves both). A SHORT blue peg is a decoy: it
fits the socket but props the lid at only ~31 deg < band_min (the near-miss).

The seed's whole strategy — stand the peg upright on a free surface, done — is
exactly this task's null outcome: a peg standing anywhere outside the socket (or in a
scene whose lid is closed) scores ~0 and is explicitly rejected in smoke.

Strategy vs the corpus tasks read this session: `pen_holder` fills a container with
pens tip-up (repeated containment); `beam_scale` (pull_cube_i20) banks weights in a
pan to tip a beam — mass accumulation, robot never touches the tilting body, no
ordering; `hasp_pin_link` (peg_insertion_side_i2, TASK.md) aligns a bar then threads
a pin through stacked holes — a fastened linkage, nothing is propped and nothing
falls; `latch_vault` (screw_nail_i59, TASK.md) is an unlock-uncover-retrieve
disassembly chain. Here the manipulated body (lid) is DIRECTLY driven through a
bistable pivot, the goal band is reachable only through a support interaction with a
second manipulated body, and the required order is enforced by the lid physically
covering the peg's socket.

success(): lid propped in the angle band, still riding its hinge, the LONG peg
upright in the socket under it, everything settled. score(): latched stage credit —
0.15 for having swung the lid past vertical (only on episodes that START closed;
some episodes start with the lid already parked on the back-stop and get no free
credit), +0.25 for the long peg standing in the socket, +0.20 once the propped band
is occupied with the peg in place, capped at 0.60; 1.0 iff success(). Null policy
scores ~0 in both start states.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - chest (KINEMATIC): sill floor + front/side walls (the closed lid's rest rim),
    two hinge towers with capped, slick pockets that trap the lid's axle (it can
    rock 0..~105 deg, not escape), a back-stop bar on posts behind the hinge, and a
    raised socket collar on the sill at `sock_x` from the hinge axis.
  - lid (dynamic, 0.5 kg): plate + capsule axle (the hinge pin) + an underside cleat
    ridge just outboard of where the long peg's top lands (blocks the prop's top
    from skating forward; the collar blocks its foot). CoM and diagonal inertia
    authored explicitly (MassAPI alone would leave the CoM at the axle).
  - peg_long (red, 32 x 32 x 240 mm, 60 g) and peg_short (blue, 32 x 32 x 120 mm,
    35 g): both fit the parallel jaw and the socket; only the long one props inside
    the band.

Per-episode randomization (readback-verified in smoke): chest xy jitter + free yaw,
both pegs scattered on a jittered ring around the chest (radius + angle + free yaw,
lying FLAT), and the lid's START state — closed (rim rest) or already parked open on
the back-stop (the solver must read it and skip stage 1). Heavy imports (isaaclab,
pxr) are deferred so importing this module stays app-free.
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
_HINGE_Z = 0.060  # axle axis height above ground (chest frame z)
_LID_LEN = 0.30  # plate x-extent from the hinge axis to the front edge
_LID_W = 0.26  # plate y-width
_LID_T = 0.016  # plate thickness -> underside plane at lid-frame z = -0.008
_AXLE_R = 0.012  # hinge pin (capsule) radius
_AXLE_TIP = 0.175  # capsule tip |y| (rests in the tower pockets at |y| ~ 0.17)
_CLEAT_XC = 0.252  # cleat ridge center x (inner face 0.246, just outboard of the prop contact)
_CLEAT_T = 0.012  # cleat x-thickness and drop below the underside

_SILL_TOP = 0.012  # sill floor top (chest frame)
_WALL_TOP = 0.050  # front/side wall rim top — the closed lid's rest
_SOCK_X = 0.150  # socket collar center, x from the hinge axis
_SOCK_IN = 0.021  # collar cavity half-width (inner faces at +/-0.021)
_SOCK_H = 0.014  # collar wall height (sits on the sill: z 0.012..0.026)
_STOP_BAR = (-0.057, 0.190)  # back-stop bar center (x, z); size (0.024, 0.36, 0.024)

_PEG_W = 0.032  # peg square cross-section (fits the 80 mm jaw and the 42 mm socket)
_PEG_L_LONG = 0.240
_PEG_L_SHORT = 0.120
_LID_MASS = 0.5
_LID_COM_X = 0.152  # plate centroid (the authored CoM; MassAPI alone would leave it at the axle)


def _prop_angle(peg_len: float) -> float:
    """Lid angle (deg) at which the underside meets the top inboard corner of a peg of
    `peg_len` standing in the socket: solve  x_c*sin(t) = (z_top - hinge_z)*cos(t) + 0.008
    with x_c the peg's inboard face x and z_top the peg top height."""
    x_c = _SOCK_X - _PEG_W / 2
    dz = _SILL_TOP + peg_len - _HINGE_Z
    lo, hi = 0.0, math.pi / 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if x_c * math.sin(mid) - dz * math.cos(mid) - _LID_T / 2 < 0.0:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi))


def _stop_angle() -> float:
    """Lid angle (deg) at which the plate's TOP face rests on the back-stop bar's
    front-top corner: solve  -dx*sin(t) + dz*cos(t) = t_half  past 90 deg."""
    dx, dz = _STOP_BAR[0] + 0.012, _STOP_BAR[1] + 0.012 - _HINGE_Z  # corner (-0.045, 0.142)
    lo, hi = math.pi / 2, math.pi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if -dx * math.sin(mid) + dz * math.cos(mid) - _LID_T / 2 > 0.0:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi))


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
    """One USD physics material (friction is load-bearing here: a slick hinge makes
    the lid fall shut reliably; grippy plate/peg faces make the prop hold)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
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


def _spawn_chest(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC chest: sill + rim walls + capped hinge pockets + back-stop
    + socket collar. Origin = the hinge axis projected to the ground; +x toward the
    front wall; the axle axis runs along y at (0, *, _HINGE_Z)."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    frame = _friction_material(stage, f"{prim_path}/frame_mat", cfg.mu_frame_s, cfg.mu_frame_d)
    slick = _friction_material(stage, f"{prim_path}/hinge_mat", cfg.mu_hinge_s, cfg.mu_hinge_d)
    col = (0.45, 0.33, 0.22)  # wooden chest
    dark = (0.30, 0.30, 0.34)
    green = (0.15, 0.50, 0.20)

    # sill floor: x 0.02..0.31, y +/-0.17, top at _SILL_TOP
    _box(stage, f"{prim_path}/sill", (0.29, 0.34, 0.012), (0.165, 0.0, 0.006), col,
         0.0015, material=frame)
    # front wall: inner face x=0.28, rim top _WALL_TOP
    _box(stage, f"{prim_path}/wall_f", (0.03, 0.34, _WALL_TOP), (0.295, 0.0, _WALL_TOP / 2),
         col, 0.0015, material=frame)
    # side walls: inner faces |y|=0.15 (lid half-width 0.13 -> 2 cm clearance)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (0.29, 0.02, _WALL_TOP),
             (0.165, sy * 0.16, _WALL_TOP / 2), col, 0.0015, material=frame)
    # hinge towers at y=+/-0.17: pocket floor top 0.048 (axle center rests at 0.060)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        y = sy * 0.17
        _box(stage, f"{prim_path}/tower_{tag}", (0.06, 0.03, 0.048), (0.0, y, 0.024),
             col, 0.001, material=slick)
        # pocket x-walls: cavity |x| < 0.016 (axle r 0.012 -> 4 mm play each side)
        _box(stage, f"{prim_path}/pkt_xp_{tag}", (0.012, 0.03, 0.024), (0.022, y, 0.060),
             dark, 0.001, material=slick)
        _box(stage, f"{prim_path}/pkt_xn_{tag}", (0.012, 0.03, 0.024), (-0.022, y, 0.060),
             dark, 0.001, material=slick)
        # cap: bottom 0.075 (axle top 0.072 -> 3 mm play; the axle is captive)
        _box(stage, f"{prim_path}/pkt_cap_{tag}", (0.056, 0.03, 0.010), (0.0, y, 0.080),
             dark, 0.001, material=slick)
        # outer y-wall: inner face |y|=0.185 (axle tip 0.175 -> 10 mm axial play)
        _box(stage, f"{prim_path}/pkt_yw_{tag}", (0.056, 0.008, 0.036), (0.0, sy * 0.189, 0.058),
             dark, 0.001, material=slick)
        # back-stop post (outside the lid's +/-0.13 sweep)
        _box(stage, f"{prim_path}/post_{tag}", (0.024, 0.02, 0.130), (_STOP_BAR[0], y, 0.113),
             col, 0.001, material=frame)
    # back-stop bar: front face x=-0.045; the open lid's top face rests on its corner
    _box(stage, f"{prim_path}/stop_bar", (0.024, 0.36, 0.024), (_STOP_BAR[0], 0.0, _STOP_BAR[1]),
         col, 0.001, material=frame)
    # socket collar on the sill (cavity 42 x 42 mm, walls 8 mm thick, 14 mm tall)
    for tag, sy in (("a", 1.0), ("b", -1.0)):
        _box(stage, f"{prim_path}/sock_y{tag}", (0.058, 0.008, _SOCK_H),
             (_SOCK_X, sy * (_SOCK_IN + 0.004), _SILL_TOP + _SOCK_H / 2),
             green, 0.001, material=frame)
        _box(stage, f"{prim_path}/sock_x{tag}", (0.008, 0.042, _SOCK_H),
             (_SOCK_X + sy * (_SOCK_IN + 0.004), 0.0, _SILL_TOP + _SOCK_H / 2),
             green, 0.001, material=frame)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dynamic lid: plate + capsule axle + underside cleat. Origin = the
    axle axis (lid local +x runs from the hinge to the front edge). CoM and diagonal
    inertia authored EXPLICITLY at the plate centroid — MassAPI mass alone would
    leave the CoM at the body origin (the axle) and the lid would never fall shut."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(_LID_COM_X, 0.0, 0.0))
    # plate-dominated diagonal inertia about the CoM (0.5 kg, 0.30 x 0.26 x 0.016)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.00283, 0.00376, 0.00657))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    # vel_iters 4: kills the GPU capsule-on-box phantom creep at the axle contact
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    # zero sleep/stabilization: the lid must respond the instant its support changes
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    plate_m = _friction_material(stage, f"{prim_path}/plate_mat", cfg.mu_plate_s, cfg.mu_plate_d)
    slick_m = _friction_material(stage, f"{prim_path}/axle_mat", cfg.mu_hinge_s, cfg.mu_hinge_d)
    wood = (0.55, 0.40, 0.25)
    # plate: x 0.002..0.302 (clear labels: the lid front edge is the grasp lip)
    _box(stage, f"{prim_path}/plate", (_LID_LEN, _LID_W, _LID_T), (_LID_LEN / 2 + 0.002, 0.0, 0.0),
         wood, 0.0015, material=plate_m)
    # underside cleat ridge (inner face x=0.246): the prop-top backstop
    _box(stage, f"{prim_path}/cleat", (_CLEAT_T, 0.24, _CLEAT_T),
         (_CLEAT_XC, 0.0, -_LID_T / 2 - _CLEAT_T / 2 + 0.002), wood, 0.001, material=plate_m)
    # axle: capsule along y, tips at +/-0.175
    cap = UsdGeom.Capsule.Define(stage, f"{prim_path}/axle")
    cap.CreateAxisAttr("Y")
    cap.CreateRadiusAttr(_AXLE_R)
    cap.CreateHeightAttr(2.0 * (_AXLE_TIP - _AXLE_R))
    cap.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.25, 0.28)])
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    pxc = PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim())
    pxc.CreateContactOffsetAttr(0.001)
    pxc.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(
        slick_m, UsdShade.Tokens.weakerThanDescendants, "physics")
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily, so the module imports
    app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chest" not in _SPAWNER_CACHE:

        @configclass
        class ChestSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chest)
            mu_frame_s: float = 0.50
            mu_frame_d: float = 0.45
            mu_hinge_s: float = 0.02
            mu_hinge_d: float = 0.02

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            mu_plate_s: float = 0.50
            mu_plate_d: float = 0.45
            mu_hinge_s: float = 0.02
            mu_hinge_d: float = 0.02

        _SPAWNER_CACHE.update(chest=ChestSpawnerCfg, lid=LidSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HoodPropSceneCfg(BaseCfg):
    """Config for `HoodPropScene`. The honesty knobs are asserted in `__post_init__`:
    the long peg props INSIDE the band with margin, the short peg props BELOW it, and
    the back-stop rest angle sits far ABOVE it — the band is reachable only through a
    real support interaction."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    band_min: float = tunable(40.0)  # propped-open band (deg): success needs theta in
    band_max: float = tunable(72.0)  # [band_min, band_max] — unstable without support
    open_latch_deg: float = tunable(88.0)  # stage-1 latch: the lid has been swung past here
    peg_tilt_max_deg: float = tunable(15.0)  # "upright" cone for the socketed peg
    sock_xy_tol: float = tunable(0.016)  # peg foot center within this of the socket axis
    hinge_pos_tol: float = tunable(0.025)  # lid origin must stay this close to the hinge axis
    lid_settle_lin: float = tunable(0.05)  # settle gates when judging success
    lid_settle_ang: float = tunable(0.20)
    peg_settle_lin: float = tunable(0.05)
    peg_settle_ang: float = tunable(0.60)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    chest_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the chest at reset (m)
    chest_yaw_deg: float = tunable(180.0)  # uniform +/- chest yaw at reset (deg)
    peg_ring_r: float = tunable(0.58)  # peg scatter ring radius around the hinge origin (m)
    peg_ring_jitter: float = tunable(0.05)  # +/- radial jitter
    peg_sep_deg: float = tunable(35.0)  # min angular separation between the two pegs
    p_start_open: float = tunable(0.35)  # probability the lid STARTS parked on the back-stop
    start_open_deg: float = tunable(100.0)  # written start angle for open starts (falls to stop)

    # --- info: structure ---------------------------------------------------------------------
    hinge_z: float = info(_HINGE_Z)
    lid_len: float = info(_LID_LEN)
    lid_mass: float = info(_LID_MASS)
    lid_com_x: float = info(_LID_COM_X)
    sock_x: float = info(_SOCK_X)
    peg_w: float = info(_PEG_W)
    peg_l_long: float = info(_PEG_L_LONG)
    peg_l_short: float = info(_PEG_L_SHORT)
    peg_mass_long: float = info(0.060)
    peg_mass_short: float = info(0.035)
    mu_peg_s: float = info(0.50)
    mu_peg_d: float = info(0.45)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    # Derived (filled in __post_init__).
    prop_deg_long: float = field(default=None, init=False)  # ~57: long-peg prop angle
    prop_deg_short: float = field(default=None, init=False)  # ~31: short-peg prop angle
    stop_deg: float = field(default=None, init=False)  # ~105: back-stop rest angle

    def __post_init__(self) -> None:
        self.prop_deg_long = _prop_angle(_PEG_L_LONG)
        self.prop_deg_short = _prop_angle(_PEG_L_SHORT)
        self.stop_deg = _stop_angle()
        # -- the band is exactly the long-peg prop, with margin on every side --
        assert self.band_min + 8.0 <= self.prop_deg_long <= self.band_max - 8.0, \
            f"long-peg prop angle {self.prop_deg_long:.1f} must sit >=8 deg inside the band"
        assert self.prop_deg_short <= self.band_min - 5.0, \
            f"short-peg prop angle {self.prop_deg_short:.1f} must fall >=5 deg below the band"
        assert self.stop_deg >= self.band_max + 20.0, \
            f"back-stop rest angle {self.stop_deg:.1f} must sit >=20 deg above the band"
        assert self.open_latch_deg < self.start_open_deg < self.stop_deg
        # -- bistability: the lid CoM must land behind the hinge at the stop --
        assert _LID_COM_X * math.cos(math.radians(self.stop_deg)) < -0.02, \
            "lid CoM must sit behind the hinge at the back-stop (the temporary hold)"
        # -- hinge slickness: pin friction torque << gravity torque near vertical --
        mu_pin = 0.02
        assert mu_pin * _LID_MASS * 9.81 * _AXLE_R < 0.2 * _LID_MASS * 9.81 * _LID_COM_X \
            * math.cos(math.radians(self.open_latch_deg)), \
            "hinge must be slick enough that the lid falls shut from the latch angle"
        # -- embodiment / geometry sanity --
        assert _PEG_W < 0.08, "pegs must fit a parallel jaw"
        assert _SOCK_IN - _PEG_W / 2 >= 0.004, "socket must leave >=4 mm insertion play"
        assert _SILL_TOP + _PEG_L_LONG < _HINGE_Z + _LID_LEN, "lid must reach over the prop"
        # closed lid covers the socket: collar top far below the closed underside sweep
        assert _SILL_TOP + _SOCK_H < _HINGE_Z - _LID_T / 2, \
            "socket collar must fit under the CLOSED lid (the ordering constraint)"
        # peg ring clears the chest footprint at any yaw
        reach = math.hypot(0.31, 0.19) + _PEG_L_LONG / 2 + 0.02
        assert self.peg_ring_r - self.peg_ring_jitter > reach, \
            "peg scatter ring would collide with the chest"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hood_prop")
class HoodPropScene(BaseScene):
    cfg: HoodPropSceneCfg

    def __init__(self, cfg: HoodPropSceneCfg | None = None) -> None:
        super().__init__(cfg or HoodPropSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        chest_cls, lid_cls = spawners["chest"], spawners["lid"]
        peg_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_peg_s, dynamic_friction=c.mu_peg_d, restitution=0.0)
        peg_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0)
        peg_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

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
            "chest": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chest",
                spawn=chest_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _HINGE_Z + 0.0005)),
            ),
            "peg_long": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PegLong",
                spawn=sim_utils.CuboidCfg(
                    size=(_PEG_W, _PEG_W, _PEG_L_LONG),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.10, 0.10)),
                    physics_material=peg_mat, rigid_props=peg_rigid, collision_props=peg_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.peg_mass_long)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.5, _PEG_W / 2 + 0.003)),
            ),
            "peg_short": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PegShort",
                spawn=sim_utils.CuboidCfg(
                    size=(_PEG_W, _PEG_W, _PEG_L_SHORT),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.25, 0.85)),
                    physics_material=peg_mat, rigid_props=peg_rigid, collision_props=peg_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.peg_mass_short)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, 0.5, _PEG_W / 2 + 0.003)),
            ),
        }
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the solve drives the lid with per-step wrenches; without this the
                # applied torque is integrated only on the first substep
                "enable_external_forces_every_iteration": True,
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
        self.chest: RigidObject = env.iscene["chest"]
        self.lid: RigidObject = env.iscene["lid"]
        self.peg_long: RigidObject = env.iscene["peg_long"]
        self.peg_short: RigidObject = env.iscene["peg_short"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.started_closed = torch.ones(n, dtype=torch.bool, device=dev)
        self.opened_latch = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter/yaw the chest (kinematic teleport), seat the lid on
        its hinge (closed, or parked past vertical on ~35% of episodes), scatter both
        pegs lying flat on a jittered ring around the chest, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        zeros = torch.zeros(m, device=dev)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- chest: xy jitter + free yaw ---
        cxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.chest_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.chest_yaw_deg)
        half = yaw / 2
        q_yaw = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        write(self.chest, torch.cat([cxy, zeros.unsqueeze(-1)], dim=-1), q_yaw)

        # --- lid: on the hinge; closed, or (p_start_open) parked past vertical ---
        open_start = torch.rand(m, device=dev) < c.p_start_open
        self.started_closed[env_ids] = ~open_start
        self.opened_latch[env_ids] = False
        theta = torch.where(open_start,
                            torch.full((m,), math.radians(c.start_open_deg), device=dev),
                            zeros)
        # q_lid = q_yaw * q_pitch(-theta about y): raises lid local +x by theta
        ph = -theta / 2
        q_pitch = torch.stack([torch.cos(ph), zeros, torch.sin(ph), zeros], dim=-1)
        q_lid = _qmul(q_yaw, q_pitch)
        lid_pos = torch.stack([cxy[:, 0], cxy[:, 1],
                               torch.full((m,), _HINGE_Z + 0.0005, device=dev)], dim=-1)
        write(self.lid, lid_pos, q_lid)

        # --- pegs: lying flat on a jittered ring, min angular separation ---
        ang_a = torch.rand(m, device=dev) * 2 * math.pi
        sep = math.radians(c.peg_sep_deg)
        off = sep + torch.rand(m, device=dev) * (2 * math.pi - 2 * sep)
        ang_b = ang_a + off
        for body, ang in ((self.peg_long, ang_a), (self.peg_short, ang_b)):
            rad = c.peg_ring_r + (torch.rand(m, device=dev) * 2 - 1) * c.peg_ring_jitter
            pos = torch.stack([cxy[:, 0] + rad * torch.cos(ang),
                               cxy[:, 1] + rad * torch.sin(ang),
                               torch.full((m,), _PEG_W / 2 + 0.003, device=dev)], dim=-1)
            pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            ph2 = pyaw / 2
            c45 = math.cos(math.pi / 4)
            # lying flat: q = qz(yaw) * qy(90 deg)
            q_peg = torch.stack([torch.cos(ph2) * c45, -torch.sin(ph2) * c45,
                                 torch.cos(ph2) * c45, torch.sin(ph2) * c45], dim=-1)
            write(body, pos, q_peg)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "chest": self.chest.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "peg_long": self.peg_long.data.root_state_w[env_ids].clone(),
            "peg_short": self.peg_short.data.root_state_w[env_ids].clone(),
            "started_closed": self.started_closed[env_ids].clone(),
            "opened_latch": self.opened_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.chest.write_root_state_to_sim(state["chest"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.peg_long.write_root_state_to_sim(state["peg_long"], env_ids)
        self.peg_short.write_root_state_to_sim(state["peg_short"], env_ids)
        self.started_closed[env_ids] = state["started_closed"]
        self.opened_latch[env_ids] = state["opened_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low wooden chest stands on the floor: an open box about 34 x 37 cm with "
            f"5 cm walls, and a heavy {c.lid_len * 100:.0f} cm lid hinged along its back "
            f"edge (a dark pin riding in capped pockets about {c.hinge_z * 100:.0f} cm "
            f"up — the lid can swing from shut to just past vertical, but cannot be "
            f"lifted off). The hinge is slick: released anywhere below vertical the lid "
            f"FALLS SHUT on its own; pushed all the way back it comes to rest leaning "
            f"against a back-stop bar and stays there. On the chest floor, "
            f"{c.sock_x * 100:.0f} cm in front of the hinge line, sits a small green "
            f"square socket collar (42 mm opening, 14 mm tall) — note the closed lid "
            f"covers it completely. On the floor around the chest lie two square pegs, "
            f"flat on the ground: a RED one ({c.peg_l_long * 100:.0f} cm long, "
            f"{c.peg_w * 1000:.0f} mm across) and a BLUE one "
            f"({c.peg_l_short * 100:.0f} cm long, same thickness). The chest's position "
            f"and heading, the pegs' spots, and whether the lid starts shut or already "
            f"parked open on the back-stop all change per episode — look first.\n"
            f"Goal: leave the lid PROPPED HALF-OPEN, resting on the red peg: the red peg "
            f"standing upright with its foot inside the green socket collar, the lid "
            f"lowered onto the peg's top so it rests at an opening angle between "
            f"{c.band_min:.0f} and {c.band_max:.0f} degrees, everything at rest with "
            f"nothing else holding the lid. Since the closed lid covers the socket, "
            f"first swing the lid past vertical onto the back-stop so it stays open by "
            f"itself, then stand the red peg in the socket, then pull the lid forward "
            f"off the stop and lower it gently until it rests on the peg. A lid left on "
            f"the back-stop is too far open to count, a shut lid counts nothing, and the "
            f"blue peg is too short — propped on it the lid sags below the required "
            f"angle. Only the peg may hold the lid up at the end."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Prop the chest lid half-open with the red peg: swing the lid back past "
            "vertical so it rests on the back-stop, stand the red peg upright in the "
            "green socket on the chest floor, then lower the lid onto the peg and let "
            "it rest there. The lid must end held up only by the red peg, neither shut "
            "nor leaning on the back-stop."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _chest_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(x_hat, y_hat) of the chest frame in world, (N, 3) each."""
        q = self.chest.data.root_quat_w
        n = q.shape[0]
        ex = _qapply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(n, 3))
        ey = _qapply(q, torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(n, 3))
        return ex, ey

    def lid_angle(self) -> torch.Tensor:
        """(N,) lid opening angle in RADIANS: the angle of the lid's local +x above
        the chest's horizontal +x (0 = shut, pi/2 = vertical, ~1.83 = back-stop)."""
        q = self.lid.data.root_quat_w
        n = q.shape[0]
        a = _qapply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(n, 3))
        ex, _ey = self._chest_axes()
        horiz = (a * ex).sum(dim=-1)
        return torch.atan2(a[:, 2], horiz)

    def hinge_ok(self) -> torch.Tensor:
        """(N,) bool: the lid origin still rides the hinge axis (a lid dislodged and
        leaned somewhere else cannot fake the propped band)."""
        hinge_w = self.chest.data.root_pos_w.clone()
        hinge_w[:, 2] += _HINGE_Z
        d = (self.lid.data.root_pos_w - hinge_w).norm(dim=-1)
        return d < self.cfg.hinge_pos_tol

    def _sock_center_w(self) -> torch.Tensor:
        ex, _ey = self._chest_axes()
        return self.chest.data.root_pos_w + ex * _SOCK_X

    def peg_socketed(self, body=None) -> torch.Tensor:
        """(N,) bool: the LONG peg (or `body`) stands upright with its foot inside
        the socket collar."""
        c = self.cfg
        body = body if body is not None else self.peg_long
        half_l = (_PEG_L_LONG if body is self.peg_long else _PEG_L_SHORT) / 2
        q = body.data.root_quat_w
        n = q.shape[0]
        axis = _qapply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(n, 3))
        up = axis[:, 2].abs() >= math.cos(math.radians(c.peg_tilt_max_deg))
        sgn = torch.sign(axis[:, 2]).unsqueeze(-1)
        foot = body.data.root_pos_w - sgn * axis * half_l
        sock = self._sock_center_w()
        near = (foot[:, :2] - sock[:, :2]).norm(dim=-1) < c.sock_xy_tol
        low = (foot[:, 2] - self.chest.data.root_pos_w[:, 2]) < _SILL_TOP + 0.02
        return up & near & low

    def lid_settled(self) -> torch.Tensor:
        return (self.lid.data.root_lin_vel_w.norm(dim=-1) < self.cfg.lid_settle_lin) \
            & (self.lid.data.root_ang_vel_w.norm(dim=-1) < self.cfg.lid_settle_ang)

    def peg_settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.peg_settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.peg_settle_ang)

    def in_band(self) -> torch.Tensor:
        c = self.cfg
        th = self.lid_angle()
        return (th >= math.radians(c.band_min)) & (th <= math.radians(c.band_max))

    def _update_latches(self) -> None:
        self.opened_latch |= self.lid_angle() >= math.radians(self.cfg.open_latch_deg)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: lid riding its hinge, at rest inside the propped band, with the
        long peg standing settled in the socket under it. The band is unreachable at
        rest without support (back-stop ~105 deg, unsupported lids fall shut), so
        this IS 'the lid rests on the peg'."""
        self._update_latches()
        return self.in_band() & self.hinge_ok() & self.peg_socketed() \
            & self.lid_settled() & self.peg_settled(self.peg_long)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 once the lid has been swung past `open_latch_deg`
        (latched; only on episodes that STARTED closed — open starts get no free
        credit), +0.25 for the long peg standing in the socket, +0.20 once the lid
        occupies the band with the peg in place, capped at 0.60; 1.0 iff success().
        Null policy ~0 in both start states."""
        self._update_latches()
        opened = (self.opened_latch & self.started_closed).float()
        pegged = self.peg_socketed().float()
        banded = (self.in_band() & self.hinge_ok() & self.peg_socketed()).float()
        base = (0.15 * opened + 0.25 * pegged + 0.20 * banded).clamp(max=0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


# ----- pure-torch quaternion helpers (shared with solve/smoke) ---------------------------------
def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) wxyz, pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a*b, wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


register_env("simgen", lambda: EnvCfg(scene="hood_prop", robot="null", env_spacing=3.0))
