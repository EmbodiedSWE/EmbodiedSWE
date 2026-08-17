"""ChannelConsoleScene — pull the interlock pin, then slide the captive channel
selector to the color-matched band (sim_gen task `change_channel_i249`).

Derived from rlbench/change_channel, but STRATEGICALLY different: the seed is a
grasp-and-press — pick the TV remote off the table, aim it at the TV and press one
channel button (one grasp, one transport, one discrete press on a hand-held device).
Here nothing is hand-held and nothing is pressed:

- the "channel control" is a FIXED kinematic console with a CAPTIVE selector: an
  orange-tabbed slider block rides inside a slotted C-channel track (overhanging lip
  strips leave only a narrow slot the tab protrudes through, so the block physically
  cannot be lifted out — proved by a force probe in smoke);
- a LOCK PIN (dark square post with a light square head) stands mid-track, seated
  36 mm deep in a socket below the channel floor; while seated its shaft fills the
  channel cross-section and the slider cannot pass (proved by a force probe in
  smoke: a pushed slider stalls well short of every band on the far side);
- the target channel is ALWAYS on the far side of the pin from the slider's start,
  so the episode has a physically-enforced execution order: extract the pin
  (vertical contact-guided lift, ~75 mm before the shaft clears the lip slot), THEN
  slide the selector into the target band;
- the target is indicated symbolically: four colored tick plates (red, green, blue,
  yellow) mark the channel positions on the track's front face, and a single
  colored INDICATOR CUBE on the rear pedestal names the target channel — the agent
  must read the cube's color and match it to a tick, per episode.

A solver needs a different PLAN from the seed (ordered mechanism sequence: read a
symbolic color cue, extract a deep-seated interlock, then continuously position a
captive slider within a tolerance band) and a different CODE STRUCTURE (a guided
vertical extraction + a closed-loop slide servo instead of a grasp pose + a button
push). Distinct from the corpus packages read while building this task:
basketball_in_hoop_i128 (push-only anti-gravity ball conveyance along a switchback)
and the pen_holder exemplar (multi-object insertion into a cup) — neither has an
interlock, a captive prismatic mechanism, or a symbolic color-matched goal.

success(): the slider block rests IN the track (canonical |y| and z windows — a
look-alike block laid ON TOP of the lip strips is rejected by z) with its center
inside the target band (|x - x_target| <= band_tol) and settled. The pin's state is
not judged directly — with the pin seated the band is physically unreachable.

score() is graded and latched (credit never evaporates): 0.20 the first time the
pin is clear of the interlock column, 0.15 the first time the slider has fully
crossed the pin column onto the target side, up to 0.25 for latched best approach
toward the target band (normalized by the episode's start distance), cap 0.60;
1.0 iff success(). The null policy scores ~0 (slider spawns >= 0.10 m from the
band, pin seated).

Per-episode randomization (readback-verified in smoke): console yaw + xy jitter,
slider start SIDE (left/right of the pin), slider start position within its side,
and the target channel (one of the two bands on the opposite side) with the
matching indicator cube swapped onto the pedestal (non-target cubes park in a
ground depot far from the console). Assets are fully procedural compound spawners
(boxes only; console kinematic, slider + pin dynamic, indicator cubes kinematic).
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


# ----- fixed geometry (single source of truth: spawners AND cfg info read these) ---------------
DECK_TOP = 0.16        # console base body top
FLOOR_TOP = 0.240      # channel floor top (block rides here)
RAIL_TOP = 0.270       # rail top = lip underside
LIP_TOP = 0.278        # lip strip top
SOCKET_FLOOR = 0.204   # pin socket floor (36 mm below FLOOR_TOP incl. floor plate)
CHAN_HALF_Y = 0.0175   # channel interior half-width
SLOT_HALF = 0.008      # lip slot interior half-width (tab + pin pass, block cannot)
HOLE_HALF = 0.008      # floor hole / socket interior half-extent
X_END = 0.160          # end-stop inner face
TRAVEL = 0.142         # max |block center x| (X_END - block half-length)
CHAN_XS = (-0.120, -0.055, 0.055, 0.120)          # channel band centers
CHAN_NAMES = ("red", "green", "blue", "yellow")   # tick colors, -x -> +x
CHAN_RGB = ((0.85, 0.12, 0.10), (0.10, 0.65, 0.15),
            (0.15, 0.30, 0.85), (0.90, 0.80, 0.10))
BLOCK_SZ = (0.036, 0.030, 0.024)   # slider block
TAB_SZ = (0.014, 0.010, 0.062)     # tab on top of the block (through the slot)
PIN_SHAFT = (0.012, 0.012, 0.116)  # lock pin shaft (root at shaft center)
PIN_HEAD = (0.020, 0.020, 0.014)   # lock pin head (cannot pass the slot)
PIN_HALF_L = PIN_SHAFT[2] / 2
BLOCK_REST_Z = FLOOR_TOP + BLOCK_SZ[2] / 2   # 0.252 — slider root at rest
PIN_SEAT_Z = SOCKET_FLOOR + PIN_HALF_L       # 0.262 — pin root when seated
PED_XY = (0.20, 0.105)             # indicator pedestal center (canonical)
PED_TOP = 0.270                    # pedestal top
FLAG_SZ = 0.030                    # indicator cube edge
FLAG_Z = PED_TOP + FLAG_SZ / 2 + 0.001
FLAG_DEPOT = (1.2, 1.2)            # env-relative ground depot for parked cubes


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
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
    """Author one axis-aligned box child prim (translate -> scale, authored once —
    the duplicate-xformOp trap is avoided by never re-authoring an existing prim)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the standard physics armor (zero sleep /
    stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on; velocity iterations 4 — the
    phantom-creep fix)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _console_boxes() -> list[tuple]:
    """The full box list of the console (canonical/local frame; z up, track along x).
    All numbers derive from the module geometry constants, so the rubric windows and
    the geometry cannot drift apart."""
    dark = (0.20, 0.20, 0.22)
    grey = (0.45, 0.46, 0.48)
    lip_c = (0.58, 0.59, 0.61)
    ped_c = (0.32, 0.33, 0.36)

    rail_ic = CHAN_HALF_Y + 0.006  # rail center |y| (12 mm thick, inner face at 0.0175)
    lip_c_y = (SLOT_HALF + rail_ic + 0.006) / 2  # lip strip spans y [SLOT_HALF, 0.0295]
    lip_w = rail_ic + 0.006 - SLOT_HALF

    boxes: list[tuple] = [
        # base body
        ("base", (0.60, 0.34, DECK_TOP), (0.0, 0.0, DECK_TOP / 2), dark),
        # plinth: two x-running walls under the rails + two end walls (hollow between,
        # leaving room for the socket well below the channel floor)
        ("plinth_p", (0.360, 0.012, 0.074), (0.0, +0.0235, 0.197), grey),
        ("plinth_n", (0.360, 0.012, 0.074), (0.0, -0.0235, 0.197), grey),
        ("plinth_e0", (0.012, 0.059, 0.074), (+0.174, 0.0, 0.197), grey),
        ("plinth_e1", (0.012, 0.059, 0.074), (-0.174, 0.0, 0.197), grey),
        # channel floor: two long plates + two bridge strips, leaving a 16x16 mm hole
        # at x=0 over the socket well
        ("floor_r", (0.166, 0.059, 0.006), (+0.091, 0.0, 0.237), grey),
        ("floor_l", (0.166, 0.059, 0.006), (-0.091, 0.0, 0.237), grey),
        ("floor_bp", (0.016, 0.0215, 0.006), (0.0, +0.01875, 0.237), grey),
        ("floor_bn", (0.016, 0.0215, 0.006), (0.0, -0.01875, 0.237), grey),
        # socket well below the hole (interior 16x16 mm, 30 mm deep + bottom plate)
        ("sock_xp", (0.008, 0.032, 0.030), (+0.012, 0.0, 0.219), grey),
        ("sock_xn", (0.008, 0.032, 0.030), (-0.012, 0.0, 0.219), grey),
        ("sock_yp", (0.016, 0.008, 0.030), (0.0, +0.012, 0.219), grey),
        ("sock_yn", (0.016, 0.008, 0.030), (0.0, -0.012, 0.219), grey),
        ("sock_bot", (0.040, 0.040, 0.006), (0.0, 0.0, 0.201), grey),
        # rails + overhanging lip strips (the slot that makes the slider captive)
        ("rail_p", (0.360, 0.012, 0.030), (0.0, +rail_ic, 0.255), grey),
        ("rail_n", (0.360, 0.012, 0.030), (0.0, -rail_ic, 0.255), grey),
        ("lip_p", (0.360, lip_w, 0.008), (0.0, +lip_c_y, 0.274), lip_c),
        ("lip_n", (0.360, lip_w, 0.008), (0.0, -lip_c_y, 0.274), lip_c),
        # end stops
        ("end_p", (0.012, 0.059, 0.038), (+0.166, 0.0, 0.259), grey),
        ("end_n", (0.012, 0.059, 0.038), (-0.166, 0.0, 0.259), grey),
        # indicator pedestal (rear)
        ("pedestal", (0.050, 0.050, PED_TOP - DECK_TOP),
         (PED_XY[0], PED_XY[1], (PED_TOP + DECK_TOP) / 2), ped_c),
    ]
    # colored tick plates on the front rail's outer face, one per channel
    for x, rgb in zip(CHAN_XS, CHAN_RGB):
        boxes.append((f"tick_{x:+.3f}".replace(".", "p").replace("+", "P")
                      .replace("-", "N"),
                      (0.030, 0.004, 0.026), (x, -(rail_ic + 0.008), 0.255), rgb))
    return boxes


def _spawn_console(prim_path: str, cfg: Any, translation=None, orientation=None):
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 40.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    for name, size, center, color in _console_boxes():
        _box(stage, f"{prim_path}/{name}", size, center, color, cfg.contact_offset,
             material=mat)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The captive slider: block + orange tab (one rigid body, root at block center)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/body", BLOCK_SZ, (0.0, 0.0, 0.0),
         (0.30, 0.30, 0.32), cfg.contact_offset, material=mat)
    _box(stage, f"{prim_path}/tab", TAB_SZ,
         (0.0, 0.0, BLOCK_SZ[2] / 2 + TAB_SZ[2] / 2),
         (0.95, 0.50, 0.10), cfg.contact_offset, material=mat)
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The lock pin: square shaft + wider square head (one rigid body, root at
    shaft center). High angular damping keeps the guided extraction stable."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.3)
    pxrb.CreateAngularDampingAttr(2.0)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/shaft", PIN_SHAFT, (0.0, 0.0, 0.0),
         (0.15, 0.15, 0.17), cfg.contact_offset, material=mat)
    _box(stage, f"{prim_path}/head", PIN_HEAD,
         (0.0, 0.0, PIN_HALF_L + PIN_HEAD[2] / 2),
         (0.88, 0.88, 0.86), cfg.contact_offset, material=mat)
    return root


def _spawn_flag(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC indicator cube (a colored lamp block — agents read it, nothing
    moves it)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 0.05,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/cube", (FLAG_SZ, FLAG_SZ, FLAG_SZ), (0.0, 0.0, 0.0),
         cfg.rgb, cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "console" not in _SPAWNER_CACHE:

        @configclass
        class ConsoleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_console)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            mass: float = 0.15
            contact_offset: float = 0.002

        @configclass
        class PinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            mass: float = 0.10
            contact_offset: float = 0.002

        @configclass
        class FlagSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flag)
            rgb: tuple = (1.0, 1.0, 1.0)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(console=ConsoleSpawnerCfg, block=BlockSpawnerCfg,
                              pin=PinSpawnerCfg, flag=FlagSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChannelConsoleSceneCfg(BaseCfg):
    """Config for `ChannelConsoleScene`. Every rubric window's honesty — captivity,
    interlock reach margin, on-lips rejection — is asserted in `__post_init__` from
    the same geometry constants the spawners author."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    band_tol: float = tunable(0.016)   # |block x - target x| for "at the channel"
    y_tol: float = tunable(0.010)      # in-track: |block canonical y| below this
    z_tol: float = tunable(0.007)      # in-track: |block z - rest z| below this
    settle_lin: float = tunable(0.04)  # max block |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.0)   # max block |ang vel| at judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_jitter_deg: float = tunable(10.0)  # +- console yaw jitter
    pos_jitter: float = tunable(0.03)      # +- console xy jitter
    start_abs_lo: float = tunable(0.060)   # slider start |x| window (its side)
    start_abs_hi: float = tunable(0.135)

    # --- info: geometry (module constants; the spawners author exactly these) ----------------
    floor_top: float = info(FLOOR_TOP)
    lip_top: float = info(LIP_TOP)
    rail_top: float = info(RAIL_TOP)
    socket_floor: float = info(SOCKET_FLOOR)
    block_rest_z: float = info(BLOCK_REST_Z)
    pin_seat_z: float = info(PIN_SEAT_Z)
    pin_half_l: float = info(PIN_HALF_L)
    chan_xs: tuple = info(CHAN_XS)
    chan_names: tuple = info(CHAN_NAMES)
    travel: float = info(TRAVEL)
    block_mass: float = info(0.15)
    pin_mass: float = info(0.10)
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening
    contact_offset: float = info(0.002)
    # pin_clear latch: pin NOT blocking = outside the interlock column OR bottom
    # endpoint above the lip slot
    clear_col_xy: float = info(0.025)
    clear_bot_z: float = info(LIP_TOP - 0.002)

    def __post_init__(self) -> None:
        c = self
        # -- bands sit inside the physical travel --
        assert max(abs(x) for x in CHAN_XS) + c.band_tol <= TRAVEL - 0.002, \
            "outermost band must be fully reachable"
        # -- captivity is real: the block cannot pass the lip slot, the tab can --
        assert 2 * SLOT_HALF < BLOCK_SZ[1] - 0.010, "block must not fit the slot"
        assert TAB_SZ[1] <= 2 * SLOT_HALF - 0.004, "tab must pass the slot freely"
        assert RAIL_TOP - (FLOOR_TOP + BLOCK_SZ[2]) >= 0.004, \
            "block needs top clearance under the lips"
        # -- the seated pin blocks the whole block cross-section --
        assert PIN_SEAT_Z - PIN_HALF_L <= FLOOR_TOP - 0.030, "pin seats deep"
        assert PIN_SEAT_Z + PIN_HALF_L >= FLOOR_TOP + BLOCK_SZ[2] + 0.02, \
            "pin shaft must top the block height"
        # -- interlock reach margin: worst-case pin tilt still stops the slider short
        #    of every far-side band (socket play amplified to block-top height) --
        play = 2 * (HOLE_HALF - PIN_SHAFT[0] / 2)
        guide = FLOOR_TOP - SOCKET_FLOOR
        tilt_shift = play * ((FLOOR_TOP + BLOCK_SZ[2] - SOCKET_FLOOR) / guide)
        reach = BLOCK_SZ[0] / 2 + PIN_SHAFT[0] / 2 + tilt_shift
        assert reach + 0.004 <= min(abs(x) for x in CHAN_XS) - c.band_tol, \
            "a seated pin must keep every far-side band unreachable"
        # -- the head cannot pass the slot/hole (no fall-through), and fits the jaw --
        assert PIN_HEAD[0] >= 2 * SLOT_HALF + 0.002, "head must not pass the slot"
        assert PIN_HEAD[0] <= c.jaw_span - 0.020, "head must fit the Franka jaw"
        # -- z window honesty: on-lips rest is rejected, in-track rest accepted --
        on_lips_z = LIP_TOP + BLOCK_SZ[2] / 2
        assert on_lips_z > BLOCK_REST_Z + c.z_tol + 0.010, "on-lips must read high"
        # -- start window: clear of the pin column and inside travel --
        assert c.start_abs_lo >= reach + 0.010, "start must not touch the pin"
        assert c.start_abs_hi <= TRAVEL - 0.004, "start must fit the travel"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("channel_console")
class ChannelConsoleScene(BaseScene):
    cfg: ChannelConsoleSceneCfg

    def __init__(self, cfg: ChannelConsoleSceneCfg | None = None) -> None:
        super().__init__(cfg or ChannelConsoleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "console": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Console",
                spawn=sp["console"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "block": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slider",
                spawn=sp["block"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.block_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.10, 0.0, BLOCK_REST_Z + 0.001)),
            ),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/LockPin",
                spawn=sp["pin"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.pin_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, PIN_SEAT_Z + 0.001)),
            ),
        }
        for i, rgb in enumerate(CHAN_RGB):
            out[f"flag_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flag" + str(i),
                spawn=sp["flag"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    rgb=rgb, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(FLAG_DEPOT[0], FLAG_DEPOT[1] + 0.12 * i, FLAG_SZ / 2)),
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
                # external-wrench plant recipe: without this the solve/smoke force
                # servos are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.console: RigidObject = env.iscene["console"]
        self.block: RigidObject = env.iscene["block"]
        self.pin: RigidObject = env.iscene["pin"]
        self.flags: list[RigidObject] = [env.iscene[f"flag_{i}"] for i in range(4)]
        self.env_origins = env.iscene.env_origins
        # per-env episode state
        self.side = torch.ones(n, device=dev)          # slider start side (+1 / -1)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.x_target = torch.zeros(n, device=dev)
        self.d0 = torch.ones(n, device=dev)            # |start - target| at reset
        # latches (post_step): pin ever clear, slider ever fully across, best approach
        self.pin_clear = torch.zeros(n, device=dev)
        self.crossed = torch.zeros(n, device=dev)
        self.appr_max = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: console yaw + xy jitter; sample the slider's start side
        (torch.rand comparison — the first-randint degeneracy), a start position on
        that side, and a target band on the OPPOSITE side; seat the pin; put the
        matching indicator cube on the pedestal, park the rest in the depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- console pose ---
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_jitter_deg) / 2
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = jit
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.console.write_root_state_to_sim(st, env_ids)
        cy, sy = torch.cos(2 * half), torch.sin(2 * half)

        def to_world(cx: torch.Tensor, cyl: torch.Tensor, cz) -> torch.Tensor:
            """Canonical (console) xy + z -> env world xyz through the fresh pose."""
            w = torch.zeros(m, 3, device=dev)
            w[:, 0] = jit[:, 0] + cy * cx - sy * cyl
            w[:, 1] = jit[:, 1] + sy * cx + cy * cyl
            w[:, 2] = cz if isinstance(cz, torch.Tensor) else float(cz)
            return w + origin

        # --- episode sampling ---
        self.side[env_ids] = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        side = self.side[env_ids]
        outer = torch.rand(m, device=dev) < 0.5  # far band on the target side?
        tside = -side
        tx = tside * torch.where(outer, tside.new_tensor(abs(CHAN_XS[3])),
                                 tside.new_tensor(abs(CHAN_XS[2])))
        # map target x -> channel index (CHAN_XS is sorted)
        xs = torch.tensor(CHAN_XS, device=dev)
        self.target_idx[env_ids] = (tx.unsqueeze(1) - xs.unsqueeze(0)).abs().argmin(dim=1)
        self.x_target[env_ids] = tx

        bx0 = side * (c.start_abs_lo
                      + torch.rand(m, device=dev) * (c.start_abs_hi - c.start_abs_lo))
        self.d0[env_ids] = (bx0 - tx).abs()

        # --- slider ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = to_world(bx0, torch.zeros(m, device=dev), BLOCK_REST_Z + 0.001)
        # yaw the body with the console so it drops straight into the channel
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self.block.write_root_state_to_sim(st, env_ids)

        # --- pin (seated) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = to_world(torch.zeros(m, device=dev), torch.zeros(m, device=dev),
                              PIN_SEAT_Z + 0.001)
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self.pin.write_root_state_to_sim(st, env_ids)

        # --- indicator cubes: target on the pedestal, the rest in the ground depot ---
        for i, flag in enumerate(self.flags):
            on_ped = self.target_idx[env_ids] == i
            ped = to_world(torch.full((m,), PED_XY[0], device=dev),
                           torch.full((m,), PED_XY[1], device=dev), FLAG_Z)
            depot = torch.zeros(m, 3, device=dev)
            depot[:, 0] = FLAG_DEPOT[0]
            depot[:, 1] = FLAG_DEPOT[1] + 0.12 * i
            depot[:, 2] = FLAG_SZ / 2
            depot += origin
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = torch.where(on_ped.unsqueeze(1), ped, depot)
            st[:, 3] = torch.where(on_ped, torch.cos(half), torch.ones(m, device=dev))
            st[:, 6] = torch.where(on_ped, torch.sin(half), torch.zeros(m, device=dev))
            flag.write_root_state_to_sim(st, env_ids)

        for latch in (self.pin_clear, self.crossed, self.appr_max):
            latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"console": self.console, "block": self.block, "pin": self.pin,
                  **{f"flag_{i}": f for i, f in enumerate(self.flags)}}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in bodies.items()},
            "side": self.side[env_ids].clone(),
            "target_idx": self.target_idx[env_ids].clone(),
            "x_target": self.x_target[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "latches": torch.stack([self.pin_clear[env_ids], self.crossed[env_ids],
                                    self.appr_max[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"console": self.console, "block": self.block, "pin": self.pin,
                  **{f"flag_{i}": f for i, f in enumerate(self.flags)}}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.side[env_ids] = state["side"]
        self.target_idx[env_ids] = state["target_idx"]
        self.x_target[env_ids] = state["x_target"]
        self.d0[env_ids] = state["d0"]
        lt = state["latches"]
        self.pin_clear[env_ids] = lt[:, 0]
        self.crossed[env_ids] = lt[:, 1]
        self.appr_max[env_ids] = lt[:, 2]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A dark CHANNEL-SELECTOR CONSOLE (a knee-high box, about 0.60 x 0.34 m, "
            "0.16 m tall) stands on the floor. On a raised plinth along its "
            "centerline runs a straight SELECTOR TRACK: a slotted C-channel whose "
            "overhanging lip strips leave only a narrow slot on top. Inside rides "
            "the SELECTOR — a small block whose bright ORANGE TAB sticks up through "
            "the slot (about 48 mm proud). The selector can only SLIDE along the "
            "track; it cannot be lifted out (the lips capture it). Four colored "
            "TICK PLATES on the track's front face mark the channel positions, in "
            "order along the track: RED, GREEN, BLUE, YELLOW. Midway along the "
            "track, between the two green/blue inner ticks, a LOCK PIN stands "
            "upright: a dark square post with a light-grey square head (head about "
            f"{PIN_HEAD[0] * 1000:.0f} mm wide, top near "
            f"{(PIN_SEAT_Z + PIN_HALF_L + PIN_HEAD[2]) * 1000:.0f} mm height). Its "
            "shaft drops through the slot into a deep socket and completely blocks "
            "the selector from passing the middle of the track. On the pedestal at "
            "the console's rear-right corner sits a single colored INDICATOR CUBE: "
            "its color names the TARGET CHANNEL. (Identical spare cubes lie far "
            "away on the floor; ignore them.) The selector always starts on the "
            "opposite side of the lock pin from the target channel.\n"
            "Goal: first pull the lock pin STRAIGHT UP out of its socket — it must "
            f"rise about {(LIP_TOP - SOCKET_FLOOR) * 1000:.0f} mm before its shaft "
            "clears the slot — and set it aside anywhere out of the track. Then "
            "slide the selector along the track and leave it at rest with its "
            "orange tab centered on the tick plate whose color MATCHES the "
            f"indicator cube (within about {c.band_tol * 1000:.0f} mm). Only the "
            "matching channel counts: the selector resting at any other tick, "
            "stopped short of the band, still moving, or a block balanced on top "
            "of the track instead of riding inside it, all score nothing. The pin "
            "cannot be skipped: while it is seated the target side of the track is "
            "physically unreachable."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the square-headed lock pin straight up out of the selector track "
            "and set it aside. Then slide the orange-tabbed selector along the "
            "track until it rests at the tick plate whose color matches the "
            "indicator cube on the rear pedestal. The selector must end at rest "
            "inside the track at that matching channel."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def canon_to_world(self, canon: torch.Tensor) -> torch.Tensor:
        """(N, 3) canonical (console-frame) points -> world."""
        from isaaclab.utils.math import quat_apply

        return self.console.data.root_pos_w + quat_apply(
            self.console.data.root_quat_w, canon)

    def block_canon(self) -> torch.Tensor:
        """(N, 3): slider root in the console frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.console.data.root_quat_w,
            self.block.data.root_pos_w - self.console.data.root_pos_w)

    def pin_canon(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(root (N,3), shaft-bottom endpoint (N,3)) of the pin, console frame."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        cq = self.console.data.root_quat_w
        cp = self.console.data.root_pos_w
        root_w = self.pin.data.root_pos_w
        tip_w = root_w + quat_apply(
            self.pin.data.root_quat_w,
            torch.tensor([0.0, 0.0, -PIN_HALF_L], device=root_w.device
                         ).expand_as(root_w))
        return (quat_apply_inverse(cq, root_w - cp),
                quat_apply_inverse(cq, tip_w - cp))

    # ----- predicates -------------------------------------------------------------------------
    def pin_blocking(self) -> torch.Tensor:
        """(N,) bool: the pin obstructs the interlock column (root over the column
        and shaft bottom below the lip slot)."""
        c = self.cfg
        root, bot = self.pin_canon()
        return (root[:, 0].abs() < c.clear_col_xy) \
            & (root[:, 1].abs() < c.clear_col_xy) \
            & (bot[:, 2] < c.clear_bot_z)

    def in_track(self) -> torch.Tensor:
        """(N,) bool: slider riding INSIDE the channel (y + z windows; an imposter
        resting on the lip tops reads ~38 mm high in z and is rejected)."""
        c = self.cfg
        p = self.block_canon()
        return (p[:, 1].abs() <= c.y_tol) \
            & ((p[:, 2] - BLOCK_REST_Z).abs() <= c.z_tol) \
            & (p[:, 0].abs() <= TRAVEL + 0.004)

    def in_band(self) -> torch.Tensor:
        """(N,) bool: slider center inside the target band."""
        return (self.block_canon()[:, 0] - self.x_target).abs() <= self.cfg.band_tol

    def settled(self) -> torch.Tensor:
        """(N,) bool: slider at rest."""
        c = self.cfg
        return (self.block.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.block.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch every physics substep: pin ever clear of the interlock column,
        slider ever fully across the pin column onto the target side, best approach
        toward the target band (normalized by the episode's start distance)."""
        c = self.cfg
        self.pin_clear = torch.maximum(self.pin_clear,
                                       (~self.pin_blocking()).float())
        bx = self.block_canon()[:, 0]
        tsign = torch.sign(self.x_target)
        self.crossed = torch.maximum(self.crossed, (bx * tsign > 0.030).float())
        d = (bx - self.x_target).abs()
        frac = ((self.d0 - d) / (self.d0 - c.band_tol).clamp_min(1e-6)).clamp(0.0, 1.0)
        self.appr_max = torch.maximum(self.appr_max, frac)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: slider at rest inside the track, centered in the TARGET band.
        The pin phase is physically necessary: with the pin seated, every far-side
        band is unreachable (asserted in cfg, proved by force probe in smoke)."""
        return self.in_band() & self.in_track() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 pin-clear latch + 0.15 crossed latch + 0.25 x
        best-approach latch, cap 0.60; 1.0 iff success(). Latched — credit never
        evaporates; the null policy scores ~0."""
        base = (0.20 * self.pin_clear + 0.15 * self.crossed
                + 0.25 * self.appr_max).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="channel_console", robot="null",
                                      env_spacing=4.0))
