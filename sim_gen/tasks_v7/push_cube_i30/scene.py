"""SiloScoopScene — ladle the amber balls out of a deep fixed silo with a long-handled
dustpan scoop, dump them into the shallow tray, then lay the scoop down clear
(sim_gen task `push_cube_i30`).

Derived from maniskill/push_cube, but STRATEGICALLY different: the seed's plan is a
single direct non-prehensile PUSH — put the gripper behind a free cube on an open
table and shove it a few centimetres into a marked goal region; one object, one
contact, one straight-line stroke, judged by 2D proximity. Here nothing can be
solved by pushing the payload along the support surface at all: the payloads (one
or two 25 mm amber balls) start at the BOTTOM of a tall, bolted-down, open-top SILO
(interior 240 x 115 mm, 300 mm deep) whose mouth is far too deep and narrow for a
hand — and the goal container (a low walled TRAY) is a separate fixture across the
floor. The only viable plan is TOOL-MEDIATED EXTRACTION AND TRANSFER: acquire the
long-handled dustpan SCOOP, dip its pan down through the silo mouth, press the pan
to the silo floor and stroke it toward the far wall so the balls are shovelled up
the ramp lip into the pan (a dustpan-against-the-wall move — pure contact), lift
the loaded pan back out of the mouth, CARRY the loose balls through the air riding
in the open pan, tip the pan nose-down over the tray so they roll out and drop in,
and finally lay the scoop down on open floor clear of both containers. A solver
needs a different PLAN (tool acquisition, constrained dip, shovel-against-wall
capture, aerial carry of an unsecured payload, pour-out, tool stow) and different
code (scoop-frame containment and 6-DoF tool control instead of a planar push).

Judged on PHYSICAL outcomes only, when settled (< 0.05 m/s):
  success() = every PRESENT ball rests inside the tray (tray-frame containment:
  centre inside the inner walls, below the wall top) AND the scoop is STOWED —
  resting low on open floor, clear of both the silo and the tray footprints.
score() is latched every physics substep and never evaporates: per present ball,
0.25 * the ball has ever been OUT of the silo's interior volume + 0.45 * the ball
has ever been IN the tray (present-mean, cap 0.70); exactly 1.0 iff success().
Doing nothing scores ~0 (the scoop starts on the floor, but the stow clause only
counts through success(), which needs the balls delivered).

Assets are fully procedural (no external files):
  - silo: ONE kinematic fixture — slate-blue walls 300 mm tall on a pale-sand
    interior floor (240 x 115 mm), wall 12 mm; the two far-end interior corners
    carry 45 deg chamfer blocks so a shovelled ball can never lodge in a corner
    outside the pan mouth; rim at 312 mm. Bolted down (kinematic): it can be
    neither lifted nor tipped to spill.
  - tray (basin): ONE kinematic fixture — green walls 50 mm on a white floor,
    inner 160 x 160 mm; low enough to dump into from above, tall enough that a
    delivered ball stays.
  - scoop: the only DYNAMIC tool — a red dustpan (pan 48 x 92 mm, 28 mm side
    walls, 36 mm back wall, full-width ramp lip that scrapes the floor) with a
    vertical black handle (14 mm dia, 400 mm long) rising from the back wall, so
    the pan reaches the silo floor while the grip stays above the rim. 150 g.
  - balls: one or two amber 25 mm spheres, 15 g — they fit the pan with room, and
    they cannot slip through the 7.5 mm gaps beside the pan during a stroke.
Contact offsets are explicit and small (2 mm): the defaults would eat the pan's
7.5 mm per-side dip clearance.

Honesty is asserted in cfg.__post_init__: the pan fits the mouth, balls cannot
slip past the pan, the ramp climb is below ball radius, the grip clears the rim at
full depth, the guaranteed ball-free entry window at the near end exceeds the pan
footprint, and the tray containment margin exceeds the largest physically-contained
offset (any ball actually inside the tray counts).

Per-episode randomization (verified by readback in smoke): silo xy + yaw, tray
xy + yaw, scoop spawn xy + free yaw (lying on its side), the PRESENT ball count
(1 or 2, judged on the sampled subset; absent balls park in a floor depot), and
each present ball's silo-local xy in the far half of the silo floor. Heavy imports
(isaaclab, pxr) are deferred so importing this module — and registering the scene —
stays app-free.
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float,
         yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
    """Unit cube scaled to `size` at `center` (parent frame), optional yaw (about z)
    then pitch (about y). Ops author translate -> rotZ -> rotY -> scale (applied
    right-to-left), each exactly once — idempotent decorations, no duplicate ops."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        sxf.AddRotateZOp().Set(float(yaw_deg))
    if pitch_deg:
        sxf.AddRotateYOp().Set(float(pitch_deg))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_silo(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC silo at `prim_path`. Local origin = bottom centre on
    the ground (z = 0 local); interior floor top at `floor_t`; rim at
    `floor_t + depth`. Chamfer blocks cut the two +x interior corners so a
    shovelled ball funnels into the pan mouth instead of lodging."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)

    co = cfg.contact_offset
    ix, iy, t, d, ft = cfg.in_x, cfg.in_y, cfg.wall_t, cfg.depth, cfg.floor_t
    wall, floor = cfg.wall_color, cfg.floor_color
    # floor plate (top at local z = ft)
    _box(stage, f"{prim_path}/floor", (ix + 2 * t, iy + 2 * t, ft),
         (0.0, 0.0, ft / 2), floor, co)
    # walls from ft to ft + d
    zc = ft + d / 2
    _box(stage, f"{prim_path}/wall_n", (ix + 2 * t, t, d),
         (0.0, iy / 2 + t / 2, zc), wall, co)
    _box(stage, f"{prim_path}/wall_s", (ix + 2 * t, t, d),
         (0.0, -iy / 2 - t / 2, zc), wall, co)
    _box(stage, f"{prim_path}/wall_e", (t, iy, d), (ix / 2 + t / 2, 0.0, zc), wall, co)
    _box(stage, f"{prim_path}/wall_w", (t, iy, d), (-ix / 2 - t / 2, 0.0, zc), wall, co)
    # +x corner chamfers (45 deg), full height: no corner pocket at the shovel wall
    leg = cfg.chamfer_leg
    cx = ix / 2 - leg / 2 + 0.004
    cy = iy / 2 - leg / 2 + 0.004
    _box(stage, f"{prim_path}/chamfer_ne", (leg * 1.5, t, d), (cx, cy, zc), wall, co,
         yaw_deg=-45.0)
    _box(stage, f"{prim_path}/chamfer_se", (leg * 1.5, t, d), (cx, -cy, zc), wall, co,
         yaw_deg=45.0)
    return root


def _spawn_basin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC tray at `prim_path`. Local origin = bottom centre on the
    ground; floor top at `b_floor_t`; wall top at `b_floor_t + b_wall_h`."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)

    co = cfg.contact_offset
    bi, t, h, ft = cfg.b_inner, cfg.b_wall_t, cfg.b_wall_h, cfg.b_floor_t
    _box(stage, f"{prim_path}/floor", (bi + 2 * t, bi + 2 * t, ft),
         (0.0, 0.0, ft / 2), cfg.b_floor_color, co)
    zc = ft + h / 2
    _box(stage, f"{prim_path}/wall_e", (t, bi + 2 * t, h), (bi / 2 + t / 2, 0.0, zc),
         cfg.b_wall_color, co)
    _box(stage, f"{prim_path}/wall_w", (t, bi + 2 * t, h), (-bi / 2 - t / 2, 0.0, zc),
         cfg.b_wall_color, co)
    _box(stage, f"{prim_path}/wall_n", (bi, t, h), (0.0, bi / 2 + t / 2, zc),
         cfg.b_wall_color, co)
    _box(stage, f"{prim_path}/wall_s", (bi, t, h), (0.0, -bi / 2 - t / 2, zc),
         cfg.b_wall_color, co)
    return root


def _spawn_scoop(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC scoop at `prim_path`. Local origin = centre of the pan
    floor TOP (z = 0 local, pan bottom at -pan_t). +x is the NOSE direction: a
    full-width ramp lip pitched down so its leading edge scrapes the ground; side
    walls, a taller back wall, and a long vertical grasp handle above the back."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.60)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    co = cfg.contact_offset
    pl, pw, pt = cfg.pan_len, cfg.pan_w, cfg.pan_t
    st, sh = cfg.side_t, cfg.side_h
    bt, bh = cfg.back_t, cfg.back_h
    rl, rt = cfg.ramp_len, cfg.ramp_t
    body, hcol = cfg.body_color, cfg.handle_color
    # pan floor: top at local z = 0
    _box(stage, f"{prim_path}/pan", (pl, pw, pt), (0.0, 0.0, -pt / 2), body, co)
    # ramp lip: pitched so the leading edge sits just below the pan bottom (scrapes)
    pitch = math.degrees(math.atan2(pt + 0.001, rl))
    _box(stage, f"{prim_path}/ramp", (rl, pw, rt),
         (pl / 2 + rl / 2 * math.cos(math.radians(pitch)), 0.0, -pt / 2 + 0.0005),
         body, co, pitch_deg=pitch)
    # side walls
    _box(stage, f"{prim_path}/side_l", (pl, st, sh),
         (0.0, pw / 2 + st / 2, sh / 2 - pt), body, co)
    _box(stage, f"{prim_path}/side_r", (pl, st, sh),
         (0.0, -pw / 2 - st / 2, sh / 2 - pt), body, co)
    # back wall (full outer width)
    _box(stage, f"{prim_path}/back", (bt, pw + 2 * st, bh),
         (-pl / 2 - bt / 2, 0.0, bh / 2 - pt), body, co)
    # handle: vertical from the back wall top
    _cyl(stage, f"{prim_path}/handle", cfg.handle_r, cfg.handle_len,
         (-pl / 2 - bt / 2, 0.0, bh - pt + cfg.handle_len / 2), hcol, co)
    return root


def _make_spawner(key: str, spawn_func: Callable, defaults: dict[str, Any],
                  overrides: dict[str, Any], *, kinematic: bool, mass: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if key not in _SPAWNER_CACHE:
        ns = {"func": clone(spawn_func), **defaults}
        ns["__annotations__"] = {"func": Callable,
                                 **{k: type(v) for k, v in defaults.items()}}
        _SPAWNER_CACHE[key] = configclass(
            type(f"{key.title()}SpawnerCfg", (RigidObjectSpawnerCfg,), ns))
    rigid = (sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
             if kinematic else sim_utils.RigidBodyPropertiesCfg())
    return _SPAWNER_CACHE[key](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass), rigid_props=rigid,
        **overrides)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SiloScoopCfg(BaseCfg):
    """Config for `SiloScoopScene`. Honesty/feasibility knobs are asserted in
    `__post_init__` (see class docstring in the module header)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    basin_margin: float = tunable(0.004)  # in-tray: |xy| < b_inner/2 - this (m)
    settle_lin: float = tunable(0.05)  # max |lin vel| of balls + scoop when judging (m/s)
    stow_z_max: float = tunable(0.09)  # stowed: scoop origin below this height (m)
    stow_clear: float = tunable(0.22)  # stowed: scoop origin farther than this from both
    # fixture centres (> each fixture's half-diagonal: the scoop is fully off both)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter: float = tunable(0.05)  # uniform +/- xy jitter of silo AND tray (m)
    silo_yaw_deg: float = tunable(20.0)  # uniform +/- silo yaw
    basin_yaw_deg: float = tunable(25.0)  # uniform +/- tray yaw
    scoop_jitter: float = tunable(0.06)  # uniform +/- xy jitter of the scoop spawn (m)
    min_present: int = tunable(1)  # sampled present-ball count: U{min_present..2}
    ball_x_range: tuple = tunable((0.010, 0.095))  # ball spawn, silo-local x (far half)
    ball_y_half: float = tunable(0.036)  # ball spawn, silo-local |y| bound
    ball_keepout: float = tunable(0.030)  # min spawn distance between balls (m)

    # --- tunable: placement (world xy nominal) -----------------------------------------------
    silo_pos: tuple = tunable((0.38, 0.26))
    basin_pos: tuple = tunable((0.38, -0.32))
    scoop_pos: tuple = tunable((0.06, -0.03))
    depot_pos: tuple = tunable((-0.85, 0.85))  # floor depot for ABSENT balls

    # --- info: silo structure ----------------------------------------------------------------
    in_x: float = info(0.240)  # interior floor length (shovel axis)
    in_y: float = info(0.115)  # interior floor width (mouth width)
    depth: float = info(0.300)  # interior depth, floor top -> rim
    wall_t: float = info(0.012)
    floor_t: float = info(0.012)  # interior floor top above ground
    chamfer_leg: float = info(0.010)  # 45 deg corner blocks at the +x (shovel) wall
    # (sized so a corner-lodged ball funnels into the ramp span, yet the full-width
    # ramp's corners CLEAR the chamfer faces at full stroke — asserted below)
    # --- info: tray (basin) ------------------------------------------------------------------
    b_inner: float = info(0.160)
    b_wall_t: float = info(0.010)
    b_wall_h: float = info(0.050)
    b_floor_t: float = info(0.010)
    # --- info: scoop (the tool) --------------------------------------------------------------
    pan_len: float = info(0.048)  # pan floor, x (nose axis)
    pan_w: float = info(0.092)  # pan floor, y
    pan_t: float = info(0.004)
    side_t: float = info(0.004)
    side_h: float = info(0.028)
    back_t: float = info(0.005)
    back_h: float = info(0.036)
    ramp_len: float = info(0.026)  # shallow (~11 deg) lip: balls roll up, never jam
    ramp_t: float = info(0.003)
    handle_r: float = info(0.007)  # 14 mm dia: a comfortable parallel-jaw grasp
    handle_len: float = info(0.400)
    mass: float = info(0.15)  # scoop mass (`cfg.mass` consumed by the spawner)
    # --- info: balls -------------------------------------------------------------------------
    ball_r: float = info(0.0125)
    ball_mass: float = info(0.015)
    n_balls: int = info(2)
    jaw_span: float = info(0.080)  # the Franka jaw the handle must fit (it does)
    # --- info: colors ------------------------------------------------------------------------
    wall_color: tuple = info((0.30, 0.36, 0.46))  # silo: slate blue
    floor_color: tuple = info((0.85, 0.78, 0.55))  # silo interior floor: pale sand
    b_wall_color: tuple = info((0.15, 0.55, 0.25))  # tray walls: green
    b_floor_color: tuple = info((0.92, 0.92, 0.92))  # tray floor: white
    body_color: tuple = info((0.75, 0.15, 0.12))  # scoop body: red
    handle_color: tuple = info((0.12, 0.12, 0.14))  # handle: near-black
    ball_color: tuple = info((0.95, 0.72, 0.10))  # balls: amber
    # Explicit small offsets: defaults would eat the 7.5 mm per-side dip clearance.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    rim_z: float = field(default=None, init=False)  # silo rim height above ground
    pan_outer_w: float = field(default=None, init=False)
    nose_x: float = field(default=None, init=False)  # scoop-local x of the ramp tip
    back_x: float = field(default=None, init=False)  # scoop-local x of the back face
    handle_top: float = field(default=None, init=False)  # scoop-local z of handle top

    def __post_init__(self) -> None:
        self.rim_z = self.floor_t + self.depth
        self.pan_outer_w = self.pan_w + 2 * self.side_t
        self.nose_x = self.pan_len / 2 + self.ramp_len * math.cos(
            math.atan2(self.pan_t + 0.001, self.ramp_len)) + 0.001
        self.back_x = -self.pan_len / 2 - self.back_t
        self.handle_top = self.back_h - self.pan_t + self.handle_len

        # the pan dips through the mouth with real (but ball-proof) side clearance
        gap = (self.in_y - self.pan_outer_w) / 2
        assert gap >= 0.006, "pan must fit the silo mouth with >= 6 mm per side"
        assert gap < 2 * self.ball_r, "balls must NOT slip through the side gaps"
        # the dustpan move works: climb below ball radius, ball fits the pan
        assert self.pan_t <= self.ball_r, "ramp climb must stay below ball radius"
        # the ramp is shallow enough that a pushed ball rolls up rather than jamming
        # in the ramp/wall wedge (a steep lip self-locks under friction)
        assert math.atan2(self.pan_t + 0.001, self.ramp_len) < math.radians(15.0), (
            "ramp must stay shallow enough to not self-lock")
        assert self.pan_len > 2 * self.ball_r + 0.010, "pan must hold a ball with room"
        assert self.side_h > self.ball_r + 0.010, "side walls must retain a riding ball"
        # the grip stays above the rim when the pan is at the silo floor
        assert self.floor_t + self.pan_t + self.handle_top >= self.rim_z + 0.10, (
            "handle grip must clear the rim at full dip depth")
        # a hand cannot plausibly reach the floor: deep, narrow mouth
        assert self.depth >= 0.28 and self.in_y <= 0.13, "silo must defeat direct reach"
        # corner chamfers: the inner faces lie on the plane x + y = c_plane
        # (silo-local, +x/+y corner; mirrored). They must (a) leave the FULL-WIDTH
        # ramp's corners clear at full stroke (nose 2 mm short of the wall) — a
        # bigger chamfer JAMS the stroke 17 mm short (observed) — and (b) still
        # funnel a corner-lodged ball into the ramp span.
        cxc = self.in_x / 2 - self.chamfer_leg / 2 + 0.004
        cyc = self.in_y / 2 - self.chamfer_leg / 2 + 0.004
        c_plane = cxc + cyc - self.wall_t / 2 * math.sqrt(2.0)
        assert c_plane >= (self.in_x / 2 - 0.002) + self.pan_w / 2 + 0.002, (
            "chamfers must not block the full-width ramp at full stroke")
        corner_ball_y = (c_plane - self.ball_r * math.sqrt(2.0)) \
            - (self.in_x / 2 - self.ball_r)
        assert corner_ball_y <= self.pan_w / 2 - 0.003, (
            "chamfers must funnel a corner-lodged ball into the ramp span")
        assert 1.5 * self.chamfer_leg >= math.sqrt(2.0) * (
            self.in_x / 2 - (c_plane - self.in_y / 2)), (
            "chamfer faces must span the whole corner pocket")
        # guaranteed ball-free ENTRY WINDOW at the -x end exceeds the pan footprint
        window = (self.ball_x_range[0] + self.in_x / 2) - self.ball_r - 0.004
        assert window >= (self.nose_x - self.back_x) + 0.010, (
            "entry window must exceed the scoop footprint")
        # tray containment margin is honest: any physically-contained ball counts
        assert self.b_inner / 2 - self.basin_margin > self.b_inner / 2 - self.ball_r, (
            "in-tray margin must admit a ball touching the wall")
        assert self.b_wall_h >= 2.5 * self.ball_r, "tray wall must retain delivered balls"
        # stow clearance ring fully clears both fixture footprints
        silo_hd = math.hypot(self.in_x / 2 + self.wall_t, self.in_y / 2 + self.wall_t)
        basin_hd = math.hypot(self.b_inner / 2 + self.b_wall_t,
                              self.b_inner / 2 + self.b_wall_t) + 0.0
        assert self.stow_clear > max(silo_hd, basin_hd) + 0.05, (
            "stow_clear must put the scoop fully off both fixtures")
        # fixtures cannot collide under jitter
        sep = math.hypot(self.silo_pos[0] - self.basin_pos[0],
                         self.silo_pos[1] - self.basin_pos[1]) - 2 * self.fix_jitter
        assert sep > silo_hd + basin_hd + 0.08, "silo and tray must never overlap"
        # the handle fits the jaw
        assert 2 * self.handle_r <= self.jaw_span - 0.02, "handle must fit the jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("silo_scoop")
class SiloScoopScene(BaseScene):
    cfg: SiloScoopCfg

    def __init__(self, cfg: SiloScoopCfg | None = None) -> None:
        super().__init__(cfg or SiloScoopCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        silo_keys = ("in_x", "in_y", "depth", "wall_t", "floor_t", "chamfer_leg",
                     "wall_color", "floor_color", "contact_offset")
        basin_keys = ("b_inner", "b_wall_t", "b_wall_h", "b_floor_t",
                      "b_wall_color", "b_floor_color", "contact_offset")
        scoop_keys = ("pan_len", "pan_w", "pan_t", "side_t", "side_h", "back_t",
                      "back_h", "ramp_len", "ramp_t", "handle_r", "handle_len",
                      "mass", "body_color", "handle_color", "contact_offset")

        def sub(keys) -> dict[str, Any]:
            return {k: getattr(c, k) for k in keys}

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "silo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Silo",
                spawn=_make_spawner("silo", _spawn_silo, sub(silo_keys), sub(silo_keys),
                                    kinematic=True, mass=30.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(*c.silo_pos, 0.0)),
            ),
            "basin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_make_spawner("basin", _spawn_basin, sub(basin_keys),
                                    sub(basin_keys), kinematic=True, mass=10.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(*c.basin_pos, 0.0)),
            ),
            "scoop": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Scoop",
                spawn=_make_spawner("scoop", _spawn_scoop, sub(scoop_keys),
                                    sub(scoop_keys), kinematic=False, mass=c.mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(*c.scoop_pos, 0.05),
                    rot=(math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)),
            ),
        }
        for i in range(c.n_balls):
            out[f"ball_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.20,
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.silo_pos[0], c.silo_pos[1] + 0.02 * i,
                         c.floor_t + c.ball_r + 0.003)),
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
        c = self.cfg
        self.silo: RigidObject = env.iscene["silo"]
        self.basin: RigidObject = env.iscene["basin"]
        self.scoop: RigidObject = env.iscene["scoop"]
        self.balls: list[RigidObject] = [env.iscene[f"ball_{i}"]
                                         for i in range(c.n_balls)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.present = torch.ones(n, c.n_balls, dtype=torch.bool, device=dev)
        self.out_latch = torch.zeros(n, c.n_balls, device=dev)
        self.basin_latch = torch.zeros(n, c.n_balls, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: silo + tray with xy jitter and yaw, the scoop lying on its
        side with free yaw (handle-tip keep-out from both fixtures), the present
        ball subset sampled (1..2) and scattered on the FAR half of the silo floor
        with mutual keep-out; absent balls park in the floor depot; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def jxy(nominal: tuple, amp: float) -> torch.Tensor:
            out = torch.tensor(nominal, device=dev).expand(m, 2).clone()
            return out + (torch.rand(m, 2, device=dev) * 2 - 1) * amp

        def write(body, xy: torch.Tensor, z, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
            q = torch.zeros(m, 4, device=dev)
            q[:, 0] = torch.cos(yaw / 2)
            q[:, 3] = torch.sin(yaw / 2)
            return q

        # --- fixtures ---
        silo_xy = jxy(c.silo_pos, c.fix_jitter)
        silo_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.silo_yaw_deg)
        write(self.silo, silo_xy, 0.0, yaw_quat(silo_yaw))
        basin_xy = jxy(c.basin_pos, c.fix_jitter)
        basin_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.basin_yaw_deg)
        write(self.basin, basin_xy, 0.0, yaw_quat(basin_yaw))

        # --- scoop: lying on its back-wall face, free yaw, handle-tip keep-out ---
        sc_xy = jxy(c.scoop_pos, c.scoop_jitter)
        sc_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        for _ in range(12):
            # after q = qz(yaw) x qy(-90), the handle (+z local) points along -x(yaw)
            tip = sc_xy - torch.stack([torch.cos(sc_yaw), torch.sin(sc_yaw)],
                                      dim=-1) * (c.handle_top + 0.02)
            bad = ((tip - silo_xy).norm(dim=-1) < c.stow_clear) \
                | ((tip - basin_xy).norm(dim=-1) < c.stow_clear) \
                | ((sc_xy - silo_xy).norm(dim=-1) < c.stow_clear) \
                | ((sc_xy - basin_xy).norm(dim=-1) < c.stow_clear)
            if not bad.any():
                break
            k = int(bad.sum())
            sc_xy[bad] = torch.tensor(c.scoop_pos, device=dev) \
                + (torch.rand(k, 2, device=dev) * 2 - 1) * c.scoop_jitter
            sc_yaw[bad] = (torch.rand(k, device=dev) * 2 - 1) * math.pi
        cy, sy = torch.cos(sc_yaw / 2), torch.sin(sc_yaw / 2)
        c45 = math.cos(math.pi / 4)
        q = torch.zeros(m, 4, device=dev)  # qz(yaw) x qy(-90 deg)
        q[:, 0] = cy * c45
        q[:, 1] = sy * c45
        q[:, 2] = -cy * c45
        q[:, 3] = sy * c45
        write(self.scoop, sc_xy, -c.back_x + 0.004, q)

        # --- balls: sample the present subset, scatter on the far half of the floor ---
        k_present = torch.randint(c.min_present, c.n_balls + 1, (m,), device=dev)
        rank = torch.rand(m, c.n_balls, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k_present.unsqueeze(1)

        x_lo, x_hi = c.ball_x_range
        cyw, syw = torch.cos(silo_yaw), torch.sin(silo_yaw)
        placed: list[torch.Tensor] = []
        for i in range(c.n_balls):
            loc = torch.zeros(m, 2, device=dev)
            loc[:, 0] = x_lo + torch.rand(m, device=dev) * (x_hi - x_lo)
            loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_half
            for _ in range(12):
                bad = torch.zeros(m, dtype=torch.bool, device=dev)
                for prev in placed:
                    bad |= (loc - prev).norm(dim=-1) < c.ball_keepout
                if not bad.any():
                    break
                kk = int(bad.sum())
                loc[bad, 0] = x_lo + torch.rand(kk, device=dev) * (x_hi - x_lo)
                loc[bad, 1] = (torch.rand(kk, device=dev) * 2 - 1) * c.ball_y_half
            placed.append(loc)
            in_silo_xy = torch.stack(
                [silo_xy[:, 0] + loc[:, 0] * cyw - loc[:, 1] * syw,
                 silo_xy[:, 1] + loc[:, 0] * syw + loc[:, 1] * cyw], dim=-1)
            depot_xy = torch.tensor(c.depot_pos, device=dev).expand(m, 2).clone()
            depot_xy[:, 0] += 0.08 * i
            pres = self.present[env_ids, i].unsqueeze(1)
            xy = torch.where(pres, in_silo_xy, depot_xy)
            z = torch.where(self.present[env_ids, i],
                            torch.full((m,), c.floor_t + c.ball_r + 0.003, device=dev),
                            torch.full((m,), c.ball_r + 0.003, device=dev))
            qi = torch.zeros(m, 4, device=dev)
            qi[:, 0] = 1.0
            write(self.balls[i], xy, z, qi)

        self.out_latch[env_ids] = 0.0
        self.basin_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "silo": self.silo.data.root_state_w[env_ids].clone(),
            "basin": self.basin.data.root_state_w[env_ids].clone(),
            "scoop": self.scoop.data.root_state_w[env_ids].clone(),
            "balls": [b.data.root_state_w[env_ids].clone() for b in self.balls],
            "present": self.present[env_ids].clone(),
            "out_latch": self.out_latch[env_ids].clone(),
            "basin_latch": self.basin_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.silo.write_root_state_to_sim(state["silo"], env_ids)
        self.basin.write_root_state_to_sim(state["basin"], env_ids)
        self.scoop.write_root_state_to_sim(state["scoop"], env_ids)
        for b, st in zip(self.balls, state["balls"]):
            b.write_root_state_to_sim(st, env_ids)
        self.present[env_ids] = state["present"]
        self.out_latch[env_ids] = state["out_latch"]
        self.basin_latch[env_ids] = state["basin_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stand two fixed containers and one loose tool. The SILO is a "
            f"tall slate-blue open-top bin (interior {c.in_x * 1000:.0f} x "
            f"{c.in_y * 1000:.0f} mm, {c.depth * 1000:.0f} mm deep, rim "
            f"{c.rim_z * 1000:.0f} mm up, pale sand floor), bolted down — it cannot be "
            f"moved or tipped, and its mouth is far too deep and narrow for a hand to "
            f"reach the bottom. Lying at the bottom of the silo are AMBER BALLS "
            f"({2 * c.ball_r * 1000:.0f} mm diameter): one or two are present — look "
            f"down into the mouth and count them. Across the floor sits the TRAY, a "
            f"shallow fixed basin with GREEN walls ({c.b_wall_h * 1000:.0f} mm tall) "
            f"and a white floor, inner {c.b_inner * 1000:.0f} x "
            f"{c.b_inner * 1000:.0f} mm. Between them lies the SCOOP: a red dustpan "
            f"(pan {c.pan_len * 1000:.0f} x {c.pan_w * 1000:.0f} mm with side walls, a "
            f"back wall, and a front ramp lip that scrapes the ground) with a straight "
            f"near-black handle ({2 * c.handle_r * 1000:.0f} mm diameter, "
            f"{c.handle_len * 1000:.0f} mm long) rising from its back — the pan fits "
            f"down through the silo mouth with the grip still above the rim.\n"
            f"Goal: every amber ball must end up resting INSIDE the tray (between its "
            f"green walls, on its white floor), and the scoop must finish laid down on "
            f"open floor, well clear of both the silo and the tray. The intended "
            f"tool-use: grasp the handle, lower the pan down the silo mouth, press it "
            f"to the silo floor and slide it toward the far wall so the balls ride up "
            f"the ramp into the pan (the far corners are chamfered so balls funnel "
            f"into the pan), tip the pan back slightly and lift it out, carry the "
            f"balls over to the tray, and tip the pan nose-down over it so they roll "
            f"out and drop in. No particular order is required. A ball left anywhere "
            f"but inside the tray does not count, and the task is not complete while "
            f"the scoop rests inside either container or leans on one. Judged only "
            f"when everything is settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Using the red long-handled scoop, ladle every amber ball out of the tall "
            "blue silo and dump them into the green-walled tray, then lay the scoop "
            "down on open floor clear of both containers. Balls left anywhere but "
            "inside the tray, or a scoop left in or against a container, fail the task."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, fixture, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture body frame (origin at its bottom centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(fixture.data.root_quat_w,
                                  p_w - fixture.data.root_pos_w)

    def _ball_pos_w(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.balls], dim=1)  # (N,B,3)

    # ----- predicates -------------------------------------------------------------------------
    def in_silo(self) -> torch.Tensor:
        """(N,B) bool: ball centre inside the silo's interior VOLUME (slack bounds;
        above the rim or beyond the walls counts as out)."""
        c = self.cfg
        pos = self._ball_pos_w()
        n, b = pos.shape[0], pos.shape[1]
        loc = self._local(self.silo, pos.reshape(n * b, 3)).reshape(n, b, 3)
        return ((loc[:, :, 0].abs() < c.in_x / 2 + c.wall_t)
                & (loc[:, :, 1].abs() < c.in_y / 2 + c.wall_t)
                & (loc[:, :, 2] > 0.0) & (loc[:, :, 2] < c.rim_z))

    def in_basin(self) -> torch.Tensor:
        """(N,B) bool: ball centre inside the tray — between the inner walls (margin
        wider than any physically-contained offset) and BELOW the wall top: real
        containment, not proximity; a ball perched on the wall does not count."""
        c = self.cfg
        pos = self._ball_pos_w()
        n, b = pos.shape[0], pos.shape[1]
        loc = self._local(self.basin, pos.reshape(n * b, 3)).reshape(n, b, 3)
        lim = c.b_inner / 2 - c.basin_margin
        return ((loc[:, :, 0].abs() < lim) & (loc[:, :, 1].abs() < lim)
                & (loc[:, :, 2] > c.b_floor_t)
                & (loc[:, :, 2] < c.b_floor_t + c.b_wall_h))

    def scoop_stowed(self) -> torch.Tensor:
        """(N,) bool: the scoop rests LOW on open floor (origin below `stow_z_max`),
        clear of BOTH fixture footprints (origin farther than `stow_clear` from each
        centre — a scoop inside, on, or leaning against a container is not stowed)."""
        c = self.cfg
        p = self.scoop.data.root_pos_w - self.env_origins
        d_silo = (self.scoop.data.root_pos_w[:, :2]
                  - self.silo.data.root_pos_w[:, :2]).norm(dim=-1)
        d_basin = (self.scoop.data.root_pos_w[:, :2]
                   - self.basin.data.root_pos_w[:, :2]).norm(dim=-1)
        return (p[:, 2] < c.stow_z_max) & (d_silo > c.stow_clear) \
            & (d_basin > c.stow_clear)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every ball AND the scoop below `settle_lin`."""
        c = self.cfg
        ok = self.scoop.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for b in self.balls:
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        return ok

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch per-ball silo escape and tray containment each physics substep, so
        demonstrated progress keeps its credit."""
        self.out_latch = torch.maximum(self.out_latch, (~self.in_silo()).float())
        self.basin_latch = torch.maximum(self.basin_latch, self.in_basin().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every PRESENT ball settled inside the tray + the scoop stowed
        on open floor, everything settled."""
        delivered = (self.in_basin() | ~self.present).all(dim=1)
        return delivered & self.scoop_stowed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: present-mean of (0.25 * ball ever OUT of the silo +
        0.45 * ball ever IN the tray), capped at 0.70; exactly 1.0 iff success().
        Doing nothing scores ~0; the seed's strategy (push the payload along the
        support surface toward the goal) cannot even reach a ball and earns ~0."""
        pres = self.present.float()
        n_pres = pres.sum(dim=1).clamp(min=1.0)
        per_ball = 0.25 * self.out_latch + 0.45 * self.basin_latch
        base = ((per_ball * pres).sum(dim=1) / n_pres).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="silo_scoop", robot="null"))
