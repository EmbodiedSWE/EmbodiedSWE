"""SkittleGalleryScene — under-wall bowling: knock over the RED pin, and ONLY the red
pin, by sending the blue ball through the floor tunnel on the red pin's side of a
sealed gallery (sim_gen task `obstacle_i17`).

Derived from pick_place/obstacle, but STRATEGICALLY different: the seed's obstacle is
a low free-standing wall and its plan is pure guarded TRANSPORT — grasp the cube,
carry it up and OVER the wall along waypoints, set it on the goal, done; the hand
holds the object through the entire interaction and the wall only shapes the path.
Here the wall is part of a ROOFED, sealed gallery and the judged objects (two skittle
pins standing in two interior cells) are physically UNREACHABLE: the roof closes the
carry-over-the-top plan entirely, the viewing slits (35 mm letterboxes) pass nothing,
and the only openings that reach the cells are two 82 mm floor tunnels too small and
too deep for a hand (the pins stand >= 150 mm beyond the 60 mm-thick wall). The only
way to affect a pin is a PROJECTILE: pick up the free blue ball, aim it down the
correct lane, impart momentum, and RELEASE it before the funnel mouth — the outcome
is decided by free-flight momentum transfer after the hand lets go, the opposite of
the seed's held-object transport. On top of the ballistic mechanic sit a selection
and a restraint problem the seed has no analogue of: the red pin's lane is coin-
flipped per episode (identify it through the slits before shooting), the WHITE pin
must remain STANDING (an irreversible wrong-lane shot fails forever), and the big
orange ball is a decoy that cannot enter either tunnel. A solver therefore needs a
different PLAN (perceive lane -> stage -> aim -> launch -> hands-off outcome) and a
different code structure (a topple predicate + a keep-standing predicate + latched
ballistic progress), not a carry-the-payload waypoint follower.

Judged in the GALLERY's body frame (kinematic, xy + yaw randomized). success() iff:
  - the RED pin is DOWN inside the chamber: tilted > `down_deg` from vertical AND its
    centre below `down_z` (truly lying, not leaning) AND inside the gallery;
  - the WHITE pin still STANDS: within `stand_deg` of vertical, centre at standing
    height within `stand_z_tol`;
  - both pins settled and the ball no longer fast.
score() is latched every physics substep: 0.10 * best ball approach toward the RED
lane's tunnel mouth (normalized by the episode's own spawn distance) + 0.35 * ball
ever inside the red pin's cell + 0.20 * red pin ever tilted > `tilt_credit_deg` while
the ball had entered that cell (ballistic credit is gated on the ball actually having
gone through — waving forces outside earns nothing), capped at 0.65; exactly 1.0 iff
success(). Doing nothing scores ~0. Knocking the white pin down kills success
permanently (it cannot re-stand) — the wrong-lane shot is an irreversible failure,
exactly like real skittles.

Assets are fully procedural (no external files):
  - gallery: ONE kinematic gray fixture — a 740 mm-wide, 200 mm-tall, 60 mm-thick
    front wall pierced at floor level by two 82 x 82 mm square tunnels (lanes at
    x = +/-170 mm), a 140 x 35 mm letterbox viewing slit above each tunnel
    (z 100..135 mm — see in, pass nothing), a full ROOF, side walls, a back wall, a
    center divider sealing the interior into two cells, low guide alleys extending
    each tunnel inside, and splayed funnel walls outside each mouth (yawed boxes)
    that catch an imperfect shot;
  - pins: RED and WHITE 26 x 26 x 130 mm square pins (35 g), one standing per cell
    on its lane axis, 225 mm behind the outer face — which cell is red is coin-
    flipped per episode;
  - ball: BLUE 60 mm sphere, 250 g — fits the tunnels with real clearance;
  - decoy: ORANGE 96 mm sphere, 350 g — cannot enter tunnel, alley, or slit.
Contact offsets are explicit and small (2 mm): the default would eat the 22 mm
tunnel clearance.

Per-episode randomization (verified by readback in smoke): gallery xy + yaw, red/
white lane assignment coin flip, pin xy jitter, ball + decoy scatter on the open
floor with keep-out resampling. Heavy imports (isaaclab, pxr) are deferred so
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


# ----- custom compound spawner -----------------------------------------------------------------
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
         yaw_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        sxf.AddRotateZOp().Set(float(yaw_deg))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia and
    COM from the child colliders — the wide head makes the pin TOP-HEAVY), damping so
    parts settle promptly, no sleeping while we judge velocities."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(1.0)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC skittle pin at `prim_path`. Origin = mid-height on the axis;
    a slim square shaft carries a wide square head on top. Uniform collider density
    puts the COM well above mid-height: the pin is deliberately TOP-HEAVY, so once
    tipped past a few degrees it falls and essentially cannot land standing again —
    'down' is an absorbing outcome, 'standing' is only lost, never regained."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co, h2 = cfg.contact_offset, cfg.pin_h / 2
    shaft_h = cfg.pin_h - cfg.head_h
    _box(stage, f"{prim_path}/shaft", (cfg.pin_sq, cfg.pin_sq, shaft_h),
         (0.0, 0.0, -h2 + shaft_h / 2), cfg.color, co)
    _box(stage, f"{prim_path}/head", (cfg.head_sq, cfg.head_sq, cfg.head_h),
         (0.0, 0.0, h2 - cfg.head_h / 2), cfg.color, co)
    return root


def _pin_spawner_cfg(*, pin_sq: float, pin_h: float, head_sq: float, head_h: float,
                     mass: float, color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pin" not in _SPAWNER_CACHE:

        @configclass
        class SkittlePinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            pin_sq: float = 0.026
            pin_h: float = 0.130
            head_sq: float = 0.042
            head_h: float = 0.035
            mass: float = 0.045
            color: tuple = (0.85, 0.12, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pin"] = SkittlePinSpawnerCfg

    return _SPAWNER_CACHE["pin"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        pin_sq=pin_sq, pin_h=pin_h, head_sq=head_sq, head_h=head_h, mass=mass,
        color=color, contact_offset=contact_offset,
    )


def _spawn_gallery(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC gallery at `prim_path`. Local origin = centre of the
    OUTER FRONT FACE at ground level; +y runs INTO the chamber. The front wall spans
    y in [0, wall_t]; the two floor tunnels, the letterbox slits, roof, side walls,
    back wall, divider, interior guide alleys and exterior funnel splays are all
    axis-aligned boxes except the funnels, which are yawed."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)

    co = cfg.contact_offset
    col, dark = cfg.color, cfg.dark_color
    hw, wt, wh = cfg.half_w, cfg.wall_t, cfg.wall_h
    lo, bh, bhw = cfg.lane_off, cfg.bore_h, cfg.bore_w / 2
    sz0, sz1, shw = cfg.slit_z0, cfg.slit_z1, cfg.slit_w / 2
    yb = wt / 2  # front-wall y centre

    # --- front wall: 4 z-bands, the slit and bore bands split around the openings ---
    _box(stage, f"{prim_path}/fw_top", (2 * hw, wt, wh - sz1),
         (0.0, yb, (sz1 + wh) / 2), col, co)
    _box(stage, f"{prim_path}/fw_web", (2 * hw, wt, sz0 - bh),
         (0.0, yb, (bh + sz0) / 2), col, co)
    for tag, x0, x1 in (("sl_l", -hw, -lo - shw), ("sl_m", -lo + shw, lo - shw),
                        ("sl_r", lo + shw, hw)):
        _box(stage, f"{prim_path}/fw_{tag}", (x1 - x0, wt, sz1 - sz0),
             ((x0 + x1) / 2, yb, (sz0 + sz1) / 2), col, co)
    for tag, x0, x1 in (("bo_l", -hw, -lo - bhw), ("bo_m", -lo + bhw, lo - bhw),
                        ("bo_r", lo + bhw, hw)):
        _box(stage, f"{prim_path}/fw_{tag}", (x1 - x0, wt, bh),
             ((x0 + x1) / 2, yb, bh / 2), col, co)

    # --- roof, side walls, back wall, divider (a sealed two-cell chamber) ---
    depth = cfg.chamber_y1 + cfg.back_t
    _box(stage, f"{prim_path}/roof", (2 * hw, depth, cfg.roof_t),
         (0.0, depth / 2, wh + cfg.roof_t / 2), cfg.roof_color, co)
    for tag, s in (("side_l", -1.0), ("side_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.side_t, depth, wh),
             (s * (hw - cfg.side_t / 2), depth / 2, wh / 2), col, co)
    _box(stage, f"{prim_path}/back", (2 * (hw - cfg.side_t), cfg.back_t, wh),
         (0.0, cfg.chamber_y1 + cfg.back_t / 2, wh / 2), col, co)
    _box(stage, f"{prim_path}/divider", (cfg.div_t, cfg.chamber_y1 - wt, wh),
         (0.0, (wt + cfg.chamber_y1) / 2, wh / 2), dark, co)

    # --- interior guide alleys: two low rails extending each tunnel into its cell ---
    ag = cfg.alley_gap / 2 + cfg.alley_t / 2
    for lane_s in (-1.0, 1.0):
        for tag, s in (("n", -1.0), ("p", 1.0)):
            _box(stage, f"{prim_path}/alley_{'l' if lane_s < 0 else 'r'}{tag}",
                 (cfg.alley_t, cfg.alley_len, cfg.alley_h),
                 (lane_s * lo + s * ag, wt + cfg.alley_len / 2, cfg.alley_h / 2), dark, co)

    # --- exterior funnel splays: yawed walls that catch an imperfect shot ---
    a0 = bhw + cfg.funnel_t / 2 + 0.001  # throat centreline (inner face flush w/ bore)
    spread = cfg.funnel_out - bhw
    a1 = a0 + spread
    fl = math.hypot(cfg.funnel_len, spread)
    fyaw = math.degrees(math.atan2(spread, cfg.funnel_len))
    for lane_s in (-1.0, 1.0):
        for s in (-1.0, 1.0):
            _box(stage, f"{prim_path}/funnel_{'l' if lane_s < 0 else 'r'}{'n' if s < 0 else 'p'}",
                 (cfg.funnel_t, fl, cfg.funnel_h),
                 (lane_s * lo + s * (a0 + a1) / 2, -cfg.funnel_len / 2, cfg.funnel_h / 2),
                 dark, co, yaw_deg=s * fyaw)
    return root


def _gallery_spawner_cfg(*, half_w: float, wall_t: float, wall_h: float, roof_t: float,
                         bore_w: float, bore_h: float, lane_off: float, slit_w: float,
                         slit_z0: float, slit_z1: float, side_t: float, back_t: float,
                         chamber_y1: float, div_t: float, alley_len: float,
                         alley_gap: float, alley_h: float, alley_t: float,
                         funnel_len: float, funnel_out: float, funnel_h: float,
                         funnel_t: float, color: tuple, dark_color: tuple,
                         roof_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gallery" not in _SPAWNER_CACHE:

        @configclass
        class SkittleGallerySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gallery)
            half_w: float = 0.37
            wall_t: float = 0.06
            wall_h: float = 0.20
            roof_t: float = 0.02
            bore_w: float = 0.082
            bore_h: float = 0.082
            lane_off: float = 0.17
            slit_w: float = 0.14
            slit_z0: float = 0.10
            slit_z1: float = 0.135
            side_t: float = 0.03
            back_t: float = 0.06
            chamber_y1: float = 0.40
            div_t: float = 0.016
            alley_len: float = 0.10
            alley_gap: float = 0.086
            alley_h: float = 0.06
            alley_t: float = 0.012
            funnel_len: float = 0.14
            funnel_out: float = 0.10
            funnel_h: float = 0.06
            funnel_t: float = 0.012
            color: tuple = (0.50, 0.50, 0.53)
            dark_color: tuple = (0.32, 0.32, 0.36)
            roof_color: tuple = (0.40, 0.40, 0.44)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["gallery"] = SkittleGallerySpawnerCfg

    return _SPAWNER_CACHE["gallery"](
        mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        half_w=half_w, wall_t=wall_t, wall_h=wall_h, roof_t=roof_t, bore_w=bore_w,
        bore_h=bore_h, lane_off=lane_off, slit_w=slit_w, slit_z0=slit_z0,
        slit_z1=slit_z1, side_t=side_t, back_t=back_t, chamber_y1=chamber_y1,
        div_t=div_t, alley_len=alley_len, alley_gap=alley_gap, alley_h=alley_h,
        alley_t=alley_t, funnel_len=funnel_len, funnel_out=funnel_out,
        funnel_h=funnel_h, funnel_t=funnel_t, color=color, dark_color=dark_color,
        roof_color=roof_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SkittleGalleryCfg(BaseCfg):
    """Config for `SkittleGalleryScene`. Honesty knobs asserted in `__post_init__`:
    the ball threads tunnel + alley with real clearance, the decoy physically cannot
    enter anything, nothing passes the slits, pins are beyond hand reach, and a pin
    can fall FLAT in every direction inside its cell (no lean-on-wall limbo — the
    down/standing predicates are mutually exclusive)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    down_deg: float = tunable(60.0)  # red pin counts as down when tilted > this from vertical
    down_z: float = tunable(0.040)  # ... AND its centre below this (truly lying) (m)
    stand_deg: float = tunable(25.0)  # white pin counts as standing within this of vertical
    stand_z_tol: float = tunable(0.020)  # ... at standing centre height within this (m)
    tilt_credit_deg: float = tunable(30.0)  # latched partial credit once red tilts past this
    settle_lin: float = tunable(0.05)  # max pin |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.5)  # max pin |ang vel| when judging (rad/s)
    ball_settle: float = tunable(0.30)  # max ball |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the gallery (m)
    fix_yaw_deg: float = tunable(12.0)  # uniform +/- gallery yaw (read it from the scene)
    shuffle_lanes: bool = tunable(True)  # coin-flip which cell holds the RED pin
    pin_jx: float = tunable(0.005)  # pin lateral jitter inside its lane (m)
    pin_jy: float = tunable(0.010)  # pin depth jitter (m)
    ball_scatter: tuple = tunable((0.20, 0.08))  # +/- xy scatter of the ball spawn (m)
    decoy_scatter: tuple = tunable((0.20, 0.08))  # +/- xy scatter of the decoy spawn (m)

    # --- tunable: placement (gallery-local xy unless noted) ----------------------------------
    fix_pos: tuple = tunable((0.0, 0.12))  # gallery front-face centre, WORLD xy nominal
    ball_pos: tuple = tunable((0.0, -0.45))  # ball nominal (local: in front of the wall)
    decoy_pos: tuple = tunable((0.0, -0.60))  # decoy nominal (behind the staging zone)
    keepout_ball_decoy: float = tunable(0.12)  # min ball<->decoy spawn distance (m)

    # --- info: gallery structure -------------------------------------------------------------
    half_w: float = info(0.37)  # gallery half-width (m)
    wall_t: float = info(0.06)  # front wall thickness — a 60 mm-deep tunnel bore
    wall_h: float = info(0.20)  # wall height (roof sits on top: nothing goes over)
    roof_t: float = info(0.02)
    bore_w: float = info(0.082)  # tunnel width — ball 60 mm fits, decoy 96 mm cannot
    bore_h: float = info(0.082)  # tunnel height
    lane_off: float = info(0.17)  # lane axes at x = +/- this
    slit_w: float = info(0.14)  # letterbox slit width (perception only)
    slit_z0: float = info(0.10)  # slit bottom height
    slit_z1: float = info(0.135)  # slit top height (35 mm: both balls too thick)
    side_t: float = info(0.03)
    back_t: float = info(0.06)
    chamber_y1: float = info(0.40)  # inner face of the back wall (chamber depth)
    div_t: float = info(0.016)  # centre divider: the two cells are mutually sealed
    alley_len: float = info(0.10)  # interior guide rails extending each tunnel
    alley_gap: float = info(0.086)
    alley_h: float = info(0.06)
    alley_t: float = info(0.012)
    funnel_len: float = info(0.14)  # exterior splayed catch walls at each mouth
    funnel_out: float = info(0.10)
    funnel_h: float = info(0.06)
    funnel_t: float = info(0.012)
    # --- info: bodies ------------------------------------------------------------------------
    pin_y: float = info(0.225)  # pin depth behind the outer face (on the lane axis)
    pin_sq: float = info(0.026)  # pin shaft square cross-section
    pin_h: float = info(0.130)  # pin height (standing centre at pin_h/2)
    head_sq: float = info(0.042)  # wide square head: top-heavy, fills the viewing slit
    head_h: float = info(0.035)  # head height (head bottom at pin_h - head_h = 0.095)
    pin_mass: float = info(0.045)
    ball_r: float = info(0.030)  # 60 mm ball: 22 mm tunnel clearance
    ball_mass: float = info(0.25)
    decoy_r: float = info(0.048)  # 96 mm decoy: cannot enter tunnel/alley/slit
    decoy_mass: float = info(0.35)
    gallery_color: tuple = info((0.50, 0.50, 0.53))
    gallery_dark_color: tuple = info((0.32, 0.32, 0.36))
    gallery_roof_color: tuple = info((0.40, 0.40, 0.44))
    red_color: tuple = info((0.85, 0.12, 0.10))
    white_color: tuple = info((0.92, 0.92, 0.92))
    ball_color: tuple = info((0.15, 0.35, 0.90))
    decoy_color: tuple = info((0.95, 0.55, 0.10))
    # Explicit small offsets: the default ~2 cm would eat the 22 mm tunnel clearance.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    pin_z0: float = field(default=None, init=False)  # standing pin centre height
    fall_reach: float = field(default=None, init=False)  # tip sweep radius of a toppling pin

    def __post_init__(self) -> None:
        self.pin_z0 = self.pin_h / 2
        # a pin tipping over a base edge: pivot offset + reach of the wide head's far top edge
        self.fall_reach = self.pin_sq / 2 + math.hypot(self.pin_h, self.head_sq)

        # ball fits everything on the shot path with real clearance
        assert self.bore_w - 2 * self.ball_r >= 0.014, "ball must clear the tunnel width"
        assert self.bore_h - 2 * self.ball_r >= 0.014, "ball must clear the tunnel height"
        assert self.alley_gap - 2 * self.ball_r >= 0.014, "ball must clear the guide alley"
        assert self.alley_gap >= self.bore_w, "alley must not constrict behind the bore"
        assert self.funnel_out >= self.bore_w / 2 + 0.02, "funnel must widen the capture zone"
        # decoy fits NOTHING
        assert 2 * self.decoy_r >= self.bore_w + 0.010, "decoy must be unable to enter the tunnel"
        assert 2 * self.decoy_r >= self.bore_h + 0.010, "decoy must be unable to enter the tunnel"
        assert 2 * self.decoy_r >= self.alley_gap + 0.008, "decoy must be unable to enter the alley"
        # slits are perception-only
        assert 2 * self.ball_r >= (self.slit_z1 - self.slit_z0) + 0.015, (
            "nothing may pass the viewing slits")
        assert self.slit_z0 >= self.bore_h + 0.008, "web must separate slit from tunnel"
        assert self.slit_z0 + 0.02 <= self.pin_h, "a standing pin must show through the slit"
        assert self.pin_h - self.head_h <= self.slit_z0, (
            "the colored head must fill the slit band (lane identification by color)")
        assert self.head_sq > self.pin_sq, "head must overhang the shaft (top-heavy pin)"
        # pins are unreachable except by projectile
        assert self.pin_y - self.pin_jy - self.wall_t >= 0.10, (
            "pins must stand well beyond finger reach through the tunnel")
        assert self.wall_h - self.pin_h >= 0.04, "pins must sit far below the roofed wall top"
        # a pin can fall FLAT in every direction (no lean-on-wall limbo states)
        m = self.fall_reach + 0.004
        assert self.pin_y - self.pin_jy - self.wall_t >= m, "pin must fall flat forward"
        assert self.chamber_y1 - self.pin_y - self.pin_jy >= m, "pin must fall flat backward"
        assert self.lane_off - self.div_t / 2 - self.pin_jx >= m, "pin must fall flat inward"
        assert self.half_w - self.side_t - self.lane_off - self.pin_jx >= m, (
            "pin must fall flat outward")
        # a forward-falling pin drops INTO the alley channel, not onto a rail
        assert self.alley_gap / 2 - (self.head_sq * math.sqrt(2) / 2 + self.pin_jx) >= 0.005, (
            "forward-falling pin (widest part: the head) must fit between the alley rails")
        # down and standing are mutually exclusive
        assert self.down_deg >= self.stand_deg + 20.0, "down/standing tilt bands must separate"
        assert self.down_z < self.pin_z0 - self.stand_z_tol, "down/standing heights must separate"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("skittle_gallery")
class SkittleGalleryScene(BaseScene):
    cfg: SkittleGalleryCfg

    def __init__(self, cfg: SkittleGalleryCfg | None = None) -> None:
        super().__init__(cfg or SkittleGalleryCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def pin_cfg(color: tuple) -> Any:
            # Compound top-heavy pin (slim shaft + wide head): the COM sits high, so once
            # tipped past ~9 deg it falls and CANNOT re-stand — "down" is absorbing.
            return _pin_spawner_cfg(
                pin_sq=c.pin_sq, pin_h=c.pin_h, head_sq=c.head_sq, head_h=c.head_h,
                mass=c.pin_mass, color=color, contact_offset=c.contact_offset)

        def ball_cfg(radius: float, mass: float, color: tuple) -> Any:
            return sim_utils.SphereCfg(
                radius=radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.0, angular_damping=0.05,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        fx, fy = c.fix_pos
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
            "gallery": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gallery",
                spawn=_gallery_spawner_cfg(
                    half_w=c.half_w, wall_t=c.wall_t, wall_h=c.wall_h, roof_t=c.roof_t,
                    bore_w=c.bore_w, bore_h=c.bore_h, lane_off=c.lane_off,
                    slit_w=c.slit_w, slit_z0=c.slit_z0, slit_z1=c.slit_z1,
                    side_t=c.side_t, back_t=c.back_t, chamber_y1=c.chamber_y1,
                    div_t=c.div_t, alley_len=c.alley_len, alley_gap=c.alley_gap,
                    alley_h=c.alley_h, alley_t=c.alley_t, funnel_len=c.funnel_len,
                    funnel_out=c.funnel_out, funnel_h=c.funnel_h, funnel_t=c.funnel_t,
                    color=c.gallery_color, dark_color=c.gallery_dark_color,
                    roof_color=c.gallery_roof_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, 0.0)),
            ),
            "red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedPin",
                spawn=pin_cfg(c.red_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx - c.lane_off, fy + c.pin_y, c.pin_z0 + 0.002)),
            ),
            "white": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WhitePin",
                spawn=pin_cfg(c.white_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.lane_off, fy + c.pin_y, c.pin_z0 + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=ball_cfg(c.ball_r, c.ball_mass, c.ball_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.ball_pos[0], fy + c.ball_pos[1], c.ball_r + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=ball_cfg(c.decoy_r, c.decoy_mass, c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.decoy_pos[0], fy + c.decoy_pos[1], c.decoy_r + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.gallery: RigidObject = env.iscene["gallery"]
        self.red: RigidObject = env.iscene["red"]
        self.white: RigidObject = env.iscene["white"]
        self.ball: RigidObject = env.iscene["ball"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.red_side = torch.ones(n, device=dev)  # +1: red in the +x lane; -1: -x lane
        self.d0 = torch.full((n,), 0.48, device=dev)  # ball-spawn -> red-mouth distance
        self.appr_latch = torch.zeros(n, device=dev)
        self.pass_latch = torch.zeros(n, device=dev)
        self.tilt_latch = torch.zeros(n, device=dev)

    def _scatter(self, m: int, nominal: tuple, scatter: tuple,
                 keepouts: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
        """(m,2) gallery-LOCAL xy around `nominal`, resampled (12 tries, batched)
        until outside every (centre, radius) keep-out."""
        dev = self.env.device
        base = torch.tensor(nominal, device=dev).expand(m, 2)
        jit = torch.tensor(scatter, device=dev)
        xy = base + (torch.rand(m, 2, device=dev) * 2 - 1) * jit
        for _ in range(12):
            bad = torch.zeros(m, dtype=torch.bool, device=dev)
            for ctr, rad in keepouts:
                bad |= (xy - ctr).norm(dim=-1) < rad
            if not bad.any():
                break
            k = int(bad.sum())
            xy[bad] = base[bad] + (torch.rand(k, 2, device=dev) * 2 - 1) * jit
        return xy

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: gallery with xy jitter + yaw (the lanes move — read the pose
        from the scene), RED pin lane coin-flipped, both pins standing with jitter,
        ball + decoy scattered on the open floor in front with keep-out resampling;
        latches zeroed and the approach baseline `d0` captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- gallery: xy jitter + yaw ---
        fix_xy = torch.tensor(c.fix_pos, device=dev).expand(m, 2).clone()
        fix_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)

        def write(body, local_xy: torch.Tensor, z: float, with_yaw: bool) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = fix_xy[:, 0] + local_xy[:, 0] * cy - local_xy[:, 1] * sy
            st[:, 1] = fix_xy[:, 1] + local_xy[:, 0] * sy + local_xy[:, 1] * cy
            st[:, 2] = z
            if with_yaw:
                st[:, 3], st[:, 6] = qw, qz
            else:
                st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.gallery, torch.zeros(m, 2, device=dev), 0.0, with_yaw=True)

        # --- pins: coin-flipped lanes, standing, jittered ---
        if c.shuffle_lanes:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               -torch.ones(m, device=dev), torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        jx = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pin_jx
        jy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pin_jy
        red_local = torch.stack([side * c.lane_off + jx[:, 0], c.pin_y + jy[:, 0]], dim=-1)
        white_local = torch.stack([-side * c.lane_off + jx[:, 1], c.pin_y + jy[:, 1]], dim=-1)
        write(self.red, red_local, c.pin_z0 + 0.002, with_yaw=True)
        write(self.white, white_local, c.pin_z0 + 0.002, with_yaw=True)

        # --- ball + decoy: scattered on the open floor, keep-out resampled ---
        ball_xy = self._scatter(m, c.ball_pos, c.ball_scatter, [])
        dec_xy = self._scatter(m, c.decoy_pos, c.decoy_scatter,
                               [(ball_xy, c.keepout_ball_decoy)])
        write(self.ball, ball_xy, c.ball_r + 0.002, with_yaw=False)
        write(self.decoy, dec_xy, c.decoy_r + 0.002, with_yaw=False)

        # --- baselines + latches ---
        mouth = torch.stack([side * c.lane_off, torch.zeros(m, device=dev)], dim=-1)
        self.red_side[env_ids] = side
        self.d0[env_ids] = (ball_xy - mouth).norm(dim=-1).clamp(min=0.10)
        self.appr_latch[env_ids] = 0.0
        self.pass_latch[env_ids] = 0.0
        self.tilt_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "gallery": self.gallery.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "white": self.white.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "red_side": self.red_side[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "appr_latch": self.appr_latch[env_ids].clone(),
            "pass_latch": self.pass_latch[env_ids].clone(),
            "tilt_latch": self.tilt_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.gallery.write_root_state_to_sim(state["gallery"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.white.write_root_state_to_sim(state["white"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.red_side[env_ids] = state["red_side"]
        self.d0[env_ids] = state["d0"]
        self.appr_latch[env_ids] = state["appr_latch"]
        self.pass_latch[env_ids] = state["pass_latch"]
        self.tilt_latch[env_ids] = state["tilt_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A sealed gray SKITTLE GALLERY stands on the floor: a "
            f"{2 * c.half_w * 1000:.0f} mm-wide, {c.wall_h * 1000:.0f} mm-tall box, "
            f"fully ROOFED over, with side walls, a back wall, and a centre divider that "
            f"splits the inside into two sealed cells (left and right). Nothing can be "
            f"carried over or reached in from above. Two square TUNNELS "
            f"({c.bore_w * 1000:.0f} x {c.bore_h * 1000:.0f} mm) pass through the "
            f"{c.wall_t * 1000:.0f} mm-thick front wall at floor level, one per cell, on "
            f"lane axes {c.lane_off * 1000:.0f} mm left and right of centre; splayed "
            f"funnel walls outside each mouth catch a slightly-off shot, and low guide "
            f"rails extend each tunnel inside. Above each tunnel a letterbox SLIT "
            f"({c.slit_w * 1000:.0f} x {(c.slit_z1 - c.slit_z0) * 1000:.0f} mm) lets you "
            f"look into the cell but pass nothing through. Inside each cell ONE top-heavy "
            f"skittle PIN ({c.pin_sq * 1000:.0f} mm shaft, {c.head_sq * 1000:.0f} mm wide "
            f"head, {c.pin_h * 1000:.0f} mm tall) stands upright on its lane axis, "
            f"{c.pin_y * 1000:.0f} mm behind the outer face — far beyond finger reach "
            f"through the tunnel. Its wide colored head sits exactly at slit height. One "
            f"pin is RED and one is WHITE, and WHICH SIDE IS RED CHANGES per episode: "
            f"look through the slits (or down the tunnels) to identify the red pin's "
            f"lane before acting. On the "
            f"open floor in front lie a BLUE ball ({2 * c.ball_r * 1000:.0f} mm, "
            f"{c.ball_mass * 1000:.0f} g) that fits the tunnels with clearance, and a "
            f"bigger ORANGE ball ({2 * c.decoy_r * 1000:.0f} mm) that is TOO BIG to enter "
            f"any tunnel, alley, or slit — a decoy.\n"
            f"Goal: knock the RED pin over while the WHITE pin stays standing. The pins "
            f"are untouchable by hand or tool, so the only way is to BOWL: aim the blue "
            f"ball at the red pin's tunnel mouth and roll or throw it through with enough "
            f"speed that it crosses the cell and topples the pin. Judged only when "
            f"settled: red pin tilted more than {c.down_deg:.0f} degrees and lying low "
            f"inside its cell, white pin still upright within {c.stand_deg:.0f} degrees "
            f"at standing height. Rolling into the WRONG lane and felling the white pin "
            f"is an unrecoverable failure — the pin cannot be re-stood. The orange ball "
            f"achieves nothing anywhere."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Look through the letterbox slits to find which cell holds the RED pin, then "
            "pick up the blue ball, line it up with that side's floor tunnel, and roll it "
            "through hard enough to knock the red pin over. Keep the white pin standing — "
            "do not roll into the wrong tunnel — and leave the big orange ball alone; it "
            "is too big to fit."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _fix_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> gallery body frame (origin = outer front face centre
        at ground level, +y into the chamber)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.gallery.data.root_quat_w,
                                  p_w - self.gallery.data.root_pos_w)

    def _up_z(self, body) -> torch.Tensor:
        """(N,) world-z component of the body's local +z (1 = upright, 0 = lying)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    def _z_rel(self, body) -> torch.Tensor:
        """(N,) body origin height above the env-origin ground plane."""
        return (body.data.root_pos_w - self.env_origins)[:, 2]

    # ----- predicates -------------------------------------------------------------------------
    def _in_chamber(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the gallery chamber (gallery frame)."""
        c = self.cfg
        loc = self._fix_local(body.data.root_pos_w)
        return ((loc[:, 1] > c.wall_t * 0.5) & (loc[:, 1] < c.chamber_y1)
                & (loc[:, 0].abs() < c.half_w - c.side_t) & (loc[:, 2] < c.wall_h))

    def red_down(self) -> torch.Tensor:
        """(N,) bool: the RED pin is genuinely felled inside the chamber — tilted past
        `down_deg` AND centre below `down_z` (lying, not leaning) AND in the gallery."""
        c = self.cfg
        return ((self._up_z(self.red) < math.cos(math.radians(c.down_deg)))
                & (self._z_rel(self.red) < c.down_z) & self._in_chamber(self.red))

    def white_standing(self) -> torch.Tensor:
        """(N,) bool: the WHITE pin still stands — near-vertical at standing height."""
        c = self.cfg
        return ((self._up_z(self.white) > math.cos(math.radians(c.stand_deg)))
                & ((self._z_rel(self.white) - c.pin_z0).abs() < c.stand_z_tol))

    def ball_in_red_cell(self) -> torch.Tensor:
        """(N,) bool: the blue ball is inside the RED pin's cell (gallery frame)."""
        c = self.cfg
        bl = self._fix_local(self.ball.data.root_pos_w)
        return ((bl[:, 1] > c.wall_t + c.ball_r) & (bl[:, 1] < c.chamber_y1)
                & (bl[:, 0] * self.red_side > 0.02)
                & (bl[:, 0].abs() < c.half_w - c.side_t) & (bl[:, 2] < c.wall_h))

    def settled(self) -> torch.Tensor:
        """(N,) bool: pins settled (lin + ang), ball no longer fast."""
        c = self.cfg
        return ((self.red.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.red.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.white.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.white.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.ball_settle))

    # ----- graded progress --------------------------------------------------------------------
    def approach_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: ball progress toward the RED lane's tunnel mouth, normalized
        by the episode's own spawn distance."""
        c = self.cfg
        bl = self._fix_local(self.ball.data.root_pos_w)
        mouth = torch.stack([self.red_side * c.lane_off,
                             torch.zeros_like(self.red_side)], dim=-1)
        d = (bl[:, :2] - mouth).norm(dim=-1)
        return (1.0 - d / self.d0).clamp(0.0, 1.0)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best ball approach, cell entry, and gated red-pin tilt each physics
        substep, so transient ballistic progress keeps its credit."""
        c = self.cfg
        self.appr_latch = torch.maximum(self.appr_latch, self.approach_frac())
        self.pass_latch = torch.maximum(self.pass_latch, self.ball_in_red_cell().float())
        tilt_now = self._up_z(self.red) < math.cos(math.radians(c.tilt_credit_deg))
        # tilt credit only counts once the ball has actually entered the red cell:
        # forces waved around outside (or a lucky nudge through the wall) earn nothing
        self.tilt_latch = torch.maximum(
            self.tilt_latch, (tilt_now & (self.pass_latch > 0.5)).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red pin felled in its cell + white pin still standing, settled."""
        return self.red_down() & self.white_standing() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched approach + 0.35 * ball entered the red
        cell + 0.20 * red pin tilted past `tilt_credit_deg` (entry-gated), capped at
        0.65; exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy
        (carry the payload over the wall and set it down) is physically impossible here
        and earns nothing."""
        base = (0.10 * self.appr_latch + 0.35 * self.pass_latch
                + 0.20 * self.tilt_latch).clamp(0.0, 0.65)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="skittle_gallery", robot="null"))
