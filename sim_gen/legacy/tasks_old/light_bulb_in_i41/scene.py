"""BulbTriageScene — hidden-state bulb triage: test, then route (sim_gen task
`light_bulb_in_i41`, derived from rlbench/light_bulb_in).

The seed (RLBench `light_bulb_in`) is a transport-and-insert task: pick the light bulb
off its stand-holder and screw it into the lamp — one object, one known goal pose, the
whole plan is a single carry with a precision insertion at the end. This task keeps the
seed's object vocabulary (bulbs, a lamp with an empty socket) but changes the PLAN
CLASS: the goal-relevant property of each bulb is HIDDEN STATE that must be discovered
by a physical test before the placement is even defined.

  - THREE visually identical frosted-globe bulbs lie scattered on the floor (their stems
    are color-coded red / green / blue so they can be referred to; the globes look the
    same). Exactly ONE bulb works — sampled uniformly per episode and NOT observable
    from any pose, color, or geometry at reset.
  - A TESTER SOCKET (dark pedestal with a cyan collar) reveals the hidden state: seat a
    bulb stem-down in the collar and hold it seated + settled for a real dwell
    (`reveal_hold_substeps` consecutive physics substeps) and the globe recolors —
    amber GLOW if it is the working bulb, smoke-gray if it is dead. The reveal latches
    (`revealed[i]`) and is permanent for the episode. A bulb teleported through the
    seat pose without the dwell reveals nothing (tested in the smoke).
  - The LAMP (disc base, bronze column, gold collar socket) accepts only a VERIFIED
    bulb: full install credit and success require `revealed[working]` — the QC
    protocol: a bulb must have been seen to glow on the tester before it is installed.
  - Every DEAD bulb must be routed to the DISPOSAL BIN (an open walled box). Dead bulbs
    do NOT have to be tested: once the working bulb has glowed, the remaining bulbs are
    dead by inference and may be binned untested — so the required plan LENGTH depends
    on the sampled hidden state (1-3 tests), and the routing of every bulb depends on
    what the tester showed.

Goal: the verified working bulb seated stem-down, upright and settled in the lamp
socket, AND both dead bulbs resting inside the disposal bin, all settled.

Why a solver needs a different plan than the seed's: the seed's carry-and-insert cannot
even be STARTED correctly — which bulb belongs in the lamp is unknowable without the
tester detour, and installing an untested bulb (even the right one, by luck) is capped
at 0.05 and can never succeed. The episode is information-gathering -> conditional
routing, not transport.

Judged on PHYSICAL outcomes plus the latched reveal history:
  - `seated_on` (tester / lamp): stem-bottom within `seat_xy_tol` of the socket axis,
    at recess-floor height (`seat_z_tol`), bulb upright within `upright_max_deg`
    (22 deg — chosen ABOVE the ~15 deg maximum physical lean of a stem captive in the
    collar, so anything genuinely seated counts; a bulb lying across the collar mouth
    is at ~90 deg and never counts).
  - `binned`: bulb centre inside the bin interior below rim height (bounds chosen so
    ANY resting pose on the bin floor counts and a bulb stacked on another does not).
  - success(): `revealed[working]` AND working bulb seated+settled in the lamp AND both
    dead bulbs binned+settled.
  - score(): 0 for doing nothing; +`w_reveal` (0.08) per latched reveal, +`w_diag`
    (0.16) once the WORKING bulb has been revealed glowing, +`w_bin_each` (0.15) per
    dead bulb currently in the bin, +`w_install` (0.30) for the working bulb seated in
    the lamp IF verified (else `w_install_untested` = 0.05, the untested-luck cap);
    capped at 0.95 unless success; exactly 1.0 iff success.

Per-episode randomization: WHICH bulb works (3), spawn-slot permutation of the three
bulbs (6), xy jitter + free yaw for every bulb, xy jitter for all three fixtures — the
test outcomes, and therefore the correct routing plan, change every episode.

Assets are fully procedural: three dynamic compound bulbs (sphere globe + colored
cylinder stem), a kinematic tester pedestal, a kinematic lamp, a kinematic open bin.
No external files. Heavy imports (isaaclab, pxr) are deferred so importing this module
— and registering the scene — stays app-free.
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
# One rigid body per object, child colliders authored with raw pxr APIs;
# `isaaclab.sim.utils.clone` supplies the regex-resolve + per-env replication (each cloned prim
# is authored fresh, so the duplicate-xformOp trap does not arise). Kinematic fixtures are
# re-posed per reset by pose writes (single bodies, no joints — the proven-safe pattern).

_SPAWNER_CACHE: dict[str, Any] = {}


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _author_root(stage, prim_path: str, translation, orientation, kinematic: bool):
    from pxr import Gf, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    return root


def _author_collar(stage, prim_path: str, inner_r: float, wall_t: float, wall_h: float,
                   color, contact_offset: float) -> None:
    """8 box wall segments forming an octagonal collar of inner inradius `inner_r`, rising
    from local z=0 to z=wall_h (the pen_holder wall layout)."""
    from pxr import Gf, UsdGeom

    n = 8
    r_mid = inner_r + wall_t / 2
    seg_len = 2 * (inner_r + wall_t) * math.tan(math.pi / n) + 0.002
    gf_color = Gf.Vec3f(*color)
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang),
                                          wall_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(wall_t, seg_len, wall_h))
        seg.CreateDisplayColorAttr([gf_color])
        _collide(seg.GetPrim(), contact_offset)


def _spawn_bulb(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One dynamic bulb. Body frame: globe sphere centred at the origin, colored stem
    cylinder along local -z spanning z in [-stem_bot, -stem_bot + stem_l]. Upright
    (stem-down) = identity orientation. The globe prim is named `globe` so the scene can
    rewrite its displayColor on reveal."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation, kinematic=False)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    # Depenetration cap (the pen_holder end-on impact lesson) + damping so the top-heavy
    # bulb stops rocking in the collar promptly and crosses the settle gates.
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.15)

    globe = UsdGeom.Sphere.Define(stage, f"{prim_path}/globe")
    globe.CreateRadiusAttr(cfg.globe_r)
    globe.CreateExtentAttr([Gf.Vec3f(-cfg.globe_r, -cfg.globe_r, -cfg.globe_r),
                            Gf.Vec3f(cfg.globe_r, cfg.globe_r, cfg.globe_r)])
    globe.CreateDisplayColorAttr([Gf.Vec3f(*cfg.globe_color)])
    _collide(globe.GetPrim(), cfg.contact_offset)

    stem = UsdGeom.Cylinder.Define(stage, f"{prim_path}/stem")
    stem.CreateRadiusAttr(cfg.stem_r)
    stem.CreateHeightAttr(cfg.stem_l)
    stem.CreateExtentAttr([Gf.Vec3f(-cfg.stem_r, -cfg.stem_r, -cfg.stem_l / 2),
                           Gf.Vec3f(cfg.stem_r, cfg.stem_r, cfg.stem_l / 2)])
    UsdGeom.Xformable(stem.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -(cfg.stem_bot - cfg.stem_l / 2)))
    stem.CreateDisplayColorAttr([Gf.Vec3f(*cfg.stem_color)])
    _collide(stem.GetPrim(), cfg.contact_offset)
    return root


def _spawn_tester(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic tester pedestal. Body origin = recess-floor centre (pedestal top face);
    the base box hangs below, the collar rises above."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation, kinematic=True)

    base = UsdGeom.Cube.Define(stage, f"{prim_path}/base")
    base.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(base.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.base_h / 2))
    bxf.AddScaleOp().Set(Gf.Vec3f(cfg.base_xy, cfg.base_xy, cfg.base_h))
    base.CreateDisplayColorAttr([Gf.Vec3f(*cfg.base_color)])
    _collide(base.GetPrim(), cfg.contact_offset)

    _author_collar(stage, prim_path, cfg.inner_r, cfg.wall_t, cfg.wall_h,
                   cfg.ring_color, cfg.contact_offset)
    return root


def _spawn_lamp(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic lamp. Body origin = socket-floor centre (column top face); base disc and
    column hang below, the gold collar rises above."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation, kinematic=True)

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/base")
    disc.CreateRadiusAttr(cfg.base_r)
    disc.CreateHeightAttr(cfg.base_h)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.base_r, -cfg.base_r, -cfg.base_h / 2),
                           Gf.Vec3f(cfg.base_r, cfg.base_r, cfg.base_h / 2)])
    UsdGeom.Xformable(disc.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -(cfg.col_h + cfg.base_h / 2)))
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.base_color)])
    _collide(disc.GetPrim(), cfg.contact_offset)

    col = UsdGeom.Cylinder.Define(stage, f"{prim_path}/column")
    col.CreateRadiusAttr(cfg.col_r)
    col.CreateHeightAttr(cfg.col_h)
    col.CreateExtentAttr([Gf.Vec3f(-cfg.col_r, -cfg.col_r, -cfg.col_h / 2),
                          Gf.Vec3f(cfg.col_r, cfg.col_r, cfg.col_h / 2)])
    UsdGeom.Xformable(col.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.col_h / 2))
    col.CreateDisplayColorAttr([Gf.Vec3f(*cfg.col_color)])
    _collide(col.GetPrim(), cfg.contact_offset)

    _author_collar(stage, prim_path, cfg.inner_r, cfg.wall_t, cfg.wall_h,
                   cfg.ring_color, cfg.contact_offset)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic open disposal bin. Body origin = inner-floor centre (floor top face);
    the floor slab hangs below, four walls rise above."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation, kinematic=True)
    outer = cfg.inner + 2 * cfg.wall_t
    gf_color = Gf.Vec3f(*cfg.color)

    floor = UsdGeom.Cube.Define(stage, f"{prim_path}/floor")
    floor.CreateSizeAttr(1.0)
    fxf = UsdGeom.Xformable(floor.GetPrim())
    fxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.floor_t / 2))
    fxf.AddScaleOp().Set(Gf.Vec3f(outer, outer, cfg.floor_t))
    floor.CreateDisplayColorAttr([gf_color])
    _collide(floor.GetPrim(), cfg.contact_offset)

    for k, (dx, dy, sx, sy) in enumerate((
            (cfg.inner / 2 + cfg.wall_t / 2, 0.0, cfg.wall_t, outer),
            (-(cfg.inner / 2 + cfg.wall_t / 2), 0.0, cfg.wall_t, outer),
            (0.0, cfg.inner / 2 + cfg.wall_t / 2, outer, cfg.wall_t),
            (0.0, -(cfg.inner / 2 + cfg.wall_t / 2), outer, cfg.wall_t))):
        wall = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        wall.CreateSizeAttr(1.0)
        wxf = UsdGeom.Xformable(wall.GetPrim())
        wxf.AddTranslateOp().Set(Gf.Vec3d(dx, dy, cfg.wall_h / 2))
        wxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, cfg.wall_h))
        wall.CreateDisplayColorAttr([gf_color])
        _collide(wall.GetPrim(), cfg.contact_offset)
    return root


def _spawner_cfgs() -> dict[str, Any]:
    """Build (lazily, app required) the four spawner cfg classes — `clone` wraps each
    author function exactly like `spawn_cuboid` is wrapped (the pen_holder pattern)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if not _SPAWNER_CACHE:

        @configclass
        class BulbSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bulb)
            globe_r: float = 0.027
            stem_r: float = 0.011
            stem_l: float = 0.045
            stem_bot: float = 0.065
            globe_color: tuple = (0.9, 0.9, 0.9)
            stem_color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.001

        @configclass
        class TesterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tester)
            base_xy: float = 0.09
            base_h: float = 0.04
            inner_r: float = 0.015
            wall_t: float = 0.008
            wall_h: float = 0.03
            base_color: tuple = (0.1, 0.1, 0.1)
            ring_color: tuple = (0.1, 0.1, 0.1)
            contact_offset: float = 0.001

        @configclass
        class LampSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lamp)
            base_r: float = 0.075
            base_h: float = 0.025
            col_r: float = 0.02
            col_h: float = 0.06
            inner_r: float = 0.015
            wall_t: float = 0.008
            wall_h: float = 0.03
            base_color: tuple = (0.1, 0.1, 0.1)
            col_color: tuple = (0.1, 0.1, 0.1)
            ring_color: tuple = (0.1, 0.1, 0.1)
            contact_offset: float = 0.001

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            inner: float = 0.19
            wall_t: float = 0.008
            wall_h: float = 0.07
            floor_t: float = 0.006
            color: tuple = (0.1, 0.1, 0.1)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(bulb=BulbSpawnerCfg, tester=TesterSpawnerCfg,
                              lamp=LampSpawnerCfg, bin=BinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BulbTriageSceneCfg(BaseCfg):
    """Config for `BulbTriageScene`. Env-local frame: everything on the ground plane;
    tester and lamp on the +x side, bulbs scattered on the -x side, bin far left."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_xy_tol: float = tunable(0.008)  # stem bottom within this of the socket axis. Honest
    # by construction: collar inner inradius 15 mm - stem r 11 mm = 4 mm max centred offset
    # (+ ~3 mm from the max captive lean) — anything genuinely seated counts; a bulb resting
    # anywhere outside the collar is >= 26 mm away.
    seat_z_tol: float = tunable(0.008)   # stem bottom within this of the recess floor
    upright_max_deg: float = tunable(22.0)  # bulb +z within this of world-up. Max physical
    # captive lean = atan(2*(inner_r - stem_r)/wall_h) ~ 15 deg < 22; lying across the
    # collar mouth is ~75-90 deg and never counts.
    settle_lin: float = tunable(0.05)    # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.80)    # max |ang vel| when judging settled (rad/s)
    reveal_hold_substeps: int = tunable(24)  # consecutive seated+settled substeps (0.2 s)
    # before the tester reveals — a teleport-through never latches (tested).
    bin_margin: float = tunable(0.025)   # xy inset from the bin's inner wall for `binned`;
    # inner half 95 mm - globe r 27 mm = 68 mm max physical centre offset < the 70 mm gate,
    # so any bulb resting on the bin floor counts.
    bin_z_max: float = tunable(0.055)    # max bulb-centre height above the bin floor: a bulb
    # resting on the floor sits at 27-33 mm; one stacked on another at ~80 mm — rejected.

    # --- tunable: score weights --------------------------------------------------------------
    w_reveal: float = tunable(0.08)          # per bulb ever revealed on the tester (latched)
    w_diag: float = tunable(0.16)            # extra: the WORKING bulb has been seen glowing
    w_bin_each: float = tunable(0.15)        # per dead bulb currently resting in the bin
    w_install: float = tunable(0.30)         # working bulb seated in the lamp, VERIFIED
    w_install_untested: float = tunable(0.05)  # ... unverified (the untested-luck cap)
    cap_non_success: float = tunable(0.95)   # hard ceiling unless success

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fixture_jitter: float = tunable(0.03)  # uniform +/- xy jitter of tester / lamp / bin
    bulb_jitter: float = tunable(0.03)     # uniform +/- xy jitter of each bulb spawn slot
    bulb_yaw_deg: float = tunable(180.0)   # uniform +/- yaw of each lying bulb

    # --- info: structure ---------------------------------------------------------------------
    globe_r: float = info(0.027)     # frosted globe radius (54 mm bulb head)
    stem_r: float = info(0.011)      # stem radius (passes the thin-cylinder pinch audit)
    stem_l: float = info(0.045)      # stem collider length
    stem_bot: float = info(0.065)    # stem bottom depth below the globe centre (body -z)
    bulb_mass: float = info(0.04)
    socket_inner_r: float = info(0.015)  # collar aperture inradius (both sockets): 4 mm
    # radial clearance around the 11 mm stem — the drop funnel measured in the smoke sweep
    socket_wall_t: float = info(0.008)
    socket_wall_h: float = info(0.030)   # collar depth; seated globe bottom clears it by 8 mm
    tester_base_xy: float = info(0.090)
    tester_floor_z: float = info(0.040)  # tester recess floor height (= pedestal top)
    lamp_base_r: float = info(0.075)
    lamp_base_h: float = info(0.025)
    lamp_col_r: float = info(0.020)
    lamp_floor_z: float = info(0.085)    # lamp socket floor height (= base_h + col_h)
    bin_inner: float = info(0.190)       # bin interior (square side); holds both dead bulbs
    bin_wall_t: float = info(0.008)
    bin_wall_h: float = info(0.070)
    bin_floor_t: float = info(0.006)
    tester_pos: tuple = info((0.32, -0.22))
    lamp_pos: tuple = info((0.32, 0.22))
    bin_pos: tuple = info((-0.38, 0.28))
    slots: tuple = info(((-0.10, -0.30), (-0.24, -0.01), (-0.10, 0.28)))
    stem_colors: tuple = info(((0.85, 0.15, 0.15), (0.15, 0.62, 0.20), (0.20, 0.35, 0.85)))
    globe_color: tuple = info((0.92, 0.90, 0.84))  # frosted — identical on all three bulbs
    glow_color: tuple = info((1.00, 0.78, 0.10))   # revealed WORKING
    dead_color: tuple = info((0.30, 0.30, 0.33))   # revealed dead
    tester_base_color: tuple = info((0.15, 0.17, 0.22))
    tester_ring_color: tuple = info((0.10, 0.62, 0.68))
    lamp_base_color: tuple = info((0.35, 0.28, 0.20))
    lamp_ring_color: tuple = info((0.85, 0.66, 0.15))
    bin_color: tuple = info((0.42, 0.40, 0.38))
    contact_offset: float = info(0.001)  # small: combined 2 mm < the 4 mm radial funnel
    n_bulbs: int = info(3)

    # Derived (filled in __post_init__).
    tester_seat_z: float = field(default=None, init=False)  # bulb-root z when seated
    lamp_seat_z: float = field(default=None, init=False)
    bin_floor_z: float = field(default=None, init=False)
    lamp_col_h: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.tester_seat_z = round(self.tester_floor_z + self.stem_bot, 4)
        self.lamp_seat_z = round(self.lamp_floor_z + self.stem_bot, 4)
        self.bin_floor_z = self.bin_floor_t
        self.lamp_col_h = round(self.lamp_floor_z - self.lamp_base_h, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bulb_triage")
class BulbTriageScene(BaseScene):
    cfg: BulbTriageSceneCfg

    def __init__(self, cfg: BulbTriageSceneCfg | None = None) -> None:
        super().__init__(cfg or BulbTriageSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the three kinematic fixtures, three dynamic bulbs lying flat at
        their nominal slots (reset() re-poses and re-randomizes everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        c45 = math.cos(math.pi / 4)
        sp = _spawner_cfgs()
        tester_cfg, lamp_cfg, bin_cfg, bulb_cfg = sp["tester"], sp["lamp"], sp["bin"], sp["bulb"]

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
            "tester": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tester",
                spawn=tester_cfg(
                    base_xy=c.tester_base_xy, base_h=c.tester_floor_z,
                    inner_r=c.socket_inner_r, wall_t=c.socket_wall_t,
                    wall_h=c.socket_wall_h, base_color=c.tester_base_color,
                    ring_color=c.tester_ring_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tester_pos[0], c.tester_pos[1], c.tester_floor_z)),
            ),
            "lamp": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lamp",
                spawn=lamp_cfg(
                    base_r=c.lamp_base_r, base_h=c.lamp_base_h, col_r=c.lamp_col_r,
                    col_h=c.lamp_col_h, inner_r=c.socket_inner_r, wall_t=c.socket_wall_t,
                    wall_h=c.socket_wall_h, base_color=c.lamp_base_color,
                    col_color=c.lamp_base_color, ring_color=c.lamp_ring_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lamp_pos[0], c.lamp_pos[1], c.lamp_floor_z)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=bin_cfg(
                    inner=c.bin_inner, wall_t=c.bin_wall_t, wall_h=c.bin_wall_h,
                    floor_t=c.bin_floor_t, color=c.bin_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bin_pos[0], c.bin_pos[1], c.bin_floor_z)),
            ),
        }
        for i in range(c.n_bulbs):
            out[f"bulb_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bulb_" + str(i),
                spawn=bulb_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bulb_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    globe_r=c.globe_r, stem_r=c.stem_r, stem_l=c.stem_l,
                    stem_bot=c.stem_bot, globe_color=c.globe_color,
                    stem_color=c.stem_colors[i], contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slots[i][0], c.slots[i][1], c.globe_r + 0.003),
                    rot=(c45, 0.0, c45, 0.0)),  # lying flat
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the hidden-state / latch buffers."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.tester: RigidObject = env.iscene["tester"]
        self.lamp: RigidObject = env.iscene["lamp"]
        self.bin: RigidObject = env.iscene["bin"]
        self.bulbs: list[RigidObject] = [env.iscene[f"bulb_{i}"] for i in range(c.n_bulbs)]
        self.env_origins = env.iscene.env_origins
        self.tester_c = torch.zeros(n, 2, device=dev)   # fixture centres (env-local xy)
        self.lamp_c = torch.zeros(n, 2, device=dev)
        self.bin_c = torch.zeros(n, 2, device=dev)
        self.working = torch.zeros(n, dtype=torch.long, device=dev)   # HIDDEN: which works
        self.slot_perm = torch.zeros(n, c.n_bulbs, dtype=torch.long, device=dev)
        self.revealed = torch.zeros(n, c.n_bulbs, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, c.n_bulbs, dtype=torch.long, device=dev)

    def _set_globe_color(self, e: int, i: int, rgb: tuple) -> None:
        """Rewrite one cloned bulb's globe displayColor (the per-env recolor pattern)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Bulb_{i}/globe")
        if prim and prim.IsValid():
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter the three fixtures (kinematic pose writes), sample the
        hidden working index + spawn-slot permutation, lay the bulbs flat with jitter +
        free yaw, clear the reveal latches and restore the frosted globes."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- fixtures: nominal + xy jitter (z fixed), identity orientation ---
        for body, nominal, store, z in (
                (self.tester, c.tester_pos, self.tester_c, c.tester_floor_z),
                (self.lamp, c.lamp_pos, self.lamp_c, c.lamp_floor_z),
                (self.bin, c.bin_pos, self.bin_c, c.bin_floor_z)):
            ctr = torch.tensor(nominal, device=dev).expand(m, 2) \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.fixture_jitter
            store[env_ids] = ctr
            st = torch.zeros(m, 7, device=dev)
            st[:, 0:2] = ctr
            st[:, 2] = z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_pose_to_sim(st, env_ids)

        # --- hidden state + spawn permutation ---
        self.working[env_ids] = torch.randint(0, c.n_bulbs, (m,), device=dev)
        self.slot_perm[env_ids] = torch.rand(m, c.n_bulbs, device=dev).argsort(dim=1)

        # --- bulbs: lying flat at their permuted slot + jitter + free yaw ---
        slots = torch.tensor(c.slots, device=dev)  # (3, 2)
        yaw_amp = math.radians(c.bulb_yaw_deg)
        c45 = math.cos(math.pi / 4)
        for i in range(c.n_bulbs):
            slot_xy = slots[self.slot_perm[env_ids, i]]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slot_xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.bulb_jitter
            st[:, 2] = c.globe_r + 0.003
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            # lying flat: q = qz(yaw) * qy(90 deg) (the pen_holder flat-spawn quaternion)
            st[:, 3] = torch.cos(half) * c45
            st[:, 4] = -torch.sin(half) * c45
            st[:, 5] = torch.cos(half) * c45
            st[:, 6] = torch.sin(half) * c45
            st[:, 0:3] += origin
            self.bulbs[i].write_root_state_to_sim(st, env_ids)

        self.revealed[env_ids] = False
        self._hold[env_ids] = 0
        for e in env_ids.tolist():
            for i in range(c.n_bulbs):
                self._set_globe_color(int(e), i, c.globe_color)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reveal bookkeeping every physics substep (buffers fresh): a bulb seated in the
        TESTER collar and settled accumulates dwell; `reveal_hold_substeps` consecutive
        substeps latch `revealed[i]` and recolor the globe (amber glow if it is the
        working bulb, smoke-gray otherwise). Latches are permanent for the episode."""
        c = self.cfg
        seated = self.seated_on("tester") & self.bulb_settled()
        self._hold = torch.where(seated, self._hold + 1, torch.zeros_like(self._hold))
        newly = (self._hold >= c.reveal_hold_substeps) & ~self.revealed
        if bool(newly.any()):
            for e, i in newly.nonzero(as_tuple=False).tolist():
                glow = int(self.working[e]) == i
                self._set_globe_color(e, i, c.glow_color if glow else c.dead_color)
            self.revealed |= newly

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bulbs": [b.data.root_state_w[env_ids].clone() for b in self.bulbs],
            "fixtures": [b.data.root_state_w[env_ids].clone()
                         for b in (self.tester, self.lamp, self.bin)],
            "tester_c": self.tester_c[env_ids].clone(),
            "lamp_c": self.lamp_c[env_ids].clone(),
            "bin_c": self.bin_c[env_ids].clone(),
            "working": self.working[env_ids].clone(),
            "slot_perm": self.slot_perm[env_ids].clone(),
            "revealed": self.revealed[env_ids].clone(),
            "hold": self._hold[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.bulbs, state["bulbs"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip((self.tester, self.lamp, self.bin), state["fixtures"]):
            b.write_root_pose_to_sim(st[:, 0:7], env_ids)
        self.tester_c[env_ids] = state["tester_c"]
        self.lamp_c[env_ids] = state["lamp_c"]
        self.bin_c[env_ids] = state["bin_c"]
        self.working[env_ids] = state["working"]
        self.slot_perm[env_ids] = state["slot_perm"]
        self.revealed[env_ids] = state["revealed"]
        self._hold[env_ids] = state["hold"]
        c = self.cfg
        for k, e in enumerate(env_ids.tolist()):
            for i in range(c.n_bulbs):
                if bool(state["revealed"][k, i]):
                    glow = int(state["working"][k]) == i
                    self._set_globe_color(int(e), i,
                                          c.glow_color if glow else c.dead_color)
                else:
                    self._set_globe_color(int(e), i, c.globe_color)

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Three frosted light bulbs (identical {2 * c.globe_r * 1000:.0f} mm globes; "
            f"stems color-coded red, green and blue) lie scattered on the floor. Exactly "
            f"ONE of them works — chosen at random each episode and impossible to tell by "
            f"looking. On the right stand a TESTER (dark pedestal with a cyan collar "
            f"socket) and a LAMP (bronze column with a gold collar socket, "
            f"{c.lamp_floor_z * 100:.0f} cm up); on the left sits an open disposal bin.\n"
            f"Goal: find the working bulb and install it. Seat a bulb stem-down in the "
            f"tester collar and hold it there a moment: its globe glows amber if it works, "
            f"turns smoke-gray if it is dead. The lamp only accepts a VERIFIED bulb — one "
            f"that has been seen to glow on the tester; installing an untested bulb never "
            f"succeeds, even if it happens to be the good one. Finish with the verified "
            f"working bulb seated upright in the lamp socket and BOTH dead bulbs dropped "
            f"in the disposal bin (once the good bulb has glowed, the rest are dead by "
            f"inference and may be binned untested). The tester holds one bulb at a time."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _bulb_pos_local(self) -> torch.Tensor:
        """(N, 3bulbs, 3) globe centres in env-local coords."""
        return torch.stack(
            [b.data.root_pos_w - self.env_origins for b in self.bulbs], dim=1)

    def _bottom_local(self) -> torch.Tensor:
        """(N, 3bulbs, 3) stem-bottom points in env-local coords."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        down = torch.tensor([0.0, 0.0, -self.cfg.stem_bot], device=self.env.device)
        cols = []
        for b in self.bulbs:
            cols.append(b.data.root_pos_w + quat_apply(b.data.root_quat_w,
                                                       down.expand(n, 3))
                        - self.env_origins)
        return torch.stack(cols, dim=1)

    def upright(self) -> torch.Tensor:
        """(N, 3bulbs) bool: bulb +z (globe up, stem down) within `upright_max_deg` of
        world-up. Signed — a globe-down bulb never counts."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        cos_max = math.cos(math.radians(self.cfg.upright_max_deg))
        cols = []
        for b in self.bulbs:
            axis = quat_apply(b.data.root_quat_w, ez)
            cols.append(axis[:, 2] >= cos_max)
        return torch.stack(cols, dim=1)

    def bulb_settled(self) -> torch.Tensor:
        """(N, 3bulbs) bool: per-bulb lin + ang velocity below the settle gates."""
        c = self.cfg
        cols = []
        for b in self.bulbs:
            lin = b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
            ang = b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
            cols.append(lin & ang)
        return torch.stack(cols, dim=1)

    def seated_on(self, which: str) -> torch.Tensor:
        """(N, 3bulbs) bool, geometric: stem bottom within `seat_xy_tol` of the socket
        axis of `which` ('tester' | 'lamp'), at recess-floor height, bulb upright."""
        c = self.cfg
        ctr = self.tester_c if which == "tester" else self.lamp_c
        floor = c.tester_floor_z if which == "tester" else c.lamp_floor_z
        bot = self._bottom_local()
        near = (bot[:, :, :2] - ctr.unsqueeze(1)).norm(dim=-1) <= c.seat_xy_tol
        z_ok = (bot[:, :, 2] - floor).abs() <= c.seat_z_tol
        return near & z_ok & self.upright()

    def binned(self) -> torch.Tensor:
        """(N, 3bulbs) bool, geometric: globe centre inside the bin interior, below the
        stacked-bulb rejection height."""
        c = self.cfg
        pos = self._bulb_pos_local()
        half = c.bin_inner / 2 - c.bin_margin
        d = pos[:, :, :2] - self.bin_c.unsqueeze(1)
        inside = (d[:, :, 0].abs() <= half) & (d[:, :, 1].abs() <= half)
        z_ok = (pos[:, :, 2] > 0.0) & (pos[:, :, 2] <= c.bin_floor_z + c.bin_z_max)
        return inside & z_ok

    def dead_mask(self) -> torch.Tensor:
        """(N, 3bulbs) bool: bulbs that are NOT the working one."""
        idx = torch.arange(self.cfg.n_bulbs, device=self.env.device)
        return idx.unsqueeze(0) != self.working.unsqueeze(1)

    def verified(self) -> torch.Tensor:
        """(N,) bool: the working bulb has been revealed glowing on the tester."""
        return self.revealed.gather(1, self.working.view(-1, 1)).squeeze(1)

    def installed_working(self) -> torch.Tensor:
        """(N,) bool: the working bulb seated + settled in the LAMP socket (geometric —
        verification is judged separately)."""
        ok = self.seated_on("lamp") & self.bulb_settled()
        return ok.gather(1, self.working.view(-1, 1)).squeeze(1)

    def dead_binned(self) -> torch.Tensor:
        """(N,) int: dead bulbs currently resting settled in the bin."""
        return (self.binned() & self.bulb_settled() & self.dead_mask()).sum(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: verified working bulb installed in the lamp AND both dead bulbs in
        the disposal bin, all settled."""
        return self.verified() & self.installed_working() \
            & (self.dead_binned() == self.cfg.n_bulbs - 1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: reveal credits (latched) + diagnosis bonus + per-dead-bulb
        disposal + verified install (unverified install capped at `w_install_untested`);
        capped at `cap_non_success`; exactly 1.0 iff success. Doing nothing scores 0."""
        c = self.cfg
        ver = self.verified()
        s = c.w_reveal * self.revealed.sum(dim=1).float() \
            + c.w_diag * ver.float() \
            + c.w_bin_each * self.dead_binned().float() \
            + torch.where(ver, c.w_install, c.w_install_untested) \
            * self.installed_working().float()
        s = s.clamp(max=c.cap_non_success)
        return torch.where(self.success(), torch.ones_like(s), s)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="bulb_triage", robot="null", env_spacing=4.0))
