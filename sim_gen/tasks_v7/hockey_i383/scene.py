"""PolarityDockScene — orient both batteries tip-toward-terminal, seat them in the
two-bay dock, then slide the lid shut; a reversed battery cannot lie flat and the
lid physically will not close over it. Derived from rlbench/hockey, but there is no
ball, no tool and no propulsion anywhere in the task.

Seed (rlbench/hockey): grasp a hockey stick and STRIKE the ball across open floor
into an open-mouthed goal — one ballistic tool swing at a passively available
receptacle; success is a bbox containment check. Every element of that plan is
removed or inverted here:

- Nothing is propelled. The two payloads (green BATTERY cylinders) must be picked
  up, REORIENTED end-for-end as needed, and LAID into open-top bays. A battery
  slid or shot along the floor at the dock just bounces off its outer wall (the
  smoke battery fires that exact probe).
- The receptacle is not passive — it is a mechanical VERIFIER. Each bay accepts a
  battery lying flat only in ONE axial orientation: the battery's silver metal TIP
  (a 12 mm nub) must point at the bay's silver SPLIT TERMINAL (two silver posts
  with an 18 mm slot between them at one end of the trough). Correct: the tip
  slides into the slot and the battery lies flat, its top 1 mm BELOW the deck.
  Reversed: body 70 + tip 12 = 82 mm will not fit the 76 mm trough — the battery
  can only rest TILTED (>= 22 deg), standing ~30 mm PROUD of the deck. Which end
  of each bay carries the terminal is randomized per bay per episode (the bay
  cassettes are teleported with a Bernoulli 180 deg yaw), so the required
  orientation must be read from the scene, not memorized.
- The final act is a CLOSURE, not a goal crossing: an amber LID (D6 prismatic
  slide, travel 108 mm, underside 4 mm above the deck) must be slid fully shut
  over both bays by its red handle. A proud (reversed / unseated) battery stands
  >= 10 mm into the lid plane and JAMS the slide — the lid can only close over a
  correctly polarized, fully seated pair. Success is judged on the settled
  physical outcome: both batteries seated flat below deck, lid at its closed
  stop, everything still.

So a solver needs a different PLAN (perceive per-bay terminal side -> reorient
each battery -> place-from-above insertion -> slide the lid shut) and different
CODE STRUCTURE (orientation readback + two seat placements + prismatic-slide
control instead of grasp-stick + swing). Nothing is ever struck.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - bay0 / bay1: KINEMATIC single-bay cassettes (floor slab, plain end wall, two
    side walls, and the silver terminal pillar pair flanking the slot). Origin at
    the footprint centre on the ground, trough along local +x, terminal at local
    +x. Teleported each reset: ensemble pose + Bernoulli 180 deg yaw per bay.
  - frame: heavy DYNAMIC rail block (40 kg — never kinematic: a kinematic body0
    would leave the lid's joint anchor world-fixed after the reset teleport).
    Origin on the ground at the compartment centre; the visible block sits on the
    +y side under the lid's open parking span.
  - lid: DYNAMIC plate + red handle bar, joined to the frame by a spawn-authored
    D6 UsdPhysics.Joint with only transY free (limits [0, stroke]); q = 0 is
    CLOSED, q = stroke is fully open. The joint carries the lid 4 mm above the
    deck, so the closed lid clears a seated battery by 4 mm and is jammed by any
    proud one.
  - battery0 / battery1: DYNAMIC compounds — green body cylinder (dia 26 mm,
    70 mm) + silver tip cylinder (dia 12 mm, 12 mm) on the local +z end.

Per-episode randomization (readback-verifiable): apparatus planar offset + yaw
(one coherent write: frame + lid + both cassettes), per-bay Bernoulli terminal
side, battery floor poses (slot jitter + free yaw, so the tip points anywhere).

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.28 * seat0   — battery ever seated flat in a bay (cassette-frame readback)
  0.28 * seat1   — the other battery ever seated flat in a bay
  0.14 * closed  — lid ever at its closed stop WHILE both batteries sit seated
  1.0 iff success() — both batteries seated flat below deck AND the lid at its
                      closed stop, everything settled. Non-success cap 0.70.

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    """Author one axis-aligned box collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable) -> None:
    """Author one z-axis cylinder collider."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateHeightAttr(float(height))
    cyl.CreateRadiusAttr(float(radius))
    r, hh = float(radius), float(height) / 2
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -hh), Gf.Vec3f(r, r, hh)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _dynamic_body(root, *, mass: float, lin_damp: float, ang_damp: float,
                  com=None, inertia=None) -> None:
    """Author a dynamic rigid body: explicit mass (root-level mass_props on custom
    spawner cfgs is silently ignored), optional explicit CoM + diagonal inertia
    (compound roots otherwise keep the CoM at the body origin), damping, zeroed
    sleep thresholds, solver velocity iterations 4 (GPU shape-creep fix)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    if com is not None:
        massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_bay(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one bay cassette: KINEMATIC compound. Origin at the footprint centre
    on the ground; trough along local +x; the silver split TERMINAL (pillar pair
    flanking an 18 mm slot) sits at local +x, the plain end wall at local -x. A
    180 deg yaw teleport at reset flips which world end carries the terminal."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    hx, hy = c.half_x, c.half_y            # 0.050, 0.024
    deck = c.floor_t + c.trough_d          # 0.037
    wz, wc = c.trough_d, c.floor_t + c.trough_d / 2  # wall height / wall centre z
    # floor slab under the whole footprint
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(2 * hx, 2 * hy, c.floor_t), color=c.color, collide=collide)
    # plain end wall (local -x)
    _add_box(stage, f"{prim_path}/end_wall",
             center=(-(hx - c.end_t / 2), 0.0, c.floor_t + wz / 2),
             size=(c.end_t, 2 * hy, wz), color=c.color, collide=collide)
    # side walls along the trough + pillar span
    side_l = 2 * hx - c.end_t
    for sgn, nm in ((1.0, "side_l"), (-1.0, "side_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(c.end_t / 2, sgn * (hy - c.wall_t / 2), c.floor_t + wz / 2),
                 size=(side_l, c.wall_t, wz), color=c.color, collide=collide)
    # the silver split terminal: two pillars flanking the slot (local +x). Each
    # pillar spans from the slot edge (y = +/- gap_w/2) to the cassette outer
    # face, so the slot is exactly gap_w wide.
    pil_w = hy - c.gap_w / 2               # slot edge -> cassette outer face
    for sgn, nm in ((1.0, "term_l"), (-1.0, "term_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(hx - c.pillar_t / 2,
                         sgn * (c.gap_w / 2 + pil_w / 2),
                         c.floor_t + wz / 2),
                 size=(c.pillar_t, pil_w, wz), color=c.term_color,
                 collide=collide)
    del deck, wc
    return root


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the lid rail frame: heavy DYNAMIC block (body0 of the lid's D6 —
    a kinematic body0 would leave the joint anchor world-fixed after the reset
    teleport). Origin on the ground at the compartment centre; the visible block
    sits on the +y side, top BELOW the lid plane."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, mass=cfg.mass, lin_damp=0.5, ang_damp=0.5,
                  com=(0.0, cfg.block_y, cfg.block_h / 2))
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/block",
             center=(0.0, cfg.block_y, cfg.block_h / 2),
             size=(0.140, 0.050, cfg.block_h), color=cfg.color, collide=collide)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sliding lid: DYNAMIC plate + red handle bar, plus the D6
    UsdPhysics.Joint to the sibling Frame — only transY free, limits
    [closed 0 .. stroke open]. The joint (not any rail geometry) carries the lid
    at a fixed height, underside `lid_clear` above the deck: it cannot ride up
    and over a proud battery, only jam against it."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, mass=cfg.mass, lin_damp=1.2, ang_damp=0.5,
                  com=(0.0, 0.0, 0.004),
                  inertia=(4.0e-4, 4.0e-4, 7.0e-4))
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(cfg.plate_x, cfg.plate_y, cfg.plate_t), color=cfg.color,
             collide=collide)
    _add_box(stage, f"{prim_path}/handle",
             center=(0.0, cfg.handle_y, cfg.plate_t / 2 + cfg.handle_h / 2),
             size=(0.090, cfg.handle_w, cfg.handle_h), color=cfg.handle_color,
             collide=collide)
    # --- the slide: generic D6 joint to the sibling Frame, authored at spawn ---
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(cfg.ride_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    for axis in ("transX", "transZ", "rotX", "rotY", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
        lim.CreateLowAttr(1.0)      # low > high == locked
        lim.CreateHighAttr(-1.0)
    lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transY")
    lim.CreateLowAttr(0.0)          # q = 0 : CLOSED
    lim.CreateHighAttr(float(cfg.stroke))
    return root


def _spawn_battery(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one battery: DYNAMIC compound — green body cylinder + silver TIP
    cylinder on the local +z end (the polarity feature)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, mass=cfg.mass, lin_damp=0.05, ang_damp=0.30,
                  com=(0.0, 0.0, 0.003))
    collide = _make_collide(cfg)
    _add_cyl(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             radius=cfg.body_r, height=cfg.body_l, color=cfg.color,
             collide=collide)
    _add_cyl(stage, f"{prim_path}/tip",
             center=(0.0, 0.0, cfg.body_l / 2 + cfg.tip_l / 2),
             radius=cfg.tip_r, height=cfg.tip_l, color=cfg.tip_color,
             collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bay" not in _SPAWNER_CACHE:

        @configclass
        class BaySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bay)
            half_x: float = 0.050
            half_y: float = 0.024
            floor_t: float = 0.010
            trough_d: float = 0.027
            end_t: float = 0.010
            wall_t: float = 0.008
            pillar_t: float = 0.014
            gap_w: float = 0.018
            color: tuple = (0.28, 0.30, 0.36)
            term_color: tuple = (0.82, 0.84, 0.88)
            contact_offset: float = 0.0015

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            mass: float = 40.0
            block_y: float = 0.087
            block_h: float = 0.036
            color: tuple = (0.20, 0.21, 0.24)
            contact_offset: float = 0.0015

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            mass: float = 0.30
            plate_x: float = 0.108
            plate_y: float = 0.104
            plate_t: float = 0.006
            handle_y: float = 0.042
            handle_w: float = 0.014
            handle_h: float = 0.030
            ride_z: float = 0.043
            stroke: float = 0.108
            color: tuple = (0.92, 0.68, 0.12)
            handle_color: tuple = (0.80, 0.12, 0.10)
            contact_offset: float = 0.0015

        @configclass
        class BatterySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_battery)
            mass: float = 0.08
            body_r: float = 0.013
            body_l: float = 0.070
            tip_r: float = 0.006
            tip_l: float = 0.012
            color: tuple = (0.10, 0.55, 0.20)
            tip_color: tuple = (0.85, 0.86, 0.88)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(bay=BaySpawnerCfg, frame=FrameSpawnerCfg,
                              lid=LidSpawnerCfg, battery=BatterySpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PolarityDockSceneCfg(BaseCfg):
    """Config for `PolarityDockScene`. The interlocks are metric: a correctly
    polarized battery (body 70 + tip 12) seats flat because the tip enters the
    18 mm terminal slot, leaving its top 1 mm below the 37 mm deck and 4 mm below
    the lid plane; a reversed battery is 82 mm against a 76 mm trough, can only
    rest tilted >= 22 deg, stands >= 10 mm into the lid plane, and jams the slide.
    The lid rides a D6 prismatic joint (z locked) so it cannot climb over a proud
    battery."""

    # --- tunable: rubric thresholds ---------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max battery |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.75)      # max battery |ang vel| when judging (rad/s)
    lid_settle: float = tunable(0.03)      # max lid |lin vel| when judging (m/s)
    closed_tol: float = tunable(0.004)     # lid q below this == closed (m)
    seat_x_tol: float = tunable(0.009)     # battery centre |x - seat_x| in the bay frame
    seat_y_tol: float = tunable(0.006)     # battery centre |y| in the bay frame
    seat_z_lo: float = tunable(0.017)      # battery centre z band in the bay frame:
    seat_z_hi: float = tunable(0.028)      # flat-on-floor is 0.023; proud rests are > 0.030
    seat_axis_min: float = tunable(0.90)   # tip axis dot bay +x (toward the terminal)
    seat_flat_max: float = tunable(0.25)   # max |tip axis dot world up| (must lie flat)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    ens_dx_max: float = tunable(0.03)      # apparatus offset along x (+/- m)
    ens_dy_max: float = tunable(0.03)      # apparatus offset along y (+/- m)
    ens_yaw_max_deg: float = tunable(10.0)  # apparatus yaw (+/- deg)
    bat_slot_x: float = tunable(0.080)     # battery spawn slots at +/- this x
    bat_jitter: float = tunable(0.020)     # +/- xy jitter about the slot
    bat_y_center: float = tunable(-0.200)  # battery spawn zone y centre
    lid_open_min: float = tunable(0.096)   # lid q at reset, uniform in
    lid_open_max: float = tunable(0.106)   # [open_min, open_max]

    # --- info: bay cassette structure ---------------------------------------------------------
    bay_half_x: float = info(0.050)        # cassette footprint 100 x 48 mm
    bay_half_y: float = info(0.024)
    bay_floor_t: float = info(0.010)       # trough floor top = 0.010
    trough_d: float = info(0.027)          # deck plane = 0.037
    bay_end_t: float = info(0.010)         # plain end wall thickness
    bay_wall_t: float = info(0.008)        # side wall thickness (trough width 32)
    pillar_t: float = info(0.014)          # terminal pillar depth (trough length 76)
    gap_w: float = info(0.018)             # terminal slot width (tip dia 12)
    bay_dy: float = info(0.0245)           # cassette centres at frame-local +/- this
    seat_x: float = info(-0.002)           # seated battery centre x in the bay frame

    # --- info: frame + lid --------------------------------------------------------------------
    frame_mass: float = info(40.0)
    frame_block_y: float = info(0.087)
    frame_block_h: float = info(0.036)     # block top 1 mm below the deck
    lid_mass: float = info(0.30)
    lid_plate_x: float = info(0.108)
    lid_plate_y: float = info(0.104)
    lid_plate_t: float = info(0.006)
    lid_ride_z: float = info(0.043)        # lid centre height: underside 4 mm above deck
    lid_stroke: float = info(0.108)        # D6 transY limits [0, stroke]; 0 = closed
    lid_handle_y: float = info(0.042)      # red handle bar, on top, lid-local +y

    # --- info: battery ------------------------------------------------------------------------
    bat_mass: float = info(0.08)
    body_r: float = info(0.013)            # dia 26 mm (parallel-jaw sized)
    body_l: float = info(0.070)
    tip_r: float = info(0.006)             # dia 12 mm silver tip
    tip_l: float = info(0.012)             # body+tip 82 mm vs 76 mm trough: the key

    # --- info: colors + misc ------------------------------------------------------------------
    bay_color: tuple = info((0.28, 0.30, 0.36))
    term_color: tuple = info((0.82, 0.84, 0.88))
    frame_color: tuple = info((0.20, 0.21, 0.24))
    lid_color: tuple = info((0.92, 0.68, 0.12))
    handle_color: tuple = info((0.80, 0.12, 0.10))
    bat_color: tuple = info((0.10, 0.55, 0.20))
    tip_color: tuple = info((0.85, 0.86, 0.88))
    contact_offset: float = info(0.0015)
    # rubric weights (0.28 + 0.28 + 0.14 = 0.70 = the non-success cap)
    w_seat: float = info(0.28)
    w_lid: float = info(0.14)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("polarity_dock")
class PolarityDockScene(BaseScene):
    cfg: PolarityDockSceneCfg

    def __init__(self, cfg: PolarityDockSceneCfg | None = None) -> None:
        super().__init__(cfg or PolarityDockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        bay_spawn = sp["bay"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            half_x=c.bay_half_x, half_y=c.bay_half_y, floor_t=c.bay_floor_t,
            trough_d=c.trough_d, end_t=c.bay_end_t, wall_t=c.bay_wall_t,
            pillar_t=c.pillar_t, gap_w=c.gap_w, color=c.bay_color,
            term_color=c.term_color, contact_offset=c.contact_offset,
        )
        frame_spawn = sp["frame"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass=c.frame_mass, block_y=c.frame_block_y, block_h=c.frame_block_h,
            color=c.frame_color, contact_offset=c.contact_offset,
        )
        lid_spawn = sp["lid"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass=c.lid_mass, plate_x=c.lid_plate_x, plate_y=c.lid_plate_y,
            plate_t=c.lid_plate_t, handle_y=c.lid_handle_y, ride_z=c.lid_ride_z,
            stroke=c.lid_stroke, color=c.lid_color, handle_color=c.handle_color,
            contact_offset=c.contact_offset,
        )
        bat_spawn = sp["battery"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass=c.bat_mass, body_r=c.body_r, body_l=c.body_l, tip_r=c.tip_r,
            tip_l=c.tip_l, color=c.bat_color, tip_color=c.tip_color,
            contact_offset=c.contact_offset,
        )
        # NOTE: Frame is declared BEFORE Lid so the lid's spawn-authored D6 can
        # target the already-existing sibling /Frame prim. Template init poses
        # are joint-consistent (lid at q = 0.10 within [0, stroke]).
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
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=frame_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.10, c.lid_ride_z)),
            ),
            "bay0": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bay0",
                spawn=bay_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, c.bay_dy, 0.0)),
            ),
            "bay1": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bay1",
                spawn=bay_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -c.bay_dy, 0.0)),
            ),
            "battery0": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Battery0",
                spawn=bat_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.bat_slot_x, c.bat_y_center, c.body_r + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
            ),
            "battery1": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Battery1",
                spawn=bat_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bat_slot_x, c.bat_y_center, c.body_r + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.frame: RigidObject = env.iscene["frame"]
        self.lid: RigidObject = env.iscene["lid"]
        self.bays: list[RigidObject] = [env.iscene["bay0"], env.iscene["bay1"]]
        self.bats: list[RigidObject] = [env.iscene["battery0"], env.iscene["battery1"]]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._seat_ever = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._lid_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the apparatus (frame + lid + both bay cassettes) re-posed
        with a random planar offset + yaw — written as ONE coherent linkage (lid at
        a random open q in the new frame; body0 is dynamic so the D6 anchor
        follows). Per bay, a Bernoulli 180 deg yaw flips which end carries the
        silver terminal. Batteries to their floor slots with jitter + free yaw."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, device=dev)  # burn the first post-seed draw (degenerate)
        dx = (torch.rand(m, device=dev) * 2 - 1) * c.ens_dx_max
        dy = (torch.rand(m, device=dev) * 2 - 1) * c.ens_dy_max
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ens_yaw_max_deg)
        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0], q_yaw[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        ens = torch.zeros(m, 3, device=dev)
        ens[:, 0], ens[:, 1] = dx, dy

        # frame (dynamic body0): origin at the compartment centre on the ground
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = ens + origin
        st[:, 3:7] = q_yaw
        self.frame.write_root_state_to_sim(st, env_ids)

        # lid: at a random OPEN q along the slide, in the new frame
        q_open = (c.lid_open_min
                  + torch.rand(m, device=dev) * (c.lid_open_max - c.lid_open_min))
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 1], loc[:, 2] = q_open, c.lid_ride_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = ens + quat_apply(q_yaw, loc) + origin
        st[:, 3:7] = q_yaw
        self.lid.write_root_state_to_sim(st, env_ids)

        # bay cassettes (kinematic): frame-local +/- bay_dy, Bernoulli 180 deg yaw
        q_pi = torch.zeros(m, 4, device=dev)
        q_pi[:, 3] = 1.0  # yaw pi
        for i, bay in enumerate(self.bays):
            flip = torch.rand(m, device=dev) < 0.5
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 1] = c.bay_dy if i == 0 else -c.bay_dy
            q_id = torch.zeros(m, 4, device=dev)
            q_id[:, 0] = 1.0
            q_bay = torch.where(flip.unsqueeze(1), quat_mul(q_yaw, q_pi), q_yaw)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = ens + quat_apply(q_yaw, loc) + origin
            st[:, 3:7] = q_bay
            bay.write_root_state_to_sim(st, env_ids)

        # batteries: floor slots + jitter, lying flat, free yaw (tip points anywhere)
        for i, bat in enumerate(self.bats):
            slot_x = -c.bat_slot_x if i == 0 else c.bat_slot_x
            bx = slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.bat_jitter
            by = c.bat_y_center + (torch.rand(m, device=dev) * 2 - 1) * c.bat_jitter
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            half = byaw / 2
            c45 = math.cos(math.pi / 4)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = bx, by, c.body_r + 0.002
            # lying flat: q = qz(yaw) * qy(90 deg) — local +z (tip) -> horizontal
            st[:, 3] = torch.cos(half) * c45
            st[:, 4] = -torch.sin(half) * c45
            st[:, 5] = torch.cos(half) * c45
            st[:, 6] = torch.sin(half) * c45
            st[:, 0:3] += origin
            bat.write_root_state_to_sim(st, env_ids)

        self._seat_ever[env_ids] = False
        self._lid_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "bays": [b.data.root_state_w[env_ids].clone() for b in self.bays],
            "bats": [b.data.root_state_w[env_ids].clone() for b in self.bats],
            "seat_ever": self._seat_ever[env_ids].clone(),
            "lid_ever": self._lid_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        for b, s in zip(self.bays, state["bays"]):
            b.write_root_state_to_sim(s, env_ids)
        for b, s in zip(self.bats, state["bats"]):
            b.write_root_state_to_sim(s, env_ids)
        self._seat_ever[env_ids] = state["seat_ever"]
        self._lid_ever[env_ids] = state["lid_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor sits an open two-bay BATTERY DOCK: two slate-gray "
            f"cassettes side by side, each with a rectangular trough "
            f"({(2 * c.bay_half_x - c.bay_end_t - c.pillar_t) * 1000:.0f} mm long, "
            f"{(2 * (c.bay_half_y - c.bay_wall_t)) * 1000:.0f} mm wide, "
            f"{c.trough_d * 1000:.0f} mm deep) open at the top. ONE end of each "
            f"trough carries a SILVER SPLIT TERMINAL — two bright silver posts "
            f"with a {c.gap_w * 1000:.0f} mm vertical slot between them; the other "
            f"end is a plain gray wall. Which end is the terminal end differs per "
            f"bay and per episode: look at the silver posts. Behind the dock "
            f"stands a dark rail block carrying an amber LID (a flat plate with a "
            f"red handle bar on top) that slides horizontally across the dock; it "
            f"starts parked fully open and its slide is the only way it moves "
            f"(it rides {(c.lid_ride_z - c.lid_plate_t / 2 - c.bay_floor_t - c.trough_d) * 1000:.0f} mm "
            f"above the deck and cannot lift). In front of the dock lie two GREEN "
            f"BATTERIES (cylinders, {2 * c.body_r * 1000:.0f} mm across, "
            f"{c.body_l * 1000:.0f} mm long), each with a small SILVER METAL TIP "
            f"({2 * c.tip_r * 1000:.0f} mm across, {c.tip_l * 1000:.0f} mm long) "
            f"on one end and a plain flat end on the other; their positions and "
            f"headings change between episodes.\n"
            f"Goal: seat BOTH batteries in the dock — one per bay, lying flat on "
            f"the trough floor with the SILVER TIP pointing INTO the silver split "
            f"terminal's slot — then slide the amber lid fully closed over both "
            f"bays by its red handle. Polarity is mechanical: with the tip toward "
            f"the terminal the tip enters the slot and the battery drops fully "
            f"below the deck; REVERSED (tip toward the plain wall) the battery is "
            f"too long for the trough, rests tilted and proud of the deck, and "
            f"the lid physically jams against it. A battery left outside, "
            f"standing upright, laid across the deck, or reversed — and a lid "
            f"not slid fully home — all mean failure. Both bays must be filled; "
            f"the lid must end fully closed with everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay each green battery flat in its own bay of the dock with the "
            "silver tip pointing into that bay's silver split terminal, then "
            "slide the amber lid fully closed by its red handle. A reversed "
            "battery cannot lie flat and will jam the lid — that is failure."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def lid_q(self) -> torch.Tensor:
        """(N,) lid slide coordinate in the FRAME's body frame: 0 = closed, stroke
        = fully open (position readback, never velocity)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.lid.data.root_pos_w - self.frame.data.root_pos_w
        return quat_apply_inverse(self.frame.data.root_quat_w, rel)[:, 1]

    def _bat_in_bay(self, bat: RigidObject, bay: RigidObject) -> torch.Tensor:
        """(N,) bool: battery seated FLAT in this bay, tip toward the terminal —
        centre inside the seat window (bay frame), z in the flat-on-floor band
        (proud/tilted rests sit higher), axis lying flat and pointing at the
        terminal (bay-local +x)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        loc = quat_apply_inverse(bay.data.root_quat_w,
                                 bat.data.root_pos_w - bay.data.root_pos_w)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        axis_w = quat_apply(bat.data.root_quat_w, ez)      # tip direction, world
        axis_b = quat_apply_inverse(bay.data.root_quat_w, axis_w)
        return ((loc[:, 0] - c.seat_x).abs() < c.seat_x_tol) \
            & (loc[:, 1].abs() < c.seat_y_tol) \
            & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi) \
            & (axis_w[:, 2].abs() < c.seat_flat_max) \
            & (axis_b[:, 0] > c.seat_axis_min)

    def seated(self) -> torch.Tensor:
        """(N, 2) bool per battery: seated flat (tip-to-terminal) in EITHER bay."""
        out = []
        for bat in self.bats:
            s = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            for bay in self.bays:
                s = s | self._bat_in_bay(bat, bay)
            out.append(s)
        return torch.stack(out, dim=1)

    def bats_settled(self) -> torch.Tensor:
        """(N, 2) bool: battery lin + ang velocities below the settle gates."""
        c = self.cfg
        out = []
        for bat in self.bats:
            out.append((bat.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                       & (bat.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))
        return torch.stack(out, dim=1)

    def lid_closed(self) -> torch.Tensor:
        """(N,) bool: lid at its closed stop (q < closed_tol) and still."""
        c = self.cfg
        return (self.lid_q() < c.closed_tol) \
            & (self.lid.data.root_lin_vel_w.norm(dim=-1) < c.lid_settle)

    def _update_latches(self) -> None:
        seat_now = self.seated() & self.bats_settled()
        self._seat_ever |= seat_now
        # lid credit only counts when the closure closes over a seated pair
        self._lid_ever |= self.lid_closed() & self.seated().all(dim=1)

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant here (the slide is passive: D6 limits + damping) — just latch
        rubric progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: both batteries seated flat below deck (tip-to-terminal) AND
        the lid at its closed stop — everything settled. Physical outcomes only:
        a reversed battery never seats, and a lid jammed on a proud battery never
        reaches the stop."""
        self._update_latches()
        return (self.seated() & self.bats_settled()).all(dim=1) & self.lid_closed()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.28 per battery ever seated + 0.14 for the lid
        ever closed over a seated pair — all latched, ~0 for doing nothing,
        capped 0.70 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._seat_ever[:, 0].float()
                + c.w_seat * self._seat_ever[:, 1].float()
                + c.w_lid * self._lid_ever.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="polarity_dock", robot="null"))
