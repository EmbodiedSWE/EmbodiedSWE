"""BananaKilnScene — ram-feed a loose banana through a low tunnel into a roofed kiln
(sim_gen task `track_banana_i240`, scene `banana_kiln`, env `simgen.banana_kiln`).

Derived from pick_place/track_banana, but STRATEGICALLY different: the seed starts
with the banana ALREADY RIGIDLY GRASPED in the closed gripper and grades dense
tracking of a prescribed free-space waypoint path — pure transport of a held object,
no contact event, no tool. Here the goal REGION is unreachable by any held object:
the drying-kiln chamber is fully roofed and walled, and its only opening is a LOW,
DEEP feed tunnel (75 mm tall, 200 mm long) that neither the gripper nor a carried
banana can pass. The banana starts loose on the open floor. The only way to deliver
it is INDIRECT, through a TOOL that is part of the scene: lay the banana into the
open feed channel, then drive the channel's RAMMER — a wide pusher head on a long
handle ending in a graspable knob — so its head bulldozes the banana down the
channel, through the tunnel, and out into the chamber; then pull the rammer back
out of the tunnel. Nothing is ever carried to the goal; the last 400+ mm of the
banana's journey happen under contact (pusher face + floor friction + tunnel walls)
while the hand stays outside at the knob. The plan (place -> ram in -> withdraw) and
the code a solver needs (tool servoing along a constrained channel, working blind
past the mouth) share nothing with the seed's waypoint tracking.

Assets are fully procedural (compound spawners; child colliders of one rigid body
never self-collide):
  - kiln: ONE heavy DYNAMIC compound (30 kg — teleport-safe for per-episode pose
    randomization, no joints anywhere in the task): floor slab, open-top dock
    channel (low side walls), covered tunnel section, and the roofed chamber with
    side walls, far wall, header and roof. Local frame: origin at the TUNNEL MOUTH
    center on the ground, +x = feed direction. The channel floor carries a
    HIGH-friction material (a thrown or flicked banana dies in the tunnel; only a
    sustained push crosses it).
  - rammer: DYNAMIC compound — blue pusher head (170 mm wide, nearly the full
    180 mm channel, 50 mm tall, fits under the 75 mm tunnel roof), a long thin
    handle, and a RED knob at the free end. The knob NEVER enters the tunnel: at
    the deepest useful insertion it still stands 180+ mm outside the mouth.
  - banana: DYNAMIC compound of three yellow angled segments (a bent banana that
    rocks flat instead of rolling) + a brown stem nub. Fits every channel/tunnel
    aperture in any flat orientation (~140 mm span < 180 mm width), and can never
    wedge diagonally (its span is smaller than every passage width; the walls are
    additionally GLAZED slick so a brushing banana slides instead of locking).

Per-episode randomization (readback-verifiable): kiln xy jitter + yaw, banana
staging pose on the open floor (xy + free yaw), rammer start depth jitter.

Rubric (0..1, latched partial credit that never evaporates):
  0.25  banana ever settled INSIDE the open feed channel (kiln frame)
  0.25  banana ever inside the covered tunnel
  0.25  banana ever inside the chamber (past the sill line, on the floor)
  cap 0.75; exactly 1.0 iff success(): banana settled ON THE CHAMBER FLOOR past the
  sill (kiln body frame; a banana perched on the rammer head is too high and does
  not count), the rammer withdrawn so its head is >= 180 mm outside the mouth, and
  banana + rammer at rest. Physical outcomes only.

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
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- compound spawner helpers -----------------------------------------------------------------

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        xf.AddRotateZOp().Set(math.degrees(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _friction_material(stage, root_path: str, name: str, mu_s: float, mu_d: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{root_path}/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_mat(stage, mat, *prim_paths: str) -> None:
    from pxr import UsdShade

    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")


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


def _rigid_dynamic(root, *, mass: float, lin_damp: float, ang_damp: float,
                   pos_iters: int = 16, vel_iters: int = 4, com_z: float = 0.0) -> None:
    """Author RigidBody + explicit MassAPI (CoM at the body ORIGIN unless com_z shifts
    it — spawners apply no cfg schemas, so mass/damping must be authored here)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(com_z)))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(pos_iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_kiln(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the kiln + feed channel as ONE heavy dynamic compound. Local origin at
    the tunnel-mouth center on the GROUND; +x = feed direction (dock at -x, chamber
    at +x). z0 = floor top."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    grey = (0.45, 0.45, 0.50)
    dark = (0.28, 0.28, 0.32)
    brick = (0.62, 0.24, 0.15)
    z0 = c.floor_t
    # floor slab (dock start .. behind the far wall)
    _add_box(stage, f"{prim_path}/floor",
             center=((c.dock_x0 + c.far_x1) / 2, 0.0, c.floor_t / 2),
             size=(c.far_x1 - c.dock_x0, 2 * c.floor_hw, c.floor_t), color=dark,
             collide=collide)
    # dock side walls (low, open top): x dock_x0..0
    for s in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/dock_w{'p' if s > 0 else 'm'}",
                 center=(c.dock_x0 / 2, s * (c.chan_hw + c.wall_t / 2),
                         z0 + c.dock_wall_h / 2),
                 size=(-c.dock_x0, c.wall_t, c.dock_wall_h), color=grey, collide=collide)
    # tunnel side walls: x 0..front_x1, full tunnel height
    for s in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/tun_w{'p' if s > 0 else 'm'}",
                 center=(c.front_x1 / 2, s * (c.chan_hw + c.wall_t / 2),
                         z0 + c.tun_h / 2),
                 size=(c.front_x1, c.wall_t, c.tun_h), color=grey, collide=collide)
    # tunnel roof: underside at z0 + tun_h
    _add_box(stage, f"{prim_path}/tun_roof",
             center=(c.front_x1 / 2, 0.0, z0 + c.tun_h + c.wall_t / 2),
             size=(c.front_x1, 2 * (c.chan_hw + c.wall_t), c.wall_t), color=dark,
             collide=collide)
    # chamber front wall (x front_x0..front_x1): two jambs flush with the tunnel
    # walls + a header from tunnel height up to the chamber roof
    for s in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/jamb_{'p' if s > 0 else 'm'}",
                 center=((c.front_x0 + c.front_x1) / 2,
                         s * (c.chan_hw + (c.cham_hw + c.wall_t - c.chan_hw) / 2),
                         z0 + c.cham_h / 2),
                 size=(c.front_x1 - c.front_x0, c.cham_hw + c.wall_t - c.chan_hw,
                       c.cham_h), color=brick, collide=collide)
    _add_box(stage, f"{prim_path}/header",
             center=((c.front_x0 + c.front_x1) / 2, 0.0,
                     z0 + (c.tun_h + c.cham_h) / 2),
             size=(c.front_x1 - c.front_x0, 2 * c.chan_hw + 0.001, c.cham_h - c.tun_h),
             color=brick, collide=collide)
    # chamber side walls: x front_x1..far_x1
    for s in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/cham_w{'p' if s > 0 else 'm'}",
                 center=((c.front_x1 + c.far_x1) / 2, s * (c.cham_hw + c.wall_t / 2),
                         z0 + c.cham_h / 2),
                 size=(c.far_x1 - c.front_x1, c.wall_t, c.cham_h), color=brick,
                 collide=collide)
    # far wall
    _add_box(stage, f"{prim_path}/far",
             center=((c.far_x0 + c.far_x1) / 2, 0.0, z0 + c.cham_h / 2),
             size=(c.far_x1 - c.far_x0, 2 * (c.cham_hw + c.wall_t), c.cham_h),
             color=brick, collide=collide)
    # chamber roof
    _add_box(stage, f"{prim_path}/cham_roof",
             center=((c.front_x0 + c.far_x1) / 2, 0.0, z0 + c.cham_h + c.wall_t / 2),
             size=(c.far_x1 - c.front_x0, 2 * (c.cham_hw + c.wall_t), c.wall_t),
             color=brick, collide=collide)
    # gritty channel floor: a thrown/flicked banana dies; a sustained push crosses
    mat = _friction_material(stage, prim_path, "gritMat", 0.90, 0.85)
    _bind_mat(stage, mat, f"{prim_path}/floor")
    # glazed walls: the guideway must GUIDE, never friction-lock a brushing banana
    slick = _friction_material(stage, prim_path, "slickMat", 0.04, 0.03)
    _bind_mat(stage, slick,
              f"{prim_path}/dock_wp", f"{prim_path}/dock_wm",
              f"{prim_path}/tun_wp", f"{prim_path}/tun_wm",
              f"{prim_path}/jamb_p", f"{prim_path}/jamb_m", f"{prim_path}/header",
              f"{prim_path}/cham_wp", f"{prim_path}/cham_wm", f"{prim_path}/far")
    return root


def _spawn_rammer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rammer: blue pusher head (origin at HEAD CENTER — CoM there, so the
    long handle overhang never tips it), thin handle along -x, red knob at the end."""
    stage, root = _root_xform(prim_path, translation, orientation)
    # CoM authored LOW (below the banana-contact height): the push reaction then
    # torques nose-DOWN into the floor instead of tipping the tail (and the knob)
    # down onto the slab's back edge.
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.4, ang_damp=1.5,
                   com_z=-0.015)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    blue = (0.15, 0.35, 0.80)
    red = (0.85, 0.12, 0.10)
    _add_box(stage, f"{prim_path}/head", center=(0.0, 0.0, 0.0),
             size=(c.head_l, c.head_w, c.head_h), color=blue, collide=collide)
    # handle spans [-head_l/2 - handle_l, -head_l/2], centered a hair above head mid
    _add_box(stage, f"{prim_path}/handle",
             center=(-c.head_l / 2 - c.handle_l / 2, 0.0, 0.010),
             size=(c.handle_l, 0.016, 0.016), color=blue, collide=collide)
    # knob bottom sits 10 mm above the head's resting-floor top: it clears the slab
    # edge even under a few degrees of transient pitch
    _add_box(stage, f"{prim_path}/knob",
             center=(-c.head_l / 2 - c.handle_l - c.knob_w / 2, 0.0, 0.0225),
             size=(c.knob_w, c.knob_w, 0.075), color=red, collide=collide)
    return root


def _spawn_banana(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the banana: three yellow angled segments (a bent fruit that rocks flat
    instead of rolling) + a brown stem nub at one tip. Root origin at the middle
    segment's center."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.10, ang_damp=0.40)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    yellow = (0.93, 0.80, 0.12)
    b = math.radians(c.bend_deg)
    _add_box(stage, f"{prim_path}/mid", center=(0.0, 0.0, 0.0),
             size=(c.seg_l, c.seg_w, c.seg_h), color=yellow, collide=collide)
    for s, nm in ((1.0, "tip_p"), (-1.0, "tip_m")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(s * c.seg_l * 0.88, 0.0085, 0.0), yaw=s * b,
                 size=(c.seg_l, c.seg_w, c.seg_h), color=yellow, collide=collide)
    _add_box(stage, f"{prim_path}/stem",
             center=(c.seg_l * 1.42, 0.024, 0.004),
             size=(0.012, 0.014, 0.014), color=(0.36, 0.26, 0.10), collide=collide)
    # matte peel: pairs with the gritty channel floor so flicks die fast
    mat = _friction_material(stage, prim_path, "peelMat", 0.80, 0.75)
    _bind_mat(stage, mat, f"{prim_path}/mid", f"{prim_path}/tip_p",
              f"{prim_path}/tip_m", f"{prim_path}/stem")
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "kiln" not in _SPAWNER_CACHE:

        @configclass
        class KilnSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_kiln)
            mass: float = 30.0
            floor_t: float = 0.015
            floor_hw: float = 0.145
            wall_t: float = 0.015
            chan_hw: float = 0.090
            dock_x0: float = -0.40
            dock_wall_h: float = 0.050
            tun_h: float = 0.075
            front_x0: float = 0.200
            front_x1: float = 0.215
            cham_hw: float = 0.120
            cham_h: float = 0.120
            far_x0: float = 0.380
            far_x1: float = 0.395
            contact_offset: float = 0.0015

        @configclass
        class RammerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rammer)
            mass: float = 0.40
            head_l: float = 0.050
            head_w: float = 0.170
            head_h: float = 0.050
            handle_l: float = 0.420
            knob_w: float = 0.022
            contact_offset: float = 0.0015

        @configclass
        class BananaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_banana)
            mass: float = 0.12
            seg_l: float = 0.050
            seg_w: float = 0.032
            seg_h: float = 0.034
            bend_deg: float = 22.0
            contact_offset: float = 0.0012

        _SPAWNER_CACHE.update(kiln=KilnSpawnerCfg, rammer=RammerSpawnerCfg,
                              banana=BananaSpawnerCfg)
    return _SPAWNER_CACHE


def _qz(yaw: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    return q


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class BananaKilnSceneCfg(BaseCfg):
    """Config for `BananaKilnScene`. Geometry honesty is asserted in __post_init__:
    the banana passes every aperture flat in any yaw but can never wedge diagonally;
    the rammer head fills the channel and fits under the tunnel roof; at the deepest
    insertion the pusher face reaches past the sill while the knob is still 200 mm
    outside the mouth; a banana perched on the rammer head reads above the on-floor
    rest gate."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    sill_margin: float = tunable(0.035)   # "inside" = this far past the tunnel's inner end
    rest_z_max: float = tunable(0.060)    # banana root local z below this = resting on the floor
    retract_x: float = tunable(-0.205)    # rammer head center local x below this = withdrawn
    settle_speed: float = tunable(0.06)   # max |lin vel| of banana + rammer when judging (m/s)
    settle_omega: float = tunable(0.60)   # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    kiln_jitter: float = tunable(0.040)   # kiln footprint xy jitter (+/- m)
    kiln_yaw_deg: float = tunable(20.0)   # kiln yaw, uniform (+/- deg)
    ban_jitter: float = tunable(0.060)    # banana staging xy jitter (+/- m)
    ram_jitter: float = tunable(0.020)    # rammer start depth jitter (+/- m)

    # --- info: layout (single Franka base beside the dock; see TASK.md) --------------------------
    kiln_pos: tuple = info((0.35, 0.18))  # tunnel-mouth world xy (before jitter)
    ban_stage: tuple = info((-0.20, -0.28))  # banana staging point, kiln-local (dock side)
    ram_face0: float = info(-0.26)        # rammer pusher-face local x at reset (before jitter)

    # --- info: kiln geometry (kiln-local frame; mirrors the spawner defaults) --------------------
    floor_t: float = info(0.015)          # floor slab thickness; floor TOP = local z 0.015
    wall_t: float = info(0.015)
    chan_hw: float = info(0.090)          # channel/tunnel interior half-width
    dock_x0: float = info(-0.40)          # dock (open channel) start
    dock_wall_h: float = info(0.050)      # dock side-wall height above the floor
    tun_h: float = info(0.075)            # tunnel interior height above the floor
    tun_x1: float = info(0.215)           # tunnel inner end (= chamber interior start)
    cham_hw: float = info(0.120)          # chamber interior half-width
    cham_h: float = info(0.120)           # chamber interior height
    cham_x1: float = info(0.380)          # chamber interior end (far wall inner face)
    kiln_mass: float = info(30.0)

    # --- info: rammer / banana --------------------------------------------------------------------
    ram_mass: float = info(0.40)
    head_l: float = info(0.050)
    head_w: float = info(0.170)
    head_h: float = info(0.050)           # head bottom rests on the floor; top = 0.065 local z
    handle_l: float = info(0.420)
    knob_w: float = info(0.022)           # knob side — well inside the 80 mm Franka jaw
    ban_mass: float = info(0.12)
    seg_l: float = info(0.050)
    seg_w: float = info(0.032)
    seg_h: float = info(0.034)
    ban_span: float = info(0.138)         # worst-case flat footprint span of the banana
    contact_offset: float = info(0.0015)

    # rubric weights (3 x 0.25 = the 0.75 non-success cap)
    w_stage: float = info(0.25)

    # Derived (filled in __post_init__).
    sill_x: float = field(default=None, init=False)      # chamber "inside" line
    head_z: float = field(default=None, init=False)      # rammer root height on the floor
    ban_rest_z: float = field(default=None, init=False)  # banana root height on the floor
    knob_dx: float = field(default=None, init=False)     # knob center offset from rammer root

    def __post_init__(self) -> None:
        self.sill_x = self.tun_x1 + self.sill_margin
        self.head_z = self.floor_t + self.head_h / 2
        self.ban_rest_z = self.floor_t + self.seg_h / 2
        self.knob_dx = -(self.head_l / 2 + self.handle_l + self.knob_w / 2)
        chan_w = 2 * self.chan_hw
        # the banana passes every aperture FLAT in any yaw, and can never wedge
        # diagonally (its span is smaller than every passage width)
        assert self.ban_span < chan_w - 0.015, "banana can wedge in the channel"
        assert self.seg_h < self.tun_h - 0.025, "banana too tall for the tunnel"
        assert self.ban_span < 2 * self.cham_hw - 0.02, "banana can wedge in the chamber"
        assert self.cham_x1 - self.tun_x1 > self.ban_span + 0.02, "chamber too short"
        # the rammer head fills the channel (nothing slips past) yet slides freely
        assert 0.004 < chan_w - self.head_w < 0.014, "head/channel clearance off"
        assert self.head_h < self.tun_h - 0.008, "head does not fit under the tunnel roof"
        assert self.head_h > self.seg_h + 0.010, "pusher face shorter than the banana"
        # full insertion: the pusher face need only reach the SILL — the banana's own
        # broadside half-depth then puts its center past it — while (a) the head's
        # REAR is still inside the tunnel-wall span (yaw-guided, so it can always be
        # pulled straight back out) and (b) the knob is still well outside the mouth
        # (graspable, the hand never enters)
        face_need = self.sill_x - 0.005
        root_need = face_need - self.head_l / 2
        assert root_need - self.head_l / 2 < self.tun_x1 - self.wall_t - 0.004, \
            "head rear leaves the tunnel guidance at full insertion"
        assert root_need + self.knob_dx + self.knob_w / 2 < -0.18, \
            "knob would reach the mouth before the face reaches the sill"
        # crush check: the banana rides BROADSIDE ahead of the face (its depth along
        # the push axis is ~2*seg_w, not the full span); the far wall must leave that
        # plus clearance even at full insertion
        assert face_need < self.far_x0() - 2 * self.seg_w - 0.02, \
            "face insertion would crush the banana against the far wall"
        # rest gate honesty: on-floor banana passes, head-perched banana is rejected
        assert self.ban_rest_z < self.rest_z_max - 0.01, "on-floor banana fails the rest gate"
        assert self.floor_t + self.head_h + self.seg_h / 2 > self.rest_z_max + 0.015, \
            "a banana riding the rammer head would pass the rest gate"
        # reset sanity: the rammer starts already withdrawn; there is room behind it
        assert self.ram_face0 + self.ram_jitter - self.head_l / 2 < self.retract_x - 0.02, \
            "rammer does not start withdrawn"
        assert self.ram_face0 - self.ram_jitter - self.head_l > self.dock_x0 + 0.02, \
            "rammer start collides with the dock end"
        # staging zone clear of the dock wall (banana + jitter + wall never author in contact)
        assert abs(self.ban_stage[1]) - self.ban_jitter \
            > self.chan_hw + self.wall_t + self.ban_span / 2 + 0.02, \
            "banana staging zone overlaps the dock wall"

    def far_x0(self) -> float:
        return self.cham_x1


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("banana_kiln")
class BananaKilnScene(BaseScene):
    cfg: BananaKilnSceneCfg

    def __init__(self, cfg: BananaKilnSceneCfg | None = None) -> None:
        super().__init__(cfg or BananaKilnSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        rigid = sim_utils.RigidBodyPropertiesCfg()
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
            "kiln": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Kiln",
                spawn=sp["kiln"](mass_props=sim_utils.MassPropertiesCfg(mass=c.kiln_mass),
                                 rigid_props=rigid, mass=c.kiln_mass,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.kiln_pos[0], c.kiln_pos[1], 0.0)),
            ),
            "rammer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rammer",
                spawn=sp["rammer"](mass_props=sim_utils.MassPropertiesCfg(mass=c.ram_mass),
                                   rigid_props=rigid, mass=c.ram_mass,
                                   contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.kiln_pos[0] + c.ram_face0 - c.head_l / 2,
                         c.kiln_pos[1], c.head_z)),
            ),
            "banana": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Banana",
                spawn=sp["banana"](mass_props=sim_utils.MassPropertiesCfg(mass=c.ban_mass),
                                   rigid_props=rigid, mass=c.ban_mass,
                                   contact_offset=0.0012),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.kiln_pos[0] + c.ban_stage[0],
                         c.kiln_pos[1] + c.ban_stage[1], c.seg_h / 2 + 0.003)),
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.kiln: RigidObject = env.iscene["kiln"]
        self.rammer: RigidObject = env.iscene["rammer"]
        self.banana: RigidObject = env.iscene["banana"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # per-episode layout (env-local), for readback + reporting
        self._kiln_xy = torch.tensor(self.cfg.kiln_pos, device=dev).repeat(n, 1)
        self._kiln_yaw = torch.zeros(n, device=dev)
        # progress latches (never cleared except at reset)
        self._fed_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tun_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cham_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the kiln (xy jitter + yaw), seat the rammer in the dock
        at its jittered start depth (aligned with the kiln), drop the banana at its
        staging point on the open floor (xy jitter + free yaw); clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        kxy = torch.tensor(c.kiln_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.kiln_jitter
        kyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.kiln_yaw_deg)
        qk = _qz(kyaw)
        cy, sy = torch.cos(kyaw), torch.sin(kyaw)

        def kframe(lx, ly):
            if not torch.is_tensor(lx):
                lx = torch.full((m,), float(lx), device=dev)
            if not torch.is_tensor(ly):
                ly = torch.full((m,), float(ly), device=dev)
            return torch.stack([kxy[:, 0] + cy * lx - sy * ly,
                                kxy[:, 1] + sy * lx + cy * ly], dim=1)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = kxy
        st[:, 3:7] = qk
        st[:, 0:3] += origin
        self.kiln.write_root_state_to_sim(st, env_ids)
        self._kiln_xy[env_ids] = kxy
        self._kiln_yaw[env_ids] = kyaw

        face = c.ram_face0 + (torch.rand(m, device=dev) * 2 - 1) * c.ram_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = kframe(face - c.head_l / 2, 0.0)
        st[:, 2] = c.head_z + 0.002
        st[:, 3:7] = qk
        st[:, 0:3] += origin
        self.rammer.write_root_state_to_sim(st, env_ids)

        bxy = torch.tensor(c.ban_stage, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.ban_jitter
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = kframe(bxy[:, 0], bxy[:, 1])
        st[:, 2] = c.seg_h / 2 + 0.003
        st[:, 3:7] = _qz(kyaw + byaw)
        st[:, 0:3] += origin
        self.banana.write_root_state_to_sim(st, env_ids)

        self._fed_ever[env_ids] = False
        self._tun_ever[env_ids] = False
        self._cham_ever[env_ids] = False

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "kiln": self.kiln.data.root_state_w[env_ids].clone(),
            "rammer": self.rammer.data.root_state_w[env_ids].clone(),
            "banana": self.banana.data.root_state_w[env_ids].clone(),
            "kiln_xy": self._kiln_xy[env_ids].clone(),
            "kiln_yaw": self._kiln_yaw[env_ids].clone(),
            "fed_ever": self._fed_ever[env_ids].clone(),
            "tun_ever": self._tun_ever[env_ids].clone(),
            "cham_ever": self._cham_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.kiln.write_root_state_to_sim(state["kiln"], env_ids)
        self.rammer.write_root_state_to_sim(state["rammer"], env_ids)
        self.banana.write_root_state_to_sim(state["banana"], env_ids)
        self._kiln_xy[env_ids] = state["kiln_xy"]
        self._kiln_yaw[env_ids] = state["kiln_yaw"]
        self._fed_ever[env_ids] = state["fed_ever"]
        self._tun_ever[env_ids] = state["tun_ever"]
        self._cham_ever[env_ids] = state["cham_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A drying KILN stands on the floor near ({c.kiln_pos[0]:.2f}, "
            f"{c.kiln_pos[1]:.2f}) (its position and heading change per episode): a "
            f"BRICK-RED fully roofed chamber ({2 * c.cham_hw * 100:.0f} cm wide, "
            f"{(c.cham_x1 - c.tun_x1) * 100:.0f} cm deep, {c.cham_h * 100:.0f} cm tall "
            f"inside) whose ONLY opening is a low feed TUNNEL "
            f"({2 * c.chan_hw * 100:.0f} cm wide, {c.tun_h * 100:.1f} cm tall, "
            f"{c.tun_x1 * 100:.0f} cm deep) — far too low and deep for a hand or a "
            f"carried object to pass. The tunnel continues outward as an OPEN-TOP grey "
            f"feed channel about {-c.dock_x0 * 100:.0f} cm long with "
            f"{c.dock_wall_h * 100:.0f} cm side walls. Riding in the channel is a "
            f"RAMMER: a BLUE pusher head ({c.head_w * 100:.0f} cm wide — it nearly "
            f"fills the channel) on a long thin handle that ends in a RED KNOB "
            f"({c.knob_w * 1000:.0f} mm square, 7.5 cm tall) hanging out past the "
            f"channel's open end; the knob always stays outside the tunnel. A single "
            f"YELLOW banana (about {c.ban_span * 100:.0f} cm long) lies loose on the "
            f"open floor beside the channel; its exact spot and heading change per "
            f"episode.\n"
            f"Goal: get the banana to REST ON THE KILN CHAMBER FLOOR, fully inside — "
            f"at least {c.sill_margin * 100:.0f} cm past the tunnel's inner end — and "
            f"then withdraw the rammer so its blue head sits at least "
            f"{-c.retract_x * 100 - c.head_l / 2 * 100:.0f} cm outside the tunnel "
            f"mouth, with everything at rest. The intended method: lay the banana flat "
            f"into the open channel in front of the pusher head, then grasp the red "
            f"knob and drive the rammer down the channel so its head bulldozes the "
            f"banana through the tunnel and out into the chamber, then pull the rammer "
            f"back out by the same knob. A banana left in the channel or tunnel, or "
            f"one perched on the rammer head, does not count; the gritty channel floor "
            f"stops thrown or flicked bananas well short of the chamber."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the yellow banana into the open feed channel in front of the blue "
            "pusher head, then drive the rammer by its red knob so the head pushes "
            "the banana through the low tunnel until it rests on the kiln chamber "
            "floor inside, and pull the rammer back until its head is well outside "
            "the tunnel mouth. The banana must end up resting fully inside the "
            "chamber, not in the tunnel and not on the rammer."
        )

    # ----- kiln-local frame -----------------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) kiln-local coordinates of world points (live kiln pose)."""
        from isaaclab.utils.math import quat_rotate_inverse

        return quat_rotate_inverse(self.kiln.data.root_quat_w,
                                   pos_w - self.kiln.data.root_pos_w)

    def banana_local(self) -> torch.Tensor:
        return self._local(self.banana.data.root_pos_w)

    def rammer_local(self) -> torch.Tensor:
        return self._local(self.rammer.data.root_pos_w)

    # ----- predicates ------------------------------------------------------------------------------
    def in_channel(self) -> torch.Tensor:
        """(N,) banana resting inside the OPEN feed channel (ahead of the dock start,
        short of the mouth), on the channel floor."""
        c = self.cfg
        p = self.banana_local()
        return ((p[:, 0] > c.dock_x0 + 0.02) & (p[:, 0] < -0.01)
                & (p[:, 1].abs() < c.chan_hw - 0.005)
                & (p[:, 2] > 0.015) & (p[:, 2] < 0.07))

    def in_tunnel(self) -> torch.Tensor:
        """(N,) banana inside the covered tunnel section."""
        c = self.cfg
        p = self.banana_local()
        return ((p[:, 0] > 0.03) & (p[:, 0] < c.tun_x1)
                & (p[:, 1].abs() < c.chan_hw - 0.005)
                & (p[:, 2] > 0.015) & (p[:, 2] < 0.09))

    def inside_chamber(self) -> torch.Tensor:
        """(N,) banana ON THE CHAMBER FLOOR past the sill line (kiln body frame). The
        z gate rejects a banana perched on the rammer head (root would read ~0.082)."""
        c = self.cfg
        p = self.banana_local()
        return ((p[:, 0] > c.sill_x) & (p[:, 0] < c.cham_x1 - 0.005)
                & (p[:, 1].abs() < c.cham_hw - 0.005)
                & (p[:, 2] > 0.020) & (p[:, 2] < c.rest_z_max))

    def retracted(self) -> torch.Tensor:
        """(N,) rammer head withdrawn clear of the tunnel (head center behind the
        retract line — its start pose satisfies this, so retraction alone earns
        nothing)."""
        return self.rammer_local()[:, 0] < self.cfg.retract_x

    def _still(self) -> torch.Tensor:
        c = self.cfg
        return ((self.banana.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.banana.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)
                & (self.rammer.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.rammer.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _update_latches(self) -> None:
        slowish = self.banana.data.root_lin_vel_w.norm(dim=-1) < 0.15
        self._fed_ever |= self.in_channel() & slowish
        self._tun_ever |= self.in_tunnel()
        self._cham_ever |= self.inside_chamber()

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: banana settled ON the chamber floor past the sill, rammer
        withdrawn clear of the tunnel, both at rest. Physical outcomes only."""
        self._update_latches()
        return self.inside_chamber() & self.retracted() & self._still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 per stage ever reached (banana settled in the
        feed channel / banana in the tunnel / banana inside the chamber) — latched,
        ~0 for doing nothing, capped 0.75 — exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_stage * self._fed_ever.float()
                + c.w_stage * self._tun_ever.float()
                + c.w_stage * self._cham_ever.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="banana_kiln", robot="null"))
