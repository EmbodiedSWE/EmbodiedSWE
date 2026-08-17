"""DecantReturnScene — pour the marble out of the narrow-necked bottle into the tray,
then stand the empty bottle back in its coaster socket (sim_gen task
`living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i417`).

Derived from libero_90 living_room_scene4 "pick up the salad dressing and put it in the
tray", but STRATEGICALLY different: the seed's whole plan is prehensile TRANSPORT OF THE
CONTAINER — grasp the dressing bottle, carry it, release it inside the tray's open
containment box. Here that exact end state is a FAILURE: the deliverable is the bottle's
CONTENT, not the bottle.

  - the payload (a red marble) starts SEALED inside the amber bottle: the bottle's only
    opening is a narrow mouth (40 mm square) above a funnel shoulder, and the marble
    (30 mm) rests 100+ mm below it — no parallel jaw can reach or pinch it (asserted in
    cfg.__post_init__). The only physical route to extract it is to reorient the WHOLE
    bottle far past horizontal so the marble rolls down the funnel and out of the mouth;
  - the marble must land INSIDE the walled tray — the pour must be AIMED from a held,
    tilted bottle hovering over the tray (the tray walls refuse a marble spilled on the
    ground: it cannot climb 55 mm);
  - the bottle itself must NOT end in the tray: success requires it standing upright,
    seated inside the raised coaster SOCKET across the rig — a drop-in registration with
    a 25 mm lip. Putting the bottle in the tray (the seed's goal, verbatim) fails.

A solver therefore needs a different PLAN (grasp the container, hover it over the tray,
roll it past ~120 deg while keeping the mouth anchored over the target, wait out the
discharge, re-erect it and seat it in the socket) and a different CODE STRUCTURE (a
6-DOF held-pose controller with a tilt trajectory and a landing check — not a
grasp-carry-release routine).

success(): marble settled INSIDE the tray (and outside the bottle) AND the bottle
settled UPRIGHT and SEATED in the socket AND the marble was OBSERVED leaving the bottle
interior continuously (the decant latch: an inside->outside transition with bounded
per-step displacement — a marble teleported out earns no latch). score() is latched
credit that never evaporates: 0.10 bottle ever lifted + 0.10 bottle ever hovered over
the tray + 0.20 decant observed + 0.15 marble ever in the tray + 0.15 bottle seated
after the decant (cap 0.70); exactly 1.0 iff success() live. Null policy scores 0.

Fully procedural geometry (boxes + one sphere; no external assets). Per-episode
randomization (readback-verified in smoke): rig yaw +/-30 deg + xy jitter (tray and
socket positions must be READ from the scene), bottle start xy + free yaw, marble
in-cavity jitter. Heavy imports (isaaclab, pxr) are deferred so importing this module
stays app-free.
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


# ----- USD authoring helpers --------------------------------------------------------------------
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
         material=None, orient=None) -> None:
    """Author one box child prim (translate -> orient -> scale; authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rig: the walled TRAY and the raised coaster SOCKET, both fixed in the
    rig frame (origin on the ground between them). +x runs from the bottle's start side
    toward the fixtures; tray at (tray_x, +tray_y), socket at (sock_x, -tray_y)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(40.0)

    co = cfg.contact_offset
    grip = _friction_material(stage, f"{prim_path}/mat_grip", 0.70, 0.60)
    wallm = _friction_material(stage, f"{prim_path}/mat_wall", 0.40, 0.35)
    wood = (0.48, 0.32, 0.16)
    wood_d = (0.38, 0.24, 0.11)
    slate = (0.25, 0.30, 0.42)

    # --- TRAY: floor plate + 4 walls; interior tray_in x tray_in, floor top at tray_f ---
    tx, ty = cfg.tray_x, cfg.tray_y
    ti, tf, tw, twt = cfg.tray_in, cfg.tray_floor, cfg.tray_wall_t, cfg.tray_wall_h
    out_w = ti + 2 * tw
    _box(stage, f"{prim_path}/tray_floor", (out_w, out_w, tf), (tx, ty, tf / 2),
         wood, co, material=grip)
    for i, (dx, dy) in enumerate(((ti / 2 + tw / 2, 0.0), (-ti / 2 - tw / 2, 0.0),
                                  (0.0, ti / 2 + tw / 2), (0.0, -ti / 2 - tw / 2))):
        sx2, sy2 = (tw, out_w) if i < 2 else (out_w, tw)
        _box(stage, f"{prim_path}/tray_wall{i}", (sx2, sy2, twt),
             (tx + dx, ty + dy, tf + twt / 2), wood_d, co, material=wallm)

    # --- SOCKET: raised plate + 4 lip walls; interior sock_in x sock_in ---
    sx0, sy0 = cfg.sock_x, -cfg.tray_y
    si, sp, sl, slh = cfg.sock_in, cfg.sock_plate, cfg.sock_lip_t, cfg.sock_lip_h
    sow = si + 2 * sl
    _box(stage, f"{prim_path}/sock_plate", (sow, sow, sp), (sx0, sy0, sp / 2),
         slate, co, material=grip)
    for i, (dx, dy) in enumerate(((si / 2 + sl / 2, 0.0), (-si / 2 - sl / 2, 0.0),
                                  (0.0, si / 2 + sl / 2), (0.0, -si / 2 - sl / 2))):
        s = (sl, sow) if i < 2 else (sow, sl)
        _box(stage, f"{prim_path}/sock_lip{i}", (s[0], s[1], slh),
             (sx0 + dx, sy0 + dy, sp + slh / 2), slate, co, material=wallm)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """FREE amber bottle, one rigid body of boxes. Local origin at the geometric
    CENTER (half-height) so the authored MassAPI CoM sits mid-body. Bottom-up: floor
    plate, 4 cavity walls, 4 funnel plates at 45 deg narrowing the cavity to the
    mouth, 4 neck walls. The interior is sealed everywhere except the mouth."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    prb.CreateLinearDampingAttr(0.05)
    prb.CreateAngularDampingAttr(0.05)
    prb.CreateSolverPositionIterationCountAttr(16)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.bottle_mass))

    co = cfg.contact_offset
    slick = _friction_material(stage, f"{prim_path}/mat_in", cfg.mu_inner, cfg.mu_inner * 0.85)
    amber = (0.85, 0.55, 0.15)
    amber_d = (0.72, 0.44, 0.10)

    w, t = cfg.body_w, cfg.wall_t                  # 0.070 outer, 0.008 walls
    cav = w - 2 * t                                # 0.054 cavity
    fh, wh, nh = cfg.floor_t, cfg.wall_h, cfg.neck_h
    mouth = cfg.mouth_ap                           # 0.040
    run = (cav - mouth) / 2                        # 0.007 funnel horizontal run (45 deg)
    H = fh + wh + run + nh                         # 0.130 total height
    zm = H / 2                                     # local origin at center
    neck_o = mouth + 2 * cfg.neck_t                # 0.050 neck outer

    _box(stage, f"{prim_path}/floor", (w, w, fh), (0, 0, fh / 2 - zm),
         amber_d, co, material=slick)
    zw = fh + wh / 2 - zm
    _box(stage, f"{prim_path}/wall_px", (t, w, wh), (cav / 2 + t / 2, 0, zw),
         amber, co, material=slick)
    _box(stage, f"{prim_path}/wall_nx", (t, w, wh), (-cav / 2 - t / 2, 0, zw),
         amber, co, material=slick)
    _box(stage, f"{prim_path}/wall_py", (cav, t, wh), (0, cav / 2 + t / 2, zw),
         amber, co, material=slick)
    _box(stage, f"{prim_path}/wall_ny", (cav, t, wh), (0, -cav / 2 - t / 2, zw),
         amber, co, material=slick)
    # funnel plates: 45 deg, slope length ~ run*sqrt(2) + overlap
    zf = fh + wh + run / 2 - zm
    rc = (cav + mouth) / 4                          # mid radius 0.0235
    sl = run * math.sqrt(2.0) + 0.004               # slope length with overlap
    c225, s225 = math.cos(math.pi / 8), math.sin(math.pi / 8)
    _box(stage, f"{prim_path}/fun_px", (sl, w, 0.003), (rc, 0, zf),
         amber_d, co, material=slick, orient=(c225, 0, s225, 0))     # Ry(+45)
    _box(stage, f"{prim_path}/fun_nx", (sl, w, 0.003), (-rc, 0, zf),
         amber_d, co, material=slick, orient=(c225, 0, -s225, 0))    # Ry(-45)
    _box(stage, f"{prim_path}/fun_py", (w, sl, 0.003), (0, rc, zf),
         amber_d, co, material=slick, orient=(c225, -s225, 0, 0))    # Rx(-45)
    _box(stage, f"{prim_path}/fun_ny", (w, sl, 0.003), (0, -rc, zf),
         amber_d, co, material=slick, orient=(c225, s225, 0, 0))     # Rx(+45)
    # neck: interior mouth x mouth, height nh
    zn = fh + wh + run + nh / 2 - zm
    nt = cfg.neck_t
    _box(stage, f"{prim_path}/neck_px", (nt, neck_o, nh), (mouth / 2 + nt / 2, 0, zn),
         amber, co, material=slick)
    _box(stage, f"{prim_path}/neck_nx", (nt, neck_o, nh), (-mouth / 2 - nt / 2, 0, zn),
         amber, co, material=slick)
    _box(stage, f"{prim_path}/neck_py", (mouth, nt, nh), (0, mouth / 2 + nt / 2, zn),
         amber, co, material=slick)
    _box(stage, f"{prim_path}/neck_ny", (mouth, nt, nh), (0, -mouth / 2 - nt / 2, zn),
         amber, co, material=slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            tray_x: float = 0.20
            tray_y: float = 0.14
            sock_x: float = 0.20
            tray_in: float = 0.170
            tray_floor: float = 0.008
            tray_wall_t: float = 0.008
            tray_wall_h: float = 0.055
            sock_in: float = 0.082
            sock_plate: float = 0.006
            sock_lip_t: float = 0.006
            sock_lip_h: float = 0.025
            contact_offset: float = 0.0015

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_w: float = 0.070
            wall_t: float = 0.008
            floor_t: float = 0.008
            wall_h: float = 0.100
            neck_h: float = 0.015
            neck_t: float = 0.005
            mouth_ap: float = 0.040
            bottle_mass: float = 0.15
            mu_inner: float = 0.15
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
        _SPAWNER_CACHE["bottle"] = BottleSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class DecantReturnSceneCfg(BaseCfg):
    """Config for `DecantReturnScene`. Honesty asserted in __post_init__: the marble
    fits through the mouth with pour clearance but cannot be pinched out through it,
    the bottle body fits a parallel jaw, the socket is a real drop-in registration,
    and the tray walls out a ground-rolling marble."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_lin: float = tunable(0.05)    # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)     # max |ang vel| when judging (rad/s)
    seat_xy_tol: float = tunable(0.010)  # bottle center within this of the socket center
    seat_z_tol: float = tunable(0.006)   # bottle base within this of the socket plate top
    seat_tilt_deg: float = tunable(8.0)  # bottle axis within this of world-up when seated

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    rig_yaw_deg: float = tunable(30.0)   # rig yaw +/- deg (fixture bearings must be read)
    rig_jitter: float = tunable(0.04)    # rig xy jitter (+/- m)
    bottle_jitter: float = tunable(0.05)  # bottle start xy (+/- m) around bottle_start
    bottle_yaw_deg: float = tunable(180.0)  # bottle start yaw (+/- deg, free)
    marble_jitter: float = tunable(0.006)   # marble in-cavity xy jitter (+/- m)

    # --- info: rig geometry (rig frame: origin on the ground) -----------------------------------
    tray_x: float = info(0.20)
    tray_y: float = info(0.14)           # tray at (+tray_x, +tray_y), socket at (+sock_x, -tray_y)
    sock_x: float = info(0.20)
    tray_in: float = info(0.170)         # tray interior width (square)
    tray_floor: float = info(0.008)      # tray floor plate thickness (interior floor top)
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.055)     # wall height above the tray floor top
    sock_in: float = info(0.082)         # socket interior width (square)
    sock_plate: float = info(0.006)      # socket plate thickness (seat rest height)
    sock_lip_t: float = info(0.006)
    sock_lip_h: float = info(0.025)      # lip height above the plate top
    # --- info: bottle ---------------------------------------------------------------------------
    body_w: float = info(0.070)          # outer square body width (fits an 80 mm jaw)
    wall_t: float = info(0.008)
    floor_t: float = info(0.008)
    wall_h: float = info(0.100)          # cavity height
    neck_h: float = info(0.015)
    neck_t: float = info(0.005)
    mouth_ap: float = info(0.040)        # mouth aperture (square, interior)
    bottle_mass: float = info(0.15)
    mu_inner: float = info(0.15)         # interior friction (marble must roll out)
    # --- info: marble ---------------------------------------------------------------------------
    marble_r: float = info(0.015)
    marble_mass: float = info(0.030)
    # --- info: starts (rig frame) ---------------------------------------------------------------
    bottle_start: tuple = info((-0.20, 0.0))
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.10+0.10+0.20+0.15+0.15 = 0.70 = non-success cap) ---------------
    w_lift: float = info(0.10)
    w_hover: float = info(0.10)
    w_decant: float = info(0.20)
    w_tray: float = info(0.15)
    w_seat: float = info(0.15)
    lift_z: float = info(0.040)          # bottle base above this = "lifted"
    hover_z: float = info(0.100)         # bottle center above this while over the tray = "hover"
    decant_jump: float = info(0.060)     # max per-step marble travel for the decant latch (m)

    # Derived (filled in __post_init__).
    bottle_h: float = info(0.0)          # total height
    bottle_zmid: float = info(0.0)       # half height (local origin above the base)
    cavity_w: float = info(0.0)

    def __post_init__(self) -> None:
        self.cavity_w = self.body_w - 2 * self.wall_t
        run = (self.cavity_w - self.mouth_ap) / 2
        self.bottle_h = self.floor_t + self.wall_h + run + self.neck_h
        self.bottle_zmid = self.bottle_h / 2
        d = 2 * self.marble_r
        # pour possible: the marble passes the mouth and the neck with real clearance
        assert self.mouth_ap >= d + 0.008, "mouth too tight to pour"
        # pinch-out impossible: no jaw finger pair fits through the mouth beside the
        # marble (needs marble + 2 fingers; 8 mm is already an optimistic finger)
        assert self.mouth_ap < d + 0.016, "marble could be pinched out through the mouth"
        # the marble rattles freely inside the cavity
        assert self.cavity_w >= d + 0.008, "cavity too tight"
        # the marble rests far below the mouth (no reach-in)
        assert self.wall_h >= 0.060, "cavity too shallow to guard the marble"
        # bottle body graspable by an 80 mm parallel jaw
        assert self.body_w <= 0.075, "bottle body exceeds the jaw stroke"
        # socket: drop-in registration with real slack and a real lip
        slack = self.sock_in - self.body_w
        assert 0.008 <= slack <= 0.020, "socket slack outside the drop-in band"
        assert self.sock_lip_h >= 0.015, "socket lip too low to register"
        # tray walls out a ground marble; interior is an arm-friendly target
        assert self.tray_wall_h >= 0.045 and self.tray_in >= 0.150
        # fixtures separated: a pour over the tray cannot land in the socket
        assert 2 * self.tray_y > self.tray_in / 2 + self.sock_in / 2 + 0.05
        # bottle start well clear of both fixtures
        assert self.bottle_start[0] + self.bottle_jitter + self.body_w \
            < min(self.tray_x - self.tray_in / 2 - self.tray_wall_t,
                  self.sock_x - self.sock_in / 2 - self.sock_lip_t) - 0.02


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
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


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("decant_return")
class DecantReturnScene(BaseScene):
    cfg: DecantReturnSceneCfg

    def __init__(self, cfg: DecantReturnSceneCfg | None = None) -> None:
        super().__init__(cfg or DecantReturnSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.80, dynamic_friction=0.70, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp["rig"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    tray_x=c.tray_x, tray_y=c.tray_y, sock_x=c.sock_x,
                    tray_in=c.tray_in, tray_floor=c.tray_floor,
                    tray_wall_t=c.tray_wall_t, tray_wall_h=c.tray_wall_h,
                    sock_in=c.sock_in, sock_plate=c.sock_plate,
                    sock_lip_t=c.sock_lip_t, sock_lip_h=c.sock_lip_h,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sp["bottle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    body_w=c.body_w, wall_t=c.wall_t, floor_t=c.floor_t,
                    wall_h=c.wall_h, neck_h=c.neck_h, neck_t=c.neck_t,
                    mouth_ap=c.mouth_ap, bottle_mass=c.bottle_mass,
                    mu_inner=c.mu_inner, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_start[0], c.bottle_start[1], c.bottle_zmid + 0.002)),
            ),
            "marble": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marble",
                spawn=sim_utils.SphereCfg(
                    radius=c.marble_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10, angular_damping=0.20,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.25, dynamic_friction=0.20, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.marble_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.10, 0.10))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_start[0], c.bottle_start[1],
                         c.floor_t + c.marble_r + 0.004)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.rig: RigidObject = env.iscene["rig"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.marble: RigidObject = env.iscene["marble"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches (partial credit survives transients; success needs the decant latch)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hover = torch.zeros(n, dtype=torch.bool, device=dev)
        self._decant = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tray_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        # decant continuity state
        self._prev_inside = torch.ones(n, dtype=torch.bool, device=dev)
        self._prev_marble = torch.zeros(n, 3, device=dev)
        self._prev_bottle = torch.zeros(n, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the rig (yaw + xy jitter), stand the bottle upright at a
        jittered start with free yaw, drop the marble inside the cavity with in-cavity
        jitter, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        from isaaclab.utils.math import quat_apply

        _ = torch.rand(m, device=dev)  # burn the first post-seed draw (degenerate)

        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        q_rig = _qz(psi)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        rp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_rig
        self.rig.write_root_state_to_sim(st, env_ids)

        # bottle: upright, jittered start, free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.bottle_start[0] + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
        loc[:, 1] = c.bottle_start[1] + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
        loc[:, 2] = c.bottle_zmid + 0.002
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bottle_yaw_deg)
        qb = _qmul(q_rig, _qz(byaw))
        stb = torch.zeros(m, 13, device=dev)
        stb[:, 0:3] = rp + origin + quat_apply(q_rig, loc)
        stb[:, 3:7] = qb
        self.bottle.write_root_state_to_sim(stb, env_ids)

        # marble: inside the cavity, small xy jitter, 4 mm drop onto the floor plate
        mloc = torch.zeros(m, 3, device=dev)
        mloc[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.marble_jitter
        mloc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.marble_jitter
        mloc[:, 2] = c.floor_t + c.marble_r + 0.004 - c.bottle_zmid  # bottle-local
        stm = torch.zeros(m, 13, device=dev)
        stm[:, 0:3] = stb[:, 0:3] + quat_apply(qb, mloc)
        stm[:, 3] = 1.0
        self.marble.write_root_state_to_sim(stm, env_ids)

        for lat in (self._lifted, self._hover, self._decant, self._tray_ever, self._seated):
            lat[env_ids] = False
        self._prev_inside[env_ids] = True
        self._prev_marble[env_ids] = stm[:, 0:3]
        self._prev_bottle[env_ids] = stb[:, 0:3]

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {n: getattr(self, n).data.root_state_w[env_ids].clone()
               for n in ("rig", "bottle", "marble")}
        out["latches"] = torch.stack(
            [self._lifted[env_ids], self._hover[env_ids], self._decant[env_ids],
             self._tray_ever[env_ids], self._seated[env_ids],
             self._prev_inside[env_ids]], dim=1).clone()
        out["prev_marble"] = self._prev_marble[env_ids].clone()
        out["prev_bottle"] = self._prev_bottle[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n in ("rig", "bottle", "marble"):
            getattr(self, n).write_root_state_to_sim(state[n], env_ids)
        lat = state["latches"]
        self._lifted[env_ids] = lat[:, 0]
        self._hover[env_ids] = lat[:, 1]
        self._decant[env_ids] = lat[:, 2]
        self._tray_ever[env_ids] = lat[:, 3]
        self._seated[env_ids] = lat[:, 4]
        self._prev_inside[env_ids] = lat[:, 5]
        self._prev_marble[env_ids] = state["prev_marble"]
        self._prev_bottle[env_ids] = state["prev_bottle"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An AMBER BOTTLE (square body {c.body_w * 1000:.0f} mm wide, "
            f"{c.bottle_h * 1000:.0f} mm tall, narrowing through a funnel shoulder to a "
            f"{c.mouth_ap * 1000:.0f} mm square mouth at the top) stands upright on the "
            f"floor. A RED MARBLE ({2 * c.marble_r * 1000:.0f} mm) is sealed inside it: "
            f"the mouth is the bottle's only opening, and the marble rests more than "
            f"{c.wall_h * 1000:.0f} mm below it — no gripper can reach in or pinch it "
            f"out. The bottle body itself fits a parallel jaw.\n"
            f"Nearby stand two fixtures whose positions and bearings change every "
            f"episode (read them from the scene): a WOODEN TRAY (interior "
            f"{c.tray_in * 1000:.0f} x {c.tray_in * 1000:.0f} mm, walls "
            f"{c.tray_wall_h * 1000:.0f} mm high — a marble on the ground cannot get "
            f"over them) and, on the opposite side, an empty BLUE-GRAY COASTER SOCKET "
            f"(a raised plate with a {c.sock_lip_h * 1000:.0f} mm lip, interior "
            f"{c.sock_in * 1000:.0f} mm square — the bottle base drops in with a few "
            f"millimetres of slack).\n"
            f"Goal: the marble at rest INSIDE the tray, and the bottle standing upright "
            f"SEATED inside the coaster socket. The only way to get the marble out is "
            f"to pick the bottle up, hold it over the tray, and tip it far past "
            f"horizontal so the marble rolls down the funnel and pours out of the mouth "
            f"— aim the pour so the marble lands between the tray walls. Then stand the "
            f"empty bottle back up and seat it in the socket. Putting the bottle itself "
            f"in the tray FAILS the task; a marble spilled outside the tray is lost. "
            f"The marble must leave the bottle through the mouth by a real pour."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the amber bottle, hold it over the wooden tray, and tip it far "
            "past horizontal so the red marble inside pours out of the narrow mouth "
            "and lands inside the tray. Then stand the empty bottle upright and seat "
            "it in the blue-gray coaster socket. Do not put the bottle in the tray; "
            "a marble spilled outside the tray fails."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rig.data.root_quat_w,
                                  pos_w - self.rig.data.root_pos_w)

    def _bottle_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.bottle.data.root_quat_w,
                                  pos_w - self.bottle.data.root_pos_w)

    def bottle_up(self) -> torch.Tensor:
        """(N,3) world direction of the bottle's local +z (mouth axis)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.bottle.data.root_quat_w, ez)

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def marble_in_bottle(self) -> torch.Tensor:
        """(N,) bool: marble center inside the bottle's interior volume (bottle frame).
        The interior is sealed except the mouth, so this is real containment."""
        c = self.cfg
        loc = self._bottle_local(self.marble.data.root_pos_w)
        half = c.bottle_zmid + 0.010
        return (loc[:, 0].abs() < c.body_w / 2 + 0.005) \
            & (loc[:, 1].abs() < c.body_w / 2 + 0.005) \
            & (loc[:, 2] > -half) & (loc[:, 2] < half)

    def marble_in_tray(self) -> torch.Tensor:
        """(N,) bool: marble center inside the tray interior box (rig frame) AND
        outside the bottle — a marble riding inside a bottle placed in the tray does
        NOT count."""
        c = self.cfg
        loc = self._rig_local(self.marble.data.root_pos_w)
        inside_box = ((loc[:, 0] - c.tray_x).abs() < c.tray_in / 2) \
            & ((loc[:, 1] - c.tray_y).abs() < c.tray_in / 2) \
            & (loc[:, 2] > c.tray_floor) \
            & (loc[:, 2] < c.tray_floor + c.tray_wall_h + 0.01)
        return inside_box & ~self.marble_in_bottle()

    def bottle_seated(self) -> torch.Tensor:
        """(N,) bool: bottle upright, its base inside the socket at plate-rest height
        (rig frame) — a genuine drop-in registration."""
        c = self.cfg
        up = self.bottle_up()
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.seat_tilt_deg))
        base_w = self.bottle.data.root_pos_w - up * c.bottle_zmid
        loc = self._rig_local(base_w)
        in_xy = ((loc[:, 0] - c.sock_x).abs() < c.seat_xy_tol) \
            & ((loc[:, 1] + c.tray_y).abs() < c.seat_xy_tol)
        at_z = (loc[:, 2] - c.sock_plate).abs() < c.seat_z_tol
        return upright & in_xy & at_z

    def _bottle_base_z(self) -> torch.Tensor:
        up = self.bottle_up()
        base_w = self.bottle.data.root_pos_w - up * self.cfg.bottle_zmid
        return (base_w - self.env_origins)[:, 2]

    def _update_latches(self) -> None:
        c = self.cfg
        inside = self.marble_in_bottle()
        mp = self.marble.data.root_pos_w
        bp = self.bottle.data.root_pos_w
        # decant: continuous inside -> outside transition. BOTH bodies must move
        # continuously: a teleported marble jumps too far, and a bottle teleported
        # away from around a stationary marble would otherwise read as an "exit"
        # with zero marble travel.
        step_ok = ((mp - self._prev_marble).norm(dim=-1) < c.decant_jump) \
            & ((bp - self._prev_bottle).norm(dim=-1) < c.decant_jump)
        self._decant |= self._prev_inside & ~inside & step_ok
        self._prev_inside = inside
        self._prev_marble = mp.clone()
        self._prev_bottle = bp.clone()

        self._lifted |= self._bottle_base_z() > c.lift_z
        bloc = self._rig_local(self.bottle.data.root_pos_w)
        over = ((bloc[:, 0] - c.tray_x).abs() < c.tray_in / 2) \
            & ((bloc[:, 1] - c.tray_y).abs() < c.tray_in / 2) \
            & (bloc[:, 2] > c.hover_z)
        self._hover |= over
        self._tray_ever |= self.marble_in_tray()
        self._seated |= self._decant & self.bottle_seated()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: marble settled inside the tray (outside the bottle) AND the
        bottle settled seated upright in the socket AND the decant was OBSERVED (the
        marble left the interior continuously through the mouth) — finite. The judged
        containment/seating is live physics; the decant latch rejects any teleport-out
        bypass."""
        self._update_latches()
        finite = torch.isfinite(self.bottle.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.marble.data.root_pos_w).all(dim=-1)
        return self.marble_in_tray() & self._settled(self.marble) \
            & self.bottle_seated() & self._settled(self.bottle) \
            & self._decant & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched stage credit (never evaporates) — lifted 0.10,
        hovered over the tray 0.10, decant observed 0.20, marble ever in the tray 0.15,
        seated after the decant 0.15 (cap 0.70); exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lifted.float()
                + c.w_hover * self._hover.float()
                + c.w_decant * self._decant.float()
                + c.w_tray * self._tray_ever.float()
                + c.w_seat * self._seated.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="decant_return", robot="null"))
