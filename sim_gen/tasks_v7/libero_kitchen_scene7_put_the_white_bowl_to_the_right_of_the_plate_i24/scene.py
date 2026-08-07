"""MugHookScene — hang the white mug on the BLUE peg of a wall rack, by its handle.

Derived from libero_90/kitchen_scene7 "put the white bowl to the right of the plate",
but the GOAL MECHANICS are replaced wholesale. The seed's plan is: grasp one rigid
object, carry it through free space, set it down on the support surface at a relative
xy offset from a reference object — success is a bbox check on the transported
object's own RESTING-ON-THE-SURFACE pose (x/y bands vs the plate, z equal to the
table). Here NOTHING may end resting on a surface: the mug must end SUSPENDED — its
rectangular handle loop threaded over a peg so the peg carries the mug's weight
through the loop while the mug dangles freely off the ground. The seed's own end
state (mug set down on the floor at the "right spot" beside/below the target peg) is
expressible in this scene and scores ~0: success requires the handle aperture around
the peg AND the mug airborne, and a floor placement is neither.

The target is IDENTIFIED BY COLOR, not by position: the rack carries three pegs —
red, green, blue — and which peg occupies which slot (and each peg's height) is
resampled every episode, so a memorized pose fails; the mug goes on the BLUE peg
only. An amber BEAKER with no handle sits near the mug as an identity distractor —
it physically cannot be hung (nothing to thread).

Assets are fully procedural (compound-spawner pattern — child colliders of one body
never self-collide):
  - rack: KINEMATIC — vertical back panel (0.40 y x 0.36 z x 0.024 m) on a base
    plate, standing on the floor, facing the robot (local -x).
  - pegs: three KINEMATIC bodies, one per color. Local origin at the peg ROOT,
    +z along the peg axis: cylinder r 8 mm x 95 mm long, plus a 20 mm root flange
    disc (the depth stop). Mounted on the panel face angled 12 deg UP so a hung
    loop slides toward the panel and stays on.
  - mug: DYNAMIC white compound — octagonal cup (inner r 26 mm, wall 5 mm x 62 mm,
    floor 8 mm; outer ~67 mm across, 70 mm tall, 120 g) + a rectangular handle
    loop on one side (three 8 mm bars + the cup wall) enclosing a 30 x 34 mm
    APERTURE. Local origin at the cup bottom center, +z up, handle along +x;
    aperture center at local (46, 0, 41) mm.
  - beaker: DYNAMIC amber cup, NO handle (slightly smaller). Distractor only.

Per-episode randomization (readback-verifiable): the color->slot permutation of the
three pegs, each peg's mounting height (0.20..0.25 m), rack xy jitter + yaw, mug xy
jitter + free yaw, beaker xy jitter + free yaw.

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory):
  0.15 * lift    — the mug was ever raised above `lift_z` (latched; null policy 0)
  0.30 * thread  — the handle aperture center ever came within `thread_gate` of the
                   BLUE peg's axis segment WITH the aperture face-on to the axis
                   (latched: the loop has been threaded over the target peg)
  1.0 iff success() — live: aperture threaded on the blue peg AND the mug airborne
                   (origin above `hang_clear_z` — nothing under it but air) AND
                   settled AND finite. Non-success capped at 0.45.

Honesty notes. `threaded` has two clauses, both honest by construction. DISTANCE:
a loop actually around the peg holds its aperture center within ~17 mm of the axis
(half the 34 mm aperture height; measured ~9 mm at the settled hang), a hang on
either neighboring peg is >= 90 mm away, and the axis segment used for the check
stops 25 mm short of the tip so an aligned-but-unthreaded loop hovering off the
tip stays outside the 30 mm gate. ALIGNMENT: the aperture NORMAL (mug local +y)
must lie within ~60 deg of the peg axis — a real hang pivots about the peg axis
and keeps |cos| ~ 1, while a mug hooked by its CUP MOUTH (peg inside the cup, the
plausible near-miss) has its cup axis along the peg, so the aperture normal stays
perpendicular to the axis (|cos| ~ 0) under both its tilt and its roll — the
settled mouth-hook can bring the aperture CENTER as close as ~29 mm to the axis,
which is why distance alone is not the test (smoke-verified). `hang_clear_z` (0.10 m) is unreachable by resting on anything in
the scene: floor rest = 0, on the beaker ~ 0.066, while a real hang on the lowest
peg sits ~ 0.126. Suspension is judged live and must survive the settle + the
solver's hands-off persistence window: an unsupported mug at the right pose simply
falls (smoke-verified).

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


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None):
    """One box child: translate [+ orient] + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    """One z-axis cylinder child: translate, displayColor, collider."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _qz_tuple(yaw: float) -> tuple:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rack: KINEMATIC compound. Local frame: origin at the bottom center
    on the ground, panel front face toward local -x. Children: panel + base plate."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/panel",
             center=(0.0, 0.0, c.panel_h / 2 + c.base_t),
             size=(c.panel_t, c.panel_w, c.panel_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/base",
             center=(-0.03, 0.0, c.base_t / 2),
             size=(0.16, c.panel_w + 0.04, c.base_t), color=c.base_color,
             collide=collide)
    return root


def _spawn_peg(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one peg: KINEMATIC compound. Local frame: origin at the ROOT (panel
    face), +z along the peg axis. Children: shaft cylinder + root flange disc."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/shaft", center=(0.0, 0.0, c.length / 2),
             radius=c.radius, height=c.length, color=c.color, collide=collide)
    _add_cyl(stage, f"{prim_path}/flange", center=(0.0, 0.0, c.flange_t / 2),
             radius=c.flange_r, height=c.flange_t, color=c.color, collide=collide)
    return root


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the mug: DYNAMIC compound. Local frame: origin at the bottom center of
    the cup floor, +z up, handle along +x. Children: floor disc + 8 wall boxes
    (octagonal cup) + 3 handle bars enclosing the aperture against the +x wall."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.4)
    px.CreateAngularDampingAttr(0.6)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             radius=c.inner_r + c.wall_t, height=c.floor_t, color=c.color,
             collide=collide)
    n_side = 8
    rmid = c.inner_r + c.wall_t / 2
    side_l = 2 * rmid * math.tan(math.pi / n_side) + 0.004
    zc = c.floor_t + c.wall_h / 2
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(rmid * math.cos(a), rmid * math.sin(a), zc),
                 size=(c.wall_t, side_l, c.wall_h), color=c.color,
                 collide=collide, orient=_qz_tuple(a))
    # handle loop against the +x wall flat (outer face at x = inner_r + wall_t + ...):
    b = c.bar_t
    x_wall = c.inner_r + c.wall_t  # +x wall flat outer face
    x_out = x_wall + c.ap_w + b / 2  # outer bar center
    z_bot = c.ap_z0 - b / 2  # bottom bar center
    z_top = c.ap_z0 + c.ap_h + b / 2  # top bar center
    x_mid = (x_wall - b / 2 + x_out) / 2
    bar_lx = x_out - x_wall + b  # horizontal bars: reach into the wall
    _add_box(stage, f"{prim_path}/handle_bot", center=(x_mid, 0.0, z_bot),
             size=(bar_lx, b, b), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/handle_top", center=(x_mid, 0.0, z_top),
             size=(bar_lx, b, b), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/handle_out",
             center=(x_out, 0.0, (z_bot + z_top) / 2),
             size=(b, b, z_top - z_bot + b), color=c.color, collide=collide)
    return root


def _spawn_beaker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the beaker: DYNAMIC octagonal cup, NO handle (identity distractor)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.4)
    px.CreateAngularDampingAttr(0.6)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             radius=c.inner_r + c.wall_t, height=c.floor_t, color=c.color,
             collide=collide)
    n_side = 8
    rmid = c.inner_r + c.wall_t / 2
    side_l = 2 * rmid * math.tan(math.pi / n_side) + 0.004
    zc = c.floor_t + c.wall_h / 2
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(rmid * math.cos(a), rmid * math.sin(a), zc),
                 size=(c.wall_t, side_l, c.wall_h), color=c.color,
                 collide=collide, orient=_qz_tuple(a))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            panel_w: float = 0.40
            panel_h: float = 0.36
            panel_t: float = 0.024
            base_t: float = 0.012
            color: tuple = (0.45, 0.33, 0.22)
            base_color: tuple = (0.35, 0.25, 0.16)
            contact_offset: float = 0.0015

        @configclass
        class PegSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_peg)
            radius: float = 0.008
            length: float = 0.095
            flange_r: float = 0.020
            flange_t: float = 0.008
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.0015

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            inner_r: float = 0.026
            wall_t: float = 0.005
            wall_h: float = 0.062
            floor_t: float = 0.008
            bar_t: float = 0.008
            ap_w: float = 0.030
            ap_h: float = 0.034
            ap_z0: float = 0.024
            mass: float = 0.12
            color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.0015

        @configclass
        class BeakerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beaker)
            inner_r: float = 0.024
            wall_t: float = 0.005
            wall_h: float = 0.050
            floor_t: float = 0.008
            mass: float = 0.10
            color: tuple = (0.85, 0.62, 0.18)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, peg=PegSpawnerCfg,
                              mug=MugSpawnerCfg, beaker=BeakerSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MugHookSceneCfg(BaseCfg):
    """Config for `MugHookScene`. The threading gate and the airborne gate are honest
    by construction (see the module docstring): a loop around the peg holds its
    aperture center within ~17 mm of the axis (gate 30), a mouth-hook keeps it
    >= 40 mm away, and no resting support in the scene reaches `hang_clear_z`."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    thread_gate: float = tunable(0.030)  # aperture center within this of the BLUE peg's
    # axis segment (segment stops `seg_tip_margin` short of the tip: an aligned loop
    # hovering just off the tip is >= 35 mm from the segment and stays out)
    hang_clear_z: float = tunable(0.10)  # mug origin above this = airborne (floor rest 0,
    # on-beaker rest ~0.066, real hang on the lowest peg ~0.126)
    align_min: float = tunable(0.5)      # min |cos(aperture normal, peg axis)| for
    # `threaded` (a real hang ~1; a mouth-hook ~0 under both its tilt and roll)
    settle_speed: float = tunable(0.05)  # max mug |lin vel| when judging (m/s)
    lift_z: float = tunable(0.10)        # latched lift credit: mug origin ever above this

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    rack_jitter: float = tunable(0.03)   # rack xy jitter (+/- m)
    rack_yaw_deg: float = tunable(8.0)   # rack yaw (+/- deg)
    slot_z0: float = tunable(0.20)       # lowest peg mounting height (m)
    slot_dz_max: float = tunable(0.05)   # per-peg mounting height spread (uniform 0..this)
    mug_jitter: float = tunable(0.04)    # mug xy jitter (+/- m)
    mug_yaw_deg: float = tunable(180.0)  # mug yaw (+/- deg)
    beaker_jitter: float = tunable(0.04)  # beaker xy jitter (+/- m)

    # --- info: layout (env-local, ground plane z = 0) ------------------------------------------
    rack_pos: tuple = info((0.42, 0.0))  # rack bottom center
    mug_pos: tuple = info((0.16, 0.10))  # mug start (floor, robot side)
    beaker_pos: tuple = info((0.16, -0.12))  # beaker start (floor, robot side)
    slot_y: tuple = info((-0.13, 0.0, 0.13))  # peg slots across the panel (rack frame y)
    # --- info: rack / peg structure -------------------------------------------------------------
    panel_w: float = info(0.40)
    panel_h: float = info(0.36)
    panel_t: float = info(0.024)
    base_t: float = info(0.012)
    peg_r: float = info(0.008)
    peg_len: float = info(0.095)
    peg_tilt_deg: float = info(12.0)     # pegs angle UP by this (hung loops slide inward)
    flange_r: float = info(0.020)
    seg_root_margin: float = info(0.010)  # threading segment: [root+this, tip-seg_tip_margin]
    seg_tip_margin: float = info(0.025)
    peg_colors: tuple = info((("red", (0.85, 0.12, 0.10)),
                              ("green", (0.10, 0.62, 0.22)),
                              ("blue", (0.12, 0.32, 0.92))))
    target_color: str = info("blue")
    # --- info: mug structure (local origin at the cup bottom center, handle along +x) ----------
    mug_inner_r: float = info(0.026)
    mug_wall_t: float = info(0.005)
    mug_wall_h: float = info(0.062)
    mug_floor_t: float = info(0.008)
    mug_h: float = info(0.070)           # floor_t + wall_h
    mug_outer_r: float = info(0.0337)    # octagon corner reach
    mug_mass: float = info(0.12)
    bar_t: float = info(0.008)
    ap_w: float = info(0.030)            # aperture width (cup wall -> outer bar inner face)
    ap_h: float = info(0.034)            # aperture height (bottom bar top -> top bar bottom)
    ap_z0: float = info(0.024)           # aperture bottom edge height (local z)
    ap_center: tuple = info((0.046, 0.0, 0.041))  # aperture center, mug local frame
    # --- info: beaker ---------------------------------------------------------------------------
    beaker_inner_r: float = info(0.024)
    beaker_wall_h: float = info(0.050)
    beaker_h: float = info(0.058)
    beaker_mass: float = info(0.10)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.15 + 0.30 = 0.45 = the non-success cap) -----------------------
    w_lift: float = info(0.15)
    w_thread: float = info(0.30)


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _seg_dist(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N,) distance from points `p` to segments [a, b], all (N, 3)."""
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)
    t = t.clamp(0.0, 1.0)
    proj = a + ab * t.unsqueeze(-1)
    return (p - proj).norm(dim=-1)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mug_hook")
class MugHookScene(BaseScene):
    cfg: MugHookSceneCfg

    def __init__(self, cfg: MugHookSceneCfg | None = None) -> None:
        super().__init__(cfg or MugHookSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
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
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=cls["rack"](panel_w=c.panel_w, panel_h=c.panel_h,
                                  panel_t=c.panel_t, base_t=c.base_t,
                                  contact_offset=c.contact_offset, rigid_props=kin),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
            "mug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mug",
                spawn=cls["mug"](inner_r=c.mug_inner_r, wall_t=c.mug_wall_t,
                                 wall_h=c.mug_wall_h, floor_t=c.mug_floor_t,
                                 bar_t=c.bar_t, ap_w=c.ap_w, ap_h=c.ap_h,
                                 ap_z0=c.ap_z0, mass=c.mug_mass,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.mug_pos[0], c.mug_pos[1], 0.002)),
            ),
            "beaker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beaker",
                spawn=cls["beaker"](inner_r=c.beaker_inner_r,
                                    wall_h=c.beaker_wall_h, mass=c.beaker_mass,
                                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.beaker_pos[0], c.beaker_pos[1], 0.002)),
            ),
        }
        for i, (name, rgb) in enumerate(c.peg_colors):
            out[f"peg_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Peg_" + name,
                spawn=cls["peg"](radius=c.peg_r, length=c.peg_len,
                                 flange_r=c.flange_r, color=rgb,
                                 contact_offset=c.contact_offset, rigid_props=kin),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0] - c.panel_t / 2 - 0.001,
                         c.slot_y[i], c.slot_z0)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.rack: RigidObject = env.iscene["rack"]
        self.mug: RigidObject = env.iscene["mug"]
        self.beaker: RigidObject = env.iscene["beaker"]
        self.pegs: dict[str, RigidObject] = {
            name: env.iscene[f"peg_{name}"] for name, _rgb in c.peg_colors}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # peg_slot[e, i]: which slot (0..2, -y to +y) peg i (red, green, blue) occupies
        self.peg_slot = torch.tensor([[0, 1, 2]], dtype=torch.long,
                                     device=dev).expand(n, 3).clone()
        self._ap_local = torch.tensor(c.ap_center, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._thread = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rack (jitter + yaw), sample the color->slot
        permutation and per-peg heights and mount the pegs on the panel face
        (angled up), place mug and beaker on the floor (jitter + free yaw), clear
        the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def jit(k: float) -> torch.Tensor:
            return (torch.rand(m, 2, device=dev) * 2 - 1) * k

        # --- rack (kinematic): pos + jitter + yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        q_rack = _qz(yaw)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rack_pos[0]
        rp[:, 1] = c.rack_pos[1]
        rp[:, :2] += jit(c.rack_jitter)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_rack
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- pegs: permute colors over slots, sample heights, mount on the face ---
        perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)  # peg i -> slot
        self.peg_slot[env_ids] = perm
        slot_y = torch.tensor(c.slot_y, device=dev)
        tilt = math.radians(c.peg_tilt_deg)
        theta = -(math.pi / 2 - tilt)  # local +z -> rack (-cos tilt, 0, +sin tilt)
        q_tilt = _qy(torch.full((m,), theta, device=dev))
        q_peg = _qmul(q_rack, q_tilt)
        face_x = -(c.panel_t / 2 + 0.001)
        for i, (name, _rgb) in enumerate(c.peg_colors):
            z_i = c.slot_z0 + torch.rand(m, device=dev) * c.slot_dz_max
            off = torch.zeros(m, 3, device=dev)
            off[:, 0] = face_x
            off[:, 1] = slot_y[perm[:, i]]
            off[:, 2] = z_i
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = rp + quat_apply(q_rack, off) + origin
            st[:, 3:7] = q_peg
            self.pegs[name].write_root_state_to_sim(st, env_ids)

        # --- mug (dynamic): floor start + jitter + free yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.mug_pos[0]
        st[:, 1] = c.mug_pos[1]
        st[:, :2] += jit(c.mug_jitter)
        st[:, 2] = 0.002
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1)
                         * math.radians(c.mug_yaw_deg))
        st[:, 0:3] += origin
        self.mug.write_root_state_to_sim(st, env_ids)

        # --- beaker (dynamic): floor start + jitter + free yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.beaker_pos[0]
        st[:, 1] = c.beaker_pos[1]
        st[:, :2] += jit(c.beaker_jitter)
        st[:, 2] = 0.002
        st[:, 3:7] = _qz(torch.rand(m, device=dev) * 2 * math.pi)
        st[:, 0:3] += origin
        self.beaker.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._lift[env_ids] = False
        self._thread[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "mug": self.mug.data.root_state_w[env_ids].clone(),
            "beaker": self.beaker.data.root_state_w[env_ids].clone(),
            "pegs": {n: b.data.root_state_w[env_ids].clone()
                     for n, b in self.pegs.items()},
            "peg_slot": self.peg_slot[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "thread": self._thread[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.mug.write_root_state_to_sim(state["mug"], env_ids)
        self.beaker.write_root_state_to_sim(state["beaker"], env_ids)
        for n, b in self.pegs.items():
            b.write_root_state_to_sim(state["pegs"][n], env_ids)
        self.peg_slot[env_ids] = state["peg_slot"]
        self._lift[env_ids] = state["lift"]
        self._thread[env_ids] = state["thread"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A brown wooden RACK stands on the floor: a vertical panel "
            f"(~{c.panel_w * 1000:.0f} mm wide, {c.panel_h * 1000:.0f} mm tall) facing "
            f"you, carrying three round PEGS that stick out toward you, angled slightly "
            f"upward — one RED, one GREEN, one BLUE (each {2 * c.peg_r * 1000:.0f} mm "
            f"thick, {c.peg_len * 1000:.0f} mm long, with a small disc of the same color "
            f"where it meets the panel). WHICH PEG IS WHERE CHANGES EVERY EPISODE — the "
            f"three colors are shuffled over the three mounting slots and each peg's "
            f"height varies — so find the BLUE peg by its color, not by position. On the "
            f"floor in front of the rack sit two vessels: a WHITE octagonal MUG "
            f"(~{2 * c.mug_outer_r * 1000:.0f} mm across, {c.mug_h * 1000:.0f} mm tall) "
            f"with a rectangular HANDLE on one side whose loop encloses an open "
            f"{c.ap_w * 1000:.0f} x {c.ap_h * 1000:.0f} mm window, and an AMBER BEAKER of "
            f"similar size with NO handle — the beaker is a distractor and cannot be "
            f"hung.\n"
            f"Goal: hang the WHITE MUG on the BLUE peg BY ITS HANDLE — pass the peg "
            f"through the handle's window and let go, so the peg carries the mug through "
            f"the loop and the mug dangles freely in the air (it will tilt as it hangs; "
            f"that is fine). The mug must not touch the floor and nothing else may hold "
            f"it up. A mug set down on the floor (anywhere, however close to the rack), "
            f"a mug hung on the red or green peg, a mug perched on the peg by its cup "
            f"opening instead of the handle, or a moved beaker all fail. Success: the "
            f"blue peg through the handle loop, the mug hanging still, clear of the "
            f"ground."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Hang the white mug on the blue peg of the rack by passing the peg "
            "through the mug's handle loop, then let it dangle freely. The mug must "
            "hang from the blue peg only — not the red or green peg — and must not "
            "touch the floor."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def aperture_w(self) -> torch.Tensor:
        """(N, 3): the handle aperture center in world coordinates."""
        from isaaclab.utils.math import quat_apply

        return self.mug.data.root_pos_w + quat_apply(
            self.mug.data.root_quat_w,
            self._ap_local.expand(self.env.num_envs, 3))

    def peg_frame(self, color: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(root_w (N,3), axis dir_w (N,3)) of peg `color` (dir points out of the
        panel, tilted up)."""
        from isaaclab.utils.math import quat_apply

        peg = self.pegs[color]
        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        return peg.data.root_pos_w, quat_apply(peg.data.root_quat_w, ez)

    def dist_to_peg(self, color: str) -> torch.Tensor:
        """(N,) distance from the aperture center to peg `color`'s axis segment
        [root + seg_root_margin, tip - seg_tip_margin]. The tip margin keeps an
        aligned-but-unthreaded loop hovering off the tip outside the gate."""
        c = self.cfg
        root, d = self.peg_frame(color)
        a = root + d * c.seg_root_margin
        b = root + d * (c.peg_len - c.seg_tip_margin)
        return _seg_dist(self.aperture_w(), a, b)

    def loop_alignment(self) -> torch.Tensor:
        """(N,) |cos| between the aperture NORMAL (mug local +y) and the target
        peg's axis. A loop threaded over the peg keeps this ~1 (its swing pivots
        about the axis); a cup mouth-hooked over the peg keeps it ~0."""
        from isaaclab.utils.math import quat_apply

        ey = torch.tensor([0.0, 1.0, 0.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        _root, d = self.peg_frame(self.cfg.target_color)
        return (quat_apply(self.mug.data.root_quat_w, ey) * d).sum(-1).abs()

    def threaded(self) -> torch.Tensor:
        """(N,) bool: the handle aperture center within `thread_gate` of the TARGET
        (blue) peg's axis segment AND the aperture face-on to the axis — the loop
        is around the target peg (distance alone would admit a settled mouth-hook
        at ~29 mm; the alignment clause rejects it)."""
        c = self.cfg
        return (self.dist_to_peg(c.target_color) < c.thread_gate) \
            & (self.loop_alignment() > c.align_min)

    def airborne(self) -> torch.Tensor:
        """(N,) bool: mug origin above `hang_clear_z` (env frame) — nothing in the
        scene can prop it that high except a peg through the loop."""
        return (self.mug.data.root_pos_w - self.env_origins)[:, 2] > self.cfg.hang_clear_z

    def mug_settled(self) -> torch.Tensor:
        """(N,) bool: mug |lin vel| below `settle_speed`."""
        return self.mug.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _update_latches(self) -> None:
        c = self.cfg
        mug_z = (self.mug.data.root_pos_w - self.env_origins)[:, 2]
        self._lift |= mug_z > c.lift_z
        self._thread |= self.threaded()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: the blue peg through the handle loop (threaded), the mug
        airborne, settled and finite — a real, load-carrying hang. All clauses are
        physical outcomes; an unsupported mug at the right pose falls out of this
        within a few steps."""
        self._update_latches()
        finite = torch.isfinite(self.mug.data.root_pos_w).all(dim=-1)
        return self.threaded() & self.airborne() & self.mug_settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*lift + 0.30*thread (latched; ~0 for the null
        policy), capped at 0.45 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lift.float()
                + c.w_thread * self._thread.float()).clamp(max=0.45)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mug_hook", robot="null"))
