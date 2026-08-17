"""CarouselCabinetScene — stow the red ketchup bottle in the GREEN sector of a
crank-driven carousel cabinet, rotated to the back.

Derived from libero_90/kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet
("put the ketchup in the top drawer of the cabinet"): the seed's plan is pull a drawer
open along its prismatic rail, pick-and-drop the bottle into the tray, judged by a
static bounding box. Here the STORAGE MECHANISM and the PLAN are replaced wholesale:
the cabinet is a round, ROOFED housing with one fixed front WINDOW and a rotating
3-sector CAROUSEL inside (a lazy susan on a vertical bearing, driven by a crank on
top). Nothing slides and nothing opens: access is granted by ROTATING the right sector
into the fixed window. The solver must (1) read the green pointer on the top crank
(it points at the green target sector) and rotate the carousel until the GREEN sector
faces the window, (2) stand the RED ketchup bottle upright inside the green sector
through the window, (3) rotate the carousel ~180 deg further so the loaded green
sector is STOWED at the back of the housing — while the YELLOW mustard bottle must
stay outside. Depositing the bottle into whatever sector happens to face the window
(the seed's "drop it in the open receptacle" plan) earns no success: the sector must
be the green one, and it must end up rotated away from the window. Both remaining
phases are physically forced: the roof and the wall make the window the only way in
(so the green sector must be AT the window to receive the bottle), and the three
divider walls make it impossible to move the bottle between sectors without rotating
the carousel (the smoke battery shoves at 3x weight and the bottle never changes
sector).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - housing: KINEMATIC round cabinet at a FIXED pose (the bearing anchor of a
    world-frame revolute joint must not teleport): 9 wall segments closing
    260 deg of arc, a 100 deg front window (edge posts, sill), a roof frame with a
    small centre hole for the crank shaft, and a pedestal column.
  - carousel: DYNAMIC compound on a frictionless vertical revolute joint to the
    world through the housing centre: platform disc (r 155 mm, top at z 100 mm),
    centre hub, three divider walls 120 deg apart (three 120-deg sectors), a GREEN
    floor mat marking sector 0, and a crank shaft + GREEN crank arm above the roof.
    The crank arm points at the green sector's centre (carousel local +x): the green
    sector's heading is always visible from outside.
  - bottles: red "ketchup" and yellow "mustard" cylinders (r 30, h 160 mm) standing
    on the ground outside; which stands left/right is shuffled per episode.

Per-episode randomization (readback-verifiable): carousel start angle: the green
sector faces a uniform random azimuth 40..150 deg away from the window (either side —
never starting aligned, never starting stowed), bottle side swap + per-bottle xy
jitter.

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory):
  0.20 aligned   — green sector centre ever within `align_tol_deg` of the window
                   azimuth (latched; unreachable at reset by construction)
  0.20 inserted  — the ketchup bottle centre ever inside the housing interior
  0.20 seated    — the ketchup bottle ever standing upright in the GREEN sector,
                   near-still (latched)
  0.25 * p_max   — stow progress: while the bottle is seated in the green sector,
                   the running max of the green sector's angular distance from the
                   window, normalized 0 (aligned) -> 1 (at the stow gate)
  1.0 iff success() — ketchup upright in the green sector, green sector within
                   `stow_tol_deg` of the BACK (180 deg from the window), mustard
                   outside the housing, everything settled and finite. Non-success
                   is capped at 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- custom compound spawners -----------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw_deg: float = 0.0):
    """One box child: translate + (optional yaw) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        half = math.radians(yaw_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
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
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the housing at `prim_path`: KINEMATIC compound. Local frame: origin at
    the bearing axis on the ground; the window faces local +x.

    Children: pedestal column, 9 wall chord segments closing the 260 deg rear arc,
    two window edge posts at +/- window_half_deg, a sill below the window, and a
    4-piece roof frame (z 315..335 mm) with a small centre hole for the crank shaft
    (hole half-width 42 mm: a 60 mm bottle cannot pass beside the 15 mm shaft)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body, trim = c.body_color, c.trim_color
    # pedestal column under the platform
    _add_cyl(stage, f"{prim_path}/pedestal", center=(0.0, 0.0, 0.0375),
             radius=0.050, height=0.075, color=body, collide=collide)
    # wall chord segments: arc from window_half..360-window_half (260 deg), 9 chords
    n_seg = 9
    arc = 360.0 - 2.0 * c.window_half_deg
    seg = arc / n_seg
    rc = c.wall_r + c.wall_t / 2
    tan_len = 2.0 * (c.wall_r + c.wall_t) * math.tan(math.radians(seg / 2)) + 0.004
    wall_h = c.roof_z - 0.020
    for i in range(n_seg):
        th = c.window_half_deg + seg * (i + 0.5)
        rad = math.radians(th)
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(rc * math.cos(rad), rc * math.sin(rad), 0.020 + wall_h / 2),
                 size=(c.wall_t, tan_len, wall_h), color=body, collide=collide,
                 yaw_deg=th)
    # window edge posts
    for sgn, s in ((1.0, "p"), (-1.0, "n")):
        th = sgn * c.window_half_deg
        rad = math.radians(th)
        _add_box(stage, f"{prim_path}/post_{s}",
                 center=(rc * math.cos(rad), rc * math.sin(rad), 0.020 + wall_h / 2),
                 size=(0.025, 0.025, wall_h), color=trim, collide=collide,
                 yaw_deg=th)
    # sill below the window (blocks under-platform access; top just below platform)
    _add_box(stage, f"{prim_path}/sill", center=(rc + 0.010, 0.0, c.sill_top / 2),
             size=(0.035, 0.27, c.sill_top), color=trim, collide=collide)
    # roof frame with a centre hole (half-width `roof_hole_half`)
    a, big = c.roof_hole_half, 0.21
    zc = c.roof_z + c.roof_t / 2
    _add_box(stage, f"{prim_path}/roof_n", center=(0.0, (a + big) / 2, zc),
             size=(2 * big, big - a, c.roof_t), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_s", center=(0.0, -(a + big) / 2, zc),
             size=(2 * big, big - a, c.roof_t), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_e", center=((a + big) / 2, 0.0, zc),
             size=(big - a, 2 * a, c.roof_t), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_w", center=(-(a + big) / 2, 0.0, zc),
             size=(big - a, 2 * a, c.roof_t), color=body, collide=collide)
    return root


def _spawn_carousel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the carousel at `prim_path`: DYNAMIC compound. Local frame: origin on
    the bearing axis at the ground; sector 0 (the GREEN sector) is centred on local
    +x, dividers at 60/180/300 deg. Children: platform disc, hub, three divider
    walls, the green sector mat, the crank shaft and the green crank arm (along +x).

    Custom spawners apply no cfg schemas, so the rigid body, damping, solver
    iterations AND the mass/CoM/diagonal-inertia are all authored here explicitly
    (solve.py asserts the mass by readback)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.20)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(c.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in c.inertia]))
    collide = _make_collide(cfg.contact_offset)
    plat, green = c.plat_color, c.green_color
    zp = c.z_plat  # platform TOP height
    # platform disc
    _add_cyl(stage, f"{prim_path}/platform", center=(0.0, 0.0, zp - c.plat_t / 2),
             radius=c.plat_r, height=c.plat_t, color=plat, collide=collide)
    # hub
    _add_cyl(stage, f"{prim_path}/hub", center=(0.0, 0.0, zp + c.div_h / 2),
             radius=c.hub_r, height=c.div_h, color=plat, collide=collide)
    # dividers at 60 / 180 / 300 deg (sector 0 spans -60..+60 about local +x)
    div_len = c.plat_r - 0.020
    div_rc = 0.020 + div_len / 2
    for k, th in enumerate((60.0, 180.0, 300.0)):
        rad = math.radians(th)
        _add_box(stage, f"{prim_path}/divider_{k}",
                 center=(div_rc * math.cos(rad), div_rc * math.sin(rad),
                         zp + c.div_h / 2),
                 size=(div_len, c.div_t, c.div_h), color=plat, collide=collide,
                 yaw_deg=th)
    # green sector mat (marks sector 0)
    _add_box(stage, f"{prim_path}/green_mat", center=(0.095, 0.0, zp + c.mat_t / 2),
             size=(0.090, 0.085, c.mat_t), color=green, collide=collide)
    # crank shaft up through the roof hole + green crank arm along local +x
    _add_cyl(stage, f"{prim_path}/shaft", center=(0.0, 0.0, (zp + c.div_h + c.crank_z) / 2),
             radius=c.shaft_r, height=c.crank_z - (zp + c.div_h), color=plat,
             collide=collide)
    _add_box(stage, f"{prim_path}/crank", center=(c.crank_len / 2 - 0.005, 0.0, c.crank_z),
             size=(c.crank_len, 0.024, 0.020), color=green, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            wall_r: float = 0.175
            wall_t: float = 0.015
            window_half_deg: float = 50.0
            sill_top: float = 0.095
            roof_z: float = 0.315
            roof_t: float = 0.020
            roof_hole_half: float = 0.042
            body_color: tuple = (0.30, 0.32, 0.38)
            trim_color: tuple = (0.55, 0.42, 0.25)
            contact_offset: float = 0.002

        @configclass
        class CarouselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carousel)
            plat_r: float = 0.155
            plat_t: float = 0.012
            z_plat: float = 0.100
            hub_r: float = 0.030
            div_t: float = 0.012
            div_h: float = 0.190
            mat_t: float = 0.005
            shaft_r: float = 0.015
            crank_z: float = 0.420
            crank_len: float = 0.100
            mass: float = 0.9
            com_z: float = 0.12
            inertia: tuple = (0.015, 0.015, 0.010)
            plat_color: tuple = (0.72, 0.70, 0.62)
            green_color: tuple = (0.10, 0.70, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg
        _SPAWNER_CACHE["carousel"] = CarouselSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CarouselCabinetSceneCfg(BaseCfg):
    """Config for `CarouselCabinetScene`. The seat tolerances are honest by
    construction: a bottle physically standing anywhere in the green sector wedge
    (dividers at +/-60 deg, hub at 30 mm, rim at 155 mm, floor at 100/105 mm)
    passes them, while a bottle in another sector, on the sill, on the hub, tipped
    over, or outside the housing cannot."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    align_tol_deg: float = tunable(15.0)   # "green sector at the window" gate (aligned latch)
    stow_tol_deg: float = tunable(20.0)    # green sector within this of the BACK for success
    sector_half_deg: float = tunable(45.0)  # bottle azimuth (carousel frame) within this of
    # the green sector centre — geometric bound: a 30 mm bottle centre >= 55 mm from the axis
    # touches a divider at ~+/-50 deg, so 45 keeps a physical margin inside the wedge
    seat_r_min: float = tunable(0.055)     # bottle radial band on the platform (hub 30 + r 30
    seat_r_max: float = tunable(0.150)     # would collide below 55; rim at 155)
    seat_z_tol: float = tunable(0.025)     # |bottle centre z - nominal standing z| gate
    upright_max_deg: float = tunable(20.0)  # bottle axis within this of world-up
    settle_speed: float = tunable(0.05)    # max bottle |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.10)    # max carousel |ang vel z| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    yaw_min_deg: float = tunable(40.0)     # green sector start azimuth magnitude, lower bound
    yaw_max_deg: float = tunable(150.0)    # upper bound (never aligned, never stowed at reset)
    bottle_jitter: float = tunable(0.05)   # per-bottle xy jitter (+/- m)
    side_swap: bool = tunable(True)        # shuffle which bottle stands left/right

    # --- info: layout (world; the housing is FIXED — the bearing anchors to the world) ----------
    hub_pos: tuple = info((0.62, 0.0))     # bearing axis on the ground (window faces +x)
    bottle_slot: tuple = info((0.88, 0.24))  # bottle slots at (x, +/-y) on the ground
    # --- info: housing structure (local frame: origin on the bearing axis, ground) --------------
    wall_r: float = info(0.175)            # wall inner radius
    wall_t: float = info(0.015)
    window_half_deg: float = info(50.0)    # window spans +/- this about local +x
    sill_top: float = info(0.095)
    roof_z: float = info(0.315)            # roof underside
    roof_t: float = info(0.020)
    roof_hole_half: float = info(0.042)
    # --- info: carousel --------------------------------------------------------------------------
    plat_r: float = info(0.155)
    plat_t: float = info(0.012)
    z_plat: float = info(0.100)            # platform TOP height
    hub_r: float = info(0.030)
    div_t: float = info(0.012)
    div_h: float = info(0.190)
    mat_t: float = info(0.005)
    shaft_r: float = info(0.015)
    crank_z: float = info(0.420)
    crank_len: float = info(0.100)
    carousel_mass: float = info(0.9)
    carousel_com_z: float = info(0.12)
    carousel_inertia: tuple = info((0.015, 0.015, 0.010))
    # --- info: bottles ---------------------------------------------------------------------------
    bot_r: float = info(0.030)
    bot_h: float = info(0.160)
    bot_mass: float = info(0.15)
    ketchup_color: tuple = info((0.80, 0.08, 0.06))
    mustard_color: tuple = info((0.90, 0.75, 0.10))
    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.20 + 0.20 + 0.25 = 0.85 = the non-success cap)
    w_aligned: float = info(0.20)
    w_inserted: float = info(0.20)
    w_seated: float = info(0.20)
    w_prog: float = info(0.25)


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


def _wrap_deg(a: torch.Tensor) -> torch.Tensor:
    """Wrap degrees to (-180, 180]."""
    return a - 360.0 * torch.floor((a + 180.0) / 360.0)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("carousel_cabinet")
class CarouselCabinetScene(BaseScene):
    cfg: CarouselCabinetSceneCfg

    def __init__(self, cfg: CarouselCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or CarouselCabinetSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        housing_spawn = cls["housing"](
            wall_r=c.wall_r, wall_t=c.wall_t, window_half_deg=c.window_half_deg,
            sill_top=c.sill_top, roof_z=c.roof_z, roof_t=c.roof_t,
            roof_hole_half=c.roof_hole_half, contact_offset=c.contact_offset)
        carousel_spawn = cls["carousel"](
            plat_r=c.plat_r, plat_t=c.plat_t, z_plat=c.z_plat, hub_r=c.hub_r,
            div_t=c.div_t, div_h=c.div_h, mat_t=c.mat_t, shaft_r=c.shaft_r,
            crank_z=c.crank_z, crank_len=c.crank_len, mass=c.carousel_mass,
            com_z=c.carousel_com_z, inertia=c.carousel_inertia,
            contact_offset=c.contact_offset)

        bottle_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
        )

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
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=housing_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hub_pos[0], c.hub_pos[1], 0.0)),
            ),
            "carousel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carousel",
                spawn=carousel_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hub_pos[0], c.hub_pos[1], 0.0)),
            ),
        }
        for name in ("ketchup", "mustard"):
            color = c.ketchup_color if name == "ketchup" else c.mustard_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.bot_r, height=c.bot_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bot_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **bottle_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_slot[0], c.bottle_slot[1] if name == "ketchup"
                         else -c.bottle_slot[1], c.bot_h / 2 + 0.002)),
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
        self.housing: RigidObject = env.iscene["housing"]
        self.carousel: RigidObject = env.iscene["carousel"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.mustard: RigidObject = env.iscene["mustard"]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        n = env.num_envs
        dev = env.device
        # ketchup_side[e] = +1 / -1: sign of the y slot the KETCHUP bottle occupies
        self.ketchup_side = torch.ones(n, dtype=torch.float, device=dev)
        # crank drive slot: post_step OWNS set_external_force_and_torque on the
        # carousel; solve.py / smoke probes write this buffer, nothing else touches it
        self.crank_tau = torch.zeros(n, device=dev)  # torque about world +z (N*m)
        # latches (partial credit survives transients; success is judged live)
        self._aligned = torch.zeros(n, dtype=torch.bool, device=dev)
        self._inserted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prog = torch.zeros(n, device=dev)  # running max stow progress (seated)

    def _author_joint(self) -> None:
        """Per env: the carousel bearing — a revolute joint carousel->WORLD, axis Z,
        anchored at the (fixed) housing centre, no limits (free spin both ways). The
        housing never teleports, so a world anchor is exact; carousel resets rotate
        it about this very axis, which keeps the joint consistent."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        origins = self.env_origins.detach().cpu().numpy()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/carousel_bearing")
            j.CreateBody1Rel().SetTargets([f"{base}/Carousel"])
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(float(origins[i][0] + c.hub_pos[0]),
                                           float(origins[i][1] + c.hub_pos[1]),
                                           float(origins[i][2])))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: spin the carousel to a random start angle (green sector
        40..150 deg away from the window, either side — never aligned, never stowed),
        stand the bottles on their ground slots (side swap + jitter), zero the crank
        drive, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- carousel: random start angle about the bearing axis ---
        sign = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        mag = c.yaw_min_deg + torch.rand(m, device=dev) * (c.yaw_max_deg - c.yaw_min_deg)
        yaw = torch.deg2rad(sign * mag)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.hub_pos[0]
        st[:, 1] = c.hub_pos[1]
        st[:, 3:7] = _qz(yaw)
        st[:, 0:3] += origin
        self.carousel.write_root_state_to_sim(st, env_ids)

        # --- bottles: ground slots at (x, +/-y), side swap + jitter ---
        if c.side_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.ketchup_side[env_ids] = swap
        for body, sgn in ((self.ketchup, swap), (self.mustard, -swap)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.bottle_slot[0] + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
            st[:, 1] = sgn * c.bottle_slot[1] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
            st[:, 2] = c.bot_h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear the drive and the latches ---
        self.crank_tau[env_ids] = 0.0
        self._aligned[env_ids] = False
        self._inserted[env_ids] = False
        self._seated[env_ids] = False
        self._prog[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "carousel": self.carousel.data.root_state_w[env_ids].clone(),
            "ketchup": self.ketchup.data.root_state_w[env_ids].clone(),
            "mustard": self.mustard.data.root_state_w[env_ids].clone(),
            "ketchup_side": self.ketchup_side[env_ids].clone(),
            "crank_tau": self.crank_tau[env_ids].clone(),
            "aligned": self._aligned[env_ids].clone(),
            "inserted": self._inserted[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
            "prog": self._prog[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.carousel.write_root_state_to_sim(state["carousel"], env_ids)
        self.ketchup.write_root_state_to_sim(state["ketchup"], env_ids)
        self.mustard.write_root_state_to_sim(state["mustard"], env_ids)
        self.ketchup_side[env_ids] = state["ketchup_side"]
        self.crank_tau[env_ids] = state["crank_tau"]
        self._aligned[env_ids] = state["aligned"]
        self._inserted[env_ids] = state["inserted"]
        self._seated[env_ids] = state["seated"]
        self._prog[env_ids] = state["prog"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A round, roofed CAROUSEL CABINET (a lazy susan inside a drum, "
            f"~{2 * (c.wall_r + c.wall_t) * 100:.0f} cm across, roof at "
            f"{(c.roof_z + c.roof_t) * 100:.0f} cm) stands on the ground. Its grey wall "
            f"has ONE open front WINDOW (spanning {2 * c.window_half_deg:.0f} deg, "
            f"between two brown edge posts, above a brown sill) — the only opening; the "
            f"roof prevents reaching in from above. Inside, a rotating CAROUSEL "
            f"platform ({2 * c.plat_r * 100:.0f} cm across, floor at "
            f"{c.z_plat * 100:.0f} cm) is split by three radial divider walls into "
            f"three equal sectors. Exactly ONE sector has a GREEN floor mat — the "
            f"target sector. The carousel spins freely BOTH ways on its centre "
            f"bearing; a vertical shaft rises through the roof to a GREEN CRANK ARM "
            f"above the roof which turns with the carousel and always POINTS AT the "
            f"green sector, so you can read the green sector's heading from outside. "
            f"Turn the carousel by the crank arm (or by pushing a divider through the "
            f"window). On the ground in front of the cabinet stand two bottles "
            f"({2 * c.bot_r * 1000:.0f} mm across, {c.bot_h * 1000:.0f} mm tall): one "
            f"RED (ketchup) and one YELLOW (mustard) — which stands left/right varies "
            f"per episode, as do the bottle spots and the carousel's start angle "
            f"(the green sector never starts at the window).\n"
            f"Goal: stow the RED ketchup bottle in the GREEN sector at the BACK of the "
            f"cabinet. That takes three steps in a forced order: (1) rotate the "
            f"carousel until the green sector faces the window (green crank arm "
            f"pointing out the window), (2) stand the RED bottle UPRIGHT on the green "
            f"sector's floor through the window, (3) rotate the carousel about half a "
            f"turn more so the loaded green sector faces the BACK (green crank arm "
            f"pointing directly away from the window, within {c.stow_tol_deg:.0f} deg). "
            f"The YELLOW mustard bottle must stay OUTSIDE the cabinet. Placing the "
            f"bottle in a non-green sector, leaving the green sector at or near the "
            f"window, tipping the bottle over, stowing the mustard instead, or "
            f"bringing the mustard inside all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Rotate the carousel until the green sector faces the window, stand the "
            "red ketchup bottle upright on the green mat through the window, then "
            "rotate the carousel half a turn so the green sector faces the back. "
            "Leave the yellow mustard bottle outside the cabinet."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def green_az_deg(self) -> torch.Tensor:
        """(N,) green sector centre azimuth in the housing frame, degrees in
        (-180, 180]; 0 = at the window, +/-180 = stowed at the back. The housing is
        axis-aligned and fixed, so this is the carousel's world yaw."""
        q = self.carousel.data.root_quat_w
        return _wrap_deg(torch.rad2deg(2.0 * torch.atan2(q[:, 3], q[:, 0])))

    def _hub_rel(self, body) -> torch.Tensor:
        """(N,3) body position relative to the bearing axis (per env)."""
        c = self.cfg
        rel = body.data.root_pos_w - self.env_origins
        rel = rel - torch.tensor([c.hub_pos[0], c.hub_pos[1], 0.0],
                                 device=rel.device)
        return rel

    def _inside(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the housing interior volume."""
        c = self.cfg
        rel = self._hub_rel(body)
        return (rel[:, :2].norm(dim=-1) < c.wall_r - 0.010) \
            & (rel[:, 2] > 0.050) & (rel[:, 2] < c.roof_z)

    def _upright(self, body, max_deg: float) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(max_deg))

    def in_green_sector(self) -> torch.Tensor:
        """(N,) bool, geometric: the ketchup bottle standing UPRIGHT in the GREEN
        sector — carousel-frame azimuth within `sector_half_deg` of the green sector
        centre, radial in [seat_r_min, seat_r_max], centre height at standing height
        (within `seat_z_tol`), axis within `upright_max_deg` of up. Honest by
        construction: the dividers, hub, rim, floor and roof bound any bottle
        physically standing in the wedge inside these gates."""
        c = self.cfg
        rel = self._hub_rel(self.ketchup)
        az_w = torch.rad2deg(torch.atan2(rel[:, 1], rel[:, 0]))
        az_local = _wrap_deg(az_w - self.green_az_deg())
        r_xy = rel[:, :2].norm(dim=-1)
        z_nom = c.z_plat + c.mat_t / 2 + c.bot_h / 2  # on the mat / bare platform
        return (az_local.abs() < c.sector_half_deg) \
            & (r_xy > c.seat_r_min) & (r_xy < c.seat_r_max) \
            & ((rel[:, 2] - z_nom).abs() < c.seat_z_tol) \
            & self._upright(self.ketchup, c.upright_max_deg)

    def stowed(self) -> torch.Tensor:
        """(N,) bool: green sector centre within `stow_tol_deg` of the BACK."""
        adist_back = 180.0 - self.green_az_deg().abs()
        return adist_back <= self.cfg.stow_tol_deg

    def aligned(self) -> torch.Tensor:
        """(N,) bool: green sector centre within `align_tol_deg` of the window."""
        return self.green_az_deg().abs() <= self.cfg.align_tol_deg

    def mustard_outside(self) -> torch.Tensor:
        """(N,) bool: the mustard bottle centre OUTSIDE the housing interior."""
        return ~self._inside(self.mustard)

    def settled(self) -> torch.Tensor:
        """(N,) bool: both bottles slow and the carousel's spin below `settle_omega`."""
        c = self.cfg
        vk = self.ketchup.data.root_lin_vel_w.norm(dim=-1)
        vm = self.mustard.data.root_lin_vel_w.norm(dim=-1)
        w = self.carousel.data.root_ang_vel_w[:, 2].abs()
        return (vk < c.settle_speed) & (vm < c.settle_speed) & (w < c.settle_omega)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.carousel, self.ketchup, self.mustard)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._aligned |= fin & self.aligned()
        self._inserted |= fin & self._inside(self.ketchup)
        seat_live = fin & self.in_green_sector()
        calm = self.ketchup.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._seated |= seat_live & calm
        # stow progress: only while the bottle rides seated in the green sector
        adist = self.green_az_deg().abs()  # 0 at the window .. 180 at the back
        p = ((adist - c.align_tol_deg)
             / (180.0 - c.stow_tol_deg - c.align_tol_deg)).clamp(0.0, 1.0)
        self._prog = torch.where(seat_live, torch.maximum(self._prog, p), self._prog)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the crank-drive buffer (a torque about the bearing axis; the
        carousel only ever rotates about world z, so the body-frame encoding of
        (0, 0, tau) is invariant), then latch rubric progress."""
        n = self.env.num_envs
        dev = self.env.device
        zero = torch.zeros(n, 1, 3, device=dev)
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 2] = self.crank_tau
        self.carousel.set_external_force_and_torque(zero, tq)
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ketchup bottle standing upright in the GREEN sector, the
        green sector stowed at the BACK, the mustard bottle outside the housing,
        everything settled and finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.in_green_sector() & self.stowed() & self.mustard_outside() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*aligned + 0.20*inserted + 0.20*seated +
        0.25*prog_max (all latched; ~0 for the null policy — the reset never starts
        aligned and the bottles start outside), capped at 0.85 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_aligned * self._aligned.float()
                + c.w_inserted * self._inserted.float()
                + c.w_seated * self._seated.float()
                + c.w_prog * self._prog).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="carousel_cabinet", robot="null"))
