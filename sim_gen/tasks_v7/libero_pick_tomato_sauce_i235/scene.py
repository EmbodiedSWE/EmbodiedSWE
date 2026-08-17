"""SauceCarouselScene — spin the roofed service carousel by its overhead crank bar
until the RED tomato-sauce can parks in the open hatch sector, lift it out over the
sill, and drop it into the free-standing basket, leaving the WHITE decoy can riding
the platter (sim_gen task `libero_pick_tomato_sauce_i235`).

Derived from libero `pick_tomato_sauce` ("pick up the tomato sauce and place it in the
basket"), but STRATEGICALLY different: the seed is identify-grasp-drop — the named can
sits exposed in tabletop grocery clutter, graspable from the first frame, and success
is a bounding-box containment readout over an open basket. Here the can is CAPTIVE: it
rides a free-spinning carousel platter inside a walled, roofed ROTUNDA. A fence ring
and a slab roof close 265-285 degrees of the circumference; the only opening is a
75-degree HATCH sector with a low SILL, and the can spawns at least 65 degrees away
from the hatch bearing, deep under the roof, where it cannot be reached from above or
from the side. The platter's central shaft rises through a hole in the roof to a green
CRANK BAR: the solver must read the target's angular position, choose a spin direction,
and drive the carousel by the crank — a rotary machine operation with a stopping
problem (overshoot puts the can back under the roof) — until the RED can sits in the
hatch, then lift it over the sill and out through the open sector, carry it to the
basket on the open floor, and drop it in. The WHITE decoy can rides the same platter
and must still be riding it at the end.

Strategy vs the corpus neighbours read for this construction: cellar_tow (i43) BUILDS
a peg-in-eye tool coupling between two free bodies and tows; rocker_lock (i104) is a
press-and-hold gravity ferry through a gated letterbox; flat_pack_crate (i151)
assembles the receptacle itself with a lid-last order; pen_holder is multi-insert into
an open cup. None of them involve angular planning on a rotary stage: here the
mechanism is a free REVOLUTE platter driven from outside the sealed volume, the core
skill is choose-direction-spin-and-stop (a park-to-tolerance problem), and the
receptacle is ordinary — the difficulty is upstream of the place, in creating access.
The seed's own plan (reach in and grab the can where it stands) is geometrically
impossible at reset: the can is under a roof 4.3 cm above its top, behind a fence,
65-145 degrees away from the only opening.

success(): the alignment latches fired in order (the can was parked in the hatch
sector, then extracted while parked), the RED can rests inside the basket volume
(basket-frame walls + height band), the sauce / basket / platter are settled, the
WHITE decoy is still upright and riding the platter, and all states are finite.
score(): latched, non-decreasing, anchored in the demonstrated solution — 0.25 *
spin-progress (running max of the fractional angular approach toward the hatch while
riding the platter), +0.15 parked in the hatch sector, +0.20 extracted after parking,
1.0 iff success(). The null policy scores ~0 (the can spawns >= 65 degrees off the
hatch and nothing moves).

Assets are fully procedural (compound spawners; child prims of one body never
self-collide):
  - rotunda (dynamic, 45 kg heavy fixture): plinth disc, slick center HUB (the thrust
    bearing), 16-segment fence ring (75-degree window), 3-segment sill plugging the
    window bottom, and 16 radial roof slabs (95-degree roof gap over the hatch) with
    a center hole for the shaft.
  - platter (dynamic, 1.2 kg): 42 cm disc with a grippy top, an 8-segment slick skirt
    collar that journals around the hub, a steel shaft rising through the roof hole,
    and the green crank bar 31.5 cm up, above the roof.
  - sauce can (RED) / decoy can (WHITE): 5.5 x 10.5 cm cylinders (< 8 cm jaw).
  - basket (dynamic, 0.7 kg): tan open box, 15 cm square interior, 8 cm walls.

Per-episode randomization (readback-verified in smoke): rotunda xy jitter + FREE yaw
(the hatch faces anywhere), platter free yaw, the two cans on random OPPOSITE sides at
random angular distances 65-145 degrees from the hatch (spin direction and amount both
vary), free can yaws, and the basket on a random world-frame ring around the rotunda.
Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
PLINTH_R = 0.280  # plinth disc radius (the rotunda ground plate)
PLINTH_T = 0.020  # plinth thickness (top = fence/sill bottom, no under-gap)
HUB_R = 0.050  # center hub (thrust bearing) radius
HUB_TOP = 0.100  # hub top surface height (platter disc rests here)
DISC_R = 0.210  # platter disc radius
DISC_T = 0.012  # platter disc thickness
BEAR_R = 0.060  # slick bearing pad under the disc center (the thrust contact —
BEAR_T = 0.004  # the grippy disc itself must never touch the slick hub, or the
BEAR_PROT = 0.002  # pair-averaged friction re-creates a ~0.2 N m brake)
PLAT_Z = HUB_TOP + BEAR_PROT + DISC_T / 2  # platter origin rest height = 0.108
PLAT_TOP = PLAT_Z + DISC_T / 2  # platter top surface = 0.114
SKIRT_IN = 0.053  # skirt collar inner apothem (3 mm play around the hub)
SKIRT_T = 0.008  # skirt wall thickness
SKIRT_D = 0.050  # skirt depth below the disc underside
SHAFT_R = 0.012  # center shaft radius
BAR_Z = 0.209  # crank bar center in the platter frame (world 0.315 at rest)
BAR_L = 0.180  # crank bar length
BAR_S = 0.016  # crank bar square section
CAN_R = 0.0275  # can radius (5.5 cm dia < ~8 cm parallel jaw)
CAN_H = 0.105  # can height (top = 0.217 riding the platter)
SLOT_R = 0.140  # riding radius of the cans on the platter
FENCE_IN = 0.240  # fence ring inner radius
FENCE_T = 0.010  # fence thickness
FENCE_Z0 = 0.020  # fence bottom (= plinth top)
FENCE_Z1 = 0.260  # fence top (= roof underside, no side gap)
N_FENCE = 16  # fence segments over (360 - 2*WIN_HALF) degrees
WIN_HALF_DEG = 37.5  # fence window half-angle (75-degree hatch) about local +x
SILL_Z1 = 0.150  # sill top: the hatch's low wall (can must be lifted >= 38 mm)
N_SILL = 3  # sill segments across the window
ROOF_UND = 0.260  # roof underside
ROOF_T = 0.015  # roof slab thickness
ROOF_R0 = 0.030  # roof center-hole radius (shaft passes with 18 mm play)
ROOF_R1 = 0.260  # roof outer radius (overlaps the fence top)
ROOF_W = 0.088  # roof slab tangential width
N_ROOF = 16  # roof slabs over (360 - 2*ROOF_HALF) degrees
ROOF_HALF_DEG = 47.5  # roof gap half-angle (95-degree open sky over the hatch)
BKT_FLOOR = 0.170  # basket floor plate side
BKT_FLOOR_T = 0.012  # basket floor thickness
BKT_WALL_H = 0.080  # basket wall height (above the floor plate)
BKT_WALL_T = 0.008  # basket wall thickness
BKT_IN_HALF = 0.075  # basket interior half-width


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


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


def _wrap(ang: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(ang), torch.cos(ang))


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


def _box(stage, path: str, size, center, yaw_deg: float, color, contact_offset: float,
         material=None) -> None:
    """Author one colliding box child prim (translate -> orient -> scale, authored
    once per prim — the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    half = math.radians(yaw_deg) / 2
    sxf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    """Author one colliding z-axis cylinder child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _body_root(stage, prim_path: str, translation, orientation, mass: float,
               lin_damp: float, ang_damp: float, iters: int = 8, viters: int = 1):
    """Author the dynamic rigid-body root xform shared by all compound spawners."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(iters)
    pxrb.CreateSolverVelocityIterationCountAttr(viters)
    return root


def _spawn_rotunda(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rotunda as ONE heavy DYNAMIC compound body (teleportable at reset).
    Origin = plinth bottom center on the ground; local +x = hatch bearing. Plinth
    disc, slick hub, fence ring with a 75-degree window about +x, a low sill plugging
    the window bottom, and radial roof slabs leaving a 95-degree gap over the hatch
    and a center hole for the platter shaft."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_hub_s, cfg.mu_hub_d)
    slate = (0.35, 0.38, 0.42)
    dark = (0.22, 0.24, 0.28)
    steel = (0.55, 0.57, 0.60)
    _cyl(stage, f"{prim_path}/plinth", PLINTH_R, PLINTH_T,
         (0.0, 0.0, PLINTH_T / 2), slate, 0.0015, material=body)
    _cyl(stage, f"{prim_path}/hub", HUB_R, HUB_TOP - PLINTH_T,
         (0.0, 0.0, PLINTH_T + (HUB_TOP - PLINTH_T) / 2), steel, 0.0015, material=slick)
    # fence ring: N_FENCE tangential chord boxes over [WIN_HALF, 360-WIN_HALF]
    fence_seg = (360.0 - 2 * WIN_HALF_DEG) / N_FENCE
    r_mid = FENCE_IN + FENCE_T / 2
    fence_chord = 2 * r_mid * math.sin(math.radians(fence_seg / 2)) + 0.003
    fence_h = FENCE_Z1 - FENCE_Z0
    for i in range(N_FENCE):
        phi = WIN_HALF_DEG + (i + 0.5) * fence_seg
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        _box(stage, f"{prim_path}/fence_{i}", (FENCE_T, fence_chord, fence_h),
             (r_mid * c, r_mid * s, FENCE_Z0 + fence_h / 2), phi, dark, 0.0015,
             material=body)
    # sill: N_SILL tangential chord boxes plugging the window below SILL_Z1
    sill_seg = 2 * WIN_HALF_DEG / N_SILL
    sill_chord = 2 * r_mid * math.sin(math.radians(sill_seg / 2)) + 0.004
    sill_h = SILL_Z1 - FENCE_Z0
    for i in range(N_SILL):
        phi = -WIN_HALF_DEG + (i + 0.5) * sill_seg
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        _box(stage, f"{prim_path}/sill_{i}", (FENCE_T, sill_chord, sill_h),
             (r_mid * c, r_mid * s, FENCE_Z0 + sill_h / 2), phi, slate, 0.0015,
             material=body)
    # roof: N_ROOF radial slabs over [ROOF_HALF, 360-ROOF_HALF], center hole ROOF_R0
    roof_seg = (360.0 - 2 * ROOF_HALF_DEG) / N_ROOF
    r_len = ROOF_R1 - ROOF_R0
    for i in range(N_ROOF):
        phi = ROOF_HALF_DEG + (i + 0.5) * roof_seg
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        rc = ROOF_R0 + r_len / 2
        _box(stage, f"{prim_path}/roof_{i}", (r_len, ROOF_W, ROOF_T),
             (rc * c, rc * s, ROOF_UND + ROOF_T / 2), phi, dark, 0.0015,
             material=body)
    return root


def _spawn_platter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the free carousel platter: grippy disc, slick 8-segment skirt collar
    (journals around the hub), center shaft, and the green crank bar above the roof.
    Origin = disc center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.1, 0.1, iters=16, viters=4)
    grip = _friction_material(stage, f"{prim_path}/grip_mat", cfg.mu_top_s, cfg.mu_top_d)
    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_hub_s, cfg.mu_hub_d)
    amber = (0.82, 0.62, 0.18)
    steel = (0.55, 0.57, 0.60)
    green = (0.10, 0.65, 0.20)
    _cyl(stage, f"{prim_path}/disc", DISC_R, DISC_T, (0.0, 0.0, 0.0), amber,
         0.0015, material=grip)
    # slick bearing pad: protrudes BEAR_PROT below the disc so the thrust contact
    # on the hub is slick-on-slick (never grippy-disc-on-hub, which PhysX would
    # pair-average into a ~0.2 N m brake)
    _cyl(stage, f"{prim_path}/bearing", BEAR_R, BEAR_T,
         (0.0, 0.0, -DISC_T / 2 - BEAR_PROT + BEAR_T / 2), steel, 0.0012,
         material=slick)
    # skirt collar: 8 chord boxes below the disc, inner apothem SKIRT_IN
    r_sk = SKIRT_IN + SKIRT_T / 2
    sk_chord = 2 * r_sk * math.tan(math.radians(22.5)) + 0.003
    for i in range(8):
        phi = i * 45.0
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        _box(stage, f"{prim_path}/skirt_{i}", (SKIRT_T, sk_chord, SKIRT_D),
             (r_sk * c, r_sk * s, -DISC_T / 2 - SKIRT_D / 2), phi, steel, 0.0012,
             material=slick)
    _cyl(stage, f"{prim_path}/shaft", SHAFT_R, BAR_Z, (0.0, 0.0, BAR_Z / 2), steel,
         0.0012, material=slick)
    _box(stage, f"{prim_path}/bar", (BAR_L, BAR_S, BAR_S), (0.0, 0.0, BAR_Z), 0.0,
         green, 0.0012, material=grip)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the free open basket: floor plate + four walls. Origin = floor bottom
    center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.2, 0.3, iters=8, viters=4)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    tan = (0.62, 0.44, 0.20)
    _box(stage, f"{prim_path}/floor", (BKT_FLOOR, BKT_FLOOR, BKT_FLOOR_T),
         (0.0, 0.0, BKT_FLOOR_T / 2), 0.0, tan, 0.0015, material=body)
    off = BKT_IN_HALF + BKT_WALL_T / 2
    span = 2 * (BKT_IN_HALF + BKT_WALL_T)
    zc = BKT_FLOOR_T + BKT_WALL_H / 2
    _box(stage, f"{prim_path}/wall_n", (span, BKT_WALL_T, BKT_WALL_H),
         (0.0, off, zc), 0.0, tan, 0.0015, material=body)
    _box(stage, f"{prim_path}/wall_s", (span, BKT_WALL_T, BKT_WALL_H),
         (0.0, -off, zc), 0.0, tan, 0.0015, material=body)
    _box(stage, f"{prim_path}/wall_e", (BKT_WALL_T, 2 * BKT_IN_HALF, BKT_WALL_H),
         (off, 0.0, zc), 0.0, tan, 0.0015, material=body)
    _box(stage, f"{prim_path}/wall_w", (BKT_WALL_T, 2 * BKT_IN_HALF, BKT_WALL_H),
         (-off, 0.0, zc), 0.0, tan, 0.0015, material=body)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass
    subclasses of RigidObjectSpawnerCfg, defined lazily so the module imports
    app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rotunda" not in _SPAWNER_CACHE:

        @configclass
        class RotundaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rotunda)
            mu_body_s: float = 0.50
            mu_body_d: float = 0.40
            mu_hub_s: float = 0.04
            mu_hub_d: float = 0.03

        @configclass
        class PlatterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platter)
            mu_top_s: float = 0.70
            mu_top_d: float = 0.60
            mu_hub_s: float = 0.04
            mu_hub_d: float = 0.03

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            mu_body_s: float = 0.50
            mu_body_d: float = 0.40

        _SPAWNER_CACHE.update(rotunda=RotundaSpawnerCfg, platter=PlatterSpawnerCfg,
                              basket=BasketSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SauceCarouselSceneCfg(BaseCfg):
    """Config for `SauceCarouselScene`. The honesty knobs are asserted in
    `__post_init__`: cans ride freely under the roof but cannot be reached outside
    the hatch sector; the hatch passes a can aligned within tolerance with real
    margin (roof-slab corner sweep included); the sill forces a deliberate lift; the
    cans spawn far enough from the hatch that the null policy scores ~0; and any
    settled interior rest of the can inside the basket passes the basket gates while
    rim perches and floor near-misses fail them."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    align_tol_deg: float = tunable(20.0)  # |hatch-frame angle| = "parked in the hatch"
    out_r: float = tunable(0.33)  # rotunda->can xy distance = "extracted"
    bkt_xy_max: float = tunable(0.062)  # basket-frame max(|x|,|y|) containment gate
    bkt_z_win: tuple = tunable((0.030, 0.105))  # basket-frame can-center height band
    plat_rad_win: tuple = tunable((0.045, 0.205))  # riding: rotunda-frame xy radius
    plat_z_win: tuple = tunable((0.130, 0.200))  # riding: can center height band
    upright_max_deg: float = tunable(15.0)  # decoy axis vs world up
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    rot_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the rotunda (m)
    can_th_range: tuple = tunable((65.0, 145.0))  # |hatch angle| of each can (deg),
    #                                               cans on opposite random sides
    basket_r_range: tuple = tunable((0.43, 0.55))  # basket ring radius around rotunda

    # --- info: structure -----------------------------------------------------------------------
    can_r: float = info(CAN_R)
    can_h: float = info(CAN_H)
    can_mass: float = info(0.20)
    platter_mass: float = info(1.2)
    rotunda_mass: float = info(45.0)  # heavy dynamic fixture (teleportable, immovable)
    basket_mass: float = info(0.70)
    slot_r: float = info(SLOT_R)
    sill_top: float = info(SILL_Z1)
    roof_und: float = info(ROOF_UND)
    mu_can_s: float = info(0.50)
    mu_can_d: float = info(0.45)
    mu_ground_s: float = info(0.50)
    mu_ground_d: float = info(0.40)

    def __post_init__(self) -> None:
        can_half = math.degrees(math.asin(CAN_R / SLOT_R))  # can angular halfwidth
        # -- embodiment: the can fits an ~80 mm parallel jaw; the bar is pinchable --
        assert 2 * CAN_R < 0.08, "can must fit an ~80 mm parallel jaw"
        assert BAR_S < 0.08, "crank bar section must fit the jaw"
        # -- the carousel guarantee: cans ride freely under the roof... --
        assert PLAT_TOP + CAN_H + 0.03 <= ROOF_UND, \
            "a riding can must clear the roof underside by >= 30 mm"
        # -- ...but cannot leave except through the hatch: fence meets roof, sill low --
        assert abs(FENCE_Z1 - ROOF_UND) < 1e-9, "fence top must meet the roof underside"
        assert PLAT_TOP + 0.02 <= SILL_Z1 <= PLAT_TOP + 0.06, \
            "sill must force a deliberate 20-60 mm lift over the hatch lip"
        assert ROOF_R0 >= SHAFT_R + 0.012, "shaft must pass the roof hole with play"
        assert BAR_Z + PLAT_Z - BAR_S / 2 > ROOF_UND + ROOF_T + 0.02, \
            "crank bar must ride clear above the roof"
        # -- hatch honesty: an aligned can lifts out without touching the roof edge --
        assert WIN_HALF_DEG >= self.align_tol_deg + can_half + 4.0, \
            "fence window must pass a tolerance-aligned can with margin"
        roof_seg = (360.0 - 2 * ROOF_HALF_DEG) / N_ROOF
        phi1 = ROOF_HALF_DEG + roof_seg / 2  # first roof slab axis bearing
        for k in range(41):
            r = (SLOT_R - CAN_R) + k * (2 * CAN_R / 40)  # can radial extent sweep
            edge = phi1 - math.degrees(math.asin(min(1.0, (ROOF_W / 2) / r)))
            cosb = (r * r + SLOT_R * SLOT_R - CAN_R * CAN_R) / (2 * r * SLOT_R)
            beta = math.degrees(math.acos(max(-1.0, min(1.0, cosb))))
            assert edge >= self.align_tol_deg + beta + 3.0, \
                "roof slab corners must clear the aligned can's lift corridor"
        # -- the platter is caged: no can-sized gap at the rim, cans fully on the disc --
        assert FENCE_IN - DISC_R < 2 * CAN_R - 0.01, \
            "the platter-to-fence moat must be too narrow for a can"
        assert SLOT_R + CAN_R <= DISC_R - 0.03, "riding cans stand fully on the disc"
        assert BAR_L / 2 + 0.02 <= SLOT_R - CAN_R, \
            "the crank bar sweep must stay clear of the can lift corridor"
        # -- free journal: bounded play around the hub --
        assert 0.002 <= SKIRT_IN - HUB_R <= 0.005, "skirt-hub play must be 2-5 mm"
        assert BEAR_R >= HUB_R + 0.005, \
            "the slick bearing pad must overhang the hub so the grippy disc never" \
            " touches it (pair-averaged friction would brake the carousel)"
        assert BEAR_PROT > 0.0 and BEAR_T > BEAR_PROT, \
            "bearing pad must protrude below the disc but stay rooted in it"
        # -- null policy scores ~0: cans spawn far outside the alignment tolerance --
        assert self.can_th_range[0] >= self.align_tol_deg + 25.0, \
            "cans must spawn well outside the hatch alignment tolerance"
        assert self.can_th_range[1] <= 180.0 - can_half
        # -- basket honesty: any settled interior rest passes, rim/floor rests fail --
        assert BKT_IN_HALF - CAN_R <= self.bkt_xy_max, \
            "every interior standing rest must pass the xy gate"
        assert BKT_IN_HALF - CAN_H / 2 <= self.bkt_xy_max, \
            "every interior lying rest must pass the xy gate"
        assert BKT_FLOOR_T + 0.01 < self.bkt_z_win[0] < BKT_FLOOR_T + CAN_R, \
            "z band must accept a lying can and reject a can far below the floor"
        assert self.bkt_z_win[1] < BKT_FLOOR_T + BKT_WALL_H + CAN_R, \
            "z band must reject a can perched on the basket rim"
        assert self.bkt_z_win[1] > BKT_FLOOR_T + CAN_H / 2 + 0.02, \
            "z band must accept a standing can with margin"
        # -- extraction gate consistent with geometry --
        bkt_circ = math.hypot(BKT_FLOOR / 2, BKT_FLOOR / 2)
        assert self.basket_r_range[0] >= PLINTH_R + bkt_circ + 0.02, \
            "the basket ring must stay clear of the rotunda"
        assert self.out_r <= self.basket_r_range[0] - BKT_IN_HALF, \
            "any can inside the basket must already count as extracted"
        assert self.out_r >= FENCE_IN + FENCE_T + CAN_R + 0.03, \
            "extracted must mean fully clear of the rotunda ring"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sauce_carousel")
class SauceCarouselScene(BaseScene):
    cfg: SauceCarouselSceneCfg

    def __init__(self, cfg: SauceCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or SauceCarouselSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        can_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_can_s, dynamic_friction=c.mu_can_d, restitution=0.0)
        can_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.1)
        can_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

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
            "rotunda": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rotunda",
                spawn=spawners["rotunda"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rotunda_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "platter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platter",
                spawn=spawners["platter"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.platter_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PLAT_Z)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=spawners["basket"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.basket_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 0.001)),
            ),
        }
        for name, color, y0 in (("sauce", (0.75, 0.10, 0.08), 0.14),
                                ("decoy", (0.92, 0.92, 0.92), -0.14)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CylinderCfg(
                    radius=CAN_R, height=CAN_H, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=can_mat, rigid_props=can_rigid,
                    collision_props=can_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, y0, PLAT_TOP + CAN_H / 2 + 0.002)),
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
        self.rotunda: RigidObject = env.iscene["rotunda"]
        self.platter: RigidObject = env.iscene["platter"]
        self.sauce: RigidObject = env.iscene["sauce"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.basket: RigidObject = env.iscene["basket"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latched-score state (zeroed / refilled in reset)
        self._prog = torch.zeros(n, device=dev)
        self._aligned = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._th0 = torch.full((n,), math.pi, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + freely yaw the rotunda (the hatch faces anywhere),
        seat the platter on the hub at a free yaw, stand the two cans on random
        OPPOSITE sides of the hatch bearing at random angular distances, and drop the
        basket on a random world-frame ring around the rotunda."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(m, device=dev) * (hi - lo) + lo

        def yaw_free() -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * math.pi

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- rotunda: xy jitter + FREE yaw ---
        rxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.rot_jitter
        q_rot = _qz(yaw_free())
        zcol = torch.zeros(m, 1, device=dev)
        rot_pos = torch.cat([rxy, zcol], dim=-1)
        write(self.rotunda, rot_pos, q_rot)

        # --- platter: seated on the hub, free yaw (0.5 mm settle drop) ---
        p_pos = rot_pos + torch.tensor([0.0, 0.0, PLAT_Z + 0.0005], device=dev)
        write(self.platter, p_pos, _qz(yaw_free()))

        # --- cans: opposite random sides of the hatch bearing (rotunda local +x) ---
        sgn = torch.where(torch.rand(m, device=dev) > 0.5, 1.0, -1.0)
        lo, hi = math.radians(c.can_th_range[0]), math.radians(c.can_th_range[1])
        th_t = sgn * u(lo, hi)
        th_d = -sgn * u(lo, hi)
        for body, th in ((self.sauce, th_t), (self.decoy, th_d)):
            loc = torch.stack([
                SLOT_R * torch.cos(th), SLOT_R * torch.sin(th),
                torch.full((m,), PLAT_TOP + CAN_H / 2 + 0.002, device=dev)], dim=-1)
            write(body, rot_pos + _qapply(q_rot, loc), _qz(yaw_free()))
        self._th0[env_ids] = th_t.abs()

        # --- basket: random world-frame ring around the rotunda ---
        ang = u(0.0, 2 * math.pi)
        rad = u(*c.basket_r_range)
        b_pos = torch.stack([rxy[:, 0] + rad * torch.cos(ang),
                             rxy[:, 1] + rad * torch.sin(ang),
                             torch.full((m,), 0.001, device=dev)], dim=-1)
        write(self.basket, b_pos, _qz(yaw_free()))

        # --- zero the latched-score state ---
        self._prog[env_ids] = 0.0
        self._aligned[env_ids] = False
        self._out[env_ids] = False

    # ----- state (full, restorable — includes the latched score state) ------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        st = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
              for nm in ("rotunda", "platter", "sauce", "decoy", "basket")}
        st["_latch"] = torch.stack([
            self._prog[env_ids], self._aligned[env_ids].float(),
            self._out[env_ids].float(), self._th0[env_ids]], dim=-1).clone()
        return st

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("rotunda", "platter", "sauce", "decoy", "basket"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        if "_latch" in state:
            lt = state["_latch"]
            self._prog[env_ids] = lt[:, 0]
            self._aligned[env_ids] = lt[:, 1] > 0.5
            self._out[env_ids] = lt[:, 2] > 0.5
            self._th0[env_ids] = lt[:, 3]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A round slate-and-dark ROTUNDA stands on the floor: a "
            f"{2 * PLINTH_R * 100:.0f} cm plinth carrying a fence ring and a slab "
            f"roof {ROOF_UND * 100:.0f} cm up that together close most of the "
            f"circumference. One 75-degree HATCH sector is open — no fence, no roof "
            f"overhead — guarded only by a low sill {SILL_Z1 * 100:.0f} cm tall. "
            f"Inside, a free-spinning amber CAROUSEL platter "
            f"({2 * DISC_R * 100:.0f} cm across) rests on a slick center hub; its "
            f"steel shaft rises through a hole in the roof to a horizontal GREEN "
            f"CRANK BAR ({BAR_L * 100:.0f} cm long) turning freely above the roof. "
            f"Two cans ride the platter {SLOT_R * 100:.0f} cm from its center: a RED "
            f"tomato-sauce can (the target) and a WHITE decoy can, both "
            f"{2 * CAN_R * 100:.1f} cm across and {CAN_H * 100:.1f} cm tall. Both "
            f"start deep under the roofed part — at least "
            f"{c.can_th_range[0]:.0f} degrees around from the hatch, on opposite "
            f"sides — where the roof (4 cm above the can tops) and the fence make "
            f"them unreachable. Out on the open floor stands a tan open BASKET "
            f"({2 * BKT_IN_HALF * 100:.0f} cm square interior, "
            f"{BKT_WALL_H * 100:.0f} cm walls). The rotunda's position and the "
            f"direction its hatch faces, the cans' angular positions and sides, and "
            f"the basket's position change every episode — read the scene by "
            f"looking.\n"
            f"Goal: put the RED sauce can into the basket, and leave the WHITE can "
            f"riding the platter. The cans cannot be reached where they start: turn "
            f"the GREEN crank bar (either direction — pick the shorter way) to spin "
            f"the carousel until the RED can is parked in the open hatch sector "
            f"(within {c.align_tol_deg:.0f} degrees of the hatch bearing), stopping "
            f"without overshooting. Then lift the can up over the "
            f"{SILL_Z1 * 100:.0f} cm sill, out through the hatch, carry it clear of "
            f"the rotunda and drop it into the basket so it rests inside, settled. "
            f"Do not pull the WHITE can off the platter — it must still be riding, "
            f"upright, at the end."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the green crank bar to spin the carousel until the red sauce can "
            "is parked in the open hatch sector, then lift the can out over the "
            "sill and drop it into the basket. Leave the white can riding the "
            "platter."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, frame_body, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> the body's frame."""
        return _qapply(_qinv(frame_body.data.root_quat_w),
                       pos_w - frame_body.data.root_pos_w)

    def hatch_angle(self, body) -> torch.Tensor:
        """(N,) the body's bearing in the rotunda frame, 0 = hatch center, (-pi,pi]."""
        loc = self._local(self.rotunda, body.data.root_pos_w)
        return torch.atan2(loc[:, 1], loc[:, 0])

    def on_platter(self, body) -> torch.Tensor:
        """(N,) bool: the body rides the platter (rotunda-frame radius + height)."""
        c = self.cfg
        loc = self._local(self.rotunda, body.data.root_pos_w)
        rad = loc[:, :2].norm(dim=-1)
        return rad.ge(c.plat_rad_win[0]) & rad.le(c.plat_rad_win[1]) \
            & (loc[:, 2] >= c.plat_z_win[0]) & (loc[:, 2] <= c.plat_z_win[1])

    def upright(self, body) -> torch.Tensor:
        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        axis = _qapply(body.data.root_quat_w, ez)
        return axis[:, 2] >= math.cos(math.radians(self.cfg.upright_max_deg))

    def aligned_now(self) -> torch.Tensor:
        """(N,) bool: sauce riding the platter, parked in the hatch tolerance."""
        return self.on_platter(self.sauce) \
            & self.hatch_angle(self.sauce).abs().le(math.radians(self.cfg.align_tol_deg))

    def out_now(self) -> torch.Tensor:
        """(N,) bool: sauce fully clear of the rotunda ring (xy)."""
        d = (self.sauce.data.root_pos_w[:, :2]
             - self.rotunda.data.root_pos_w[:, :2]).norm(dim=-1)
        return d >= self.cfg.out_r

    def in_basket(self) -> torch.Tensor:
        """(N,) bool: sauce center inside the basket volume (walls + height band)."""
        c = self.cfg
        loc = self._local(self.basket, self.sauce.data.root_pos_w)
        return torch.maximum(loc[:, 0].abs(), loc[:, 1].abs()).le(c.bkt_xy_max) \
            & (loc[:, 2] >= c.bkt_z_win[0]) & (loc[:, 2] <= c.bkt_z_win[1])

    def decoy_ok(self) -> torch.Tensor:
        """(N,) bool: decoy still upright and riding the platter."""
        return self.on_platter(self.decoy) & self.upright(self.decoy)

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for nm in ("rotunda", "platter", "sauce", "decoy", "basket"):
            ok &= getattr(self, nm).data.root_state_w.isfinite().all(dim=-1)
        return ok

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ordered latches fired (parked in the hatch, THEN extracted),
        the RED can rests inside the basket with the sauce / basket / platter settled,
        the WHITE decoy still rides the platter upright, all states finite."""
        return self._out & self.in_basket() & self.settled(self.sauce) \
            & self.settled(self.basket) & self.settled(self.platter) \
            & self.decoy_ok() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched and non-decreasing along the demonstrated
        solution: 0.25 * spin progress (running max of the fractional angular
        approach from the spawn bearing toward the hatch, while riding the platter);
        +0.15 parked within the hatch tolerance; +0.20 extracted from the rotunda
        after parking; 1.0 iff success(). The null policy holds ~0 (cans spawn
        >= 65 degrees off the hatch; nothing moves)."""
        c = self.cfg
        tol = math.radians(c.align_tol_deg)
        riding = self.on_platter(self.sauce)
        th = self.hatch_angle(self.sauce).abs()
        prog_now = ((self._th0 - th) / (self._th0 - tol).clamp(min=1e-3)).clamp(0.0, 1.0)
        self._prog = torch.maximum(self._prog, prog_now * riding.float())
        self._aligned |= self.aligned_now()
        self._out |= self._aligned & self.out_now()
        base = 0.25 * self._prog + 0.15 * self._aligned.float() + 0.20 * self._out.float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="sauce_carousel", robot="null", env_spacing=3.0))
