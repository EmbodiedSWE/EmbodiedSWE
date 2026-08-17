"""CaptureArenaScene — capture the black king down the chute, then crown the white king.

Derived from rlbench/setup_chess ("set up the chess board"), but the PLAN is inverted
and re-mechanised. The seed's plan is: 32 independent, unordered pick-and-place moves
that ADD pieces to an open board. Here nothing is added to an empty square: the one
goal square (a gold DAIS) starts OCCUPIED by the opposing black king, and the task is
a CAPTURE followed by a coronation:

  1. CAPTURE — get the black king off the board so it ends up INSIDE an open capture
     BOX standing on the floor beside the platform. The board is fenced by a low rim
     wall on three sides; the only way off is a 16 cm OPENING in the rim (which side
     it is on is randomized per episode) that feeds a descending CAPTURE CHUTE into
     the box. The intended manipulation is a contact push across the board and
     through the opening; gravity does the rest. The captured king may end up in any
     orientation — capture is judged as real containment (settled inside the box,
     below its rim).
  2. ENTHRONE — stand the white king upright, centred on the vacated gold dais, at
     rest. The dais seats exactly one piece: while the black king occupies it, the
     white king physically cannot be seated (base diameter 44 mm vs a 25 mm centring
     tolerance — two bases cannot both be within tolerance), so the capture MUST
     precede the coronation.

Strategic deltas vs the seed: removal-by-ejection through a randomized aperture with
a gravity chute + containment rubric (the seed never removes a piece); an
occupancy-enforced two-stage order (the seed's 32 moves are independent); success on
ONE placement whose target starts blocked (the seed's squares are all free).

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide):
  - platform: KINEMATIC compound — dark slab (0.46 x 0.46 x 0.12 m) with a visual
    4x4 checkerboard (collision comes from the slab), fenced by a 35 mm rim wall;
    both +/-y rims have a central 16 cm gap.
  - blocker: KINEMATIC wall segment that plugs the INACTIVE gap each episode (so
    exactly one opening exists).
  - chute: KINEMATIC compound posed at the ACTIVE gap — a slick 24 deg ramp with
    guard walls descending into an open-top capture box (grippy floor, 9 cm walls)
    standing on the floor.
  - dais: KINEMATIC gold square (0.10 x 0.10 x 6 mm) = the throne square, position
    randomized on the board.
  - black king / white king: DYNAMIC compounds (flanged base, shaft, ball crown),
    base-weighted with authored CoM + diagonal inertia. Black is near-black with a
    crimson crown; white is ivory with a gold crown.

Per-episode randomization (readback-verifiable): platform yaw +/-20 deg + xy jitter,
the ACTIVE gap side (+y/-y; chute and blocker follow), the dais position on the
board, and the white king's start spot along the opposite (walled) side.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.25  l1 — the black king has left the board through the opening (descending in
             the chute frame) (latched)
  0.30  l2 — the black king is contained in the capture box (latched)
  0.20  l3 — with l2 latched, the white king is near the dais (within 10 cm,
             upright) (latched)
  1.0 iff success() — live: black king settled inside the capture box AND white
       king standing upright centred on the gold dais, everything settled, states
       finite. Non-success capped at 0.75.

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


def _friction_material(stage, path: str, mu_s: float, mu_d: float):
    """A physics material prim (custom-spawner colliders otherwise get ~0.5 friction)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _add_box(stage, path: str, *, center, size, color, contact_offset=None,
             material=None, orient=None, collide=True):
    """One box child: translate (+ optional orient) + scale, displayColor, optional
    collider. `collide=False` authors a purely visual prim (checker tiles)."""
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
    if collide:
        _collide(box.GetPrim(), contact_offset, material)
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, contact_offset, material=None):
    """One z-axis cylinder child with authored extent, displayColor, collider."""
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
    _collide(cyl.GetPrim(), contact_offset, material)
    return cyl.GetPrim()


def _add_ball(stage, path: str, *, center, radius, color, contact_offset, material=None):
    from pxr import Gf, UsdGeom

    ball = UsdGeom.Sphere.Define(stage, path)
    ball.CreateRadiusAttr(float(radius))
    r = float(radius)
    ball.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    xf = UsdGeom.Xformable(ball.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    ball.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(ball.GetPrim(), contact_offset, material)
    return ball.GetPrim()


def _spawn_platform(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the board platform: KINEMATIC compound. Local frame: origin at the
    centre of the slab footprint ON THE GROUND, z up. Children: slab, a 4x4 VISUAL
    checkerboard (no colliders — the slab's flat top is the walking surface, so no
    proud collider edge can ever park a slide), rim walls on all four sides with a
    central gap in BOTH +/-y rims (one gets plugged per episode)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    c = cfg
    co = c.contact_offset
    top = c.board_h
    half = c.board_xy / 2
    t = c.rim_t
    rim_zc = top + c.rim_h / 2
    gap_half = c.gap_w / 2

    slab_c = (0.28, 0.20, 0.14)
    rim_c = (0.55, 0.55, 0.60)
    tile_a = (0.90, 0.86, 0.74)
    tile_b = (0.22, 0.42, 0.28)

    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, top / 2), size=(c.board_xy, c.board_xy, top),
             color=slab_c, contact_offset=co, material=mat)
    # visual checkerboard (no colliders)
    tile = (c.board_xy - 2 * t) / 4
    for i in range(4):
        for j in range(4):
            _add_box(stage, f"{prim_path}/tile_{i}_{j}",
                     center=((i - 1.5) * tile, (j - 1.5) * tile, top + 0.0004),
                     size=(tile - 0.003, tile - 0.003, 0.0008),
                     color=tile_a if (i + j) % 2 == 0 else tile_b, collide=False)
    # rim walls: +/-x full; +/-y split by a central gap
    for tag, sx in (("p", 1.0), ("n", -1.0)):
        _add_box(stage, f"{prim_path}/rimx_{tag}",
                 center=(sx * (half - t / 2), 0.0, rim_zc),
                 size=(t, c.board_xy, c.rim_h), color=rim_c,
                 contact_offset=co, material=mat)
    for tag, sy in (("p", 1.0), ("n", -1.0)):
        for seg, sxx in (("a", 1.0), ("b", -1.0)):
            x0, x1 = gap_half, half
            _add_box(stage, f"{prim_path}/rimy_{tag}_{seg}",
                     center=(sxx * (x0 + x1) / 2, sy * (half - t / 2), rim_zc),
                     size=(x1 - x0, t, c.rim_h), color=rim_c,
                     contact_offset=co, material=mat)
    return root


def _spawn_chute(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the capture chute: KINEMATIC compound. Local frame: origin at the
    active gap's centre AT BOARD-TOP HEIGHT, +y pointing OUTWARD (away from the
    board), z up. Children: a slick inclined ramp (entry 4 mm BELOW board top so no
    proud edge can wall the push), two guard walls, and the capture box (grippy
    floor + near/far/side walls) standing on the ground."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    c = cfg
    co = c.contact_offset
    mat_wall = _friction_material(stage, f"{prim_path}/mat_wall", 0.30, 0.25)
    mat_ramp = _friction_material(stage, f"{prim_path}/mat_ramp", c.ramp_mu, c.ramp_mu * 0.9)
    mat_floor = _friction_material(stage, f"{prim_path}/mat_floor", c.floor_mu, c.floor_mu * 0.9)

    th = math.radians(c.ramp_deg)
    sin_t, cos_t, tan_t = math.sin(th), math.cos(th), math.tan(th)
    # ramp top surface: from (y=0.002, z=-entry_drop) descending over run `ramp_run`
    y0, z0 = 0.002, -c.ramp_entry_drop
    y1 = y0 + c.ramp_run
    z1 = z0 - c.ramp_run * tan_t
    length = c.ramp_run / cos_t
    my, mz = (y0 + y1) / 2, (z0 + z1) / 2
    # box centre = surface midpoint - normal * thickness/2 ; orient = rot about x by -theta
    cy = my - sin_t * c.ramp_t / 2
    cz = mz - cos_t * c.ramp_t / 2
    hh = math.cos(th / 2)
    hs = -math.sin(th / 2)
    ramp_c = (0.62, 0.62, 0.66)
    guard_c = (0.55, 0.55, 0.60)
    box_c = (0.45, 0.30, 0.16)

    _add_box(stage, f"{prim_path}/ramp",
             center=(0.0, cy, cz), size=(c.ramp_w, length, c.ramp_t),
             color=ramp_c, contact_offset=co, material=mat_ramp,
             orient=(hh, hs, 0.0, 0.0))
    for tag, sx in (("p", 1.0), ("n", -1.0)):
        _add_box(stage, f"{prim_path}/guard_{tag}",
                 center=(sx * (c.ramp_w / 2 + 0.006), 0.09, -0.045),
                 size=(0.012, 0.20, 0.13), color=guard_c,
                 contact_offset=co, material=mat_wall)
    # capture box on the ground (ground is at local z = -board_h)
    zg = -c.board_h
    floor_top = zg + 0.016
    _add_box(stage, f"{prim_path}/bin_floor",
             center=(0.0, 0.28, zg + 0.008), size=(0.27, 0.264, 0.016),
             color=box_c, contact_offset=co, material=mat_floor)
    wall_top = c.bin_wall_top          # local z of the box wall rims
    wh = wall_top - floor_top
    _add_box(stage, f"{prim_path}/bin_far",
             center=(0.0, 0.406, floor_top + wh / 2), size=(0.27, 0.012, wh),
             color=box_c, contact_offset=co, material=mat_wall)
    for tag, sx in (("p", 1.0), ("n", -1.0)):
        _add_box(stage, f"{prim_path}/bin_side_{tag}",
                 center=(sx * 0.121, 0.28, floor_top + wh / 2), size=(0.012, 0.264, wh),
                 color=box_c, contact_offset=co, material=mat_wall)
    # near wall stays BELOW the ramp underside (7 mm clearance) so nothing escapes
    # back out underneath the ramp exit
    _add_box(stage, f"{prim_path}/bin_near",
             center=(0.0, 0.154, floor_top + 0.006), size=(0.27, 0.012, 0.012),
             color=box_c, contact_offset=co, material=mat_wall)
    return root


def _spawn_piece(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one chess king: DYNAMIC compound rigid body. Local frame: origin at
    the CENTRE of the base bottom face. Children: base cylinder, shaft cylinder,
    ball crown. Mass, CoM (base-weighted) and a diagonal inertia are authored
    EXPLICITLY (MassAPI mass alone leaves the CoM at the body origin)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.20)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)  # kills GPU phantom-creep artifacts
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    co = cfg.contact_offset
    _add_cyl(stage, f"{prim_path}/base",
             center=(0.0, 0.0, cfg.base_h / 2), radius=cfg.base_r,
             height=cfg.base_h, color=cfg.color, contact_offset=co, material=mat)
    _add_cyl(stage, f"{prim_path}/shaft",
             center=(0.0, 0.0, cfg.base_h + cfg.shaft_h / 2), radius=cfg.shaft_r,
             height=cfg.shaft_h, color=cfg.color, contact_offset=co, material=mat)
    _add_ball(stage, f"{prim_path}/crown",
              center=(0.0, 0.0, cfg.base_h + cfg.shaft_h + cfg.crown_r * 0.8),
              radius=cfg.crown_r, color=cfg.accent, contact_offset=co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "platform" not in _SPAWNER_CACHE:

        @configclass
        class PlatformSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platform)
            board_xy: float = 0.46
            board_h: float = 0.12
            rim_t: float = 0.012
            rim_h: float = 0.035
            gap_w: float = 0.16
            mu_s: float = 0.35
            mu_d: float = 0.30
            contact_offset: float = 0.002

        @configclass
        class ChuteSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chute)
            board_h: float = 0.12
            ramp_deg: float = 24.0
            ramp_run: float = 0.17
            ramp_w: float = 0.15
            ramp_t: float = 0.012
            ramp_entry_drop: float = 0.004
            ramp_mu: float = 0.12
            floor_mu: float = 0.65
            bin_wall_top: float = -0.015
            contact_offset: float = 0.002

        @configclass
        class PieceSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_piece)
            base_r: float = 0.022
            base_h: float = 0.014
            shaft_r: float = 0.012
            shaft_h: float = 0.075
            crown_r: float = 0.017
            mass: float = 0.15
            com_z: float = 0.012
            inertia: tuple = (1.6e-4, 1.6e-4, 4.0e-5)
            color: tuple = (0.93, 0.91, 0.82)
            accent: tuple = (0.85, 0.68, 0.15)
            mu_s: float = 0.30
            mu_d: float = 0.25
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(platform=PlatformSpawnerCfg, chute=ChuteSpawnerCfg,
                              piece=PieceSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CaptureArenaSceneCfg(BaseCfg):
    """Config for `CaptureArenaScene`. The tolerances are honest by construction:
    the capture-box interior (23 x 23 cm, walls 8.9 cm above its floor) is judged
    with the containment sill 3 cm BELOW the wall rims, and the 25 mm dais centring
    tolerance keeps a fully-supported base (base r 22 mm, dais half-width 50 mm) —
    while making double occupancy geometrically impossible (two 44 mm bases cannot
    both centre within 25 mm)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    cell_tol: float = tunable(0.025)       # white base centre within this of the dais centre (m)
    upright_max_deg: float = tunable(12.0)  # white axis within this of world-up when enthroned
    settle_speed: float = tunable(0.04)    # max |lin vel| of both pieces when judging (m/s)
    capture_z_max: float = tunable(-0.045)  # chute-frame root z below this = below the box rim
    approach_r: float = tunable(0.10)      # white within this of the dais = "approach" latch

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    plat_yaw_deg: float = tunable(20.0)    # platform yaw about nominal (+/- deg)
    plat_jitter: float = tunable(0.04)     # platform xy jitter (+/- m)
    throne_x_range: tuple = tunable((-0.06, 0.06))  # dais centre x (board frame)
    throne_y_range: tuple = tunable((-0.05, 0.05))  # dais centre y (board frame)
    white_x_range: tuple = tunable((-0.13, 0.13))   # white start x (board frame)
    white_y_abs: float = tunable(0.17)     # white start |y| — always on the WALLED side

    # --- info: layout (board frame; origin at slab footprint centre on the ground) --------------
    plat_pos: tuple = info((0.0, 0.0))
    board_xy: float = info(0.46)
    board_h: float = info(0.12)            # slab top = playing surface height
    rim_t: float = info(0.012)
    rim_h: float = info(0.035)             # rim wall height above the top
    gap_w: float = info(0.16)              # opening width in the +/-y rims
    dais_s: float = info(0.10)             # dais square side
    dais_h: float = info(0.006)            # dais thickness (top at board_h + dais_h)
    # chute (chute frame: origin at gap centre at board-top height, +y outward)
    ramp_deg: float = info(24.0)
    ramp_run: float = info(0.17)
    bin_x_half: float = info(0.115)        # capture-box interior half-width
    bin_y0: float = info(0.165)            # capture-box interior near/far bounds
    bin_y1: float = info(0.398)
    bin_wall_top: float = info(-0.015)     # box wall rims (chute-frame z)
    # pieces
    base_r: float = info(0.022)
    base_h: float = info(0.014)
    shaft_r: float = info(0.012)
    shaft_h: float = info(0.075)
    crown_r: float = info(0.017)
    piece_mass: float = info(0.15)
    black_color: tuple = info((0.09, 0.09, 0.11))
    black_accent: tuple = info((0.55, 0.10, 0.10))
    white_color: tuple = info((0.93, 0.91, 0.82))
    white_accent: tuple = info((0.85, 0.68, 0.15))
    dais_color: tuple = info((0.85, 0.68, 0.15))
    # rubric weights (0.25 + 0.30 + 0.20 = 0.75 = the non-success cap)
    w1: float = info(0.25)
    w2: float = info(0.30)
    w3: float = info(0.20)
    contact_offset: float = info(0.002)


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


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("capture_arena")
class CaptureArenaScene(BaseScene):
    cfg: CaptureArenaSceneCfg

    def __init__(self, cfg: CaptureArenaSceneCfg | None = None) -> None:
        super().__init__(cfg or CaptureArenaSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        plat_spawn = cls["platform"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            board_xy=c.board_xy, board_h=c.board_h, rim_t=c.rim_t, rim_h=c.rim_h,
            gap_w=c.gap_w, contact_offset=c.contact_offset)
        chute_spawn = cls["chute"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            board_h=c.board_h, ramp_deg=c.ramp_deg, ramp_run=c.ramp_run,
            bin_wall_top=c.bin_wall_top, contact_offset=c.contact_offset)
        black_spawn = cls["piece"](
            base_r=c.base_r, base_h=c.base_h, shaft_r=c.shaft_r, shaft_h=c.shaft_h,
            crown_r=c.crown_r, mass=c.piece_mass, color=c.black_color,
            accent=c.black_accent, contact_offset=c.contact_offset)
        white_spawn = cls["piece"](
            base_r=c.base_r, base_h=c.base_h, shaft_r=c.shaft_r, shaft_h=c.shaft_h,
            crown_r=c.crown_r, mass=c.piece_mass, color=c.white_color,
            accent=c.white_accent, contact_offset=c.contact_offset)

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
            "platform": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platform",
                spawn=plat_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plat_pos[0], c.plat_pos[1], 0.0)),
            ),
            "chute": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chute",
                spawn=chute_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.12)),
            ),
            "blocker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Blocker",
                spawn=sim_utils.CuboidCfg(
                    size=(0.17, c.rim_t, c.rim_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.55, 0.55, 0.60)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.6, 0.2)),
            ),
            "dais": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dais",
                spawn=sim_utils.CuboidCfg(
                    size=(c.dais_s, c.dais_s, c.dais_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.dais_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
            ),
            "black": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlackKing",
                spawn=black_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, 0.6, 0.02)),
            ),
            "white": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WhiteKing",
                spawn=white_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.6, 0.02)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.platform: RigidObject = env.iscene["platform"]
        self.chute: RigidObject = env.iscene["chute"]
        self.blocker: RigidObject = env.iscene["blocker"]
        self.dais: RigidObject = env.iscene["dais"]
        self.black: RigidObject = env.iscene["black"]
        self.white: RigidObject = env.iscene["white"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.side = torch.ones(n, device=dev)        # active gap side: +1 or -1 (board +/-y)
        self.throne = torch.zeros(n, 2, device=dev)  # dais centre (board frame xy)
        # latches (partial credit survives transients; success is judged live)
        self._l1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l3 = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the platform (yaw + xy jitter), draw the active gap
        side (chute at that gap, blocker plugging the mirror gap), sample the dais
        position and stand the BLACK king on it (the throne starts occupied), stand
        the WHITE king near the opposite walled side, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 4, device=dev)  # burn draws (first post-seed draws degenerate)

        # --- platform: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.plat_yaw_deg)
        q = _qz(psi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plat_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.plat_jitter
        st[:, 1] = c.plat_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.plat_jitter
        st[:, 3:7] = q
        pxy = st[:, 0:2].clone()
        st[:, 0:3] += origin
        self.platform.write_root_state_to_sim(st, env_ids)

        # --- active gap side ---
        s = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        self.side[env_ids] = s

        from isaaclab.utils.math import quat_apply

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            w = torch.zeros(m, 3, device=dev)
            w[:, 0:2] = pxy
            return w + quat_apply(q, loc) + origin

        def write_kin(body, loc: torch.Tensor, quat: torch.Tensor) -> None:
            stk = torch.zeros(m, 13, device=dev)
            stk[:, 0:3] = to_world(loc)
            stk[:, 3:7] = quat
            body.write_root_state_to_sim(stk, env_ids)

        half = c.board_xy / 2
        # --- chute at the ACTIVE gap (+y outward; rotated pi when on the -y side) ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 1] = s * half
        loc[:, 2] = c.board_h
        yaw_off = torch.where(s > 0, torch.zeros(m, device=dev),
                              torch.full((m,), math.pi, device=dev))
        write_kin(self.chute, loc, _qmul(q, _qz(yaw_off)))
        # --- blocker plugs the INACTIVE gap ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 1] = -s * (half - c.rim_t / 2)
        loc[:, 2] = c.board_h + c.rim_h / 2
        write_kin(self.blocker, loc, q)

        # --- dais (throne square), position sampled on the board ---
        tx = c.throne_x_range[0] + torch.rand(m, device=dev) \
            * (c.throne_x_range[1] - c.throne_x_range[0])
        ty = c.throne_y_range[0] + torch.rand(m, device=dev) \
            * (c.throne_y_range[1] - c.throne_y_range[0])
        self.throne[env_ids, 0] = tx
        self.throne[env_ids, 1] = ty
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = tx
        loc[:, 1] = ty
        loc[:, 2] = c.board_h + c.dais_h / 2
        write_kin(self.dais, loc, q)

        # --- black king ON the dais (occupied throne); white king on the walled side ---
        yaw_amp = math.pi
        for body, bx, by, bz in (
            (self.black, tx + (torch.rand(m, device=dev) * 2 - 1) * 0.004,
             ty + (torch.rand(m, device=dev) * 2 - 1) * 0.004,
             c.board_h + c.dais_h + 0.0015),
            (self.white,
             c.white_x_range[0] + torch.rand(m, device=dev)
             * (c.white_x_range[1] - c.white_x_range[0]),
             -s * c.white_y_abs, c.board_h + 0.0015),
        ):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = bx
            loc[:, 1] = by
            loc[:, 2] = bz
            stp = torch.zeros(m, 13, device=dev)
            stp[:, 0:3] = to_world(loc)
            stp[:, 3:7] = _qmul(q, _qz((torch.rand(m, device=dev) * 2 - 1) * yaw_amp))
            body.write_root_state_to_sim(stp, env_ids)

        # --- clear latches ---
        self._l1[env_ids] = False
        self._l2[env_ids] = False
        self._l3[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "platform": self.platform.data.root_state_w[env_ids].clone(),
            "chute": self.chute.data.root_state_w[env_ids].clone(),
            "blocker": self.blocker.data.root_state_w[env_ids].clone(),
            "dais": self.dais.data.root_state_w[env_ids].clone(),
            "black": self.black.data.root_state_w[env_ids].clone(),
            "white": self.white.data.root_state_w[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "throne": self.throne[env_ids].clone(),
            "l1": self._l1[env_ids].clone(),
            "l2": self._l2[env_ids].clone(),
            "l3": self._l3[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.platform.write_root_state_to_sim(state["platform"], env_ids)
        self.chute.write_root_state_to_sim(state["chute"], env_ids)
        self.blocker.write_root_state_to_sim(state["blocker"], env_ids)
        self.dais.write_root_state_to_sim(state["dais"], env_ids)
        self.black.write_root_state_to_sim(state["black"], env_ids)
        self.white.write_root_state_to_sim(state["white"], env_ids)
        self.side[env_ids] = state["side"]
        self.throne[env_ids] = state["throne"]
        self._l1[env_ids] = state["l1"]
        self._l2[env_ids] = state["l2"]
        self._l3[env_ids] = state["l3"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A square dark-wood chess PLATFORM ({c.board_xy * 100:.0f} x "
            f"{c.board_xy * 100:.0f} cm, {c.board_h * 100:.0f} cm tall) stands on the "
            f"floor, its top painted as a pale-and-green checkerboard and fenced by a low "
            f"grey RIM WALL ({c.rim_h * 1000:.0f} mm tall). Three sides are fully walled; "
            f"on exactly ONE side the rim has a {c.gap_w * 100:.0f} cm OPENING that leads "
            f"onto a descending grey CAPTURE CHUTE ending inside an open-top CAPTURE BOX "
            f"standing on the floor beside the platform. Which side has the opening, the "
            f"platform's position and heading, the gold square's spot and the pieces' "
            f"start spots change every episode.\n"
            f"A square GOLD DAIS ({c.dais_s * 100:.0f} cm wide, {c.dais_h * 1000:.0f} mm "
            f"thick — the throne square) lies near the middle of the board. The BLACK "
            f"KING — the near-black piece with a CRIMSON ball crown — starts standing ON "
            f"the gold dais. The WHITE KING — the ivory piece with a GOLD ball crown — "
            f"waits on the board near the walled side opposite the opening.\n"
            f"Goal — a capture, then a coronation, in that order: (1) CAPTURE the black "
            f"king: remove it from the board so it ends up settled INSIDE the capture "
            f"box (any orientation counts; the rim walls retain pieces everywhere except "
            f"the opening, so push it through the opening and let it slide down the "
            f"chute into the box — a piece left on the board, on the floor, or anywhere "
            f"outside the box does not count). (2) ENTHRONE the white king: stand it "
            f"upright (within ~{c.upright_max_deg:.0f} deg) with its base centred on the "
            f"gold dais (within ~{c.cell_tol * 100:.1f} cm), at rest. The dais seats "
            f"only one piece, so the black king must be gone before the white king can "
            f"take the throne. Success: black king inside the capture box AND white "
            f"king standing centred on the gold dais, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Capture the black king: push it across the board and out through the rim "
            "opening so it falls down the chute into the capture box beside the "
            "platform. Then stand the white king upright, centred on the gold dais. "
            "The black king must end inside the box and the white king must stand "
            "on the gold square; a captured king left outside the box fails."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local_of(self, body, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body.data.root_quat_w, pos_w - body.data.root_pos_w)

    def board_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the platform's local frame (origin at footprint
        centre on the ground)."""
        return self._local_of(self.platform, pos_w)

    def chute_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the chute's local frame (origin at the active gap
        centre at board-top height, +y outward)."""
        return self._local_of(self.chute, pos_w)

    def board_to_world(self, loc: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        return self.platform.data.root_pos_w + quat_apply(self.platform.data.root_quat_w, loc)

    def upright(self, body) -> torch.Tensor:
        """(N,) bool: piece axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def on_board(self, body) -> torch.Tensor:
        """(N,) bool: root over the playing surface at playing height."""
        c = self.cfg
        loc = self.board_local(body.data.root_pos_w)
        lim = c.board_xy / 2 - c.rim_t
        return (loc[:, 0].abs() < lim) & (loc[:, 1].abs() < lim) \
            & (loc[:, 2] > c.board_h - 0.005) & (loc[:, 2] < c.board_h + 0.030)

    def black_ejected(self) -> torch.Tensor:
        """(N,) bool: the black king is past the gap line and DESCENDING in the
        chute (chute-frame y beyond the board edge, z below board top)."""
        loc = self.chute_local(self.black.data.root_pos_w)
        return (loc[:, 1] > 0.03) & (loc[:, 2] < -0.015)

    def black_captured(self) -> torch.Tensor:
        """(N,) bool, geometric containment: black king root inside the capture-box
        interior, BELOW the containment sill (3 cm under the box wall rims) — a
        piece on the ramp, on a wall rim, or on the floor outside cannot pass."""
        c = self.cfg
        loc = self.chute_local(self.black.data.root_pos_w)
        return (loc[:, 0].abs() < c.bin_x_half) & (loc[:, 1] > c.bin_y0) \
            & (loc[:, 1] < c.bin_y1) & (loc[:, 2] < c.capture_z_max)

    def white_enthroned(self) -> torch.Tensor:
        """(N,) bool: white king standing upright with its base centred on the dais
        and RESTING ON the dais top (z window rejects hovering / on-board-near-dais
        / lying poses)."""
        c = self.cfg
        loc = self.board_local(self.white.data.root_pos_w)
        near = (loc[:, :2] - self.throne).norm(dim=-1) < c.cell_tol
        z_top = c.board_h + c.dais_h
        seated = (loc[:, 2] > z_top - 0.005) & (loc[:, 2] < z_top + 0.010)
        return near & seated & self.upright(self.white)

    def settled(self) -> torch.Tensor:
        """(N,) bool: both kings slower than `settle_speed`."""
        c = self.cfg
        bv = self.black.data.root_lin_vel_w.norm(dim=-1)
        wv = self.white.data.root_lin_vel_w.norm(dim=-1)
        return (bv < c.settle_speed) & (wv < c.settle_speed)

    def _update_latches(self) -> None:
        c = self.cfg
        self._l1 |= self.black_ejected()
        self._l2 |= self.black_captured() \
            & (self.black.data.root_lin_vel_w.norm(dim=-1) < 0.5)
        wl = self.board_local(self.white.data.root_pos_w)
        self._l3 |= self._l2 & ((wl[:, :2] - self.throne).norm(dim=-1) < c.approach_r) \
            & self.upright(self.white)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: black king contained in the capture box AND white king
        standing upright centred on the gold dais, both settled, states finite.
        All clauses are physical outcomes (real containment, real seated rest)."""
        self._update_latches()
        finite = torch.isfinite(self.black.data.root_state_w).all(dim=-1) \
            & torch.isfinite(self.white.data.root_state_w).all(dim=-1)
        return self.black_captured() & self.white_enthroned() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*ejected + 0.30*captured + 0.20*approach (all
        latched; ~0 for doing nothing), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w1 * self._l1.float() + c.w2 * self._l2.float()
                + c.w3 * self._l3.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="capture_arena", robot="null"))
