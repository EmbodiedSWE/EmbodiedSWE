"""CastlingGalleryScene — garage the rook, then slide the king home ("castling").

Derived from rlbench/setup_chess ("set up the chess board"), but the MANIPULATION
MODEL is replaced wholesale. The seed's plan is: 32 independent, unordered
pick-and-place moves — lift an exposed piece off the table, carry it through free
space, lay it down on a marked square of an open horizontal board. Here NO piece can
be picked up at all: both pieces have flanged bases captured UNDER overhanging lip
rails of a channel network milled into a plinth (the "castling gallery"). The only
possible manipulation is SLIDING along the channels — every centimetre of every
placement is contact-guided translation, and the final poses are produced by pushing
the pieces to their stops, never by placing them. The gallery is exactly one piece
wide, so the rook (which starts between the king and the king's home cell) makes the
goal UNREACHABLE for the king until the rook is first slid aside into a side POCKET
— a strict, geometry-enforced execution order the seed has no analogue of. Which of
the two mirrored pockets is the correct garage is randomized and marked only by a
green beacon tile: the other pocket is a decoy, and a rook parked there fails.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - plinth: KINEMATIC slate block (0.60 x 0.44 x 0.12 m). On its top, a channel
    network built from wall boxes (16 mm tall, 64 mm inner gap — the flange level)
    topped by overhanging LIP rails (z 16..30 mm, 38 mm inner slot — the shaft
    level). Main GALLERY along local +x (interior x -0.24..+0.24); at x_j = -0.02 a
    junction opens into two mirrored POCKETS along +/-y (interior out to |y|=0.16).
    The +x gallery end wall is GOLD and taller: the king's home cell lies against
    it. A flange (56 mm dia, 12 mm tall) slides under the lips with 4 mm of head
    room and 9 mm of overlap per side: the pieces physically cannot leave the
    channels — even at the junction, whose cross-shaped lip opening inscribes only
    a 53.7 mm circle (< the 56 mm flange).
  - king: DYNAMIC compound — flanged base, tall ivory shaft (30 mm dia, 110 mm),
    gold ball crown. Base-weighted (authored CoM in the flange) like a real
    weighted chess piece.
  - rook: DYNAMIC compound — flanged base, short dark-red shaft (60 mm), wide flat
    turret drum on top.
  - beacon: a small KINEMATIC green tile placed each episode on the plinth just
    beyond the end wall of the TARGET pocket.

Per-episode randomization (readback-verifiable): plinth yaw +/-25 deg + xy jitter,
king and rook start positions along the gallery (king always nearer the -x end,
rook always between king and home), and the target-pocket SIDE (+y or -y, beacon
readback).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve
trajectory):
  0.20  l1 — the rook has entered the TARGET pocket (flange fully clear of the
             gallery, on the beacon side) (latched)
  0.25  l2 — the rook is seated at the target pocket's end cell (latched)
  0.30  l3 — with l2 latched, the king has passed the junction toward home
             (latched)
  1.0 iff success() — live: rook at the target pocket end cell, king at the gold
       home cell, both upright, both still captive in the channels, everything
       settled. Non-success capped at 0.75.

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


def _add_box(stage, path: str, *, center, size, color, contact_offset, material=None):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
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


def _spawn_plinth(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the plinth at `prim_path`: KINEMATIC compound. Local frame: origin at
    the centre of the block's footprint ON THE GROUND, +x along the gallery toward
    the gold home end, z up. The channel network sits on the block top (z = top).

    Children: block, gallery/pocket wall boxes (flange level), lip rail boxes
    (shaft level, overhanging the flanges), and three end walls (+x GOLD)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    c = cfg
    co = c.contact_offset
    top = c.plinth_h
    a = c.chan_gap / 2         # wall inner half-gap (flange level)
    b = c.slot_gap / 2         # lip inner half-gap (shaft level)
    t = c.wall_t
    xj = c.junction_x
    gx = c.gallery_half        # gallery interior half-length
    ye = c.pocket_end_y        # pocket end wall inner face |y|
    wall_zc = top + c.wall_h / 2
    lip_zc = top + c.wall_h + c.lip_h / 2
    lip_w = (a + t) - b        # lip strip width (inner edge b .. outer a+t)
    lip_yc = b + lip_w / 2

    slate = (0.34, 0.35, 0.38)
    wallc = (0.52, 0.53, 0.56)
    lipc = (0.44, 0.45, 0.50)
    gold = (0.85, 0.68, 0.15)

    # the block
    _add_box(stage, f"{prim_path}/block",
             center=(0.0, 0.0, top / 2), size=(c.plinth_x, c.plinth_y, top),
             color=slate, contact_offset=co, material=mat)

    # ---- flange-level walls (z top..top+wall_h) --------------------------------------------
    # gallery side walls, split at the pocket openings (|x - xj| < a)
    segs = [(-gx - t, xj - a), (xj + a, gx + t)]
    for si, (x0, x1) in enumerate(segs):
        for tag, sy in (("p", 1.0), ("n", -1.0)):
            _add_box(stage, f"{prim_path}/gwall_{si}_{tag}",
                     center=((x0 + x1) / 2, sy * (a + t / 2), wall_zc),
                     size=(x1 - x0, t, c.wall_h), color=wallc,
                     contact_offset=co, material=mat)
    # pocket side walls (both pockets, both sides), y from a to ye + t
    for ptag, sp in (("p", 1.0), ("n", -1.0)):
        for tag, sx in (("p", 1.0), ("n", -1.0)):
            _add_box(stage, f"{prim_path}/pwall_{ptag}_{tag}",
                     center=(xj + sx * (a + t / 2), sp * (a + ye + t) / 2, wall_zc),
                     size=(t, ye + t - a, c.wall_h), color=wallc,
                     contact_offset=co, material=mat)
    # ---- lip rails (z top+wall_h .. +lip_h), overhanging the flange ------------------------
    lsegs = [(-gx - t, xj - b), (xj + b, gx + t)]
    for si, (x0, x1) in enumerate(lsegs):
        for tag, sy in (("p", 1.0), ("n", -1.0)):
            _add_box(stage, f"{prim_path}/glip_{si}_{tag}",
                     center=((x0 + x1) / 2, sy * lip_yc, lip_zc),
                     size=(x1 - x0, lip_w, c.lip_h), color=lipc,
                     contact_offset=co, material=mat)
    for ptag, sp in (("p", 1.0), ("n", -1.0)):
        for tag, sx in (("p", 1.0), ("n", -1.0)):
            _add_box(stage, f"{prim_path}/plip_{ptag}_{tag}",
                     center=(xj + sx * lip_yc, sp * (b + ye + t) / 2, lip_zc),
                     size=(lip_w, ye + t - b, c.lip_h), color=lipc,
                     contact_offset=co, material=mat)
    # ---- end walls -------------------------------------------------------------------------
    # start end (-x), plain
    _add_box(stage, f"{prim_path}/end_start",
             center=(-gx - t / 2, 0.0, top + 0.015),
             size=(t, 2 * (a + t), 0.030), color=wallc, contact_offset=co, material=mat)
    # HOME end (+x): GOLD and taller — the landmark of the king's home cell
    _add_box(stage, f"{prim_path}/end_home",
             center=(gx + t / 2, 0.0, top + 0.025),
             size=(t, 2 * (a + t), 0.050), color=gold, contact_offset=co, material=mat)
    # pocket end walls
    for ptag, sp in (("p", 1.0), ("n", -1.0)):
        _add_box(stage, f"{prim_path}/end_pocket_{ptag}",
                 center=(xj, sp * (ye + t / 2), top + 0.015),
                 size=(2 * (a + t), t, 0.030), color=wallc, contact_offset=co, material=mat)
    return root


def _spawn_piece(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one chess piece: DYNAMIC compound rigid body. Local frame: origin at
    the CENTRE of the flange bottom face (so root z ~ channel-floor height when
    seated). Children: flange cylinder, shaft cylinder, and either a ball crown
    (king) or a turret drum (rook). Mass, CoM (base-weighted) and a diagonal
    inertia are authored EXPLICITLY (MassAPI mass alone leaves the CoM at the body
    origin, and PhysX would otherwise guess the inertia from the hulls)."""
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
    _add_cyl(stage, f"{prim_path}/flange",
             center=(0.0, 0.0, cfg.flange_h / 2), radius=cfg.flange_r,
             height=cfg.flange_h, color=cfg.color, contact_offset=co, material=mat)
    _add_cyl(stage, f"{prim_path}/shaft",
             center=(0.0, 0.0, cfg.flange_h + cfg.shaft_h / 2), radius=cfg.shaft_r,
             height=cfg.shaft_h, color=cfg.color, contact_offset=co, material=mat)
    z_top = cfg.flange_h + cfg.shaft_h
    if cfg.top == "ball":  # king: gold ball crown
        _add_ball(stage, f"{prim_path}/crown",
                  center=(0.0, 0.0, z_top + cfg.top_r * 0.8), radius=cfg.top_r,
                  color=cfg.accent, contact_offset=co, material=mat)
    else:  # rook: wide flat turret drum
        _add_cyl(stage, f"{prim_path}/turret",
                 center=(0.0, 0.0, z_top + cfg.top_h / 2), radius=cfg.top_r,
                 height=cfg.top_h, color=cfg.accent, contact_offset=co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plinth" not in _SPAWNER_CACHE:

        @configclass
        class PlinthSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plinth)
            plinth_x: float = 0.60
            plinth_y: float = 0.44
            plinth_h: float = 0.12
            chan_gap: float = 0.064
            slot_gap: float = 0.038
            wall_h: float = 0.016
            lip_h: float = 0.014
            wall_t: float = 0.012
            junction_x: float = -0.02
            gallery_half: float = 0.24
            pocket_end_y: float = 0.16
            mu_s: float = 0.25
            mu_d: float = 0.20
            contact_offset: float = 0.002

        @configclass
        class PieceSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_piece)
            flange_r: float = 0.028
            flange_h: float = 0.012
            shaft_r: float = 0.015
            shaft_h: float = 0.110
            top: str = "ball"
            top_r: float = 0.020
            top_h: float = 0.020
            mass: float = 0.20
            com_z: float = 0.010
            inertia: tuple = (2.2e-4, 2.2e-4, 7.0e-5)
            color: tuple = (0.92, 0.90, 0.80)
            accent: tuple = (0.85, 0.68, 0.15)
            mu_s: float = 0.25
            mu_d: float = 0.20
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(plinth=PlinthSpawnerCfg, piece=PieceSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CastlingGallerySceneCfg(BaseCfg):
    """Config for `CastlingGalleryScene`. The cell tolerances are honest by
    construction: the channel walls bound a captive flange centre to +/-4 mm off
    the channel axis and the end walls bound the along-axis coordinate to 132 mm
    (pocket) / 212 mm (home), both INSIDE their 25 mm goal bands — a piece pushed
    to its stop passes; a piece a train-length short, on the lips, or in the decoy
    pocket cannot."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    cell_tol: float = tunable(0.025)      # goal-cell tolerance around the cell centre (m)
    upright_max_deg: float = tunable(15.0)  # piece axis within this of world-up
    settle_speed: float = tunable(0.03)   # max |lin vel| of both pieces when judging (m/s)
    z_in_max: float = tunable(0.014)      # root height above the channel floor below this =
    # still captive IN the channel (a piece resting on the lip rails sits at +30 mm)
    garage_y: float = tunable(0.065)      # |y| beyond this on the beacon side = rook garaged
    pass_x_margin: float = tunable(0.07)  # king x beyond junction_x + this = past the junction

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    plinth_yaw_deg: float = tunable(25.0)  # plinth yaw about nominal (+/- deg)
    plinth_jitter: float = tunable(0.05)   # plinth xy jitter (+/- m)
    king_x_range: tuple = tunable((-0.20, -0.14))  # king start x (plinth frame)
    sep_range: tuple = tunable((0.09, 0.24))       # rook start = king + sep (clamped)
    rook_x_max: float = tunable(0.13)      # rook never starts inside the home approach

    # --- info: layout (plinth local frame; origin at footprint centre on the ground) ------------
    plinth_pos: tuple = info((0.0, 0.0))   # plinth footprint centre (nominal, world xy)
    plinth_x: float = info(0.60)
    plinth_y: float = info(0.44)
    plinth_h: float = info(0.12)           # block top = channel floor height
    chan_gap: float = info(0.064)          # wall inner gap (flange level)
    slot_gap: float = info(0.038)          # lip inner slot (shaft level)
    wall_h: float = info(0.016)            # lip underside height above the floor
    lip_h: float = info(0.014)             # lip rail thickness (lip top at 30 mm)
    wall_t: float = info(0.012)
    junction_x: float = info(-0.02)        # pocket centreline x
    gallery_half: float = info(0.24)       # gallery interior half-length
    pocket_end_y: float = info(0.16)       # pocket end wall inner face
    home_x: float = info(0.206)            # home cell centre (stop bounds x <= 0.212)
    pocket_y: float = info(0.128)          # pocket end cell centre (stop bounds |y| <= 0.132)
    # --- info: pieces ----------------------------------------------------------------------------
    flange_r: float = info(0.028)
    flange_h: float = info(0.012)
    shaft_r: float = info(0.015)
    king_shaft_h: float = info(0.110)
    rook_shaft_h: float = info(0.060)
    piece_mass: float = info(0.20)
    king_color: tuple = info((0.92, 0.90, 0.80))
    king_accent: tuple = info((0.85, 0.68, 0.15))
    rook_color: tuple = info((0.50, 0.10, 0.10))
    rook_accent: tuple = info((0.62, 0.16, 0.16))
    # --- info: beacon ----------------------------------------------------------------------------
    beacon_size: tuple = info((0.055, 0.055, 0.008))
    beacon_y: float = info(0.205)          # |y| of the beacon tile centre (plinth frame)
    beacon_color: tuple = info((0.10, 0.72, 0.20))
    # rubric weights (0.20 + 0.25 + 0.30 = 0.75 = the non-success cap)
    w1: float = info(0.20)
    w2: float = info(0.25)
    w3: float = info(0.30)
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
@SCENES.register("castling_gallery")
class CastlingGalleryScene(BaseScene):
    cfg: CastlingGallerySceneCfg

    def __init__(self, cfg: CastlingGallerySceneCfg | None = None) -> None:
        super().__init__(cfg or CastlingGallerySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        plinth_spawn = cls["plinth"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            plinth_x=c.plinth_x, plinth_y=c.plinth_y, plinth_h=c.plinth_h,
            chan_gap=c.chan_gap, slot_gap=c.slot_gap, wall_h=c.wall_h,
            lip_h=c.lip_h, wall_t=c.wall_t, junction_x=c.junction_x,
            gallery_half=c.gallery_half, pocket_end_y=c.pocket_end_y,
            contact_offset=c.contact_offset)
        king_spawn = cls["piece"](
            flange_r=c.flange_r, flange_h=c.flange_h, shaft_r=c.shaft_r,
            shaft_h=c.king_shaft_h, top="ball", top_r=0.020,
            mass=c.piece_mass, color=c.king_color, accent=c.king_accent,
            contact_offset=c.contact_offset)
        rook_spawn = cls["piece"](
            flange_r=c.flange_r, flange_h=c.flange_h, shaft_r=c.shaft_r,
            shaft_h=c.rook_shaft_h, top="turret", top_r=0.024, top_h=0.020,
            mass=c.piece_mass, color=c.rook_color, accent=c.rook_accent,
            contact_offset=c.contact_offset)

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
            "plinth": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plinth",
                spawn=plinth_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plinth_pos[0], c.plinth_pos[1], 0.0)),
            ),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=sim_utils.CuboidCfg(
                    size=c.beacon_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.beacon_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 0.2)),
            ),
            "king": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/King",
                spawn=king_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, 0.6, 0.02)),
            ),
            "rook": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rook",
                spawn=rook_spawn,
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
        self.plinth: RigidObject = env.iscene["plinth"]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.king: RigidObject = env.iscene["king"]
        self.rook: RigidObject = env.iscene["rook"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.tgt = torch.ones(n, device=dev)  # target pocket side: +1 or -1 (plinth +y/-y)
        # latches (partial credit survives transients; success is judged live)
        self._l1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l3 = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the plinth (yaw + xy jitter), sample the target
        pocket side and park the beacon tile beside that pocket's end, then stand
        the king and rook in the gallery — king toward -x, rook between king and
        home — and clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- plinth: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.plinth_yaw_deg)
        q = _qz(psi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plinth_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.plinth_jitter
        st[:, 1] = c.plinth_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.plinth_jitter
        st[:, 3:7] = q
        pxy = st[:, 0:2].clone()
        st[:, 0:3] += origin
        self.plinth.write_root_state_to_sim(st, env_ids)

        # --- target pocket side (torch.rand comparison, not randint) ---
        s = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        self.tgt[env_ids] = s

        from isaaclab.utils.math import quat_apply

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            w = torch.zeros(m, 3, device=dev)
            w[:, 0:2] = pxy
            return w + quat_apply(q, loc) + origin

        # --- beacon: kinematic tile beside the target pocket's end wall ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.junction_x
        loc[:, 1] = s * c.beacon_y
        loc[:, 2] = c.plinth_h + c.beacon_size[2] / 2 + 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = to_world(loc)
        st[:, 3:7] = q
        self.beacon.write_root_state_to_sim(st, env_ids)

        # --- pieces: standing in the gallery, king behind rook ---
        kx = c.king_x_range[0] + torch.rand(m, device=dev) \
            * (c.king_x_range[1] - c.king_x_range[0])
        sep = c.sep_range[0] + torch.rand(m, device=dev) \
            * (c.sep_range[1] - c.sep_range[0])
        rx = torch.clamp(kx + sep, max=c.rook_x_max)
        for body, x in ((self.king, kx), (self.rook, rx)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = x
            loc[:, 2] = c.plinth_h + 0.002
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._l1[env_ids] = False
        self._l2[env_ids] = False
        self._l3[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plinth": self.plinth.data.root_state_w[env_ids].clone(),
            "beacon": self.beacon.data.root_state_w[env_ids].clone(),
            "king": self.king.data.root_state_w[env_ids].clone(),
            "rook": self.rook.data.root_state_w[env_ids].clone(),
            "tgt": self.tgt[env_ids].clone(),
            "l1": self._l1[env_ids].clone(),
            "l2": self._l2[env_ids].clone(),
            "l3": self._l3[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plinth.write_root_state_to_sim(state["plinth"], env_ids)
        self.beacon.write_root_state_to_sim(state["beacon"], env_ids)
        self.king.write_root_state_to_sim(state["king"], env_ids)
        self.rook.write_root_state_to_sim(state["rook"], env_ids)
        self.tgt[env_ids] = state["tgt"]
        self._l1[env_ids] = state["l1"]
        self._l2[env_ids] = state["l2"]
        self._l3[env_ids] = state["l3"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-grey PLINTH ({c.plinth_x * 100:.0f} x {c.plinth_y * 100:.0f} cm, "
            f"{c.plinth_h * 100:.0f} cm tall) carries a channel network on its top: one "
            f"long straight MAIN GALLERY, and, partway along it, a JUNCTION where two "
            f"identical short SIDE POCKETS branch off to the left and right. One gallery "
            f"end is capped by a tall GOLD wall — the cell touching the gold wall is the "
            f"king's HOME square. A small bright-GREEN BEACON TILE lies on the plinth "
            f"just beyond the end of exactly ONE side pocket: that is the TARGET pocket; "
            f"the mirror-image pocket with no tile is a decoy. Which side the beacon is "
            f"on, the plinth's position and heading, and the pieces' starting spots "
            f"change every episode.\n"
            f"Two chess pieces stand in the main gallery: the KING — the taller piece, "
            f"ivory shaft with a GOLD BALL crown — nearer the plain end, and the ROOK — "
            f"the shorter DARK-RED piece with a wide flat turret drum on top — between "
            f"the king and the gold wall. Each piece stands on a wide flanged base that "
            f"is captured UNDER the channels' overhanging lip rails: the pieces can "
            f"only SLIDE along the channels (push or hold the exposed shaft above the "
            f"rails and translate it); they physically cannot be lifted out, and the "
            f"channel is exactly one piece wide, so pieces can never pass each other.\n"
            f"Goal ('castling'): first slide the rook from the gallery through the "
            f"junction into the BEACON-marked pocket, all the way to the pocket's end "
            f"wall (within ~{c.cell_tol * 100:.0f} cm of the end cell); then slide the "
            f"king along the gallery, past the junction, up to the GOLD wall (within "
            f"~{c.cell_tol * 100:.0f} cm of the home cell). Because the rook blocks the "
            f"one-piece-wide gallery, it MUST be garaged first — and it must go into "
            f"the beacon pocket: a rook parked in the decoy pocket does not count. "
            f"Success: rook at the target pocket's end, king at the gold home cell, "
            f"both standing upright and at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the dark-red rook along the channel and garage it in the side "
            "pocket marked by the green tile, pushing it to the pocket's end. Then "
            "slide the white king along the main channel to the gold wall at its "
            "end. The pieces cannot be lifted out of the channels, and a rook left "
            "in the unmarked pocket does not count."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the plinth's local frame (origin at footprint
        centre on the ground)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.plinth.data.root_quat_w,
                                  pos_w - self.plinth.data.root_pos_w)

    def local_to_world(self, loc: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        return self.plinth.data.root_pos_w + quat_apply(self.plinth.data.root_quat_w, loc)

    def _piece_loc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(king_loc, rook_loc): piece root positions in the plinth frame, (N,3)."""
        return (self._local(self.king.data.root_pos_w),
                self._local(self.rook.data.root_pos_w))

    def upright(self, body) -> torch.Tensor:
        """(N,) bool: piece axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def in_channel(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,) bool: root height says the flange is still ON the channel floor,
        under the lips (a piece standing on top of the lip rails sits ~30 mm up)."""
        c = self.cfg
        dz = loc[:, 2] - c.plinth_h
        return (dz > -0.006) & (dz < c.z_in_max)

    def rook_in_pocket(self) -> torch.Tensor:
        """(N,) bool: rook garaged in the TARGET pocket (flange fully clear of the
        gallery, on the beacon side, captive)."""
        c = self.cfg
        _k, r = self._piece_loc()
        return ((r[:, 0] - c.junction_x).abs() < 0.035) \
            & (self.tgt * r[:, 1] > c.garage_y) & self.in_channel(r)

    def rook_seated(self) -> torch.Tensor:
        """(N,) bool: rook at the target pocket's end cell."""
        c = self.cfg
        _k, r = self._piece_loc()
        return ((r[:, 0] - c.junction_x).abs() < c.cell_tol) \
            & ((self.tgt * r[:, 1] - c.pocket_y).abs() < c.cell_tol) \
            & self.in_channel(r) & self.upright(self.rook)

    def king_home(self) -> torch.Tensor:
        """(N,) bool: king at the gold home cell."""
        c = self.cfg
        k, _r = self._piece_loc()
        return ((k[:, 0] - c.home_x).abs() < c.cell_tol) & (k[:, 1].abs() < 0.020) \
            & self.in_channel(k) & self.upright(self.king)

    def settled(self) -> torch.Tensor:
        """(N,) bool: both pieces slower than `settle_speed`."""
        c = self.cfg
        kv = self.king.data.root_lin_vel_w.norm(dim=-1)
        rv = self.rook.data.root_lin_vel_w.norm(dim=-1)
        return (kv < c.settle_speed) & (rv < c.settle_speed)

    def _update_latches(self) -> None:
        c = self.cfg
        k, _r = self._piece_loc()
        self._l1 |= self.rook_in_pocket()
        self._l2 |= self.rook_seated()
        self._l3 |= self._l2 & (k[:, 0] > c.junction_x + c.pass_x_margin) \
            & self.in_channel(k)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: rook at the TARGET pocket end cell AND king at the gold
        home cell, both upright, both still captive in the channels, both settled,
        states finite. All clauses are physical outcomes."""
        self._update_latches()
        k, r = self._piece_loc()
        finite = torch.isfinite(k).all(dim=-1) & torch.isfinite(r).all(dim=-1)
        return self.rook_seated() & self.king_home() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*garaged + 0.25*seated + 0.30*king-past-junction
        (all latched; ~0 for doing nothing), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w1 * self._l1.float() + c.w2 * self._l2.float()
                + c.w3 * self._l3.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="castling_gallery", robot="null"))
