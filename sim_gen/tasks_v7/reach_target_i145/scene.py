"""ColorCarouselScene — align a covered rotary carousel to the cue color, then feed the
white token through the loading chute into the matching cell (sim_gen task
`reach_target_i145`).

Derived from rlbench/reach_target, but STRATEGICALLY different: the seed is a pure
REACHING task — move the gripper until it touches the one red sphere among colored
distractors; nothing is grasped, nothing moves, the world state is untouched. Only the
seed's identification kernel survives here ("the colored one that matters this
episode"), and it is re-embedded in a mechanism task with a forced two-stage plan: a
sorting MACHINE stands on the floor — a four-cell rotary carousel under a fixed square
roof. Each open-top cell has colored walls (RED, GREEN, BLUE, YELLOW); the roof covers
every cell mouth at every angle except through one fixed loading chute (a walled
aperture in the roof). A cue card in the machine's corner tray shows this episode's
target color. The solver must (1) READ the cue, (2) ROTATE the carousel by its rim
pegs until the matching cell sits under the chute — the carousel is damped and stays
where it is left, and alignment must be within a tolerance window — and then
(3) FEED the white token through the chute so it falls into the target cell. Touching
things achieves nothing (proved in smoke): success requires a changed world state that
the seed never demands, reached through an ordered rotate-then-insert plan with a real
alignment sub-goal. A solver needs a different PLAN (cue reading, mechanism actuation
to a tolerance, then a gated insertion) and a different code structure (a yaw-alignment
latch plus a rotating-frame containment predicate — not a static reach check).

The cover story is real geometry, not a scripted gate: the roof underside sits 10 mm
above the cell walls (a 30 mm token cannot pass), every cell-mouth corner stays under
the roof at every angle (asserted), so the ONLY way into any cell is the chute — and
the chute admits the token into the target cell only when that cell is rotated under
it. The carousel is a plain dynamic body on a spawn-authored frictionless revolute
joint (kinematic-frame body0, collision-disabled pair); angular damping is what makes
it "stay where you leave it" (no motor, no detent).

success(): the white token rests INSIDE the target-colored cell — carousel-frame
containment window that by construction accepts every physically-in-cell resting pose
and rejects wall-top / just-outside / deck poses (asserted) — with token and carousel
settled. Identity: a wrong-color cell never counts.

score() is graded and latched (credit never evaporates): 0.25 target cell ever aligned
under the chute (held still for 10 substeps) + 0.45 token ever inside the target cell
= 0.70 cap; 1.0 iff success(). The null policy scores ~0 (the carousel spawns >= 25
degrees off alignment; the token spawns on the floor off the machine).

Per-episode randomization (readback-verified in smoke): the target color (cue card in
the tray), the carousel's starting angle (uniform, never aligned), and the token's
floor slot + jitter + yaw. Assets are fully procedural compound-box spawners (one
rigid body each). Heavy imports (isaaclab, pxr) are deferred so importing this module
— and registering the scene — stays app-free.
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

CELL_NAMES = ("red", "green", "blue", "yellow")
CELL_COLORS = ((0.85, 0.10, 0.10), (0.10, 0.70, 0.15), (0.15, 0.35, 0.85), (0.92, 0.80, 0.10))

# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         yaw: float = 0.0) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once — the
    duplicate xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw:
        h = 0.5 * float(yaw)
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, 0.0, math.sin(h))))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False, inertia=None):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    ma = UsdPhysics.MassAPI.Apply(root)
    ma.CreateMassAttr(float(mass))
    if inertia is not None:
        ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
        ma.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_machine(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC machine frame: base plate, center hub, the fixed square roof with
    ONE aperture on the -x side (the loading window), the collar chute walls standing
    on the roof around that aperture, and the cue-card tray in the near-right corner.
    One rigid body; origin = machine center at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 20.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    grey, slate, dark = (0.55, 0.55, 0.58), (0.40, 0.42, 0.48), (0.25, 0.25, 0.28)
    boxes = [
        ("plate", (0.60, 0.60, 0.02), (0.0, 0.0, 0.010), grey),
        ("hub", (0.04, 0.04, 0.085), (0.0, 0.0, 0.0625), dark),
        # roof: underside z=0.105, top z=0.117; aperture hole x in [-0.125, -0.075],
        # y in [-0.0325, 0.0325] (the ONLY way past the roof)
        ("roof_n", (0.32, 0.1275, 0.012), (0.0, +0.09625, 0.111), slate),
        ("roof_s", (0.32, 0.1275, 0.012), (0.0, -0.09625, 0.111), slate),
        ("roof_e", (0.235, 0.065, 0.012), (+0.0425, 0.0, 0.111), slate),
        ("roof_w", (0.035, 0.065, 0.012), (-0.1425, 0.0, 0.111), slate),
        # collar chute on the roof around the aperture (interior = the aperture)
        ("collar_e", (0.008, 0.081, 0.050), (-0.071, 0.0, 0.142), dark),
        ("collar_w", (0.008, 0.081, 0.050), (-0.129, 0.0, 0.142), dark),
        ("collar_n", (0.074, 0.008, 0.050), (-0.100, +0.0365, 0.142), dark),
        ("collar_s", (0.074, 0.008, 0.050), (-0.100, -0.0365, 0.142), dark),
        # cue-card tray (interior 0.062 x 0.062 on the plate top)
        ("tray_n", (0.078, 0.008, 0.012), (-0.26, -0.225, 0.026), dark),
        ("tray_s", (0.078, 0.008, 0.012), (-0.26, -0.295, 0.026), dark),
        ("tray_e", (0.008, 0.062, 0.012), (-0.225, -0.26, 0.026), dark),
        ("tray_w", (0.008, 0.062, 0.012), (-0.295, -0.26, 0.026), dark),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    return root


def _spawn_carousel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The carousel rotor: an annular deck (8 overlapping rotated slabs), four
    open-top colored cells at 90-degree spacing, and three rim tabs carrying upright
    orange handle pegs that stick out past the roof edge. One dynamic rigid body;
    origin = rotation axis at deck mid-height (the joint anchor)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass,
                             inertia=(0.02, 0.02, 0.03))
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(cfg.ang_damping)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    deck_col, tab_col, peg_col = (0.45, 0.45, 0.48), (0.45, 0.45, 0.48), (0.85, 0.50, 0.15)

    def rot(th: float, u: float, v: float) -> tuple:
        return (u * math.cos(th) - v * math.sin(th), u * math.sin(th) + v * math.cos(th))

    # deck annulus r 0.04..0.165, top local z = +0.0075
    for k in range(8):
        th = k * math.pi / 4
        x, y = rot(th, 0.1025, 0.0)
        _box(stage, f"{prim_path}/deck_{k}", (0.125, 0.145, 0.015), (x, y, 0.0),
             deck_col, co, material=mat, yaw=th)
    # four cells: interior 0.090 x 0.090, walls 0.008 thick, local z 0.0075..0.0525
    for i, col in enumerate(CELL_COLORS):
        th = i * math.pi / 2
        for tag, du, dv, size in (
                ("out", +0.049, 0.0, (0.008, 0.106, 0.045)),
                ("in", -0.049, 0.0, (0.008, 0.106, 0.045)),
                ("l", 0.0, +0.049, (0.090, 0.008, 0.045)),
                ("r", 0.0, -0.049, (0.090, 0.008, 0.045))):
            x, y = rot(th, cfg.cell_r + du, dv)
            _box(stage, f"{prim_path}/cell{i}_{tag}", size, (x, y, 0.030), col, co,
                 material=mat, yaw=th)
    # three rim tabs + handle pegs (the graspable actuation surface, outside the roof)
    for j, deg in enumerate((60.0, 180.0, 300.0)):
        th = math.radians(deg)
        x, y = rot(th, 0.2125, 0.0)
        _box(stage, f"{prim_path}/tab_{j}", (0.105, 0.030, 0.015), (x, y, 0.0),
             tab_col, co, material=mat, yaw=th)
        px, py = rot(th, cfg.peg_r, 0.0)
        _box(stage, f"{prim_path}/peg_{j}", (0.020, 0.020, 0.075), (px, py, 0.045),
             peg_col, co, material=mat, yaw=th)
    return root


def _spawn_token(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The white 30 mm token cube."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.1)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/body", (cfg.edge, cfg.edge, cfg.edge), (0.0, 0.0, 0.0),
         (0.95, 0.95, 0.97), cfg.contact_offset, material=mat)
    return root


def _spawn_card(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One colored cue card (a thin 50 x 50 x 8 mm plate)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/body", (0.050, 0.050, 0.008), (0.0, 0.0, 0.0),
         cfg.color, cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "machine" not in _SPAWNER_CACHE:

        @configclass
        class MachineSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_machine)
            contact_offset: float = 0.002

        @configclass
        class CarouselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carousel)
            mass: float = 1.5
            ang_damping: float = 4.0
            cell_r: float = 0.10
            peg_r: float = 0.245
            contact_offset: float = 0.002

        @configclass
        class TokenSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_token)
            mass: float = 0.05
            edge: float = 0.030
            contact_offset: float = 0.005

        @configclass
        class CardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_card)
            mass: float = 0.02
            color: tuple = (0.85, 0.10, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(machine=MachineSpawnerCfg, carousel=CarouselSpawnerCfg,
                              token=TokenSpawnerCfg, card=CardSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ColorCarouselSceneCfg(BaseCfg):
    """Config for `ColorCarouselScene`. The cover-and-chute honesty is asserted in
    `__post_init__`: the roof blocks every cell mouth at every angle except through
    the chute, the chute admits the token into the aligned cell across the whole
    alignment tolerance, and the cell containment window accepts every
    physically-in-cell resting pose while rejecting wall-top / just-outside poses."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    align_tol_deg: float = tunable(7.0)  # cell counts as under the chute within this yaw err
    cell_xy_tol: float = tunable(0.040)  # |token - cell center|, cell frame, per axis (m)
    cell_z_lo: float = tunable(0.010)  # token center height window, carousel frame:
    cell_z_hi: float = tunable(0.033)  # ... deck rest reads 0.0225; a wall-top rest 0.0675
    settle_lin: float = tunable(0.05)  # max token |lin vel| at judging (m/s)
    settle_ang: float = tunable(0.30)  # max carousel |ang vel| at judging (rad/s)
    latch_ang: float = tunable(0.15)  # align latch: carousel |ang vel| below this ...
    latch_count: int = tunable(10)  # ... for this many consecutive substeps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    offset_min_deg: float = tunable(25.0)  # carousel spawn offset from alignment, min
    offset_max_deg: float = tunable(335.0)  # ... and max (uniform in between)
    token_slots: tuple = tunable(((0.06, 0.24), (0.06, -0.24), (0.16, 0.36), (0.16, -0.36)))
    token_jitter: float = tunable(0.02)  # +-xy jitter of the token at its slot (m)
    card_jitter: float = tunable(0.004)  # +-xy jitter of the cue card in the tray (m)

    # --- info: structure (the geometry the spawners author) ----------------------------------
    hub_pos: tuple = info((0.42, 0.0))  # machine center (env frame; never moves)
    root_z: float = info(0.0425)  # carousel root / joint anchor height
    deck_top: float = info(0.050)  # deck top (world z) = cell floor
    cell_r: float = info(0.10)  # cell center radius
    cell_inner: float = info(0.090)  # cell interior width
    cell_wall_t: float = info(0.008)  # cell wall thickness
    cell_wall_top: float = info(0.095)  # cell wall top (world z)
    roof_under: float = info(0.105)  # roof underside (world z)
    roof_half: float = info(0.16)  # roof half-extent (square)
    ap_azimuth: float = info(math.pi)  # aperture azimuth in the machine frame (-x side)
    ap_r_lo: float = info(0.075)  # aperture radial band (machine frame)
    ap_r_hi: float = info(0.125)
    ap_half_w: float = info(0.0325)  # aperture tangential half-width
    collar_top: float = info(0.167)  # chute collar top (world z)
    peg_r: float = info(0.245)  # handle peg radius (outside the roof)
    peg_azimuths: tuple = info((60.0, 180.0, 300.0))  # carousel-frame, degrees
    tray_pos: tuple = info((-0.26, -0.26))  # tray center (machine frame, on the plate)
    plate_top: float = info(0.020)
    token_edge: float = info(0.030)
    token_mass: float = info(0.05)
    card_mass: float = info(0.02)
    carousel_mass: float = info(1.5)
    ang_damping: float = info(4.0)
    card_park: tuple = info((-0.70, 0.55, 0.12))  # unused cards: (x, y0, dy) on the ground
    contact_offset: float = info(0.002)
    token_contact_offset: float = info(0.005)

    def __post_init__(self) -> None:
        c = self
        tol = math.radians(c.align_tol_deg)
        # -- the chute is the only way in, and it admits the token --
        assert c.ap_r_hi - c.ap_r_lo >= c.token_edge + 0.012, "chute radial band fits token"
        assert 2 * c.ap_half_w >= c.token_edge + 0.012, "chute tangential width fits token"
        gap = c.roof_under - c.cell_wall_top
        assert 0.003 <= gap < c.token_edge / 2, \
            "roof-wall gap must clear rotation but block the token (cover is real)"
        mouth_corner = math.hypot(c.cell_r + c.cell_inner / 2, c.cell_inner / 2)
        assert mouth_corner <= c.roof_half - 0.005, \
            "every cell-mouth corner stays under the roof at every angle"
        assert math.hypot(c.peg_r - 0.010, 0.010) > c.roof_half * math.sqrt(2) + 0.004, \
            "handle pegs must clear the roof corners while rotating"
        # -- the chute drops into the aligned cell across the WHOLE tolerance window --
        assert c.ap_r_lo >= c.cell_r - c.cell_inner / 2 + 0.010, "chute inside cell, radially"
        assert c.ap_r_hi <= c.cell_r + c.cell_inner / 2 - 0.010, "chute inside cell, radially"
        tang = c.cell_r * math.sin(tol) + (2 * c.ap_half_w - c.token_edge) / 2
        assert tang + c.token_edge / 2 <= c.cell_inner / 2, \
            "a token dropped at max tolerated misalignment still enters the cell"
        # -- the containment window is honest by construction --
        assert c.cell_inner / 2 - c.token_edge / 2 <= c.cell_xy_tol, \
            "every physically-in-cell resting pose must be inside the window"
        assert c.cell_inner / 2 + c.cell_wall_t / 2 > c.cell_xy_tol + 0.005, \
            "a token centered on / beyond a cell wall must be outside the window"
        rest_z = c.deck_top - c.root_z + c.token_edge / 2
        assert c.cell_z_lo + 0.002 < rest_z < c.cell_z_hi - 0.002, \
            "the deck-rest pose must sit inside the height window"
        wall_rest = (c.cell_wall_top - c.root_z) + c.token_edge / 2
        assert wall_rest > c.cell_z_hi + 0.020, \
            "a token resting on a wall top must be above the height window"
        # -- randomization can never spawn aligned --
        assert c.offset_min_deg > c.align_tol_deg + 10.0, "reset is never aligned"
        assert c.offset_max_deg == 360.0 - c.offset_min_deg, "offset band symmetric"
        # -- layout: tray and token slots clear of the machine's moving parts --
        tx, ty = c.tray_pos
        assert math.hypot(abs(tx) - 0.039, abs(ty) - 0.039) > 0.265 + 0.010, \
            "tray corner clear of the carousel sweep"
        hx, hy = c.hub_pos
        for sx, sy in c.token_slots:
            margin = c.token_jitter + c.token_edge / 2
            off_plate = abs(sx - hx) > 0.30 + margin or abs(sy - hy) > 0.30 + margin
            assert off_plate, f"token slot ({sx},{sy}) must be off the machine plate"
            cxx, cyy = hx + tx, hy + ty
            assert max(abs(sx - cxx), abs(sy - cyy)) > 0.039 + margin + 0.005, \
                f"token slot ({sx},{sy}) clear of the cue tray"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("color_carousel")
class ColorCarouselScene(BaseScene):
    cfg: ColorCarouselSceneCfg

    def __init__(self, cfg: ColorCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or ColorCarouselSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        hx, hy = c.hub_pos

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "machine": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Machine",
                spawn=sp["machine"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0)),
            ),
            "carousel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carousel",
                spawn=sp["carousel"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.carousel_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.carousel_mass, ang_damping=c.ang_damping,
                    cell_r=c.cell_r, peg_r=c.peg_r, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, c.root_z)),
            ),
            "token": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Token",
                spawn=sp["token"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.token_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.token_mass, edge=c.token_edge,
                    contact_offset=c.token_contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.token_slots[0][0], c.token_slots[0][1],
                         c.token_edge / 2 + 0.002)),
            ),
        }
        px, py0, pdy = c.card_park
        for j, name in enumerate(CELL_NAMES):
            out[f"card_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Card" + name.title(),
                spawn=sp["card"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.card_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.card_mass, color=CELL_COLORS[j],
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py0 + j * pdy, 0.006)),
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
        n, dev = env.num_envs, env.device
        self.machine: RigidObject = env.iscene["machine"]
        self.carousel: RigidObject = env.iscene["carousel"]
        self.token: RigidObject = env.iscene["token"]
        self.cards: dict[str, RigidObject] = {
            nm: env.iscene[f"card_{nm}"] for nm in CELL_NAMES}
        self.env_origins = env.iscene.env_origins
        self.target = torch.zeros(n, dtype=torch.long, device=dev)
        # progress latches (post_step): target cell ever aligned+still, token ever in cell
        self.align_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.align_latch = torch.zeros(n, device=dev)
        self.drop_latch = torch.zeros(n, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the carousel's frictionless revolute pivot on the kinematic machine
        frame — pair collision disabled (the joint constrains all other DOF; the roof
        and the rotor never need to touch), NO limits (continuous rotation). Angular
        damping on the body is what makes the rotor stay where it is left."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/carousel_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Machine"])
            j.CreateBody1Rel().SetTargets([f"{base}/Carousel"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.root_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the target color, drop its cue card into the tray
        (others parked far away), spin the carousel to a uniformly random NEVER-aligned
        angle (written along its own joint DOF), and set the token at a sampled floor
        slot off the machine with jitter + free yaw; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        hx, hy = c.hub_pos

        def write(body, pos: torch.Tensor, quat: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            if quat is None:
                st[:, 3] = 1.0
            else:
                st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # target color (torch.rand, not randint — the first-randint degeneracy)
        t = (torch.rand(m, device=dev) * 4).long().clamp(max=3)
        self.target[env_ids] = t

        # carousel: yaw = alignment angle for the target PLUS a never-aligned offset
        off = math.radians(c.offset_min_deg) + torch.rand(m, device=dev) * (
            math.radians(c.offset_max_deg) - math.radians(c.offset_min_deg))
        psi = self._wrap(c.ap_azimuth - t.float() * (math.pi / 2) + off)
        pos = torch.tensor([hx, hy, c.root_z], device=dev).expand(m, 3)
        half = psi / 2
        quat = torch.stack([torch.cos(half), torch.zeros(m, device=dev),
                            torch.zeros(m, device=dev), torch.sin(half)], dim=-1)
        write(self.carousel, pos, quat)

        # token: sampled floor slot + jitter + free yaw
        slots = torch.tensor(c.token_slots, device=dev)
        pick = (torch.rand(m, device=dev) * len(c.token_slots)).long().clamp(
            max=len(c.token_slots) - 1)
        tok = torch.zeros(m, 3, device=dev)
        tok[:, 0:2] = slots[pick] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.token_jitter
        tok[:, 2] = c.token_edge / 2 + 0.002
        tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
        tq = torch.stack([torch.cos(tyaw / 2), torch.zeros(m, device=dev),
                          torch.zeros(m, device=dev), torch.sin(tyaw / 2)], dim=-1)
        write(self.token, tok, tq)

        # cue cards: the target's card in the tray, the rest parked off-scene
        trx, try_ = hx + c.tray_pos[0], hy + c.tray_pos[1]
        px, py0, pdy = c.card_park
        for j, nm in enumerate(CELL_NAMES):
            in_tray = (t == j).unsqueeze(-1)
            tray = torch.zeros(m, 3, device=dev)
            tray[:, 0] = trx
            tray[:, 1] = try_
            tray[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.card_jitter
            tray[:, 2] = c.plate_top + 0.004 + 0.002
            park = torch.tensor([px, py0 + j * pdy, 0.006], device=dev).expand(m, 3)
            write(self.cards[nm], torch.where(in_tray, tray, park))

        self.align_count[env_ids] = 0
        self.align_latch[env_ids] = 0.0
        self.drop_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"carousel": self.carousel, "token": self.token,
                  **{f"card_{nm}": b for nm, b in self.cards.items()}}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "target": self.target[env_ids].clone(),
            "align_count": self.align_count[env_ids].clone(),
            "align_latch": self.align_latch[env_ids].clone(),
            "drop_latch": self.drop_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"carousel": self.carousel, "token": self.token,
                  **{f"card_{nm}": b for nm, b in self.cards.items()}}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.target[env_ids] = state["target"]
        self.align_count[env_ids] = state["align_count"]
        self.align_latch[env_ids] = state["align_latch"]
        self.drop_latch[env_ids] = state["drop_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A sorting machine stands on a square base plate on the floor. Under its "
            "fixed slate-grey square roof spins a four-cell carousel: four open-top "
            "square cells (walls colored RED, GREEN, BLUE and YELLOW) at 90-degree "
            "spacing on a rotating deck. The roof covers every cell mouth at every "
            "angle — the ONLY opening is one square loading window on the machine's "
            "near side, ringed by a short dark chute collar standing on the roof. "
            "Three ORANGE handle pegs on rim tabs stick out past the roof edge at "
            "deck level; pushing a peg sideways spins the carousel about its center "
            "pivot. The carousel turns freely but is damped: it stays where you leave "
            "it. In a small dark tray on the corner of the base plate lies ONE "
            "colored cue card — its color names this episode's target cell. A WHITE "
            f"token cube ({c.token_edge * 1000:.0f} mm) lies loose on the floor near "
            "the machine. The cue color, the carousel's starting angle and the "
            "token's spot change every episode: read them by looking.\n"
            "Goal: rotate the carousel by its orange pegs until the cell whose color "
            "matches the cue card sits under the loading chute (within about "
            f"{c.align_tol_deg:.0f} degrees), then drop the white token down through "
            "the chute so it lands INSIDE that cell, and leave everything at rest. "
            "The token cannot enter any cell except through the chute, and the chute "
            "feeds only the aligned cell. A token in a wrong-color cell, on the deck "
            "between cells, on the roof, or still moving counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Look at the cue card in the tray, spin the carousel by its orange pegs "
            "until the matching colored cell sits under the loading chute, then drop "
            "the white token through the chute into that cell."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def carousel_yaw(self) -> torch.Tensor:
        """(N,) carousel yaw about +z (the joint constrains all other DOF)."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        e = quat_apply(self.carousel.data.root_quat_w, ex)
        return torch.atan2(e[:, 1], e[:, 0])

    def align_err(self) -> torch.Tensor:
        """(N,) signed yaw error of the TARGET cell from the chute azimuth."""
        c = self.cfg
        return self._wrap(self.carousel_yaw()
                          + self.target.float() * (math.pi / 2) - c.ap_azimuth)

    def aligned(self) -> torch.Tensor:
        return self.align_err().abs() <= math.radians(self.cfg.align_tol_deg)

    def _token_cell_uvz(self, theta: torch.Tensor) -> torch.Tensor:
        """(N,3) token position in the frame of the cell at carousel-frame azimuth
        `theta`: u radial (cell center at u = cell_r), v tangential, z above the
        joint anchor plane."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = quat_apply_inverse(
            self.carousel.data.root_quat_w,
            self.token.data.root_pos_w - self.carousel.data.root_pos_w)
        ct, st = torch.cos(theta), torch.sin(theta)
        u = rel[:, 0] * ct + rel[:, 1] * st
        v = -rel[:, 0] * st + rel[:, 1] * ct
        return torch.stack([u, v, rel[:, 2]], dim=-1)

    def token_in_cell(self, i: int) -> torch.Tensor:
        """(N,) bool, geometric: token inside cell `i` (0=red 1=green 2=blue
        3=yellow) — carousel-frame containment window (honesty asserted in
        `__post_init__`)."""
        theta = torch.full((self.env.num_envs,), i * math.pi / 2,
                           device=self.env.device)
        return self._in_window(self._token_cell_uvz(theta))

    def token_in_target_cell(self) -> torch.Tensor:
        """(N,) bool, geometric: token inside THIS episode's target cell."""
        theta = self.target.float() * (math.pi / 2)
        return self._in_window(self._token_cell_uvz(theta))

    def _in_window(self, uvz: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return ((uvz[:, 0] - c.cell_r).abs() <= c.cell_xy_tol) \
            & (uvz[:, 1].abs() <= c.cell_xy_tol) \
            & (uvz[:, 2] >= c.cell_z_lo) & (uvz[:, 2] <= c.cell_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: token AND carousel at rest."""
        c = self.cfg
        return (self.token.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.carousel.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress every physics substep: target cell aligned AND the rotor
        near-still for `latch_count` consecutive substeps (a drive-by sweep through
        the window does not count), and the token ever inside the target cell."""
        c = self.cfg
        still = self.carousel.data.root_ang_vel_w.norm(dim=-1) < c.latch_ang
        hold = self.aligned() & still
        self.align_count = torch.where(hold, self.align_count + 1,
                                       torch.zeros_like(self.align_count))
        self.align_latch = torch.maximum(
            self.align_latch, (self.align_count >= c.latch_count).float())
        self.drop_latch = torch.maximum(
            self.drop_latch, self.token_in_target_cell().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the white token rests inside the TARGET-color cell, token and
        carousel settled. Judged by identity — a wrong-color cell never counts; the
        roof makes the chute (and therefore prior alignment) physically necessary."""
        return self.token_in_target_cell() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 target cell ever aligned+held under the chute +
        0.45 token ever inside the target cell (cap 0.70); 1.0 iff success().
        Latched — credit never evaporates; the null policy scores ~0 (spawns
        misaligned with the token far from the machine)."""
        base = (0.25 * self.align_latch + 0.45 * self.drop_latch).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="color_carousel", robot="null", env_spacing=3.0))
