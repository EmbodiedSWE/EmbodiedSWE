"""FoldRackScene — the drying rack starts FOLDED FLAT: erect the gravity-bistable fin
comb past over-center to CREATE the color slots, then insert the plate into the BLUE
gap (sim_gen task `put_plate_in_colored_dish_rack_i384`).

Derived from rlbench/put_plate_in_colored_dish_rack, but STRATEGICALLY different: the
seed's dish rack is a static prop whose color-named slot pre-exists, open to the sky —
one prehensile transport (lift THE plate off its stand, lower it into the open slot)
solves it. Here the rack is a FOLD-FLAT rack and the episode starts with its hinged
FIN COMB folded DOWN over the deck: the slots do not exist yet, and the folded comb
bodily covers the deposit area — the seed's entire strategy has no target to aim at.
The comb is a real articulated body on a spawn-authored revolute hinge with a
deliberate over-center geometry (its center of mass crosses the hinge vertical at
~`overcenter_deg`): gravity holds it CLOSED at the 0-deg stop and, once pushed past
over-center, gravity slams it OPEN against the `joint_hi_deg` stop — both end states
are self-holding, and only the OPEN state erects the three fins that form the two
color-coded plate gaps (BLUE gap on the blue-fin side, YELLOW gap on the yellow-fin
side). The white plate starts flat on a separate pedestal stand.

A solver therefore needs a different PLAN and different code structure than the seed:
(1) ERECT the receptacle — push/pull the comb's red handle tab through >
    `overcenter_deg` of hinge travel so gravity parks it at the open stop (a
    mechanism-driving phase with a bistable hand-off, nothing like a pick-and-place);
(2) only THEN transport the plate and thread it edgewise down into the BLUE gap —
    an on-edge insertion between fin posts, nothing like the seed's flat lowering.
The order is enforced by PHYSICS, not the rubric: while the comb is folded there is
no gap anywhere in the world (smoke drops a plate on the folded rack and it just
lands on the comb cover), and the seat predicate itself only exists when the comb is
OPEN (`deployed`), so no arrangement made before erecting can ever count.

success(): plate seated ON EDGE in the BLUE gap in the CHASSIS body frame (x within
the blue slot band, y inside the spine-to-lip roll range, plate-center z in the
on-edge rest window, |plate axis dot z| small) AND the comb OPEN (hinge angle >=
`deploy_min_deg`) AND everything PERSISTENTLY still (counter-latch).

score(), latched in post_step (credit never evaporates): +0.35 once the comb has ever
been OPEN for `latch_steps` consecutive slow steps (the comb starts folded and
gravity-held, so a null policy can never earn it), +0.20 once the plate has ever been
seated in EITHER color gap for `latch_steps` consecutive slow steps (no gap exists at
reset — this is unearnable before deploying); capped at 0.55; exactly 1.0 iff
success() (BLUE gap specifically). Null policy scores ~0.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - chassis (DYNAMIC, 4 kg — dynamic so reset may teleport the whole linkage
    consistently; the kinematic-anchor-stays-world-fixed trap): deck slab (origin at
    the DECK-TOP center), front retaining lip, and visual-only blue/yellow slot tabs
    on the deck in front of where each gap erects;
  - comb (dynamic, one rigid body on a spawn-authored revolute hinge to the chassis,
    axis = chassis x, limits [0, `joint_hi_deg`] deg): a spine bar at the hinge, three
    fins (BLUE side fin, wide GRAY center fin, YELLOW side fin) whose two gaps become
    the plate slots when erected, a full-width TOP COVER plate that roofs the fin
    gaps while folded (erected it becomes a backing wall behind the posts) — the
    member that makes the deploy-first order PHYSICS-forced, asserted in
    `__post_init__` — and a red HANDLE tab standing proud of the folded cover.
    Mass/CoM/inertia authored explicitly (the MassAPI CoM-at-origin trap would
    kill the over-center bistability);
  - stand (KINEMATIC pedestal cylinder) + plate (dynamic white disc, spawned FLAT so
    in-episode flat poses differ from spawn by pure yaw — vertical forces stay
    vertical under either wrench-frame encoding).
Friction materials are bound explicitly in the spawners (the default-material trap).

Per-episode randomization (readback-verified in smoke): the chassis pose (xy jitter +
FULL random yaw — where the handle and each color gap sit in the world changes every
episode), the comb's start angle (small fold-side range that settles onto the closed
stop), the stand's xy, and the plate's on-stand xy + free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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

BLUE = (0.15, 0.35, 0.90)
YELLOW = (0.92, 0.80, 0.12)
GRAY = (0.55, 0.55, 0.58)
RED = (0.85, 0.15, 0.15)
WOOD = (0.62, 0.48, 0.30)
WHITE = (0.93, 0.93, 0.90)
BLUE_GAP_SIGN = -1.0  # blue gap center at chassis-frame x = -slot_dx (yellow at +slot_dx)


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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         collide: bool = True) -> None:
    """Author one axis-aligned box child prim (translate then scale, authored once each —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(seg.GetPrim(), contact_offset, material)


def _spawn_chassis(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC rack chassis; body origin at the DECK-TOP center. Deck slab (bottom on
    the ground), front retaining lip, visual-only colored slot tabs. Dynamic (not
    kinematic) so per-reset teleports can move chassis+comb as one consistent linkage;
    heavy + low CoM so it stays planted while the comb is driven."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -cfg.deck_size[2] / 2))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.017, 0.027, 0.043))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.2)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    dx, dy, dz = cfg.deck_size
    _box(stage, f"{prim_path}/deck", (dx, dy, dz), (0.0, 0.0, -dz / 2), WOOD, co, mat)
    lx, ly, lz = cfg.lip_size
    _box(stage, f"{prim_path}/lip", (lx, ly, lz), (0.0, cfg.lip_y, lz / 2),
         (0.48, 0.36, 0.22), co, mat)
    # visual-only slot color tabs, on the deck strip the folded comb does NOT cover
    for tag, sgn, col in (("blue", BLUE_GAP_SIGN, BLUE), ("yellow", -BLUE_GAP_SIGN, YELLOW)):
        _box(stage, f"{prim_path}/tab_{tag}", (0.030, 0.014, 0.003),
             (sgn * cfg.slot_dx, 0.052, 0.0015), col, co, collide=False)
    return root


def _spawn_comb(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The hinged fin comb, ONE dynamic rigid body; body origin ON the hinge axis
    (chassis-local (0, hinge_y, hinge_z)); comb-local x = hinge axis. Folded (0 deg)
    the comb lies flat over the deck; erected it stands the fins up. Parts: spine bar
    at the hinge, BLUE / GRAY(center) / YELLOW fins forming the two plate gaps, red
    handle tab. Mass/CoM/inertia authored EXPLICITLY (over-center bistability lives in
    the CoM). Revolute joint to the sibling Chassis authored IN THE SPAWNER so it
    exists before the physics parse; the joint pair's collisions are filtered."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, float(cfg.com_y), float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(4.3e-4, 5.5e-4, 9.0e-4))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.3)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    # spine bar (along the hinge)
    sx, sy, sz = cfg.spine_size
    _box(stage, f"{prim_path}/spine", (sx, sy, sz),
         (0.0, cfg.spine_yc, sz / 2), (0.40, 0.40, 0.44), co, mat)
    # fins: BLUE at -side_fin_x, YELLOW at +side_fin_x, wide GRAY center
    fin_yc = cfg.fin_y0 + cfg.fin_len / 2
    _box(stage, f"{prim_path}/fin_blue", (cfg.fin_t_side, cfg.fin_len, cfg.fin_th),
         (BLUE_GAP_SIGN * cfg.side_fin_x, fin_yc, cfg.fin_th / 2), BLUE, co, mat)
    _box(stage, f"{prim_path}/fin_yellow", (cfg.fin_t_side, cfg.fin_len, cfg.fin_th),
         (-BLUE_GAP_SIGN * cfg.side_fin_x, fin_yc, cfg.fin_th / 2), YELLOW, co, mat)
    _box(stage, f"{prim_path}/fin_center", (cfg.fin_t_center, cfg.fin_len, cfg.fin_th),
         (0.0, fin_yc, cfg.fin_th / 2), GRAY, co, mat)
    # full-width top COVER plate: folded it ROOFS the fin gaps (nothing can be slotted
    # while the comb is closed — the order-forcing member); erected it becomes a
    # backing wall behind the fin posts, clear of the plate's lean.
    _box(stage, f"{prim_path}/cover", (cfg.spine_size[0], cfg.fin_len, cfg.cover_th),
         (0.0, fin_yc, cfg.fin_th + cfg.cover_th / 2), (0.68, 0.54, 0.34), co, mat)
    # red handle tab: a post standing proud of the folded cover
    hx, hy, hz = cfg.handle_size
    _box(stage, f"{prim_path}/handle", (hx, hy, hz),
         tuple(float(v) for v in cfg.handle_center), RED, co, mat)

    # Revolute hinge to the sibling Chassis — authored IN THE SPAWNER so it exists
    # BEFORE the physics parse. Axis X; anchor at chassis-local (0, hinge_y, hinge_z)
    # = comb-local origin. Limits [0, joint_hi_deg] deg (well below the 180-deg wrap).
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Chassis"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(cfg.hinge_y), float(cfg.hinge_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(cfg.joint_hi_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chassis" not in _SPAWNER_CACHE:

        @configclass
        class ChassisSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chassis)
            deck_size: tuple = (0.28, 0.22, 0.024)
            lip_y: float = 0.075
            lip_size: tuple = (0.24, 0.012, 0.016)
            slot_dx: float = 0.035
            mass: float = 4.0
            mu_static: float = 0.7
            mu_dynamic: float = 0.6
            contact_offset: float = 0.0015

        @configclass
        class CombSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_comb)
            spine_size: tuple = (0.14, 0.024, 0.016)
            spine_yc: float = -0.002
            fin_y0: float = 0.010
            fin_len: float = 0.090
            fin_th: float = 0.016
            fin_t_side: float = 0.012
            fin_t_center: float = 0.036
            side_fin_x: float = 0.058
            cover_th: float = 0.004
            handle_size: tuple = (0.012, 0.014, 0.036)
            handle_center: tuple = (0.0, 0.088, 0.038)
            hinge_y: float = -0.060
            hinge_z: float = 0.019
            joint_hi_deg: float = 98.0
            mass: float = 0.35
            com_y: float = 0.0404
            com_z: float = 0.0115
            mu_static: float = 0.7
            mu_dynamic: float = 0.6
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(chassis=ChassisSpawnerCfg, comb=CombSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FoldRackSceneCfg(BaseCfg):
    """Config for `FoldRackScene`. `__post_init__` computes the comb CoM from the part
    volumes and asserts the honesty invariants: the hinge is genuinely over-center
    bistable with margin on both sides, the folded comb covers the deposit area and
    leaves no plate-passing gap underneath, the erected gaps pass exactly one plate
    with margin, the seat bands cover every physically-seated pose yet reject
    perches/flat rests, the comb's swept corners clear the deck, and the seated top
    rim stands proud of the fins for a parallel-jaw rim pinch."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    x_tol: float = tunable(0.021)  # |x - gap center| (chassis frame) counted seated
    y_s: float = tunable(0.005)  # seat y-band center (chassis frame)
    y_tol: float = tunable(0.050)  # |y - y_s| counted seated
    z_lo: float = tunable(0.078)  # plate-center z window (over deck top) ...
    z_hi: float = tunable(0.093)  # ... rejects folded-cover / lip / fin-top rests
    axis_z_max: float = tunable(0.30)  # |plate axis dot world z| counted ON EDGE
    deploy_min_deg: float = tunable(92.0)  # hinge angle counted OPEN (deployed)
    fold_max_deg: float = tunable(8.0)  # hinge angle counted folded (smoke readback)
    settle_lin: float = tunable(0.08)  # max |lin vel| counted still (m/s)
    settle_ang: float = tunable(1.2)  # max |ang vel| counted still (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must persist this many steps
    latch_steps: int = tunable(24)  # deploy/seat must persist this many slow steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rack_pos: tuple = tunable((0.0, 0.06))  # chassis nominal center (world xy)
    rack_jitter: float = tunable(0.03)  # uniform +/- xy jitter at reset
    rack_yaw_deg: float = tunable(180.0)  # uniform +/- yaw (FULL circle by default)
    comb_start_deg: tuple = tunable((2.0, 6.0))  # start angle range (settles closed)
    stand_pos: tuple = tunable((0.0, 0.38))  # stand nominal center (world xy)
    stand_jitter: float = tunable(0.02)  # uniform +/- xy jitter at reset
    plate_jxy: float = tunable(0.008)  # plate on-stand xy jitter (+ free yaw)

    # --- info: structure ---------------------------------------------------------------------
    deck_size: tuple = info((0.28, 0.22, 0.024))  # deck slab (top = chassis origin)
    lip_y: float = info(0.075)  # front lip center y (chassis frame)
    lip_size: tuple = info((0.24, 0.012, 0.016))
    hinge_y: float = info(-0.060)  # hinge axis at chassis-local (y, z)
    hinge_z: float = info(0.019)
    spine_size: tuple = info((0.14, 0.024, 0.016))  # comb-local, folded pose
    spine_yc: float = info(-0.002)
    fin_y0: float = info(0.010)  # fins span comb-local y in [fin_y0, fin_y0+fin_len]
    fin_len: float = info(0.090)
    fin_th: float = info(0.016)  # folded z-thickness of the fins
    fin_t_side: float = info(0.012)  # side fin thickness (x)
    fin_t_center: float = info(0.036)  # center fin thickness (x)
    side_fin_x: float = info(0.058)  # side fin centers at x = +/- side_fin_x
    slot_dx: float = info(0.035)  # gap centers at chassis-frame x = +/- slot_dx
    cover_th: float = info(0.004)  # top cover plate (roofs the gaps when folded)
    handle_size: tuple = info((0.012, 0.014, 0.036))
    handle_center: tuple = info((0.0, 0.088, 0.038))
    joint_hi_deg: float = info(98.0)  # open stop (gravity parks the comb here)
    comb_mass: float = info(0.35)
    chassis_mass: float = info(4.0)
    plate_r: float = info(0.085)
    plate_t: float = info(0.020)
    plate_m: float = info(0.30)
    stand_r: float = info(0.045)
    stand_h: float = info(0.060)
    mu_static: float = info(0.7)
    mu_dynamic: float = info(0.6)
    contact_offset: float = info(0.0015)
    franka_base: tuple = info((0.42, 0.12, 0.0))  # proposed robot base (embodiment)

    # Derived (filled in __post_init__).
    gap: float = field(default=None, init=False)  # fin-to-fin gap width (x)
    lean_max: float = field(default=None, init=False)  # max in-gap lean (rad)
    com_y: float = field(default=None, init=False)  # comb CoM (comb-local)
    com_z: float = field(default=None, init=False)
    overcenter_deg: float = field(default=None, init=False)  # CoM crosses hinge vertical

    def __post_init__(self) -> None:
        # --- gap geometry: the two gaps between center fin and each side fin -------------
        inner_side = self.side_fin_x - self.fin_t_side / 2
        outer_center = self.fin_t_center / 2
        self.gap = inner_side - outer_center
        assert abs(self.slot_dx - (inner_side + outer_center) / 2) < 1e-9, \
            "slot_dx must be the gap center"
        slack = self.gap - self.plate_t
        # insertable: per-side clearance clears both contact offsets with margin
        assert slack / 2 >= 2 * self.contact_offset + 0.003, "gap must pass one plate"
        # one plate only: two plate thicknesses exceed the gap
        assert 2 * self.plate_t >= self.gap + 0.005, "gap must hold exactly one plate"
        # --- comb CoM from part volumes (authored into the spawner) ----------------------
        parts = [  # (volume, y_center, z_center) in the comb frame, folded pose
            (self.spine_size[0] * self.spine_size[1] * self.spine_size[2],
             self.spine_yc, self.spine_size[2] / 2),
            (2 * self.fin_t_side * self.fin_len * self.fin_th,
             self.fin_y0 + self.fin_len / 2, self.fin_th / 2),
            (self.fin_t_center * self.fin_len * self.fin_th,
             self.fin_y0 + self.fin_len / 2, self.fin_th / 2),
            (self.spine_size[0] * self.fin_len * self.cover_th,
             self.fin_y0 + self.fin_len / 2, self.fin_th + self.cover_th / 2),
            (self.handle_size[0] * self.handle_size[1] * self.handle_size[2],
             self.handle_center[1], self.handle_center[2]),
        ]
        vol = sum(v for v, _y, _z in parts)
        self.com_y = sum(v * y for v, y, _z in parts) / vol
        self.com_z = sum(v * z for v, _y, z in parts) / vol
        # --- over-center bistability with margin on BOTH sides ---------------------------
        self.overcenter_deg = math.degrees(math.atan2(self.com_y, self.com_z))
        assert 40.0 <= self.overcenter_deg <= self.deploy_min_deg - 8.0, \
            f"over-center angle {self.overcenter_deg:.1f} must sit well inside travel"
        assert self.joint_hi_deg >= self.deploy_min_deg + 4.0, \
            "gravity must park the comb PAST deploy_min"
        assert self.joint_hi_deg <= 175.0, "stay clear of the PhysX 180-deg wrap"
        assert self.fold_max_deg <= self.overcenter_deg - 30.0
        # --- swept back corners of the spine clear the deck across the whole travel ------
        corners = [(self.spine_yc - self.spine_size[1] / 2, 0.0),
                   (self.spine_yc - self.spine_size[1] / 2, self.spine_size[2])]
        min_clear = min(
            self.hinge_z + (y * math.sin(th) + z * math.cos(th))
            for y, z in corners
            for th in (math.radians(t) for t in range(0, int(self.joint_hi_deg) + 1)))
        assert min_clear >= 0.002, f"spine sweeps into the deck (clear {min_clear:.4f})"
        # --- folded comb: covers the deposit area, passes nothing underneath -------------
        assert self.hinge_z < self.plate_t, "a flat plate must NOT slide under the cover"
        folded_tip_y = self.hinge_y + self.fin_y0 + self.fin_len
        assert folded_tip_y >= 0.035, "folded comb must cover the gap/post region"
        assert folded_tip_y <= self.lip_y - self.lip_size[1] / 2 - 0.010, \
            "folded comb must clear the front lip"
        # spine and fin/cover region are contiguous (no folded channel between them)
        assert self.fin_y0 <= self.spine_yc + self.spine_size[1] / 2 + 1e-9, \
            "fins must start where the spine ends (no folded gap)"
        # ORDER IS PHYSICS-FORCED: while folded, the cover roofs the fin gaps, so no
        # physically-free on-edge stand exists anywhere inside the seat bands —
        # in front of the folded comb the disc chord over the cover pushes the stand
        # center beyond the y band; behind, beyond the spine, likewise.
        r = self.plate_r
        cover_top = self.hinge_z + self.fin_th + self.cover_th
        assert cover_top < r, "cover must sit below the plate radius"
        stand_clear_front = folded_tip_y + math.sqrt(r**2 - (r - cover_top)**2)
        assert stand_clear_front >= self.y_s + self.y_tol + 0.010, \
            "folded comb must deny every in-band stand in front"
        spine_rear_y = self.hinge_y + self.spine_yc - self.spine_size[1] / 2
        spine_top = self.hinge_z + self.spine_size[2]
        stand_clear_rear = spine_rear_y - math.sqrt(r**2 - (r - spine_top)**2)
        assert stand_clear_rear <= self.y_s - self.y_tol - 0.010, \
            "folded comb must deny every in-band stand behind"
        # --- erected geometry -------------------------------------------------------------
        th_open = math.radians(self.joint_hi_deg)
        s, co = math.sin(th_open), math.cos(th_open)
        post_top = self.hinge_z + (self.fin_y0 + self.fin_len) * s + self.fin_th * co
        post_bot = self.hinge_z + self.fin_y0 * s  # bottom of the erected fin posts
        post_front = self.hinge_y + self.fin_y0 * co - self.fin_th * s  # front face y
        assert post_bot <= 0.035, "erected posts must reach low enough to catch the rim"
        assert post_top >= 0.10, "erected posts must be tall enough to arrest tipping"
        # seat kinematics: lean bounded by the gap slack against the tall posts
        self.lean_max = math.atan2(slack / 2, 0.06)
        # --- x band: covers seated leans, the two gap bands stay disjoint ----------------
        assert self.x_tol >= slack / 2 + 0.010, "x_tol must cover the seated lean shift"
        assert 2 * (self.slot_dx - self.x_tol) >= 0.02, "gap bands must be disjoint"
        assert self.slot_dx + self.x_tol <= self.spine_size[0] / 2, \
            "seat bands must lie within the comb span"
        # --- z window: accepts on-edge deck rests, rejects every perch -------------------
        assert self.z_lo <= self.plate_r * math.cos(self.lean_max) - 0.004, \
            "z window must accept the max-lean rest"
        assert self.z_hi >= self.plate_r + 0.005, "z window must accept upright rest"
        assert self.z_hi <= self.hinge_z + self.fin_th + self.plate_r - 0.02, \
            "z window must reject an on-edge plate standing ON the folded cover"
        assert self.z_hi <= post_top + self.plate_r - 0.05, \
            "z window must reject an on-edge plate perched on the erected fin tops"
        assert self.z_lo >= self.lip_size[2] + self.plate_r - 0.02 - 0.004 or \
            self.z_hi <= self.lip_size[2] + self.plate_r - 0.005, \
            "z window must reject an on-edge plate standing on the lip"
        # on-edge test: accepts the max lean, rejects flat poses
        assert math.sin(self.lean_max) + 0.05 <= self.axis_z_max <= 0.7
        # --- y band: covers the full physical roll range (spine <-> lip) -----------------
        r = self.plate_r
        spine_top = self.hinge_z + max(
            y * math.sin(th) + z * math.cos(th)
            for y, z in ((self.spine_yc + self.spine_size[1] / 2, 0.0),
                         (self.spine_yc + self.spine_size[1] / 2, self.spine_size[2]),
                         (self.spine_yc - self.spine_size[1] / 2, self.spine_size[2]))
            for th in (th_open,))
        y_min = post_front + math.sqrt(max(r**2 - (r - min(spine_top, 0.05))**2, 0.0))
        lip_in = self.lip_y - self.lip_size[1] / 2
        y_max = lip_in - math.sqrt(r**2 - (r - self.lip_size[2])**2)
        assert self.y_s - self.y_tol <= y_min - 0.008, "y band must cover the back rest"
        assert self.y_s + self.y_tol >= y_max + 0.008, "y band must cover the lip rest"
        # --- embodiment: rim pinch + stand overhang + reach ------------------------------
        assert self.plate_r * (1 + math.cos(self.lean_max)) + self.z_lo - self.plate_r \
            >= post_top + 0.04 or (self.z_lo + 2 * r - r) >= post_top + 0.04, \
            "seated top rim must stand proud of the fin posts"
        assert self.plate_t <= 0.075, "rim must fit a parallel jaw"
        assert self.plate_r - self.stand_r >= 0.025, "stand must leave a pinchable rim"
        assert self.plate_jxy <= self.stand_r - 0.010, "plate must stay on the stand"
        half_diag = math.hypot(self.deck_size[0] / 2, self.deck_size[1] / 2)
        d_sep = math.hypot(self.stand_pos[0] - self.rack_pos[0],
                           self.stand_pos[1] - self.rack_pos[1])
        assert d_sep - self.stand_jitter - self.rack_jitter - self.stand_r - half_diag \
            >= 0.02, "stand and chassis must never collide at reset"
        bx, by = self.franka_base[0], self.franka_base[1]
        reach_rack = math.hypot(bx - self.rack_pos[0], by - self.rack_pos[1]) \
            + self.rack_jitter + half_diag
        reach_stand = math.hypot(bx - self.stand_pos[0], by - self.stand_pos[1]) \
            + self.stand_jitter + self.stand_r + self.plate_r
        assert max(reach_rack, reach_stand) <= 0.72, "everything inside Franka reach"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("fold_rack")
class FoldRackScene(BaseScene):
    cfg: FoldRackSceneCfg

    def __init__(self, cfg: FoldRackSceneCfg | None = None) -> None:
        super().__init__(cfg or FoldRackSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # chassis FIRST: the comb's spawner authors a joint targeting .../Chassis.
            "chassis": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chassis",
                spawn=spawners["chassis"](
                    deck_size=c.deck_size, lip_y=c.lip_y, lip_size=c.lip_size,
                    slot_dx=c.slot_dx, mass=c.chassis_mass,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], c.deck_size[2] + 0.0015)),
            ),
            "comb": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Comb",
                spawn=spawners["comb"](
                    spine_size=c.spine_size, spine_yc=c.spine_yc, fin_y0=c.fin_y0,
                    fin_len=c.fin_len, fin_th=c.fin_th, fin_t_side=c.fin_t_side,
                    fin_t_center=c.fin_t_center, side_fin_x=c.side_fin_x,
                    cover_th=c.cover_th,
                    handle_size=c.handle_size, handle_center=c.handle_center,
                    hinge_y=c.hinge_y, hinge_z=c.hinge_z, joint_hi_deg=c.joint_hi_deg,
                    mass=c.comb_mass, com_y=c.com_y, com_z=c.com_z,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1] + c.hinge_y,
                         c.deck_size[2] + 0.0015 + c.hinge_z)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=sim_utils.CylinderCfg(
                    radius=c.stand_r, height=c.stand_h, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], c.stand_h / 2)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_t, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.15, angular_damping=0.30),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_m),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=WHITE),
                ),
                # spawn parked high, FLAT (identity): every in-episode flat pose then
                # differs from spawn by pure yaw, so a world-vertical force stays
                # vertical under either wrench-frame encoding.
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.30, 0.60, 0.30), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "enable_external_forces_every_iteration": True,
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
        self.chassis: RigidObject = env.iscene["chassis"]
        self.comb: RigidObject = env.iscene["comb"]
        self.stand: RigidObject = env.iscene["stand"]
        self.plate: RigidObject = env.iscene["plate"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.rack_yaw0 = torch.zeros(n, device=dev)  # sampled chassis yaw (readback)
        self.comb_th0 = torch.zeros(n, device=dev)  # sampled comb start angle (deg)
        self.deploy_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        self.deploy_cnt = torch.zeros(n, device=dev)
        self.seat_cnt = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, device=dev)

    def _yaw_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
        q[:, 0] = torch.cos(yaw / 2)
        q[:, 3] = torch.sin(yaw / 2)
        return q

    def _hinge_quat(self, base_quat: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """(M,4) comb quat = chassis quat composed with a rotation of theta about x."""
        from isaaclab.utils.math import quat_mul

        qx = torch.zeros(base_quat.shape[0], 4, device=base_quat.device)
        qx[:, 0] = torch.cos(theta / 2)
        qx[:, 1] = torch.sin(theta / 2)
        return quat_mul(base_quat, qx)

    def _seat_quat(self, frame_quat: torch.Tensor) -> torch.Tensor:
        """(M,4) plate quat standing on edge, plate axis along the frame's +x."""
        from isaaclab.utils.math import quat_mul

        m = frame_quat.shape[0]
        qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                            device=frame_quat.device).expand(m, 4)
        return quat_mul(frame_quat, qy90)

    def comb_pose_for(self, ch_pos: torch.Tensor, ch_quat: torch.Tensor,
                      theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Comb (pos, quat) consistent with the hinge at angle theta (rad) for the
        given chassis pose — used by reset and by probe writes (the whole linkage is
        always written together; the teleport-vs-linkage-depenetration trap)."""
        from isaaclab.utils.math import quat_apply

        m = ch_pos.shape[0]
        local = torch.tensor([0.0, self.cfg.hinge_y, self.cfg.hinge_z],
                             device=ch_pos.device).expand(m, 3)
        return ch_pos + quat_apply(ch_quat, local), self._hinge_quat(ch_quat, theta)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the chassis (xy jitter + FULL random yaw) with the
        comb FOLDED (small start angle that settles onto the closed stop, written as
        one consistent linkage), the stand (xy jitter), and the plate FLAT on the
        stand (xy jitter + free yaw); zero the latches and streaks."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(1, device=dev)  # burn the degenerate first post-seed draw

        # --- chassis ---
        yaw_r = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        self.rack_yaw0[env_ids] = yaw_r
        q_ch = self._yaw_quat(yaw_r)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.rack_pos[0]
        st[:, 1] = c.rack_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        st[:, 2] = c.deck_size[2] + 0.0015
        st[:, 0:3] += origin
        st[:, 3:7] = q_ch
        ch_pos = st[:, 0:3].clone()
        self.chassis.write_root_state_to_sim(st, env_ids)

        # --- comb: FOLDED, written consistently with the chassis linkage ---
        lo, hi = c.comb_start_deg
        th0 = math.radians(lo) + torch.rand(m, device=dev) * math.radians(hi - lo)
        self.comb_th0[env_ids] = torch.rad2deg(th0)
        pos_cb, q_cb = self.comb_pose_for(ch_pos, q_ch, th0)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pos_cb
        st[:, 3:7] = q_cb
        self.comb.write_root_state_to_sim(st, env_ids)

        # --- stand ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stand_pos[0]
        st[:, 1] = c.stand_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        st[:, 2] = c.stand_h / 2
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        stand_pos = st[:, 0:3].clone()
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- plate: FLAT on the stand, xy jitter + free yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = stand_pos[:, 0:2] \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jxy
        st[:, 2] = c.stand_h + c.plate_t / 2 + 0.002
        st[:, 3:7] = self._yaw_quat((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.plate.write_root_state_to_sim(st, env_ids)

        for buf in (self.deploy_latch, self.seat_latch, self.deploy_cnt,
                    self.seat_cnt, self.still_count):
            buf[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "chassis": self.chassis.data.root_state_w[env_ids].clone(),
            "comb": self.comb.data.root_state_w[env_ids].clone(),
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "rack_yaw0": self.rack_yaw0[env_ids].clone(),
            "comb_th0": self.comb_th0[env_ids].clone(),
            "deploy_latch": self.deploy_latch[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "deploy_cnt": self.deploy_cnt[env_ids].clone(),
            "seat_cnt": self.seat_cnt[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.chassis.write_root_state_to_sim(state["chassis"], env_ids)
        self.comb.write_root_state_to_sim(state["comb"], env_ids)
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.rack_yaw0[env_ids] = state["rack_yaw0"]
        self.comb_th0[env_ids] = state["comb_th0"]
        self.deploy_latch[env_ids] = state["deploy_latch"]
        self.seat_latch[env_ids] = state["seat_latch"]
        self.deploy_cnt[env_ids] = state["deploy_cnt"]
        self.seat_cnt[env_ids] = state["seat_cnt"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A fold-flat wooden drying rack stands on the floor: a low deck with a "
            f"thin retaining lip along its front edge, and a hinged FIN COMB that "
            f"starts FOLDED DOWN flat over the deck like a closed lid — a spine bar "
            f"along the hinge, three fins (one BLUE, one wide gray in the middle, one "
            f"YELLOW) under a full-width top cover plate that ROOFS the fin gaps "
            f"while folded, and a small red HANDLE tab standing up from the cover. "
            f"While folded there is no slot anywhere: the comb bodily covers the "
            f"deck, and nothing can be slotted into or through the closed lid. The "
            f"hinge is over-center bistable: gravity holds the comb closed, and once "
            f"it is rotated past about {c.overcenter_deg:.0f} degrees gravity carries "
            f"it the rest of the way and parks it firmly OPEN at its "
            f"{c.joint_hi_deg:.0f}-degree stop — no one needs to hold it. Erecting "
            f"the comb stands the three fins up as posts and CREATES two open-top "
            f"gaps between them, each {c.gap * 1000:.0f} mm wide: the gap beside the "
            f"BLUE fin is the blue slot, the gap beside the YELLOW fin the yellow "
            f"slot (matching color tabs sit on the deck in front of each). About "
            f"{abs(c.stand_pos[1] - c.rack_pos[1]) * 100:.0f} cm away a round white "
            f"plate ({2 * c.plate_r * 1000:.0f} mm across, {c.plate_t * 1000:.0f} mm "
            f"thick) lies FLAT on a low pedestal stand, its rim overhanging all "
            f"around.\n"
            f"Goal: put the white plate away in the BLUE slot of the rack — standing "
            f"ON EDGE in the blue gap, leaning on the erected posts above the deck, "
            f"at rest, with the comb still open. The order is forced by physics: the "
            f"slot does not exist until the comb has been erected past over-center "
            f"(push or pull the red handle; either direction of approach works, and "
            f"gravity finishes and holds the motion once past the tipping angle), so "
            f"the comb must be opened FIRST and the plate inserted SECOND. A plate "
            f"laid on the folded cover, left flat on the deck, perched on the fin "
            f"tops or on the lip, standing anywhere while the comb is closed, or "
            f"put in the YELLOW gap does not finish the task (the yellow gap earns "
            f"only partial credit). The rack's position and heading change every "
            f"episode — read the fin colors and deck tabs to find the blue gap."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The drying rack's fin comb starts folded flat, covering the deck. First "
            "erect the comb by its red handle past the tipping angle so gravity "
            "parks it open and the fin gaps appear, then take the white plate from "
            "the pedestal stand and stand it on edge in the BLUE fin gap. Finish "
            "with the comb open and the plate seated at rest in the blue slot."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body, ref) -> torch.Tensor:
        """(N,3) body center in `ref`'s body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - ref.data.root_pos_w
        return quat_apply_inverse(ref.data.root_quat_w, rel)

    def comb_angle(self) -> torch.Tensor:
        """(N,) hinge angle in RADIANS (0 = folded, joint_hi = open), from the
        chassis->comb relative quaternion (pure x rotation)."""
        from isaaclab.utils.math import quat_inv, quat_mul

        q_rel = quat_mul(quat_inv(self.chassis.data.root_quat_w),
                         self.comb.data.root_quat_w)
        return 2.0 * torch.atan2(q_rel[:, 1], q_rel[:, 0])

    def deployed(self) -> torch.Tensor:
        """(N,) bool: the comb is OPEN (erected past deploy_min_deg)."""
        return self.comb_angle() >= math.radians(self.cfg.deploy_min_deg)

    def folded(self) -> torch.Tensor:
        """(N,) bool: the comb is CLOSED (within fold_max_deg of the folded stop)."""
        return self.comb_angle() <= math.radians(self.cfg.fold_max_deg)

    def _on_edge(self) -> torch.Tensor:
        """(N,) bool: plate axis within axis_z_max of horizontal (standing on edge)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        axis = quat_apply(self.plate.data.root_quat_w, ez)
        return axis[:, 2].abs() <= self.cfg.axis_z_max

    def seat_bands(self, cx: float) -> torch.Tensor:
        """(N,) bool: the pure GEOMETRIC seat bands for the gap at chassis-frame x=cx
        (exposed separately so smoke can prove the deployed gate is load-bearing)."""
        c = self.cfg
        loc = self._local(self.plate, self.chassis)
        return ((loc[:, 0] - cx).abs() <= c.x_tol) \
            & ((loc[:, 1] - c.y_s).abs() <= c.y_tol) \
            & (loc[:, 2] >= c.z_lo) & (loc[:, 2] <= c.z_hi) \
            & self._on_edge()

    def seated(self, slot: str) -> torch.Tensor:
        """(N,) bool: plate seated ON EDGE in the `slot` gap — the gap only EXISTS
        while the comb is deployed (the folded comb covers this volume; the gate is
        what makes pre-erection arrangements worthless)."""
        sign = BLUE_GAP_SIGN if slot == "blue" else -BLUE_GAP_SIGN
        return self.seat_bands(sign * self.cfg.slot_dx) & self.deployed()

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.plate, self.comb, self.chassis):
            ok &= (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED settle_steps_min consecutive steps
        (counter-latch in post_step — teleport writes and pushes reset it)."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: comb ever OPEN for latch_steps
        consecutive slow steps; plate ever seated in EITHER gap for latch_steps
        consecutive slow steps (a fly-through never latches)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        comb_slow = (self.comb.data.root_ang_vel_w.norm(dim=-1) < 2.0 * c.settle_ang)
        self.deploy_cnt = (self.deploy_cnt + 1.0) * (self.deployed() & comb_slow).float()
        self.deploy_latch = torch.maximum(
            self.deploy_latch, (self.deploy_cnt >= c.latch_steps).float())
        plate_slow = (self.plate.data.root_lin_vel_w.norm(dim=-1) < 2.0 * c.settle_lin) \
            & (self.plate.data.root_ang_vel_w.norm(dim=-1) < 2.0 * c.settle_ang)
        seated_any = self.seated("blue") | self.seated("yellow")
        self.seat_cnt = (self.seat_cnt + 1.0) * (seated_any & plate_slow).float()
        self.seat_latch = torch.maximum(
            self.seat_latch, (self.seat_cnt >= c.latch_steps).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: plate seated in the BLUE gap with the comb OPEN, everything
        persistently still."""
        return self.seated("blue") & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.35 comb ever OPEN + 0.20 plate ever seated in
        either gap (both latched — credit never evaporates); capped 0.55; 1.0 iff
        success(). Null policy ~0 (the comb starts folded and gravity-held; no gap
        exists at reset)."""
        base = (0.35 * self.deploy_latch + 0.20 * self.seat_latch).clamp(0.0, 0.55)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="fold_rack", robot="null",
                                      env_spacing=3.0))
