"""SortingTowerScene — gauge-sort three graded discs down a keyed tower shaft
(sim_gen task `two_robot_pick_cube_i165`).

Derived from maniskill/two_robot_pick_cube, but STRATEGICALLY different: the seed's
whole skill is a two-arm relay — arm A picks a cube and lifts it to the midpoint,
arm B takes it and holds it at a floating elevated goal. Success there is a pose
match reached by carrying, and the implicit sequencing (A before B) lives in the
agents. Here there is ONE implied arm, NO handover and NO floating goal: each of
three GREEN discs must come to rest at its own prescribed height INSIDE a gauge
tower — heights that are reached not by holding but by SIZE-KEYED SEATING. The
tower's internal funnels admit a disc exactly as deep as its diameter allows, so
"deliver each object to its elevated goal" becomes "insert the discs through the
one top mouth in ASCENDING size order": the ordering the seed put into two
cooperating agents is here PHYSICALLY FORCED by the geometry — a seated disc
blocks the shaft, so any larger-first order strands the rest on top (smoke proves
it with real drops). A RED disc (same shape family, size sampled per episode)
must be left alone: red inside the shaft at judging time fails the episode.

The scene is fully procedural (compound-spawner pattern; children of one body
never self-collide): the tower is ONE kinematic compound — slab, three square
tube sections of decreasing width, three 45-degree four-plate funnels (mouth,
mid, lower), and 45-degree corner chamfer posts under each interior throat that
octagonalize the corners (a square throat's diagonal would otherwise pass a
near-vertical disc one size too big — the chamfers close that leak, audited in
`__post_init__`). Discs are plain rigid cylinders.

Per-episode randomization (readback-verified by smoke): tower yaw FREE
(+/-180 deg) + xy jitter, the 4 slab slots are permuted over {small, mid, large,
red}, WHICH size the red distractor is (three red discs exist; the two inactive
park in a ground depot), per-disc xy jitter + free yaw. All discrete draws derive
from torch.rand (the first torch.randint after manual_seed is degenerate on this
stack).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.22 per green disc EVER seated at its own band (flat, on the tower axis, at its
       keyed height, tower frame) — latched, 3 x 0.22 = 0.66 cap.
Exactly 1.0 iff success(): all three green discs seated at their bands, NO red
disc inside the shaft, everything settled and finite. Null policy ~0. The seed's
strategy — carry each object straight to the destination, no size reasoning
(largest first / any descending order) — seats only the first disc and strands
the rest on top of it: ~0.22, never success.

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


# ----- custom compound spawner ------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}

_SQ2 = math.sqrt(2.0)


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


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _obox(stage, path: str, *, center, size, quat, color, collide: Callable):
    """Oriented box child (funnel plates / chamfer posts). `quat` is wxyz."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    w, x, y, z = (float(v) for v in quat)
    xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _tube(stage, base: str, *, b, t, z0, z1, color, collide: Callable) -> None:
    """Four walls of a square tube: interior half-width b, thickness t, z span."""
    _span(stage, f"{base}_xp", x=(b, b + t), y=(-b - t, b + t), z=(z0, z1),
          color=color, collide=collide)
    _span(stage, f"{base}_xn", x=(-b - t, -b), y=(-b - t, b + t), z=(z0, z1),
          color=color, collide=collide)
    _span(stage, f"{base}_yp", x=(-b, b), y=(b, b + t), z=(z0, z1),
          color=color, collide=collide)
    _span(stage, f"{base}_yn", x=(-b, b), y=(-b - t, -b), z=(z0, z1),
          color=color, collide=collide)


def _funnel(stage, base: str, *, bi, bo, z0, z1, t, color, collide: Callable) -> None:
    """Four 45-degree plates narrowing (bo at z1) -> (bi at z0). Each plate's INNER
    surface contains the slope line from (bi, z0) to (bo, z1); the plate's lower end
    stops 1 mm past the throat (so it cannot narrow the tube below), the upper end
    overlaps 7 mm into the wall band of the tube above."""
    rise = z1 - z0
    ell = rise * _SQ2 + 0.008  # slope length + (1 mm low, 7 mm high) extension
    shift = 0.003              # slide the box +3 mm up-slope: asymmetric extension
    wid = 2 * (bo + t)
    half = math.cos(math.pi / 8), math.sin(math.pi / 8)  # cos/sin of 22.5 deg
    mx, mz = (bi + bo) / 2, (z0 + z1) / 2
    # +x / -x plates: rotate about y by -45 / +45 deg
    for tag, sgn in (("xp", 1.0), ("xn", -1.0)):
        # slope unit s = (sgn, 0, 1)/sqrt2; outward-down normal n = (sgn, 0, -1)/sqrt2
        cx = sgn * mx + sgn * shift / _SQ2 + sgn * t / (2 * _SQ2)
        cz = mz + shift / _SQ2 - t / (2 * _SQ2)
        quat = (half[0], 0.0, -sgn * half[1], 0.0)
        _obox(stage, f"{base}_{tag}", center=(cx, 0.0, cz), size=(ell, wid, t),
              quat=quat, color=color, collide=collide)
    # +y / -y plates: rotate about x by +45 / -45 deg
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        cy = sgn * mx + sgn * shift / _SQ2 + sgn * t / (2 * _SQ2)
        cz = mz + shift / _SQ2 - t / (2 * _SQ2)
        quat = (half[0], sgn * half[1], 0.0, 0.0)
        _obox(stage, f"{base}_{tag}", center=(0.0, cy, cz), size=(wid, ell, t),
              quat=quat, color=color, collide=collide)


def _chamfers(stage, base: str, *, c, z0, z1, t, color, collide: Callable) -> None:
    """Four 45-deg corner posts below a throat: vertical boxes yawed 45 deg whose
    inner faces cut each corner at diagonal reach `c` (square -> octagon), closing
    the diagonal corridor a near-vertical oversize disc could otherwise slip
    through."""
    half = math.cos(math.pi / 8), math.sin(math.pi / 8)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        d = c + t / 2
        yaw_half = a / 2
        quat = (math.cos(yaw_half), 0.0, 0.0, math.sin(yaw_half))
        _obox(stage, f"{base}_{k}",
              center=(d * math.cos(a), d * math.sin(a), (z0 + z1) / 2),
              size=(t, 0.036, z1 - z0), quat=quat, color=color, collide=collide)
    del half


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gauge tower at `prim_path`: KINEMATIC compound (a fixture; no
    joints, so a kinematic root teleports cleanly at reset). Local frame: origin at
    the shaft axis on the slab TOP (z=0)."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    S = c.slab_half
    t = c.wall_t

    # --- slab -----------------------------------------------------------------------
    _span(stage, f"{prim_path}/slab", x=(-S, S), y=(-S, S), z=(-c.slab_t, 0.0),
          color=c.slab_color, collide=collide)

    # --- shaft stack (bottom to top) --------------------------------------------------
    _tube(stage, f"{prim_path}/tube1", b=c.b1, t=t, z0=0.0, z1=c.f2z0,
          color=c.wall_color, collide=collide)
    _funnel(stage, f"{prim_path}/funnel2", bi=c.b1, bo=c.b2, z0=c.f2z0, z1=c.f2z1,
            t=t, color=c.funnel_color, collide=collide)
    _tube(stage, f"{prim_path}/tube2", b=c.b2, t=t, z0=c.f2z1, z1=c.f3z0,
          color=c.wall_color, collide=collide)
    _funnel(stage, f"{prim_path}/funnel3", bi=c.b2, bo=c.b3, z0=c.f3z0, z1=c.f3z1,
            t=t, color=c.funnel_color, collide=collide)
    _tube(stage, f"{prim_path}/tube3", b=c.b3, t=t, z0=c.f3z1, z1=c.mz0,
          color=c.wall_color, collide=collide)
    _funnel(stage, f"{prim_path}/mouth", bi=c.b3, bo=c.bm, z0=c.mz0, z1=c.mz1,
            t=t, color=c.mouth_color, collide=collide)

    # --- corner chamfers under the two interior throats -------------------------------
    _chamfers(stage, f"{prim_path}/cham2", c=c.c2, z0=c.f2z0 - 0.020, z1=c.f2z0,
              t=t, color=c.funnel_color, collide=collide)
    _chamfers(stage, f"{prim_path}/cham3", c=c.c3, z0=c.f3z0 - 0.020, z1=c.f3z0,
              t=t, color=c.funnel_color, collide=collide)

    # --- kinematic root (fixture: heavy, never moves, teleports cleanly at reset) -----
    UsdPhysics.RigidBodyAPI.Apply(root)
    api = UsdPhysics.RigidBodyAPI(root)
    api.CreateKinematicEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.tower_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSleepThresholdAttr(0.0)
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/towermat"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=c.tower_mu_s, dynamic_friction=c.tower_mu_d, restitution=0.0,
        friction_combine_mode="average"))
    bind_physics_material(prim_path, mat_path)
    return root


def _spawner_class() -> Any:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tower" not in _SPAWNER_CACHE:

        @configclass
        class TowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tower)
            slab_half: float = 0.30
            slab_t: float = 0.04
            wall_t: float = 0.008
            b1: float = 0.019
            b2: float = 0.029
            b3: float = 0.040
            bm: float = 0.058
            f2z0: float = 0.034
            f2z1: float = 0.044
            f3z0: float = 0.082
            f3z1: float = 0.093
            mz0: float = 0.133
            mz1: float = 0.151
            c2: float = 0.021
            c3: float = 0.030
            tower_mass: float = 25.0
            contact_offset: float = 0.0015
            tower_mu_s: float = 0.50
            tower_mu_d: float = 0.45
            slab_color: tuple = (0.45, 0.42, 0.36)
            wall_color: tuple = (0.36, 0.40, 0.48)
            funnel_color: tuple = (0.55, 0.56, 0.60)
            mouth_color: tuple = (0.72, 0.60, 0.20)

        _SPAWNER_CACHE["tower"] = TowerSpawnerCfg
    return _SPAWNER_CACHE["tower"]


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SortingTowerSceneCfg(BaseCfg):
    """Config for `SortingTowerScene`. The keying contract is asserted in
    `__post_init__`: every disc passes every throat ABOVE its own seat flat-on
    (>= 4 mm side clearance, including the chamfered diagonals), no disc passes its
    own seat (>= 8 mm total rim overlap), no near-vertical disc slips a throat
    diagonally (octagon max chord < the next disc diameter), perch states (a disc
    resting on a seated larger disc) sit far outside the z band, tubes are wide
    enough that a tipped disc can right itself, and slots/depot clear the tower."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    xy_tol: float = tunable(0.012)      # seated: disc centre radial offset from the shaft axis (m)
    z_tol: float = tunable(0.008)       # seated: |z - z_seat| tolerance (m)
    align_max_deg: float = tunable(15.0)  # seated: disc axis within this of the tower axis
    settle_lin: float = tunable(0.05)   # max disc |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.80)   # max disc |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)     # tower yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)    # tower xy jitter (+/- m)
    randomize_slots: bool = tunable(True)   # permute the 4 discs over the 4 slab slots
    randomize_red: bool = tunable(True)     # sample WHICH size the red distractor is
    disc_jitter: float = tunable(0.012)     # per-disc xy jitter in its slot (+/- m)

    # --- info: tower geometry (local frame: origin at shaft axis on slab top) --------------------
    base_z: float = info(0.040)         # kinematic root height: slab bottom flush on ground
    slab_half: float = info(0.30)
    slab_t: float = info(0.04)
    wall_t: float = info(0.008)
    b1: float = info(0.019)             # tube1 interior half-width (38 mm)
    b2: float = info(0.029)             # tube2 interior half-width (58 mm)
    b3: float = info(0.040)             # tube3 interior half-width (80 mm)
    bm: float = info(0.058)             # mouth top half-width (116 mm)
    f2z0: float = info(0.034)           # lower funnel throat plane
    f2z1: float = info(0.044)
    f3z0: float = info(0.082)           # mid funnel throat plane
    f3z1: float = info(0.093)
    mz0: float = info(0.133)
    mz1: float = info(0.151)            # mouth rim
    c2: float = info(0.021)             # chamfer diagonal reach under throat 2
    c3: float = info(0.030)             # chamfer diagonal reach under throat 3
    tower_mass: float = info(25.0)
    # --- info: discs (diameter, height, mass); index 0 = small ... 2 = large ---------------------
    disc_d: tuple = info((0.028, 0.048, 0.068))
    disc_h: tuple = info((0.016, 0.018, 0.020))
    disc_m: tuple = info((0.05, 0.10, 0.18))
    green: tuple = info((0.10, 0.62, 0.18))
    red: tuple = info((0.82, 0.10, 0.10))
    # --- info: slots / depot ---------------------------------------------------------------------
    slot_r: float = info(0.21)          # 4 slab slots around the tower
    slot_az_deg: tuple = info((45.0, 135.0, 225.0, 315.0))
    depot_xy: tuple = info(((0.85, 0.70), (0.85, -0.70)))  # ground depot, world (per env)
    # --- info: materials -------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    disc_mu_s: float = info(0.40)       # must slide down the 45 deg funnels: mu < tan(45)
    disc_mu_d: float = info(0.35)
    tower_mu_s: float = info(0.50)
    tower_mu_d: float = info(0.45)
    # --- info: rubric weights --------------------------------------------------------------------
    w_seat: float = info(0.22)
    score_cap: float = info(0.66)

    def __post_init__(self) -> None:
        c = self
        r = [d / 2 for d in c.disc_d]
        # seat heights (derived): floor rest; rim-on-45-deg-bevel rests
        self.z_seat = (
            c.disc_h[0] / 2,
            c.f2z0 + (r[1] - c.b1) + c.disc_h[1] / 2,
            c.f3z0 + (r[2] - c.b2) + c.disc_h[2] / 2,
        )
        # flat pass clearances at every throat above each disc's seat
        assert c.b1 - r[0] >= 0.004, "small disc must pass tube1/throat2 flats"
        assert c.c2 - r[0] >= 0.005, "small disc must pass throat-2 chamfered diagonals"
        assert c.b2 - r[1] >= 0.004, "mid disc must pass tube2/throat3 flats"
        assert c.c3 - r[1] >= 0.005, "mid disc must pass throat-3 chamfered diagonals"
        assert c.b3 - r[2] >= 0.005, "large disc must pass tube3"
        assert c.bm - r[2] >= 0.015, "mouth must catch the large disc generously"
        # seating: each disc's rim overlaps its own throat
        assert c.disc_d[1] - 2 * c.b1 >= 0.008, "mid disc must NOT pass throat 2"
        assert c.disc_d[2] - 2 * c.b2 >= 0.008, "large disc must NOT pass throat 3"
        # seat contact stays inside the funnel band
        assert 0.0 < (r[1] - c.b1) < (c.f2z1 - c.f2z0), "mid seat inside funnel 2"
        assert 0.0 < (r[2] - c.b2) < (c.f3z1 - c.f3z0), "large seat inside funnel 3"
        # diagonal-escape audit: octagon max chord (2c) < the excluded disc diameter
        assert 2 * c.c2 <= c.disc_d[1] - 0.004, "throat-2 diagonal must reject the mid disc"
        assert 2 * c.c3 <= c.disc_d[2] - 0.006, "throat-3 diagonal must reject the large disc"
        assert c.c2 >= c.b1 and c.c3 >= c.b2, "chamfers must not narrow the flats"
        # a tipped disc can right itself inside its tube (swept diagonal < tube width)
        for k, b in ((0, c.b1), (1, c.b2), (2, c.b3)):
            assert math.hypot(c.disc_d[k], c.disc_h[k]) < 2 * b - 0.004, \
                f"disc {k} must be able to flop flat in its tube"
        # perch states sit far outside the band: disc k resting on seated disc j > k
        for k in range(3):
            for j in range(k + 1, 3):
                perch = self.z_seat[j] + c.disc_h[j] / 2 + c.disc_h[k] / 2
                assert perch - self.z_seat[k] > 3 * c.z_tol, \
                    f"perch of disc {k} on {j} must clear the z band"
        # disc k cannot slip past a seated larger disc j (annular gap < diameter)
        for j, b in ((1, c.b2), (2, c.b3)):
            gap = b - c.disc_d[j] / 2
            assert gap < min(c.disc_d[:j]) / 2, f"annular gap at seated disc {j} traps smaller discs"
        # bands are distinct beyond tolerance
        assert self.z_seat[1] - self.z_seat[0] > 4 * c.z_tol
        assert self.z_seat[2] - self.z_seat[1] > 4 * c.z_tol
        # embodiment: parallel jaw span (80 mm) takes the largest disc with margin
        assert c.disc_d[2] <= 0.072, "largest disc must fit an 80 mm jaw with margin"
        # slots on the slab, clear of the tower mouth footprint and of each other
        guard = c.bm + c.wall_t
        assert c.slot_r - r[2] - c.disc_jitter > guard + 0.030, "slots clear the tower"
        assert c.slot_r + r[2] + c.disc_jitter < c.slab_half - 0.010, "slots stay on the slab"
        pitch = 2 * c.slot_r * math.sin(math.radians(45.0))
        assert pitch > c.disc_d[2] + 2 * c.disc_jitter + 0.010, "adjacent slots never overlap"
        # depot clears the slab and the env cell
        for dx, dy in c.depot_xy:
            assert math.hypot(dx, dy) > c.slab_half * _SQ2 + 0.10, "depot off the slab"
            assert abs(dx) < 0.95 and abs(dy) < 0.95, "depot inside the env cell"
        assert abs(3 * c.w_seat - c.score_cap) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sorting_tower")
class SortingTowerScene(BaseScene):
    cfg: SortingTowerSceneCfg

    def __init__(self, cfg: SortingTowerSceneCfg | None = None) -> None:
        super().__init__(cfg or SortingTowerSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        tower_spawn = _spawner_class()(
            slab_half=c.slab_half, slab_t=c.slab_t, wall_t=c.wall_t,
            b1=c.b1, b2=c.b2, b3=c.b3, bm=c.bm,
            f2z0=c.f2z0, f2z1=c.f2z1, f3z0=c.f3z0, f3z1=c.f3z1,
            mz0=c.mz0, mz1=c.mz1, c2=c.c2, c3=c.c3,
            tower_mass=c.tower_mass, contact_offset=c.contact_offset,
            tower_mu_s=c.tower_mu_s, tower_mu_d=c.tower_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.disc_mu_s, dynamic_friction=c.disc_mu_d, restitution=0.0,
            friction_combine_mode="average")

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "tower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tower", spawn=tower_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.base_z))),
        }
        for k in range(3):
            for fam, col in (("cargo", c.green), ("red", c.red)):
                out[f"{fam}_{k}"] = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{fam.capitalize()}_{k}",
                    spawn=sim_utils.CylinderCfg(
                        radius=c.disc_d[k] / 2, height=c.disc_h[k], axis="Z",
                        mass_props=sim_utils.MassPropertiesCfg(mass=c.disc_m[k]),
                        rigid_props=rigid, collision_props=coll,
                        physics_material=mat,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col)),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(1.2 + 0.2 * k, 0.8 if fam == "cargo" else -0.8, 0.05)))
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
        self.tower: RigidObject = env.iscene["tower"]
        self.cargo: list[RigidObject] = [env.iscene[f"cargo_{k}"] for k in range(3)]
        self.reds: list[RigidObject] = [env.iscene[f"red_{k}"] for k in range(3)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # episode readbacks (verified by smoke)
        self.slot_of = torch.zeros(n, 4, dtype=torch.long, device=dev)  # disc -> slot az idx
        self.active_red = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._seat = torch.zeros(n, 3, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the tower (free yaw + xy jitter), permute the four
        discs {small, mid, large, red} over the four slab slots, sample WHICH size
        the red distractor is (the other two reds park in the ground depot), clear
        latches. All discrete draws derive from torch.rand."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_h = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.base_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_h
        self.tower.write_root_state_to_sim(st, env_ids)

        if c.randomize_slots:
            perm = torch.argsort(torch.rand(m, 4, device=dev), dim=1)
        else:
            perm = torch.arange(4, device=dev).expand(m, 4).contiguous()
        self.slot_of[env_ids] = perm
        act = (torch.rand(m, device=dev) * 3).long().clamp(max=2)
        if not c.randomize_red:
            act = torch.zeros(m, dtype=torch.long, device=dev)
        self.active_red[env_ids] = act

        az_all = torch.tensor([math.radians(a) for a in c.slot_az_deg], device=dev)

        def slot_state(slot_idx: torch.Tensor, h: float) -> torch.Tensor:
            az = az_all[slot_idx]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.slot_r * torch.cos(az) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.disc_jitter
            loc[:, 1] = c.slot_r * torch.sin(az) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.disc_jitter
            loc[:, 2] = h / 2 + 0.002
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            return s

        for k in range(3):
            self.cargo[k].write_root_state_to_sim(slot_state(perm[:, k], c.disc_h[k]), env_ids)

        for k in range(3):
            s_slot = slot_state(perm[:, 3], c.disc_h[k])
            on_slot = act == k
            dxy = c.depot_xy[k % 2]
            park = torch.tensor([dxy[0], dxy[1] + 0.12 * (k // 2), c.disc_h[k] / 2 + 0.002],
                                device=dev)
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = torch.where(on_slot.unsqueeze(-1), s_slot[:, 0:3], origin + park)
            s[:, 3] = 1.0
            s[:, 3:7] = torch.where(on_slot.unsqueeze(-1), s_slot[:, 3:7], s[:, 3:7])
            self.reds[k].write_root_state_to_sim(s, env_ids)

        self._seat[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "tower": self.tower.data.root_state_w[env_ids].clone(),
            "slot_of": self.slot_of[env_ids].clone(),
            "active_red": self.active_red[env_ids].clone(),
            "seat": self._seat[env_ids].clone(),
        }
        for k in range(3):
            out[f"cargo_{k}"] = self.cargo[k].data.root_state_w[env_ids].clone()
            out[f"red_{k}"] = self.reds[k].data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tower.write_root_state_to_sim(state["tower"], env_ids)
        for k in range(3):
            self.cargo[k].write_root_state_to_sim(state[f"cargo_{k}"], env_ids)
            self.reds[k].write_root_state_to_sim(state[f"red_{k}"], env_ids)
        self.slot_of[env_ids] = state["slot_of"]
        self.active_red[env_ids] = state["active_red"]
        self._seat[env_ids] = state["seat"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        d = [f"{1000 * v:.0f}" for v in c.disc_d]
        return (
            f"A GAUGE TOWER stands at the centre of a square counter slab: a vertical "
            f"shaft with one golden funnel MOUTH on top (its only opening, rim about "
            f"{1000 * (c.base_z + c.mz1):.0f} mm above the ground). Inside, the shaft "
            f"narrows in stages: two hidden internal funnels admit a disc exactly as "
            f"deep as its diameter allows, so a dropped disc comes to rest at the one "
            f"depth keyed to its size — small discs fall to the shaft floor, larger "
            f"ones seat higher up. A disc that seats WEDGES the shaft shut: nothing "
            f"can pass a seated disc, so discs must go in from SMALLEST to LARGEST — "
            f"any larger disc inserted early strands the smaller ones on top of it, "
            f"where they do not count.\n"
            f"On the slab around the tower lie four flat discs: three GREEN cargo "
            f"discs of graded diameters ({d[0]}, {d[1]} and {d[2]} mm) and one RED "
            f"disc (its size varies by episode and can match any of the green ones — "
            f"tell them apart by COLOUR, not shape). Which disc lies at which slot, "
            f"the red disc's size, and the whole tower's heading vary by episode.\n"
            f"Goal: drop the three GREEN discs, one at a time, through the tower's "
            f"top mouth in ascending size order — smallest first, largest last — so "
            f"that each ends up resting FLAT and centred at its own keyed depth "
            f"inside the shaft. Leave the RED disc alone: any red disc inside the "
            f"shaft at the end fails the episode. Release each disc centred over the "
            f"mouth; the funnels guide it down. The episode ends settled: all three "
            f"green discs seated at their depths, red outside, nothing moving."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the three green discs into the gauge tower's top mouth in "
            "ascending size order — smallest first, largest last — so each seats "
            "flat at its own depth inside the shaft. Do not put the red disc in."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _tower_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tower.data.root_quat_w,
                                  pos_w - self.tower.data.root_pos_w)

    def tower_world(self, local_xyz) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        loc = torch.as_tensor(local_xyz, device=self.env.device, dtype=torch.float)
        loc = loc.expand(self.env.num_envs, 3)
        return self.tower.data.root_pos_w + quat_apply(self.tower.data.root_quat_w, loc)

    def _up_z(self, body) -> torch.Tensor:
        """(N,) world-z component of a disc's local +z axis (tower is upright)."""
        q = body.data.root_quat_w
        w, x, y, z = q.unbind(-1)
        return 1.0 - 2.0 * (x * x + y * y)

    def seated(self) -> torch.Tensor:
        """(N, 3) bool, geometric: green disc k resting FLAT (axis within
        `align_max_deg` of vertical), centred on the shaft axis (radial < `xy_tol`)
        at its keyed height (|z - z_seat[k]| < `z_tol`), tower frame."""
        c = self.cfg
        cos_max = math.cos(math.radians(c.align_max_deg))
        cols = []
        for k in range(3):
            loc = self._tower_local(self.cargo[k].data.root_pos_w)
            near = loc[:, :2].norm(dim=-1) < c.xy_tol
            band = (loc[:, 2] - self.z_seat_t[k]).abs() < c.z_tol
            flat = self._up_z(self.cargo[k]).abs() > cos_max
            cols.append(near & band & flat)
        return torch.stack(cols, dim=1)

    @property
    def z_seat_t(self):
        return self.cfg.z_seat

    def red_inside(self) -> torch.Tensor:
        """(N,) bool: ANY red disc inside the shaft column (tower frame: within the
        mouth's outer square, above the slab top)."""
        c = self.cfg
        guard = c.bm + c.wall_t
        out = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for k in range(3):
            loc = self._tower_local(self.reds[k].data.root_pos_w)
            out |= (loc[:, 0].abs() < guard) & (loc[:, 1].abs() < guard) \
                & (loc[:, 2] > 0.002) & (loc[:, 2] < 0.40)
        return out

    def settled(self) -> torch.Tensor:
        """(N,) bool: every disc slow (linear and angular)."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (*self.cargo, *self.reds):
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def _finite(self) -> torch.Tensor:
        ps = [self.tower.data.root_pos_w] + [b.data.root_pos_w for b in (*self.cargo, *self.reds)]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    # ----- step-coupled latches (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def _update_latches(self) -> None:
        self._seat |= self.seated() & self._finite().unsqueeze(-1)

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three green discs seated at their keyed bands, no red disc
        inside the shaft, everything settled and finite — all live physical
        outcomes. A disc got to its band only by descending the shaft through every
        wider stage above it; the keying (and hence the smallest-first order) is
        enforced by the throats themselves (smoke proves the wrong orders strand
        discs outside their bands)."""
        self._update_latches()
        return self.seated().all(dim=1) & ~self.red_inside() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.22 per green disc EVER seated at its own band
        (latched), capped at 0.66; exactly 1.0 iff success() holds live. Doing
        nothing scores ~0; the seed's carry-to-destination strategy (no size
        reasoning: largest/descending order) seats only its first disc, ~0.22."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._seat.float().sum(dim=1)).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="sorting_tower", robot="null"))
