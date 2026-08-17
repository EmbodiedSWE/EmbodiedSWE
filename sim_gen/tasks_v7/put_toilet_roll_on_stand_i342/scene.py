"""RailRingScene — slide the captive ring along the bent guide rail, around two flat
corners and over the far downturn, so it drops down the end post and lands flat on the
landing plate, encircling the post (sim_gen task `put_toilet_roll_on_stand_i342`).

Derived from rlbench/put_toilet_roll_on_stand, but STRATEGICALLY different: the seed is
pick-and-place-in-free-space — grab a free roll, carry it through open air, and slide
its core sideways onto a fixed cantilever peg; any 6-DOF transport that ends on the peg
works, and the approach direction is unconstrained. Here the ring instead starts
ALREADY on its "peg": it is topologically CAPTIVE on a bent rail whose start end is
sealed by a ball cap wider than the ring's bore, and whose far end is the goal post.
Free-space transport does not exist for this object — the only way the ring can reach
the stand is to TRAVERSE the rail's full course:

  - a long straight run at 190 mm height,
  - a chamfered 90-degree corner (two 45-degree turns) in the horizontal plane,
  - a second straight run,
  - a 45-degree downturn onto a vertical end post,
  - a free drop down the post onto the green landing plate.

A solver needs a different PLAN (a monotone guided traversal with in-contact corner
negotiation and a committed release over the crest — not grasp/carry/insert) and
different CODE (path-following control of a body that is never free, plus a
delivered-around-the-post predicate), not different constants. The seed's strategy
literally cannot be expressed: there is nothing to pick up (the ring cannot leave the
rail) and nothing to insert (the ring is already threaded from the first frame).

success(): the ring rests FLAT on the landing plate encircling the post — center
within `deliver_xy_tol` of the post axis and within `deliver_z_tol` of the flat-rest
height `LAND_Z` (a ring standing anywhere on the post above the plate, leaning on the
post, or resting anywhere off the plate reads outside), axis within
`deliver_tilt_max_deg` of vertical, and SETTLED (sustained stillness). Nothing is
welded or held; the final state is a free rest.

score() is graded and latched (credit never evaporates): a running-max of arc-length
progress along the rail's course polyline, gated by staying within `prog_gate_dist` of
the course (a teleported-away ring gains nothing), normalized from the episode's own
start station to the course end and scaled to `prog_cap`; 1.0 iff success(). The null
policy scores ~0 (the ring spawns hanging at its start station; the first
`prog_deadband` of arc-length earns nothing).

Per-episode randomization (readback-verified in smoke): the whole rail fixture is
re-placed every reset (xy jitter + yaw), and the ring's start station along the first
straight is re-sampled (plus a free roll angle about the rail axis). Assets are fully
procedural compound spawners (an 8-sided box tube for the ring; capsules + spheres +
boxes for the one-piece kinematic rail fixture). Heavy imports (isaaclab, pxr) are
deferred so importing this module stays app-free.
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


# ----- shared geometry constants (spawners + cfg assertions + solve/smoke) ---------------------
RING_N = 8                        # ring tube facets
BORE_R = 0.024                    # bore inradius (the captive aperture)
WALL_T = 0.010                    # ring wall thickness
OUT_FLAT = BORE_R + WALL_T        # outer inradius (grasp width across flats = 68 mm)
OUT_CORNER = OUT_FLAT / math.cos(math.pi / RING_N)   # outer corner radius (~36.8 mm)
BORE_CORNER = BORE_R / math.cos(math.pi / RING_N)    # bore corner radius (~26.0 mm)
RING_HALF = 0.014                 # ring half length (28 mm long)

RAIL_R = 0.008                    # rail tube radius (16 mm rod)
RAIL_H = 0.190                    # rail run height (rail AXIS z)
CAP_R = 0.032                     # start-end ball cap radius (blocks the bore)
PLATE_T = 0.016                   # landing plate thickness
PLATE_HALF = 0.085                # landing plate half extent
POST_XY = (0.0, 0.245)            # end post axis (rail-fixture local frame)
LAND_Z = PLATE_T + RING_HALF      # delivered ring CENTER height (flat rest on plate)
HANG_DROP = BORE_R - RAIL_R       # hanging ring center sits this far below the rail axis

# the course polyline (rail-fixture local frame): start cap -> corner chamfer ->
# second run -> downturn chamfer -> down the post to the flat-rest center height
PATH_LOCAL = (
    (-0.260, 0.000, RAIL_H),          # A: start (ball cap center)
    (-0.032, 0.000, RAIL_H),          # B: corner chamfer entry
    (0.000, 0.032, RAIL_H),           # C: corner chamfer exit (45+45 = 90 deg turn)
    (0.000, 0.200, RAIL_H),           # D: downturn entry (crest)
    (0.000, 0.245, RAIL_H - 0.045),   # E: downturn exit = post top
    (POST_XY[0], POST_XY[1], LAND_Z), # G: course end (delivered ring center)
)


def _seg_lengths() -> list[float]:
    out = []
    for a, b in zip(PATH_LOCAL[:-1], PATH_LOCAL[1:]):
        out.append(math.dist(a, b))
    return out


SEG_LEN = _seg_lengths()
S_TOTAL = float(sum(SEG_LEN))
S_AT = [0.0]
for _l in SEG_LEN:
    S_AT.append(S_AT[-1] + _l)  # cumulative arc length at each path point


# ----- torch course helpers (shared by scene rubric, solve pursuit, smoke probes) --------------
_PATH_CACHE: dict[str, tuple] = {}


def _path_dev(device) -> tuple:
    key = str(device)
    if key not in _PATH_CACHE:
        pts = torch.tensor(PATH_LOCAL, dtype=torch.float32, device=device)
        seg = pts[1:] - pts[:-1]
        seglen = seg.norm(dim=-1)
        dirs = seg / seglen.unsqueeze(-1)
        cum = torch.cat([torch.zeros(1, device=device), torch.cumsum(seglen, dim=0)])
        _PATH_CACHE[key] = (pts, dirs, seglen, cum)
    return _PATH_CACHE[key]


def path_s_dist(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(s, dist) of points p (N,3) in the RAIL-LOCAL frame: arc-length station of the
    closest course point and the distance to it."""
    pts, dirs, seglen, cum = _path_dev(p.device)
    rel = p.unsqueeze(1) - pts[:-1].unsqueeze(0)               # (N,S,3)
    t = (rel * dirs.unsqueeze(0)).sum(dim=-1)                  # (N,S)
    t = t.clamp(min=0.0)
    t = torch.minimum(t, seglen.unsqueeze(0).expand_as(t))
    proj = pts[:-1].unsqueeze(0) + dirs.unsqueeze(0) * t.unsqueeze(-1)
    d = (p.unsqueeze(1) - proj).norm(dim=-1)                   # (N,S)
    i = d.argmin(dim=1)
    dist = d.gather(1, i.unsqueeze(1)).squeeze(1)
    s = cum[i] + t.gather(1, i.unsqueeze(1)).squeeze(1)
    return s, dist


def path_point(s: torch.Tensor, device) -> torch.Tensor:
    """(N,3) rail-local course point at station s (clamped to [0, S_TOTAL])."""
    pts, dirs, seglen, cum = _path_dev(device)
    s = s.clamp(0.0, float(cum[-1]) - 1e-6)
    idx = torch.searchsorted(cum[1:].contiguous(), s, right=True)
    idx = idx.clamp(max=len(SEG_LEN) - 1)
    t = s - cum[idx]
    return pts[idx] + dirs[idx] * t.unsqueeze(-1)


def path_tangent(s: torch.Tensor, device) -> torch.Tensor:
    """(N,3) rail-local course tangent (unit, pointing toward the course end)."""
    _pts, dirs, _seglen, cum = _path_dev(device)
    s = s.clamp(0.0, float(cum[-1]) - 1e-6)
    idx = torch.searchsorted(cum[1:].contiguous(), s, right=True)
    idx = idx.clamp(max=len(SEG_LEN) - 1)
    return dirs[idx]


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


def _collide(prim, color, contact_offset: float, material) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         quat=None) -> None:
    """One box child prim (translate -> orient -> scale, authored exactly once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _collide(seg, color, contact_offset, material)


def _capsule(stage, path: str, radius: float, height: float, center, quat, color,
             contact_offset: float, material=None) -> None:
    from pxr import UsdGeom

    seg = UsdGeom.Capsule.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    _apply_xform(seg, center, quat)
    _collide(seg, color, contact_offset, material)


def _sphere(stage, path: str, radius: float, center, color, contact_offset: float,
            material=None) -> None:
    from pxr import UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    _apply_xform(seg, center, None)
    _collide(seg, color, contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the physics armor (zero sleep/stabilization
    thresholds: a sleeping body silently ignores the applied wrenches the solve and
    smoke probes depend on; velocity iterations 4 kill the capsule-contact creep)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _quat_z_to(d) -> tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) rotating local +z onto the unit direction d."""
    dx, dy, dz = (float(v) for v in d)
    if dz > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if dz < -1.0 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    ax, ay = -dy, dx  # z cross d (unnormalized), z component 0
    n = math.hypot(ax, ay)
    ang = math.acos(max(-1.0, min(1.0, dz)))
    s = math.sin(ang / 2)
    return (math.cos(ang / 2), s * ax / n, s * ay / n, 0.0)


def _spawn_ring(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The BLUE ring: an 8-sided box TUBE (real 48 mm bore — captivity is enforced by
    collision, not bookkeeping). Local axis +z; origin = tube center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(1.50)  # arrests ring-on-rail pendulum/spin cycles
    mat = _material(stage, f"{prim_path}/phys_mat", 0.20, 0.18)
    co = cfg.contact_offset
    blue = (0.18, 0.34, 0.72)
    w = 2.0 * OUT_FLAT * math.tan(math.pi / RING_N) + 0.001  # tangential closure
    r_mid = BORE_R + WALL_T / 2
    for i in range(RING_N):
        ang = 2.0 * math.pi * i / RING_N
        cx, cy = r_mid * math.cos(ang), r_mid * math.sin(ang)
        q = (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))
        _box(stage, f"{prim_path}/wall_{i}", (WALL_T, w, 2 * RING_HALF),
             (cx, cy, 0.0), blue, co, material=mat, quat=q)
    return root


def _spawn_rail(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The one-piece KINEMATIC rail fixture: slick capsule rail segments spanning the
    course polyline (hemisphere ends fill the corners), a RED ball cap sealing the
    start end (wider than the ring's bore — the ring is captive), a support column
    under the cap, the vertical end POST (the last course segment, extended down into
    the plate), and the grippy GREEN landing plate. Origin = fixture local origin at
    ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 20.0,
                              kinematic=True)
    co = cfg.contact_offset
    slick = _material(stage, f"{prim_path}/slick_mat", 0.05, 0.04)
    grip = _material(stage, f"{prim_path}/grip_mat", 0.60, 0.50)
    steel = (0.72, 0.74, 0.78)
    red = (0.78, 0.14, 0.12)
    green = (0.22, 0.46, 0.28)
    dark = (0.20, 0.22, 0.26)
    # rail segments A..E (the E->G post is authored separately, extended into the plate)
    for i, (a, b) in enumerate(zip(PATH_LOCAL[:-2], PATH_LOCAL[1:-1])):
        d = tuple((bb - aa) / SEG_LEN[i] for aa, bb in zip(a, b))
        mid = tuple((aa + bb) / 2 for aa, bb in zip(a, b))
        _capsule(stage, f"{prim_path}/rail_{i}", RAIL_R, SEG_LEN[i], mid,
                 _quat_z_to(d), steel, co, material=slick)
    # end post: from the downturn exit E down into the plate (one slick capsule)
    e = PATH_LOCAL[4]
    post_top, post_bot = e[2], 0.012
    _capsule(stage, f"{prim_path}/post", RAIL_R, post_top - post_bot,
             (POST_XY[0], POST_XY[1], (post_top + post_bot) / 2),
             (1.0, 0.0, 0.0, 0.0), steel, co, material=slick)
    # start cap: RED ball, wider than the bore corner radius -> the ring cannot leave
    _sphere(stage, f"{prim_path}/cap", CAP_R, PATH_LOCAL[0], red, co, material=slick)
    # support column under the cap (behind the ring's reachable span)
    _box(stage, f"{prim_path}/column", (0.024, 0.040, RAIL_H - CAP_R),
         (PATH_LOCAL[0][0] - 0.012, PATH_LOCAL[0][1], (RAIL_H - CAP_R) / 2),
         dark, co, material=grip)
    # landing plate: grippy green slab centered on the post
    _box(stage, f"{prim_path}/plate", (2 * PLATE_HALF, 2 * PLATE_HALF, PLATE_T),
         (POST_XY[0], POST_XY[1], PLATE_T / 2), green, co, material=grip)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ring" not in _SPAWNER_CACHE:

        @configclass
        class RingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ring)
            mass: float = 0.06
            contact_offset: float = 0.002

        @configclass
        class RailSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rail)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(ring=RingSpawnerCfg, rail=RailSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RailRingSceneCfg(BaseCfg):
    """Config for `RailRingScene`. The rubric honesty is asserted in `__post_init__`:
    every encircling flat rest on the plate reads inside every tolerance by
    construction, while each wrong outcome (ring against the post but not around it,
    ring on the ground, ring lying on its side on the plate, ring still up on the
    post) reads outside with margin — and the ring can physically negotiate every
    corner of the course while the cap can never let it off."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    deliver_xy_tol: float = tunable(0.024)  # ring center to the post axis (m)
    deliver_z_tol: float = tunable(0.010)  # |ring center z - LAND_Z| (m)
    deliver_tilt_max_deg: float = tunable(25.0)  # ring axis vs vertical
    settle_lin: float = tunable(0.04)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.50)  # max |ang vel| when judging (rad/s)
    settle_steps: int = tunable(30)  # substeps of SUSTAINED stillness (0.25 s at 120 Hz)
    # progress-latch geometry
    prog_gate_dist: float = tunable(0.045)  # max distance to the course to earn progress
    prog_deadband: float = tunable(0.015)  # first arc-length that earns nothing (m)
    prog_cap: float = tunable(0.85)  # graded-credit cap (1.0 reserved for success)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rail_jitter: float = tunable(0.05)  # +-xy jitter of the rail fixture
    rail_yaw_deg: float = tunable(30.0)  # +-yaw of the rail fixture
    start_min: float = tunable(0.060)  # ring start station band along the first run (m
    start_max: float = tunable(0.150)  # of arc length from the cap center)

    # --- info: layout (env frame; the Franka base pose argument lives in TASK.md) ------------
    rail_pos: tuple = info((0.40, -0.08))  # nominal rail-fixture origin
    ring_mass: float = info(0.06)
    contact_offset: float = info(0.002)
    land_z: float = info(LAND_Z)
    post_xy: tuple = info(POST_XY)
    s_total: float = info(S_TOTAL)

    def __post_init__(self) -> None:
        c = self
        # -- captivity: the cap seals the bore with margin --
        assert CAP_R > BORE_CORNER + 0.005, "the cap must never pass the bore"
        # -- corner negotiability: the tilted ring always keeps the rail in its bore --
        # at a 45-deg turn the two arms deviate 22.5 deg from the bisector; at the ring
        # face the rail centerline is off-axis by RING_HALF*tan(22.5) plus its radius
        assert RAIL_R + RING_HALF * math.tan(math.pi / 8) < BORE_R - 0.006, \
            "the ring must pass every 45-deg corner with margin"
        assert 2 * RING_HALF < 2 * math.sqrt(BORE_R**2 - RAIL_R**2) - 0.010, \
            "the ring length must clear the corner chord bound"
        # -- delivered tolerance honest: every encircling rest reads inside --
        assert BORE_CORNER - RAIL_R < c.deliver_xy_tol - 0.004, \
            "a ring resting anywhere around the post must read inside deliver_xy_tol"
        # -- a ring AGAINST the post (not around it) reads far outside --
        assert OUT_FLAT + RAIL_R > c.deliver_xy_tol + 0.015, \
            "a ring leaning on / touching the post from outside must be rejected"
        # -- the z band rejects a ring flat on the GROUND at the post xy --
        assert abs(RING_HALF - LAND_Z) > c.deliver_z_tol + 0.004, \
            "a ring on the ground must read outside the z band"
        # -- the z band + tilt reject a ring lying on its SIDE on the plate --
        assert (PLATE_T + OUT_FLAT) - LAND_Z > c.deliver_z_tol + 0.008, \
            "a side-lying ring on the plate must read outside the z band"
        assert c.deliver_tilt_max_deg < 60.0, \
            "a side-lying ring (axis horizontal) must fail the tilt gate"
        # -- a ring still on the POST above the plate reads outside the z band --
        assert (PATH_LOCAL[4][2] - LAND_Z) > 4 * c.deliver_z_tol, \
            "the post must be tall enough that 'parked on the post' is rejected"
        # -- the delivered footprint stays fully on the plate --
        assert c.deliver_xy_tol + OUT_CORNER < PLATE_HALF - 0.020, \
            "the delivered ring footprint must stay on the plate with margin"
        # -- the hanging ring always earns progress (the gate is honest) --
        assert BORE_CORNER - RAIL_R < c.prog_gate_dist - 0.010, \
            "a ring hanging anywhere on the rail must sit inside the progress gate"
        # -- the null policy earns nothing: the deadband out-spans any settle wiggle --
        assert c.prog_deadband >= 0.010, "spawn-settle wiggle must earn no progress"
        assert 0.0 < c.prog_cap < 1.0, "graded credit must stay below success"
        # -- the start band sits on the first run, clear of the cap and the corner --
        assert c.start_min - RING_HALF > CAP_R + 0.008, \
            "the spawned ring must not touch the cap"
        assert c.start_max + RING_HALF < SEG_LEN[0] - 0.030, \
            "the spawned ring must sit well before the corner chamfer"
        # -- stillness must be sustained past a swing turning point --
        assert c.settle_steps >= 12, "settled() must out-last a turning point"
        # -- the column under the cap stays behind the ring's reachable span --
        # (ring center can never go below A_x + cap clearance; its face stays clear)
        col_face = PATH_LOCAL[0][0]  # column +x face at the cap center plane
        ring_min_face = PATH_LOCAL[0][0] + math.sqrt(CAP_R**2 - BORE_R**2)
        assert ring_min_face - col_face > 0.010, \
            "the support column must stay behind the cap-blocked ring"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rail_ring")
class RailRingScene(BaseScene):
    cfg: RailRingSceneCfg

    def __init__(self, cfg: RailRingSceneCfg | None = None) -> None:
        super().__init__(cfg or RailRingSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
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
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rail": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rail",
                spawn=sp["rail"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rail_pos[0], c.rail_pos[1], 0.0)),
            ),
            "ring": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ring",
                spawn=sp["ring"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ring_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.ring_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rail_pos[0] - 0.16, c.rail_pos[1], RAIL_H),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
        self.rail: RigidObject = env.iscene["rail"]
        self.ring: RigidObject = env.iscene["ring"]
        self.env_origins = env.iscene.env_origins
        self.prog_latch = torch.zeros(n, device=dev)
        self.s0 = torch.zeros(n, device=dev)  # per-episode start station
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the rail fixture re-placed (jitter + yaw), the ring hung on
        the first run at a sampled start station (free roll angle), latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def qz(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            z = torch.zeros_like(half)
            return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

        def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            aw, ax, ay, az = a.unbind(-1)
            bw, bx, by, bz = b.unbind(-1)
            return torch.stack([
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw], dim=-1)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # burn one draw: the FIRST post-seed draw is near-constant across seeds
        torch.rand(m, device=dev)

        # rail fixture: nominal + jitter + yaw
        rail_p = torch.zeros(m, 3, device=dev)
        rail_p[:, 0] = c.rail_pos[0]
        rail_p[:, 1] = c.rail_pos[1]
        rail_p[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rail_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rail_yaw_deg)
        q_rail = qz(yaw)
        write(self.rail, rail_p, q_rail)

        # ring: hung concentric on the first run at station s0 (falls HANG_DROP to
        # hang), axis along the run, free roll angle about its own axis
        off = c.start_min + torch.rand(m, device=dev) * (c.start_max - c.start_min)
        local = torch.zeros(m, 3, device=dev)
        local[:, 0] = PATH_LOCAL[0][0] + off
        local[:, 2] = RAIL_H
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        world = torch.zeros(m, 3, device=dev)
        world[:, 0] = rail_p[:, 0] + local[:, 0] * cy
        world[:, 1] = rail_p[:, 1] + local[:, 0] * sy
        world[:, 2] = RAIL_H
        c45 = math.cos(math.pi / 4)
        q_y90 = torch.tensor([c45, 0.0, c45, 0.0], device=dev).expand(m, 4)
        roll = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        q_ring = qmul(qmul(q_rail, q_y90), qz(roll))
        write(self.ring, world, q_ring)

        self.prog_latch[env_ids] = 0.0
        self.s0[env_ids] = off
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in (("rail", self.rail), ("ring", self.ring))},
            "prog_latch": self.prog_latch[env_ids].clone(),
            "s0": self.s0[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in (("rail", self.rail), ("ring", self.ring)):
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.prog_latch[env_ids] = state["prog_latch"]
        self.s0[env_ids] = state["s0"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A steel guide rail (a 16 mm round bar) is mounted at 190 mm height, and "
            "a BLUE RING — a short 8-sided tube, 68 mm across, 28 mm long, with a "
            "48 mm bore — hangs on it. The ring is CAPTIVE: the rail's near end is "
            "sealed by a RED BALL wider than the ring's bore (a dark column supports "
            "it), so the ring can never be taken off; it can only slide along the "
            "bar. From the ball the rail runs straight, turns 90 degrees through two "
            "45-degree chamfers (staying level), runs straight again, then bends 45 "
            "degrees DOWNWARD and becomes a vertical END POST that drops onto a "
            "GREEN LANDING PLATE (a 170 mm square slab, 16 mm thick, centered on the "
            "post). The whole fixture's position and heading, and the ring's start "
            "point along the first straight, change every episode — read them by "
            "looking. The rail is slick; the plate is grippy.\n"
            "Goal: deliver the ring to the plate — slide it along the bar the whole "
            "way (around the level corner, over the downward bend, down the post) so "
            "it drops off the post's lower end onto the plate and comes to rest FLAT "
            "on the plate with the post standing through its bore. A ring left "
            "hanging anywhere on the bar or parked on the post above the plate, a "
            "ring leaning against the post, or a ring resting tilted or on its side "
            "does not count. Everything must be at rest when judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the captive blue ring all the way along the bent guide rail — "
            "around the level corner, over the downward bend, and down the vertical "
            "end post — so it drops onto the green landing plate and rests flat with "
            "the post through its bore. The ring cannot leave the rail; the red ball "
            "seals the near end."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _axis_w(self, body) -> torch.Tensor:
        """(N,3): body local +z axis in world."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def ring_in_rail(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,3), axis (N,3)): ring center and axis in the RAIL fixture frame."""
        from isaaclab.utils.math import quat_apply_inverse

        q = self.rail.data.root_quat_w
        p = quat_apply_inverse(q, self.ring.data.root_pos_w - self.rail.data.root_pos_w)
        a = quat_apply_inverse(q, self._axis_w(self.ring))
        return p, a

    def course_station(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(s, dist): the ring's arc-length station along the course and its distance
        to the course polyline (rail frame)."""
        p, _a = self.ring_in_rail()
        return path_s_dist(p)

    def delivered(self) -> torch.Tensor:
        """(N,) bool: the ring rests flat on the plate encircling the post — center on
        the post axis within `deliver_xy_tol`, at flat-rest height within
        `deliver_z_tol`, axis within `deliver_tilt_max_deg` of vertical."""
        c = self.cfg
        p, a = self.ring_in_rail()
        post = torch.tensor([POST_XY[0], POST_XY[1]], device=p.device)
        on_axis = (p[:, 0:2] - post).norm(dim=-1) < c.deliver_xy_tol
        at_rest_z = (p[:, 2] - LAND_Z).abs() < c.deliver_z_tol
        upright = a[:, 2].abs() > math.cos(math.radians(c.deliver_tilt_max_deg))
        return on_axis & at_rest_z & upright

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: the ring instantaneously below the stillness thresholds (a swing
        crosses this briefly at turning points — never judge on it directly)."""
        c = self.cfg
        return (self.ring.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.ring.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness SUSTAINED for `settle_steps` consecutive substeps."""
        return self.still_count >= float(self.cfg.settle_steps)

    # ----- progress latch (step-coupled) ------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the ring's best on-course arc-length progress every substep (gated:
        off-course positions earn nothing), plus the sustained-stillness counter."""
        c = self.cfg
        s, dist = self.course_station()
        denom = (S_TOTAL - self.s0 - c.prog_deadband).clamp(min=1e-4)
        prog = ((s - self.s0 - c.prog_deadband) / denom).clamp(0.0, 1.0)
        on_course = (dist < c.prog_gate_dist).float()
        self.prog_latch = torch.maximum(self.prog_latch, prog * on_course)
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ring rests flat on the landing plate around the post,
        settled — the delivered end state of the full traversal."""
        return self.delivered() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: `prog_cap` x latched on-course arc-length progress
        (normalized from the episode's start station); 1.0 iff success(). Latched —
        credit never evaporates; the null policy scores ~0."""
        base = (self.cfg.prog_cap * self.prog_latch).clamp(0.0, self.cfg.prog_cap)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="rail_ring", robot="null", env_spacing=3.0))
