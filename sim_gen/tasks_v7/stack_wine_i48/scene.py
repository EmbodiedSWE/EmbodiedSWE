"""CaskWeightSortScene — weigh three identical-looking kegs by NUDGING them, then rack
them sorted by weight into color-coded cradles (sim_gen task `stack_wine_i48`).

Derived from rlbench/stack_wine, but STRATEGICALLY different: the seed is a pure
prehensile pick-and-place — grasp THE wine bottle (unique, visually given) and lay it
on THE rack; nothing ever has to be discovered, and any bottle-shaped grasp-carry-set
plan solves it. Here the load-bearing problem is EPISTEMIC: the three kegs are
VISUALLY IDENTICAL (same capsule geometry, same color) but their MASSES differ by 3x
steps (0.3 / 0.9 / 2.7 kg) and are RE-SHUFFLED across the three bodies every episode
(runtime PhysX mass+inertia writes, verified by readback). Which keg goes to which
cradle is decided by mass rank — red=heaviest, yellow=middle, green=lightest — so a
solver that does not first PROBE the kegs (nudge each one and compare the responses:
the same push rolls the light keg roughly 3x farther than the middle one and ~9x
farther than the heavy one) cannot beat a 1-in-6 permutation guess no matter how good
its pick-and-place is. The plan is measure -> rank -> place; the seed's plan is place.

success() (all live, judged on physical poses):
  for EVERY keg: it lies seated in the V-groove of ITS OWN cradle (the one whose
  color matches the keg's mass rank) — center within `seat_x_tol`/`seat_y_tol` of the
  cradle center, center height within `seat_z_pad` of the V-seat height `seat_z`
  (i.e. cradled on the V faces, not perched on a rail, not on the floor or plate, not
  held aloft), axis within `seat_axis_deg` of the cradle's trough direction — and
  every keg is settled (lin + ang velocity thresholds).
score() = latched stage credit anchored in the demonstrated solution:
  0.05 per keg ever DISPLACED >= `disp_latch` from its spawn while staying low (the
  probe/manipulation onset a null policy never produces) + 0.20 per keg ever SEATED IN
  ITS CORRECT cradle continuously for `seat_latch_steps` substeps while slow, capped
  at 0.75; exactly 1.0 iff success(). Doing nothing scores ~0; racking the kegs in a
  wrong order (the seed's identity-blind strategy) leaves the 0.60 identity credit on
  the table and never reaches success.

Assets are fully procedural (no external files):
  - kegs: three DYNAMIC capsules (r=26 mm, cylindrical section 120 mm, total length
    172 mm — native PhysX capsule, so they roll smoothly), identical oak color with
    two visual-only hoop rings; spawn mass is the middle value, per-episode masses are
    written into PhysX at reset (inertias rescaled proportionally, CoM at the body
    origin) and VERIFIED by readback.
  - cradles ("bays"): three KINEMATIC compound bodies — a colored base plate with two
    45-degree tilted rails forming a V-groove along x: a keg dropped in rests on two
    FACE-tangent contacts (centre at ~49 mm), is self-centred by the V and cannot
    roll out; RED / YELLOW / GREEN mark the heaviest / middle / lightest role. (Face
    contacts, not edge contacts: a keg chocked on two sharp box edges excites a PhysX
    velocity limit cycle — constant phantom velocity readback on a stationary body.)
Grasp: the keg's 52 mm body fits the 80 mm Franka jaw with margin; the V is open and
shallow, so the jaw releases the keg from just above the seat without entering it.

Per-episode randomization (verified by READBACK in smoke): the mass permutation
across the three keg bodies (PhysX mass readback), the cradle color arrangement
(the three bays are permuted across the row slots + jittered), and keg spawn slots
(permuted + jittered + small yaw jitter). Heavy imports (isaaclab, pxr) are deferred
so importing this module — and registering the scene — stays app-free.
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

# Role order used EVERYWHERE: mass rank 0 = heaviest -> RED, 1 -> YELLOW, 2 -> GREEN.
ROLE_NAMES = ("RED", "YELLOW", "GREEN")
ROLE_COLORS = (
    (0.85, 0.10, 0.10),  # red   <- heaviest
    (0.90, 0.78, 0.08),  # yellow<- middle
    (0.10, 0.62, 0.18),  # green <- lightest
)

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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         orient=None) -> None:
    """A colored box prim; collides iff `contact_offset` is not None. `orient` is an
    optional wxyz quat applied between translate and scale."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_cask(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC keg at `prim_path`. Origin = capsule centre, axis = body +x
    (native PhysX capsule -> smooth rolling; the probe physics depends on it). Two
    visual-only hoop rings so it reads as a little barrel. Spawn mass is the MIDDLE
    value; per-episode masses are written at reset via the PhysX view."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.spawn_mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(cfg.damping))
    px.CreateAngularDampingAttr(float(cfg.damping))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    r, half_l = cfg.cask_r, cfg.cask_len / 2
    cap = UsdGeom.Capsule.Define(stage, f"{prim_path}/body")
    cap.CreateRadiusAttr(float(r))
    cap.CreateHeightAttr(float(cfg.cask_len))
    cap.CreateAxisAttr("X")
    cap.CreateExtentAttr([Gf.Vec3f(-half_l - r, -r, -r), Gf.Vec3f(half_l + r, r, r)])
    cap.CreateDisplayColorAttr([Gf.Vec3f(*cfg.cask_color)])
    _collide(cap.GetPrim(), cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction_static,
                         cfg.friction_dynamic)
    _bind_material(cap.GetPrim(), mat)

    # visual-only hoops (no colliders — they must not spoil the smooth capsule roll)
    for j, hx in enumerate((-0.6 * half_l, 0.6 * half_l)):
        hoop = UsdGeom.Cylinder.Define(stage, f"{prim_path}/hoop_{j}")
        hoop.CreateRadiusAttr(float(r + 0.0008))
        hoop.CreateHeightAttr(0.008)
        hoop.CreateAxisAttr("X")
        hoop.CreateExtentAttr([Gf.Vec3f(-0.004, -r - 0.0008, -r - 0.0008),
                               Gf.Vec3f(0.004, r + 0.0008, r + 0.0008)])
        UsdGeom.Xformable(hoop.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(hx, 0.0, 0.0))
        hoop.CreateDisplayColorAttr([Gf.Vec3f(*cfg.hoop_color)])
    return root


def _spawn_bay(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one KINEMATIC cradle at `prim_path`. Origin = plate centre at floor
    level. A colored base plate + two 45-degree tilted rails forming a V-groove
    along x: a keg dropped in rests on two FACE-tangent contacts (never on box
    edges — sharp edge contacts drive a PhysX velocity limit cycle) and is centred
    by the V; it cannot roll out."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(5.0)

    color = cfg.role_color
    plate_c = tuple(0.55 * v for v in color)
    _box(stage, f"{prim_path}/plate",
         (cfg.plate_x, cfg.plate_y, cfg.plate_t),
         (0.0, 0.0, cfg.plate_t / 2), plate_c, cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction_static,
                         cfg.friction_dynamic)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/plate"), mat)
    # V-groove rails: apex at (y=0, z=plate_t); each rail's INNER FACE is a plane
    # tilted `v_angle` from horizontal, box local +y = up-slope, local +z = inner
    # normal (toward the keg).
    s = math.sin(math.radians(cfg.v_angle_deg))
    cs = math.cos(math.radians(cfg.v_angle_deg))
    half = math.radians(cfg.v_angle_deg) / 2
    for j, sgn in enumerate((-1.0, 1.0)):
        cy = sgn * (cs * cfg.rail_slope_w / 2 + s * cfg.rail_t / 2)
        cz = cfg.plate_t + s * cfg.rail_slope_w / 2 - cs * cfg.rail_t / 2
        # rotation about x: +v_angle for the +y rail, -v_angle for the -y rail
        quat = (math.cos(half), sgn * math.sin(half), 0.0, 0.0)
        _box(stage, f"{prim_path}/rail_{j}",
             (cfg.rail_len, cfg.rail_slope_w, cfg.rail_t),
             (0.0, cy, cz), color, cfg.contact_offset, orient=quat)
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/rail_{j}"), mat)
    return root


def _cask_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cask" not in _SPAWNER_CACHE:

        @configclass
        class CaskSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cask)
            cask_r: float = 0.026
            cask_len: float = 0.120
            spawn_mass: float = 0.9
            damping: float = 0.5
            cask_color: tuple = (0.46, 0.26, 0.14)
            hoop_color: tuple = (0.20, 0.14, 0.10)
            friction_static: float = 0.9
            friction_dynamic: float = 0.8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cask"] = CaskSpawnerCfg
    return _SPAWNER_CACHE["cask"](**kw)


def _bay_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bay" not in _SPAWNER_CACHE:

        @configclass
        class BaySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bay)
            role_color: tuple = (0.8, 0.1, 0.1)
            plate_x: float = 0.18
            plate_y: float = 0.11
            plate_t: float = 0.012
            rail_len: float = 0.16
            rail_slope_w: float = 0.044
            rail_t: float = 0.010
            v_angle_deg: float = 45.0
            friction_static: float = 0.9
            friction_dynamic: float = 0.8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bay"] = BaySpawnerCfg
    return _SPAWNER_CACHE["bay"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CaskWeightSortSceneCfg(BaseCfg):
    """Config for `CaskWeightSortScene`. Honesty knobs asserted in `__post_init__`:
    kegs are graspable but indistinguishable, mass steps are far apart (probe
    separability), the wedge geometry chocks a seated keg above the plate, and the
    seat tolerances reject on-a-rail / beside-the-bay / across-the-rails states."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_x_tol: float = tunable(0.050)  # keg centre along the trough (m)
    seat_y_tol: float = tunable(0.012)  # keg centre across the trough (m) — a keg
    # perched on one rail's top edge sits at |y| ~ cos(v)*slope_w ~ 31 mm: rejected
    seat_z_pad: float = tunable(0.007)  # +/- band around the V-seat height (m) —
    # must exclude floor (26 mm), on-the-plate (38 mm) and perched-on-a-rail-edge
    # (~69 mm) rests, asserted in __post_init__
    seat_axis_deg: float = tunable(15.0)  # keg axis within this of the trough (x) axis
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.50)  # max |ang vel| when judging (rad/s)
    disp_latch: float = tunable(0.030)  # "was manipulated": ever displaced this far
    low_z: float = tunable(0.080)  # ...while the keg centre stayed under this (on floor)
    seat_latch_steps: int = tunable(24)  # consecutive slow-seated substeps to latch credit

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    masses: tuple = tunable((2.7, 0.9, 0.3))  # heavy/middle/light (kg), 3x steps
    bay_row_jitter: float = tunable(0.025)  # +/- y jitter of the whole cradle row
    bay_x_jitter: float = tunable(0.008)  # +/- x jitter per cradle (keeps end gaps safe)
    cask_xy_jitter: tuple = tunable((0.010, 0.035))  # +/- (x, y) jitter per keg spawn
    cask_yaw_deg: float = tunable(8.0)  # +/- yaw jitter per keg spawn

    # --- tunable: placement ------------------------------------------------------------------
    slots_x: tuple = tunable((-0.20, 0.0, 0.20))  # shared x slots (bays AND keg spawns)
    bay_row_y: float = tunable(0.20)  # cradle row y (env frame)
    cask_row_y: float = tunable(-0.12)  # keg spawn row y; probes roll kegs toward -y

    # --- info: structure ----------------------------------------------------------------------
    cask_r: float = info(0.026)  # capsule radius — 52 mm dia fits the 80 mm Franka jaw
    cask_len: float = info(0.120)  # cylindrical section; total length = len + 2 r = 172 mm
    spawn_mass: float = info(0.9)  # authored mass (middle value); reset overwrites
    damping: float = info(0.5)  # lin+ang damping: a probed keg coasts ~2 s, not forever
    jaw_max: float = info(0.080)  # Franka parallel-jaw stroke (embodiment honesty)
    payload_max: float = info(3.0)  # Franka payload (embodiment honesty)
    plate_x: float = info(0.18)
    plate_y: float = info(0.11)
    plate_t: float = info(0.012)
    rail_len: float = info(0.16)
    rail_slope_w: float = info(0.044)  # slope width of each V face (apex -> top edge)
    rail_t: float = info(0.010)  # rail board thickness
    v_angle_deg: float = info(45.0)  # V-face angle from horizontal — FACE-tangent
    # contacts (a keg chocked on two sharp box EDGES excites a PhysX velocity limit
    # cycle: constant phantom lin/ang readback on a visually stationary body)
    friction_static: float = info(0.9)
    friction_dynamic: float = info(0.8)
    cask_color: tuple = info((0.46, 0.26, 0.14))  # identical for all three — that is the point
    hoop_color: tuple = info((0.20, 0.14, 0.10))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    seat_z: float = field(default=None, init=False)  # V-seat height (keg centre)

    def __post_init__(self) -> None:
        v = math.radians(self.v_angle_deg)
        # Keg centred in the V: centre at apex_z + r/sin(v); contact points on each
        # face at slope distance r/tan(v) from the apex.
        self.seat_z = self.plate_t + self.cask_r / math.sin(v)
        assert self.cask_r / math.tan(v) < self.rail_slope_w - 0.005, (
            "seat contacts must land ON the V faces, not on the rail top edges")
        assert self.seat_z - self.cask_r > self.plate_t + 0.002, (
            "a seated keg must hang clear of the plate (held by the V faces)")
        assert self.seat_z - self.seat_z_pad > self.cask_r + 0.004, (
            "the seat band must exclude a keg lying on the FLOOR")
        assert self.seat_z - self.seat_z_pad > self.plate_t + self.cask_r + 0.002, (
            "the seat band must exclude a keg lying flat ON THE PLATE")
        edge_top_z = self.plate_t + math.sin(v) * self.rail_slope_w
        assert self.seat_z + self.seat_z_pad < edge_top_z + self.cask_r - 0.002, (
            "the seat band must exclude a keg perched ON TOP of one rail")
        assert self.seat_y_tol < math.cos(v) * self.rail_slope_w - 0.005, (
            "the y tolerance must reject a keg perched on one rail's top edge")
        assert 2 * self.cask_r < self.jaw_max - 0.015, "kegs must be trivially graspable"
        assert max(self.masses) <= self.payload_max, "heaviest keg must be liftable"
        m = sorted(self.masses)
        assert m[1] / m[0] >= 2.5 and m[2] / m[1] >= 2.5, (
            "mass steps must be far apart — probe separability is the task's core")
        total_l = self.cask_len + 2 * self.cask_r
        gap = (self.slots_x[1] - self.slots_x[0])
        assert gap - 2 * self.bay_x_jitter > total_l + 0.008, "cradle end gaps must be safe"
        assert gap - 2 * self.cask_xy_jitter[0] > total_l + 0.006, (
            "keg spawn slots must never overlap end-to-end")
        assert (self.bay_row_y - self.bay_row_jitter
                - (self.cask_row_y + self.cask_xy_jitter[1])) > (
            self.plate_y / 2 + self.cask_r + 0.04), "keg spawns stay clear of the cradles"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cask_weight_sort")
class CaskWeightSortScene(BaseScene):
    cfg: CaskWeightSortSceneCfg

    def __init__(self, cfg: CaskWeightSortSceneCfg | None = None) -> None:
        super().__init__(cfg or CaskWeightSortSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        for j, role in enumerate(ROLE_NAMES):
            out[f"bay_{role.lower()}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bay_" + role.lower(),
                spawn=_bay_spawner_cfg(
                    role_color=ROLE_COLORS[j], plate_x=c.plate_x, plate_y=c.plate_y,
                    plate_t=c.plate_t, rail_len=c.rail_len, rail_slope_w=c.rail_slope_w,
                    rail_t=c.rail_t, v_angle_deg=c.v_angle_deg,
                    friction_static=c.friction_static, friction_dynamic=c.friction_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slots_x[j], c.bay_row_y, 0.0)),
            )
        for i in range(3):
            out[f"cask_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cask_" + str(i),
                spawn=_cask_spawner_cfg(
                    cask_r=c.cask_r, cask_len=c.cask_len, spawn_mass=c.spawn_mass,
                    damping=c.damping, cask_color=c.cask_color, hoop_color=c.hoop_color,
                    friction_static=c.friction_static, friction_dynamic=c.friction_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slots_x[i], c.cask_row_y, c.cask_r + 0.002)),
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
        self.bays: list[RigidObject] = [
            env.iscene[f"bay_{r.lower()}"] for r in ROLE_NAMES]
        self.casks: list[RigidObject] = [env.iscene[f"cask_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # rank_of_cask[e, i] in {0,1,2}: 0 = heaviest (RED bay), 2 = lightest (GREEN).
        self.rank_of_cask = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self.mass_of_cask = torch.full((n, 3), self.cfg.spawn_mass, device=dev)
        self.spawn_xy = torch.zeros(n, 3, 2, device=dev)
        self.disp_latch = torch.zeros(n, 3, device=dev)
        self.correct_latch = torch.zeros(n, 3, device=dev)
        self.seat_ctr = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self.max_seat_ctr = torch.zeros(n, 3, dtype=torch.long, device=dev)  # telemetry
        # spawn-time inertia/mass baseline (same authored geometry for all kegs)
        view = self.casks[0].root_physx_view
        self._m0 = float(view.get_masses().cpu().view(-1)[0])
        self._i0 = view.get_inertias().cpu().view(n, -1)[0].clone()

    def _apply_masses(self, env_ids: torch.Tensor) -> None:
        """Write the per-episode masses into PhysX (mass + proportionally rescaled
        inertia; CoM stays at the body origin) and VERIFY by readback — a silent no-op
        voids the whole task (the balance_scale precedent)."""
        n = self.env.num_envs
        all_ids = torch.arange(n)
        ids_cpu = env_ids.cpu()
        for i, cask in enumerate(self.casks):
            view = cask.root_physx_view
            masses = view.get_masses().clone().cpu().view(-1)
            masses[ids_cpu] = self.mass_of_cask[env_ids, i].cpu()
            view.set_masses(masses.view_as(view.get_masses()), all_ids)
            inert = view.get_inertias().clone().cpu().view(n, -1)
            ratio = (self.mass_of_cask[env_ids, i].cpu() / self._m0).view(-1, 1)
            inert[ids_cpu] = self._i0.view(1, -1) * ratio
            view.set_inertias(inert.view_as(view.get_inertias()), all_ids)
        back = self.casks[0].root_physx_view.get_masses().cpu().view(-1)[int(env_ids[0])]
        expect = float(self.mass_of_cask[int(env_ids[0]), 0])
        if abs(float(back) - expect) > 1e-4:
            print(f"[cask_weight_sort] MASS APPLY FAILED: readback {float(back):.3f} "
                  f"!= {expect:.3f} — verdicts are void", flush=True)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: shuffle the mass permutation across keg bodies (PhysX
        write + readback), shuffle the cradle color arrangement across the row slots
        (+ jitter), shuffle keg spawn slots (+ jitter + yaw), zero all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        slots = torch.tensor(c.slots_x, device=dev)

        # --- masses: permutation of the 3 values across the 3 keg bodies ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # rank of cask i
        mass_table = torch.tensor(c.masses, device=dev)  # index by rank: 0 = heaviest
        self.rank_of_cask[env_ids] = perm
        self.mass_of_cask[env_ids] = mass_table[perm]
        self._apply_masses(env_ids)

        # --- cradles: role j -> slot bay_slot[e, j], one shared row-y jitter ---
        bay_slot = torch.rand(m, 3, device=dev).argsort(dim=1)
        row_y = c.bay_row_y + (torch.rand(m, device=dev) * 2 - 1) * c.bay_row_jitter
        for j, bay in enumerate(self.bays):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slots[bay_slot[:, j]]
            st[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.bay_x_jitter
            st[:, 1] = row_y
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            bay.write_root_state_to_sim(st, env_ids)

        # --- kegs: slot permutation + jitter + small yaw, lying axis ~x ---
        cask_slot = torch.rand(m, 3, device=dev).argsort(dim=1)
        yaw_amp = math.radians(c.cask_yaw_deg)
        for i, cask in enumerate(self.casks):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slots[cask_slot[:, i]]
            st[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.cask_xy_jitter[0]
            st[:, 1] = c.cask_row_y + (torch.rand(m, device=dev) * 2 - 1) * c.cask_xy_jitter[1]
            st[:, 2] = c.cask_r + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            cask.write_root_state_to_sim(st, env_ids)
            self.spawn_xy[env_ids, i] = st[:, 0:2] - origin[:, 0:2]

        # --- latches ---
        self.disp_latch[env_ids] = 0.0
        self.correct_latch[env_ids] = 0.0
        self.seat_ctr[env_ids] = 0
        self.max_seat_ctr[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bays": [b.data.root_state_w[env_ids].clone() for b in self.bays],
            "casks": [k.data.root_state_w[env_ids].clone() for k in self.casks],
            "rank_of_cask": self.rank_of_cask[env_ids].clone(),
            "mass_of_cask": self.mass_of_cask[env_ids].clone(),
            "spawn_xy": self.spawn_xy[env_ids].clone(),
            "disp_latch": self.disp_latch[env_ids].clone(),
            "correct_latch": self.correct_latch[env_ids].clone(),
            "seat_ctr": self.seat_ctr[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.bays, state["bays"]):
            b.write_root_state_to_sim(st, env_ids)
        for k, st in zip(self.casks, state["casks"]):
            k.write_root_state_to_sim(st, env_ids)
        self.rank_of_cask[env_ids] = state["rank_of_cask"]
        self.mass_of_cask[env_ids] = state["mass_of_cask"]
        self._apply_masses(env_ids)
        self.spawn_xy[env_ids] = state["spawn_xy"]
        self.disp_latch[env_ids] = state["disp_latch"]
        self.correct_latch[env_ids] = state["correct_latch"]
        self.seat_ctr[env_ids] = state["seat_ctr"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        total_l = c.cask_len + 2 * c.cask_r
        return (
            f"Three IDENTICAL-LOOKING oak kegs (capsule-shaped, {2 * c.cask_r * 1000:.0f} mm "
            f"across, {total_l * 1000:.0f} mm long) lie on the floor with their long axes "
            f"roughly parallel, in a loose row. Behind them stands a row of three cradles, "
            f"one RED, one YELLOW, one GREEN (their left-to-right order changes between "
            f"episodes): each cradle is a colored plate with two tilted rails forming a "
            f"V-groove; a keg laid lengthwise into the V settles into the groove and is "
            f"chocked there.\n"
            f"The kegs look the same but WEIGH very differently — one is heavy "
            f"({max(c.masses):.1f} kg), one middling, one light ({min(c.masses):.1f} kg), "
            f"and which body is which is shuffled every episode. There is NO visual cue: "
            f"the only way to tell them apart is to interact — give each keg the same "
            f"gentle nudge and compare the responses (the same push rolls the light keg "
            f"about three times farther than the middle one and about nine times farther "
            f"than the heavy one; they roll freely across the floor, perpendicular to "
            f"their axis).\n"
            f"Goal: rack the kegs SORTED BY WEIGHT — the HEAVIEST keg seated in the RED "
            f"cradle, the MIDDLE one in the YELLOW cradle, the LIGHTEST in the GREEN "
            f"cradle. Each keg must lie along its cradle's groove, wedged between the two "
            f"rails (not balanced on a rail, not beside the cradle), and everything must "
            f"come to rest. Any keg racked in the wrong color fails the task — probing "
            f"before racking is how you beat the guess. No particular order of operations "
            f"is required."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Nudge each keg to judge its weight by how far it rolls, then lay the heaviest "
            "keg in the red cradle, the middle-weight keg in the yellow cradle, and the "
            "lightest keg in the green cradle, each wedged between its cradle's rails. "
            "The kegs look identical — racking any keg in a wrong color fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _cask_pos(self) -> torch.Tensor:
        """(N,3,3) keg centres, env-local."""
        return torch.stack(
            [k.data.root_pos_w - self.env_origins for k in self.casks], dim=1)

    def _cask_axis(self) -> torch.Tensor:
        """(N,3,3) world direction of each keg's body +x (long) axis."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        return torch.stack(
            [quat_apply(k.data.root_quat_w, ex) for k in self.casks], dim=1)

    def _bay_pos(self) -> torch.Tensor:
        """(N,3,3) cradle origins (plate centre at floor level), env-local; row j =
        role j (0 RED, 1 YELLOW, 2 GREEN)."""
        return torch.stack(
            [b.data.root_pos_w - self.env_origins for b in self.bays], dim=1)

    # ----- predicates -------------------------------------------------------------------------
    def seated_matrix(self) -> torch.Tensor:
        """(N,3,3) bool: keg i geometrically seated in the groove of role-j's cradle
        (centred, at wedge height, axis along the trough). Identity-blind — the
        identity gate lives in `seated_correct`."""
        c = self.cfg
        rel = self._cask_pos().unsqueeze(2) - self._bay_pos().unsqueeze(1)  # (N,3,3,3)
        near = ((rel[..., 0].abs() < c.seat_x_tol)
                & (rel[..., 1].abs() < c.seat_y_tol)
                & ((rel[..., 2] - c.seat_z).abs() < c.seat_z_pad))
        ax = self._cask_axis()  # (N,3,3)
        aligned = ax[..., 0].abs() >= math.cos(math.radians(c.seat_axis_deg))
        return near & aligned.unsqueeze(2)

    def seated_correct(self) -> torch.Tensor:
        """(N,3) bool: keg i seated in ITS OWN cradle (the role = its mass rank)."""
        mat = self.seated_matrix()
        return mat.gather(2, self.rank_of_cask.unsqueeze(-1)).squeeze(-1)

    def settled(self) -> torch.Tensor:
        """(N,3) bool: keg lin AND ang velocity below thresholds."""
        lin = torch.stack(
            [k.data.root_lin_vel_w.norm(dim=-1) for k in self.casks], dim=1)
        ang = torch.stack(
            [k.data.root_ang_vel_w.norm(dim=-1) for k in self.casks], dim=1)
        return (lin < self.cfg.settle_lin) & (ang < self.cfg.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch manipulation onset (ever displaced while low) and correct seating
        (continuously slow-seated for `seat_latch_steps`), every physics substep."""
        c = self.cfg
        pos = self._cask_pos()  # (N,3,3)
        moved = (pos[..., :2] - self.spawn_xy).norm(dim=-1) > c.disp_latch
        low = pos[..., 2] < c.low_z
        self.disp_latch = torch.maximum(self.disp_latch, (moved & low).float())
        lin = torch.stack(
            [k.data.root_lin_vel_w.norm(dim=-1) for k in self.casks], dim=1)
        ok = self.seated_correct() & (lin < 0.10)
        self.seat_ctr = torch.where(ok, self.seat_ctr + 1, torch.zeros_like(self.seat_ctr))
        self.max_seat_ctr = torch.maximum(self.max_seat_ctr, self.seat_ctr)
        self.correct_latch = torch.maximum(
            self.correct_latch, (self.seat_ctr >= c.seat_latch_steps).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every keg seated in its own (mass-rank) cradle AND settled —
        judged live on physical poses."""
        return (self.seated_correct() & self.settled()).all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.05 per keg ever displaced (manipulation onset)
        + 0.20 per keg ever slow-seated in its CORRECT cradle (latched), capped at
        0.75; exactly 1.0 iff success(). Doing nothing scores ~0; the seed's
        identity-blind racking strategy leaves the identity credit on the table."""
        base = (0.05 * self.disp_latch.sum(dim=1)
                + 0.20 * self.correct_latch.sum(dim=1)).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="cask_weight_sort", robot="null"))
