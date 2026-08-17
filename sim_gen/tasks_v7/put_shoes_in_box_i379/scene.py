"""CrateFlipPackScene — uncover the shoes by INVERTING the crate that traps them, then
pack them into it (sim_gen task `put_shoes_in_box_i379`).

Derived from rlbench/put_shoes_in_box, but STRATEGICALLY different: the seed is a bare
pick-and-drop — an always-OPEN box and two free shoes; carry each shoe over the mouth,
release, done. The container is never operated and every object is accessible from the
first frame. Here the crate spawns UPSIDE-DOWN with its mouth sealed to the floor and
the PAIR OF SHOES TRAPPED UNDERNEATH it: the goal objects are initially unreachable,
and the crate is initially an OBSTACLE, not a receptacle. The solver must first
INVERT the crate — a nonprehensile maneuver: the mouth-down shell offers no graspable
plate, so it is rolled mouth-up by tipping it over its ground edge (two quarter-rolls,
or one push-over then another), each tip a real contact-dynamics event (pivot on the
rim edge, fall through the balance angle, land on the next face) — and only then does
the classic transport-and-release phase exist at all. The rubric refuses geometric
containment without erection: a mouth-down crate standing over the shoes contains
them in its body frame exactly like a packed crate does, and scores nothing.

Execution order is enforced by GEOMETRY, not rubric fiat: while the crate is inverted
the mouth is sealed against the floor (smoke pushes a trapped shoe with ~3x its
weight — it slides into the interior wall and never leaves the footprint), and
nothing can be placed "in" a container whose cavity opens downward. The two shoes
themselves may be inserted in either order.

success(): crate standing UPRIGHT on its base (up axis within `up_tol_deg` of
world-up AND root height at the resting height — a crate perched on a shoe or held
in the air fails the height clause), both shoes fully inside the cavity below the
rim (three body points each, in the CRATE FRAME), everything settled and finite —
all live physical outcomes.

score() (latched credit never evaporates): 0.15 * shoes ever UNCOVERED (the crate
rolled off / away from both) + 0.15 * crate ever ERECTED (upright at rest on its
base) + 0.20 * each shoe ever inside-the-erected-crate-and-settled, capped at 0.70;
exactly 1.0 iff success() live. Doing nothing scores ~0 (the crate starts inverted
over the shoes).

Assets are fully procedural (no external files):
  - crate (dynamic, 0.6 kg): slate-blue compound — floor slab + four walls; interior
    340 x 260 x 120 mm, 12 mm walls. Local frame: origin at the interior floor-top
    center, mouth up +z. CoM authored at (0,0,+45 mm) (walls taller than the slab).
    Tipping it over the long ground edge takes a ~7 N push at the top edge.
  - shoe (x2, dynamic, 0.14 kg): bright-orange sole 140 x 55 x 16 mm with a darker
    heel block 50 x 55 x 30 mm at the rear (total height 46 mm — fits under the
    120 mm cavity with a wide margin, and both shoes fit flat side by side in the
    upright interior with >= 20 mm to every wall: there is deliberately NO packing
    puzzle; the task is the container inversion).
All spawners author mass/CoM/inertia + friction material + contact offsets (1.5 mm)
in-func (custom spawn funcs apply no cfg schemas); bind() asserts masses by readback.

Per-episode randomization (readback-verified in smoke): crate center xy jitter +
FREE yaw, shoe under-crate offsets (per-slot jitter + free relative yaw,
segment-distance resampled with a deterministic fallback). Heavy imports (isaaclab,
pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawners (crate / shoe) --------------------------------------------------
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


def _box_prim(stage, path: str, size, center, color, contact_offset: float,
              material=None) -> None:
    """Author one axis-aligned box child prim (translate -> scale; authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_armor(root, mass: float, com, inertia) -> None:
    """RigidBody + explicit MassAPI (mass, CoM, diagonal inertia) + PhysX armor,
    authored in-func (custom spawn funcs apply no cfg schemas)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.08)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC open crate compound. Local frame: origin at the INTERIOR FLOOR-TOP
    CENTER, mouth up +z; interior spans |x| < L/2, |y| < W/2, 0 < z < H."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_armor(root, cfg.crate_mass, (0.0, 0.0, cfg.crate_com_z),
                 (cfg.crate_ix, cfg.crate_iy, cfg.crate_iz))
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    L, W, H, tw, tf = cfg.in_l, cfg.in_w, cfg.in_h, cfg.wall_t, cfg.floor_t
    co = cfg.contact_offset
    _box_prim(stage, f"{prim_path}/floor", (L + 2 * tw, W + 2 * tw, tf),
              (0.0, 0.0, -tf / 2), cfg.crate_color2, co, mat)
    _box_prim(stage, f"{prim_path}/wall_yn", (L + 2 * tw, tw, H),
              (0.0, -(W + tw) / 2, H / 2), cfg.crate_color, co, mat)
    _box_prim(stage, f"{prim_path}/wall_yp", (L + 2 * tw, tw, H),
              (0.0, (W + tw) / 2, H / 2), cfg.crate_color, co, mat)
    _box_prim(stage, f"{prim_path}/wall_xn", (tw, W, H),
              (-(L + tw) / 2, 0.0, H / 2), cfg.crate_color, co, mat)
    _box_prim(stage, f"{prim_path}/wall_xp", (tw, W, H),
              ((L + tw) / 2, 0.0, H / 2), cfg.crate_color, co, mat)
    return root


def _spawn_shoe(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic shoe. Local frame: origin at the SOLE CENTER (mid-thickness); toe at
    +x, the darker heel block at -x on top of the sole rear."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_armor(root, cfg.shoe_mass, (-0.015, 0.0, 0.008),
                 (cfg.shoe_ix, cfg.shoe_iy, cfg.shoe_iz))
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _box_prim(stage, f"{prim_path}/sole", (cfg.sole_l, cfg.sole_w, cfg.sole_t),
              (0.0, 0.0, 0.0), cfg.shoe_color, co, mat)
    _box_prim(stage, f"{prim_path}/heel",
              (cfg.heel_l, cfg.heel_w, cfg.heel_h),
              (-(cfg.sole_l - cfg.heel_l) / 2, 0.0, (cfg.sole_t + cfg.heel_h) / 2),
              cfg.heel_color, co, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            in_l: float = 0.34
            in_w: float = 0.26
            in_h: float = 0.12
            wall_t: float = 0.012
            floor_t: float = 0.012
            crate_mass: float = 0.6
            crate_com_z: float = 0.045
            crate_ix: float = 0.0049
            crate_iy: float = 0.0075
            crate_iz: float = 0.0107
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            crate_color: tuple = (0.32, 0.40, 0.52)
            crate_color2: tuple = (0.42, 0.50, 0.62)
            contact_offset: float = 0.0015

        @configclass
        class ShoeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shoe)
            sole_l: float = 0.14
            sole_w: float = 0.055
            sole_t: float = 0.016
            heel_l: float = 0.050
            heel_w: float = 0.055
            heel_h: float = 0.030
            shoe_mass: float = 0.14
            shoe_ix: float = 6.0e-5
            shoe_iy: float = 2.5e-4
            shoe_iz: float = 2.6e-4
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            shoe_color: tuple = (0.95, 0.45, 0.10)
            heel_color: tuple = (0.70, 0.26, 0.05)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(crate=CrateSpawnerCfg, shoe=ShoeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CrateFlipPackSceneCfg(BaseCfg):
    """Config for `CrateFlipPackScene`. The captivity / erection honesty conditions are
    asserted in __post_init__."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    in_margin: float = tunable(0.002)     # shoe body points inside the interior minus this (m)
    in_z_min: float = tunable(-0.005)     # shoe points above the crate floor minus this (m)
    up_tol_deg: float = tunable(8.0)      # crate up-axis within this of world-up (erected)
    rest_z_tol: float = tunable(0.008)    # crate root height within this of the base rest height
    cover_up_max: float = tunable(-0.10)  # crate up_z below this still counts as "inverted over"
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.6)      # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    crate_jitter: float = tunable(0.06)   # crate center xy jitter (+/- m)
    crate_yaw_deg: float = tunable(180.0)  # crate yaw, uniform +/- deg (FREE yaw)
    slot_x: float = tunable(0.075)        # shoe slots under the crate: +/- this along local x
    slot_x_jitter: float = tunable(0.008)
    slot_y_jitter: float = tunable(0.030)
    shoe_sep: float = tunable(0.061)      # min shoe segment-to-segment distance (m)

    # --- info: layout ---------------------------------------------------------------------------
    crate_pos: tuple = info((0.40, 0.0))  # crate center on the ground (nominal)
    # --- info: crate geometry (local frame: origin at interior floor-top center, mouth +z) ------
    in_l: float = info(0.34)              # interior length (x)
    in_w: float = info(0.26)              # interior width (y)
    in_h: float = info(0.12)              # interior height (rim above the floor)
    wall_t: float = info(0.012)
    floor_t: float = info(0.012)
    crate_mass: float = info(0.6)
    crate_com_z: float = info(0.045)
    crate_ix: float = info(0.0049)
    crate_iy: float = info(0.0075)
    crate_iz: float = info(0.0107)
    # --- info: shoe -----------------------------------------------------------------------------
    sole_l: float = info(0.14)
    sole_w: float = info(0.055)
    sole_t: float = info(0.016)
    heel_l: float = info(0.050)
    heel_w: float = info(0.055)
    heel_h: float = info(0.030)
    shoe_mass: float = info(0.14)
    shoe_ix: float = info(6.0e-5)
    shoe_iy: float = info(2.5e-4)
    shoe_iz: float = info(2.6e-4)
    # --- info: physics / rubric weights ---------------------------------------------------------
    mu_static: float = info(0.6)
    mu_dynamic: float = info(0.5)
    contact_offset: float = info(0.0015)
    w_uncover: float = info(0.15)         # shoes ever uncovered (latched)
    w_erect: float = info(0.15)           # crate ever upright at rest on its base (latched)
    w_shoe: float = info(0.20)            # each shoe ever inside-the-erected-crate (latched)

    # Derived in __post_init__.
    outer_l: float = 0.0                  # outer footprint (x)
    outer_w: float = 0.0                  # outer footprint (y)
    outer_h: float = 0.0                  # overall crate height
    rest_z: float = 0.0                   # upright root rest height (origin = floor top)
    inv_z: float = 0.0                    # mouth-down root height (rim on the ground)
    shoe_h: float = 0.0                   # total shoe height lying flat
    shoe_half_diag: float = 0.0           # top-view half diagonal of a lying shoe

    def __post_init__(self) -> None:
        self.outer_l = self.in_l + 2 * self.wall_t
        self.outer_w = self.in_w + 2 * self.wall_t
        self.outer_h = self.in_h + self.floor_t
        self.rest_z = self.floor_t
        self.inv_z = self.in_h
        self.shoe_h = self.sole_t + self.heel_h
        self.shoe_half_diag = math.hypot(self.sole_l / 2, self.sole_w / 2)
        # --- captivity honesty: shoes fit UNDER the inverted cavity with margin ---
        assert self.shoe_h + 0.02 < self.in_h, "shoe must fit under the inverted crate"
        # spawn zone keeps every shoe fully inside the interior at any yaw
        zx = self.slot_x + self.slot_x_jitter + self.shoe_half_diag
        zy = self.slot_y_jitter + self.shoe_half_diag
        assert zx < self.in_l / 2 - 0.006 and zy < self.in_w / 2 - 0.006, \
            "shoe spawn zone must stay clear of the interior walls"
        # both shoes fit flat in the upright interior at the solve's drop slots
        assert 0.08 + self.sole_l / 2 < self.in_l / 2 - 0.015, "drop slots must fit"
        # a crate perched on a shoe is outside the rest-height tolerance
        assert self.rest_z_tol < self.shoe_h / 2, "perched crate must fail the z clause"
        # a shoe flat on the rim top is above the interior (z > in_h)
        assert self.sole_t / 2 > 0.004, "rim-lying shoe must sit above the cavity"
        # tipping is within a Franka push: force at the top edge to start the tip
        tip_force = self.crate_mass * 9.81 * (self.outer_w / 2) / self.outer_h
        assert tip_force < 12.0, f"tipping push {tip_force:.1f} N too high"
        # graspable by the 80 mm Franka jaw: sole width / heel length
        assert self.sole_w <= 0.078 and self.heel_l <= 0.078
        # "covered" (mouth-down over the shoes) and "upright receptacle" are far apart:
        # cover requires the crate tilted > 90 deg from up, upright allows < up_tol_deg,
        # so no pose can ever satisfy both and the uncover credit needs a real roll.
        assert -0.9 < self.cover_up_max < 0.0
        assert math.cos(math.radians(self.up_tol_deg)) > 0.9


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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("crate_flip_pack")
class CrateFlipPackScene(BaseScene):
    cfg: CrateFlipPackSceneCfg

    SHOE_NAMES = ("shoe_a", "shoe_b")

    def __init__(self, cfg: CrateFlipPackSceneCfg | None = None) -> None:
        super().__init__(cfg or CrateFlipPackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        crate_spawn = sp["crate"](
            in_l=c.in_l, in_w=c.in_w, in_h=c.in_h, wall_t=c.wall_t, floor_t=c.floor_t,
            crate_mass=c.crate_mass, crate_com_z=c.crate_com_z,
            crate_ix=c.crate_ix, crate_iy=c.crate_iy, crate_iz=c.crate_iz,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)

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
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=crate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos[0], c.crate_pos[1], c.inv_z + 0.0015),
                    rot=(0.0, 1.0, 0.0, 0.0)),  # mouth-down: Rx(pi)
            ),
        }
        for i, name in enumerate(self.SHOE_NAMES):
            shoe_spawn = sp["shoe"](
                sole_l=c.sole_l, sole_w=c.sole_w, sole_t=c.sole_t,
                heel_l=c.heel_l, heel_w=c.heel_w, heel_h=c.heel_h,
                shoe_mass=c.shoe_mass, shoe_ix=c.shoe_ix, shoe_iy=c.shoe_iy,
                shoe_iz=c.shoe_iz, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                contact_offset=c.contact_offset)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shoe_" + name,
                spawn=shoe_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos[0] - 0.075 + 0.15 * i, c.crate_pos[1],
                         c.sole_t / 2 + 0.002)),
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
        self.crate: RigidObject = env.iscene["crate"]
        self.shoes: dict[str, RigidObject] = {n: env.iscene[n] for n in self.SHOE_NAMES}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # authored-mass readback (custom spawn funcs apply no cfg schemas — verify)
        for body, m_want in ((self.crate, c.crate_mass),
                             (self.shoes["shoe_a"], c.shoe_mass),
                             (self.shoes["shoe_b"], c.shoe_mass)):
            m_got = float(body.root_physx_view.get_masses().reshape(-1)[0])
            assert abs(m_got - m_want) < 0.02, f"authored mass lost: {m_got} vs {m_want}"
        # shoe body points judged for containment (toe end, heel end, heel top), local
        self._pts_local = torch.tensor(
            [[(c.sole_l / 2, 0.0, 0.0),
              (-c.sole_l / 2, 0.0, 0.0),
              (-(c.sole_l - c.heel_l) / 2, 0.0, c.sole_t / 2 + c.heel_h)]],
            device=dev).expand(n, 3, 3).contiguous()
        # latches (partial credit survives transients; success is judged live)
        self._uncovered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._erected = torch.zeros(n, dtype=torch.bool, device=dev)
        self._packed = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: crate MOUTH-DOWN at a jittered center with FREE yaw, both
        shoes flat on the ground UNDERNEATH it (slot jitter + free relative yaw,
        segment-distance resampled with a deterministic fallback), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(8, device=dev)  # burn: first post-seed draws are near-constant

        # --- crate: mouth-down (Rx(pi)), center jitter + free yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.crate_yaw_deg)
        q_crate = _qmul(_qz(yaw), _qx(torch.full((m,), math.pi, device=dev)))
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.crate_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        bp[:, 1] = c.crate_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        bp[:, 2] = c.inv_z + 0.0015
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_crate
        self.crate.write_root_state_to_sim(st, env_ids)

        # --- shoes: flat on the ground under the crate footprint ---
        # slots at +/- slot_x along the crate's long axis, jittered, free relative yaw;
        # resampled so the two sole segments stay > shoe_sep apart, deterministic
        # fallback (crosswise at the slots) if sampling fails.
        sx = torch.stack([
            -c.slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.slot_x_jitter,
            c.slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.slot_x_jitter], dim=1)
        sy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_y_jitter
        ry = (torch.rand(m, 2, device=dev) * 2 - 1) * math.pi

        def seg_dist() -> torch.Tensor:
            """Min distance between the two sole segments in the slot plane, (m,)."""
            h = c.sole_l / 2
            pc = torch.stack([sx[:, 0], sy[:, 0]], dim=-1)
            qc = torch.stack([sx[:, 1], sy[:, 1]], dim=-1)
            su = torch.stack([torch.cos(ry[:, 0]), torch.sin(ry[:, 0])], dim=-1)
            tv = torch.stack([torch.cos(ry[:, 1]), torch.sin(ry[:, 1])], dim=-1)
            best = torch.full((m,), torch.inf, device=dev)
            for fa in (-1.0, -0.5, 0.0, 0.5, 1.0):
                pa = pc + su * (fa * h)
                w = pa - qc
                t = (w * tv).sum(-1).clamp(-h, h)
                best = torch.minimum(best, (w - tv * t.unsqueeze(-1)).norm(dim=-1))
            for fb in (-1.0, -0.5, 0.0, 0.5, 1.0):
                qb = qc + tv * (fb * h)
                w = qb - pc
                t = (w * su).sum(-1).clamp(-h, h)
                best = torch.minimum(best, (w - su * t.unsqueeze(-1)).norm(dim=-1))
            return best

        for _try in range(24):
            bad = seg_dist() < c.shoe_sep
            if not bad.any():
                break
            k = int(bad.sum())
            sy[bad, 1] = (torch.rand(k, device=dev) * 2 - 1) * c.slot_y_jitter
            ry[bad, :] = (torch.rand(k, 2, device=dev) * 2 - 1) * math.pi
        bad = seg_dist() < c.shoe_sep
        if bad.any():  # deterministic fallback: crosswise shoes at the two slots
            sx[bad, 0], sx[bad, 1] = -c.slot_x, c.slot_x
            sy[bad, 0], sy[bad, 1] = -0.02, 0.02
            ry[bad, :] = math.pi / 2

        cpsi, spsi = torch.cos(yaw), torch.sin(yaw)
        for i, name in enumerate(self.SHOE_NAMES):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = bp[:, 0] + sx[:, i] * cpsi - sy[:, i] * spsi
            st[:, 1] = bp[:, 1] + sx[:, i] * spsi + sy[:, i] * cpsi
            st[:, 2] = c.sole_t / 2 + 0.002
            st[:, 3:7] = _qz(yaw + ry[:, i])
            st[:, 0:3] += origin
            self.shoes[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._uncovered[env_ids] = False
        self._erected[env_ids] = False
        self._packed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "shoes": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.shoes.items()},
            "uncovered": self._uncovered[env_ids].clone(),
            "erected": self._erected[env_ids].clone(),
            "packed": self._packed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        for n, b in self.shoes.items():
            b.write_root_state_to_sim(state["shoes"][n], env_ids)
        self._uncovered[env_ids] = state["uncovered"]
        self._erected[env_ids] = state["erected"]
        self._packed[env_ids] = state["packed"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A SLATE-BLUE OPEN CRATE (interior {c.in_l * 1000:.0f} x {c.in_w * 1000:.0f} mm, "
            f"{c.in_h * 1000:.0f} mm deep, lighter-blue base) sits UPSIDE-DOWN on the floor: "
            f"its mouth is sealed against the ground and its flat base faces up. Trapped "
            f"UNDERNEATH it — invisible until the crate moves — lies a pair of BRIGHT-ORANGE "
            f"SHOES (each a flat sole {c.sole_l * 1000:.0f} x {c.sole_w * 1000:.0f} mm with a "
            f"darker heel block at the rear, {c.shoe_h * 1000:.0f} mm tall lying flat). The "
            f"crate's position and heading change every episode — read them from the scene.\n"
            f"Goal: both shoes INSIDE the crate with the crate standing UPRIGHT on its base. "
            f"While the crate is upside-down nothing can be put in it and the shoes cannot be "
            f"reached, so first TURN THE CRATE MOUTH-UP: it has no lid or handle — push high "
            f"on one side wall (a ~7 N push near the top edge) so it tips over its opposite "
            f"bottom edge onto its side, then tip it once more in the same direction onto its "
            f"base; this also uncovers the shoes. Then place both shoes (either order) fully "
            f"inside the cavity, below the rim. A shoe lying across the rim, leaning outside "
            f"the crate, or left on the floor does not count; a crate left on its side, "
            f"upside-down, tilted, or perched on a shoe does not count — even with the shoes "
            f"geometrically under/inside it, the crate must stand level on its base. Success: "
            f"crate upright at rest on its base, both shoes fully inside below the rim, "
            f"everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip the upside-down blue crate over its bottom edge twice so it lands "
            "mouth-up — the pair of orange shoes is trapped beneath it. Then put both "
            "shoes fully inside the upright crate. The crate must end standing level on "
            "its base with both shoes below its rim, everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _crate_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> crate body frame (live-read)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.crate.data.root_quat_w,
                                  pos_w - self.crate.data.root_pos_w)

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _status(self) -> dict[str, torch.Tensor]:
        """Live geometric predicates ((N,) or (N,2))."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        up_z = quat_apply(self.crate.data.root_quat_w, ez)[:, 2]
        crate_z = (self.crate.data.root_pos_w - self.env_origins)[:, 2]

        inside_l, covered_l, settle_l = [], [], []
        for name in self.SHOE_NAMES:
            b = self.shoes[name]
            q = b.data.root_quat_w[:, None, :].expand(n, 3, 4).reshape(n * 3, 4)
            pts_w = b.data.root_pos_w[:, None, :] + quat_apply(
                q, self._pts_local.reshape(n * 3, 3)).reshape(n, 3, 3)
            loc = self._crate_local(pts_w.reshape(n * 3, 3)).reshape(n, 3, 3)
            ok = (loc[:, :, 0].abs() < c.in_l / 2 - c.in_margin) \
                & (loc[:, :, 1].abs() < c.in_w / 2 - c.in_margin) \
                & (loc[:, :, 2] < c.in_h) & (loc[:, :, 2] > c.in_z_min)
            inside_l.append(ok.all(dim=1))
            # covered: the crate is inverted over this shoe (shoe in the sealed cavity)
            ctr = self._crate_local(b.data.root_pos_w)
            cov = (up_z < c.cover_up_max) \
                & (ctr[:, 0].abs() < c.in_l / 2 + c.wall_t) \
                & (ctr[:, 1].abs() < c.in_w / 2 + c.wall_t) \
                & (ctr[:, 2] > -0.002) & (ctr[:, 2] < c.in_h)
            covered_l.append(cov)
            settle_l.append(self._settled(b))
        inside = torch.stack(inside_l, dim=1)
        covered = torch.stack(covered_l, dim=1)
        sh_settled = torch.stack(settle_l, dim=1)

        upright = up_z.clamp(-1.0, 1.0) >= math.cos(math.radians(c.up_tol_deg))
        at_rest_z = (crate_z - c.rest_z).abs() < c.rest_z_tol
        receptacle = upright & at_rest_z & self._settled(self.crate)
        return {"inside": inside, "covered": covered, "sh_settled": sh_settled,
                "receptacle": receptacle, "up_z": up_z, "crate_z": crate_z}

    def _update_latches(self) -> None:
        s = self._status()
        self._uncovered |= ~s["covered"].any(dim=1)
        self._erected |= s["receptacle"]
        self._packed |= s["inside"] & s["sh_settled"] & s["receptacle"].unsqueeze(-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: crate standing upright AT REST ON ITS BASE (up-axis + resting
        height + settled), both shoes fully inside the cavity below the rim in the
        crate frame, shoes settled, everything finite — all live physical outcomes.
        Geometric containment alone is NOT success: a mouth-down crate standing over
        the shoes contains them in its body frame and fails the erection clauses."""
        self._update_latches()
        s = self._status()
        pos = torch.stack([b.data.root_pos_w for b in self.shoes.values()]
                          + [self.crate.data.root_pos_w], dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return (s["inside"] & s["sh_settled"]).all(dim=1) & s["receptacle"] & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*shoes ever uncovered + 0.15*crate ever erected +
        0.20*each shoe ever inside-the-erected-crate-and-settled (all latched; ~0 for
        doing nothing — the crate starts inverted over the shoes), capped at 0.70 —
        and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_uncover * self._uncovered.float()
                + c.w_erect * self._erected.float()
                + c.w_shoe * self._packed.float().sum(dim=1)).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="crate_flip_pack", robot="null"))
