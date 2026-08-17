"""CartonFlipPackScene — roll the OVERTURNED carton upright (180 deg, two edge-pivots),
then pack both items into it through the mouth.

Derived from the `box_task/box_task_replay` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is "bimanual pick-and-place of loose products into an
already-open, ready cardboard box" — the container is passive scenery and every action
is a prehensile item transport. Here the container itself is the mechanism: the carton
starts MOUTH-DOWN on the floor (an overturned empty carton), and no item can enter it
until it has been physically turned over — a 180-degree NONPREHENSILE reorientation by
two chained edge-pivots (push near the top edge above the tipping threshold, let the
carton fall onto its side, pivot again onto its base), with the carton translating
about a quarter metre along the roll corridor as it goes. Only then are the two items
dropped in through the open mouth. Plan skeleton: large-body nonprehensile
reorientation (tip-over dynamics, balance angles, momentum management) + terminal
drop-in loading — not a different-numbers packing run.

Mechanics (plain rigid bodies, no joints): the carton is a dynamic COMPOUND body
(base plate + 4 walls, authored by a custom spawner: MassAPI mass/CoM/inertia and the
friction material are authored in the func — custom spawners apply no cfg schemas).
A `post_step` plant consumes a `roll_tau` buffer: a torque about the carton's BODY-X
axis (the ridge axis — self-aligned with the pivot edges, so the command is immune to
the pods' wrench-frame quirks), clamped to `tau_max` = the moment a ~4.5 N fingertip
push at the top edge produces. Tipping, the fall onto the next face, the landing
impact and the final rock-to-rest all go through contact dynamics; sub-threshold
torque provably cannot right the carton (smoke check).

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py trajectory):
  - `righted` (latch, 30-substep streak): the carton rests UPRIGHT on the floor
    (mouth-up within `upright_deg`, centre at rest height, settled).
  - `packed_can` / `packed_candle` (latch, 30-substep streak): the item is inside the
    upright carton (carton-frame bounds, centre below the rim) with carton + item
    settled.
  - success(): CURRENT physical state — carton upright at rest height and settled,
    BOTH items contained, items settled.
  - score() = 1.0 iff success(), else 0.30*righted + 0.25*packed_can
    + 0.25*packed_candle. ~0 for the null policy; a knock-over after success falls
    back to 0.80, never 0.

Per-episode randomization (verified by READBACK in smoke.py): carton xy + free yaw,
each item's polar offset (radius + angle in the carton's yaw frame, sampled in the
two ridge-side cones so both roll corridors stay clear), and which item spawns on
which side.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

_SPAWNER_CACHE: dict[str, Any] = {}


# ----- carton spawner (compound dynamic body; schemas authored HERE) ----------------------------
def _spawn_carton(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the open carton: DYNAMIC compound — base plate + 4 walls, origin at the
    OUTER-box centre, mouth at body +z. Custom spawners apply no cfg schemas, so mass,
    CoM, diagonal inertia, damping, solver iterations and the friction material are
    all authored here."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    # CoM measured from the panel volumes (base plate pulls it 11.5 mm below centre);
    # hollow-box diagonal inertia (uniform-solid formula x 1.4 shell factor).
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(c.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in c.inertia]))

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.08)
    pxrb.CreateAngularDampingAttr(float(c.ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)

    mat = UsdShade.Material.Define(stage, f"{prim_path}/card_mat")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(float(c.mu[0]))
    mapi.CreateDynamicFrictionAttr(float(c.mu[1]))
    mapi.CreateRestitutionAttr(0.0)
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pxm.CreateRestitutionCombineModeAttr().Set("min")

    def add_box(name: str, center, size) -> None:
        box = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        box.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(box.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        box.CreateDisplayColorAttr([Gf.Vec3f(*c.color)])
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        px.CreateContactOffsetAttr(float(c.contact_offset))
        px.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(mat, materialPurpose="physics")

    hx, hy, hz = c.outer[0] / 2, c.outer[1] / 2, c.outer[2] / 2
    t = c.wall_t
    wall_h = c.outer[2] - t  # walls stand ON the base plate, top flush with the rim
    wall_cz = -hz + t + wall_h / 2
    add_box("base", (0.0, 0.0, -hz + t / 2), (c.outer[0], c.outer[1], t))
    add_box("wall_yp", (0.0, hy - t / 2, wall_cz), (c.outer[0], t, wall_h))
    add_box("wall_yn", (0.0, -hy + t / 2, wall_cz), (c.outer[0], t, wall_h))
    add_box("wall_xp", (hx - t / 2, 0.0, wall_cz), (t, c.outer[1] - 2 * t, wall_h))
    add_box("wall_xn", (-hx + t / 2, 0.0, wall_cz), (t, c.outer[1] - 2 * t, wall_h))
    return root


def _carton_spawner_cls() -> Any:
    """Declare (once) the carton spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "carton" not in _SPAWNER_CACHE:

        @configclass
        class CartonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carton)
            outer: tuple = (0.18, 0.14, 0.10)
            wall_t: float = 0.008
            mass: float = 0.30
            com_z: float = -0.0115
            inertia: tuple = (0.00104, 0.00148, 0.00182)
            mu: tuple = (0.85, 0.75)
            ang_damp: float = 0.5
            color: tuple = (0.62, 0.44, 0.24)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["carton"] = CartonSpawnerCfg
    return _SPAWNER_CACHE["carton"]


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CartonFlipPackSceneCfg(BaseCfg):
    """Config for `CartonFlipPackScene`. Geometry is derived once in `__post_init__` so
    the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    upright_deg: float = tunable(10.0)  # carton mouth-up within this of world-up
    rest_z_tol: float = tunable(0.015)  # carton centre within this of upright rest height
    contain_x: float = tunable(0.070)  # |item x| in carton frame (interior half-x 0.082;
    # max physical in-carton offset = 0.082 - r_can = 0.056 < this, so anything
    # physically inside counts; anything outside the shell is >= 0.116 away)
    contain_y: float = tunable(0.050)  # |item y| in carton frame (interior half-y 0.062)
    contain_z: tuple = tunable((-0.055, 0.028))  # item centre band: above the floor
    # (local -0.042), below the rim (local +0.050) by > 2.2 cm — containment judged
    # BELOW the aperture, not at it
    settle_v: float = tunable(0.10)  # max |lin vel| when judging (above the GPU phantom band)
    settle_w: float = tunable(0.60)  # max carton |ang vel| when judging (rad/s)
    latch_streak: int = tunable(30)  # substeps a latch condition must hold (0.25 s)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    carton_pos: tuple = tunable((0.0, 0.0))  # nominal carton centre
    pos_jitter: float = tunable(0.04)  # +- xy jitter of the carton at reset
    yaw_deg: float = tunable(180.0)  # +- carton yaw at reset (free)
    item_r: tuple = tunable((0.20, 0.27))  # item polar radius from the carton centre
    item_cone_deg: float = tunable(45.0)  # items sampled within +-this of the +-body-x
    # (ridge) directions, so BOTH +-body-y roll corridors stay clear by construction

    # --- tunable: plant (difficulty dials) -------------------------------------------------------
    tau_max: float = tunable(0.45)  # roll-torque clamp (~4.5 N fingertip at the 0.10 m edge)
    carton_mass: float = tunable(0.30)
    can_mass: float = tunable(0.15)
    candle_mass: float = tunable(0.12)
    carton_mu: tuple = tunable((0.85, 0.75))  # tipping needs the base NOT to slide out
    item_mu: tuple = tunable((0.60, 0.50))

    # --- info: structure -------------------------------------------------------------------------
    outer: tuple = info((0.18, 0.14, 0.10))  # carton outer size (body frame, mouth at +z)
    wall_t: float = info(0.008)
    can_r: float = info(0.026)
    can_h: float = info(0.085)  # fits standing inside (interior depth 0.092)
    candle_r: float = info(0.028)
    candle_h: float = info(0.050)
    drop_dx: tuple = info((0.040, -0.040))  # in-carton drop offsets (can, candle) along body x:
    # wall clearance >= 14 mm each, inter-item gap 26 mm (no seam wedging)
    drop_clear: float = info(0.020)  # release height of the item bottom above the rim
    base_pos: tuple = info((0.42, 0.0))  # documented Franka base xy (TASK.md; on a ridge side)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    int_hx: float = field(default=None, init=False)  # interior half extents
    int_hy: float = field(default=None, init=False)
    floor_z_local: float = field(default=None, init=False)  # interior floor top (body frame)
    rim_z_local: float = field(default=None, init=False)
    rest_z_up: float = field(default=None, init=False)  # carton centre resting upright
    rest_z_down: float = field(default=None, init=False)  # carton centre resting mouth-down
    tau_tip1: float = field(default=None, init=False)  # torque to start pivot 1 (mouth-down)
    tau_tip2: float = field(default=None, init=False)  # torque to start pivot 2 (on side)

    def __post_init__(self) -> None:
        lx, ly, h = self.outer
        self.int_hx = lx / 2 - self.wall_t
        self.int_hy = ly / 2 - self.wall_t
        self.floor_z_local = -h / 2 + self.wall_t
        self.rim_z_local = h / 2
        self.rest_z_up = h / 2
        self.rest_z_down = h / 2
        g = 9.81
        # pivot 1 (mouth-down, roll about a ridge edge): restoring moment mg * (ly/2)
        self.tau_tip1 = self.carton_mass * g * (ly / 2)  # 0.206 N m
        # pivot 2 (on side, over the base edge): mg * (h/2 - |com_z|)
        self.tau_tip2 = self.carton_mass * g * (h / 2 - 0.0115)  # 0.113 N m


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("carton_flip_pack")
class CartonFlipPackScene(BaseScene):
    cfg: CartonFlipPackSceneCfg

    def __init__(self, cfg: CartonFlipPackSceneCfg | None = None) -> None:
        super().__init__(cfg or CartonFlipPackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the compound carton (custom spawner; reset() overturns it),
        and the two cylinder items (built-in CylinderCfg so cfg schemas apply)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        CartonSpawnerCfg = _carton_spawner_cls()
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

        def item_cfg(radius: float, height: float, mass: float, color) -> Any:
            return sim_utils.CylinderCfg(
                radius=radius, height=height, axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.10, angular_damping=0.50, **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.item_mu[0], dynamic_friction=c.item_mu[1],
                    restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.85, dynamic_friction=0.75, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "carton": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton",
                spawn=CartonSpawnerCfg(
                    outer=c.outer, wall_t=c.wall_t, mass=c.carton_mass,
                    mu=c.carton_mu, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.carton_pos[0], c.carton_pos[1], c.rest_z_down + 0.002),
                    rot=(0.0, 1.0, 0.0, 0.0)),  # mouth-down
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=item_cfg(c.can_r, c.can_h, c.can_mass, (0.90, 0.45, 0.08)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.24, 0.0, c.can_h / 2 + 0.002)),
            ),
            "candle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Candle",
                spawn=item_cfg(c.candle_r, c.candle_h, c.candle_mass, (0.15, 0.65, 0.20)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.24, 0.0, c.candle_h / 2 + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the solve drives the carton with per-substep wrenches; without this
                # an applied wrench is integrated only on the first solver iteration
                # and the edge-pivot plant rings/stalls (rock-and-scoot limit cycle)
                "enable_external_forces_every_iteration": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.carton: RigidObject = env.iscene["carton"]
        self.can: RigidObject = env.iscene["can"]
        self.candle: RigidObject = env.iscene["candle"]
        self.env_origins = env.iscene.env_origins
        # External drive input (solve/smoke write; post_step consumes + owns the wrench
        # slot — never call set_external_force_and_torque on the carton directly).
        self.roll_tau = torch.zeros(n, device=dev)  # torque about the carton BODY-X axis
        # Rubric latches + streak counters.
        self.righted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.packed_can = torch.zeros(n, dtype=torch.bool, device=dev)
        self.packed_candle = torch.zeros(n, dtype=torch.bool, device=dev)
        self._streak_right = torch.zeros(n, dtype=torch.long, device=dev)
        self._streak_can = torch.zeros(n, dtype=torch.long, device=dev)
        self._streak_candle = torch.zeros(n, dtype=torch.long, device=dev)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: carton MOUTH-DOWN with xy jitter + free yaw; items standing
        upright at sampled polar offsets in the two ridge-side cones (both roll
        corridors clear by construction); latches, streaks and the drive buffer
        zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(8, device=dev)  # burn post-seed draws (first-draw degeneracy)

        # --- carton: mouth-down = qz(yaw) * qx(pi) = (0, cos y/2, sin y/2, 0) ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        half = yaw / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.carton_pos[0]
        st[:, 1] = c.carton_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        st[:, 2] = c.rest_z_down + 0.002
        st[:, 4] = torch.cos(half)
        st[:, 5] = torch.sin(half)
        cxy = st[:, 0:2].clone()
        st[:, 0:3] += origin
        self.carton.write_root_state_to_sim(st, env_ids)

        # --- items: polar offsets in the +-body-x cones; side assignment random ---
        swap = torch.rand(m, device=dev) < 0.5
        cone = math.radians(c.item_cone_deg)
        for i, (body, h) in enumerate(((self.can, c.can_h), (self.candle, c.candle_h))):
            side = torch.where(swap ^ (i == 1), torch.zeros(m, device=dev),
                               torch.full((m,), math.pi, device=dev))
            phi = yaw + side + (torch.rand(m, device=dev) * 2 - 1) * cone
            r = c.item_r[0] + torch.rand(m, device=dev) * (c.item_r[1] - c.item_r[0])
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = cxy[:, 0] + r * torch.cos(phi)
            st[:, 1] = cxy[:, 1] + r * torch.sin(phi)
            st[:, 2] = h / 2 + 0.002
            st[:, 0:3] += origin
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        # --- rubric + drive state ---
        for t in (self.righted, self.packed_can, self.packed_candle):
            t[env_ids] = False
        for t in (self._streak_right, self._streak_can, self._streak_candle):
            t[env_ids] = 0
        self.roll_tau[env_ids] = 0.0

    # ----- readings -----------------------------------------------------------------------------
    def carton_pos(self) -> torch.Tensor:
        """(N, 3) carton centre (env-origin corrected)."""
        return self.carton.data.root_pos_w - self.env_origins

    def up_z(self) -> torch.Tensor:
        """(N,) world-z component of the carton's body +z (mouth normal):
        -1 mouth-down, 0 on side, +1 upright."""
        q = self.carton.data.root_quat_w
        # (R ez)_z = 1 - 2 (qx^2 + qy^2)
        return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)

    def roll_axis_w(self) -> torch.Tensor:
        """(N, 3) the carton's BODY-X (ridge) axis in world frame — the roll axis."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self.carton.data.root_quat_w, ex)

    def roll_rate(self) -> torch.Tensor:
        """(N,) angular velocity about the roll (body-x) axis, signed."""
        return (self.carton.data.root_ang_vel_w * self.roll_axis_w()).sum(dim=-1)

    def items_local(self) -> torch.Tensor:
        """(N, 2, 3) item centres in the CARTON BODY frame (can, candle)."""
        from isaaclab.utils.math import quat_apply_inverse

        pos = torch.stack([self.can.data.root_pos_w, self.candle.data.root_pos_w], dim=1)
        n = pos.shape[0]
        hq = self.carton.data.root_quat_w[:, None, :].expand(n, 2, 4).reshape(n * 2, 4)
        hp = self.carton.data.root_pos_w[:, None, :]
        return quat_apply_inverse(hq, (pos - hp).reshape(n * 2, 3)).reshape(n, 2, 3)

    def carton_upright(self) -> torch.Tensor:
        """(N,) bool: mouth-up within `upright_deg` AND centre at upright rest height
        (resting on the floor, not perched)."""
        c = self.cfg
        up = self.up_z() >= math.cos(math.radians(c.upright_deg))
        at_rest_h = (self.carton_pos()[:, 2] - c.rest_z_up).abs() < c.rest_z_tol
        return up & at_rest_h

    def carton_settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.carton.data.root_lin_vel_w.norm(dim=-1) < c.settle_v)
                & (self.carton.data.root_ang_vel_w.norm(dim=-1) < c.settle_w))

    def items_settled(self) -> torch.Tensor:
        """(N, 2) bool."""
        v = torch.stack([self.can.data.root_lin_vel_w.norm(dim=-1),
                         self.candle.data.root_lin_vel_w.norm(dim=-1)], dim=1)
        return v < self.cfg.settle_v

    def contained(self) -> torch.Tensor:
        """(N, 2) bool, geometric: item centre inside the UPRIGHT carton's interior in
        the carton's body frame, below the rim by > 2.2 cm (containment below the
        aperture) and above the floor. An item on the rim, leaning outside a wall, on
        the ground beside the carton, on top of — or trapped under — an overturned
        carton all fail (the upright gate or the bounds)."""
        c = self.cfg
        loc = self.items_local()
        inside = ((loc[:, :, 0].abs() < c.contain_x)
                  & (loc[:, :, 1].abs() < c.contain_y)
                  & (loc[:, :, 2] > c.contain_z[0]) & (loc[:, :, 2] < c.contain_z[1]))
        return inside & self.carton_upright().unsqueeze(-1)

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, CURRENT physical state: carton upright at rest on the floor and
        settled, BOTH items contained, items settled."""
        return (self.carton_upright() & self.carton_settled()
                & self.contained().all(dim=1) & self.items_settled().all(dim=1))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); otherwise 0.30 * righted latch
        + 0.25 * packed_can latch + 0.25 * packed_candle latch. ~0 for the null
        policy; a knock-over after success falls back to 0.80 (latched credit does
        not evaporate)."""
        partial = (0.30 * self.righted.float() + 0.25 * self.packed_can.float()
                   + 0.25 * self.packed_candle.float())
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Roll plant: torque `roll_tau` about the carton's body-x (ridge) axis —
        commanded in the BODY frame (the house wrench convention), which IS the roll
        axis at every roll angle, so no world-frame encoding is needed. Then the
        streak-gated rubric latches (a fly-through state earns nothing; NaN
        comparisons are False, so a garbage frame earns nothing either)."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 0] = self.roll_tau.clamp(-c.tau_max, c.tau_max)
        self.carton.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), tq)

        right_now = self.carton_upright() & self.carton_settled()
        self._streak_right = torch.where(right_now, self._streak_right + 1,
                                         torch.zeros_like(self._streak_right))
        self.righted = self.righted | (self._streak_right >= c.latch_streak)

        packed_now = self.contained() & self.carton_settled().unsqueeze(-1) \
            & self.items_settled()
        self._streak_can = torch.where(packed_now[:, 0], self._streak_can + 1,
                                       torch.zeros_like(self._streak_can))
        self._streak_candle = torch.where(packed_now[:, 1], self._streak_candle + 1,
                                          torch.zeros_like(self._streak_candle))
        self.packed_can = self.packed_can | (self._streak_can >= c.latch_streak)
        self.packed_candle = self.packed_candle | (self._streak_candle >= c.latch_streak)

    # ----- state (full, restorable) -------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"carton": self.carton, "can": self.can, "candle": self.candle}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("righted", "packed_can", "packed_candle", "roll_tau",
                               "_streak_right", "_streak_can", "_streak_candle")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        lx, ly, h = (v * 100 for v in c.outer)
        return (
            f"An empty brown cardboard carton ({lx:.0f} x {ly:.0f} cm footprint, "
            f"{h:.0f} cm tall) lies OVERTURNED on the floor: its open mouth faces the "
            f"floor and its closed base faces up. Its position and heading vary per "
            f"episode. On the floor near it, one to each side along the carton's long "
            f"(ridge) direction, stand two loose items: an ORANGE can (cylinder, "
            f"{2 * c.can_r * 100:.0f} cm across, {c.can_h * 100:.1f} cm tall) and a "
            f"GREEN candle (cylinder, {2 * c.candle_r * 100:.0f} cm across, "
            f"{c.candle_h * 100:.0f} cm tall). Which item is on which side varies.\n"
            f"Goal: turn the carton over so it stands upright with its mouth open to "
            f"the ceiling, then put BOTH items inside it through the mouth, and leave "
            f"everything at rest. Nothing can enter the carton while it is overturned "
            f"— turn it first: push near its top edge, above the tipping threshold, "
            f"so it pivots over a floor edge onto its side, then push again so it "
            f"pivots onto its base (it walks about 25 cm along the floor as it rolls "
            f"— roll it across its SHORT width, in a direction clear of the items). "
            f"Gentle pushes below the tipping threshold do nothing. An item left "
            f"leaning on the carton, perched on its rim, on top of the overturned "
            f"base, or trapped underneath it does not count; only items resting "
            f"inside the upright carton, below its rim, count. The carton must end "
            f"upright and settled on the floor with both items settled inside."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Turn the overturned cardboard carton over so its mouth faces up, then "
            "put the orange can and the green candle inside it. Both items must end "
            "settled inside the upright carton; items on, under, or beside the "
            "carton do not count."
        )


# ----- runnable env: scene physics only (NullRobot solve/smoke) -> "simgen.carton_flip_pack" ----
register_env("simgen", lambda: EnvCfg(scene="carton_flip_pack", robot="null"))
