"""PressurePlateHeistScene — swap the idol's weight with a counterweight BEFORE
lifting it, or the pressure-plate alarm trips ("beat the buzz", museum-heist form).

Derived from the RLBench `beat_the_buzz` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is a keep-out traversal — grasp a wand and guide its loop
along a buzz wire without ever touching it; "don't trigger the buzzer" is enforced
by proximity along a path. Here NOTHING is won by careful avoidance: the alarm is a
spring-loaded PRESSURE PLATE under the golden idol, and it fires the moment the
plate is UNLOADED — however slowly and precisely the idol is lifted. The only way
to beat the buzz is a weight substitution: put a heavy-enough counterweight on the
plate FIRST (the dark granite block; the visually similar white foam block is far
too light), and only then carry the idol to the delivery pad. The plan skeleton is
target-mass selection + an ordered swap that maintains a load invariant — not a
guided collision-free sweep.

Mechanics (plain rigid bodies + one authored USD D6 joint, the lamp_off/door-hinge
pattern): the bronze plate is a dynamic slab on a D6 joint (pedestal -> plate) that
frees exactly ONE translation, the vertical, with travel [0, 35 mm]. `post_step`
owns the plate's wrench slot and applies a constant upward spring force
`spring_f` = 3.5 N — chosen strictly between (plate + foam) * g ~ 2.6 N and
(plate + idol) * g ~ 6.9 N, so the idol or the granite (0.7 kg) pins the plate down
while the bare or foam-loaded plate rises through the 15 mm trigger height within
~0.1 s. The alarm is a LATCH on that physical plate extension: once tripped it
never resets within the episode, so an end state that is geometrically identical
to the goal still fails if the plate ever rose. `post_step` also applies the
solver's `drive_f` buffers (the stand-in for the Franka's grasp-lift wrench on
idol / granite) and latches rubric progress.

Rubric (graded 0..1, latching transient achievement — anchored in the demonstrated
solve.py trajectory: counterweight on, idol clear, idol delivered):
  - `hold_latch`  (0.35): the granite block has rested fully ON the depressed
    plate, settled, alarm unfired;
  - `clear_latch` (0.15): the idol has left the pedestal region (or been lifted
    high) while the plate stayed down and the alarm stayed unfired;
  - `pad_latch`   (0.15): the idol has stood upright, settled, on the delivery pad;
  - success (-> 1.0): idol upright + settled on the pad, plate currently held
    down, alarm NEVER fired, everything still. score() == 1.0 iff success().
  - The whole score is multiplied by (1 - alarm): tripping the alarm zeroes it —
    credit only evaporates under incorrect behavior. Null policy scores ~0.

Per-episode randomization (readback-verified in smoke): which side of the table
the granite is on (foam takes the other), both blocks' positions and yaws, the
delivery pad's position, and the idol's jitter + yaw on the plate.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PressurePlateHeistSceneCfg(BaseCfg):
    """Config for `PressurePlateHeistScene`. Geometry is derived once in `__post_init__` so
    the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    trigger_h: float = tunable(0.015)  # plate risen past this = alarm (latched forever)
    down_tol: float = tunable(0.006)  # plate extension below this counts as "held down"
    pad_xy_tol: float = tunable(0.050)  # idol centre within this of the pad centre (per axis)
    pad_z_tol: float = tunable(0.012)  # idol base within this of the pad top
    upright_max_deg: float = tunable(15.0)  # idol axis within this of world-up on the pad
    on_plate_xy: float = tunable(0.065)  # block centre within this of the plate axis (per axis)
    on_plate_z_tol: float = tunable(0.012)  # block bottom within this of the plate top
    clear_r: float = tunable(0.17)  # idol this far (xy) from the pedestal axis = removed...
    clear_z: float = tunable(0.62)  # ...or lifted above this height (world z - origin)
    settle_lin: float = tunable(0.05)  # max |lin vel| (m/s) when judging
    settle_ang: float = tunable(1.0)  # max |ang vel| (rad/s) when judging

    # --- tunable: plant ----------------------------------------------------------------------
    spring_f: float = tunable(3.5)  # N, constant upward spring on the plate (post_step)
    plate_mass: float = tunable(0.20)
    idol_mass: float = tunable(0.50)
    granite_mass: float = tunable(0.70)  # (plate + granite) * g ~ 8.8 N >> spring_f: holds
    foam_mass: float = tunable(0.06)  # (plate + foam) * g ~ 2.6 N < spring_f: plate rises

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    block_x_range: tuple = tunable((0.00, 0.18))  # block slots on the table, robot side
    block_y_range: tuple = tunable((0.20, 0.30))  # |y| band; granite side sampled per episode
    pad_x_range: tuple = tunable((0.08, 0.20))  # delivery pad centre bands
    pad_y_range: tuple = tunable((-0.06, 0.06))
    idol_jitter: float = tunable(0.015)  # idol xy jitter on the plate

    # --- info: structure (env-local coordinates; table top at z0) ----------------------------
    table_center: tuple = info((0.05, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.00, 0.08))  # top at z0 = 0.40
    pedestal_xy: tuple = info((-0.18, 0.0))
    pedestal_size: tuple = info((0.22, 0.22, 0.10))  # top at z0 + 0.10
    plate_size: tuple = info((0.19, 0.19, 0.02))
    plate_gap: float = info(0.002)  # joint-limit floor above the pedestal top
    travel: float = info(0.035)  # D6 vertical travel of the plate
    idol_base: float = info(0.045)  # idol footprint (square)
    idol_h: float = info(0.110)
    block_s: float = info(0.055)  # granite / foam cube edge
    pad_size: tuple = info((0.15, 0.15, 0.012))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    z0: float = field(default=None, init=False)  # table top
    plate_z_lo: float = field(default=None, init=False)  # plate centre z at extension 0
    plate_top_lo: float = field(default=None, init=False)  # plate top face at extension 0
    pad_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.z0 = self.table_center[2] + self.table_size[2] / 2
        ped_top = self.z0 + self.pedestal_size[2]
        self.plate_z_lo = ped_top + self.plate_gap + self.plate_size[2] / 2
        self.plate_top_lo = self.plate_z_lo + self.plate_size[2] / 2
        self.pad_top = self.z0 + self.pad_size[2]


# ----- scene -----------------------------------------------------------------------------------
class PressurePlateHeistScene(BaseScene):
    cfg: PressurePlateHeistSceneCfg

    def __init__(self, cfg: PressurePlateHeistSceneCfg | None = None) -> None:
        super().__init__(cfg or PressurePlateHeistSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + stone pedestal + green delivery pad, the dynamic
        bronze pressure plate (D6-hung in bind), the golden idol on it, and the two candidate
        counterweights (dark granite, white foam) on the table."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.33, 0.20))
        stone = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.50, 0.53))
        bronze = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15))
        gold = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.75, 0.20))
        granite_c = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.22, 0.24))
        foam_c = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.93, 0.90))
        green = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.50, 0.15))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives idol/granite/plate with external wrenches, which do NOT wake a
        # sleeping body — sleep_threshold=0 keeps the plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.7, restitution=0.05)

        def dyn(mass: float, damp: float = 0.2) -> sim_utils.RigidBodyPropertiesCfg:
            return sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                linear_damping=damp, angular_damping=damp, **live)

        px, py = c.pedestal_xy
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size, rigid_props=kin, collision_props=coll,
                    visual_material=wood),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=c.pedestal_size, rigid_props=kin, collision_props=coll,
                    visual_material=stone),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py, c.z0 + c.pedestal_size[2] / 2)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=c.pad_size, rigid_props=kin, collision_props=coll,
                    visual_material=green),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.14, 0.0, c.z0 + c.pad_size[2] / 2)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CuboidCfg(
                    size=c.plate_size,
                    rigid_props=dyn(c.plate_mass, damp=1.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=bronze),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.plate_z_lo)),
            ),
            "idol": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Idol",
                spawn=sim_utils.CuboidCfg(
                    size=(c.idol_base, c.idol_base, c.idol_h),
                    rigid_props=dyn(c.idol_mass),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.idol_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=gold),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py, c.plate_top_lo + c.idol_h / 2 + 0.001)),
            ),
            "granite": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Granite",
                spawn=sim_utils.CuboidCfg(
                    size=(c.block_s,) * 3,
                    rigid_props=dyn(c.granite_mass),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.granite_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=granite_c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.10, 0.25, c.z0 + c.block_s / 2 + 0.002)),
            ),
            "foam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Foam",
                spawn=sim_utils.CuboidCfg(
                    size=(c.block_s,) * 3,
                    rigid_props=dyn(c.foam_mass),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.foam_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=foam_c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.10, -0.25, c.z0 + c.block_s / 2 + 0.002)),
            ),
        }
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.plate: RigidObject = env.iscene["plate"]
        self.idol: RigidObject = env.iscene["idol"]
        self.granite: RigidObject = env.iscene["granite"]
        self.foam: RigidObject = env.iscene["foam"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        self._author_d6()
        self._author_idol_head()
        # Episode state.
        self.granite_side = torch.ones(n, dtype=torch.long, device=dev)  # +1 / -1 y side
        self.alarm = torch.zeros(n, device=dev)  # LATCH: plate ever rose past trigger_h
        self.hold_latch = torch.zeros(n, device=dev)
        self.clear_latch = torch.zeros(n, device=dev)
        self.pad_latch = torch.zeros(n, device=dev)
        # External drive input (solve.py and smoke probes write; post_step consumes and
        # owns idol/granite/foam/plate wrench slots — never call
        # set_external_force_and_torque on those bodies directly).
        self.drive_f = torch.zeros(n, 3, 3, device=dev)  # [idol, granite, foam] world force

    def _author_d6(self) -> None:
        """Per env: a D6 joint pedestal -> plate freeing exactly transZ in [0, travel].
        The joint pair never collides; the plate floats on the joint-limit floor 2 mm
        above the pedestal top."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        ped_cz = c.z0 + c.pedestal_size[2] / 2
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            if stage.GetPrimAtPath(f"{base}/plate_joint").IsValid():
                continue
            j = UsdPhysics.Joint.Define(stage, f"{base}/plate_joint")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.plate_z_lo - ped_cz))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transY", "rotX", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)  # low > high = locked
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transZ")
            lim.CreateLowAttr(0.0)
            lim.CreateHighAttr(c.travel)

    def _author_idol_head(self) -> None:
        """Visual-only gold head sphere on the idol (env_0; others compose by reference).
        Authored idempotently; no collision — pure identification dressing."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        path = "/World/envs/env_0/Idol/head"
        if stage.GetPrimAtPath(path).IsValid():
            return
        sp = UsdGeom.Sphere.Define(stage, path)
        sp.CreateRadiusAttr(0.020)
        UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, c.idol_h / 2 + 0.014))
        sp.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.80, 0.25)])

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plate seated at the bottom of its travel, idol on it (jitter +
        yaw), granite on a sampled side of the table with foam on the other (positions +
        yaws sampled), pad placed; latches and drives cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        px, py = c.pedestal_xy

        # NOTE: the FIRST randint draw after manual_seed is degenerate on this stack —
        # all discrete draws come from torch.rand comparisons instead.
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        self.granite_side[env_ids] = side.long()
        self.alarm[env_ids] = 0.0
        self.hold_latch[env_ids] = 0.0
        self.clear_latch[env_ids] = 0.0
        self.pad_latch[env_ids] = 0.0
        self.drive_f[env_ids] = 0.0

        def yaw_quat(m_: int) -> torch.Tensor:
            half = (torch.rand(m_, device=dev) * 2 - 1) * math.pi / 2
            q = torch.zeros(m_, 4, device=dev)
            q[:, 0] = torch.cos(half)
            q[:, 3] = torch.sin(half)
            return q

        # --- plate: bottom of travel, identity ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = c.plate_z_lo
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- idol: on the plate, jitter + yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px + (torch.rand(m, device=dev) * 2 - 1) * c.idol_jitter
        st[:, 1] = py + (torch.rand(m, device=dev) * 2 - 1) * c.idol_jitter
        st[:, 2] = c.plate_top_lo + c.idol_h / 2 + 0.001
        st[:, 3:7] = yaw_quat(m)
        st[:, 0:3] += origin
        self.idol.write_root_state_to_sim(st, env_ids)

        # --- blocks: granite on the sampled side, foam on the other ---
        xr, yr = c.block_x_range, c.block_y_range
        for body, s in ((self.granite, side), (self.foam, -side)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xr[0] + (xr[1] - xr[0]) * torch.rand(m, device=dev)
            st[:, 1] = s * (yr[0] + (yr[1] - yr[0]) * torch.rand(m, device=dev))
            st[:, 2] = c.z0 + c.block_s / 2 + 0.002
            st[:, 3:7] = yaw_quat(m)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- pad: sampled position on the table ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pad_x_range[0] + (c.pad_x_range[1] - c.pad_x_range[0]) \
            * torch.rand(m, device=dev)
        st[:, 1] = c.pad_y_range[0] + (c.pad_y_range[1] - c.pad_y_range[0]) \
            * torch.rand(m, device=dev)
        st[:, 2] = c.z0 + c.pad_size[2] / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def plate_ext(self) -> torch.Tensor:
        """(N,) plate extension above the bottom of its travel (m)."""
        return (self.plate.data.root_pos_w[:, 2] - self.env_origins[:, 2]) - self.cfg.plate_z_lo

    def plate_down(self) -> torch.Tensor:
        return self.plate_ext() < self.cfg.down_tol

    def _still(self, body: RigidObject) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,) bool: idol, granite and plate at rest."""
        return self._still(self.idol) & self._still(self.granite) & self._still(self.plate)

    def granite_on_plate(self) -> torch.Tensor:
        """(N,) bool: granite fully on the plate footprint, its bottom at the plate top."""
        c = self.cfg
        g = self.granite.data.root_pos_w - self.env_origins
        p = self.plate.data.root_pos_w - self.env_origins
        on_xy = ((g[:, 0:2] - p[:, 0:2]).abs() < c.on_plate_xy).all(dim=1)
        g_bot = g[:, 2] - c.block_s / 2
        p_top = p[:, 2] + c.plate_size[2] / 2
        return on_xy & ((g_bot - p_top).abs() < c.on_plate_z_tol)

    def idol_up(self) -> torch.Tensor:
        """(N,) bool: idol long axis within `upright_max_deg` of world-up."""
        q = self.idol.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return up_z >= math.cos(math.radians(self.cfg.upright_max_deg))

    def idol_clear(self) -> torch.Tensor:
        """(N,) bool: idol out of the pedestal region (xy) or lifted high."""
        c = self.cfg
        r = self.idol.data.root_pos_w - self.env_origins
        ped = torch.tensor(c.pedestal_xy, device=r.device)
        far = (r[:, 0:2] - ped).norm(dim=-1) > c.clear_r
        high = r[:, 2] > c.clear_z
        return far | high

    def idol_on_pad(self) -> torch.Tensor:
        """(N,) bool: idol upright, centred on the pad, its base on the pad top."""
        c = self.cfg
        r = self.idol.data.root_pos_w - self.env_origins
        p = self.pad.data.root_pos_w - self.env_origins
        on_xy = ((r[:, 0:2] - p[:, 0:2]).abs() < c.pad_xy_tol).all(dim=1)
        base = r[:, 2] - c.idol_h / 2
        on_z = (base - c.pad_top).abs() < c.pad_z_tol
        return on_xy & on_z & self.idol_up()

    def success(self) -> torch.Tensor:
        """(N,) bool: idol delivered (upright + settled on the pad), plate currently held
        down, alarm NEVER fired, everything still."""
        return self.idol_on_pad() & self.plate_down() & (self.alarm < 0.5) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: (1 - alarm) * (1.0 if success else 0.35 * hold_latch
        + 0.15 * clear_latch + 0.15 * pad_latch). Exactly 1.0 iff success(); zeroed
        forever once the alarm fires; ~0 for the null policy."""
        partial = 0.35 * self.hold_latch + 0.15 * self.clear_latch + 0.15 * self.pad_latch
        s = torch.where(self.success(), torch.ones_like(partial), partial)
        return (1.0 - self.alarm) * s

    # ----- step-coupled mechanics (every step) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """The plant: constant upward spring on the plate + the solver's drive wrenches on
        idol / granite (owns those wrench slots), then latch the alarm and rubric progress."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        zero = torch.zeros(n, 1, 3, device=dev)
        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 2] = c.spring_f
        self.plate.set_external_force_and_torque(f, zero)
        self.idol.set_external_force_and_torque(self.drive_f[:, 0:1, :], zero)
        self.granite.set_external_force_and_torque(self.drive_f[:, 1:2, :], zero)
        self.foam.set_external_force_and_torque(self.drive_f[:, 2:3, :], zero)

        ext = self.plate_ext()
        fired = (ext > c.trigger_h).float()
        armed = self.alarm < 0.5
        down = self.plate_down()
        hold = (self.granite_on_plate() & down & armed & self._still(self.granite)).float()
        clear = (self.idol_clear() & down & armed).float()
        pad = (self.idol_on_pad() & armed & self._still(self.idol)).float()
        # A diverged step must not latch: torch.maximum propagates NaN. Garbage earns 0.
        fired = torch.nan_to_num(fired, nan=0.0, posinf=0.0, neginf=0.0)
        hold = torch.nan_to_num(hold, nan=0.0, posinf=0.0, neginf=0.0)
        clear = torch.nan_to_num(clear, nan=0.0, posinf=0.0, neginf=0.0)
        pad = torch.nan_to_num(pad, nan=0.0, posinf=0.0, neginf=0.0)
        self.alarm = torch.maximum(self.alarm, fired)
        self.hold_latch = torch.maximum(self.hold_latch, hold)
        self.clear_latch = torch.maximum(self.clear_latch, clear)
        self.pad_latch = torch.maximum(self.pad_latch, pad)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("granite_side", "alarm", "hold_latch", "clear_latch",
                               "pad_latch", "drive_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"plate": self.plate, "idol": self.idol, "granite": self.granite,
                "foam": self.foam, "pad": self.pad}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden table holds, toward its rear, a gray stone pedestal "
            f"({c.pedestal_size[0] * 100:.0f} cm square) topped by a BRONZE PRESSURE PLATE "
            f"({c.plate_size[0] * 100:.0f} cm square) that can slide vertically about "
            f"{c.travel * 100:.1f} cm. A golden idol (a {c.idol_base * 100:.1f} cm square, "
            f"{c.idol_h * 100:.0f} cm tall gold column with a small gold head-sphere) stands "
            f"on the plate; its weight holds the plate pressed down against an internal "
            f"spring. On the table nearby lie two {c.block_s * 100:.1f} cm cubes: a DARK "
            f"GRAY GRANITE block (heavy, about {c.granite_mass * 1000:.0f} g) on one side "
            f"and a WHITE FOAM block (very light, about {c.foam_mass * 1000:.0f} g) on the "
            f"other — which side each is on changes every episode. A flat GREEN DELIVERY "
            f"PAD ({c.pad_size[0] * 100:.0f} cm square) lies on the table; its position "
            f"also changes per episode.\n"
            f"The plate is alarmed: if it ever rises more than {c.trigger_h * 100:.1f} cm "
            f"the alarm trips PERMANENTLY — the episode is failed even if the plate is "
            f"pushed back down afterwards. The spring is strong enough to lift the bare "
            f"plate and also the plate with only the foam block on it, but NOT the plate "
            f"with the granite block (or the idol) on it.\n"
            f"Goal: steal the idol without tripping the alarm. First place the granite "
            f"block onto the pressure plate beside the idol (the plate is large enough for "
            f"both), so the plate stays held down; only then lift the idol off and stand it "
            f"UPRIGHT, centred on the green delivery pad, and let everything settle. "
            f"Lifting the idol before a sufficient counterweight is on the plate — however "
            f"slowly or carefully — lets the plate rise and trips the alarm. Using the foam "
            f"block as the counterweight also trips it (too light). The task is complete "
            f"when the idol stands still and upright on the pad, the plate is still held "
            f"down, and the alarm never fired."
        )

    def instruction(self) -> str:
        return (
            "Place the dark granite block onto the bronze pressure plate next to the "
            "golden idol, then lift the idol and stand it upright on the green delivery "
            "pad. If the plate is unloaded before the granite is on it — the white foam "
            "block is too light to count — the plate rises and the alarm trips, failing "
            "the task permanently."
        )


# Guarded registration: the forge may import this module under two names.
if "pressure_plate_heist" not in SCENES.list():
    SCENES.register("pressure_plate_heist", PressurePlateHeistScene)
if "simgen.pressure_plate_heist" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="pressure_plate_heist", robot="null"))
