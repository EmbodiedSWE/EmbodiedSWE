"""WeightAirlockScene — lock the red cargo cube inside a roofed vault whose sliding
gate is interlocked to a weight-operated pressure plate.

Derived from the ManiSkill `pick_cube_v1` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's whole plan is ONE grasp of a red cube and ONE guided free-space
lift to a floating goal sphere — position change, no other object matters, no
mechanism, no order. Here the destination is the interior of a ROOFED vault: lifting
the cargo and dropping it at the target region lands it on the roof and earns nothing
(the seed's plan is reproduced and rejected in smoke). The only way in is a side
doorway barred by a sliding GATE, and the gate is interlocked: it slides open only
while a heavy BLUE weight cube presses a spring-loaded pressure PLATE down on its
pedestal, and springs shut when the weight is removed. A same-size GRAY decoy cube is
hollow and too light to depress the plate (physically — its weight is under the
spring preload). Success is the LOCKED end state: red cargo fully inside the vault,
gate closed, plate back up, everything settled — which physically forces the order
weight ON -> cargo THROUGH -> weight OFF.

Mechanics (plain rigid bodies + authored USD D6 joints — the proven pattern):
  - GATE: one dynamic slab hung on a D6 (back wall -> gate) freeing exactly transX
    within [0, gate_travel] (physical end stops); post_step pushes it toward OPEN
    while the plate is depressed, toward CLOSED otherwise (the interlock).
  - PLATE: one dynamic slab on a D6 (pedestal -> plate) freeing exactly transZ within
    [-plate_travel, 0]; post_step applies a constant upward spring force. Empty (or
    decoy-loaded) the spring holds it at the TOP stop; the blue weight's gravity
    exceeds the preload and drives it to the BOTTOM stop through real contact.
  - post_step also owns the three cubes' wrench slots: it consumes the `push_f`
    world-frame buffers (solve.py's stand-in for fingertip pushes), pre-encoded
    against the wrench frame drag (applied wrenches rotate with rotation-since-reset
    on this stack).

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.25 * latch_open: the interlock was operated (plate depressed AND gate at its
    open stop, simultaneously — latched);
  - 0.35 * latch_inside: the red cargo was fully inside the vault interior while
    slow (latched — a fast transit that ends outside latches nothing);
  - 1.0 iff success(): red inside NOW + gate closed + plate up + all settled.
  ~0 for the null policy; latched credit never evaporates under correct behavior.

Per-episode randomization (readback-verified in smoke): the three cubes are dealt to
three staging slots by a sampled permutation (so position never identifies a cube —
color does), each with xy jitter and free yaw.

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
class WeightAirlockSceneCfg(BaseCfg):
    """Config for `WeightAirlockScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.05)  # max |lin vel| on every dynamic body when judging (m/s)
    red_slow_latch: float = tunable(0.10)  # red counts as slow below this when latching (m/s)
    gate_closed_tol: float = tunable(0.010)  # gate within this of the closed stop = closed (m)
    open_latch_disp: float = tunable(0.130)  # gate past this = at the open stop (m)
    plate_gap_tol: float = tunable(0.005)  # plate within this of a stop = at that stop (m)
    inside_margin: float = tunable(0.005)  # extra clearance beyond the cube half-size (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    slot_jitter: float = tunable(0.04)  # uniform +/- xy jitter per cube at reset (m)
    yaw_deg: float = tunable(180.0)  # uniform +/- yaw per cube at reset

    # --- tunable: plant ----------------------------------------------------------------------
    red_size: float = tunable(0.05)
    red_mass: float = tunable(0.06)
    blue_size: float = tunable(0.06)
    blue_mass: float = tunable(0.80)  # >> spring preload margin: depresses the plate
    decoy_size: float = tunable(0.06)  # same size/shape as blue — color is the only cue
    decoy_mass: float = tunable(0.08)  # << spring preload margin: cannot depress it
    spring_force: float = tunable(2.5)  # N, constant upward on the plate (post_step)
    gate_force: float = tunable(2.0)  # N, interlock drive toward open/closed (post_step)
    gate_mass: float = tunable(0.15)
    plate_mass: float = tunable(0.06)  # empty preload margin = 2.5 - 0.59 = 1.9 N

    # --- info: structure (env-local coordinates; table top at z0) ----------------------------
    table_center: tuple = info((0.05, 0.0, 0.36))
    table_size: tuple = info((0.90, 1.00, 0.08))  # top at z0 = 0.40
    z0: float = info(0.40)
    vault_center: tuple = info((0.0, 0.16))  # interior x in +-0.10, y in [0.06, 0.26]
    inner_x_half: float = info(0.10)
    inner_y: tuple = info((0.06, 0.26))  # inner faces of front / back walls
    inner_h: float = info(0.12)  # interior height (roof underside)
    wall_t: float = info(0.02)
    door_half: float = info(0.06)  # doorway x in +-this (gap in the front wall)
    gate_size: tuple = info((0.135, 0.015, 0.120))
    gate_y: float = info(0.0265)  # gate slab plane (6 mm clear of the wall face)
    gate_lift: float = info(0.004)  # gate bottom above the table (no rubbing contact)
    gate_travel: float = info(0.14)  # closed stop at x = 0, open stop at x = +travel
    ped_center: tuple = info((0.22, -0.07))  # clear of the gate's slide path (gate y~0.03)
    ped_size: tuple = info((0.10, 0.10, 0.05))  # top at z0 + 0.05
    plate_size: tuple = info((0.09, 0.09, 0.012))
    plate_travel: float = info(0.02)  # transZ stops; rest = TOP stop (spring holds)
    plate_clear: float = info(0.006)  # plate bottom above the pedestal at the BOTTOM stop
    slots: tuple = info(((-0.22, -0.14), (0.00, -0.20), (0.24, -0.22)))  # staging slots
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    gate_center: tuple = field(default=None, init=False)  # closed pose (spawn = joint zero)
    plate_rest_z: float = field(default=None, init=False)  # plate centre at the TOP stop
    plate_bottom_z: float = field(default=None, init=False)  # plate centre at the BOTTOM stop
    inside_x_half: float = field(default=None, init=False)  # rubric bound for the red centre
    inside_y_min: float = field(default=None, init=False)
    inside_y_max: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.gate_center = (0.0, self.gate_y, self.z0 + self.gate_lift + self.gate_size[2] / 2)
        ped_top = self.z0 + self.ped_size[2]
        self.plate_bottom_z = ped_top + self.plate_clear + self.plate_size[2] / 2
        self.plate_rest_z = self.plate_bottom_z + self.plate_travel
        rh = self.red_size / 2
        self.inside_x_half = self.inner_x_half - rh - self.inside_margin
        self.inside_y_min = self.inner_y[0] + rh + 0.010  # fully past the inner wall face
        self.inside_y_max = self.inner_y[1] - rh - self.inside_margin


def _quat_z(rad: torch.Tensor) -> torch.Tensor:
    """(N,) angle about +z -> (N, 4) wxyz."""
    half = rad / 2
    q = torch.zeros(rad.shape[0], 4, device=rad.device)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


# ----- scene -----------------------------------------------------------------------------------
class WeightAirlockScene(BaseScene):
    cfg: WeightAirlockSceneCfg

    def __init__(self, cfg: WeightAirlockSceneCfg | None = None) -> None:
        super().__init__(cfg or WeightAirlockSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + vault shell (walls, roof, rail, pedestal), the
        dynamic gate + plate (D6 joints authored in bind), three dynamic cubes."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.z0
        vx, vy = c.vault_center
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))
        concrete = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.50))
        roof_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.33))
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.20))
        yellow = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.80, 0.10))
        red_m = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.08))
        blue_m = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.25, 0.85))
        gray_m = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.62, 0.64))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives gate/plate/cubes with external wrenches that do NOT wake a
        # sleeping body — sleep_threshold=0 keeps the plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

        def kin_box(name: str, center: tuple, size: tuple, mat) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(size=size, rigid_props=kin, collision_props=coll,
                                          visual_material=mat),
                init_state=RigidObjectCfg.InitialStateCfg(pos=center),
            )

        wz = z0 + c.inner_h / 2  # wall centre height
        seg_w = (c.inner_x_half + c.wall_t) - c.door_half  # front wall segment width
        seg_x = c.door_half + seg_w / 2
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
            "table": kin_box("Table", c.table_center, c.table_size, wood),
            # --- the vault shell (all kinematic; never moves) ---
            "back_wall": kin_box(
                "BackWall", (vx, c.inner_y[1] + c.wall_t / 2, wz),
                (2 * (c.inner_x_half + c.wall_t), c.wall_t, c.inner_h), concrete),
            "left_wall": kin_box(
                "LeftWall", (vx - c.inner_x_half - c.wall_t / 2, vy, wz),
                (c.wall_t, c.inner_y[1] - c.inner_y[0] + 2 * c.wall_t, c.inner_h), concrete),
            "right_wall": kin_box(
                "RightWall", (vx + c.inner_x_half + c.wall_t / 2, vy, wz),
                (c.wall_t, c.inner_y[1] - c.inner_y[0] + 2 * c.wall_t, c.inner_h), concrete),
            "front_left": kin_box(
                "FrontLeft", (vx - seg_x, c.inner_y[0] - c.wall_t / 2, wz),
                (seg_w, c.wall_t, c.inner_h), concrete),
            "front_right": kin_box(
                "FrontRight", (vx + seg_x, c.inner_y[0] - c.wall_t / 2, wz),
                (seg_w, c.wall_t, c.inner_h), concrete),
            "roof": kin_box(
                "Roof", (vx, vy, z0 + c.inner_h + 0.01),
                (2 * (c.inner_x_half + c.wall_t), c.inner_y[1] - c.inner_y[0] + 2 * c.wall_t,
                 0.02), roof_col),
            # decor rail bar above the gate's full travel (visual sense for the slide)
            "rail": kin_box(
                "Rail", (vx + 0.07, c.gate_y, z0 + 0.137), (0.30, 0.02, 0.010), roof_col),
            "pedestal": kin_box(
                "Pedestal", (c.ped_center[0], c.ped_center[1], z0 + c.ped_size[2] / 2),
                c.ped_size, concrete),
        }

        dyn = dict(max_depenetration_velocity=0.5, solver_position_iteration_count=16,
                   solver_velocity_iteration_count=4, **live)
        out["gate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Gate",
            spawn=sim_utils.CuboidCfg(
                size=c.gate_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=15.0, angular_damping=5.0, **dyn),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                collision_props=coll, visual_material=steel),
            init_state=RigidObjectCfg.InitialStateCfg(pos=c.gate_center),
        )
        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.CuboidCfg(
                size=c.plate_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=12.0, angular_damping=5.0, **dyn),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                collision_props=coll, visual_material=yellow),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.ped_center[0], c.ped_center[1], c.plate_rest_z)),
        )
        for name, size, mass, mat, damp in (
                ("red_cube", c.red_size, c.red_mass, red_m, 0.10),
                ("blue_cube", c.blue_size, c.blue_mass, blue_m, 0.20),
                ("decoy_cube", c.decoy_size, c.decoy_mass, gray_m, 0.10)):
            sx, sy = c.slots[("red_cube", "blue_cube", "decoy_cube").index(name)]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=(size, size, size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=damp, angular_damping=damp, **dyn),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=coll, visual_material=mat),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, z0 + size / 2 + 0.003)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.gate: RigidObject = env.iscene["gate"]
        self.plate: RigidObject = env.iscene["plate"]
        self.red: RigidObject = env.iscene["red_cube"]
        self.blue: RigidObject = env.iscene["blue_cube"]
        self.decoy: RigidObject = env.iscene["decoy_cube"]
        self.cubes = [self.red, self.blue, self.decoy]  # push_f order: 0=red 1=blue 2=decoy
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # Episode state.
        self.spawn_xy = torch.zeros(n, 3, 2, device=dev)  # sampled cube slots (readback ref)
        self.spawn_yaw = torch.zeros(n, 3, device=dev)
        self.latch_open = torch.zeros(n, dtype=torch.bool, device=dev)
        self.latch_inside = torch.zeros(n, dtype=torch.bool, device=dev)
        # External push input (solve.py and smoke probes write WORLD-frame forces; post_step
        # consumes + owns the cubes' wrench slots — never call set_external_* directly).
        self.push_f = torch.zeros(n, 3, 3, device=dev)
        self._q_ref = torch.zeros(n, 3, 4, device=dev)  # cube quats at reset (drag pre-encode)
        self._q_ref[:, :, 0] = 1.0

    def _author_joints(self) -> None:
        """Per env: two D6 joints. Gate: BackWall -> Gate freeing exactly transX within
        [0, gate_travel] (0 = the closed spawn pose). Plate: Pedestal -> Plate freeing
        exactly transZ within [-plate_travel, 0] (0 = the TOP-stop spawn pose). Both
        joints filter their pair's collisions (the plate's bottom stop is the joint
        limit, never a contact)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        bw_c = (c.vault_center[0], c.inner_y[1] + c.wall_t / 2, c.z0 + c.inner_h / 2)
        gc = c.gate_center
        pc = (c.ped_center[0], c.ped_center[1], c.z0 + c.ped_size[2] / 2)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.Joint.Define(stage, f"{base}/gate_joint")
            j.CreateBody0Rel().SetTargets([f"{base}/BackWall"])
            j.CreateBody1Rel().SetTargets([f"{base}/Gate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(gc[0] - bw_c[0], gc[1] - bw_c[1], gc[2] - bw_c[2]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transY", "transZ", "rotX", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)  # low > high = locked
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transX")
            lim.CreateLowAttr(0.0)
            lim.CreateHighAttr(c.gate_travel)

            j = UsdPhysics.Joint.Define(stage, f"{base}/plate_joint")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.plate_rest_z - pc[2]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transY", "rotX", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transZ")
            lim.CreateLowAttr(-c.plate_travel)
            lim.CreateHighAttr(0.0)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: deal the three cubes to the three staging slots by a sampled
        permutation (position never identifies a cube — color does), each with xy jitter
        and free yaw; gate to the closed stop, plate to the top stop; clear latches and
        drives. Uses torch.rand throughout (the first randint after manual_seed is
        degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # cube i -> slot perm[:, i]
        slots = torch.tensor(c.slots, device=dev)  # (3, 2)
        yaw_amp = math.radians(c.yaw_deg)
        sizes = (c.red_size, c.blue_size, c.decoy_size)
        for i, body in enumerate(self.cubes):
            xy = slots[perm[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.z0 + sizes[i] / 2 + 0.003
            st[:, 3:7] = _quat_z(yaw)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)
            self.spawn_xy[env_ids, i] = xy
            self.spawn_yaw[env_ids, i] = yaw
            self._q_ref[env_ids, i] = st[:, 3:7]

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = torch.tensor(c.gate_center, device=dev) + origin
        st[:, 3] = 1.0
        self.gate.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.ped_center[0]
        st[:, 1] = c.ped_center[1]
        st[:, 2] = c.plate_rest_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        self.latch_open[env_ids] = False
        self.latch_inside[env_ids] = False
        self.push_f[env_ids] = 0.0

    # ----- readings -----------------------------------------------------------------------------
    def gate_disp(self) -> torch.Tensor:
        """(N,) gate displacement from the closed stop along +x (m)."""
        return (self.gate.data.root_pos_w - self.env_origins)[:, 0] - self.cfg.gate_center[0]

    def plate_z(self) -> torch.Tensor:
        """(N,) plate centre height (env-local)."""
        return (self.plate.data.root_pos_w - self.env_origins)[:, 2]

    def plate_depressed(self) -> torch.Tensor:
        """(N,) bool: plate at (near) the BOTTOM stop — the interlock's OPEN input."""
        return self.plate_z() < self.cfg.plate_bottom_z + self.cfg.plate_gap_tol

    def plate_up(self) -> torch.Tensor:
        """(N,) bool: plate back at (near) the TOP stop."""
        return self.plate_z() > self.cfg.plate_rest_z - self.cfg.plate_gap_tol

    def gate_closed(self) -> torch.Tensor:
        """(N,) bool: gate at (near) the closed stop, covering the doorway."""
        return self.gate_disp() < self.cfg.gate_closed_tol

    def gate_open(self) -> torch.Tensor:
        """(N,) bool: gate at (near) the open stop, clear of the doorway."""
        return self.gate_disp() > self.cfg.open_latch_disp

    def red_inside(self) -> torch.Tensor:
        """(N,) bool, geometric: red centre fully inside the vault interior (past the
        inner wall face with margin, under the roof)."""
        c = self.cfg
        p = self.red.data.root_pos_w - self.env_origins
        return ((p[:, 0] - c.vault_center[0]).abs() < c.inside_x_half) \
            & (p[:, 1] > c.inside_y_min) & (p[:, 1] < c.inside_y_max) \
            & (p[:, 2] > c.z0) & (p[:, 2] < c.z0 + c.inner_h - 0.02)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body |lin vel| below settle_lin."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.gate, self.plate, self.red, self.blue, self.decoy):
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin
        return ok

    def success(self) -> torch.Tensor:
        """(N,) bool: the LOCKED end state — red cargo fully inside the vault, gate at
        the closed stop, plate back at the top stop (weight removed), all settled
        (current, physical state)."""
        return self.red_inside() & self.gate_closed() & self.plate_up() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); else 0.25 * latch_open (the interlock
        was operated) + 0.35 * latch_inside (the cargo was fully inside while slow).
        Latched credit never evaporates; ~0 for the null policy."""
        partial = 0.25 * self.latch_open.float() + 0.35 * self.latch_inside.float()
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """The plant: plate spring (constant up force), gate interlock drive (toward open
        while the plate is depressed, toward closed otherwise), the cubes' push buffers
        (world-frame intent pre-encoded against wrench frame drag), and the rubric
        latches."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        zt = torch.zeros(n, 1, 3, device=dev)

        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 2] = c.spring_force
        self.plate.set_external_force_and_torque(f, zt)

        depressed = self.plate_depressed()
        fg = torch.zeros(n, 1, 3, device=dev)
        fg[:, 0, 0] = torch.where(depressed,
                                  torch.full((n,), c.gate_force, device=dev),
                                  torch.full((n,), -c.gate_force, device=dev))
        self.gate.set_external_force_and_torque(fg, zt)

        # Cube pushes: applied wrenches rotate with rotation-since-reset on this stack —
        # pre-encode with q_ref * q_now^-1 so the WORLD-frame intent lands as written.
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

        for i, b in enumerate(self.cubes):
            q_now = b.data.root_quat_w
            q_fix = quat_mul(self._q_ref[:, i, :], quat_conjugate(q_now))
            fc = quat_apply(q_fix, self.push_f[:, i, :])
            b.set_external_force_and_torque(fc.unsqueeze(1), zt)

        # Rubric latches (bool; NaN-state comparisons are False, so garbage latches nothing).
        self.latch_open |= depressed & self.gate_open()
        slow = self.red.data.root_lin_vel_w.norm(dim=-1) < c.red_slow_latch
        self.latch_inside |= self.red_inside() & slow

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("spawn_xy", "spawn_yaw", "latch_open", "latch_inside",
                               "push_f", "_q_ref")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"gate": self.gate, "plate": self.plate, "red_cube": self.red,
                "blue_cube": self.blue, "decoy_cube": self.decoy}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On a wooden table stands a small concrete VAULT: four walls and a flat roof "
            f"(interior {2 * c.inner_x_half * 100:.0f} x "
            f"{(c.inner_y[1] - c.inner_y[0]) * 100:.0f} cm, {c.inner_h * 100:.0f} cm tall). "
            f"Its only opening is a doorway in the front wall, {2 * c.door_half * 100:.0f} cm "
            f"wide and full height, barred by a dark steel GATE that slides sideways on a "
            f"rail. To the right of the vault a concrete pedestal carries a YELLOW "
            f"spring-loaded PRESSURE PLATE. The gate is interlocked to the plate: it slides "
            f"open only while the plate is pressed all the way down, and springs shut again "
            f"the moment the plate comes back up. Three cubes wait on the table, dealt to "
            f"random spots each episode: a small RED cargo cube "
            f"({c.red_size * 100:.0f} cm), a BLUE weight cube ({c.blue_size * 100:.0f} cm, "
            f"solid and heavy — the only object heavy enough to hold the plate down), and a "
            f"GRAY decoy cube (same size and shape as the blue one but hollow and light — "
            f"it cannot depress the plate; only color tells them apart).\n"
            f"Goal: leave the RED cube locked inside the vault — red cube fully inside, "
            f"gate closed, pressure plate back up, everything at rest. The roof means the "
            f"red cube cannot be dropped in from above; it only fits through the doorway "
            f"while the gate is open. So: set the BLUE cube onto the yellow plate to hold "
            f"the gate open, move the RED cube through the doorway well inside the vault "
            f"(slide it along the table — the doorway is {2 * c.door_half * 100:.0f} cm "
            f"wide, plenty for the {c.red_size * 100:.0f} cm cube), then lift the BLUE "
            f"cube off the plate so the gate slides shut. This order is forced by the "
            f"mechanism itself. The gray decoy is a trap: it opens nothing, and a cube "
            f"left in the doorway blocks the gate and fails."
        )

    def instruction(self) -> str:
        return (
            "Put the heavy blue cube on the yellow pressure plate so the vault gate slides "
            "open, move the red cube through the doorway fully inside the vault, then take "
            "the blue cube off the plate so the gate closes. Finish with the red cube shut "
            "inside: gate closed, plate up, everything at rest. The gray cube is a decoy — "
            "too light to open the gate."
        )


# Guarded registration: the forge may import this module under two names.
if "weight_airlock" not in SCENES.list():
    SCENES.register("weight_airlock", WeightAirlockScene)
if "simgen.weight_airlock" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="weight_airlock", robot="null"))
