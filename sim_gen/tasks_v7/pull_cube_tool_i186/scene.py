"""ShelfDropDispatchScene — pull the falsework column out from under a raised hinged
shelf so gravity delivers the out-of-reach cube down the shelf into a catch pit, then
set the cube on the yellow pedestal.

Derived from the `maniskill/pull_cube_tool` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is "grasp a portable L-shaped tool, hook it behind the far
cube, drag the cube inward" — reach extension with a hand-held implement, the cube's
whole trajectory under direct quasi-static control, goal = a proximity disc around the
base. Here there is NO portable tool and the arm NEVER moves the cube while it is far
away: the red cube rests on an elevated steel shelf beyond arm reach. The shelf is
hinged at its FAR end and held level by a single removable support column (blue, the
falsework) standing under its midspan — within reach. The solver must slide the loaded
column sideways out of its floor channel (a genuine extraction under load: the shelf
presses down on the column while it slides), whereupon the shelf swings down on its
damped hinge into a ramp and the cube slides down UNDER GRAVITY into a walled catch
pit near the robot; the solver then picks the cube out of the pit and sets it centered
on top of the yellow pedestal (whose position varies per episode). Plan skeleton:
stored-energy RELEASE + passive gravity transport + a terminal precision place — not a
different-numbers drag.

Mechanics (plain rigid bodies + one authored USD revolute joint, the
carousel_ferry-proven pattern): kinematic post --revolute Y (limits [-15deg, +0.5deg])
--> shelf (dynamic box). A `post_step` plant applies viscous torque about the hinge
axis (so the release swings down instead of slamming) and consumes a `prop_force`
buffer (world force on the column CoM) that solve/smoke probes write — never call
set_external_force_and_torque on the bodies directly.

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py trajectory):
  - `released` (latch): the shelf has physically swung down (pitch <= -release_deg).
    The column physically prevents this until it is out from under the shelf.
  - `delivered` (latch): AFTER release, the cube is inside the catch-pit footprint
    below the containment height (z < pit_z_max — containment judged below the walls,
    not at the aperture).
  - success(): `released` AND the cube currently rests settled, centered on the
    pedestal TOP (per-axis xy within ped_xy_tol, z at rest height, |v|/|w| settled).
  - score() = 1.0 if success else 0.25*released + 0.30*delivered — exactly 1.0 iff
    success(); ~0 for the null policy; latched credit survives a later knock-off
    (falls back to 0.55, never 0).

Per-episode randomization (verified by readback in smoke.py): cube spawn position and
yaw on the shelf, the column's channel position, and the pedestal position.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


def _wrap(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return (x + math.pi) % (2 * math.pi) - math.pi


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShelfDropDispatchSceneCfg(BaseCfg):
    """Config for `ShelfDropDispatchScene`. Geometry is derived once in `__post_init__` so
    the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    release_deg: float = tunable(10.0)  # released latch: shelf pitch at/below -this (deg)
    pit_x: tuple = tunable((0.29, 0.57))  # delivered latch: cube x window (pit interior)
    pit_y_half: float = tunable(0.075)  # delivered latch: |cube y| below this
    pit_z_max: float = tunable(0.05)  # delivered latch: cube centre below this (walls are 0.09)
    ped_xy_tol: float = tunable(0.045)  # success: |cube - pedestal centre| per axis (top half 0.08)
    rest_z_tol: float = tunable(0.012)  # success: cube centre within this of pedestal rest z
    settle_v: float = tunable(0.04)  # success: max |lin vel| (m/s)
    settle_w: float = tunable(0.6)  # success: max |ang vel| (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    cube_x_range: tuple = tunable((0.84, 0.96))  # cube spawn x on the shelf (out of reach)
    cube_y_jitter: float = tunable(0.04)  # cube spawn |y| bound on the shelf
    prop_y_jitter: float = tunable(0.015)  # column spawn y in its channel
    ped_center: tuple = tunable((0.26, -0.34))  # pedestal nominal centre
    ped_jitter: float = tunable(0.05)  # +- xy jitter of the pedestal at reset

    # --- tunable: plant (difficulty dials) -------------------------------------------------------
    # Viscous hinge damping sized against the shelf inertia about the hinge
    # (I ~ m L^2 / 3 ~ 0.045 kg m^2, gravity torque ~ 1.2 N m): the released shelf
    # swings down in ~0.5-1 s instead of slamming the joint limit.
    visc: float = tunable(2.0)  # hinge viscous damping (N*m*s/rad), applied in post_step
    shelf_mass: float = tunable(0.4)
    prop_mass: float = tunable(0.6)
    cube_mass: float = tunable(0.05)
    # Slick shelf top / channel floor: the 15 deg release ramp must beat friction with
    # margin (tan 15 deg = 0.27 vs combined mu ~ 0.12), and the loaded column must
    # slide out without tipping (low floor mu keeps the drag moment under the
    # gravity-restoring moment).
    shelf_mu: tuple = tunable((0.10, 0.08))
    cube_mu: tuple = tunable((0.15, 0.12))
    plate_mu: tuple = tunable((0.08, 0.06))
    prop_mu: tuple = tunable((0.25, 0.20))
    ped_mu: tuple = tunable((0.7, 0.6))

    # --- info: structure --------------------------------------------------------------------------
    hinge_x: float = info(1.06)  # hinge axis x (world), axis along y
    hinge_z: float = info(0.22)  # hinge axis height
    shelf_len: float = info(0.55)  # shelf extends from the hinge toward the base
    shelf_w: float = info(0.14)
    shelf_t: float = info(0.02)
    drop_deg: float = info(15.0)  # hinge lower limit: the released ramp angle
    post_y: float = info(0.10)  # the two hinge posts stand at y = +-this
    post_size: tuple = info((0.05, 0.04, 0.26))
    prop_x: float = info(0.66)  # column channel x (under the shelf midspan)
    prop_size: tuple = info((0.06, 0.09, 0.20))  # the blue falsework column
    plate_top: float = info(0.010)  # slick channel floor top height
    plate_size: tuple = info((0.14, 0.60, 0.02))
    rail_size: tuple = info((0.02, 0.60, 0.03))
    rail_dx: float = info(0.046)  # rail centres at prop_x +- this (~6 mm slide clearance)
    pit_back_x: float = info(0.26)  # catch pit: back (-x) wall centre x
    pit_wall_h: float = info(0.09)
    pit_back_size: tuple = info((0.04, 0.20, 0.09))
    pit_side_size: tuple = info((0.32, 0.02, 0.09))
    pit_side_cx: float = info(0.42)  # side wall centre x (span 0.26..0.58)
    pit_side_y: float = info(0.09)  # side wall centres at y = +-this (inner faces +-0.08)
    ped_size: tuple = info((0.16, 0.16, 0.12))  # the yellow pedestal (top face at 0.12)
    cube_size: float = info(0.045)
    base_pos: tuple = info((0.0, 0.0))  # the documented Franka base xy (TASK.md)
    reach: float = info(0.75)  # documented comfortable arm envelope from base_pos
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    shelf_cx: float = field(default=None, init=False)  # shelf centre x when level
    shelf_top: float = field(default=None, init=False)  # shelf top face z when level
    shelf_bot: float = field(default=None, init=False)  # shelf underside z when level
    prop_z: float = field(default=None, init=False)  # column centre z standing in the channel
    prop_top: float = field(default=None, init=False)
    ped_top: float = field(default=None, init=False)
    ped_rest_z: float = field(default=None, init=False)  # cube centre resting on the pedestal
    prop_clear_y: float = field(default=None, init=False)  # |y| beyond which the column is clear

    def __post_init__(self) -> None:
        self.shelf_cx = self.hinge_x - self.shelf_len / 2
        self.shelf_top = self.hinge_z + self.shelf_t / 2
        self.shelf_bot = self.hinge_z - self.shelf_t / 2
        self.prop_z = self.plate_top + self.prop_size[2] / 2
        self.prop_top = self.plate_top + self.prop_size[2]
        self.ped_top = self.ped_size[2]
        self.ped_rest_z = self.ped_top + self.cube_size / 2
        self.prop_clear_y = self.shelf_w / 2 + self.prop_size[1] / 2 + 0.01


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shelf_drop_dispatch")
class ShelfDropDispatchScene(BaseScene):
    cfg: ShelfDropDispatchSceneCfg

    def __init__(self, cfg: ShelfDropDispatchSceneCfg | None = None) -> None:
        super().__init__(cfg or ShelfDropDispatchSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the two kinematic hinge posts (+ a visual axle rod), the dynamic
        shelf (revolute joint authored in bind()), the blue falsework column standing in
        its slick floor channel (plate + two guide rails), the red cargo cube on the shelf,
        the three-walled catch pit, and the yellow kinematic pedestal (re-posed per
        reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.48, 0.55))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.17))
        blue = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.25, 0.85))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.08))
        yellow = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.75, 0.05))
        gray = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38))
        pale = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.62, 0.66))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        # post_step drives shelf/column with external wrenches that do NOT wake a
        # sleeping body — sleep_threshold=0 keeps the plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "axle_rod": AssetBaseCfg(  # visual only: shows where the shelf hinges
                prim_path="{ENV_REGEX_NS}/AxleRod",
                spawn=sim_utils.CylinderCfg(radius=0.008, height=2 * c.post_y + 0.04,
                                            axis="Y", visual_material=dark),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(c.hinge_x, 0.0, c.hinge_z)),
            ),
            "shelf": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shelf",
                spawn=sim_utils.CuboidCfg(
                    size=(c.shelf_len, c.shelf_w, c.shelf_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.shelf_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.shelf_mu[0], dynamic_friction=c.shelf_mu[1]),
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.shelf_cx, 0.0, c.hinge_z)),
            ),
            "prop": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Prop",
                spawn=sim_utils.CuboidCfg(
                    size=c.prop_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.prop_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.prop_mu[0], dynamic_friction=c.prop_mu[1]),
                    visual_material=blue,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.prop_x, 0.0, c.prop_z)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu[0], dynamic_friction=c.cube_mu[1]),
                    visual_material=red,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.90, 0.0, c.shelf_top + c.cube_size / 2 + 0.003)),
            ),
            "channel_plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ChannelPlate",
                spawn=sim_utils.CuboidCfg(
                    size=c.plate_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.plate_mu[0], dynamic_friction=c.plate_mu[1]),
                    visual_material=pale,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.prop_x, 0.0, c.plate_top - c.plate_size[2] / 2)),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=c.ped_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ped_mu[0], dynamic_friction=c.ped_mu[1]),
                    visual_material=yellow,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_center[0], c.ped_center[1], c.ped_size[2] / 2)),
            ),
        }
        for s, ys in (("l", 1.0), ("r", -1.0)):
            out[f"post_{s}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post_" + s,
                spawn=sim_utils.CuboidCfg(
                    size=c.post_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=dark,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hinge_x, ys * c.post_y, c.post_size[2] / 2)),
            )
        for s, xs in (("a", -1.0), ("b", 1.0)):
            out[f"rail_{s}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rail_" + s,
                spawn=sim_utils.CuboidCfg(
                    size=c.rail_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=gray,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.prop_x + xs * c.rail_dx, 0.0,
                         c.plate_top + c.rail_size[2] / 2)),
            )
        pit = [
            ("back", c.pit_back_size, (c.pit_back_x, 0.0, c.pit_wall_h / 2)),
            ("sl", c.pit_side_size, (c.pit_side_cx, c.pit_side_y, c.pit_wall_h / 2)),
            ("sr", c.pit_side_size, (c.pit_side_cx, -c.pit_side_y, c.pit_wall_h / 2)),
        ]
        for nm, size, pos in pit:
            out[f"pit_{nm}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pit_" + nm,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5),
                    visual_material=gray,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
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

    # ----- lifecycle -------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.shelf: RigidObject = env.iscene["shelf"]
        self.prop: RigidObject = env.iscene["prop"]
        self.cube: RigidObject = env.iscene["cube"]
        self.ped: RigidObject = env.iscene["pedestal"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # External drive input (solve/smoke write; post_step consumes + owns the wrench
        # slots — never call set_external_force_and_torque on shelf/prop directly).
        self.prop_force = torch.zeros(n, 3, device=dev)  # world force at the column CoM
        # Rubric latches.
        self.released = torch.zeros(n, dtype=torch.bool, device=dev)
        self.delivered = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: a damped revolute Y hinge post->shelf with limits
        [-drop_deg, +0.5 deg]. The kinematic post anchors the axis world-fixed; the
        jointed pair never collides; the shelf spawns level, matching the authored local
        poses (reset re-poses it back to level about the unchanged axis)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/shelf_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Post_l"])
            j.CreateBody1Rel().SetTargets([f"{base}/Shelf"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            # Anchor: the hinge point (hinge_x, 0, hinge_z) expressed in each body frame.
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, -c.post_y,
                                           c.hinge_z - c.post_size[2] / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(c.shelf_len / 2, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.drop_deg)  # released ramp angle (hard stop)
            j.CreateUpperLimitAttr(0.5)

    # ----- reset -----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: shelf back to level about the unchanged hinge, the column back
        under the midspan (sampled channel y), the cube sampled on the shelf's far half
        (position + yaw), the pedestal jittered, latches and drive buffers zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- shelf: level (identity rotation about the authored axis) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.shelf_cx
        st[:, 2] = c.hinge_z
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.shelf.write_root_state_to_sim(st, env_ids)

        # --- column: standing in the channel under the shelf midspan ---
        py = (torch.rand(m, device=dev) * 2 - 1) * c.prop_y_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.prop_x
        st[:, 1] = py
        st[:, 2] = c.prop_z
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.prop.write_root_state_to_sim(st, env_ids)

        # --- cube: on the shelf's far half, free yaw ---
        cx = c.cube_x_range[0] + torch.rand(m, device=dev) * (c.cube_x_range[1] - c.cube_x_range[0])
        cy = (torch.rand(m, device=dev) * 2 - 1) * c.cube_y_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        half = yaw / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = cx
        st[:, 1] = cy
        st[:, 2] = c.shelf_top + c.cube_size / 2 + 0.003
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self.cube.write_root_state_to_sim(st, env_ids)

        # --- pedestal: jittered (kinematic) ---
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.ped_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.ped_center[0] + jit[:, 0]
        st[:, 1] = c.ped_center[1] + jit[:, 1]
        st[:, 2] = c.ped_size[2] / 2
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.ped.write_root_state_to_sim(st, env_ids)

        # --- rubric + drive state ---
        self.released[env_ids] = False
        self.delivered[env_ids] = False
        self.prop_force[env_ids] = 0.0

    # ----- readings --------------------------------------------------------------------------------
    def shelf_pitch(self) -> torch.Tensor:
        """(N,) shelf pitch about the hinge axis (rad; 0 = level, negative = swung down).
        The hinge is a pure Y revolute, so the quat is (cos a/2, 0, sin a/2, 0)."""
        q = self.shelf.data.root_quat_w
        return _wrap(2.0 * torch.atan2(q[:, 2], q[:, 0]))

    def prop_y(self) -> torch.Tensor:
        """(N,) column y (env-origin corrected) — extraction progress readback."""
        return self.prop.data.root_pos_w[:, 1] - self.env_origins[:, 1]

    def ped_xy(self) -> torch.Tensor:
        """(N, 2) live pedestal centre (env-origin corrected) — the rubric's goal readback."""
        return self.ped.data.root_pos_w[:, :2] - self.env_origins[:, :2]

    def cube_pos(self) -> torch.Tensor:
        """(N, 3) cube position (env-origin corrected)."""
        return self.cube.data.root_pos_w - self.env_origins

    def in_pit(self) -> torch.Tensor:
        """(N,) bool: cube inside the catch-pit footprint below the containment height
        (z-based containment — a cube perched on a wall or hovering at the mouth does
        not count)."""
        c = self.cfg
        p = self.cube_pos()
        return ((p[:, 0] > c.pit_x[0]) & (p[:, 0] < c.pit_x[1])
                & (p[:, 1].abs() < c.pit_y_half) & (p[:, 2] < c.pit_z_max))

    def on_pedestal(self) -> torch.Tensor:
        """(N,) bool: cube resting settled, centered on the pedestal top (current,
        physical)."""
        c = self.cfg
        p = self.cube_pos()
        d = (p[:, :2] - self.ped_xy()).abs()
        v = self.cube.data.root_lin_vel_w.norm(dim=-1)
        w = self.cube.data.root_ang_vel_w.norm(dim=-1)
        return ((d < c.ped_xy_tol).all(dim=-1)
                & ((p[:, 2] - c.ped_rest_z).abs() < c.rest_z_tol)
                & (v < c.settle_v) & (w < c.settle_w))

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the shelf was physically released (latch — the column bars this
        until it is out) and the cube now rests settled, centered on the pedestal top."""
        return self.released & self.on_pedestal()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); otherwise 0.25 * released latch
        + 0.30 * delivered latch. ~0 for the null policy; a knock-off after success
        falls back to 0.55 (latched credit does not evaporate)."""
        partial = 0.25 * self.released.float() + 0.30 * self.delivered.float()
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Hinge damping plant (viscous torque about the hinge axis) + the column force
        buffer, then the release/delivery latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        wy = self.shelf.data.root_ang_vel_w[:, 1]
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 1] = -c.visc * wy
        self.shelf.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), tq)
        self.prop.set_external_force_and_torque(
            self.prop_force.reshape(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

        # --- latches (NaN comparisons are False -> a garbage frame earns nothing) ---
        self.released = self.released | (self.shelf_pitch() < -math.radians(c.release_deg))
        self.delivered = self.delivered | (self.released & self.in_pit())

    # ----- state (full, restorable) -------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"shelf": self.shelf, "prop": self.prop, "cube": self.cube,
                "pedestal": self.ped}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("released", "delivered", "prop_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel shelf ({c.shelf_len * 100:.0f} x {c.shelf_w * 100:.0f} cm) stands "
            f"level about {c.shelf_top * 100:.0f} cm above the floor in front of the "
            f"robot. Its FAR end is hinged on a dark axle between two dark posts about "
            f"{c.hinge_x:.2f} m out; its free NEAR end points back toward the robot and "
            f"is held up only by a single BLUE support column "
            f"({c.prop_size[0] * 100:.0f} x {c.prop_size[1] * 100:.0f} x "
            f"{c.prop_size[2] * 100:.0f} cm) standing under the shelf's midspan, about "
            f"{c.prop_x:.2f} m out — the column sits in a slick floor channel between "
            f"two low gray guide rails that run sideways (either direction works). A "
            f"single RED cube ({c.cube_size * 100:.1f} cm) rests on the shelf near its "
            f"far end — beyond arm reach where it starts, and nothing you can hold "
            f"reaches it. On the floor between the robot and the column sits an "
            f"open-topped catch pit: a back wall and two side walls "
            f"({c.pit_wall_h * 100:.0f} cm high) whose open mouth faces the shelf's "
            f"free end. To one side stands a YELLOW pedestal "
            f"({c.ped_size[0] * 100:.0f} cm square, top face {c.ped_top * 100:.0f} cm "
            f"high); its position varies per episode — find the yellow block.\n"
            f"Goal: get the red cube to rest centered on TOP of the yellow pedestal. "
            f"Slide the blue column sideways along its channel, out from under the "
            f"shelf (it carries load — keep pulling until it is fully clear of the "
            f"shelf's width); the shelf then swings down on its hinge into a ramp and "
            f"the cube slides down by itself into the catch pit. Then pick the cube "
            f"out of the pit and set it down centered on the pedestal top. The release "
            f"must happen first — the cube cannot be reached where it starts. The cube "
            f"must end resting on the pedestal top, centered within about "
            f"{c.ped_xy_tol * 100:.0f} cm and settled; a cube left in the pit, on the "
            f"floor, beside the pedestal, or perched off-center on its edge does not "
            f"succeed."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Pull the blue support column sideways out of its floor channel so the "
            "raised shelf swings down and the red cube slides off into the walled "
            "catch pit, then pick up the cube and set it centered on top of the "
            "yellow pedestal. The cube must end resting settled on the pedestal top."
        )


# ----- runnable env: scene physics only (NullRobot solve/smoke) -> "simgen.shelf_drop_dispatch" --
register_env("simgen", lambda: EnvCfg(scene="shelf_drop_dispatch", robot="null"))
