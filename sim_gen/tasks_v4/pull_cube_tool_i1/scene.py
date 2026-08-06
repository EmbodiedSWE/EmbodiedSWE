"""CarouselFerryScene — rotate a peg-driven carousel to ferry an out-of-reach cube into
reach, then place it into the walled green tray.

Derived from the `maniskill/pull_cube_tool` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is "grasp a portable L-shaped tool, hook it behind the far
cube, drag the cube inward" — reach extension with a hand-held implement, one
quasi-static drag, goal = a proximity disc around the base. Here there is NO portable
tool and NO contact the arm can make moves the cube directly: the red cube rides the
far rim of a large rotating platter (a lazy-susan carousel on a fixed pedestal, free
revolute Z axle) far beyond arm reach. Four vertical capstan pegs stand on the platter;
the solver must work the pegs that pass through the near quadrant — push one through an
arc, let the next peg rotate into reach, re-engage — until the platter has carried the
cube around to the near side, then PICK the cube off the platter and PLACE it inside a
small walled tray with a green floor mat. Plan skeleton: anchored-mechanism actuation
with cyclic re-engagement + a terminal pick-and-place — not a different-numbers drag.

Mechanics (the oven_dials-proven pattern — plain rigid bodies + authored USD joints;
platter "feel" is an external viscous torque applied in `post_step`, which also consumes
external drive/force buffers (`platter_drive` / `cube_force`) that solve/smoke probes may
write): pedestal (kinematic) --revolute Z--> platter (dynamic cylinder), four pegs
fixed-jointed onto the platter top. The cube is a free rigid body riding the platter by
friction.

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py trajectory):
  - RIDE tracking (`post_step`, per substep): the cube counts as "riding" when it rests
    on the platter top, inside the rim, its platter-frame xy is steady (< `ride_slip`
    per substep) and the platter turns < `ride_dth_deg` per substep. While riding, the
    SIGNED platter rotation accumulates into `net_sweep` (rocking cancels; a kinematic
    teleport across the scene never accumulates).
  - `ferried` (latch): the cube is riding INSIDE the near sector (azimuth within
    `sector_deg` of the base-facing direction) with |net_sweep| >= `sweep_req_deg`.
    Physically the spawn sector forces >= ~100 deg of true carried rotation, so a
    teleport straight to the near side / into the tray never latches.
  - `ferry_prog` (latched): min(|net_sweep| / sweep_full_deg, 1), forced to 1.0 once
    `ferried` — partial real rotation earns partial credit, doing nothing earns ~0.
  - `placed` (latch): after `ferried`, the cube enters the tray footprint below wall
    top.
  - success(): `ferried` AND the cube currently rests settled inside the tray (xy within
    `in_tray_xy` of the mat centre, resting z, |v| / |w| settled).
  - score() = 0.3 * ferry_prog + 0.2 * placed + 0.5 * success -> exactly 1.0 iff
    success(); latched credit survives a later knock-out (falls back to 0.5, never 0).

Per-episode randomization (verified by readback in smoke.py): cube spawn azimuth on the
far rim, spawn radius, cube yaw, the platter's initial yaw (pegs pattern), and the tray
position — a memorized fixed trajectory fails across episodes.

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
class CarouselFerrySceneCfg(BaseCfg):
    """Config for `CarouselFerryScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    sector_deg: float = tunable(55.0)  # near sector: cube azimuth within this of base-facing dir
    sweep_req_deg: float = tunable(80.0)  # min |net carried rotation| for the ferried latch
    sweep_full_deg: float = tunable(100.0)  # ferry_prog saturates here (spawn needs >= ~100)
    ride_dz: float = tunable(0.02)  # riding: |cube z - ride height| below this (m)
    ride_slip: float = tunable(0.008)  # riding: max platter-frame xy motion per substep (m)
    ride_dth_deg: float = tunable(3.0)  # riding: max platter rotation per substep (deg)
    in_tray_xy: float = tunable(0.050)  # success: |cube - mat centre| per axis (cube fully in)
    placed_xy: float = tunable(0.070)  # placed latch: footprint bound (inner half is 0.075)
    rest_z_tol: float = tunable(0.012)  # success: cube centre within this of its tray rest z
    settle_v: float = tunable(0.04)  # success: max |lin vel| (m/s)
    settle_w: float = tunable(0.6)  # success: max |ang vel| (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    spawn_az_deg: float = tunable(25.0)  # cube azimuth ~ U(+-this) about the FAR direction (+x)
    spawn_r_min: float = tunable(0.23)  # cube spawn radius on the platter
    spawn_r_max: float = tunable(0.27)
    tray_jitter: float = tunable(0.04)  # +- xy jitter of the whole tray at reset

    # --- tunable: plant (difficulty dials) -------------------------------------------------------
    # Viscous axle friction sized against the platter inertia (I ~ 0.5*3*0.34^2 + pegs
    # ~ 0.20 kg m^2): tau = I/b ~ 0.5 s spin-down, so peg pushes produce controlled,
    # promptly-settling rotation instead of a flywheel.
    visc: float = tunable(0.40)  # axle viscous friction (N*m*s/rad), applied in post_step
    platter_mass: float = tunable(3.0)
    peg_mass: float = tunable(0.08)
    cube_mass: float = tunable(0.06)
    # Ride friction: mu_eff ~ 0.7 -> the cube stays platter-fixed up to w ~ 5 rad/s at
    # r=0.27 (w^2 r < mu g); the plant + solver never exceed ~2.5 rad/s.
    platter_mu: tuple = tunable((0.85, 0.75))  # (static, dynamic)
    cube_mu: tuple = tunable((0.70, 0.60))

    # --- info: structure --------------------------------------------------------------------------
    platter_center: tuple = info((0.68, 0.0))  # axle xy; base-facing (near) direction = -x
    platter_r: float = info(0.34)
    platter_th: float = info(0.03)
    platter_gap: float = info(0.004)  # pedestal top to platter bottom (jointed pair, no contact)
    pedestal_r: float = info(0.07)
    pedestal_h: float = info(0.08)
    n_pegs: int = info(4)
    peg_mount_r: float = info(0.26)  # peg circle radius on the platter
    peg_r: float = info(0.016)
    peg_h: float = info(0.11)
    cube_size: float = info(0.045)
    tray_center: tuple = info((0.22, -0.30))  # nominal; jittered per episode
    tray_inner: float = info(0.075)  # inner half-extent of the tray floor (square)
    wall_t: float = info(0.012)
    wall_h: float = info(0.05)  # wall height above ground
    mat_th: float = info(0.006)  # green floor mat thickness
    base_pos: tuple = info((-0.10, 0.0))  # the documented Franka base xy (TASK.md)
    reach: float = info(0.75)  # documented comfortable arm envelope from base_pos
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    platter_z: float = field(default=None, init=False)  # platter centre z
    platter_top: float = field(default=None, init=False)
    peg_z: float = field(default=None, init=False)  # peg centre z
    ride_z: float = field(default=None, init=False)  # cube centre z when riding
    mat_top: float = field(default=None, init=False)
    rest_z: float = field(default=None, init=False)  # cube centre z resting on the mat
    wall_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.platter_z = self.pedestal_h + self.platter_gap + self.platter_th / 2
        self.platter_top = self.platter_z + self.platter_th / 2
        self.peg_z = self.platter_top + self.peg_h / 2 + 0.002
        self.ride_z = self.platter_top + self.cube_size / 2
        self.mat_top = self.mat_th
        self.rest_z = self.mat_top + self.cube_size / 2
        self.wall_top = self.wall_h

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ------------------
    def peg_offset(self, k: int) -> tuple:
        """Peg k's mount offset in the PLATTER frame (platter yaw 0)."""
        a = math.radians(90.0 * k)
        return (self.peg_mount_r * math.cos(a), self.peg_mount_r * math.sin(a),
                self.peg_z - self.platter_z)

    def wall_specs(self) -> list[tuple[tuple, tuple]]:
        """[(size, offset-from-tray-centre), ...] for the four tray walls (on the ground)."""
        i, t, h = self.tray_inner, self.wall_t, self.wall_h
        long = 2 * i + 2 * t
        return [
            ((long, t, h), (0.0, i + t / 2, h / 2)),
            ((long, t, h), (0.0, -(i + t / 2), h / 2)),
            ((t, 2 * i, h), (i + t / 2, 0.0, h / 2)),
            ((t, 2 * i, h), (-(i + t / 2), 0.0, h / 2)),
        ]


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("carousel_ferry")
class CarouselFerryScene(BaseScene):
    cfg: CarouselFerrySceneCfg

    def __init__(self, cfg: CarouselFerrySceneCfg | None = None) -> None:
        super().__init__(cfg or CarouselFerrySceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic pedestal, the free-spinning platter + four capstan
        pegs (joints authored in bind()), the red cargo cube, and the walled tray with its
        green floor mat (kinematic; re-posed per reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.48, 0.55))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.17))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.60, 0.28))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.08))
        green = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.15))
        gray = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        # post_step drives the platter with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live (drawer_stash lesson).
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)
        px, py = c.platter_center
        tx, ty = c.tray_center

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
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pedestal_r, height=c.pedestal_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=dark,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.pedestal_h / 2)),
            ),
            "platter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platter",
                spawn=sim_utils.CylinderCfg(
                    radius=c.platter_r, height=c.platter_th, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.platter_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.platter_mu[0], dynamic_friction=c.platter_mu[1]),
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.platter_z)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu[0], dynamic_friction=c.cube_mu[1]),
                    visual_material=red,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.peg_mount_r, py, c.ride_z + 0.003)),
            ),
            "tray_mat": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TrayMat",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.tray_inner, 2 * c.tray_inner, c.mat_th),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6),
                    visual_material=green,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(tx, ty, c.mat_th / 2)),
            ),
        }
        for k in range(c.n_pegs):
            ox, oy, _oz = c.peg_offset(k)
            out[f"peg_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Peg_" + str(k),
                spawn=sim_utils.CylinderCfg(
                    radius=c.peg_r, height=c.peg_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.peg_mass),
                    collision_props=coll,
                    visual_material=brass,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + ox, py + oy, c.peg_z)),
            )
        for w, (size, off) in enumerate(c.wall_specs()):
            out[f"wall_{w}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wall_" + str(w),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=gray,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx + off[0], ty + off[1], off[2])),
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
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.platter: RigidObject = env.iscene["platter"]
        self.cube: RigidObject = env.iscene["cube"]
        self.mat: RigidObject = env.iscene["tray_mat"]
        self.pegs: list[RigidObject] = [env.iscene[f"peg_{k}"] for k in range(c.n_pegs)]
        self.walls: list[RigidObject] = [env.iscene[f"wall_{w}"] for w in range(4)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # External drive inputs (solve/smoke write; post_step consumes + owns the wrench
        # slots — never call set_external_force_and_torque on platter/cube directly).
        self.platter_drive = torch.zeros(n, device=dev)  # torque about the axle (N*m)
        self.cube_force = torch.zeros(n, 3, device=dev)  # world force at the cube CoM (probes)
        # Rubric state.
        self.net_sweep = torch.zeros(n, device=dev)  # SIGNED carried platter rotation (rad)
        self.ferry_prog = torch.zeros(n, device=dev)  # latched progress in [0, 1]
        self.ferried = torch.zeros(n, dtype=torch.bool, device=dev)
        self.placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._th_prev = torch.zeros(n, device=dev)  # platter yaw last substep
        self._pf_prev = torch.zeros(n, 2, device=dev)  # cube xy in the platter frame
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: a free Z revolute axle pedestal->platter and a fixed joint platter->peg
        for each capstan peg. Joint pairs never collide; bodies spawn at yaw 0, matching the
        authored local poses (reset re-poses the whole follower assembly about the unchanged
        axle — the fridge_clearway/oven_dials-proven safe teleport)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/platter_axle")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Platter"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.platter_z - c.pedestal_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # no limit attrs -> free continuous axle
            for k in range(c.n_pegs):
                ox, oy, oz = c.peg_offset(k)
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/peg_fix_{k}")
                j.CreateBody0Rel().SetTargets([f"{base}/Platter"])
                j.CreateBody1Rel().SetTargets([f"{base}/Peg_{k}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(ox, oy, oz))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ----- reset -----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the cube's far-rim spawn (azimuth about +x, radius, yaw),
        set the platter yaw so the peg pattern straddles the cube (nearest peg 45 deg away),
        re-pose platter + pegs as one rigid assembly about the unchanged axle, jitter the
        tray, zero every latch and drive buffer."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        px, py = c.platter_center

        az = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_az_deg)
        r0 = c.spawn_r_min + torch.rand(m, device=dev) * (c.spawn_r_max - c.spawn_r_min)
        yaw0 = az + math.radians(45.0)  # pegs at az +- 45 deg (+ k*90): clear of the cube

        # --- platter + pegs: one rigid assembly at yaw0 about the fixed axle ---
        half = yaw0 / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = c.platter_z
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self.platter.write_root_state_to_sim(st, env_ids)
        for k in range(c.n_pegs):
            a = yaw0 + math.radians(90.0 * k)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = px + c.peg_mount_r * torch.cos(a)
            st[:, 1] = py + c.peg_mount_r * torch.sin(a)
            st[:, 2] = c.peg_z
            st[:, 0:3] += origin
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.pegs[k].write_root_state_to_sim(st, env_ids)

        # --- cube: far rim, free yaw ---
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        ch = cyaw / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px + r0 * torch.cos(az)
        st[:, 1] = py + r0 * torch.sin(az)
        st[:, 2] = c.ride_z + 0.003
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(ch)
        st[:, 6] = torch.sin(ch)
        self.cube.write_root_state_to_sim(st, env_ids)

        # --- tray (mat + walls): jittered as one unit ---
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        tc = torch.tensor(c.tray_center, device=dev).expand(m, 2) + jit
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = tc
        st[:, 2] = c.mat_th / 2
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.mat.write_root_state_to_sim(st, env_ids)
        for w, (_size, off) in enumerate(c.wall_specs()):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = tc[:, 0] + off[0]
            st[:, 1] = tc[:, 1] + off[1]
            st[:, 2] = off[2]
            st[:, 0:3] += origin
            st[:, 3] = 1.0
            self.walls[w].write_root_state_to_sim(st, env_ids)

        # --- rubric + drive state ---
        self.net_sweep[env_ids] = 0.0
        self.ferry_prog[env_ids] = 0.0
        self.ferried[env_ids] = False
        self.placed[env_ids] = False
        self._th_prev[env_ids] = yaw0
        self._pf_prev[env_ids] = 0.0
        self._prev_valid[env_ids] = False
        self.platter_drive[env_ids] = 0.0
        self.cube_force[env_ids] = 0.0

    # ----- readings --------------------------------------------------------------------------------
    def platter_yaw(self) -> torch.Tensor:
        """(N,) platter yaw (rad, wrapped). The axle is world-Z; the platter never tilts."""
        q = self.platter.data.root_quat_w
        return _wrap(2.0 * torch.atan2(q[:, 3], q[:, 0]))

    def platter_rate(self) -> torch.Tensor:
        """(N,) signed axle rate (rad/s)."""
        return self.platter.data.root_ang_vel_w[:, 2]

    def cube_azimuth(self) -> torch.Tensor:
        """(N,) cube azimuth about the axle (rad; 0 = +x = FAR, pi = base-facing = NEAR)."""
        rel = self.cube_rel()
        return torch.atan2(rel[:, 1], rel[:, 0])

    def cube_rel(self) -> torch.Tensor:
        """(N, 2) cube xy relative to the axle (env-origin corrected)."""
        c = self.cfg
        rel = self.cube.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        return rel - torch.tensor(c.platter_center, device=rel.device)

    def tray_xy(self) -> torch.Tensor:
        """(N, 2) live tray-mat centre (env-origin corrected) — the rubric's goal readback."""
        return self.mat.data.root_pos_w[:, :2] - self.env_origins[:, :2]

    def riding(self) -> torch.Tensor:
        """(N,) bool: cube resting on the platter top inside the rim (z + radius only — the
        per-substep slip clause lives in post_step where consecutive frames exist)."""
        c = self.cfg
        z = self.cube.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        r = self.cube_rel().norm(dim=-1)
        return ((z - c.ride_z).abs() < c.ride_dz) & (r < c.platter_r - 0.01)

    def in_tray(self) -> torch.Tensor:
        """(N,) bool: cube settled resting inside the tray (current, physical)."""
        c = self.cfg
        d = (self.cube.data.root_pos_w[:, :2] - self.env_origins[:, :2] - self.tray_xy()).abs()
        z = self.cube.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        v = self.cube.data.root_lin_vel_w.norm(dim=-1)
        w = self.cube.data.root_ang_vel_w.norm(dim=-1)
        return ((d < c.in_tray_xy).all(dim=-1)
                & ((z - c.rest_z).abs() < c.rest_z_tol)
                & (v < c.settle_v) & (w < c.settle_w))

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cube was physically FERRIED around the carousel (latch) and now
        rests settled inside the tray."""
        return self.ferried & self.in_tray()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.3 * latched ferry progress + 0.2 * latched placement +
        0.5 * current success. Exactly 1.0 iff success(); ~0 for the null policy; a
        knock-out after success falls back to 0.5 (latched credit does not evaporate)."""
        return (0.3 * self.ferry_prog + 0.2 * self.placed.float()
                + 0.5 * self.success().float())

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Axle plant (viscous friction + external drive buffer), cube probe-force buffer,
        then the ride/ferry/placement latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev)

        th = self.platter_yaw()
        om = self.platter_rate()
        tq = self.platter_drive - c.visc * om
        self.platter.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tq.reshape(n, 1, 1) * ez.view(1, 1, 3))
        self.cube.set_external_force_and_torque(
            self.cube_force.reshape(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

        # --- ride tracking ---
        dth = _wrap(th - self._th_prev)
        rel = self.cube_rel()
        cth, sth = torch.cos(th), torch.sin(th)
        pf = torch.stack([cth * rel[:, 0] + sth * rel[:, 1],
                          -sth * rel[:, 0] + cth * rel[:, 1]], dim=-1)  # platter-frame xy
        slip = (pf - self._pf_prev).norm(dim=-1)
        riding = (self.riding() & self._prev_valid
                  & (slip < c.ride_slip) & (dth.abs() < math.radians(c.ride_dth_deg)))
        # A diverged substep must not latch (NaN propagates through maximum and the latches
        # are checkpointed via get_state/set_state) — a garbage frame earns NO progress.
        dth = torch.nan_to_num(dth, nan=0.0, posinf=0.0, neginf=0.0)
        self.net_sweep = self.net_sweep + torch.where(riding, dth, torch.zeros_like(dth))
        prog = (self.net_sweep.abs() / math.radians(c.sweep_full_deg)).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)

        az = torch.atan2(rel[:, 1], rel[:, 0])
        in_sector = _wrap(az - math.pi).abs() <= math.radians(c.sector_deg)
        self.ferried = self.ferried | (
            riding & in_sector
            & (self.net_sweep.abs() >= math.radians(c.sweep_req_deg)))
        self.ferry_prog = torch.maximum(
            self.ferry_prog, torch.where(self.ferried, torch.ones_like(prog), prog))

        # --- placement latch: after the ferry, the cube enters the tray footprint below
        # the wall top (the drop is credited before it fully settles) ---
        d = (self.cube.data.root_pos_w[:, :2] - self.env_origins[:, :2]
             - self.tray_xy()).abs()
        z = self.cube.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self.placed = self.placed | (
            self.ferried & (d < c.placed_xy).all(dim=-1) & (z < c.wall_top + 0.01))

        self._th_prev = th
        self._pf_prev = pf
        self._prev_valid = torch.ones_like(self._prev_valid)

    # ----- state (full, restorable) -------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        out = {"platter": self.platter, "cube": self.cube, "tray_mat": self.mat}
        out.update({f"peg_{k}": p for k, p in enumerate(self.pegs)})
        out.update({f"wall_{w}": b for w, b in enumerate(self.walls)})
        return out

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("net_sweep", "ferry_prog", "ferried", "placed",
                               "_th_prev", "_pf_prev", "_prev_valid",
                               "platter_drive", "cube_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        px, _ = c.platter_center
        return (
            f"A large steel carousel (a flat platter, {2 * c.platter_r * 100:.0f} cm across, "
            f"top face {c.platter_top * 100:.0f} cm above the floor) spins freely about a "
            f"fixed vertical axle on a low pedestal about {px:.2f} m in front of the robot. "
            f"Four brass pegs ({2 * c.peg_r * 1000:.0f} mm across, {c.peg_h * 100:.0f} cm "
            f"tall, one every 90 degrees at radius {c.peg_mount_r * 100:.0f} cm) stand on "
            f"the platter and turn with it — they are handles: push or pull a peg sideways "
            f"and the whole carousel rotates. A single red cube "
            f"({c.cube_size * 100:.1f} cm) rides on the platter near its FAR rim, on the "
            f"side of the axle away from the robot — beyond arm reach where it starts. On "
            f"the floor near the robot, offset to its right, sits a small open-top tray: "
            f"four gray walls ({c.wall_h * 100:.0f} cm high) around a GREEN floor mat "
            f"({2 * c.tray_inner * 100:.0f} cm square; its position varies per episode — "
            f"find the green mat).\n"
            f"Goal: get the red cube to rest inside the green-floored tray. The cube "
            f"cannot be reached where it starts and nothing you can throw or poke reaches "
            f"it: rotate the carousel by working the pegs that pass through the near "
            f"quadrant (push one through an arc, let the next peg come around, re-engage) "
            f"until the platter has carried the cube around to the near side, then pick "
            f"the cube off the platter and set it down inside the tray walls. Either "
            f"rotation direction works. The cube must arrive riding the platter and must "
            f"end resting fully inside the tray, settled; a cube dropped beside the tray "
            f"or left anywhere else scores nothing further."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Rotate the carousel by pushing its brass pegs until the red cube riding the "
            "far rim comes around within reach, then pick up the cube and place it inside "
            "the green-floored tray. The cube starts beyond reach and must ride the "
            "carousel in; it must end resting inside the tray walls."
        )


# ----- runnable env: scene physics only (NullRobot solve/smoke) -> "simgen.carousel_ferry" -----
register_env("simgen", lambda: EnvCfg(scene="carousel_ferry", robot="null"))
