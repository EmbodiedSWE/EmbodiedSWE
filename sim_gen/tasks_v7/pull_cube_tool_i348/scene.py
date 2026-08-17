"""BoomCorralScene — herd an out-of-reach cube into an out-of-reach corral pen with the
far blade of an anchored, mirror-inverting boom lever.

Derived from the `maniskill/pull_cube_tool` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is "grasp a portable L-shaped tool, hook it behind the far
cube, DRAG the cube inward into reach" — reach extension with a hand-held implement,
goal = a proximity disc around the base. Here the cube is NEVER brought into reach and
is never picked: a rigid boom pivots on a fixed post between the robot and the far
field. Its NEAR end carries a graspable handle (always within reach); its FAR end
carries a shallow pusher pocket (a blade with two short end lips) that sweeps a circular
arc through the far field, where both the red cargo cube and a green three-walled corral
pen sit — both permanently beyond arm reach. Sweeping the handle one way swings the
blade the OPPOSITE way (mirror inversion about the pivot) with amplified displacement
(far arm 0.56 m vs near arm 0.42 m). The solver must read which side of the cube the pen
is on (the side flips per episode), sweep the blade in behind the cube from the
anti-pen side, and herd the cube along the arc through the pen's open mouth until it
rests inside. Plan skeleton: remote lever manipulation of an untouchable object into an
untouchable goal — no tool is carried, nothing is retrieved, nothing is picked.

Mechanics (the carousel-proven pattern — plain rigid bodies + authored USD joints; boom
"feel" is a viscous yaw torque applied in `post_step`, which also consumes the external
drive buffer `boom_drive` that solve/smoke probes write): pedestal (kinematic)
--revolute Z, limits +-70 deg--> spar (dynamic), with handle / blade face / two pocket
lips fixed-jointed onto the spar (authored with clearances so non-paired pieces never
interpenetrate). The cube is a free rigid body on the ground; the pen walls are
kinematic, re-posed per episode.

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py trajectory):
  - HERD tracking (`post_step`, per substep): the cube's azimuth about the pivot is
    accumulated SIGNED toward the pen (`arc_prog` source) only while the cube is on the
    ground inside the sweep annulus and the per-substep azimuth change is below a jump
    cap — a teleport across the far field never accumulates.
  - `swept` (latch): accumulated toward-pen arc >= (required arc - 6 deg). The pen is
    authored 30-40 deg from the cube spawn, so only real continuous transport latches.
  - `entered` (latch): swept AND the cube crosses the pen mouth into the footprint.
  - success(): `entered` AND the cube currently rests settled inside the pen interior.
  - score() = 0.3 * arc_prog + 0.2 * entered + 0.5 * success -> exactly 1.0 iff
    success(); latched credit survives a later knock-out (falls to 0.5, never 0).

Per-episode randomization (verified by readback in smoke.py): cube spawn azimuth /
radius / yaw, the SIDE the pen is on (flips the required sweep direction), the pen's
arc offset, and the boom's initial yaw.

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
class BoomCorralSceneCfg(BaseCfg):
    """Config for `BoomCorralScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    swept_margin_deg: float = tunable(6.0)  # swept latch: accumulated arc >= required - this
    jump_cap_deg: float = tunable(2.0)  # max per-substep cube azimuth change that accumulates
    band_r_min: float = tunable(0.42)  # sweep annulus (accumulation gate), radius about pivot
    band_r_max: float = tunable(0.72)
    band_z_max: float = tunable(0.06)  # cube counts as "on the ground" below this (m)
    in_pen_u: tuple = tunable((0.020, 0.170))  # success: cube centre depth past the mouth (m)
    in_pen_w: float = tunable(0.090)  # success: |lateral offset| from the pen axis (m)
    entered_u: float = tunable(0.010)  # entered latch: depth past the mouth (m)
    rest_z_tol: float = tunable(0.012)  # success: cube centre within this of rest height
    settle_v: float = tunable(0.04)  # success: max |lin vel| (m/s)
    settle_w: float = tunable(0.6)  # success: max |ang vel| (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    spawn_az_deg: float = tunable(14.0)  # cube azimuth ~ U(+-this) about +x from the pivot
    spawn_r_min: float = tunable(0.535)  # cube spawn radius about the pivot (inside the pocket
    spawn_r_max: float = tunable(0.585)  # interior [0.4945, 0.6255])
    pen_arc_min_deg: float = tunable(30.0)  # pen mouth this far along the arc from the cube
    pen_arc_max_deg: float = tunable(40.0)
    boom_back_min_deg: float = tunable(13.0)  # blade starts this far behind the cube
    boom_back_max_deg: float = tunable(18.0)  # (on the anti-pen side)

    # --- tunable: plant (difficulty dials) -------------------------------------------------------
    # Viscous yaw friction sized against the boom inertia (I ~ 0.074 kg m^2): tau = I/b
    # ~ 0.6 s spin-down — handle pushes give controlled, promptly-settling swings.
    visc: float = tunable(0.12)  # boom viscous yaw friction (N*m*s/rad), applied in post_step
    spar_mass: float = tunable(0.40)
    handle_mass: float = tunable(0.05)
    face_mass: float = tunable(0.08)
    lip_mass: float = tunable(0.02)
    cube_mass: float = tunable(0.05)
    ground_mu: tuple = tunable((0.50, 0.45))  # (static, dynamic): cube slides when pushed,
    cube_mu: tuple = tunable((0.50, 0.45))  # parks where released
    boom_mu: tuple = tunable((0.30, 0.25))

    # --- info: structure --------------------------------------------------------------------------
    pivot: tuple = info((0.62, 0.0))  # boom pivot post xy; +x = away from the robot
    ped_r: float = info(0.035)
    ped_h: float = info(0.07)  # post top flush with the spar bottom
    # Spar local x extent [-0.44, 0.66] about the pivot; it rides at z 0.07-0.10, ABOVE
    # the 6 cm pen walls (10 mm clearance), so only the blade pocket shares the walls'
    # height band — the spar can overfly the pen (the first build jammed the spar's side
    # face on the inner wall's near corner at radius 0.445).
    spar_size: tuple = info((1.10, 0.035, 0.03))
    spar_off: tuple = info((0.11, 0.0, 0.085))  # spar centre in the boom (pivot) frame
    handle_r: float = info(0.014)
    handle_h: float = info(0.16)
    handle_off: tuple = info((-0.42, 0.0, 0.18))  # handle centre (bottom sits on the spar top)
    face_size: tuple = info((0.12, 0.02, 0.075))  # blade face: radial x tangential x tall
    face_off: tuple = info((0.56, 0.0, 0.0455))  # bottom edge 8 mm above the floor
    lip_size: tuple = info((0.015, 0.09, 0.075))  # pocket end lips (protrude +-35 mm past face)
    lip_off_in: tuple = info((0.487, 0.0, 0.0455))  # 5 mm radial gap to the face (never touch)
    lip_off_out: tuple = info((0.633, 0.0, 0.0455))
    boom_limit_deg: float = info(70.0)  # revolute travel +-; also the blade's hard stop
    cube_size: float = info(0.045)
    pen_r: float = info(0.56)  # pen mouth centre radius about the pivot (= blade circle)
    pen_depth: float = info(0.18)  # interior depth along the arrival direction
    pen_halfw: float = info(0.115)  # interior half-width (radial)
    wall_t: float = info(0.015)
    wall_h: float = info(0.06)
    base_pos: tuple = info((-0.10, 0.0))  # the documented Franka base xy (TASK.md)
    reach: float = info(0.75)  # documented comfortable arm envelope from base_pos
    contact_offset: float = info(0.002)
    drive_cap: float = info(1.5)  # honest |boom_drive| bound (N*m): ~3.6 N at the handle

    # Derived (filled in __post_init__).
    handle_arm: float = field(default=None, init=False)  # handle radius about the pivot
    blade_arm: float = field(default=None, init=False)  # blade face centre radius
    rest_z: float = field(default=None, init=False)  # cube centre resting on the ground

    def __post_init__(self) -> None:
        self.handle_arm = abs(self.handle_off[0])
        self.blade_arm = self.face_off[0]
        self.rest_z = self.cube_size / 2

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ------------------
    def boom_pieces(self) -> list[tuple[str, tuple]]:
        """[(asset name, centre offset in the boom/pivot frame), ...] for the jointed boom
        assembly (the spar is the jointed root; the rest are fixed-jointed to it)."""
        return [
            ("spar", self.spar_off),
            ("handle", self.handle_off),
            ("face", self.face_off),
            ("lip_in", self.lip_off_in),
            ("lip_out", self.lip_off_out),
        ]

    def pen_pieces(self) -> list[tuple[str, tuple, tuple]]:
        """[(asset name, size, centre offset in the PEN frame), ...]. Pen frame: origin at
        the mouth centre on the blade circle, +x = arrival (sweep) direction, +y = radial
        outward from the pivot. Mouth fully open; three walls."""
        d, hw, t, h = self.pen_depth, self.pen_halfw, self.wall_t, self.wall_h
        return [
            ("pen_back", (t, 2 * hw + 2 * t, h), (d + t / 2, 0.0, h / 2)),
            ("pen_side_out", (d + t, t, h), ((d + t) / 2, hw + t / 2, h / 2)),
            ("pen_side_in", (d + t, t, h), ((d + t) / 2, -(hw + t / 2), h / 2)),
        ]


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("boom_corral")
class BoomCorralScene(BaseScene):
    cfg: BoomCorralSceneCfg

    def __init__(self, cfg: BoomCorralSceneCfg | None = None) -> None:
        super().__init__(cfg or BoomCorralSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic pivot post, the free-yaw boom assembly (joints
        authored in bind()), the red cargo cube, and the green corral pen (kinematic;
        re-posed per reset). Spawn poses are the boom-yaw-0 / pen-at-+35-deg nominal;
        reset() re-places everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.48, 0.55))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.17))
        black = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.06, 0.06))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.08))
        green = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.15))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        boom_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.boom_mu[0], dynamic_friction=c.boom_mu[1])
        # post_step drives the boom with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)
        dyn = dict(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5,
            **live,
        )
        px, py = c.pivot
        masses = {"spar": c.spar_mass, "handle": c.handle_mass, "face": c.face_mass,
                  "lip_in": c.lip_mass, "lip_out": c.lip_mass}

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu[0], dynamic_friction=c.ground_mu[1])),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CylinderCfg(
                    radius=c.ped_r, height=c.ped_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=dark,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.ped_h / 2)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**dyn),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu[0], dynamic_friction=c.cube_mu[1]),
                    visual_material=red,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + 0.56, py, c.rest_z + 0.003)),
            ),
        }
        # --- boom pieces (spar + handle + blade pocket), authored at boom yaw 0 ---
        for name, off in c.boom_pieces():
            if name == "spar":
                spawn = sim_utils.CuboidCfg(
                    size=c.spar_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**dyn),
                    mass_props=sim_utils.MassPropertiesCfg(mass=masses[name]),
                    collision_props=coll, physics_material=boom_mat, visual_material=steel)
            elif name == "handle":
                spawn = sim_utils.CylinderCfg(
                    radius=c.handle_r, height=c.handle_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**dyn),
                    mass_props=sim_utils.MassPropertiesCfg(mass=masses[name]),
                    collision_props=coll, physics_material=boom_mat, visual_material=black)
            else:
                size = c.face_size if name == "face" else c.lip_size
                spawn = sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**dyn),
                    mass_props=sim_utils.MassPropertiesCfg(mass=masses[name]),
                    collision_props=coll, physics_material=boom_mat, visual_material=steel)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Boom_" + name,
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + off[0], py + off[1], off[2])),
            )
        # --- pen walls (kinematic; nominal pose, re-posed per reset) ---
        nom_u = (-math.sin(math.radians(35.0)), math.cos(math.radians(35.0)))
        nom_w = (math.cos(math.radians(35.0)), math.sin(math.radians(35.0)))
        nom_org = (px + c.pen_r * nom_w[0], py + c.pen_r * nom_w[1])
        nom_yaw = math.atan2(nom_u[1], nom_u[0])
        for name, size, off in c.pen_pieces():
            wx = nom_org[0] + off[0] * nom_u[0] + off[1] * nom_w[0]
            wy = nom_org[1] + off[0] * nom_u[1] + off[1] * nom_w[1]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pen_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=green,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx, wy, off[2]),
                    rot=(math.cos(nom_yaw / 2), 0.0, 0.0, math.sin(nom_yaw / 2)),
                ),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
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

    # ----- lifecycle -------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.cube: RigidObject = env.iscene["cube"]
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.boom: dict[str, RigidObject] = {nm: env.iscene[nm] for nm, _o in c.boom_pieces()}
        self.pen: dict[str, RigidObject] = {nm: env.iscene[nm] for nm, _s, _o in c.pen_pieces()}
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # External drive input (solve/smoke write; post_step consumes + owns the wrench
        # slot — never call set_external_force_and_torque on the spar directly).
        self.boom_drive = torch.zeros(n, device=dev)  # yaw torque about the pivot (N*m)
        # Per-episode goal geometry (sampled at reset; smoke reads it back).
        self.side = torch.ones(n, device=dev)  # +1: pen counter-clockwise of the cube
        self.pen_yaw = torch.zeros(n, device=dev)  # pen +x (arrival dir) world yaw
        self.pen_org = torch.zeros(n, 2, device=dev)  # pen mouth centre xy (env-local)
        self.arc_req = torch.ones(n, device=dev)  # required herd arc (rad), cube spawn -> mouth
        # Rubric state.
        self.arc_done = torch.zeros(n, device=dev)  # accumulated toward-pen cube arc (rad)
        self.arc_prog = torch.zeros(n, device=dev)  # latched progress in [0, 1]
        self.swept = torch.zeros(n, dtype=torch.bool, device=dev)
        self.entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._phi_prev = torch.zeros(n, device=dev)  # cube azimuth last substep
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: a limited Z revolute pedestal->spar (the boom pivot, +-boom_limit_deg)
        and a fixed joint spar->piece for the handle and the blade pocket pieces. Joint
        pairs never collide; non-paired pieces are authored with real clearances (5 mm+)
        so the assembly never fights itself. Bodies spawn at yaw 0, matching the authored
        local poses (reset re-poses the whole assembly about the unchanged pivot)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        sx, sy, sz = c.spar_off
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/boom_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Boom_spar"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, sz - c.ped_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(-sx, -sy, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.boom_limit_deg)
            j.CreateUpperLimitAttr(c.boom_limit_deg)
            for name, off in c.boom_pieces():
                if name == "spar":
                    continue
                fj = UsdPhysics.FixedJoint.Define(stage, f"{base}/boom_fix_{name}")
                fj.CreateBody0Rel().SetTargets([f"{base}/Boom_spar"])
                fj.CreateBody1Rel().SetTargets([f"{base}/Boom_{name}"])
                fj.CreateCollisionEnabledAttr(False)
                fj.CreateLocalPos0Attr(Gf.Vec3f(off[0] - sx, off[1] - sy, off[2] - sz))
                fj.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                fj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ----- reset -----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the cube's far-field spawn (azimuth about +x from the
        pivot, radius, yaw), the pen SIDE and arc offset, and the boom's start yaw (blade
        behind the cube on the anti-pen side); re-pose the boom as one rigid assembly
        about the unchanged pivot, place the pen walls, zero every latch and buffer."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        px, py = c.pivot

        # First post-seed draws are degenerate across seeds — burn before the discrete
        # side choice.
        _ = torch.rand(m, 4, device=dev)
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        az = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_az_deg)
        r0 = c.spawn_r_min + torch.rand(m, device=dev) * (c.spawn_r_max - c.spawn_r_min)
        arc = math.radians(c.pen_arc_min_deg) + torch.rand(m, device=dev) * math.radians(
            c.pen_arc_max_deg - c.pen_arc_min_deg)
        back = math.radians(c.boom_back_min_deg) + torch.rand(m, device=dev) * math.radians(
            c.boom_back_max_deg - c.boom_back_min_deg)
        pen_az = az + side * arc
        boom0 = az - side * back

        # --- boom assembly at yaw boom0 about the fixed pivot ---
        half = boom0 / 2
        cs, sn = torch.cos(boom0), torch.sin(boom0)
        for name, off in c.boom_pieces():
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = px + off[0] * cs - off[1] * sn
            st[:, 1] = py + off[0] * sn + off[1] * cs
            st[:, 2] = off[2]
            st[:, 0:3] += origin
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.boom[name].write_root_state_to_sim(st, env_ids)

        # --- cube: far field, free yaw ---
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        ch = cyaw / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px + r0 * torch.cos(az)
        st[:, 1] = py + r0 * torch.sin(az)
        st[:, 2] = c.rest_z + 0.003
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(ch)
        st[:, 6] = torch.sin(ch)
        self.cube.write_root_state_to_sim(st, env_ids)

        # --- pen: mouth on the blade circle at pen_az, +x = arrival direction ---
        u = torch.stack([-side * torch.sin(pen_az), side * torch.cos(pen_az)], dim=-1)
        w = torch.stack([torch.cos(pen_az), torch.sin(pen_az)], dim=-1)
        org = torch.stack([px + c.pen_r * torch.cos(pen_az),
                           py + c.pen_r * torch.sin(pen_az)], dim=-1)
        pyaw = torch.atan2(u[:, 1], u[:, 0])
        ph = pyaw / 2
        for name, _size, off in c.pen_pieces():
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = org + off[0] * u + off[1] * w
            st[:, 2] = off[2]
            st[:, 0:3] += origin
            st[:, 3] = torch.cos(ph)
            st[:, 6] = torch.sin(ph)
            self.pen[name].write_root_state_to_sim(st, env_ids)

        # --- goal geometry + rubric + drive state ---
        self.side[env_ids] = side
        self.pen_yaw[env_ids] = pyaw
        self.pen_org[env_ids] = org
        self.arc_req[env_ids] = arc
        self.arc_done[env_ids] = 0.0
        self.arc_prog[env_ids] = 0.0
        self.swept[env_ids] = False
        self.entered[env_ids] = False
        self._phi_prev[env_ids] = az
        self._prev_valid[env_ids] = False
        self.boom_drive[env_ids] = 0.0

    # ----- readings --------------------------------------------------------------------------------
    def boom_yaw(self) -> torch.Tensor:
        """(N,) boom yaw (rad, wrapped; 0 = blade pointing +x/away). The pivot is world-Z."""
        q = self.boom["spar"].data.root_quat_w
        return _wrap(2.0 * torch.atan2(q[:, 3], q[:, 0]))

    def boom_rate(self) -> torch.Tensor:
        """(N,) signed boom yaw rate (rad/s)."""
        return self.boom["spar"].data.root_ang_vel_w[:, 2]

    def cube_rel(self) -> torch.Tensor:
        """(N, 2) cube xy relative to the pivot (env-origin corrected)."""
        rel = self.cube.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        return rel - torch.tensor(self.cfg.pivot, device=rel.device)

    def cube_azimuth(self) -> torch.Tensor:
        """(N,) cube azimuth about the pivot (rad; 0 = +x = straight away from the robot)."""
        rel = self.cube_rel()
        return torch.atan2(rel[:, 1], rel[:, 0])

    def cube_pen_frame(self) -> torch.Tensor:
        """(N, 2) cube position in the PEN frame: x = depth past the mouth along the
        arrival direction, y = lateral (radial) offset from the pen axis."""
        p = self.cube.data.root_pos_w[:, :2] - self.env_origins[:, :2] - self.pen_org
        cu, su = torch.cos(self.pen_yaw), torch.sin(self.pen_yaw)
        return torch.stack([cu * p[:, 0] + su * p[:, 1],
                            -su * p[:, 0] + cu * p[:, 1]], dim=-1)

    def in_pen(self) -> torch.Tensor:
        """(N,) bool: cube settled resting inside the pen interior (current, physical)."""
        c = self.cfg
        uw = self.cube_pen_frame()
        z = self.cube.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        v = self.cube.data.root_lin_vel_w.norm(dim=-1)
        w = self.cube.data.root_ang_vel_w.norm(dim=-1)
        return ((uw[:, 0] > c.in_pen_u[0]) & (uw[:, 0] < c.in_pen_u[1])
                & (uw[:, 1].abs() < c.in_pen_w)
                & ((z - c.rest_z).abs() < c.rest_z_tol)
                & (v < c.settle_v) & (w < c.settle_w))

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cube was physically HERDED along the arc into the pen (latch
        chain) and now rests settled inside it."""
        return self.entered & self.in_pen()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.3 * latched herd progress + 0.2 * latched pen entry +
        0.5 * current success. Exactly 1.0 iff success(); ~0 for the null policy; a
        knock-out after success falls back to 0.5 (latched credit does not evaporate)."""
        return (0.3 * self.arc_prog + 0.2 * self.entered.float()
                + 0.5 * self.success().float())

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Boom plant (viscous yaw friction + external drive buffer), then the herd /
        entry latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev)

        om = self.boom_rate()
        tq = self.boom_drive.clamp(-c.drive_cap, c.drive_cap) - c.visc * om
        self.boom["spar"].set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tq.reshape(n, 1, 1) * ez.view(1, 1, 3))

        # --- herd tracking: signed toward-pen cube arc, gated + jump-capped ---
        phi = self.cube_azimuth()
        dphi = _wrap(phi - self._phi_prev)
        rel = self.cube_rel()
        r = rel.norm(dim=-1)
        z = self.cube.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        ok = (self._prev_valid
              & (dphi.abs() < math.radians(c.jump_cap_deg))
              & (r > c.band_r_min) & (r < c.band_r_max) & (z < c.band_z_max))
        dphi = torch.nan_to_num(dphi, nan=0.0, posinf=0.0, neginf=0.0)
        self.arc_done = self.arc_done + torch.where(ok, self.side * dphi,
                                                    torch.zeros_like(dphi))
        prog = (self.arc_done / self.arc_req).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self.swept = self.swept | (
            self.arc_done >= self.arc_req - math.radians(c.swept_margin_deg))
        self.arc_prog = torch.maximum(
            self.arc_prog, torch.where(self.swept, torch.ones_like(prog), prog))

        # --- entry latch: after the sweep, the cube crosses the mouth into the footprint
        # (credited on transit, before it settles) ---
        uw = self.cube_pen_frame()
        self.entered = self.entered | (
            self.swept & (uw[:, 0] > c.entered_u)
            & (uw[:, 1].abs() < c.pen_halfw) & (z < c.band_z_max))

        self._phi_prev = phi
        self._prev_valid = torch.ones_like(self._prev_valid)

    # ----- state (full, restorable) -------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        out = {"cube": self.cube, "pedestal": self.pedestal}
        out.update(self.boom)
        out.update(self.pen)
        return out

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("side", "pen_yaw", "pen_org", "arc_req", "arc_done",
                               "arc_prog", "swept", "entered", "_phi_prev",
                               "_prev_valid", "boom_drive")},
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
            f"A long steel boom pivots freely about a fixed vertical post "
            f"{c.pivot[0]:.2f} m in front of the robot (it can only yaw, between hard "
            f"stops at +-{c.boom_limit_deg:.0f} degrees). The boom's NEAR end, always "
            f"within reach, carries a black vertical handle (a "
            f"{2 * c.handle_r * 100:.0f} cm post, top {(c.handle_off[2] + c.handle_h / 2) * 100:.0f} cm "
            f"above the floor) at {c.handle_arm * 100:.0f} cm from the pivot. The FAR "
            f"end carries a shallow steel pusher pocket (a {c.face_size[0] * 100:.0f} cm "
            f"blade with two short end lips, open on both sweep sides, riding "
            f"{(c.face_off[2] - c.face_size[2] / 2) * 1000:.0f} mm above the floor) at "
            f"{c.blade_arm * 100:.0f} cm from the pivot, on the far side — because the "
            f"boom pivots between them, moving the handle one way swings the blade the "
            f"OPPOSITE way, and the blade travels farther than the handle.\n"
            f"In the far field, permanently beyond arm reach, a single red cube "
            f"({c.cube_size * 100:.1f} cm) sits on the floor on the blade's sweep "
            f"circle, and a GREEN corral pen (three green walls "
            f"{c.wall_h * 100:.0f} cm high around a {2 * c.pen_halfw * 100:.0f} cm wide, "
            f"{c.pen_depth * 100:.0f} cm deep patch of floor, mouth wide open toward "
            f"the cube along the sweep circle) stands 30-40 degrees of arc away from "
            f"the cube — on the robot's LEFT in some episodes and on its RIGHT in "
            f"others: look before sweeping. The blade starts on the opposite side of "
            f"the cube from the pen.\n"
            f"Goal: get the red cube to rest inside the green pen. The cube and the pen "
            f"cannot be reached and nothing you can throw or poke reaches them: sweep "
            f"the handle so the blade advances along the arc, takes the cube into its "
            f"pocket, and herds it along the sweep circle through the pen's open mouth; "
            f"then ease off and let it settle inside. Remember the mirror inversion: to "
            f"send the blade toward the pen, the handle must move the other way. The "
            f"cube must travel the arc pushed by the blade — it must end resting "
            f"settled between the pen walls; a cube left anywhere else, or shoved past "
            f"or beside the pen, scores nothing further."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Sweep the black handle of the pivoting boom so its far blade pushes the "
            "red cube along the arc into the green three-walled pen. The handle and "
            "blade move in opposite directions about the pivot; the cube and pen are "
            "out of reach — only the blade can move the cube. Leave the cube resting "
            "inside the pen."
        )


# ----- runnable env: scene physics only (NullRobot solve/smoke) -> "simgen.boom_corral" --------
register_env("simgen", lambda: EnvCfg(scene="boom_corral", robot="null"))
