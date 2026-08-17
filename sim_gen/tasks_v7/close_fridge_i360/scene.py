"""SpringLatchFridgeScene — retract the slide bolt, press the spring-loaded fridge
door shut, and throw the bolt home into the keeper to SECURE it.

Derived from the RLBench `close_fridge` seed but STRATEGICALLY DIFFERENT (see TASK.md):
the seed's plan is one uncontrolled gross-motion push on a hinged panel — swing the
fridge door shut, done; the closed state is passively stable and any push wins. Here
the door hinge carries a RETURN SPRING with an OPEN equilibrium (40-72 deg, sampled
per episode): an unlatched door does not stay closed — release it anywhere short of
mechanically secured and the spring swings it back open. The only stable closed state
is the door pressed flush with the slide bolt thrown into the keeper bore on the
cabinet wall. And the bolt starts EXTENDED: closing the door with the bolt out slams
the protruding tip into the proud keeper housing >= ~4.5 deg short of flush. The plan
skeleton therefore changes from "push panel" (one action, memoryless) to a forced
three-step serial program on TWO coupled DOFs — (1) retract the bolt, (2) press the
door flush AND HOLD it against the live spring, (3) slide the bolt into the bore while
holding, then release — with steps (1) and (3) ordered by collision geometry and step
(2)'s hold made mandatory by the spring. No rubric clause encodes the order; the
mechanism does.

Mechanics (plain rigid bodies + authored USD joints, the i89 hinge pattern): the
cabinet is five kinematic slabs; the door is ONE dynamic panel on a Z-axis revolute
joint (limits [0, 105] deg, 0 = closed = the joint's own lower stop). The bolt is a
second dynamic body riding the door's outer face on an authored PRISMATIC joint (axis
= door-local +y, toward the free edge; travel [0, 50] mm, 0 = retracted). The keeper
is a compound housing proud of the free-side wall with a blind rectangular bore: 4 mm
of play around the 20 mm bolt section, so the bolt can only enter with the door pressed
essentially flush (<= ~1 deg — swing misalignment is ~0.49 m of tip motion per rad).
`post_step` applies the hinge return spring k*(theta_eq - theta) - c*omega plus the
`door_drive` probe torque, and the `bolt_force` buffer (a BODY-frame force on the bolt:
+x presses the door closed through the prismatic joint, +/-y slides the bolt). Nothing
else touches either wrench slot.

Rubric (graded 0..1, latching transient achievement — anchored in the demonstrated
solve.py trajectory: retract, press flush, engage):
  - `retract_latch` (0.25): the bolt has ever been fully retracted (ext <= 10 mm);
  - `close_latch`   (0.35): the door has ever been within 2 deg of flush (unreachable
    with the bolt out — the jam arrests the swing >= ~4.5 deg);
  - current success (0.40): bolt tip engaged >= 18 mm into the keeper bore AND door
    within `closed_tol_deg` AND everything settled -> score == 1.0 iff success().
    Both latches are forced full the moment success() holds (the legit path implies
    them). Null policy: door rests at its spring equilibrium, bolt out — score ~0.
    A push-only strategy (the seed's) caps at 0.60 and the door ends OPEN.

Per-episode randomization (readback-verified in smoke): spring equilibrium angle
theta_eq (= the door's rest angle), spring stiffness k, and the bolt's initial
extension.

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


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SpringLatchFridgeSceneCfg(BaseCfg):
    """Config for `SpringLatchFridgeScene`. Geometry is derived once in `__post_init__` so
    the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    closed_tol_deg: float = tunable(3.0)  # door within this of the 0 deg stop = closed
    close_latch_deg: float = tunable(2.0)  # flush latch band (extended-bolt jam >= ~4.5 deg)
    retract_thresh: float = tunable(0.010)  # bolt ext <= this = retracted (start >= 0.032)
    engage_min: float = tunable(0.018)  # bolt tip this far past the bore mouth = engaged
    settle_door: float = tunable(0.08)  # max hinge rate |omega_z| (rad/s) when judging
    settle_bolt: float = tunable(0.05)  # max bolt |lin vel| (m/s) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    theta_eq_min_deg: float = tunable(42.0)  # spring equilibrium (= door rest angle)
    theta_eq_max_deg: float = tunable(72.0)
    k_min: float = tunable(1.0)  # spring stiffness (N*m/rad)
    k_max: float = tunable(1.6)
    ext0_min: float = tunable(0.032)  # initial bolt extension (m) — always jam-extended
    ext0_max: float = tunable(0.048)

    # --- tunable: plant (difficulty dials) ---------------------------------------------------
    door_damping: float = tunable(0.5)  # hinge damper (N*m*s/rad); zeta ~0.4 with k~1.3
    door_mass: float = tunable(2.0)
    bolt_mass: float = tunable(0.15)

    # --- info: cabinet (env-local coordinates; interior x [0.35,0.75] y [-0.20,0.16]) --------
    cab_floor_pos: tuple = info((0.565, -0.02, 0.10))
    cab_floor_size: tuple = info((0.43, 0.42, 0.20))
    cab_back_pos: tuple = info((0.765, -0.02, 0.31))
    cab_back_size: tuple = info((0.03, 0.42, 0.62))
    cab_wall_hinge_pos: tuple = info((0.565, -0.215, 0.41))  # hinge-side (y-) wall = hinge body0
    cab_wall_free_pos: tuple = info((0.565, 0.175, 0.41))  # free-side wall = keeper carrier
    cab_wall_size: tuple = info((0.43, 0.03, 0.42))
    cab_top_pos: tuple = info((0.565, -0.02, 0.635))
    cab_top_size: tuple = info((0.43, 0.42, 0.03))
    door_center_closed: tuple = info((0.330, -0.02, 0.42))  # panel centre at theta = 0
    door_size: tuple = info((0.024, 0.46, 0.46))
    hinge_y: float = info(-0.235)  # hinge axis at (door_center_closed.x, hinge_y), world-z
    door_limit_deg: float = info(105.0)

    # --- info: bolt (door-local frame; +x = inner normal when closed, -x = outward) ----------
    bolt_size: tuple = info((0.020, 0.100, 0.020))
    bolt_local_x: float = info(-0.025)  # bolt centre x in the door frame (1 mm off the face)
    bolt_local_z: float = info(0.030)  # bolt centre z in the door frame (world z 0.45)
    bolt_retract_y: float = info(0.170)  # bolt centre y at ext = 0 (tip then at +0.220)
    bolt_stroke: float = info(0.050)  # prismatic travel; ext in [0, stroke]
    knob_center: tuple = info((-0.017, -0.030, 0.0))  # bolt-local push knob (outward face)
    knob_size: tuple = info((0.014, 0.030, 0.030))

    # --- info: keeper housing (env-local; mounted proud of the free wall's front face) -------
    # Housing outer box x [0.262, 0.350], y [0.215, 0.275], z [0.410, 0.490]; blind bore
    # x [0.291, 0.319], z [0.436, 0.464], mouth at y = 0.215 (faces the hinge), end 0.252.
    # Door free edge sweeps y <= 0.21, so the housing never obstructs the door itself —
    # only a bolt tip protruding >= ~5 mm past the edge collides with it.
    keeper_mouth_y: float = info(0.215)
    keeper_bore_cx: float = info(0.305)  # = door x + bolt_local_x at theta = 0 (dead centre)
    keeper_bore_cz: float = info(0.450)
    keeper_bore_half: float = info(0.014)  # bore half-width/height: 4 mm play per side
    keeper_bore_end_y: float = info(0.252)  # blind end
    keeper_front_x: float = info(0.262)  # proud face: extended-bolt jam at ~5 deg
    keeper_back_x: float = info(0.350)
    keeper_z_lo: float = info(0.410)
    keeper_z_hi: float = info(0.490)
    keeper_y_hi: float = info(0.275)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    hinge_xy: tuple = field(default=None, init=False)
    bolt_tip_half: float = field(default=None, init=False)  # tip = centre + this along +y
    door_free_edge_y: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.hinge_xy = (self.door_center_closed[0], self.hinge_y)
        self.bolt_tip_half = self.bolt_size[1] / 2  # 0.050
        self.door_free_edge_y = self.door_center_closed[1] + self.door_size[1] / 2  # 0.21

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ----------------
    def door_center_at(self, theta_rad: float) -> tuple:
        """World centre of the door panel at hinge angle theta (0 = closed)."""
        hx, hy = self.hinge_xy
        arm = self.door_center_closed[1] - self.hinge_y  # 0.215, along +y at closed
        return (hx - arm * math.sin(theta_rad), hy + arm * math.cos(theta_rad),
                self.door_center_closed[2])

    def bolt_center_local(self, ext: float) -> tuple:
        """Bolt centre in the DOOR frame at extension `ext`."""
        return (self.bolt_local_x, self.bolt_retract_y + ext, self.bolt_local_z)


def _quat_z(rad: float) -> tuple:
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("spring_latch_fridge")
class SpringLatchFridgeScene(BaseScene):
    cfg: SpringLatchFridgeSceneCfg

    def __init__(self, cfg: SpringLatchFridgeSceneCfg | None = None) -> None:
        super().__init__(cfg or SpringLatchFridgeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, five kinematic cabinet slabs (keeper housing authored onto the
        free wall in bind), the dynamic door panel and the dynamic slide bolt (knob child
        authored in bind)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cream = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.88, 0.84))
        door_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.78, 0.80))
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.55, 0.30))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives door and bolt with external wrenches that do NOT wake a
        # sleeping body — sleep_threshold=0 keeps the plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

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
        }
        slabs = {
            "cab_floor": (c.cab_floor_pos, c.cab_floor_size),
            "cab_back": (c.cab_back_pos, c.cab_back_size),
            "cab_wall_hinge": (c.cab_wall_hinge_pos, c.cab_wall_size),
            "cab_wall_free": (c.cab_wall_free_pos, c.cab_wall_size),
            "cab_top": (c.cab_top_pos, c.cab_top_size),
        }
        for name, (pos, size) in slabs.items():
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=size, rigid_props=kin, collision_props=coll, visual_material=cream),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        out["door"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Door",
            spawn=sim_utils.CuboidCfg(
                size=c.door_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.door_mass),
                collision_props=coll,
                visual_material=door_col,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=c.door_center_closed),
        )
        bl = c.bolt_center_local(0.0)
        dc = c.door_center_closed
        out["bolt"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bolt",
            spawn=sim_utils.CuboidCfg(
                size=c.bolt_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    linear_damping=0.05,
                    angular_damping=0.05,
                    **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.bolt_mass),
                collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.3, dynamic_friction=0.25, restitution=0.0),
                visual_material=steel,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(dc[0] + bl[0], dc[1] + bl[1], dc[2] + bl[2])),
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
        self.door: RigidObject = env.iscene["door"]
        self.bolt: RigidObject = env.iscene["bolt"]
        self.env_origins = env.iscene.env_origins
        self._author_keeper()
        self._author_knob()
        self._author_joints()
        # Episode state.
        self.theta_eq = torch.full((n,), 1.0, device=dev)  # spring equilibrium (rad)
        self.k_spring = torch.full((n,), 1.3, device=dev)  # spring stiffness (N*m/rad)
        self.ext0 = torch.zeros(n, device=dev)  # authored initial bolt extension
        self.retract_latch = torch.zeros(n, device=dev)  # bolt ever fully retracted
        self.close_latch = torch.zeros(n, device=dev)  # door ever within close_latch_deg
        # External inputs (solve.py and smoke probes write; post_step consumes + owns both
        # wrench slots — never call set_external_force_and_torque directly).
        self.door_drive = torch.zeros(n, device=dev)  # torque about the hinge (N*m, + opens)
        self.bolt_force = torch.zeros(n, 3, device=dev)  # BODY-frame force on the bolt (N)

    def _author_keeper(self) -> None:
        """Keeper housing: five collision boxes on the free-side wall (env_0 only — env_1..
        compose from env_0 by reference; authored idempotently) forming a blind rectangular
        bore that opens toward the hinge (-y). The housing's proud front face (x = 0.262)
        is what an extended bolt tip slams into ~5 deg short of flush."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/CabWallFree/keeper_front").IsValid():
            return
        wx, wy, wz = c.cab_wall_free_pos

        def box(name: str, lo: tuple, hi: tuple, color: tuple) -> None:
            cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/CabWallFree/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d((lo[0] + hi[0]) / 2 - wx,
                                             (lo[1] + hi[1]) / 2 - wy,
                                             (lo[2] + hi[2]) / 2 - wz))
            xf.AddScaleOp().Set(Gf.Vec3f(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            px.CreateContactOffsetAttr(c.contact_offset)
            px.CreateRestOffsetAttr(0.0)

        dark = (0.25, 0.27, 0.30)
        bx0, bx1 = c.keeper_bore_cx - c.keeper_bore_half, c.keeper_bore_cx + c.keeper_bore_half
        bz0, bz1 = c.keeper_bore_cz - c.keeper_bore_half, c.keeper_bore_cz + c.keeper_bore_half
        box("keeper_front", (c.keeper_front_x, c.keeper_mouth_y, c.keeper_z_lo),
            (bx0, c.keeper_y_hi, c.keeper_z_hi), dark)
        box("keeper_back", (bx1, c.keeper_mouth_y, c.keeper_z_lo),
            (c.keeper_back_x, c.keeper_y_hi, c.keeper_z_hi), dark)
        box("keeper_top", (bx0, c.keeper_mouth_y, bz1),
            (bx1, c.keeper_y_hi, c.keeper_z_hi), dark)
        box("keeper_bot", (bx0, c.keeper_mouth_y, c.keeper_z_lo),
            (bx1, c.keeper_y_hi, bz0), dark)
        box("keeper_end", (bx0, c.keeper_bore_end_y, bz0),
            (bx1, c.keeper_y_hi, bz1), dark)

    def _author_knob(self) -> None:
        """Push knob on the bolt's outward face (env_0 only, idempotent) — the fingertip
        contact point the embodiment argument uses; also the visual handle."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Bolt/knob").IsValid():
            return
        cube = UsdGeom.Cube.Define(stage, "/World/envs/env_0/Bolt/knob")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*c.knob_center))
        xf.AddScaleOp().Set(Gf.Vec3f(*c.knob_size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.18, 0.20)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(c.contact_offset)
        px.CreateRestOffsetAttr(0.0)

    def _author_joints(self) -> None:
        """Per env: (a) Z-axis revolute hinge wall -> door, limits [0, door_limit_deg]
        (0 = closed; the lower limit IS the door stop); (b) Y-axis prismatic door -> bolt,
        limits [0, bolt_stroke] (0 = retracted). Both joint pairs collision-filtered."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        hx, hy = c.hinge_xy
        wx, wy, wz = c.cab_wall_hinge_pos
        dx, dy, dz = c.door_center_closed
        bl = c.bolt_center_local(0.0)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/door_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/CabWallHinge"])
            j.CreateBody1Rel().SetTargets([f"{base}/Door"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(hx - wx, hy - wy, dz - wz))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(hx - dx, hy - dy, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(c.door_limit_deg)
            p = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/bolt_slide")
            p.CreateBody0Rel().SetTargets([f"{base}/Door"])
            p.CreateBody1Rel().SetTargets([f"{base}/Bolt"])
            p.CreateCollisionEnabledAttr(False)
            p.CreateAxisAttr("Y")
            p.CreateLocalPos0Attr(Gf.Vec3f(*bl))
            p.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            p.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            p.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            p.CreateLowerLimitAttr(0.0)
            p.CreateUpperLimitAttr(c.bolt_stroke)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the spring (equilibrium angle + stiffness) and the bolt's
        initial extension; pose the door AT its spring equilibrium (zero net torque — the
        start is settled) with the bolt written CONSISTENTLY in the same door frame (the
        whole linkage is teleported together); clear latches and drive buffers."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(1, device=dev)  # burn the degenerate first post-seed draw
        u = torch.rand(m, 3, device=dev)
        theta = torch.deg2rad(
            c.theta_eq_min_deg + (c.theta_eq_max_deg - c.theta_eq_min_deg) * u[:, 0])
        self.theta_eq[env_ids] = theta
        self.k_spring[env_ids] = c.k_min + (c.k_max - c.k_min) * u[:, 1]
        ext = c.ext0_min + (c.ext0_max - c.ext0_min) * u[:, 2]
        self.ext0[env_ids] = ext
        self.retract_latch[env_ids] = 0.0
        self.close_latch[env_ids] = 0.0
        self.door_drive[env_ids] = 0.0
        self.bolt_force[env_ids] = 0.0

        # --- door at the spring equilibrium ---
        st = torch.zeros(m, 13, device=dev)
        hx, hy = c.hinge_xy
        arm = c.door_center_closed[1] - c.hinge_y
        st[:, 0] = hx - arm * torch.sin(theta)
        st[:, 1] = hy + arm * torch.cos(theta)
        st[:, 2] = c.door_center_closed[2]
        st[:, 3] = torch.cos(theta / 2)
        st[:, 6] = torch.sin(theta / 2)
        st[:, 0:3] += origin
        self.door.write_root_state_to_sim(st, env_ids)

        # --- bolt riding the door at the sampled extension (same frame, same quat) ---
        door_pos = st[:, 0:3]
        cth, sth = torch.cos(theta), torch.sin(theta)
        lx = torch.full((m,), c.bolt_local_x, device=dev)
        ly = c.bolt_retract_y + ext
        bs = torch.zeros(m, 13, device=dev)
        bs[:, 0] = door_pos[:, 0] + lx * cth - ly * sth
        bs[:, 1] = door_pos[:, 1] + lx * sth + ly * cth
        bs[:, 2] = door_pos[:, 2] + c.bolt_local_z
        bs[:, 3] = torch.cos(theta / 2)
        bs[:, 6] = torch.sin(theta / 2)
        self.bolt.write_root_state_to_sim(bs, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def door_angle(self) -> torch.Tensor:
        """(N,) hinge angle in rad (0 = closed; the door only ever rotates about world z)."""
        q = self.door.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def door_rate(self) -> torch.Tensor:
        """(N,) signed hinge rate (rad/s) — the door's world-z angular velocity."""
        return self.door.data.root_ang_vel_w[:, 2]

    def bolt_ext(self) -> torch.Tensor:
        """(N,) bolt extension in m (0 = retracted): door-frame y of the bolt centre minus
        the retracted mount y."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.bolt.data.root_pos_w - self.door.data.root_pos_w
        return quat_apply_inverse(self.door.data.root_quat_w, rel)[:, 1] - self.cfg.bolt_retract_y

    def bolt_tip(self) -> torch.Tensor:
        """(N, 3) env-local world position of the bolt TIP (+y end of the bar)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        off = torch.zeros(n, 3, device=self.env.device)
        off[:, 1] = self.cfg.bolt_tip_half
        return self.bolt.data.root_pos_w + quat_apply(self.bolt.data.root_quat_w, off) \
            - self.env_origins

    def engaged(self) -> torch.Tensor:
        """(N,) bool, geometric: the bolt tip is inside the keeper bore, at least
        `engage_min` past the mouth. Judged in the static keeper frame (env-local world)."""
        c = self.cfg
        tip = self.bolt_tip()
        return (tip[:, 1] >= c.keeper_mouth_y + c.engage_min) \
            & ((tip[:, 0] - c.keeper_bore_cx).abs() <= c.keeper_bore_half) \
            & ((tip[:, 2] - c.keeper_bore_cz).abs() <= c.keeper_bore_half)

    def door_closed(self) -> torch.Tensor:
        return self.door_angle().abs() <= math.radians(self.cfg.closed_tol_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: door AND bolt at rest."""
        c = self.cfg
        return (self.door_rate().abs() < c.settle_door) \
            & (self.bolt.data.root_lin_vel_w.norm(dim=-1) < c.settle_bolt)

    def success(self) -> torch.Tensor:
        """(N,) bool: bolt engaged in the keeper, door flush against its stop, everything
        settled (current, physical state — the spring makes any unsecured closed door
        revert, so success is only reachable via the full retract/press/throw program)."""
        return self.engaged() & self.door_closed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 * retract_latch + 0.35 * close_latch + 0.40 *
        current success. Exactly 1.0 iff success() (success forces both latches full);
        ~0 for the null policy; a push-only (seed) strategy caps at 0.60 with the door
        left open by the spring."""
        succ = self.success().float()
        return (0.25 * torch.maximum(self.retract_latch, succ)
                + 0.35 * torch.maximum(self.close_latch, succ)
                + 0.40 * succ)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Plant: hinge return spring + damper + the `door_drive` probe torque on the door
        (world z == body z — the door only rotates about z), and the BODY-frame `bolt_force`
        buffer on the bolt (body frame == door frame: +x presses the door closed through the
        prismatic joint, +/-y slides the bolt). Then latch rubric progress."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        theta = self.door_angle()
        tq = self.door_drive + self.k_spring * (self.theta_eq - theta) \
            - c.door_damping * self.door_rate()
        torque = torch.zeros(n, 1, 3, device=dev)
        torque[:, 0, 2] = tq
        self.door.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), torque)
        self.bolt.set_external_force_and_torque(
            self.bolt_force.unsqueeze(1), torch.zeros(n, 1, 3, device=dev))

        retracted = (self.bolt_ext() <= c.retract_thresh).float()
        flush = (theta.abs() <= math.radians(c.close_latch_deg)).float()
        succ = self.success().float()  # the legit path implies both latches
        retracted = torch.maximum(retracted, succ)
        flush = torch.maximum(flush, succ)
        # A diverged substep must not latch: torch.maximum propagates NaN. A garbage
        # frame earns NO progress.
        retracted = torch.nan_to_num(retracted, nan=0.0, posinf=0.0, neginf=0.0)
        flush = torch.nan_to_num(flush, nan=0.0, posinf=0.0, neginf=0.0)
        self.retract_latch = torch.maximum(self.retract_latch, retracted)
        self.close_latch = torch.maximum(self.close_latch, flush)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in (("door", self.door), ("bolt", self.bolt))},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("theta_eq", "k_spring", "ext0", "retract_latch",
                               "close_latch", "door_drive", "bolt_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in (("door", self.door), ("bolt", self.bolt)):
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A cream mini-fridge cabinet stands with its doorway facing you; its gray door "
            f"hangs on a vertical hinge at the door's RIGHT edge (as you face the fridge). "
            f"The hinge carries a RETURN SPRING with an OPEN equilibrium: the door rests "
            f"swung open by {c.theta_eq_min_deg:.0f}-{c.theta_eq_max_deg:.0f} degrees "
            f"(sampled fresh every episode) and any unsecured door you let go of swings "
            f"back there. On the door's outer face near the free (LEFT) edge, at height "
            f"{c.door_center_closed[2] + c.bolt_local_z:.2f} m, rides a green steel slide "
            f"bolt with a dark knob: it slides {c.bolt_stroke * 1000:.0f} mm along the "
            f"door edge direction and starts partly thrown, its tip protruding past the "
            f"door edge. On the cabinet wall beside the doorway sits a dark keeper housing, "
            f"proud of the wall, with a blind bore facing the bolt.\n"
            f"Goal: secure the fridge door — press it fully shut (within "
            f"{c.closed_tol_deg:.0f} deg of its stop) and hold it there against the "
            f"spring while you throw the bolt so its tip seats at least "
            f"{c.engage_min * 1000:.0f} mm into the keeper bore, then let go; the bolt in "
            f"the keeper is the only thing that keeps the door closed. The bolt must be "
            f"fully retracted BEFORE the door swings shut — a protruding tip slams into "
            f"the proud keeper housing and arrests the door short of flush — and the bore "
            f"has only {(c.keeper_bore_half - c.bolt_size[0] / 2) * 1000:.0f} mm of play, "
            f"so the bolt only enters while the door is pressed essentially flush. A door "
            f"pushed shut but not bolted does not count: the spring reopens it."
        )

    def instruction(self) -> str:
        return (
            "Secure the spring-loaded fridge door: retract the slide bolt fully, press the "
            "door flush shut and hold it against its return spring, then slide the bolt "
            "into the keeper bore on the cabinet wall and let go. Only the bolted door "
            "stays closed — pushing the door shut without the bolt lets the spring swing "
            "it back open."
        )


# ----- runnable env: scene physics only (NullRobot smoke) -> "simgen.spring_latch_fridge" ------
register_env("simgen", lambda: EnvCfg(scene="spring_latch_fridge", robot="null"))
