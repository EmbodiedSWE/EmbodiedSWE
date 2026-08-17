"""EggShelfFridgeScene — stow the egg in the marked door-shelf cup, then close the
fridge door GENTLY enough that the egg stays seated.

Derived from the RLBench `close_fridge` seed but STRATEGICALLY DIFFERENT (see TASK.md):
the seed's plan is one uncontrolled gross-motion push on a hinged panel — swing the
fridge door shut, done; HOW the door arrives does not matter and nothing rides on it.
Here the door is a moving VEHICLE for a loose payload: a fresh egg (a ball) must first
be seated into one of two shallow egg cups on the door's inner shelf (the cup whose rim
color matches the beacon block on the pedestal — sampled fresh every episode), and only
then may the door be swung closed — SLOWLY. The egg is held in its cup by nothing but
gravity and an 8 mm rim: a seed-style slam is physically self-defeating, because when
the door slaps its end stop the egg keeps its tangential velocity, pivots over the low
rim corner, and is gone (escape needs ~0.8 m/s at the cup, reached by any arrival rate
above ~3 rad/s; a careful close keeps cup speeds 8x lower). The plan skeleton changes
from "push
panel" to "load a payload onto the closure member, then execute a speed-bounded
transport with the articulation" — the MANNER of the motion is load-bearing and is
enforced by ejection physics, not by a rubric clause.

Mechanics (plain rigid bodies + authored USD joints, the oven_dials pattern): the
cabinet is five kinematic slabs; the door is ONE dynamic body (panel) with the shelf,
two cup-wall frames and a handle authored as compound collision children; a Z-axis
revolute joint (limits [0 deg, 105 deg], 0 = closed) hangs it on the hinge-side wall.
`post_step` applies viscous hinge friction and consumes the `door_drive` torque buffer
(solve.py's fingertip-scale push stand-in and smoke's probes write it; nothing else
touches the door's wrench slot). The joint pair is collision-filtered, so the closed
stop is the joint's own lower limit; the egg collides with everything.

Rubric (graded 0..1, latching transient achievement — anchored in the demonstrated
solve.py trajectory: seat, carry, closed):
  - `seat_latch`  (0.30): the egg has ever rested SETTLED inside the target cup;
  - `carry_latch` (0.30): best closure progress (1 - theta/theta0) reached WHILE the
    egg was in the target cup; forced to 1.0 the moment the door is closed with the
    egg aboard — an ejection en route keeps the credit earned up to the eject angle;
  - current success (0.40): egg in the target cup AND door within `closed_tol_deg` of
    the stop AND everything settled  ->  score == 1.0 iff success(), ~0 for the null
    policy (the egg starts on the counter, the door starts >= 55 deg open).

Per-episode randomization (readback-verified in smoke): door start angle, egg start
position on the counter, and WHICH cup is the target (the matching beacon block is
placed on the pedestal; the other beacon parks in an off-view depot).

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
class EggShelfFridgeSceneCfg(BaseCfg):
    """Config for `EggShelfFridgeScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    closed_tol_deg: float = tunable(3.0)  # door within this of the 0 deg stop = closed
    cup_xy_tol: float = tunable(0.014)  # egg centre per-axis offset from cup centre (door frame)
    # Honest by construction: the egg's max physical in-well offset is cup_half - egg_r
    # = 9 mm (~11 mm with contact-offset compliance, measured mid-carry), while an egg
    # ON the rim is >= 34 mm off and one outside the walls >= 45 mm off.
    cup_z_lo: float = tunable(-0.108)  # egg centre band in the door frame (rest = -0.099)
    cup_z_hi: float = tunable(-0.080)
    settle_egg: float = tunable(0.05)  # max egg |lin vel| (m/s) when judging
    settle_door: float = tunable(0.08)  # max hinge rate |omega_z| (rad/s) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    door_open_min_deg: float = tunable(55.0)  # door start angle sampled in [min, max]
    door_open_max_deg: float = tunable(100.0)
    egg_jitter: float = tunable(0.05)  # uniform +- xy jitter of the egg on the counter

    # --- tunable: door plant (difficulty dials) ----------------------------------------------
    # Viscous-only hinge friction: no static term, so a gentle drive never stalls and the
    # door rests wherever it stops. Slam audit (see TASK.md, measured in smoke check 6):
    # tau=4 N*m from >= 55 deg arrives at >= 6 rad/s; cup speeds 0.9-2.3 m/s exceed the
    # ~0.8 m/s rim-escape speed of the 8 mm corner (the stop-arrest impulse at the low
    # rim corner redirects ~half the tangential speed upward), so a seed-style slam
    # ejects the egg from EITHER cup; a 0.35 rad/s careful close stays ~8x under.
    door_damping: float = tunable(0.15)  # viscous hinge friction (N*m*s/rad)
    door_mass: float = tunable(2.0)
    egg_mass: float = tunable(0.055)
    egg_r: float = tunable(0.021)

    # --- info: structure (env-local coordinates) ---------------------------------------------
    # Cabinet interior x [0.35, 0.75], y [-0.20, 0.16], z [0.20, 0.62]; front aperture is
    # the full x=0.35 plane; the closed door covers it with >= 20 mm overlap on every
    # edge and a <= 8 mm slit — nothing egg-sized enters a closed fridge.
    cab_floor_pos: tuple = info((0.565, -0.02, 0.10))
    cab_floor_size: tuple = info((0.43, 0.42, 0.20))
    cab_back_pos: tuple = info((0.765, -0.02, 0.31))
    cab_back_size: tuple = info((0.03, 0.42, 0.62))
    cab_wall_hinge_pos: tuple = info((0.565, -0.215, 0.41))  # hinge-side (y-) wall = joint body0
    cab_wall_free_pos: tuple = info((0.565, 0.175, 0.41))
    cab_wall_size: tuple = info((0.43, 0.03, 0.42))
    cab_top_pos: tuple = info((0.565, -0.02, 0.635))
    cab_top_size: tuple = info((0.43, 0.42, 0.03))
    door_center_closed: tuple = info((0.330, -0.02, 0.42))  # panel centre at theta = 0
    door_size: tuple = info((0.024, 0.46, 0.46))
    hinge_y: float = info(-0.235)  # hinge axis at (door_center_closed.x, hinge_y), world-z
    door_limit_deg: float = info(105.0)
    # Door-local fixture (origin = panel centre; +x = inner normal when closed):
    shelf_center: tuple = info((0.057, -0.01, -0.126))  # slab under both cups
    shelf_size: tuple = info((0.090, 0.28, 0.012))  # top face at local z = -0.120
    cup_x: float = info(0.057)  # both cup centres sit at this local x
    cup_y: tuple = info((-0.075, 0.065))  # local y: cup 0 (BLUE, near hinge), cup 1 (ORANGE)
    cup_half: float = info(0.030)  # clear half-width of the square well
    cup_wall_t: float = info(0.008)
    # 8 mm walls: deep enough that a careful close (cup speed <= 0.10 m/s) can never
    # lift the egg out, shallow enough that a slam's stop-arrest impulse at the rim
    # corner (13 mm below the egg centre) vaults it — this height IS the task's
    # gentleness boundary (16 mm walls were measured to retain even a 1.7 m/s slam).
    cup_wall_h: float = info(0.008)  # rim top at local z = -0.112
    handle_center: tuple = info((-0.027, 0.17, 0.0))  # grab/push bar on the outer face
    handle_size: tuple = info((0.03, 0.024, 0.14))
    counter_pos: tuple = info((0.10, 0.34, 0.15))  # side table the egg starts on
    counter_size: tuple = info((0.26, 0.24, 0.30))
    pedestal_pos: tuple = info((-0.10, 0.34, 0.11))
    pedestal_size: tuple = info((0.07, 0.07, 0.22))
    beacon_size: float = info(0.045)
    beacon_depot: tuple = info(((0.0, 1.6), (0.3, 1.6)))  # off-view park for the unused beacon
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    hinge_xy: tuple = field(default=None, init=False)  # world (x, y) of the hinge axis
    egg_rest_local: float = field(default=None, init=False)  # egg centre z in the door frame
    cup_rim_z: float = field(default=None, init=False)  # rim top, door frame
    counter_top_z: float = field(default=None, init=False)
    beacon_on_pedestal: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.hinge_xy = (self.door_center_closed[0], self.hinge_y)
        shelf_top = self.shelf_center[2] + self.shelf_size[2] / 2  # -0.120
        self.egg_rest_local = shelf_top + self.egg_r  # -0.099
        self.cup_rim_z = shelf_top + self.cup_wall_h  # -0.112
        self.counter_top_z = self.counter_pos[2] + self.counter_size[2] / 2
        self.beacon_on_pedestal = (
            self.pedestal_pos[0], self.pedestal_pos[1],
            self.pedestal_pos[2] + self.pedestal_size[2] / 2 + self.beacon_size / 2 + 0.001)

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ----------------
    def cup_center_local(self, which: int) -> tuple:
        """Door-frame centre of cup `which` at the egg's REST height."""
        return (self.cup_x, self.cup_y[which], self.egg_rest_local)

    def cup_radius(self, which: int) -> float:
        """Distance of cup `which` from the hinge axis (the ejection lever arm)."""
        # door centre local y of the hinge = hinge_y - door_center_closed.y
        return abs(self.cup_y[which] - (self.hinge_y - self.door_center_closed[1]))

    def door_center_at(self, theta_rad: float) -> tuple:
        """World centre of the door panel at hinge angle theta (0 = closed)."""
        hx, hy = self.hinge_xy
        arm = self.door_center_closed[1] - self.hinge_y  # 0.215, along +y at closed
        return (hx - arm * math.sin(theta_rad), hy + arm * math.cos(theta_rad),
                self.door_center_closed[2])


def _quat_z(rad: float) -> tuple:
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("egg_shelf_fridge")
class EggShelfFridgeScene(BaseScene):
    cfg: EggShelfFridgeSceneCfg

    def __init__(self, cfg: EggShelfFridgeSceneCfg | None = None) -> None:
        super().__init__(cfg or EggShelfFridgeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, five kinematic cabinet slabs, the dynamic door panel (fixture
        children are authored in bind), the egg, the counter, the pedestal and the two
        beacon blocks (kinematic — landmarks, not pucks)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cream = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.88, 0.84))
        door_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.78, 0.80))
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.18))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.20))
        white = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.94, 0.88))
        blue = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95))
        orange = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.08))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives the door with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live (the oven_dials lesson).
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
            "cab_floor": (c.cab_floor_pos, c.cab_floor_size, cream),
            "cab_back": (c.cab_back_pos, c.cab_back_size, cream),
            "cab_wall_hinge": (c.cab_wall_hinge_pos, c.cab_wall_size, cream),
            "cab_wall_free": (c.cab_wall_free_pos, c.cab_wall_size, cream),
            "cab_top": (c.cab_top_pos, c.cab_top_size, cream),
            "counter": (c.counter_pos, c.counter_size, wood),
            "pedestal": (c.pedestal_pos, c.pedestal_size, dark),
        }
        for name, (pos, size, mat) in slabs.items():
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=size, rigid_props=kin, collision_props=coll, visual_material=mat),
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
        out["egg"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Egg",
            spawn=sim_utils.SphereCfg(
                radius=c.egg_r,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,  # kills the GPU sphere-creep artifact
                    linear_damping=0.03,
                    angular_damping=0.03,
                    **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.egg_mass),
                collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.55, restitution=0.1),
                visual_material=white,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.counter_pos[0], c.counter_pos[1], c.counter_top_z + c.egg_r + 0.002)),
        )
        for name, mat in (("beacon_blue", blue), ("beacon_orange", orange)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=(c.beacon_size,) * 3, rigid_props=kin, collision_props=coll,
                    visual_material=mat),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.beacon_on_pedestal),
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
        self.egg: RigidObject = env.iscene["egg"]
        self.beacons: dict[str, RigidObject] = {
            "blue": env.iscene["beacon_blue"], "orange": env.iscene["beacon_orange"]}
        self.env_origins = env.iscene.env_origins
        self._author_door_fixture()
        self._author_hinge()
        # Episode state.
        self.target_cup = torch.zeros(n, dtype=torch.long, device=dev)  # 0 = blue, 1 = orange
        self.theta0 = torch.full((n,), 1.0, device=dev)  # authored start angle (rad)
        self.seat_latch = torch.zeros(n, device=dev)  # egg ever settled in the target cup
        self.carry_latch = torch.zeros(n, device=dev)  # best closure progress with egg aboard
        # External drive input (solve.py and smoke probes write; post_step consumes + owns
        # the door's wrench slot — never call set_external_force_and_torque directly).
        self.door_drive = torch.zeros(n, device=dev)  # torque about the hinge (N*m, + opens)

    def _author_door_fixture(self) -> None:
        """Compound collision children on the door panel (env_0 only — env_1.. compose from
        env_0 by reference; authored idempotently): the shelf slab, the two cup-wall frames
        (BLUE = cup 0 near the hinge, ORANGE = cup 1) and the outer handle bar."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Door/shelf").IsValid():
            return

        def box(name: str, center: tuple, size: tuple, color: tuple) -> None:
            cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Door/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            px.CreateContactOffsetAttr(c.contact_offset)
            px.CreateRestOffsetAttr(0.0)

        gray = (0.55, 0.58, 0.60)
        box("shelf", c.shelf_center, c.shelf_size, gray)
        box("handle", c.handle_center, c.handle_size, (0.18, 0.18, 0.20))
        wall_z = c.shelf_center[2] + c.shelf_size[2] / 2 + c.cup_wall_h / 2
        frame_len = 2 * c.cup_half + 2 * c.cup_wall_t
        for which, color in ((0, (0.15, 0.35, 0.95)), (1, (0.95, 0.45, 0.08))):
            cy = c.cup_y[which]
            off = c.cup_half + c.cup_wall_t / 2
            box(f"cup{which}_yneg", (c.cup_x, cy - off, wall_z),
                (frame_len, c.cup_wall_t, c.cup_wall_h), color)
            box(f"cup{which}_ypos", (c.cup_x, cy + off, wall_z),
                (frame_len, c.cup_wall_t, c.cup_wall_h), color)
            box(f"cup{which}_xneg", (c.cup_x - off, cy, wall_z),
                (c.cup_wall_t, frame_len, c.cup_wall_h), color)
            box(f"cup{which}_xpos", (c.cup_x + off, cy, wall_z),
                (c.cup_wall_t, frame_len, c.cup_wall_h), color)

    def _author_hinge(self) -> None:
        """Per env: a Z-axis revolute joint hinge-wall -> door, limits [0, door_limit_deg]
        (0 = closed; the lower limit IS the door stop — the joint pair never collides)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        hx, hy = c.hinge_xy
        wx, wy, wz = c.cab_wall_hinge_pos
        dx, dy, dz = c.door_center_closed
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

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the target cup, place the matching beacon on the pedestal
        (other one to the depot), pose the door at a sampled open angle (a pure
        joint-coordinate re-pose about the unchanged hinge), drop the egg on the counter
        with xy jitter, clear latches and drive."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        self.target_cup[env_ids] = torch.randint(0, 2, (m,), device=dev)
        self.seat_latch[env_ids] = 0.0
        self.carry_latch[env_ids] = 0.0
        self.door_drive[env_ids] = 0.0

        # --- door at the sampled open angle ---
        theta = torch.deg2rad(
            c.door_open_min_deg
            + (c.door_open_max_deg - c.door_open_min_deg) * torch.rand(m, device=dev))
        self.theta0[env_ids] = theta
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

        # --- egg on the counter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.counter_pos[0]
        st[:, 1] = c.counter_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.egg_jitter
        st[:, 2] = c.counter_top_z + c.egg_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.egg.write_root_state_to_sim(st, env_ids)

        # --- beacons: target's color on the pedestal, the other in the depot ---
        for which, name in ((0, "blue"), (1, "orange")):
            on = (self.target_cup[env_ids] == which).unsqueeze(1)
            ped = torch.tensor(c.beacon_on_pedestal, device=dev).expand(m, 3)
            depot = torch.tensor(
                (*c.beacon_depot[which], c.beacon_size / 2 + 0.001), device=dev).expand(m, 3)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(on, ped, depot)
            st[:, 3] = 1.0
            self.beacons[name].write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def door_angle(self) -> torch.Tensor:
        """(N,) hinge angle in rad (0 = closed; the door only ever rotates about world z)."""
        q = self.door.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def door_rate(self) -> torch.Tensor:
        """(N,) signed hinge rate (rad/s) — the door's world-z angular velocity."""
        return self.door.data.root_ang_vel_w[:, 2]

    def egg_local(self) -> torch.Tensor:
        """(N, 3) egg centre in the DOOR body frame (the cups live in this frame, so a
        moving door judges identically to a parked one)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.egg.data.root_pos_w - self.door.data.root_pos_w
        return quat_apply_inverse(self.door.data.root_quat_w, rel)

    def egg_in_cup(self, which: torch.Tensor) -> torch.Tensor:
        """(N,) bool, geometric: egg centre inside cup `which` (per-env index tensor) —
        per-axis xy within `cup_xy_tol` of the cup centre and z inside the well band, all
        in the door frame. No velocity clause (used while the door is moving)."""
        c = self.cfg
        loc = self.egg_local()
        cup_y = torch.where(which == 0,
                            torch.full_like(loc[:, 1], c.cup_y[0]),
                            torch.full_like(loc[:, 1], c.cup_y[1]))
        return ((loc[:, 0] - c.cup_x).abs() < c.cup_xy_tol) \
            & ((loc[:, 1] - cup_y).abs() < c.cup_xy_tol) \
            & (loc[:, 2] > c.cup_z_lo) & (loc[:, 2] < c.cup_z_hi)

    def egg_in_target(self) -> torch.Tensor:
        return self.egg_in_cup(self.target_cup)

    def door_closed(self) -> torch.Tensor:
        return self.door_angle().abs() <= math.radians(self.cfg.closed_tol_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: egg AND door at rest."""
        c = self.cfg
        return (self.egg.data.root_lin_vel_w.norm(dim=-1) < c.settle_egg) \
            & (self.door_rate().abs() < c.settle_door)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 * seat_latch + 0.30 * carry_latch + 0.40 * current
        success. Exactly 1.0 iff success() (success forces both latches to 1); ~0 for the
        null policy; an egg ejected after a full carry keeps 0.60 (latched credit)."""
        return (0.30 * self.seat_latch + 0.30 * self.carry_latch
                + 0.40 * (self.egg_in_target() & self.door_closed() & self.settled()).float())

    def success(self) -> torch.Tensor:
        """(N,) bool: egg seated in the TARGET cup, door closed against its stop,
        everything settled (current, physical state)."""
        return self.egg_in_target() & self.door_closed() & self.settled()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Door plant: viscous hinge friction + the external `door_drive` torque buffer
        (owns the door's wrench slot); then latch rubric progress. Torque is pure world-z
        and the door only rotates about z, so the body-frame wrench never drifts."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        tq = self.door_drive - c.door_damping * self.door_rate()
        torque = torch.zeros(n, 1, 3, device=dev)
        torque[:, 0, 2] = tq
        self.door.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), torque)

        in_cup = self.egg_in_target()
        seated = (in_cup & self.settled()).float()
        theta = self.door_angle().abs()
        prog = (1.0 - theta / self.theta0).clamp(0.0, 1.0) * in_cup.float()
        prog = torch.where(in_cup & self.door_closed(), torch.ones_like(prog), prog)
        # A diverged substep must not latch: torch.maximum propagates NaN and the latches
        # are checkpointed via get_state/set_state. A garbage frame earns NO progress.
        seated = torch.nan_to_num(seated, nan=0.0, posinf=0.0, neginf=0.0)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self.seat_latch = torch.maximum(self.seat_latch, seated)
        self.carry_latch = torch.maximum(self.carry_latch, prog)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"door": self.door, "egg": self.egg,
                  "beacon_blue": self.beacons["blue"], "beacon_orange": self.beacons["orange"]}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("target_cup", "theta0", "seat_latch", "carry_latch",
                               "door_drive")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"door": self.door, "egg": self.egg,
                  "beacon_blue": self.beacons["blue"], "beacon_orange": self.beacons["orange"]}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A cream mini-fridge cabinet stands with its doorway facing you; its gray door "
            f"hangs on a vertical hinge at the door's RIGHT edge (as you face the fridge) "
            f"and starts swung open by {c.door_open_min_deg:.0f}-{c.door_open_max_deg:.0f} "
            f"degrees. On the door's inner face, a small shelf (at height "
            f"{c.shelf_center[2] + c.door_center_closed[2]:.2f} m) carries two shallow "
            f"square egg cups, open side up: the cup NEARER the hinge has BLUE walls, the "
            f"cup nearer the door's free edge has ORANGE walls. A white egg (a "
            f"{2 * c.egg_r * 100:.1f} cm ball) lies on the wooden counter to the left of "
            f"the fridge, and on the dark pedestal beside the counter sits one colored "
            f"beacon block — its color (blue or orange, sampled fresh every episode) names "
            f"the cup the egg belongs in.\n"
            f"Goal: place the egg into the door cup whose wall color matches the beacon, "
            f"then swing the fridge door fully closed (within {c.closed_tol_deg:.0f} deg "
            f"of its stop) with the egg still seated in that cup, and let everything come "
            f"to rest. The egg is held only by gravity and a "
            f"{c.cup_wall_h * 1000:.0f} mm cup rim: slamming the door ejects it when the "
            f"door hits its stop, so close SLOWLY (cup speeds must stay well under "
            f"~0.8 m/s). The egg must be loaded while the door is open — a closed door "
            f"hides the cups inside the cabinet. The egg in the wrong cup, the egg "
            f"anywhere else, or a door left ajar does not count."
        )

    def instruction(self) -> str:
        return (
            "Put the egg into the door-shelf cup whose rim color matches the beacon block "
            "on the pedestal, then gently swing the fridge door fully closed so the egg "
            "stays seated in its cup. Slamming the door ejects the egg and fails the task."
        )


# ----- runnable env: scene physics only (NullRobot smoke) -> "simgen.egg_shelf_fridge" ---------
register_env("simgen", lambda: EnvCfg(scene="egg_shelf_fridge", robot="null"))
