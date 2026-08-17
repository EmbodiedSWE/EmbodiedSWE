"""BatchScaleScene — weigh out the batch on the baker's spring scale (open_oven_i213).

Derived from the RLBench `open_oven` seed but STRATEGICALLY DIFFERENT (see TASK.md): the
seed's plan is "grasp the door handle, pull the hinged panel through a large free arc" —
one prehensile act on the appliance's single moving part, judged by a binary joint
reading. Here NOTHING is pulled open and the manipulands are not part of the appliance
at all: a spring SCALE stands on the floor (kinematic base + tick mast; a dynamic TRAY
hangs on an authored vertical prismatic joint, suspended by a spring applied in
`post_step`). Five loose counterweights lie on the floor: three small brass cubes
(1 unit = 120 g each) and two large iron cubes (2 units = 240 g each). Every episode
samples a target load u* in {2..6} units and physically parks a green target tab on the
mast at the tick the pointer must reach. **Goal (carried here, no task layer): place
counterweights in the tray until the orange pointer rests level with the green tab —
the right TOTAL MASS, resting in the tray, everything settled.** Goal-conditioned
discrete mass selection with indirect indicator control: the pointer cannot be "stopped"
anywhere — only the resting load determines where it settles; too little AND too much
both fail, and a wrong load must be exchanged, not nudged.

Mechanics (the oven_dials-proven pattern — plain rigid bodies + authored USD joints;
the spring/damper is an external force applied in `post_step`, overwritten every
substep): tray root z is the single mechanical DOF; deflection d = z_eq - z where z_eq
is the empty-tray equilibrium (computed in cfg, marked by the white zero tick). One
unit of load sinks the tray by delta_unit = unit_mass*g/k = 15.0 mm = one black tick.
`post_step` also consumes `ext_force`/`ext_torque` probe buffers (writers: smoke's
rubric probes ONLY — solve.py never touches them; nothing else writes the tray's
wrench slot).

Rubric honesty — the "accounted" clause: success and all graded credit require the
measured deflection to MATCH the weight actually resting inside the tray
(|d - m_on_tray*g/k| <= consist_tol). Pressing the tray down (seed-style pulling,
hand pressure, a hovering pinned weight) produces deflection with no resting load, or
load with no deflection, and earns nothing. Credit latches only after `acc_streak`
consecutive settled+accounted substeps, so fly-through band crossings and
turning-point stillness (velocity-zero at the bottom of a bounce) latch nothing.

Rubric (graded 0..1, latched, anchored in the demonstrated solve.py trajectory):
  - stage1 (0.15, latched): some weight ever rested in the tray (accounted, settled);
  - best_prog (0.45, latched): max over settled+accounted frames of
    1 - |units_on - u*|/u* — progress of the RESTING LOAD toward the target load;
  - +0.40 iff success() NOW: |d - u**delta| <= band_tol AND accounted AND settled
    (acc_streak >= band_streak). score == 1.0 iff success(); null policy ~0; a
    correct load later removed leaves the latched 0.60.

Per-episode randomization: u* (5 values), weight->floor-slot permutation, per-weight xy
jitter + free yaw, and the green tab physically re-parks at the target tick (verified
by readback in smoke) — a memorized fixed placement count fails across episodes.

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

G = 9.81


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BatchScaleSceneCfg(BaseCfg):
    """Config for `BatchScaleScene`. All geometry/plant numbers are derived once in
    `__post_init__` so the scene, the smoke AND the solver read the same values."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    band_tol: float = tunable(0.006)  # |d - target| <= this = pointer on the tab (0.4 units)
    consist_tol: float = tunable(0.007)  # |d - resting_load*g/k| <= this = "accounted"
    settle_vz: float = tunable(0.02)  # max |tray vz| (m/s) on frames that count
    weight_settle_v: float = tunable(0.05)  # max |weight v| to count as RESTING in the tray
    band_streak: int = tunable(24)  # consecutive good substeps (0.2 s) for success()
    latch_streak: int = tunable(10)  # consecutive good substeps before credit latches

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    u_min: int = tunable(2)  # min target load (units)
    u_max: int = tunable(6)  # max target load (units)
    slot_jitter: float = tunable(0.015)  # +- xy jitter of the floor slots (m)

    # --- tunable: spring plant (difficulty dials) -----------------------------------------------
    # k chosen so one unit (120 g) = 15.0 mm = one tick; discrete-stability audit at 120 Hz:
    # omega_n*dt = sqrt(k/m_tray)/120 ~ 0.12 << 2, c*dt/m ~ 0.33 < 1; zeta 0.7..1.3 over the
    # load range -> a drop rings once and settles inside a second.
    k_spring: float = tunable(78.48)  # N/m
    c_damp: float = tunable(14.0)  # N*s/m

    # --- info: weights ---------------------------------------------------------------------------
    unit_mass: float = info(0.12)  # one unit (small brass cube), kg
    small_size: float = info(0.040)  # brass cube edge (3x, 1 unit each)
    big_size: float = info(0.050)  # iron cube edge (2x, 2 units each)
    weight_friction: float = info(0.8)

    # --- info: scale structure -------------------------------------------------------------------
    scale_xy: tuple = info((0.50, -0.20))  # tray/base centre
    base_size: tuple = info((0.26, 0.26, 0.04))  # kinematic pedestal slab, on the ground
    mast_center: tuple = info((0.66, -0.20, 0.25))
    mast_size: tuple = info((0.05, 0.12, 0.50))
    plate_size: tuple = info((0.20, 0.20, 0.012))  # dynamic tray floor plate
    rim_h: float = info(0.032)  # tray rim wall height
    rim_t: float = info(0.008)  # tray rim wall thickness
    tray_mass: float = info(0.35)
    z_nat: float = info(0.254)  # tray ROOT z with the spring at natural length
    joint_up: float = info(0.004)  # prismatic upper stop above natural (m)
    joint_down: float = info(0.185)  # prismatic lower stop below natural (m)
    pointer_len: float = info(0.032)
    n_ticks: int = info(8)  # ticks 0..7 units down the mast
    marker_size: tuple = info((0.014, 0.036, 0.010))  # green target tab
    contact_offset: float = info(0.002)
    # floor slots for the five weights (row, robot side): x = slot_x0 + i*slot_pitch, y = slot_y
    slot_x0: float = info(0.27)
    slot_pitch: float = info(0.09)
    slot_y: float = info(0.16)

    # Derived (filled in __post_init__).
    delta_unit: float = field(default=None, init=False)  # tray sink per unit (m) = tick pitch
    sag0: float = field(default=None, init=False)  # empty-tray static sag below natural
    z_eq: float = field(default=None, init=False)  # empty-tray equilibrium ROOT z
    plate_top_eq: float = field(default=None, init=False)  # empty tray floor top z
    z_zero: float = field(default=None, init=False)  # pointer centre z at empty rest (tick 0)
    mast_face_x: float = field(default=None, init=False)
    tray_inner_half: float = field(default=None, init=False)  # rim inner half-extent
    manifest: tuple = field(default=None, init=False)  # ((name, size, mass, units), ...)

    def __post_init__(self) -> None:
        self.delta_unit = self.unit_mass * G / self.k_spring
        self.sag0 = self.tray_mass * G / self.k_spring
        self.z_eq = self.z_nat - self.sag0
        self.plate_top_eq = self.z_eq + self.plate_size[2] / 2
        self.z_zero = self.z_eq + self.plate_size[2] / 2 + 0.005  # pointer centre, empty rest
        self.mast_face_x = self.mast_center[0] - self.mast_size[0] / 2
        self.tray_inner_half = self.plate_size[0] / 2 - self.rim_t
        self.manifest = tuple(
            [(f"small_{i}", self.small_size, self.unit_mass, 1) for i in range(3)]
            + [(f"big_{i}", self.big_size, 2 * self.unit_mass, 2) for i in range(2)]
        )
        # ----- honesty-by-construction asserts -----
        assert abs(self.delta_unit - 0.015) < 1e-6, "one unit must be one 15 mm tick"
        # the band admits exactly ONE integer load: the neighbour loads sit delta away
        assert self.band_tol < self.delta_unit - self.band_tol, "band must exclude u*+-1"
        assert self.consist_tol < self.delta_unit / 2, "consistency must resolve one unit"
        # travel: the lower stop sits below the deepest sampled load with margin
        assert self.joint_down > self.sag0 + 7 * self.delta_unit + 0.02
        # deepest tray floor stays above the pedestal top
        assert (self.z_nat - self.joint_down - self.plate_size[2] / 2
                > self.base_size[2] + 0.01)
        # a parallel jaw (80 mm) fits both cube sizes with room
        assert self.big_size < 0.065
        # tray opening comfortably admits the biggest cube
        assert self.tray_inner_half * 2 > self.big_size + 0.10

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ------------------
    def tick_z(self, units: float) -> float:
        return self.z_zero - units * self.delta_unit

    def marker_pos(self, units: int) -> tuple:
        return (self.mast_face_x + self.marker_size[0] / 2 - 0.002,
                self.scale_xy[1] + 0.032, self.tick_z(units))

    def slot_xy(self, i: int) -> tuple:
        return (self.slot_x0 + i * self.slot_pitch, self.slot_y)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("batch_scale")
class BatchScaleScene(BaseScene):
    cfg: BatchScaleSceneCfg

    def __init__(self, cfg: BatchScaleSceneCfg | None = None) -> None:
        super().__init__(cfg or BatchScaleSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic pedestal + tick mast, the dynamic tray plate (rim
        walls and the pointer are authored as child prims in bind()), the green target tab
        (kinematic — a landmark, not a puck), and the five counterweights."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.20, 0.22))
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.58, 0.62))
        blue = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.35, 0.55))
        green = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.80, 0.20))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.62, 0.24))
        iron = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.28, 0.32))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        grippy = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.weight_friction, dynamic_friction=c.weight_friction,
            restitution=0.0)
        cx, cy = c.scale_xy

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
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=sim_utils.CuboidCfg(
                    size=c.base_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, visual_material=dark,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, c.base_size[2] / 2)),
            ),
            "mast": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mast",
                spawn=sim_utils.CuboidCfg(
                    size=c.mast_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.mast_center),
            ),
            # Tray plate: the ONLY dynamic scale part. post_step drives it with (deprecated)
            # external wrenches that do NOT wake a sleeping body — sleep_threshold=0 keeps
            # the plant live (the oven_dials lesson).
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=sim_utils.CuboidCfg(
                    size=c.plate_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=12,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
                    collision_props=coll, visual_material=blue,
                    physics_material=grippy,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, c.z_eq)),
            ),
            "marker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marker",
                spawn=sim_utils.CuboidCfg(
                    size=c.marker_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, visual_material=green,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.marker_pos(c.u_min)),
            ),
        }
        for i, (name, size, mass, units) in enumerate(self.cfg.manifest):
            sx, sy = c.slot_xy(i)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/W_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(size, size, size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=12,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=coll,
                    visual_material=brass if units == 1 else iron,
                    physics_material=grippy,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, size / 2 + 0.002)),
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
        self.base: RigidObject = env.iscene["base"]
        self.mast: RigidObject = env.iscene["mast"]
        self.tray: RigidObject = env.iscene["tray"]
        self.marker: RigidObject = env.iscene["marker"]
        self.weights: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _s, _m, _u in c.manifest}
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        self._author_decorations()
        self._masses = torch.tensor([m for _n, _s, m, _u in c.manifest], device=dev)
        self._sizes = torch.tensor([s for _n, s, _m, _u in c.manifest], device=dev)
        # Episode state.
        self.target_units = torch.full((n,), c.u_min, dtype=torch.long, device=dev)
        self.best_prog = torch.zeros(n, device=dev)  # latched load progress in [0, 1]
        self.stage1 = torch.zeros(n, dtype=torch.bool, device=dev)  # ever loaded anything
        self.acc_streak = torch.zeros(n, dtype=torch.long, device=dev)  # settled+accounted run
        # Probe buffers (smoke's rubric probes write these; post_step consumes + owns the
        # tray's external-wrench slot — never call set_external_force_and_torque directly).
        self.ext_force = torch.zeros(n, 3, device=dev)
        self.ext_torque = torch.zeros(n, 3, device=dev)

    def _author_joint(self) -> None:
        """Per env: a Z-axis prismatic joint Base->Tray (the scale's suspension DOF).
        Joint pair never collides; the kinematic base is never teleported, so the
        world-fixed anchor is safe."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        cx, cy = c.scale_xy
        bz = c.base_size[2] / 2
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/tray_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Base"])
            j.CreateBody1Rel().SetTargets([f"{base}/Tray"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.z_nat - bz))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.joint_down)
            j.CreateUpperLimitAttr(c.joint_up)

    def _author_decorations(self) -> None:
        """Child prims on env_0 (isaaclab composes env_1.. from env_0 by reference — the
        oven_dials idempotency precedent): COLLIDING tray rim walls (part of the tray's
        rigid body -> a real tray, weights cannot slide off), the visual-only orange
        pointer on the tray, and the visual-only tick column on the mast."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Tray/rim_px").IsValid():
            return
        ph, pt = c.plate_size[0] / 2, c.plate_size[2] / 2
        wall_z = pt + c.rim_h / 2
        blue = Gf.Vec3f(0.25, 0.35, 0.55)
        # rim walls: (name, translate, scale) — full boxes, UsdGeom.Cube size 1
        walls = (
            ("rim_px", (ph - c.rim_t / 2, 0.0, wall_z), (c.rim_t, 2 * ph, c.rim_h)),
            ("rim_mx", (-ph + c.rim_t / 2, 0.0, wall_z), (c.rim_t, 2 * ph, c.rim_h)),
            ("rim_py", (0.0, ph - c.rim_t / 2, wall_z), (2 * ph - 2 * c.rim_t, c.rim_t, c.rim_h)),
            ("rim_my", (0.0, -ph + c.rim_t / 2, wall_z), (2 * ph - 2 * c.rim_t, c.rim_t, c.rim_h)),
        )
        for name, tr, sc in walls:
            cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Tray/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*tr))
            xf.AddScaleOp().Set(Gf.Vec3f(*sc))
            cube.CreateDisplayColorAttr([blue])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # pointer: visual only (NO CollisionAPI) — reaches toward the mast face
        tip_x = c.mast_face_x - self.cfg.scale_xy[0] - 0.003
        ptr = UsdGeom.Cube.Define(stage, "/World/envs/env_0/Tray/pointer")
        ptr.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(ptr.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(tip_x - c.pointer_len / 2, 0.0, pt + 0.005))
        xf.AddScaleOp().Set(Gf.Vec3f(c.pointer_len, 0.008, 0.010))
        ptr.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.45, 0.05)])
        # tick column on the mast face: tick 0 (empty rest) WHITE and long, 1..7 black
        mx, my, mz = c.mast_center
        for j in range(c.n_ticks):
            tick = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Mast/tick_{j}")
            tick.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(tick.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(
                -c.mast_size[0] / 2 + 0.0015, -0.012, c.tick_z(j) - mz))
            long = j == 0
            xf.AddScaleOp().Set(Gf.Vec3f(0.004, 0.070 if long else 0.045, 0.003))
            tick.CreateDisplayColorAttr(
                [Gf.Vec3f(0.95, 0.95, 0.95) if long else Gf.Vec3f(0.05, 0.05, 0.05)])

    # ----- reset -----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the target load u* in {u_min..u_max}, park the green tab at
        its tick (kinematic teleport, readback-verified in smoke), re-pose the tray at the
        empty equilibrium (pure joint-coordinate re-pose of the follower about the unmoved
        kinematic base), scatter the weights over a permuted floor-slot row with jitter +
        free yaw. Discrete draws use torch.rand (first-randint-after-seed degeneracy)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        cx, cy = c.scale_xy

        n_u = c.u_max - c.u_min + 1
        self.target_units[env_ids] = (
            (torch.rand(m, device=dev) * n_u).long().clamp(0, n_u - 1) + c.u_min)
        self.best_prog[env_ids] = 0.0
        self.stage1[env_ids] = False
        self.acc_streak[env_ids] = 0
        self.ext_force[env_ids] = 0.0
        self.ext_torque[env_ids] = 0.0

        # tray: empty equilibrium, zero velocity
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = cx
        st[:, 1] = cy
        st[:, 2] = c.z_eq
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # marker: kinematic park at the target tick
        st = torch.zeros(m, 13, device=dev)
        st[:, 3] = 1.0
        for row in range(m):
            pos = c.marker_pos(int(self.target_units[env_ids[row]]))
            st[row, 0:3] = origin[row] + torch.tensor(pos, device=dev)
        self.marker.write_root_state_to_sim(st, env_ids)

        # weights: permuted slots + jitter + free yaw, resting on the ground
        perm = torch.rand(m, len(c.manifest), device=dev).argsort(dim=1)
        for i, (name, size, _mass, _u) in enumerate(c.manifest):
            slot = perm[:, i]  # (m,) slot index for this weight
            sx = c.slot_x0 + slot.float() * c.slot_pitch
            sy = torch.full((m,), c.slot_y, device=dev)
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx + jit[:, 0]
            st[:, 1] = sy + jit[:, 1]
            st[:, 2] = size / 2 + 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.weights[name].write_root_state_to_sim(st, env_ids)

    # ----- readings / rubric -----------------------------------------------------------------------
    def deflection(self) -> torch.Tensor:
        """(N,) tray sink below the empty-rest equilibrium (m); one unit = delta_unit."""
        z = self.tray.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return self.cfg.z_eq - z

    def target_deflection(self) -> torch.Tensor:
        return self.target_units.float() * self.cfg.delta_unit

    def on_tray(self) -> torch.Tensor:
        """(N, 5) bool: weight RESTING inside the tray — xy inside the rim, z from the tray
        floor up through one stacked layer, and slow. A hovering (pinned) weight passes
        this only while its velocity is ~0, and then fails `accounted` instead."""
        c = self.cfg
        cx, cy = c.scale_xy
        pos = torch.stack([b.data.root_pos_w for b in self.weights.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.weights.values()], dim=1)
        rel = pos - self.env_origins[:, None, :]
        plate_top = (self.tray.data.root_pos_w[:, 2] - self.env_origins[:, 2]
                     + c.plate_size[2] / 2)
        in_x = (rel[:, :, 0] - cx).abs() < (c.tray_inner_half - 0.005)
        in_y = (rel[:, :, 1] - cy).abs() < (c.tray_inner_half - 0.005)
        lo = plate_top[:, None] + self._sizes[None, :] / 2 - 0.006
        hi = plate_top[:, None] + 0.115
        in_z = (rel[:, :, 2] > lo) & (rel[:, :, 2] < hi)
        return in_x & in_y & in_z & (vel < c.weight_settle_v)

    def load_mass(self) -> torch.Tensor:
        """(N,) total mass resting in the tray (kg)."""
        return (self.on_tray().float() * self._masses[None, :]).sum(dim=1)

    def units_on(self) -> torch.Tensor:
        """(N,) resting load in units (float; integer at rest by construction)."""
        return self.load_mass() / self.cfg.unit_mass

    def accounted(self) -> torch.Tensor:
        """(N,) bool: the measured deflection matches the resting load — the honesty
        clause. Pressing the tray (deflection, no load) and hovering/pinning a weight
        (load, no deflection) both fail it."""
        want = self.load_mass() * G / self.cfg.k_spring
        return (self.deflection() - want).abs() <= self.cfg.consist_tol

    def in_band(self) -> torch.Tensor:
        """(N,) bool: pointer level with the green tab (deflection within band_tol)."""
        return (self.deflection() - self.target_deflection()).abs() <= self.cfg.band_tol

    def success(self) -> torch.Tensor:
        """(N,) bool: pointer resting on the tab AND the deflection is accounted for by
        weights resting in the tray AND that has held for band_streak substeps."""
        return (self.in_band() & self.accounted()
                & (self.acc_streak >= self.cfg.band_streak))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 stage1 (ever loaded, latched) + 0.45 best latched
        load-progress + 0.40 iff success() now. Exactly 1.0 iff success (progress of the
        exact load is exactly 1); ~0 for the null policy; 0.60 after a knock-off."""
        return (0.15 * self.stage1.float() + 0.45 * self.best_prog
                + 0.40 * self.success().float())

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self) -> None:
        """Scale plant: spring + damper on the tray's z plus the smoke probe buffers; then
        the rubric bookkeeping (streak-gated latches). Owns the tray's wrench slot."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        z = self.tray.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        vz = self.tray.data.root_lin_vel_w[:, 2]
        f = self.ext_force.clone()
        f[:, 2] += c.k_spring * (c.z_nat - z) - c.c_damp * vz
        self.tray.set_external_force_and_torque(
            f.view(n, 1, 3), self.ext_torque.view(n, 1, 3))

        # rubric bookkeeping: only settled AND accounted frames count
        still = vz.abs() < c.settle_vz
        acc = self.accounted() & still
        self.acc_streak = torch.where(
            acc, self.acc_streak + 1, torch.zeros_like(self.acc_streak))
        good = self.acc_streak >= c.latch_streak
        units = self.units_on()
        prog = (1.0 - (units - self.target_units.float()).abs()
                / self.target_units.float()).clamp(0.0, 1.0)
        # A diverged substep must not latch (oven_dials NaN-poisoning lesson).
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self.best_prog = torch.maximum(
            self.best_prog, torch.where(good, prog, torch.zeros_like(prog)))
        self.stage1 |= good & (units > 0.5)

    # ----- state (full, restorable) -------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"tray": self.tray, "marker": self.marker, **self.weights}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("target_units", "best_prog", "stage1", "acc_streak",
                               "ext_force", "ext_torque")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"tray": self.tray, "marker": self.marker, **self.weights}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description -----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A spring scale stands on the floor: a dark pedestal at "
            f"({c.scale_xy[0]:.2f}, {c.scale_xy[1]:.2f}) carrying a blue open-top TRAY "
            f"({c.plate_size[0] * 100:.0f} cm square, rim {c.rim_h * 100:.1f} cm, floor "
            f"~{c.plate_top_eq:.2f} m up) that hangs on an internal vertical spring — "
            f"loading the tray sinks it. An orange POINTER fixed to the tray reaches to a "
            f"grey tick MAST just behind it. On the mast face: a long WHITE tick level "
            f"with the pointer when the tray is empty, and {c.n_ticks - 1} BLACK ticks "
            f"below it, one per {c.delta_unit * 1000:.0f} mm. Each black tick is the sink "
            f"caused by {c.unit_mass * 1000:.0f} g of load. A GREEN TAB is fixed beside "
            f"one tick: that is this episode's target weight — it is sampled fresh every "
            f"episode, so read it from the scene (count ticks from the white zero down to "
            f"the tab).\n"
            f"On the floor nearby lie five counterweights: three small BRASS cubes "
            f"({c.small_size * 100:.0f} cm, {c.unit_mass * 1000:.0f} g — one tick each) and "
            f"two large IRON cubes ({c.big_size * 100:.0f} cm, "
            f"{2 * c.unit_mass * 1000:.0f} g — two ticks each).\n"
            f"Goal: place counterweights INSIDE the tray until the orange pointer rests "
            f"level with the green tab (within about a third of a tick), then leave the "
            f"scale alone. Any combination with the right total weight counts; order is "
            f"free. Too little OR too much both fail — if you overshoot, take weight back "
            f"out. Only weight resting in the tray counts: pressing the tray or pointer "
            f"down by hand, holding a weight above the tray, or leaning weights against "
            f"the scale does nothing. Leave unused weights on the floor, clear of the "
            f"scale."
        )

    def instruction(self) -> str:
        return (
            "Place counterweights into the blue tray of the spring scale until its orange "
            "pointer rests level with the green target tab on the mast: small brass cubes "
            "sink it one black tick, large iron cubes two. Exactly on the tab — too little "
            "or too much fails; pressing the tray down or holding weights does not count. "
            "Leave unused weights on the floor."
        )


# ----- runnable env: scene physics only (NullRobot) -> "simgen.batch_scale" ---------------------
register_env("simgen", lambda: EnvCfg(scene="batch_scale", robot="null"))
