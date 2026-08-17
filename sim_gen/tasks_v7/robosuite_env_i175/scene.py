"""CounterBalanceScene — level a beam balance by loading the exact counterweight subset.

Derived from the robosuite suite seed (Lift / Stack / Door / PickPlaceCan /
NutAssemblySquare) but STRATEGICALLY DIFFERENT (see TASK.md): every seed task's plan
is "move THE object to ITS goal pose" (lift it, stack it, thread it, swing the door)
— the object's own pose IS the success predicate. Here no manipulated object has a
goal pose at all: the judged state is a MECHANISM EQUILIBRIUM. A balance scale
stands on the table — a heavy-bobbed beam on a +-12 deg pivot with a pan HANGING
from each end — and a red CARGO cylinder of sampled mass (2..6 units, readable from
its height: 15 mm per unit) sits in one pan, pinning the beam at its stop. Three
brass counterweights (1, 2 and 4 units — mass readable from diameter) lie on the
table. The solver must choose the subset that sums EXACTLY to the cargo's mass
(binary decomposition — unique), load it into the EMPTY pan only, and let physics
level the beam: success is the beam resting level within `tol_deg`, settled, cargo
still seated, no weight anywhere else on the scale, spare weights off the scale.
One unit too much or too little settles the beam at ~7.7 deg — physically far
outside the 3 deg window — so only the exact subset can ever pass. Hanging pans
(the real reason balance scales hang their pans) make the measurement exact:
wherever a weight sits inside a pan, the pan swings until its load acts through the
hang point, so the lever arm is exactly L and placement sloppiness costs nothing.
No execution order is required.

Mechanics (plain rigid bodies + authored USD D6 joints — the proven pattern):
post->beam D6 frees exactly rotY within +-12 deg (the end stops the cargo pins the
beam against); beam->pan D6s free rotY within +-25 deg so each pan hangs plumb.
The beam's authored MassAPI puts 2.2 kg at 91 mm below the pivot: the pendulum bob
is the restoring spring (K ~ 2.0 N*m/rad), one unit of imbalance = ~7.7 deg, and a
needle under the bob sweeps across tick marks on the base so levelness is VISIBLE.
The task is pure passive physics — no wrench interface, no post_step drives; the
only inputs are where bodies get placed.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.25 * A_latch: some counterweight rests in the counter pan while the scale is
    quiet and the cargo is seated (the first load-bearing drop);
  - 0.35 * B_latch: best QUIET levelness progress (1 - |tilt|/12deg), gated on
    cargo seated + weights only in the counter pan + spares clear — latched only
    through a 30-substep quiet streak, so swing-through apexes latch nothing;
  - 1.0 iff success(). ~0 for the null policy (the beam rests pinned at its stop);
    latched credit never evaporates under correct behavior.

Per-episode randomization (readback-verified in smoke): cargo mass k in {2..6}
(five distinct cargo cylinders; the active one is seated, spares park in a ground
depot), WHICH side carries the cargo, and the table scatter (slot permutation +
jitter) of the three counterweights.

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
class CounterBalanceSceneCfg(BaseCfg):
    """Config for `CounterBalanceScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tol_deg: float = tunable(3.0)  # |beam tilt| within this = level (one unit off ~ 7.7 deg)
    settle_lin: float = tunable(0.05)  # max |lin vel| on beam/pans/loads when judging (m/s)
    settle_ang: float = tunable(0.15)  # max beam |ang vel| when judging (rad/s)
    pan_settle_ang: float = tunable(0.40)  # max pan |ang vel| when judging (rad/s)
    quiet_steps: int = tunable(30)  # consecutive settled substeps before latches mature (0.25 s)
    inventory_tol: float = tunable(0.075)  # |counter-pan mass - cargo mass| < this (= 0.5 unit)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    k_min: int = tunable(2)  # cargo mass lower bound (units)
    k_max: int = tunable(6)  # cargo mass upper bound (units)
    slot_jitter_x: float = tunable(0.030)  # weight scatter jitter on the table (m)
    slot_jitter_y: float = tunable(0.020)

    # --- tunable: plant ----------------------------------------------------------------------
    unit_mass: float = tunable(0.15)  # kg per unit; weights 1/2/4, cargo k units
    beam_mass: float = tunable(2.2)  # authored MassAPI mass (bob-dominated)
    beam_com_z: float = tunable(-0.091)  # authored CoM below the pivot -> K ~ 2.0 N*m/rad
    beam_ang_damp: float = tunable(3.0)  # rings down in a couple of seconds
    beam_lin_damp: float = tunable(0.2)
    pan_ang_damp: float = tunable(4.0)
    pan_lin_damp: float = tunable(0.5)
    pan_mass: float = tunable(0.10)

    # --- info: structure (env-local coordinates) ---------------------------------------------
    table_center: tuple = info((0.05, 0.0, 0.36))
    table_size: tuple = info((0.90, 1.00, 0.08))  # top at z = 0.40
    base_center: tuple = info((0.0, 0.0, 0.411))
    base_size: tuple = info((0.18, 0.16, 0.022))  # base plate, top at z = 0.422
    upright_y: float = info(0.065)  # the two uprights straddle the beam/bob
    upright_size: tuple = info((0.04, 0.014, 0.24))  # z 0.40..0.64
    pivot_z: float = info(0.62)  # beam root origin = the pivot
    arm: float = info(0.18)  # hang points at beam-local x = +-arm
    beam_bar: tuple = info((0.40, 0.03, 0.014))  # crossbar (the beam root prim)
    bob_size: tuple = info((0.06, 0.06, 0.10))  # pendulum bob, centre (0,0,-0.10)
    bob_cz: float = info(-0.10)
    needle_tip_z: float = info(-0.185)  # needle tip (beam-local); sweeps over the base ticks
    tilt_limit_deg: float = info(12.0)  # pivot D6 rotY end stops
    pan_swing_deg: float = info(25.0)  # pan hinge rotY end stops
    hang_z: float = info(-0.004)  # hang point, beam-local z (just under the crossbar)
    hang_drop: float = info(0.120)  # pan floor centre hangs this far below the hang point
    # (crossbar underside to pan-floor top = hang_drop - 0.011 = 109 mm: the tallest
    # cargo, 6 units = 90 mm, stands clear of the beam with 19 mm to spare)
    pan_floor: tuple = info((0.09, 0.17, 0.008))
    pan_rim_h: float = info(0.026)
    pan_rim_t: float = info(0.006)
    weight_h: float = info(0.030)
    weight_r: tuple = info((0.012, 0.017, 0.024))  # 1 / 2 / 4 units (mass ~ r^2 at equal h)
    weight_units: tuple = info((1, 2, 4))
    cargo_r: float = info(0.023)
    cargo_unit_h: float = info(0.015)  # cargo height = k * this (mass readable from height)
    slot_x: tuple = info((-0.15, 0.0, 0.15))  # weight scatter slots on the table
    slot_y: float = info(-0.28)
    depot_pos: tuple = info((1.1, 1.1))  # ground depot for the four spare cargos
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    table_top_z: float = field(default=None, init=False)
    pan_root_z: float = field(default=None, init=False)  # pan floor centre at reset (level beam)
    pan_inner_x: float = field(default=None, init=False)  # inner half-extents between the rims
    pan_inner_y: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.table_top_z = self.table_center[2] + self.table_size[2] / 2
        self.pan_root_z = self.pivot_z + self.hang_z - self.hang_drop
        self.pan_inner_x = self.pan_floor[0] / 2 - self.pan_rim_t
        self.pan_inner_y = self.pan_floor[1] / 2 - self.pan_rim_t

    def cargo_h(self, k: int) -> float:
        return k * self.cargo_unit_h

    def cargo_mass(self, k: int) -> float:
        return k * self.unit_mass


# ----- scene -----------------------------------------------------------------------------------
class CounterBalanceScene(BaseScene):
    cfg: CounterBalanceSceneCfg

    def __init__(self, cfg: CounterBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or CounterBalanceSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + scale post, the dynamic beam and two hanging
        pans (compound children + joints authored in bind), three counterweights and five
        cargo variants (the active one is seated at reset; spares park in a ground depot)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))
        charcoal = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.14, 0.14, 0.16))
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.55, 0.20))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.10, 0.08))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        solid = dict(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5,
        )

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
            # scale post: base plate root; uprights/crossmember/ticks authored in bind
            "post": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post",
                spawn=sim_utils.CuboidCfg(
                    size=c.base_size, rigid_props=kin, collision_props=coll,
                    visual_material=charcoal),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.base_center),
            ),
            # the beam: crossbar root; bob/needle/lugs + MassAPI authored in bind
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=sim_utils.CuboidCfg(
                    size=c.beam_bar,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=c.beam_lin_damp,
                        angular_damping=c.beam_ang_damp,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        **solid),
                    collision_props=coll,
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.pivot_z)),
            ),
        }
        # --- the two hanging pans: floor root; rims + strut visuals authored in bind ---
        for side, name in ((-1.0, "pan_l"), (1.0, "pan_r")):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + ("PanL" if side < 0 else "PanR"),
                spawn=sim_utils.CuboidCfg(
                    size=c.pan_floor,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=c.pan_lin_damp,
                        angular_damping=c.pan_ang_damp,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        **solid),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pan_mass),
                    collision_props=coll,
                    visual_material=charcoal,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(side * c.arm, 0.0, c.pan_root_z)),
            )
        # --- the three counterweights (mass ~ diameter^2: 1 / 2 / 4 units) ---
        for i, (r, u) in enumerate(zip(c.weight_r, c.weight_units)):
            out[f"w{u}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Weight{u}",
                spawn=sim_utils.CylinderCfg(
                    radius=r, height=c.weight_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.3, angular_damping=2.0, **solid),
                    mass_props=sim_utils.MassPropertiesCfg(mass=u * c.unit_mass),
                    collision_props=coll,
                    visual_material=brass,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x[i], c.slot_y, c.table_top_z + c.weight_h / 2 + 0.002)),
            )
        # --- the five cargo variants (height = k * 15 mm; only one is active per episode) ---
        for k in range(c.k_min, c.k_max + 1):
            out[f"cargo{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Cargo{k}",
                spawn=sim_utils.CylinderCfg(
                    radius=c.cargo_r, height=c.cargo_h(k), axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.3, angular_damping=2.0, **solid),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_mass(k)),
                    collision_props=coll,
                    visual_material=red,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.18 * (k - c.k_min), c.depot_pos[1],
                         c.cargo_h(k) / 2 + 0.003)),
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
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.beam: RigidObject = env.iscene["beam"]
        self.pans: dict[float, RigidObject] = {-1.0: env.iscene["pan_l"],
                                               1.0: env.iscene["pan_r"]}
        self.weights: dict[int, RigidObject] = {u: env.iscene[f"w{u}"] for u in c.weight_units}
        self.cargos: dict[int, RigidObject] = {k: env.iscene[f"cargo{k}"]
                                               for k in range(c.k_min, c.k_max + 1)}
        self.env_origins = env.iscene.env_origins
        self._author_dressing()
        self._author_joints()
        # Episode state.
        self.k = torch.full((n,), c.k_min, dtype=torch.long, device=dev)  # cargo units
        self.side = torch.ones(n, device=dev)  # +1: cargo in the RIGHT (+x) pan
        self.quiet_ctr = torch.zeros(n, dtype=torch.long, device=dev)
        self.a_latch = torch.zeros(n, device=dev)
        self.b_latch = torch.zeros(n, device=dev)
        self._unit_masses = torch.tensor([u * c.unit_mass for u in c.weight_units], device=dev)

    def _author_dressing(self) -> None:
        """Compound children, authored idempotently PER ENV: post uprights (collide) +
        crossmember + level ticks (visual); beam bob (collides) + needle + hang lugs
        (visual); pan rims (collide) + hanger struts (visual); beam MassAPI (the
        pendulum restoring spring lives in the authored CoM)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Beam/bob").IsValid():
            return

        def box(root: str, name: str, center: tuple, size: tuple, color: tuple,
                collide: bool = False) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                px.CreateContactOffsetAttr(c.contact_offset)
                px.CreateRestOffsetAttr(0.0)

        tol_x = -c.needle_tip_z * math.sin(math.radians(c.tol_deg))  # needle sweep at tol
        for i in range(self.env.num_envs):
            post = f"/World/envs/env_{i}/Post"
            bz = c.base_center[2]
            for s in (-1.0, 1.0):
                box(post, f"upright{'L' if s < 0 else 'R'}",
                    (0.0, s * c.upright_y, 0.52 - bz), c.upright_size, (0.14, 0.14, 0.16),
                    collide=True)
            box(post, "crossmember", (0.0, 0.0, 0.646 - bz), (0.03, 0.144, 0.012),
                (0.14, 0.14, 0.16))
            # level ticks on the base plate: white centre, amber +-tol marks
            top = c.base_size[2] / 2
            box(post, "tick_c", (0.0, 0.0, top + 0.002), (0.003, 0.03, 0.004),
                (0.95, 0.95, 0.95))
            box(post, "tick_p", (tol_x, 0.0, top + 0.002), (0.003, 0.03, 0.004),
                (0.95, 0.75, 0.20))
            box(post, "tick_m", (-tol_x, 0.0, top + 0.002), (0.003, 0.03, 0.004),
                (0.95, 0.75, 0.20))

            beam = f"/World/envs/env_{i}/Beam"
            box(beam, "bob", (0.0, 0.0, c.bob_cz), c.bob_size, (0.30, 0.30, 0.33),
                collide=True)
            box(beam, "needle", (0.0, 0.0, (c.bob_cz - c.bob_size[2] / 2 + c.needle_tip_z) / 2),
                (0.006, 0.006, abs(c.needle_tip_z) - (abs(c.bob_cz) + c.bob_size[2] / 2)),
                (0.85, 0.08, 0.05))
            for s in (-1.0, 1.0):
                box(beam, f"lug{'L' if s < 0 else 'R'}", (s * c.arm, 0.0, c.hang_z),
                    (0.016, 0.020, 0.016), (0.35, 0.35, 0.38))
            # authored mass: the bob IS the restoring spring (CoM below the pivot)
            prim = stage.GetPrimAtPath(beam)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(c.beam_mass)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, c.beam_com_z))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.004, 0.006, 0.0035))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

            for pan in ("PanL", "PanR"):
                root = f"/World/envs/env_{i}/{pan}"
                fx, fy, fz = c.pan_floor
                rim_z = fz / 2 + c.pan_rim_h / 2
                for s in (-1.0, 1.0):  # x rims
                    box(root, f"rimX{'L' if s < 0 else 'R'}",
                        (s * (fx / 2 - c.pan_rim_t / 2), 0.0, rim_z),
                        (c.pan_rim_t, fy, c.pan_rim_h), (0.20, 0.20, 0.22), collide=True)
                for s in (-1.0, 1.0):  # y rims
                    box(root, f"rimY{'L' if s < 0 else 'R'}",
                        (0.0, s * (fy / 2 - c.pan_rim_t / 2), rim_z),
                        (fx - 2 * c.pan_rim_t, c.pan_rim_t, c.pan_rim_h),
                        (0.20, 0.20, 0.22), collide=True)
                for s in (-1.0, 1.0):  # hanger struts + yoke (visual)
                    box(root, f"strut{'L' if s < 0 else 'R'}",
                        (0.0, s * (fy / 2 - c.pan_rim_t / 2), c.hang_drop / 2 + 0.008),
                        (0.006, 0.006, c.hang_drop - 0.016), (0.35, 0.35, 0.38))
                box(root, "yoke", (0.0, 0.0, c.hang_drop), (0.006, fy, 0.006),
                    (0.35, 0.35, 0.38))

    def _author_joints(self) -> None:
        """Per env: post->beam pivot D6 (rotY free within +-tilt_limit_deg) and
        beam->pan hang D6s (rotY free within +-pan_swing_deg). All other axes locked;
        joint pairs never collide."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()

        def d6(path: str, b0: str, b1: str, p0: tuple, p1: tuple, lim_deg: float) -> None:
            j = UsdPhysics.Joint.Define(stage, path)
            j.CreateBody0Rel().SetTargets([b0])
            j.CreateBody1Rel().SetTargets([b1])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(*p0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(*p1))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transY", "transZ", "rotX", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)  # low > high = locked
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "rotY")
            lim.CreateLowAttr(-lim_deg)
            lim.CreateHighAttr(lim_deg)

        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            d6(f"{base}/pivot_joint", f"{base}/Post", f"{base}/Beam",
               (0.0, 0.0, c.pivot_z - c.base_center[2]), (0.0, 0.0, 0.0),
               c.tilt_limit_deg)
            for side, pan in ((-1.0, "PanL"), (1.0, "PanR")):
                d6(f"{base}/hang_joint{'L' if side < 0 else 'R'}",
                   f"{base}/Beam", f"{base}/{pan}",
                   (side * c.arm, 0.0, c.hang_z), (0.0, 0.0, c.hang_drop),
                   c.pan_swing_deg)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample cargo mass k and WHICH pan carries it; write the whole
        linkage at its consistent level pose (memory rule: teleport all linked bodies
        together); seat the active cargo just above its pan floor; park the spares in
        the ground depot; scatter the weights on permuted, jittered table slots. Uses
        torch.rand throughout (first-randint degeneracy on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        n_k = c.k_max - c.k_min + 1
        k = c.k_min + (torch.rand(m, device=dev) * n_k).long().clamp(max=n_k - 1)
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           -torch.ones(m, device=dev), torch.ones(m, device=dev))
        self.k[env_ids] = k
        self.side[env_ids] = side
        self.quiet_ctr[env_ids] = 0
        self.a_latch[env_ids] = 0.0
        self.b_latch[env_ids] = 0.0

        def write(body: RigidObject, pos: torch.Tensor,
                  quat: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + pos
            if quat is None:
                st[:, 3] = 1.0
            else:
                st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- the linkage, at its PINNED equilibrium (beam already resting toward the
        # cargo side, just inside the stop; pans plumb under the tilted hang points) —
        # writing the level pose instead would slam the loaded beam onto its stop and
        # jolt the tall cargo into an undamped rolling precession ---
        phi = side * math.radians(c.tilt_limit_deg - 0.5)  # signed beam pitch
        cph, sph = torch.cos(phi), torch.sin(phi)
        beam_q = torch.zeros(m, 4, device=dev)
        beam_q[:, 0] = torch.cos(phi / 2)
        beam_q[:, 2] = torch.sin(phi / 2)  # rotY
        write(self.beam, torch.tensor([0.0, 0.0, c.pivot_z], device=dev).expand(m, 3),
              beam_q)
        pan_pos = {}
        for s, pan in self.pans.items():
            pos = torch.zeros(m, 3, device=dev)
            pos[:, 0] = s * c.arm * cph + c.hang_z * sph
            pos[:, 2] = c.pivot_z - s * c.arm * sph + c.hang_z * cph - c.hang_drop
            pan_pos[s] = pos
            write(pan, pos)

        # --- cargos: the active one seated in the (lowered) cargo pan, spares parked ---
        cargo_pan = torch.where((side > 0).unsqueeze(1), pan_pos[1.0], pan_pos[-1.0])
        for kk, body in self.cargos.items():
            act = (k == kk).unsqueeze(1)
            seat = cargo_pan.clone()
            seat[:, 2] += c.pan_floor[2] / 2 + c.cargo_h(kk) / 2 + 0.0005
            park = torch.tensor([c.depot_pos[0] + 0.18 * (kk - c.k_min), c.depot_pos[1],
                                 c.cargo_h(kk) / 2 + 0.003], device=dev).expand(m, 3)
            write(body, torch.where(act, seat, park))

        # --- weights: permuted slots + jitter (slots 0.15 apart >> max diameter: no overlap) ---
        perm_rank = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per weight
        for i, u in enumerate(c.weight_units):
            slot = perm_rank[:, i]
            pos = torch.zeros(m, 3, device=dev)
            pos[:, 0] = torch.tensor(c.slot_x, device=dev)[slot] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter_x
            pos[:, 1] = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter_y
            pos[:, 2] = c.table_top_z + c.weight_h / 2 + 0.002
            write(self.weights[u], pos)

    # ----- readings -----------------------------------------------------------------------------
    def tilt(self) -> torch.Tensor:
        """(N,) signed beam tilt (rad): positive = the +x end is DOWN (rotY frees only
        pitch, so the beam x-axis' world z-component is the whole story)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        ex_w = quat_apply(self.beam.data.root_quat_w, ex)
        return torch.asin((-ex_w[:, 2]).clamp(-1.0, 1.0))

    def level(self) -> torch.Tensor:
        """(N,) bool: |tilt| < tol_deg."""
        return self.tilt().abs() < math.radians(self.cfg.tol_deg)

    def _local(self, pos_w: torch.Tensor, ref) -> torch.Tensor:
        """World points (N, 3) -> ref body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(ref.data.root_quat_w, pos_w - ref.data.root_pos_w)

    def _in_pan(self, body: RigidObject, s: float) -> torch.Tensor:
        """(N,) bool: body centre inside pan s's inner box (pan-local)."""
        c = self.cfg
        loc = self._local(body.data.root_pos_w, self.pans[s])
        return (loc[:, 0].abs() < c.pan_inner_x + 0.004) \
            & (loc[:, 1].abs() < c.pan_inner_y + 0.004) \
            & (loc[:, 2] > c.pan_floor[2] / 2) & (loc[:, 2] < 0.11)

    def _on_beam_top(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre resting on the crossbar (beam-local) — a parked-on-the-
        beam cheat region distinct from both pans (which hang BELOW the beam origin)."""
        loc = self._local(body.data.root_pos_w, self.beam)
        return (loc[:, 0].abs() < 0.22) & (loc[:, 1].abs() < 0.06) \
            & (loc[:, 2] > 0.0) & (loc[:, 2] < 0.09)

    def _on_scale(self, body: RigidObject) -> torch.Tensor:
        return self._in_pan(body, -1.0) | self._in_pan(body, 1.0) | self._on_beam_top(body)

    def weight_in_pan(self, s_sign: torch.Tensor) -> torch.Tensor:
        """(N, 3) bool: weight i inside the pan on the per-env side `s_sign` (+-1)."""
        cols = []
        for u in self.cfg.weight_units:
            in_l = self._in_pan(self.weights[u], -1.0)
            in_r = self._in_pan(self.weights[u], 1.0)
            cols.append(torch.where(s_sign > 0, in_r, in_l))
        return torch.stack(cols, dim=1)

    def counter_mass(self) -> torch.Tensor:
        """(N,) total counterweight mass resting inside the COUNTER pan."""
        inw = self.weight_in_pan(-self.side).float()
        return (inw * self._unit_masses.unsqueeze(0)).sum(dim=1)

    def cargo_mass_t(self) -> torch.Tensor:
        """(N,) the active cargo's mass."""
        return self.k.float() * self.cfg.unit_mass

    def cargo_seated(self) -> torch.Tensor:
        """(N,) bool: the ACTIVE cargo rests inside the cargo-side pan."""
        seated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for kk, body in self.cargos.items():
            in_l = self._in_pan(body, -1.0)
            in_r = self._in_pan(body, 1.0)
            in_side = torch.where(self.side > 0, in_r, in_l)
            seated |= (self.k == kk) & in_side
        return seated

    def spares_clear(self) -> torch.Tensor:
        """(N,) bool: no INACTIVE cargo anywhere on the scale."""
        clear = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for kk, body in self.cargos.items():
            clear &= (self.k == kk) | ~self._on_scale(body)
        return clear

    def weights_ok(self) -> torch.Tensor:
        """(N,) bool: every counterweight is either inside the COUNTER pan or entirely
        off the scale (nothing in the cargo pan, nothing parked on the crossbar)."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        in_counter = self.weight_in_pan(-self.side)
        for i, u in enumerate(self.cfg.weight_units):
            ok &= in_counter[:, i] | ~self._on_scale(self.weights[u])
        return ok

    def inventory_ok(self) -> torch.Tensor:
        """(N,) bool: counter-pan mass matches the cargo mass to half a unit (units are
        integers, so only the EXACT subset passes; also rejects propping the beam level
        by hand — a held-level beam with the wrong mass aboard is not a balance)."""
        return (self.counter_mass() - self.cargo_mass_t()).abs() < self.cfg.inventory_tol

    def settled(self) -> torch.Tensor:
        """(N,) bool: beam, pans, weights and active cargo all slow RIGHT NOW."""
        c = self.cfg
        ok = (self.beam.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        for pan in self.pans.values():
            ok &= (pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (pan.data.root_ang_vel_w.norm(dim=-1) < c.pan_settle_ang)
        for w in self.weights.values():
            ok &= w.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for kk, body in self.cargos.items():
            ok &= (self.k != kk) | (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        return ok

    def quiet(self) -> torch.Tensor:
        """(N,) bool: settled for `quiet_steps` CONSECUTIVE substeps — velocity passes
        through zero at every swing apex, so a streak (not an instant) is what
        distinguishes rest from a turning point (memory rule)."""
        return self.quiet_ctr >= self.cfg.quiet_steps

    def success(self) -> torch.Tensor:
        """(N,) bool: the beam RESTS level with the books straight — level, quiet, cargo
        seated in its pan, >= 1 counterweight in the counter pan with the inventory
        matching the cargo to half a unit, nothing else anywhere on the scale."""
        return self.level() & self.quiet() & self.cargo_seated() & self.weights_ok() \
            & self.spares_clear() & (self.counter_mass() > 0.5 * self.cfg.unit_mass) \
            & self.inventory_ok()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); else 0.25 * A_latch + 0.35 * B_latch
        (both latched through the quiet streak in post_step — see the module docstring).
        The null policy scores 0.000: the beam rests pinned at its stop with an empty
        counter pan, so neither latch ever arms."""
        base = 0.25 * self.a_latch + 0.35 * self.b_latch
        return torch.where(self.success(), torch.ones_like(base), base)

    # ----- step-coupled bookkeeping (every substep) ----------------------------------------------
    def post_step(self) -> None:
        """Advance the quiet streak and mature the rubric latches. No drives — the task
        is pure passive physics."""
        c = self.cfg
        self.quiet_ctr = torch.where(self.settled(), self.quiet_ctr + 1,
                                     torch.zeros_like(self.quiet_ctr))
        q = self.quiet()
        any_counter = self.counter_mass() > 0.5 * c.unit_mass
        gate_a = (q & self.cargo_seated() & any_counter).float()
        self.a_latch = torch.maximum(self.a_latch, torch.nan_to_num(gate_a, nan=0.0))
        prog = (1.0 - self.tilt().abs() / math.radians(c.tilt_limit_deg)).clamp(0.0, 1.0)
        gate_b = (q & self.cargo_seated() & any_counter & self.weights_ok()
                  & self.spares_clear()).float()
        self.b_latch = torch.maximum(
            self.b_latch, torch.nan_to_num(prog * gate_b, nan=0.0, posinf=0.0, neginf=0.0))

    # ----- state (full, restorable) --------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        out = {"beam": self.beam, "pan_l": self.pans[-1.0], "pan_r": self.pans[1.0]}
        out.update({f"w{u}": b for u, b in self.weights.items()})
        out.update({f"cargo{k}": b for k, b in self.cargos.items()})
        return out

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("k", "side", "quiet_ctr", "a_latch", "b_latch")},
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
            f"A beam balance stands on a table: a steel beam on a centre pivot between two "
            f"dark uprights, a pan HANGING from each end ({c.arm * 200:.0f} cm apart), a heavy "
            f"gray bob under the pivot, and a red needle under the bob sweeping over tick "
            f"marks on the base (white centre tick = level; amber ticks = the "
            f"+-{c.tol_deg:.0f} degree pass band). One pan holds a RED CARGO cylinder; the "
            f"beam rests tipped toward it against its +-{c.tilt_limit_deg:.0f} degree stop. "
            f"On the table lie three BRASS counterweights, all {c.weight_h * 1000:.0f} mm "
            f"tall, telling their masses by diameter: {2 * c.weight_r[0] * 1000:.0f} mm = 1 "
            f"unit, {2 * c.weight_r[1] * 1000:.0f} mm = 2 units, "
            f"{2 * c.weight_r[2] * 1000:.0f} mm = 4 units. The cargo's mass in units equals "
            f"its height divided by {c.cargo_unit_h * 1000:.0f} mm (between {c.k_min} and "
            f"{c.k_max} units, sampled per episode; which pan carries it is also sampled).\n"
            f"Goal: place counterweights summing EXACTLY to the cargo's mass into the EMPTY "
            f"pan — and nothing anywhere else on the scale — so the beam settles level "
            f"(needle inside the amber ticks) with everything at rest, hands off. One unit "
            f"too much or too little leaves the beam resting visibly tipped (~8 degrees) and "
            f"fails. The cargo must stay in its own pan (taking it off does not count as "
            f"balancing); no counterweight may ride in the cargo's pan or sit on the beam "
            f"itself; unused counterweights stay on the table. Either compute the subset "
            f"from the sizes, or watch the needle: the beam always tips toward the heavier "
            f"side."
        )

    def instruction(self) -> str:
        return (
            "Balance the scale: place brass counterweights totalling exactly the red "
            "cargo's mass into the empty hanging pan, leaving the cargo in its pan and "
            "unused weights on the table, so the beam rests level between the amber "
            "ticks. Weights are 1, 2 and 4 units by diameter; the cargo weighs one unit "
            "per 15 mm of height."
        )


# Guarded registration: the forge may import this module under two names.
if "counter_balance" not in SCENES.list():
    SCENES.register("counter_balance", CounterBalanceScene)
if "simgen.counter_balance" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="counter_balance", robot="null"))
