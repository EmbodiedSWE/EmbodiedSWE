"""BalanceVerdictScene — weigh two identical canisters, put the HEAVIER one on the pedestal.

Derived from the LIBERO-plus seed (`libero/native_liberoplus`) but STRATEGICALLY
DIFFERENT (see TASK.md): every LIBERO-plus task is a visually-grounded pick-and-place
under scene perturbations — the target's identity is always GIVEN visually and the
distractors/perturbations are there to be ignored. Here the perturbation IS the task
and cannot be ignored, because it is invisible: two canisters stand on marked pads,
IDENTICAL in size, shape and color, but one is ~3.5x heavier than the other, and which
pad holds the heavy one is sampled per episode. No amount of looking reveals the
target. The scene provides an instrument instead: a two-pan balance comparator (a
free-tilting beam on a pivot with hard stops at +-10 deg). The solver must run a
physical EXPERIMENT — place BOTH canisters on opposite pans at the same time and let
the beam settle tipped (the heavy side drops) — then CONDITION its plan on the
readout: place the canister the scale showed heavier upright on the red pedestal,
leaving the lighter one clear of it. Weighing BEFORE the pedestal placement is a
declared, latched ordering requirement: a lucky unweighed guess earns no success.

Mechanics (plain rigid bodies + one authored D6 joint — the proven pattern): the beam
is ONE dynamic compound body (bar + two pan discs + pan rims, authored per env) hung
on a D6 joint (pillar -> beam) that frees exactly rotX within +-10 deg (physical end
stops). The beam's CoM sits `com_drop` BELOW the pivot, so gravity restores level when
unloaded, while any unequal pan load (>= one canister's weight) overwhelms that
restoring torque and parks the beam on a stop — a comparator, not a measuring scale.
`post_step` owns the canisters' wrench slots (probe buffers for smoke) and matures the
two latches through streak counters: `pan_latch` (a canister rested on a pan) and
`weighed` (both canisters at rest on OPPOSITE pans with the beam settled tipped past
`tip_min_deg`).

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.15 * mean(pan_latch): per-canister, has rested on a pan (latched);
  - 0.25 * weighed: the comparison experiment physically happened (latched);
  - 0.20 * heavy_on_pedestal(): CURRENT state — the heavy canister upright, centered,
    settled on the pedestal top;
  - 0.40 * success(): weighed AND heavy on the pedestal AND the light canister clear
    of the pedestal (>= `clear_r` from its axis) and slow.
  score == 1.0 iff success(); ~0 for the null policy; latched credit never evaporates
  under correct behavior (smoke proves the rejections).

Per-episode randomization (readback-verified in smoke): WHICH pad holds the heavy
canister (the hidden bit), xy jitter on both canisters, xy jitter on the pedestal.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
class BalanceVerdictSceneCfg(BaseCfg):
    """Config for `BalanceVerdictScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tip_min_deg: float = tunable(6.0)  # |beam tilt| beyond this counts as a decisive verdict
    pan_xy_tol: float = tunable(0.045)  # canister within this of a pan centre (beam frame)
    pan_align_max_deg: float = tunable(18.0)  # canister axis within this of the beam's up axis
    ped_xy_tol: float = tunable(0.030)  # heavy canister centre within this of the pedestal axis
    ped_z_tol: float = tunable(0.012)  # ...and its centre height within this of the seated height
    upright_max_deg: float = tunable(15.0)  # heavy canister axis within this of world-up
    clear_r: float = tunable(0.095)  # light canister must be at least this far from the pedestal axis
    settle_lin: float = tunable(0.05)  # max |lin vel| of the heavy canister when judged seated (m/s)
    light_slow: float = tunable(0.15)  # max |lin vel| of the light canister at success (m/s)
    can_slow: float = tunable(0.08)  # canister counts as at-rest on a pan below this (m/s)
    beam_slow: float = tunable(0.30)  # beam counts as settled below this |ang vel| (rad/s)
    latch_steps: int = tunable(24)  # consecutive calm substeps before a latch matures (0.2 s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    pad_jitter: float = tunable(0.020)  # uniform +-xy jitter on each canister around its pad
    ped_jitter: float = tunable(0.025)  # uniform +-xy jitter on the pedestal

    # --- tunable: plant ----------------------------------------------------------------------
    heavy_mass: float = tunable(0.42)  # kg — the hidden property (3.5x the light one)
    light_mass: float = tunable(0.12)
    beam_mass: float = tunable(0.35)
    beam_ang_damp: float = tunable(3.0)  # tames the slam onto the stops
    beam_lin_damp: float = tunable(0.2)
    can_lin_damp: float = tunable(0.05)
    can_ang_damp: float = tunable(0.2)

    # --- info: structure (env-local coordinates; ground z = 0, table top z = 0.40) -----------
    table_center: tuple = info((0.0, 0.0, 0.36))
    table_size: tuple = info((0.90, 1.00, 0.08))  # top at z = 0.40
    scale_xy: tuple = info((0.02, 0.0))  # pillar / pivot axis position on the table
    pillar_size: tuple = info((0.04, 0.04, 0.09))  # kinematic support column (top 0.49)
    pivot_z: float = info(0.53)  # world height of the D6 pivot axis (along x)
    com_drop: float = info(0.025)  # beam origin (=CoM) sits this far BELOW the pivot
    bar_size: tuple = info((0.05, 0.34, 0.02))  # the beam bar (root prim)
    arm: float = info(0.13)  # pan centres at beam-local y = +-arm
    pan_r: float = info(0.055)
    pan_h: float = info(0.008)
    rim_r: float = info(0.049)  # 8 rim blocks around each pan at this radius
    rim_size: tuple = info((0.012, 0.026, 0.018))  # radial x tangential x tall
    tilt_lim_deg: float = info(10.0)  # D6 rotX limit — the physical end stops
    can_r: float = info(0.024)  # canister: 48 mm dia, 100 mm tall — jaw-sized
    can_h: float = info(0.100)
    pads: tuple = info(((-0.20, -0.13), (-0.20, 0.13)))  # marked start pads (index 0 / 1)
    ped_xy: tuple = info((0.06, 0.30))  # pedestal nominal position
    ped_r: float = info(0.05)
    ped_h: float = info(0.05)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    surface_z: float = field(default=None, init=False)  # table top
    beam_z: float = field(default=None, init=False)  # beam origin height when level
    pan_top_local: float = field(default=None, init=False)  # pan top, beam frame
    pan_rest_local: float = field(default=None, init=False)  # canister CENTRE on a pan, beam frame
    ped_top_z: float = field(default=None, init=False)  # pedestal top height

    def __post_init__(self) -> None:
        self.surface_z = self.table_center[2] + self.table_size[2] / 2
        self.beam_z = self.pivot_z - self.com_drop
        self.pan_top_local = self.bar_size[2] / 2 + self.pan_h  # 0.010 + 0.008 = 0.018
        self.pan_rest_local = self.pan_top_local + self.can_h / 2
        self.ped_top_z = self.surface_z + self.ped_h


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----- scene -----------------------------------------------------------------------------------
class BalanceVerdictScene(BaseScene):
    cfg: BalanceVerdictSceneCfg

    def __init__(self, cfg: BalanceVerdictSceneCfg | None = None) -> None:
        super().__init__(cfg or BalanceVerdictSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + pillar + pedestal, the dynamic beam (pans/rims
        authored per env in bind), and the two IDENTICAL-looking canisters (masses differ —
        the hidden property)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))
        charcoal = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.17))
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.46, 0.50))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.10, 0.08))
        blue = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.34, 0.72))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step writes external wrenches (probe buffers) that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live.
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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size, rigid_props=kin, collision_props=coll,
                    visual_material=wood),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center),
            ),
            "pillar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pillar",
                spawn=sim_utils.CuboidCfg(
                    size=c.pillar_size, rigid_props=kin, collision_props=coll,
                    visual_material=charcoal),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.scale_xy[0], c.scale_xy[1], c.surface_z + c.pillar_size[2] / 2)),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CylinderCfg(
                    radius=c.ped_r, height=c.ped_h, axis="Z",
                    rigid_props=kin, collision_props=coll, visual_material=red),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_xy[0], c.ped_xy[1], c.surface_z + c.ped_h / 2)),
            ),
            # --- the beam: root prim = the bar (pans/rims authored per env in bind) ---
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=sim_utils.CuboidCfg(
                    size=c.bar_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=c.beam_lin_damp,
                        angular_damping=c.beam_ang_damp,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
                    collision_props=coll,
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.scale_xy[0], c.scale_xy[1], c.beam_z)),
            ),
        }
        # --- the two canisters: IDENTICAL spawn cfgs except the (invisible) mass ---
        for j, mass in ((0, c.heavy_mass), (1, c.light_mass)):
            out[f"can{j}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Can{j}",
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=c.can_lin_damp,
                        angular_damping=c.can_ang_damp,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=coll,
                    visual_material=blue,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pads[j][0], c.pads[j][1], c.surface_z + c.can_h / 2 + 0.003)),
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
        self.beam: RigidObject = env.iscene["beam"]
        self.cans: list[RigidObject] = [env.iscene[f"can{j}"] for j in (0, 1)]
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.env_origins = env.iscene.env_origins
        self._author_children()
        self._author_d6()
        # Episode state.
        self.heavy_pad = torch.zeros(n, dtype=torch.long, device=dev)  # pad index of can0 (heavy)
        self.pan_latch = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.weighed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pan_streak = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.weigh_streak = torch.zeros(n, dtype=torch.long, device=dev)
        # External probe input (smoke writes; post_step consumes + owns the canisters'
        # wrench slots — never call set_external_force_and_torque directly).
        self.drive_f = torch.zeros(n, 2, 3, device=dev)  # force on canister j (N)

    def _author_children(self) -> None:
        """Compound/visual children, authored idempotently PER ENV: on the beam, a pan
        disc + 8 rim blocks at each end (all collide — the weighing surfaces); on the
        table, two charcoal pad discs (visual start markers)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Beam/panP").IsValid():
            return

        def box(root: str, name: str, center: tuple, size: tuple, color: tuple,
                yaw_deg: float = 0.0, collide: bool = False) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            if yaw_deg:
                xf.AddRotateZOp().Set(yaw_deg)
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                px.CreateContactOffsetAttr(c.contact_offset)
                px.CreateRestOffsetAttr(0.0)

        def disc(root: str, name: str, center: tuple, radius: float, height: float,
                 color: tuple, collide: bool = False) -> None:
            cy = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
            cy.CreateAxisAttr("Z")
            cy.CreateRadiusAttr(radius)
            cy.CreateHeightAttr(height)
            UsdGeom.Xformable(cy.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
            cy.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(cy.GetPrim())
                px = PhysxSchema.PhysxCollisionAPI.Apply(cy.GetPrim())
                px.CreateContactOffsetAttr(c.contact_offset)
                px.CreateRestOffsetAttr(0.0)

        pan_z = c.bar_size[2] / 2 + c.pan_h / 2
        rim_z = c.pan_top_local + c.rim_size[2] / 2
        pale = (0.78, 0.78, 0.80)
        for i in range(self.env.num_envs):
            beam_root = f"/World/envs/env_{i}/Beam"
            for tag, s in (("P", 1.0), ("M", -1.0)):
                disc(beam_root, f"pan{tag}", (0.0, s * c.arm, pan_z),
                     c.pan_r, c.pan_h, pale, collide=True)
                for k in range(8):
                    phi = k * 45.0
                    a = math.radians(phi)
                    box(beam_root, f"rim{tag}{k}",
                        (c.rim_r * math.cos(a), s * c.arm + c.rim_r * math.sin(a), rim_z),
                        c.rim_size, pale, yaw_deg=phi, collide=True)
            table_root = f"/World/envs/env_{i}/Table"
            tz = c.table_size[2] / 2 + 0.002
            for j in (0, 1):
                disc(table_root, f"pad{j}", (c.pads[j][0], c.pads[j][1], tz),
                     0.05, 0.004, (0.20, 0.20, 0.22))

    def _author_d6(self) -> None:
        """Per env: a D6 joint pillar -> beam freeing exactly rotX within +-tilt_lim_deg
        (physical end stops); all translation and the other rotations locked. The pivot
        sits `com_drop` ABOVE the beam origin, so gravity restores level when unloaded
        and any unequal pan load parks the beam on a stop. The joint pair never collides."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        pillar_cz = c.surface_z + c.pillar_size[2] / 2
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.Joint.Define(stage, f"{base}/beam_joint")
            j.CreateBody0Rel().SetTargets([f"{base}/Pillar"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.pivot_z - pillar_cz))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, c.com_drop))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transY", "transZ", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)  # low > high = locked
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "rotX")
            lim.CreateLowAttr(-c.tilt_lim_deg)
            lim.CreateHighAttr(c.tilt_lim_deg)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample which pad gets the heavy canister (the hidden bit), pose
        both canisters on their pads with xy jitter, jitter the pedestal, level the beam,
        clear latches and probe buffers. Uses torch.rand throughout (the first randint
        after manual_seed is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        hp = (torch.rand(m, device=dev) < 0.5).long()  # 0 -> heavy at pad0, 1 -> pad1
        self.heavy_pad[env_ids] = hp
        self.pan_latch[env_ids] = False
        self.weighed[env_ids] = False
        self.pan_streak[env_ids] = 0
        self.weigh_streak[env_ids] = 0
        self.drive_f[env_ids] = 0.0

        pads = torch.tensor(c.pads, device=dev)  # (2, 2)
        for j, can in enumerate(self.cans):  # can0 = heavy, can1 = light
            pad_idx = hp if j == 0 else 1 - hp
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = pads[pad_idx] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
            st[:, 2] = c.surface_z + c.can_h / 2 + 0.003
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            can.write_root_state_to_sim(st, env_ids)

        # pedestal: nominal + jitter (kinematic re-pose)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.ped_xy[0]
        st[:, 1] = c.ped_xy[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.ped_jitter
        st[:, 2] = c.surface_z + c.ped_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pedestal.write_root_state_to_sim(st, env_ids)

        # beam: level, at rest
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.scale_xy[0]
        st[:, 1] = c.scale_xy[1]
        st[:, 2] = c.beam_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def beam_tilt(self) -> torch.Tensor:
        """(N,) beam rotation about +x in rad (the D6 frees only rotX)."""
        q = self.beam.data.root_quat_w
        return _wrap(2.0 * torch.atan2(q[:, 1], q[:, 0]))

    def pan_center_w(self, side: float) -> torch.Tensor:
        """(N, 3) world position of the pan TOP centre on beam-local side +1 / -1."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        loc = torch.tensor([0.0, side * c.arm, c.pan_top_local],
                           device=self.env.device).expand(n, 3)
        return self.beam.data.root_pos_w + quat_apply(self.beam.data.root_quat_w, loc)

    def down_side(self) -> torch.Tensor:
        """(N,) +1.0 if the +y pan is currently the LOW one, else -1.0."""
        zp = self.pan_center_w(+1.0)[:, 2]
        zm = self.pan_center_w(-1.0)[:, 2]
        return torch.where(zp < zm, 1.0, -1.0)

    def _pan_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(on_pan (N,2) bool, side (N,2) float): per canister, is it seated on a pan
        (beam-frame xy near a pan centre, centre height at the seated band, axis aligned
        with the beam's up), and on which beam-local side its centre lies."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        bq = self.beam.data.root_quat_w
        bp = self.beam.data.root_pos_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        beam_up = quat_apply(bq, ez)
        on = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        side = torch.zeros(n, 2, device=dev)
        cos_align = math.cos(math.radians(c.pan_align_max_deg))
        for j, can in enumerate(self.cans):
            loc = quat_apply_inverse(bq, can.data.root_pos_w - bp)
            side[:, j] = torch.where(loc[:, 1] >= 0, 1.0, -1.0)
            near = (loc[:, 0].abs() < c.pan_xy_tol) \
                & ((loc[:, 1].abs() - c.arm).abs() < c.pan_xy_tol)
            z_ok = (loc[:, 2] > c.pan_rest_local - 0.013) \
                & (loc[:, 2] < c.pan_rest_local + 0.012)
            can_up = quat_apply(can.data.root_quat_w, ez)
            aligned = (can_up * beam_up).sum(-1) > cos_align
            on[:, j] = near & z_ok & aligned
        return on, side

    def heavy_on_pedestal(self) -> torch.Tensor:
        """(N,) bool: the HEAVY canister (can0) upright, centered on the pedestal top at
        the seated height, and settled — a physical, current-state judgement."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        p = self.cans[0].data.root_pos_w
        ped = self.pedestal.data.root_pos_w
        xy_ok = (p[:, :2] - ped[:, :2]).norm(dim=-1) < c.ped_xy_tol
        seat_z = ped[:, 2] + c.ped_h / 2 + c.can_h / 2
        z_ok = (p[:, 2] - seat_z).abs() < c.ped_z_tol
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.cans[0].data.root_quat_w, ez)
        upright = up[:, 2] > math.cos(math.radians(c.upright_max_deg))
        still = self.cans[0].data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return xy_ok & z_ok & upright & still

    def light_clear(self) -> torch.Tensor:
        """(N,) bool: the LIGHT canister (can1) at least `clear_r` from the pedestal axis."""
        p = self.cans[1].data.root_pos_w
        ped = self.pedestal.data.root_pos_w
        return (p[:, :2] - ped[:, :2]).norm(dim=-1) > self.cfg.clear_r

    def success(self) -> torch.Tensor:
        """(N,) bool: the weighing experiment happened (latched), the heavy canister is
        seated on the pedestal NOW, and the light canister is clear of it and slow."""
        light_ok = self.cans[1].data.root_lin_vel_w.norm(dim=-1) < self.cfg.light_slow
        return self.weighed & self.heavy_on_pedestal() & self.light_clear() & light_ok

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 * mean(pan_latch) + 0.25 * weighed
        + 0.20 * heavy_on_pedestal + 0.40 * success. ~0 for the null policy; 1.0 iff
        success(); the latched shares never evaporate under correct behavior."""
        return 0.15 * self.pan_latch.float().mean(dim=1) \
            + 0.25 * self.weighed.float() \
            + 0.20 * self.heavy_on_pedestal().float() \
            + 0.40 * self.success().float()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Consume the probe-force buffers (owns the canisters' wrench slots), then mature
        the latches through streak counters: `pan_latch` (canister at rest on a pan) and
        `weighed` (both canisters at rest on OPPOSITE pans, beam settled past tip_min)."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        zt = torch.zeros(n, 1, 3, device=dev)
        for j, can in enumerate(self.cans):
            can.set_external_force_and_torque(self.drive_f[:, j].unsqueeze(1), zt)

        on, side = self._pan_metrics()
        slow = torch.stack(
            [can.data.root_lin_vel_w.norm(dim=-1) < c.can_slow for can in self.cans], dim=1)
        rest = on & slow
        self.pan_streak = torch.where(rest, self.pan_streak + 1,
                                      torch.zeros_like(self.pan_streak))
        self.pan_latch |= self.pan_streak >= c.latch_steps

        tipped = self.beam_tilt().abs() > math.radians(c.tip_min_deg)
        beam_calm = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_slow
        w_ok = rest.all(dim=1) & (side[:, 0] * side[:, 1] < 0) & tipped & beam_calm
        self.weigh_streak = torch.where(w_ok, self.weigh_streak + 1,
                                        torch.zeros_like(self.weigh_streak))
        self.weighed |= self.weigh_streak >= c.latch_steps

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("heavy_pad", "pan_latch", "weighed", "pan_streak",
                               "weigh_streak", "drive_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"beam": self.beam, "can0": self.cans[0], "can1": self.cans[1],
                "pedestal": self.pedestal}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden table carries a two-pan BALANCE SCALE at its centre: a charcoal "
            f"pillar supporting a steel beam that tilts freely left/right between hard "
            f"stops at +-{c.tilt_lim_deg:.0f} degrees, with a shallow rimmed pan at each "
            f"end ({c.pan_r * 200:.0f} cm across, {2 * c.arm * 100:.0f} cm apart). The beam "
            f"rests LEVEL when the pans carry equal weight and tips DOWN on the side "
            f"carrying more weight. On two dark round pads on the near side stand two "
            f"BLUE CANISTERS, visually IDENTICAL ({2 * c.can_r * 100:.1f} cm across, "
            f"{c.can_h * 100:.0f} cm tall) — but one is roughly 3.5 times heavier than the "
            f"other, and which pad holds the heavy one changes every episode. You CANNOT "
            f"tell them apart by looking; the balance scale is the only way to find out. "
            f"A short RED PEDESTAL ({c.ped_r * 200:.0f} cm across) stands to one side.\n"
            f"Goal, in this required order: FIRST weigh the canisters — place BOTH of them "
            f"on the two pans at the same time (one per pan) and let the beam settle "
            f"clearly tipped; the pan that sinks holds the HEAVIER canister. THEN place "
            f"that heavier canister upright on top of the red pedestal, centered within "
            f"about {c.ped_xy_tol * 100:.0f} cm of its middle, and leave the lighter "
            f"canister anywhere at least {c.clear_r * 100:.0f} cm away from the pedestal "
            f"(leaving it on its pan is fine). Hands off at the end, everything at rest. "
            f"Putting a canister on the pedestal WITHOUT having weighed both first does "
            f"not count, even if you guessed the heavy one; putting the lighter canister "
            f"on the pedestal, or leaving it against the pedestal, fails."
        )

    def instruction(self) -> str:
        return (
            "Weigh the two identical blue canisters by setting both on the balance "
            "scale's pans at the same time and letting the beam settle tipped; then "
            "place the heavier one (the sunken pan's canister) upright on the red "
            "pedestal, keeping the lighter one at least 10 cm away. Weighing must come "
            "first — an unweighed guess or the wrong canister fails."
        )


# Guarded registration: the forge may import this module under two names.
if "balance_verdict" not in SCENES.list():
    SCENES.register("balance_verdict", BalanceVerdictScene)
if "simgen.balance_verdict" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="balance_verdict", robot="null"))
