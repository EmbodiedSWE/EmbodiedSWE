"""ToolPackingScene — stow three tools into an articulated three-drawer toolbox, then shut it.

The refined successor of the pen-holder floor task, on real scanned assets: a plastic parts
cabinet (two hinged translucent doors in front of three sliding drawers) stands on the work
table with a stapler, a pair of scissors and a utility knife scattered in front of it.
**Goal (carried here, no task layer): open the box, put the stapler in the TOP drawer, the
scissors in the MIDDLE drawer and the knife in the BOTTOM drawer, close every drawer and
both doors.**

Assets are vendored PartNet-mobility-style scans (`suites/packing/assets/{toolbox,stapler,
scissors,knife}`, built by `scripts/vendor_tool_packing_assets.py` in cap-x):
  - toolbox: the AUTHORED articulation is kept — 2 revolute doors (|limit| 130 deg, they
    cover the drawer faces: drawers cannot open before the doors) + 3 prismatic drawers
    (0.25 m travel), SDF mesh collision (the trays are genuinely concave and hold what is
    put in them), drives freed in vendoring (stiffness zeroed, authored damping kept, so a
    released drawer coasts to a stop instead of springing shut). The base link is FIXED to
    the world — a cabinet that slides around while its drawers are pulled is a later
    difficulty variant (`fix_base=False`), not the floor tier.
  - items: each flattened in vendoring to ONE rigid body frozen at rest pose (the scans are
    articulated — stapler hinge, scissors pivot, retractable blade — but loose floppy items
    would tax grasping, settle time and memory (per-link SDF), all off-goal here); collision
    is convexDecomposition. The stapler spawns scaled `stapler_scale` (0.85): at full size
    it is 81 mm tall against a ~75 mm drawer bay — a real stapler that does not fit the real
    box is an asset mismatch, not a task.

Judged in each DRAWER'S BODY FRAME (the pen-holder lesson: a moving container judges
identically wherever it is): an item is STOWED in its assigned drawer when its origin lies
inside that drawer's tray-interior box (`tray_center` +/- `tray_half`, measured from the
vendored asset), whatever the drawer's slide position. A drawer is CLOSED when its joint is
within `drawer_closed_tol` of 0 (limits [-0.25, 0], 0 = shut), a door when within
`door_closed_deg` of 0. Score: 20 per stowed item, +10 per stowed item whose drawer is shut,
+10 once everything is stowed and shut (doors included) -> 0..100. `success()` = full score,
settled. Items must go through the real openings — the box interior is only reachable past
the doors and out-slid drawers; nothing is welded, ordering emerges from the geometry.

Per-episode randomization (task-family knobs): item scatter poses (slot row + xy jitter +
free yaw, lying flat) and optional item->slot permutation, so a memorized fixed pick order
fails. The box pose gets xy jitter + yaw jitter around its nominal.

Embodiment-agnostic: items and joints are scene objects the robot reaches through
`env.scene`; a NullRobot smoke drives drawers/doors by joint writes and items by kinematic
staging. Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from robobench.core import BaseEnv


@dataclass
class ToolPackingSceneCfg(BaseCfg):
    """Config for `ToolPackingScene`. Each field is a `tunable()` curriculum/difficulty dial
    or an `info()` structural constant (see `robobench.core.BaseCfg`)."""

    # --- tunable: rubric thresholds ---------------------------------------------------------
    drawer_closed_tol: float = tunable(0.015)  # drawer joint within this of 0 = closed (m)
    door_closed_deg: float = tunable(6.0)  # door joint within this of 0 = closed (deg)
    settle_speed: float = tunable(0.05)  # max item |v| when judging (m/s)
    settle_joint_speed: float = tunable(0.05)  # max |drawer v| (m/s) / |door w| (rad/s) x10
    reset_pos_jitter: float = tunable(0.02)  # uniform +/- xy jitter per item at reset (m)
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per item at reset
    box_pos_jitter: float = tunable(0.02)  # uniform +/- xy jitter of the toolbox at reset (m)
    box_yaw_deg: float = tunable(10.0)  # uniform +/- yaw of the toolbox at reset
    shuffle_slots: bool = tunable(True)  # per-episode random item->scatter-slot permutation
    fix_base: bool = tunable(True)  # False: the cabinet is free-standing (difficulty variant)

    # --- tunable: placement (table-relative xy; the lab table itself sits at TABLES pos) -------
    box_pos: tuple = tunable((0.12, 0.0))  # toolbox centre on the table
    box_yaw_deg_nominal: float = tunable(-90.0)  # doors/drawers face local -y; -90 -> toward -x
    items_center: tuple = tunable((-0.52, 0.0))  # scatter row centre — OUTSIDE the doors'
    # swing sweep WITH margins: the door edge reaches box_centre_x - 0.465 (half-depth
    # 0.245 + panel 0.22), worst-case -0.365 under the +/-10 deg box yaw and 2 cm box
    # jitter; an item's own reach from its slot is up to 0.132 (scissors half-length
    # + slot jitter), so the row centre needs x <= -0.50 (-0.28 put scattered items
    # INSIDE the arc — the sweeping door shoved a knife through the -1.5 mm gate)
    item_spacing: float = tunable(0.16)  # y gap between scatter slots
    # Explicit per-item scatter slots (table-relative xy, manifest order), overriding the
    # items_center row. Robot bindings need this: the doors sweep the whole front strip when
    # they open, so items must start on an arc OUTSIDE the sweep yet inside reach.
    item_slots: tuple | None = tunable(None)

    # --- info: the assignment (item -> drawer), asset paths, structure ------------------------
    # manifest: (item name, assigned drawer, mass kg, spawn scale, rest z-lift m).
    # Stapler at 0.55: the bays' REAL interior height above the raised tray floor is
    # ~4.9 cm (top bay) — the scan's full-size 8.1 cm stapler cannot close inside any
    # drawer, so it ships as a pocket stapler (4.5 cm tall flat, ~4 mm closing margin).
    manifest: tuple = info((
        ("stapler", "top", 0.10, 0.55, 0.005),
        ("scissors", "mid", 0.05, 1.0, 0.006),
        ("knife", "bot", 0.09, 1.0, 0.004),
    ))
    drawers: tuple = info(("top", "mid", "bot"))  # rubric order; joints joint_drawer_<k>
    # Drawer-frame tray interior (vendoring script measurement on the BAKED asset, xy inset
    # 15 mm for the walls): an item origin inside this box is IN the tray. z spans the
    # MEASURED inner floor plane (+0.004 — the tray floor is raised above its outer
    # bottom) up past the rim.
    tray_center: tuple = info((0.0, 0.137, 0.028))
    tray_half: tuple = info((0.198, 0.127, 0.026))
    drawer_travel: float = info(0.25)  # prismatic limits [-travel, 0]; 0 = shut
    door_limits_deg: tuple = info((-130.0, 130.0))  # left opens negative, right positive
    box_body: str = info("E_bodyM1_10")  # articulation link names in the vendored USD
    drawer_bodies: tuple = info(("E_drawer_1_7", "E_drawer_2_4", "E_drawer_3_1"))  # top/mid/bot
    door_joints: tuple = info(("joint_door_left", "joint_door_right"))
    contact_offset: float = info(0.003)  # items: the closed-drawer stapler headroom is
    # ~5 mm — the ~2 cm default would press phantom contact through every shut drawer
    box_contact_offset: float = info(0.001)  # toolbox links: the cabinet's INTERNAL design
    # gaps (drawer-to-shell, door-to-corner) are 1-2 mm, so with self-collision enabled a
    # 3 mm speculative margin would hold every closed drawer in permanent phantom contact
    light_intensity: float = info(2500.0)
    # Selectable work surface (same presets as the assembly scenes). Default = the
    # general-purpose packing table (the ikea/microwave bench; the lab table is an
    # industrial GPU-assembly bench and stays available as a preset).
    table: str = info("packing")
    table_depth_scale: float = info(1.5)  # y-stretch: the packing top is 2.47 x 0.76 m,
    # and toolbox + scatter row + an on-table robot base need ~1 m of depth (the
    # microwave task's deepening, adopted with it)
    surface_z: float | None = info(None)
    workbench_pos: tuple[float, float] | None = info(None)
    workbench_usd: str = info("")
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.5, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    asset_dir: str = info("")
    toolbox_usd: str = info("")
    item_usds: dict = field(default=None, init=False)

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        self.toolbox_usd = self.toolbox_usd or str(Path(self.asset_dir) / "toolbox" / "toolbox.usd")
        self.item_usds = {name: str(Path(self.asset_dir) / name / f"{name}.usd")
                          for name, _d, _m, _s, _z in self.manifest}
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(
            assets.parents[1] / "assembly" / "assets" / "props" / preset["usd"][0] / preset["usd"][1])


@SCENES.register("tool_packing")
class ToolPackingScene(BaseScene):
    cfg: ToolPackingSceneCfg

    def __init__(self, cfg: ToolPackingSceneCfg | None = None) -> None:
        super().__init__(cfg or ToolPackingSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, table, the articulated toolbox (fixed base, doors/drawers shut),
        and the three loose items in a row in front of it."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.toolbox_usd, *c.item_usds.values()):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — vendor the tool_packing assets first "
                    f"(scripts/vendor_tool_packing_assets.py)")
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = c.surface_z - preset["top_offset"]
        ground_z = c.surface_z - preset["height"]
        s = preset["scale"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd,
                                           scale=(s, s * c.table_depth_scale, s))
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        half = math.radians(c.box_yaw_deg_nominal) / 2

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(c.asset_dir).parents[1] / "assembly" / "assets" / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z), rot=preset["orient"]),
                spawn=table_spawn,
            ),
            "toolbox": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Toolbox",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.toolbox_usd,
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=True, fix_root_link=c.fix_base,
                        # PhysX articulations do NOT collide with their own links unless
                        # asked: with this off (the silent default), doors/drawers/shell
                        # passed freely through each other — every penetration incident
                        # traced back here. The vendored asset filters the one DESIGNED
                        # mm-clearance pair (door<->door).
                        enabled_self_collisions=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.box_contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(wx + c.box_pos[0], wy + c.box_pos[1], c.surface_z),
                    rot=(math.cos(half), 0.0, 0.0, math.sin(half)),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                # Explicit passive-joint physics (implicit actuators, target 0 = "no
                # motor"). The scan's authored angular drives are per-DEGREE and load
                # per-RADIAN (x57.3): the doors came up as SPRING-LOADED SELF-CLOSING
                # hinges (stiffness 1.15 N*m/rad, damping 114.6 — holding one at 88 deg
                # takes 1.8 N*m, far beyond a fingertip crack; measured live, driver
                # jobs 0102-0106). Vertical hinges need no spring to hold an angle, so:
                # zero stiffness, light viscosity, real PhysX joint FRICTION for
                # stay-where-put (doors 0.02 N*m — a ~2 N fingertip push at the handle
                # overcomes it; drawers 0.3 N — kills the residual self-close creep).
                actuators={
                    # doors: DAMPED HINGES (soft-close cabinet regime, user 2026-08-06:
                    # "the door just stays wherever it is") — heavy damping absorbs
                    # knocks (a 2 N*m graze for 0.1 s moves <4 deg, then the door
                    # stops dead) while a deliberate sustained push still opens at
                    # ~0.6 rad/s; friction (written in reset()) holds it parked.
                    # damping 1.5: at 3.0 the door's velocity ceiling (torque/damping)
                    # fell below the opening stream's rate and the pusher outran the
                    # handle's 20 mm depth (door stalled ~25 deg); 1.5 still kills a
                    # knock within ~7 deg while a sustained push opens at ~1.5 rad/s
                    "doors": ImplicitActuatorCfg(
                        joint_names_expr=["joint_door_.*"],
                        stiffness=0.0, damping=1.5, friction=0.3),
                    "drawers": ImplicitActuatorCfg(
                        joint_names_expr=["joint_drawer_.*"],
                        stiffness=0.0, damping=2.0, friction=0.3),
                },
            ),
        }
        for i, (name, _drawer, mass, scale, zlift) in enumerate(c.manifest):
            sx, sy = self._slot_xy(i)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.item_usds[name],
                    scale=(scale,) * 3,
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.05,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + sx, wy + sy, c.surface_z + zlift)),
            )
        return out

    def _slot_xy(self, i: int) -> tuple[float, float]:
        """Table-relative xy of scatter slot `i`: explicit `item_slots` when given, else the
        y-row around `items_center`."""
        c = self.cfg
        if c.item_slots is not None:
            return tuple(c.item_slots[i])
        n = len(c.manifest)
        return (c.items_center[0], c.items_center[1] + (i - (n - 1) / 2) * c.item_spacing)

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            # the door panels are tinted glass — without translucency they render invisible
            render={"enable_translucency": True},
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab the articulation + item handles and resolve joint/body indices by name."""
        super().bind(env)
        c = self.cfg
        self.box: Articulation = env.iscene["toolbox"]
        self.items: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _d, _m, _s, _z in c.manifest}
        self.env_origins = env.iscene.env_origins
        drawer_joints = [f"joint_drawer_{k}" for k in c.drawers]
        self._drawer_j, _ = self.box.find_joints(drawer_joints, preserve_order=True)
        self._door_j, _ = self.box.find_joints(list(c.door_joints), preserve_order=True)
        self._drawer_b = [self.box.body_names.index(n) for n in c.drawer_bodies]
        # item -> assigned drawer column (rubric order = manifest order)
        self._assigned = [c.drawers.index(d) for _n, d, _m, _s, _z in c.manifest]
        self._tray_c = torch.tensor(c.tray_center, device=env.device)
        self._tray_h = torch.tensor(c.tray_half, device=env.device)
        # (joint friction is written in reset(), not here: a bind-time write gets
        # clobbered by actuator initialization on fresh boots — measured, jobs 0123/0127)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: toolbox shut (all joints 0) at its jittered pose, items scattered
        lying flat on their (optionally permuted) row slots."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        wx, wy = c.workbench_pos

        # Joint friction, RAW physx view (CPU tensors are the view's contract), written
        # HERE because every other home fails: the ImplicitActuatorCfg `friction` and
        # write_joint_friction_coefficient_to_sim never reach PhysX (live coefficients
        # 0.0 — jobs 0106/0123), and a bind-time raw write is clobbered by actuator init
        # on fresh boots (job 0127: a door parked at 20 deg fell shut). Idempotent, so
        # per-reset is fine.
        # Doors 0.3 N*m friction + 3.0 damping (the actuator cfg): a DAMPED soft-close
        # hinge — the door moves only while deliberately pushed and stays wherever it
        # is released (user 2026-08-06). Friction holds it parked; damping eats knock
        # impulses (an arm graze moves it a few degrees at most, then it stops dead).
        fr = torch.zeros(self.box.num_instances, self.box.num_joints, device="cpu")
        for j in self._door_j:
            fr[:, j] = 0.3
        for j in self._drawer_j:
            fr[:, j] = 0.3
        self.box.root_physx_view.set_dof_friction_coefficients(
            fr, torch.arange(self.box.num_instances, device="cpu"))

        # --- toolbox: joints shut; root at nominal pose + jitter (root write also covers
        # fix_base=False, where the cabinet may have been shoved around last episode) ---
        half = (math.radians(c.box_yaw_deg_nominal)
                + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.box_yaw_deg)) / 2
        root = torch.zeros(m, 13, device=dev)
        root[:, 0] = wx + c.box_pos[0]
        root[:, 1] = wy + c.box_pos[1]
        root[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.box_pos_jitter
        root[:, 2] = c.surface_z
        root[:, 3] = torch.cos(half)
        root[:, 6] = torch.sin(half)
        root[:, 0:3] += origin
        self.box.write_root_pose_to_sim(root[:, 0:7], env_ids)
        self.box.write_root_velocity_to_sim(torch.zeros(m, 6, device=dev), env_ids)
        nj = self.box.num_joints
        self.box.write_joint_state_to_sim(
            torch.zeros(m, nj, device=dev), torch.zeros(m, nj, device=dev), env_ids=env_ids)

        # --- items: scatter slots (optionally permuted) + jitter + free yaw, lying flat ---
        n_items = len(c.manifest)
        if c.shuffle_slots:
            perm = torch.rand(m, n_items, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(n_items, device=dev).expand(m, n_items)
        slots = torch.tensor([self._slot_xy(i) for i in range(n_items)], device=dev)  # (I, 2)
        yaw_amp = math.radians(c.reset_yaw_deg)
        for i, (name, _d, _mass, _s, zlift) in enumerate(c.manifest):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = wx + slots[perm[:, i], 0]
            st[:, 1] = wy + slots[perm[:, i], 1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = c.surface_z + zlift
            h = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(h)
            st[:, 6] = torch.sin(h)
            st[:, 0:3] += origin
            self.items[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "box_root": self.box.data.root_state_w[env_ids].clone(),
            "box_joint_pos": self.box.data.joint_pos[env_ids].clone(),
            "box_joint_vel": self.box.data.joint_vel[env_ids].clone(),
            "items": {n: b.data.root_state_w[env_ids].clone() for n, b in self.items.items()},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.box.write_root_pose_to_sim(state["box_root"][:, 0:7], env_ids)
        self.box.write_root_velocity_to_sim(state["box_root"][:, 7:13], env_ids)
        self.box.write_joint_state_to_sim(
            state["box_joint_pos"], state["box_joint_vel"], env_ids=env_ids)
        for n, b in self.items.items():
            b.write_root_state_to_sim(state["items"][n], env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A plastic parts cabinet (a small toolbox, ~46 cm wide) stands on the table: two "
            "translucent hinged doors in front, and behind them three sliding drawers stacked "
            "top / middle / bottom (each ~44 x 28 cm inside, pulling out toward the doors). "
            "All doors and drawers start shut. In front of the box lie three tools: a green "
            "stapler, a pair of scissors, and a utility knife.\n"
            "Goal: open the doors, slide out drawers as needed, and pack the tools — the "
            "STAPLER goes in the TOP drawer, the SCISSORS in the MIDDLE drawer, the KNIFE in "
            "the BOTTOM drawer — then push every drawer shut and close both doors. An item "
            "counts only inside its own drawer's tray; the job is done when all three tools "
            "are stowed and the box is fully shut."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def _item_in_tray(self) -> torch.Tensor:
        """(N, I) bool: each item's origin inside its ASSIGNED drawer's tray-interior box,
        computed in that drawer's body frame (drawer slide position is irrelevant)."""
        from isaaclab.utils.math import quat_apply_inverse

        pos = self.box.data.body_pos_w  # (N, B, 3)
        quat = self.box.data.body_quat_w  # (N, B, 4)
        cols = []
        for i, (name, _d, _m, _s, _z) in enumerate(self.cfg.manifest):
            b = self._drawer_b[self._assigned[i]]
            loc = quat_apply_inverse(quat[:, b], self.items[name].data.root_pos_w - pos[:, b])
            cols.append(((loc - self._tray_c).abs() <= self._tray_h).all(dim=-1))
        return torch.stack(cols, dim=1)

    def drawer_pos(self) -> torch.Tensor:
        """(N, 3) drawer joint positions, rubric order (0 = shut, -travel = fully out)."""
        return self.box.data.joint_pos[:, self._drawer_j]

    def door_pos(self) -> torch.Tensor:
        """(N, 2) door joint angles in rad (0 = shut; left opens negative, right positive)."""
        return self.box.data.joint_pos[:, self._door_j]

    def stowed(self) -> torch.Tensor:
        """(N, I) bool: item settled inside its assigned drawer's tray."""
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.items.values()], dim=1)
        return self._item_in_tray() & (vel < self.cfg.settle_speed)

    def drawers_closed(self) -> torch.Tensor:
        """(N, 3) bool: each drawer within `drawer_closed_tol` of shut."""
        return self.drawer_pos() > -self.cfg.drawer_closed_tol

    def doors_closed(self) -> torch.Tensor:
        """(N,) bool: both doors within `door_closed_deg` of shut."""
        return (self.door_pos().abs() <= math.radians(self.cfg.door_closed_deg)).all(dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: items and box joints all quiet."""
        item_v = torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in self.items.values()], dim=1)
        jv = self.box.data.joint_vel
        drawers_still = jv[:, self._drawer_j].abs().amax(dim=1) < self.cfg.settle_joint_speed
        doors_still = jv[:, self._door_j].abs().amax(dim=1) < self.cfg.settle_joint_speed * 10
        return (item_v < self.cfg.settle_speed).all(dim=1) & drawers_still & doors_still

    def score(self) -> torch.Tensor:
        """(N,) int 0..100: 20 per stowed item, +10 per stowed item whose drawer is shut,
        +10 once all three are stowed AND every drawer and both doors are shut."""
        stowed = self.stowed()  # (N, I)
        closed = self.drawers_closed()  # (N, 3)
        per_item_closed = closed[:, self._assigned]  # (N, I)
        base = 20 * stowed.sum(dim=1) + 10 * (stowed & per_item_closed).sum(dim=1)
        all_shut = stowed.all(dim=1) & closed.all(dim=1) & self.doors_closed()
        return base + torch.where(all_shut, 10, 0)

    def success(self) -> torch.Tensor:
        """(N,) bool: all items stowed in their assigned drawers, box fully shut, settled
        (scene-level success; the oracle's target)."""
        return (self.stowed().all(dim=1) & self.drawers_closed().all(dim=1)
                & self.doors_closed() & self.settled())
