"""SO101AssemblyScene — assemble the SO101 arm: seat + screw the elbow servo, attach the distal.

The assembly object-world:

  - `proximal`: base + shoulder + upper-arm shell as a FLOATING-base articulation (joints 1-2
    live).
  - `motor`: the bare elbow STS3215 as a free rigid body. KEY FRAME FACT: its body frame IS the
    upper_arm link frame, so "seated in the pocket" = identical body poses. The pocket is a
    slip fit — a straight, centered push inserts it.
  - `screw_0..3`: four IDENTICAL slim-head M2x6, free rigid bodies — the servo's four tab
    screws. Any screw fastens into any free hole (the pairing is order-independent).
  - `drill`: the compact power screwdriver articulation (body/trigger/bit).
  - `distal`: lower_arm..gripper as a floating articulation. Its root link `lower_arm` is a
    closed fork that clips over the seated servo — the drive-side cup onto the output horn,
    the far plate onto the back bearing post — and is locked by M3 screws (`horn_screw_0..3`)
    on the four peripheral screw lines around the elbow axis: near side through the fork's
    inner plate into the horn's metal holes (reached through the outer skin's access
    channels), far side through the far plate into the servo's case-back holes. The center
    bore on the axis is only driver access. The seated lower_arm pose relative to the motor
    is the elbow_flex joint transform at joint zero.

The servo is held by four M2 screws (upper_arm LINK frame): the NEAR pair enters the countersunk
outer wall from link -Z; the FAR pair enters from link +Z through the ring bosses on that face.
Each hole carries its own seat pose; the out-of-hole axis is derived from it, so both facings
run the same mechanic.

Everything rests on a WORKBENCH (a static table under the whole layout; its top is at
`surface_z` — by default the ikea_table scene's packing table standing on the floor, top at
0.994 m). Every body is free-floating; the only pre-authored joints are the DISABLED fastening
welds (the mechanic below).

THE FASTENING MECHANIC (rule-based for fast simulation — see `_fasten_rule`): one pre-authored
DISABLED FixedJoint per screw, its seat frame authored at enable time (so any screw can take any
hole in its own group); a per-step gate (screw in a free hole + parts aligned + bit on that
screw + trigger on) advances the screw kinematically with a latched depth and snaps its weld on
at the seat. The holes come in per-joint GROUPS, each with its own seat link and part-alignment
check: any fastened elbow tab screw welds motor<->upper_arm; the fastened horn screws CLOSE THE
DRIVEN ELBOW JOINT (a revolute about the horn axis with the servo's drive — the assembled robot
articulates; command it via set_elbow_target). THE MAGNETIC BIT (same section): a free screw whose head
touches the bit tip attaches and rides it, coaxial and spinning, until driven home — real M2
driving carries the screw on a magnetized bit, and any embodiment holding the drill can use it.
Everything else is real collision against the parts' actual holes and walls.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from robobench.core import BaseEnv


@dataclass
class SO101SceneCfg(BaseCfg):
    """Config for `SO101AssemblyScene`. Nothing is locked — a variant is just a copy with a few fields changed."""

    # --- the fastening gate + drive dials (see _fasten_rule for THE RULE) -----------------
    drive_rate: float = 0.020  # screw advance speed while driving (m/s)
    min_drive_s: float = 0.25  # minimum accumulated drive time before the weld engages (s)
    gate_axis_deg: float = 15.0  # max screw-vs-hole axis misalignment (deg)
    gate_radial: float = 0.0025  # max head-center offset from the hole axis (m)
    gate_window: float = 0.014  # engagement window above the seat, along the axis (m)
    gate_window_below: float = 0.008  # engagement window BELOW the seat (m): a screw
    # that slid deep into its hole still drives. Below the seat the bore itself constrains the
    # screw, so the axis check is waived there — inside the hole, messy is acceptable; only
    # what happens above the surface must be precise.
    motor_align_pos: float = 0.004  # max servo-vs-pocket position error to fasten (m)
    motor_align_deg: float = 10.0  # max servo-vs-pocket orientation error to fasten (deg)
    bit_on_head: float = 0.004  # bit tip -> screw head-top distance for the gate (m)
    bit_axis_deg: float = 30.0  # max bit-vs-screw axis misalignment (deg)
    spin_min: float = 3.0  # bit speed that counts as "spinning" (rad/s)
    weld_snap: float = 0.005  # fastened part farther than this off its weld frame snaps back (m)

    # --- the drill (powered screwdriver) -----------------------------------------------------
    bit_speed: float = 15.0  # bit spin speed while the trigger is squeezed (rad/s)
    trigger_swing: float = math.radians(14.0)  # trigger travel, rest -> full squeeze (rad)
    bit_tip: tuple[float, float, float] = (0.0, 0.055, 0.0)  # bit tip point, in the bit's own link frame

    # --- fastening welds — where each screw seats in the upper_arm (LINK frame) --------------
    # Per-joint convention: each joint's screw(s) carry its <joint>_ prefix; adding wrist_*/shoulder_*
    # later is additive. Pose-agnostic — a screw seats relative to its link, wherever the arm is.
    # One loose screw is spawned per hole (identical screws — any screw may take any hole).
    # Each seat is the driven head-top, placed so the seated screw rests contact-free (the weld
    # holds it): the near heads sit ~2 mm proud of the countersunk wall, the far heads sit
    # inside the ring bosses with the tips just above the servo's far tab.
    elbow_screw_seat_pts: tuple[tuple[float, float, float], ...] = (
        (-0.1227, 0.0010, -0.0040),  # near pair: the countersunk holes in the -Z wall
        (-0.1022, 0.0010, -0.0040),
        (-0.1228, 0.0052, 0.0428),   # far pair: the ring-boss holes through the +Z wall
        (-0.1023, 0.0052, 0.0428),
    )
    elbow_screw_seat_quats: tuple[tuple[float, float, float, float], ...] = (
        (0.0, 0.0, 1.0, 0.0),  # near: screw +Z (out of the head) -> link -Z
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),  # far: screw +Z -> link +Z
        (1.0, 0.0, 0.0, 0.0),
    )
    # free spawn xy of each loose screw (env frame, resting on the workbench top; z from the asset)
    screw_spawn_pts: tuple[tuple[float, float], ...] = (
        (0.25, -0.15), (0.31, -0.15), (0.25, -0.22), (0.31, -0.22))

    # --- the elbow HORN fastening — the lower_arm clips onto the motor's output horn ---
    # Seated lower_arm pose in the MOTOR (upper_arm-link) frame: the elbow_flex joint transform
    # at joint zero. The lower_arm origin sits ON the elbow axis, so this pose is also where the
    # pre-authored elbow JOINT is framed.
    elbow_lower_arm_seat_pos: tuple[float, float, float] = (-0.11257, -0.028, 0.0)
    elbow_lower_arm_seat_quat: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, 0.7071068)
    # Fastening the horn screws doesn't weld the forearm rigid — it closes the REAL elbow
    # joint: a revolute about the horn axis, driven with the URDF elbow_flex servo drive.
    # Enabled while any horn screw is fastened; command it via set_elbow_target().
    elbow_joint_stiffness: float = 5.2859  # USD angular drive units (per-degree); the
    # URDF elbow_flex servo stiffness
    elbow_joint_damping: float = 0.025  # near-critical for the forearm about this axis —
    # the URDF's 0.0021 is tuned for the implicit articulation solver and leaves this LOOSE
    # joint ~10x underdamped (the forearm rings as a pendulum)
    elbow_joint_max_force: float = 10.0
    elbow_joint_limit_deg: float = 96.83
    # The M3 horn screws' seats (driven head-top poses), in the LOWER_ARM link frame: the FOUR
    # peripheral screw lines around the elbow axis, on EACH side of the fork (the center bore is
    # only driver access). NEAR (horn) side, holes 0-3: the head seats on the fork's inner plate
    # — reached through the outer skin's access channels — and the shaft threads into the horn's
    # metal holes; out-of-hole is lower_arm -Z. FAR side, holes 4-7: the head seats on the far
    # plate's outer face, threading into the servo's case-back holes; out-of-hole is +Z.
    elbow_horn_screw_seat_pts: tuple[tuple[float, float, float], ...] = (
        (0.00497, -0.00497, -0.0066), (-0.00495, -0.00498, -0.0066),   # near (horn) side
        (0.00498, 0.00495, -0.0066), (-0.00490, 0.00495, -0.0066),
        (0.00497, -0.00497, 0.0431), (-0.00495, -0.00498, 0.0431),     # far (case-back) side
        (0.00498, 0.00495, 0.0431), (-0.00490, 0.00495, 0.0431))
    elbow_horn_screw_seat_quats: tuple[tuple[float, float, float, float], ...] = (
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0),    # near: screw +Z (out of head) -> link -Z
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0),    # far: screw +Z -> link +Z
        (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    # eight loose M3s spawn (one per horn-line hole, near + far)
    horn_screw_spawn_pts: tuple[tuple[float, float], ...] = (
        (0.37, -0.15), (0.43, -0.15), (0.37, -0.22), (0.43, -0.22),
        (0.49, -0.15), (0.55, -0.15), (0.49, -0.22), (0.55, -0.22))

    # --- scene assets ------------------------------------------------------------------------
    light_intensity: float = 2500.0
    # Selectable work surface (the same vendored presets as bulb/nut_thread). All spawn poses are
    # env-frame xy with heights ABOVE the top, so the whole layout rides `surface_z`. Default:
    # the ikea_table scene's Heavy-Duty packing table standing on the floor (top at 0.994 m —
    # per Haoxiang's preference); "lab_table" (top at z = 0, ground sunk to its feet) stays an
    # option. The smokes read `surface_z` too, so their choreography rides along.
    table: str = "packing"  # which work surface: "packing" | "lab_table"
    surface_z: float | None = None  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = None  # xy the table sits at; None -> preset
    workbench_usd: str = ""  # empty -> the preset's vendored USD
    # so101 defaults differ from bulb/nut_thread: identity orient lays the lab table's LONG side
    # (1.28 m vs 0.91 m) along the arm->parts spread (x in [-0.3, 0.62]), and `pos` centres the
    # physical top under the layout. MEASURED (physics probe, 2026-07-09): the spawner REPLACES
    # the USD root prim's authored xform (DemoTable carries a +0.55 x offset), so the collision
    # cube spans x [-0.796, 0.484], y [-0.455, 0.455] about the SPAWNED origin — pos (0.30,
    # 0.075) puts the top at x [-0.50, 0.78], y [-0.38, 0.53].
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.0, "pos": (0.30, 0.075),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        # packing top (root xform is identity): +/-1.237 x +/-0.381 about the origin
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.2, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    asset_dir: str = ""
    proximal_usd: str = ""  # floating-base build
    distal_usd: str = ""
    motor_usd: str = ""
    screw_usd: str = ""
    horn_screw_usd: str = ""
    drill_usd: str = ""

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "so101")
        a = Path(self.asset_dir)
        self.proximal_usd = self.proximal_usd or str(a / "so101_proximal.usd")
        self.distal_usd = self.distal_usd or str(a / "so101_distal.usd")
        self.motor_usd = self.motor_usd or str(a / "sts3215_03a.usd")
        self.screw_usd = self.screw_usd or str(a / "screw" / "screw_m2.usd")
        self.horn_screw_usd = self.horn_screw_usd or str(a / "screw" / "screw_m3.usd")
        self.drill_usd = self.drill_usd or str(a / "drill" / "power_drill.usd")
        # Fill the table placement from the chosen preset wherever the user left it unset.
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(
            assets / "props" / preset["usd"][0] / preset["usd"][1])

    # Screws are counted by spawn list, holes by seat table — the counts may differ (e.g. four
    # loose M3s vs eight horn-line holes). The flat index orders are [elbow tab screws...,
    # horn screws...] for screws (self.screws, `fastened` rows) and [elbow tab holes..., horn
    # holes...] for holes (`fastened` values, seat frames).
    @property
    def num_elbow_screws(self) -> int:
        return len(self.screw_spawn_pts)

    @property
    def num_horn_screws(self) -> int:
        return len(self.horn_screw_spawn_pts)

    @property
    def num_screws(self) -> int:
        return self.num_elbow_screws + self.num_horn_screws

    @property
    def num_elbow_holes(self) -> int:
        return len(self.elbow_screw_seat_pts)

    @property
    def num_horn_holes(self) -> int:
        return len(self.elbow_horn_screw_seat_pts)

    @property
    def num_holes(self) -> int:
        return self.num_elbow_holes + self.num_horn_holes


@SCENES.register("so101")
class SO101AssemblyScene(BaseScene):
    cfg: SO101SceneCfg

    def __init__(self, cfg: SO101SceneCfg | None = None) -> None:
        super().__init__(cfg or SO101SceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        sz = c.surface_z
        # Place the table so its top surface lands at `surface_z`, and sink the ground to its feet.
        table_z = sz - preset["top_offset"]
        ground_z = sz - preset["height"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd, scale=(preset["scale"],) * 3)
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        contact = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        )
        usd_drives = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)}
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(__file__).resolve().parents[1] / "assets" / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9))),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z), rot=preset["orient"]),
                spawn=table_spawn),
            # every body spawns FREE and spread out on the workbench top (see reset() for SPAWN);
            # a test re-stages them.
            "proximal": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Proximal",
                spawn=sim_utils.UsdFileCfg(usd_path=c.proximal_usd, rigid_props=contact),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, sz), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
                actuators=usd_drives),
            "motor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Motor",
                spawn=sim_utils.UsdFileCfg(usd_path=c.motor_usd, rigid_props=contact),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.15, sz + 0.06))),
            "drill": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Drill",
                spawn=sim_utils.UsdFileCfg(usd_path=c.drill_usd, rigid_props=contact),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.5, 0.0, sz + 0.12), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
                actuators=usd_drives),  # gains None -> the USD drives (the trigger spring!)
            # the NOT-yet-tested half: present in the world, free-floating. Spawned on the
            # roomy east side, well inboard of the table edges — its free chain WANDERS when
            # its joints are driven (e.g. a smoke's wiggle) and the old (0.5, 0.35) spot was
            # 3 cm from the packing top's +y edge (it crawled off).
            "distal": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Distal",
                spawn=sim_utils.UsdFileCfg(usd_path=c.distal_usd, rigid_props=contact),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.75, 0.15, sz + 0.06), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
                actuators=usd_drives),
        }
        for s, (x, y) in enumerate(c.screw_spawn_pts):
            out[f"screw_{s}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Screw_%d" % s,
                spawn=sim_utils.UsdFileCfg(usd_path=c.screw_usd, rigid_props=contact),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, sz + 0.02)))
        for s, (x, y) in enumerate(c.horn_screw_spawn_pts):
            out[f"horn_screw_{s}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/HornScrew_%d" % s,
                spawn=sim_utils.UsdFileCfg(usd_path=c.horn_screw_usd, rigid_props=contact),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, sz + 0.02)))
        return out

    def sim_cfg(self) -> SimCfg:
        # dt=1/240 + CCD on the screws (see _precreate_joints): a fast-moving M2 must not
        # tunnel the 1-3 mm printed walls.
        return SimCfg(dt=1.0 / 240.0, physx={"enable_ccd": True})

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.proximal: Articulation = env.iscene["proximal"]
        self.distal: Articulation = env.iscene["distal"]
        self.motor: RigidObject = env.iscene["motor"]
        self.screws: list[RigidObject] = (
            [env.iscene[f"screw_{s}"] for s in range(c.num_elbow_screws)]
            + [env.iscene[f"horn_screw_{s}"] for s in range(c.num_horn_screws)])
        self.drill: Articulation = env.iscene["drill"]
        self.env_origins = env.iscene.env_origins
        self.i_trig = self.drill.find_joints("trigger")[0][0]
        self.i_bit = self.drill.find_joints("bit_spin")[0][0]
        self.b_bit = self.drill.find_bodies("bit")[0][0]
        self.b_ua = self.proximal.find_bodies("upper_arm")[0][0]
        n, dev, ns = env.num_envs, env.device, c.num_screws
        self.fastened = torch.full((n, ns), -1, dtype=torch.long, device=dev)  # hole per screw, -1 = free
        self.attached = torch.zeros(n, ns, dtype=torch.bool, device=dev)  # riding the magnetic bit
        self.drive_t = torch.zeros(n, ns, device=dev)  # LATCHED drive depth per screw (see _fasten_rule)
        self.driving_prev = torch.zeros(n, ns, dtype=torch.bool, device=dev)
        self.drive_time = torch.zeros(n, ns, device=dev)
        self.spin_ang = torch.zeros(n, ns, device=dev)
        self.state = torch.zeros(n, dtype=torch.long, device=dev)  # 0 free, 1 driving, 2 all fastened
        self._elbow_screw_seat_pts = torch.tensor(c.elbow_screw_seat_pts, device=dev)
        self._elbow_screw_seat_quats = torch.tensor(c.elbow_screw_seat_quats, device=dev)
        self._elbow_horn_screw_seat_pts = torch.tensor(c.elbow_horn_screw_seat_pts, device=dev)
        self._elbow_horn_screw_seat_quats = torch.tensor(c.elbow_horn_screw_seat_quats, device=dev)
        self._seat_quats_all = torch.cat(
            [self._elbow_screw_seat_quats, self._elbow_horn_screw_seat_quats])
        self._la_seat_pos = torch.tensor(c.elbow_lower_arm_seat_pos, device=dev)
        self._la_seat_quat = torch.tensor(c.elbow_lower_arm_seat_quat, device=dev)
        # screw<->hole compatibility: each group's screws only fit its own holes (an M2 fits
        # nothing on the horn lines; an M3 fits no tab hole)
        self._pair_ok = torch.zeros(ns, c.num_holes, dtype=torch.bool, device=dev)
        self._pair_ok[:c.num_elbow_screws, :c.num_elbow_holes] = True
        self._pair_ok[c.num_elbow_screws:, c.num_elbow_holes:] = True
        self._bit_tip = torch.tensor(c.bit_tip, device=dev)
        self._ey = torch.tensor((0.0, 1.0, 0.0), device=dev)
        self._ez = torch.tensor((0.0, 0.0, 1.0), device=dev)
        self._precreate_joints()

    def _screw_prim(self, base: str, s: int) -> str:
        ne = self.cfg.num_elbow_screws
        return f"{base}/Screw_{s}" if s < ne else f"{base}/HornScrew_{s - ne}"

    def _precreate_joints(self) -> None:
        """Pre-author the normally-DISABLED fastening welds: one screw<->seat-link joint per
        screw (elbow tab screws seat in the upper_arm, the horn screw in the lower_arm; the seat
        frame is authored when the drive rule enables it, so any screw may take any hole in its
        group), plus the motor<->upper_arm and lower_arm<->motor part welds. Bit<->screw
        collision is filtered: that pair is rule-based (the drive gate owns it). Bit<->shell,
        bit<->motor and bit<->lower_arm are filtered too: the drive corridors graze the walls,
        the servo case and the fork skin, and contact just winds the free-spinning bit to
        hundreds of rad/s and bats the screw."""
        import omni.usd
        from pxr import Gf, Sdf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        self._weld_paths: list[list[str]] = []  # [env][screw]
        self._motor_weld_paths: list[str] = []
        self._elbow_joint_paths: list[str] = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            flt = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{base}/Drill/bit"))
            flt.CreateFilteredPairsRel().AddTarget(f"{base}/Proximal/upper_arm")
            flt.CreateFilteredPairsRel().AddTarget(f"{base}/Motor/upper_arm")
            flt.CreateFilteredPairsRel().AddTarget(f"{base}/Distal/lower_arm")
            for s in range(c.num_screws):  # CCD: a fast-moving screw must not tunnel the walls
                stage.GetPrimAtPath(self._screw_prim(base, s)).CreateAttribute(
                    "physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
            paths = []
            for s in range(c.num_screws):
                seat_link = (f"{base}/Proximal/upper_arm" if s < c.num_elbow_screws
                             else f"{base}/Distal/lower_arm")
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/screw_weld_{s}")
                j.CreateBody0Rel().SetTargets([seat_link])
                j.CreateBody1Rel().SetTargets([self._screw_prim(base, s)])
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateJointEnabledAttr(False)
                paths.append(f"{base}/screw_weld_{s}")
                flt.CreateFilteredPairsRel().AddTarget(self._screw_prim(base, s))
            self._weld_paths.append(paths)
            mj = UsdPhysics.FixedJoint.Define(stage, f"{base}/motor_weld")
            mj.CreateBody0Rel().SetTargets([f"{base}/Proximal/upper_arm"])
            mj.CreateBody1Rel().SetTargets([f"{base}/Motor/upper_arm"])
            mj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))  # same link frame on both sides
            mj.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            mj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            mj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            mj.CreateJointEnabledAttr(False)
            self._motor_weld_paths.append(f"{base}/motor_weld")
            lj = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/elbow_joint")
            lj.CreateBody0Rel().SetTargets([f"{base}/Motor/upper_arm"])
            lj.CreateBody1Rel().SetTargets([f"{base}/Distal/lower_arm"])
            lj.CreateAxisAttr("Z")  # the horn axis — the servo's output DOF
            lj.CreateLocalPos0Attr(Gf.Vec3f(*c.elbow_lower_arm_seat_pos))
            q = c.elbow_lower_arm_seat_quat
            lj.CreateLocalRot0Attr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
            lj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            lj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            lj.CreateLowerLimitAttr(-c.elbow_joint_limit_deg)
            lj.CreateUpperLimitAttr(c.elbow_joint_limit_deg)
            drv = UsdPhysics.DriveAPI.Apply(lj.GetPrim(), "angular")
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(c.elbow_joint_stiffness)
            drv.CreateDampingAttr(c.elbow_joint_damping)
            drv.CreateMaxForceAttr(c.elbow_joint_max_force)
            drv.CreateTargetPositionAttr(0.0)
            lj.CreateJointEnabledAttr(False)
            self._elbow_joint_paths.append(f"{base}/elbow_joint")

    def reset(self, env_ids: torch.Tensor) -> None:
        """Every body reset to its free spawn pose (spread out, upright, resting on the
        workbench top); welds released; drive state cleared."""
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        surf = torch.tensor((0.0, 0.0, self.cfg.surface_z), device=dev)

        def place(body, pos):  # env-local free spawn (matches assets() init_state), identity
            # quat; z rides the workbench top
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + surf + torch.tensor(pos, device=dev)
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        place(self.proximal, (0.0, 0.0, 0.0))  # free floating arm at the origin
        zp = torch.zeros(m, self.proximal.num_joints, device=dev)
        self.proximal.write_joint_state_to_sim(zp, zp, env_ids=env_ids)
        self.proximal.set_joint_position_target(zp, env_ids=env_ids)

        place(self.motor, (0.25, 0.15, 0.06))
        for s, (x, y) in enumerate(self.cfg.screw_spawn_pts):
            place(self.screws[s], (x, y, 0.02))
        for s, (x, y) in enumerate(self.cfg.horn_screw_spawn_pts):
            place(self.screws[self.cfg.num_elbow_screws + s], (x, y, 0.02))

        place(self.drill, (0.5, 0.0, 0.12))
        zdr = torch.zeros(m, self.drill.num_joints, device=dev)
        self.drill.write_joint_state_to_sim(zdr, zdr, env_ids=env_ids)

        place(self.distal, (0.75, 0.15, 0.06))
        zdi = torch.zeros(m, self.distal.num_joints, device=dev)
        self.distal.write_joint_state_to_sim(zdi, zdi, env_ids=env_ids)
        self.distal.set_joint_position_target(zdi, env_ids=env_ids)

        for i in env_ids.tolist():  # release every weld (nothing fastened)
            for s in range(self.cfg.num_screws):
                self._set_weld(int(i), s, 0, False)
        self.attached[env_ids] = False
        self.drive_t[env_ids] = 0.0
        self.driving_prev[env_ids] = False
        self.drive_time[env_ids] = 0.0
        self.spin_ang[env_ids] = 0.0
        self.state[env_ids] = 0

    # ----- frames (world, live) -------------------------------------------------------------------
    def upper_arm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the upper_arm link — the motor seat frame (identical by construction)."""
        return (self.proximal.data.body_pos_w[:, self.b_ua],
                self.proximal.data.body_quat_w[:, self.b_ua])

    def lower_arm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the lower_arm link — the distal articulation's root body."""
        return self.distal.data.root_pos_w, self.distal.data.root_quat_w

    def lower_arm_seat_w(self, motor_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where the lower_arm SEATS on the motor's output horn (world pose): the elbow joint
        transform applied to the (live or given) motor pose. `motor_pose` overrides the live
        one when it is stale or the caller wants the expected chain (see _reconcile_fastened)."""
        from isaaclab.utils.math import quat_apply, quat_mul

        mp, mq = motor_pose if motor_pose is not None else (
            self.motor.data.root_pos_w, self.motor.data.root_quat_w)
        pos = mp + quat_apply(mq, self._la_seat_pos.expand(len(mp), 3))
        return pos, quat_mul(mq, self._la_seat_quat.expand(len(mp), 4))

    def _seats_w(self, pose: tuple[torch.Tensor, torch.Tensor], pts: torch.Tensor,
                 quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World head-top seat points (m, nh, 3) and out-of-hole axes (m, nh, 3) of one hole
        group, from its seat-link pose and link-frame seat tables. The axis is derived from
        each hole's seat quat (screw local +Z points out of the head), so every hole gets the
        right approach direction whatever its facing."""
        from isaaclab.utils.math import quat_apply, quat_mul

        p, q = pose
        m, nh = len(p), len(pts)
        q_rep = q.repeat_interleave(nh, 0)
        seats = p.unsqueeze(1) + quat_apply(q_rep, pts.repeat(m, 1)).view(m, nh, 3)
        seat_q = quat_mul(q_rep, quats.repeat(m, 1))
        return seats, quat_apply(seat_q, self._ez.expand(m * nh, 3)).view(m, nh, 3)

    def elbow_screw_seats_w(self, arm_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """The elbow tab holes' world seats/axes — see _seats_w. `arm_pose` overrides the live
        upper_arm pose — pass it when the live link FK is stale (right after root writes, e.g.
        inside set_state)."""
        return self._seats_w(arm_pose if arm_pose is not None else self.upper_arm_pose(),
                             self._elbow_screw_seat_pts, self._elbow_screw_seat_quats)

    def elbow_horn_screw_seats_w(self, la_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """The horn screw holes' world seats/axes — see _seats_w. `la_pose` overrides the live
        lower_arm pose."""
        return self._seats_w(la_pose if la_pose is not None else self.lower_arm_pose(),
                             self._elbow_horn_screw_seat_pts, self._elbow_horn_screw_seat_quats)

    def _hole_frames_w(self, arm_pose=None, la_pose=None):
        """All holes' world seats (n, nh, 3), axes (n, nh, 3) and seat-link quats (n, nh, 4),
        concatenated in the flat hole order [elbow tabs..., horn...]."""
        ap, aq = arm_pose if arm_pose is not None else self.upper_arm_pose()
        lp, lq = la_pose if la_pose is not None else self.lower_arm_pose()
        e_s, e_a = self.elbow_screw_seats_w(arm_pose=(ap, aq))
        h_s, h_a = self.elbow_horn_screw_seats_w(la_pose=(lp, lq))
        ne, nh = self.cfg.num_elbow_holes, self.cfg.num_horn_holes
        link_q = torch.cat([aq.unsqueeze(1).expand(-1, ne, -1),
                            lq.unsqueeze(1).expand(-1, nh, -1)], dim=1)
        return torch.cat([e_s, h_s], dim=1), torch.cat([e_a, h_a], dim=1), link_q

    # ----- the fastening mechanic (runs every physics step) ----------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._reconcile_fastened()  # fastened parts follow the arm through teleports
        squeezed = self._spin_bit()  # the screwdriver: trigger squeeze -> bit spins
        self._fasten_rule(squeezed)

    def _reconcile_fastened(self, env_ids: torch.Tensor | None = None,
                            arm_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
                            motor_pose: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        """Fastened parts FOLLOW their seat link. The welds keep the physics tight, but an
        external write of a root (a test's vise, a reset-to-pose, a user teleport, a curriculum
        reset to a phase) moves that body alone and the enabled welds would yank the welded
        parts violently across the workspace. The expected poses form a CHAIN — the motor
        follows the arm (while tab-screwed), the lower_arm follows the motor (while
        horn-screwed), each screw follows its own hole's link — and any welded part found more
        than `weld_snap` off its weld frame is snapped back onto it with zero velocity, so
        teleporting the assembled robot is always legal. Runs each post_step against the live
        poses; set_state calls it with the SNAPSHOT's upper_arm + motor poses (rows aligned to
        `env_ids`), since the live FK is stale right after root writes."""
        from isaaclab.utils.math import quat_mul

        c, dev = self.cfg, self.env.device
        ids = torch.arange(self.env.num_envs, device=dev) if env_ids is None else env_ids
        f = self.fastened[ids]
        if not bool((f >= 0).any()):
            return
        ne = c.num_elbow_holes
        ap, aq = arm_pose if arm_pose is not None else self.upper_arm_pose()
        if arm_pose is None:
            ap, aq = ap[ids], aq[ids]
        mp, mq = motor_pose if motor_pose is not None else (
            self.motor.data.root_pos_w[ids], self.motor.data.root_quat_w[ids])
        motor_welded = ((f >= 0) & (f < ne)).any(dim=1)
        la_welded = (f >= ne).any(dim=1)
        exp_mp = torch.where(motor_welded.unsqueeze(-1), ap, mp)
        exp_mq = torch.where(motor_welded.unsqueeze(-1), aq, mq)
        fix = motor_welded & ((self.motor.data.root_pos_w[ids] - ap).norm(dim=-1) > c.weld_snap)
        if fix.any():
            rows = fix.nonzero(as_tuple=False).squeeze(-1)
            st = torch.zeros(len(rows), 13, device=dev)
            st[:, 0:3] = ap[rows]
            st[:, 3:7] = aq[rows]
            self.motor.write_root_state_to_sim(st, ids[rows])
        la_seat_p, la_seat_q = self.lower_arm_seat_w(motor_pose=(exp_mp, exp_mq))
        # the fork hangs on the assembled elbow JOINT (a revolute): a small heal keeps the live
        # joint ANGLE, but a row that needs a real SNAP (> weld_snap = a teleport) resets the
        # angle to ZERO — the pre-teleport pose it would be extracted from is stale garbage,
        # and an arbitrary angle can violate the joint limits at re-enable; the drive re-tracks
        # its target from zero
        qz, _ = self._elbow_angle_split(self.distal.data.root_quat_w[ids], la_seat_q)
        la_exp_q = quat_mul(la_seat_q, qz)
        fix = la_welded & ((self.distal.data.root_pos_w[ids] - la_seat_p).norm(dim=-1)
                           > c.weld_snap)
        if fix.any():
            rows = fix.nonzero(as_tuple=False).squeeze(-1)
            st = torch.zeros(len(rows), 13, device=dev)
            st[:, 0:3] = la_seat_p[rows]
            st[:, 3:7] = la_seat_q[rows]
            self.distal.write_root_state_to_sim(st, ids[rows])
            la_exp_q[rows] = la_seat_q[rows]
        exp_lp = torch.where(la_welded.unsqueeze(-1), la_seat_p, self.distal.data.root_pos_w[ids])
        exp_lq = torch.where(la_welded.unsqueeze(-1), la_exp_q, self.distal.data.root_quat_w[ids])
        seats, _, link_q = self._hole_frames_w(arm_pose=(ap, aq), la_pose=(exp_lp, exp_lq))
        arange = torch.arange(len(ids), device=dev)
        for s in range(c.num_screws):
            h = self.fastened[ids, s]
            exp = seats[arange, h.clamp_min(0)]
            fix = (h >= 0) & ((self.screws[s].data.root_pos_w[ids] - exp).norm(dim=-1) > c.weld_snap)
            if not fix.any():
                continue
            rows = fix.nonzero(as_tuple=False).squeeze(-1)
            st = torch.zeros(len(rows), 13, device=dev)
            st[:, 0:3] = exp[rows]
            st[:, 3:7] = quat_mul(link_q[arange, h.clamp_min(0)][rows],
                                  self._seat_quats_all[h.clamp_min(0)[rows]])
            self.screws[s].write_root_state_to_sim(st, ids[rows])

    def _spin_bit(self) -> torch.Tensor:
        """Screwdriver trigger -> bit coupling: squeezing the trigger past 70% of its swing spins
        the bit at `bit_speed`. Returns the per-env "trigger squeezed" mask (the fastening gate
        reads it too)."""
        c = self.cfg
        trig = self.drill.data.joint_pos[:, self.i_trig]
        squeezed = trig < -0.7 * c.trigger_swing
        spin_t = torch.where(squeezed, torch.full_like(trig, c.bit_speed), torch.zeros_like(trig))
        self.drill.set_joint_velocity_target(spin_t.unsqueeze(-1), joint_ids=[self.i_bit])
        return squeezed

    def _fasten_rule(self, squeezed: torch.Tensor) -> None:
        """THE FASTENING RULE (rule-based, no thread simulation). Within each hole GROUP the
        screws are identical and the pairing is order-independent (ikea-style): any free screw
        drives into any FREE, COMPATIBLE hole h while ALL of these hold, re-checked every step:

          1. SCREW IN A FREE HOLE — screw axis within `gate_axis_deg` of the hole axis, head
             center within `gate_radial` of the axis, inside the engagement window
             (`gate_window_below` below the seat .. `gate_window` above it — the axis check
             is waived below the seat, where the bore constrains the screw), no other screw
             fastened there, and the screw fits the hole (`_pair_ok`: M2 tab screws vs the
             elbow tab holes, the M3 horn screw vs the horn hole);
          2. PARTS ALIGNED      — the hole's JOINT is seated within `motor_align_pos` /
             `motor_align_deg`: a tab hole needs the servo in the arm pocket (their frames
             coincide exactly when seated); the horn hole needs the lower_arm on the motor's
             horn (at the elbow joint transform);
          3. DRIVER ON THE SCREW — bit tip within `bit_on_head` of that screw's head-top, bit
             axis within `bit_axis_deg` of the screw axis;
          4. TRIGGER ON          — squeezed past 70% AND the bit actually spinning (> `spin_min`).

        While driving, the screw moves along its hole's axis TOWARD the seat at `drive_rate` —
        descending if engaged above it, drawn back up if it lies deep in the hole — spinning
        with the bit (kinematic-follow with a LATCHED depth; pilot pushback must not slow the
        schedule). Reaching the seat after >= `min_drive_s` enables that screw's pre-authored
        weld; any fastened tab screw also holds the motor<->arm weld on, and the fastened horn
        screw the lower_arm<->motor weld: FASTENED until reset. Any condition breaking mid-drive
        returns the screw to free dynamics on the spot.
        """
        from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

        c, n, dt = self.cfg, self.env.num_envs, self.env.dt
        ns, nh, ne = c.num_screws, c.num_holes, c.num_elbow_holes
        bit_vel = self.drill.data.joint_vel[:, self.i_bit]
        spinning = bit_vel.abs() > c.spin_min

        ap, aq = self.upper_arm_pose()
        lp, lq = self.lower_arm_pose()
        sp = torch.stack([s.data.root_pos_w for s in self.screws], dim=1)  # (n, ns, 3)
        sq = torch.stack([s.data.root_quat_w for s in self.screws], dim=1)  # (n, ns, 4)
        bit_q = self.drill.data.body_quat_w[:, self.b_bit]
        seats, axis, link_q = self._hole_frames_w(arm_pose=(ap, aq), la_pose=(lp, lq))
        s_axis = quat_apply(sq.view(-1, 4), self._ez.expand(n * ns, 3)).view(n, ns, 3)  # out of head
        tip = self.drill.data.body_pos_w[:, self.b_bit] + quat_apply(
            bit_q, self._bit_tip.expand(n, 3))
        bit_dir = quat_apply(bit_q, self._ey.expand(n, 3))

        # 1. screw in a free, compatible hole — all pairs at once: (n, ns, nh)
        delta = sp.unsqueeze(2) - seats.unsqueeze(1)
        ax = axis.unsqueeze(1)
        t = (delta * ax).sum(-1)
        radial = (delta - t.unsqueeze(-1) * ax).norm(dim=-1)
        hole_free = torch.ones(n, nh, dtype=torch.bool, device=sp.device)
        taken = self.fastened >= 0
        hole_free.scatter_(1, self.fastened.clamp_min(0), ~taken)
        # below its seat the screw is inside the bore, which constrains it better than the
        # axis check could — the check applies only above
        axis_ok = ((s_axis.unsqueeze(2) * ax).sum(-1)
                   >= math.cos(math.radians(c.gate_axis_deg))) | (t < 0)
        in_hole = ((t > -c.gate_window_below) & (t < c.gate_window)
                   & (radial < c.gate_radial) & axis_ok & hole_free.unsqueeze(1)
                   & self._pair_ok.unsqueeze(0))
        # 2. parts aligned — per hole: the tab holes need the seated servo, the horn hole the
        # seated lower_arm
        motor_aligned = (((self.motor.data.root_pos_w - ap).norm(dim=-1) < c.motor_align_pos)
                         & (quat_error_magnitude(self.motor.data.root_quat_w, aq)
                            < math.radians(c.motor_align_deg)))
        la_exp_p, la_exp_q = self.lower_arm_seat_w()
        # angle-agnostic: the horn holes rotate WITH the assembled elbow joint, so only the
        # position and the OFF-AXIS orientation must match the seat
        _, la_residual = self._elbow_angle_split(lq, la_exp_q)
        la_aligned = (((lp - la_exp_p).norm(dim=-1) < c.motor_align_pos)
                      & (la_residual < math.radians(c.motor_align_deg)))
        aligned_h = torch.cat([motor_aligned.unsqueeze(1).expand(-1, ne),
                               la_aligned.unsqueeze(1).expand(-1, nh - ne)], dim=1)  # (n, nh)
        # 3. driver on the screw — per screw: (n, ns)
        on_head = (((tip.unsqueeze(1) - sp).norm(dim=-1) < c.bit_on_head)
                   & ((bit_dir.unsqueeze(1) * s_axis).sum(-1)
                      <= -math.cos(math.radians(c.bit_axis_deg))))
        # 4. trigger on; plus only a FREE screw can drive
        screw_ok = ~taken & on_head & (squeezed & spinning).unsqueeze(1)
        gate = in_hole & screw_ok.unsqueeze(-1) & aligned_h.unsqueeze(1)  # (n, ns, nh)
        driving = gate.any(dim=2)  # (n, ns)
        hsel = torch.where(gate, radial, torch.full_like(radial, torch.inf)).argmin(dim=2)

        # accumulated drive time survives a momentary gate break while the screw stays ON the
        # bit (the magnet re-takes it instantly and the driver never stopped spinning on it) —
        # requiring an uninterrupted window starves the weld whenever a contact flicker at the
        # seat breaks the gate for single steps
        self.drive_time = torch.where(
            driving, self.drive_time + dt,
            torch.where(self.attached, self.drive_time, torch.zeros_like(self.drive_time)))
        self.spin_ang = torch.where(driving, self.spin_ang + bit_vel.unsqueeze(1) * dt, self.spin_ang)
        t_sel = t.gather(2, hsel.unsqueeze(-1)).squeeze(-1)
        self.drive_t = torch.where(driving & ~self.driving_prev, t_sel, self.drive_t)
        # the drive works TOWARD the seat from either side: a screw engaged above descends, one
        # that slid deep into its hole is drawn back up as it is driven
        step_d = c.drive_rate * dt
        toward = torch.where(self.drive_t.abs() <= step_d, torch.zeros_like(self.drive_t),
                             self.drive_t - self.drive_t.sign() * step_d)
        self.drive_t = torch.where(driving, toward, self.drive_t)
        self.driving_prev = driving.clone()

        # THE MAGNETIC BIT (rule-based, like the gate): a free screw whose head-top comes within
        # `bit_on_head` of the bit tip ATTACHES and rides the bit — head on the tip, coaxial,
        # spinning with it — until it is driven home. Real M2 driving carries the screw on a
        # magnetized bit; nothing perches loose 2 mm screws in vertical holes. One screw per
        # bit; a driving screw is owned by the drive (the bit re-takes it if the gate breaks).
        near_tip = (tip.unsqueeze(1) - sp).norm(dim=-1) < c.bit_on_head  # (n, ns)
        grab_new = near_tip & ~taken & ~driving & ~self.attached.any(dim=1, keepdim=True)
        first = grab_new & (grab_new.cumsum(dim=1) == 1)  # at most one new screw per env
        self.attached |= first
        self.attached &= ~taken  # a fastened screw leaves the bit
        q_align = torch.tensor((0.7071068, 0.7071068, 0.0, 0.0), device=sp.device)  # +Z -> -Y
        for s in range(ns):
            carry = self.attached[:, s] & ~driving[:, s]
            if not carry.any():
                continue
            idx = carry.nonzero(as_tuple=False).squeeze(-1)
            st = torch.zeros(len(idx), 13, device=sp.device)
            st[:, 0:3] = tip[idx]
            st[:, 3:7] = quat_mul(bit_q[idx], q_align.expand(len(idx), 4))
            self.screws[s].write_root_state_to_sim(st, idx)

        for s in range(ns):  # kinematic-follow each driving screw to its selected hole
            if not driving[:, s].any():
                continue
            idx = driving[:, s].nonzero(as_tuple=False).squeeze(-1)
            h = hsel[idx, s]
            t_new = self.drive_t[idx, s]
            pos = seats[idx, h] + t_new.unsqueeze(-1) * axis[idx, h]
            half = 0.5 * self.spin_ang[idx, s]
            zero = torch.zeros_like(half)
            q_spin = torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)
            quat = quat_mul(quat_mul(link_q[idx, h], self._seat_quats_all[h]), q_spin)
            st = torch.cat([pos, quat, torch.zeros(len(idx), 6, device=pos.device)], dim=-1)
            self.screws[s].write_root_state_to_sim(st, idx)
            done = (t_new.abs() <= 1e-6) & (self.drive_time[idx, s] >= c.min_drive_s)
            for k in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                self._set_weld(int(idx[k]), s, int(h[k]), True)

        self.state.zero_()  # 0 free
        self.state[driving.any(dim=1)] = 1
        self.state[(self.fastened >= 0).all(dim=1)] = 2  # 2 = fully fastened

    def _set_weld(self, env_i: int, screw: int, hole: int, on: bool) -> None:
        """Toggle screw `screw`'s weld; enabling authors its seat frame from `hole` (any screw
        can lock into any hole of its group; hole indices follow the flat order, elbow tabs then
        horn). The screws are what fasten the parts, so the pre-authored motor<->arm weld stays
        on while ANY tab screw is fastened, and the lower_arm<->motor weld while any horn screw
        is."""
        from pxr import Gf, UsdPhysics

        stage = self.env.stage
        c = self.cfg
        ne = c.num_elbow_holes
        j = UsdPhysics.FixedJoint.Get(stage, self._weld_paths[env_i][screw])
        if on:
            if hole < ne:
                seat, q = c.elbow_screw_seat_pts[hole], c.elbow_screw_seat_quats[hole]
            else:
                seat = c.elbow_horn_screw_seat_pts[hole - ne]
                q = c.elbow_horn_screw_seat_quats[hole - ne]
            j.CreateLocalPos0Attr(Gf.Vec3f(*seat))
            j.CreateLocalRot0Attr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        j.GetJointEnabledAttr().Set(on)
        self.fastened[env_i, screw] = hole if on else -1
        f = self.fastened[env_i]
        self._set_part_weld(self._motor_weld_paths[env_i], bool(((f >= 0) & (f < ne)).any()))
        self._set_part_weld(self._elbow_joint_paths[env_i], bool((f >= ne).any()))

    def _set_part_weld(self, path: str, on: bool) -> None:
        from pxr import UsdPhysics

        UsdPhysics.Joint.Get(self.env.stage, path).GetJointEnabledAttr().Set(on)

    def set_elbow_target(self, target_rad: float) -> None:
        """Position target for the ASSEMBLED elbow joint (live once a horn screw is fastened).
        The servo drive tracks it exactly like the arm's own joint drives."""
        from pxr import UsdPhysics

        deg = math.degrees(target_rad)
        for path in self._elbow_joint_paths:
            UsdPhysics.DriveAPI.Get(self.env.stage.GetPrimAtPath(path),
                                    "angular").GetTargetPositionAttr().Set(deg)

    def _elbow_angle_split(self, lq: torch.Tensor, la_seat_q: torch.Tensor,
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the fork's orientation relative to its seat into the elbow-axis rotation
        Rz(theta) (the assembled joint's DOF) and the off-axis residual (rad)."""
        from isaaclab.utils.math import quat_conjugate, quat_error_magnitude, quat_mul

        rel = quat_mul(quat_conjugate(la_seat_q), lq)
        half = torch.atan2(rel[:, 3], rel[:, 0])
        zero = torch.zeros_like(half)
        qz = torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)
        return qz, quat_error_magnitude(rel, qz)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state. `upper_arm` (the link's world pose) is derived, not written
        back — set_state reconciles the fastened parts against it, so a restored (or hand-built,
        e.g. a curriculum/RL reset to a phase) state comes up self-consistent."""
        return {
            "proximal_root": self.proximal.data.root_state_w[env_ids].clone(),
            "proximal_joint_pos": self.proximal.data.joint_pos[env_ids].clone(),
            "proximal_joint_vel": self.proximal.data.joint_vel[env_ids].clone(),
            "upper_arm": torch.cat([self.proximal.data.body_pos_w[env_ids, self.b_ua],
                                    self.proximal.data.body_quat_w[env_ids, self.b_ua]], dim=-1),
            "distal_root": self.distal.data.root_state_w[env_ids].clone(),
            "distal_joint_pos": self.distal.data.joint_pos[env_ids].clone(),
            "distal_joint_vel": self.distal.data.joint_vel[env_ids].clone(),
            "motor": self.motor.data.root_state_w[env_ids].clone(),
            "screws": torch.stack([s.data.root_state_w[env_ids].clone() for s in self.screws], dim=1),
            "drill_root": self.drill.data.root_state_w[env_ids].clone(),
            "drill_joint_pos": self.drill.data.joint_pos[env_ids].clone(),
            "drill_joint_vel": self.drill.data.joint_vel[env_ids].clone(),
            "fastened": self.fastened[env_ids].clone(),
            "attached": self.attached[env_ids].clone(),
            "drive_t": self.drive_t[env_ids].clone(),
            "drive_time": self.drive_time[env_ids].clone(),
            "spin_ang": self.spin_ang[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what get_state returned, in three stages: bodies, then welds, then the drive
        state. Finally the fastened parts are reconciled against the state's OWN upper_arm pose
        (the live link FK is stale until the sim steps) — so a hand-edited state (e.g. flipping
        `fastened` flags to reset a curriculum to a later phase) still comes up with the welded
        parts seated consistently instead of being yanked on the first step."""
        self.proximal.write_root_state_to_sim(state["proximal_root"], env_ids)
        self.proximal.write_joint_state_to_sim(
            state["proximal_joint_pos"], state["proximal_joint_vel"], env_ids=env_ids)
        self.distal.write_root_state_to_sim(state["distal_root"], env_ids)
        self.distal.write_joint_state_to_sim(
            state["distal_joint_pos"], state["distal_joint_vel"], env_ids=env_ids)
        self.motor.write_root_state_to_sim(state["motor"], env_ids)
        for s, screw in enumerate(self.screws):
            screw.write_root_state_to_sim(state["screws"][:, s], env_ids)
        self.drill.write_root_state_to_sim(state["drill_root"], env_ids)
        self.drill.write_joint_state_to_sim(
            state["drill_joint_pos"], state["drill_joint_vel"], env_ids=env_ids)
        for row, i in enumerate(env_ids.tolist()):
            for s in range(self.cfg.num_screws):
                h = int(state["fastened"][row, s])
                self._set_weld(int(i), s, max(h, 0), h >= 0)
        self.attached[env_ids] = state["attached"]
        self.drive_t[env_ids] = state["drive_t"]
        self.driving_prev[env_ids] = False
        self.drive_time[env_ids] = state["drive_time"]
        self.spin_ang[env_ids] = state["spin_ang"]
        self._reconcile_fastened(
            env_ids, arm_pose=(state["upper_arm"][:, 0:3], state["upper_arm"][:, 3:7]),
            motor_pose=(state["motor"][:, 0:3], state["motor"][:, 3:7]))

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "The lower half of an SO101 robot arm (base + shoulder + upper arm), a bare elbow "
            "servo, four identical loose M2x6 screws, one loose M3 horn screw, a compact power "
            "screwdriver with a magnetic bit, and the not-yet-attached distal half "
            "(forearm..gripper) all rest free on a workbench. The upper arm has four M2 screw "
            "holes over the elbow servo's pocket (joint 3): a countersunk pair in the near "
            "outer wall and the mirrored pair through the far wall. The distal half's forearm "
            "fork clips over the seated servo — its cup onto the output horn — and carries "
            "four M3 screw holes around the elbow axis on each side (near: into the horn, "
            "reached through the skin's access channels; far: into the servo's case back). "
            "Goal: seat the servo into the pocket and drive the four M2 tab screws (any tab "
            "screw fits any tab hole), then clip the forearm onto the horn and drive the M3s "
            "home (any M3 fits any horn-line hole). A driven screw locks in place; the servo "
            "is fastened by the tab screws, the forearm by the horn screws."
        )
