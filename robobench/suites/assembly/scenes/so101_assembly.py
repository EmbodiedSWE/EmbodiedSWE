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
  - `distal`: lower_arm..gripper as a floating articulation — present but NOT yet under test
    (its fastening story comes later); its assets already carry the same collision hygiene.

The servo is held by four M2 screws (upper_arm LINK frame): the NEAR pair enters the countersunk
outer wall from link -Z; the FAR pair enters from link +Z through the ring bosses on that face.
Each hole carries its own seat pose; the out-of-hole axis is derived from it, so both facings
run the same mechanic.

Every body is free-floating; the only pre-authored joints are the DISABLED fastening welds (the
mechanic below).

THE FASTENING MECHANIC (rule-based for fast simulation — see `_fasten_rule`): one pre-authored
DISABLED FixedJoint per screw, its seat frame authored at enable time (so any screw can take any
hole); a per-step gate (screw in a free hole + parts aligned + bit on that screw + trigger on)
advances the screw kinematically with a latched depth and snaps its weld on at the seat — any
fastened screw also welds motor<->arm. THE MAGNETIC BIT (same section): a free screw whose head
touches the bit tip attaches and rides it, coaxial and spinning, until driven home — real M2
driving carries the screw on a magnetized bit, and any embodiment holding the drill can use it.
Everything else is real collision against the parts' actual holes and walls.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from robobench.core import BaseEnv


@dataclass
class SO101SceneCfg(BaseCfg):
    """Config for `SO101AssemblyScene`. `tunable()` fields are curriculum/difficulty dials;
    `info()` fields are structural constants (see `robobench.core.BaseCfg`)."""

    # --- tunable: the fastening gate + drive dials (see _fasten_rule for THE RULE) -----------------
    drive_rate: float = tunable(0.020)  # screw advance speed while driving (m/s)
    min_drive_s: float = tunable(0.25)  # minimum accumulated drive time before the weld engages (s)
    gate_axis_deg: float = tunable(15.0)  # max screw-vs-hole axis misalignment (deg)
    gate_radial: float = tunable(0.0025)  # max head-center offset from the hole axis (m)
    gate_window: float = tunable(0.014)  # engagement window above the seat, along the axis (m)
    gate_window_below: float = tunable(0.008)  # engagement window BELOW the seat (m): a screw
    # that slid deep into its hole still drives. Below the seat the bore itself constrains the
    # screw, so the axis check is waived there — inside the hole, messy is acceptable; only
    # what happens above the surface must be precise.
    motor_align_pos: float = tunable(0.004)  # max servo-vs-pocket position error to fasten (m)
    motor_align_deg: float = tunable(10.0)  # max servo-vs-pocket orientation error to fasten (deg)
    bit_on_head: float = tunable(0.004)  # bit tip -> screw head-top distance for the gate (m)
    bit_axis_deg: float = tunable(30.0)  # max bit-vs-screw axis misalignment (deg)
    spin_min: float = tunable(3.0)  # bit speed that counts as "spinning" (rad/s)
    weld_snap: float = info(0.005)  # fastened part farther than this off its weld frame snaps back (m)

    # --- info: the drill (powered screwdriver) -----------------------------------------------------
    bit_speed: float = info(15.0)  # bit spin speed while the trigger is squeezed (rad/s)
    trigger_swing: float = info(math.radians(14.0))  # trigger travel, rest -> full squeeze (rad)
    bit_tip: tuple[float, float, float] = info((0.0, 0.055, 0.0))  # bit tip point, in the bit's own link frame

    # --- info: fastening welds — where each screw seats in the upper_arm (LINK frame) --------------
    # Per-joint convention: each joint's screw(s) carry its <joint>_ prefix; adding wrist_*/shoulder_*
    # later is additive. Pose-agnostic — a screw seats relative to its link, wherever the arm is.
    # One loose screw is spawned per hole (identical screws — any screw may take any hole).
    # Each seat is the driven head-top, placed so the seated screw rests contact-free (the weld
    # holds it): the near heads sit ~2 mm proud of the countersunk wall, the far heads sit
    # inside the ring bosses with the tips just above the servo's far tab.
    elbow_screw_seat_pts: tuple[tuple[float, float, float], ...] = info((
        (-0.1227, 0.0010, -0.0040),  # near pair: the countersunk holes in the -Z wall
        (-0.1022, 0.0010, -0.0040),
        (-0.1228, 0.0052, 0.0428),   # far pair: the ring-boss holes through the +Z wall
        (-0.1023, 0.0052, 0.0428),
    ))
    elbow_screw_seat_quats: tuple[tuple[float, float, float, float], ...] = info((
        (0.0, 0.0, 1.0, 0.0),  # near: screw +Z (out of the head) -> link -Z
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),  # far: screw +Z -> link +Z
        (1.0, 0.0, 0.0, 0.0),
    ))
    # free spawn xy of each loose screw (env frame, resting on the ground; z from the asset)
    screw_spawn_pts: tuple[tuple[float, float], ...] = info(
        ((0.25, -0.15), (0.31, -0.15), (0.25, -0.22), (0.31, -0.22)))

    # --- info: scene assets ------------------------------------------------------------------------
    light_intensity: float = info(2500.0)
    asset_dir: str = info("")
    proximal_usd: str = info("")  # floating-base build
    distal_usd: str = info("")
    motor_usd: str = info("")
    screw_usd: str = info("")
    drill_usd: str = info("")

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "so101")
        a = Path(self.asset_dir)
        self.proximal_usd = self.proximal_usd or str(a / "so101_proximal.usd")
        self.distal_usd = self.distal_usd or str(a / "so101_distal.usd")
        self.motor_usd = self.motor_usd or str(a / "sts3215_03a.usd")
        self.screw_usd = self.screw_usd or str(a / "screw" / "screw_m2.usd")
        self.drill_usd = self.drill_usd or str(a / "drill" / "power_drill.usd")

    @property
    def num_screws(self) -> int:
        return len(self.elbow_screw_seat_pts)  # one identical loose screw per hole


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
        contact = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        )
        usd_drives = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)}
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9))),
            # every body spawns FREE and spread out (see reset() for SPAWN); a test re-stages them.
            "proximal": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Proximal",
                spawn=sim_utils.UsdFileCfg(usd_path=c.proximal_usd, rigid_props=contact),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
                actuators=usd_drives),
            "motor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Motor",
                spawn=sim_utils.UsdFileCfg(usd_path=c.motor_usd, rigid_props=contact),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.15, 0.06))),
            "drill": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Drill",
                spawn=sim_utils.UsdFileCfg(usd_path=c.drill_usd, rigid_props=contact),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.5, 0.0, 0.12), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
                actuators=usd_drives),  # gains None -> the USD drives (the trigger spring!)
            # the NOT-yet-tested half: present in the world, free-floating
            "distal": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Distal",
                spawn=sim_utils.UsdFileCfg(usd_path=c.distal_usd, rigid_props=contact),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.5, 0.35, 0.06), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
                actuators=usd_drives),
        }
        for s, (x, y) in enumerate(c.screw_spawn_pts):
            out[f"screw_{s}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Screw_%d" % s,
                spawn=sim_utils.UsdFileCfg(usd_path=c.screw_usd, rigid_props=contact),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.02)))
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
        self.screws: list[RigidObject] = [env.iscene[f"screw_{s}"] for s in range(c.num_screws)]
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
        self._bit_tip = torch.tensor(c.bit_tip, device=dev)
        self._ey = torch.tensor((0.0, 1.0, 0.0), device=dev)
        self._ez = torch.tensor((0.0, 0.0, 1.0), device=dev)
        self._precreate_joints()

    def _precreate_joints(self) -> None:
        """Pre-author the normally-DISABLED fastening welds: one screw<->upper_arm joint per screw
        (the seat frame is authored when the drive rule enables it, so any screw may take any
        hole), plus the single motor<->upper_arm weld. Bit<->screw collision is filtered: that
        pair is rule-based (the drive gate owns it). Bit<->shell and bit<->motor are filtered
        too: the drive corridors graze both walls and the servo case, and contact just winds the
        free-spinning bit to hundreds of rad/s and bats the screw."""
        import omni.usd
        from pxr import Gf, Sdf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        self._weld_paths: list[list[str]] = []  # [env][screw]
        self._motor_weld_paths: list[str] = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            flt = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{base}/Drill/bit"))
            flt.CreateFilteredPairsRel().AddTarget(f"{base}/Proximal/upper_arm")
            flt.CreateFilteredPairsRel().AddTarget(f"{base}/Motor/upper_arm")
            for s in range(self.cfg.num_screws):  # CCD: a fast-moving M2 must not tunnel the walls
                stage.GetPrimAtPath(f"{base}/Screw_{s}").CreateAttribute(
                    "physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
            paths = []
            for s in range(self.cfg.num_screws):
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/screw_weld_{s}")
                j.CreateBody0Rel().SetTargets([f"{base}/Proximal/upper_arm"])
                j.CreateBody1Rel().SetTargets([f"{base}/Screw_{s}"])
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateJointEnabledAttr(False)
                paths.append(f"{base}/screw_weld_{s}")
                flt.CreateFilteredPairsRel().AddTarget(f"{base}/Screw_{s}")
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

    def reset(self, env_ids: torch.Tensor) -> None:
        """Every body reset to its free spawn pose (spread out, upright, resting on the ground);
        welds released; drive state cleared."""
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, pos):  # env-local free spawn (matches assets() init_state), identity quat
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor(pos, device=dev)
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        place(self.proximal, (0.0, 0.0, 0.0))  # free floating arm at the origin
        zp = torch.zeros(m, self.proximal.num_joints, device=dev)
        self.proximal.write_joint_state_to_sim(zp, zp, env_ids=env_ids)
        self.proximal.set_joint_position_target(zp, env_ids=env_ids)

        place(self.motor, (0.25, 0.15, 0.06))
        for s, (x, y) in enumerate(self.cfg.screw_spawn_pts):
            place(self.screws[s], (x, y, 0.02))

        place(self.drill, (0.5, 0.0, 0.12))
        zdr = torch.zeros(m, self.drill.num_joints, device=dev)
        self.drill.write_joint_state_to_sim(zdr, zdr, env_ids=env_ids)

        place(self.distal, (0.5, 0.35, 0.06))
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

    def elbow_screw_seats_w(self, arm_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """World head-top seat points of the screw holes (m, nh, 3) and each hole's out-of-hole
        axis (m, nh, 3) — derived from that hole's seat quat (screw local +Z points out of the
        head), so the -Z near holes and the +Z far holes each get the right approach direction.
        `arm_pose` (pos (m, 3), quat (m, 4)) overrides the live upper_arm pose — pass it when the
        live link FK is stale (right after root writes, e.g. inside set_state)."""
        from isaaclab.utils.math import quat_apply, quat_mul

        ap, aq = arm_pose if arm_pose is not None else self.upper_arm_pose()
        m, nh = len(ap), len(self.cfg.elbow_screw_seat_pts)
        aq_rep = aq.repeat_interleave(nh, 0)
        seats = ap.unsqueeze(1) + quat_apply(
            aq_rep, self._elbow_screw_seat_pts.repeat(m, 1)).view(m, nh, 3)
        seat_q = quat_mul(aq_rep, self._elbow_screw_seat_quats.repeat(m, 1))
        return seats, quat_apply(seat_q, self._ez.expand(m * nh, 3)).view(m, nh, 3)

    # ----- the fastening mechanic (runs every physics step) ----------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._reconcile_fastened()  # fastened parts follow the arm through teleports
        squeezed = self._spin_bit()  # the screwdriver: trigger squeeze -> bit spins
        self._fasten_rule(squeezed)

    def _reconcile_fastened(self, env_ids: torch.Tensor | None = None,
                            arm_pose: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        """Fastened parts FOLLOW the arm. The welds keep the physics tight, but an external write
        of the proximal root (a test's vise, a reset-to-pose, a user teleport, a curriculum reset
        to a phase) moves the arm alone and the enabled welds would yank the parts violently
        across the workspace. Any fastened screw — and the welded motor — found more than
        `weld_snap` off its weld frame is snapped back onto it with zero velocity, so teleporting
        the assembled robot is always legal. Runs each post_step against the live link pose;
        set_state calls it with the SNAPSHOT's upper_arm pose (rows aligned to `env_ids`), since
        the live link FK is stale right after root writes."""
        from isaaclab.utils.math import quat_mul

        c, dev = self.cfg, self.env.device
        ids = torch.arange(self.env.num_envs, device=dev) if env_ids is None else env_ids
        welded = (self.fastened[ids] >= 0).any(dim=1)
        if not bool(welded.any()):
            return
        ap, aq = arm_pose if arm_pose is not None else self.upper_arm_pose()
        if arm_pose is None:
            ap, aq = ap[ids], aq[ids]
        seats, _ = self.elbow_screw_seats_w(arm_pose=(ap, aq))
        fix = welded & ((self.motor.data.root_pos_w[ids] - ap).norm(dim=-1) > c.weld_snap)
        if fix.any():
            rows = fix.nonzero(as_tuple=False).squeeze(-1)
            st = torch.zeros(len(rows), 13, device=dev)
            st[:, 0:3] = ap[rows]
            st[:, 3:7] = aq[rows]
            self.motor.write_root_state_to_sim(st, ids[rows])
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
            st[:, 3:7] = quat_mul(aq[rows], self._elbow_screw_seat_quats[h.clamp_min(0)[rows]])
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
        """THE FASTENING RULE (rule-based, no thread simulation). The screws are identical and
        the pairing is order-independent (ikea-style): any free screw drives into any FREE hole h
        while ALL of these hold, re-checked every step:

          1. SCREW IN A FREE HOLE — screw axis within `gate_axis_deg` of the hole axis, head
             center within `gate_radial` of the axis, inside the engagement window
             (`gate_window_below` below the seat .. `gate_window` above it — the axis check
             is waived below the seat, where the bore constrains the screw), no other screw
             fastened there;
          2. PARTS ALIGNED      — the servo sits in the arm pocket within `motor_align_pos` /
             `motor_align_deg` (their frames coincide exactly when seated);
          3. DRIVER ON THE SCREW — bit tip within `bit_on_head` of that screw's head-top, bit
             axis within `bit_axis_deg` of the screw axis;
          4. TRIGGER ON          — squeezed past 70% AND the bit actually spinning (> `spin_min`).

        While driving, the screw moves along its hole's axis TOWARD the seat at `drive_rate` —
        descending if engaged above it, drawn back up if it lies deep in the hole — spinning
        with the bit (kinematic-follow with a LATCHED depth; pilot pushback must not slow the
        schedule). Reaching the seat after >= `min_drive_s` enables that screw's pre-authored
        weld; any fastened screw also holds the motor<->arm weld on: FASTENED until reset. Any
        condition breaking mid-drive returns the screw to free dynamics on the spot.
        """
        from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

        c, n, dt = self.cfg, self.env.num_envs, self.env.dt
        ns, nh = c.num_screws, len(c.elbow_screw_seat_pts)
        bit_vel = self.drill.data.joint_vel[:, self.i_bit]
        spinning = bit_vel.abs() > c.spin_min

        ap, aq = self.upper_arm_pose()
        sp = torch.stack([s.data.root_pos_w for s in self.screws], dim=1)  # (n, ns, 3)
        sq = torch.stack([s.data.root_quat_w for s in self.screws], dim=1)  # (n, ns, 4)
        bit_q = self.drill.data.body_quat_w[:, self.b_bit]
        seats, axis = self.elbow_screw_seats_w()  # (n, nh, 3) both
        s_axis = quat_apply(sq.view(-1, 4), self._ez.expand(n * ns, 3)).view(n, ns, 3)  # out of head
        tip = self.drill.data.body_pos_w[:, self.b_bit] + quat_apply(
            bit_q, self._bit_tip.expand(n, 3))
        bit_dir = quat_apply(bit_q, self._ey.expand(n, 3))

        # 1. screw in a free hole — all pairs at once: (n, ns, nh)
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
                   & (radial < c.gate_radial) & axis_ok & hole_free.unsqueeze(1))
        # 2. parts aligned
        parts_aligned = (((self.motor.data.root_pos_w - ap).norm(dim=-1) < c.motor_align_pos)
                         & (quat_error_magnitude(self.motor.data.root_quat_w, aq)
                            < math.radians(c.motor_align_deg)))
        # 3. driver on the screw — per screw: (n, ns)
        on_head = (((tip.unsqueeze(1) - sp).norm(dim=-1) < c.bit_on_head)
                   & ((bit_dir.unsqueeze(1) * s_axis).sum(-1)
                      <= -math.cos(math.radians(c.bit_axis_deg))))
        # 4. trigger on; plus only a FREE screw can drive
        screw_ok = ~taken & on_head & (parts_aligned & squeezed & spinning).unsqueeze(1)
        gate = in_hole & screw_ok.unsqueeze(-1)  # (n, ns, nh)
        driving = gate.any(dim=2)  # (n, ns)
        hsel = torch.where(gate, radial, torch.full_like(radial, torch.inf)).argmin(dim=2)

        self.drive_time = torch.where(driving, self.drive_time + dt, torch.zeros_like(self.drive_time))
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
            quat = quat_mul(quat_mul(aq[idx], self._elbow_screw_seat_quats[h]), q_spin)
            st = torch.cat([pos, quat, torch.zeros(len(idx), 6, device=pos.device)], dim=-1)
            self.screws[s].write_root_state_to_sim(st, idx)
            done = (t_new.abs() <= 1e-6) & (self.drive_time[idx, s] >= c.min_drive_s)
            for k in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                self._set_weld(int(idx[k]), s, int(h[k]), True)

        self.state.zero_()  # 0 free
        self.state[driving.any(dim=1)] = 1
        self.state[(self.fastened >= 0).all(dim=1)] = 2  # 2 = fully fastened

    def _set_weld(self, env_i: int, screw: int, hole: int, on: bool) -> None:
        """Toggle screw `screw`'s weld; enabling authors its seat frame from `hole` (any screw can
        lock into any hole). The screws are what fasten the servo, so the pre-authored motor<->arm
        weld stays on while ANY screw is fastened."""
        from pxr import Gf, UsdPhysics

        stage = self.env.stage
        j = UsdPhysics.FixedJoint.Get(stage, self._weld_paths[env_i][screw])
        if on:
            seat, q = self.cfg.elbow_screw_seat_pts[hole], self.cfg.elbow_screw_seat_quats[hole]
            j.CreateLocalPos0Attr(Gf.Vec3f(*seat))
            j.CreateLocalRot0Attr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        j.GetJointEnabledAttr().Set(on)
        self.fastened[env_i, screw] = hole if on else -1
        self._set_motor_weld(env_i, bool((self.fastened[env_i] >= 0).any()))

    def _set_motor_weld(self, env_i: int, on: bool) -> None:
        from pxr import UsdPhysics

        UsdPhysics.FixedJoint.Get(
            self.env.stage, self._motor_weld_paths[env_i]).GetJointEnabledAttr().Set(on)

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
            env_ids, arm_pose=(state["upper_arm"][:, 0:3], state["upper_arm"][:, 3:7]))

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "The lower half of an SO101 robot arm (base + shoulder + upper arm), a bare elbow "
            "servo, four identical loose M2x6 screws, a compact power screwdriver with a "
            "magnetic bit, and the not-yet-attached distal half (forearm..gripper) all rest "
            "free in the workspace. The upper arm has four M2 screw holes over the elbow "
            "servo's pocket (joint 3): a countersunk pair in the near outer wall and the "
            "mirrored pair through the far wall. Goal: seat the servo into the pocket, then "
            "for each hole pick a screw up with the magnetic bit, insert it, and drive it home. "
            "Any screw fits any hole; a driven screw locks in place. The servo is fastened once "
            "all four screws are driven."
        )
