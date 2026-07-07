"""SO101AssemblyScene — assemble the SO101 arm: seat + screw the elbow servo, attach the distal.

The assembly object-world:

  - `proximal`: base + shoulder + upper-arm shell as a FLOATING-base articulation (joints 1-2
    live).
  - `motor`: the bare elbow STS3215 as a free rigid body. KEY FRAME FACT: its body frame IS the
    upper_arm link frame, so "seated in the pocket" = identical body poses. Its collision mesh
    is baked 3% smaller (slip fit) so a straight-line push inserts it.
  - `screw`: one M2x6 (slim-head bake), free rigid body.
  - `drill`: the compact power screwdriver articulation (body/trigger/bit).
  - `distal`: lower_arm..gripper as a floating articulation — present but NOT yet under test
    (its fastening story comes later); its assets already carry the same collision hygiene.

Every body is free-floating; the only pre-authored joints are the DISABLED fastening welds (the
mechanic below).

THE FASTENING MECHANIC (rule-based for fast simulation — see `_fasten_rule`): pre-authored
DISABLED FixedJoints at the canonical seated frames; a per-step gate (screw in hole + parts
aligned + bit on head + trigger on) advances the screw kinematically with a latched depth and
snaps the welds on at the seat — the screw weld also welds motor<->arm. Everything else is real
collision: SDF meshes with the actual holes, the countersink funnels a dropped screw, tight
explicit contact offsets.

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
    motor_align_pos: float = tunable(0.004)  # max servo-vs-pocket position error to fasten (m)
    motor_align_deg: float = tunable(10.0)  # max servo-vs-pocket orientation error to fasten (deg)
    bit_on_head: float = tunable(0.004)  # bit tip -> screw head-top distance for the gate (m)
    bit_axis_deg: float = tunable(30.0)  # max bit-vs-screw axis misalignment (deg)
    spin_min: float = tunable(3.0)  # bit speed that counts as "spinning" (rad/s)

    # --- info: the drill (powered screwdriver) -----------------------------------------------------
    bit_speed: float = info(15.0)  # bit spin speed while the trigger is squeezed (rad/s)
    trigger_swing: float = info(math.radians(14.0))  # trigger travel, rest -> full squeeze (rad)
    bit_tip: tuple[float, float, float] = info((0.0, 0.055, 0.0))  # bit tip point, in the bit's own link frame

    # --- info: fastening welds — where the screw seats in the upper_arm (LINK frame) ---------------
    # Per-joint convention: each joint's screw(s) carry its <joint>_ prefix; adding wrist_*/shoulder_*
    # later is additive. Pose-agnostic — a screw seats relative to its link, wherever the arm is.
    elbow_screw_seat_pts: tuple[tuple[float, float, float], ...] = info(((-0.1227, 0.0010, -0.0035),))  # M2 head-top seated, per hole (z below flush -0.0015 -> proud, bit clears the arm)
    elbow_screw_seat_quat: tuple[float, float, float, float] = info((0.0, 0.0, 1.0, 0.0))  # seated orientation (link frame)

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
        return {
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
            "screw": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Screw",
                spawn=sim_utils.UsdFileCfg(usd_path=c.screw_usd, rigid_props=contact),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.15, 0.02))),
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

    def sim_cfg(self) -> SimCfg:
        # dt=1/240: no tunneling for a dropped M2 against the 1-3 mm printed walls (verified);
        # 1/480 is the fallback if drop/perch contacts ever misbehave.
        return SimCfg(dt=1.0 / 240.0)

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.proximal: Articulation = env.iscene["proximal"]
        self.distal: Articulation = env.iscene["distal"]
        self.motor: RigidObject = env.iscene["motor"]
        self.screw: RigidObject = env.iscene["screw"]
        self.drill: Articulation = env.iscene["drill"]
        self.env_origins = env.iscene.env_origins
        self.i_trig = self.drill.find_joints("trigger")[0][0]
        self.i_bit = self.drill.find_joints("bit_spin")[0][0]
        self.b_bit = self.drill.find_bodies("bit")[0][0]
        self.b_ua = self.proximal.find_bodies("upper_arm")[0][0]
        n, dev = env.num_envs, env.device
        self.fastened = torch.full((n,), -1, dtype=torch.long, device=dev)  # hole idx, -1 = free
        self.drive_t = torch.zeros(n, device=dev)  # LATCHED drive depth (see _fasten_rule)
        self.driving_prev = torch.zeros(n, dtype=torch.bool, device=dev)
        self.drive_time = torch.zeros(n, device=dev)
        self.spin_ang = torch.zeros(n, device=dev)
        self.state = torch.zeros(n, dtype=torch.long, device=dev)  # 0 free, 1 driving, 2 fastened
        self._elbow_screw_seat_pts = torch.tensor(self.cfg.elbow_screw_seat_pts, device=dev)
        self._elbow_screw_seat_quat = torch.tensor(self.cfg.elbow_screw_seat_quat, device=dev)
        self._bit_tip = torch.tensor(self.cfg.bit_tip, device=dev)
        self._ey = torch.tensor((0.0, 1.0, 0.0), device=dev)
        self._ez = torch.tensor((0.0, 0.0, 1.0), device=dev)
        self._precreate_joints()

    def _precreate_joints(self) -> None:
        """Pre-author the normally-DISABLED fastening welds (screw<->upper_arm at each seat, plus
        motor<->upper_arm); the drive rule enables them at the seat. Bit<->screw collision is
        filtered: that pair is rule-based (the drive gate owns it)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        screw_q = self.cfg.elbow_screw_seat_quat
        self._weld_paths: list[list[str]] = []
        self._motor_weld_paths: list[str] = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            paths = []
            for h, seat in enumerate(self.cfg.elbow_screw_seat_pts):
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/screw_weld_{h}")
                j.CreateBody0Rel().SetTargets([f"{base}/Proximal/upper_arm"])
                j.CreateBody1Rel().SetTargets([f"{base}/Screw"])
                j.CreateLocalPos0Attr(Gf.Vec3f(*seat))
                j.CreateLocalRot0Attr(Gf.Quatf(screw_q[0], Gf.Vec3f(*screw_q[1:])))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateJointEnabledAttr(False)
                paths.append(f"{base}/screw_weld_{h}")
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
            flt = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{base}/Drill/bit"))
            flt.CreateFilteredPairsRel().AddTarget(f"{base}/Screw")

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
        place(self.screw, (0.25, -0.15, 0.02))

        place(self.drill, (0.5, 0.0, 0.12))
        zdr = torch.zeros(m, self.drill.num_joints, device=dev)
        self.drill.write_joint_state_to_sim(zdr, zdr, env_ids=env_ids)

        place(self.distal, (0.5, 0.35, 0.06))
        zdi = torch.zeros(m, self.distal.num_joints, device=dev)
        self.distal.write_joint_state_to_sim(zdi, zdi, env_ids=env_ids)
        self.distal.set_joint_position_target(zdi, env_ids=env_ids)

        for i in env_ids.tolist():  # release every weld (nothing fastened)
            for h in range(len(self.cfg.elbow_screw_seat_pts)):
                self._set_weld(int(i), h, False)
        self.fastened[env_ids] = -1
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

    def elbow_screw_seats_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World head-top seat points of the screw hole(s) (n, nh, 3) + the hole axis (n, 3)."""
        from isaaclab.utils.math import quat_apply

        n, nh = self.env.num_envs, len(self.cfg.elbow_screw_seat_pts)
        ap, aq = self.upper_arm_pose()
        seats = ap.unsqueeze(1) + quat_apply(
            aq.repeat_interleave(nh, 0), self._elbow_screw_seat_pts.repeat(n, 1)).view(n, nh, 3)
        return seats, quat_apply(aq, -self._ez.expand(n, 3))  # out of the hole = link -Z

    # ----- the fastening mechanic (runs every physics step) ----------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        squeezed = self._spin_bit()  # the screwdriver: trigger squeeze -> bit spins
        self._fasten_rule(squeezed)

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
        """THE FASTENING RULE (rule-based, no thread simulation). A free screw is DRIVING into
        hole h while ALL of these hold, re-checked every step:

          1. SCREW IN THE HOLE  — screw axis within `gate_axis_deg` of the hole axis, head
             center within `gate_radial` of the axis, inside the engagement window
             (-1 mm .. `gate_window` above the seat);
          2. PARTS ALIGNED      — the servo sits in the arm pocket within `motor_align_pos` /
             `motor_align_deg` (their frames coincide exactly when seated);
          3. DRIVER ON THE SCREW — bit tip within `bit_on_head` of the screw head-top, bit axis
             within `bit_axis_deg` of the screw axis;
          4. TRIGGER ON          — squeezed past 70% AND the bit actually spinning (> `spin_min`).

        While driving, the screw advances along the hole axis at `drive_rate`, spinning with the
        bit (kinematic-follow with a LATCHED depth — pilot pushback must not slow the schedule).
        Reaching the seat after >= `min_drive_s` enables the pre-baked welds (screw<->arm AND
        motor<->arm): FASTENED until reset. Any condition breaking mid-drive returns the screw
        to free dynamics on the spot.
        """
        from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

        c, n, dt = self.cfg, self.env.num_envs, self.env.dt
        bit_vel = self.drill.data.joint_vel[:, self.i_bit]
        spinning = bit_vel.abs() > c.spin_min

        ap, aq = self.upper_arm_pose()
        sp, sq = self.screw.data.root_pos_w, self.screw.data.root_quat_w
        bit_q = self.drill.data.body_quat_w[:, self.b_bit]
        seats, axis = self.elbow_screw_seats_w()
        s_axis = quat_apply(sq, self._ez.expand(n, 3))  # screw axis, out of the head
        tip = self.drill.data.body_pos_w[:, self.b_bit] + quat_apply(
            bit_q, self._bit_tip.expand(n, 3))
        bit_dir = quat_apply(bit_q, self._ey.expand(n, 3))

        # 1. screw in the hole
        delta = sp.unsqueeze(1) - seats
        t = (delta * axis.unsqueeze(1)).sum(-1)
        radial = (delta - t.unsqueeze(-1) * axis.unsqueeze(1)).norm(dim=-1)
        in_hole = ((t > -1e-3) & (t < c.gate_window) & (radial < c.gate_radial)
                   & ((s_axis * axis).sum(-1)
                      >= math.cos(math.radians(c.gate_axis_deg))).unsqueeze(1))
        # 2. parts aligned
        parts_aligned = (((self.motor.data.root_pos_w - ap).norm(dim=-1) < c.motor_align_pos)
                         & (quat_error_magnitude(self.motor.data.root_quat_w, aq)
                            < math.radians(c.motor_align_deg)))
        # 3. driver on the screw
        on_head = ((tip - sp).norm(dim=-1) < c.bit_on_head) & (
            (bit_dir * s_axis).sum(-1) <= -math.cos(math.radians(c.bit_axis_deg)))
        # 4. trigger on; plus only a FREE screw can drive
        free = self.fastened < 0
        gate = in_hole & (free & parts_aligned & on_head & squeezed & spinning).unsqueeze(1)
        driving = gate.any(dim=1)
        hsel = torch.where(gate, radial, torch.full_like(radial, torch.inf)).argmin(dim=1)

        self.drive_time = torch.where(driving, self.drive_time + dt, torch.zeros_like(self.drive_time))
        self.spin_ang = torch.where(driving, self.spin_ang + bit_vel * dt, self.spin_ang)
        t_sel = t.gather(1, hsel.unsqueeze(-1)).squeeze(-1)
        self.drive_t = torch.where(driving & ~self.driving_prev, t_sel, self.drive_t)
        self.drive_t = torch.where(driving, (self.drive_t - c.drive_rate * dt).clamp_min(0.0),
                                   self.drive_t)
        self.driving_prev = driving.clone()
        if driving.any():
            idx = driving.nonzero(as_tuple=False).squeeze(-1)
            h = hsel[idx]
            t_new = self.drive_t[idx]
            pos = seats[idx, h] + t_new.unsqueeze(-1) * axis[idx]
            half = 0.5 * self.spin_ang[idx]
            zero = torch.zeros_like(half)
            q_spin = torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)
            quat = quat_mul(quat_mul(aq[idx], self._elbow_screw_seat_quat.expand(len(idx), 4)), q_spin)
            st = torch.cat([pos, quat, torch.zeros(len(idx), 6, device=pos.device)], dim=-1)
            self.screw.write_root_state_to_sim(st, idx)
            done = (t_new <= 1e-6) & (self.drive_time[idx] >= c.min_drive_s)
            for k in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                self._set_weld(int(idx[k]), int(h[k]), True)

        self.state.zero_()  # 0 free
        self.state[driving] = 1
        self.state[self.fastened >= 0] = 2

    def _set_weld(self, env_i: int, hole: int, on: bool) -> None:
        """Toggle the screw<->arm weld; the screw is what fastens the servo, so the pre-baked
        motor<->arm weld toggles with it."""
        from pxr import UsdPhysics

        stage = self.env.stage
        UsdPhysics.FixedJoint.Get(stage, self._weld_paths[env_i][hole]).GetJointEnabledAttr().Set(on)
        UsdPhysics.FixedJoint.Get(stage, self._motor_weld_paths[env_i]).GetJointEnabledAttr().Set(on)
        if on:
            self.fastened[env_i] = hole

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "proximal_joint_pos": self.proximal.data.joint_pos[env_ids].clone(),
            "proximal_joint_vel": self.proximal.data.joint_vel[env_ids].clone(),
            "distal_root": self.distal.data.root_state_w[env_ids].clone(),
            "distal_joint_pos": self.distal.data.joint_pos[env_ids].clone(),
            "distal_joint_vel": self.distal.data.joint_vel[env_ids].clone(),
            "motor": self.motor.data.root_state_w[env_ids].clone(),
            "screw": self.screw.data.root_state_w[env_ids].clone(),
            "drill_root": self.drill.data.root_state_w[env_ids].clone(),
            "drill_joint_pos": self.drill.data.joint_pos[env_ids].clone(),
            "drill_joint_vel": self.drill.data.joint_vel[env_ids].clone(),
            "fastened": self.fastened[env_ids].clone(),
            "drive_t": self.drive_t[env_ids].clone(),
            "drive_time": self.drive_time[env_ids].clone(),
            "spin_ang": self.spin_ang[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.proximal.write_joint_state_to_sim(
            state["proximal_joint_pos"], state["proximal_joint_vel"], env_ids=env_ids)
        self.distal.write_root_state_to_sim(state["distal_root"], env_ids)
        self.distal.write_joint_state_to_sim(
            state["distal_joint_pos"], state["distal_joint_vel"], env_ids=env_ids)
        self.motor.write_root_state_to_sim(state["motor"], env_ids)
        self.screw.write_root_state_to_sim(state["screw"], env_ids)
        self.drill.write_root_state_to_sim(state["drill_root"], env_ids)
        self.drill.write_joint_state_to_sim(
            state["drill_joint_pos"], state["drill_joint_vel"], env_ids=env_ids)
        for row, i in enumerate(env_ids.tolist()):
            want = int(state["fastened"][row])
            for h in range(len(self.cfg.elbow_screw_seat_pts)):
                self._set_weld(int(i), h, h == want)
            self.fastened[int(i)] = want
        self.drive_t[env_ids] = state["drive_t"]
        self.driving_prev[env_ids] = False
        self.drive_time[env_ids] = state["drive_time"]
        self.spin_ang[env_ids] = state["spin_ang"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "The lower half of an SO101 robot arm (base + shoulder + upper arm), a bare elbow "
            "servo, a loose M2x6 screw, a compact power screwdriver, and the not-yet-attached "
            "distal half (forearm..gripper) all rest free in the workspace. In the upper arm's "
            "outer wall, above the elbow servo's pocket, is a countersunk "
            "M2 hole (joint 3). Goal: seat the servo into the pocket, drop the screw into the "
            "countersunk hole — the cone funnels it upright and the servo's pin perches it — and "
            "drive it home with the screwdriver, fastening the servo into the arm. A seated screw "
            "locks in place; an unseated one falls away."
        )
