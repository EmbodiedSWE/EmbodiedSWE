"""Franka OSC session substrate — the shared layer every scripted solution re-implements.

Distilled from the corpus of working Franka solutions (examples/solve_pen_holder.py and the
19 sim_gen campaign solutions, which copied it from there with per-task drift): the servo
loop, the phase kernel, grasp mechanics, and the wound-arm recovery, with the measured
constants that made them work. Import AFTER AppLauncher has booted (Isaac import order).

What this module deliberately does NOT contain: task-specific contact primitives (friction
drags, poke controllers, counterweight placement) — those are geometry-bound and live with
their tasks. The pick_and_place() at the bottom is the one multi-task skill: the same phase
sequence appeared, parameters aside, in six independent solutions.

Constant provenance labels (transferability):
  [ROBOT]  a property of the Franka hardware/geometry — transfers across tasks;
  [CORPUS] default that held across the solved-task corpus — a starting point, retune
           when contact loading differs (two tasks did);
  [FAMILY] tuned only within the pick-place family envelope (top-down grasps of roughly
           40-90 mm, 50-500 g objects) — no evidence outside it, retune there.

Typical use:

    s = FrankaSession.make("my_scene", base_pos=(-0.45, 0.0, 0.0))
    s.hold(s.origin + V3(0.2, 0.0, 0.30), s.jaw_quat(0.0), OPEN, secs=1.0)
    ok = s.run_phase(goal_fn, gate_fn, grip=OPEN, timeout_s=8.0, tag="approach")
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

OPEN = 0.04          # [ROBOT] per-finger position target, fully open (80 mm aperture)
CLOSED = 0.0
# [ROBOT] panda_hand frame -> fingertip along the approach axis; calibrate +/-5 mm for the
# intended contact point on the pad (corpus used 0.107-0.112).
FINGER_LEN = 0.112

# [CORPUS] OSC gain defaults; kd = 2*sqrt(kp) (critical damping). Tasks with stiff contact
# loading retuned kp (500/400 and 320 appear in the corpus).
DEFAULT_KP = (220.0, 220.0, 220.0, 600.0, 600.0, 600.0)
DEFAULT_ROT_SCALE = 0.15   # [CORPUS]
DEFAULT_KP_NULL = 3.0      # [CORPUS]


def V3(x: float, y: float, z: float, device: str = "cuda:0") -> torch.Tensor:
    return torch.tensor([float(x), float(y), float(z)], device=device)


class FrankaSession:
    """One Franka + OSC control session over a robobench env (num_envs=1)."""

    def __init__(self, env, kp=DEFAULT_KP, rot_scale=DEFAULT_ROT_SCALE,
                 kp_null=DEFAULT_KP_NULL, finger_len=FINGER_LEN):
        self.env = env
        self.robot = env.robot
        self.scene = env.scene
        self.device = env.device
        art = self.robot.articulation
        self.art = art
        self.ee_idx = art.body_names.index("panda_hand")
        self.fj1 = art.find_joints(["panda_finger_joint1"])[0]
        self.finger_len = finger_len

        osc = self.robot.controller.controllers[0]
        osc._kp = torch.tensor(list(kp), device=self.device)
        osc._kd = 2.0 * osc._kp.sqrt()
        osc.cfg.rot_scale = rot_scale
        osc.cfg.kp_null = kp_null
        osc.cfg.kd_null = 2.0 * math.sqrt(kp_null)
        self.osc = osc

        self.n_act = self.robot.action_dim
        self.ctrl_hz = 1.0 / (env.dt * self.robot.control_period)
        self.origin = env.iscene.env_origins[0]
        self.sim_t = 0.0
        self.on_step = None   # optional hook(session) after every tick, e.g. video capture

    @classmethod
    def make(cls, scene_name: str, base_pos=(-0.45, 0.0, 0.0), base_rot=None,
             gripper_effort_limit=120.0, gripper_stiffness=4000.0, device=None, **kw):
        """Build the env with the known-good Franka config and wrap it.

        nullspace_dof_pos=() is load-bearing: the default posture target winds the arm
        against its joint limits during long lateral servos.
        """
        import robobench
        from robobench.core import EnvCfg
        from robobench.robots.franka import FrankaRobotCfg

        robobench.discover()
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        rcfg = FrankaRobotCfg(base_pos=base_pos, nullspace_dof_pos=(),
                              gripper_effort_limit=gripper_effort_limit,
                              gripper_stiffness=gripper_stiffness)
        if base_rot is not None:
            rcfg.base_rot = base_rot
        env = EnvCfg(scene=scene_name, robot="franka", control_mode="osc",
                     env_spacing=3, robot_cfg=rcfg).build(num_envs=1, device=device)
        env.reset()
        return cls(env, **kw)

    # ----- state reads ---------------------------------------------------------------
    def ee_pose(self):
        return self.art.data.body_pos_w[0, self.ee_idx], self.art.data.body_quat_w[0, self.ee_idx]

    def width(self) -> float:
        """Current jaw opening (both fingers)."""
        return 2.0 * self.art.data.joint_pos[0, self.fj1].item()

    def tip_pos(self) -> torch.Tensor:
        p, q = self.ee_pose()
        zh = quat_apply(q.unsqueeze(0), V3(0, 0, 1, self.device).unsqueeze(0))[0]
        return p + self.finger_len * zh

    def hand_for_tip(self, tip: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Hand position that puts the fingertip midpoint at `tip` for hand quat `quat`."""
        zh = quat_apply(quat.unsqueeze(0), V3(0, 0, 1, self.device).unsqueeze(0))[0]
        return tip - self.finger_len * zh

    # ----- orientation constructors ----------------------------------------------------
    def jaw_quat(self, azimuth: float) -> torch.Tensor:
        """Top-down hand orientation with the jaw axis at world `azimuth`."""
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0, self.device)
        zh = V3(0.0, 0.0, -1.0, self.device)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    def tilt_quat(self, azimuth: float, target_xy: torch.Tensor,
                  near: float = 0.37, max_tilt: float = 0.35) -> torch.Tensor:
        """jaw_quat tilted away from the base column for close-in targets: a strict
        top-down pose within ~0.37 m of the base folds q4 onto its limit.
        `near` is [ROBOT] (the arm's own geometry); the tilt ramp and cap are [CORPUS]."""
        gq = self.jaw_quat(azimuth)
        base_xy = self.art.data.root_pos_w[0, :2]
        d = float((target_xy - base_xy).norm())
        if d < near:
            tilt = min(max_tilt, (near - d) * 5.0)
            u = (target_xy - base_xy) / max(d, 1e-6)
            axis = V3(-u[1], u[0], 0.0, self.device)
            gq = quat_mul(quat_from_angle_axis(
                torch.tensor([tilt], device=self.device), axis.unsqueeze(0))[0].unsqueeze(0),
                gq.unsqueeze(0))[0]
        return gq

    # ----- the servo/tick/phase kernel --------------------------------------------------
    def SEC(self, s: float) -> int:
        return max(1, round(s * self.ctrl_hz))

    def servo(self, goal_pos, goal_quat, grip, a, xy_boost: float = 1.0):
        """Fill action `a` toward the pose goal. xy_boost > 1 counteracts the OSC's
        lateral sag while carrying a load (the pick-place solutions used 1.6); leave
        at 1.0 for free-space motion and pushes."""
        p, q = self.ee_pose()
        err = goal_pos - p
        err = torch.cat([err[:2] * xy_boost, err[2:3]])
        a[0, 0:3] = (err / self.osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / self.osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip

    def tick(self, a, render: bool = False):
        self.env.step(a, render=render)
        self.sim_t += self.env.dt * self.robot.control_period
        if self.on_step is not None:
            self.on_step(self)

    def hold(self, pos, quat, grip, secs: float, xy_boost: float = 1.0):
        for _ in range(self.SEC(secs)):
            a = torch.zeros(1, self.n_act, device=self.device)
            self.servo(pos, quat, grip, a, xy_boost)
            self.tick(a)

    def run_phase(self, goal_fn, gate_fn, grip, timeout_s: float, tag: str = "",
                  xy_boost: float = 1.0) -> bool:
        """Closed-loop phase: servo toward goal_fn() (recomputed every tick from measured
        state) until gate_fn() or timeout. The pattern that makes solutions robust:
        goals track live object poses, gates read measured outcomes."""
        deadline = self.sim_t + timeout_s
        while self.sim_t < deadline:
            a = torch.zeros(1, self.n_act, device=self.device)
            gp, gq = goal_fn()
            self.servo(gp, gq, grip, a, xy_boost)
            self.tick(a)
            if gate_fn():
                return True
        if tag:
            p, _ = self.ee_pose()
            gp, _ = goal_fn()
            print(f"[phase:{tag}] timeout: ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                  f"err={float((gp - p).norm()) * 1000:.0f}mm w={self.width()*1000:.1f}mm",
                  flush=True)
        return False

    # ----- grasp mechanics ---------------------------------------------------------------
    def close_ramp(self, pos, quat, grip_target: float, secs: float = 1.2):
        """Close the jaw as a slow ramp to a non-zero target. Commanding 0 outright pops
        light/box objects out of the jaw at the effort limit. The ramp time is [CORPUS];
        grip_target is task-specific by design (corpus used 0.020-0.029 for 40-70 mm
        objects)."""
        n = self.SEC(secs)
        for k in range(n):
            a = torch.zeros(1, self.n_act, device=self.device)
            f = min(1.0, (k + 1) / (n * 0.7))
            self.servo(pos, quat, OPEN + (grip_target - OPEN) * f, a)
            self.tick(a)

    def lift_verdict(self, obj_z: float, min_rise: float,
                     width_lo: float, width_hi: float, obj_z_now) -> bool:
        """The grasp test that actually works: lift, then check the OBJECT rose and the
        jaw width sits in the expected band (too narrow = slipped out; too wide = jammed)."""
        w = self.width()
        return (obj_z_now() - obj_z) > min_rise and width_lo < w < width_hi

    # ----- recovery ------------------------------------------------------------------------
    def dewind(self, grip: float = OPEN) -> bool:
        """Reset a wound-up arm. Long lateral servos walk q1/q4/q7 onto their limits, after
        which the OSC stalls; a joint reset in place is cheap and the phase loop re-acquires.
        Thresholds from the solution corpus."""
        q = self.art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[dewind] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — reset",
                  flush=True)
            self.robot.reset(torch.tensor([0], device=self.device, dtype=torch.long))
            p, qq = self.ee_pose()
            self.hold(p, qq, grip, 0.5)
            return True
        return False


# ---- Layer 2: the one skill six solutions implemented independently -----------------------

def pick_and_place(s: FrankaSession, obj_pos_fn, place_xy: torch.Tensor, *,
                   jaw_azimuth: float, grasp_tip_z_fn, grip: float,
                   width_lo: float, width_hi: float,
                   carry_z: float, place_z: float,
                   place_tol: float = 0.008, attempts: int = 3) -> bool:
    """Hover -> descend -> ramped close -> lift verdict -> closed-loop carry on the OBJECT
    -> closed-loop lower on the OBJECT -> slow release. Goals always track obj_pos_fn()
    live; per-attempt retry with a dewind between tries.

    obj_pos_fn() -> (3,) live object position; grasp_tip_z_fn() -> world z for the
    fingertips at grasp. All heights are world z. Returns True when the object was
    released within place_tol of place_xy.

    Every internal constant here (correction gains 1.3/1.2 with 15/10 mm clamps, 6-8 mm
    gates, 8-12 s phase timeouts, 50 mm lift-rise test, the 1.6 carry xy-boost) is
    [FAMILY]: consistent across the six pick-place solutions but validated only in their
    envelope — retune outside it.
    """
    for attempt in range(attempts):
        s.dewind()
        gq = s.tilt_quat(jaw_azimuth, obj_pos_fn()[:2])

        def tip_goal(z):
            o = obj_pos_fn()
            return s.hand_for_tip(V3(o[0], o[1], z, s.device), gq), gq

        hover_z = carry_z + s.finger_len
        if not s.run_phase(lambda: tip_goal(hover_z),
                           lambda: (s.ee_pose()[0][:2] - obj_pos_fn()[:2]).norm() < 0.006,
                           OPEN, 8.0, tag=f"hover{attempt}"):
            continue
        if not s.run_phase(lambda: tip_goal(grasp_tip_z_fn()),
                           lambda: abs(s.tip_pos()[2] - grasp_tip_z_fn()) < 0.006,
                           OPEN, 8.0, tag=f"descend{attempt}"):
            continue

        gp, _ = tip_goal(grasp_tip_z_fn())
        s.close_ramp(gp, gq, grip)

        z0 = float(obj_pos_fn()[2])
        p, _ = s.ee_pose()
        s.hold(V3(p[0], p[1], carry_z + s.finger_len, s.device), gq, grip, 1.0)
        if not s.lift_verdict(z0, 0.05, width_lo, width_hi,
                              lambda: float(obj_pos_fn()[2])):
            p, _ = s.ee_pose()
            s.hold(V3(p[0], p[1], hover_z, s.device), gq, OPEN, 0.8)
            continue

        def carry_goal():
            o, h = obj_pos_fn(), s.ee_pose()[0]
            corr = (1.3 * (place_xy - o[:2])).clamp(-0.015, 0.015)
            return V3(h[0] + corr[0], h[1] + corr[1], carry_z + s.finger_len, s.device), gq

        s.run_phase(carry_goal,
                    lambda: (obj_pos_fn()[:2] - place_xy).norm() < place_tol,
                    grip, 12.0, tag=f"carry{attempt}", xy_boost=1.6)

        def lower_goal():
            o, h = obj_pos_fn(), s.ee_pose()[0]
            corr = (1.2 * (place_xy - o[:2])).clamp(-0.01, 0.01)
            return V3(h[0] + corr[0], h[1] + corr[1],
                      h[2] - (o[2] - place_z), s.device), gq

        s.run_phase(lower_goal,
                    lambda: abs(float(obj_pos_fn()[2]) - place_z) < 0.006,
                    grip, 10.0, tag=f"lower{attempt}", xy_boost=1.6)

        p, q = s.ee_pose()
        s.hold(p, q, OPEN, 0.8)                                # slow release in place
        s.hold(V3(p[0], p[1], hover_z, s.device), q, OPEN, 1.0)  # retreat up
        if (obj_pos_fn()[:2] - place_xy).norm() < place_tol * 2:
            return True
    return False
