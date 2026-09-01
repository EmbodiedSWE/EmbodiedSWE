"""Screw the light bulb into the lamp socket (assembly.bulb.franka.osc).

Strategy (pure 8-D OSC actions; controller/control-mode are frozen):
  1. Read the lying bulb's pose; approach along its axis from the glass end,
     hand z antiparallel to the bulb axis, and grasp the glass envelope.
  2. Lift, pitch the hand to vertical (bulb axis up, cap down).
  3. Move over the socket (servo on the bulb origin), descend until the cap
     rests in the bore.
  4. Ratchet-screw: press down lightly and rotate CW (about world -z) with the
     wrist until q7 nears its limit; open, wind back, re-grip, repeat until
     scene.seated().

All servoing is closed-loop at the env control rate (~15 Hz): each step we
command the remaining pose delta (clamped), so the controller's EMA and
latching just shape the approach.

Bulb geometry (calibrated from the asset, local frame, origin at the cap tip):
  cap z 0..0.02 (Ø22) | thread collider Ø30 | glass z 0.015..0.083 (Ø48.2).
Socket: origin on the table, thread nut z 0.0225..0.0385, bore mouth 0.0385;
seated when bulb origin is <= 0.027 above the socket origin (rest ~0.035).
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------- quat utils


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qw, qv = q[..., 0:1], q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def _axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    q = torch.where(q[..., 0:1] >= 0, q, -q)
    w = q[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt((1.0 - w * w).clamp_min(1e-12))
    axis = q[..., 1:4] / s.unsqueeze(-1)
    small = angle < 1e-5
    return torch.where(small.unsqueeze(-1), q[..., 1:4] * 2.0, axis * angle.unsqueeze(-1))


def _quat_from_axis_angle(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = angle / 2.0
    return torch.cat(
        (torch.full_like(axis[..., :1], math.cos(half)), axis * math.sin(half)), dim=-1
    )


def _quat_from_two_axes(z_axis: torch.Tensor, y_axis_hint: torch.Tensor) -> torch.Tensor:
    """wxyz quat whose frame has z = z_axis and y ~ y_axis_hint (orthogonalized).
    Shepperd's branch method — the naive trace formula degenerates near 180 deg."""
    z = z_axis / z_axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x = torch.cross(y_axis_hint, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    y = torch.cross(z, x, dim=-1)
    # column-major rotation matrix entries m[r][c] with columns (x, y, z)
    m00, m01, m02 = float(x[0]), float(y[0]), float(z[0])
    m10, m11, m12 = float(x[1]), float(y[1]), float(z[1])
    m20, m21, m22 = float(x[2]), float(y[2]), float(z[2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 >= m11 and m00 >= m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 >= m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = torch.tensor([qw, qx, qy, qz], device=z_axis.device)
    return q / q.norm().clamp_min(1e-9)


GRIP_OPEN = 0.04
GRIP_CAGE = 0.031  # per-finger opening that still cages the Ø48 glass (6 mm/side clearance)
GRIP_CLOSED = 0.0

# calibrated bulb geometry (local frame, origin at the cap tip)
GLASS_EQUATOR_Z = 0.052  # grip height above the bulb origin, along its axis
FINGER_REACH = 0.100  # hand-frame z from panda_hand origin to the fingertip pad centre
Q7_LO, Q7_HI = -2.89, 2.89  # panda_joint7 limits


class BulbTask:
    """Closed-loop scripted solver for one env (num_envs == 1)."""

    def __init__(self, env, log=None) -> None:
        self.env = env
        self.robot = env.robot
        self.art = env.robot.articulation
        self.scene = env.scene
        self.dev = env.device
        self.ee_idx = self.art.body_names.index("panda_hand")
        self.q7_idx = self.art.joint_names.index("panda_joint7")
        self.fin_idx = [
            self.art.joint_names.index("panda_finger_joint1"),
            self.art.joint_names.index("panda_finger_joint2"),
        ]
        self.adim = self.robot.action_dim
        self.pos_scale = 0.02
        self.rot_scale = 0.097
        self.steps = 0
        self._log = log or (lambda *a, **k: None)
        self._grip = GRIP_OPEN

    # ---- state readers ------------------------------------------------------
    @property
    def ee_pos(self) -> torch.Tensor:
        return self.art.data.body_pos_w[0, self.ee_idx]

    @property
    def ee_quat(self) -> torch.Tensor:
        return self.art.data.body_quat_w[0, self.ee_idx]

    @property
    def q7(self) -> float:
        return float(self.art.data.joint_pos[0, self.q7_idx])

    @property
    def finger_gap(self) -> float:
        jp = self.art.data.joint_pos[0]
        return float(jp[self.fin_idx[0]] + jp[self.fin_idx[1]])

    @property
    def bulb_pos(self) -> torch.Tensor:
        return self.scene.bulbs[0].data.root_pos_w[0]

    @property
    def bulb_quat(self) -> torch.Tensor:
        return self.scene.bulbs[0].data.root_quat_w[0]

    @property
    def bulb_axis(self) -> torch.Tensor:
        """Bulb local +z (cap at the origin end, glass toward +z) in world."""
        return _quat_apply(self.bulb_quat, torch.tensor([0.0, 0.0, 1.0], device=self.dev))

    @property
    def socket_pos(self) -> torch.Tensor:
        return self.scene.sockets[0].data.root_pos_w[0]

    def bulb_depth(self) -> float:
        """Bulb origin height above the socket origin (socket frame z)."""
        return float(self.scene._bulb_offsets_in_socket()[0, 0, 0, 2])

    def bulb_xy_err(self) -> float:
        return float(self.scene._bulb_offsets_in_socket()[0, 0, 0, :2].norm())

    def seated(self) -> bool:
        return bool(self.scene.seated().all())

    def ee_z_axis(self) -> torch.Tensor:
        return _quat_apply(self.ee_quat, torch.tensor([0.0, 0.0, 1.0], device=self.dev))

    # ---- low-level action ---------------------------------------------------
    def act(self, dpos: torch.Tensor, drot: torch.Tensor, grip: float | None = None) -> None:
        if grip is not None:
            self._grip = grip
        a = torch.zeros(1, self.adim, device=self.dev)
        a[0, 0:3] = dpos / self.pos_scale
        a[0, 3:6] = drot / self.rot_scale
        a[0, 6:8] = self._grip
        self.env.step(a)
        self.steps += 1

    def hold(self, grip: float | None = None, n: int = 5) -> None:
        z = torch.zeros(3, device=self.dev)
        for _ in range(n):
            self.act(z, z, grip)

    def servo(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        grip: float | None = None,
        max_steps: int = 100,
        pos_tol: float = 0.004,
        rot_tol: float = 0.03,
        max_dpos: float = 0.08,
        max_drot: float = 0.5,
        settle: int = 0,
    ) -> bool:
        """Servo the EE to (pos, quat) with proportional saturated deltas."""
        ok = False
        for _ in range(max_steps):
            perr = pos - self.ee_pos
            rerr = _axis_angle_from_quat(_quat_mul(quat, _quat_conj(self.ee_quat)))
            if perr.norm() < pos_tol and rerr.norm() < rot_tol:
                ok = True
                break
            dpos = perr.clamp(-max_dpos, max_dpos)
            n = rerr.norm()
            drot = rerr if n <= max_drot else rerr * (max_drot / n)
            self.act(dpos, drot, grip)
        for _ in range(settle):
            perr = (pos - self.ee_pos).clamp(-max_dpos, max_dpos)
            rerr = _axis_angle_from_quat(_quat_mul(quat, _quat_conj(self.ee_quat)))
            n = rerr.norm()
            drot = rerr if n <= max_drot else rerr * (max_drot / n)
            self.act(perr, drot, grip)
        return ok

    # ---- phases --------------------------------------------------------------
    def grasp_pose(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(grasp hand pos, hand quat, glass centre) for the current bulb pose."""
        dev = self.dev
        axis = self.bulb_axis
        hand_z = -axis  # approach from the glass end toward the cap
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        y_hint = torch.cross(up, hand_z, dim=-1)  # fingers close horizontally
        if y_hint.norm() < 1e-3:
            y_hint = torch.tensor([0.0, 1.0, 0.0], device=dev)
        quat = _quat_from_two_axes(hand_z, y_hint)
        glass = self.bulb_pos + axis * GLASS_EQUATOR_Z
        hand_pos = glass - hand_z * FINGER_REACH
        return hand_pos, quat, glass

    def pick(self) -> bool:
        """Approach along the bulb axis, grasp the glass, lift; True if the bulb came up."""
        for attempt in range(3):
            hand_pos, quat, glass = self.grasp_pose()
            pre = hand_pos - _quat_apply(quat, torch.tensor([0.0, 0.0, 0.12], device=self.dev))
            pre = pre.clone()
            pre[2] = max(float(pre[2]), 0.06)
            self._log(f"pick attempt {attempt}: pre-grasp")
            self.servo(pre, quat, GRIP_OPEN, max_steps=140, pos_tol=0.008, settle=2)
            # re-read the bulb (it may have been nudged) and go in
            hand_pos, quat, glass = self.grasp_pose()
            self.servo(hand_pos, quat, GRIP_OPEN, max_steps=80, pos_tol=0.0035, rot_tol=0.02, max_dpos=0.03, settle=4)
            self.hold(GRIP_CLOSED, 12)
            # lift straight up
            lift = self.ee_pos.clone()
            lift[2] = 0.22
            self.servo(lift, quat, GRIP_CLOSED, max_steps=60, pos_tol=0.01)
            gap = self.finger_gap
            # gap 0.0465 = glass equator; below 0.045 the grip is off-equator and
            # tends to slip later under screwing loads — reject and retry
            if float(self.bulb_pos[2]) > 0.10 and 0.045 < gap < 0.055:
                self._log(f"pick OK (bulb z={float(self.bulb_pos[2]):.3f}, gap={gap:.3f})")
                return True
            self._log(f"pick FAILED (bulb z={float(self.bulb_pos[2]):.3f}, gap={gap:.3f}); retrying")
            self.hold(GRIP_OPEN, 6)
            up = self.ee_pos.clone()
            up[2] = 0.25
            self.servo(up, quat, GRIP_OPEN, max_steps=40, pos_tol=0.02)
        return False

    def reposition(self) -> bool:
        """Fallback when the axial pick is unreachable (glass end facing away from
        the base): top-down cross-axis grasp, lift, yaw so the glass end faces the
        base again, set down near the default spot, release. Then pick() can retry."""
        dev = self.dev
        axis = self.bulb_axis
        axis_h = torch.tensor([float(axis[0]), float(axis[1]), 0.0], device=dev)
        if axis_h.norm() < 0.2:  # bulb standing upright — nothing sensible to do here
            return False
        axis_h = axis_h / axis_h.norm()
        down = torch.tensor([0.0, 0.0, -1.0], device=dev)
        y_hint = torch.tensor([-float(axis_h[1]), float(axis_h[0]), 0.0], device=dev)
        quat = _quat_from_two_axes(down, y_hint)  # fingers close across the bulb axis
        glass = self.bulb_pos + self.bulb_axis * GLASS_EQUATOR_Z
        above = torch.tensor([float(glass[0]), float(glass[1]), FINGER_REACH + 0.10], device=dev)
        self.servo(above, quat, GRIP_OPEN, max_steps=80, pos_tol=0.006)
        grasp = above.clone()
        grasp[2] = float(glass[2]) + FINGER_REACH - 0.004
        self.servo(grasp, quat, GRIP_OPEN, max_steps=50, pos_tol=0.004, max_dpos=0.03, settle=3)
        self.hold(GRIP_CLOSED, 10)
        lift = grasp.clone()
        lift[2] = 0.20
        self.servo(lift, quat, GRIP_CLOSED, max_steps=40, pos_tol=0.01)
        if float(self.bulb_pos[2]) < 0.08:
            self._log("reposition: lift failed")
            self.hold(GRIP_OPEN, 5)
            return False
        # yaw so the bulb's glass end points toward -y (the well-reachable direction)
        axis_now = self.bulb_axis
        want = math.atan2(-1.0, 0.0)  # -y
        have = math.atan2(float(axis_now[1]), float(axis_now[0]))
        dpsi_total = (want - have + math.pi) % (2 * math.pi) - math.pi
        n_steps = max(1, int(abs(dpsi_total) / 0.35) + 1)
        for _ in range(n_steps):
            self.act(torch.zeros(3, device=dev), self._down_yaw_rot(dpsi_total / n_steps), GRIP_CLOSED)
        for _ in range(4):
            self.act(torch.zeros(3, device=dev), self._down_yaw_rot(0.0), GRIP_CLOSED)
        # set it down near the default pick spot and release (glass centre ~3 cm up)
        drop = torch.tensor([0.26, 0.25, 0.060 + FINGER_REACH], device=dev)
        self.servo(drop, self.ee_quat.clone(), GRIP_CLOSED, max_steps=60, pos_tol=0.008)
        low = drop.clone()
        low[2] = 0.030 + FINGER_REACH
        self.servo(low, self.ee_quat.clone(), GRIP_CLOSED, max_steps=30, pos_tol=0.006)
        self.hold(GRIP_OPEN, 6)
        up = low.clone()
        up[2] = 0.22
        self.servo(up, self.ee_quat.clone(), GRIP_OPEN, max_steps=30, pos_tol=0.02)
        self.hold(GRIP_OPEN, 10)  # let the bulb settle
        self._log(f"repositioned: bulb at {self.bulb_pos.tolist()}, axis {self.bulb_axis.tolist()}")
        return True

    def reorient(self) -> None:
        """Pitch the hand so the held bulb's axis points up (hand z down), in stages."""
        dev = self.dev
        down = torch.tensor([0.0, 0.0, -1.0], device=dev)
        # rotate about the (horizontal) finger-closing axis so the grip is not twisted
        for _ in range(6):
            axis_now = self.bulb_axis
            cosv = float(axis_now[2])
            if cosv > 0.995:
                break
            hand_y = _quat_apply(self.ee_quat, torch.tensor([0.0, 1.0, 0.0], device=dev))
            # desired hand z: -bulb axis rotated to vertical -> just aim hand z at 'down'
            # step: rotate current hand quat by up to 30 deg toward (bulb axis -> up)
            rot_axis = torch.cross(axis_now, -down, dim=-1)  # axis to bring bulb axis to +z
            n = rot_axis.norm()
            if n < 1e-4:
                break
            rot_axis = rot_axis / n
            ang = min(0.5, math.acos(max(-1.0, min(1.0, cosv))))
            dq = _quat_from_axis_angle(rot_axis, ang)
            target_q = _quat_mul(dq, self.ee_quat)
            keep = self.ee_pos.clone()
            keep[2] = max(float(keep[2]), 0.20)
            self.servo(keep, target_q, GRIP_CLOSED, max_steps=25, pos_tol=0.01, rot_tol=0.05)
        self._log(f"reorient done: bulb axis z={float(self.bulb_axis[2]):.3f}")

    def move_over_socket(self, hover_depth: float = 0.075) -> None:
        """Bring the bulb origin over the socket axis at hover_depth above the socket origin."""
        dev = self.dev
        sock = self.socket_pos
        for _ in range(3):
            # keep the hand orientation that points the bulb axis straight up
            axis_now = self.bulb_axis
            corr = torch.cross(axis_now, torch.tensor([0.0, 0.0, 1.0], device=dev), dim=-1)
            n = float(corr.norm())
            dq = (
                _quat_from_axis_angle(corr / n, min(0.3, math.asin(min(1.0, n))))
                if n > 1e-3
                else torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
            )
            target_q = _quat_mul(dq, self.ee_quat)
            offset = self.ee_pos - self.bulb_pos  # hand relative to bulb origin
            target_p = sock + torch.tensor([0.0, 0.0, hover_depth], device=dev) + offset
            self.servo(target_p, target_q, GRIP_CLOSED, max_steps=80, pos_tol=0.004, rot_tol=0.03, settle=2)
            if self.bulb_xy_err() < 0.004:
                break
        self._log(
            f"over socket: xy_err={self.bulb_xy_err()*1000:.1f}mm depth={self.bulb_depth():.4f} "
            f"axis_z={float(self.bulb_axis[2]):.3f}"
        )

    def descend_to_rest(self) -> None:
        """Lower until the cap rests in the bore (depth stops decreasing)."""
        dev = self.dev
        last = self.bulb_depth()
        stall = 0
        for _ in range(60):
            if self.bulb_depth() <= 0.0365:
                break
            # xy correction on the bulb origin, z down a small step
            off = self.scene._bulb_offsets_in_socket()[0, 0, 0]
            dpos = torch.tensor([-float(off[0]), -float(off[1]), -0.004], device=dev)
            dpos[0:2] = dpos[0:2].clamp(-0.004, 0.004)
            self.act(dpos, self._down_yaw_rot(0.0), GRIP_CLOSED)
            d = self.bulb_depth()
            if d > last - 0.0004:
                stall += 1
                if stall > 6:
                    break
            else:
                stall = 0
            last = d
        self._log(f"rested: depth={self.bulb_depth():.4f} xy={self.bulb_xy_err()*1000:.1f}mm")

    def _down_yaw_rot(self, dpsi: float, max_rot: float = 0.55) -> torch.Tensor:
        """Rotation delta toward the target 'hand z straight down, yaw advanced by
        dpsi' — corrects accumulated tilt every step instead of blindly adding yaw."""
        dev = self.dev
        hy = _quat_apply(self.ee_quat, torch.tensor([0.0, 1.0, 0.0], device=dev))
        c, s = math.cos(dpsi), math.sin(dpsi)
        y_hint = torch.tensor([c * float(hy[0]) - s * float(hy[1]),
                               s * float(hy[0]) + c * float(hy[1]), 0.0], device=dev)
        if y_hint.norm() < 1e-4:
            y_hint = torch.tensor([0.0, 1.0, 0.0], device=dev)
        target = _quat_from_two_axes(torch.tensor([0.0, 0.0, -1.0], device=dev), y_hint)
        rerr = _axis_angle_from_quat(_quat_mul(target, _quat_conj(self.ee_quat)))
        n = float(rerr.norm())
        return rerr if n <= max_rot else rerr * (max_rot / n)

    def _yaw_to(
        self,
        q7_target: float,
        grip: float,
        press: float,
        max_steps: int = 40,
        follow_xy: bool = False,
    ) -> None:
        """Turn about the vertical axis until q7 reaches q7_target, holding the hand
        pointing straight down. Uses the world-z -> q7 sign from prewind (w_per_q7).
        With follow_xy, keep the hand centred on the live bulb axis while turning."""
        dev = self.dev
        best = abs(q7_target - self.q7)
        stuck = 0
        for _ in range(max_steps):
            err = q7_target - self.q7
            if abs(err) < 0.08:
                break
            if abs(err) < best - 0.02:
                best, stuck = abs(err), 0
            else:
                stuck += 1
                if stuck > 14:  # not converging — don't keep spinning the wrist
                    self._log(f"_yaw_to stuck at q7={self.q7:.2f} (target {q7_target:.2f})")
                    break
            dpsi = max(-0.45, min(0.45, self.w_per_q7 * err))
            dpos = torch.tensor([0.0, 0.0, press], device=dev)
            if follow_xy:
                dxy = (self.bulb_pos[:2] - self.ee_pos[:2]).clamp(-0.004, 0.004)
                dpos[0], dpos[1] = float(dxy[0]), float(dxy[1])
            self.act(dpos, self._down_yaw_rot(dpsi), grip)

    def regrasp(self, loose_end: float) -> None:
        """Open to a cage (glass stays surrounded), wind the wrist back in place, re-grip."""
        self.hold(GRIP_CAGE, 4)
        self._yaw_to(loose_end, GRIP_CAGE, press=0.0, max_steps=45, follow_xy=True)
        self.hold(GRIP_CLOSED, 8)

    def prewind(self) -> None:
        """While the bulb is airborne over the socket: discover the yaw->q7 sign and
        wind the wrist to the loose end so the first stroke has full range."""
        dev = self.dev
        zero = torch.zeros(3, device=dev)
        # With the hand pointing straight down, the joint-7 axis is along -z world,
        # so a world yaw of +dpsi moves q7 by ~-dpsi: fixed kinematics, not probed.
        # (A probe here proved unreliable — EMA lag once read the sign backwards.)
        q0 = self.q7
        for _ in range(3):
            self.act(zero, self._down_yaw_rot(-0.3), GRIP_CLOSED)
        self._log(f"prewind probe: q7 {q0:.2f}->{self.q7:.2f} (expect increase)")
        self._yaw_to(-2.55 * self.cw_q7_sign, GRIP_CLOSED, press=0.0, max_steps=60)
        self._log(f"prewound: q7={self.q7:.2f}")

    cw_q7_sign: float = 1.0
    w_per_q7: float = -1.0

    def screw(self, max_cycles: int = 45) -> bool:
        """Ratchet the bulb down: grip+turn CW to the wrist limit, regrasp, repeat."""
        dev = self.dev
        zero = torch.zeros(3, device=dev)
        cw_q7_sign = self.cw_q7_sign
        tight_end = 2.55 * cw_q7_sign  # q7 value to stop a tightening stroke at
        loose_end = -2.40 * cw_q7_sign  # wind-back end (nullspace resists the last bit)

        for cyc in range(max_cycles):
            if self.seated():
                break
            d0 = self.bulb_depth()
            # tightening stroke: press down and turn CW until the wrist limit stalls.
            # NOTE the controller's EMA (0.2) needs ~12 steps after a direction flip
            # before torque builds — the stall check must wait it out.
            q_hist: list[float] = []
            for i in range(60):
                if self.seated():
                    break
                if (tight_end - self.q7) * cw_q7_sign < 0.1:
                    break
                if self.bulb_xy_err() > 0.010 or float(self.bulb_axis[2]) < 0.95:
                    self._log("stroke abort: bulb drifting")
                    break
                q_hist.append(self.q7)
                if len(q_hist) >= 16 and abs(q_hist[-1] - q_hist[-13]) < 0.04:
                    self._log(f"stroke stalled at q7={self.q7:.2f}")
                    break
                # follow the BULB axis (the thread centres it; forcing the socket
                # axis while threaded just wedges the cap), light press when engaged
                dz = -0.0020 if self.bulb_depth() > 0.0340 else -0.0008
                dxy = (self.bulb_pos[:2] - self.ee_pos[:2]).clamp(-0.003, 0.003)
                dpos = torch.tensor([float(dxy[0]), float(dxy[1]), dz], device=dev)
                self.act(dpos, self._down_yaw_rot(-0.5), GRIP_CLOSED)
            d1 = self.bulb_depth()
            self._log(
                f"cycle {cyc}: depth {d0:.4f}->{d1:.4f} xy={self.bulb_xy_err()*1000:.1f}mm "
                f"q7={self.q7:.2f} gap={self.finger_gap:.3f} seated={self.seated()}"
            )
            if self.seated():
                break
            if float(self.bulb_pos[2]) < 0.02 or self.bulb_xy_err() > 0.05:
                self._log("bulb lost from the socket — recovering via a fresh pick")
                return False
            self.regrasp(loose_end)
        ok = self.seated()
        # let go and retreat either way
        self.hold(GRIP_OPEN, 5)
        up = self.ee_pos.clone()
        up[2] = float(up[2]) + 0.12
        self.servo(up, self.ee_quat.clone(), GRIP_OPEN, max_steps=30, pos_tol=0.02)
        return ok

    def bulb_in_socket(self) -> bool:
        return (
            self.bulb_depth() < 0.060
            and self.bulb_xy_err() < 0.020
            and float(self.bulb_axis[2]) > 0.90
        )

    def re_engage(self) -> bool:
        """Grip a bulb standing in the socket (e.g. after a screw() cycle budget ran
        out and the hand retreated): come down around the glass and close."""
        if not self.bulb_in_socket():
            return False
        dev = self.dev
        grip_z = float(self.bulb_pos[2]) + GLASS_EQUATOR_Z + FINGER_REACH
        hy = _quat_apply(self.ee_quat, torch.tensor([0.0, 1.0, 0.0], device=dev))
        y_hint = torch.tensor([float(hy[0]), float(hy[1]), 0.0], device=dev)
        quat = _quat_from_two_axes(torch.tensor([0.0, 0.0, -1.0], device=dev), y_hint)
        above = torch.tensor([float(self.bulb_pos[0]), float(self.bulb_pos[1]), grip_z + 0.05], device=dev)
        self.servo(above, quat, GRIP_OPEN, max_steps=60, pos_tol=0.005)
        down = above.clone()
        down[2] = grip_z
        self.servo(down, quat, GRIP_CAGE, max_steps=40, pos_tol=0.004, max_dpos=0.02, settle=2)
        self.hold(GRIP_CLOSED, 8)
        return True

    def run(self) -> bool:
        self.hold(GRIP_OPEN, 8)  # settle
        ok = False
        for round_ in range(4):  # full-task retries (screw() may lose the bulb)
            if round_ > 0 and self.re_engage():
                self._log(f"round {round_}: re-engaged the bulb in the socket")
            else:
                picked = self.pick()
                tries = 0
                while not picked and tries < 2:
                    if not self.reposition():
                        break
                    picked = self.pick()
                    tries += 1
                if not picked:
                    self._log("giving up on pick")
                    break
                self.reorient()
                self.move_over_socket()
                self.prewind()
                self.move_over_socket()
                self.descend_to_rest()
            ok = self.screw()
            if ok:
                break
            self._log(f"round {round_} failed; retrying")
        self._log(f"finished: seated={self.seated()} steps={self.steps}")
        return self.seated()


def solve(env) -> None:
    task = BulbTask(env, log=print)
    task.run()
