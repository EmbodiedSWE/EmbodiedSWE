"""assembly.pc_ram.franka.joint — seat both RAM sticks in their DIMM slots, by differential IK.

Per stick (far slot first): the arm picks it out of its foam holder (the scene's
weld-on-closure grasp contract holds the 7.3 mm blade), carries it over the case wall,
aligns its gold edge over the slot at the align hover, and presses it home. The first stick
is pressed to full depth while gripped. The second, beside its seated neighbour, is
pressed gripped to the pinch's physical floor (~1.3 mm, where the fingertip meets the
neighbour's top edge), released IN PLACE — the aligned blade then slides home under its
own weight through the scene's slick-stick / grippy-case channel — and the fingertip
re-form + seat-press lands on the seated stick as the verification-by-contact (and the
retry path). Every long move glides; per-env biases are learned in free air, gated on
near-convergence.

Servo: closed-loop damped-least-squares differential IK from the MEASURED joint positions
(non-integrating — contact cannot wind up), emitting 9-D joint-position targets (7 arm +
2 finger) at the preset's 48 Hz latch under the stock arm PD. For the data engine this
makes `raw_command` the portable `joint_pos` convention. Phase budgets are authored at the
OSC solve's 15 Hz and scale by 16/control_period at runtime.

Protocol form of the verified run (experiments/2026-08-24_pc_ram_franka_ik, runs 21-23):
module level is import-safe stdlib; all mission logic lives in solve(env); no controller
overrides. The scene's seated() is the graded criterion.
"""

from __future__ import annotations

import math

DT = 1.0 / 240.0  # sim timestep (matches the registered env's dt override)

# Stick-local grasp geometry — identical to the OSC solve (from ram_tridentz.usd /ram/collision).
GRIP_TOP_ZC = 0.0401     # body-slab TOP face above the stick origin (= the visual top edge)
GRIP_XC = -0.00005       # body-slab mid-plane (faces at x -0.0037 / +0.0036)
GRIP_DEPTH = 0.005       # pad-centre depth below the slab top at the pinch
HOVER_CLEAR = 0.05       # pad hover height above the slab top before the descent
FINGER_TO_PAD = 0.045
PAD_TO_TIP = 0.0088
OPEN_W = 0.04
STRADDLE_W = 0.011       # per-finger width while descending AROUND the 7.3 mm slab
GRIP_W = 0.0035          # per-finger closed width (slab half-width minus a 0.15 mm kiss)
GRIP_DEPTH_HI = -0.00575  # SECOND-stick top-edge pinch: pad centre 5.75 mm ABOVE the slab top
CAPTURE_PRESS = 0.0005   # commanded blade-bottom z for the gripped capture press: well past
                         # the gate, so the press keeps a standing lead the whole way down (the
                         # closed-loop press force is proportional to the REMAINING error).
CAPTURE_GATE = 0.0034    # hand off to the fingertips at depth >= 1.25 mm — just above the
                         # gripped press's PHYSICAL floor at ~1.4 mm, where the fingertip
                         # (3.05 mm below the gripped slab top) lands on the SEATED NEIGHBOUR's
                         # top edge (measured stalls 1.27-1.46 mm, runs 4/6/7). The deepest
                         # reachable capture: a freed blade has no upright equilibrium anywhere
                         # the pinch can reach (run 11 trace: heels 0.06 -> 11 deg in 20 ticks
                         # untouched), so the handoff plans FOR the stable ~11 deg wedge rather
                         # than fighting it: offset rise past it, tips onto the live top,
                         # tip-follow press — the channel rights it (proven from 25 deg).
TIP_W = 0.0005           # per-finger width for the fingertip seat-press
CAGE_W = 0.006           # per-finger cage width for righting the freed stick (12 mm sum: above
                         # the weld window top, so the cage never re-welds; sweeps an 11.5 deg
                         # lean back to <= ~3.5 deg)
CLOSED_MIN, CLOSED_MAX = 0.005, 0.010  # finger-joint-sum closure window at the pinch (m)

# Differential-IK servo (replaces the OSC action semantics).
IK_LAMBDA2 = 0.0025      # DLS damping^2 (lambda 0.05)
ERR_LIN = 0.03           # per-tick position error clamp (m): bounds speed AND press force
ERR_ANG = 0.12           # per-tick orientation error clamp (rad)
IK_GAIN = 0.85           # fraction of the (clamped) error closed per tick
DQ_MAX = 0.09            # per-joint step clamp (rad/tick)
NULL_GAIN = 0.02         # nullspace posture pull per tick (toward the cfg elbow posture)
LEARN_GATE = 0.04        # m: integrate a free-air bias only this close to the unbiased target.
# The IK+PD arm LAGS its glide (unlike the OSC torque servo, which is settled by glide-end), and
# mid-flight tracking error poisons the integrator — run 1's carry learner ate ~100 mm of lag,
# saturated its clamp, inflated the goal outside the reachable envelope, and deadlocked there.
BIAS_CAP = 0.08          # bias clamp (m): joint-PD sag is mm-scale; bound the blast radius

# Insertion flight plan — identical to the OSC solve (m; case-frame blade-bottom heights).
ORDER = (1, 0)
CROSS_Z = 0.240
ALIGN_Z = 0.017
PRESS_TGT = -0.002
RESEAT_Z = 0.010
PRESS_DONE = 0.0042
MAX_RETRIES = 2

# Waypoint tolerances (control-rate independent).
TOL_P, TOL_R = 0.004, 0.06
ALIGN_TOL = 0.0004
ALIGN_ROT_TOL = math.radians(1.0)
PICK_RETRIES = 3

# Phase budgets, authored in the OSC solve's 15 Hz control steps; scaled at runtime (see S()).
SHOW_END_15 = 20
WP_TIMEOUT_15, SETTLE_15 = 75, 45
HOVER_15, DOWN_15, CLOSE_15 = 110, 40, 18
LIFT_15, CARRY_15, RETREAT_15 = 60, 100, 50
DROP_15, PRESS_15, PRESS_MAX_15 = 70, 60, 180
REFORM_15, SEAT_15, SEAT_MAX_15 = 42, 40, 120
LOG_EVERY_15 = 45

# Per-batch solve-hyperparameter bands (data_engine sampler grammar; the values above ARE the
# nominal, one set is drawn per batch and written onto these constants before solve(env)).
# Only trajectory SHAPE and timing knobs — never the grip geometry, capture gate, or press leads,
# which are physics-critical. Budgets are the 15 Hz-authored numbers the runtime scaling reads.
SOLVE_PARAMS = {
    "ALIGN_Z": {"dist": "uniform", "lo": 0.014, "hi": 0.020,
                "reason": "align-hover height; lower loses lateral authority, higher lengthens the press"},
    "CROSS_Z": {"dist": "uniform", "lo": 0.240, "hi": 0.270,
                "reason": "rim-crossing height above the 195 mm walls; lo raised 0.220->0.240 — "
                          "crossing at 0.222 stalls the far stick-0 descent at hand z ~0.31 "
                          "(elbow trapped, campaign solve_4 0/4, 2026-08-25); 0.24-0.26 pass"},
    "HOVER_CLEAR": {"dist": "uniform", "lo": 0.044, "hi": 0.060,
                    "reason": "pick approach hover above the slab top; lo raised 0.040->0.044 — "
                              "the 41.5mm/94.5-tick low-corner set missed the far stick-0 pick "
                              "(campaign probe_solve0, 2026-08-25), 43.1mm/99.1 passed"},
    "HOVER_15": {"dist": "uniform", "lo": 100, "hi": 140,
                 "reason": "pick approach glide budget; lo raised 90->100 (same probe finding — "
                           "the stick-0 bias learner needs approach time to converge)"},
    "CARRY_15": {"dist": "uniform", "lo": 80, "hi": 130, "reason": "carry glide budget"},
    "LIFT_15": {"dist": "uniform", "lo": 50, "hi": 80, "reason": "lift glide budget"},
    "DROP_15": {"dist": "uniform", "lo": 55, "hi": 90, "reason": "drop-to-align glide budget"},
}


def solve(env) -> None:
    """Install both sticks on `env` (already built and reset by the harness). Vectorized over
    env.num_envs (lockstep phases, per-env learned biases)."""
    import torch
    from isaaclab.utils.math import (
        axis_angle_from_quat,
        quat_apply,
        quat_apply_inverse,
        quat_conjugate,
        quat_from_angle_axis,
        quat_mul,
    )

    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return (a + math.pi) % (2 * math.pi) - math.pi

    sc = env.scene
    n = env.num_envs
    dev = env.device
    case = sc.case
    art = env.robot.articulation
    hand_idx = art.body_names.index("panda_hand")
    jac_idx = hand_idx - 1  # fixed-base articulation: jacobian rows exclude the root body
    arm_ids = art.find_joints(["panda_joint[1-7]"])[0]
    j7 = art.joint_names.index("panda_joint7")
    fingers = art.find_joints(["panda_finger_joint.*"])[0]
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    eye6 = (IK_LAMBDA2 * torch.eye(6, device=dev)).unsqueeze(0).expand(n, 6, 6)
    eye7 = torch.eye(len(arm_ids), device=dev).unsqueeze(0).expand(n, len(arm_ids), len(arm_ids))
    q_null = torch.tensor(env.robot.cfg.nullspace_dof_pos or env.robot.cfg.default_dof_pos,
                          device=dev).expand(n, len(arm_ids))

    # Budgets scale from the OSC solve's 15 Hz authoring rate to the live joint-mode rate.
    period = env.robot.control_period  # physics substeps per control step (5 -> 48 Hz)
    scale = 16.0 / period

    def S(x: float) -> int:
        return max(1, int(round(x * scale)))

    ctrl_hz = 1.0 / (period * DT)
    # Contact-critical phases (press/reform/seatpress) are TICK-invariant, not wall-invariant
    # (native-20 fix, 2026-08-28): their carrier steps, standing press lead, and trim cadences
    # were tuned per-tick at 48 Hz (period 5); at lower rates they run proportionally slower in
    # wall time so the per-tick geometry against contact is unchanged. Factor 1 at 48 Hz.
    contact_slow = max(1.0, period / 5.0)

    def C(x: float) -> int:
        return S(x * contact_slow)

    SHOW_END = S(SHOW_END_15)
    WP_TIMEOUT, SETTLE_STEPS = S(WP_TIMEOUT_15), S(SETTLE_15)
    HOVER_STEPS, DOWN_STEPS, CLOSE_STEPS = S(HOVER_15), S(DOWN_15), S(CLOSE_15)
    LIFT_STEPS, CARRY_STEPS, RETREAT_STEPS = S(LIFT_15), S(CARRY_15), S(RETREAT_15)
    DROP_STEPS, PRESS_STEPS, PRESS_MAX = S(DROP_15), C(PRESS_15), C(PRESS_MAX_15)
    REFORM_STEPS, SEAT_STEPS, SEAT_MAX = C(REFORM_15), C(SEAT_15), C(SEAT_MAX_15)
    LOG_EVERY = S(LOG_EVERY_15)
    # Free-air bias integrators: per-tick gains divide by the rate scale (same per-second dynamics
    # as the OSC solve); cadenced learners keep their gain and scale the cadence instead.
    G_HOVER, G_DOWN, G_CARRY, G_DROP, G_ALIGN = (g / scale for g in (0.3, 0.25, 0.25, 0.2, 0.1))
    G_ROT, G_ROT_ALIGN = 0.2 / scale, 0.05 / scale

    case_pos = case.data.root_pos_w.clone()  # (n, 3): origin ON the board face, at its centre
    board_z = case_pos[:, 2].clone()
    seats_w = [case_pos + torch.tensor(p, device=dev) for p in sc.cfg.seat_pos]  # per-slot, world

    seq = 0                 # index into ORDER; k = ORDER[seq] is the active stick/slot
    k = ORDER[seq]

    def ram():  # the active stick's asset
        return sc.rams[k]

    def smoothstep(t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        return t * t * (3 - 2 * t)

    print(env.describe(), flush=True)
    print(f"joint-mode control: period {period} substeps -> {ctrl_hz:.1f} Hz, budgets x{scale:.2f}", flush=True)

    # Measure hand-frame -> finger-pad-centre once from the live articulation.
    lf = art.body_names.index("panda_leftfinger")
    _hq0 = art.data.body_quat_w[:, hand_idx]
    _rel = quat_apply_inverse(_hq0, art.data.body_pos_w[:, lf] - art.data.body_pos_w[:, hand_idx])
    hand_to_pad = float(_rel[0, 2]) + FINGER_TO_PAD
    hand_to_tip = hand_to_pad + PAD_TO_TIP
    print(f"hand->pad centre: {hand_to_pad:.4f} m", flush=True)

    # ----- grasp transform (the physical weld is the SCENE contract's) ----------------------------
    rel_p = torch.zeros(n, 3, device=dev)  # stick pose in the hand frame, captured at grasp time
    rel_q = torch.zeros(n, 4, device=dev)
    welded = torch.zeros(n, dtype=torch.bool, device=dev)

    def hand_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return art.data.body_pos_w[:, hand_idx], art.data.body_quat_w[:, hand_idx]

    def weld_on() -> None:
        held = bool(sc.grasp_held[:, k].all())
        assert held, f"scene grasp-weld contract did not engage on stick {k}"
        hp, hq = hand_pose()
        rel_p[:] = quat_apply_inverse(hq, ram().data.root_pos_w - hp)
        rel_q[:] = quat_mul(quat_conjugate(hq), ram().data.root_quat_w)
        welded[:] = True

    def weld_off() -> None:
        welded[:] = False  # physically released by the scene contract as the fingers open

    # ----- servo layer: waypoints -> differential-IK joint action ---------------------------------
    def _clamp_norm(v: torch.Tensor, cap: float) -> torch.Tensor:
        nrm = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return v * (nrm.clamp(max=cap) / nrm)

    def servo(tp: torch.Tensor, tq: torch.Tensor, grip, null_pull: bool = True) -> torch.Tensor:
        """9-D action toward a hand waypoint: DLS differential-IK joint targets + finger targets.
        Closed loop from the MEASURED joint positions (non-integrating): at contact the standing
        command offset — and so the press force — is bounded by the error clamps.
        `null_pull=False` for the contact-critical phases (press/reform/seatpress): the posture
        pull's self-motion is EE-preserving only to FIRST order, and its measured second-order
        creep (~30 um/tick — 1.4 mm across run 12's reform freeze+open) dragged the welded stick
        off the funnel cradle before the weld cut, seeding the fall."""
        hp, hq = hand_pose()
        e_p = _clamp_norm(tp - hp, ERR_LIN)
        qe = quat_mul(tq, quat_conjugate(hq))
        qe = torch.where(qe[:, :1] >= 0, qe, -qe)
        e_r = _clamp_norm(axis_angle_from_quat(qe), ERR_ANG)
        err = torch.cat([e_p, e_r], dim=-1).unsqueeze(-1)          # (n, 6, 1), world frame
        J = art.root_physx_view.get_jacobians()[:, jac_idx, :, :][:, :, arm_ids]  # (n, 6, 7), world
        Jt = J.transpose(1, 2)
        pinv_err = Jt @ torch.linalg.solve(J @ Jt + eye6, err)     # (n, 7, 1)  DLS
        q = art.data.joint_pos[:, arm_ids]
        dq = IK_GAIN * pinv_err
        if null_pull:
            dq = dq + (eye7 - Jt @ torch.linalg.solve(J @ Jt + eye6, J)) @ (
                NULL_GAIN * (q_null - q)).unsqueeze(-1)            # posture pull, task-null only
        dq = dq.squeeze(-1).clamp(-DQ_MAX, DQ_MAX)
        q_des = q + dq
        lims = art.data.soft_joint_pos_limits[:, arm_ids]          # (n, 7, 2)
        q_des = q_des.clamp(lims[..., 0], lims[..., 1])
        a = torch.zeros(n, 9, device=dev)
        a[:, 0:7] = q_des
        a[:, 7:9] = grip
        return a

    def at(tp: torch.Tensor, tq: torch.Tensor) -> torch.Tensor:
        hp, hq = hand_pose()
        qe = quat_mul(tq, quat_conjugate(hq))
        ang = 2.0 * torch.arccos(qe[:, 0].abs().clamp(max=1.0))
        return ((tp - hp).norm(dim=-1) < TOL_P) & (ang < TOL_R)

    def hand_for_stick(kp: torch.Tensor, kq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hand waypoint that puts the WELDED stick at pose (kp, kq), via the captured grasp frame."""
        hq = quat_mul(kq, quat_conjugate(rel_q))
        return kp - quat_apply(hq, rel_p), hq

    def q_down(yaw: torch.Tensor) -> torch.Tensor:
        """Top-down hand orientation (approach -z) with the given world yaw."""
        flip = torch.zeros(n, 4, device=dev)
        flip[:, 1] = 1.0  # 180 deg about x: hand +z -> world -z
        return quat_mul(quat_from_angle_axis(yaw, ez), flip)

    def j7_after(cand: torch.Tensor) -> torch.Tensor:
        """Predicted joint-7 angle after the shortest-path yaw to `cand` (near a top-down hand)."""
        ex = torch.zeros(n, 3, device=dev)
        ex[:, 0] = 1.0
        hx = quat_apply(hand_pose()[1], ex)
        cur = torch.atan2(hx[:, 1], hx[:, 0])
        return art.data.joint_pos[:, j7] - _wrap(cand - cur)

    def nearest_parity(psi: torch.Tensor) -> torch.Tensor:
        """The gripper is 180-deg symmetric: psi or psi-pi, whichever centres the wrist roll."""
        alt = _wrap(psi - math.pi)
        return torch.where(j7_after(psi).abs() <= j7_after(alt).abs(), psi, alt)

    def grip_point() -> torch.Tensor:
        """World grip point on the ACTIVE stick: the body slab's TOP face centre."""
        off = torch.tensor([GRIP_XC, 0.0, GRIP_TOP_ZC], device=dev).expand(n, 3)
        return ram().data.root_pos_w + quat_apply(ram().data.root_quat_w, off)

    def pad_centre() -> torch.Tensor:
        """World centre of the closed finger pads, along the hand's approach axis."""
        hp, hq = hand_pose()
        return hp + quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)) * hand_to_pad

    def depth() -> torch.Tensor:  # blade depth below the ACTIVE slot's mouth (m), per env
        return sc.engaged()[:, k]

    def xy_err() -> torch.Tensor:
        return (ram().data.root_pos_w[:, 0:2] - seats_w[k][:, 0:2]).norm(dim=-1)

    def rot_err() -> torch.Tensor:
        """Axis-angle error norm from the ACTIVE stick's orientation to seated (world identity)."""
        return axis_angle_from_quat(quat_conjugate(ram().data.root_quat_w)).norm(dim=-1)

    def step(action: torch.Tensor) -> None:
        env.step(action)

    # ----- run ------------------------------------------------------------------------------------
    home_p, home_q = None, None
    picks = 0
    retries = 0
    pos_off = torch.zeros(n, 3, device=dev)   # INTEGRATED free-air bias (desired - achieved): the
    # joint PD sags under gravity like the OSC did; commands near contact add this learned offset
    rot_bias = torch.zeros(n, 3, device=dev)  # the same for ORIENTATION (axis-angle, free air)
    grip_freeze = torch.zeros(n, 2, device=dev)  # finger targets latched at release
    grip_pt = torch.zeros(n, 3, device=dev)
    grip_yaw = torch.zeros(n, device=dev)
    close_ok = torch.zeros(n, device=dev)  # consecutive ticks the closure has verified
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)
    hover_from = torch.zeros(n, 3, device=dev)
    hover_yaw0 = torch.zeros(n, device=dev)
    carry_from = torch.zeros(n, 3, device=dev)
    retreat_from = torch.zeros(n, 3, device=dev)
    retreat_q0 = torch.zeros(n, 4, device=dev)
    retreat_aa = torch.zeros(n, 3, device=dev)
    lift_xy = torch.zeros(n, 2, device=dev)
    lift_from = torch.zeros(n, device=dev)
    drop_from = torch.zeros(n, device=dev)
    release_p = torch.zeros(n, 3, device=dev)
    release_q = torch.zeros(n, 4, device=dev)
    press_from = torch.zeros(n, device=dev)
    reform_p = torch.zeros(n, 3, device=dev)
    reform_q = torch.zeros(n, 4, device=dev)
    seat_from = torch.zeros(n, device=dev)

    def upright_cmd() -> torch.Tensor:
        """Commanded stick orientation: upright, pre-rotated by the learned droop bias."""
        ang = rot_bias.norm(dim=-1).clamp_min(1e-9)
        return quat_from_angle_axis(ang, rot_bias / ang.unsqueeze(-1))

    def learn_pos(want: torch.Tensor, achieved: torch.Tensor, gain: float) -> torch.Tensor:
        """Integrate (want - achieved) into `pos_off`, gated on NEAR-convergence (LEARN_GATE) so
        the bias learns steady-state sag only, never glide lag. Returns the per-env gate."""
        e = want - achieved
        near = e.norm(dim=-1, keepdim=True) < LEARN_GATE
        pos_off[:] = (pos_off + gain * torch.where(near, e, torch.zeros_like(e))).clamp(-BIAS_CAP, BIAS_CAP)
        return near.squeeze(-1)

    def learn_xy(want_xy: torch.Tensor, achieved_xy: torch.Tensor, gain: float) -> None:
        """The xy-only trim variant of `learn_pos` (contact-adjacent phases trim xy only)."""
        e = want_xy - achieved_xy
        near = e.norm(dim=-1, keepdim=True) < LEARN_GATE
        pos_off[:, 0:2] = (pos_off[:, 0:2]
                           + gain * torch.where(near, e, torch.zeros_like(e))).clamp(-BIAS_CAP, BIAS_CAP)

    def learn_rot(gain: float, near: torch.Tensor | None = None) -> None:
        """Integrate the stick's residual orientation error (free air, near-converged only)."""
        r = axis_angle_from_quat(quat_conjugate(ram().data.root_quat_w))
        if near is not None:
            r = torch.where(near.unsqueeze(-1), r, torch.zeros_like(r))
        rot_bias[:] = (rot_bias + gain * r).clamp(-0.2, 0.2)

    phase, marker = "show", 0
    reform_close_t = -1  # tick the reform's tip-close ramp began (-1 = still converging)
    rise_off = torch.zeros(n, 2, device=dev)  # reform rise offset, away from the live lean
    i = 0
    while True:
        i += 1
        t_in = i - marker
        hp, hq = hand_pose()
        if home_p is None:
            home_p, home_q = hp.clone(), hq.clone()
        act = servo(home_p, home_q, OPEN_W)  # default: hold home, fingers open
        gd = GRIP_DEPTH if seq == 0 else GRIP_DEPTH_HI  # second stick: top-edge pinch

        if phase == "show":
            if t_in >= SHOW_END:
                phase, marker = "pick_hover", i
                picks += 1
        elif phase == "pick_hover":  # glide to a top-down hover over the active standing stick
            if t_in == 1:
                pos_off.zero_()  # fresh site, fresh bias
                hover_from[:] = hp
                ky = quat_apply(ram().data.root_quat_w, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
                ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
                hx = quat_apply(hq, ex)
                hover_yaw0[:] = torch.atan2(hx[:, 1], hx[:, 0])
                grip_yaw[:] = nearest_parity(torch.atan2(ky[:, 1], ky[:, 0]))  # fingers across the slab
            grip_pt[:] = grip_point()
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR
            s = smoothstep(t_in / HOVER_STEPS)
            wp_q[:] = q_down(hover_yaw0 + _wrap(grip_yaw - hover_yaw0) * s)
            if s >= 1.0:  # arrived, free air: learn the pad-centre bias for the descent
                want = grip_pt.clone()
                want[:, 2] += HOVER_CLEAR
                learn_pos(want, pad_centre(), G_HOVER)
            goal = wp_p + pos_off
            tp = hover_from + (goal - hover_from) * s
            # Rim-safe transit (native-20 fix, 2026-08-28): the straight slot->holder glide
            # grazes the case wall with the fingertips (48 Hz log: gap 22->12.4->22 mid-hover;
            # at 20 Hz the extra latch sag catches the rim and jams the pick). Hold the carry
            # height until the xy leg is nearly done, then descend; also makes a retry's first
            # move UP and out of any jam.
            far = (goal[:, 0:2] - hp[:, 0:2]).norm(dim=-1) > 0.03
            safe = board_z + CROSS_Z + hand_to_pad
            tp[:, 2] = torch.where(far, torch.maximum(tp[:, 2], safe), tp[:, 2])
            act = servo(tp, wp_q, STRADDLE_W)
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= HOVER_STEPS + S(10) and bool((pad_err < 0.004).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= HOVER_STEPS + 2 * WP_TIMEOUT:
                phase, marker = "pick_down", i
        elif phase == "pick_down":  # descend AROUND the stick until the pads flank the grip band
            s = smoothstep(t_in / DOWN_STEPS)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR - s * (HOVER_CLEAR + gd)
            if s >= 1.0:
                want = grip_pt.clone()
                want[:, 2] -= gd
                learn_pos(want, pad_centre(), G_DOWN)
            act = servo(wp_p + pos_off, wp_q, STRADDLE_W)
            want = grip_pt.clone()
            want[:, 2] -= gd
            pad_on = (pad_centre() - want).norm(dim=-1) < 0.0015  # on-band before the close: the
            # top-edge pinch has only ~3 mm of pad engagement to spare (run 2 wedged at 3 mm)
            if (t_in >= DOWN_STEPS + S(8) and bool(pad_on.all())) or t_in >= 2 * WP_TIMEOUT:
                close_ok.zero_()
                phase, marker = "pick_close", i
        elif phase == "pick_close":  # close on the slab; the scene weld engages on verified closure
            s = smoothstep(t_in / CLOSE_STEPS)
            width = STRADDLE_W + (GRIP_W - STRADDLE_W) * s
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            want = grip_pt.clone()
            want[:, 2] -= gd
            learn_pos(want, pad_centre(), G_DOWN)  # keep trimming through the close: the IK's
            # proportional loop carries a standing mm-residual (run 2: the shallow top-edge pinch
            # wedged on the slab's top corners at gap 10.3-10.8 mm from ~2 mm of pad height error)
            act = servo(wp_p + pos_off, wp_q, width)
            near = (pad_centre() - want).norm(dim=-1) < 0.004
            ok = near & (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
            close_ok[:] = torch.where(ok, close_ok + 1, torch.zeros_like(close_ok))
            if t_in >= CLOSE_STEPS + S(6) and bool((close_ok >= S(4)).all()):
                weld_on()
                print(f"  grasped stick {k}: pads on the slab faces, finger gap "
                      f"{[f'{float(g)*1e3:.1f}' for g in gap]} mm (slab 7.3), pad err "
                      f"{float((pad_centre() - want).norm(dim=-1).max()) * 1e3:.1f} mm", flush=True)
                phase, marker = "lift", i
            elif t_in >= CLOSE_STEPS + S(30):
                if picks < PICK_RETRIES * len(ORDER):
                    print(f"  pick of stick {k} missed (gap {[f'{float(g)*1e3:.1f}' for g in gap]} mm), "
                          f"retrying", flush=True)
                    phase, marker = "pick_hover", i
                    picks += 1
                else:
                    print("  ABORT: pick failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "lift":  # straight up out of the holder to rim-crossing height (glided)
            if t_in == 1:
                lift_xy[:] = ram().data.root_pos_w[:, 0:2]
                lift_from[:] = ram().data.root_pos_w[:, 2]
            s = smoothstep(t_in / LIFT_STEPS)
            kp = ram().data.root_pos_w.clone()
            kp[:, 0:2] = lift_xy
            kp[:, 2] = lift_from + s * (board_z + CROSS_Z - lift_from)
            tp, tq = hand_for_stick(kp, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            if (t_in >= LIFT_STEPS and bool(((board_z + CROSS_Z - ram().data.root_pos_w[:, 2]).abs() < 0.01).all())) \
                    or t_in >= LIFT_STEPS + S(45):
                pos_off.zero_()  # pick-spot bias is stale here; re-learn on the carry
                phase, marker = "carry", i
        elif phase == "carry":  # translate to above the active slot, at crossing height (glided)
            if t_in == 1:
                carry_from[:] = ram().data.root_pos_w
            goal = seats_w[k].clone()
            goal[:, 2] = board_z + CROSS_Z
            s = smoothstep(t_in / CARRY_STEPS)
            if s >= 1.0:  # arrived, free air near the case: learn the stick-frame biases
                near = learn_pos(goal, ram().data.root_pos_w, G_CARRY)
                learn_rot(G_ROT, near)
            kp = carry_from + (goal - carry_from) * s
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            arrived = (ram().data.root_pos_w[:, 0:2] - seats_w[k][:, 0:2]).norm(dim=-1) < 0.003
            if (t_in >= CARRY_STEPS + S(5) and bool(arrived.all())) or t_in >= CARRY_STEPS + WP_TIMEOUT:
                drop_from[:] = ram().data.root_pos_w[:, 2]
                phase, marker = "drop", i
        elif phase == "drop":  # descend over the slot to this sequence's release hover
            s = smoothstep(t_in / DROP_STEPS)
            kp = seats_w[k].clone()
            kp[:, 2] = drop_from + s * (board_z + ALIGN_Z - drop_from)
            if s >= 1.0:  # learn only once the target is stationary
                near = learn_pos(kp, ram().data.root_pos_w, G_DROP)
                learn_rot(G_ROT, near)
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            if t_in >= DROP_STEPS + S(15):
                phase, marker = "align", i
        elif phase == "align":  # settle at the align hover: the 0.4 mm gate beats the funnel
            kp = seats_w[k].clone()
            kp[:, 2] = board_z + ALIGN_Z
            near = learn_pos(kp, ram().data.root_pos_w, G_ALIGN)
            learn_rot(G_ROT_ALIGN, near)  # gentle: each correction sways the stick
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            still = ram().data.root_lin_vel_w.norm(dim=-1) < 0.01
            z_ok = (ram().data.root_pos_w[:, 2] - (board_z + ALIGN_Z)).abs() < 0.001
            ok = (xy_err() < ALIGN_TOL) & z_ok & (rot_err() < ALIGN_ROT_TOL) & still
            if bool(ok.all()) or t_in >= 6 * WP_TIMEOUT:
                print(f"  aligned stick {k}: xy err {float(xy_err().max()) * 1e3:.2f} mm, rot "
                      f"{float(torch.rad2deg(rot_err()).max()):.2f} deg", flush=True)
                press_from[:] = ram().data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "press":  # straight down, STILL GRIPPED (stick 0 to full depth; stick 1 to
            # the slot mouth's capture height, then the fingertips take over)
            s = smoothstep(t_in / PRESS_STEPS)
            z_end = board_z + sc.cfg.seat_pos[k][2] + (PRESS_TGT if seq == 0 else CAPTURE_PRESS)
            kp = seats_w[k].clone()
            kp[:, 2] = press_from + s * (z_end - press_from)
            if t_in % C(3) == 0 and s < 0.6:  # free air until the blade meets the mouth: keep the
                # xy trim live AND re-learn the orientation droop at the PRESS pose — it differs
                # from the align pose's (run 6: a 0.47 deg free-air tilt at the outer slot jammed
                # the 1.6 mm blade 1.4 mm into the 1.9 mm channel, twice, exactly where the
                # parallel grip begins; stick 1 pressed through at 0.05 deg). Also makes the
                # press RETRIES self-correcting: each re-entry re-converges the bias.
                learn_xy(kp[:, 0:2], ram().data.root_pos_w[:, 0:2], 0.1)
                near = (kp - ram().data.root_pos_w).norm(dim=-1) < LEARN_GATE
                learn_rot(0.2, near)  # cadenced (S(3)): gain stays, cadence scales
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            bz = ram().data.root_pos_w[:, 2] - (board_z + sc.cfg.seat_pos[k][2])
            if seq == 0 and bool((depth() >= PRESS_DONE).all()):
                phase, marker = "release", i
            elif seq > 0 and t_in >= C(10) and bool((bz <= CAPTURE_GATE).all()) \
                    and bool((xy_err() < 0.0015).all()) and bool((rot_err() < 0.035).all()):
                # xy/tilt guards (2026-08-28): depth alone false-captured a blade that skated
                # off the funnel and heeled over the wall (20 Hz gate v2: xy 4.8 mm, rot 0.9 deg
                # at "capture"); a miss now rides to PRESS_MAX and the reseat retry re-converges.
                print(f"  stick {k} captured: blade {float(bz.mean() * 1e3):.2f} mm in the mouth, "
                      f"handing over to the fingertips", flush=True)
                phase, marker = "reform", i
            elif t_in >= PRESS_MAX:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"  WARN: press on stick {k} stalled at {float(depth().mean() * 1e3):+.2f} mm "
                          f"— retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reseat", i
                else:
                    print(f"  WARN: press on stick {k} exhausted its retries — releasing as-is", flush=True)
                    phase, marker = "release", i
        elif phase == "reseat":  # rise back to just above the mouth, still gripped, and re-press
            kp = seats_w[k].clone()
            kp[:, 2] = board_z + RESEAT_Z
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            if t_in >= WP_TIMEOUT // 2:
                press_from[:] = ram().data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "reform":  # bleed the press, let go, re-form into a two-fingertip press
            if t_in == 1:
                grip_freeze[:] = art.data.joint_pos[:, fingers]
                reform_p[:] = hp
                reform_q[:] = hq
            if t_in <= C(6):
                act = servo(reform_p, reform_q, grip_freeze, null_pull=False)
            elif t_in <= C(20):  # open IN PLACE — the weld cuts at aperture 18 mm, so the open
                # must FINISH before any rise: run 9 staged the open ACROSS the rise and the
                # still-welded stick was lifted ~7 mm out of its 1.3 mm capture before the cut,
                # then dropped and spun to the deterministic 90-deg rest. Opening with the hand
                # motionless cuts the weld while the channel walls still hold the blade.
                if welded.any():
                    weld_off()
                s = smoothstep((t_in - C(6)) / C(10))
                act = servo(reform_p, reform_q, grip_freeze + (STRADDLE_W - grip_freeze) * s, null_pull=False)
            elif t_in <= C(34):  # OFFSET RISE, 3.5 mm to the ANTI-LEAN side: the freed stick
                # heels to its stable ~11 deg wedge (top edge at ~11.6 mm, past the STRADDLE
                # corridor the seated neighbour caps at 11 mm), so a centred rise hooks it
                # (runs 16-18: seated-or-flat on render timing), a cage-guided rise lifts it
                # out (run 19), and an end-slide walked it out (run 20). Rising shifted AWAY
                # from the measured live lean clears the leaning top on the lean side and the
                # near-upright face on the other. The offset's component TOWARD the seated
                # neighbour (+x) is capped at 0.5 mm (finger outer 14.3 vs neighbour 15.26).
                if t_in == C(20) + 1:
                    d_xy = grip_point()[:, 0:2] - ram().data.root_pos_w[:, 0:2]
                    rise_off[:] = -0.0035 * d_xy / d_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    rise_off[:, 0] = rise_off[:, 0].clamp(max=0.0005)
                s = smoothstep((t_in - C(20)) / C(14))
                wp_p[:] = reform_p
                wp_p[:, 0:2] = reform_p[:, 0:2] + s * rise_off
                wp_p[:, 2] = reform_p[:, 2] + s * 0.008
                act = servo(wp_p, reform_q, STRADDLE_W, null_pull=False)
            else:  # hover the tips 1.5 mm over the top edge, CONVERGE IN 3D, then close. Closing
                # while the glide is still a millimetre low clamps the 7.3 mm slab in the 1 mm
                # tip gap and SPINS the barely-captured stick out about the blade line (runs 3+8
                # both ended rot 90.01 deg — the torsional watermelon-seed; the z droop at this
                # pose is unlearned, so the hover must learn it BEFORE any close).
                if t_in == C(34) + 1:
                    hover_from[:] = wp_p
                    reform_close_t = -1
                tip_p = grip_point()
                tip_p[:, 2] = tip_p[:, 2] + 0.0015 + hand_to_tip
                s = smoothstep((t_in - C(34)) / (C(52) - C(34)))  # glide end shifted by
                # the in-place-open + offset-rise stages; the hover approaches the live
                # (wedge-leaning) top FROM ABOVE and tracks it
                wp_p[:] = hover_from + (tip_p - hover_from) * s
                if s >= 1.0 and t_in % C(2) == 0:  # full-3D trim: z is the ejection axis
                    learn_pos(tip_p, hp, 0.15)
                on_tgt = bool(((tip_p - hp).norm(dim=-1) < 0.0012).all())
                if reform_close_t < 0:
                    w = STRADDLE_W  # tips stay wide until the hover has truly converged
                    if s >= 1.0 and on_tgt:
                        reform_close_t = t_in
                else:
                    sc_ = smoothstep((t_in - reform_close_t) / C(14))
                    w = STRADDLE_W + (TIP_W - STRADDLE_W) * sc_
                act = servo(wp_p + pos_off, reform_q, w, null_pull=False)
                if (reform_close_t > 0 and t_in >= reform_close_t + C(16) and on_tgt) \
                        or t_in >= 3 * WP_TIMEOUT:
                    phase, marker = "seatpress", i
        elif phase == "seatpress":  # fingertips down on the top edge until the blade bottoms out
            if t_in == 1:
                seat_from[:] = hp[:, 2]
            z_end = board_z + sc.cfg.seat_pos[k][2] + GRIP_TOP_ZC + PRESS_TGT + hand_to_tip
            s = smoothstep(t_in / SEAT_STEPS)
            wp_p[:, 0:2] = grip_point()[:, 0:2]  # FOLLOW the live top: press straight down on
            # wherever the (possibly wedge-leaning) top actually is — the CHANNEL rights the
            # blade as it descends and the top converges over the slot by itself. Targeting the
            # seat's xy instead sweeps the tips off the 7.3 mm edge righting a >5 deg lean
            # (run 14), and makes the press depend on the exact post-release lean.
            wp_p[:, 2] = seat_from + s * (z_end - seat_from)
            if t_in % C(2) == 0:
                learn_xy(wp_p[:, 0:2], hp[:, 0:2], 0.1)
            cmd = wp_p.clone()
            cmd[:, 0:2] += pos_off[:, 0:2]
            act = servo(cmd, reform_q, TIP_W, null_pull=False)
            if bool((depth() >= PRESS_DONE).all()):
                phase, marker = "release", i
            elif t_in >= SEAT_MAX:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"  WARN: seat-press on stick {k} stalled at {float(depth().mean() * 1e3):+.2f} mm "
                          f"— retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reform", i
                else:
                    print(f"  WARN: seat-press on stick {k} exhausted its retries — releasing as-is", flush=True)
                    phase, marker = "release", i
        elif phase == "release":  # bleed the stored press, let go, rise clear (then peel west)
            if t_in == 1:
                grip_freeze[:] = art.data.joint_pos[:, fingers]
                release_p[:] = hp
                release_q[:] = hq
            if t_in <= S(8):
                act = servo(release_p, release_q, grip_freeze)
            else:
                if welded.any():
                    weld_off()
                wp2 = release_p.clone()
                if t_in > S(14):  # rise first (glided): the fingers back off the seated top edge
                    s2 = smoothstep((t_in - S(14)) / S(16))
                    wp2[:, 2] = release_p[:, 2] + s2 * 0.045
                if t_in > S(26):  # then peel WEST while still rising
                    s3 = smoothstep((t_in - S(26)) / S(14))
                    wp2[:, 0] = release_p[:, 0] - s3 * 0.035
                act = servo(wp2, release_q, STRADDLE_W if t_in > S(20) else grip_freeze)
            if t_in >= S(44):
                phase, marker = "clear", i
        elif phase == "clear":  # rise straight off the seated stick, then the next stick/retreat
            if t_in == 1:
                drop_from[:] = hp[:, 2]
                wp_p[:] = hp  # hold the peeled-away xy; rise only
            s = smoothstep(t_in / S(30))
            kp = wp_p.clone()
            kp[:, 2] = drop_from + s * (board_z + CROSS_Z + hand_to_tip - drop_from)
            act = servo(kp, release_q, OPEN_W)
            if t_in >= S(34):
                print(f"  stick {k} pressed: depth {float(depth().mean() * 1e3):+.2f} mm, xy "
                      f"{float(xy_err().mean() * 1e3):.2f} mm", flush=True)
                seq += 1
                if seq < len(ORDER):
                    k = ORDER[seq]
                    retries = 0
                    picks += 1
                    phase, marker = "pick_hover", i
                else:
                    phase, marker = "retreat", i
        elif phase == "retreat":  # glide home — position AND orientation
            if t_in == 1:
                if welded.any():
                    weld_off()
                retreat_from[:] = hp
                retreat_q0[:] = hq
                qe = quat_mul(home_q, quat_conjugate(hq))
                qe = torch.where(qe[:, :1] >= 0, qe, -qe)
                retreat_aa[:] = axis_angle_from_quat(qe)
            s = smoothstep(t_in / RETREAT_STEPS)
            ang = retreat_aa.norm(dim=-1).clamp_min(1e-9)
            q_cmd = quat_mul(quat_from_angle_axis(ang * s, retreat_aa / ang.unsqueeze(-1)), retreat_q0)
            act = servo(retreat_from + (home_p - retreat_from) * s, q_cmd, OPEN_W)
            if t_in >= RETREAT_STEPS + S(10):
                phase, marker = "settle", i
        else:  # settle: hands off — both seated sticks must hold on their own
            act = servo(home_p, home_q, OPEN_W)
            if t_in >= SETTLE_STEPS:
                break

        step(act)

        bad = not torch.isfinite(ram().data.root_pos_w).all()
        if bad:
            print("  ABORT: stick state went non-finite", flush=True)
            break
        if i % LOG_EVERY == 0:
            gap = art.data.joint_pos[:, fingers].sum(dim=-1) * 1e3
            print(f"  ctrl {i:4d} [{phase:9s} stick {k}] | depth {float(depth().mean() * 1e3):+6.2f}mm | "
                  f"xy err {float(xy_err().mean() * 1e3):6.2f}mm | rot {float(torch.rad2deg(rot_err()).max()):5.2f}deg | "
                  f"stick z {float(ram().data.root_pos_w[:, 2].mean()):.3f} | hand z {float(hp[:, 2].mean()):.3f} | "
                  f"gap {float(gap.mean()):4.1f}mm | bias ({float(pos_off[0, 0]) * 1e3:+.0f},"
                  f"{float(pos_off[0, 1]) * 1e3:+.0f},{float(pos_off[0, 2]) * 1e3:+.0f})mm | "
                  f"j4 {float(art.data.joint_pos[0, arm_ids[3]]):+.2f}", flush=True)

    seated = sc.seated()  # (n, S) — re-checked for BOTH sticks after the final settle
    all_ok = seated.all(dim=1)
    depths = sc.engaged()
    per_slot = " | ".join(
        f"slot{j}: depth {float(depths[:, j].mean() * 1e3):+.2f} mm, xy "
        f"{float((sc.rams[j].data.root_pos_w[:, 0:2] - seats_w[j][:, 0:2]).norm(dim=-1).mean() * 1e3):.2f} mm"
        for j in range(sc.cfg.num_slots)
    )
    print(f"  scene grasp-weld holds at end: {int(sc.grasp_held.sum())} (expect 0)", flush=True)
    print(f"PC-RAM-FRANKA-IK | seated {int(all_ok.sum())}/{n} envs ({int(seated.sum())}/{n * sc.cfg.num_slots} "
          f"sticks) | stroke 4.44, seat >= {sc.cfg.seat_depth * 1e3:.1f} | {per_slot} | "
          f"{picks} picks, {retries} press retries", flush=True)


if __name__ == "__main__":
    # Dev-only harness: build the graded preset, reset, run solve(env). The graded run imports
    # this module and calls solve(env) on its own env.
    import argparse

    from isaaclab.app import AppLauncher

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(_parser)
    _args = _parser.parse_args()
    _args.headless = True
    _app = AppLauncher(_args).app

    import torch

    import robobench
    from robobench.core import ENVS

    robobench.discover()
    _env = ENVS.get("assembly.pc_ram.franka.joint")().build(
        num_envs=_args.num_envs, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    _env.reset()
    solve(_env)

    import os

    os._exit(0)
