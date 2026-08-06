"""The nut-threading policy, run N-wide on ONE GPU: same machine, per-env theta.

Measured on this scene (bench_width.py, L20): 14.3 ms/step at 1 env, 17.2 ms/step at 100 envs —
0.17 ms/step/env. The physics is nowhere near contact-saturated at this width, so a 100-episode
batch costs one GPU-hour here instead of a hundred. That is what makes the repair round of
Proposal 1 affordable: a new hypothesis costs one pod and 40 minutes.

Fidelity is the whole game, so this is the scalar machine of nut_thread_policy.py transposed, not
a reimplementation: same phases in the same order, same transition tests, same frozen constants,
same closed-loop targets. Every scalar in `st` becomes a per-env array and every `if` becomes a
mask. `--check` runs one env and prints the same trace as the scalar port for comparison.

Per-env theta means the phase DURATIONS differ per env (phase_mult), so the schedule thresholds
are arrays too, and envs sit in different phases at the same step. That is fine: physics steps
once for everyone, and each env reads its own row.
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

OPEN, CLOSE = 0.04, 0.0
HAND_OFFSET = 0.058
GRIP_KP = 8000.0
DOWN_YAW = -0.5
STOP_DZ = 0.005

# phase codes
SETTLE, HOVER, DESCEND, GRASP, LIFT, OVER, LOWER, THREAD, FINISH, DONE = range(10)
# thread sub-state codes
PINCH, SEEK, WIND, OPENJ, REWIND, RECLOSE = range(6)
PHASE_NAME = ["settle", "hover", "descend", "grasp", "lift", "over", "lower", "thread",
              "finish", "done"]
TSTATE_NAME = ["pinch", "seek", "wind", "open", "rewind", "reclose"]


def solve_batch(env, thetas: list[dict], rec=None, max_sec: float = 300.0,
                log_every_s: float = 8.0, verbose: bool = True) -> list[dict]:
    """Run one episode per env, each with its own theta. Returns one verdict per env."""
    dev = env.device
    n = env.num_envs
    assert len(thetas) == n, f"{len(thetas)} thetas for {n} envs"
    scene, robot = env.scene, env.robot
    osc = robot.controller.controllers[0]
    osc._kp = torch.tensor([150.0, 150.0, 150.0, 600.0, 600.0, 600.0], device=dev)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    art = robot.articulation
    ee_idx = art.body_names.index("panda_hand")
    lf_idx = art.body_names.index("panda_leftfinger")
    rf_idx = art.body_names.index("panda_rightfinger")
    fj1 = art.find_joints(["panda_finger_joint1"])[0]
    dim = robot.action_dim
    DECIM = robot.control_period
    CTRL_HZ = 1.0 / (env.dt * DECIM)
    down = quat_from_euler_xyz(*(torch.tensor([v], device=dev)
                                 for v in (3.14159, 0.0, DOWN_YAW)))[0]
    env.reset()

    T = lambda key: torch.tensor([float(t[key]) for t in thetas], device=dev)
    pinch_n, pinch_wind_n = T("pinch_n"), T("pinch_wind_n")
    lean_cmd, lower_mps = T("lean"), T("lower_mps")
    hover_dz, carry_dz = T("hover_dz"), T("carry_dz")
    seek_max = T("seek_deg") * math.pi / 180.0
    sweep = T("sweep_deg") * math.pi / 180.0
    w_land_rad = T("w_land_deg") * math.pi / 180.0
    d_wind, d_rewind = T("wind_rate") / CTRL_HZ, T("rewind_rate") / CTRL_HZ
    pm = T("phase_mult")

    SEC = lambda s: max(1, round(s * CTRL_HZ))
    steps_of = lambda s: torch.clamp((pm * s * CTRL_HZ).round(), min=1)
    T_SETTLE, T_HOVER, T_DESCEND, T_GRASP, T_LIFT, T_OVER = (
        steps_of(s) for s in (0.5, 3.0, 3.0, 1.0, 2.5, 1.8))
    T_PINCH, T_OPEN, T_RECLOSE, T_QUIET = SEC(0.5), SEC(0.4), SEC(0.5), SEC(0.5)
    T_FLAT, T_STALL, T_BOUND, T_RCBAND, T_RCSEARCH = (
        SEC(1.5), SEC(1.5), SEC(2.5), SEC(2.0), SEC(2.5))
    T_FINISH, LOST_FOR, LOG_EVERY = SEC(2.0), SEC(3.0), SEC(log_every_s)

    Z0 = torch.zeros(n, device=dev)
    phase = torch.zeros(n, dtype=torch.long, device=dev)
    tstate = torch.zeros(n, dtype=torch.long, device=dev)
    ph_t = torch.zeros(n, device=dev)
    sub_t = torch.zeros(n, device=dev)          # st["t"]
    stall_t, stall_run, lost_t = Z0.clone(), Z0.clone(), Z0.clone()
    wound, stroke = Z0.clone(), torch.zeros(n, dtype=torch.long, device=dev)
    p_flat, grip_off, z_contact, lower_z = Z0.clone(), Z0.clone(), Z0.clone(), Z0.clone()
    dz_touch, dz_seek0, ee0, w_start = Z0.clone(), Z0.clone(), Z0.clone(), Z0.clone()
    flat_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    rc_at_band = torch.zeros(n, dtype=torch.bool, device=dev)
    caught = torch.zeros(n, dtype=torch.bool, device=dev)
    lost = torch.zeros(n, dtype=torch.bool, device=dev)
    xy_tgt, xy_tgt_g = torch.zeros(n, 2, device=dev), torch.zeros(n, 2, device=dev)
    have_xy = torch.zeros(n, dtype=torch.bool, device=dev)
    have_xy_g = torch.zeros(n, dtype=torch.bool, device=dev)
    hist = torch.zeros(n, T_QUIET, device=dev)
    hist_k = torch.zeros(n, dtype=torch.long, device=dev)
    ee_acc, ee_prev = Z0.clone(), None
    ctrl_steps = Z0.clone()

    def yaw_deg(q):
        ex = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(q.shape[0], 3))
        return torch.atan2(ex[:, 1], ex[:, 0]) * 180.0 / math.pi

    def yaw_about_z(ang):
        z = torch.zeros_like(ang)
        yz = quat_from_euler_xyz(z, z, ang)
        return quat_mul(yz, down.unsqueeze(0).expand(n, 4))

    for i in range(1, SEC(max_sec) + 1):
        p = art.data.body_pos_w[:, ee_idx]
        q = art.data.body_quat_w[:, ee_idx]
        nx = scene.nuts[0].data.root_pos_w
        bx = scene.bolts[0].data.root_pos_w
        gc = 0.5 * (art.data.body_pos_w[:, lf_idx] + art.data.body_pos_w[:, rf_idx])
        gpos = art.data.joint_pos[:, fj1].squeeze(-1)
        dz = (nx - bx)[:, 2]
        lat = (nx - bx)[:, :2].norm(dim=1)

        ee_u = yaw_deg(q)
        if ee_prev is not None:
            ee_acc = ee_acc + torch.remainder(ee_u - ee_prev + 180.0, 360.0) - 180.0
        ee_prev = ee_u

        live = phase < DONE
        if not bool(live.any()):
            break

        # ---- integrators (same gains and the same anti-windup clamps as the scalar port) ------
        in_pick = (phase == HOVER) | (phase == DESCEND)
        fresh_g = in_pick & ~have_xy_g
        xy_tgt_g = torch.where(fresh_g.unsqueeze(1), nx[:, :2], xy_tgt_g)
        have_xy_g = have_xy_g | fresh_g
        step_g = (0.01 * (nx[:, :2] - gc[:, :2])).clamp(-3e-4, 3e-4)
        upd_g = in_pick.unsqueeze(1)
        cand_g = nx[:, :2] + (xy_tgt_g + step_g - nx[:, :2]).clamp(-0.03, 0.03)
        xy_tgt_g = torch.where(upd_g, cand_g, xy_tgt_g)

        in_place = (phase == OVER) | (phase == LOWER)
        fresh = in_place & ~have_xy
        xy_tgt = torch.where(fresh.unsqueeze(1), bx[:, :2], xy_tgt)
        have_xy = have_xy | fresh
        step_p = (0.01 * (bx[:, :2] - nx[:, :2])).clamp(-3e-4, 3e-4)
        cand_p = bx[:, :2] + (xy_tgt + step_p - bx[:, :2]).clamp(-0.025, 0.025)
        xy_tgt = torch.where(in_place.unsqueeze(1), cand_p, xy_tgt)
        # thread phase: the tiny 2e-6 cap (a big clamp at 480 Hz ejects the nut — r6)
        err_t = bx[:, :2] - nx[:, :2]
        thr_upd = (phase == THREAD) & have_xy & (err_t.norm(dim=1) > 5e-4)
        xy_tgt = torch.where(thr_upd.unsqueeze(1),
                             xy_tgt + (0.01 * err_t).clamp(-2e-6, 2e-6), xy_tgt)

        # ---- per-phase position/orientation/grip targets --------------------------------------
        goal = p.clone()
        quat = down.unsqueeze(0).expand(n, 4).clone()
        grip = torch.full((n,), OPEN, device=dev)

        m = phase == HOVER
        if bool(m.any()):
            g = nx + torch.stack([Z0, Z0, HAND_OFFSET + hover_dz], 1)
            g[:, :2] = xy_tgt_g
            goal = torch.where(m.unsqueeze(1), g, goal)

        m = phase == DESCEND
        if bool(m.any()):
            g = nx + torch.stack([Z0, Z0, Z0 + HAND_OFFSET], 1)
            g[:, :2] = xy_tgt_g
            goal = torch.where(m.unsqueeze(1), g, goal)

        m = phase == GRASP
        if bool(m.any()):
            g = nx + torch.stack([Z0, Z0, Z0 + HAND_OFFSET], 1)
            g[:, :2] = torch.where(have_xy_g.unsqueeze(1), xy_tgt_g, g[:, :2])
            goal = torch.where(m.unsqueeze(1), g, goal)
            grip = torch.where(m, torch.full_like(grip, CLOSE), grip)

        m = phase == LIFT
        if bool(m.any()):
            g = bx + torch.stack([Z0, Z0, HAND_OFFSET + carry_dz], 1)
            goal = torch.where(m.unsqueeze(1), g, goal)
            grip = torch.where(m, torch.full_like(grip, CLOSE), grip)

        m = phase == OVER
        if bool(m.any()):
            g = bx + torch.stack([Z0, Z0, HAND_OFFSET + carry_dz], 1)
            g[:, :2] = xy_tgt
            goal = torch.where(m.unsqueeze(1), g, goal)
            grip = torch.where(m, torch.full_like(grip, CLOSE), grip)

        m = phase == LOWER
        if bool(m.any()):
            lower_z = torch.where(m, lower_z - lower_mps / CTRL_HZ, lower_z)
            g = torch.stack([xy_tgt[:, 0], xy_tgt[:, 1], lower_z], 1)
            goal = torch.where(m.unsqueeze(1), g, goal)
            grip = torch.where(m, torch.full_like(grip, CLOSE), grip)

        m = phase == THREAD
        if bool(m.any()):
            soft = (tstate == PINCH) | (tstate == SEEK)
            newtons = torch.where(soft, pinch_n, pinch_wind_n)
            g_thread = p_flat - newtons / GRIP_KP
            wide = p_flat + 0.005
            lean = torch.where((tstate == SEEK) | ((tstate == WIND) & flat_ok),
                               lean_cmd, torch.zeros_like(lean_cmd))
            # wind's flat-contact gate stays soft; open/rewind go wide; reclose is width-gated
            g_thread = torch.where((tstate == WIND) & ~flat_ok, p_flat - pinch_n / GRIP_KP, g_thread)
            g_thread = torch.where((tstate == OPENJ) | (tstate == REWIND), wide, g_thread)
            rc_soft = (tstate == RECLOSE) & rc_at_band & (sub_t <= T_RECLOSE)
            rc_dither = ((tstate == RECLOSE) & rc_at_band & (sub_t > T_RECLOSE)
                         & (gpos > p_flat + 3e-4) & (sub_t < T_RECLOSE + T_RCSEARCH))
            g_thread = torch.where((tstate == RECLOSE) & ~rc_at_band, wide, g_thread)
            g_thread = torch.where(rc_soft | rc_dither, p_flat - pinch_n / GRIP_KP, g_thread)
            wound = torch.where(m & rc_dither, wound - 0.6 / CTRL_HZ, wound)

            zt = nx[:, 2] + grip_off - lean
            swing = ((tstate == OPENJ) | (tstate == REWIND)) & (gpos > p_flat + 3.5e-3)
            zt = torch.where(swing, zt + 0.008, zt)          # frozen: width-gated rewind lift
            zt = torch.maximum(zt, z_contact - 0.021)        # frozen: z floor
            g = torch.stack([xy_tgt[:, 0], xy_tgt[:, 1], zt], 1)
            goal = torch.where(m.unsqueeze(1), g, goal)
            quat = torch.where(m.unsqueeze(1), yaw_about_z(wound), quat)
            grip = torch.where(m, g_thread, grip)

        m = phase == FINISH
        if bool(m.any()):
            zt = torch.minimum(p[:, 2] + 0.03 / CTRL_HZ, z_contact + 0.08)
            g = torch.stack([bx[:, 0], bx[:, 1], zt], 1)
            goal = torch.where(m.unsqueeze(1), g, goal)

        # ---- one servo for everybody ----------------------------------------------------------
        a = torch.zeros(n, dim, device=dev)
        a[:, 0:3] = ((goal - p) / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(quat, quat_conjugate(q))
        a[:, 3:6] = (axis_angle_from_quat(qe) / osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[:, 6:8] = grip.unsqueeze(1)
        a[~live] = 0.0

        render = rec.observe(env, a, PHASE_NAME[int(phase[0])],
                             TSTATE_NAME[int(tstate[0])] if phase[0] == THREAD else "") \
            if rec is not None else False
        env.step(a, render=render)
        if rec is not None:
            rec.capture(env)
        ph_t = ph_t + live.float()
        ctrl_steps = ctrl_steps + live.float()

        # ---- transitions (evaluated on the state the step produced) ---------------------------
        adv = lambda cur, nxt: torch.where(cur, torch.full_like(phase, nxt), phase)

        go = (phase == SETTLE) & (ph_t >= T_SETTLE)
        phase, ph_t = adv(go, HOVER), torch.where(go, torch.zeros_like(ph_t), ph_t)
        go = (phase == HOVER) & (ph_t >= T_HOVER)
        phase, ph_t = adv(go, DESCEND), torch.where(go, torch.zeros_like(ph_t), ph_t)
        go = (phase == DESCEND) & (ph_t >= T_DESCEND)
        phase, ph_t = adv(go, GRASP), torch.where(go, torch.zeros_like(ph_t), ph_t)

        go = (phase == GRASP) & (ph_t >= T_GRASP)
        if bool(go.any()):
            p_flat = torch.where(go, art.data.joint_pos[:, fj1].squeeze(-1), p_flat)
            grip_off = torch.where(go, art.data.body_pos_w[:, ee_idx, 2]
                                   - scene.nuts[0].data.root_pos_w[:, 2], grip_off)
        phase, ph_t = adv(go, LIFT), torch.where(go, torch.zeros_like(ph_t), ph_t)

        go = (phase == LIFT) & (ph_t >= T_LIFT)
        phase, ph_t = adv(go, OVER), torch.where(go, torch.zeros_like(ph_t), ph_t)
        go = (phase == OVER) & (ph_t >= T_OVER)
        lower_z = torch.where(go, art.data.body_pos_w[:, ee_idx, 2], lower_z)
        phase, ph_t = adv(go, LOWER), torch.where(go, torch.zeros_like(ph_t), ph_t)

        # lower -> thread: the quiet-nut touch test, with the same ring buffer
        inl = phase == LOWER
        if bool(inl.any()):
            nz = scene.nuts[0].data.root_pos_w[:, 2]
            hist[torch.arange(n, device=dev), hist_k] = torch.where(inl, nz,
                                                                    hist[torch.arange(n, device=dev), hist_k])
            hist_k = torch.where(inl, (hist_k + 1) % T_QUIET, hist_k)
            filled = ph_t >= T_QUIET
            quiet = filled & ((hist.max(1).values - hist.min(1).values) < 2e-4)
            dz_now = (scene.nuts[0].data.root_pos_w - scene.bolts[0].data.root_pos_w)[:, 2]
            touch = inl & (((quiet & (lower_z < art.data.body_pos_w[:, ee_idx, 2] - 0.002)
                             & (dz_now < 0.035)) | (dz_now < 0.0245)))
            if bool(touch.any()):
                z_contact = torch.where(touch, art.data.body_pos_w[:, ee_idx, 2], z_contact)
                dz_touch = torch.where(touch, dz_now, dz_touch)
                tstate = torch.where(touch, torch.full_like(tstate, PINCH), tstate)
                sub_t = torch.where(touch, torch.zeros_like(sub_t), sub_t)
            phase, ph_t = adv(touch, THREAD), torch.where(touch, torch.zeros_like(ph_t), ph_t)

        # ---- the thread sub-machine ----------------------------------------------------------
        inth = phase == THREAD
        if bool(inth.any()):
            gpos_now = art.data.joint_pos[:, fj1].squeeze(-1)
            dz_now = (scene.nuts[0].data.root_pos_w - scene.bolts[0].data.root_pos_w)[:, 2]
            sub_t = sub_t + (inth & ((tstate == PINCH) | (tstate == OPENJ)
                                     | (tstate == RECLOSE))).float()

            # pinch -> wind (pre-engaged) or seek
            m = inth & (tstate == PINCH) & (sub_t >= T_PINCH)
            pre = m & ((dz_touch - dz_now) > 8e-4)
            caught = caught | pre
            tstate = torch.where(pre, torch.full_like(tstate, WIND), tstate)
            to_seek = m & ~pre
            dz_seek0 = torch.where(to_seek, dz_now, dz_seek0)
            tstate = torch.where(to_seek, torch.full_like(tstate, SEEK), tstate)
            sub_t = torch.where(m, torch.zeros_like(sub_t), sub_t)

            # seek: back-rotate under lean until the start clicks, the nut rises, or the cap
            m = inth & (tstate == SEEK)
            wound = torch.where(m, wound + d_wind, wound)
            hit = m & (((dz_seek0 - dz_now) > 4e-4) | ((dz_now - dz_seek0) > 5e-4))
            caught = caught | hit
            tstate = torch.where(hit | (m & (wound >= seek_max)),
                                 torch.full_like(tstate, WIND), tstate)

            # wind: flat gate, then EE-paced tighten stroke
            m = inth & (tstate == WIND)
            fresh_w = m & (ee0 == 0.0) & ~flat_ok & (sub_t == 0)
            starting = m & ~flat_ok & (sub_t == 0)
            ee0 = torch.where(starting, ee_acc, ee0)
            w_start = torch.where(starting, wound, w_start)
            stall_t = torch.where(starting, torch.zeros_like(stall_t), stall_t)
            stall_run = torch.where(starting, torch.zeros_like(stall_run), stall_run)
            gate = m & ~flat_ok
            sub_t = sub_t + gate.float()
            done_gate = gate & ((gpos_now < p_flat + 5e-4) | (sub_t >= T_FLAT))
            flat_ok = flat_ok | done_gate
            ee0 = torch.where(done_gate, ee_acc, ee0)

            paced = m & flat_ok
            swept = (ee0 - ee_acc) * math.pi / 180.0
            lag = w_start - wound - swept
            stall_run = torch.where(paced, torch.where(lag >= 0.35, stall_run + 1.0,
                                                       torch.zeros_like(stall_run)), stall_run)
            w_end = w_start - sweep
            adv_w = paced & (wound > w_end) & (lag < 0.35)
            wound = torch.where(adv_w, torch.maximum(w_end, wound - d_wind), wound)
            dwell = paced & (wound <= w_end)
            stall_t = torch.where(dwell, stall_t + 1.0, stall_t)
            end = (paced & (stall_run >= T_BOUND)) | (dwell & ((lag < 0.26) | (stall_t >= T_STALL)))
            tstate = torch.where(end, torch.full_like(tstate, OPENJ), tstate)
            sub_t = torch.where(end, torch.zeros_like(sub_t), sub_t)
            ee0 = torch.where(end, torch.zeros_like(ee0), ee0)
            flat_ok = flat_ok & ~end
            stroke = stroke + end.long()

            # open -> rewind -> reclose -> wind
            m = inth & (tstate == OPENJ) & (sub_t >= T_OPEN)
            tstate = torch.where(m, torch.full_like(tstate, REWIND), tstate)
            m = inth & (tstate == REWIND)
            wound = torch.where(m, torch.minimum(w_land_rad, wound + d_rewind), wound)
            land = m & (wound >= w_land_rad)
            tstate = torch.where(land, torch.full_like(tstate, RECLOSE), tstate)
            sub_t = torch.where(land, torch.zeros_like(sub_t), sub_t)
            rc_at_band = rc_at_band & ~land

            m = inth & (tstate == RECLOSE)
            band = m & ~rc_at_band & (((art.data.body_pos_w[:, ee_idx, 2]
                                        - (scene.nuts[0].data.root_pos_w[:, 2] + grip_off)) < 1.5e-3)
                                      | (sub_t >= T_RCBAND))
            rc_at_band = rc_at_band | band
            sub_t = torch.where(band, torch.zeros_like(sub_t), sub_t)
            found = (m & rc_at_band & (sub_t > T_RECLOSE)
                     & ~((gpos_now > p_flat + 3e-4) & (sub_t < T_RECLOSE + T_RCSEARCH)))
            tstate = torch.where(found, torch.full_like(tstate, WIND), tstate)
            sub_t = torch.where(found, torch.zeros_like(sub_t), sub_t)

            # thread -> finish, and the lost-nut stop (orchestration, as in the scalar port)
            fin = inth & (dz_now <= STOP_DZ)
            phase, ph_t = adv(fin, FINISH), torch.where(fin, torch.zeros_like(ph_t), ph_t)
            latn = (scene.nuts[0].data.root_pos_w - scene.bolts[0].data.root_pos_w)[:, :2].norm(dim=1)
            lost_t = torch.where(inth & (latn > 0.020), lost_t + 1.0, torch.zeros_like(lost_t))
            gone = inth & (lost_t >= LOST_FOR)
            lost = lost | gone
            phase = adv(gone, DONE)

        go = (phase == FINISH) & (ph_t >= T_FINISH)
        phase = adv(go, DONE)

        if verbose and i % LOG_EVERY == 0:
            alive = int(live.sum())
            print(f"  {i:6d} live={alive:4d} "
                  + " ".join(f"{PHASE_NAME[k][:4]}={int((phase == k).sum())}"
                             for k in (HOVER, LOWER, THREAD, FINISH, DONE))
                  + f" | seated={int(scene.seated().any(dim=1).sum())}"
                  + f" dz[min/med]={dz.min() * 1e3:.1f}/{dz.median() * 1e3:.1f}mm"
                  + f" lat_med={lat.median() * 1e3:.1f}mm strokes_max={int(stroke.max())}",
                  flush=True)

    off = scene.nuts[0].data.root_pos_w - scene.bolts[0].data.root_pos_w
    succ = scene.success()
    seat = scene.seated()
    return [{
        "index": thetas[k].get("_index", k),
        "success": bool(succ[k].all() if succ[k].ndim else succ[k]),
        "seated": int(seat[k, 0]) if seat.ndim > 1 else int(seat[k]),
        "dz_mm": round(off[k, 2].item() * 1e3, 2),
        "lat_mm": round(off[k, :2].norm().item() * 1e3, 2),
        "strokes": int(stroke[k]),
        "caught": bool(caught[k]),
        "ctrl_steps": int(ctrl_steps[k]),
        "final_phase": PHASE_NAME[int(phase[k])],
        "t_state": TSTATE_NAME[int(tstate[k])],
        "lost_nut": bool(lost[k]),
        "theta": thetas[k],
    } for k in range(n)]
