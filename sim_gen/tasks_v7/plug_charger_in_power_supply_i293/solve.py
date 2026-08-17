"""Teleport solution for TotePourDockScene (sim_gen task
`plug_charger_in_power_supply_i293`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY — two carries, both through free space:
  T1 — the tote AND its contents are moved as ONE RIGID ASSEMBLY (the bank's
  tote-local pose is read back and preserved exactly across the write, so no
  interaction is skipped) from the floor to an airborne pour stance beside the mat;
  T2 — the EMPTY tote (freed of the bank, still held in mid-air) is carried to its
  park spot well clear of the mat and set down 3 mm up for a real gravity set-down.
Every load-bearing interaction goes through CONTACT DYNAMICS:
  P2 — POUR: a floating-hand wrench controller (gravity feedforward + pose PD,
  force/torque capped) holds the airborne tote and rotates it about its own rim
  edge past vertical. The bank is NEVER touched: it slides across the real tote
  floor, is caught by the real pour-side wall, rides that wall out of the mouth
  when the tilt beats the friction release angle, free-falls, and lands on the
  floor by itself. Escalation if a seed is sticky: deeper tilt, then a small
  reference shake along the pour direction.
  P4 — NUDGE: a velocity-regulated horizontal push (~1 N against the bank's own
  ground friction, capped at 2.5 N — enough to topple an on-edge bank flat, far
  too weak to flip a flat one) walks the landed bank to the mat centre patch.
The bank is never teleported anywhere, and never teleported out of the tote; the
pour — the interaction the deep unreachable tote exists to force — is pure
gravity-driven contact physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.plug_charger_in_power_supply_i293.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

DT = 1.0 / 120.0
G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tote_pour_dock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import (axis_angle_from_quat, quat_apply,
                                     quat_apply_inverse, quat_inv, quat_mul)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        bl = scene._mat_local(scene.bank.data.root_pos_w)[0]
        tl = scene._mat_local(scene.tote.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | bank_mat_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) tote_mat_local=({float(tl[0]):+.3f},{float(tl[1]):+.3f}) "
              f"tilt={math.degrees(float(scene.tote_tilt()[0])):.1f}deg "
              f"in_tote={bool(scene.bank_in_tote(c.freed_expand)[0])} "
              f"freed={bool(scene.bank_freed()[0])} placed={bool(scene.bank_placed()[0])} "
              f"clear={bool(scene.tote_clear()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_wrench(body, f_world: torch.Tensor, tq_world: torch.Tensor) -> None:
        # Frame law on these pods (hard-won): the DEFAULT call applies the wrench in the
        # body's LIVE frame, and `is_global=True` drags by a stale reset reference — a
        # tilted pour hold under is_global=True turned the restoring force into a
        # constant thruster (runaway exactly along the pour direction). Per-step
        # pre-encode into the live body frame makes effective world == commanded world.
        q_now = body.data.root_quat_w[0:1]
        f_b = quat_apply_inverse(q_now, f_world.view(1, 3))
        t_b = quat_apply_inverse(q_now, tq_world.view(1, 3))
        body.set_external_force_and_torque(
            f_b.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            t_b.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    t0 = (scene.tote.data.root_pos_w - scene.env_origins)[0].clone()
    mp = (scene.mat.data.root_pos_w - scene.env_origins)[0].clone()
    mq = scene.mat.data.root_quat_w[0].clone()
    bl0 = scene._tote_local(scene.bank.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): tote=({float(t0[0]):+.3f},"
          f"{float(t0[1]):+.3f}) mat=({float(mp[0]):+.3f},{float(mp[1]):+.3f}) "
          f"bank_tote_local=({float(bl0[0]):+.4f},{float(bl0[1]):+.4f},{float(bl0[2]):.4f})",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport #1 — one rigid assembly) ----------------
    # Carry the tote WITH the bank inside (relative pose read back and preserved
    # exactly) to the airborne pour stance: tote yawed so its local +y (the pour
    # side) points along d_hat toward the mat, rim pivot 0.10 m short of the mat
    # centre at z = 0.21 (the full body sweep about that pivot — max radius
    # hypot(tote_h, outer_y) = 0.169 m — stays > 4 cm off the ground, clearing
    # even a bank already lying below it, 22 mm thick).
    d = mp[:2] - t0[:2]
    d_hat = d / d.norm().clamp_min(1e-6)
    dx, dy = float(d_hat[0]), float(d_hat[1])
    yaw_pour = math.atan2(dy, dx) - math.pi / 2  # local +y -> d_hat
    q0 = torch.tensor([math.cos(yaw_pour / 2), 0.0, 0.0, math.sin(yaw_pour / 2)],
                      device=device)
    pivot = torch.tensor([float(mp[0]) - 0.10 * dx, float(mp[1]) - 0.10 * dy, 0.21],
                         device=device)
    b0 = torch.tensor([float(pivot[0]) - (c.outer_y / 2) * dx,
                       float(pivot[1]) - (c.outer_y / 2) * dy,
                       float(pivot[2]) - c.tote_h], device=device)
    u_hat = torch.tensor([-dy, dx, 0.0], device=device)  # rotation axis: z-hat x d-hat

    # current rigid relative pose of the bank in the tote (preserved across the write)
    r_rel = scene._tote_local(scene.bank.data.root_pos_w)[0].clone()
    q_rel = quat_mul(quat_inv(scene.tote.data.root_quat_w[0:1]),
                     scene.bank.data.root_quat_w[0:1])[0]

    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b0
    st[:, 3:7] = q0
    st[:, 0:3] += scene.env_origins
    scene.tote.write_root_state_to_sim(st, all_ids)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b0 + quat_apply(q0.unsqueeze(0), r_rel.unsqueeze(0))[0]
    st[:, 3:7] = quat_mul(q0.unsqueeze(0), q_rel.unsqueeze(0))[0]
    st[:, 0:3] += scene.env_origins
    scene.bank.write_root_state_to_sim(st, all_ids)
    rl = scene._tote_local(scene.bank.data.root_pos_w)[0]
    print(f"[solve] transport: assembly carried; bank_tote_local now "
          f"({float(rl[0]):+.4f},{float(rl[1]):+.4f},{float(rl[2]):.4f}) "
          f"(was ({float(r_rel[0]):+.4f},{float(r_rel[1]):+.4f},{float(r_rel[2]):.4f}))",
          flush=True)

    # ---------------- phase 2: POUR (contact dynamics, wrench-held tote) -------------------
    kp, kd, kr, kw = 20.0, 6.0, 1.2, 0.05
    f_cap, tq_cap = 18.0, 0.6
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    def hold_step(phi: float, shake: float = 0.0) -> None:
        half = phi / 2
        q_rot = torch.cat([torch.tensor([math.cos(half)], device=device),
                           math.sin(half) * u_hat]).unsqueeze(0)
        q_ref = quat_mul(q_rot, q0.unsqueeze(0))[0]
        p_ref = pivot + quat_apply(q_rot, (b0 - pivot).unsqueeze(0))[0]
        p_ref = p_ref + shake * torch.tensor([dx, dy, 0.0], device=device)
        p = (scene.tote.data.root_pos_w - scene.env_origins)[0]
        v = scene.tote.data.root_lin_vel_w[0]
        q = scene.tote.data.root_quat_w[0]
        w = scene.tote.data.root_ang_vel_w[0]
        m_eff = c.tote_mass + (0.0 if bool(scene.bank_freed()[0]) else c.bank_mass)
        f = m_eff * G * ez + kp * (p_ref - p) - kd * v
        fn = float(f.norm())
        if fn > f_cap:
            f = f * (f_cap / fn)
        q_err = quat_mul(q_ref.unsqueeze(0), quat_inv(q.unsqueeze(0)))
        rot = axis_angle_from_quat(q_err)[0]
        tq = kr * rot - kw * w
        tn = float(tq.norm())
        if tn > tq_cap:
            tq = tq * (tq_cap / tn)
        apply_wrench(scene.tote, f, tq)
        env.step(no_action)

    # 2a: grab-settle at the stance (upright hold, half a second)
    for _ in range(60):
        hold_step(0.0)
    report("stance")

    # 2b: ramp the tilt to the hold reference; wait for the pour at each stage
    phi, rate = 0.0, 0.0
    rate_max, rate_acc = math.radians(40.0), math.radians(80.0)
    freed_at = None
    for stage, (target_deg, wait_s, shake_amp) in enumerate(
            ((c.pour_hold_deg, 2.0, 0.0), (135.0, 2.0, 0.0), (135.0, 3.0, 0.012))):
        target = math.radians(target_deg)
        while phi < target - 1e-6:
            rate = min(rate + rate_acc * DT, rate_max)
            phi = min(phi + rate * DT, target)
            hold_step(phi)
            if bool(scene.bank_freed()[0]):
                break
        t_shake = 0.0
        while not bool(scene.bank_freed()[0]) and t_shake < wait_s:
            amp = shake_amp * math.sin(2 * math.pi * 2.0 * t_shake) if shake_amp else 0.0
            hold_step(phi, shake=amp)
            t_shake += DT
        if bool(scene.bank_freed()[0]):
            freed_at = math.degrees(phi)
            break
        print(f"[solve] pour stage {stage} done without release (tilt {math.degrees(phi):.1f} "
              f"deg) — escalating", flush=True)
    if freed_at is None:
        clear_wrench(scene.tote)
        print("SIM_GEN_SOLVE: FAIL (bank never poured out)", flush=True)
        os._exit(1)
    print(f"[solve] bank released at tilt {freed_at:.1f} deg", flush=True)

    # 2c: keep holding the tote while the bank free-falls and settles on the floor
    landed = False
    for i in range(360):
        hold_step(phi)
        bz = float((scene.bank.data.root_pos_w - scene.env_origins)[0, 2])
        bv = float(scene.bank.data.root_lin_vel_w[0].norm())
        if bz < 0.05 and bv < 0.10:
            landed = True
            break
    report("poured")
    if not landed:
        clear_wrench(scene.tote)
        print("SIM_GEN_SOLVE: FAIL (poured bank did not settle)", flush=True)
        os._exit(1)
    s2 = print_score("P2 pour (contact)")
    assert s2 >= s0 - 1e-6, "score decreased across the pour"

    # ---------------- phase 3: PARK (teleport #2 — carry the empty tote away) --------------
    # The tote is airborne, held, and empty. Carry it to a spot 0.32 m from the mat
    # centre on the origin side (mat-frame Chebyshev >= 0.32/sqrt(2) = 0.226 >
    # clear_cheb, and it cannot reach the bank), 3 mm up for a real gravity set-down.
    e_hat = -mp[:2] / mp[:2].norm().clamp_min(1e-6)
    park = mp[:2] + 0.32 * e_hat
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(park[0])
    st[:, 1] = float(park[1])
    st[:, 2] = 0.003
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.tote.write_root_state_to_sim(st, all_ids)
    clear_wrench(scene.tote)
    step(90)  # gravity set-down + settle
    report("parked")
    s3 = print_score("P3 park the tote (transport + gravity set-down)")
    assert s3 >= s2 - 1e-6, "score decreased across the park"

    # ---------------- phase 4: NUDGE (contact dynamics) ------------------------------------
    # Velocity-regulated horizontal push on the landed bank toward the mat centre.
    # ~1 N feedforward against its own ground friction, capped at 2.5 N: enough to
    # topple an on-edge bank flat (needs > 0.8 N) and to walk a flat one (needs
    # ~1.0 N), far below anything that could flip a flat bank (> 4.9 N).
    push_ff, v_des = 0.9, 0.06
    best_err, last_gain = 1e9, 0
    done_nudge = False
    for i in range(2400):
        mloc = scene._mat_local(scene.bank.data.root_pos_w)[0]
        err_vec = -mloc[:2]
        err = float(err_vec.norm())
        if err < 0.010 and bool(scene.bank_flat()[0]):
            done_nudge = True
            break
        dir_m = torch.cat([err_vec / max(err, 1e-6), torch.zeros(1, device=device)])
        dir_w = quat_apply(mq.unsqueeze(0), dir_m.unsqueeze(0))[0]
        dir_w[2] = 0.0
        dir_w = dir_w / dir_w.norm().clamp_min(1e-6)
        v = scene.bank.data.root_lin_vel_w[0]
        v_along = float((v * dir_w).sum())
        f_mag = max(0.0, min(2.5, push_ff + c.bank_mass * 40.0 * (v_des - v_along)))
        apply_wrench(scene.bank, dir_w * f_mag, torch.zeros(3, device=device))
        env.step(no_action)
        if err < best_err - 0.001:
            best_err, last_gain = err, i
        elif i - last_gain > 240:  # stalled: lean a little harder
            push_ff = min(push_ff + 0.3, 2.2)
            last_gain = i
            print(f"[solve] nudge stalled at err={err:.4f}, push_ff={push_ff:.2f} N", flush=True)
    clear_wrench(scene.bank)
    step(120)  # release + settle (placed latch needs a settled bank)
    report("nudged")
    print(f"[solve] nudge loop done (done_nudge={done_nudge})", flush=True)
    s4 = print_score("P4 nudge to centre (contact)")
    assert s4 >= s3 - 1e-6, "score decreased across the nudge"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after nudge+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.4 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.4 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.4 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
