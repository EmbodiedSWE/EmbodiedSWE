"""Walking solution for SteleWalkScene (sim_gen task
`living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_i419`)
— the task's legitimacy certificate, with ZERO teleports of anything.

The LEFT stele is WALKED the way movers walk furniture, entirely through contact
dynamics driven by an external wrench (the sanctioned proxy for two hands on the
stone): a PD orientation servo tips the stele a few degrees onto one base edge
(never past the 26.6-deg edge-topple line, far short of the 30-deg foul), yaws it
about the planted corner so the raised side swings forward, sets it down, and
alternates sides. Each half-step advances ~2-3 cm and steps the base over the
20 mm kerbs and the 12 mm socket threshold (a 15-deg tip raises the free edge
31 mm). A small fixture-frame trim force (<= 2 N, ~0.09 g lateral) keeps it in
lane — it can never lift or drag the 2.2 kg stone, and the scene's own foul
detectors (carried / toppled, evaluated every substep) police the walk.

Prints `SIM_GEN_SCORE <s>` whenever a progress latch fires (non-decreasing by
construction), then holds >= 3.3 simulated seconds fully hands-off and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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

threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ---- gait constants ----------------------------------------------------------------------------
# Lean direction d = (pitch, 0.6*s): the stele pivots on the base corner it leans toward.
# REAR lean (pitch -0.8) plants a rear corner and raises the WHOLE front edge >= 25 mm
# (clears the 20 mm kerb + 12 mm threshold); FORWARD lean (+0.8) plants a front corner
# and raises the whole rear edge — used only when the rear edge is the one at an obstacle.
TILT_MAX = math.radians(15.0)   # walking tip (far corner up 44 mm, near free corner 25 mm)
PITCH = 0.8                     # |pitch| of the unit lean direction; side = 0.6*s
KP, KD = 14.0, 1.0              # orientation PD gains (N*m/rad, N*m*s/rad)
TAU_MAX = 3.0                   # torque cap (N*m) — tips 1.29 N*m static margin, can't fling
FF_TILT = 0.9                   # gravity feed-forward along the lean axis at full tip (N*m)
F_TRIM = 2.0                    # lateral lane-trim force cap (N) ~ 0.09 g sideways
K_TRIM = 3.0                    # N per m of lane error
PSI_AMP = 0.60                  # commanded yaw swing (rad); contact lag yields ~0.4 achieved
PSI_AMP_SOCK = 0.35             # smaller swings near/inside the walled socket
PSI_STEER = 0.30                # heading clamp toward the lane (rad)
K_STEER = 1.6                   # heading per m of lane error
N_TIP, N_SWING, N_DOWN, N_SETTLE = 30, 55, 25, 20   # substeps per phase
MAX_HALF_STEPS = 64


def main() -> None:
    from isaaclab.utils.math import (axis_angle_from_quat, quat_apply, quat_apply_inverse,
                                     quat_from_angle_axis, quat_inv, quat_mul)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stele_walk")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    env.reset(seed=args.seed)  # seed AFTER build (reseed trap)
    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    tgt = scene.stele_a if bool(scene.tgt_is_a[0]) else scene.stele_b

    # fixture-x windows of the step obstacles: kerb 1, kerb 2, socket threshold
    obstacles = tuple((kx - c.kerb_w / 2, kx + c.kerb_w / 2) for kx in c.kerb_x) + (
        (c.sock_x - c.sock_in / 2 - c.wall_t, c.sock_x - c.sock_in / 2),)
    x_sock_zone = c.sock_x - 0.16    # damp swings from here on (walled approach)

    def rz(psi: float) -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 3] = math.cos(psi / 2), math.sin(psi / 2)
        return q

    def state():
        """fixture-frame (x, y), body yaw rel. fixture, tilt deg, min corner clear."""
        tf = scene.target_fix()[0]
        q_rel = quat_mul(quat_inv(scene.cway.data.root_quat_w), tgt.data.root_quat_w)[0]
        w, x, y, z = (float(v) for v in q_rel)
        psi = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(scene.target_tilt_cos()[0])))))
        return float(tf[0]), float(tf[1]), psi, tilt, float(scene.min_corner_clear()[0])

    def y_lane(x: float) -> float:
        return y0 * max(0.0, 1.0 - x / 0.35)

    def ctl_step(psi_cmd: float, theta: float, s: float, pitch: float,
                 fx: float = 0.0) -> None:
        """One physics substep under the gait wrench (PD orientation + trim force)."""
        q_fix = scene.cway.data.root_quat_w
        d = torch.tensor([pitch, 0.6 * s, 0.0], device=device)  # unit: |(0.8, 0.6)| = 1
        ax = torch.tensor([-float(d[1]), float(d[0]), 0.0], device=device).expand(n, 3)
        q_tilt = quat_from_angle_axis(
            torch.full((n,), theta, device=device), ax)
        q_des = quat_mul(q_fix, quat_mul(rz(psi_cmd), q_tilt))
        rot = axis_angle_from_quat(quat_mul(q_des, quat_inv(tgt.data.root_quat_w)))
        tau = KP * rot - KD * tgt.data.root_ang_vel_w
        # gravity feed-forward along the world lean axis
        if theta > 1e-4:
            ax_w = quat_apply(quat_mul(q_fix, rz(psi_cmd)), ax)
            tau = tau + FF_TILT * (theta / TILT_MAX) * ax_w
        nrm = tau.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tau = tau * (nrm.clamp(max=TAU_MAX) / nrm)
        # fixture-frame trim force (lane keeping + optional forward assist)
        tf = scene.target_fix()
        y_err = y_lane(float(tf[0, 0])) - tf[:, 1]
        f_fix = torch.zeros(n, 3, device=device)
        f_fix[:, 0] = fx
        f_fix[:, 1] = (K_TRIM * y_err).clamp(-F_TRIM, F_TRIM)
        f_w = quat_apply(q_fix, f_fix)
        tgt.set_external_force_and_torque(f_w.view(n, 1, 3), tau.view(n, 1, 3),
                                          env_ids=all_ids, is_global=True)
        env.step(no_action)

    def free_step(k: int) -> None:
        z3 = torch.zeros(n, 1, 3, device=device)
        tgt.set_external_force_and_torque(z3, z3, env_ids=all_ids, is_global=True)
        for _ in range(k):
            env.step(no_action)

    prev_score = 0.0

    def maybe_score(tag: str) -> None:
        nonlocal prev_score
        s = float(scene.score()[0])
        if s > prev_score + 1e-6:
            print(f"[solve] phase boundary: {tag}", flush=True)
            print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
            assert s >= prev_score, "score decreased"
            prev_score = s

    def fail(msg: str) -> None:
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        os._exit(1)

    # ---------------- P0: settle + layout readback ----------------------------------------
    free_step(60)
    cwp = (scene.cway.data.root_pos_w - scene.env_origins)[0]
    qf = scene.cway.data.root_quat_w[0]
    cyaw = math.degrees(2.0 * math.atan2(float(qf[3]), float(qf[0])))
    x, y, psi, tilt, clr = state()
    dfix = scene._fix_local(scene.decoy_pos())[0]
    print(f"[solve] layout (seed {args.seed}): cway=({float(cwp[0]):+.3f},"
          f"{float(cwp[1]):+.3f}) yaw={cyaw:+.1f}deg target={'A' if bool(scene.tgt_is_a[0]) else 'B'}"
          f" tgt_fix=({x:+.3f},{y:+.3f}) decoy_fix=({float(dfix[0]):+.3f},{float(dfix[1]):+.3f})"
          f" score={float(scene.score()[0]):.3f}", flush=True)
    assert y > 0.04, "target must have spawned on the LEFT (fixture +y)"
    assert float(dfix[1]) < -0.04, "decoy must be on the RIGHT"
    print(f"SIM_GEN_SCORE {float(scene.score()[0]):.4f}", flush=True)
    y0 = y

    # ---------------- P1..: the walk -------------------------------------------------------
    side = 1.0          # first half-step: swing CCW -> tip the LEFT edge
    hist_x = [x]
    hist: list[tuple[float, float]] = []   # (x, psi) at half-step START, for wedge detect
    for hs in range(MAX_HALF_STEPS):
        if bool(scene.fouled[0]):
            fail(f"fouled during half-step {hs}")
        x, y, psi, tilt, clr = state()
        if bool(scene.target_in_socket()[0]):
            break
        # lean mode: rear lean by default (front edge flies over obstacles); forward
        # lean when the REAR edge is the one at an obstacle and the front is past it
        cfx = scene.target_corners_fix()[0, :, 0]
        rear_x, front_x = float(cfx.min()), float(cfx.max())
        # a blocked FRONT edge always wins (rear lean lifts it); only when the front is
        # clear and the REAR edge is the one at an obstacle do we lean forward
        front_block = any(a - 0.025 < front_x < b + 0.01 for a, b in obstacles)
        rear_block = any(a - 0.03 < rear_x < b + 0.015 for a, b in obstacles)
        fwd_lean = rear_block and not front_block
        pitch = PITCH if fwd_lean else -PITCH
        # heading toward the lane, alternating swing around it
        amp = PSI_AMP_SOCK if x > x_sock_zone else PSI_AMP
        psi_star = max(-PSI_STEER, min(PSI_STEER, K_STEER * (y_lane(x) - y)))
        if x > x_sock_zone:
            psi_star = max(-0.12, min(0.12, psi_star))
        dpsi = (psi_star + side * amp / 2) - psi
        if abs(dpsi) < 0.12:            # degenerate swing: force the alternation
            dpsi = side * 0.18
        s = 1.0 if dpsi > 0 else -1.0   # pivot rule: CCW swing -> plant the LEFT corner
        # forward assist if the walk has stalled; hard wedge -> back-off half-step
        # (rear lean, swing the yaw square, pull gently backward to free the base)
        fx = 2.5 if (len(hist_x) >= 8 and x - hist_x[-8] < 0.01) else 0.0
        if len(hist) >= 2 and all(abs(x - hx) < 0.002 and abs(psi - hp) < 0.03
                                  for hx, hp in hist[-2:]):
            print(f"[solve] hs{hs:02d} WEDGED -> back-off", flush=True)
            pitch, fwd_lean = -PITCH, False
            dpsi = -psi                 # square the yaw while backing out
            s = 1.0 if dpsi > 0 else -1.0
            fx = -3.5
        hist.append((x, psi))
        # TIP: ramp the lean on at fixed yaw
        for i in range(N_TIP):
            ctl_step(psi, TILT_MAX * (i + 1) / N_TIP, s, pitch, fx)
        # SWING: hold the lean, ramp yaw through dpsi (pivot about the planted corner)
        for i in range(N_SWING):
            ctl_step(psi + dpsi * (i + 1) / N_SWING, TILT_MAX, s, pitch, fx)
        # DOWN: ramp the lean off at the new yaw
        for i in range(N_DOWN):
            ctl_step(psi + dpsi, TILT_MAX * (1.0 - (i + 1) / N_DOWN), s, pitch, 0.0)
        # SETTLE: hands off
        free_step(N_SETTLE)
        side = -side
        x, y, psi, tilt, clr = state()
        hist_x.append(x)
        print(f"[solve] hs{hs:02d} {'FWD' if fwd_lean else 'REAR'} x={x:+.3f} y={y:+.3f} "
              f"rx={rear_x:+.3f} fxx={front_x:+.3f} psi={math.degrees(psi):+5.1f} "
              f"tilt={tilt:4.1f} clr={clr * 1000:5.1f}mm "
              f"dep={int(scene.dep_latch[0])} k1={int(scene.k1_latch[0])} "
              f"k2={int(scene.k2_latch[0])} ap={int(scene.apron_latch[0])} "
              f"score={float(scene.score()[0]):.2f}", flush=True)
        maybe_score(f"half-step {hs}")
    else:
        fail("walk budget exhausted before the socket")

    # ---------------- final settle + persistence ------------------------------------------
    free_step(120)
    if bool(scene.fouled[0]):
        fail("fouled at arrival")
    if not bool(scene.success()[0]):
        free_step(240)
    if not bool(scene.success()[0]):
        x, y, psi, tilt, clr = state()
        print(f"[solve] arrival state x={x:+.3f} y={y:+.3f} tilt={tilt:.1f} "
              f"in_socket={bool(scene.target_in_socket()[0])} "
              f"decoy_in={bool(scene.decoy_in_socket()[0])} "
              f"settled={bool(scene.settled()[0])}", flush=True)
        fail("no success at the socket")
    maybe_score("arrived: left stele standing in the socket")

    hold = True
    for _ in range(10):                 # 400 substeps = 3.33 s hands-off
        free_step(40)
        hold = hold and bool(scene.success()[0])
    s4 = float(scene.score()[0])
    print(f"SIM_GEN_SCORE {s4:.4f}", flush=True)
    ok = hold and bool(scene.success()[0]) and s4 >= prev_score - 1e-6
    print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL (did not persist)",
          flush=True)

    code = 0 if ok else 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
