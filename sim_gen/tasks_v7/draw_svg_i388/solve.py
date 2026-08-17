"""Teleport solution for GantryStampScene (sim_gen task `draw_svg_i388`) — the
task's legitimacy certificate.

There is NOTHING to transport in this task (every body is captive in the
machine or kinematic), so no teleports are used after reset at all: the whole
solution is delivered through contact-dynamics-equivalent applied forces — the
same pushes and presses a Franka would exert on the three handles:

  read    — settle, then read the pad layout and the machine configuration
            back from the scene (frame pose, pad centres frame-local, bridge /
            carriage coordinates).
  aim     — for each RED pad, decompose its centre into machine coordinates
            (bridge target = pad_x - TIP_OFF, carriage target = pad_y) and run
            a dual-axis PD force servo: bounded horizontal forces on the BRIDGE
            (along the frame's x-axis, as a push on the green mast) and on the
            CARRIAGE (along the frame's y-axis, as a push on the yellow knob)
            until both coordinates are within tolerance and slow.
  press   — hold the axes and push the stylus HEAD straight down (~6 N, the
            fingertip press) against its return spring until the scene's
            sustained-press latch fires; release and let the spring retract
            the tip.  Gray pads are simply never pressed.
  finish  — zero every force, hands off; the spring parks the stylus, the
            machine settles, success() turns True.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric is latched), then holds HANDS-OFF >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it
still holds.

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
    from .scene import N_RED, TIP_OFF, _qapply  # noqa: F401
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import N_RED, TIP_OFF, _qapply  # noqa: F401
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# --- servo constants (K*dt/m well under 1; wrench acts one substep late) ---
KP_B, KD_B, FMAX_B = 30.0, 6.0, 8.0  # bridge axis (eff. mass ~1.1 kg)
KP_C, KD_C, FMAX_C = 12.0, 4.0, 5.0  # carriage axis (eff. mass ~0.3 kg)
TOL = 0.004  # in-position tolerance (m); stamp_r is 0.018
V_OK = 0.02  # in-position velocity gate (m/s)
PRESS_F = 6.0  # stylus head press (N); full-stroke spring+weight ~3.5 N


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gantry_stamp")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    ex_l = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3)
    ey_l = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(n, 3)
    zero3 = torch.zeros(n, 1, 3, device=device)

    def wrench(body, force_nx3: torch.Tensor) -> None:
        f = force_nx3.view(n, 1, 3)
        try:
            body.set_external_force_and_torque(f, zero3, body_ids=[0], is_global=True)
        except TypeError:  # older API: world-frame is the default
            body.set_external_force_and_torque(f, zero3, body_ids=[0])

    def hands_off() -> None:
        wrench(scene.bridge, zero3.view(n, 3))
        wrench(scene.carriage, zero3.view(n, 3))
        wrench(scene.stylus, zero3.view(n, 3))

    def servo_step(bx_t: float, cy_t: float, press_f: float,
                   kp_b: float = KP_B, kp_c: float = KP_C) -> tuple[float, float]:
        """One sim step of the dual-axis hold/track servo (+ optional press).
        Returns current (err_b, err_c)."""
        q = scene.frame.data.root_quat_w
        ex, ey = _qapply(q, ex_l), _qapply(q, ey_l)
        bx, cy, _tz = scene.machine_coords()
        vb = (scene.bridge.data.root_lin_vel_w * ex).sum(-1)
        vc = (scene.carriage.data.root_lin_vel_w * ey).sum(-1)
        fb = (kp_b * (bx_t - bx) - KD_B * vb).clamp(-FMAX_B, FMAX_B)
        fc = (kp_c * (cy_t - cy) - KD_C * vc).clamp(-FMAX_C, FMAX_C)
        wrench(scene.bridge, fb.unsqueeze(-1) * ex)
        wrench(scene.carriage, fc.unsqueeze(-1) * ey)
        fz = torch.zeros(n, 3, device=device)
        fz[:, 2] = -press_f
        wrench(scene.stylus, fz)
        env.step(no_action)
        return float((bx_t - bx).abs()[0]), float((cy_t - cy).abs()[0])

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def coords() -> tuple[float, float, float]:
        bx, cy, tz = scene.machine_coords()
        return float(bx[0]), float(cy[0]), float(tz[0])

    def report(tag: str) -> None:
        bx, cy, tz = coords()
        print(f"[solve] {tag:12s} | bridge={bx * 1000:+7.1f}mm car={cy * 1000:+7.1f}mm "
              f"tip_z={tz * 1000:6.1f}mm "
              f"red={int(scene.stamped_red()[0])}/3 foul={bool(scene.fouled()[0])} "
              f"retracted={bool(scene.stylus_retracted()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)
    layout = scene.pad_layout()[0]  # (5,2) frame-local
    yaw = 2.0 * math.atan2(float(scene.frame.data.root_quat_w[0, 3]),
                           float(scene.frame.data.root_quat_w[0, 0]))
    fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
    bx0, cy0, tz0 = coords()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"frame=({float(fp[0]):+.3f},{float(fp[1]):+.3f}) yaw={math.degrees(yaw):+.0f}deg "
          f"bridge0={bx0 * 1000:+.1f}mm car0={cy0 * 1000:+.1f}mm tip_z0={tz0 * 1000:.1f}mm",
          flush=True)
    for i, nm in enumerate(("red0", "red1", "red2", "gray0", "gray1")):
        print(f"[solve]   pad {nm}: frame-local "
              f"({float(layout[i, 0]) * 1000:+7.1f}, {float(layout[i, 1]) * 1000:+7.1f}) mm",
              flush=True)
    report("reset")
    assert bool(scene.stylus_retracted()[0]), "spring must park the stylus retracted"
    assert tz0 > c.pad_top + 0.012, "retracted tip must clear the pads"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (stylus spring-parked, nothing stamped)")
    assert s0 <= 0.01, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1..3: per red pad — aim, press, retract -----------------------
    s_prev = s0
    # visit reds in bridge-coordinate order (shortest total travel; order is free)
    order = sorted(range(N_RED), key=lambda i: float(layout[i, 0]))
    for k, idx in enumerate(order, start=1):
        bx_t = float(layout[idx, 0]) - TIP_OFF
        cy_t = float(layout[idx, 1])

        # --- aim: dual-axis PD servo, gain escalation on stall ---
        kp_b, kp_c = KP_B, KP_C
        good, moved, esc = 0, 0, 0
        while True:
            eb, ec = servo_step(bx_t, cy_t, 0.0, kp_b, kp_c)
            moved += 1
            good = good + 1 if (eb < TOL and ec < TOL
                                and float(scene.bridge.data.root_lin_vel_w[0].norm()) < V_OK
                                and float(scene.carriage.data.root_lin_vel_w[0].norm()) < V_OK
                                ) else 0
            if good >= 12:
                break
            if moved % 300 == 0 and esc < 3:  # stall — escalate GAIN, not the cap
                esc += 1
                kp_b, kp_c = kp_b * 1.5, kp_c * 1.5
                print(f"[solve] pad red{idx}: servo escalation {esc} "
                      f"(err {eb * 1000:.1f}/{ec * 1000:.1f} mm)", flush=True)
            assert moved < 1500, f"servo never converged on red{idx} " \
                                 f"(err {eb * 1000:.1f}/{ec * 1000:.1f} mm)"
        report(f"aim red{idx}")

        # --- press: hold axes, push the head down until the scene latch fires ---
        pressed = 0
        while not bool(scene.stamp_latch[0, idx]):
            servo_step(bx_t, cy_t, PRESS_F, kp_b, kp_c)
            pressed += 1
            assert pressed < 360, f"press on red{idx} never latched " \
                                  f"(tip_z {coords()[2] * 1000:.1f} mm)"
        for _ in range(6):  # dwell a moment at depth (real presses do)
            servo_step(bx_t, cy_t, PRESS_F, kp_b, kp_c)
        assert not bool(scene.fouled()[0]), f"foul latched while pressing red{idx}"

        # --- retract: release the head, keep holding the axes, spring lifts the tip ---
        lifted = 0
        while coords()[2] < c.retract_z + 0.002:
            servo_step(bx_t, cy_t, 0.0, kp_b, kp_c)
            lifted += 1
            assert lifted < 240, "spring failed to retract the stylus"
        report(f"stamped red{idx}")
        s_now = print_score(f"P{k} red pad {idx} stamped (press #{k} of {N_RED}) and released")
        assert s_now >= s_prev - 1e-6, "score decreased across a stamp"
        assert s_now >= 0.25 * k - 1e-6, f"stamp {k} credit missing, got {s_now}"
        s_prev = s_now

    assert int(scene.stamped_red()[0]) == N_RED, "not all red pads latched"
    assert not bool(scene.fouled()[0]), "gray pad fouled"

    # ---------------- phase 4: hands off, settle, success + persistence --------------------
    hands_off()
    step(2)
    hands_off()  # ensure the buffers really are zero before the free run
    for _ in range(20):  # up to ~3.3 s hands-off settling
        if bool(scene.success()[0]):
            break
        step(20)
    report("settled")
    s4 = print_score("P4 goal state: 3 red stamped, no foul, stylus parked, machine still")
    assert s4 >= s_prev - 1e-6, "score decreased across final settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stamping)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for i in range(420):  # 420 steps = 3.5 s at 120 Hz, hands off
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                bx, cy, tz = coords()
                print(f"[solve] persist flicker @step {i}: tip_z={tz * 1000:.1f}mm "
                      f"retracted={bool(scene.stylus_retracted()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"red={int(scene.stamped_red()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s5 = print_score("P-persist persistence 3.5 s (latches hold, spring holds the stylus)")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
