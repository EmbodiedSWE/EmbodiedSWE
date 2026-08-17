"""Teleport solution for RiddleTrayScene (sim_gen task `sweep_to_dustpan_i333`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY: each present cube is carried (pose write through free
space) to a hover above the open crate and DROPPED in — the settling is contact
physics. The load-bearing interaction — getting the marbles into the sealed hopper —
is executed through the tray mechanism, every time:

1. TILT DISCHARGE (applied hinge torque + rolling contact): the tray is driven
   about its dock hinge by a torque servo (gravity feedforward from the measured
   payload + PD, capped at 3.5 N*m — the moment a fingertip pressing the red paddle
   at 0.18 m produces with ~20 N of margin). The bed tilts to the upper stop, every
   marble ROLLS down the bed, under the canopy, out through the 28 mm end slot, and
   falls into the hopper mouth; the cubes could not follow even if they were still
   aboard (slot 28 mm < cube 32 mm, friction holds them regardless). If a marble
   straggles, the servo ROCKS the tray (drop to a shallow angle and re-tilt) — the
   riddling motion — until the hopper latch has fired for every present marble.
   No marble is ever teleported into, over, or near the hopper.
2. RELEASE: the servo ramps the tray back down and clears the torque; gravity seats
   it on its rest stop. Success requires the tray back at rest — holding it tilted
   forever cannot finish the episode.
3. NO teleport ever enters the hopper or crosses a wall: cube hover targets sit in
   free air above the open crate mouth.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.sweep_to_dustpan_i333.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.riddle_tray")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_tray_torque() -> None:
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def angle() -> float:
        return float(scene.tray_angle_deg()[0])

    def report(tag: str) -> None:
        binned = scene.balls_in_hopper()[0].tolist()
        crated = scene.cubes_in_crate()[0].tolist()
        print(f"[solve] {tag:16s} | tray={angle():+6.2f}deg "
              f"pb={scene.present_ball[0].tolist()} pc={scene.present_cube[0].tolist()} "
              f"binned={binned} crated={crated} "
              f"ball_l={scene._ball_l[0].tolist()} cube_l={scene._cube_l[0].tolist()} "
              f"tilt_l={bool(scene._tilt[0])} rest={bool(scene.tray_at_rest()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback --------------------------------
    step(180)
    pb = scene.present_ball[0].tolist()
    pc = scene.present_cube[0].tolist()
    kb = int(scene.present_ball[0].sum())
    kc = int(scene.present_cube[0].sum())
    print(f"[solve] layout readback (seed {args.seed}): balls={pb} (kb={kb}) "
          f"cubes={pc} (kc={kc}) tray={angle():+.2f}deg", flush=True)
    try:
        m_tray = float(scene.tray.root_physx_view.get_masses()[0].sum())
        print(f"[solve] tray mass readback: {m_tray:.3f} kg", flush=True)
    except Exception:  # noqa: BLE001
        m_tray = c.tray_mass
    for name, body in list(scene.balls.items()) + list(scene.cubes.items()):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   {name}: ({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f})", flush=True)
    cp = (scene.crate.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve]   crate: ({float(cp[0]):+.3f},{float(cp[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert c.rest_deg - 1.5 < angle() < c.rest_max_deg, \
        f"tray must rest reclined on its stop, got {angle():+.2f} deg"
    assert kb >= c.min_balls and kc >= c.min_cubes, "counts below the sampled minima"
    assert int(scene.balls_in_hopper()[0].sum()) == 0, "no marble may start in the hopper"
    assert int(scene.cubes_in_crate()[0].sum()) == 0, "no cube may start in the crate"
    aboard0 = (scene.balls_on_tray()[0] & scene.present_ball[0]).sum()
    assert int(aboard0) == kb, "every present marble must start on the tray bed"
    s_prev = print_score("P0 reset+settle (mixed load on the reclined tray)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.03, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: pick each present cube into the crate ------------------------
    crate_slots = ((-0.045, -0.045), (0.045, -0.045), (0.0, 0.045))
    placed = 0
    for j, (name, body) in enumerate(scene.cubes.items()):
        if not pc[j]:
            continue
        ok = False
        for attempt in range(3):
            sx, sy = crate_slots[placed]
            tgt = torch.tensor([sx, sy, 0.0], device=device).expand(n, 3)
            xy = scene.crate.data.root_pos_w + quat_apply(scene.crate.data.root_quat_w, tgt)
            st = torch.zeros(n, 13, device=device)
            st[:, 0:2] = xy[:, 0:2]
            # hover above the open crate mouth (free air), env-origin aware
            st[:, 2] = scene.env_origins[:, 2] + 0.10 + 0.01 * attempt
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, all_ids)
            step(80)
            loc = scene._local(scene.crate, body.data.root_pos_w)[0]
            v = float(body.data.root_lin_vel_w[0].norm())
            ok = bool(scene.cubes_in_crate()[0, j]) and v < 0.10
            print(f"[solve]   {name} drop -> crate loc=({float(loc[0]):+.3f},"
                  f"{float(loc[1]):+.3f},{float(loc[2]):+.3f}) v={v:.3f} in={ok}", flush=True)
            if ok:
                break
        if not ok:
            print("SIM_GEN_SOLVE: FAIL (cube did not settle in the crate)", flush=True)
            os._exit(1)
        assert bool(scene._cube_l[0, j]), f"crate latch must have fired for {name}"
        placed += 1
        s = print_score(f"P1.{placed} {name} dropped into the crate ({placed}/{kc})")
        assert s >= s_prev - 1e-6, f"score decreased: {s_prev} -> {s}"
        s_prev = s

    # ---------------- phase 2: paddle-press tilt discharge ----------------------------------
    # Torque servo about the world-y hinge axis: gravity feedforward from the MEASURED
    # payload + PD. The rotation axis is y itself, so a y-torque is immune to any
    # wrench frame-mode drift. Cap 3.5 N*m ~ a 19 N fingertip press at the paddle.
    g = 9.81
    kp, kd, tau_max = 12.0, 0.5, 3.5
    tilt_hi = c.tilt_max_deg - 0.5
    ramp_steps = 48  # fast ramp (0.4 s): the tilt latch arms while marbles still ride

    def payload_ff(th_rad: float) -> float:
        """Hold torque: tray CoM + every piece currently riding the bed."""
        tau = m_tray * (-c.tray_com[0]) * g * math.cos(th_rad)
        bloc, _ = scene._family(scene.balls, scene.tray)
        onb = scene.balls_on_tray()[0] & scene.present_ball[0]
        tau += float((-bloc[0, :, 0] * onb.float()).clamp(min=0).sum()) * c.ball_mass * g
        cloc, _ = scene._family(scene.cubes, scene.tray)
        onc = (cloc[0, :, 0] > -(c.floor_len)) & (cloc[0, :, 0] < 0.0) \
            & (cloc[0, :, 2] > -0.005) & (cloc[0, :, 2] < 0.06) & scene.present_cube[0]
        tau += float((-cloc[0, :, 0] * onc.float()).clamp(min=0).sum()) * c.cube_mass * g
        return tau

    def servo_step(theta_des_deg: float) -> None:
        th = math.radians(angle())
        wy = float(scene.tray.data.root_ang_vel_w[0, 1])
        tau = payload_ff(th) + kp * (math.radians(theta_des_deg) - th) - kd * wy
        tau = max(-tau_max, min(tau_max, tau))
        t = torch.zeros(n, 1, 3, device=device)
        t[0, 0, 1] = tau
        scene.tray.set_external_force_and_torque(zero_wrench, t, env_ids=all_ids,
                                                 is_global=True)
        env.step(no_action)

    def all_binned() -> bool:
        return bool(((scene._ball_l[0] | ~scene.present_ball[0])).all())

    max_theta = angle()
    # ramp up
    for i in range(ramp_steps):
        servo_step(c.rest_deg + (tilt_hi - c.rest_deg) * (i + 1) / ramp_steps)
        max_theta = max(max_theta, angle())
    # hold at full tilt until every present marble's hopper latch fires; rock if stuck
    held, rocks = 0, 0
    while not all_binned():
        servo_step(tilt_hi)
        max_theta = max(max_theta, angle())
        held += 1
        if held % 120 == 0:
            report(f"hold t+{held}")
        if held % 360 == 0:
            rocks += 1
            if rocks > 3:
                break
            print(f"[solve]   straggler: rock cycle {rocks} (drop to 4 deg, re-tilt)",
                  flush=True)
            for i in range(60):
                servo_step(tilt_hi + (4.0 - tilt_hi) * (i + 1) / 60)
            for _ in range(60):
                servo_step(4.0)
            for i in range(30):
                servo_step(4.0 + (tilt_hi - 4.0) * (i + 1) / 30)
    report("discharged")
    assert max_theta > c.tilt_credit_deg + 1.0, \
        f"the tray was never really tilted (max {max_theta:.1f} deg)"
    assert bool(scene._tilt[0]), "tilt latch must have fired during the discharge"
    if not all_binned():
        print("SIM_GEN_SOLVE: FAIL (a marble never reached the hopper)", flush=True)
        os._exit(1)
    s = print_score(f"P2 all {kb} marbles rolled through the slot into the hopper")
    assert s >= s_prev - 1e-6, f"score decreased: {s_prev} -> {s}"
    s_prev = s

    # ---------------- phase 3: release, seat on the stop, judge, persist --------------------
    for i in range(120):
        servo_step(tilt_hi + (c.rest_deg - tilt_hi) * (i + 1) / 120)
    clear_tray_torque()
    step(180)
    report("released")
    assert bool(scene.tray_at_rest()[0]), \
        f"tray must fall back onto its rest stop, got {angle():+.2f} deg"
    assert int((scene.cubes_in_crate()[0] & scene.present_cube[0]).sum()) == kc, \
        "every present cube must still rest in the crate"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after discharge + release)", flush=True)
        os._exit(1)
    s_f = print_score(f"P3 sorted: {kc} cubes crated, {kb} marbles hoppered, tray at rest")
    assert s_f >= 0.999, f"final score {s_f} (expect success=1.0)"

    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_p = print_score("persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_p >= s_f - 1e-6
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
