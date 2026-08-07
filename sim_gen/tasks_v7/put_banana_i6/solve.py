"""Teleport solution for MugTipoutScene (sim_gen task `put_banana_i6`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): one consistent pair pose-write carries the mug WITH the orange
   still seated at the bottom of its bore from the spawn to a hold pose hovering above
   the GREEN dish. This bypasses nothing: the orange is inside the mug at both endpoints
   (its start state), and hovering above the dish satisfies no rubric clause (all credit
   is gated on the orange LEAVING the bore, which has not happened).
2. POUR (contact dynamics — the core interaction; no teleport can produce it without
   bypassing the task): the mug is pose-HELD each step (the kinematic-hold emulation of
   a rigid grasp — exactly what a hand does to a held mug) and slowly tilted about a
   horizontal axis. The orange is NEVER pose-written here: it rolls along the real bore
   wall under gravity, exits over the rim when the tilt passes ~100 deg, free-falls into
   the dish, and settles through real impacts (restitution 0). The tilt freezes the
   moment the orange leaves the bore; the orange's landing and settling are pure physics.
3. MUG SET-DOWN (teleport transport + contact dynamics): the now-empty mug is carried by
   one pose write from its held pose to 25 mm ABOVE the blue home pad — upright, but
   deliberately OUTSIDE the 10 mm base-height tolerance, so the freshly-teleported state
   does not satisfy the mug-home clause. Gravity drops it onto the pad; it beds down and
   settles under physics. success() first turns True only here, judged on settled poses.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_banana_i6.solve --headless [--seed N]
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
G_DT = 9.81 * DT  # gravity-compensated kinematic hold: write vz=+g*dt so net vz ~ 0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_tipout_rehome")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

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
        b = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        m_ = (scene.mug.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | ball=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) mug=({float(m_[0]):+.3f},{float(m_[1]):+.3f},"
              f"{float(m_[2]):.3f}) in_mug={bool(scene._in_mug()[0])} "
              f"in_dish={bool(scene._in_dish_now(scene.dish_t)[0])} "
              f"mug_home={bool(scene._mug_home_now()[0])} "
              f"out={bool(scene._out[0])} app={float(scene._app_max[0]):.3f} "
              f"in={bool(scene._in[0])} home={float(scene._home_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hold_mug(pos: torch.Tensor, quat: torch.Tensor, omega: torch.Tensor) -> None:
        """One kinematic-hold write: pose imposed, vz=+g*dt cancels the gravity kick, ang
        vel consistent with the tilt trajectory (zero when frozen)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        st[:, 9] = G_DT
        st[:, 10:13] = omega
        scene.mug.write_root_state_to_sim(st, all_ids)

    def tilt_quat(axis: torch.Tensor, theta: float) -> torch.Tensor:
        h = theta / 2
        return torch.tensor([math.cos(h), float(axis[0]) * math.sin(h),
                             float(axis[1]) * math.sin(h), float(axis[2]) * math.sin(h)],
                            device=device)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    mug0 = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    dish_t = (scene.dish_t.data.root_pos_w - scene.env_origins)[0]
    dish_d = (scene.dish_d.data.root_pos_w - scene.env_origins)[0]
    pad0 = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): mug=({float(mug0[0]):+.3f},"
          f"{float(mug0[1]):+.3f}) green_dish=({float(dish_t[0]):+.3f},{float(dish_t[1]):+.3f}) "
          f"white_dish=({float(dish_d[0]):+.3f},{float(dish_d[1]):+.3f}) "
          f"pad=({float(pad0[0]):+.3f},{float(pad0[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._in_mug()[0]), "orange did not settle seated inside the mug bore"
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport, consistent pair) ------------------------
    # Carry the mug WITH its seated orange across free space to a hold pose above the
    # GREEN dish: mug centre 100 mm above the dish floor, offset ~35 mm back along the
    # pour direction so the rim ends up over the dish centre at exit tilt. The orange is
    # written at the exact bore-bottom seat of the new mug pose (same relative state as
    # at spawn — containment is neither created nor bypassed). Hovering satisfies no
    # rubric clause: every credit term is gated on the orange leaving the bore.
    dish_w = scene.dish_t.data.root_pos_w[0]  # world coords (env origin included)
    mug_w = scene.mug.data.root_pos_w[0]
    tdir = dish_w[:2] - mug_w[:2]
    tdir = tdir / tdir.norm().clamp(min=1e-6)  # pour direction (horizontal, unit)
    axis = torch.tensor([-float(tdir[1]), float(tdir[0]), 0.0], device=device)  # z x tdir
    hold_pos = torch.zeros(n, 3, device=device)
    hold_pos[:, 0] = dish_w[0] - tdir[0] * 0.035
    hold_pos[:, 1] = dish_w[1] - tdir[1] * 0.035
    hold_pos[:, 2] = dish_w[2] + 0.100
    q_up = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    hold_mug(hold_pos, q_up.expand(n, 4), torch.zeros(n, 3, device=device))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = hold_pos
    st[:, 2] += -c.mug_h / 2 + c.ball_seat_z  # bore-bottom seat of the new mug pose
    st[:, 3] = 1.0
    scene.ball.write_root_state_to_sim(st, all_ids)
    for _ in range(30):  # re-seat under the hold
        hold_mug(hold_pos, q_up.expand(n, 4), torch.zeros(n, 3, device=device))
        env.step(no_action)
    report("transported")
    assert bool(scene._in_mug()[0]), "orange lost from the bore during transport"
    s1 = print_score("P1 transport (mug+orange pair, still contained)")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: POUR through contact dynamics --------------------------------
    # Slow kinematic tilt about the horizontal axis, 0 -> up to 150 deg at ~40 deg/s. The
    # orange is never pose-written: it rides the real bore wall, exits over the rim under
    # gravity, falls into the dish and settles through real impacts. The tilt FREEZES the
    # moment the orange leaves the bore.
    theta, theta_dot = 0.0, math.radians(40.0)
    d_theta = theta_dot * DT
    exit_theta = None
    for i in range(1200):
        if not bool(scene._in_mug()[0]):
            exit_theta = theta
            break
        theta = min(theta + d_theta, math.radians(150.0))
        hold_mug(hold_pos, tilt_quat(axis, theta).expand(n, 4),
                 (axis * theta_dot).expand(n, 3))
        env.step(no_action)
        if i == 600 and theta >= math.radians(150.0) - 1e-6:
            print("[solve] full tilt reached, orange still inside — continuing hold",
                  flush=True)
    assert exit_theta is not None, "orange never left the bore under a 150 deg tilt"
    print(f"[solve] orange exited the bore at tilt {math.degrees(exit_theta):.1f} deg",
          flush=True)
    # Freeze the tilt; let the orange fall, impact the dish and settle under physics.
    q_frozen = tilt_quat(axis, exit_theta).expand(n, 4)
    settled = False
    for i in range(360):
        hold_mug(hold_pos, q_frozen, torch.zeros(n, 3, device=device))
        env.step(no_action)
        if i > 30 and float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.03:
            settled = True
            break
    report("poured")
    assert settled, "orange did not settle after the pour"
    assert bool(scene._in_dish_now(scene.dish_t)[0]), \
        "orange did not come to rest inside the green dish"
    s2 = print_score("P2 contact-dynamics pour into the green dish")
    assert s2 >= s1 - 1e-6, "score decreased across the pour"

    # ---------------- phase 3: MUG SET-DOWN (teleport transport + gravity) ------------------
    # One pose write carries the empty mug from its held pose to 25 mm ABOVE the blue
    # home pad, upright — outside the 10 mm base-height tolerance, so this state does
    # not satisfy the mug-home clause. Gravity drops it onto the pad; it beds down and
    # settles under contact. (The upright carry orientation is transport: a hand
    # re-orients a held mug in free space the same way.)
    pad_w = scene.pad.data.root_pos_w[0]
    pad_top = float(pad_w[2]) + c.pad_h / 2
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = pad_w[0]
    st[:, 1] = pad_w[1]
    st[:, 2] = pad_top + 0.025 + c.mug_h / 2
    st[:, 3] = 1.0
    scene.mug.write_root_state_to_sim(st, all_ids)
    report("release")
    assert not bool(scene._mug_home_now()[0]), \
        "released mug already satisfies the home clause (release band too low)"
    step(180)  # 1.5 s: drop 25 mm, impact the pad, bed down, settle
    report("set-down")
    s3 = print_score("P3 gravity set-down on the home pad + settle")
    assert s3 >= s2 - 1e-6, "score decreased across set-down"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after set-down)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
