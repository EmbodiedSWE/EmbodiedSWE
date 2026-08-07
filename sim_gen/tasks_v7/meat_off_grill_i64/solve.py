"""Teleport solution for KebabSpitScene (sim_gen task `meat_off_grill_i64`) — the
task's legitimacy certificate.

There are NO teleports in this solution at all: every judged body starts threaded on
the spit rail and the entire task IS the load-bearing interaction, executed through
contact dynamics end to end:

1. METERED AXIAL PUSH (contact dynamics — never teleported): the innermost present
   COOKED piece (the cooked/raw boundary — exactly where the arm's fingertip reaches
   into the inter-piece gap) receives a velocity-limited axial force along the
   fixture's +x. It conveys the whole cooked group ahead of it through ring-on-rail
   sliding contact and ring-on-ring pushing contact; each piece that reaches the
   rail's free tip tips over the end under gravity and falls into the serving tray.
   The RAW pieces behind the push point are NEVER touched (no force is ever applied
   to them; nothing pushes them) — the stop condition is built into WHERE the push
   is applied, exactly the plan the task demands.
2. The force is cut the moment the boundary piece's CoM passes the tip; gravity and
   momentum finish the last drop. From the cut to the verdict nothing touches any
   piece: they tumble, land, and settle hands-off.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.kebab_spit")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    nc = int(scene.n_cooked[0])
    nr = int(scene.n_raw[0])

    def cooked_x(i: int) -> float:
        return float(scene._fix_local(scene.cooked[i])[0, 0])

    def report(tag: str) -> None:
        cx = ", ".join(f"{cooked_x(i):+.3f}" for i in range(nc))
        rl = ", ".join(
            f"({float(scene._fix_local(scene.raw[i])[0, 0]):+.3f},"
            f"on_rail={bool(scene._on_rail(scene.raw[i])[0])})" for i in range(nr))
        tray = [bool(scene._in_tray(scene.cooked[i])[0]) for i in range(nc)]
        print(f"[solve] {tag:12s} | cooked_x=[{cx}] in_tray={tray} raw=[{rl}] "
              f"adv={float(scene._adv_max[0]):.3f} "
              f"off={scene._off_ever[0, :nc].tolist()} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def report_motion(tag: str) -> None:
        """Per-piece pose + velocity diagnostic (stillness debugging)."""
        for nm, b in ([(f"cooked_{i}", scene.cooked[i]) for i in range(nc)]
                      + [(f"raw_{i}", scene.raw[i]) for i in range(nr)]):
            loc = scene._fix_local(b)[0]
            lv = float(b.data.root_lin_vel_w[0].norm())
            av = float(b.data.root_ang_vel_w[0].norm())
            print(f"[solve] {tag} {nm}: loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                  f"{float(loc[2]):+.3f}) |v|={lv:.4f} |w|={av:.4f} "
                  f"still={bool(scene._still(b)[0])}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    f = (scene.spit.data.root_pos_w - scene.env_origins)[0]
    fq = scene.spit.data.root_quat_w[0]
    f_yaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
    print(f"[solve] layout readback (seed {args.seed}): n_cooked={nc} n_raw={nr} "
          f"fix=({float(f[0]):+.3f},{float(f[1]):+.3f}) yaw={math.degrees(f_yaw):+.1f}deg "
          f"cooked_x=[{', '.join(f'{cooked_x(i):+.3f}' for i in range(nc))}]", flush=True)
    report("reset")
    for i in range(nc):
        assert bool(scene._on_rail(scene.cooked[i])[0]), f"cooked_{i} not threaded at reset"
    for i in range(nr):
        assert bool(scene._on_rail(scene.raw[i])[0]), f"raw_{i} not threaded at reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: METERED AXIAL PUSH at the boundary (contact dynamics) --------
    # The push target is the INNERMOST present cooked piece (index nc-1: the train is
    # threaded outermost-first). A velocity-limited force along the fixture's +x at
    # its CoM — the wrench emulation of a fingertip reaching into the boundary gap —
    # conveys the cooked group; the raw group behind is never touched.
    boundary = scene.cooked[nc - 1]
    push_dir = quat_apply(scene.spit.data.root_quat_w,
                          torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    f_push, v_des = 1.2, 0.12
    first_off_printed = False
    best_x, last_bump = -1.0, 0
    done = False
    s_mid = s0
    for i in range(3000):
        bx = float(scene._fix_local(boundary)[0, 0])
        if bx > c.tip_x + 0.004:
            done = True
            break
        vel = boundary.data.root_lin_vel_w[0]
        u_vel = float((vel * push_dir[0]).sum())
        f_axis = f_push if u_vel < v_des else 0.0
        boundary.set_external_force_and_torque(
            (push_dir * f_axis).view(n, 1, 3).contiguous(), zero_wrench,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if not first_off_printed and bool(scene._off_ever[0, 0]):
            first_off_printed = True
            report("first-off")
            s_mid = print_score("P1a first cooked piece off the tip")
            assert s_mid >= s0 - 1e-6, "score decreased at first-off"
        if bx > best_x + 0.002:
            best_x, last_bump = bx, i
        elif i - last_bump > 300:  # stalled: push harder
            f_push = min(f_push + 0.6, 6.0)
            last_bump = i
            print(f"[solve] stall at boundary_x={bx:+.3f} -> f_push={f_push:.1f} N",
                  flush=True)
    boundary.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("all-pushed")
    assert done, "the metered push never carried the boundary piece past the tip"
    s1 = print_score("P1 whole cooked group conveyed past the tip")
    assert s1 >= s_mid - 1e-6, "score decreased across the push"

    # ---------------- phase 2: hands-off — fall, land, settle in the tray -------------------
    quiet = 0
    for _ in range(1200):
        env.step(no_action)
        ok = all(bool((scene._in_tray(scene.cooked[i]) & scene._still(scene.cooked[i]))[0])
                 for i in range(nc))
        quiet = quiet + 1 if ok else 0
        if quiet >= 30:
            break
    report("settled")
    for i in range(nc):
        assert bool(scene._in_tray(scene.cooked[i])[0]), \
            f"cooked_{i} did not settle inside the tray"
    for i in range(nr):
        assert bool(scene._on_rail(scene.raw[i])[0]), f"raw_{i} left the rail (overrun?!)"
    s2 = print_score("P2 cooked group settled in the tray, raw group untouched")
    assert s2 >= s1 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        report_motion("FAIL")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
