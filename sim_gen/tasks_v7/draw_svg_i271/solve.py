"""Teleport solution for DominoRelayScene (sim_gen task `draw_svg_i271`) — the
task's legitimacy certificate.

Teleportation is TRANSPORT ONLY: each domino is teleported from the depot row to
an empty upright pose on the bench (never into contact with anything — the chain
spacing is asserted wider than a domino thickness plus 2 cm). Everything
load-bearing happens through CONTACT DYNAMICS: after the chain is built the
solver applies ONE small trigger push (~0.10 N at the first domino's CoM, cut as
soon as it passes its tipping angle, well below the sliding-friction budget) and
then goes fully HANDS-OFF — the topple wave travels domino to domino by impact,
the last domino's falling face bats the red ball over its retaining lip, and the
ball flies off the pedestal into the walled catch pocket where restitution-zero
contacts and bench friction settle it.

Plan (oracle side):
  1. read back the episode's randomized run (pad centre, bearing, span, depot
     side) from the scene's live tensors;
  2. derive the chain spacing s = (span - standoff) / 5 — asserted at scene
     level to sit inside the robust topple band for the whole span range —
     and teleport the six dominoes upright onto the line pad -> pedestal,
     thickness facing the run direction;
  3. wait for the STOOD latches (12 consecutive upright steps each);
  4. trigger: push domino 0 at its CoM along the run direction with ~0.10 N
     (tips at 0.039 N, slides at 0.18 N), cut the force at ~12 deg tilt,
     zero the wrench, hands-off;
  5. watch the cascade: all six FELL latches inside the collapse window, ball's
     first pocket entry inside the delivery window;
  6. retry from a fresh reset (same seed -> same episode) with a different
     standoff / trigger force if the cascade stalls;
then wait for success() (ball settled in the pocket, everything down) and hold
>= 3.3 simulated seconds hands-off, printing SIM_GEN_SCORE at each phase
boundary (non-decreasing: the scene's credit is latched) and
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds at the end.

Run (forge): python -u -m simgen_tasks.draw_svg_i271.solve --headless [--seed N]
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
# daemon=True: a crashed main thread must NOT idle until the timer fires.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# (standoff, trigger newtons) per attempt: nominal first, then a closer chain
# with a firmer push, then a wider chain. Trigger window for the 50 g domino:
# tips above mg*t/h = 0.065 N, slides above mu*mg = 0.30 N.
ATTEMPTS = ((0.065, 0.12), (0.060, 0.18), (0.070, 0.18))
TILT_CUT_DEG = 12.0        # cut the trigger force once domino 0 passes this tilt


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.domino_relay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(kk: int) -> None:
        for _ in range(kk):
            env.step(no_action)

    def report(tag: str) -> None:
        upz = scene._dom_up_z()[0]
        tilt = [f"{math.degrees(math.acos(max(-1.0, min(1.0, float(u))))):5.1f}" for u in upz]
        loc = scene.ball_local()[0]
        print(f"[solve] {tag:16s} | tilt(deg)=[{' '.join(tilt)}] "
              f"was_up={scene.was_up[0].tolist()} "
              f"fell={[bool(v) for v in torch.isfinite(scene.fall_t[0])]} "
              f"ball_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"in_pocket={bool(scene.ball_in_pocket()[0])} "
              f"bin_streak={int(scene.bin_streak[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero3 = torch.zeros(n, 1, 3, device=device)

    def push(newtons: float) -> None:
        """Trigger force on domino 0, at its CoM, along its local +x (= the run
        direction while it stands upright). Body-frame and persistent: cut with
        push(0.0)."""
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = newtons
        scene.doms[0].set_external_force_and_torque(f, zero3, env_ids=all_ids)

    def teleport_dom(i: int, xy_world: torch.Tensor, z: float, quat: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = xy_world
        st[:, 2] = scene.env_origins[:, 2] + z
        st[:, 3:7] = quat
        scene.doms[i].write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset readback ----------------------------------------------
    step(30)   # let the depot dominoes and the seated ball settle
    pad = scene.pad_xy[0]
    span = float(scene.span[0])
    bear = math.degrees(float(scene.term_yaw[0]))
    print(f"[solve] readback (seed {args.seed}): pad=({float(pad[0]):+.3f},"
          f"{float(pad[1]):+.3f}) bearing={bear:+.1f} deg span={span:.3f} m "
          f"depot_y={float(scene._dom_xy()[0, 0, 1]):+.3f}", flush=True)
    assert bool(scene.ball_on_pedestal()[0]), "ball not seated on the pedestal at reset"
    assert not bool(scene.was_up[0].any()), "a depot domino latched STOOD at reset"
    report("reset")
    s_prev = print_score("P0 reset (dominoes flat in depot, ball on pedestal)")
    assert s_prev < 0.005, "reset score must be ~0"

    # ---------------- attempts: build the chain, trigger, watch the cascade ----------------
    from isaaclab.utils.math import quat_apply  # noqa: F401  (parity with siblings)

    built_score = None
    ok_cascade = False
    for attempt, (standoff, newtons) in enumerate(ATTEMPTS):
        if attempt > 0:
            print(f"[solve] attempt {attempt}: fresh reset, standoff={standoff:.3f} "
                  f"trigger={newtons:.2f} N", flush=True)
            env.reset(seed=args.seed)   # same episode layout, cleared latches
            step(30)

        # ---- build: teleport the six dominoes upright along pad -> pedestal ----
        u = (scene.term_xy - scene.pad_xy) / scene.span.unsqueeze(-1)
        s = (scene.span - standoff) / (c.n_dominoes - 1)
        q_up = scene_mod._qz(scene.term_yaw)
        for i in range(c.n_dominoes):
            base = scene.pad_xy + (i * s).unsqueeze(-1) * u
            xy = scene.env_origins[:, :2] + base
            # domino 0 stands on the (0.9 mm proud) pad plate
            z = c.bench_top + c.dom_h / 2 + (0.0014 if i == 0 else 0.0005)
            teleport_dom(i, xy, z, q_up)
            step(12)
        spacing = float(s[0])
        print(f"[solve] chain built: spacing={spacing * 1000:.1f} mm "
              f"(topple band ({c.dom_t + 0.020:.3f}, {0.65 * c.dom_h:.3f}) m), "
              f"last base -> pedestal centre {standoff * 1000:.0f} mm", flush=True)

        # ---- wait for all STOOD latches ----
        for _ in range(60):
            if bool(scene.was_up[0].all()):
                break
            step(5)
        if not bool(scene.was_up[0].all()):
            report("build-failed")
            print(f"[solve] attempt {attempt}: not all dominoes latched STOOD, retrying",
                  flush=True)
            continue
        report("built")
        if built_score is None:
            built_score = print_score("P1 chain built (six STOOD latches)")
            assert built_score >= s_prev - 1e-6, "score decreased across build"
            assert abs(built_score - c.n_dominoes * c.w_stood) < 1e-3, \
                "build credit mismatch"
            s_prev = built_score

        # ---- trigger: one push on domino 0, cut past the tipping angle ----
        push(newtons)
        cut = math.cos(math.radians(TILT_CUT_DEG))
        tipped = False
        for _ in range(90):                     # <= 0.75 s of pushing
            step(1)
            if float(scene._dom_up_z()[0, 0]) < cut:
                tipped = True
                break
        push(0.0)
        if not tipped:
            report("trigger-failed")
            print(f"[solve] attempt {attempt}: domino 0 did not tip at "
                  f"{newtons:.2f} N, retrying", flush=True)
            continue
        print(f"[solve] trigger cut at {math.degrees(math.acos(float(scene._dom_up_z()[0, 0]))):.1f} deg tilt "
              f"-- hands-off from here", flush=True)

        # ---- hands-off: watch the cascade ----
        done = False
        for _ in range(30):                     # <= 7.5 s
            step(30)
            if bool((scene.relay_ok() & scene.ball_ok())[0]):
                done = True
                break
        report("cascade")
        if done:
            ok_cascade = True
            break
        print(f"[solve] attempt {attempt}: cascade incomplete "
              f"(fell={[bool(v) for v in torch.isfinite(scene.fall_t[0])]}, "
              f"ball_t={'inf' if math.isinf(float(scene.ball_t[0])) else float(scene.ball_t[0])}), "
              f"retrying", flush=True)

    if not ok_cascade:
        print("SIM_GEN_SOLVE: FAIL (cascade never completed)", flush=True)
        os._exit(1)

    tmin = float(scene.fall_t[0].min())
    tmax = float(scene.fall_t[0].max())
    bt = float(scene.ball_t[0])
    print(f"[solve] cascade: collapse window {(tmax - tmin) / 120.0:.2f} s "
          f"(limit {c.window_s:.1f}), ball entry {(bt - tmax) / 120.0:+.2f} s after last "
          f"fall (limit {c.ball_window_s:.1f}), first faller {float(scene.fall_dpad[0].min()) * 1000:.0f} mm "
          f"from pad (limit {c.start_r * 1000:.0f})", flush=True)
    s_now = print_score("P2 cascade complete (relay + ball latched)")
    assert s_now >= s_prev - 1e-6, "score decreased across cascade"
    s_prev = s_now

    # ---------------- wait for full success (ball settled in the pocket) -------------------
    for _ in range(20):
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (ball never settled in the pocket)", flush=True)
        os._exit(1)
    s_now = print_score("P3 success (ball settled in pocket)")
    assert s_now >= s_prev - 1e-6, "score decreased at success"
    s_prev = s_now

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) --------------
    hold = True
    for _ in range(10):       # 10 x 40 steps = 400 steps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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
    except Exception:  # noqa: BLE001 — die fast, don't let Kit teardown hang
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
