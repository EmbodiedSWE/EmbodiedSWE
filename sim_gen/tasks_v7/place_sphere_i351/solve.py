"""Teleport solution for BallPumpScene (sim_gen task `place_sphere_i351`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. FEED (teleport, whites): each white ball is carried by one root-state write from
   the supply tray to a zero-velocity HOVER above the intake funnel, then released.
   This is exactly a grasp-carry-release: the freshly written state is airborne and
   satisfies no rubric clause. Gravity drops it through the funnel into the shaft;
   the column's weight quasi-statically drives the single-file chain — around the
   45-deg bend, along the flat passage, up the 30-deg incline — all pure contact
   dynamics. The RED ball's every millimetre of conduit progress is earned by the
   pushing chain; no pose write ever touches the red ball while it is inside the
   machine.
2. EJECT: after enough feeds the red ball tips over the crest and drops into the
   machine's exit tray by gravity alone. The solver only WATCHES for this (readback)
   — how many feeds it takes varies with the episode's random start depth x0.
3. DELIVER (teleport, red): once the red ball rests in the OPEN exit tray it is
   reachable; one root-state write carries it to a hover above the green goal bin
   and releases it. The landing and settling in the bin are earned by gravity.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.place_sphere_i351.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ball_pump")().build(num_envs=args.num_envs, device=device)
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

    def pump_pose() -> tuple[torch.Tensor, float]:
        pp = (scene.pump.data.root_pos_w - scene.env_origins)[0]
        q = scene.pump.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return pp, yaw

    def to_world(fix, lx: float, ly: float) -> tuple[float, float]:
        fp = (fix.data.root_pos_w - scene.env_origins)[0]
        q = fix.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        wx = float(fp[0]) + math.cos(yaw) * lx - math.sin(yaw) * ly
        wy = float(fp[1]) + math.sin(yaw) * lx + math.cos(yaw) * ly
        return wx, wy

    def hover_release(body, fix, lx: float, ly: float, z: float) -> None:
        """TRANSPORT: write `body` to a zero-velocity hover at fixture-frame
        (lx, ly), world height z; gravity does all the rest."""
        wx, wy = to_world(fix, lx, ly)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def red_local() -> torch.Tensor:
        return scene._local(scene.red, scene.pump)[0]

    def report_whites(k: int) -> None:
        parts = []
        for i in range(k):
            wl = scene._local(scene.whites[i], scene.pump)[0]
            parts.append(f"w{i}=({float(wl[0]):+.3f},{float(wl[1]):+.3f},{float(wl[2]):.3f})")
        print("[solve] whites: " + " ".join(parts), flush=True)

    def report(tag: str) -> None:
        rl = red_local()
        print(f"[solve] {tag:14s} | red_l=({float(rl[0]):+.4f},{float(rl[1]):+.4f},"
              f"{float(rl[2]):.4f}) prog={float(scene._prog[0]):.3f} "
              f"eject_now={bool(scene._eject_now()[0])} "
              f"eject_latch={bool(scene._eject_latch[0])} "
              f"deliver_latch={bool(scene._deliver_latch[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        t = threading.Timer(10.0, lambda: os._exit(1))
        t.daemon = True
        t.start()
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline ------------------------------------
    step(60)
    pp, pyaw = pump_pose()
    x0 = float(scene._x0[0])
    tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    bp = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): pump=({float(pp[0]):+.3f},"
          f"{float(pp[1]):+.3f}) yaw={math.degrees(pyaw):+.1f}deg x0={x0:+.4f} "
          f"tray=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) "
          f"bin=({float(bp[0]):+.3f},{float(bp[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    if bool(scene.success()[0]):
        fail("fresh reset is already success")
    if s0 > 0.02:
        fail(f"baseline score should be ~0, got {s0}")
    rl = red_local()
    if abs(float(rl[0]) - x0) > 0.01 or float(rl[2]) > 0.05:
        fail("red ball did not settle at its lodged start")

    # ---------------- phase 1: feed the pump until the red ball is ejected -----------------
    # Each white: one transport hover above the funnel centre (shaft axis), release,
    # let the machine settle the chain. Watch the red ball's readback for the crest
    # drop — the required count varies with x0.
    hover_z = 0.320  # world z: ~46 mm above the funnel top plane (~0.274)
    s_prev = s0
    ejected = False
    for i in range(c.n_white):
        hover_release(scene.whites[i], scene.pump, c.shaft_cx, 0.0, hover_z)
        step(180)  # 1.5 s: fall + chain drive + settle
        report(f"feed{i}")
        report_whites(i + 1)
        s_i = print_score(f"P1.{i + 1} fed white ball {i + 1}")
        if s_i < s_prev - 1e-6:
            fail(f"score decreased across feed {i}")
        s_prev = s_i
        rl = red_local()
        if float(rl[0]) > c.eject_x[0] and float(rl[2]) < c.eject_z[1]:
            ejected = True
            print(f"[solve] red ball ejected after {i + 1} feeds", flush=True)
            break
    if not ejected:
        # give the chain a last long settle — a slow creep over the crest
        step(360)
        rl = red_local()
        ejected = float(rl[0]) > c.eject_x[0] and float(rl[2]) < c.eject_z[1]
    if not ejected:
        fail("red ball never ejected from the conduit")

    # wait for the eject latch (still-streak in the exit tray)
    latched = False
    for _ in range(30):  # up to 5 s
        step(20)
        if bool(scene._eject_latch[0]):
            latched = True
            break
    report("ejected")
    if not latched:
        fail("eject latch never earned (red not settled in the exit tray?)")
    s1 = print_score("P1 red ball ejected into the exit tray (eject latch)")
    if s1 < s_prev - 1e-6:
        fail("score decreased across ejection")

    # ---------------- phase 2: deliver the red ball to the goal bin ------------------------
    # The red ball now rests in the OPEN exit tray — reachable. One transport hover
    # above the goal bin centre, release; gravity lands it in the bin.
    hover_release(scene.red, scene.bin, 0.0, 0.0, 0.008 + c.ball_r + 0.12)
    delivered = False
    for _ in range(30):  # up to 5 s
        step(20)
        if bool(scene._deliver_latch[0]):
            delivered = True
            break
    report("delivered")
    if not delivered:
        fail("deliver latch never earned (red not settled in the goal bin?)")
    if not bool(scene.success()[0]):
        fail("no success with the red ball delivered")
    s2 = print_score("P2 red ball placed in the goal bin")
    if s2 < s1 - 1e-6:
        fail("score decreased across delivery")

    # ---------------- deep settle: kill residual motion before judging ---------------------
    def max_speed() -> float:
        v = float(scene.red.data.root_lin_vel_w[0].norm())
        for wb in scene.whites:
            v = max(v, float(wb.data.root_lin_vel_w[0].norm()))
        return v

    quiet = 0
    for _ in range(600):
        env.step(no_action)
        quiet = quiet + 1 if max_speed() < 0.03 else 0
        if quiet >= 30:
            break
    print(f"[solve] deep settle: max body speed {max_speed():.4f} m/s "
          f"(quiet streak {quiet})", flush=True)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for b in range(10):  # 10 x 42 steps = 3.5 s at 120 Hz
        step(42)
        ok_b = bool(scene.success()[0])
        print(f"[solve] persist block {b}: success={ok_b} max_speed={max_speed():.4f}",
              flush=True)
        hold = hold and ok_b
    report("persist")
    s3 = print_score("P3 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - die loudly, never idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {e})", flush=True)
        os._exit(1)
