"""Teleport solution for DrainPlugScene (sim_gen task `place_sphere_i196`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, ball): one root-state write carries the red ball from the
   tray to a HOVER a few millimetres above the tilted false floor, well UPSLOPE of
   the drain corner, with zero velocity. This is exactly a grasp-carry-release: the
   freshly written state satisfies no rubric clause (the ball is airborne, not
   seated). Everything that follows is gravity + contact: the ball drops, ROLLS
   ~12 cm down the fall line, funnels into the drain corner and SEATS itself in the
   55 mm gap — the sinking into the seat window is earned, never written.
2. TRANSPORT (teleport, marbles): each live marble is likewise written to a hover
   just above the floor at a distinct upslope point and released; it rolls down and
   settles against the seated ball (contact). No pose write ever places a body in a
   scoring region: the seat and every containment rest are reached by rolling.

The ORDER is the task: the ball goes first. (smoke.py proves the converse — a
marble released before the plug drains into the sealed vault and is lost.)

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.place_sphere_i196.solve --headless [--seed N]
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
    env = ENVS.get("simgen.drain_plug")().build(num_envs=args.num_envs, device=device)
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

    def hopper_pose() -> tuple[torch.Tensor, float]:
        hp = (scene.hopper.data.root_pos_w - scene.env_origins)[0]
        q = scene.hopper.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return hp, yaw

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        hp, hyaw = hopper_pose()
        wx = float(hp[0]) + math.cos(hyaw) * lx - math.sin(hyaw) * ly
        wy = float(hp[1]) + math.sin(hyaw) * lx + math.cos(hyaw) * ly
        return wx, wy

    def hover_release(body, lx: float, ly: float, z: float) -> None:
        """TRANSPORT: write `body` to a zero-velocity hover at fixture-frame (lx,
        ly), world height z (just above the floor plane there); gravity + rolling
        contact do all the rest."""
        wx, wy = to_world(lx, ly)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def plane_z(lx: float, ly: float) -> float:
        return c.z_mid + c.slope_s * (lx + ly)

    def report(tag: str) -> None:
        bl = scene._fixture_local(scene.ball)[0]
        k = int(scene._k[0])
        beads = [f"{'L' if bool(scene._bead_latch[0, i]) else ('i' if bool(scene._bead_in(i)[0]) else '.')}"
                 for i in range(k)]
        print(f"[solve] {tag:14s} | ball_l=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.4f}) plug_now={bool(scene._plug_now()[0])} "
              f"plug_latch={bool(scene._plug_latch[0])} beads[{k}]={''.join(beads)} "
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
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    hp, hyaw = hopper_pose()
    k = int(scene._k[0])
    tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    cwx, cwy = to_world(-c.h, -c.h)
    print(f"[solve] layout readback (seed {args.seed}): hopper=({float(hp[0]):+.3f},"
          f"{float(hp[1]):+.3f}) yaw={math.degrees(hyaw):+.1f}deg "
          f"drain_corner_world=({cwx:+.3f},{cwy:+.3f}) "
          f"tray=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) K={k}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    if bool(scene.success()[0]):
        fail("fresh reset is already success")
    if s0 > 0.02:
        fail(f"baseline score should be ~0, got {s0}")

    # ---------------- phase 1: plug the drain (transport hover, then gravity rolls it in) --
    # Release the ball just above the floor plane ~12 cm upslope of the corner; it
    # rolls down the fall line and seats itself in the gap. The write itself is
    # airborne and satisfies nothing.
    bx, by = -c.h + 0.12, -c.h + 0.12
    hover_release(scene.ball, bx, by, plane_z(bx, by) + c.ball_r + 0.006)
    seated = False
    for _ in range(30):  # up to 5 s
        step(20)
        if bool(scene._plug_latch[0]):
            seated = True
            break
    report("plugged")
    if not seated:
        fail("ball did not seat in the drain gap")
    bl = scene._fixture_local(scene.ball)[0]
    print(f"[solve] seat readback: ball fixture z={float(bl[2]):.4f} "
          f"(expected ~{c.z_seat:.4f}), corner dist="
          f"{math.hypot(float(bl[0]) + c.h, float(bl[1]) + c.h):.4f}", flush=True)
    s1 = print_score("P1 ball rolled in and seated (plug latch)")
    if s1 < s0 - 1e-6:
        fail("score decreased across P1")

    # ---------------- phase 2: pour the marbles (one transport hover each) -----------------
    # Distinct upslope release points staggered PERPENDICULAR to the fall line so
    # each marble rolls its own lane down into the corner pocket.
    base_u = -c.h + 0.14
    offsets = (0.0, 0.04, -0.04, 0.08)
    s_prev = s1
    for i in range(k):
        d = offsets[i]
        lx = base_u + d * 0.7071
        ly = base_u - d * 0.7071
        hover_release(scene.marbles[i], lx, ly, plane_z(lx, ly) + c.marble_r + 0.005)
        latched = False
        for _ in range(30):  # up to 5 s
            step(20)
            if bool(scene._bead_latch[0, i]):
                latched = True
                break
        report(f"marble{i}")
        if not latched:
            fail(f"marble {i} did not settle contained (drained or still rolling?)")
        s_i = print_score(f"P2.{i + 1} marble {i + 1}/{k} rolled in and settled")
        if s_i < s_prev - 1e-6:
            fail(f"score decreased across marble {i}")
        s_prev = s_i

    if not bool(scene.success()[0]):
        fail("no success with plug seated and all marbles contained")

    # ---------------- deep settle: kill any residual pocket oscillation before judging -----
    def max_speed() -> float:
        v = float(scene.ball.data.root_lin_vel_w[0].norm())
        for i in range(k):
            v = max(v, float(scene.marbles[i].data.root_lin_vel_w[0].norm()))
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
    ok = hold and bool(scene.success()[0]) and s3 >= s_prev - 1e-6
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
