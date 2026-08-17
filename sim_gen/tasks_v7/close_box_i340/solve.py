"""Teleport SOLUTION for SlamShutCourierScene — NullRobot, force-driven end to end.

Phases (SIM_GEN_SCORE printed at each boundary, non-decreasing):
  P0  reset + settle: box on the launch pad under the low roof, lid at its open
      over-center rest (-120 deg), cargo cube inside; score 0.
  P1  LAUNCH (applied force): a velocity-regulated CoM force on the box along the
      alley (+x fixture frame, force-frame mode probed from progress) accelerates it
      to the computed launch speed, then cuts at the release line. The launch speed
      is derived from the cfg at runtime: v_launch^2 = v_margin * v_req^2 +
      2 mu g d_coast, where v_req is the arrest speed that just carries the lid over
      its balance point (rod-arrest model) — energy margin `v_margin` (2.2x).
  P2  COAST + SLAM (hands off): the box slides free down the slick alley, hits the
      red bumper, arrests dead (restitution 0) — the lid's angular momentum about
      the suddenly-stopped hinge carries it over the apex and gravity slams it shut
      over the cargo. Wait for everything to settle; success() must hold.
  P3  persistence: >= 3.3 simulated seconds hands-off; success must hold throughout.
      Then print `SIM_GEN_SOLVE: SUCCESS`.

NO teleports are used after reset — the entire solution is one honest shove: every
rubric fact (entered / arrived / shut / delivered / closed) is produced by the
applied launch force, sliding contact, and the arrest dynamics.

Run: python -m simgen_tasks.close_box_i340.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qapply, encode_force = task_scene._qapply, task_scene.encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.slam_shut_courier")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    ids = torch.arange(n, device=device)
    no_action = torch.empty(0, device=device)
    zero = torch.zeros(n, 1, 3, device=device)

    def step(k: int = 1) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.box.set_external_force_and_torque(zero, zero, env_ids=ids)

    last_score = -1.0

    def print_score(tag: str) -> None:
        nonlocal last_score
        s = float(scene.score()[0])
        assert s >= last_score - 1e-6, f"score regressed at {tag}: {last_score} -> {s}"
        last_score = max(last_score, s)
        print(f"SIM_GEN_SCORE {s:.2f}", flush=True)

    def report(tag: str) -> None:
        p = scene.box_local()[0]
        v = float(scene.box.data.root_lin_vel_w[0].norm())
        print(f"[solve] {tag:12s} | box_x={float(p[0]):+.3f} y={float(p[1]):+.3f} "
              f"v={v:.3f} lid={math.degrees(float(scene.lid_angle()[0])):+.1f}deg "
              f"cargo_in={bool(scene.cargo_in()[0])} settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    # ---------------- P0: reset + settle --------------------------------------------------------
    env.reset(seed=args.seed)
    step(90)
    report("P0 settle")
    p0 = scene.box_local()[0]
    lid0 = math.degrees(float(scene.lid_angle()[0]))
    assert abs(float(p0[0]) - c.box_spawn_x) < c.box_x_jitter + 0.01, "box off the pad"
    assert abs(lid0 + c.open_deg) < 6.0, f"lid not at its open rest ({lid0:.1f} deg)"
    assert bool(scene.cargo_in()[0]), "cargo not inside the box"
    assert float(scene.score()[0]) <= 0.01, "nonzero score at spawn"
    print_score("P0")

    # ---------------- P1: LAUNCH (velocity-regulated CoM force) ---------------------------------
    d_coast = c.x_bump - c.box_hx - c.release_x
    v_launch = math.sqrt(c.v_margin * c.v_req ** 2 + 2 * c.mu_dynamic * 9.81 * d_coast)
    print(f"[solve] v_req={c.v_req:.3f} m/s, coast={d_coast:.3f} m -> "
          f"v_launch={v_launch:.3f} m/s (margin {c.v_margin:.1f}x)", flush=True)

    q_fix = scene.fixture.data.root_quat_w.clone()
    u3 = _qapply(q_fix, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
    u_xy = u3[0, :2] / max(float(u3[0, :2].norm()), 1e-6)
    q_ref = scene.box.data.root_quat_w.clone()
    mode = 0
    floor_f = 1.0
    win_i, win_px = 0, float(scene.box_local()[0, 0])
    launched = False
    for i in range(1200):
        px = float(scene.box_local()[0, 0])
        v = scene.box.data.root_lin_vel_w[0, :2]
        v_along = float((v * u_xy).sum())
        if px >= c.release_x:
            clear_forces()
            launched = True
            print(f"[solve] release at x={px:.3f}, v_along={v_along:.3f} m/s "
                  f"(need >= {math.sqrt(max(c.v_margin - 0.4, 1.2)) * c.v_req:.3f})",
                  flush=True)
            break
        f_along = 15.0 * (v_launch - v_along)
        if abs(v_along) < 0.02 and f_along < floor_f:
            f_along = floor_f
        f_along = min(max(f_along, -2.0), 8.0)
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, :2] = u_xy * f_along
        f_arg = encode_force(mode, q_ref, scene.box.data.root_quat_w, f_world)
        scene.box.set_external_force_and_torque(
            f_arg.view(n, 1, 3), zero, env_ids=ids, is_global=True)
        step(1)
        if i - win_i >= 30:
            px2 = float(scene.box_local()[0, 0])
            if px2 < win_px - 0.004:
                mode = 1 - mode
                print(f"[solve] moving backward; force-frame mode -> {mode}", flush=True)
            elif px2 < win_px + 0.004:
                floor_f = min(floor_f + 0.5, 4.0)
                print(f"[solve] stalled at x={px2:.3f}; floor -> {floor_f:.1f} N",
                      flush=True)
            win_i, win_px = i, px2
    clear_forces()
    assert launched, "launch never reached the release line"
    report("P1 launch")
    print_score("P1")

    # ---------------- P2: COAST + SLAM (hands off) -----------------------------------------------
    for i in range(720):
        step(1)
        if i > 60 and bool(scene.settled()[0]):
            break
    step(60)
    report("P2 slam")
    px = float(scene.box_local()[0, 0])
    lid_deg = math.degrees(float(scene.lid_angle()[0]))
    assert px >= c.x_delivered, f"box stopped short (x={px:.3f} < {c.x_delivered:.3f})"
    assert lid_deg > -c.closed_deg, f"lid did not flip shut ({lid_deg:.1f} deg)"
    assert bool(scene.cargo_in()[0]), "cargo not retained through the slam"
    assert bool(scene.success()[0]), "success() does not hold after the slam"
    print_score("P2")

    # ---------------- P3: persistence ------------------------------------------------------------
    for k in range(10):
        step(40)
        assert bool(scene.success()[0]), f"success dropped in persistence block {k}"
    report("P3 persist")
    print_score("P3")
    print("SIM_GEN_SOLVE: SUCCESS", flush=True)

    threading.Timer(10.0, lambda: os._exit(0)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
