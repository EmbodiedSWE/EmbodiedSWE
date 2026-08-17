"""Teleport solution for CleatPierScene (sim_gen task
`libero_pick_chocolate_pudding_i407`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY; every load-bearing interaction goes through
CONTACT DYNAMICS:
  P1 — CANTILEVER BUILD. The plank is slid ~0.5 m forward along the guided lane by
  a horizontal velocity-servo force at its CoM along the fixture-local +x axis
  (is_global — the body-frame default drags with the body), passing UNDER the red
  cleat bar, until its nose overhangs the cliff edge with its tail still beneath
  the cleat (tail in the anchored stop window). The force is cut and the plank
  settles flat — the anchoring itself is pure geometry, nothing is attached.
  P2 — LOADING. The payload crate is teleported from its side-strip spawn to a
  HOVER pose 40 mm above the plank's overhanging nose (transport only — open air,
  touching nothing), then dropped: gravity + contact put it on the nose, the plank
  pitches by the 6 mm cleat gap, the rising tail presses the cleat's underside and
  the couple holds the cantilever. The payload is never teleported into a
  supported/goal contact state — its final rest pose exists only because the
  physics of the anchored plank holds it.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene
latches its credit), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_chocolate_pudding_i407.solve --headless [--seed N]
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
    env = ENVS.get("simgen.cleat_pier")().build(num_envs=args.num_envs, device=device)
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

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def fix_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        q = scene.fixture.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return fp, yaw

    def to_world(local_xy: tuple) -> tuple[float, float]:
        fp, yaw = fix_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + local_xy[0] * cy - local_xy[1] * sy,
                float(fp[1]) + local_xy[0] * sy + local_xy[1] * cy)

    def report(tag: str) -> None:
        nose, tail = scene.plank_ends()
        pl = scene._fix_local(scene.payload.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | plank nose={float(nose[0]):+.3f} "
              f"tail={float(tail[0]):+.3f} flat={bool(scene.plank_flat()[0])} "
              f"payload_local=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
              f"ovh={float(scene.ovh_latch[0]):.0f} anc={float(scene.anchor_latch[0]):.0f} "
              f"brd={float(scene.board_latch[0]):.0f} "
              f"at_goal={bool(scene.payload_at_goal_geom()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fp, fyaw = fix_pose()
    nose0, tail0 = scene.plank_ends()
    pay0 = scene._fix_local(scene.payload.data.root_pos_w)[0]
    dec0 = scene._fix_local(scene.decoy.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): fixture=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg "
          f"plank_tail={float(tail0[0]):+.4f} "
          f"payload_local=({float(pay0[0]):+.3f},{float(pay0[1]):+.3f}) "
          f"decoy_local=({float(dec0[0]):+.3f},{float(dec0[1]):+.3f})", flush=True)
    report("reset")
    s_prev = print_score("P0 reset+settle")

    # Fixture is kinematic — its frame is constant for the whole episode.
    fwd = torch.tensor([math.cos(fyaw), math.sin(fyaw), 0.0], device=device)  # local +x
    lat = torch.tensor([-math.sin(fyaw), math.cos(fyaw), 0.0], device=device)  # local +y

    # ---------------- phase 1: slide the plank under the cleat to the stop window ----------
    # Velocity servo + friction feed-forward at the plank CoM. The bias breaks static
    # friction (~0.5 * 0.4 kg * g ~ 2.0 N); the cap keeps the push gentle. Target: tail
    # in the middle of the anchored stop window.
    tail_stop = c.anchor_tail - 0.020  # 20 mm under the cleat: comfortably anchored
    m_plank, v_des = c.plank_mass, 0.10
    reached = False
    for i in range(1200):
        nose, tail = scene.plank_ends()
        if float(tail[0]) >= tail_stop:
            reached = True
            break
        if i % 200 == 199:
            print(f"[solve] slide i={i} tail={float(tail[0]):+.3f} "
                  f"nose={float(nose[0]):+.3f}", flush=True)
        v_w = scene.plank.data.root_lin_vel_w[0]
        v_fwd = float(torch.dot(v_w, fwd))
        v_lat = float(torch.dot(v_w, lat))
        py = float(scene._fix_local(scene.plank.data.root_pos_w)[0][1])
        gain = 40.0 + 10.0 * (i // 300)  # escalate past static friction if bound
        cap = 3.5 + 0.5 * (i // 300)
        f_fwd = max(0.0, min(cap, 1.2 + m_plank * gain * (v_des - v_fwd)))
        f_lat = max(-1.0, min(1.0, m_plank * (-30.0 * py - 8.0 * v_lat)))
        f_world = f_fwd * fwd + f_lat * lat
        scene.plank.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
    clear_wrench(scene.plank)
    assert reached, "plank never reached the anchored stop window"
    step(80)  # coast + settle flat
    nose, tail = scene.plank_ends()
    assert float(tail[0]) <= c.anchor_tail, "plank overshot the anchored window"
    assert float(nose[0]) >= c.min_ovh + c.payload_size / 2 + 0.01, (
        "plank nose does not offer the goal overhang")
    report("cantilevered")
    s = print_score("P1 plank slid under the cleat, nose cantilevered (anchored)")
    assert s >= s_prev - 1e-6, "score decreased across the slide"
    s_prev = s

    # ---------------- phase 2: load the nose (teleport = hover transport, drop = physics) ---
    # Hover pose: 40 mm above the nose's resting payload height, centred between the
    # latch line and the nose end — open air, touching nothing.
    nose_x = float(nose[0])
    px = max(c.min_ovh + 0.015, min(nose_x - c.payload_size / 2 - 0.01, 0.14))
    wx, wy = to_world((px, 0.0))
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = wx, wy
    st[:, 2] = c.deck_h + c.z_goal + 0.040
    st[:, 3], st[:, 6] = math.cos(fyaw / 2), math.sin(fyaw / 2)  # square to the lane
    st[:, 0:3] += scene.env_origins
    scene.payload.write_root_state_to_sim(st, all_ids)
    print(f"[solve] payload hover-dropped at local x={px:+.3f} (nose={nose_x:+.3f})",
          flush=True)
    step(120)  # gravity drop onto the nose + cantilever settling
    report("loaded")
    s = print_score("P2 payload dropped onto the overhanging nose + settled")
    assert s >= s_prev - 1e-6, "score decreased across the loading"
    s_prev = s
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after loading settled)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except BaseException as exc:  # noqa: BLE001 — die NOW, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
