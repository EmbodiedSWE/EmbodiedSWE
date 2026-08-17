"""Teleport solution for SieveSorterScene (sim_gen task `track_spoon_i335`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, always ending in a non-contact hover; every
load-bearing interaction — and the entire SORTING itself — goes through CONTACT
DYNAMICS:
  P0 — SETTLE: reset scatters the apparatus (xy + yaw), the tray (xy + free yaw) and
  the per-episode batch (2-3 beads + 1-2 marbles in random tray slots). Everything
  settles. Nothing is touched. Score ~0.
  P1 — FEED MARBLES: each present marble is lifted from the tray to a hover 20 mm
  above a V-groove on the OPEN upstream stretch of the rails and RELEASED. The
  machine does the rest: the 34 mm marble cannot pass the 21 mm gap, seats on the
  45-deg rail faces, rolls downhill, clears the divider sill and drops off the rail
  ends into the roofed end bin, where the tilted floor pins it against the end wall.
  The per-object latch is polled by READBACK before the next feed.
  P2 — FEED BEADS: each present bead is hovered 30 mm above the SAME groove line and
  released. The 13 mm bead falls straight through the 21 mm gap into the lower bin
  and rolls to rest against the upstream wall. Same feed zone, opposite verdict —
  the classification is the machine's, not the teleport's.
  P3 — persistence: hands off >= 3.6 simulated seconds after success() turns True;
  `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched per object).

Run (forge): python -u -m simgen_tasks.track_spoon_i335.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sieve_sorter")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        d = scene.obj_local()[0]
        cb = scene.correct_bin()[0]
        for j, nm in enumerate(scene.OBJ_NAMES):
            if not bool(scene._present[0, j]):
                continue
            print(f"[solve] {tag:10s} | {nm:8s} loc=({float(d[j, 0]):+.3f},"
                  f"{float(d[j, 1]):+.3f},{float(d[j, 2]):+.3f}) "
                  f"v={float(scene.objs[j].data.root_lin_vel_w.norm(dim=-1)[0]):.3f} "
                  f"correct={bool(cb[j])} binned={bool(scene._binned[0, j])}", flush=True)
        print(f"[solve] {tag:10s} | upright={bool(scene.upright()[0])} "
              f"settled={bool(scene.settled()[0])} still={bool(scene._still_pos[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hover_world(x: float, y: float, z: float) -> torch.Tensor:
        """Apparatus-frame point -> world (per env)."""
        local = torch.tensor([x, y, z], device=device).expand(n, 3)
        return scene.apparatus.data.root_pos_w \
            + quat_apply(scene.apparatus.data.root_quat_w, local)

    def write_state(body, pos_w: torch.Tensor) -> None:
        """Transport-only teleport to a WORLD position with zero velocities."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def feed(j: int, groove_y: float, feed_x: float, drop: float, tag: str) -> bool:
        """Hover object j above the groove line at feed_x and RELEASE; poll the
        per-object latch by readback. One retry (re-hover) on timeout."""
        r = c.marble_r if j >= 3 else c.bead_r
        for attempt in range(3):
            # hover just above the rail tops, centred on a groove: a clean
            # non-contact release for both types (the marble then seats in the
            # groove ~25 mm below; the bead falls straight through the gap)
            z = c.waist_z(feed_x) + c.rail_half_w + r + drop
            write_state(scene.objs[j], hover_world(feed_x, groove_y, z))
            step(30)  # release + first roll/fall
            for _ in range(50):  # up to ~12.5 s sim for the machine to deliver
                step(30)
                if bool(scene._binned[0, j]):
                    print(f"[solve] {tag}: {scene.OBJ_NAMES[j]} delivered "
                          f"(attempt {attempt})", flush=True)
                    return True
                v = float(scene.objs[j].data.root_lin_vel_w.norm(dim=-1)[0])
                d = scene.obj_local()[0, j]
                if v < 0.02 and not bool(scene.correct_bin()[0, j]):
                    print(f"[solve] {tag}: {scene.OBJ_NAMES[j]} stalled at "
                          f"({float(d[0]):+.3f},{float(d[1]):+.3f},{float(d[2]):+.3f}) "
                          f"— re-feeding", flush=True)
                    break  # stalled somewhere wrong: retry from a fresh hover
        return False

    # ---------------- phase 0: reset, settle, readback -------------------------------------
    step(240)  # tray batch + machine come to rest (~2 s)
    nb = int(scene._present[0, :3].sum())
    nm = int(scene._present[0, 3:].sum())
    print(f"[solve] layout readback (seed {args.seed}): "
          f"app=({float(scene._app_xy[0, 0]):+.3f},{float(scene._app_xy[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene._app_yaw[0])):+.1f}deg "
          f"tray=({float(scene._tray_xy[0, 0]):+.3f},{float(scene._tray_xy[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene._tray_yaw[0])):+.1f}deg "
          f"beads={nb} marbles={nm}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    if bool(scene.success()[0]) or s0 > 0.03:
        print("SIM_GEN_SOLVE: FAIL (reset state already scores)", flush=True)
        os._exit(1)

    # ---------------- phase 1: FEED MARBLES through the classifier -------------------------
    feed_x = -0.24  # open upstream stretch, well clear of the roof lip
    g = c.groove_ys  # V-groove centrelines: (-0.061, -0.020, +0.020, +0.061)
    ok = True
    for k in range(nm):
        ok = ok and feed(3 + k, g[2] if k == 0 else g[1], feed_x, 0.020, "P1")
    report("marbles")
    s1 = print_score("P1 marbles ridden to the end bin")
    if not ok or s1 < s0 - 1e-6:
        print("SIM_GEN_SOLVE: FAIL (a marble was not delivered)", flush=True)
        os._exit(1)

    # ---------------- phase 2: FEED BEADS through the SAME machine -------------------------
    bead_grooves = (g[1], g[3], g[0])
    for k in range(nb):
        ok = ok and feed(k, bead_grooves[k], -0.20, 0.030, "P2")
    report("beads")
    s2 = print_score("P2 beads dropped through to the lower bin")
    if not ok or s2 < s1 - 1e-6:
        print("SIM_GEN_SOLVE: FAIL (a bead was not delivered)", flush=True)
        os._exit(1)

    # ---------------- wait for the settled success verdict ---------------------------------
    reached = False
    for _ in range(40):  # up to 10 s sim
        step(30)
        if bool(scene.success()[0]):
            reached = True
            break
    report("verdict")
    if not reached:
        print("SIM_GEN_SOLVE: FAIL (no settled success after feeding)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.6 simulated seconds, hands off) ----------
    hold = True
    for i in range(12):  # 12 x 36 steps = 432 substeps = 3.6 s at 120 Hz
        step(36)
        ok_i = bool(scene.success()[0])
        if not ok_i:
            print(f"[solve] persist blip @block {i}: "
                  f"settled={bool(scene.settled()[0])} still={bool(scene._still_pos[0])} "
                  f"upright={bool(scene.upright()[0])}", flush=True)
            report(f"blip{i}")
        hold = hold and ok_i
    report("persist")
    s3 = print_score("P3 persistence 3.6 s")
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
    main()
