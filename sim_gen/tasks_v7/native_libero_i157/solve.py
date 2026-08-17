"""Teleport solution for DominoRelayScene (sim_gen task `native_libero_i157`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact dynamics:
  1. BUILD (transport + gravity/contact): each tile is teleported from the depot to a
     point in FREE SPACE standing upright a few millimetres ABOVE the ground at its
     planned lane station (foot of tile 0 at the sampled A, even pitch
     (D - reach) / (n - 1) along the sampled heading u, flat faces normal to u),
     released with zero velocity, and allowed to DROP the last millimetres and settle
     on the ground. Nothing is ever spawned in contact; the standing rest pose is made
     by gravity.
  2. TRIGGER (applied force): a small horizontal world force along u (0.12 N — above
     the 0.059 N tipping threshold m*g*t/h, below the 0.16 N sliding threshold
     mu*m*g) is applied at tile 0's CoM, re-set every physics step in the tile's
     CURRENT body frame (house wrench convention), and CUT as soon as tile 0 has
     tilted past ~20 deg — well beyond the 11.3 deg critical angle atan(t/h), so
     gravity finishes the topple.
  3. CASCADE (pure dynamics, hands-off): the chain reaction runs tile to tile on
     gravity + tile-on-tile contact alone. No body is touched again; the run is
     judged from the settled lapped carpet.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.native_libero_i157.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.domino_relay")().build(num_envs=args.num_envs, device=device)
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

    def wrench(body, f3: torch.Tensor) -> None:
        """Apply a WORLD force to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing (`control_period == 1`
        for the null robot, so each env.step is one physics substep)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def report(tag: str) -> None:
        r = scene.tile_report()
        stand = int((r["standing"][0] & r["in_lane"][0]).sum())
        fall = int((r["fallen"][0] & r["in_lane"][0]).sum())
        print(f"[solve] {tag:16s} | stand_in_lane={stand} fallen_in_lane={fall}"
              f" pad={bool(r['head_on_pad'][0].any())}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)  # depot tiles drop the last 2 mm and settle flat
    a, b = scene.A[0], scene.B[0]
    u, d = scene.U[0], float(scene.D[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"A=({float(a[0]):+.3f},{float(a[1]):+.3f}) "
          f"B=({float(b[0]):+.3f},{float(b[1]):+.3f}) "
          f"u=({float(u[0]):+.3f},{float(u[1]):+.3f}) D={d:.3f}", flush=True)
    r0 = scene.tile_report()
    for i in range(c.n_tiles):
        p = r0["pos"][0, i]
        print(f"[solve]   tile{i} depot=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) axis_z={float(r0['axis'][0, i, 2]):+.2f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: build the row (transport + gravity settle) ------------------
    reach = math.sqrt(c.tile_h ** 2 - c.tile_t ** 2)
    pitch = (d - reach) / (c.n_tiles - 1)
    gap = pitch - c.tile_t
    print(f"[solve] plan: {c.n_tiles} tiles, pitch={pitch * 1000:.1f}mm "
          f"gap={gap * 1000:.1f}mm reach={reach * 1000:.1f}mm", flush=True)
    assert c.gap_min - 1e-6 <= gap <= c.gap_max + 1e-6, f"planned gap {gap:.4f} off-window"

    yaw = math.atan2(float(u[1]), float(u[0]))  # thickness axis (body x) along u
    qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    origin = scene.env_origins
    scores = [s0]
    for i in range(c.n_tiles):
        tile = scene.tiles[i]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = scene.A + scene.U * (i * pitch)  # foot xy == center xy (upright)
        st[:, 2] = c.tile_h / 2 + 0.003  # 3 mm of free air below — gravity seats it
        st[:, 3:7] = torch.tensor(qz, device=device)
        st[:, 0:3] += origin
        tile.write_root_state_to_sim(st, all_ids)
        # settle: the tile drops the last mm and stands
        for _ in range(10):
            step(15)
            if float(tile.data.root_lin_vel_w[0].norm()) < 0.03 \
                    and float(tile.data.root_ang_vel_w[0].norm()) < 0.3:
                break
        r = scene.tile_report()
        assert bool(r["standing"][0, i] & r["in_lane"][0, i]), \
            f"tile {i} not standing in lane after placement " \
            f"(axis_z={float(r['axis'][0, i, 2]):+.2f})"
        si = print_score(f"P1 tile {i} stood at station {i} (gravity-seated)")
        assert si >= scores[-1] - 1e-6
        scores.append(si)
    report("row built")

    # ---------------- phase 2: trigger tile 0 (applied force, then cut) --------------------
    tile0 = scene.tiles[0]
    f_world = torch.zeros(3, device=device)
    f_world[0:2] = scene.U[0] * 0.12  # tips (>0.059 N) without sliding (<0.16 N)
    tipped = False
    for _ in range(600):  # 5 s cap on the nudge
        axis_z = float(scene.tile_report()["axis"][0, 0, 2])
        if axis_z < 0.94:  # ~20 deg — past the 11.3 deg critical angle; gravity finishes
            tipped = True
            break
        wrench(tile0, f_world)
        env.step(no_action)
    wrench(tile0, torch.zeros(3, device=device))
    assert tipped, "tile 0 never tipped under the trigger force"
    s2 = print_score("P2 tile 0 tipped by CoM force (cut at ~20 deg)")
    assert s2 >= scores[-1] - 1e-6

    # ---------------- phase 3: hands-off cascade -> settle -> success ----------------------
    for _ in range(32):  # up to 8 s: cascade + settling
        step(30)
        if bool(scene.settled()[0]) and bool(scene.success()[0]):
            break
    report("cascade done")
    r = scene.tile_report()
    for i in range(c.n_tiles):
        h = r["head"][0, i]
        print(f"[solve]   tile{i} head=({float(h[0]):+.3f},{float(h[1]):+.3f},"
              f"{float(h[2]) * 1000:.1f}mm) fallen={bool(r['fallen'][0, i])} "
              f"dir_dot={float(r['dir_dot'][0, i]):+.2f}", flush=True)
    s3 = print_score("P3 cascade ran hands-off, all settled")
    assert s3 >= s2 - 1e-6, "score decreased across cascade"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after build+trigger+cascade)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                lin = max(float(t.data.root_lin_vel_w[0].norm()) for t in scene.tiles)
                ang = max(float(t.data.root_ang_vel_w[0].norm()) for t in scene.tiles)
                print(f"[solve] persist flicker @step {i}: "
                      f"chain={bool(scene.chain_ok()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"max_lin={lin:.4f} max_ang={ang:.4f}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
