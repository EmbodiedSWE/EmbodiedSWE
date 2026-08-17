"""Teleport solution for BallastPressCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i319`) — the task's
legitimacy certificate.

This solve applies ZERO forces and ZERO wrenches — ever. Its TWO teleports are pure
TRANSPORT: one ballast block at a time is lifted off the deck, lowered into an open
hopper cell and released (zero velocity, attitude-aligned, 3 mm above the cell floor) —
exactly what a hand does when it picks a block up and sets it down in the cell. Everything judged
happens through contact dynamics under gravity:

1. PERCEPTION: base pose/yaw, the drawer's sampled opening q0 and the block dock
   positions are read back from the episode state — never hard-coded. Release points
   are hopper-cell centres mapped through the MEASURED lever pose.
2. TRANSPORT A (teleport 1): block A released above the CENTRE cell. It drops in and
   the lever DOES NOT MOVE — the counterweight margin is demonstrated live: one block
   is below the press threshold, the drawer has not moved, only ballast credit is
   earned.
3. TRANSPORT B (teleport 2): block B released above the +y cell. From release on the
   episode is hands off: two blocks now out-torque the counterweight, the lever heels
   over under the ballast's weight, and the press blade sweeps the drawer to its rear
   hard stop. The judged "drawer closes" outcome is delivered ENTIRELY by the
   machine — no wrench ever touches the drawer or the lever.
4. The settled end state is judged LIVE: drawer seated in its channel AND lever
   heeled past press_min_deg AND both blocks riding inside their cells.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches), then holds HANDS-OFF >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i319.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_press_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        inc = scene.blocks_in_cells()[0]
        print(f"[solve] {tag:12s} | q={float(scene.drawer_q()[0]):+.4f} "
              f"theta={float(scene.lever_theta_deg()[0]):+.2f}deg "
              f"in_cells={[bool(v) for v in inc]} "
              f"drawer_closed={bool(scene.drawer_closed()[0])} "
              f"in_channel={bool(scene.drawer_in_channel()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drop_block(blk, cell_y: float) -> None:
        """The transport teleport: the block lowered INTO the cell and released (zero
        velocity, attitude-aligned) 3 mm above the cell floor — a hand placing a block,
        not hurling it — mapped through the MEASURED lever pose. The gentle release
        matters physically: a block dropped from above the wall top delivers an impulsive
        torque spike that jolts the lever far past its static threshold (verified on the
        forge: the blade transiently tapped the drawer 15-27 mm). Whether the placed
        weight tips the lever is decided by the hands-off statics that follow."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = (c.tray_x0 + c.tray_x1) / 2
        loc[:, 1] = cell_y
        loc[:, 2] = 0.003            # rest height is z=0 (cell floor top at -cube_s/2)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.lever.data.root_pos_w + quat_apply(scene.lever.data.root_quat_w, loc)
        st[:, 3:7] = scene.lever.data.root_quat_w
        blk.write_root_state_to_sim(st)

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(150)   # drawer seats on the slab; lever settles on its raised stop; blocks rest
    bp0 = (scene.base.data.root_pos_w - scene.env_origins)[0]
    bq0 = scene.base.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(bq0[3]), float(bq0[0])))
    q0 = float(scene.q0[0])
    q_meas = float(scene.drawer_q()[0])
    th0 = float(scene.lever_theta_deg()[0])
    docks = [scene._station_local(b.data.root_pos_w)[0] for b in scene.blocks]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"base=({float(bp0[0]):+.3f},{float(bp0[1]):+.3f}) yaw~{yaw:+.1f}deg "
          f"q0={q0:+.4f} (measured {q_meas:+.4f}) theta0={th0:+.2f}deg "
          + " ".join(f"blk{k}=({float(d[0]):+.3f},{float(d[1]):+.3f})"
                     for k, d in enumerate(docks)), flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.drawer_in_channel()[0]), "drawer must ride in its channel"
    assert not bool(scene.drawer_closed()[0]), "drawer must start open"
    assert abs(q_meas - q0) < 0.010, "drawer must rest at its sampled opening"
    assert -1.5 < th0 < 1.5, f"lever must rest on its raised stop, theta={th0:+.2f}"
    assert not bool(scene.blocks_in_cells()[0].any()), "blocks must start on the deck"
    for k, d in enumerate(docks):
        assert c.dock_x_range[0] - 0.02 < float(d[0]) < c.dock_x_range[1] + 0.02, k
        assert c.dock_y_range[0] - 0.02 < abs(float(d[1])) < c.dock_y_range[1] + 0.02, k
    s_prev = print_score("P0 reset+settle (lever up, hopper empty, drawer out)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: TRANSPORT A — one block aboard, lever provably stays up -----------
    cy = c.cell_centers_y()
    drop_block(scene.blocks[0], cy[1])      # centre cell
    step(300)   # 2.5 s: the block drops in, everything re-settles — and nothing moves
    report("ballastA")
    th1 = float(scene.lever_theta_deg()[0])
    q1 = float(scene.drawer_q()[0])
    assert bool(scene.blocks_in_cells()[0, 0]), "block A must ride in the centre cell"
    assert th1 < 2.0, f"ONE block must NOT tip the lever (theta={th1:+.2f}deg)"
    assert abs(q1 - q_meas) < 0.005, \
        f"drawer must not have moved under one block (q {q_meas:+.4f} -> {q1:+.4f})"
    assert not bool(scene.success()[0])
    s = print_score("P1 one ballast block aboard (below threshold: lever still up)")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert 0.13 <= s <= 0.22, f"P1 must earn one ballast credit only, got {s:.3f}"
    s_prev = s

    # ---------------- phase 2: TRANSPORT B — second block; the machine presses, hands off --------
    drop_block(scene.blocks[1], cy[2])      # +y cell
    closed_at = -1
    for i in range(720):    # up to 6 s — the heel-over + press takes ~1-2 s
        env.step(no_action)
        if closed_at < 0 and bool(scene.drawer_closed()[0]):
            closed_at = i
        if closed_at >= 0 and bool(scene.settled()[0]):
            break
    print(f"[solve] press: drawer first seated at step {closed_at}", flush=True)
    report("press")
    assert closed_at >= 0, \
        f"ballast press failed to seat the drawer (q={float(scene.drawer_q()[0]):+.4f} " \
        f"theta={float(scene.lever_theta_deg()[0]):+.2f})"
    step(120)   # hands-off settle: everything comes to rest
    report("settled")
    assert bool(scene.drawer_closed()[0]), \
        f"drawer did not stay seated: q={float(scene.drawer_q()[0]):+.4f}"
    th2 = float(scene.lever_theta_deg()[0])
    assert th2 >= c.press_min_deg, f"lever must hold the press (theta={th2:+.2f}deg)"
    assert int(scene.blocks_in_cells()[0].sum()) >= 2, "both blocks must ride their cells"
    assert bool(scene.success()[0]), "success() must hold on the settled end state"
    s = print_score("P2 second block aboard: ballast heels the lever, drawer pressed home")
    assert s >= s_prev - 1e-6, "score decreased across P2"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold_ok = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold_ok = hold_ok and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold_ok and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
