"""Teleport solution for TileShuntScene (sim_gen task `empty_dishwasher_i23`) — the
task's legitimacy certificate.

This task has NOTHING to transport: every tile is captive under the frame's lid, so
there are ZERO pose writes after reset. The entire solution is contact dynamics:

  P0 — READ + PLAN: settle, read the arrangement back from the scene (nearest-cell
  occupancy of the five tiles, the empty cell, the goal corner from the green
  marker's position), then BFS over the (red_cell, empty_cell) puzzle graph — the
  classic 15-puzzle abstraction with interchangeable distractors — to get the
  shortest legal move sequence. Most seeds require moving WHITE tiles first to walk
  the empty cell onto the red tile's path (the make-way plan).
  P1..Pk — EXECUTE each move as a real constrained slide: a floating-hand force
  controller (position-PD toward the target cell centre + friction-breaking bias
  with stall escalation, lateral PD onto the slot line, yaw-steadying torque)
  pushes ONE tile from its cell into the CURRENT empty cell. The plate drags on the
  real frame floor, guided by the real walls, neighbours and the knob-in-slot
  clearance the whole way; occupancy is re-verified by READBACK after every move
  and the plan is recomputed if a move did not land.
  Pk+1 — final settle; success() must hold with everything at rest.

Prints `SIM_GEN_SCORE <score>` after reset and after EVERY executed move
(non-decreasing: the scene's credit is latched), then holds HANDS-OFF for >= 3.4
simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.empty_dishwasher_i23.solve --headless [--seed N]
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
from collections import deque

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


def neighbors(cell: int) -> list[int]:
    """Adjacent cells in the 2x3 lattice (cells k = row*3 + col)."""
    i, j = cell % 3, cell // 3
    out = []
    if i > 0:
        out.append(cell - 1)
    if i < 2:
        out.append(cell + 1)
    out.append(cell + 3 if j == 0 else cell - 3)
    return out


def bfs_plan(red: int, empty: int, goal: int) -> list[tuple[int, int]] | None:
    """Shortest move list [(from_cell, to_cell), ...] over the (red, empty) puzzle
    abstraction (white tiles interchangeable). A move slides the tile at from_cell
    into the empty to_cell."""
    start = (red, empty)
    prev: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]] | None] = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        r, e = s
        if r == goal:
            moves: list[tuple[int, int]] = []
            while prev[s] is not None:
                s, mv = prev[s]  # type: ignore[misc]
                moves.append(mv)
            return list(reversed(moves))
        for a in neighbors(e):
            ns = (e, a) if a == r else (r, a)
            if ns not in prev:
                prev[ns] = (s, (a, e))
                q.append(ns)
    return None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tile_shunt")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    m_tile, g = c.tile_mass, 9.81

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def frame_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
        q = scene.frame.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def occupancy() -> tuple[list[int], int]:
        """READBACK: nearest cell of each tile; plus the one empty cell."""
        loc = scene.tiles_local()[0]  # (T, 3)
        d = (loc[:, None, :2] - scene.cell_xy[None, :, :]).norm(dim=-1)  # (T, 6)
        cells = d.argmin(dim=1).tolist()
        empty = ({0, 1, 2, 3, 4, 5} - set(cells)).pop()
        return cells, empty

    def goal_from_marker() -> int:
        """READBACK: the goal corner cell nearest the green post (frame frame)."""
        ml = scene._frame_local(scene.marker.data.root_pos_w)[0]
        col = 0 if float(ml[0]) < 0 else 2
        row = 0 if float(ml[1]) < 0 else 1
        return row * 3 + col

    def report(tag: str) -> None:
        loc = scene.tiles_local()[0]
        cells, empty = occupancy()
        print(f"[solve] {tag:12s} | cells={cells} empty={empty} "
              f"red_local=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},"
              f"{float(loc[0, 2]):.3f}) red_at_goal={bool(scene.red_at_goal()[0])} "
              f"in_band={scene.in_band()[0].tolist()} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(t: int) -> None:
        scene.tiles[t].set_external_force_and_torque(zero_wrench, zero_wrench,
                                                     env_ids=all_ids)

    def push_move(t: int, target_cell: int) -> bool:
        """Slide tile t into `target_cell` under contact dynamics: position-PD
        toward the cell centre + friction-breaking bias (stall-escalated), lateral
        PD onto the slot line, yaw-steadying torque. Returns True when the tile
        parks within 6 mm of the target centre."""
        tgt = scene.cell_xy[target_cell]  # (2,) frame frame
        body = scene.tiles[t]
        bias, best_err, last_gain = 0.5, 1e9, 0
        done = False
        for i in range(1500):
            _fp, fyaw = frame_pose()
            ca, sa = math.cos(fyaw), math.sin(fyaw)
            loc = scene.tiles_local()[0, t]
            err = tgt - loc[:2]  # frame frame
            e = float(err.norm())
            v_w = body.data.root_lin_vel_w[0]
            # world lin vel -> frame frame (pure yaw)
            v_l = torch.tensor([ca * float(v_w[0]) + sa * float(v_w[1]),
                                -sa * float(v_w[0]) + ca * float(v_w[1])], device=device)
            if e < 0.006 and float(v_l.norm()) < 0.03:
                done = True
                break
            f_l = m_tile * (70.0 * err - 16.0 * v_l)
            if e > 1e-6:
                f_l = f_l + bias * err / e
            fn = float(f_l.norm())
            if fn > 2.5:
                f_l = f_l * (2.5 / fn)
            f_w = torch.tensor([ca * float(f_l[0]) - sa * float(f_l[1]),
                                sa * float(f_l[0]) + ca * float(f_l[1]), 0.0], device=device)
            # yaw-steadying torque (what a hand on the knob does) + tilt damping
            q = body.data.root_quat_w[0]
            tyaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
            dyaw = math.atan2(math.sin(fyaw - tyaw), math.cos(fyaw - tyaw))
            w_w = body.data.root_ang_vel_w[0]
            tq = torch.tensor([-0.01 * float(w_w[0]), -0.01 * float(w_w[1]),
                               0.04 * dyaw - 0.01 * float(w_w[2])], device=device)
            tq = tq.clamp(-0.06, 0.06)
            body.set_external_force_and_torque(
                f_w.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if e < best_err - 0.002:
                best_err, last_gain = e, i
            elif i - last_gain > 240:  # stalled: lean a little harder
                bias = min(bias + 0.35, 2.0)
                last_gain = i
                print(f"[solve] move stalled at err={e:.4f}, bias={bias:.2f} N", flush=True)
        clear_wrench(t)
        step(30)
        return done

    # ---------------- phase 0: reset, settle, read, plan ------------------------------------
    step(60)
    fp, fyaw = frame_pose()
    cells, empty = occupancy()
    goal = goal_from_marker()
    goal_buf = int(scene.goal_cell[0])
    print(f"[solve] layout readback (seed {args.seed}): frame=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg cells={cells} "
          f"empty={empty} goal(marker)={goal} goal(buffer)={goal_buf} "
          f"d0={float(scene.d0[0]):.0f}", flush=True)
    assert goal == goal_buf, "marker readback disagrees with the scene's goal buffer"
    report("reset")
    prev_s = print_score("P0 reset+settle")

    # ---------------- phases 1..k: execute the shunt plan (contact dynamics) ----------------
    replans = 0
    moves = bfs_plan(cells[0], empty, goal)
    assert moves is not None, "puzzle graph must be solvable"
    print(f"[solve] plan: {len(moves)} moves {moves}", flush=True)
    k = 0
    while moves:
        a, b = moves.pop(0)
        cells, empty = occupancy()
        assert b == empty, f"plan desync: move targets {b} but empty is {empty}"
        t = cells.index(a)
        k += 1
        tag = "RED" if t == 0 else f"white#{t}"
        print(f"[solve] move {k}: slide {tag} {a} -> {b}", flush=True)
        ok = push_move(t, b)
        cells2, empty2 = occupancy()
        if not ok or cells2[t] != b:
            replans += 1
            print(f"[solve] move {k} did not land (cells={cells2}, empty={empty2}) — "
                  f"replan {replans}", flush=True)
            if replans > 4:
                print("SIM_GEN_SOLVE: FAIL (too many replans)", flush=True)
                os._exit(1)
            moves = bfs_plan(cells2[0], empty2, goal)
            assert moves is not None
            continue
        report(f"move {k}")
        s = print_score(f"P{k} move {a}->{b} ({tag})")
        assert s >= prev_s - 1e-6, "score decreased across a move"
        prev_s = s

    # ---------------- final settle + judge ---------------------------------------------------
    step(90)
    report("final")
    s_fin = print_score("Pfinal settle")
    assert s_fin >= prev_s - 1e-6, "score decreased across the final settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the plan + settle)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.4 simulated seconds, no intervention) ---------------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.4 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_p = print_score("Ppersist 3.4 s")
    ok = hold and bool(scene.success()[0]) and s_p >= s_fin - 1e-6
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — Kit teardown hangs; fail fast, don't wait for watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
