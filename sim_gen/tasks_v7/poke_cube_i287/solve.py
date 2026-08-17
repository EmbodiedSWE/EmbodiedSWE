"""Teleport solution for ShuntTrayScene (sim_gen task `poke_cube_i287`) — the task's
legitimacy certificate.

The task's load-bearing interaction is ORDERED SLIDING UNDER CONTACT: tiles are
captive in the tray (slotted cover — asserted in the scene cfg), so the only way
the red tile can reach the blue cell is a sequence of in-plane shunts through the one
empty cell, and the covering gray tile must vacate the blue cell first. How this
solution executes it:

1. NO TRANSPORT TELEPORT IS NEEDED OR USED: every object already starts inside the
   tray, and no pose of any tile is ever written after reset. The entire solution is
   contact dynamics.
2. PLAN (breadth-first search): occupancy is read back from the physical tile poses
   (nearest cell center in the tray frame); BFS over the 2x2 vacancy puzzle returns
   the shortest legal shunt sequence that puts red on the goal cell. The plan is
   recomputed from readback before every move, so an under/overshot slide is simply
   replanned.
3. SHUNT (contact dynamics): each move applies a velocity-capped horizontal force at
   the moving tile's CoM (the applied-wrench emulation of a fingertip pushing the
   knob), with a lateral P-servo and a small yaw-correcting torque to keep the tile
   in its corridor (the knob centered in its cover slot). The force is CUT near
   the target cell
   center; the tile coasts and settles on friction. Stalls escalate the drive force
   (velocity-servo stall memory: escalate the gain, not the cap). Each move is
   verified by occupancy readback.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after
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
from collections import deque

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

TILE_NAMES = scene_mod.TILE_NAMES
CELL_ADJ = scene_mod.CELL_ADJ

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def bfs_plan(occ: tuple, goal_cell: int) -> list[tuple[int, int, int]]:
    """Shortest shunt sequence for the 2x2 vacancy puzzle. `occ` is a 4-tuple
    cell -> tile index (-1 empty, 0 red). Returns [(tile, from_cell, to_cell), ...]
    ending with red (tile 0) on `goal_cell`."""
    start = tuple(occ)
    if start[goal_cell] == 0:
        return []
    seen = {start}
    q = deque([(start, [])])
    while q:
        state, path = q.popleft()
        empty = state.index(-1)
        for c_from in CELL_ADJ[empty]:
            tile = state[c_from]
            if tile < 0:
                continue
            nxt = list(state)
            nxt[c_from], nxt[empty] = -1, tile
            nxt = tuple(nxt)
            npath = path + [(tile, c_from, empty)]
            if nxt[goal_cell] == 0:
                return npath
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, npath))
    raise RuntimeError(f"BFS found no plan from {occ} to red@{goal_cell}")


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shunt_tray")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the scene's own readouts.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def tray_yaw() -> float:
        q = scene.tray.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def w_from_t(vx: float, vy: float) -> tuple[float, float]:
        yw = tray_yaw()
        return (math.cos(yw) * vx - math.sin(yw) * vy,
                math.sin(yw) * vx + math.cos(yw) * vy)

    def t_from_w(vx: float, vy: float) -> tuple[float, float]:
        yw = tray_yaw()
        return (math.cos(yw) * vx + math.sin(yw) * vy,
                -math.sin(yw) * vx + math.cos(yw) * vy)

    def read_occ() -> tuple:
        return tuple(int(v) for v in scene.occupancy()[0])

    def report(tag: str) -> None:
        occ = read_occ()
        xy = scene.tile_local_xy("red_tile")[0]
        print(f"[solve] {tag:14s} | occ={occ} goal_cell={int(scene._t_cell[0])} "
              f"red_loc=({float(xy[0]):+.3f},{float(xy[1]):+.3f}) "
              f"d_goal={float(scene.red_dist()[0]):.3f} "
              f"vac={bool(scene._vacated_ever[0])} moved={bool(scene._red_moved_ever[0])} "
              f"app={float(scene._app_max[0]):.3f} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        sc = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {sc:.4f}", flush=True)
        assert sc >= last_score[0] - 1e-6, \
            f"score decreased across {tag}: {last_score[0]:.4f} -> {sc:.4f}"
        last_score[0] = sc
        return sc

    def settle_wait(max_steps: int = 600) -> bool:
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            still = all(
                float(b.data.root_lin_vel_w[0].norm()) < 0.025
                and float(b.data.root_ang_vel_w[0].norm()) < 0.40
                for b in scene.tiles.values())
            quiet = quiet + 1 if still else 0
            if quiet >= 25:
                return True
        return False

    centers = scene._centers  # (4, 2) tray-frame cell centers

    def shunt(tile_i: int, c_from: int, c_to: int, tag: str) -> bool:
        """One shunt through contact: velocity-capped CoM force along the corridor,
        lateral P-servo, small yaw-correcting torque; force cut near the target cell
        center; coast + settle; verified by occupancy readback."""
        name = TILE_NAMES[tile_i]
        body = scene.tiles[name]
        a = centers[c_from].cpu()
        b = centers[c_to].cpu()
        d_t = (b - a) / (b - a).norm()
        n_t = torch.tensor([-float(d_t[1]), float(d_t[0])])
        f0, v_cap = 1.6, 0.12
        mark_prog, mark_i = -1.0, 0
        done = False
        for i in range(1200):
            p = scene.tray_local(body.data.root_pos_w)[0].cpu()
            rem = float((b - p[:2]) @ d_t)
            if rem < 0.006:
                done = True
                break
            v_w = body.data.root_lin_vel_w[0]
            vtx, vty = t_from_w(float(v_w[0]), float(v_w[1]))
            v_along = vtx * float(d_t[0]) + vty * float(d_t[1])
            e_lat = float((p[:2] - b) @ n_t)
            v_lat = vtx * float(n_t[0]) + vty * float(n_t[1])
            f_along = f0 if v_along < v_cap else 0.0
            f_lat = max(-0.5, min(0.5, -30.0 * e_lat - 1.5 * v_lat))
            ftx = f_along * float(d_t[0]) + f_lat * float(n_t[0])
            fty = f_along * float(d_t[1]) + f_lat * float(n_t[1])
            fwx, fwy = w_from_t(ftx, fty)
            # yaw correction: tile body x-axis vs the tray grid, mod 90 deg
            ex = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3)
            bx = quat_apply(body.data.root_quat_w, ex)[0]
            ang_rel = math.atan2(float(bx[1]), float(bx[0])) - tray_yaw()
            err = ((ang_rel + math.pi / 4) % (math.pi / 2)) - math.pi / 4
            wz = float(body.data.root_ang_vel_w[0, 2])
            tz = max(-0.06, min(0.06, -0.30 * err - 0.02 * wz))
            F = torch.zeros(n, 1, 3, device=device)
            F[:, 0, 0], F[:, 0, 1] = fwx, fwy
            T = torch.zeros(n, 1, 3, device=device)
            T[:, 0, 2] = tz
            body.set_external_force_and_torque(F, T, env_ids=all_ids, is_global=True)
            env.step(no_action)
            prog = float((p[:2] - a) @ d_t)
            if prog > mark_prog + 0.004:
                mark_prog, mark_i = prog, i
            elif i - mark_i > 200:
                if f0 < 5.0:
                    f0 += 0.6
                    print(f"[solve] {tag}: stall at rem={rem:.3f} -> f0={f0:.1f} N",
                          flush=True)
                mark_i = i
        body.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        if not done:
            print(f"[solve] {tag}: never reached the target cell", flush=True)
        settle_wait(400)
        p = scene.tray_local(body.data.root_pos_w)[0, :2].cpu()
        miss = float((p - b).norm())
        print(f"[solve] {tag}: {name} cell{c_from}->cell{c_to}, landed "
              f"{miss * 1000:.1f} mm off center", flush=True)
        return done and miss < 0.024

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    settle_wait(300)
    occ = read_occ()
    t_cell = int(scene._t_cell[0])
    report("reset")
    assert occ[t_cell] in (1, 2), \
        f"goal cell {t_cell} not covered by a gray tile at reset (occ={occ})"
    assert occ[t_cell] == 1, "the covering tile should be gray_a by construction"
    assert 0 in occ and occ.index(0) != t_cell, "red spawned on the goal cell"
    assert -1 in occ, "no empty cell at reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phases 1..k: BFS-planned shunts (all contact dynamics) ----------------
    moves_done = 0
    for it in range(14):
        occ = read_occ()
        if occ[t_cell] == 0:
            break
        if occ.count(-1) != 1 or len({v for v in occ if v >= 0}) != 3:
            report("bad-occ")
            print("SIM_GEN_SOLVE: FAIL (occupancy readback degenerate)", flush=True)
            os._exit(1)
        plan = bfs_plan(occ, t_cell)
        print(f"[solve] plan from occ={occ}: "
              f"{[(TILE_NAMES[t], a, b) for t, a, b in plan]}", flush=True)
        tile_i, c_from, c_to = plan[0]
        ok = shunt(tile_i, c_from, c_to, f"move{moves_done}")
        moves_done += 1
        report(f"after-move{moves_done - 1}")
        print_score(f"P{moves_done} shunted {TILE_NAMES[tile_i]} "
                    f"cell{c_from}->cell{c_to}")
        if not ok:
            print(f"[solve] move{moves_done - 1} imperfect — replanning from readback",
                  flush=True)
    occ = read_occ()
    if occ[t_cell] != 0:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (red never reached the goal cell)", flush=True)
        os._exit(1)

    # ---------------- final: settle + success ------------------------------------------------
    settle_wait(600)
    report("settled")
    s_fin = print_score(f"P{moves_done + 1} red on the blue cell, settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.5 simulated seconds, no intervention) ----------------
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score(f"P{moves_done + 2} persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_fin - 1e-6
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
