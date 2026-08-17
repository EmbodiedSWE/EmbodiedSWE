"""solve — solution for TileShuffleScene (hand_trajectory_i374): the legitimacy certificate.

This solution uses ZERO teleports. Every tile move is contact dynamics: a velocity-servoed
horizontal WORLD-frame force at the moving tile's CoM (the applied-wrench emulation of a
fingertip pushing the protruding knob) slides it one lattice cell into the current vacancy.
The tiles are topologically captive under the slotted roof, so nothing could be carried even
if we wanted to.

There is no memorized move list: the per-episode layout (gold cell, target cell, vacancy,
frame pose) is read back from the scene, and a fresh BFS over the 30-state sliding-puzzle
graph (gold cell x vacancy cell; the four grey tiles are interchangeable) is planned BEFORE
EVERY MOVE from the current physical readback — so a disturbed or imperfect move simply gets
replanned. Most moves push GREY tiles, to route the vacancy around the board; the gold tile
itself only ever slides into an adjacent vacancy.

Force discipline: servo gain K = 8 N/(m/s) keeps K*dt/m = 0.44 < 1 (one-substep wrench delay
stable); the cap starts at 1.5 N (sliding needs ~0.4 N) and stall-escalates to at most 2.5 N.
A CoM-height planar force cannot tip the tile (quasi-static tip threshold ~2.9 N at the CoM,
and the roof leaves only 5 mm of headroom regardless). A small yaw-keeper torque about z
(cap 0.02 N m) keeps the square tile lattice-aligned so it cannot wedge in its corridor.

Phases (SIM_GEN_SCORE printed at each boundary, asserted non-decreasing):
  P0 reset + settle + layout readback              -> 0.000
  P1..Pn one lattice move each (replanned)         -> monotone latched credit
  Pn+1 gold seated in the target cell, settled     -> 1.000
  final: hands-off persistence >= 3.5 simulated seconds, success() must still hold
     -> exactly `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.hand_trajectory_i374.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=1350.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.hand_trajectory_i374 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

TILE_NAMES = scene_mod.TILE_NAMES

# Watchdog: never leave a GPU zombie if anything below wedges.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def _neighbors(i: int) -> list[int]:
    col, row = i // 2, i % 2
    out = []
    if col > 0:
        out.append((col - 1) * 2 + row)
    if col < 2:
        out.append((col + 1) * 2 + row)
    if row == 0:
        out.append(col * 2 + 1)
    else:
        out.append(col * 2)
    return out


def _bfs(gold: int, blank: int, target: int) -> list[tuple[int, int]]:
    """Shortest move list [(from_cell, to_cell), ...] on the (gold, blank) state graph;
    grey tiles are interchangeable. A move slides the tile at from_cell into to_cell
    (the vacancy at that time)."""
    start = (gold, blank)
    if gold == target:
        return []
    prev: dict[tuple[int, int], Any] = {start: None}
    dq = deque([start])
    while dq:
        s = dq.popleft()
        g, b = s
        for n in _neighbors(b):
            ns = (b, n) if n == g else (g, n)
            if ns in prev:
                continue
            prev[ns] = (s, (n, b))
            if ns[0] == target:
                moves: list[tuple[int, int]] = []
                cur = ns
                while prev[cur] is not None:
                    cur, mv = prev[cur]
                    moves.append(mv)
                moves.reverse()
                return moves
            dq.append(ns)
    raise RuntimeError(f"BFS: no path from {start} to gold={target}")


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tile_shuffle")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, 1, 3, device=device)
    cells = c.cells

    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_force(name: str) -> None:
        scene.tiles[name].set_external_force_and_torque(zero, zero, env_ids=ids)
        env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        gl = scene.world_to_local(scene.tiles["gold"].data.root_pos_w)[0]
        print(f"[solve] {tag:16s} gold_local=({float(gl[0]):+.3f},{float(gl[1]):+.3f},"
              f"{float(gl[2]):.3f}) gold_cell={int(scene.gold_cell()[0])} "
              f"target={int(scene.target[0])} moved={bool(scene.moved[0])} "
              f"d_min={float(scene.d_min[0]):.0f}/{float(scene.d0[0]):.0f} "
              f"settled={bool(scene.settled()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = [-1.0]

    def phase_score(tag: str) -> float:
        s = sc()
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)
        assert s >= last_score[0] - 1e-6, f"score decreased at {tag}: {last_score[0]} -> {s}"
        last_score[0] = s
        return s

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def local_dir_to_world(dx: float, dy: float) -> tuple[float, float]:
        cy, sy = float(torch.cos(scene.f_yaw[0])), float(torch.sin(scene.f_yaw[0]))
        return cy * dx - sy * dy, sy * dx + cy * dy

    def tile_yaw_err(name: str) -> float:
        """Tile yaw relative to the frame, wrapped to the nearest square symmetry."""
        q = scene.tiles[name].data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        err = yaw - float(scene.f_yaw[0])
        return (err + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def read_state() -> tuple[int, int, dict[int, str]]:
        """Physical readback: which tile sits in which cell; returns (gold_cell, blank_cell,
        {cell: tile_name}). Retries with extra settling if any tile is between cells."""
        for attempt in range(6):
            p = scene.tiles_local()[0]  # (5, 3)
            occ: dict[int, str] = {}
            good = True
            for j, name in enumerate(TILE_NAMES):
                d = [math.hypot(float(p[j, 0]) - cx, float(p[j, 1]) - cy)
                     for cx, cy in cells]
                cell = min(range(6), key=lambda i: d[i])
                if d[cell] > c.cell_tol or cell in occ:
                    good = False
                    break
                occ[cell] = name
            if good:
                blank = next(i for i in range(6) if i not in occ)
                gold = next(i for i, n in occ.items() if n == "gold")
                return gold, blank, occ
            step(30)
        raise RuntimeError("read_state: tiles not seated after settling")

    def _servo_step(name: str, vlx: float, vly: float, gain: float, cap: float) -> None:
        """One substep of the velocity-servoed WORLD-frame planar force + yaw keeper."""
        body = scene.tiles[name]
        dwx, dwy = local_dir_to_world(vlx, vly)
        v_now = body.data.root_lin_vel_w[0]
        fx = gain * (dwx - float(v_now[0]))
        fy = gain * (dwy - float(v_now[1]))
        fm = math.hypot(fx, fy)
        if fm > cap:
            fx, fy = fx * cap / fm, fy * cap / fm
        yerr = tile_yaw_err(name)
        wz = float(body.data.root_ang_vel_w[0, 2])
        tz = max(-0.02, min(0.02, -0.30 * yerr - 0.010 * wz))
        f = torch.zeros(1, 1, 3, device=device)
        f[0, 0, 0], f[0, 0, 1] = fx, fy
        t = torch.zeros(1, 1, 3, device=device)
        t[0, 0, 2] = tz
        body.set_external_force_and_torque(f, t, env_ids=ids, is_global=True)
        env.step(no_action)

    def push_tile(name: str, frm: int, to: int, max_steps: int = 2400) -> bool:
        """Axis-decomposed servo push: the tile is first CENTRED on the move line (the knob
        must ride the middle of its roof slot, or it corner-catches a slot-guide edge at a
        junction — the split-guide re-entry jam), then driven along the lattice axis with a
        continuous cross-axis correction. Arrival requires BOTH axes within 4 mm so the next
        move starts centred. On stall: back off along the move axis while re-centring, then
        re-approach with an escalated cap (never above 2.5 N)."""
        j = TILE_NAMES.index(name)
        fx0, fy0 = cells[frm]
        tx, ty = cells[to]
        along_x = abs(tx - fx0) > abs(ty - fy0)
        # gain * v_slow (12 * 0.04 = 0.48 N) must exceed sliding friction (~0.37 N) or the
        # P-servo stalls at the slow-mode threshold; K*dt/m = 0.67 stays wrench-delay stable
        gain, cap = 12.0, 1.5
        best, best_i = float("inf"), 0
        i = 0
        backoffs = 0
        while i < max_steps:
            p = scene.tiles_local()[0, j]
            px, py = float(p[0]), float(p[1])
            e_along = (tx - px) if along_x else (ty - py)
            e_cross = (ty - py) if along_x else (tx - px)
            d = math.hypot(tx - px, ty - py)
            if abs(e_along) < 0.004 and abs(e_cross) < 0.004:
                clear_force(name)
                print(f"[solve] {name} {frm}->{to}: reached (d={d * 1000:.1f} mm, "
                      f"step {i})", flush=True)
                return True
            # centre first, traverse second: full speed only when the knob is mid-slot
            if abs(e_cross) > 0.006:
                v_a = 0.0
            elif abs(e_cross) > 0.003:
                v_a = 0.04 * math.copysign(1.0, e_along)
            else:
                v_a = (0.08 if abs(e_along) > 0.030 else 0.04) * math.copysign(1.0, e_along)
            v_c = max(-0.05, min(0.05, 6.0 * e_cross))
            _servo_step(name, v_a if along_x else v_c, v_c if along_x else v_a, gain, cap)
            i += 1
            if d < best - 0.003:
                best, best_i = d, i
            elif i - best_i > 240:
                backoffs += 1
                cap = min(cap + 0.25, 2.5)
                gain = min(gain * 1.3, 18.0)
                print(f"[solve] {name} {frm}->{to}: stall at d={d * 1000:.0f} mm "
                      f"(cross={e_cross * 1000:+.1f} mm) -> backoff #{backoffs}, "
                      f"cap={cap:.2f} N", flush=True)
                for _ in range(80):  # reverse along the move axis while re-centring
                    p = scene.tiles_local()[0, j]
                    px, py = float(p[0]), float(p[1])
                    e_along = (tx - px) if along_x else (ty - py)
                    e_cross = (ty - py) if along_x else (tx - px)
                    v_a = -0.05 * math.copysign(1.0, e_along)
                    v_c = max(-0.05, min(0.05, 6.0 * e_cross))
                    _servo_step(name, v_a if along_x else v_c,
                                v_c if along_x else v_a, gain, cap)
                    i += 1
                best, best_i = float("inf"), i
        clear_force(name)
        print(f"[solve] {name} {frm}->{to}: NOT reached in {max_steps} steps", flush=True)
        return False

    def settle_until(pred, max_steps: int = 900, poll: int = 10) -> bool:
        waited = 0
        while waited <= max_steps:
            if pred():
                return True
            step(poll)
            waited += poll
        return pred()

    # ================= P0: reset + settle + layout readback ====================================
    step(90)
    g0, b0, _occ = read_state()
    print(f"[solve] layout readback (seed {args.seed}): gold={g0} blank={b0} "
          f"target={int(scene.target[0])} d0={float(scene.d0[0]):.0f} "
          f"frame=({float(scene.f_pos[0, 0]):+.3f},{float(scene.f_pos[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene.f_yaw[0])):+.1f}deg", flush=True)
    assert g0 == int(scene.gold_start[0]), "readback gold cell mismatch"
    assert b0 == int(scene.blank[0]), "readback blank cell mismatch"
    report("reset")
    assert sc() <= 1e-6, f"reset score must be 0, got {sc()}"
    assert not bool(scene.success()[0]), "success at reset"
    phase_score("P0 reset")  # 0.000

    # ================= P1..Pn: replanned lattice moves =========================================
    target = int(scene.target[0])
    n_moves = 0
    while True:
        gold, blank, occ = read_state()
        if gold == target:
            break
        moves = _bfs(gold, blank, target)
        frm, to = moves[0]
        name = occ[frm]
        assert to == blank, "planned move must slide into the current vacancy"
        n_moves += 1
        if n_moves > 40:
            print("[solve] FAILED: move budget exhausted", flush=True)
            verdict(False)
        print(f"[solve] move {n_moves}: {name} {frm}->{to} "
              f"(plan length {len(moves)})", flush=True)
        ok = push_tile(name, frm, to)
        step(30)  # let the moved tile seat
        if not ok:
            print(f"[solve] move {n_moves} did not converge; replanning", flush=True)
        report(f"after move {n_moves}")
        phase_score(f"P{n_moves} move {name} {frm}->{to}")

    # ================= gold seated: settle to live success =====================================
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    report("seated")
    if not ok:
        print("[solve] FAILED: gold seated but success() never latched live", flush=True)
        verdict(False)
    phase_score("P_final gold in target")  # 1.000

    # ================= hands-off persistence >= 3.5 simulated seconds ==========================
    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("final")
    phase_score("P_persist final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)
