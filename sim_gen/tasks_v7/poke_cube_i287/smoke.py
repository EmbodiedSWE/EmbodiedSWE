"""Smoke / rubric-REJECTION battery for ShuntTrayScene (sim_gen task
`poke_cube_i287`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — BFS-planned force-servo shunts that end with the
red tile on the blue cell — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/layout    — reset settles finite: occupancy readback matches the
                          sampled (goal, red, empty) cells, the goal cell covered
                          by gray_a, marker sunk at the goal cell, all still, at
                          rest height; score ~0, no success;
  3-4. randomization    — READBACK over 8 seeded resets: >= 3 distinct
                          (goal, red, empty) triples, goal ALWAYS covered by a
                          gray tile; tray xy jitter (> 8 mm) and free yaw
                          (> 10 deg) are real;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  captivity probe   — a strong up + tray-center-ward yank on the red tile
                          (the pick-it-out bypass): the tile lifts off (probe
                          engaged, non-vacuous) but the slotted cover keeps it
                          inside the tray;
  7.  seed strategy     — the end state the SEED's plan produces here (one
                          straight push of red at the goal): red jammed against
                          the gray tile still covering the blue cell -> NOT
                          success, score <= 0.71;
  8.  position near-miss— goal cell free, red settled 33 mm from the blue cell
                          center (just past the 25 mm tolerance), flat, upright,
                          still -> NOT success;
  9.  wrong tile        — a GRAY tile centered on the blue cell (red elsewhere)
                          -> NOT success (the rubric keys on the RED tile);
  10. stacked           — red resting ON TOP of the gray tile that covers the
                          goal (in-tolerance xy, quiet) judged on write -> the
                          floor-height gate alone rejects;
  11. tilt gate         — goal cell freed, red AT the blue center at rest height
                          but pitched 25 deg (> 15 deg tolerance), judged on
                          write -> the upright gate alone rejects;
  12. latched credit    — vacate + approach credit earned, then red regressed to
                          its start cell: latched score UNCHANGED, no success;
  13. rejection audit   — success() was never True at ANY judged point;
  14. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shunt_tray")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.85, 0.80)) + o),
                                tuple(np.array((0.40, 0.00, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        sc, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return sc, ok

    def read_occ() -> tuple:
        return tuple(int(v) for v in scene.occupancy()[0])

    def cells() -> tuple[int, int, int]:
        return (int(scene._t_cell[0]), int(scene._r_cell[0]), int(scene._e_cell[0]))

    def report(tag: str) -> None:
        t, r, e = cells()
        xy = scene.tile_local_xy("red_tile")[0]
        sc, ok = judge()
        print(f"[smoke] {tag:16s} | occ={read_occ()} t/r/e=({t},{r},{e}) "
              f"red_loc=({float(xy[0]):+.3f},{float(xy[1]):+.3f}) "
              f"d_goal={float(scene.red_dist()[0]):.3f} "
              f"vac={bool(scene._vacated_ever[0])} moved={bool(scene._red_moved_ever[0])} "
              f"app={float(scene._app_max[0]):.3f} settled={bool(scene.settled()[0])} "
              f"score={sc:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_tile(name: str, x_loc: float, y_loc: float, z_loc: float,
                   pitch_deg: float = 0.0) -> None:
        """Probe constructor: teleport a tile to a TRAY-LOCAL pose (tray-yaw
        aligned, optional extra pitch about the tray x-axis), zero velocity."""
        tq = scene.tray.data.root_quat_w
        tp = scene.tray.data.root_pos_w
        local = torch.tensor([x_loc, y_loc, z_loc], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = tp + quat_apply(tq, local)
        if pitch_deg:
            h = math.radians(pitch_deg) / 2
            qp = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0],
                              device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(tq, qp)
        else:
            st[:, 3:7] = tq
        scene.tiles[name].write_root_state_to_sim(st, all_ids)

    def cellc(i: int) -> tuple[float, float]:
        return (float(scene._centers[i, 0]), float(scene._centers[i, 1]))

    def toward(i: int, j: int, dist: float) -> tuple[float, float]:
        """Tray-local point `dist` from cell i's center toward cell j's center."""
        ax, ay = cellc(i)
        bx, by = cellc(j)
        L = math.hypot(bx - ax, by - ay)
        return (ax + (bx - ax) / L * dist, ay + (by - ay) / L * dist)

    def finite_all() -> bool:
        ok = bool(torch.isfinite(scene.tray.data.root_state_w).all()
                  and torch.isfinite(scene.marker.data.root_state_w).all())
        for b in scene.tiles.values():
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def red_z_loc() -> float:
        return float(scene.tray_local(scene.tiles["red_tile"].data.root_pos_w)[0, 2])

    # =========================== 1-2. settle / layout / no-NaN ==============================
    env.reset(seed=11)
    report("reset")
    step(90)
    report("show")
    t, r, e = cells()
    occ = read_occ()
    mk_loc = scene.tray_local(scene.marker.data.root_pos_w)[0]
    mk_off = math.hypot(float(mk_loc[0]) - cellc(t)[0], float(mk_loc[1]) - cellc(t)[1])
    z_ok = all(
        abs(float(scene.tray_local(b.data.root_pos_w)[0, 2]) - c.rest_z) < 0.005
        for b in scene.tiles.values())
    check("settle: states finite; occupancy readback matches the sampled cells "
          "(gray_a ON the goal cell, red at its cell, the empty cell empty); the "
          "blue marker sunk at the goal cell; every tile at rest height and still",
          finite_all() and occ[t] == 1 and occ[r] == 0 and occ[e] == -1
          and mk_off < 0.006 and z_ok and bool(scene.settled()[0]))
    sc, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", sc <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        t, r, e = cells()
        tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        q = scene.tray.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        reads.append((t, r, e, float(tp[0]), float(tp[1]), yaw, read_occ()[t]))
    print(f"[smoke] randomization readback (t, r, e, tray_x, tray_y, yaw, occ[t]):",
          flush=True)
    for rr in reads:
        print(f"[smoke]   {rr}", flush=True)
    triples = {rr[0:3] for rr in reads}
    check("randomization: >= 3 distinct (goal, red, empty) cell triples across 8 "
          "seeded resets, and the goal cell is ALWAYS covered by gray_a (readback)",
          len(triples) >= 3 and all(rr[6] == 1 for rr in reads))
    xy = np.array([rr[3:5] for rr in reads])
    spread = float((xy.max(axis=0) - xy.min(axis=0)).max())
    uv = np.array([(math.cos(rr[5]), math.sin(rr[5])) for rr in reads])
    min_dot = min(float(uv[i] @ uv[j])
                  for i in range(len(uv)) for j in range(i + 1, len(uv)))
    check("randomization: tray xy jitter (> 8 mm) and free tray yaw (two draws "
          "differ > 10 deg) are real (readback)",
          spread > 0.008 and min_dot < math.cos(math.radians(10.0)))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    sc, ok = judge()
    check("null policy: score ~0 (<= 0.02) and no success after 240 idle steps",
          sc <= 0.02 and not ok)

    # =========================== 6. captivity: the pick-it-out bypass =======================
    # A strong upward + tray-center-ward yank on the red tile (what a gripper
    # pinching the knob and pulling would do). Non-vacuous: the tile must actually
    # lift off the floor — and the slotted cover must keep it inside the tray.
    env.reset(seed=101)
    step(60)
    z_max = red_z_loc()
    red = scene.tiles["red_tile"]
    for i in range(120):
        p = scene.tile_local_xy("red_tile")[0]
        L = math.hypot(float(p[0]), float(p[1])) + 1e-9
        ux, uy = -float(p[0]) / L, -float(p[1]) / L  # tray-local, toward the center
        tq = scene.tray.data.root_quat_w
        fdir = quat_apply(tq, torch.tensor([ux, uy, 0.0], device=device).expand(n, 3))
        F = torch.zeros(n, 1, 3, device=device)
        F[:, 0, :] = fdir * 2.5
        F[:, 0, 2] += 6.0
        red.set_external_force_and_torque(F, zero_w, env_ids=all_ids, is_global=True)
        env.step(no_action)
        z_max = max(z_max, red_z_loc())
    red.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(200)
    report("captivity-yank")
    p = scene.tile_local_xy("red_tile")[0]
    inside = (abs(float(p[0])) < c.inner_half and abs(float(p[1])) < c.inner_half
              and red_z_loc() < c.wall_h)
    check("captivity: a 6 N up + 2.5 N center-ward yank lifts the red tile off the "
          f"floor (peak z {z_max * 1000:.1f} mm — probe engaged) but the slotted "
          "cover keeps it inside the tray (no pick-and-place bypass exists)",
          z_max > c.rest_z + 0.003 and z_max < c.wall_h - c.tile_h / 2 + 0.006
          and inside)
    sc, ok = judge()
    check("captivity: the yank itself earns no success and stays under the "
          "non-success cap", not ok and sc <= 0.71)

    # =========================== 7. seed strategy: straight push jams =======================
    # The seed's entire plan — one straight push of the red object at the goal —
    # executed here as its END STATE: red pressed against the gray tile that still
    # covers the blue cell (2 mm shy of face contact), settled. The goal stays
    # covered; the distance gate rejects.
    env.reset(seed=41)
    step(60)
    t, r, e = cells()
    nb = [x for x in CELL_ADJ[t] if x in (r, e)][0]  # approach lane free of gray_b
    px, py = toward(t, nb, 0.058)
    place_tile("red_tile", px, py, c.rest_z + 0.001)
    step(150)
    report("seed-jam")
    d = float(scene.red_dist()[0])
    sc, ok = judge()
    check("seed strategy: red pushed straight at the occupied blue cell jams "
          f"against the covering gray tile ({d * 1000:.0f} mm from the center, goal "
          "still covered by gray_a) — NOT success, score <= 0.71",
          read_occ()[t] == 1 and 0.045 < d < 0.075 and bool(scene.settled()[0])
          and not ok and sc <= 0.71)

    # =========================== 8. position near-miss ======================================
    # Goal cell freed (grays parked on the two cells off the approach lane), red
    # flat and upright 33 mm from the blue center — just past the 25 mm tolerance.
    # Every other gate reads True; the distance tolerance alone rejects.
    env.reset(seed=51)
    step(60)
    t, r, e = cells()
    n2 = CELL_ADJ[t][0]
    others = [i for i in range(4) if i not in (t, n2)]
    ax, ay = cellc(others[0])
    bx, by = cellc(others[1])
    place_tile("gray_a", ax, ay, c.rest_z + 0.001)
    place_tile("gray_b", bx, by, c.rest_z + 0.001)
    px, py = toward(t, n2, 0.033)
    place_tile("red_tile", px, py, c.rest_z + 0.001)
    step(150)
    report("near-miss")
    d = float(scene.red_dist()[0])
    sc, ok = judge()
    check("position near-miss: goal cell free, red flat/upright/still "
          f"{d * 1000:.0f} mm from the blue center (tolerance 25 mm) — NOT success "
          "(the distance tolerance is load-bearing)",
          0.027 < d < 0.042 and bool(scene.red_on_floor()[0])
          and bool(scene.red_upright()[0]) and bool(scene.settled()[0]) and not ok)

    # =========================== 9. wrong tile on the goal ==================================
    env.reset(seed=61)
    step(60)
    t, r, e = cells()
    place_tile("gray_a", *cellc(e), c.rest_z + 0.001)
    place_tile("gray_b", *cellc(t), c.rest_z + 0.001)
    step(120)
    report("wrong-tile")
    gb = float((scene.tile_local_xy("gray_b")[0]
                - scene._centers[t]).norm())
    sc, ok = judge()
    check("wrong tile: a GRAY tile centered on the blue cell "
          f"({gb * 1000:.0f} mm off center) with red elsewhere — NOT success, "
          "score <= 0.71 (the rubric keys on the RED tile)",
          gb <= c.success_tol and read_occ()[r] == 0 and bool(scene.settled()[0])
          and not ok and sc <= 0.71)

    # =========================== 10. stacked on the covering tile ===========================
    # Red resting ON TOP of the gray tile that covers the goal: xy in tolerance,
    # zero velocity -> in-goal, upright, settled all read True; the floor-height
    # gate ALONE rejects. Judged on the written state with NO intervening step,
    # then relocated before physics reacts to the (cover-penetrating) stack.
    env.reset(seed=71)
    step(60)
    t, r, e = cells()
    place_tile("red_tile", *cellc(t), c.rest_z + c.tile_h + 0.002)
    report("stacked")
    sc, ok = judge()
    stack_ok = (bool(scene.red_in_goal()[0]) and bool(scene.red_upright()[0])
                and bool(scene.settled()[0]) and not bool(scene.red_on_floor()[0])
                and not ok)
    place_tile("red_tile", *cellc(r), c.rest_z + 0.001)  # relocate pre-reaction
    step(60)
    check("stacked: red ON TOP of the gray tile covering the goal (in-tolerance "
          "xy, quiet) judged on write — in-goal/upright/settled all read True, yet "
          "NOT success (the floor-height gate is load-bearing)", stack_ok)

    # =========================== 11. tilt gate ==============================================
    # Goal cell freed, red AT the blue center at rest height but pitched 25 deg
    # (> the 15 deg tolerance): in-goal, on-floor, settled all read True; the
    # upright gate ALONE rejects. Judged on write (a tile cannot rest at 25 deg),
    # then relocated before it can slap flat.
    env.reset(seed=81)
    step(60)
    t, r, e = cells()
    place_tile("gray_a", *cellc(e), c.rest_z + 0.001)
    place_tile("red_tile", *cellc(t), c.rest_z, pitch_deg=25.0)
    report("tilt-25deg")
    sc, ok = judge()
    tilt_ok = (bool(scene.red_in_goal()[0]) and bool(scene.red_on_floor()[0])
               and bool(scene.settled()[0]) and not bool(scene.red_upright()[0])
               and not ok)
    place_tile("red_tile", *cellc(r), c.rest_z + 0.001)  # relocate pre-slap
    step(60)
    check("tolerance gate: red AT the blue center, rest height, quiet, pitched "
          "25 deg — in-goal/on-floor/settled read True, yet NOT success (the "
          "15 deg upright tolerance is load-bearing)", tilt_ok)

    # =========================== 12. latched credit survives regression =====================
    # Construct partial credit on write (gray_a vacated, red advanced to 40 mm
    # from the blue center), then regress red all the way back to its start cell:
    # the latched score must be UNCHANGED — and still no success.
    env.reset(seed=91)
    step(60)
    t, r, e = cells()
    place_tile("gray_a", *cellc(e), c.rest_z + 0.001)
    px, py = toward(t, CELL_ADJ[t][0], 0.040)
    place_tile("red_tile", px, py, c.rest_z + 0.001)
    report("credit")
    sc_a, ok_a = judge()
    place_tile("red_tile", *cellc(r), c.rest_z + 0.001)  # regress beyond the latch
    step(30)
    report("regressed")
    sc_b, ok_b = judge()
    check("latched credit: vacate + approach credit earned "
          f"({sc_a:.3f} >= 0.35), then red regressed to its start cell — latched "
          f"score UNCHANGED ({sc_a:.3f} -> {sc_b:.3f}), never success",
          sc_a >= 0.35 and abs(sc_b - sc_a) < 1e-3 and not ok_a and not ok_b
          and float(scene.red_dist()[0]) > c.success_tol + 0.020)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shunt_tray")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
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
