"""smoke — REJECTION battery for the TileShuffleScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py — the contact-only knob-push shuffle — already
proves the rubric ACCEPTS the correct outcome). Every check here CONSTRUCTS a wrong
strategy or a near-miss (teleports are instrumentation) and asserts the rubric REJECTS
it. success() is audited at every judged point and must NEVER be True anywhere:

  1. settle/no-NaN    — reset settles finite, all five tiles seated in their assigned
                        cells (gold in gold_start, none in the vacancy), score 0;
  2. randomization    — READBACK: the frame physically moves across seeds (wall piece
                        world pose + yaw spread);
  3. layout diversity — READBACK: >= 3 distinct (gold, target, blank) triples across 8
                        seeds; each seed's PHYSICAL gold pose matches the claimed start
                        cell and the pads' centroid sits on the claimed target cell;
  4. null policy      — 240 idle steps -> score 0, no latches, no success;
  5. seed strategy    — the seed's plan (CARRY the object through free space along a
                        route): the gold tile is pinned over every cell centre in turn,
                        ending over the target, 47+ mm above the z gate (altitude
                        verified by the probe) -> d_min untouched, moved unlatched,
                        score 0;
  6. captivity        — the gold tile released above the target cell lands ON the
                        slotted roof and stays there (root z above the gate): the cell
                        cannot be entered from above, score 0, no success;
  7. straddle         — gold parked at the midpoint between the target and a neighbour
                        cell (45 mm from both centres): seated nowhere, no success,
                        score = the 0.08 moved latch only;
  8. wrong cell       — gold seated in a NEIGHBOUR of the target: no success (progress
                        credit is latched by design but stays < 0.45);
  9. imposter         — a GREY tile seated in the target cell: no success, no credit;
 10. occupancy (neg)  — a knobbed tile pushed toward a fully occupied chain (two tiles
                        + wall) advances < 30 mm under the same force that moves a free
                        tile a full cell — the "only into the vacancy" move law is
                        physical, not scripted (tile stays below the z gate throughout);
 11. occupancy (pos)  — the same tile, same force, pushed toward the vacancy travels a
                        full cell and seats there (positive control of check 10);
 12. settle gate      — gold written INTO the target cell sliding at 0.35 m/s: success
                        is refused while it moves (progress latches the 0.60 cap);
 13. latched credit   — gold yanked back out to its start cell: the 0.60 survives,
                        success never fires;
 14. audit            — success() was never True at any judged point, max score 0.60;
 15. final no-NaN; frames.npz saved.

Run (forge): python -u -m simgen_tasks.hand_trajectory_i374.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20/4090 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.hand_trajectory_i374 import scene as scene_mod
except ImportError:  # standalone fallback
    import scene as scene_mod

_mdist = scene_mod._mdist
TILE_NAMES = scene_mod.TILE_NAMES

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tile_shuffle")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.45, -0.85, 0.70)) + o),
                                tuple(np.array((0.45, 0.0, 0.03)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

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

    never_success = [True]
    max_score = [0.0]

    def judge() -> tuple[float, bool]:
        """Sample the rubric at a judged point and feed the global audit."""
        s = float(scene.score()[0])
        ok = bool(scene.success()[0])
        never_success[0] &= not ok
        max_score[0] = max(max_score[0], s)
        return s, ok

    def report(tag: str) -> None:
        p = scene.world_to_local(scene.tiles["gold"].data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} gold_local=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) gold_cell={int(scene.gold_cell()[0])} "
              f"target={int(scene.target[0])} moved={bool(scene.moved[0])} "
              f"d_min={float(scene.d_min[0]):.0f}/{float(scene.d0[0]):.0f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def tile_state(x: float, y: float, z: float,
                   vel: tuple = (0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.local_to_world(
            torch.tensor([[x, y, z]], device=device), ids)
        half = float(scene.f_yaw[0]) / 2
        st[0, 3] = math.cos(half)
        st[0, 6] = math.sin(half)
        st[0, 7:10] = torch.tensor(vel, device=device)
        return st

    def put_tile(name: str, x: float, y: float, z: float = 0.001, settle: int = 40,
                 vel: tuple = (0.0, 0.0, 0.0)) -> None:
        """Teleport a tile to a frame-local pose (instrumentation only)."""
        scene.tiles[name].write_root_state_to_sim(tile_state(x, y, z, vel), ids)
        step(settle)

    def pin_tile(name: str, x: float, y: float, z: float, n: int) -> None:
        """Hold a tile at a pose for n substeps (rewrite every step, zero velocity)."""
        for _ in range(n):
            scene.tiles[name].write_root_state_to_sim(tile_state(x, y, z), ids)
            step(1)

    def tile_cell(name: str) -> int:
        """Physical occupancy readback: nearest cell if within tol AND below the z gate."""
        p = scene.world_to_local(scene.tiles[name].data.root_pos_w)[0]
        dm, i = min((math.hypot(float(p[0]) - cx, float(p[1]) - cy), i)
                    for i, (cx, cy) in enumerate(c.cells))
        return i if (dm < c.cell_tol and float(p[2]) < c.z_gate) else -1

    def local_dir_to_world(dx: float, dy: float) -> tuple[float, float]:
        cy, sy = math.cos(float(scene.f_yaw[0])), math.sin(float(scene.f_yaw[0]))
        return (cy * dx - sy * dy, sy * dx + cy * dy)

    def push(name: str, dlx: float, dly: float, n: int, force: float) -> float:
        """Constant WORLD-frame force along a frame-local direction for n steps; returns
        the along-axis displacement measured AT force-off (before any rebound)."""
        wx, wy = local_dir_to_world(dlx, dly)
        f = torch.zeros(1, 1, 3, device=device)
        f[0, 0, 0], f[0, 0, 1] = force * wx, force * wy
        body = scene.tiles[name]
        p0 = scene.world_to_local(body.data.root_pos_w)[0, :2].clone()
        for _ in range(n):
            body.set_external_force_and_torque(f, zero, env_ids=ids, is_global=True)
            step(1)
        p1 = scene.world_to_local(body.data.root_pos_w)[0, :2].clone()
        body.set_external_force_and_torque(zero, zero, env_ids=ids)
        step(1)
        return float((p1[0] - p0[0]) * dlx + (p1[1] - p0[1]) * dly)

    def arrange(gold_at, grey_cells: list[int]) -> None:
        """Teleport all five tiles into distinct free spots: greys at the four given cell
        centres, gold at `gold_at` (a cell index or a raw local (x, y)). ALL writes land
        before any step, so transient overlaps with not-yet-moved tiles cannot
        depenetration-launch anything."""
        for name, cell in zip(TILE_NAMES[1:], grey_cells):
            put_tile(name, *c.cells[cell], settle=0)
        gx, gy = c.cells[gold_at] if isinstance(gold_at, int) else gold_at
        put_tile("gold", gx, gy, settle=0)
        step(50)

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    occ = [tile_cell(n) for n in TILE_NAMES]
    check("settle: finite state; all five tiles seated in distinct assigned cells "
          "(gold in gold_start, vacancy empty); settled; score 0",
          all(bool(torch.isfinite(scene.tiles[n].data.root_state_w).all())
              for n in TILE_NAMES)
          and all(o >= 0 for o in occ) and len(set(occ)) == 5
          and occ[0] == int(scene.gold_start[0])
          and int(scene.blank[0]) not in occ
          and bool(scene.settled()[0]) and float(scene.score()[0]) <= 1e-6)

    # =========================== 2./3. randomization + layout (readback) ====================
    frame_reads, layouts, phys_ok = [], [], []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(5)
        w = (scene.frame["wall_s"].data.root_pos_w - scene.env_origins)[0]
        frame_reads.append((float(w[0]), float(w[1]), float(scene.f_yaw[0])))
        g, t, b = int(scene.gold_start[0]), int(scene.target[0]), int(scene.blank[0])
        layouts.append((g, t, b))
        # physical readbacks: the gold TILE really sits in the claimed start cell, and the
        # marker PADS really centre on the claimed target cell.
        pads = torch.stack([p.data.root_pos_w for p in scene.pads]).mean(dim=0)
        pl = scene.world_to_local(pads)[0]
        tc = c.cells[t]
        phys_ok.append(tile_cell("gold") == g
                       and math.hypot(float(pl[0]) - tc[0], float(pl[1]) - tc[1]) < 0.010)
    arr = np.array(frame_reads)
    spread = arr.max(axis=0) - arr.min(axis=0)
    print(f"[smoke] frame readback (wall_s x, y, yaw):\n{arr}", flush=True)
    print(f"[smoke] layouts (gold, target, blank)={layouts} phys_ok={phys_ok}", flush=True)
    check("randomization: the frame physically moves across seeded resets "
          "(wall readback spread)",
          spread[0] > 0.010 and spread[1] > 0.010 and spread[2] > 0.05)
    check("layout: >= 3 distinct (gold, target, blank) triples across 8 seeds; physical "
          "gold pose and pad centroid match the claimed layout every seed",
          len(set(layouts)) >= 3 and all(phys_ok)
          and all(_mdist(g, t) >= c.min_dist for g, t, _b in layouts))

    # =========================== 4. null policy =============================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score 0, no latches, no success",
          float(scene.score()[0]) <= 1e-6 and not bool(scene.moved[0])
          and float(scene.d_min[0]) == float(scene.d0[0])
          and not bool(scene.success()[0]))

    # =========================== 5. seed strategy: aerial carry =============================
    # The seed's whole plan — carry the object through free space along a route to the
    # goal. The gold tile visits EVERY cell centre in turn (a superset of any route),
    # ending directly over the target, but 65 mm above the z gate; the probe asserts its
    # own altitude so the rejection cannot be vacuous.
    env.reset(seed=41)
    step(60)
    z_fly = 0.10  # roof top is 0.053; z gate is 0.035
    tgt = int(scene.target[0])
    route = [i for i in range(6) if i != tgt] + [tgt]
    alt_ok = True
    for cell in route:
        pin_tile("gold", *c.cells[cell], z=z_fly, n=20)
        p = scene.world_to_local(scene.tiles["gold"].data.root_pos_w)[0]
        alt_ok &= float(p[2]) > c.z_gate + 0.02
    report("aerial-carry")
    check("seed strategy: flying the gold tile over every cell (ending over the target) "
          "latches NOTHING (z gate; probe altitude verified)",
          alt_ok and not bool(scene.moved[0])
          and float(scene.d_min[0]) == float(scene.d0[0])
          and float(scene.score()[0]) <= 1e-6)

    # =========================== 6. captivity: the roof blocks entry from above =============
    step(150)  # release: the tile falls from z_fly over the target cell
    report("roof-drop")
    p = scene.world_to_local(scene.tiles["gold"].data.root_pos_w)[0]
    check("captivity: the dropped tile lands ON the slotted roof (root z above the "
          "gate) — the target cell cannot be entered from above; score 0, no success",
          float(p[2]) > c.z_gate + 0.010 and int(scene.gold_cell()[0]) == -1
          and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 7./8./9. straddle + wrong cell + imposter ==================
    env.reset(seed=51)
    step(60)
    tgt = int(scene.target[0])
    nbr = next(j for j in range(6) if _mdist(j, tgt) == 1)
    others = [j for j in range(6) if j not in (tgt, nbr)]
    mid = ((c.cells[tgt][0] + c.cells[nbr][0]) / 2, (c.cells[tgt][1] + c.cells[nbr][1]) / 2)
    arrange(mid, others)
    report("straddle")
    check("straddle: gold at the midpoint between the target and a neighbour (45 mm "
          "from both centres) is seated NOWHERE — no success, score = moved latch only",
          int(scene.gold_cell()[0]) == -1 and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.08) < 1e-3)
    put_tile("gold", *c.cells[nbr], settle=60)
    report("wrong-cell")
    s_wrong = float(scene.score()[0])
    check("wrong cell: gold seated one cell from the target — no success, score < 0.45",
          int(scene.gold_cell()[0]) == nbr and not bool(scene.success()[0])
          and s_wrong < 0.45)
    put_tile("grey0", *c.cells[tgt], settle=60)
    report("imposter")
    check("imposter: a GREY tile seated in the target cell earns nothing — no success, "
          "score unchanged",
          tile_cell("grey0") == tgt and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - s_wrong) < 1e-3)

    # =========================== 10./11. occupancy move law (physical) ======================
    # Canonical arrangement: vacancy at cell 5; gold parked in a non-target low cell;
    # greys fill the rest, one of them in cell 4. Pushing the cell-4 tile toward -x meets
    # a two-tile chain (cells 2, 0) against the west wall; pushing it toward +y meets the
    # vacancy. Same force both times — only the vacancy direction moves a full cell.
    env.reset(seed=61)
    step(60)
    tgt = int(scene.target[0])
    gold_home = next(j for j in (0, 1, 2, 3) if j != tgt)
    grey_homes = [j for j in (0, 1, 2, 3, 4) if j != gold_home]
    arrange(gold_home, grey_homes)
    pusher = TILE_NAMES[1 + grey_homes.index(4)]  # the grey seated in cell 4
    zs = []
    d_blocked = push(pusher, -1.0, 0.0, 300, force=0.8)
    p = scene.world_to_local(scene.tiles[pusher].data.root_pos_w)[0]
    zs.append(float(p[2]))
    print(f"[smoke] blocked push: along-axis displacement {d_blocked * 1000:.1f} mm, "
          f"still cell {tile_cell(pusher)}", flush=True)
    report("blocked-push")
    check("occupancy (neg): pushing into a fully occupied chain advances < 30 mm and "
          "never leaves cell 4 (tile below the z gate throughout)",
          d_blocked < 0.030 and tile_cell(pusher) == 4 and max(zs) < c.z_gate)
    # Re-square the pusher on the cell-4 centre before the positive control: the blocked
    # push left it ~14 mm off the column slot line, and the knob-in-slot move law
    # (correctly) refuses lateral travel from an off-line pose.
    put_tile(pusher, *c.cells[4], settle=30)
    d_free = push(pusher, 0.0, 1.0, 300, force=0.8)
    print(f"[smoke] vacancy push: along-axis displacement {d_free * 1000:.1f} mm, "
          f"now cell {tile_cell(pusher)}", flush=True)
    report("vacancy-push")
    check("occupancy (pos): the SAME force toward the vacancy moves the tile a full "
          "cell and seats it there (positive control)",
          d_free > 0.055 and tile_cell(pusher) == 5)

    # =========================== 12./13. settle gate + latched credit =======================
    env.reset(seed=71)
    step(60)
    tgt = int(scene.target[0])
    g0 = int(scene.gold_start[0])
    occ_t = next((n for n in TILE_NAMES[1:] if tile_cell(n) == tgt), None)
    if occ_t is not None:  # clear the target cell: park that grey in the vacancy
        put_tile(occ_t, *c.cells[int(scene.blank[0])], settle=30)
    scene.tiles["gold"].write_root_state_to_sim(
        tile_state(*c.cells[tgt], 0.001, vel=(0.35, 0.0, 0.0)), ids)
    step(1)
    v_now = float(scene.tiles["gold"].data.root_lin_vel_w[0].norm())
    s_move, ok_move = judge()
    print(f"[smoke] settle-gate probe: |v|={v_now:.2f} gold_cell={int(scene.gold_cell()[0])} "
          f"score={s_move:.3f} success={ok_move}", flush=True)
    check("settle gate: gold IN the target cell but sliding at speed is refused "
          "(progress latches the 0.60 cap, success stays False)",
          v_now > c.settle_speed and not ok_move and abs(s_move - 0.60) < 2e-3)
    put_tile("gold", *c.cells[g0], settle=90)  # yank it back out before it can settle
    report("yanked-out")
    check("latched credit: gold yanked back to its start cell keeps the 0.60; no "
          "success without the live seated-in-target state",
          int(scene.gold_cell()[0]) == g0
          and abs(float(scene.score()[0]) - 0.60) < 2e-3
          and not bool(scene.success()[0]))

    # =========================== 14./15. audit + final no-NaN ===============================
    check("audit: success() was never True at any judged point; max score 0.60",
          never_success[0] and max_score[0] <= 0.60 + 2e-3)
    check("final: all tile states finite",
          all(bool(torch.isfinite(scene.tiles[n].data.root_state_w).all())
              for n in TILE_NAMES))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tile_shuffle")
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
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
