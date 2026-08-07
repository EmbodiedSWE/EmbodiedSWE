"""Smoke / rubric-REJECTION battery for TileShuntScene (sim_gen task
`empty_dishwasher_i23`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — plan a make-way move sequence and execute every
one-cell slide under a floating-hand force controller — is the acceptance evidence
that the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no
probe in this battery ever reaches success(), and a final audit check asserts
exactly that.

  1.  settle/no-NaN       — reset layout settles finite: all five tiles verified at
                            their sampled cells by frame-frame READBACK, score ~0;
  2.  SEED strategy       — the seed task's plan (take the payload OUT of the
                            appliance) teleport-constructed: the RED tile placed on
                            the floor OUTSIDE the frame -> out of the captive band,
                            score ~0, no success — extraction earns NOTHING here;
  3-4. randomization      — READBACK over 8 seeded resets: frame xy + yaw vary; the
                            red start cell, the empty cell and the goal corner all
                            take several values; the goal derived from the MARKER
                            position always matches the rubric's goal buffer, and
                            the red tile never starts on the goal;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  captivity yank      — 2 simulated seconds of an 8 N straight-up pull on the
                            RED tile (5.4x its weight, pure contact): the 60 mm
                            plate bears on the 24 mm lid slots, the tile origin
                            never rises above the lid underside -> lift-out is
                            physically impossible;
  7.  blocked line        — 2 s of a 2.5 N push (the solver's own force budget) on
                            the middle tile of the FULL row, along the row: the
                            line jams against the frame wall — readback shows the
                            tile barely moves, occupancy unchanged -> only slides
                            INTO the empty cell are physically possible;
  8.  near-miss cell      — RED tile parked centered in a cell ADJACENT to the
                            goal: partial progress only, NOT success;
  9.  near-miss offset    — RED tile 28 mm off the goal cell centre (> 18 mm
                            tolerance, goal cell left empty): NOT at goal, NOT
                            success;
  10. lid percher         — RED tile balanced ON TOP of the lid directly above the
                            goal cell: correct xy, but above the captive band ->
                            NOT success (transient judged probe);
  11. wrong object        — a WHITE tile placed in the goal cell, red untouched:
                            NOT success, score <= 0.20 (identity matters);
  12. constraint          — RED tile placed at the goal but the goal cell's WHITE
                            occupant teleported OUT of the frame: red-at-goal is
                            true yet success is rejected — every tile must stay
                            captive (you cannot make way by removing tiles);
  13. latched credit      — progress credit survives sliding the red tile BACK to
                            its start cell (credit never evaporates, no success);
  14. monotonicity        — a deeper advance toward the goal latches strictly more
                            progress credit;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.empty_dishwasher_i23.smoke --headless
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
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tile_shunt")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.50, -0.50, 0.42)) + o),
                                tuple(np.array((0.00, 0.00, 0.03)) + o),
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

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def frame_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
        q = scene.frame.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def occupancy() -> tuple[list[int], int]:
        """Nearest-cell READBACK of all five tiles + the empty cell."""
        loc = scene.tiles_local()[0, :, :2]
        d = (loc[:, None, :] - scene.cell_xy[None, :, :]).norm(dim=-1)
        cells = [int(v) for v in d.argmin(dim=1)]
        empty = ({0, 1, 2, 3, 4, 5} - set(cells)).pop()
        return cells, empty

    def marker_goal_cell() -> int:
        """Goal corner derived from the MARKER position alone (frame-frame signs)."""
        ml = scene._frame_local(scene.marker.data.root_pos_w)[0]
        col = 0 if float(ml[0]) < 0 else 2
        row = 0 if float(ml[1]) < 0 else 1
        return row * 3 + col

    def report(tag: str) -> None:
        cells, empty = occupancy()
        rl = scene.tiles_local()[0, 0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | cells={cells} empty={empty} "
              f"goal={int(scene.goal_cell[0])} "
              f"red_local=({float(rl[0]):+.3f},{float(rl[1]):+.3f},{float(rl[2]):.3f}) "
              f"in_band={[bool(v) for v in scene.in_band()[0]]} "
              f"latches=({float(scene.moved_latch[0]):.2f},{float(scene.prog_latch[0]):.2f},"
              f"{float(scene.atgoal_latch[0]):.0f}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_tile_local(t: int, lx: float, ly: float, z: float,
                         settle_steps: int = 30) -> None:
        """Probe write of tile t at FRAME-LOCAL (lx, ly), world height z, yaw-aligned
        with the frame, then REAL physics steps before judging (the zero-step trap)."""
        fp, fyaw = frame_pose()
        ca, sa = math.cos(fyaw), math.sin(fyaw)
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(fp[0]) + ca * lx - sa * ly
        st[:, 1] = float(fp[1]) + sa * lx + ca * ly
        st[:, 2] = z
        st[:, 3] = math.cos(fyaw / 2)
        st[:, 6] = math.sin(fyaw / 2)
        st[:, 0:3] += scene.env_origins
        scene.tiles[t].write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def place_at_cell(t: int, cell: int, settle_steps: int = 30) -> None:
        lx, ly = (float(v) for v in scene.cell_xy[cell])
        write_tile_local(t, lx, ly, c.rest_z + 0.001, settle_steps)

    def arrange(red_lxy: tuple[float, float], white_cells: list[int],
                red_z: float | None = None, settle_steps: int = 40) -> None:
        """Write ALL five tiles into one constructed probe arrangement in one go
        (no interpenetration bookkeeping): red at frame-local (x, y), the four
        whites centered at `white_cells`, then real settle steps."""
        write_tile_local(0, red_lxy[0], red_lxy[1],
                         c.rest_z + 0.001 if red_z is None else red_z, settle_steps=0)
        for w, cell in zip((1, 2, 3, 4), white_cells):
            lx, ly = (float(v) for v in scene.cell_xy[cell])
            write_tile_local(w, lx, ly, c.rest_z + 0.001, settle_steps=0)
        step(settle_steps)

    def cell_cr(k: int) -> tuple[int, int]:
        return k % 3, k // 3

    def next_cell_toward(r: int, g: int) -> int:
        """One-cell manhattan step from r toward g (column first)."""
        rc, rr = cell_cr(r)
        gc, gr = cell_cr(g)
        if gc != rc:
            return r + (1 if gc > rc else -1)
        return r + (3 if gr > rr else -3)

    def vacate(cell: int) -> None:
        """Make `cell` empty by moving its WHITE occupant into the current empty
        cell (probe bookkeeping so constructed poses never interpenetrate)."""
        cells, empty = occupancy()
        if empty == cell:
            return
        w = cells.index(cell)
        place_at_cell(w, empty, settle_steps=15)

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    cells, empty = occupancy()
    want = [int(v) for v in scene.tile_cell0[0]]
    loc = scene.tiles_local()[0]
    off = (loc[:, :2] - scene.cell_xy[torch.tensor(want, device=device)]).norm(dim=-1)
    fin0 = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in scene.tiles)
    fin0 = fin0 and bool(torch.isfinite(scene.frame.data.root_state_w).all())
    s, ok = judge()
    check("settle: states finite; all five tiles verified at their sampled cells by "
          "frame-frame readback (< 6 mm), settled, score ~0, no success",
          fin0 and bool(scene.settled()[0]) and cells == want
          and float(off.max()) < 0.006 and s <= 0.02 and not ok)

    # =========================== 2. SEED strategy rejected ==================================
    # The seed (rlbench/empty_dishwasher) ends with the payload OUT of the appliance.
    # Constructing that end state here — the red tile on the floor outside the frame —
    # must earn NOTHING: progress credit is gated on the tile staying captive.
    env.reset(seed=13)
    step(30)
    write_tile_local(0, c.inner_hx + 0.20, 0.0, c.tile_h / 2 + 0.001, settle_steps=60)
    report("red-extracted")
    s, ok = judge()
    check("SEED strategy: red tile taken OUT of the frame (the seed's extraction end "
          "state) — out of the captive band, score ~0 (<= 0.02), no success",
          not bool(scene.in_band()[0, 0]) and s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        fp, fyaw = frame_pose()
        cells, empty = occupancy()
        reads.append((float(fp[0]), float(fp[1]), fyaw, cells[0], empty,
                      int(scene.goal_cell[0]), marker_goal_cell()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (frame_x, frame_y, frame_yaw, red_cell, "
          f"empty_cell, goal_cell, marker_goal):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: frame xy + yaw vary across 8 seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.3)
    check("randomization: red start cell, empty cell and goal corner each take "
          "several values; marker-derived goal always matches the rubric goal; red "
          "never starts on the goal (readback)",
          len(set(arr[:, 3])) >= 3 and len(set(arr[:, 4])) >= 3
          and len(set(arr[:, 5])) >= 2 and (arr[:, 5] == arr[:, 6]).all()
          and (arr[:, 3] != arr[:, 5]).all())

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. captivity yank (contact, no teleport) ===================
    # 2 simulated seconds of an 8 N straight-up pull on the red tile — 5.4x its
    # weight. The 60 mm plate bears on the 24 mm lid slots: the tile origin must
    # never rise above the lid underside; lift-out is physically impossible.
    env.reset(seed=41)
    step(30)
    fup = torch.zeros(n, 1, 3, device=device)
    fup[:, 0, 2] = 8.0
    scene.tiles[0].set_external_force_and_torque(fup, zero_wrench, env_ids=all_ids,
                                                 is_global=True)
    step(240)
    z_pulled = float(scene.tiles_local()[0, 0, 2])
    scene.tiles[0].set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(60)
    report("yank-captive")
    s, ok = judge()
    check("captivity yank: 2 s of an 8 N straight-up pull — the plate bears on the "
          f"lid (readback z={z_pulled:.4f} < lid underside {c.lid_z0 - c.tile_h / 2 + 0.004:.4f}), "
          "tile still captive, score <= 0.05, no success",
          z_pulled < c.lid_z0 - c.tile_h / 2 + 0.004 and bool(scene.in_band()[0, 0])
          and s <= 0.05 and not ok)

    # =========================== 7. blocked line jams =======================================
    # Push the middle tile of the FULL row (the row not holding the empty cell)
    # along the row with the solver's whole 2.5 N force budget: the line jams
    # against the frame wall. Only slides INTO the empty cell are possible.
    env.reset(seed=51)
    step(30)
    cells0, empty0 = occupancy()
    full_row = 1 - empty0 // 3
    mid = full_row * 3 + 1
    t_mid = cells0.index(mid)
    p0 = scene.tiles_local()[0, t_mid, :2].clone()
    _fp, fyaw = frame_pose()
    fpush = torch.zeros(n, 1, 3, device=device)
    fpush[:, 0, 0] = 2.5 * math.cos(fyaw)
    fpush[:, 0, 1] = 2.5 * math.sin(fyaw)
    scene.tiles[t_mid].set_external_force_and_torque(fpush, zero_wrench, env_ids=all_ids,
                                                     is_global=True)
    step(240)
    scene.tiles[t_mid].set_external_force_and_torque(zero_wrench, zero_wrench,
                                                     env_ids=all_ids)
    step(30)
    moved_d = float((scene.tiles_local()[0, t_mid, :2] - p0).norm())
    cells1, _e1 = occupancy()
    report("blocked-line")
    _s, ok = judge()
    check("blocked line: 2 s of a 2.5 N push on the middle tile of the FULL row — "
          f"the line jams on the wall (moved {moved_d * 1000:.1f} mm < 30 mm, vs 70 mm "
          "for a real move), occupancy unchanged, no success",
          moved_d < 0.030 and cells1 == cells0 and not ok)

    # =========================== 8. near-miss: adjacent cell ================================
    env.reset(seed=61)
    step(10)
    g = int(scene.goal_cell[0])
    adj = next_cell_toward(g, int(scene.tile_cell0[0, 0]))  # neighbor of g toward red
    whites = [k for k in range(6) if k not in (adj, g)]  # exactly 4 cells
    arrange(tuple(float(v) for v in scene.cell_xy[adj]), whites)
    report("adjacent-cell")
    s, ok = judge()
    check("near-miss cell: red tile parked centered in a cell ADJACENT to the goal — "
          "partial progress only (score <= 0.85), NOT at goal, NOT success",
          not bool(scene.red_at_goal()[0]) and s <= 0.85 and not ok)

    # =========================== 9. near-miss: 28 mm offset =================================
    env.reset(seed=71)
    step(10)
    g = int(scene.goal_cell[0])
    adj = next_cell_toward(g, int(scene.tile_cell0[0, 0]))
    gx, gy = (float(v) for v in scene.cell_xy[g])
    ax, ay = (float(v) for v in scene.cell_xy[adj])
    ux, uy = (ax - gx) / c.pitch, (ay - gy) / c.pitch  # unit toward the free neighbor
    whites = [k for k in range(6) if k not in (adj, g)]
    arrange((gx + 0.028 * ux, gy + 0.028 * uy), whites)
    d_goal = float((scene.tiles_local()[0, 0, :2] - scene.cell_xy[g]).norm())
    report("offset-28mm")
    s, ok = judge()
    check("near-miss offset: red tile 28 mm off the goal cell centre (readback "
          f"{d_goal * 1000:.1f} mm > {c.goal_tol * 1000:.0f} mm tolerance) — NOT at "
          "goal, NOT success",
          d_goal > c.goal_tol and not bool(scene.red_at_goal()[0]) and not ok)

    # =========================== 10. lid percher ============================================
    env.reset(seed=81)
    step(10)
    g = int(scene.goal_cell[0])
    gx, gy = (float(v) for v in scene.cell_xy[g])
    write_tile_local(0, gx, gy, c.lid_z1 + c.tile_h / 2 + 0.001,
                     settle_steps=5)  # transient judged probe: perched, not captive
    zl = float(scene.tiles_local()[0, 0, 2])
    report("lid-percher")
    _s, ok = judge()
    check("lid percher: red tile balanced ON TOP of the lid directly above the goal "
          f"cell — correct xy but above the captive band (readback z={zl:.3f} > "
          f"{c.band_z_hi:.3f}) => NOT success",
          zl > c.band_z_hi and not bool(scene.red_at_goal()[0]) and not ok)

    # =========================== 11. wrong object ===========================================
    env.reset(seed=91)
    step(10)
    g = int(scene.goal_cell[0])
    cells, _e = occupancy()
    if g not in cells:  # goal cell empty at reset: walk white #1 into it
        place_at_cell(1, g, settle_steps=20)
    step(30)
    report("wrong-object")
    s, ok = judge()
    cells, _e = occupancy()
    check("wrong object: a WHITE tile occupies the goal cell while the red tile "
          "never moved — NOT success, score <= 0.20",
          cells.index(g) != 0 and s <= 0.20 and not ok)

    # =========================== 12. constraint: tile removed ===============================
    # Red AT the goal — but the goal's white occupant was teleported OUT of the
    # frame to make room. red_at_goal alone is not success: every tile must stay
    # captive (you cannot make way by removing tiles — the no-extraction contract
    # cuts both ways).
    env.reset(seed=101)
    step(10)
    g = int(scene.goal_cell[0])
    cells, empty = occupancy()
    if empty != g:
        w = cells.index(g)
    else:
        w = 1  # goal already empty: any white becomes the removed tile
    write_tile_local(w, c.inner_hx + 0.25, -0.10, c.tile_h / 2 + 0.001, settle_steps=10)
    place_at_cell(0, g, settle_steps=40)
    report("white-removed")
    s, ok = judge()
    check("constraint: red tile centered at the goal, but a WHITE tile was removed "
          "from the frame — red_at_goal true yet NOT success, score <= 0.85",
          bool(scene.red_at_goal()[0]) and not bool(scene.in_band()[0].all())
          and s <= 0.85 and not ok)

    # =========================== 13-14. latched credit + monotonicity =======================
    # Need an episode where the red tile starts >= 2 cells from the goal, so a
    # one-cell advance is not already the goal (this battery must never succeed).
    d0_seed = None
    for sd in (111, 112, 113, 114, 115, 116, 117, 118, 119, 120):
        env.reset(seed=sd)
        step(3)
        if float(scene.d0[0]) >= 2.0:
            d0_seed = sd
            break
    assert d0_seed is not None, "no seed with d0 >= 2 in the scan window"
    r = int(scene.tile_cell0[0, 0])
    g = int(scene.goal_cell[0])
    nc = next_cell_toward(r, g)
    vacate(nc)
    place_at_cell(0, nc, settle_steps=25)
    s_fwd, _ok1 = judge()
    place_at_cell(0, r, settle_steps=25)
    report("moved-back")
    s_back, ok = judge()
    check("latched credit: advancing the red tile one cell latches credit "
          f"({s_fwd:.3f}), sliding it BACK to its start cell keeps it "
          f"({s_back:.3f}), still no success",
          s_fwd >= 0.10 and abs(s_back - s_fwd) < 0.02 and not ok)

    env.reset(seed=d0_seed)
    step(10)
    r = int(scene.tile_cell0[0, 0])
    g = int(scene.goal_cell[0])
    nc = next_cell_toward(r, g)
    vacate(nc)
    rx, ry = (float(v) for v in scene.cell_xy[r])
    nx, ny = (float(v) for v in scene.cell_xy[nc])
    write_tile_local(0, rx + 0.35 * (nx - rx), ry + 0.35 * (ny - ry),
                     c.rest_z + 0.001, settle_steps=10)
    a_part = float(scene.prog_latch[0])
    write_tile_local(0, rx + 0.80 * (nx - rx), ry + 0.80 * (ny - ry),
                     c.rest_z + 0.001, settle_steps=10)
    a_deep = float(scene.prog_latch[0])
    judge()
    check("monotonicity: a deeper advance toward the goal latches strictly more "
          f"progress credit ({a_part:.3f} < {a_deep:.3f})", a_part + 0.05 < a_deep)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (bool(torch.isfinite(scene.frame.data.root_state_w).all())
           and bool(torch.isfinite(scene.marker.data.root_state_w).all())
           and all(bool(torch.isfinite(b.data.root_state_w).all()) for b in scene.tiles))
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tile_shunt")
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — Kit teardown hangs; fail fast, don't wait for watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
