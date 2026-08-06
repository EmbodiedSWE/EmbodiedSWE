"""Smoke / oracle test for SowingTrayScene (sim_gen task `sweep_to_dustpan_i38`) —
NullRobot, teleport-oracle, RECORDED.

Battery (registry_i28 / pen_holder skeleton):
  1-2. reset-sane      — states finite; every present pellet genuinely inside the hopper
                         (readback: xy within the inner radius of the hopper axis, z below
                         the rim), absent pellets parked in the depot; zero good cells,
                         score exactly 0, no success;
  3-4. randomization   — READBACK across 8 seeded resets: pellet count varies, tray yaw
                         genuinely rotates, tray AND hopper xy jitter move;
  5. null-policy-fails — 2 s idle: score stays 0, no latch, no success;
  6-8. oracle x3 seeds — extract one pellet at a time (kinematic lift, latching
                         extraction), carry over the bed, real 38 mm drop into its own
                         cell; success() and score 1.0 on 3 seeds;
  9-11. rubric ladder  — extraction latch alone -> 0.10; each planted pellet adds
                         0.70/k; partials strictly increase and stay < 1.0; the last
                         placement flips success -> 1.0;
 12. negative A (seed) — the seed's plan class, AGGREGATE the pile into the container:
                         every pellet dumped into ONE cell -> the cell is over-occupied
                         and VOIDED, no success, score pinned near 0;
 13. negative B (sweep)— the seed's transport channel, PLANAR SWEEPING: pellets moved
                         along the ground flush against the planter's base -> score
                         exactly 0 (the raised walled bed is unreachable from the plane);
 14. near-miss (perch) — a pellet authored balanced on the wall top between two cells:
                         inside both cells' z band story but rejected by the xy gate of
                         BOTH (counts nowhere, occupies nothing);
 15. near-miss (hover) — a pellet in a cell's airspace but NOT on the floor: occupies
                         (would void the cell for others) yet earns nothing; the same
                         pellet then FALLS 26 mm and becomes good — the z tolerance is a
                         real physical boundary;
 16-17. exact-occupancy— two pellets properly on ONE cell floor -> cell voided (0 good);
                         moving one to a free cell -> BOTH cells good, score strictly
                         increases (over-filling is recoverable, and counted exactly);
 18. reset-clears      — a fresh reset after a latched episode reads clean;
 19-20. calibration    — drop sweep at growing tray-frame xy offset into one cell
                         (3 seeds each, tray yaw re-sampled per seed): centred / 6 mm /
                         10 mm drops (funnel edge 18 mm) plant 3/3; a 45 mm offset (past
                         the wall) NEVER ends in the target cell; middles published.

Run (forge): python -u -m simgen_tasks.sweep_to_dustpan_i38.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
    from simgen_tasks.sweep_to_dustpan_i38 import scene as scene_mod  # noqa: F401 (registers)
except ImportError:  # pragma: no cover - local run fallback
    import scene as scene_mod  # noqa: F401


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _state13(scene, xy, z: float) -> torch.Tensor:
    """A 13-dim root state at env-local (xy, z), identity rotation, zero velocities."""
    env = scene.env
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = float(xy[0])
    st[:, 1] = float(xy[1])
    st[:, 2] = float(z)
    st[:, 3] = 1.0
    st[:, 0:3] += env.iscene.env_origins
    return st


def _drop_z(c) -> float:
    """Release height: pellet centre 6 mm above the wall rim — a real ~38 mm fall."""
    return c.bed_z + c.wall_h + c.pellet_r + 0.006


def oracle_solution(scene_or_env, step_fn=None, verbose: bool = True) -> bool:
    """Teleport-oracle for the CURRENT episode: for each present pellet in turn, lift it
    kinematically straight up out of the hopper (well above `extract_z` — the latch fires
    on real state), carry it over the bed, and release it 6 mm above the rim of its own
    cell (cells assigned in index order); the 38 mm drop and settling are REAL physics,
    judged by `good_cells`. One centred retry per pellet. Returns True iff scene.success()."""
    env = _env_of(scene_or_env)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=env.device)
    all_ids = torch.arange(env.num_envs, device=env.device)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def _until(pred, max_steps: int = 300, poll: int = 12) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            _step(poll)
            waited += poll
            if pred():
                return True
        return False

    present_ids = [i for i in range(c.n_pellets) if bool(scene.present[0, i])]
    centers = scene.cell_centers_local()[0]  # (C, 2)
    z_hi = 0.20

    for m, i in enumerate(present_ids):
        pos = scene._pellet_pos_local()[0, i]
        # straight up out of the hopper mouth (extraction latches on the real transit)
        scene.pellets[i].write_root_state_to_sim(
            _state13(scene, (float(pos[0]), float(pos[1])), z_hi), all_ids)
        _step(2)
        tgt = (float(centers[m, 0]), float(centers[m, 1]))
        scene.pellets[i].write_root_state_to_sim(_state13(scene, tgt, z_hi), all_ids)
        _step(2)
        planted = False
        for attempt in range(2):
            scene.pellets[i].write_root_state_to_sim(
                _state13(scene, tgt, _drop_z(c)), all_ids)
            if _until(lambda m=m: bool(scene.good_cells()[0, m]), max_steps=240):
                planted = True
                break
            if verbose and attempt == 0:
                print(f"[oracle]   drop retry: pellet_{i} missed cell {m}", flush=True)
        if verbose:
            print(f"[oracle] sow {m + 1}/{len(present_ids)}: pellet_{i} -> cell {m} "
                  f"ok={planted} score={float(scene.score()[0]):.3f}", flush=True)
    done = _until(lambda: bool(scene.success()[0]))
    if verbose:
        print(f"[oracle] finished: success={done} score={float(scene.score()[0]):.3f}",
              flush=True)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sowing_tray")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.35, 0.90)) + o),
                                tuple(np.array((-0.05, 0.0, 0.08)) + o),
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

    def settle_until(pred, max_steps: int = 300, poll: int = 12) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def score0() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        occ = scene.cell_occupancy()[0].tolist()
        good = scene.good_cells()[0].int().tolist()
        print(f"[smoke] {tag:14s} | present={scene.present[0].int().tolist()} "
              f"occ={occ} good={good} extracted={bool(scene.extracted[0])} "
              f"score={score0():.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def cell_xy(k: int, off_tray=(0.0, 0.0)) -> tuple:
        """Env-local xy of cell k's centre displaced by a TRAY-FRAME offset."""
        ctr = scene.cell_centers_local()[0, k]
        yaw = float(scene.tray_yaw[0])
        dx = off_tray[0] * math.cos(yaw) - off_tray[1] * math.sin(yaw)
        dy = off_tray[0] * math.sin(yaw) + off_tray[1] * math.cos(yaw)
        return (float(ctr[0]) + dx, float(ctr[1]) + dy)

    def sow_kin(i: int, k: int, off_tray=(0.0, 0.0), dz: float = 0.0) -> None:
        """Kinematic release of pellet i above cell k (tray-frame offset), real fall."""
        scene.pellets[i].write_root_state_to_sim(
            _state13(scene, cell_xy(k, off_tray), _drop_z(c) + dz), all_ids)

    def present_ids() -> list:
        return [i for i in range(c.n_pellets) if bool(scene.present[0, i])]

    def all_settled(ids_) -> bool:
        return all(bool(scene.settled()[0, j]) for j in ids_)

    # =========================== 1-2. reset-sane ===========================================
    torch.manual_seed(11)
    env.reset()
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in scene.pellets)
    hop = (scene.hopper.data.root_pos_w[0] - scene.env_origins[0])
    pos = scene._pellet_pos_local()[0]
    in_hopper = True
    for i in present_ids():
        d = float((pos[i, :2] - hop[:2]).norm())
        in_hopper &= d < c.hopper_inner_r and float(pos[i, 2]) < c.hopper_rim_z
    check("reset: states finite and every present pellet rests INSIDE the hopper "
          "(readback: within the inner radius, below the rim)", finite and in_hopper)
    parked = all(float(pos[i, 2]) < 0.05 and float(pos[i, 0]) > 0.8
                 for i in range(c.n_pellets) if not bool(scene.present[0, i]))
    kk = int(scene.present[0].sum())
    check("reset: pellet count in {4..6}, absent pellets parked in the depot, zero good "
          "cells, score exactly 0, no success",
          c.min_present <= kk <= c.n_pellets and parked
          and int(scene.good_cells()[0].sum()) == 0 and score0() == 0.0
          and not bool(scene.success()[0]))

    # =========================== 3-4. randomization is real ================================
    counts, yaws, trays, hops = [], [], [], []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(s)
        env.reset()
        step(1)
        counts.append(int(scene.present[0].sum()))
        yaws.append(float(scene.tray_yaw[0]))
        trays.append(scene.tray_c[0].tolist())
        hops.append(scene.hopper_c[0].tolist())
    print(f"[smoke] randomization readback: counts={counts} "
          f"yaws={[round(v, 2) for v in yaws]}", flush=True)
    yaw_spread = max(yaws) - min(yaws)
    check("randomization: pellet count varies (>= 2 sizes) AND the tray yaw genuinely "
          f"rotates (spread {yaw_spread:.2f} rad)",
          len(set(counts)) >= 2 and yaw_spread > 0.6)
    ta, ha = np.array(trays), np.array(hops)
    t_spread = float((ta.max(axis=0) - ta.min(axis=0)).min())
    h_spread = float((ha.max(axis=0) - ha.min(axis=0)).min())
    check(f"randomization: tray xy (min spread {t_spread:.3f} m) and hopper xy "
          f"(min spread {h_spread:.3f} m) both jitter (readback)",
          t_spread > 0.015 and h_spread > 0.015)

    # =========================== 5. null policy fails ======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: 2 s idle -> score 0, no extraction latch, no success",
          score0() == 0.0 and not bool(scene.extracted[0])
          and not bool(scene.success()[0]))

    # =========================== 6-8. oracle x3 ============================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(40)
        kk = int(scene.present[0].sum())
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score 1.0, {kk} pellets sown)",
              ok and bool(scene.success()[0]) and score0() == 1.0)

    # =========================== 9-11. rubric ladder =======================================
    torch.manual_seed(41)
    env.reset()
    step(40)
    ids = present_ids()
    kk = len(ids)
    assert score0() == 0.0, "ladder must start from a clean 0"
    # extraction alone: hover the first pellet above the rim, then put it back in the cup
    hop = (scene.hopper.data.root_pos_w[0] - scene.env_origins[0])
    scene.pellets[ids[0]].write_root_state_to_sim(
        _state13(scene, (float(hop[0]), float(hop[1])), 0.20), all_ids)
    step(3)
    scene.pellets[ids[0]].write_root_state_to_sim(
        _state13(scene, (float(hop[0]), float(hop[1])), 0.075), all_ids)
    ok = settle_until(lambda: abs(score0() - c.w_extract) < 2e-3, max_steps=180)
    report("ladder-extract")
    check("ladder: a genuine lift out of the hopper latches extraction -> score 0.10 "
          "(and nothing is planted yet)", ok and int(scene.good_cells()[0].sum()) == 0)
    mono = True
    s_prev = score0()
    for m, i in enumerate(ids[:-1]):
        sow_kin(i, m)
        expect = c.w_extract + c.w_frac * (m + 1) / kk
        ok = settle_until(lambda e=expect: abs(score0() - e) < 2e-3, max_steps=300)
        mono = mono and ok and score0() >= s_prev - 1e-6 and score0() < 1.0
        s_prev = score0()
        report(f"ladder-sow{m + 1}")
    check(f"ladder: each planted pellet adds 0.70/{kk} — partials strictly increase "
          "and stay < 1.0", mono)
    sow_kin(ids[-1], kk - 1)
    ok = settle_until(lambda: bool(scene.success()[0]) and score0() == 1.0, max_steps=300)
    report("ladder-done")
    check("ladder: the last pellet planted -> success, score exactly 1.0", ok)

    # =========================== 12. negative A: the seed's aggregation ====================
    # The seed's plan class: collect the WHOLE pile into the container in bulk. Dump every
    # pellet into one cell (gentle sequential releases — the kindest possible dump): the
    # cell is over-occupied and VOIDED by exact-occupancy judging.
    torch.manual_seed(51)
    env.reset()
    step(40)
    ids = present_ids()
    for t, i in enumerate(ids):
        sow_kin(i, 0, off_tray=(0.006 * ((t % 3) - 1), 0.006 * ((t // 3) - 0.5)))
        step(50)
    settle_until(lambda: all_settled(ids), max_steps=240)
    report("seed-dump")
    check("negative A (seed strategy): the whole clump dumped into ONE cell -> the cell "
          "is over-occupied and voided, no success, score pinned near 0",
          int(scene.cell_occupancy()[0, 0]) >= 2 and not bool(scene.success()[0])
          and score0() <= 0.25)

    # =========================== 13. negative B: planar sweeping ===========================
    # The seed's transport channel: push the material along the plane to the container.
    # Pellets end flush against the planter's base — at ground level, 41 mm below the
    # lowest counted pose, outside every cell. Sweeping cannot even latch extraction.
    torch.manual_seed(61)
    env.reset()
    step(40)
    ids = present_ids()
    base = scene.tray_c[0].tolist()
    for t, i in enumerate(ids):
        scene.pellets[i].write_root_state_to_sim(
            _state13(scene, (base[0] - 0.17, base[1] - 0.09 + 0.035 * t),
                     c.pellet_r + 0.002), all_ids)
    settle_until(lambda: all_settled(ids), max_steps=240)
    report("planar-sweep")
    check("negative B (seed transport): pellets swept along the ground flush against the "
          "planter's base -> score exactly 0, no latch, no success",
          score0() == 0.0 and not bool(scene.extracted[0])
          and not bool(scene.success()[0]))

    # =========================== 14-15. near-miss tolerance controls =======================
    torch.manual_seed(71)
    env.reset()
    step(40)
    ids = present_ids()
    # perch: balanced on the wall top between cells 0 and 1 (tray-frame x = -pitch/2)
    scene.pellets[ids[0]].write_root_state_to_sim(
        _state13(scene, cell_xy(0, off_tray=(c.pitch / 2, 0.0)),
                 c.bed_z + c.wall_h + c.pellet_r), all_ids)
    env.iscene.update(0.0)  # judge the authored pose (zero velocities)
    check("near-miss (perch): a pellet on the wall top between two cells counts in "
          "NEITHER cell and occupies neither (xy gate rejects both at 36 mm > 28 mm)",
          int(scene.cell_occupancy()[0].sum()) == 0
          and int(scene.good_cells()[0].sum()) == 0)
    # hover: inside cell 2's airspace but off the floor -> occupies, earns nothing; then
    # the same pellet falls 26 mm and becomes good (the z boundary is physical)
    scene.pellets[ids[0]].write_root_state_to_sim(
        _state13(scene, cell_xy(2), c.bed_z + 0.040), all_ids)
    env.iscene.update(0.0)
    mid_air = (int(scene.cell_occupancy()[0, 2]) == 1
               and not bool(scene.good_cells()[0, 2]))
    ok = settle_until(lambda: bool(scene.good_cells()[0, 2]), max_steps=240)
    report("near-miss")
    check("near-miss (hover): a pellet in the cell's airspace but NOT on the floor "
          "occupies-yet-earns-nothing; after its real 26 mm fall it becomes good",
          mid_air and ok)

    # =========================== 16-17. exact-occupancy + recovery =========================
    # (continues the same episode: pellet ids[0] already good in cell 2)
    sow_kin(ids[1], 3, off_tray=(-0.011, 0.0))
    step(40)
    sow_kin(ids[2], 3, off_tray=(0.011, 0.0))
    settle_until(lambda: all_settled([ids[1], ids[2]]), max_steps=240)
    report("two-in-one")
    s_before = score0()
    check("exact-occupancy: TWO pellets resting properly on one cell floor -> the cell "
          "is voided (occupancy 2, not good)",
          int(scene.cell_occupancy()[0, 3]) == 2 and not bool(scene.good_cells()[0, 3]))
    sow_kin(ids[2], 4)
    ok = settle_until(lambda: bool(scene.good_cells()[0, 3])
                      and bool(scene.good_cells()[0, 4]), max_steps=300)
    report("split-up")
    check("exact-occupancy recovery: moving the extra pellet to a free cell makes BOTH "
          "cells good and the score strictly increases",
          ok and score0() > s_before + 0.1)

    # =========================== 18. reset clears latches ==================================
    env.reset()
    step(5)
    check("reset clears the latches (fresh episode reads clean, score 0)",
          not bool(scene.extracted[0]) and score0() == 0.0
          and int(scene.good_cells()[0].sum()) == 0)

    # =========================== 19-20. calibration: drop-capture sweep ====================
    # Raw single drops (no retry) released 6 mm above the rim over cell 1, displaced along
    # the TRAY-FRAME x axis (tray yaw re-sampled per seed). Geometric funnel = cell/2 -
    # pellet_r = 18 mm; the wall spans 32-40 mm. Middles published, deterministic ends
    # asserted.
    print("[smoke] CALIBRATION SWEEP (tray-frame drop offset -> plant rate, 3 seeds each)",
          flush=True)
    results: dict[float, int] = {}
    escapes = 0
    for off_mm in (0.0, 6.0, 10.0, 14.0, 18.0, 24.0, 32.0, 45.0):
        hits = in_target = 0
        for seed in range(3):
            torch.manual_seed(100 + 10 * int(off_mm) + seed)
            env.reset()
            step(10)
            i = present_ids()[0]
            sow_kin(i, 1, off_tray=(off_mm / 1000.0, 0.0))
            hit = settle_until(lambda: bool(scene.good_cells()[0, 1]), max_steps=240)
            hits += int(hit)
            in_target += int(scene.cell_occupancy()[0, 1])
            print(f"[smoke]   off={off_mm:.0f}mm seed={seed}: planted={hit} "
                  f"occ={scene.cell_occupancy()[0].tolist()}", flush=True)
        results[off_mm] = hits
        if off_mm == 45.0:
            escapes = in_target
    print("[smoke] SWEEP RESULT: "
          + " | ".join(f"{k:.0f}mm: {v}/3" for k, v in results.items()), flush=True)
    check("sweep: centred, 6 mm and 10 mm drops (funnel edge 18 mm) plant 3/3",
          results[0.0] == 3 and results[6.0] == 3 and results[10.0] == 3)
    check("sweep: a 45 mm offset (past the wall) never ends in the target cell",
          results[45.0] == 0 and escapes == 0)

    # =========================== save + verdict ============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sowing_tray")
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
