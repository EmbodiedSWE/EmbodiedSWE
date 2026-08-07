"""Smoke / rubric-REJECTION battery for CaskWeightSortScene (sim_gen task
`stack_wine_i48`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — probe each keg with the same nudge, rank by roll
distance, drop each keg into its color cradle — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: kegs on the floor at capsule
                            radius height, settled; score ~0 at rest, no success;
  3.  randomization A     — READBACK over 6 seeded resets: the cradle color
                            arrangement (bay x positions) and keg spawn xy vary;
  4.  randomization B     — the MASS permutation is real: PhysX mass READBACK equals
                            the nominal {heavy, middle, light} table under the scene's
                            rank bookkeeping, and the permutation varies across seeds;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's identity-blind racking ("put each object on a
                            rack") = all three kegs PHYSICALLY WEDGED in cradles but
                            in a wrong permutation (a derangement): every keg seated
                            geometrically, settled, yet NOT success, score <= 0.20;
  7.  one-correct swap    — only the heavy keg in RED, the other two swapped: NOT
                            success, score <= 0.40 (one identity credit, not three);
  8.  position near-miss  — the right keg lying on the floor BESIDE its cradle
                            (60 mm off in y): not seated, NOT success;
  9.  across-rails        — the right keg centred on its cradle but with its axis
                            ACROSS the trough (90 deg yaw), resting on the rails'
                            top edges: axis + z clauses reject -> NOT success;
  10. two-kegs-one-bay    — the right keg seated, a SECOND keg dropped on top of it
                            in the same bay: the second keg (on top at ~seat+2r, or
                            rolled off) is never seated in that bay, NOT success;
  11. hover loophole      — the right keg held in the AIR over its cradle (judged
                            transiently): NOT success and the seat latch stays 0 (the
                            24-substep continuity gate never fires);
  12. latched credit      — seating one correct keg then REMOVING it leaves the
                            latched score unchanged and success stays gone;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_wine_i48.smoke --headless
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

ROLE_NAMES = scene_mod.ROLE_NAMES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cask_weight_sort")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.75)) + o),
                                tuple(np.array((0.0, 0.05, 0.05)) + o),
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

    def finite() -> bool:
        return bool(all(torch.isfinite(b.data.root_state_w).all() for b in scene.bays)
                    and all(torch.isfinite(k.data.root_state_w).all() for k in scene.casks))

    def report(tag: str) -> None:
        s, ok = judge()
        seated = [bool(v) for v in scene.seated_correct()[0]]
        lin = [float(k.data.root_lin_vel_w[0].norm()) for k in scene.casks]
        ang = [float(k.data.root_ang_vel_w[0].norm()) for k in scene.casks]
        print(f"[smoke] {tag:16s} | seated_correct={seated} "
              f"disp={[f'{float(v):.0f}' for v in scene.disp_latch[0]]} "
              f"corr={[f'{float(v):.0f}' for v in scene.correct_latch[0]]} "
              f"maxctr={scene.max_seat_ctr[0].tolist()} "
              f"lin={[f'{v:.3f}' for v in lin]} ang={[f'{v:.2f}' for v in ang]} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def place_cask(i: int, x: float, y: float, z: float,
                   quat=(1.0, 0.0, 0.0, 0.0), settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        scene.casks[i].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_into_bay(i: int, role: int, settle_steps: int = 240) -> None:
        """Hover ABOVE role's cradle and DROP through contact so the keg physically
        wedges (same mechanism as the solution — used here to build WRONG racks)."""
        tgt = scene._bay_pos()[0, role]
        place_cask(i, float(tgt[0]), float(tgt[1]), c.seat_z + 0.060,
                   settle_steps=settle_steps)

    def cask_state(i: int) -> torch.Tensor:
        return (scene.casks[i].data.root_pos_w - scene.env_origins)[0]

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    heights = [float(cask_state(i)[2]) for i in range(3)]
    check("settle: states finite; all three kegs on the floor at capsule-radius "
          f"height (readback z={[f'{h:.3f}' for h in heights]}) and settled",
          finite() and all(abs(h - c.cask_r) < 0.006 for h in heights)
          and bool(scene.settled()[0].all()))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    bay_reads, cask_reads, perms, mass_reads = [], [], [], []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        bp = scene._bay_pos()[0]
        bay_reads.append([float(bp[j, 0]) for j in range(3)] + [float(bp[0, 1])])
        cp = scene._cask_pos()[0]
        cask_reads.append([float(v) for v in cp[:, :2].reshape(-1)])
        perms.append(tuple(scene.rank_of_cask[0].tolist()))
        mb = [float(scene.casks[i].root_physx_view.get_masses().cpu().view(-1)[0])
              for i in range(3)]
        mass_reads.append(mb)
    barr = np.array(bay_reads)
    carr = np.array(cask_reads)
    print(f"[smoke] bay readback (red_x, yellow_x, green_x, row_y):\n{barr}", flush=True)
    print(f"[smoke] mass permutations across seeds: {perms}", flush=True)
    print(f"[smoke] PhysX mass readback across seeds:\n{np.array(mass_reads)}", flush=True)
    bay_orderings = {tuple(np.argsort(row[:3]).tolist()) for row in barr}
    cask_spread = float((carr.max(axis=0) - carr.min(axis=0)).max())
    check("randomization A: the cradle color arrangement varies across seeded resets "
          "(readback) and keg spawn xy varies",
          len(bay_orderings) >= 2 and (barr.max(0) - barr.min(0))[:3].max() > 0.05
          and cask_spread > 0.02)
    nominal = sorted(c.masses)
    masses_ok = all(
        all(abs(a - b) < 1e-3 for a, b in zip(sorted(mb), nominal))
        and all(abs(mb[i] - c.masses[perm[i]]) < 1e-3 for i in range(3))
        for mb, perm in zip(mass_reads, perms))
    check("randomization B: PhysX mass READBACK matches the nominal mass table under "
          "the scene's rank bookkeeping, and the permutation varies across seeds",
          masses_ok and len(set(perms)) >= 2)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy (identity-blind racking) ==================
    # The seed's whole plan is "put each object on a rack" without ever asking WHICH.
    # Build the physically perfect version of that: all three kegs wedged in cradles,
    # settled — but in a DERANGEMENT of the correct assignment. Zero identity credit.
    env.reset(seed=41)
    step(10)
    rank = scene.rank_of_cask[0].tolist()  # rank[i] = correct role of cask i
    derange = [(r + 1) % 3 for r in rank]  # every keg in a WRONG cradle
    for i in range(3):
        drop_into_bay(i, derange[i])
    step(120)
    report("seed-derangement")
    s, ok = judge()
    seated_any = scene.seated_matrix()[0]  # (3 casks, 3 roles)
    for i in range(3):
        rel = scene._cask_pos()[0, i] - scene._bay_pos()[0, derange[i]]
        ax = scene._cask_axis()[0, i]
        print(f"[smoke]   derange cask_{i} -> {ROLE_NAMES[derange[i]]}: "
              f"rel=({float(rel[0]):+.3f},{float(rel[1]):+.3f},{float(rel[2]):+.3f}) "
              f"axis_x={float(ax[0]):+.2f} seated={bool(seated_any[i, derange[i]])} "
              f"settled={bool(scene.settled()[0, i])}", flush=True)
    geom_ok = all(bool(seated_any[i, derange[i]]) for i in range(3))
    check("seed strategy: all three kegs PHYSICALLY WEDGED in cradles (verified "
          "seated geometrically) but in a wrong permutation — NOT success, "
          f"score <= 0.20 (got {s:.3f})",
          geom_ok and bool(scene.settled()[0].all()) and not ok and s <= 0.20)

    # =========================== 7. one-correct swap ========================================
    env.reset(seed=51)
    step(10)
    rank = scene.rank_of_cask[0].tolist()
    heavy = rank.index(0)  # cask with rank 0 (RED)
    others = [i for i in range(3) if i != heavy]
    drop_into_bay(heavy, 0)  # correct
    drop_into_bay(others[0], rank[others[1]])  # swapped
    drop_into_bay(others[1], rank[others[0]])
    step(120)
    report("one-correct")
    s, ok = judge()
    check("one-correct swap: heavy keg in RED but the other two swapped — NOT "
          f"success, exactly one identity credit (score <= 0.40, got {s:.3f})",
          bool(scene.seated_correct()[0, heavy]) and not ok and 0.15 <= s <= 0.40)

    # =========================== 8. position near-miss ======================================
    env.reset(seed=61)
    step(10)
    rank = scene.rank_of_cask[0].tolist()
    heavy = rank.index(0)
    tgt = scene._bay_pos()[0, 0]
    place_cask(heavy, float(tgt[0]), float(tgt[1]) - c.plate_y / 2 - 0.060,
               c.cask_r + 0.002, settle_steps=120)  # on the floor BESIDE the cradle
    report("near-miss-pos")
    s, ok = judge()
    check("position near-miss: the right keg on the floor 60 mm beside its cradle — "
          "not seated, NOT success",
          not bool(scene.seated_correct()[0, heavy]) and not ok)

    # =========================== 9. across-rails axis loophole ==============================
    env.reset(seed=71)
    step(10)
    rank = scene.rank_of_cask[0].tolist()
    heavy = rank.index(0)
    tgt = scene._bay_pos()[0, 0]
    place_cask(heavy, float(tgt[0]), float(tgt[1]), c.seat_z + 0.030,
               quat=yaw_quat(math.pi / 2), settle_steps=120)  # axis ACROSS the trough
    report("across-rails")
    s, ok = judge()
    ax = scene._cask_axis()[0, heavy]
    check("across-rails: the right keg centred on its cradle but lying ACROSS the "
          f"trough (axis_x={float(ax[0]):+.2f}) — axis clause rejects, NOT success",
          abs(float(ax[0])) < math.cos(math.radians(c.seat_axis_deg))
          and not bool(scene.seated_correct()[0, heavy]) and not ok)

    # =========================== 10. two-kegs-one-bay loophole ==============================
    env.reset(seed=81)
    step(10)
    rank = scene.rank_of_cask[0].tolist()
    heavy = rank.index(0)
    other = [i for i in range(3) if i != heavy][0]
    drop_into_bay(heavy, 0)  # the right keg, correctly seated
    tgt = scene._bay_pos()[0, 0]
    place_cask(other, float(tgt[0]), float(tgt[1]), c.seat_z + 2 * c.cask_r + 0.060,
               settle_steps=240)  # a SECOND keg dropped onto the same bay
    report("two-kegs-one-bay")
    s, ok = judge()
    rel = cask_state(other) - scene._bay_pos()[0, 0]
    check("two-kegs-one-bay: a second keg dropped onto an occupied bay (rests at "
          f"rel=({float(rel[0]):+.3f},{float(rel[1]):+.3f},{float(rel[2]):+.3f})) is "
          "never seated in that bay — rejected, NOT success",
          not bool(scene.seated_matrix()[0, other, 0]) and not ok)

    # =========================== 11. hover loophole =========================================
    env.reset(seed=91)
    step(5)
    rank = scene.rank_of_cask[0].tolist()
    heavy = rank.index(0)
    tgt = scene._bay_pos()[0, 0]
    place_cask(heavy, float(tgt[0]), float(tgt[1]), 0.30, settle_steps=2)
    s, ok = judge()
    corr_hover = float(scene.correct_latch[0, heavy])
    z_high = float(cask_state(heavy)[2])
    # remove it before it can fall into the cradle
    place_cask(heavy, float(tgt[0]) + 0.30, float(tgt[1]) - 0.35, c.cask_r + 0.002,
               settle_steps=10)
    check("hover loophole: the right keg held in the air over its cradle — NOT "
          "success while aloft and the seat latch stays 0 (continuity gate)",
          z_high > 0.15 and not ok and corr_hover == 0.0)

    # =========================== 12. latched credit survives regression =====================
    env.reset(seed=101)
    step(10)
    rank = scene.rank_of_cask[0].tolist()
    heavy = rank.index(0)
    drop_into_bay(heavy, 0)
    step(60)
    report("one-seated")
    s_in, _ = judge()
    tgt = scene._bay_pos()[0, 0]
    place_cask(heavy, float(tgt[0]) + 0.30, float(tgt[1]) - 0.40, c.cask_r + 0.002,
               settle_steps=60)
    report("seated-removed")
    s_out, ok = judge()
    check("latched credit: removing the one correctly seated keg leaves the latched "
          f"score unchanged ({s_in:.3f} -> {s_out:.3f}) and success stays gone",
          s_in >= 0.20 and abs(s_out - s_in) < 0.02 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cask_weight_sort")
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
    except BaseException:  # noqa: BLE001 — die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
