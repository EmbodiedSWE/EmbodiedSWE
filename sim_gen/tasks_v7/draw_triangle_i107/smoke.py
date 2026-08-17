"""Smoke / rubric-REJECTION battery for BeamBalanceScene (sim_gen task
`draw_triangle_i107`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — closed-loop greedy binary weighing with the
4/2/1-unit brass weights — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) settled outcome and asserts the rubric REJECTS it; no probe in this
battery ever reaches success(), and a final audit asserts exactly that.

Seed-strategy check: the seed (maniskill/draw_triangle) judges a robot-drawn
TRACE along a prescribed path; here there is no path and no marker — tracing is
inexpressible. In its place the battery attacks this task's own cheat surface:

  1-2. settle/premise    — states finite; the hidden stone VERIFIABLY pins the
                           beam on a hard stop with the tilt sign pointing at
                           the stone's side (readback); stone home, weights
                           staged clean, stand home; score ~0, not level, no
                           success.
  3-4. randomization     — READBACK over 10 seeded resets: the stone's unit
                           mass takes >= 3 distinct values 1..7, the stone side
                           flips, stand xy + yaw jitter, the weight slot
                           assignment and weight yaws vary; invariants hold
                           every seed (stone home, weights clean, stand home).
  5.  null policy        — 360 idle steps: the beam is STILL pinned on the
                           stop, never level, score exactly 0, no success.
  6.  swing-through      — the pinned beam is kicked toward level (velocity
                           write, no pose change): level() instants are
                           WITNESSED mid-swing, yet the sustained-still gate
                           stays closed and success never fires — a beam
                           swinging through level earns nothing.
  7.  under-load         — brass totalling (k-1) units seated: the beam stays
                           pinned on the stone's stop — one missing unit is
                           REJECTED by physics (equilibrium past the stop).
  8.  over-load          — brass totalling (k+1) units seated: the beam slams
                           to the OPPOSITE stop — one extra unit is rejected.
  9.  clause D (parked)  — the EXACT load seated (level, sustained-still,
                           stone home, stand home) but a leftover brass weight
                           parked 0.18 m from the stand (touching nothing):
                           weights_clean False -> NOT success, score capped.
  10. clause C (stone)   — the stone teleported off to the floor: the EMPTY
                           balance levels itself and stills — stone_home False
                           -> NOT success (removing the load is not weighing).
  11. clause E (dragged) — the whole balance (with the exact brass load seated,
                           level and still) dragged 0.10 m from its spawn pose:
                           stand_home False -> NOT success.
  12. latched credit     — an under-load seated weight latches the 0.15 seating
                           credit; removing it again leaves the latched score
                           unchanged; never 1.0 without success.
  13. rejection audit    — success() was never True at ANY judged point.
  14. final no-NaN       — all task-object states finite; frames recorded.

Run (forge): python -u -m simgen_tasks.draw_triangle_i107.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

PARK_XY = ((0.45, -0.35), (-0.45, -0.35), (0.45, 0.35))   # far bench corners
TRAY_SPOT = ((0.026, -0.022), (0.026, 0.020), (-0.023, 0.0))
SUBSET = {1: [0], 2: [1], 3: [0, 1], 4: [2], 5: [0, 2], 6: [1, 2], 7: [0, 1, 2]}
UNITS = (1, 2, 4)


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.beam_balance")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 1.05)) + o),
                                tuple(np.array((0.0, 0.0, 0.35)) + o),
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

    def tilt_deg() -> float:
        return math.degrees(float(scene.tilt()[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:18s} | tilt={tilt_deg():+7.2f} deg "
              f"in_tray={scene.weight_in_tray()[0].tolist()} "
              f"stone_home={bool(scene.stone_home()[0])} "
              f"clean={bool(scene.weights_clean()[0])} "
              f"stand_home={bool(scene.stand_home()[0])} "
              f"still={bool(scene.still()[0])} (n={int(scene.still_count[0])}) "
              f"success={ok} score={s:.3f} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- probe placement (instrumentation, not a solution) --------------------------------
    def counter_pan():
        return scene.pans[1 - int(scene.side_idx[0])]

    def drop_weight(widx: int, pan=None) -> None:
        """Hover weight `widx` 12 mm above its spot in the LIVE pan frame, zero
        velocity, and let it fall onto the floor plate."""
        pan = pan if pan is not None else counter_pan()
        h = c.weight_specs[widx][3]
        loc = torch.tensor([TRAY_SPOT[widx][0], TRAY_SPOT[widx][1],
                            -c.stem_len + h / 2 + 0.012], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pan.data.root_pos_w + quat_apply(pan.data.root_quat_w, loc)
        st[:, 3:7] = pan.data.root_quat_w
        scene.weights[widx].write_root_state_to_sim(st, all_ids)

    def park_weight(widx: int, xy=None) -> None:
        h = c.weight_specs[widx][3]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = torch.tensor(xy or PARK_XY[widx], device=device).expand(n, 2)
        st[:, 2] = c.bench_top + h / 2 + 0.003
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.weights[widx].write_root_state_to_sim(st, all_ids)

    def load_units(total: int) -> None:
        """Seat the exact binary subset for `total` units in the counter tray,
        one drop at a time with a short settle between drops."""
        for widx in SUBSET[total]:
            drop_weight(widx)
            step(180)

    def settle(max_rounds: int = 30) -> bool:
        """REAL physics steps until sustained-still (or budget ends)."""
        for _ in range(max_rounds):
            step(30)
            if bool(scene.still()[0]):
                return True
        return False

    def find_seed(lo: int, pred) -> tuple[int, int]:
        """Reset over seeds from `lo` until the stone's unit count satisfies
        `pred` (readback); bounded search."""
        for sd in range(lo, lo + 40):
            env.reset(seed=sd)
            k = int(scene.active_idx[0]) + 1
            if pred(k):
                return sd, k
        print("SIM_GEN_SMOKE: FAIL (no seed found for probe)", flush=True)
        os._exit(1)

    def stand_xy_yaw() -> tuple[float, float, float]:
        p = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        w, x, y, z = (float(v) for v in scene.stand.data.root_quat_w[0])
        return float(p[0]), float(p[1]), math.atan2(2 * (w * z + x * y),
                                                    1 - 2 * (y * y + z * z))

    def weight_yaw(j: int) -> float:
        w, x, y, z = (float(v) for v in scene.weights[j].data.root_quat_w[0])
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def finite_all() -> bool:
        ok = True
        for b in [scene.stand, scene.beam] + scene.pans + scene.stones + scene.weights:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    # =========================== 1-2. settle / premise ======================================
    env.reset(seed=11)
    step(240)
    t0 = tilt_deg()
    k0 = int(scene.active_idx[0]) + 1
    side0 = int(scene.side_idx[0])
    report("reset-pinned")
    s, ok = judge()
    check("premise: states finite; the hidden stone (readback "
          f"{k0}u in {'PanP' if side0 == 0 else 'PanN'}) pins the beam on a hard stop "
          f"({t0:+.1f} deg, |tilt| > 9) with the tilt sign pointing at the stone; "
          "stone home, weights staged clean, stand home",
          finite_all() and abs(t0) > 9.0 and ((t0 < 0) == (side0 == 0))
          and bool(scene.stone_home()[0]) and bool(scene.weights_clean()[0])
          and bool(scene.stand_home()[0]))
    check("reset: score exactly 0, not level, no success",
          s <= 1e-6 and not bool(scene.level()[0]) and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    inv_ok = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
        env.reset(seed=sd)
        step(2)
        sx, sy, syaw = stand_xy_yaw()
        reads.append((int(scene.active_idx[0]) + 1, int(scene.side_idx[0]),
                      sx, sy, syaw,
                      float((scene.weights[0].data.root_pos_w - scene.env_origins)[0, 0]),
                      weight_yaw(2)))
        inv_ok = inv_ok and bool(scene.stone_home()[0]) \
            and bool(scene.weights_clean()[0]) and bool(scene.stand_home()[0])
    arr = np.array(reads)
    print("[smoke] randomization readback (units, side, stand_x, stand_y, "
          f"stand_yaw, w1_x, w4_yaw):\n{arr.round(3)}", flush=True)
    check("randomization: the stone's hidden unit count takes >= 3 distinct values "
          "in 1..7 and the stone side flips across seeds",
          len(set(arr[:, 0])) >= 3 and set(arr[:, 0]) <= set(range(1, 8))
          and len(set(arr[:, 1])) == 2)
    check("randomization: stand xy + yaw jitter, weight slot assignment and "
          "weight yaws vary; invariants hold every seed (stone home, weights "
          "clean, stand home)",
          arr[:, 2].ptp() > 0.008 and arr[:, 3].ptp() > 0.008
          and arr[:, 4].ptp() > 0.05 and arr[:, 5].ptp() > 0.05
          and arr[:, 6].ptp() > 0.5 and inv_ok)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(360)
    report("null-policy")
    s, ok = judge()
    check("null policy: after 360 idle steps the beam is still pinned "
          f"({tilt_deg():+.1f} deg), never level, score exactly 0, no success",
          abs(tilt_deg()) > 9.0 and not bool(scene.level()[0]) and s <= 1e-6 and not ok)

    # =========================== 6. swinging through level earns nothing ====================
    # Kick the pinned beam toward level (pure velocity write about the live pivot
    # axis, no pose change): it sweeps through the level window fast. Sample
    # EVERY step: level() instants must be witnessed, success must never fire.
    sd6, k6 = find_seed(41, lambda k: k <= 2)
    step(240)
    t0 = tilt_deg()
    ydir = quat_apply(scene.stand.data.root_quat_w,
                      torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    st = scene.beam.data.root_state_w.clone()
    st[:, 7:10] = 0.0
    st[:, 10:13] = math.copysign(2.4, t0) * ydir
    scene.beam.write_root_state_to_sim(st, all_ids)
    level_hits, level_still = 0, False
    for _ in range(300):
        step(1)
        if bool(scene.level()[0]):
            level_hits += 1
            level_still = level_still or bool(scene.still()[0])
        judge()
    step(300)  # let it fall back
    report("swing-through")
    s, ok = judge()
    check(f"swing-through (seed {sd6}, {k6}u): the kicked beam swept through the "
          f"level window ({level_hits} sampled level() instants) but the "
          "sustained-still gate never opened at level and success never fired; "
          f"the beam fell back to the stop ({tilt_deg():+.1f} deg) with the "
          "stone still home",
          level_hits > 0 and not level_still and not ever_success[0]
          and abs(tilt_deg()) > 9.0 and bool(scene.stone_home()[0]))

    # =========================== 7. one unit short stays pinned =============================
    sd7, k7 = find_seed(51, lambda k: k >= 2)
    step(240)
    t0 = tilt_deg()
    load_units(k7 - 1)
    settle(20)
    report("under-load")
    s, ok = judge()
    check(f"under-load (seed {sd7}): brass totalling {k7 - 1}u vs a {k7}u stone "
          f"leaves the beam pinned on the stone's stop ({tilt_deg():+.1f} deg, "
          "same sign) — not level, NOT success",
          abs(tilt_deg()) > 9.0 and (tilt_deg() < 0) == (t0 < 0)
          and not bool(scene.level()[0]) and not ok)

    # =========================== 8. one unit over slams the other stop ======================
    sd8, k8 = find_seed(61, lambda k: k <= 6)
    step(240)
    t0 = tilt_deg()
    load_units(k8 + 1)
    settle(20)
    report("over-load")
    s, ok = judge()
    check(f"over-load (seed {sd8}): brass totalling {k8 + 1}u vs a {k8}u stone "
          f"slams the beam to the OPPOSITE stop ({tilt_deg():+.1f} deg, sign "
          "flipped) — not level, NOT success",
          abs(tilt_deg()) > 9.0 and (tilt_deg() < 0) != (t0 < 0)
          and not bool(scene.level()[0]) and not ok)

    # =========================== 9. clause D: leftover weight parked close ==================
    sd9, k9 = find_seed(71, lambda k: k <= 6)
    step(240)
    load_units(k9)
    spare = [j for j in range(3) if j not in SUBSET[k9]][0]
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    park_weight(spare, (float(sp[0]), float(sp[1]) - 0.18))
    settled = settle(30)
    d = float((scene.weights[spare].data.root_pos_w[:, :2]
               - scene.stand.data.root_pos_w[:, :2]).norm(dim=-1)[0])
    report("parked-close")
    s, ok = judge()
    check(f"clause D (seed {sd9}): the EXACT {k9}u load is seated — level, "
          f"sustained-still, stone home, stand home — but the leftover "
          f"{c.weight_specs[spare][0]} rests {d:.2f} m (< {c.exclusion_r} m) from "
          "the stand: weights_clean False, NOT success, score capped at 0.25",
          settled and bool(scene.level()[0]) and bool(scene.stone_home()[0])
          and bool(scene.stand_home()[0]) and d < c.exclusion_r
          and not bool(scene.weights_clean()[0]) and not ok and s <= 0.251)

    # =========================== 10. clause C: stone removed ================================
    env.reset(seed=81)
    step(240)
    stone = scene.stones[int(scene.active_idx[0])]
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = scene.env_origins[:, 0] + 1.30
    st[:, 1] = scene.env_origins[:, 1] - 0.90
    st[:, 2] = scene.env_origins[:, 2] + c.stone_size / 2 + 0.003
    st[:, 3] = 1.0
    stone.write_root_state_to_sim(st, all_ids)
    settled = settle(40)
    report("stone-removed")
    s, ok = judge()
    check("clause C: with the stone teleported off to the floor the EMPTY "
          f"balance levels itself ({tilt_deg():+.1f} deg) and stills, weights "
          "staged clean, stand home — but stone_home False: NOT success "
          "(removing the load is not weighing it)",
          settled and bool(scene.level()[0]) and bool(scene.weights_clean()[0])
          and bool(scene.stand_home()[0]) and not bool(scene.stone_home()[0])
          and not ok)

    # =========================== 11. clause E: balance dragged from home ====================
    sd11, k11 = find_seed(91, lambda k: True)
    step(240)
    linkage = [scene.stand, scene.beam, scene.pans[0], scene.pans[1],
               scene.stones[int(scene.active_idx[0])]]
    for body in linkage:  # drag the WHOLE linkage coherently (depenetration trap)
        st = body.data.root_state_w.clone()
        st[:, 0] += 0.10
        st[:, 7:13] = 0.0
        body.write_root_state_to_sim(st, all_ids)
    step(120)
    load_units(k11)
    settled = settle(30)
    report("stand-dragged")
    s, ok = judge()
    check(f"clause E (seed {sd11}): the dragged balance carries the EXACT {k11}u "
          f"load — level ({tilt_deg():+.1f} deg), sustained-still, stone home, "
          "weights clean — but the stand sits 0.10 m (> 0.04 m tol) from its "
          "spawn pose: stand_home False, NOT success",
          settled and bool(scene.level()[0]) and bool(scene.stone_home()[0])
          and bool(scene.weights_clean()[0]) and not bool(scene.stand_home()[0])
          and not ok)

    # =========================== 12. latched credit survives removal ========================
    sd12, k12 = find_seed(101, lambda k: k >= 2)
    step(240)
    drop_weight(0)  # 1u: under-load, beam stays pinned, but the seat latches
    step(180)
    settle(10)
    s1, _ = judge()
    park_weight(0)
    step(120)
    s2, ok = judge()
    report("latch-removed")
    check(f"latched credit (seed {sd12}, {k12}u stone): seating one under-load "
          f"unit latches the seating credit (score {s1:.2f} >= 0.15); removing "
          f"it leaves the latched score unchanged ({s2:.2f}); never success",
          s1 >= 0.15 - 1e-6 and abs(s2 - s1) < 1e-6 and not ok and not ever_success[0])

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN); frames recorded",
          finite_all() and (annot is None or len(frames) > 0))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.beam_balance")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    except Exception:  # noqa: BLE001 - fail fast and loud, never hang on teardown
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
