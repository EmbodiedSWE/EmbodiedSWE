"""Smoke / rubric-REJECTION battery for CounterweightScaleScene (sim_gen task
`push_buttons_i40`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop the binary-decomposition subset of weight
blocks into the blue pan and let the beam settle level — is the acceptance evidence
that the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) settled outcome and asserts the rubric REJECTS
it; no probe in this battery ever reaches success(), and a final audit check asserts
exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: every reference cube seated in
                           the red pan, the empty-blue beam resting on the RED-side
                           joint limit, everything at sustained rest; score ~0.
  3-4. randomization     — READBACK over 8 seeded resets: k (reference count), scale
                           xy, the red-pan side/heading and the staging permutation all
                           move; invariants hold (exactly k refs inside the red pan,
                           all candidates beyond the exclusion radius).
  5.  null policy        — 360 idle steps -> score ~0, no success (beam stays on the
                           red-side limit; the empty blue pan earns nothing).
  6.  mid-swing          — the correct block teleported into the blue pan, judged
                           WHILE the beam is still swinging: level() is seen True at
                           some sampled instant, yet success() is never True and no
                           ever-success latches ("swinging through level does not
                           count" — the sustained-stillness gate is load-bearing).
  7.  under-load by 1    — blocks totalling k-1 units settled in the blue pan -> the
                           beam still rests far on the RED side, NOT success.
  8.  over-load by 1     — blocks totalling k+1 units settled in the blue pan -> the
                           beam tips past level to the BLUE side, NOT success.
  9.  clause C (empty)   — all reference cubes teleported OUT of the red pan onto the
                           bench: the empty scale balances level and still, yet
                           refs_home is False -> NOT success, score ~0.
  10. clause C (moved)   — k=2: one reference cube moved INTO the blue pan (1 vs 1
                           balances level and still) -> refs_home False, NOT success.
  11. clause D (parked)  — the correct subset in the blue pan AND the unused block
                           parked on the bench INSIDE the exclusion radius: level,
                           still, refs undisturbed — yet cand_ok False -> NOT success.
  12. clause D (red pan) — k=1: medium (2u) into the blue pan and small (1u) into the
                           RED pan balances 2 vs 2 level and still -> the candidate in
                           the red pan violates clause D, NOT success.
  13. latched credit     — loading 4/6 then 5/6 units latches strictly increasing
                           progress; removing a block leaves the latch unchanged; the
                           under-loaded beam never reaches success.
  14. rejection audit    — success() was never True at ANY judged point.
  15. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.push_buttons_i40.smoke --headless
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
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterweight_scale")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    tol = math.radians(c.level_tol_deg)
    units = [u for _nm, _s, u in c.cand_specs]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.80, -0.80, 0.72)) + o),
                                tuple(np.array((0.0, 0.02, 0.24)) + o),
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

    def settle(max_rounds: int = 22) -> bool:
        """REAL physics steps until sustained rest (or the budget ends)."""
        for _ in range(max_rounds):
            step(60)
            if bool(scene.still()[0]):
                return True
        return False

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:18s} | tilt={math.degrees(float(scene.tilt()[0])):+6.2f} deg "
              f"blue_units={float(scene.blue_units()[0]):.1f} "
              f"refs_home={bool(scene.refs_home()[0])} cand_ok={bool(scene.cand_ok()[0])} "
              f"level={bool(scene.level()[0])} still={bool(scene.still()[0])} "
              f"success={ok} score={s:.3f} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- probe placement (instrumentation, not a solution) --------------------------------
    def put_in_pan(body, size: float, pan, ox: float = 0.0, oy: float = 0.0) -> None:
        """Teleport a block gently onto the pan floor (pan-local xy offset), judged
        only after REAL settle steps (the zero-step trap)."""
        loc = torch.tensor([ox, oy, -c.pan_drop + size / 2 + 0.004],
                           device=device).expand(n, 3)
        pq, pp = pan.data.root_quat_w, pan.data.root_pos_w
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pp + quat_apply(pq, loc)
        st[:, 3:7] = pq
        body.write_root_state_to_sim(st, all_ids)

    def to_staging(j: int) -> None:
        """Park candidate j on a FREE bench spot far outside the exclusion radius
        (the nominal slots are shuffled, so slot j may be occupied by another block)."""
        size = c.cand_specs[j][1]
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1] = 0.34 + 0.07 * j, -0.26
        st[:, 2] = c.bench_top + size / 2 + 0.003
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.cands[j].write_root_state_to_sim(st, all_ids)

    tile = {"small": (0.030, -0.030), "medium": (0.030, 0.025), "large": (-0.025, -0.025)}

    def load_blue(total: int) -> None:
        """Teleport the (unique) subset of blocks summing to `total` units onto the
        blue pan floor, largest first."""
        sel = [j for j in range(3) if total & units[j]]
        for j in sorted(sel, key=lambda jj: -units[jj]):
            nm, size, _u = c.cand_specs[j]
            ox, oy = tile[nm]
            put_in_pan(scene.cands[j], size, scene.pan_blue, ox, oy)
            step(30)

    def ref_in_red_count() -> int:
        pos, _v = scene._blocks(scene.refs)
        inn = scene._in_pan(scene.pan_red, pos)[0]
        return int((inn & scene.ref_present[0]).sum())

    def base_xy_yaw() -> tuple[float, float, float]:
        p = (scene.base.data.root_pos_w - scene.env_origins)[0]
        q = scene.base.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(p[0]), float(p[1]), yaw

    def red_azimuth() -> float:
        """World heading of the red pan as seen from the base axis (flip readback)."""
        r = (scene.pan_red.data.root_pos_w - scene.base.data.root_pos_w)[0]
        return math.atan2(float(r[1]), float(r[0]))

    def small_slot() -> int:
        sx = float((scene.cands[0].data.root_pos_w - scene.env_origins)[0, 0])
        return min(range(3), key=lambda i: abs(sx - c.slot_xs[i]))

    def finite_all() -> bool:
        ok = True
        for b in [scene.base, scene.beam, scene.pan_red, scene.pan_blue] + scene.refs + scene.cands:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    # ----- seed hunt: the clause probes need specific k values ------------------------------
    need = {1: None, 2: None, 6: None}
    for sd in range(100, 200):
        if all(v is not None for v in need.values()):
            break
        env.reset(seed=sd)
        step(2)
        k = int(scene.n_ref[0])
        if k in need and need[k] is None:
            need[k] = sd
    print(f"[smoke] seed hunt: k->seed {need}", flush=True)
    assert all(v is not None for v in need.values()), f"seed hunt failed: {need}"
    seed_k1, seed_k2, seed_k6 = need[1], need[2], need[6]

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settled = settle()
    report("reset-settled")
    k = int(scene.n_ref[0])
    lim = math.radians(c.limit_deg)
    check("settle: states finite; all k reference cubes seated in the red pan; the "
          "empty-blue beam rests on the RED-side joint limit at sustained rest",
          finite_all() and settled and ref_in_red_count() == k
          and bool(scene.refs_home()[0]) and float(scene.tilt()[0]) < -(lim - math.radians(2.0)))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    inv_ok = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        kk = int(scene.n_ref[0])
        bx, by, byaw = base_xy_yaw()
        reads.append((kk, bx, by, byaw, red_azimuth(), small_slot()))
        inv_ok = inv_ok and ref_in_red_count() == kk
        pos, _v = scene._blocks(scene.cands)
        axis = scene.base.data.root_pos_w[:, None, :2]
        inv_ok = inv_ok and bool(
            ((pos[:, :, :2] - axis).norm(dim=-1) > c.exclusion_r).all())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (k, base_x, base_y, base_yaw, red_az, "
          f"small_slot):\n{arr.round(3)}", flush=True)
    check("randomization: k (reference count) takes >= 3 distinct values and the scale "
          "base xy moves across seeded resets (readback)",
          len(set(arr[:, 0].astype(int))) >= 3
          and arr[:, 1].ptp() > 0.01 and arr[:, 2].ptp() > 0.01)
    check("randomization: the red-pan side/heading and the staging permutation vary; "
          "invariants hold (exactly k refs inside the red pan, all candidates beyond "
          "the exclusion radius)",
          arr[:, 4].ptp() > 0.3 and len(set(arr[:, 5].astype(int))) >= 2 and inv_ok)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(360)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 360 idle steps (empty blue pan, "
          "beam on the red-side limit)", s <= 0.02 and not ok)

    # =========================== 6. mid-swing does not count ================================
    # Teleport the CORRECT load (k=2 -> the medium block) into the blue pan, then judge
    # every few steps WHILE the beam swings up from the red-side limit. level() must be
    # seen True at some instant, yet success() must never fire and no ever-success may
    # latch: "a beam merely swinging through level does not count".
    env.reset(seed=seed_k2)
    settle(8)
    put_in_pan(scene.cands[1], c.cand_specs[1][1], scene.pan_blue,
               *tile["medium"])
    any_level, any_success = False, False
    for _ in range(48):  # 240 steps = 2 s: far shorter than the sustained-rest horizon
        step(5)
        any_level = any_level or bool(scene.level()[0])
        _s, ok = judge()
        any_success = any_success or ok
    report("mid-swing")
    check("mid-swing: with the correct load the swinging beam is seen level() at some "
          "sampled instant, yet success() never fires and no ever-success latches "
          "(sustained-stillness gate)",
          any_level and not any_success and float(scene.level_latch[0]) == 0.0)
    to_staging(1)  # remove the block again before the scale can come to rest
    step(30)

    # =========================== 7. under-load by one unit ==================================
    env.reset(seed=seed_k6)
    settle(8)
    load_blue(5)  # k = 6
    settle()
    report("under-load")
    s, ok = judge()
    check("under-load: 5 of 6 units settled in the blue pan leave the beam far on the "
          "RED side (tilt < -level_tol), NOT success, score capped below 0.90",
          float(scene.tilt()[0]) < -tol and not bool(scene.level()[0]) and not ok
          and s <= 0.90)

    # =========================== 8. over-load by one unit ===================================
    env.reset(seed=seed_k6)
    settle(8)
    load_blue(7)  # k = 6
    settle()
    report("over-load")
    s, ok = judge()
    check("over-load: 7 units against 6 tip the beam past level to the BLUE side "
          "(tilt > +level_tol), NOT success",
          float(scene.tilt()[0]) > tol and not bool(scene.level()[0]) and not ok)

    # =========================== 9. clause C: emptied red pan ===============================
    # Cheat: make the scale level by REMOVING the reference cubes instead of matching
    # them. The empty scale balances perfectly — and must earn nothing.
    env.reset(seed=41)
    settle(8)
    for i, body in enumerate(scene.refs):
        if not bool(scene.ref_present[0, i]):
            continue
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = -0.42 + 0.05 * i
        st[:, 1] = -0.05
        st[:, 2] = c.bench_top + c.ref_size / 2 + 0.003
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
    settle()
    report("refs-removed")
    s, ok = judge()
    check("clause C (emptied): all reference cubes moved out of the red pan — the "
          "empty scale balances level and still, yet refs_home is False -> NOT "
          "success, score <= 0.02",
          bool(scene.level()[0]) and bool(scene.still()[0])
          and not bool(scene.refs_home()[0]) and not ok and s <= 0.02)

    # =========================== 10. clause C: reference moved to the blue pan ==============
    env.reset(seed=seed_k2)
    settle(8)
    put_in_pan(scene.refs[0], c.ref_size, scene.pan_blue, 0.030, 0.030)
    settle()
    report("ref-to-blue")
    s, ok = judge()
    check("clause C (moved): one of two reference cubes teleported into the BLUE pan "
          "— 1 vs 1 balances level and still, yet refs_home is False -> NOT success",
          bool(scene.level()[0]) and bool(scene.still()[0])
          and not bool(scene.refs_home()[0]) and not ok)

    # =========================== 11. clause D: unused block parked too close ================
    # Perfect balance PLUS an unused block parked on the bench inside the exclusion
    # radius (perpendicular to the beam, clear of both pans and of the base foot).
    env.reset(seed=seed_k6)
    settle(8)
    # world-frame offset toward the bench front: 0.20 m from the axis (< exclusion_r),
    # perpendicular-ish to the beam for any yaw in +/-15 deg (+ flip), so it clears
    # both pans, the base foot and the staging row for every scale pose.
    park = scene.base.data.root_pos_w + torch.tensor([0.0, -0.20, 0.0], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = park[:, 0:2]
    st[:, 2] = scene.env_origins[:, 2] + c.bench_top + c.cand_specs[0][1] / 2 + 0.003
    st[:, 3] = 1.0
    scene.cands[0].write_root_state_to_sim(st, all_ids)  # small = the unused block
    step(30)
    load_blue(6)  # the correct subset (medium + large)
    settle(40)  # the fully-loaded balanced beam rings longest — generous budget
    report("parked-inside")
    s, ok = judge()
    d_park = float((scene.cands[0].data.root_pos_w[:, :2]
                    - scene.base.data.root_pos_w[:, :2]).norm(dim=-1)[0])
    check("clause D (parked): correct subset balanced level and still, refs "
          "undisturbed — but the unused block rests inside the exclusion radius "
          f"({d_park:.2f} m < {c.exclusion_r} m) -> cand_ok False, NOT success, "
          "score < 1",
          bool(scene.level()[0]) and bool(scene.still()[0])
          and bool(scene.refs_home()[0]) and d_park < c.exclusion_r
          and not bool(scene.cand_ok()[0]) and not ok and s <= 0.90)

    # =========================== 12. clause D: compensation via the red pan =================
    # Cheat: k=1 — put the medium block (2u) in the blue pan and the small block (1u)
    # in the RED pan: 2 vs 2 balances level. The candidate in the red pan violates
    # clause D even though the reference cube itself was never touched.
    env.reset(seed=seed_k1)
    settle(8)
    put_in_pan(scene.cands[0], c.cand_specs[0][1], scene.pan_red, 0.030, 0.030)
    step(30)
    put_in_pan(scene.cands[1], c.cand_specs[1][1], scene.pan_blue, *tile["medium"])
    settle()
    report("red-compensated")
    s, ok = judge()
    check("clause D (red pan): k=1 — medium (2u) in blue and small (1u) dropped in "
          "with the reference: 2 vs 2 balances level and still, refs_home True, yet "
          "the candidate in the red pan fails cand_ok -> NOT success",
          bool(scene.level()[0]) and bool(scene.still()[0])
          and bool(scene.refs_home()[0]) and not bool(scene.cand_ok()[0]) and not ok)

    # =========================== 13. latched credit + monotonicity ==========================
    env.reset(seed=seed_k6)
    settle(8)
    put_in_pan(scene.cands[2], c.cand_specs[2][1], scene.pan_blue, *tile["large"])
    step(120)  # 4 of 6 units
    l1 = float(scene.prog_latch[0])
    put_in_pan(scene.cands[0], c.cand_specs[0][1], scene.pan_blue, *tile["small"])
    step(120)  # 5 of 6 units
    l2 = float(scene.prog_latch[0])
    to_staging(0)  # take the small block back out
    step(120)
    l3 = float(scene.prog_latch[0])
    report("latch-regressed")
    _s, ok = judge()
    check("latched credit: loading 4/6 then 5/6 units latches strictly more progress "
          f"({l1:.3f} < {l2:.3f}); removing the block leaves the latch unchanged "
          f"({l3:.3f}); the under-loaded beam never reaches success",
          l1 + 0.10 < l2 and abs(l3 - l2) < 1e-6 and not ok and not ever_success[0])

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.counterweight_scale")
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
