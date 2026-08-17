"""Smoke / rubric-REJECTION battery for GradeBeaconScene (sim_gen task
`stack_blocks_i415`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — read the collar grade, solve the exact block
subset-sum, force-build the pillar, force-crown the beacon — is the acceptance
evidence that the rubric ACCEPTS a correct outcome; it passes on seeds 0/1/2
across three different grades). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it. No
probe in this battery ever reaches success(), and a final audit asserts exactly
that.

  1-2. settle/no-NaN    — reset layout settles finite: 4 blocks + beacon scattered
                          on the ring OUTSIDE the pad column, collar readback at
                          pad_t + grade + beacon_s/2 with grade ON the 5-value
                          ladder, pad `mast_gap` from the mast; score ~0;
  3-4. randomization    — READBACK over 8 seeded resets: the collar GRADE varies
                          (>= 3 distinct rungs), the pad's bearing from the mast
                          (rig yaw) and the scatter permutation vary; every reset
                          sane;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  seed-strategy     — the seed family's plan: stack ALL FOUR blocks on the pad
                          (fixed "stack the blocks" loop, 210 mm tower) + beacon on
                          top -> OVERSHOOTS every grade, no success, score <=
                          build_credit (overshot pillar earns nothing more);
  7.  single block      — beacon on the TALLEST single block (75 mm): below every
                          grade band by >= 15 mm -> rejected (a pillar must be
                          COMPOSED);
  8.  neighbor grade    — the exact pillar for the NEIGHBORING rung (+/-15 mm) +
                          beacon -> |dz| = 15 mm > band -> rejected (the grade must
                          be READ, not guessed);
  9.  off-column        — correct-HEIGHT pillar + beacon built on the floor AWAY
                          from the pad: beacon lands IN band (floor build differs
                          only by the 6 mm pad) but out of column -> rejected;
  10. held at grade     — beacon teleport-HELD in the air at the exact collar
                          height in-column (velocity re-zeroed every substep, 40
                          substeps): in band, in column, but supported() False ->
                          success never fires; released, it falls;
  11. sustain gate      — genuinely correct crown judged 6 substeps after contact:
                          geometry all True but hold < settle_steps -> NOT success
                          (then the beacon is removed before the window fills);
  12. latched credit    — after removing that beacon, score == build_credit
                          exactly: build progress stays latched, place credit was
                          never earned;
  13. no invisible shelf — beacon teleported BESIDE the mast at collar height (the
                          collar is visual-only, NO collider): nothing supports it
                          -> it falls to the floor, no success;
  14. rejection audit   — success() was never True at ANY judged point;
  15. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_blocks_i415.smoke --headless
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

BLOCK_HS = scene_mod.BLOCK_HS
subset_for = scene_mod.subset_for

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.grade_beacon")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.10, 0.90)) + o),
                                tuple(np.array((0.25, 0.00, 0.12)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:16s} | grade={float(scene.grade[0]) * 1000:.0f}mm "
              f"pillar={float(scene.pillar_top()[0]) * 1000:.0f}mm "
              f"dz={float(scene.beacon_dz()[0]) * 1000:+.0f}mm "
              f"in_col={bool(scene.beacon_in_col()[0])} "
              f"sup={bool(scene.supported()[0])} hold={float(scene.hold_count[0]):.0f} "
              f"| score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in xyz], device=device)
        st[:, 3] = 1.0
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def build(idxs, xy, base_z: float, drop: float = 0.008,
              settle_steps: int = 40) -> float:
        """Drop-stack the given blocks at `xy` (relative coords), bottom at
        `base_z`; returns the planned top height."""
        z = base_z
        for i in idxs:
            h = BLOCK_HS[i]
            place(scene.blocks[i], (xy[0], xy[1], z + h / 2 + drop),
                  settle_steps=settle_steps)
            z += h
        return z

    def pad_rel_xy() -> tuple[float, float]:
        p = (scene.pad_xy[0] - scene.env_origins[0, 0:2])
        return float(p[0]), float(p[1])

    def neighbor_grade(g: float) -> float:
        return g + c.grade_step if abs(g - c.grade_min) < 1e-6 else g - c.grade_step

    def layout_sane(tag: str) -> bool:
        """Reset honesty: movers on the scatter ring OUTSIDE the column; collar at
        pad_t + grade + beacon_s/2 with grade on the ladder; pad `mast_gap` out."""
        mast_xy = rel(scene.mast)[0:2]
        grade = float(scene.grade[0])
        rung = (grade - c.grade_min) / c.grade_step
        collar_z = float(rel(scene.collar)[2])
        pad_d = float((scene.pad_xy[0] - scene.mast.data.root_pos_w[0, 0:2]).norm())
        ok = (abs(collar_z - (c.pad_t + grade + c.beacon_s / 2)) < 0.002
              and abs(rung - round(rung)) < 1e-4 and 0 <= round(rung) < c.n_grades
              and abs(pad_d - c.mast_gap) < 0.005
              and not bool(scene.block_in_col()[0].any())
              and not bool(scene.beacon_in_col()[0]))
        for b in [*scene.blocks, scene.beacon]:
            r = float((rel(b)[0:2] - mast_xy).norm())
            ok = ok and abs(r - c.scatter_r) < c.scatter_jitter + 0.03
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    bodies = [*scene.blocks, scene.beacon]

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; blocks + beacon scattered on the ring outside "
          "the pad column; collar readback on the grade ladder; pad mast_gap out",
          fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        mast_xy = rel(scene.mast)[0:2]
        pxy = torch.tensor(pad_rel_xy(), device=device) - mast_xy
        b0 = rel(scene.blocks[0])[0:2] - mast_xy
        reads.append([float(scene.grade[0]) * 1000,
                      math.atan2(float(pxy[1]), float(pxy[0])),
                      math.atan2(float(b0[1]), float(b0[0]))])
    arr = np.array(reads)
    print("[smoke] randomization readback (grade_mm, pad_bearing, block30_bearing):\n"
          f"{np.round(arr, 3)}", flush=True)
    grades_seen = {round(v) for v in arr[:, 0]}
    bear = np.ptp(np.sort(arr[:, 1]))
    slot = np.ptp(np.sort(arr[:, 2]))
    check("randomization: the collar GRADE varies across seeds "
          f"({sorted(grades_seen)} mm — >= 3 of the 5 rungs seen)", len(grades_seen) >= 3)
    check("randomization: the pad's bearing from the mast (rig yaw) and the scatter "
          f"permutation vary (spreads {bear:.2f} / {slot:.2f} rad), every reset sane",
          bear > 1.0 and slot > 1.0 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed-strategy: stack ALL the blocks ====================
    # rlbench/stack_blocks succeeds by piling a FIXED set of blocks on the marked
    # plane — the stack itself is the goal. Express that plan here: stack all four
    # blocks on the pad (210 mm tower) and crown it with the beacon. The pillar
    # overshoots every grade; nothing above build progress is earned.
    env.reset(seed=41)
    step(30)
    px, py = pad_rel_xy()
    top = build([3, 2, 1, 0], (px, py), c.pad_t)  # 75+60+45+30 = 210 mm
    place(scene.beacon, (px, py, top + c.beacon_s / 2 + 0.008), settle_steps=90)
    report("all-blocks")
    s, ok = judge()
    dz = float(scene.beacon_dz()[0])
    check("seed-strategy: ALL FOUR blocks stacked on the pad + beacon on top "
          f"(dz={dz * 1000:+.0f}mm above the collar) -> overshoot rejected, no "
          f"success, score {s:.3f} <= build credit (overshot pillar earns no more)",
          dz > c.band and not ok and s <= c.build_credit + 0.02)

    # =========================== 7. single block is never enough ============================
    env.reset(seed=51)
    step(30)
    px, py = pad_rel_xy()
    build([3], (px, py), c.pad_t)  # tallest single block: 75 mm
    place(scene.beacon, (px, py, c.pad_t + BLOCK_HS[3] + c.beacon_s / 2 + 0.008),
          settle_steps=90)
    report("single-75")
    s, ok = judge()
    dz = float(scene.beacon_dz()[0])
    check("single block: beacon on the TALLEST block (75 mm) reads "
          f"dz={dz * 1000:+.0f}mm — below every grade band by >= 15 mm -> rejected "
          "(a pillar must be COMPOSED)", dz < -(c.band + 0.003) and not ok)

    # =========================== 8. neighbor-grade near-miss ================================
    env.reset(seed=61)
    step(30)
    grade = float(scene.grade[0])
    wrong = neighbor_grade(grade)
    idxs = sorted(subset_for(wrong), key=lambda i: -BLOCK_HS[i])
    px, py = pad_rel_xy()
    top = build(idxs, (px, py), c.pad_t)
    place(scene.beacon, (px, py, top + c.beacon_s / 2 + 0.008), settle_steps=90)
    report("neighbor")
    s, ok = judge()
    dz = float(scene.beacon_dz()[0])
    check(f"neighbor grade: the exact pillar for the ADJACENT rung "
          f"({round(wrong * 1000)} vs {round(grade * 1000)} mm, "
          f"dz={dz * 1000:+.0f}mm) -> outside the +/-{c.band * 1000:.0f} mm band, "
          "rejected: the grade must be READ, not guessed",
          abs(dz) > c.band and not ok)

    # =========================== 9. right height, OFF the pad column ========================
    env.reset(seed=71)
    step(30)
    grade = float(scene.grade[0])
    idxs = sorted(subset_for(grade), key=lambda i: -BLOCK_HS[i])
    mast_xy = rel(scene.mast)[0:2]
    # bare floor, well OUTSIDE the 0.34 m scatter ring (no slot collisions)
    fx, fy = float(mast_xy[0]) - 0.45, float(mast_xy[1]) - 0.25
    top = build(idxs, (fx, fy), 0.0)
    place(scene.beacon, (fx, fy, top + c.beacon_s / 2 + 0.008), settle_steps=90)
    report("off-column")
    s, ok = judge()
    dz = float(scene.beacon_dz()[0])
    dcol = float((scene.beacon.data.root_pos_w[0, 0:2] - scene.pad_xy[0]).norm())
    check("off-column: the correct-HEIGHT pillar + beacon built on bare floor away "
          f"from the pad (dz={dz * 1000:+.0f}mm — in/near band; column distance "
          f"{dcol * 1000:.0f}mm) -> the pad-column gate rejects it",
          dcol > c.col_r and not ok and not bool(scene.beacon_in_col()[0]))

    # =========================== 10. held in the air at grade ===============================
    # The perch cheat: hold the beacon at the exact collar height over the pad with
    # NO built support (velocity re-zeroed every substep — a "grip"). In band, in
    # column, still — but supported() is False and success never fires.
    env.reset(seed=81)
    step(30)
    px, py = pad_rel_xy()
    grade = float(scene.grade[0])
    hold_ok = True
    sup_seen = False
    for _ in range(40):
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = c.pad_t + grade + c.beacon_s / 2
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.beacon.write_root_state_to_sim(st, all_ids)
        step(1)
        _s, okh = judge()
        hold_ok = hold_ok and not okh
        sup_seen = sup_seen or bool(scene.supported()[0])
    in_band_held = bool(scene.in_band()[0]) and bool(scene.beacon_in_col()[0])
    step(60)  # release: it falls
    report("held-at-grade")
    fell = float(rel(scene.beacon)[2]) < grade / 2
    check("held at grade: beacon HELD in the air at the collar height in-column for "
          "40 substeps (in band, in column, still) -> supported() stays False, "
          "success never fires; released, it falls to the floor",
          in_band_held and not sup_seen and hold_ok and fell)

    # =========================== 11-12. sustain gate + latched credit =======================
    env.reset(seed=91)
    step(30)
    grade = float(scene.grade[0])
    idxs = sorted(subset_for(grade), key=lambda i: -BLOCK_HS[i])
    px, py = pad_rel_xy()
    top = build(idxs, (px, py), c.pad_t)
    step(60)  # column settles; build latch fills
    place(scene.beacon, (px, py, top + c.beacon_s / 2 + 0.001), settle_steps=6)
    report("early-judge")
    geom = (bool(scene.in_band()[0]) and bool(scene.beacon_in_col()[0])
            and bool(scene.supported()[0]))
    _s, ok_early = judge()
    hold_now = float(scene.hold_count[0])
    check("sustain gate: a genuinely correct crown judged 6 substeps after contact "
          f"(band/column/support all True, hold={hold_now:.0f} < "
          f"{c.settle_steps}) is NOT yet success — stillness must be SUSTAINED",
          geom and not ok_early and hold_now < c.settle_steps)
    # remove the beacon BEFORE the window can fill: success is never reached
    place(scene.beacon, (px - 0.45, py - 0.35, c.beacon_s / 2 + 0.003),
          settle_steps=60)
    report("removed")
    s_after, ok = judge()
    check("latched credit: with the beacon removed, score == build credit exactly "
          f"({s_after:.3f} ~= {c.build_credit:.2f}) — pillar progress stays "
          "latched, place credit was never earned",
          abs(s_after - c.build_credit) < 0.02 and not ok)

    # =========================== 13. no invisible shelf on the mast/collar ==================
    env.reset(seed=101)
    step(30)
    mast_xy = rel(scene.mast)[0:2]
    grade = float(scene.grade[0])
    collar_z = c.pad_t + grade + c.beacon_s / 2
    place(scene.beacon,
          (float(mast_xy[0]) + c.mast_r + c.beacon_s / 2 + 0.002, float(mast_xy[1]),
           collar_z), settle_steps=80)
    report("beside-collar")
    z_now = float(rel(scene.beacon)[2])
    s, ok = judge()
    check("no invisible shelf: beacon released BESIDE the mast at collar height "
          f"(the collar is visual-only) falls to the floor (z={z_now * 1000:.0f}mm), "
          "no success", z_now < collar_z - 0.05 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.grade_beacon")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
