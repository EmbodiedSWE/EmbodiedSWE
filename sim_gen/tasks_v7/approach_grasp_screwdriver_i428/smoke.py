"""Smoke / rubric-REJECTION battery for CorbelReachScene (sim_gen task
`approach_grasp_screwdriver_i428`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — build the harmonic corbel by hover-drop, then
slide the top course past the line by contact — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: all planks flat on the deck,
                            pose-still; score ~0, no success;
  3-4. randomization      — READBACK over 6 seeded resets: the beacon line R varies;
                            every plank's xy jitter AND yaw vary; planks always spawn
                            fully BEHIND the cliff edge (gated reach starts at 0);
  5.  null policy         — 240 idle steps: the stillness latch fires, but at reach 0
                            -> score ~0, no success;
  6.  seed strategy       — the seed's whole plan ("grasp the graspable object, carry
                            it to the target") = ONE plank carried to the beacon line
                            and released at deck height: its centre of mass is past
                            the cliff, so it TIPS and falls -> no credit;
  7.  transient at line   — a plank written hovering AT deck height past the line
                            reads gated reach >= R for its first substeps, yet the
                            pose-stillness streak refuses it: no success, no latch;
  8.  beacon intangible   — the same plank falls THROUGH the red beacon to the floor
                            (the beacon has no collider): reach back to 0, score ~0;
  9.  greedy stack        — a teleported 4-course stack with UNIFORM 60 mm offsets
                            (deck interface at 2x its critical overhang) transiently
                            shows reach >= R while numerically still, then collapses:
                            the streak gate must keep the latch at ~0 (the flagship
                            anti-latch-poisoning check);
  10. short corbel        — a REAL stable corbel (drop+settle, the solve's own
                            construction) scaled 15 mm SHORT of the line: settled,
                            honest partial credit in [0.50, 0.75], NOT success;
  11. latched credit      — removing the top course afterwards collapses the reach,
                            but the latched score survives unchanged; success stays
                            gone;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.approach_grasp_screwdriver_i428.smoke --headless
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

HARMONIC_BASE = (0.025, 1.0 / 30.0, 0.05, 0.10)  # same plan as solve.py


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.corbel_reach")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -1.10, 0.95)) + o),
                                tuple(np.array((-0.15, 0.00, 0.42)) + o),
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

    def plank_p(k: int) -> torch.Tensor:
        return (scene.planks[k].data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cx, cz = scene._plank_extremes()
        s, ok = judge()
        print(f"[smoke] {tag:18s} | reach={float(scene.reach_now()[0]):+.4f} "
              f"R={float(scene.reach_R[0]):.4f} corner_x="
              f"{[f'{float(v):+.3f}' for v in cx[0]]} min_z="
              f"{[f'{float(v):.3f}' for v in cz[0]]} "
              f"latch={float(scene.reach_latch[0]):.3f} streak={int(scene._still[0])} "
              f"settled={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_course(k: int, edge_x: float, settle_steps: int = 60) -> None:
        """The solve's own construction move: hover the plank 3 mm over its course
        and let it fall + settle through real contact."""
        place(scene.planks[k], edge_x - c.plank_l / 2, 0.0,
              c.ped_h + c.plank_t / 2 + k * c.plank_t + 0.003,
              settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(100)
    report("reset-settled")
    fin0 = all(bool(torch.isfinite(pl.data.root_state_w).all()) for pl in scene.planks)
    _cx0, cz0 = scene._plank_extremes()
    flat = bool((cz0[0] > c.ped_h - 0.004).all()) and bool((cz0[0] < c.ped_h + 0.006).all())
    check("settle: states finite; all four planks flat on the deck (readback) and "
          "pose-still", fin0 and flat and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads, yaws = [], []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        cx, _cz = scene._plank_extremes()
        p0, p2 = plank_p(0), plank_p(2)
        q0 = scene.planks[0].data.root_quat_w[0]
        reads.append((float(scene.reach_R[0]), float(p0[0]), float(p0[1]),
                      float(p2[0]), float(p2[1]), float(cx[0].max())))
        yaws.append(abs(float(q0[3])))  # |sin(yaw/2)|: quat double-cover safe
    arr = np.array(reads)
    print(f"[smoke] randomization readback (R, p0_x, p0_y, p2_x, p2_y, max_corner_x):"
          f"\n{arr}", flush=True)
    print(f"[smoke] plank0 |q_z| across seeds: {[f'{y:.3f}' for y in yaws]}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: the beacon line R varies across seeded resets (readback)",
          spread[0] > 0.005)
    check("randomization: plank xy jitter and yaw vary; planks always spawn fully "
          "BEHIND the cliff edge (max corner x < 0)",
          spread[1] > 0.01 and spread[2] > 0.005 and spread[3] > 0.01
          and spread[4] > 0.005 and (max(yaws) - min(yaws)) > 0.10
          and bool((arr[:, 5] < 0.0).all()))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: the stillness latch fires but at reach 0 — score ~0 and no "
          "success after 240 idle steps",
          bool(scene.settled()[0]) and s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ============================================
    # The seed's whole plan is "grasp the graspable object and carry it to the target".
    # Here: ONE plank carried to the beacon line and released at deck height. Its
    # centre of mass is past the cliff edge (R - L/2 + 20 mm > 0), so physics tips it
    # into the void — the strategy earns nothing.
    env.reset(seed=41)
    step(10)
    r_line = float(scene.reach_R[0])
    place(scene.planks[0], r_line - c.plank_l / 2 + 0.020, 0.0,
          c.ped_h + c.plank_t / 2 + 0.003, settle_steps=150)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (single graspable plank carried to the line and released): "
          "it tips off the cliff and falls — NOT success, score <= 0.02, plank low",
          not ok and s <= 0.02 and float(plank_p(0)[2]) < c.ped_h - 0.05)

    # =========================== 7-8. transient at the line + intangible beacon =============
    # A plank written hovering AT deck height past the line: for its first substeps the
    # gated reach reads >= R (min corner still at deck height), but the pose-stillness
    # streak refuses to latch or succeed. It then free-falls THROUGH the red beacon
    # (no collider) to the floor.
    env.reset(seed=51)
    step(10)
    r_line = float(scene.reach_R[0])
    place(scene.planks[1], r_line, 0.0, c.ped_h + c.plank_t / 2, settle_steps=2)
    report("transient-at-line")
    s, ok = judge()
    transient_reach = float(scene.reach_now()[0])
    check("transient at the line: gated reach >= R for the first substeps after the "
          "write, yet NOT success and score <= 0.02 (stillness streak gates both)",
          transient_reach >= r_line - 0.005 and not ok and s <= 0.02
          and float(scene.reach_latch[0]) <= 0.02)
    step(150)
    report("through-beacon")
    s, ok = judge()
    check("beacon intangible: the plank fell THROUGH the red beacon to the floor "
          "(final z < 0.10 at x near R), reach back to 0, score still <= 0.02",
          float(plank_p(1)[2]) < 0.10 and abs(float(plank_p(1)[0]) - r_line) < 0.25
          and float(scene.reach_now()[0]) <= 0.001 and s <= 0.02 and not ok)

    # =========================== 9. greedy infeasible stack ==================================
    # Four courses with UNIFORM 60 mm offsets: the top edge reads past the line, and
    # for the first substeps the freshly written stack is numerically still — but the
    # deck interface is at 2x its critical overhang (60 mm vs 30 mm) and the stack
    # centre of mass is past the cliff, so it collapses. The streak gate must keep
    # both the latch and success at zero (anti-latch-poisoning).
    env.reset(seed=61)
    step(10)
    r_line = float(scene.reach_R[0])
    st = torch.zeros(n, 13, device=device)
    for k in range(4):
        st.zero_()
        st[:, 0] = 0.060 * (k + 1) - c.plank_l / 2
        st[:, 2] = c.ped_h + c.plank_t / 2 + k * c.plank_t + 0.003
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.planks[k].write_root_state_to_sim(st, all_ids)
    step(2)
    report("greedy-written")
    greedy_transient = float(scene.reach_now()[0]) >= r_line - 0.005
    step(200)
    report("greedy-collapsed")
    s, ok = judge()
    _cx, cz = scene._plank_extremes()
    check("greedy uniform-offset stack: transiently past the line when written, then "
          "COLLAPSES (some plank far below deck) — latch stays ~0, never success",
          greedy_transient and float(cz[0].min()) < c.ped_h - 0.10
          and float(scene.reach_latch[0]) <= 0.05 and s <= 0.05 and not ok)

    # =========================== 10. stable corbel SHORT of the line ========================
    # The solve's own construction, scaled 15 mm short of the line: a real, settled
    # corbel that earns honest partial credit but NOT success.
    env.reset(seed=71)
    step(10)
    r_line = float(scene.reach_R[0])
    s_scale = (r_line - 0.015) / sum(HARMONIC_BASE)
    acc = 0.0
    for k, b in enumerate(HARMONIC_BASE):
        acc += b * s_scale
        drop_course(k, acc, settle_steps=60)
    step(100)  # mature the stillness streak -> the latch fires at the short reach
    report("short-corbel")
    s_short, ok = judge()
    check("stable corbel 15 mm short of the line: settled, honest partial credit "
          "(0.50 <= score <= 0.75), NOT success",
          bool(scene.settled()[0]) and not ok and 0.50 <= s_short <= 0.75)

    # =========================== 11. latched credit survives regression =====================
    place(scene.planks[3], -0.65, 0.0, c.ped_h + c.plank_t / 2 + 0.003,
          settle_steps=100)
    report("top-removed")
    s_out, ok = judge()
    check("latched credit: removing the top course collapses the reach, but the "
          f"latched score survives ({s_short:.3f} -> {s_out:.3f}); success stays gone",
          float(scene.reach_now()[0]) < s_short * float(scene.reach_R[0])
          and abs(s_out - s_short) < 0.02 and not ok)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(pl.data.root_state_w).all()) for pl in scene.planks)
    fin = fin and bool(torch.isfinite(scene.beacon.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.corbel_reach")
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
