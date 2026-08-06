"""Smoke battery for OvenDialsScene — REJECTION tests for the rubric, NullRobot, RECORDED.

This is NOT a solution (the real solution is solve.py — a Franka pinch-turns the bars).
Teleported dial states here are rubric INSTRUMENTATION: construct an outcome as a settled
state under the scene's live detent plant, then assert the rubric's verdict on it.

One linear run, 14 named checks:
  1. settle      — clean reset: finite state, pointers resting on their start settings,
                   score ~0 (predicates read a clean slate);
  2. random      — start/target draws differ across seeds (randomization by READBACK);
  3. indicator   — the amber lamps physically sit at the sampled target tick marks;
  4. null        — 2 s of nothing: score < 0.05, no success;
  5. negative A  — the SEED's plan (grasp + pull the panel open through a hinge arc):
                   sustained 25 N upward pull + 2.5 N*m prying torque on a knob moves
                   nothing (nothing on this oven opens; the spindle resists every
                   off-axis wrench) — no lift, no turn, no score;
  6. negative B  — near miss: pointer left on the ADJACENT setting (40 deg off) with the
                   other dial correct — rejected by the 8 deg stop tolerance;
  7. negative C  — dials set to EACH OTHER's targets — goal assignment matters;
  8. monotone    — teleport knob 0 detent-by-detent toward its target: the latched score
                   never decreases and climbs;
  9. partial     — knob 0 alone on target: score ~0.5, no success;
 10. exactness   — knob 1 set too: success() and score == 1.0 exactly (the rubric's top
                   is reserved for the full goal state);
 11. latch       — knock knob 0 back off: success revoked, score 0.85 (latched progress
                   does not evaporate; the at-target bonus does);
 12/13. calib    — capture probe: offsets <= 15 deg snap back into the target detent,
                   offsets >= 25 deg fall to the neighbour (20 deg = basin edge, reported);
 14. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.open_oven_i7.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the L20 driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
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

from . import scene as scene_mod  # noqa: E402  (registers "oven_dials" + "simgen.oven_dials")

assert scene_mod.OvenDialsScene is not None  # keep the import explicit


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.oven_dials")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.15, -0.45, 0.75)) + o),
                                tuple(np.array((0.46, 0.0, 0.18)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        r = scene.readings_deg()[0].tolist()
        t = scene.target_deg()[0].tolist()
        print(f"[smoke] {tag:12s} | read=({r[0]:7.1f},{r[1]:7.1f}) "
              f"tgt=({t[0]:6.1f},{t[1]:6.1f}) at={scene.at_target()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 360, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def teleport_dial(k: int, deg: float) -> None:
        """Kinematic write: knob k + its grip bar as ONE rigid assembly at angle `deg`
        about the unchanged spindle, zero velocity (a pure joint-coordinate re-pose of the
        followers; the physical verdict comes from settling under the live detent plant)."""
        half = math.radians(deg) / 2
        for body, center in ((scene.knobs[k], c.knob_center(k)),
                             (scene.bars[k], c.bar_center(k))):
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = scene.env_origins + torch.tensor(center, device=device)
            st[:, 3] = math.cos(half)
            st[:, 6] = math.sin(half)
            body.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    S = c.settings_deg  # tuple of detent angles

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    report("reset")
    step(60)
    report("settled")
    read = scene.readings_deg()[0]
    start_ang = scene._settings[scene.start_idx[0]]
    finite = bool(torch.isfinite(scene.readings_deg()).all()) and \
        bool(torch.isfinite(scene.score()).all()) and \
        all(bool(torch.isfinite(b.data.root_state_w).all())
            for b in (*scene.knobs, *scene.bars, *scene.lamps))
    on_start = bool(((read - start_ang).abs() < 3.0).all())
    check("settle: clean reset (finite, pointers on start settings, score ~0)",
          finite and on_start and float(scene.score()[0]) < 0.05)

    # ========================= 2+3. randomization + goal indicator ============================
    draws = []
    lamp_err_max = 0.0
    for seed in (11, 12, 13):
        torch.manual_seed(seed)
        env.reset()
        step(10)
        draws.append((scene.start_idx[0].tolist(), scene.target_idx[0].tolist()))
        for k in range(2):
            want = torch.tensor(c.lamp_pos(k, int(scene.target_idx[0, k])), device=device)
            got = scene.lamps[k].data.root_pos_w[0] - scene.env_origins[0]
            lamp_err_max = max(lamp_err_max, float((got - want).norm()))
    print(f"[smoke] draws across seeds: {draws} | lamp readback err={lamp_err_max * 1000:.1f}mm",
          flush=True)
    check("randomization is real (start/target draws differ across seeds)",
          len({str(d) for d in draws}) >= 2)
    check("goal indicator: lamps sit at the sampled target marks (readback)",
          lamp_err_max < 0.005)

    # ========================= 4. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative A: the seed's plan =================================
    # open_oven's strategy — grasp and PULL the panel open through a hinge arc. Here: a
    # sustained 25 N upward pull + 2.5 N*m prying (world-x) torque on knob 0. Nothing on
    # this oven opens and the spindle resists every off-axis wrench: no lift, no turn.
    torch.manual_seed(7)
    env.reset()
    step(30)
    pos_before = scene.knobs[0].data.root_pos_w[0].clone()
    read_before = float(scene.readings_deg()[0, 0])
    scene.knob_force[:, 0] = torch.tensor([0.0, 0.0, 25.0], device=device)
    scene.knob_torque_ext[:, 0] = torch.tensor([2.5, 0.0, 0.0], device=device)
    step(240)
    scene.knob_force[:] = 0.0
    scene.knob_torque_ext[:] = 0.0
    step(30)
    drift = float((scene.knobs[0].data.root_pos_w[0] - pos_before).norm())
    turn = abs(float(scene.readings_deg()[0, 0]) - read_before)
    report("pull-test")
    print(f"[smoke]   pull aftermath: drift={drift * 1000:.1f}mm turn={turn:.1f}deg", flush=True)
    check("negative (seed strategy): pulling/prying a knob opens nothing, score ~0",
          drift < 0.010 and turn < 10.0 and float(scene.score()[0]) < 0.06
          and not bool(scene.success()[0]))

    # ========================= 6. negative B: near miss =======================================
    torch.manual_seed(8)
    env.reset()
    step(30)
    t0b, t1b = int(scene.target_idx[0, 0]), int(scene.target_idx[0, 1])
    adj = t0b + 1 if t0b + 1 < c.n_settings else t0b - 1
    teleport_dial(0, S[adj])  # one detent off the target
    teleport_dial(1, S[t1b])  # the other dial fully correct
    settle_until(lambda: bool(scene.at_target()[0, 1]))
    step(45)
    report("near-miss")
    check("negative (near miss): adjacent setting rejected by the stop tolerance",
          not bool(scene.at_target()[0, 0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.90)

    # ========================= 7. negative C: swapped targets ==================================
    for seed in range(20, 40):
        torch.manual_seed(seed)
        env.reset()
        if int(scene.target_idx[0, 0]) != int(scene.target_idx[0, 1]):
            break
    step(30)
    teleport_dial(0, S[int(scene.target_idx[0, 1])])  # each dial on the OTHER's target
    teleport_dial(1, S[int(scene.target_idx[0, 0])])
    step(90)
    report("swapped")
    at = scene.at_target()[0]
    check("negative (swap): dials on each other's targets rejected",
          not bool(at[0]) and not bool(at[1]) and not bool(scene.success()[0]))

    # ========================= 8+9. monotone climb, knob 0 ====================================
    torch.manual_seed(3)
    env.reset()
    step(30)
    s0, t0 = int(scene.start_idx[0, 0]), int(scene.target_idx[0, 0])
    sgn = 1 if t0 > s0 else -1
    scores = [float(scene.score()[0])]
    for idx in range(s0 + sgn, t0 + sgn, sgn):
        teleport_dial(0, S[idx])
        step(45)
        scores.append(float(scene.score()[0]))
    print(f"[smoke] climb scores (knob 0, {s0}->{t0}): "
          + " ".join(f"{s:.3f}" for s in scores), flush=True)
    mono = all(b >= a - 1e-4 for a, b in zip(scores, scores[1:]))
    check("rubric monotonicity: latched score never decreases along the approach",
          mono and scores[-1] > scores[0] + 0.2)
    settle_until(lambda: bool(scene.at_target()[0, 0]))
    report("knob0-set")
    sc = float(scene.score()[0])
    check("partial credit: knob 0 on target -> score ~0.5, no success",
          0.45 <= sc <= 0.60 and not bool(scene.success()[0]))

    # ========================= 10. exactness: success == score 1.0 ============================
    teleport_dial(1, S[int(scene.target_idx[0, 1])])
    step(30)  # judge only AFTER real physics has run on the placed state (latches update)
    ok = settle_until(lambda: bool(scene.success()[0]))
    report("both-set")
    check("exactness: both dials set -> success() and score == 1.0",
          ok and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 11. achievement latch ==========================================
    teleport_dial(0, S[s0])  # knock knob 0 back to where it started (>= 2 detents off)
    step(45)
    report("knock-off")
    check("achievement latch: knock-off revokes success, score 0.85 (latched progress)",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.85) < 0.02)

    # ========================= 12+13. calibration: detent capture probe =======================
    # Teleport knob 0 to target + offset (toward dial centre, so extremes stay in range) and
    # let the detent plant settle: the capture basin is half the 40 deg spacing, so <= 15 deg
    # must snap back onto the target, >= 25 deg must fall to the neighbour. 20 deg is the
    # basin edge — measured and reported, not asserted.
    torch.manual_seed(5)
    env.reset()
    step(30)
    tgt = float(scene.target_deg()[0, 0])
    toward0 = -1.0 if tgt > 0 else 1.0
    results: dict[float, tuple[bool, float]] = {}
    for off in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 40.0):
        teleport_dial(0, tgt + toward0 * off)
        step(130)
        results[off] = (bool(scene.at_target()[0, 0]), float(scene.errors_deg()[0, 0]))
        print(f"[smoke]   calib off={off:4.0f}deg -> at_target={results[off][0]} "
              f"final_err={results[off][1]:.1f}deg", flush=True)
    print(f"[smoke] CALIBRATION: capture basin edge (nominal) = {c.spacing_deg / 2:.0f}deg; "
          f"20deg measured -> {results[20.0]}", flush=True)
    check("calibration: offsets <= 15 deg are captured back to the target detent",
          all(results[o][0] for o in (0.0, 5.0, 10.0, 15.0)))
    check("calibration: offsets >= 25 deg fall to the neighbour detent (rejected)",
          not results[25.0][0] and not results[40.0][0])

    # ========================= 14. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.oven_dials")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
