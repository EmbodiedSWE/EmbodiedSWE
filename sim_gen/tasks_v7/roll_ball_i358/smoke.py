"""Smoke / rubric-REJECTION battery for WindmillTollgateScene (sim_gen task
`roll_ball_i358`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — brake beam threaded through the wall bore by a
held wrench until the blade slams into it and the motor stalls, then the ball bowled
hands-off up the apron, through the doorway and down the tilted floor onto the
corner pad — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
as a settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1.  settle/no-NaN    — reset layout settles finite: ball and beam at rest on the
                         open ground OUTSIDE, the motor-driven mill demonstrably
                         TURNING, score ~0, no success;
  2.  seed strategy    — the seed's entire skill (ball delivered to the goal region)
                         constructed: ball parked ON the green pad while the mill
                         runs. It STAYS there (the pad is outside the sweep circle —
                         the blade cannot touch it, so it also cannot be credited
                         with stalling anything), yet score stays ~0 and no clause
                         fires: delivery without silencing is worthless;
  3.  shallow beam     — beam resting in the bore with its tip SHORT of the sweep
                         circle: the mill keeps turning right past it, engagement
                         never latches, score ~0 — partial insertion earns nothing;
  4.  bore vs ball     — the 60 mm ball pushed straight at the bore side of the
                         courtyard travels freely (> 4 cm — the probe is not
                         vacuous), then presses against the wall and NEVER gets
                         inside: the seed's object cannot do the beam's job;
  5.  null policy      — 300 idle steps: mill still turning, score ~0, no success;
  6.  randomization    — READBACK over 8 seeded resets: ball xy, beam xy, mill angle
                         spreads are real; the motor spin direction lands on BOTH
                         signs;
  7.  determinism      — same seed twice -> identical readback;
  8.  conjunction      — a REAL stall constructed (beam placed at full depth, the
                         blade slams into it and the motor stalls) while the ball
                         still rests outside: engaged+silenced credit only (0.50),
                         NOT success — silencing without delivery is half the task;
  9.  wrong level      — ball settled on the ROOF directly above the pad (pad-band
                         x,y!): the z clause rejects — no over-the-wall credit;
  10. reversible stall — beam withdrawn (teleport instrumentation): the motor spins
                         the mill back up — the stall is a LIVE physical state, not
                         a latch — while the latched partial credit stays exactly
                         0.50 (credit does not evaporate);
  11. live-stall gate  — ball then parked ON the pad with engaged+silenced latched
                         but the mill RUNNING again: every latch and the pad clause
                         hold, success still refuses — it demands the stall NOW;
  12. rejection audit  — success() was never True at ANY judged point;
  13. final no-NaN     — all task-object states finite at the end.

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
    from .scene import _qapply  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.windmill_tollgate")().build(num_envs=args.num_envs,
                                                      device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.45, -1.05, 0.95)) + o),
                                tuple(np.array((0.38, 0.06, 0.08)) + o),
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
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def loc(body) -> torch.Tensor:
        return scene._local(body.data.root_pos_w)[0]

    def report(tag: str) -> None:
        bl = loc(scene.ball)
        kl = loc(scene.beam)
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) beam=({float(kl[0]):+.3f},{float(kl[1]):+.3f},"
              f"{float(kl[2]):.3f}) tip_x={float(scene.beam_tip_local()[0, 0]):+.3f} "
              f"yaw={math.degrees(float(scene.mill_yaw()[0])):+.1f}deg "
              f"stalled={bool(scene.stalled()[0])} eng={bool(scene._engaged[0])} "
              f"sil={bool(scene._silenced[0])} ent={bool(scene._entered[0])} "
              f"on_pad={bool(scene.ball_on_pad()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Teleport `body` to a court-local pose (the court never moves, yaw 0)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene._court_pos(all_ids)
        st[:, 0] += x
        st[:, 1] += y
        st[:, 2] += z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        body.write_root_state_to_sim(st, all_ids)

    def spin_progress(k: int) -> float:
        """Accumulated (unwrapped) mill yaw progress over k steps, radians."""
        tot = 0.0
        prev = float(scene.mill_yaw()[0])
        for _ in range(k):
            step(1)
            cur = float(scene.mill_yaw()[0])
            tot += math.atan2(math.sin(cur - prev), math.cos(cur - prev))
            prev = cur
        return tot

    def still(body, thr: float = 0.05) -> bool:
        return float(body.data.root_lin_vel_w[0].norm()) < thr

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.court.data.root_state_w).all()
                    and torch.isfinite(scene.mill.data.root_state_w).all()
                    and torch.isfinite(scene.beam.data.root_state_w).all()
                    and torch.isfinite(scene.ball.data.root_state_w).all())

    def readback() -> tuple:
        bl, kl = loc(scene.ball), loc(scene.beam)
        return (float(bl[0]), float(bl[1]), float(kl[0]), float(kl[1]),
                math.degrees(float(scene.mill_yaw()[0])),
                float(torch.sign(scene.mill.data.root_ang_vel_w[0, 2])))

    def construct_stall(max_steps: int = 1500) -> bool:
        """Instrumentation: the beam placed at FULL engagement depth in the bore
        (a pose the solve reaches by held-wrench insertion); the blade slam and the
        motor stall that follow are pure physics."""
        place_local(scene.beam, c.engage_x - 0.007 + c.beam_len / 2, c.chan_y,
                    (c.chan_z0 + c.chan_z1) / 2)
        for _ in range(max_steps):
            step(1)
            if bool(scene.stalled()[0]) and bool(scene._silenced[0]):
                return True
        return False

    # =========================== 1. settle / no-NaN + live mill =============================
    env.reset(seed=11)
    step(150)
    report("reset")
    bl, kl = loc(scene.ball), loc(scene.beam)
    prog = spin_progress(60)
    s, ok = judge()
    check("settle: states finite, ball and beam at rest on the open ground OUTSIDE "
          f"the courtyard, the mill demonstrably TURNING ({math.degrees(prog):+.0f} "
          "deg / 0.5 s), score ~0, no success",
          finite_all() and float(bl[1]) < -0.30 and still(scene.ball)
          and float(kl[0]) > 0.25 and still(scene.beam) and abs(prog) > 0.5
          and s <= 0.02 and not ok)

    # =========================== 2. seed strategy: delivery without silencing ==============
    # The seed's entire plan — ball in the goal region — constructed while the mill
    # runs. The pad lies OUTSIDE the sweep circle: the blade cannot touch the ball
    # there (so the parked ball can also never be what stalls the mill), and no
    # rubric clause fires.
    env.reset(seed=12)
    step(120)
    place_local(scene.ball, 0.140, 0.140, 0.037)
    step(300)
    report("seed-strategy")
    prog = spin_progress(60)
    s, ok = judge()
    check("seed strategy: ball parked ON the green pad while the mill RUNS — it "
          f"stays on the pad untouched (mill still turning {math.degrees(prog):+.0f} "
          "deg / 0.5 s), yet score ~0 and no success: delivery without silencing "
          "is worthless",
          bool(scene.ball_on_pad()[0]) and still(scene.ball) and abs(prog) > 0.5
          and not bool(scene._entered[0]) and s <= 0.02 and not ok)

    # =========================== 3. shallow beam: short of the sweep ========================
    env.reset(seed=13)
    step(120)
    tip_short = c.engage_x + 0.020                      # 0.085: outside the sweep
    place_local(scene.beam, tip_short + c.beam_len / 2, c.chan_y,
                (c.chan_z0 + c.chan_z1) / 2)
    step(60)
    prog = spin_progress(360)
    report("shallow-beam")
    tip_x = float(scene.beam_tip_local()[0, 0])
    s, ok = judge()
    check("shallow beam: beam resting in the bore with its tip SHORT of the sweep "
          f"circle (tip_x={tip_x:+.3f}) — the mill keeps turning right past it "
          f"({math.degrees(prog):+.0f} deg / 3 s), engagement never latches, "
          "score ~0, no success",
          0.075 < tip_x < 0.105 and abs(prog) > 4.0
          and not bool(scene._engaged[0]) and not bool(scene.stalled()[0])
          and s <= 0.02 and not ok)

    # =========================== 4. the bore refuses the ball ===============================
    env.reset(seed=14)
    step(120)
    place_local(scene.ball, 0.42, c.chan_y, c.ball_r + 0.001)
    step(30)
    x0 = float(loc(scene.ball)[0])
    f_in = torch.zeros(n, 3, device=device)
    f_in[:, 0] = -2.0
    for _ in range(300):
        scene.ball.set_external_force_and_torque(f_in.view(n, 1, 3).contiguous(),
                                                 zero_wrench, env_ids=all_ids,
                                                 is_global=True)
        env.step(no_action)
    scene.ball.set_external_force_and_torque(zero_wrench, zero_wrench,
                                             env_ids=all_ids)
    step(60)
    report("bore-vs-ball")
    bl = loc(scene.ball)
    s, ok = judge()
    check("bore vs ball: the 60 mm ball pushed straight at the bore side travels "
          f"freely ({100 * (x0 - float(bl[0])):.1f} cm — not vacuous) then presses "
          f"the wall and NEVER gets inside (x={float(bl[0]):+.3f} > 0.185, "
          f"z={float(bl[2]):.3f} stays low): the ball cannot do the beam's job",
          x0 - float(bl[0]) > 0.04 and float(bl[0]) > 0.185 and float(bl[2]) < 0.08
          and not bool(scene.ball_inside()[0]) and s <= 0.02 and not ok)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    prog = spin_progress(60)
    s, ok = judge()
    check("null policy: after 300 idle steps the mill still turns "
          f"({math.degrees(prog):+.0f} deg / 0.5 s), score ~0, no success",
          abs(prog) > 0.5 and s <= 0.02 and not ok)

    # =========================== 6-7. randomization + determinism ===========================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(10)
        reads.append(readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (ball_x, ball_y, beam_x, beam_y, "
          f"mill_yaw_deg, spin_sign):\n{arr}", flush=True)
    ball_span = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    beam_span = float((arr[:, 2:4].max(axis=0) - arr[:, 2:4].min(axis=0)).max())
    yaw_span = float(arr[:, 4].max() - arr[:, 4].min())
    check("randomization: ball xy span > 2 cm, beam xy span > 2 cm, mill angle "
          "span > 40 deg, motor spin lands on BOTH signs (readback)",
          ball_span > 0.02 and beam_span > 0.02 and yaw_span > 40.0
          and (arr[:, 5] > 0).any() and (arr[:, 5] < 0).any())
    env.reset(seed=77)
    step(10)
    r1 = np.array(readback())
    env.reset(seed=77)
    step(10)
    r2 = np.array(readback())
    check("determinism: the same seed reproduces the same layout (readback, <1e-4)",
          bool(np.abs(r1 - r2).max() < 1e-4))

    # =========================== 8. conjunction: silenced but ball absent ===================
    env.reset(seed=41)
    step(120)
    stalled = construct_stall()
    report("stall-only")
    bl = loc(scene.ball)
    s8, ok = judge()
    check("conjunction: beam at full depth -> the blade SLAMS into it and the motor "
          "stalls (a real, live jam) while the ball still rests outside — "
          f"engaged+silenced credit only (score {s8:.2f}), NOT success",
          stalled and bool(scene.beam_engaged_now()[0]) and float(bl[1]) < -0.30
          and abs(s8 - 0.50) < 0.01 and not ok)

    # =========================== 9. wrong level: ball on the roof ===========================
    place_local(scene.ball, 0.140, 0.140, c.wall_h + c.roof_t + c.ball_r + 0.002)
    step(240)
    report("roof-park")
    bl = loc(scene.ball)
    s9, ok = judge()
    check("wrong level: ball settled on the ROOF directly above the pad (x,y inside "
          f"the pad band, z={float(bl[2]):.3f}) — the z clause rejects: no "
          "over-the-wall credit, NOT success",
          c.pad_lo < float(bl[0]) < c.pad_hi and c.pad_lo < float(bl[1]) < c.pad_hi
          and float(bl[2]) > 0.15 and still(scene.ball)
          and not bool(scene.ball_on_pad()[0]) and not ok and abs(s9 - 0.50) < 0.01)

    # =========================== 10. reversible stall + latched credit ======================
    place_local(scene.beam, 0.42, 0.12, c.beam_s / 2 + 0.002)  # withdraw the brake
    step(240)
    prog = spin_progress(120)
    report("beam-withdrawn")
    s10, ok = judge()
    check("reversible stall: with the beam withdrawn the motor spins the mill back "
          f"up ({math.degrees(prog):+.0f} deg / 1 s — the stall is a LIVE state, "
          f"not a latch) while the latched credit stays exactly {s8:.2f} "
          f"(-> {s10:.2f}), still no success",
          abs(prog) > 1.0 and not bool(scene.stalled()[0])
          and abs(s10 - s8) < 1e-3 and not ok)

    # =========================== 11. live-stall gate ========================================
    place_local(scene.ball, 0.140, 0.140, 0.037)
    step(300)
    report("latches-vs-live")
    s11, ok = judge()
    check("live-stall gate: ball ON the pad with engaged+silenced already latched "
          "but the mill RUNNING again — every latch and the pad clause hold, yet "
          "success refuses: it demands the mill be held stalled NOW",
          bool(scene.ball_on_pad()[0]) and still(scene.ball)
          and bool(scene._engaged[0]) and bool(scene._silenced[0])
          and not bool(scene.stalled()[0]) and not ok and s11 <= 0.76)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.windmill_tollgate")
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
    except BaseException as exc:  # noqa: BLE001 — Kit teardown hangs; die loudly now
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
