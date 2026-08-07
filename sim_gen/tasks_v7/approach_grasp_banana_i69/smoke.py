"""Smoke battery for TiltMazeScene — REJECTION tests for the rubric, NullRobot, RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the correct
outcome). Teleported states here are rubric INSTRUMENTATION: construct a wrong outcome
as a settled state, then assert the rubric REJECTS it. Every force-driven negative also
asserts the actuator REALLY moved (no vacuous probes).

One linear run, 15 named checks:
  1. settle      — clean reset: finite state, ball resting in the open start bay, score ~0;
  2. random      — goal side flips and ball spawn varies across seeds; beacon READBACK
                   matches the sampled side;
  3. null        — 2 s of nothing: score ~0, no success;
  4. negative A  — the SEED's plan (pick the object up and carry it to the target):
                   ball dropped from above onto the goal well — the grate keeps it out
                   of the maze (it perches on the slats or rolls off the tray entirely;
                   its own weight tilts the tray, so it may not sit still) -> rejected,
                   score ~0;
  5. negative B  — teleport straight INTO the goal well, settled + tray level (the
                   success end-state, photographically): traversal latches never fired
                   -> rejected, score ~0;
  6. negative C  — real tilt-driven run into the WRONG well (tray verifiably tilted):
                   traversal credit only, branch/potted/success all zero -> cap ~0.25;
  7. trap        — irreversibility: full drive toward the goal cannot extract the ball
                   from the decoy well (escape needs ~25 deg, stops at 12 deg);
  8. oracle      — real tilt-steered traverse into the GOAL well: departed + crossed +
                   branch + potted latch, score 0.70 while still held;
  9. monotone    — the score never decreases along the oracle drive;
 10. exactness   — release the drive: spring re-levels, ball settles -> success() and
                   score == 1.0 exactly (ladder 0 -> 0.25 -> 0.70 -> 1.00);
 11. persist     — success holds 2 further seconds with no flicker;
 12. latch       — pin the gimbal at its stops: success revoked, latched 0.70 remains;
                   release again: success returns (live, not sticky);
 13. anchored    — the gimbal holds: tray pivot drifts < 5 mm and combined tilt stays
                   within the two-axis stop envelope (+ solver compliance) during every
                   drive — far below the well-escape angle;
 14. near-miss   — south leg only, then release: ball settled in the corridor on the
                   FLOOR, not in a well -> partial branch credit at most, strictly
                   below the potted plateau, no potted, no success;
 15. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.approach_grasp_banana_i69.smoke --headless
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
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
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

try:
    from .scene import TiltMazeScene  # registers "tilt_maze" + env
except ImportError:  # direct-file fallback
    from scene import TiltMazeScene

assert TiltMazeScene is not None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.tilt_maze")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    one = torch.arange(1, device=device)
    post = torch.tensor(c.post_pos, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.05, -0.80, 0.75)) + o),
                                tuple(np.array((0.50, -0.02, 0.10)) + o),
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
        p = scene.ball_local()[0].tolist()
        print(f"[smoke] {tag:12s} | ball_loc=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
              f"tilt={math.degrees(float(scene.tray_tilt()[0])):5.2f} "
              f"side={float(scene.side[0]):+.0f} dep={bool(scene.departed[0])} "
              f"cross={bool(scene.crossed[0])} branch={float(scene.branch[0]):.2f} "
              f"pot={bool(scene.potted[0])} goal={bool(scene.in_goal_well()[0])} "
              f"decoy={bool(scene.in_decoy_well()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport_ball(x: float, y: float, z: float) -> None:
        """World-frame (env-local) ball teleport, zero velocity."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.env_origins[0] + torch.tensor([x, y, z], device=device)
        st[0, 3] = 1.0
        scene.ball.write_root_state_to_sim(st, one)
        env.iscene.update(0.0)

    def drive(fn, stop, max_steps: int, trace: list | None = None):
        """Write `fn() -> (tau_x, tau_y)` into the plant buffer until `stop()` or
        timeout. Returns (stopped, max_tilt_deg, max_pivot_drift_m)."""
        max_tilt = 0.0
        max_drift = 0.0
        done = False
        for i in range(max_steps):
            if stop():
                done = True
                break
            tx, ty = fn()
            scene.tilt_drive[:, 0] = tx
            scene.tilt_drive[:, 1] = ty
            step(1)
            max_tilt = max(max_tilt, math.degrees(float(scene.tray_tilt()[0])))
            max_drift = max(max_drift, float(
                (scene.tray.data.root_pos_w[0, :2] - scene.env_origins[0, :2] - post).norm()))
            if trace is not None and i % 20 == 0:
                trace.append(float(scene.score()[0]))
        scene.tilt_drive[:] = 0.0
        return done, max_tilt, max_drift

    def south_fn():
        px = float(scene.ball_local()[0, 0])
        return (c.kappa * math.sin(math.radians(6.0)),
                -c.kappa * max(-0.05, min(0.05, 1.5 * px)))

    def branch_fn(sgn: float):
        return lambda: (c.kappa * math.sin(math.radians(2.5)),
                        sgn * c.kappa * math.sin(math.radians(7.0)))

    # two-axis stop envelope + solver limit compliance under sustained max drive
    # (measured ~1.5 deg/axis of soft overshoot); must stay far below escape_deg.
    tilt_env = math.degrees(math.acos(math.cos(math.radians(c.limit_deg)) ** 2)) + 3.5
    assert tilt_env < c.escape_deg - 4.0

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.pedestal, scene.cradle, scene.tray, scene.ball, scene.beacon)) \
        and bool(torch.isfinite(scene.score()).all())
    p = scene.ball_local()[0]
    in_bay = (abs(float(p[0])) < c.chan_hw and c.slat_y_hi < float(p[1]) < c.bay_y_hi
              and abs(float(p[2]) - c.ball_rest_floor) < 0.01)
    check("settle: clean reset (finite state, ball resting in the start bay, score ~0)",
          finite and in_bay and float(scene.score()[0]) < 0.02)

    # ========================= 2. randomization (side + spawn, beacon readback) ===============
    draws = []
    sides = set()
    beacon_ok = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        env.reset(seed=seed)
        step(10)
        p = scene.ball_local()[0]
        sd = float(scene.side[0])
        sides.add(sd)
        bx = float(scene.beacon.data.root_pos_w[0, 0] - scene.env_origins[0, 0])
        beacon_ok = beacon_ok and abs(bx - (c.post_pos[0] + sd * c.beacon_dx)) < 0.005
        draws.append((round(float(p[0]), 3), round(float(p[1]), 3), int(sd)))
    print(f"[smoke] draws (ball_xy_local, side) across seeds: {draws}", flush=True)
    check("randomization is real (goal side flips, ball spawn varies, beacon matches side)",
          len(sides) == 2 and len({d[:2] for d in draws}) >= 4 and beacon_ok)

    # ========================= 3. null policy =================================================
    env.reset(seed=7)
    step(240)
    report("null")
    check("null policy: 2 s of nothing -> score ~0, no success",
          float(scene.score()[0]) < 0.02 and not bool(scene.success()[0]))

    # ========================= 4. negative A: the seed's plan (carry + drop from above) =======
    # approach_grasp_banana's plan: pick the object up and carry it to the target. Here
    # the carry ends ON the roof grate — the ball cannot be inserted from above.
    env.reset(seed=21)
    step(30)
    sd = float(scene.side[0])
    teleport_ball(c.post_pos[0] + sd * 0.125, c.post_pos[1] - 0.07,
                  c.pivot_z + c.slat_z + c.slat_th / 2 + c.ball_r + 0.004)
    step(360)
    report("roof-drop")
    p = scene.ball_local()[0]
    entered = (float(p[2]) < 0.015 and abs(float(p[0])) < c.well_x1 + 0.002
               and c.cross_y0 - 0.002 < float(p[1]) < c.bay_y_hi)
    check("negative (seed strategy): ball carried above the goal well is kept out by "
          "the grate (perches or rolls off; never enters the maze) -> rejected, score ~0",
          not entered and not bool(scene.in_goal_well()[0]) and not bool(scene.departed[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 5. negative B: teleport into the goal well =====================
    env.reset(seed=8)
    step(30)
    sd = float(scene.side[0])
    teleport_ball(c.post_pos[0] + sd * (c.well_x0 + c.well_x1) / 2, c.post_pos[1] - 0.07,
                  c.pivot_z + c.ball_rest_well + 0.003)
    step(240)
    report("tp-well")
    physically_in = (bool(scene.in_goal_well()[0])
                     and float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_v
                     and math.degrees(float(scene.tray_tilt()[0])) < c.level_tol_deg)
    check("anti-cheat: teleport straight into the goal well (settled, tray level) -> "
          "rejected, score ~0",
          physically_in and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 6. negative C: real drive into the WRONG well ==================
    env.reset(seed=9)
    step(30)
    sd = float(scene.side[0])
    ok_s, tilt_s, _ = drive(south_fn, lambda: float(scene.ball_local()[0, 1]) < -0.055, 1800)
    ok_w, tilt_w, _ = drive(branch_fn(-sd), lambda: bool(scene.in_decoy_well()[0]), 1800)
    step(300)  # release: spring re-levels, ball settles in the decoy well
    report("wrong-well")
    check("negative (wrong branch): tray verifiably tilted (real actuation), ball settled "
          "in the DECOY well -> traversal credit only, capped ~0.25",
          ok_s and ok_w and max(tilt_s, tilt_w) > 3.0
          and bool(scene.in_decoy_well()[0]) and bool(scene.crossed[0])
          and not bool(scene.potted[0]) and not bool(scene.success()[0])
          and 0.20 <= float(scene.score()[0]) <= 0.27)

    # ========================= 7. the decoy trap is irreversible ==============================
    for k in range(6):  # pin toward the goal, rocking the other axis — try to shake it out
        sx = 1.0 if k % 2 == 0 else -1.0
        scene.tilt_drive[:, 0] = sx * c.drive_max
        scene.tilt_drive[:, 1] = sd * c.drive_max
        step(60)
    scene.tilt_drive[:] = 0.0
    step(120)
    report("trap")
    check("irreversibility: full drive toward the goal cannot extract the ball from the "
          "decoy well",
          bool(scene.in_decoy_well()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.27)

    # ========================= 8-10. oracle traverse + release ================================
    env.reset(seed=3)
    step(30)
    sd = float(scene.side[0])
    s_start = float(scene.score()[0])
    trace: list[float] = [s_start]
    ok_s, tilt_a, drift_a = drive(
        south_fn, lambda: float(scene.ball_local()[0, 1]) < -0.055, 1800, trace)
    s_mid = float(scene.score()[0])
    ok_g, tilt_b, drift_b = drive(
        branch_fn(sd), lambda: bool(scene.potted[0]), 1800, trace)
    report("oracle-pot")
    s_pot = float(scene.score()[0])
    check("oracle: real tilt-steered traverse into the GOAL well latches the full path, "
          "score 0.70 while still held",
          ok_s and ok_g and max(tilt_a, tilt_b) > 3.0 and bool(scene.departed[0])
          and bool(scene.crossed[0]) and bool(scene.potted[0])
          and 0.68 <= s_pot <= 0.72)
    trace.append(s_pot)
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] oracle score trace: "
          f"{' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace) // 12)])}", flush=True)
    check("rubric monotonicity: score never decreases along the oracle drive", mono)
    step(300)  # release
    report("released")
    ladder = [s_start, s_mid, s_pot, float(scene.score()[0])]
    print(f"[smoke] ladder: {' -> '.join(f'{s:.3f}' for s in ladder)}", flush=True)
    check("exactness: release -> spring re-levels, success() and score == 1.0 exactly "
          "(ladder monotone)",
          bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3
          and all(b >= a - 1e-4 for a, b in zip(ladder, ladder[1:])))

    # ========================= 11. persistence ================================================
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further seconds with no flicker", flicker == 0)

    # ========================= 12. achievement latch (disturb + recover) ======================
    scene.tilt_drive[:, 0] = c.drive_max
    scene.tilt_drive[:, 1] = c.drive_max
    step(240)
    report("disturbed")
    tilted = math.degrees(float(scene.tray_tilt()[0]))
    revoked = (not bool(scene.success()[0])
               and abs(float(scene.score()[0]) - 0.70) < 0.02 and tilted > 3.0)
    scene.tilt_drive[:] = 0.0
    step(300)
    report("recovered")
    check("achievement latch: pinned gimbal revokes success (latched 0.70 remains); "
          "releasing restores it (success is live)",
          revoked and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 13. mechanism anchored =========================================
    check("anchored: tray pivot drift < 5 mm and tilt within the two-axis stop envelope "
          "(+ compliance) during every drive, far below the well-escape angle",
          max(drift_a, drift_b) < 0.005 and max(tilt_a, tilt_b, tilted) < tilt_env)

    # ========================= 14. near-miss: parked beside the wells =========================
    env.reset(seed=5)
    step(30)
    ok_s, _t, _d = drive(south_fn, lambda: float(scene.ball_local()[0, 1]) < -0.055, 1800)
    step(300)  # release with the ball mid-corridor
    report("near-miss")
    p = scene.ball_local()[0]
    check("near-miss: ball settled on the corridor FLOOR beside the wells -> partial "
          "branch credit at most, strictly below the potted plateau, no success",
          ok_s and bool(scene.crossed[0]) and float(p[2]) > c.pot_z
          and not bool(scene.potted[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.56)

    # ========================= 15. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tilt_maze")
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
