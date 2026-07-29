"""Smoke / oracle test for CombinationSafeScene — NullRobot, torque-driven, RECORDED.

The full crack-the-safe pipeline, exactly the strategy an agent should discover:
  1. calibration — sweep the dial at 3 speeds (slow/medium/fast) and print the
     rotation-rate trace stats: the detent clicks must be UNAMBIGUOUS dips at slow
     speed and smeared at fast speed (the designed speed/information tradeoff);
  2. explore — one slow RIGHTWARD sweep >360 deg and one slow LEFTWARD sweep,
     recording (reading, omega); cluster the rate dips -> candidate numbers
     ({A, C} from the right sweep, {B} from the left);
  3. verify detected candidates against the scene's true combination (privileged,
     smoke-only — the agent never sees this);
  4. enter — right to a1, left to B, right to a2; if the lock stage says no, enter
     the swapped hypothesis (a2, B, a1);
  5. open — torque the handle past 60 deg, torque the door open, teleport the prize
     out (NullRobot has no hands), settle, assert `success()`;
  6. negative control — reset, enter a deliberately WRONG combination, verify the
     handle stays bolted and success() is False.

Video is recorded throughout (viewport rgb annotator, the proven server recipe).

    python -m robobench.suites.articulated.smokes.combination_safe_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful crack (explore, enter, open, "
                         "prize out) — no speed calibration, no negative control")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="safe_smoke_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import json
import math
import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[safe-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    from robobench.suites.articulated.scenes import CombinationSafeSceneCfg

    # Demo lamps via the scene cfg: visual stage indicators for humans watching the
    # render. Smoke-only — the eval harness never enables them (they leak lock state).
    env = ENVS.get("articulated.safe")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=CombinationSafeSceneCfg(demo_lamps=True))
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    # --- recording ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        # Demo framing: close on the safe front so the numbered dial is READABLE
        # (at the old 1.1 m wide shot the face was a 100 px blur).
        cam = (0.30, -0.62, 0.42) if args.demo else (0.55, -0.85, 0.55)
        tgt = (0.0, 0.05, 0.22) if args.demo else (0.0, 0.15, 0.20)
        env.sim.set_camera_view(tuple(np.array(cam) + o),
                                tuple(np.array(tgt) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[safe-smoke] camera ready shape={np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[safe-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0
    frame_steps: list[int] = []  # sim step of each saved frame (video <-> trace sync)
    # Full per-step telemetry for the annotated demo video: (step, reading, omega,
    # stage, phase). Phase labels let the compositor caption what is happening.
    telem: list[tuple[int, float, float, int]] = []
    phases: list[tuple[int, str]] = []  # (start step, label)

    def phase(label: str) -> None:
        phases.append((step_i, label))
        print(f"[safe-smoke] PHASE @{step_i}: {label}", flush=True)

    def step(k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=True)
            telem.append((step_i, float(scene.dial_reading_deg()[0]),
                          float(scene._omega[0]), int(scene._stage[0])))
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
                    frame_steps.append(step_i)
            step_i += 1

    # --- dial driving: POSITION-RAMP servo (the way a finger turns a dial) --------------
    # A rate servo either stalls in a detent (weak) or limit-cycles (strong kick — v3
    # smoke had false clicks everywhere). A position ramp is stable IF the gains respect
    # the dial's inertia (I ~ 3.7e-4 kg m^2): KP=0.01 N*m/deg -> natural freq ~6 Hz,
    # KD=0.02 N*m/(rad/s) ~ critical damping (v4 ran KP=0.02/undertuned KD and
    # oscillated at 570 deg/s). In a detent the dial lags ~5 deg, then pops through —
    # a crisp local rate dip.
    KP, KD, CLAMP = 0.01, 0.02, 0.50

    def dial_track(theta_target: float) -> None:
        err = theta_target - float(scene._theta[0])
        w_rad = math.radians(float(scene._omega[0]))
        scene.dial_drive[0] = float(np.clip(KP * err - KD * w_rad, -CLAMP, CLAMP))
        step(1)

    def sweep(direction: float, deg: float, rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Sweep the dial `deg` degrees in `direction` (+1 left, -1 right) at `rate` deg/s.
        Returns (reading, omega) samples for the CONSTANT-RATE portion only: once the
        ramp target clamps at the end, the dial decelerates to a stall — v6 smoke found
        those tail samples cluster into FALSE detent candidates (and drag min|omega| to
        0 at every speed), so they are cut from the returned trace."""
        reads, omegas = [], []
        tgt = float(scene._theta[0])
        end = tgt + direction * deg
        guard = int(1.8 * deg / rate * 120) + 400
        for _ in range(guard):
            tgt += direction * rate / 120.0
            if direction * (tgt - end) > 0:
                tgt = end
            if tgt == end:
                break  # stop sampling at ramp end; let the dial coast to a stop below
            dial_track(tgt)
            reads.append(float(scene.dial_reading_deg()[0]))
            omegas.append(float(scene._omega[0]))
        for _ in range(150):  # settle onto the end target (unsampled)
            dial_track(end)
            if abs(float(scene._theta[0]) - end) < 1.0 and abs(float(scene._omega[0])) < 2.0:
                break
        scene.dial_drive[0] = 0.0
        # Head-cut spin-up (dial accelerating from rest reads as a dip too). 420-deg
        # sweeps re-pass any number hidden in the cut arc a full turn later.
        return np.array(reads[60:]), np.array(omegas[60:])

    def rest(steps: int = 90) -> None:
        scene.dial_drive[0] = 0.0
        step(steps)

    def dial_to(target: float, direction: float, rate: float = 40.0) -> None:
        """Turn the dial in `direction` to reading `target` (position-ramp all the way),
        then RELEASE and rest a full dwell so the stop registers. Do NOT servo-hold
        through the dwell: the hold creeps the last fraction of a degree at ~2 deg/s —
        just over the lock's 1.7 deg/s rest threshold — and no stop ever fires
        (measured v7: detection perfect, zero stop events)."""
        r = float(scene.dial_reading_deg()[0])
        delta = (direction * (target - r)) % 360.0  # travel along `direction`, in [0,360)
        goal = float(scene._theta[0]) + direction * delta
        tgt = float(scene._theta[0])
        guard = int(1.8 * delta / rate * 120) + 200
        for _ in range(guard):
            tgt += direction * rate / 120.0
            if direction * (tgt - goal) > 0:
                tgt = goal
            dial_track(tgt)
            if tgt == goal and abs(float(scene._theta[0]) - goal) < 1.0 \
                    and abs(float(scene._omega[0])) < 2.0:
                break
        rest(int(c.rest_dwell_steps * 1.6) + 20)  # free rest: the stop registers here
        print(f"[safe-smoke]   stop @{float(scene.dial_reading_deg()[0]):.1f} "
              f"(target {target:.1f}) stage={int(scene._stage[0])} "
              f"ctr={int(scene._rest_ctr[0])} armed={bool(scene._stop_armed[0])}", flush=True)

    def find_dips(reads: np.ndarray, omegas: np.ndarray, expect_dir: float) -> list[float]:
        """Cluster rate-dip samples into candidate dial numbers."""
        w = np.abs(omegas)
        base = np.median(w[w > 1.0]) if (w > 1.0).any() else 1.0
        dip = (w < 0.45 * base) & (np.abs(omegas) > 0.0)
        cand = reads[dip]
        clusters: list[list[float]] = []
        for r in cand:
            for cl in clusters:
                if abs((r - cl[0] + 180.0) % 360.0 - 180.0) < 6.0:
                    cl.append(r)
                    break
            else:
                clusters.append([r])
        # A real detent drags for ~detent_window*2 deg -> several samples.
        out = []
        for cl in clusters:
            if len(cl) >= 3:
                ang = np.array(cl)
                ref = ang[0]
                ang = (ang - ref + 180.0) % 360.0 - 180.0
                out.append(float((ref + ang.mean()) % 360.0))
        return sorted(out)

    def save_npz() -> None:
        """Frames + synced telemetry (per-step reading/omega/stage, frame<->step map,
        phase labels) — everything the annotated-demo compositor needs."""
        if not frames:
            return
        arr = np.stack(frames, axis=0)
        t = np.array(telem, dtype=np.float32)
        np.savez_compressed(
            args.out, frames=arr, env="articulated.safe",
            frame_steps=np.array(frame_steps, dtype=np.int64),
            telem_step=t[:, 0], telem_read=t[:, 1], telem_omega=t[:, 2],
            telem_stage=t[:, 3],
            phases=json.dumps(phases), combo=json.dumps(true_combo))
        print(f"[safe-smoke] saved {arr.shape} (+telemetry) -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[safe-smoke] hdfs upload rc={rc}", flush=True)

    env.reset()
    step(60)
    true_combo = [float(x) for x in scene._combo[0].tolist()]
    print(f"[safe-smoke] true combo (privileged, smoke-only): {true_combo}", flush=True)
    print(f"[safe-smoke] initial reading={float(scene.dial_reading_deg()[0]):.1f} "
          f"handle={float(scene.handle_angle_deg()[0]):.1f} "
          f"door={float(scene.door_angle_deg()[0]):.1f}", flush=True)

    # 1. calibration: 3 speeds, right sweeps; report dip contrast (skipped in demo)
    if not args.demo:
        print("[safe-smoke] CALIBRATION (speed -> dip contrast)", flush=True)
        for rate in (30.0, 90.0, 240.0):
            reads, om = sweep(-1.0, 380.0, rate)
            w = np.abs(om)
            base = np.median(w[len(w) // 10:])
            wmin = w[len(w) // 10:].min() if len(w) > 10 else 0.0
            print(f"[safe-smoke]   rate={rate:.0f}deg/s: n={len(om)} median|w|={base:.1f} "
                  f"min|w|={wmin:.1f} contrast={wmin / max(base, 1e-6):.2f}", flush=True)
            rest(60)

    # 2. explore: slow right + slow left
    phase("explore: slow RIGHT sweep — feel for clicks")
    reads_r, om_r = sweep(-1.0, 420.0, 30.0)
    rest(60)
    right_cands = find_dips(reads_r, om_r, -1.0)
    phase("explore: slow LEFT sweep")
    reads_l, om_l = sweep(+1.0, 420.0, 30.0)
    rest(60)
    left_cands = find_dips(reads_l, om_l, +1.0)
    print(f"[safe-smoke] right-sweep candidates (A,C): {right_cands}", flush=True)
    print(f"[safe-smoke] left-sweep candidates (B): {left_cands}", flush=True)
    phase(f"clicks found: R at {', '.join(f'{x:.0f}' for x in right_cands)}"
          f" | L at {', '.join(f'{x:.0f}' for x in left_cands)}")

    def match(dets: list[float], truth: list[float]) -> bool:
        return all(any(abs((d - t + 180.0) % 360.0 - 180.0) < 6.0 for d in dets) for t in truth)

    check("detect-right", match(right_cands, [true_combo[0], true_combo[2]])
          and len(right_cands) == 2, f"{right_cands} vs {[true_combo[0], true_combo[2]]}")
    check("detect-left", match(left_cands, [true_combo[1]]) and len(left_cands) == 1,
          f"{left_cands} vs {[true_combo[1]]}")

    # 3./4. enter the combination (try both orders of the two right-clicks)
    def enter(a: float, b: float, cc: float) -> bool:
        dwell = 150 if args.demo else 0  # demo: hold each stop ~1.2 s extra so the
        dial_to(a, -1.0)                 # lamp flipping green is unmissable
        rest(dwell)
        dial_to(b, +1.0)
        rest(dwell)
        dial_to(cc, -1.0)
        rest(dwell)
        return bool(scene.combo_entered()[0])

    if len(right_cands) == 2 and len(left_cands) == 1:
        a1, a2 = right_cands
        b = left_cands[0]
        phase(f"try order 1: R{a1:.0f} L{b:.0f} R{a2:.0f}")
        ok = enter(a1, b, a2)
        print(f"[safe-smoke] hypothesis 1 ({a1:.0f} R, {b:.0f} L, {a2:.0f} R): "
              f"entered={ok} stage={int(scene._stage[0])}", flush=True)
        if not ok:
            print(f"[safe-smoke] lock trace after hyp 1: {scene._trace}", flush=True)
            phase(f"no — try order 2: R{a2:.0f} L{b:.0f} R{a1:.0f}")
            ok = enter(a2, b, a1)
            print(f"[safe-smoke] hypothesis 2 ({a2:.0f} R, {b:.0f} L, {a1:.0f} R): "
                  f"entered={ok} stage={int(scene._stage[0])}", flush=True)
            if not ok:
                print(f"[safe-smoke] lock trace after hyp 2: {scene._trace}", flush=True)
    else:  # fall back to the true combo so the downstream mechanics still get exercised
        print("[safe-smoke] detection incomplete -> entering true combo to test mechanics",
              flush=True)
        ok = enter(*true_combo)
    check("combo-entered", bool(scene.combo_entered()[0]), f"stage={int(scene._stage[0])}")

    # 5. handle -> door -> prize out
    phase("combination entered — turn the handle, open the door")
    for _ in range(600):
        if float(scene.handle_angle_deg()[0]) >= c.handle_open_deg + 10:
            break
        scene.handle_drive[0] = 0.6
        step(1)
    scene.handle_drive[0] = 0.15  # keep it turned while the door opens
    check("handle-open", float(scene.handle_angle_deg()[0]) >= c.handle_open_deg,
          f"handle={float(scene.handle_angle_deg()[0]):.1f}deg")
    for _ in range(900):
        if float(scene.door_angle_deg()[0]) >= c.door_open_deg + 15:
            break
        scene.door_drive[0] = -1.2  # negative = outward
        step(1)
    scene.handle_drive[0] = 0.0
    check("door-open", bool(scene.door_open()[0]),
          f"door={float(scene.door_angle_deg()[0]):.1f}deg")

    # NullRobot prize extraction: teleport it out the open front, settle. Keep a light
    # outward door drive through the settle — the frictionless hinge lets a freely
    # coasting door swing back below door_open_deg mid-check (nondeterministic flake
    # seen once in the demo render).
    scene.door_drive[0] = -0.15
    W, D, H = c.outer
    cx, cy = c.safe_pos
    st = torch.zeros(env.num_envs, 13, device=device)
    st[:, 0:3] = env.iscene.env_origins + torch.tensor(
        [cx + 0.25, cy - D / 2 - 0.25, c.surface_z + c.prize_size / 2 + 0.05], device=device)
    st[:, 3] = 1.0
    scene.prize.write_root_state_to_sim(st, torch.arange(env.num_envs, device=device))
    step(80)
    check("success", bool(scene.success()[0]),
          f"prize_out={bool(scene.prize_out()[0])} opened={bool(scene.opened()[0])}")
    scene.door_drive[0] = 0.0

    if args.demo:  # deliverable video ends on the open safe + prize out
        phase("prize retrieved — safe cracked")
        step(40)
        save_npz()
        print("SAFE_SMOKE_DONE", flush=True)
        env.close()
        return

    # 6. negative control: fresh episode, wrong combo must NOT open the handle
    env.reset()
    step(30)
    truth = [float(x) for x in scene._combo[0].tolist()]
    wrong = [(t + 40.0) % 360.0 for t in truth]
    enter(*wrong)
    for _ in range(240):
        scene.handle_drive[0] = 0.6
        step(1)
    scene.handle_drive[0] = 0.0
    h = float(scene.handle_angle_deg()[0])
    check("negative-control", (not bool(scene.combo_entered()[0])) and h < 25.0,
          f"stage={int(scene._stage[0])} handle={h:.1f}deg success={bool(scene.success()[0])}")

    print(f"[safe-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    save_npz()
    print("SAFE_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin),
    wedging headless runs after everything is printed — the repo's standard hard-exit
    (see robobench/scripts/smoke.py): a watchdog guarantees the process ends."""
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()
