"""Smoke / oracle test for MicrowaveMealScene — NullRobot, drive-tensor driven, RECORDED.

The full two-cycle meal pipeline plus every appliance rule, exercised in order:
  1. door/latch feasibility — 20 scripted open/close cycles (the brief's robot-free
     spike): the latch must pop under a firm pull and re-catch every close;
  2. button calibration — force ramp on the TIME key: a brushing contact must NOT
     register, a firm fingertip poke must; the measured force knee is published;
  3. state-machine checks, exactly the ported robocasa semantics + labeled extensions:
     start with the door open is a no-op; START with a wrong entry clears it (wrong-key
     penalty); START with an off-centre bowl is REFUSED (entry preserved); opening the
     door clears the entry; a started cycle spins the turntable; opening mid-cycle
     ABORTS with no heat;
  4. oracle solve — two full cycles: place bowl centred, close, key the sampled
     program, start, wait out the cycle (WAIT/MONITOR as first-class steps), retrieve
     the hot bowl to the serving mat; assert all 12 staged flags, score 100, success();
  5. negative control — unheated bowls parked on the mat must NOT succeed, and an
     EMPTY completed cycle must heat nothing;
  6. calibration sweep — centring offset vs start-acceptance: the measured boundary is
     the published number for the r_tol knob.

Video is recorded throughout (viewport rgb annotator, the proven server recipe).

    python -m robobench.suites.articulated.smokes.microwave_meal_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean two-cycle meal (no calibration ramps, no "
                         "negative control, no sweep)")
parser.add_argument("--only_sweep", action="store_true", default=False,
                    help="skip straight to the centring calibration sweep (fast "
                         "iteration on the knee check)")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="microwave_smoke_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import json
import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[mw-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("articulated.microwave")().build(num_envs=args.num_envs, device=device)
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
        # Framing: microwave front + counter, keypad and mat both in shot.
        cam = (0.55, -0.85, 0.55) if not args.demo else (0.45, -0.75, 0.45)
        tgt = (0.05, 0.10, 0.18)
        env.sim.set_camera_view(tuple(np.array(cam) + o), tuple(np.array(tgt) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[mw-smoke] camera ready shape={np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[mw-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0
    frame_steps: list[int] = []
    telem: list[tuple[int, float, int, int, int, float, int]] = []
    phases: list[tuple[int, str]] = []

    def phase(label: str) -> None:
        phases.append((step_i, label))
        print(f"[mw-smoke] PHASE @{step_i}: {label}", flush=True)

    def step(k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=True)
            telem.append((step_i, float(scene.door_angle_deg()[0]),
                          int(scene._entry[0]), int(scene._running[0]),
                          int(scene._timer[0]), float(scene.turntable_rate_dps()[0]),
                          int(scene.score()[0])))
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
                    frame_steps.append(step_i)
            step_i += 1

    def settle_until(pred, max_steps: int = 300) -> bool:
        """Poll `pred()` instead of asserting on a clock tick (the pen-holder lesson: fixed
        settle windows call slow settling a miss)."""
        for _ in range(max_steps):
            if bool(pred()):
                return True
            step(1)
        return bool(pred())

    # --- drivers (write the scene's drive tensors; post_step owns force buffers) -------
    def door_to_open(target: float = None, tq: float = -1.0, guard: int = 600) -> None:
        target = c.door_open_deg + 12.0 if target is None else target
        for _ in range(guard):
            if float(scene.door_angle_deg()[0]) >= target:
                break
            scene.door_drive[0] = tq
            step(1)
        scene.door_drive[0] = -0.10  # park lightly against the swing so it stays open
        step(20)

    def door_close(guard: int = 600) -> bool:
        for _ in range(guard):
            if float(scene.door_angle_deg()[0]) <= 0.8:
                break
            scene.door_drive[0] = 0.55
            step(1)
        scene.door_drive[0] = 0.0
        settle_until(lambda: bool(scene.door_closed()[0])
                     and abs(float(scene.door.data.root_ang_vel_w[0, 2])) < 0.05, 240)
        return bool(scene.door_closed()[0])

    def press(k: int, force: float = 3.0, guard: int = 120) -> float:
        """Press button k (0=TIME, 1=START) with `force` N until it registers (or the
        guard ends), then release until it re-arms. Returns the max depth reached."""
        dmax = 0.0
        for _ in range(guard):
            scene.btn_drive[0, k] = force
            step(1)
            dmax = max(dmax, float(scene.button_depth()[0, k]))
            if bool(scene._btn_pressed[0, k]):
                break
        scene.btn_drive[0, k] = 0.0
        for _ in range(guard):
            step(1)
            if float(scene.button_depth()[0, k]) < 0.0005:
                break
        step(10)
        return dmax

    def disc_top_z() -> float:
        return c.surface_z + c.wall_t + c.tt_clear + c.tt_h

    def put_bowl(b: int, dx: float = 0.0, dy: float = 0.0, on_disc: bool = True,
                 pos: tuple | None = None) -> None:
        """Teleport bowl b (NullRobot has no hands): default = onto the turntable at
        (dx, dy) off the axis; or to an explicit surface `pos`."""
        st = torch.zeros(env.num_envs, 13, device=device)
        if pos is None:
            x = c.mw_pos[0] + c.tt_off_x + dx
            y = c.mw_pos[1] + dy
            z = disc_top_z() + c.bowl_h / 2 + 0.003
        else:
            x, y = pos
            z = c.surface_z + c.bowl_h / 2 + 0.003
        st[:, 0:3] = env.iscene.env_origins + torch.tensor([x, y, z], device=device)
        st[:, 3] = 1.0
        scene.bowls[b].write_root_state_to_sim(
            st, torch.arange(env.num_envs, device=device))
        step(40)  # settle onto the disc / surface

    # Parking spots OUTSIDE the door's swing envelope (root cause of the run-2 sweep
    # failure): the 0.35 m door hinges at x = mw_x - W/2 and parks across the LEFT
    # front counter — the default bowl slots sit inside that arc, and a bowl
    # teleported back there gets bulldozed by the closing door and WEDGES it a few
    # degrees ajar, silently voiding every keypad press. Front-RIGHT, clear of the
    # arc (>= 0.6 m from the hinge) and clear of the serving mat's footprint.
    SAFE_PARK = ((0.30, -0.34), (0.45, -0.34))

    def park_bowls(*bs: int) -> None:
        for b in bs:
            put_bowl(b, pos=SAFE_PARK[b])

    def key_program(n_time: int) -> None:
        for _ in range(n_time):
            press(0)

    def run_cycle_wait(guard_extra: int = 400) -> bool:
        """WAIT out the running cycle, MONITORING the turntable mid-cycle."""
        total = int(scene._timer[0]) + guard_extra
        spun = False
        for _ in range(total):
            if not bool(scene._running[0]):
                break
            step(1)
            spun = spun or abs(float(scene.turntable_rate_dps()[0])) > 0.5 * c.spin_rate_dps
        return spun

    env.reset()
    step(60)
    requested = int(scene._requested[0])
    print(f"[mw-smoke] sampled program: TIME x{requested} "
          f"(cycle {requested * c.unit_steps / 120.0:.1f} s)", flush=True)
    print(f"[mw-smoke] door={float(scene.door_angle_deg()[0]):.1f}deg "
          f"offsets={[round(float(v), 3) for v in scene.bowl_offsets()[0]]}", flush=True)
    print(f"[mw-smoke] describe():\n{scene.describe()}", flush=True)

    # 1. door/latch feasibility: 20 scripted open/close cycles (skipped in demo)
    if not (args.demo or args.only_sweep):
        park_bowls(0, 1)  # clear the swing arc: 20 cycles must test the LATCH, not plowing
        phase("door/latch: 20 open-close cycles")
        good = 0
        for i in range(20):
            door_to_open()
            opened = bool(scene.door_open()[0])
            closed = door_close()
            good += int(opened and closed)
        check("door-latch-cycles", good == 20, f"{good}/20 clean cycles")

        # 2. button calibration: brush vs poke, published knee (door OPEN so the keypad
        # ignores the presses and the machine state stays clean)
        phase("button calibration: force ramp on TIME")
        door_to_open()
        knee = c.btn_k * c.press_depth
        ramp = []
        for force in (0.2, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0):
            scene._btn_pressed[:, 0] = False
            dmin, dmax = 1e9, -1e9
            for _ in range(60):
                scene.btn_drive[0, 0] = force
                step(1)
                disp = float(scene.buttons[0].data.root_pos_w[0, 1]
                             - scene._btn_home_y[0])
                dmin, dmax = min(dmin, disp), max(dmax, disp)
            reg = bool(scene._btn_pressed[0, 0])
            scene.btn_drive[0, 0] = 0.0
            for _ in range(60):
                step(1)
                if float(scene.button_depth()[0, 0]) < 0.0005:
                    break
            step(10)
            # dmin/dmax expose the SIGNED travel: a joint pinned at a limit shows a
            # dead [0, 0] range, a sign inversion shows negative motion (run-1 lesson).
            print(f"[mw-smoke]   F={force:.1f}N: disp=[{dmin * 1000:+.2f}, "
                  f"{dmax * 1000:+.2f}]mm registered={reg}", flush=True)
            ramp.append((force, round(dmax * 1000, 2), reg))
        print(f"[mw-smoke] force(N) -> depth(mm), registered: {ramp} "
              f"(spring knee = k*press_depth = {knee:.2f} N)", flush=True)
        check("button-brush-ignored", not any(r for f, _d, r in ramp if f <= 0.4),
              f"{[r for r in ramp if r[0] <= 0.4]}")
        check("button-poke-registers", all(r for f, _d, r in ramp if f >= 1.6),
              f"{[r for r in ramp if r[0] >= 1.6]}")
        door_close()

        # 3. state-machine checks
        phase("state machine: door-open START is a no-op")
        door_to_open()
        press(1)
        check("no-start-door-open", not bool(scene._running[0]))
        door_close()

        phase("state machine: wrong entry clears (wrong-key penalty)")
        key_program(requested + 1)  # one press too many
        wk0 = int(scene._wrong_key[0])
        press(1)
        check("wrong-key-clears", (not bool(scene._running[0]))
              and int(scene._entry[0]) == 0 and int(scene._wrong_key[0]) == wk0 + 1,
              f"entry={int(scene._entry[0])} wrong_key={int(scene._wrong_key[0])}")

        phase("state machine: off-centre bowl -> start REFUSED, entry kept")
        door_to_open()
        put_bowl(0, dx=0.05)
        door_close()
        key_program(requested)
        rf0 = int(scene._refusals[0])
        press(1)
        check("off-centre-refused", (not bool(scene._running[0]))
              and int(scene._refusals[0]) == rf0 + 1 and int(scene._entry[0]) == requested,
              f"refusals={int(scene._refusals[0])} entry={int(scene._entry[0])}")

        phase("state machine: door open clears entry; centred start runs + spins")
        door_to_open()
        check("door-open-clears-entry", int(scene._entry[0]) == 0)
        put_bowl(0)
        door_close()
        key_program(requested)
        press(1)
        started = bool(scene._running[0])
        for _ in range(90):
            step(1)
        spinning = abs(float(scene.turntable_rate_dps()[0])) > 0.5 * c.spin_rate_dps
        check("centred-start-runs", started and bool(scene._running[0]) and spinning,
              f"running={bool(scene._running[0])} "
              f"tt={float(scene.turntable_rate_dps()[0]):.0f}dps")

        phase("state machine: mid-cycle door open ABORTS, no heat")
        ab0 = int(scene._aborted[0])
        door_to_open()
        check("abort-on-open", (not bool(scene._running[0]))
              and int(scene._aborted[0]) == ab0 + 1 and not bool(scene._heated[0, 0]),
              f"aborted={int(scene._aborted[0])} heated={bool(scene._heated[0, 0])}")
        door_close()

    if not args.only_sweep:
        # 4. oracle: fresh episode, two full cycles + serving
        phase("ORACLE: fresh episode, two full heat-and-serve cycles")
        env.reset()
        step(60)
        requested = int(scene._requested[0])
        print(f"[mw-smoke] oracle program: TIME x{requested}", flush=True)
        park_bowls(0, 1)  # both wait clear of the door arc until their own cycle
        for b in range(2):
            phase(f"cycle {b + 1}: load bowl {b}, key program, run, serve")
            door_to_open()
            put_bowl(b)
            check(f"bowl{b}-centred", bool(scene.bowl_centred()[0, b]),
                  f"offset={float(scene.bowl_offsets()[0, b]) * 100:.1f}cm")
            ok_close = door_close()
            check(f"bowl{b}-enclosed", ok_close and bool(scene.bowl_centred()[0, b]))
            key_program(requested)
            press(1)
            check(f"bowl{b}-cycle-started", bool(scene._running[0]),
                  f"entry_was={requested} timer={int(scene._timer[0])}")
            spun = run_cycle_wait()
            check(f"bowl{b}-cycle-completed", (not bool(scene._running[0]))
                  and bool(scene._heated[0, b]) and spun,
                  f"heated={bool(scene._heated[0, b])} spun_mid_cycle={spun}")
            door_to_open()
            put_bowl(b, pos=(c.mat_pos[0] + (b * 2 - 1) * 0.06, c.mat_pos[1]))
            served = settle_until(
                lambda bb=b: bool((scene.bowl_on_mat() & scene.bowls_settled())[0, bb]))
            check(f"bowl{b}-served", served,
                  f"on_mat={bool(scene.bowl_on_mat()[0, b])}")
            door_close()

        flags = int(scene.stage_flags()[0].long().sum())
        check("stage-flags-12", flags == 12, f"{flags}/12: {scene._flags[0].tolist()}")
        check("score-100", int(scene.score()[0]) == 100, f"score={int(scene.score()[0])}")
        check("success", bool(scene.success()[0]))
        print(f"[mw-smoke] metrics: aborted={int(scene._aborted[0])} "
              f"wrong_key={int(scene._wrong_key[0])} refusals={int(scene._refusals[0])} "
              f"cycles={int(scene._cycles_done[0])}", flush=True)

    def save_npz() -> None:
        if not frames:
            return
        arr = np.stack(frames, axis=0)
        t = np.array(telem, dtype=np.float32)
        np.savez_compressed(
            args.out, frames=arr, env="articulated.microwave",
            frame_steps=np.array(frame_steps, dtype=np.int64),
            telem_step=t[:, 0], telem_door=t[:, 1], telem_entry=t[:, 2],
            telem_running=t[:, 3], telem_timer=t[:, 4], telem_tt=t[:, 5],
            telem_score=t[:, 6], phases=json.dumps(phases), requested=requested)
        print(f"[mw-smoke] saved {arr.shape} (+telemetry) -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[mw-smoke] hdfs upload rc={rc}", flush=True)

    if args.demo:  # deliverable video ends on the served meal
        phase("both bowls heated and served")
        step(60)
        save_npz()
        print("MICROWAVE_SMOKE_DONE", flush=True)
        env.close()
        return

    if not args.only_sweep:
        # 5. negative controls
        phase("NEGATIVE: unheated bowls on the mat must not succeed")
        env.reset()
        step(30)
        requested = int(scene._requested[0])
        put_bowl(0, pos=(c.mat_pos[0] - 0.06, c.mat_pos[1]))
        put_bowl(1, pos=(c.mat_pos[0] + 0.06, c.mat_pos[1]))
        step(60)
        check("negative-unheated", (not bool(scene.success()[0]))
              and int(scene.score()[0]) < 20,
              f"success={bool(scene.success()[0])} score={int(scene.score()[0])}")

        phase("NEGATIVE: an empty completed cycle heats nothing")
        key_program(requested)
        press(1)
        check("empty-start-allowed", bool(scene._running[0]))
        run_cycle_wait()
        check("empty-cycle-no-heat", (not bool(scene._running[0]))
              and not bool(scene._heated[0].any()) and not bool(scene.success()[0]),
              f"heated={scene._heated[0].tolist()} cycles={int(scene._cycles_done[0])}")

    # 6. calibration sweep: centring offset -> start accepted? (the r_tol knee)
    phase("calibration: centring offset vs start acceptance")
    env.reset()
    step(30)
    requested = int(scene._requested[0])
    park_bowls(0, 1)  # both clear of the swing arc; the loop only moves bowl 0
    sweep = []
    for off_cm in (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        door_to_open()  # also aborts/clears any prior state
        put_bowl(0, dx=off_cm / 100.0)
        if not door_close():
            door_close()  # a bounced latch would silently void the START press
        key_program(requested)
        wk0, rf0 = int(scene._wrong_key[0]), int(scene._refusals[0])
        entry_at_press = int(scene._entry[0])
        press(1)
        accepted = bool(scene._running[0])
        meas = float(scene.bowl_offsets()[0, 0]) * 100
        # Per-iteration diagnosis: entry@press vs requested (keying), door angle
        # (latch), centred/load_ok (geometry), and which machine event fired.
        print(f"[mw-smoke]   off={off_cm:.1f}cm: accepted={accepted} "
              f"meas={meas:.2f}cm door={float(scene.door_angle_deg()[0]):.2f}deg "
              f"entry@press={entry_at_press}/{requested} "
              f"centred={bool(scene.bowl_centred()[0, 0])} "
              f"load_ok={bool(scene.load_ok()[0])} "
              f"d_wrong={int(scene._wrong_key[0]) - wk0} "
              f"d_refused={int(scene._refusals[0]) - rf0}", flush=True)
        sweep.append((off_cm, round(meas, 2), accepted))
        if accepted:  # stop the cycle so the next iteration starts idle
            press(1)
    print(f"[mw-smoke] offset(cm) -> measured(cm), accepted: {sweep} "
          f"(r_tol = {c.r_tol * 100:.1f} cm)", flush=True)
    ok_knee = all(acc for off, _m, acc in sweep if off <= 2.0) \
        and not any(acc for off, _m, acc in sweep if off >= 4.0)
    check("centring-knee", ok_knee, f"boundary should sit at ~{c.r_tol * 100:.1f} cm")

    print(f"[mw-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)
    save_npz()
    print("MICROWAVE_SMOKE_DONE", flush=True)
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
