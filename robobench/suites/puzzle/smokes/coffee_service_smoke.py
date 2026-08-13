"""Smoke / oracle test for CoffeeServiceScene — NullRobot, drive-tensor driven, RECORDED.

The full one-cup service pipeline plus every appliance rule, exercised in order:
  1. cover/detent feasibility — 20 scripted slide open/close cycles: the detent must
     pop under a firm pull and the cover must re-seat every close;
  2. key calibration — force ramp on the START key (top-facing, presses DOWN): a
     brushing contact must NOT register, a firm poke must; the knee is published;
  3. state-machine checks, the ported semantics + labeled extensions: START with the
     cover open is a no-op; START without a pod (or without the cup) is REFUSED; a
     started brew glows + streams; sliding the cover open mid-brew ABORTS with no
     fill; removing the cup mid-brew ABORTS (the spill rule);
  4. oracle solve — one full service: pod into the bay, cover shut, cup under the
     spout, START, wait out the brew (WAIT/MONITOR as first-class steps), serve the
     filled cup to the tray; assert all 7 staged flags, score 100, success();
  5. negative control — an unfilled cup parked on the tray must NOT succeed;
  6. calibration sweep — cup offset from the spout axis vs start-acceptance: the
     measured boundary is the published number for the cup_r_tol knob.

Video is recorded throughout (viewport rgb annotator, the proven server recipe).

    python -m robobench.suites.puzzle.smokes.coffee_service_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean service (no calibration ramps, no negative "
                         "control, no sweep)")
parser.add_argument("--only_sweep", action="store_true", default=False,
                    help="skip straight to the centring calibration sweep")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="coffee_smoke_frames.npz")
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
    print(f"[coffee-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("puzzle.coffee")().build(num_envs=args.num_envs, device=device)
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
        o[2] += scene.cfg.surface_z  # the work rides the table top
        # Framing: machine front + tray both in shot (the machine is tall and narrow;
        # the camera rides higher than the microwave's to keep the top slider visible).
        cam = (0.85, -1.10, 0.85) if not args.demo else (0.72, -0.98, 0.72)
        tgt = (0.12, 0.02, 0.22)
        env.sim.set_camera_view(tuple(np.array(cam) + o), tuple(np.array(tgt) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[coffee-smoke] camera ready shape={np.asarray(annot.get_data()).shape}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[coffee-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0
    frame_steps: list[int] = []
    telem: list[tuple[int, float, float, int, int, int]] = []
    phases: list[tuple[int, str]] = []

    def phase(label: str) -> None:
        phases.append((step_i, label))
        print(f"[coffee-smoke] PHASE @{step_i}: {label}", flush=True)

    def step(k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=True)
            telem.append((step_i, float(scene.cover_pos()[0]),
                          float(scene.button_depth()[0]), int(scene._running[0]),
                          int(scene._timer[0]), int(scene.score()[0])))
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
                    frame_steps.append(step_i)
            step_i += 1

    def settle_until(pred, max_steps: int = 300) -> bool:
        """Poll `pred()` instead of asserting on a clock tick (the pen-holder lesson)."""
        for _ in range(max_steps):
            if bool(pred()):
                return True
            step(1)
        return bool(pred())

    # --- drivers (write the scene's drive tensors; post_step owns force buffers) -------
    def cover_to_open(force: float = -6.0, guard: int = 600) -> bool:
        """Pull the cover south past the open threshold, then park it lightly open."""
        for _ in range(guard):
            if float(scene.cover_pos()[0]) <= c.cover_open_pos - 0.005:
                break
            scene.cover_drive[0] = force
            step(1)
        scene.cover_drive[0] = -0.5  # park against the end stop so it stays open
        step(20)
        return bool(scene.cover_open()[0])

    def cover_close(force: float = 6.0, guard: int = 600) -> bool:
        for _ in range(guard):
            if float(scene.cover_pos()[0]) >= -0.004:
                break
            scene.cover_drive[0] = force
            step(1)
        scene.cover_drive[0] = 0.0
        settle_until(lambda: bool(scene.cover_closed()[0])
                     and abs(float(scene.cover.data.root_lin_vel_w[0, 1])) < 0.02, 240)
        return bool(scene.cover_closed()[0])

    def press(force: float = 3.0, guard: int = 120) -> float:
        """Press the START key with `force` N until it registers (or the guard ends),
        then release until it re-arms. Returns the max depth reached."""
        dmax = 0.0
        for _ in range(guard):
            scene.btn_drive[0] = force
            step(1)
            dmax = max(dmax, float(scene.button_depth()[0]))
            if bool(scene._btn_pressed[0]):
                break
        scene.btn_drive[0] = 0.0
        for _ in range(guard):
            step(1)
            if float(scene.button_depth()[0]) < 0.0003:
                break
        step(10)
        return dmax

    def put_pod(in_bay: bool = True, pos: tuple | None = None) -> None:
        """Teleport the pod (NullRobot has no hands): into the bay pocket, or to a
        surface/tray `pos`."""
        st = torch.zeros(env.num_envs, 13, device=device)
        if in_bay and pos is None:
            x = c.cm_pos[0] + c.pocket_off[0]
            y = c.cm_pos[1] + c.pocket_off[1]
            z = c.surface_z + c.pocket_floor_z + c.pod_h / 2 + 0.003
        else:
            x, y = pos
            z = c.surface_z + c.tray_h + c.pod_h / 2 + 0.003
        st[:, 0:3] = env.iscene.env_origins + torch.tensor([x, y, z], device=device)
        st[:, 3] = 1.0
        scene.pod.write_root_state_to_sim(st, torch.arange(env.num_envs, device=device))
        step(40)

    def put_cup(under_spout: bool = True, pos: tuple | None = None,
                dx: float = 0.0, dy: float = 0.0) -> None:
        """Teleport the cup: onto the platform under the spout (+ optional offset), or
        to a surface/tray `pos`."""
        st = torch.zeros(env.num_envs, 13, device=device)
        if under_spout and pos is None:
            # the STAGED point sits cup_stage_y_off south of the spout axis (a mug's
            # centre cannot reach the axis — the wall is 13 mm behind it)
            x = c.cm_pos[0] + c.spout_off[0] + dx
            y = c.cm_pos[1] + c.spout_off[1] + c.cup_stage_y_off + dy
            z = c.surface_z + c.platform_top_z + c.cup_h / 2 + 0.003
        else:
            x, y = pos
            z = c.surface_z + c.tray_h + c.cup_h / 2 + 0.003
        st[:, 0:3] = env.iscene.env_origins + torch.tensor([x, y, z], device=device)
        st[:, 3] = 1.0
        scene.cup.write_root_state_to_sim(st, torch.arange(env.num_envs, device=device))
        step(40)

    def run_brew_wait(guard_extra: int = 400) -> bool:
        """WAIT out the running brew, MONITORING the run lamp state mid-brew."""
        total = int(scene._timer[0]) + guard_extra
        lamp_on = False
        for _ in range(total):
            if not bool(scene._running[0]):
                break
            step(1)
            lamp_on = lamp_on or bool(scene._running[0])
        return lamp_on

    env.reset()
    step(60)
    print(f"[coffee-smoke] brew {c.brew_steps / 120.0:.1f} s; cover_pos="
          f"{float(scene.cover_pos()[0]) * 1000:.1f}mm "
          f"pod_seated={bool(scene.pod_seated()[0])} "
          f"cup_under_spout={bool(scene.cup_under_spout()[0])}", flush=True)
    print(f"[coffee-smoke] describe():\n{scene.describe()}", flush=True)

    # 1. cover/detent feasibility: 20 scripted open/close cycles (skipped in demo)
    if not (args.demo or args.only_sweep):
        phase("cover/detent: 20 open-close cycles")
        good = 0
        for i in range(20):
            opened = cover_to_open()
            closed = cover_close()
            good += int(opened and closed)
        check("cover-detent-cycles", good == 20, f"{good}/20 clean cycles")

        # 2. key calibration: brush vs poke, published knee (cover OPEN so the machine
        # ignores the presses and the state stays clean)
        phase("key calibration: force ramp on START")
        cover_to_open()
        knee = c.btn_k * c.press_depth
        ramp = []
        for force in (0.1, 0.2, 0.4, 0.8, 1.4, 2.2, 3.5):
            scene._btn_pressed[:] = False
            dmin, dmax = 1e9, -1e9
            for _ in range(60):
                scene.btn_drive[0] = force
                step(1)
                disp = float(scene._btn_home_z[0] - scene.button.data.root_pos_w[0, 2])
                dmin, dmax = min(dmin, disp), max(dmax, disp)
            reg = bool(scene._btn_pressed[0])
            scene.btn_drive[0] = 0.0
            for _ in range(60):
                step(1)
                if float(scene.button_depth()[0]) < 0.0003:
                    break
            step(10)
            print(f"[coffee-smoke]   F={force:.1f}N: disp=[{dmin * 1000:+.2f}, "
                  f"{dmax * 1000:+.2f}]mm registered={reg}", flush=True)
            ramp.append((force, round(dmax * 1000, 2), reg))
        print(f"[coffee-smoke] force(N) -> depth(mm), registered: {ramp} "
              f"(spring knee = k*press_depth = {knee:.2f} N)", flush=True)
        check("key-brush-ignored", not any(r for f, _d, r in ramp if f <= 0.2),
              f"{[r for r in ramp if r[0] <= 0.2]}")
        check("key-poke-registers", all(r for f, _d, r in ramp if f >= 1.4),
              f"{[r for r in ramp if r[0] >= 1.4]}")
        cover_close()

        # 3. state-machine checks
        phase("state machine: cover-open START is a no-op")
        cover_to_open()
        put_pod(in_bay=True)
        put_cup(under_spout=True)
        press()
        check("no-start-cover-open", not bool(scene._running[0]))

        phase("state machine: no pod -> START refused")
        put_pod(pos=c.pod_slot)  # pod back on the tray
        cover_close()
        rf0 = int(scene._refusals[0])
        press()
        check("no-pod-refused", (not bool(scene._running[0]))
              and int(scene._refusals[0]) == rf0 + 1,
              f"refusals={int(scene._refusals[0])}")

        phase("state machine: no cup -> START refused")
        cover_to_open()
        put_pod(in_bay=True)
        put_cup(pos=c.cup_slot)  # cup back on the tray
        cover_close()
        rf0 = int(scene._refusals[0])
        press()
        check("no-cup-refused", (not bool(scene._running[0]))
              and int(scene._refusals[0]) == rf0 + 1,
              f"refusals={int(scene._refusals[0])}")

        phase("state machine: full preconditions -> brew starts")
        put_cup(under_spout=True)
        press()
        check("preconditions-start", bool(scene._running[0]),
              f"timer={int(scene._timer[0])}")

        phase("state machine: mid-brew cover open ABORTS, no fill")
        ab0 = int(scene._aborted[0])
        cover_to_open()
        check("abort-on-open", (not bool(scene._running[0]))
              and int(scene._aborted[0]) == ab0 + 1 and not bool(scene._filled[0]),
              f"aborted={int(scene._aborted[0])} filled={bool(scene._filled[0])}")
        cover_close()

        phase("state machine: mid-brew cup removal ABORTS (spill)")
        press()
        check("restart-ok", bool(scene._running[0]))
        ab0 = int(scene._aborted[0])
        put_cup(pos=c.cup_slot)
        check("abort-on-spill", (not bool(scene._running[0]))
              and int(scene._aborted[0]) == ab0 + 1 and not bool(scene._filled[0]),
              f"aborted={int(scene._aborted[0])} filled={bool(scene._filled[0])}")

    if not args.only_sweep:
        # 4. oracle: fresh episode, one full service
        phase("ORACLE: fresh episode, one full brew-and-serve service")
        env.reset()
        step(60)
        phase("service: open bay, load pod, close, stage cup, start, wait, serve")
        check("oracle-cover-opens", cover_to_open())
        put_pod(in_bay=True)
        check("oracle-pod-seated", bool(scene.pod_seated()[0]))
        check("oracle-cover-closes", cover_close())
        put_cup(under_spout=True)
        check("oracle-cup-staged", bool(scene.cup_under_spout()[0]))
        press()
        check("oracle-brew-started", bool(scene._running[0]),
              f"timer={int(scene._timer[0])}")
        lamp = run_brew_wait()
        check("oracle-brew-completed", (not bool(scene._running[0]))
              and bool(scene._filled[0]) and lamp,
              f"filled={bool(scene._filled[0])}")
        put_cup(pos=c.cup_slot)
        served = settle_until(
            lambda: bool((scene.cup_on_tray() & scene.cup_settled())[0]))
        check("oracle-served", served, f"on_tray={bool(scene.cup_on_tray()[0])}")

        flags = int(scene.stage_flags()[0].long().sum())
        check("stage-flags-7", flags == 7, f"{flags}/7: {scene._flags[0].tolist()}")
        check("score-100", int(scene.score()[0]) == 100, f"score={int(scene.score()[0])}")
        check("success", bool(scene.success()[0]))
        print(f"[coffee-smoke] metrics: aborted={int(scene._aborted[0])} "
              f"refusals={int(scene._refusals[0])} "
              f"cycles={int(scene._cycles_done[0])}", flush=True)

    def save_npz() -> None:
        if not frames:
            return
        arr = np.stack(frames, axis=0)
        t = np.array(telem, dtype=np.float32)
        np.savez_compressed(
            args.out, frames=arr, env="puzzle.coffee",
            frame_steps=np.array(frame_steps, dtype=np.int64),
            telem_step=t[:, 0], telem_cover=t[:, 1], telem_btn=t[:, 2],
            telem_running=t[:, 3], telem_timer=t[:, 4], telem_score=t[:, 5],
            phases=json.dumps(phases))
        print(f"[coffee-smoke] saved {arr.shape} (+telemetry) -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(
            f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
            f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[coffee-smoke] hdfs upload rc={rc}", flush=True)

    if args.demo:  # deliverable video ends on the served coffee
        phase("the coffee is served")
        step(60)
        save_npz()
        print("COFFEE_SMOKE_DONE", flush=True)
        env.close()
        return

    if not args.only_sweep:
        # 5. negative control
        phase("NEGATIVE: an unfilled cup on the tray must not succeed")
        env.reset()
        step(30)
        put_cup(pos=c.cup_slot)
        step(60)
        check("negative-unfilled", (not bool(scene.success()[0]))
              and int(scene.score()[0]) < 20,
              f"success={bool(scene.success()[0])} score={int(scene.score()[0])}")

    # 6. calibration sweep: cup offset from the spout axis -> start accepted?
    phase("calibration: cup offset vs start acceptance")
    env.reset()
    step(30)
    cover_to_open()
    put_pod(in_bay=True)
    cover_close()
    sweep = []
    for off_cm in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        put_cup(under_spout=True, dx=off_cm / 100.0)
        rf0 = int(scene._refusals[0])
        press()
        accepted = bool(scene._running[0])
        meas = float((scene.cup.data.root_pos_w[0, :2]
                      - scene._spout_axis_xy()[0]).norm()) * 100
        print(f"[coffee-smoke]   off={off_cm:.1f}cm: accepted={accepted} "
              f"meas={meas:.2f}cm cup_ok={bool(scene.cup_under_spout()[0])} "
              f"d_refused={int(scene._refusals[0]) - rf0}", flush=True)
        sweep.append((off_cm, round(meas, 2), accepted))
        if accepted:  # stop the brew so the next iteration starts idle
            press()
    print(f"[coffee-smoke] offset(cm) -> measured(cm), accepted: {sweep} "
          f"(cup_r_tol = {c.cup_r_tol * 100:.1f} cm)", flush=True)
    # the staged baseline already sits |cup_stage_y_off| off-axis; judge the knee by
    # the MEASURED radial offset against the 5 cm gate
    ok_knee = all(acc for _o, meas, acc in sweep if meas <= c.cup_r_tol * 100 - 0.5) \
        and not any(acc for _o, meas, acc in sweep if meas >= c.cup_r_tol * 100 + 0.5)
    check("centring-knee", ok_knee, f"boundary should sit at ~{c.cup_r_tol * 100:.1f} cm")

    print(f"[coffee-smoke] RESULT: "
          f"{'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}", flush=True)
    save_npz()
    print("COFFEE_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() — the repo's
    standard hard-exit: a watchdog guarantees the process ends."""
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
