"""Recorded NullRobot oracle/smoke for :mod:`puzzle.push_t`.

The clean path visibly slides the T block along the work surface onto the physical target,
then checks the source task's 7 mm / 7 degree / no-lift rubric and the settled success gate.
The full run also proves independent XY, yaw, lift, motion, and state-roundtrip controls.

Every invocation records both a compressed frame archive and an MP4. ``--demo`` runs only
the clean solve. The NullRobot applies no action; oracle state writes are confined to this
smoke and never appear in the reference solution.

Run on a Linux NVIDIA host with the Isaac Lab environment:

    python -m robobench.suites.puzzle.smokes.push_t_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False)
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--out", type=str, default="push_t_frames.npz")
parser.add_argument("--video", type=str, default="push_t_smoke.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import shutil
import subprocess
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS


def main() -> bool:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("puzzle.push_t")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    count = env.num_envs
    all_ids = torch.arange(count, device=device)
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_mul

    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, condition))
        print(f"[smoke] {'PASS' if condition else 'FAIL'}: {name}", flush=True)

    frames: list[np.ndarray] = []

    def report(tag: str) -> None:
        print(
            f"[smoke] {tag:22s} | xy={float(scene.position_error()[0]) * 1000:5.1f}mm "
            f"angle={math.degrees(float(scene.orientation_error()[0])):4.1f}deg "
            f"flat={bool(scene.not_lifted()[0])} settled={bool(scene.settled()[0])} "
            f"score={int(scene.score()[0])} success={bool(scene.success()[0])} "
            f"frames={len(frames)}",
            flush=True,
        )

    # ----- recording ---------------------------------------------------------------------------
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        origin = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        task_y = (c.block_pos[1] + c.target_pos[1]) / 2
        eye = np.array((0.58, task_y - 0.70, c.surface_z + 0.60)) + origin
        look = np.array((0.0, task_y, c.surface_z + 0.01)) + origin
        env.sim.set_camera_view(tuple(eye), tuple(look), camera_prim_path="/OmniverseKit_Persp")
        render_product = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([render_product])
        for _ in range(6):
            env.sim.render()
        print(
            f"[smoke] camera warmup frame shape={np.asarray(annot.get_data()).shape}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED: {exc!r}", flush=True)

    step_index = 0

    def step(steps: int) -> None:
        nonlocal step_index
        for _ in range(steps):
            env.step(no_action)
            if annot is not None and step_index % args.record_every == 0:
                for _flush in range(3):
                    env.sim.render()
                image = np.asarray(annot.get_data())
                if image.size:
                    frames.append(image[..., :3].astype(np.uint8).copy())
            step_index += 1

    def write_block(
        xy: torch.Tensor,
        quat: torch.Tensor,
        *,
        lift: float = 0.0,
        linear_speed: float = 0.0,
    ) -> None:
        state = torch.zeros(count, 13, device=device)
        state[:, 0:2] = xy
        state[:, 2] = (
            env.iscene.env_origins[:, 2]
            + c.surface_z
            + c.block_half_height
            + lift
        )
        state[:, 3:7] = quat
        state[:, 7] = linear_speed
        scene.block.write_root_state_to_sim(state, all_ids)
        env.iscene.update(0.0)

    def target_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return scene.target.data.root_pos_w[:, :2].clone(), scene.target.data.root_quat_w.clone()

    def yaw_offset(degrees: float) -> torch.Tensor:
        radians = math.radians(degrees)
        delta = torch.tensor(
            [math.cos(radians / 2), 0.0, 0.0, math.sin(radians / 2)], device=device
        ).expand(count, 4)
        return quat_mul(scene.target.data.root_quat_w, delta)

    # =========================== clean oracle ================================================
    env.reset()
    step(90)
    report("reset tableau")
    check("reset score is zero", bool((scene.score() == 0).all()))
    check("reset block starts flat", bool(scene.not_lifted().all()))

    start = scene.block.data.root_state_w.clone()
    target_xy, target_quat = target_pose()
    move_steps = 120
    physics_dt = env.sim.get_physics_dt()
    for index in range(move_steps):
        alpha = (index + 1) / move_steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        state = start.clone()
        state[:, 0:2] = start[:, 0:2] * (1 - alpha) + target_xy * alpha
        state[:, 2] = (
            env.iscene.env_origins[:, 2] + c.surface_z + c.block_half_height
        )
        quat = start[:, 3:7] * (1 - alpha) + target_quat * alpha
        state[:, 3:7] = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-9)
        state[:, 7:13] = 0.0
        # Cancel one gravity kick while the smoke visibly re-pins the sliding body.
        state[:, 9] = 9.81 * physics_dt
        scene.block.write_root_state_to_sim(state, all_ids)
        step(1)

    write_block(target_xy, target_quat)
    step(90)
    report("oracle complete")
    check("aligned block scores 100", bool((scene.score() == 100).all()))
    check("aligned block reaches success", bool(scene.success().all()))
    solved_state = scene.get_state(all_ids)

    if not args.demo:
        # ======================= negative: XY ------------------------------------------------
        write_block(target_xy + torch.tensor([[0.020, 0.0]], device=device), target_quat)
        report("negative: XY")
        check("20 mm XY error fails source tolerance", not bool(scene.success().any()))
        check("20 mm XY error cannot score above 40", bool((scene.score() <= 40).all()))

        # ======================= negative: yaw -----------------------------------------------
        write_block(target_xy, yaw_offset(12.0))
        report("negative: yaw")
        check("12 degree yaw error fails source tolerance", not bool(scene.success().any()))
        check("wrong yaw cannot score above 80", bool((scene.score() <= 80).all()))

        # ======================= negative: lift ----------------------------------------------
        write_block(target_xy, target_quat, lift=0.020)
        report("negative: lift")
        check("20 mm lift is rejected", not bool(scene.not_lifted().any()))
        check("lifted alignment cannot score above 80", bool((scene.score() <= 80).all()))

        # ======================= negative: moving --------------------------------------------
        write_block(target_xy, target_quat, linear_speed=0.20)
        report("negative: moving")
        check("moving aligned block is not settled", not bool(scene.settled().any()))
        check("moving aligned block scores 90", bool((scene.score() == 90).all()))

        # ======================= state round-trip --------------------------------------------
        env.reset()
        step(30)
        check("reset clears solved state", bool((scene.score() == 0).all()))
        scene.set_state(solved_state, all_ids)
        env.iscene.update(0.0)
        report("state restored")
        check("state round-trip restores score 100", bool((scene.score() == 100).all()))
        check("state round-trip restores success", bool(scene.success().all()))

    # =========================== artifacts + verdict ========================================
    check("viewport produced recorded frames", bool(frames))
    video_ok = False
    if frames:
        array = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=array, env="puzzle.push_t")
        print(f"[smoke] saved {array.shape} -> {args.out}", flush=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            height, width = array.shape[1:3]
            fps = max(1, round(120 / args.record_every))
            try:
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s:v",
                        f"{width}x{height}",
                        "-r",
                        str(fps),
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        args.video,
                    ],
                    input=array.tobytes(),
                    check=True,
                )
                video_ok = os.path.isfile(args.video) and os.path.getsize(args.video) > 0
                print(f"[smoke] encoded -> {args.video}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[smoke] MP4 encode FAILED: {exc!r}", flush=True)
        else:
            print("[smoke] ffmpeg unavailable; cannot encode MP4", flush=True)
    check("MP4 video encoded", video_ok)

    passed = sum(ok for _name, ok in checks)
    all_ok = passed == len(checks)
    print(
        f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
        f"({passed}/{len(checks)} checks)",
        flush=True,
    )
    print("PUSH_T_SMOKE_DONE", flush=True)
    return all_ok


def hard_exit(code: int) -> None:
    """Guarantee Kit teardown cannot wedge a headless validation job."""
    watchdog = threading.Timer(10.0, lambda: os._exit(code))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    os._exit(code)


if __name__ == "__main__":
    try:
        ok = main()
    except BaseException as exc:  # noqa: BLE001
        print(f"[smoke] UNCAUGHT FAILURE: {exc!r}", flush=True)
        ok = False
    hard_exit(0 if ok else 1)
