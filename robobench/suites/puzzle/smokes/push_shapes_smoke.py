"""Recorded NullRobot oracle/smoke for :mod:`puzzle.push_shapes`.

The clean path drives each of the three plates through the two sub-skills the task is built
from — a pivot to its pad's yaw, then a slide onto the pad — checking that the matching stage
flag latches at each step and that all seven latch by the end.

The full run additionally proves the controls that make the staging meaningful: a block on
the WRONG pad does not seat (so shape correspondence is load-bearing), disturbing an
already-seated neighbour fails the live success gate while leaving latched credit intact,
and the usual yaw / lift / motion / state-roundtrip controls hold per piece.

Every invocation records both a compressed frame archive and an MP4. ``--demo`` runs only the
clean solve. The NullRobot applies no action; oracle state writes are confined to this smoke
and never appear in a reference solution.

Run on a Linux NVIDIA host with the Isaac Lab environment:

    python -m robobench.suites.puzzle.smokes.push_shapes_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False)
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--out", type=str, default="push_shapes_frames.npz")
parser.add_argument("--video", type=str, default="push_shapes_smoke.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.suites.puzzle.scenes.push_shapes import PIECES  # noqa: E402


def main() -> bool:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("puzzle.push_shapes")().build(num_envs=args.num_envs, device=device)
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
        parts = " ".join(
            f"{key}={float(scene.position_error(key)[0]) * 1000:4.1f}mm/"
            f"{math.degrees(float(scene.orientation_error(key)[0])):4.1f}deg"
            for key in PIECES
        )
        flags = "".join("1" if bool(v) else "0" for v in scene.stage_flags()[0])
        print(
            f"[smoke] {tag:24s} | {parts} flags={flags} "
            f"score={int(scene.score()[0])} success={bool(scene.success()[0])} "
            f"frames={len(frames)}",
            flush=True,
        )

    # ----- recording ------------------------------------------------------------------
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        origin = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        station_ys = [py for _, py in c.pad_stations]
        mid_y = (min(station_ys) - max(c.push_lengths) + max(station_ys)) / 2
        # STEEP view from just beyond the pad row, looking back down at the bench. What has
        # to be judgeable here is whether each outline sits on its own pad, and that is a
        # top-down question: a shallow angle (the first attempt used ~40 degrees) foreshortens
        # the two rows into one cluttered band and a shallow over-the-shoulder angle also lets
        # the hands hide the contact, which is how the push_t deliverable went wrong.
        # ~65 degrees keeps the pieces readable while retaining enough obliquity to show
        # thickness, and the small +x offset stops the robot's arms from covering a lane.
        eye = np.array((0.10, mid_y + 0.30, c.surface_z + 0.64)) + origin
        look = np.array((0.0, mid_y, c.surface_z + 0.005)) + origin
        env.sim.set_camera_view(tuple(eye), tuple(look), camera_prim_path="/OmniverseKit_Persp")
        render_product = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([render_product])
        for _ in range(6):
            env.sim.render()
        print(f"[smoke] camera warmup frame shape={np.asarray(annot.get_data()).shape}",
              flush=True)
    except Exception as exc:  # noqa: BLE001 -- a dead camera must fail the run, not wedge it
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

    physics_dt = env.sim.get_physics_dt()

    def write_piece(
        key: str,
        xy: torch.Tensor,
        quat: torch.Tensor,
        *,
        lift: float = 0.0,
        linear_speed: float = 0.0,
    ) -> None:
        state = torch.zeros(count, 13, device=device)
        state[:, 0:2] = xy
        # Pieces are written at BOARD-TOP sliding height; a piece written over its cutout
        # then falls in under gravity during the following steps, which is exactly the
        # physical event seated() grades.
        state[:, 2] = (
            env.iscene.env_origins[:, 2] + c.surface_z + c.board_thickness
            + c.block_half_height * c.piece_scale + lift
        )
        state[:, 3:7] = quat
        state[:, 7] = linear_speed
        # Cancel one gravity kick while the oracle re-pins a sliding body.
        state[:, 9] = 9.81 * physics_dt
        scene.blocks[key].write_root_state_to_sim(state, all_ids)
        env.iscene.update(0.0)

    def pad_pose(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        pad = scene.pads[key].data
        return pad.root_pos_w[:, :2].clone(), pad.root_quat_w.clone()

    def yaw_delta(key: str, degrees: float) -> torch.Tensor:
        radians = math.radians(degrees)
        delta = torch.tensor(
            [math.cos(radians / 2), 0.0, 0.0, math.sin(radians / 2)], device=device
        ).expand(count, 4)
        return quat_mul(scene.pads[key].data.root_quat_w, delta)

    def slerp_to(key: str, goal_xy: torch.Tensor, goal_quat: torch.Tensor,
                 steps: int) -> None:
        """Carry one piece to a pose over `steps`, so the motion is visible on camera."""
        start = scene.blocks[key].data.root_state_w.clone()
        start_xy = start[:, 0:2].clone()
        start_quat = start[:, 3:7].clone()
        for index in range(steps):
            alpha = (index + 1) / steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)     # smoothstep
            xy = start_xy * (1 - alpha) + goal_xy * alpha
            quat = start_quat * (1 - alpha) + goal_quat * alpha
            quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-9)
            write_piece(key, xy, quat)
            step(1)

    # =========================== clean oracle =========================================
    env.reset()
    step(90)
    report("reset tableau")
    check("reset score is zero", bool((scene.score() == 0).all()))
    check("reset leaves every block flat", bool(all(scene.not_lifted(k).all() for k in PIECES)))
    check(
        "reset yaw is outside the stage tolerance",
        bool(not any(scene.yaw_matched(k).any() for k in PIECES)),
    )

    for order, key in enumerate(PIECES):
        pad_xy, pad_quat = pad_pose(key)
        block_xy = scene.blocks[key].data.root_pos_w[:, :2].clone()
        # Sub-skill 1: pivot in place to the pad's yaw.
        slerp_to(key, block_xy, pad_quat, 45)
        step(20)
        report(f"{key}: pivoted")
        check(f"{key} yaw stage latches after the pivot", bool(scene.yaw_matched(key).all()))
        # Sub-skill 2: slide onto the pad.
        slerp_to(key, pad_xy, pad_quat, 60)
        step(45)
        report(f"{key}: seated")
        check(f"{key} seats on its own pad", bool(scene.seated(key).all()))
        check(f"{key} is settled once seated", bool(scene.settled(key).all()))
        check(
            f"score rises with {key} seated ({order + 1}/3)",
            bool((scene.score() > 0).all()),
        )

    report("all three seated")
    check("all three pieces seated gives success", bool(scene.success().all()))
    check("all seven stages latch", bool((scene.score() == 100).all()))
    solved_state = scene.get_state(all_ids)

    if not args.demo:
        # ==================== negative: wrong pad ==================================
        # Put the T on the X pad: correspondence must be what seats a piece.
        wrong_xy, wrong_quat = pad_pose("x")
        write_piece("t", wrong_xy, wrong_quat)
        step(30)
        report("negative: wrong pad")
        check("a block on another shape's pad does not seat", bool(not scene.seated("t").any()))
        check("wrong-pad placement fails success", bool(not scene.success().any()))
        scene.set_state(solved_state, all_ids)
        env.iscene.update(0.0)
        step(20)

        # ==================== negative: disturbed neighbour ========================
        pad_xy, pad_quat = pad_pose("x")
        write_piece("x", pad_xy + torch.tensor([[0.020, 0.0]], device=device), pad_quat)
        step(30)
        report("negative: neighbour hit")
        check("disturbing a seated piece fails success", bool(not scene.success().any()))
        check(
            "latched credit survives the disturbance",
            bool((scene.score() == 100).all()),
        )
        scene.set_state(solved_state, all_ids)
        env.iscene.update(0.0)
        step(20)

        # ==================== negative: yaw ========================================
        pad_xy, _ = pad_pose("l")
        write_piece("l", pad_xy, yaw_delta("l", 12.0))
        step(30)
        report("negative: yaw")
        check("12 degree yaw error unseats a piece", bool(not scene.seated("l").any()))
        check("yaw error fails success", bool(not scene.success().any()))
        scene.set_state(solved_state, all_ids)
        env.iscene.update(0.0)
        step(20)

        # ==================== negative: lift =======================================
        pad_xy, pad_quat = pad_pose("t")
        write_piece("t", pad_xy, pad_quat, lift=0.020)
        # Two steps only: enough to record the frame, far too few to land. A 20 mm drop
        # closes in ~64 ms of free fall, so stepping 10 here measured a piece that had
        # already settled back onto the pad and the control passed vacuously.
        step(2)
        report("negative: lift")
        check("20 mm lift is rejected", bool(not scene.not_lifted("t").any()))
        check("lifted piece fails success", bool(not scene.success().any()))
        scene.set_state(solved_state, all_ids)
        env.iscene.update(0.0)
        step(20)

        # ==================== negative: moving =====================================
        pad_xy, pad_quat = pad_pose("t")
        write_piece("t", pad_xy, pad_quat, linear_speed=0.20)
        report("negative: moving")
        check("a moving aligned piece is not settled", bool(not scene.settled("t").any()))
        check("unsettled piece fails success", bool(not scene.success().any()))

        # ==================== state round-trip =====================================
        env.reset()
        step(30)
        check("reset clears solved state", bool((scene.score() == 0).all()))
        scene.set_state(solved_state, all_ids)
        env.iscene.update(0.0)
        step(10)
        report("state restored")
        check("state round-trip restores score 100", bool((scene.score() == 100).all()))
        check("state round-trip restores success", bool(scene.success().all()))

    # =========================== artifacts + verdict ==================================
    check("viewport produced recorded frames", bool(frames))
    video_ok = False
    if frames:
        array = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=array, env="puzzle.push_shapes")
        print(f"[smoke] saved {array.shape} -> {args.out}", flush=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            height, width = array.shape[1:3]
            fps = max(1, round(120 / args.record_every))
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps),
                     "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.video],
                    input=array.tobytes(), check=True,
                )
                video_ok = os.path.isfile(args.video) and os.path.getsize(args.video) > 0
                print(f"[smoke] encoded -> {args.video}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- report, do not hide a dead encoder
                print(f"[smoke] MP4 encode FAILED: {exc!r}", flush=True)
        else:
            print("[smoke] ffmpeg unavailable; cannot encode MP4", flush=True)
    check("MP4 video encoded", video_ok)

    passed = sum(ok for _name, ok in checks)
    all_ok = passed == len(checks)
    print(
        f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} ({passed}/{len(checks)} checks)",
        flush=True,
    )
    print("PUSH_SHAPES_SMOKE_DONE", flush=True)
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
    except BaseException as exc:  # noqa: BLE001 -- always print, then hard-exit
        print(f"[smoke] UNCAUGHT FAILURE: {exc!r}", flush=True)
        ok = False
    hard_exit(0 if ok else 1)
