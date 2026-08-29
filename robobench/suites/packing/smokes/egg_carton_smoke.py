"""Recorded NullRobot oracle/smoke for :mod:`packing.egg_carton`.

The main path settles the reset tableau, kinematically carries one egg above a physical pocket,
RELEASES it through the real opening, and verifies the 0 -> 90 -> 100 seating rubric.

The full run additionally proves that an egg lying sideways at a cavity centre does not count
and that get_state/set_state restores a quiet full-score state.  The other three eggs and the
open articulated lid remain visible scene geometry but are not success requirements.

Every invocation records viewport RGB frames. ``--demo`` runs only the clean solve.  Bodies
and joints are driven through scene handles; NullRobot contributes no action.

Run on a Linux NVIDIA host with the Isaac Lab environment:

    python -m robobench.suites.packing.smokes.egg_carton_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--demo",
    action="store_true",
    default=False,
    help="record only the clean successful solve",
)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="egg_carton_frames.npz")
parser.add_argument(
    "--hdfs_dir", type=str, default="", help="optional HDFS directory for the frames archive"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    # L20 render nodes can have a driver that Kit mis-decodes and silently disables RTX for.
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
    env = ENVS.get("packing.egg_carton")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, condition))
        print(f"[smoke] {'PASS' if condition else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(
            f"[smoke] {tag:22s} | seated={scene.seated()[0].int().tolist()} "
            f"occupied={scene.occupied()[0].int().tolist()} "
            f"lid={math.degrees(float(scene.lid_pos()[0])):6.1f}deg "
            f"score={int(scene.score()[0])} success={bool(scene.success()[0])} "
            f"frames={len(frames)}",
            flush=True,
        )

    # ----- recording ---------------------------------------------------------------------------
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        origin = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        wx, wy = c.workbench_pos
        eye = np.array((wx + 0.72, wy - 0.80, c.surface_z + 0.68)) + origin
        look = np.array((wx, wy + 0.10, c.surface_z + 0.04)) + origin
        env.sim.set_camera_view(
            tuple(eye), tuple(look), camera_prim_path="/OmniverseKit_Persp"
        )
        render_product = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([render_product])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED: {exc!r}", flush=True)

    step_index = 0

    def step(count: int, render: bool = True) -> None:
        nonlocal step_index
        for _ in range(count):
            env.step(no_action, render=render)
            if annot is not None and step_index % args.record_every == 0:
                # Flush sparse RTX accumulation so teleported/carry poses do not ghost.
                for _flush in range(3):
                    env.sim.render()
                image = np.asarray(annot.get_data())
                if image.size:
                    frames.append(image[..., :3].astype(np.uint8).copy())
            step_index += 1

    def wait_until(predicate, max_steps: int = 360, poll: int = 15) -> bool:
        if predicate():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if predicate():
                return True
        return False

    # ----- oracle manipulation -----------------------------------------------------------------
    def cavity_target(cavity: int, local_z: float) -> tuple[torch.Tensor, torch.Tensor]:
        """World target position/quaternion for an upright egg in a base-frame cavity."""
        body_pos = scene.carton.data.body_pos_w[:, scene._body_b]
        body_quat = scene.carton.data.body_quat_w[:, scene._body_b]
        xy = c.cavity_centers[cavity]
        local = torch.tensor([xy[0], xy[1], local_z], device=device).expand(n, 3)
        return body_pos + quat_apply(body_quat, local), body_quat.clone()

    def carry_to(
        name: str, target_pos: torch.Tensor, target_quat: torch.Tensor, steps: int = 72
    ) -> None:
        """Visible kinematic carry with normalized quaternion interpolation."""
        egg = scene.eggs[name]
        start_pos = egg.data.root_pos_w.clone()
        start_quat = egg.data.root_quat_w.clone()
        # q and -q encode the same rotation; take the shorter interpolation hemisphere.
        target_quat = target_quat.clone()
        target_quat[(start_quat * target_quat).sum(dim=1) < 0] *= -1
        for index in range(steps):
            alpha = (index + 1) / steps
            pos = start_pos * (1 - alpha) + target_pos * alpha
            quat = start_quat * (1 - alpha) + target_quat * alpha
            quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-9)
            state = torch.zeros(n, 13, device=device)
            state[:, 0:3] = pos
            state[:, 3:7] = quat
            # Counter one gravity kick while the state is re-pinned each physics step.
            state[:, 9] = 9.81 * env.sim.get_physics_dt()
            egg.write_root_state_to_sim(state, all_ids)
            step(1)
        released = torch.zeros(n, 13, device=device)
        released[:, 0:3] = target_pos
        released[:, 3:7] = target_quat
        egg.write_root_state_to_sim(released, all_ids)

    def drop_egg(name: str, cavity: int) -> bool:
        """Carry above a pocket, release through its opening, retry one honest drop if needed."""
        for attempt, release_z in enumerate((0.082, 0.072), start=1):
            target_pos, target_quat = cavity_target(cavity, release_z)
            carry_to(name, target_pos, target_quat)
            if wait_until(lambda: bool(scene.occupied()[:, cavity].all())):
                print(
                    f"[smoke] {name} seated in cavity {cavity} on attempt {attempt}", flush=True
                )
                return True
            print(f"[smoke] retrying {name} -> cavity {cavity}", flush=True)
        return False

    egg_names = list(scene.eggs)

    def fill(count: int) -> None:
        for cavity, name in enumerate(egg_names[:count]):
            check(f"drop {name} into distinct cavity {cavity}", drop_egg(name, cavity))
            check(
                f"score after {cavity + 1} egg(s) is 100",
                bool((scene.score() == 100).all()),
            )
            report(f"after {cavity + 1} egg(s)")

    # =========================== clean oracle ================================================
    env.reset()
    step(120)
    report("reset tableau")
    check("reset score is zero", bool((scene.score() == 0).all()))
    fill(1)
    step(120)
    report("oracle complete")
    check("one seated egg scores 100", bool((scene.score() == 100).all()))
    check("one seated egg reaches success", bool(scene.success().all()))
    solved_state = scene.get_state(all_ids)

    if not args.demo:
        # ======================= state round-trip ===========================================
        env.reset()
        step(30)
        check("reset clears solved state", bool((scene.score() == 0).all()))
        scene.set_state(solved_state, all_ids)
        step(30)
        report("state restored")
        check("state round-trip restores score 100", bool((scene.score() == 100).all()))
        check("state round-trip restores success", bool(scene.success().all()))

        # ======================= negative: sideways egg =====================================
        env.reset()
        step(60)
        target_pos, upright_quat = cavity_target(0, 0.030)
        half_turn = torch.tensor(
            [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], device=device
        ).expand(n, 4)
        sideways = torch.zeros(n, 13, device=device)
        sideways[:, 0:3] = target_pos
        sideways[:, 3:7] = quat_mul(upright_quat, half_turn)
        scene.eggs[egg_names[0]].write_root_state_to_sim(sideways, all_ids)
        env.iscene.update(0.0)
        report("negative: sideways")
        check(
            "sideways egg at cavity centre is not seated",
            not bool(scene.seated()[:, 0].any()),
        )
        check("sideways egg leaves score at zero", bool((scene.score() == 0).all()))
        check("sideways egg blocks success", not bool(scene.success().any()))

    # =========================== recording + verdict ========================================
    check("viewport produced recorded frames", bool(frames))
    if frames:
        array = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=array, env="packing.egg_carton")
        print(f"[smoke] saved {array.shape} -> {args.out}", flush=True)
        if args.hdfs_dir and shutil.which("hdfs"):
            subprocess.run(
                ["hdfs", "dfs", "-mkdir", "-p", args.hdfs_dir], check=True
            )
            subprocess.run(
                ["hdfs", "dfs", "-put", "-f", args.out, args.hdfs_dir + "/"], check=True
            )
            print(f"[smoke] uploaded {args.out} -> {args.hdfs_dir}/", flush=True)

    passed = sum(ok for _name, ok in checks)
    all_ok = passed == len(checks)
    print(
        f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
        f"({passed}/{len(checks)} checks)",
        flush=True,
    )
    print("EGG_CARTON_SMOKE_DONE", flush=True)
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
