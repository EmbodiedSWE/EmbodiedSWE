#!/usr/bin/env python3
"""Grade one delivered solve.py — runs INSIDE the grading container.

Builds the registered preset from /bench (the same read-only tree the agent
had), wraps it in GradedEnv with the scene's grader from /graders, runs the
delivery's solve(env), writes verdict.json + progress.jsonl to /out. One
trajectory, no seeds yet. verdict.json is always written — success False
with the traceback if solve raises, and on the host's budget kill (SIGTERM,
30 s grace) the verdict is taken from the state at that moment
(timed_out: true). Only a hard hang inside the sim dies verdict-less.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

SOLUTION = "/solution/solve.py"  # fixed mount points: eval/scripts/run_grade.py
GRADERS_DIR = Path("/graders")
OUT = Path("/out")


def load_graders(grader_dir: Path) -> dict:
    """Import the suite's grader package from a path outside the /bench tree."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "suite_grader", grader_dir / "__init__.py",
        submodule_search_locations=[str(grader_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["suite_grader"] = mod  # registered first so its relative imports resolve
    spec.loader.exec_module(mod)
    return mod.GRADERS


def load_solve(path: str):
    """Import the delivery's entry point and hand back its solve(env). The
    whole solution folder is the deliverable: its dir goes on sys.path so
    solve.py's sibling modules import normally."""
    import importlib.util

    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location("solve", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "solve", None)):
        raise AttributeError(f"{path} does not expose solve(env)")
    return mod.solve


def start_renderer(env, out: Path, fps: float = 10.0, size: tuple = (1280, 720),
                   eye: tuple = (0.9, -1.1, 0.65), target_at: tuple = (0.30, -0.05, 0.10)):
    """Frame renderer: patch env.step to grab a JPEG into out/frames/ every
    1/fps of SIM time — frame timestamps (out/frames.jsonl) align with
    progress.jsonl, so a video or synced progress GUI can be assembled later.
    out/render.json records the camera/rendering args. Call after build,
    BEFORE the graded reset (the camera needs a sim re-parse). Returns
    flush() -> frame count."""
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import Camera, CameraCfg

    cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0.0,
                           height=size[1], width=size[0], data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                                            clipping_range=(0.01, 100.0))))
    env.sim.reset()  # re-parse so the camera is picked up
    anchor = env.iscene.env_origins[0].tolist()  # 3/4 view over the work surface
    anchor[2] += float(getattr(env.scene.cfg, "surface_z", 0.0) or 0.0)
    eye_w = [anchor[i] + eye[i] for i in range(3)]
    target_w = [anchor[i] + target_at[i] for i in range(3)]
    dev = env.device
    cam.set_world_poses_from_view(torch.tensor([eye_w], device=dev),
                                  torch.tensor([target_w], device=dev))

    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    (out / "render.json").write_text(json.dumps({
        "fps_sim": fps, "size": list(size),
        "camera_eye_world": eye_w, "camera_target_world": target_w,
        "focal_length": 24.0,
        "frames": "frames/<idx>.jpg; frames.jsonl maps each to sim_time_s/wall_s "
                  "(join with progress.jsonl on sim_time_s)",
    }, indent=2) + "\n")

    state = {"t": 0.0, "next": 0.0, "idx": 0, "t0": time.monotonic()}
    real_step = env.step

    def step(action, render: bool = False):
        state["t"] += env.dt * env.robot.control_period  # live: solves may retune decimation
        grab = state["t"] >= state["next"]
        real_step(action, render=render or grab)
        if grab:
            state["next"] += 1.0 / fps
            cam.update(env.dt)
            frame = cam.data.output["rgb"][0].detach().cpu().numpy()
            if frame.size and frame.any():  # warm-up frames come back empty
                name = f"{state['idx']:05d}.jpg"
                imageio.imwrite(frames / name, frame[..., :3], quality=90)
                with (out / "frames.jsonl").open("a") as f:
                    f.write(json.dumps({"frame": state["idx"], "file": f"frames/{name}",
                                        "sim_time_s": round(state["t"], 4),
                                        "wall_s": round(time.monotonic() - state["t0"], 3)}) + "\n")
                state["idx"] += 1

    env.step = step

    def flush() -> int:
        print(f"[render] {state['idx']} frames in {frames}", flush=True)
        return state["idx"]

    return flush


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--render", action="store_true",
                    help="render the run: /out/frames/*.jpg + frames.jsonl + render.json")
    args = ap.parse_args()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=args.render).app  # noqa: F841 — before isaaclab.sim

    import robobench

    robobench.discover()
    from robobench.core import GradedEnv
    from robobench.core.registries import ENVS

    grader_cls = load_graders(GRADERS_DIR)[args.scene]
    env = ENVS.get(args.preset)().build(num_envs=1)
    flush = start_renderer(env, out) if args.render else None  # re-parses sim: before the graded reset
    env.reset()
    grader = grader_cls(env)  # one grader instance = this trajectory

    import signal

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    # docker stop = SIGTERM + 30 s grace. The handler only sets a flag: raising
    # inside it can land in a Kit C++ callback frame and never reach our try.
    # The raise happens below, from our own per-step hook — always inside
    # solve's Python call chain.

    def on_record(rec: dict) -> None:
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if stop["flag"]:
            raise TimeoutError("grading budget reached")

    genv = GradedEnv(env, grader, on_record=on_record)

    result = {"preset": args.preset, "scene": args.scene, "criteria": grader.describe()}
    try:
        solve = load_solve(SOLUTION)  # the delivery

        t0 = time.monotonic()
        solve(genv)
        result["solve_wall_s"] = round(time.monotonic() - t0, 3)
        result["solve_sim_steps"] = grader.steps

        success, score = genv.verdict()
        result.update(success=success, score=score)
    except TimeoutError:  # verdict from the state at kill — peak score is already real
        success, score = genv.verdict()
        result.update(success=success, score=score, timed_out=True,
                      solve_sim_steps=grader.steps)
    except Exception:
        result.update(success=False, score=0.0, error=traceback.format_exc())

    if flush is not None:
        result["frames"] = flush()  # also on a crashed solve — partial frames show what happened
    (out / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"GRADE_DONE success={result['success']} score={result['score']}", flush=True)
    os._exit(0)  # Kit sometimes hangs on close; the verdict is on disk


if __name__ == "__main__":
    main()
