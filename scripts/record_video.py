"""Record any smoke / experiment run to an mp4, without modifying it.

Runs the target headless and grabs the viewport camera every ~1/fps of sim time into the mp4,
so the video plays at real-time speed. Fully headless — safe to run next to another Isaac Sim.

The target can be given three ways:
  - a bare smoke name          -> resolved under --package (default: the assembly scripts pkg)
  - a full dotted module path  -> run as `python -m <module>`
  - a path to a .py file       -> run as a standalone script (e.g. an experiments/ solver)

Run from the repo root (the repo is pip-installed editable, so no PYTHONPATH needed):

    # an assembly smoke (bare name):
    .venv/bin/python scripts/record_video.py so101_smoke

    # an experiment solver (.py path), only the first 20s, higher quality:
    .venv/bin/python scripts/record_video.py \
        experiments/nut_thread_osc/solve.py --max-seconds 20 --quality 9

    # a smoke in another suite (dotted module), custom camera + output:
    .venv/bin/python scripts/record_video.py \
        robobench.suites.assembly.scripts.pc_gpu_smoke \
        --video videos/gpu.mp4 --eye 0.9 -1.1 0.65 --target-at 0.3 -0.05 0.1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import runpy
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("target", help="smoke name (bare, resolved under --package), a dotted module "
                                    "path, or a path to a standalone .py script")
parser.add_argument("--package", default="robobench.suites.assembly.scripts",
                    help="package a bare smoke name is resolved under")
parser.add_argument("--video", default=None,
                    help="output mp4 path (default: videos/<target>-<timestamp>.mp4)")
parser.add_argument("--fps", type=int, default=30, help="capture rate in sim time = playback rate")
parser.add_argument("--quality", type=int, default=8, metavar="0-10",
                    help="imageio mp4 quality (0=worst/smallest, 10=best/largest); "
                         "resolution (--size) is the other big lever")
parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                    help="stop recording and exit after this many seconds of VIDEO (playback = "
                         "sim time); omit to record the whole run")
parser.add_argument("--eye", type=float, nargs=3, default=(0.9, -1.1, 0.65),
                    help="camera eye, relative to the env origin on the work surface")
parser.add_argument("--target-at", type=float, nargs=3, default=(0.30, -0.05, 0.10), dest="target_at",
                    help="camera look-at point, same frame")
a, rest = parser.parse_known_args()  # extra args pass through to the target

_base = os.path.splitext(os.path.basename(a.target))[0]  # strip dir + .py
smoke_name = _base.rsplit(".", 1)[-1]                     # last part of a dotted module
if a.video:
    video_path = os.path.abspath(a.video)
else:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    video_path = os.path.abspath(os.path.join("videos", f"{smoke_name}-{stamp}.mp4"))
os.makedirs(os.path.dirname(video_path), exist_ok=True)

import robobench.core.env as _envmod  # noqa: E402  (app-free, safe to import now)

_rec = {"writer": None, "annot": None, "every": 1, "count": 0, "frames": 0, "max_frames": None}


def _ensure(env) -> None:
    """Set up on the first step, once the app and env exist."""
    if _rec["writer"] is not None:
        return
    import imageio
    import omni.replicator.core as rep

    rp = rep.create.render_product("/OmniverseKit_Persp", tuple(a.size))
    annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annot.attach([rp])
    anchor = env.iscene.env_origins[0].tolist()
    anchor[2] += float(getattr(env.scene.cfg, "surface_z", 0.0))
    env.sim.set_camera_view(eye=[anchor[i] + a.eye[i] for i in range(3)],
                            target=[anchor[i] + a.target_at[i] for i in range(3)])
    sim_dt = env.dt * env.robot.control_period  # sim time of one step() call
    _rec["annot"] = annot
    _rec["every"] = max(1, round(1.0 / (sim_dt * a.fps)))
    if a.max_seconds is not None:
        _rec["max_frames"] = max(1, round(a.max_seconds * a.fps))
    _rec["writer"] = imageio.get_writer(video_path, fps=a.fps, quality=a.quality)
    cap = f", stop @ {_rec['max_frames']} frames ({a.max_seconds}s)" if _rec["max_frames"] else ""
    print(f"[record] {video_path}: {a.size[0]}x{a.size[1]} @ {a.fps} fps q{a.quality} "
          f"(1 frame / {_rec['every']} steps){cap}", flush=True)


_orig_step = _envmod.BaseEnv.step


def _step(self, action, render: bool = False) -> None:
    _ensure(self)
    _rec["count"] += 1
    grab = _rec["count"] % _rec["every"] == 0
    _orig_step(self, action, render=render or grab)
    if grab:
        frame = np.asarray(_rec["annot"].get_data())
        if frame.size:  # the first few frames can come back empty
            _rec["writer"].append_data(frame[..., :3])
            _rec["frames"] += 1
            if _rec["max_frames"] is not None and _rec["frames"] >= _rec["max_frames"]:
                print(f"[record] --max-seconds reached at {_rec['frames']} frames, stopping.",
                      flush=True)
                _exit_flush(0)  # flush the mp4, then hard-exit the whole run


_envmod.BaseEnv.step = _step

_orig_close = _envmod.BaseEnv.close


def _close(self) -> None:
    """Flush the mp4 before teardown (all frames are in by now), then guard against Kit's
    close() sometimes hanging with a 30s force-exit watchdog."""
    _flush_writer()
    import threading

    watchdog = threading.Timer(30.0, lambda: _os_exit(0))
    watchdog.daemon = True
    watchdog.start()
    _orig_close(self)


_envmod.BaseEnv.close = _close

_os_exit = os._exit


def _flush_writer() -> None:
    w, _rec["writer"] = _rec["writer"], None
    if w is not None:
        w.close()
        print(f"[record] video written: {video_path}", flush=True)


def _exit_flush(code: int) -> None:
    try:
        _flush_writer()
    finally:
        _os_exit(code)


os._exit = _exit_flush

sys.argv = [smoke_name, "--headless", "--enable_cameras"] + rest
try:
    if a.target.endswith(".py"):
        runpy.run_path(a.target, run_name="__main__")
    elif "." in a.target:
        runpy.run_module(a.target, run_name="__main__")
    else:
        runpy.run_module(f"{a.package}.{a.target}", run_name="__main__")
finally:
    _flush_writer()  # crash path; the normal path exits through _exit_flush above
