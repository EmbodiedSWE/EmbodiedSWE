"""Render recorded episodes to RGB frames + previews (L5 visual replay; boots Isaac).

    # normal use — render every episode of a batch with the defaults:
    .venv/bin/python data_engine/scripts/render.py --headless \\
        data_gen_test/experiments/bulb_franka_osc/runs/test/data_gen/gen_e2e \\
        --batches bf_scene3

    # quick smoke (a few frames per episode), picking views, or another look:
    #   … --batches bf_scene3 --max-frames 40
    #   … --cams front wrist              (subset of the declared cameras)
    #   … --eye 1.0 -0.7 0.5 --target 0.22 0.1 0.18   (probe a one-time ad-hoc view, named `cam`)
    #   … --visual_draw 2                 (scene-owned VISUAL_PARAMS look)

    All flags: --help, or the args table in data_engine/README.html (tab 05).

Replays recorded states (`ep_NNNN/traj.npz`) kinematically — physics decides nothing —
and renders every env in one pass per frame, `num_envs` episodes at a time, one
TiledCamera per view. Writes per (episode, view): `imgs/<view>.mp4` (the dataset
video, streamed during the replay — no per-frame image files) and
`imgs/render_<view>.json` (the export contract, written last = the completion
marker); per batch: `replay_sheet_<view>.png`. The scene's `post_step` runs on every restored
state, so state-coupled visuals (the bulb glow) render correctly.

Cameras are DECLARED, not flag defaults: the scene's `CAMERAS` are external views
(eye/target env-origin-relative on the work surface), the robot's `CAMERAS` are ego
views (`link` mounts them on that body, riding the replayed motion). Default = all
declared; `--cams` selects; `--eye/--target` adds a one-time ad-hoc view for probing.
Per-view `bands` randomize the pose PER EPISODE with the engine's sampling grammar
({eye,target}_{x,y,z}, nominal = the declared value).

Each frame shows ONE standalone robot: RTX renders a single shared stage (Isaac has
no per-env world isolation), so `--env-spacing` (default 50 m) spreads the replay
grid beyond the camera's 40 m far clip — neighbors are clipped before rasterization,
pixel-identical to rendering truly independent worlds.

Visual diversification is scene-owned, like world physics: the cell's scene.py
declares `VISUAL_PARAMS` bands, and `--visual_draw k` applies draw k as ONE
stage-wide look for the pass (recorded into render_<cam>.json). The draw is written
onto the scene cfg BEFORE the build (build-consumed knobs — a table preset — work
with no extra code) and handed to `scene.apply_visual_params` after it (live knobs:
lights, textures, material rebinds). `--visual file.py` is the escape hatch for
looks that don't band-ify: its `setup(env)` runs after the build, `per_frame(env, t)`
before each rendered frame.

One scene per process: inputs spanning several scene cells are split automatically,
one subprocess per scene group.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="replay recorded episodes to RGB (L5)")
parser.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
parser.add_argument("--batches", nargs="*", default=[], help="batch names under data/ (default: all)")
parser.add_argument("--episodes", nargs="*", default=[], help="explicit ep dirs (override --batches)")
parser.add_argument("--num_envs", type=int, default=8, help="episodes replayed in parallel")
parser.add_argument("--fps", type=int, default=None,
                    help="dataset frame rate; default = the batch's recorded control rate "
                         "(one frame per latch — the matched regime; pass e.g. 30 to subsample)")
parser.add_argument("--size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
parser.add_argument("--cams", nargs="*", default=None,
                    help="declared cameras to render, by name (the scene's + robot's CAMERAS; "
                         "default: all declared)")
parser.add_argument("--eye", type=float, nargs=3, default=None,
                    help="ad-hoc ONE-TIME camera eye, env-origin-relative on the work surface "
                         "(for probing a view before writing it into the scene's CAMERAS)")
parser.add_argument("--target", type=float, nargs=3, default=None, help="ad-hoc camera look-at point")
parser.add_argument("--focal", type=float, default=16.0,
                    help="ad-hoc camera focal length, USD mm (24 ≈ 47° hFOV, 16 ≈ 66°)")
parser.add_argument("--cam", default="cam", help="name for the ad-hoc --eye/--target camera")
parser.add_argument("--env-spacing", type=float, default=50.0, dest="env_spacing",
                    help="replay grid spacing (m); beyond the 40 m camera far clip, so each "
                         "frame shows ONLY its own env — one standalone robot per image")
parser.add_argument("--warmup", type=int, default=12,
                    help="throwaway renders per chunk start (temporal-denoiser ghost flush)")
parser.add_argument("--no-sheet", action="store_true", help="skip the per-batch contact sheet")
parser.add_argument("--crf", type=int, default=18,
                    help="x264 crf of the dataset video (18 ≈ visually lossless; "
                         "LeRobot's own storage default is more aggressive)")
parser.add_argument("--max-frames", type=int, default=0, dest="max_frames",
                    help="cap frames per episode (0 = all) — smoke tests")
parser.add_argument("--visual_draw", type=int, default=None,
                    help="sample the scene's own VISUAL_PARAMS bands at this index and apply the "
                         "look stage-wide for the whole pass (omit = the nominal look); K looks of "
                         "the same episodes = K runs, kept apart with distinct --cam names")
parser.add_argument("--visual", default="", help="visual-diversify hook: py file with setup(env) / per_frame(env, t)")
parser.add_argument("--_scene", default="", help=argparse.SUPPRESS)  # internal: single-scene worker

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if (args.eye is None) != (args.target is None):
    parser.error("--eye and --target go together")
adhoc = ({"name": args.cam, "eye": tuple(args.eye), "target": tuple(args.target),
          "focal": args.focal} if args.eye else None)

DATA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_ENGINE_ROOT.parent))
sys.path.insert(0, str(DATA_ENGINE_ROOT))
from engine.replay import batch_scene, collect_episodes, group_by_scene  # noqa: E402

gen_root = Path(args.gen_root)
eps = collect_episodes(gen_root, args.batches or None, args.episodes or None)
if not eps:
    raise SystemExit("no episodes found")
groups = group_by_scene(eps)
if args._scene:
    groups = {args._scene: groups[args._scene]}

if len(groups) > 1:
    # one Isaac boot per scene cell — run each group in its own subprocess, sequentially
    print(f"[render] {len(eps)} episodes across {len(groups)} scenes — one worker per scene", flush=True)
    for scene, group in groups.items():
        cmd = [sys.executable, __file__, str(gen_root), "--_scene", scene,
               "--episodes", *[str(e) for e in group]]
        cmd += ["--num_envs", str(args.num_envs),
                "--size", *map(str, args.size),
                "--env-spacing", str(args.env_spacing),
                "--warmup", str(args.warmup),
                "--crf", str(args.crf), "--max-frames", str(args.max_frames)]
        if args.fps is not None:
            cmd += ["--fps", str(args.fps)]
        if args.cams is not None:
            cmd += ["--cams", *args.cams]
        if adhoc:
            cmd += ["--eye", *map(str, args.eye), "--target", *map(str, args.target),
                    "--focal", str(args.focal), "--cam", args.cam]
        if args.visual:
            cmd += ["--visual", args.visual]
        if args.visual_draw is not None:
            cmd += ["--visual_draw", str(args.visual_draw)]
        for f, on in [("--no-sheet", args.no_sheet), ("--headless", args.headless)]:
            if on:
                cmd.append(f)
        print(f"[render] scene {scene}: {len(group)} episodes", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"worker for {scene} failed ({r.returncode})")
    print("[render] ALL SCENES DONE", flush=True)
    sys.exit(0)

app = AppLauncher(args).app

import torch  # noqa: E402

from engine.replay import contact_sheet, replay_scene  # noqa: E402

(scene, group), = groups.items()
rendered, view_names = replay_scene(
    gen_root, scene, group,
    num_envs=args.num_envs, fps=args.fps, size=tuple(args.size),
    cams=args.cams, adhoc=adhoc,
    warmup=args.warmup, crf=args.crf, max_frames=args.max_frames,
    visual=args.visual or None, visual_draw=args.visual_draw, env_spacing=args.env_spacing,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
if not args.no_sheet:
    for batch_dir in sorted({ep.parent for ep in rendered}):
        for view in view_names:
            sheet = contact_sheet(batch_dir, view)
            if sheet:
                print(f"[render] sheet -> {sheet}", flush=True)
print(f"[render] DONE: {len(rendered)} episodes ({scene})", flush=True)

# Kit teardown regularly hangs inside app.close() under --enable_cameras — same
# watchdog hard-exit as scripts/generate.py; everything is written by now.
import threading  # noqa: E402

watchdog = threading.Timer(10.0, lambda: os._exit(0))
watchdog.daemon = True
watchdog.start()
app.close()
os._exit(0)
