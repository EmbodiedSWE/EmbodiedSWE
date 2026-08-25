"""Bake canonical episodes into a LeRobotDataset (+ dialect metadata; no Isaac).

Run with lerobot's OWN venv:

    ~/Documents/Research/lerobot/.venv/bin/python vla/convert/convert.py \\
        <…/data_gen/<gen_name>> --repo-id cosigen/bulb_franka_osc \\
        [--control_space joint_vel] [--control_freq 15]  (default space: joint_target) [--batches …] \\
        [--cams front wrist] [--root <out dir>] [--task "…"] [--include-failures]

Episodes are read through the canonical Episode form (episode.py), projected into
ONE control space (conventions.py) at ONE control frequency, streamed frame-by-frame
into a LeRobotDataset — videos are decoded sequentially, an episode never sits in
RAM. Sim episodes additionally carry the verbatim recorded command as a
`raw_command` column. Successful episodes only by default.

Besides the dataset, the bake writes into its meta/:
  modality.json   GR00T's named-parts map (state/action slices, video keys)
  bake.json       THE provenance stamp: control space + frequency, parts, origin,
                  robot_type, gripper convention, the recorded control law
                  (when the episodes carry one), source episodes + git shas.
The stamp is the single source of truth an eval bridge reads to build the
matching tracking controller — train/eval convention drift is structurally
impossible. Episodes whose stamped control law differs are refused (bake them
separately; mixed-law pools must stay partitionable).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conventions import CONVENTIONS  # noqa: E402
from episode import read_sim_episode  # noqa: E402

parser = argparse.ArgumentParser(description="bake episodes into a LeRobotDataset")
parser.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
parser.add_argument("--repo-id", required=True, dest="repo_id")
parser.add_argument("--control_space", default="joint_target", choices=sorted(CONVENTIONS),
                    help="what the action column means (see conventions.py)")
parser.add_argument("--control_freq", type=float, default=None,
                    help="control frequency of the baked labels (Hz); default: the episodes' "
                         "native control rate (one tick per latch); a lower frequency must "
                         "divide the video fps (pi-DROID: 15)")
parser.add_argument("--batches", nargs="*", default=[], help="batch names under data/ (default: all)")
parser.add_argument("--episodes", nargs="*", default=[], help="explicit ep dirs (override --batches)")
parser.add_argument("--cams", nargs="*", default=None, help="views to pack (default: all rendered)")
parser.add_argument("--root", default="", help="dataset dir (default: <gen_root>/datasets/<repo_id>)")
parser.add_argument("--task", default="", help="language instruction (default: the scene's Goal sentence)")
parser.add_argument("--robot-type", default="", dest="robot_type")
parser.add_argument("--include-failures", action="store_true", dest="include_failures")
parser.add_argument("--filter-idle", action="store_true", dest="filter_idle",
                    help="drop dead ticks (robot AND objects static AND no command intent); "
                         "presses and active settling are kept — see filters.py")
args = parser.parse_args()

import imageio.v2 as imageio  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

gen_root = Path(args.gen_root)


def ep_dirs() -> list[Path]:
    if args.episodes:
        return [Path(e) for e in args.episodes]
    data = gen_root / "data"
    dirs = [data / b for b in args.batches] if args.batches else sorted(d for d in data.iterdir() if d.is_dir())
    return [ep for d in dirs for ep in sorted(d.glob("ep_*")) if (ep / "imgs").is_dir()]


def _law(c: dict | None) -> dict | None:
    """A controller block normalized for identity comparison: `control_dt` is DERIVED
    (env.dt x control_period, added to stamps 2026-08-19), so blocks from before and
    after that date describing the same law must compare equal."""
    if not c:
        return c
    return {**c, "leaves": [{k: v for k, v in l.items() if k != "control_dt"}
                            for l in c.get("leaves", [])]}


eps = [read_sim_episode(d, args.cams) for d in ep_dirs()]
if not args.include_failures:
    skipped = sum(not e.success for e in eps)
    eps = [e for e in eps if e.success]
    print(f"[convert] {skipped} failed episodes skipped (--include-failures keeps them)")
if not eps:
    raise SystemExit("nothing to convert")
e0 = eps[0]
views = sorted(e0.videos)
for e in eps:
    if sorted(e.videos) != views or e.size != e0.size or e.arm_joints != e0.arm_joints:
        raise SystemExit(f"{e.ep_dir}: views/size/joints differ from {e0.ep_dir} — bake separately")
    if _law(e.controller) != _law(e0.controller):
        raise SystemExit(f"{e.ep_dir}: recorded control law differs from {e0.ep_dir} — "
                         f"mixed-law pools must be baked separately")
mid_solve_changed = [str(e.ep_dir) for e in eps if e.meta.get("controller_changes")]
if mid_solve_changed:
    print(f"[convert] WARNING: {len(mid_solve_changed)} episodes changed control law MID-SOLVE "
          f"(raw_cmd labels span several laws; stamped block = the final one): "
          f"{mid_solve_changed[:3]}{'…' if len(mid_solve_changed) > 3 else ''}")

rate = args.control_freq or float(e0.fps)  # no --control_freq = native rate (tick per latch)
project = CONVENTIONS[args.control_space]
proj0 = project(e0, rate)
W, H = e0.size
features = {
    **{f"observation.images.{v}": {"dtype": "video", "shape": (H, W, 3),
                                   "names": ["height", "width", "channels"]} for v in views},
    "observation.state": {"dtype": "float32", "shape": (proj0.state.shape[1],),
                          "names": proj0.state_names},
    "action": {"dtype": "float32", "shape": (proj0.action.shape[1],),
               "names": proj0.action_names},
}
if proj0.raw_command is not None:
    features["raw_command"] = {"dtype": "float32", "shape": (proj0.raw_command.shape[1],),
                               "names": [f"cmd_{i}" for i in range(proj0.raw_command.shape[1])]}

root = Path(args.root) if args.root else gen_root / "datasets" / args.repo_id
print(f"[convert] {len(eps)} episodes, control_space={args.control_space} @ {rate:g}Hz, "
      f"views: {', '.join(views)}")
ds = LeRobotDataset.create(args.repo_id, fps=int(rate), features=features, root=root,
                           robot_type=args.robot_type or e0.robot_type, use_videos=True)


def frame_stream(video: Path, wanted: list[int]):
    """Yield the frames at `wanted` (increasing) indices, decoding sequentially."""
    reader = imageio.get_reader(str(video))
    try:
        it, k = iter(reader), -1
        for w in wanted:
            while k < w:
                frame = next(it)
                k += 1
            yield frame
    except StopIteration:
        raise SystemExit(f"{video}: ends before frame {w} — re-render") from None
    finally:
        reader.close()


n_ticks, n_dropped = 0, 0
for e in eps:
    proj = project(e, rate)
    if args.filter_idle:
        from filters import apply_mask, idle_keep_mask

        keep = idle_keep_mask(e, proj)
        n_dropped += int((~keep).sum())
        proj = apply_mask(proj, keep)
    streams = {v: frame_stream(e.videos[v], proj.video_ticks) for v in views}
    task = args.task or e.task
    for i in range(len(proj.video_ticks)):
        frame = {
            **{f"observation.images.{v}": next(streams[v]) for v in views},
            "observation.state": proj.state[i],
            "action": proj.action[i],
            "task": task,
        }
        if proj.raw_command is not None:
            frame["raw_command"] = proj.raw_command[i]
        ds.add_frame(frame)
    ds.save_episode()
    n_ticks += len(proj.video_ticks)
    print(f"[convert] {e.ep_dir.parent.name}/{e.ep_dir.name}: {len(proj.video_ticks)} ticks "
          f"x {len(views)} views", flush=True)

ds.finalize()

meta_dir = root / "meta"
(meta_dir / "modality.json").write_text(json.dumps({
    "state": {n: {"start": s, "end": e} for n, (s, e) in proj0.state_parts.items()},
    "action": {n: {"start": s, "end": e} for n, (s, e) in proj0.action_parts.items()},
    "video": {v: {"original_key": f"observation.images.{v}"} for v in views},
    "annotation": {"annotation.human.task_description": {}},
}, indent=2) + "\n")
(meta_dir / "bake.json").write_text(json.dumps({
    "control_space": args.control_space, "control_freq_hz": rate,
    "origin": sorted({e.origin for e in eps}), "robot_type": e0.robot_type,
    "state_names": proj0.state_names, "action_names": proj0.action_names,
    "state_parts": proj0.state_parts, "action_parts": proj0.action_parts,
    "gripper": "closedness in [0,1]: 0 = fully open, 1 = fully closed",
    "idle_filter": {"enabled": args.filter_idle, "dropped_ticks": n_dropped},
    "raw_command": proj0.raw_command is not None,
    "controller": e0.controller,
    "mid_solve_controller_changes": mid_solve_changed,
    "episodes": [str(e.ep_dir) for e in eps],
    "git_shas": sorted({e.meta.get("git_sha", "?") for e in eps}),
    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}, indent=2) + "\n")
print(f"[convert] DONE: {len(eps)} episodes, {n_ticks} ticks, {len(views)} views, "
      f"{args.control_space} @ {rate:g}Hz -> {root}", flush=True)
