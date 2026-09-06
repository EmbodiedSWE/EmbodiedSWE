"""Bake canonical episodes into a LeRobotDataset (+ dialect metadata; no Isaac).

Run with lerobot's OWN venv:

    .venv-lerobot/bin/python vla/convert/convert.py \\
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
parser.add_argument("--vcodec", default="h264",
                    help="dataset video codec (lerobot names: h264, hevc, libsvtav1/av1, auto=hardware). "
                         "Default h264: decodes everywhere (torchcodec/pyav/decord, every NVDEC "
                         "generation); lerobot's own default is libsvtav1")
parser.add_argument("--crf", type=float, default=23,
                    help="video quality (codec-specific; x264: 18 ≈ visually lossless, 23 = standard)")
parser.add_argument("--gop-seconds", type=float, default=0.25, dest="gop_seconds",
                    help="keyframe interval in SECONDS (g = round(fps * this), min 1): bounds the "
                         "worst-case random-access seek at a fixed wall-clock cost regardless of the "
                         "control rate. 0.25 s → g=15 at 60 Hz, g=4 at 15 Hz. lerobot's default is "
                         "g=2 (fastest seeks, ~4-5x the bytes)")
parser.add_argument("--pix-fmt", default="yuv420p", dest="pix_fmt", help="video pixel format")
parser.add_argument("--preset", default=None,
                    help="encoder speed/quality preset (codec-specific; default: codec's own)")
parser.add_argument("--max-video-file-seconds", type=float, default=800.0, dest="max_video_file_seconds",
                    help="cap on the DURATION of each packed video file (default 800 s: 25 % margin under the 1024 s cliff, must exceed one episode). LeRobot only "
                         "caps by MB and stores timestamps as float32: past 1024 s into a file their "
                         "step (1.2e-4 s) exceeds its default tolerance_s=1e-4 and training dies with "
                         "FrameTimestampError. Bitrate is measured on the first episode and the MB cap "
                         "derived from it; a final ffprobe pass fails the bake if any file exceeds 1000 s")
parser.add_argument("--workers", default="1",
                    help="parallel bake: N worker processes each bake a temporary shard, then the shards are "
                         "merged (lerobot aggregate_datasets) into --root with the video-duration cap applied "
                         "and verified; temp shards are deleted. 'auto' = SLURM_CPUS_PER_TASK or os.cpu_count(). "
                         "Default 1 = sequential (no shards, no merge)")
parser.add_argument("--filter-idle", action="store_true", dest="filter_idle",
                    help="drop dead ticks (robot AND objects static AND no command intent); "
                         "presses and active settling are kept — see filters.py")
args = parser.parse_args()

import imageio.v2 as imageio  # noqa: E402

try:  # lerobot with the encoder config object: full control (codec, crf, GOP, pix_fmt, preset)
    from lerobot.configs.video import RGBEncoderConfig  # noqa: E402
except ImportError:  # lerobot 0.4.x: LeRobotDataset.create() takes only `vcodec`
    RGBEncoderConfig = None
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

gen_root = Path(args.gen_root)

VIDEO_FILE_HARD_LIMIT_S = 1000.0  # float32 timestamp step exceeds lerobot's 1e-4 tolerance past 1024 s


def video_durations_s(root: Path) -> dict[Path, float]:
    """Container duration of every packed video file (ffprobe); {} if ffprobe is unavailable."""
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        print("[convert] WARNING: ffprobe not found — video file durations not verified", flush=True)
        return {}
    out = {}
    for f in sorted(root.glob("videos/**/*.mp4")):
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(f)], capture_output=True, text=True)
        out[f] = float(r.stdout.strip() or "nan")
    return out




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
def _n_workers(spec: str) -> int:
    if spec != "auto":
        return max(1, int(spec))
    import os
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1))


def _child_argv(episodes: list[Path], shard_root: Path) -> list[str]:
    """This invocation's options, re-targeted at one shard: explicit episode list, own --root,
    --workers 1. Rebuilt from the parsed namespace so new options are forwarded automatically."""
    argv = [sys.executable, str(Path(__file__).resolve()), args.gen_root]
    skip = {"gen_root", "episodes", "batches", "root", "workers"}
    for act in parser._actions:
        if not act.option_strings or act.dest in skip or act.dest == "help":
            continue
        val = getattr(args, act.dest)
        if val is None or val == act.default and not isinstance(act, argparse._StoreTrueAction):
            continue
        if isinstance(act, argparse._StoreTrueAction):
            if val:
                argv.append(act.option_strings[0])
        elif isinstance(val, (list, tuple)):
            if val:
                argv += [act.option_strings[0], *map(str, val)]
        else:
            argv += [act.option_strings[0], str(val)]
    argv += ["--root", str(shard_root), "--workers", "1", "--episodes", *(str(e.ep_dir) for e in episodes)]
    return argv


def _merge_bakes(bakes: list[dict], extra: dict) -> dict:
    merged = dict(bakes[0])
    merged["episodes"] = [e for b in bakes for e in b.get("episodes", [])]
    merged["git_shas"] = sorted({g for b in bakes for g in b.get("git_shas", [])})
    merged["mid_solve_controller_changes"] = [e for b in bakes for e in b.get("mid_solve_controller_changes", [])]
    merged["idle_filter"] = {**bakes[0]["idle_filter"],
                             "dropped_ticks": sum(b["idle_filter"]["dropped_ticks"] for b in bakes)}
    merged.update(extra)
    return merged


n_workers = min(_n_workers(args.workers), len(eps))
if n_workers > 1:
    import shutil
    import subprocess

    from lerobot.datasets.aggregate import aggregate_datasets

    shards_root = root.parent / f"{root.name}.shards"
    if root.exists() or shards_root.exists():
        raise SystemExit(f"{root} / {shards_root} already exist — remove them first")
    # balance by episode length (ticks), largest first
    order = sorted(eps, key=lambda e: -len(e.frame_indices))
    buckets: list[list] = [[] for _ in range(n_workers)]
    for i, e in enumerate(order):
        buckets[i % n_workers].append(e)
    shard_roots = [shards_root / f"w{i:02d}" for i in range(n_workers)]
    print(f"[convert] {n_workers} workers x ~{len(eps) / n_workers:.1f} episodes -> temp shards under "
          f"{shards_root}", flush=True)
    procs = [subprocess.Popen(_child_argv(b, r)) for b, r in zip(buckets, shard_roots)]
    codes = [pr.wait() for pr in procs]
    if any(codes):
        raise SystemExit(f"{sum(bool(c) for c in codes)} worker(s) failed (exit codes {codes}); "
                         f"shards kept under {shards_root} for inspection")
    # merge: derive lerobot's MB cap from the shards' real bitrate so files stay under the duration cap
    tot_bytes: dict[str, int] = {}
    tot_s = 0.0
    for r in shard_roots:
        info = json.loads((r / "meta/info.json").read_text())
        tot_s += info["total_frames"] / info["fps"]
        for f in (r / "videos").rglob("*.mp4"):
            tot_bytes[f.parent.parent.name] = tot_bytes.get(f.parent.parent.name, 0) + f.stat().st_size
    mb_cap = int(max(1, min(200, args.max_video_file_seconds * max(tot_bytes.values()) / 1e6 / tot_s)))
    print(f"[convert] merging {n_workers} shards -> {root} (video files capped at {mb_cap} MB "
          f"~ {args.max_video_file_seconds:g} s)", flush=True)
    aggregate_datasets(repo_ids=[args.repo_id] * n_workers, aggr_repo_id=args.repo_id,
                       roots=shard_roots, aggr_root=root, video_files_size_in_mb=mb_cap)
    durations = video_durations_s(root)
    longest = max(durations.values()) if durations else None
    if durations:
        print(f"[convert] {len(durations)} video files, longest {longest:.0f} s "
              f"(cap {args.max_video_file_seconds:g} s)", flush=True)
        if longest > VIDEO_FILE_HARD_LIMIT_S:
            raise SystemExit(f"merged video file {longest:.0f} s > {VIDEO_FILE_HARD_LIMIT_S:g} s — re-run with a "
                             f"smaller --max-video-file-seconds; shards kept under {shards_root}")
    meta_dir = root / "meta"
    shutil.copy(shard_roots[0] / "meta/modality.json", meta_dir / "modality.json")
    bakes = [json.loads((r / "meta/bake.json").read_text()) for r in shard_roots]
    (meta_dir / "bake.json").write_text(json.dumps(_merge_bakes(bakes, {
        "video_files": {"max_seconds": args.max_video_file_seconds, "size_mb": mb_cap, "longest_s": longest},
        "workers": n_workers,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }), indent=2) + "\n")
    shutil.rmtree(shards_root)
    n_ticks = sum(json.loads((r / "meta/info.json").read_text())["total_frames"] for r in [root])
    print(f"[convert] DONE: {len(eps)} episodes, {n_ticks} ticks, {len(views)} views, "
          f"{args.control_space} @ {rate:g}Hz -> {root} ({n_workers} workers)", flush=True)
    sys.exit(0)

gop = max(1, round(rate * args.gop_seconds))
if RGBEncoderConfig is not None:
    encoder = RGBEncoderConfig(vcodec=args.vcodec, pix_fmt=args.pix_fmt, crf=args.crf,
                               g=gop, preset=args.preset)
    create_kwargs = {"rgb_encoder": encoder}
    encoder_meta = {"vcodec": encoder.vcodec, "pix_fmt": encoder.pix_fmt, "crf": encoder.crf,
                    "g": encoder.g, "gop_seconds": args.gop_seconds, "preset": encoder.preset}
    print(f"[convert] video: {encoder.vcodec} crf={encoder.crf:g} g={encoder.g} "
          f"({args.gop_seconds:g}s keyframe interval) {encoder.pix_fmt}", flush=True)
else:
    # lerobot 0.4.x applies its own crf/GOP/pix_fmt defaults for the chosen codec
    create_kwargs = {"vcodec": args.vcodec}
    encoder_meta = {"vcodec": args.vcodec, "pix_fmt": None, "crf": None, "g": None,
                    "gop_seconds": None, "preset": None,
                    "note": "lerobot without configs.video: only the codec was set"}
    print(f"[convert] video: {args.vcodec} (lerobot {'0.4.x'}: codec only; crf/GOP = lerobot "
          f"defaults — --crf/--gop-seconds/--pix-fmt/--preset ignored)", flush=True)
ds = LeRobotDataset.create(args.repo_id, fps=int(rate), features=features, root=root,
                           robot_type=args.robot_type or e0.robot_type, use_videos=True,
                           image_writer_threads=4 * len(views),  # async PNG staging (default is synchronous)
                           **create_kwargs)


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


def fit_video_file_size_mb(ds: LeRobotDataset, first_ep_seconds: float, max_seconds: float) -> int | None:
    """Derive the MB cap that keeps every video file under max_seconds, from the first
    episode's measured bitrate (per view; the fattest view decides). lerobot re-reads the
    cap at every save, so setting it after episode 0 governs the packing from episode 1 on."""
    rates = []
    for key in ds.meta.video_keys:
        f = ds.root / ds.meta.video_path.format(video_key=key, chunk_index=0, file_index=0)
        if f.is_file() and first_ep_seconds > 0:
            rates.append(f.stat().st_size / 1e6 / first_ep_seconds)
    if not rates:
        return None
    mb = int(max(1, min(ds.meta.video_files_size_in_mb, max_seconds * max(rates))))
    ds.meta.info.video_files_size_in_mb = mb
    return mb


n_ticks, n_dropped = 0, 0
video_file_size_mb = None
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
    if video_file_size_mb is None:
        video_file_size_mb = fit_video_file_size_mb(ds, len(proj.video_ticks) / rate,
                                                    args.max_video_file_seconds)
        if video_file_size_mb is not None:
            print(f"[convert] video files capped at {video_file_size_mb} MB "
                  f"(~{args.max_video_file_seconds:g} s at the measured bitrate)", flush=True)
    n_ticks += len(proj.video_ticks)
    print(f"[convert] {e.ep_dir.parent.name}/{e.ep_dir.name}: {len(proj.video_ticks)} ticks "
          f"x {len(views)} views", flush=True)

ds.finalize()

durations = video_durations_s(root)
if durations:
    longest = max(durations.values())
    print(f"[convert] {len(durations)} video files, longest {longest:.0f} s "
          f"(cap {args.max_video_file_seconds:g} s)", flush=True)
    too_long = {f: d for f, d in durations.items() if d > VIDEO_FILE_HARD_LIMIT_S}
    if too_long:
        for f, d in too_long.items():
            print(f"[convert] ERROR video file {f.relative_to(root)} is {d:.0f} s > {VIDEO_FILE_HARD_LIMIT_S:g} s",
                  flush=True)
        raise SystemExit("video files too long for lerobot's float32 timestamps + default tolerance_s; "
                         "re-run with a smaller --max-video-file-seconds (bitrate rose after the first episode)")

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
    "video_encoder": encoder_meta,
    "video_files": {"max_seconds": args.max_video_file_seconds,
                    "size_mb": ds.meta.video_files_size_in_mb,
                    "longest_s": max(durations.values()) if durations else None},
    "workers": 1,
    "raw_command": proj0.raw_command is not None,
    "controller": e0.controller,
    "mid_solve_controller_changes": mid_solve_changed,
    "episodes": [str(e.ep_dir) for e in eps],
    "git_shas": sorted({e.meta.get("git_sha", "?") for e in eps}),
    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}, indent=2) + "\n")
print(f"[convert] DONE: {len(eps)} episodes, {n_ticks} ticks, {len(views)} views, "
      f"{args.control_space} @ {rate:g}Hz -> {root}", flush=True)
