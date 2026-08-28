"""rollout_chunked — closed-loop eval client that executes whole predicted action CHUNKS.

Drives the sim served by vla/eval/serve.py (through the lerobot_env_cosigen plugin's
CosigenEnv) with a LeRobot policy, one predicted chunk at a time:

    obs -> preprocessor -> policy.predict_action_chunk -> postprocessor (whole chunk)
        -> execute the first `--exec-steps` actions -> repeat

Why not `lerobot-eval`: it draws actions through `policy.select_action`, which GR00T
REFUSES for state-relative (delta) action policies — a cached relative chunk would be
decoded against a newer observation state. Postprocessing the full chunk right after
prediction anchors every action to the state it was predicted from (the relative ->
absolute step reads the state the preprocessor cached), so `--exec-steps` below the
chunk length is a legitimate re-plan horizon here — each new chunk re-anchors to the
fresh state — where truncating an ABSOLUTE-target chunk merely creeps (STAGE3.md).
Works unchanged for absolute-action policies (no relative steps in their pipelines).

Outputs (in --out): eval_info.json (lerobot-eval-shaped: per_task[0].metrics with
sum_rewards / max_rewards / successes / video_paths + overall), videos/episode_<k>.mp4
(front camera), trace/episode_<k>.npz (per-step action, state, progress).

    .venv-lerobot/bin/python vla/eval/rollout_chunked.py \\
        --policy-path experiments/<run>/checkpoints/<step>/pretrained_model \\
        --dataset-root vla/datasets/pc_ram_ik --repo-id cosigen/pc_ram_ik \\
        --port 5555 --fps 48 --episode-length 5760 --episodes 10 --exec-steps 40 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--policy-path", required=True, dest="policy_path")
parser.add_argument("--dataset-root", required=True, dest="dataset_root")
parser.add_argument("--repo-id", required=True, dest="repo_id")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--fps", type=int, default=48)
parser.add_argument("--episode-length", type=int, default=5760, dest="episode_length")
parser.add_argument("--cameras", nargs="*", default=["front", "wrist"])
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--seed0", type=int, default=1000)
parser.add_argument("--exec-steps", type=int, default=0, dest="exec_steps",
                    help="actions executed per predicted chunk (0 = the whole chunk)")
parser.add_argument("--video-stride", type=int, default=1, dest="video_stride",
                    help="record every Nth tick to the episode video")
parser.add_argument("--stop-on-success", action="store_true", dest="stop_on_success")
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

import sys  # noqa: E402

# Run as a script, sys.path[0] is THIS directory, where the plugin's PROJECT dir
# (vla/eval/lerobot_env_cosigen/, holding pyproject.toml) would shadow the installed package as
# an empty namespace package ("cannot import name 'CosigenEnv' ... (unknown location)").
_here = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != _here]

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot_env_cosigen import CosigenEnv, CosigenEnvConfig  # noqa: E402

out = Path(args.out); (out / "videos").mkdir(parents=True, exist_ok=True); (out / "trace").mkdir(exist_ok=True)

# ---- policy (same construction path as offline_eval / the bulb chunk server) ----------------------
ckpt = Path(args.policy_path).resolve()
meta = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav", episodes=[0]).meta
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.pretrained_path = str(ckpt)
policy = make_policy(cfg=cfg, ds_meta=meta)
policy.eval()
pre, post = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=str(ckpt),
    preprocessor_overrides={"device_processor": {"device": args.device}},
)
relative = bool(getattr(cfg, "use_relative_actions", False))
chunk_len = int(getattr(cfg, "n_action_steps", 1))
exec_steps = args.exec_steps or chunk_len
print(f"[chunked] policy {ckpt.parents[2].name}/{ckpt.parent.name}: chunk={chunk_len} "
      f"exec_steps={exec_steps} relative_actions={relative} "
      f"exclude={getattr(cfg, 'relative_exclude_joints', None)}", flush=True)

# ---- env (the plugin's gym env, unbatched; we add the batch dim ourselves) ----------------------
env_cfg = CosigenEnvConfig(host=args.host, port=args.port, cameras=tuple(args.cameras),
                           fps=args.fps, episode_length=args.episode_length)
env = CosigenEnv(env_cfg)
task = env.task
print(f"[chunked] sim ready: task '{task[:60]}…' cams={args.cameras} fps={args.fps} cap={args.episode_length}", flush=True)


def to_batch(obs: dict) -> dict:
    """Plugin obs -> lerobot observation batch (B=1): pixels (B,H,W,3) u8, agent_pos (B,D)."""
    batched = {"pixels": {cam: np.asarray(img)[None] for cam, img in obs["pixels"].items()},
               "agent_pos": np.asarray(obs["agent_pos"], dtype=np.float32)[None]}
    observation = preprocess_observation(batched)
    observation["task"] = [task]
    return observation


def predict_chunk(obs: dict) -> np.ndarray:
    """One re-plan: (chunk_len, A) ABSOLUTE actions, anchored to this observation's state."""
    observation = pre(to_batch(obs))
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(observation)  # (1, T, A) model space
    chunk = post(chunk)  # unnormalize (+ relative -> absolute against the cached state) -> cpu
    return chunk[0].float().numpy()


def write_video(frames: list, path: Path, fps: int) -> None:
    import av

    if not frames:
        return
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "23", "preset": "medium"}
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


sum_rewards, max_rewards, successes, video_paths = [], [], [], []
t_start = time.time()
for ep in range(args.episodes):
    seed = args.seed0 + ep
    policy.reset()
    obs, info = env.reset(seed=seed)
    frames = {cam: [np.asarray(obs["pixels"][cam])] for cam in args.cameras}  # every served view
    actions, states, progress = [], [], [float(info.get("progress", 0.0))]
    success, ticks, queue = False, 0, []
    t0 = time.time()
    while ticks < args.episode_length:
        if not queue:
            queue = list(predict_chunk(obs)[:exec_steps])
        a = queue.pop(0)
        obs, reward, terminated, truncated, info = env.step(a)
        ticks += 1
        actions.append(a); states.append(np.asarray(obs["agent_pos"], dtype=np.float32)); progress.append(float(reward))
        if ticks % args.video_stride == 0:
            for cam in args.cameras:
                frames[cam].append(np.asarray(obs["pixels"][cam]))
        if info.get("is_success") or terminated:
            success = True
            if args.stop_on_success:
                break
    video = out / "videos" / f"episode_{ep}.mp4"  # first camera keeps the legacy name
    write_video(frames[args.cameras[0]], video, max(1, args.fps // args.video_stride))
    for cam in args.cameras[1:]:
        write_video(frames[cam], out / "videos" / f"episode_{ep}_{cam}.mp4",
                    max(1, args.fps // args.video_stride))
    np.savez(out / "trace" / f"episode_{ep}.npz", action=np.stack(actions), state=np.stack(states),
             progress=np.array(progress, dtype=np.float32), seed=seed)
    sum_rewards.append(float(np.sum(progress))); max_rewards.append(float(np.max(progress)))
    successes.append(bool(success)); video_paths.append(str(video))
    print(f"[chunked] episode {ep} seed {seed}: {'SUCCESS' if success else 'fail'} | ticks {ticks} | "
          f"max progress {max(progress):.3f} | {time.time() - t0:.0f}s wall | "
          f"running {sum(successes)}/{ep + 1}", flush=True)

overall = {"pc_success": 100.0 * float(np.mean(successes)), "avg_max_reward": float(np.mean(max_rewards)),
           "avg_sum_reward": float(np.mean(sum_rewards)), "n_episodes": args.episodes,
           "eval_s": time.time() - t_start, "exec_steps": exec_steps, "chunk": chunk_len,
           "relative_actions": relative, "policy_path": str(ckpt)}
(out / "eval_info.json").write_text(json.dumps({
    "per_task": [{"task_group": "cosigen", "task_id": 0, "metrics": {
        "sum_rewards": sum_rewards, "max_rewards": max_rewards, "successes": successes,
        "video_paths": video_paths}}],
    "overall": overall}, indent=1))
print(f"[chunked] DONE: {sum(successes)}/{args.episodes} success, avg max progress "
      f"{np.mean(max_rewards):.3f} -> {out}", flush=True)
env.close()
