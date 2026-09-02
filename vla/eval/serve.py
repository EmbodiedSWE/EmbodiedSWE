"""serve — the eval sim behind a socket, for the closed-loop VLA eval (boots Isaac).

    .venv/bin/python vla/eval/serve.py bulb_jointpd_60hz --headless

    # then, in the lerobot venv (plugin installed -e once):
    lerobot-eval --policy.path=<ckpt> --env.type=cosigen \\
        --eval.n_episodes=20 --eval.batch_size=1 --eval.use_async_envs=false

A thin loop over load_sim/EvalSim: boot once, then answer handshake / reset /
step over a local TCP socket (protocol.py framing). All eval semantics live in
sim.py — the executor, cameras, grader success, warmup, grip margin; this file
only makes EvalSim reachable from another process. The eval CONDITION is pinned
here at launch (source + overrides); the client supplies only policy, seeds and
episode count, so a result can never half-override the sim it claims it ran on.

The server stays warm across client runs (outer accept loop): iterate on the
policy side without ever re-paying the Isaac boot. Sim time freezes while the
policy thinks (blocking socket) = instant-inference semantics; action chunking
is the policy's own affair (lerobot's select_action queue).

--init dataset starts each episode from a recorded state instead of scene
randomization: episode index = seed %% len(episodes), so lerobot's seeding
stays meaningful and runs reproduce.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description="serve an eval sim for lerobot-eval")
parser.add_argument("source", help="registered sim name | ENVS preset | bake.json path")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--num_envs", type=int, default=1, help="v1: the client consumes slot 0")
parser.add_argument("--init", choices=("scene", "dataset"), default="scene",
                    help="episode starts: the scene's own randomization, or recorded states")
parser.add_argument("--init-batch", default="", dest="init_batch",
                    help="ep_* dir for --init dataset (episode = seed %% n)")
parser.add_argument("--grip-margin", type=float, default=None, dest="grip_margin",
                    help="SimSpec.grip_margin override (echoed into provenance)")
parser.add_argument("--record-dir", default="", dest="record_dir",
                    help="where rollouts are recorded (default vla/eval/_out/serve_<source>/rollouts). "
                         "Every served episode is written on its reset/close as ep_NNNN/{traj.npz, "
                         "meta.json} in the generation layout (states per latch + the policy's action "
                         "rows), so data_engine/scripts/render.py <gen_root> --episodes … renders it at "
                         "the control rate and vla/eval/replay_actions.py can re-drive it")
parser.add_argument("--no-record-states", dest="record_states", action="store_false",
                    help="do not record served rollouts")
parser.add_argument("--record-cell", default="scene_0/eval", dest="record_cell",
                    help="the `cell` stamped on the rollout batch meta.json — render.py resolves its "
                         "scene part against <gen_root>/scenes/<scene>/")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

import socket  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol  # noqa: E402
from sim import _JOINT_SPACES, load_sim  # noqa: E402

overrides = {} if args.grip_margin is None else {"grip_margin": args.grip_margin}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sim = load_sim(args.source, num_envs=args.num_envs, device=device, **overrides)
action_dim = len(sim.arm_names) + 1 if sim.spec.control_space in _JOINT_SPACES \
    else sim.env.robot.action_dim

episodes: list[Path] = []
if args.init == "dataset":
    episodes = sorted(p for p in Path(args.init_batch).glob("ep_*") if (p / "traj.npz").is_file())
    if not episodes:
        raise SystemExit(f"--init dataset: no episodes under '{args.init_batch}'")

W, H = sim.spec.size
HANDSHAKE = {
    "task": sim.task, "rate_hz": sim.rate_hz, "fps": int(round(sim.rate_hz)),
    "state_names": sim.state_names, "action_dim": int(action_dim),
    "views": {n: [H, W] for n in sim.sensors},
    "control_space": sim.spec.control_space, "source": str(args.source),
    "init": args.init, "num_envs": sim.env.num_envs,
    "grip_margin": sim.spec.grip_margin,
}


# ----- rollout recorder ---------------------------------------------------------------------------
import json  # noqa: E402
import re  # noqa: E402

from engine.generation import _flat as _flat_states  # noqa: E402  (the data recorder's flatten)


class RolloutRecorder:
    """States per latch (+ the policy's action rows) of every served episode, flushed on the next
    reset / client close as <dir>/ep_NNNN/{traj.npz, meta.json} — the generation layout, so the
    render and replay tools accept a policy rollout exactly like a demo."""

    def __init__(self, root: Path, cell: str) -> None:
        self.root, self.cell, self.n = root, cell, 0
        self.cur: dict | None = None
        root.mkdir(parents=True, exist_ok=True)
        (root / "meta.json").write_text(json.dumps({
            "cell": cell, "origin": "vla/eval/serve.py rollouts", "source": str(args.source),
            "control_space": sim.spec.control_space, "rate_hz": sim.rate_hz,
            "init": args.init, "grip_margin": sim.spec.grip_margin}, indent=2) + "\n")

    def start(self, seed, init_ep: Path | None, obs: dict) -> None:
        self.flush(obs=None)
        self.cur = {"seed": seed, "init": str(init_ep) if init_ep else None,
                    "states": [_flat_states(sim.env.get_states())], "actions": [], "last": obs}

    def step(self, action, obs: dict) -> None:
        if self.cur is None:
            return
        self.cur["actions"].append(np.asarray(action, dtype=np.float32).reshape(sim.env.num_envs, -1))
        self.cur["states"].append(_flat_states(sim.env.get_states()))
        self.cur["last"] = obs

    def flush(self, obs: dict | None = None) -> None:
        c, self.cur = self.cur, None
        if c is None or len(c["states"]) < 2:
            return
        last = obs or c["last"]
        keys = [k for k in c["states"][0] if not k.startswith("robot/controller")]
        stacked = {k: np.stack([r[k].numpy() for r in c["states"]]) for k in keys}  # (T, E, …)
        act = np.stack(c["actions"])                                                # (T-1, E, A)
        act = np.concatenate([act, act[-1:]], axis=0)                              # hold on the last row
        decim = max(1, round(1.0 / (sim.env.dt * sim.rate_hz)))
        for e in range(sim.env.num_envs):
            d = self.root / f"ep_{self.n:04d}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "traj.npz", action=act[:, e], **{k: v[:, e] for k, v in stacked.items()})
            (d / "meta.json").write_text(json.dumps({
                "episode": self.n, "env_index": e, "seed": c["seed"], "init_episode": c["init"],
                "steps": int(len(c["states"])), "success": bool(last["success"][e]),
                "progress_final": float(last["progress"][e]),
                "sim_dt": float(sim.env.dt), "decimation": int(decim),
                "control_space": sim.spec.control_space, "source": str(args.source),
                "preset": sim.spec.preset, "cell": self.cell, "origin": "policy rollout (serve.py)",
            }, indent=2) + "\n")
            self.n += 1
        print(f"[serve] recorded rollout -> {d}", flush=True)


recorder = None
if args.record_states:
    rdir = Path(args.record_dir) if args.record_dir else \
        Path(__file__).parent / "_out" / ("serve_" + re.sub(r"[^\w.-]+", "_", str(args.source)).strip("_")) / "rollouts"
    recorder = RolloutRecorder(rdir, args.record_cell)
    print(f"[serve] recording rollouts under {rdir}", flush=True)


def reply_obs(conn, obs: dict) -> None:
    arrays = {"state": obs["state"].astype(np.float32),
              "progress": obs["progress"].astype(np.float32)}
    for view, imgs in obs["images"].items():
        arrays[f"img/{view}"] = imgs
    protocol.send_msg(conn, {"is_success": obs["success"].tolist()}, arrays)


srv = socket.create_server((args.host, args.port))
print(f"[serve] {args.source} ready on {args.host}:{args.port} — task: {sim.task[:60]}…", flush=True)

try:
    while True:
        conn, addr = srv.accept()
        print(f"[serve] client {addr}", flush=True)
        try:
            while True:
                header, arrays = protocol.recv_msg(conn)
                cmd = header.get("cmd")
                try:
                    if cmd == "handshake":
                        protocol.send_msg(conn, HANDSHAKE)
                    elif cmd == "reset":
                        seed = header.get("seed")
                        if episodes:
                            ep = episodes[(seed or 0) % len(episodes)]
                            print(f"[serve] reset seed={seed} -> {ep.parent.name}/{ep.name}",
                                  flush=True)
                            obs = sim.init_from_episode(ep)
                        else:
                            ep, obs = None, sim.reset(seed=seed)
                        if recorder:
                            recorder.start(seed, ep, obs)
                        reply_obs(conn, obs)
                    elif cmd == "step":
                        obs = sim.step(arrays["action"])
                        if recorder:
                            recorder.step(arrays["action"], obs)
                        reply_obs(conn, obs)
                    elif cmd == "close":
                        if recorder:
                            recorder.flush()
                        break
                    else:
                        protocol.send_msg(conn, {"error": f"unknown cmd {cmd!r}"})
                except Exception:  # noqa: BLE001 — report to the client, keep serving
                    err = traceback.format_exc()
                    print(f"[serve] ERROR on {cmd}:\n{err}", flush=True)
                    protocol.send_msg(conn, {"error": err})
        except (ConnectionError, OSError):
            pass  # client went away; wait for the next one
        finally:
            if recorder:
                recorder.flush()
            conn.close()
            print("[serve] client disconnected — sim stays warm", flush=True)
except KeyboardInterrupt:
    print("[serve] shutting down", flush=True)

import os  # noqa: E402
import threading  # noqa: E402

watchdog = threading.Timer(10.0, lambda: os._exit(0))
watchdog.daemon = True
watchdog.start()
app.close()
os._exit(0)
