"""lerobot eval plugin for CoSiGen scenes — `--env.type=cosigen`.

Install once into the lerobot venv (auto-discovered by name prefix):

    .venv-lerobot/bin/pip install -e vla/eval/lerobot_env_cosigen   (scripts/bootstrap_lerobot.sh does this)

Then, with `vla/eval/serve.py <sim>` running in the CoSiGen venv:

    lerobot-eval --policy.path=<ckpt> --env.type=cosigen \\
        --eval.n_episodes=20 --eval.batch_size=1 --eval.use_async_envs=false

CosigenEnv is a thin gym client: reset/step travel over a local socket to the
sim server, which owns every eval semantic (executor, cameras, grader success,
warmup, control rate). Obs follow lerobot's classic route ({"pixels": {cam: HWC
uint8}, "agent_pos": vec} -> observation.images.<cam> / observation.state).

The client declares NOTHING about the sim: cameras, image size, state/action
dims and the control rate are all taken from the server's handshake when the
env is created (lerobot builds the policy processors only after make_env, so
the features are bound in time). Any --env.* field you do set explicitly is an
assertion, validated against the server and failing loudly on mismatch. The
only eval knob on this side is --env.max_episode_seconds (the per-episode cap,
in sim seconds, converted with the served rate). The eval condition is pinned
server-side; this client supplies policy, seeds, episode count and horizon.
"""

from __future__ import annotations

import importlib.util
import socket as _socket
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# single-source wire format: vla/eval/protocol.py, loaded by path (this package
# is installed -e from vla/eval/lerobot_env_cosigen, so the repo layout holds)
_spec = importlib.util.spec_from_file_location(
    "cosigen_protocol", Path(__file__).resolve().parents[2] / "protocol.py")
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)


@EnvConfig.register_subclass("cosigen")
@dataclass
class CosigenEnvConfig(EnvConfig):
    """The served sim's interface. Every field below defaults to None = "whatever
    serve.py is running" (bound from the handshake in create_envs); set one
    explicitly only to ASSERT it — a mismatch with the server is a hard error."""

    host: str = "127.0.0.1"
    port: int = 5555
    cameras: tuple[str, ...] | None = None
    observation_height: int | None = None
    observation_width: int | None = None
    state_dim: int | None = None
    action_dim: int | None = None
    fps: int | None = None
    max_episode_seconds: float = 480.0  # per-episode cap in SIM SECONDS; ticks = seconds x served fps

    _INTERFACE = ("cameras", "observation_height", "observation_width", "state_dim", "action_dim", "fps")

    def __post_init__(self):
        pass  # features are bound from the handshake (bind_handshake), not declared here

    @property
    def gym_kwargs(self) -> dict:
        return {}

    @staticmethod
    def _served(hs: dict) -> dict:
        h, w = next(iter(hs["views"].values()))
        return {"cameras": tuple(hs["views"]), "observation_height": int(h), "observation_width": int(w),
                "state_dim": len(hs["state_names"]), "action_dim": int(hs["action_dim"]),
                "fps": int(hs["fps"])}

    def bind_handshake(self, hs: dict) -> None:
        """Fill unset interface fields from the served sim; assert the set ones match.
        Idempotent: a second call (from each CosigenEnv) is pure validation."""
        served = self._served(hs)
        mismatch = {}
        for k, v in served.items():
            mine = getattr(self, k)
            if mine is None:
                setattr(self, k, v)
            elif (sorted(mine) if k == "cameras" else type(v)(mine)) != (sorted(v) if k == "cameras" else v):
                mismatch[k] = {"config": mine, "server": v}
        if mismatch:
            raise RuntimeError(f"served sim != env config: {mismatch} (server source: {hs.get('source')})")
        self.features.clear()
        self.features_map.clear()
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))
        self.features_map.update({ACTION: ACTION, "agent_pos": OBS_STATE})
        for cam in self.cameras:
            self.features[f"pixels/{cam}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3))
            self.features_map[f"pixels/{cam}"] = f"{OBS_IMAGES}.{cam}"

    def handshake(self) -> dict:
        """One round-trip to the server; the connection is closed again (sim stays warm)."""
        with _socket.create_connection((self.host, self.port)) as sock:
            protocol.send_msg(sock, {"cmd": "handshake"})
            hs, _ = protocol.recv_msg(sock)
            if "error" in hs:
                raise RuntimeError(f"sim server error on handshake:\n{hs['error']}")
            try:
                protocol.send_msg(sock, {"cmd": "close"})
            except OSError:
                pass
        return hs

    def create_envs(self, n_envs: int = 1, use_async_envs: bool = False, **kwargs):
        if n_envs != 1 or use_async_envs:
            raise ValueError("cosigen serves ONE env over one socket (v1): run with "
                             "--eval.batch_size=1 --eval.use_async_envs=false")
        self.bind_handshake(self.handshake())  # before lerobot reads cfg.features
        from gymnasium.vector import SyncVectorEnv

        return {"cosigen": {0: SyncVectorEnv([lambda: CosigenEnv(self)])}}


class CosigenEnv(gym.Env):
    """Socket client for one served EvalSim episode stream."""

    metadata = {"render_modes": ["rgb_array"]}  # render_fps is filled from the handshake

    def __init__(self, cfg: CosigenEnvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._sock = _socket.create_connection((cfg.host, cfg.port))
        hs = self._rpc({"cmd": "handshake"})[0]
        cfg.bind_handshake(hs)  # fills the interface on first contact, validates afterwards
        self.task = self.task_description = hs["task"]
        self._max_episode_steps = int(round(cfg.max_episode_seconds * cfg.fps))
        self.metadata = {**type(self).metadata, "render_fps": int(cfg.fps)}
        h, w = cfg.observation_height, cfg.observation_width
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({cam: spaces.Box(0, 255, (h, w, 3), np.uint8)
                                   for cam in cfg.cameras}),
            "agent_pos": spaces.Box(-np.inf, np.inf, (cfg.state_dim,), np.float64),
        })
        self.action_space = spaces.Box(-1e3, 1e3, (cfg.action_dim,), np.float32)
        self._last_front: np.ndarray | None = None

    def _rpc(self, header: dict, arrays: dict | None = None):
        protocol.send_msg(self._sock, header, arrays)
        reply, reply_arrays = protocol.recv_msg(self._sock)
        if "error" in reply:
            raise RuntimeError(f"sim server error:\n{reply['error']}")
        return reply, reply_arrays

    def _obs(self, arrays: dict) -> dict:
        pixels = {cam: arrays[f"img/{cam}"][0] for cam in self.cfg.cameras}
        self._last_front = pixels[self.cfg.cameras[0]]
        return {"pixels": pixels, "agent_pos": arrays["state"][0].astype(np.float64)}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        header, arrays = self._rpc({"cmd": "reset",
                                    "seed": int(seed) if seed is not None else None})
        return self._obs(arrays), {"is_success": False,
                                   "progress": self._progress(arrays)}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(1, -1)
        header, arrays = self._rpc({"cmd": "step"}, {"action": a})
        success = bool(header["is_success"][0])
        progress = self._progress(arrays)
        # reward = the grader's rubric progress, so lerobot's per-episode
        # max_reward IS the score (generation's own `score` scale) and
        # sum_reward the area under the progress curve
        return self._obs(arrays), progress, success, False, \
            {"is_success": success, "progress": progress}

    @staticmethod
    def _progress(arrays: dict) -> float:
        p = arrays.get("progress")
        return float(p[0]) if p is not None else 0.0

    def render(self):
        return self._last_front

    def close(self):
        try:
            protocol.send_msg(self._sock, {"cmd": "close"})
        except OSError:
            pass
        self._sock.close()
