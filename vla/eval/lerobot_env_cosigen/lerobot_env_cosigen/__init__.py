"""lerobot eval plugin for CoSiGen scenes — `--env.type=cosigen`.

Install once into the lerobot venv (auto-discovered by name prefix):

    ~/Documents/Research/lerobot/.venv/bin/pip install -e vla/eval/lerobot_env_cosigen

Then, with `vla/eval/serve.py <sim>` running in the CoSiGen venv:

    lerobot-eval --policy.path=<ckpt> --env.type=cosigen \\
        --eval.n_episodes=20 --eval.batch_size=1 --eval.use_async_envs=false

CosigenEnv is a thin gym client: reset/step travel over a local socket to the
sim server, which owns every eval semantic (executor, cameras, grader success,
warmup). Obs follow lerobot's classic route ({"pixels": {cam: HWC uint8},
"agent_pos": vec} -> observation.images.<cam> / observation.state); the
handshake validates the config's dims against the served sim and fails loudly
on mismatch. The eval condition is pinned server-side; this client only
supplies policy, seeds and episode count.
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
    """Declares the served sim's interface; defaults = the bulb 60 Hz bake."""

    host: str = "127.0.0.1"
    port: int = 5555
    cameras: tuple[str, ...] = ("front", "wrist")
    observation_height: int = 480
    observation_width: int = 640
    state_dim: int = 8
    action_dim: int = 8
    fps: int = 60
    episode_length: int = 10800  # max ticks per episode (@60 Hz: 180 s, the demo horizon)

    def __post_init__(self):
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))
        self.features_map.update({ACTION: ACTION, "agent_pos": OBS_STATE})
        for cam in self.cameras:
            self.features[f"pixels/{cam}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3))
            self.features_map[f"pixels/{cam}"] = f"{OBS_IMAGES}.{cam}"

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def create_envs(self, n_envs: int = 1, use_async_envs: bool = False, **kwargs):
        if n_envs != 1 or use_async_envs:
            raise ValueError("cosigen serves ONE env over one socket (v1): run with "
                             "--eval.batch_size=1 --eval.use_async_envs=false")
        from gymnasium.vector import SyncVectorEnv

        return {"cosigen": {0: SyncVectorEnv([lambda: CosigenEnv(self)])}}


class CosigenEnv(gym.Env):
    """Socket client for one served EvalSim episode stream."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, cfg: CosigenEnvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._sock = _socket.create_connection((cfg.host, cfg.port))
        hs = self._rpc({"cmd": "handshake"})[0]
        self._check(hs)
        self.task = self.task_description = hs["task"]
        self._max_episode_steps = cfg.episode_length
        self.metadata = {**type(self).metadata, "render_fps": int(hs["fps"])}
        h, w = cfg.observation_height, cfg.observation_width
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({cam: spaces.Box(0, 255, (h, w, 3), np.uint8)
                                   for cam in cfg.cameras}),
            "agent_pos": spaces.Box(-np.inf, np.inf, (cfg.state_dim,), np.float64),
        })
        self.action_space = spaces.Box(-1e3, 1e3, (cfg.action_dim,), np.float32)
        self._last_front: np.ndarray | None = None

    def _check(self, hs: dict) -> None:
        want = {"state": self.cfg.state_dim, "action": self.cfg.action_dim,
                "views": sorted(self.cfg.cameras),
                "size": [self.cfg.observation_height, self.cfg.observation_width]}
        got = {"state": len(hs["state_names"]), "action": hs["action_dim"],
               "views": sorted(hs["views"]),
               "size": next(iter(hs["views"].values()))}
        if want != got:
            raise RuntimeError(f"served sim != env config: server {got} vs config {want} "
                               f"(server source: {hs.get('source')})")

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
        return self._obs(arrays), {"is_success": False}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(1, -1)
        header, arrays = self._rpc({"cmd": "step"}, {"action": a})
        success = bool(header["is_success"][0])
        # terminal obs returned unchanged; autoreset is the vector env's job
        return self._obs(arrays), float(success), success, False, {"is_success": success}

    def render(self):
        return self._last_front

    def close(self):
        try:
            protocol.send_msg(self._sock, {"cmd": "close"})
        except OSError:
            pass
        self._sock.close()
