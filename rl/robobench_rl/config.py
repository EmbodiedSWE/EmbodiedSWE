"""Layered yaml config: base <- task env `defaults()` <- tasks/<task>.yaml <- rewards/<reward>.yaml <- `key.sub=value` overrides.

The layering is the audit trail: the base holds the defaults and the observation rule; a task file
sets the preset, horizon, env count, action mapping, and any per-task PPO overrides (same key path,
deep-merged); a reward file selects the reward source. The merged config is dumped with each run."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(task: str, reward: str | None = None, overrides: list[str] | None = None) -> dict:
    """`task` / `reward` are file stems under configs/tasks and configs/rewards (or explicit paths).
    Overrides are `a.b.c=value` with the value parsed as yaml (so `num_envs=64`, `x=[1,2]`)."""
    cfg = _load(CONFIG_DIR / "base.yaml")
    task_path = Path(task) if task.endswith(".yaml") else CONFIG_DIR / "tasks" / f"{task}.yaml"
    task_cfg = _load(task_path)
    task_env = (task_cfg.get("task") or {}).get("task_env")
    if task_env:  # a tuned env's own defaults (horizon, finger PD, PPO steps/entropy, ...) beat base.yaml, lose to the yaml
        from .tasks import load_task_env_cls

        cfg = _deep_merge(cfg, load_task_env_cls(task_env).defaults())
    cfg = _deep_merge(cfg, task_cfg)
    if reward:
        reward_path = Path(reward) if reward.endswith(".yaml") else CONFIG_DIR / "rewards" / f"{reward}.yaml"
        cfg = _deep_merge(cfg, _load(reward_path))
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    cfg.setdefault("_layers", {"base": "base.yaml", "task": task_path.name, "reward": reward})
    return cfg


def dump_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
