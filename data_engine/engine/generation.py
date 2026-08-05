"""generation — run one batch on a baked cell and write graded episodes.

A cell is a (scene × strategy × phase) triple in a campaign; the phase is optional —
without one the strategy's solve.py runs from scratch off the scene's own reset. With
one, the phase cell declares itself in code: reset/ holds one file per phase of the
cell's division, named exactly as the phase — the file sampled each round (uniformly)
both chooses which phase to enter and builds its entry state: it holds one or more
initial-condition builders reset_0(env), reset_1(env), … — generate targets one by
index (default 0); their randomness uses the globally seeded RNGs. No port in the
cell = plain solve.py from its natural start. One batch =
rounds × num_envs episodes:

    build the env from the campaign preset on the cell's LOCAL scene copy
    per round: reset(seed+round) [→ phase reset] → grader → noise → recorder → solve
    grade every trajectory, write data/<batch>/ep_NNNN/{traj.npz, meta.json}
    finish with the batch meta.json: config, yield, per-episode verdicts

The stack around the unmodified solve:  solve(Recorder(NoisyActionEnv(env))).
Grading is generation's own job, no env wrapper: the cell's grader is
constructed at the entry state and its verdict() read from the final state.
States are recorded BEFORE each step (state_t, action_t pairs); the recorded action
is the solve's commanded (clean) one — the noise wrapper perturbs only what executes.
Episode states come from env.get_states(), so any recorded step can later be
restored with set_states (phase resets draw their entry states from these).

Needs a running AppLauncher (see scripts/generate.py). Sampling is nominal-only
for now: engine/sampler.py is a placeholder until the tunable rebuild.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .meta import refresh_metas

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    """Exec a file as module `name` once; later calls return the same module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_env(scene_dir: Path, num_envs: int, device: str, seed: int):
    """The campaign preset's binding (robot, control mode, layout) on the LOCAL scene."""
    import dataclasses

    import robobench
    import yaml

    robobench.discover()
    from robobench.core.registries import ENVS

    _load("datagen_local_scene", scene_dir / "scene" / "scene.py")
    scene_name = re.search(r'@SCENES\.register\("([\w.]+)"\)',
                           (scene_dir / "scene" / "scene.py").read_text()).group(1)
    gen = yaml.safe_load((scene_dir.parents[1] / "gen.yaml").read_text())
    cfg = dataclasses.replace(ENVS.get(gen["preset"])(), scene=scene_name)
    return cfg.build(num_envs=num_envs, device=device, seed=seed), gen


def load_grader_cls(scene_dir: Path):
    """The judge is always the CELL's grader/grader.py (scene and grader move
    as a pair). Generation owns grading directly: the grader is constructed at
    the entry state and its verdict() read from the FINAL state — the solve
    runs unwrapped by any grading env."""
    from robobench.core.grader import BaseGrader

    mod = _load("datagen_grader", scene_dir / "grader" / "grader.py")
    return next(v for v in vars(mod).values()
                if isinstance(v, type) and issubclass(v, BaseGrader) and v is not BaseGrader)


def _flat(d: dict, prefix: str = "") -> dict:
    """Nested get_states dict -> {"scene/nut/root_state": (E, …) cpu tensor, …}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "/"))
        else:
            out[key] = v.detach().cpu().clone()
    return out


class Recorder:
    """Outermost wrapper: records (state_t, commanded action_t) before delegating."""

    def __init__(self, env, raw_env) -> None:
        self._env, self._raw = env, raw_env
        self.states: list[dict] = []
        self.actions: list = []

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action, render: bool = False):
        self.states.append(_flat(self._raw.get_states()))
        self.actions.append(action.detach().cpu().clone())
        return self._env.step(action, render)


def run_batch(gen_root: str | Path, batch: str | None = None, scene: str = "scene_0",
              strategy: str = "strategy_0", phase: str | None = None, reset: int = 0,
              num_envs: int = 4, rounds: int = 1, seed: int = 0,
              noise: dict | None = None, device: str = "cuda:0") -> Path:
    import random

    import numpy as np
    import torch


    from .noise import NoisyActionEnv

    noise = noise or {}
    gen_root = Path(gen_root)
    scene_dir = gen_root / "scenes" / scene
    strategy_dir = scene_dir / "strategies" / strategy
    # a session's workspace view aliases cells (scene_0 -> the real start scene);
    # lineage always records the CAMPAIGN names, so resolve through any symlinks
    real_root = (gen_root / "gen.yaml").resolve().parent
    cell = (f"{scene_dir.resolve().name}/{strategy_dir.resolve().name}"
            + (f"/{phase}" if phase else ""))
    batch = batch or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    out = real_root / "data" / batch
    if out.exists():
        raise SystemExit(f"{out} already exists — batches are append-only")

    env, gen = build_env(scene_dir, num_envs, device, seed)
    grader_cls = load_grader_cls(scene_dir)
    if phase is None:
        solve = _load("datagen_solve", strategy_dir / "solve.py").solve
        entry, conditions = None, []
    else:
        # reset/ holds one file per phase, named as the phase: the sampled file
        # chooses the entry and builds its state. No port = plain solve.py.
        phase_dir = strategy_dir / "phases" / phase
        port = phase_dir / "solve_by_phase.py"
        solve = _load("datagen_solve", port if port.is_file()
                      else strategy_dir / "solve.py").solve
        has_port = port.is_file()
        entry = None
        conditions = [_load(f"datagen_reset_{f.stem}", f)
                      for f in sorted((phase_dir / "reset").glob("*.py"))]
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dims = noise.get("dims")

    verdicts_all = []
    for rnd in range(rounds):
        env.reset(seed=seed + rnd)
        reset_name = None
        if conditions:
            rng = random.Random(seed + rnd)
            cond = rng.choice(conditions)
            reset_name = cond.__name__.removeprefix("datagen_reset_")
            # a phase file holds reset_0(env), reset_1(env), … — take the targeted
            # one, falling back to reset_0. Randomness inside uses the global RNGs,
            # already seeded by env.reset(seed=seed+round).
            fn = getattr(cond, f"reset_{reset}", None) or cond.reset_0
            fn(env)
            entry = reset_name if has_port else None  # the file IS the phase
        grader = grader_cls(env)
        grader.setup()  # baselines captured at the entry state
        stack = NoisyActionEnv(env, dims=slice(*dims) if dims else slice(0, 0),
                               sigma=noise.get("sigma", 0.0), prob=noise.get("prob", 1.0),
                               duration=noise.get("duration", 0.0), seed=seed + rnd)
        rec = Recorder(stack, env)
        print(f"[batch {batch}] round {rnd}: solve on {num_envs} envs …", flush=True)
        solve(rec) if entry is None else solve(rec, entry=entry)

        verdicts = grader.verdict()
        T = len(rec.actions)
        arrays = {k: np.stack([s[k].numpy() for s in rec.states]) for k in rec.states[0]}
        arrays["action"] = np.stack([a.numpy() for a in rec.actions])
        for e in range(num_envs):
            ep = rnd * num_envs + e
            ep_dir = out / f"ep_{ep:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ep_dir / "traj.npz",
                                **{k: v[:, e] for k, v in arrays.items()})
            meta = {
                "episode": ep, "round": rnd, "env_index": e,
                "success": verdicts[e]["success"], "score": verdicts[e]["score"],
                "parameters": {},  # nominal — sampler is a placeholder
                "reset": reset_name, "reset_fn": (f"reset_{reset}" if reset_name else None),
                "entry": entry,
                "seed": seed + rnd, "steps": T,
                "sim_dt": env.dt, "decimation": env.robot.control_period,
                "noise": {k: v for k, v in noise.items() if v},
                "preset": gen["preset"],
                "cell": cell,
                "git_sha": sha,
            }
            (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        verdicts_all += [{"episode": rnd * num_envs + e, **v} for e, v in enumerate(verdicts)]
        ok = sum(v["success"] for v in verdicts)
        print(f"[batch {batch}] round {rnd}: {ok}/{num_envs} succeeded, {T} steps", flush=True)

    n_ok = sum(v["success"] for v in verdicts_all)
    (out / "meta.json").write_text(json.dumps({
        "batch": batch, "cell": cell,
        "preset": gen["preset"], "num_envs": num_envs, "rounds": rounds, "seed": seed,
        "noise": {k: v for k, v in noise.items() if v},
        "episodes": len(verdicts_all), "successes": n_ok,
        "success_rate": round(n_ok / max(1, len(verdicts_all)), 4),
        "verdicts": verdicts_all,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
    }, indent=2) + "\n")
    refresh_metas(real_root)
    print(f"[batch {batch}] DONE: {n_ok}/{len(verdicts_all)} -> {out}", flush=True)
    return out
