"""generation — run one batch on a baked cell and write graded episodes.

A cell is a (scene × strategy × phase) triple in a campaign; the phase is optional —
without one the strategy's solve.py runs from scratch off the scene's own reset. With
one, the phase cell declares itself in code: reset/ holds one file per phase of the
cell's division, named exactly as the phase — each batch sweeps ALL the files, one
rollout per file (in name order), so every entry is covered: a
file both chooses which phase to enter and builds its entry state via one or more
initial-condition builders reset_0(env), reset_1(env), … — a rollout runs them ALL,
dividing the batch's envs evenly among them (remainder to the earliest; fewer envs
than builders fills them in order); each episode's meta records its (file, builder)
lineage. Their randomness uses the globally seeded RNGs. No port in the
cell = plain solve.py from its natural start. One batch =
reset-files × num_envs episodes (no phase: num_envs — scale comes from MORE
BATCHES, each with fresh world draws, not from repeating rollouts in one boot):

    build the env from the campaign preset on the cell's LOCAL scene copy
    per rollout: reset(seed+rollout) [→ phase reset] → grader → noise → recorder → solve
    grade every trajectory, write data/<batch>/ep_NNNN/{traj.npz, meta.json}
    finish with the batch meta.json: config, yield, per-episode verdicts

The stack around the unmodified solve:  solve(Recorder(NoisyActionEnv(env))).
Grading is generation's own job, no env wrapper: the cell's grader is
constructed at the entry state and its verdict() read from the final state.
States are recorded BEFORE each step (state_t, action_t pairs); the recorded action
is the solve's commanded (clean) one — the noise wrapper perturbs only what executes.
Episode states come from env.get_states(), so any recorded step can later be
restored with set_states (phase resets draw their entry states from these).

Needs a running AppLauncher (see scripts/generate.py). Sampling (engine/sampler.py)
is driven by the cells' optional params.yaml files; a yaml present means every index
is a genuine draw (`--nominal` ignores the yamls — the baseline batch mechanism):

  - WORLD (scene cell's yaml): if the cell's scene class defines
    `apply_world_params(env, values)`, worlds vary PER ENV in one parallel batch —
    env slot 0 keeps the nominal world (the in-batch canary), slot e >= 1 draws index
    `env_draw + e - 1`, and the hook writes the per-env values through the PhysX views
    (mass / material friction are per-env settable; the cfg-field -> view mapping is
    the cell's own knowledge). Without the hook: one draw (`env_draw`) patches the
    scene cfg before build — world physics becomes a per-batch axis.
  - SOLVE (strategy cell's yaml): drawn ONCE per rollout (one batched solve() call
    parameterizes all its envs together), rollout r drawing index `index0 + r`,
    passed as solve(...) kwargs — banded names must be kwargs of the solve.

Drawn values land in every episode meta; the full declarations land in the batch
meta. No params.yaml -> everything nominal, exactly as before.
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


def build_env(scene_dir: Path, num_envs: int, device: str, seed: int,
              env_decl=None, env_draw: int = 0):
    """The campaign preset's binding (robot, control mode, layout) on the LOCAL scene.

    `env_decl` (the scene cell's parsed params.yaml) picks the world physics. Two modes
    (see the module docstring): the cell's scene class defining `apply_world_params`
    gets PER-ENV worlds — nominal build, slot 0 stays nominal, slots 1.. drawn from
    `env_draw` on, values written through the hook after build; otherwise ONE draw
    (`env_draw`) patches the scene cfg before build. Returns (env, gen, drawn, per_env)
    where `drawn` is the per-slot list (per-env mode, slot 0 = {}) or the single dict.
    Validation is against the LOCAL scene's own cfg class, so a band naming a
    nonexistent field fails here — before the expensive build."""
    import dataclasses

    import robobench
    import yaml

    from .sampler import sample, validate_env_keys

    robobench.discover()
    from robobench.core.registries import ENVS, SCENES

    _load("datagen_local_scene", scene_dir / "scene" / "scene.py")
    scene_name = re.search(r'@SCENES\.register\("([\w.]+)"\)',
                           (scene_dir / "scene" / "scene.py").read_text()).group(1)
    gen = yaml.safe_load((scene_dir.parents[1] / "gen.yaml").read_text())
    drawn: dict | list = {}
    scene_cfg = None
    per_env = False
    cfg_cls = None
    if env_decl is not None and (env_decl.params or env_decl.frozen):
        scene_cls = SCENES.get(scene_name)
        cfg_cls = type(scene_cls().cfg)
        validate_env_keys(env_decl, cfg_cls())
        per_env = bool(env_decl.params) and hasattr(scene_cls, "apply_world_params")
        if per_env:  # slot 0 = nominal canary; slot e >= 1 draws index env_draw + e - 1
            drawn = [{}] + [sample(env_decl, env_draw + e) for e in range(num_envs - 1)]
        else:
            drawn = sample(env_decl, env_draw)
            if drawn:
                scene_cfg = cfg_cls(**drawn)  # fresh cfg so __post_init__ derives from the draw
    cfg = dataclasses.replace(ENVS.get(gen["preset"])(), scene=scene_name, scene_cfg=scene_cfg)
    env = cfg.build(num_envs=num_envs, device=device, seed=seed)
    if per_env:
        nominal = cfg_cls()
        values = {n: [getattr(nominal, n)] + [d[n] for d in drawn[1:]] for n in env_decl.params}
        env.scene.apply_world_params(env, values)
    return env, gen, drawn, per_env


def load_grader_cls(scene_dir: Path):
    """The judge is always the CELL's grader/grader.py (scene and grader move
    as a pair). Generation owns grading directly: the grader is constructed at
    the entry state and its verdict() read from the FINAL state — the solve
    runs unwrapped by any grading env."""
    from robobench.core.grader import BaseGrader

    mod = _load("datagen_grader", scene_dir / "grader" / "grader.py")
    return next(v for v in vars(mod).values()
                if isinstance(v, type) and issubclass(v, BaseGrader) and v is not BaseGrader)


def _clone_states(d: dict) -> dict:
    """Deep-clone a get_states tree so a snapshot survives further sim stepping."""
    return {k: _clone_states(v) if isinstance(v, dict) else v.detach().clone()
            for k, v in d.items()}


def _slice_assign(dst: dict, src: dict, sl: slice) -> None:
    """Copy src's env-slice into dst across the whole states tree (dim 0 = env)."""
    for k, v in src.items():
        if isinstance(v, dict):
            _slice_assign(dst[k], v, sl)
        else:
            dst[k][sl] = v[sl]


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
              strategy: str = "strategy_0", phase: str | None = None,
              num_envs: int = 4, seed: int = 0,
              noise: dict | None = None, device: str = "cuda:0",
              env_draw: int = 0, index0: int = 0, nominal: bool = False) -> Path:
    import numpy as np
    import torch

    from .noise import NoisyActionEnv
    from .sampler import Declaration, load_declaration, sample

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

    env_decl = Declaration() if nominal else load_declaration(scene_dir)
    solve_decl = Declaration() if nominal else load_declaration(strategy_dir)
    env, gen, env_drawn, per_env_world = build_env(scene_dir, num_envs, device, seed,
                                                   env_decl, env_draw)
    if env_decl.params:
        mode = "per-env (slot 0 nominal)" if per_env_world else "per-batch"
        print(f"[batch {batch}] world params {mode}: {env_drawn}", flush=True)
    grader_cls = load_grader_cls(scene_dir)
    if phase is None:
        solve = _load("datagen_solve", strategy_dir / "solve.py").solve
        entry, conditions = None, []
    else:
        # reset/ holds one file per phase, named as the phase: each batch
        # sweeps all the files, one rollout per file — a file chooses the
        # entry and builds its state. No port = plain solve.py.
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
    # one rollout per reset file (no phase = a single rollout) — every entry
    # covered once per batch; more episodes = more batches
    rollouts = max(1, len(conditions))
    for rnd in range(rollouts):
        env.reset(seed=seed + rnd)
        reset_name, fn_of_env = None, None
        if conditions:
            cond = conditions[rnd % len(conditions)]
            reset_name = cond.__name__.removeprefix("datagen_reset_")
            # a phase file holds reset_0(env), reset_1(env), … — ALL of them run,
            # the batch's envs divided evenly among them: each builder shapes
            # (and settles) the whole batch; its settled snapshot supplies its
            # env-slice of the composed entry state, so a later builder's settle
            # never disturbs an earlier builder's envs. Randomness inside uses
            # the global RNGs, already seeded by env.reset(seed=seed+rollout).
            names = sorted((n for n in vars(cond) if re.fullmatch(r"reset_\d+", n)),
                           key=lambda n: int(n[6:]))
            if len(names) == 1:
                getattr(cond, names[0])(env)
                fn_of_env = [names[0]] * num_envs
            else:
                # even split, remainder to the earliest; fewer envs than
                # builders fills them in order (later builders get none and
                # are skipped)
                base, rem = divmod(num_envs, len(names))
                counts = [base + (1 if i < rem else 0) for i in range(len(names))]
                composed, fn_of_env, lo = None, [], 0
                for n, c in zip(names, counts):
                    if c == 0:
                        continue
                    getattr(cond, n)(env)
                    snap = _clone_states(env.get_states())
                    if composed is None:
                        composed = snap
                    else:
                        _slice_assign(composed, snap, slice(lo, lo + c))
                    fn_of_env += [n] * c
                    lo += c
                env.set_states(composed)
            entry = reset_name if has_port else None  # the file IS the phase
        grader = grader_cls(env)
        grader.setup()  # baselines captured at the entry state
        stack = NoisyActionEnv(env, dims=slice(*dims) if dims else slice(0, 0),
                               sigma=noise.get("sigma", 0.0), prob=noise.get("prob", 1.0),
                               duration=noise.get("duration", 0.0), seed=seed + rnd)
        rec = Recorder(stack, env)
        solve_params = sample(solve_decl, index0 + rnd)
        print(f"[batch {batch}] rollout {rnd + 1}/{rollouts}"
              + (f" ({reset_name})" if reset_name else "")
              + (f" solve_params={solve_params}" if solve_params else "")
              + f": solve on {num_envs} envs …", flush=True)
        solve(rec, **solve_params) if entry is None else solve(rec, entry=entry, **solve_params)

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
                "episode": ep, "rollout": rnd, "env_index": e,
                "success": verdicts[e]["success"], "score": verdicts[e]["score"],
                # the actual draws this episode ran under ({} = nominal on that axis)
                "parameters": {"env": env_drawn[e] if per_env_world else env_drawn,
                               "solve": solve_params},
                "reset": reset_name, "reset_fn": (fn_of_env[e] if fn_of_env else None),
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
        print(f"[batch {batch}] rollout {rnd + 1}/{rollouts}: {ok}/{num_envs} succeeded, "
              f"{T} steps", flush=True)

    n_ok = sum(v["success"] for v in verdicts_all)
    (out / "meta.json").write_text(json.dumps({
        "batch": batch, "cell": cell,
        "preset": gen["preset"], "num_envs": num_envs, "seed": seed,
        "noise": {k: v for k, v in noise.items() if v},
        # the full declarations + this batch's slice of the index space (provenance)
        "params": {"scene": env_decl.to_meta(), "strategy": solve_decl.to_meta(),
                   "env_draw": env_draw, "index0": index0, "nominal": nominal,
                   "per_env_world": per_env_world, "world_drawn": env_drawn},
        "episodes": len(verdicts_all), "successes": n_ok,
        "success_rate": round(n_ok / max(1, len(verdicts_all)), 4),
        "verdicts": verdicts_all,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
    }, indent=2) + "\n")
    refresh_metas(real_root)
    print(f"[batch {batch}] DONE: {n_ok}/{len(verdicts_all)} -> {out}", flush=True)
    return out
