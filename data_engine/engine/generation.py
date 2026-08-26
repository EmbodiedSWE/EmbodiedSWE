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
is driven by the LOCAL scene's own `PHYSICAL_PARAMS` bands — no side files: worlds vary
PER ENV in one parallel batch. Env slot 0 keeps the nominal world (the in-batch
canary), slot e >= 1 draws index `env_draw + e - 1`, written after the build via
`scene.apply_physical_params(env, values)` (the scene owns its cfg-field -> PhysX-view
mapping; `bind()` routes the nominal application through the same hook, so the code
path is exercised by every batch ever run). Solve hyperparameters sample ONE set per
batch from the solve module's `SOLVE_PARAMS` bands (`--solve_draw`) and are written
onto the module's CONSTANTS before solve(env) runs — the solve signature never
changes; the values in the file are the nominal. `--nominal` skips ALL sampling
(the baseline batch: plain world, the file's own values).
Drawn values land in every episode meta; the band specs land in the batch meta. No
bands declared -> nominal at every index, exactly as before.
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
              env_draw: int = 0, nominal: bool = False, solo_draw: bool = False,
              phys_nominal: bool = False,
              env_spacing: float | None = None,
              scene_overrides: dict | None = None):
    """The campaign preset's binding (robot, control mode, layout) on the LOCAL scene.

    World physics comes from the LOCAL scene's own `PHYSICAL_PARAMS` bands (see the module
    docstring): slot 0 nominal, slot e >= 1 at index `env_draw + e - 1`, written through
    `scene.apply_physical_params` after the build. `nominal=True` skips sampling. Returns
    (env, gen, bands, slot_drawn): the validated band specs and the per-slot draws
    (slot 0 = {}; both empty when nominal or band-less). A bad band fails here — before
    the expensive build.

    `scene_overrides` (replay's visual draw) constructs the scene cfg WITH those field
    values — through the constructor, not setattr, so `__post_init__` derivations (a
    table preset filling its usd/height) see them — and the build consumes them."""
    import dataclasses

    import robobench
    import yaml

    from .sampler import sample, scene_bands

    robobench.discover()
    from robobench.core.registries import ENVS, SCENES

    _load("datagen_local_scene", scene_dir / "scene" / "scene.py")
    scene_name = re.search(r'@SCENES\.register\("([\w.]+)"\)',
                           (scene_dir / "scene" / "scene.py").read_text()).group(1)
    gen = yaml.safe_load((scene_dir.parents[1] / "gen.yaml").read_text())
    scene_cls = SCENES.get(scene_name)
    bands = {} if (nominal or phys_nominal) else scene_bands(scene_cls, scene_cls().cfg)
    # slot 0 = nominal canary; slot e >= 1 draws index env_draw + e - 1. solo_draw ON (single-env
    # diversified batches that sidestep the lockstep phase coupling): EVERY slot draws.
    if not bands:
        slot_drawn = []
    elif solo_draw:
        slot_drawn = [sample(bands, env_draw + e) for e in range(num_envs)]
    else:
        slot_drawn = [{}] + [sample(bands, env_draw + e) for e in range(num_envs - 1)]
    cfg = dataclasses.replace(ENVS.get(gen["preset"])(), scene=scene_name)
    # env_spacing: None keeps the preset's grid; replay overrides it (recorded states
    # shift onto whatever grid the replay builds, so spacing is free there)
    extra = {} if env_spacing is None else {"env_spacing": env_spacing}
    if scene_overrides:
        extra["scene_cfg"] = type(scene_cls().cfg)(**scene_overrides)
    env = cfg.build(num_envs=num_envs, device=device, seed=seed, **extra)
    if slot_drawn:
        c = env.scene.cfg  # nominal source for slot 0
        values = {n: [getattr(c, n)] + [d[n] for d in slot_drawn[1:]] for n in bands}
        env.scene.apply_physical_params(env, values)
    return env, gen, bands, slot_drawn


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


def _controller_info(robot) -> dict:
    """The EFFECTIVE control law the episode ran under, captured after the solve:
    setup-time overrides are live writes on the controller/articulation (never in a
    cfg file), so generation is the only moment they can be recorded. This block is
    what makes episodes from different controllers (other presets, real teleop)
    distinguishable downstream — the action label only means anything under it."""
    import torch

    def leaf(c) -> dict:
        d: dict = {"class": type(c).__name__, "control_period": c._control_period,
                   # THE effective rate — cfg's `dt` is only the pre-bind preference and
                   # goes stale when a solve writes `_control_period` directly
                   "control_dt": robot.env.dt * c._control_period}
        cfg = getattr(c, "cfg", None)
        if cfg is not None:
            import dataclasses as _dc

            def _safe(v):  # JSON-safe, recursively: nested cfg dataclasses (e.g. pink's
                # FrameTaskCfg frames) flatten to dicts instead of crashing json.dumps
                if isinstance(v, torch.Tensor):
                    return v.tolist()
                if _dc.is_dataclass(v) and not isinstance(v, type):
                    return {k2: _safe(v2) for k2, v2 in vars(v).items()}
                if isinstance(v, (tuple, list)):
                    return [_safe(x) for x in v]
                return v

            d["cfg"] = {k: _safe(v)
                        for k, v in vars(cfg).items()
                        if isinstance(v, (int, float, bool, str, tuple, list, torch.Tensor))
                        or (_dc.is_dataclass(v) and not isinstance(v, type))}
        for name in ("_kp", "_kd"):  # task-space gains live on the instance, not the cfg
            v = getattr(c, name, None)
            if isinstance(v, torch.Tensor):
                d[name.lstrip("_")] = v.tolist()
        return d

    leaves = getattr(robot.controller, "controllers", None) or [robot.controller]
    data = robot.articulation.data
    return {
        "leaves": [leaf(c) for c in leaves],
        "joint_names": list(robot.articulation.joint_names),
        # per-joint drive gains (env 0 — identical across envs): captures e.g. the
        # gripper stiffness the solve wrote to sim, which sets what a position
        # target means in force terms
        "joint_stiffness": data.joint_stiffness[0].tolist(),
        "joint_damping": data.joint_damping[0].tolist(),
    }


def _ctrl_diff(a: dict, b: dict, prefix: str = "") -> dict:
    """{path: [old, new]} for every leaf that differs between two _controller_info dicts."""
    out: dict = {}
    for k in set(a) | set(b):
        va, vb, p = a.get(k), b.get(k), f"{prefix}{k}"
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(_ctrl_diff(va, vb, p + "."))
        elif (isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb)
              and va and isinstance(va[0], dict)):
            for i, (x, y) in enumerate(zip(va, vb)):
                out.update(_ctrl_diff(x, y, f"{p}[{i}]."))
        elif va != vb:
            out[p] = [va, vb]
    return out


class Recorder:
    """Outermost wrapper: records (state_t, commanded action_t) before delegating.
    Also WATCHES the control law: the stamped block is one post-solve snapshot, so a
    solve that re-gains mid-episode (phase-wise kp/kd) would otherwise be silently
    misdescribed — sampled every CTRL_CHECK steps, each change lands in ctrl_changes."""

    CTRL_CHECK = 25  # latches between law checks (phases last hundreds; cost ~0.5 ms/check)

    def __init__(self, env, raw_env) -> None:
        self._env, self._raw = env, raw_env
        self.states: list[dict] = []
        self.actions: list = []
        self.joint_targets: list = []  # COMMANDED joint targets per tick (see step)
        self._intent_ok: bool | None = None  # all-position-mode leaves? resolved lazily
        self.ctrl_changes: list[dict] = []
        self._ctrl_ref: dict | None = None

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def watch_controller(self, step: int | None = None) -> None:
        snap = _controller_info(self._raw.robot)
        if self._ctrl_ref is None:
            self._ctrl_ref = snap
        elif snap != self._ctrl_ref:
            self.ctrl_changes.append({"step": len(self.actions) if step is None else step,
                                      "changed": _ctrl_diff(self._ctrl_ref, snap)})
            self._ctrl_ref = snap

    def step(self, action, render: bool = False):
        if len(self.actions) % self.CTRL_CHECK == 0:
            self.watch_controller()
        self.states.append(_flat(self._raw.get_states()))
        self.actions.append(action.detach().cpu().clone())
        ret = self._env.step(action, render)
        # The COMMANDED joint targets that governed this step (written by the controller
        # during it, held by the actuator PD) — controller INTENT, which achieved-state
        # labels flatten: a press/squeeze is a sustained target offset past contact.
        # Recorded ONLY when every leaf controller writes position targets (joint/diff_ik/
        # pink_ik): under a torque-mode arm (osc/impedance) the arm columns of the target
        # buffer are the inert reset pose — stamping them would let a joint_target bake
        # silently produce frozen-home actions. Absent channel -> the convention refuses
        # loudly instead. Single-articulation robots only (a MultiRobot has no one vector).
        if self._intent_ok is None:
            ctrl = getattr(self._raw.robot, "controller", None)
            leaves = getattr(ctrl, "controllers", [ctrl] if ctrl is not None else [])
            self._intent_ok = (
                getattr(self._raw.robot, "articulation", None) is not None
                and bool(leaves)
                and all(getattr(c, "command_type", None) == "position" for c in leaves)
            )
        if self._intent_ok:
            art = self._raw.robot.articulation
            if art.data.joint_pos_target is not None:
                self.joint_targets.append(art.data.joint_pos_target.detach().cpu().clone())
        return ret


def run_batch(gen_root: str | Path, batch: str | None = None, scene: str = "scene_0",
              strategy: str = "strategy_0", phase: str | None = None,
              num_envs: int = 4, seed: int = 0,
              noise: dict | None = None, device: str = "cuda:0",
              env_draw: int = 0, solve_draw: int = 0, nominal: bool = False,
              solo_draw: bool = False, phys_nominal: bool = False,
              solve_nominal: bool = False) -> Path:
    import numpy as np
    import torch

    from .noise import NoisyActionEnv
    from .sampler import sample, solve_bands

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

    env, gen, bands, slot_drawn = build_env(scene_dir, num_envs, device, seed,
                                            env_draw, nominal, solo_draw=solo_draw,
                                            phys_nominal=phys_nominal)
    if slot_drawn:
        _c = "every slot drawn" if solo_draw else "slot 0 nominal"
        print(f"[batch {batch}] physical params per-env ({_c}): {slot_drawn}", flush=True)
    grader_cls = load_grader_cls(scene_dir)
    if phase is None:
        solve_mod = _load("datagen_solve", strategy_dir / "solve.py")
        solve = solve_mod.solve
        entry, conditions = None, []
    else:
        # reset/ holds one file per phase, named as the phase: each batch
        # sweeps all the files, one rollout per file — a file chooses the
        # entry and builds its state. No port = plain solve.py.
        phase_dir = strategy_dir / "phases" / phase
        port = phase_dir / "solve_by_phase.py"
        solve_mod = _load("datagen_solve", port if port.is_file()
                          else strategy_dir / "solve.py")
        solve = solve_mod.solve
        has_port = port.is_file()
        entry = None
        conditions = [_load(f"datagen_reset_{f.stem}", f)
                      for f in sorted((phase_dir / "reset").glob("*.py"))]
    # ONE set of solve hyperparameters per batch (see sampler.solve_bands): drawn at
    # --solve_draw and WRITTEN ONTO THE MODULE's constants before solve(env) runs —
    # the solve signature never changes. --nominal (or no SOLVE_PARAMS) -> file values.
    s_bands = {} if (nominal or solve_nominal) else solve_bands(solve_mod)
    solve_drawn = sample(s_bands, solve_draw) if s_bands else {}
    for n, v in solve_drawn.items():
        setattr(solve_mod, n, v)
    if solve_drawn:
        print(f"[batch {batch}] solve params (one set, whole batch): {solve_drawn}", flush=True)
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
        print(f"[batch {batch}] rollout {rnd + 1}/{rollouts}"
              + (f" ({reset_name})" if reset_name else "")
              + f": solve on {num_envs} envs …", flush=True)
        solve(rec) if entry is None else solve(rec, entry=entry)

        verdicts = grader.verdict()
        ctrl_info = _controller_info(env.robot)  # after the solve = overrides included
        T = len(rec.actions)
        rec.watch_controller(step=T)  # catch a change in the last <CTRL_CHECK latches
        if rec.ctrl_changes:
            print(f"[batch {batch}] WARNING: control law changed MID-SOLVE at steps "
                  f"{[c['step'] for c in rec.ctrl_changes]} — the stamped `controller` is the "
                  f"FINAL law; per-change diffs are in meta `controller_changes`", flush=True)
        arrays = {k: np.stack([s[k].numpy() for s in rec.states]) for k in rec.states[0]}
        arrays["action"] = np.stack([a.numpy() for a in rec.actions])
        if len(rec.joint_targets) == T:  # commanded-target channel (see Recorder.step)
            arrays["robot/joint_target"] = np.stack([t.numpy() for t in rec.joint_targets])
        for e in range(num_envs):
            ep = rnd * num_envs + e
            ep_dir = out / f"ep_{ep:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ep_dir / "traj.npz",
                                **{k: v[:, e] for k, v in arrays.items()})
            meta = {
                "episode": ep, "rollout": rnd, "env_index": e,
                "success": verdicts[e]["success"], "score": verdicts[e]["score"],
                # the draws this episode ran under ({} = nominal on that axis)
                "parameters": {"physical": slot_drawn[e] if slot_drawn else {},
                               "solve": solve_drawn},
                "reset": reset_name, "reset_fn": (fn_of_env[e] if fn_of_env else None),
                "entry": entry,
                "seed": seed + rnd, "steps": T,
                "sim_dt": env.dt, "decimation": env.robot.control_period,
                "controller": ctrl_info,
                "controller_changes": rec.ctrl_changes,
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
        # the scene's band specs + this batch's slice of the index space (provenance)
        "params": {"physical_params": bands, "env_draw": env_draw, "nominal": nominal,
                   "per_env": slot_drawn,
                   "solve_params": s_bands, "solve_draw": solve_draw, "solve_drawn": solve_drawn},
        "episodes": len(verdicts_all), "successes": n_ok,
        "success_rate": round(n_ok / max(1, len(verdicts_all)), 4),
        "verdicts": verdicts_all,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
    }, indent=2) + "\n")
    refresh_metas(real_root)
    print(f"[batch {batch}] DONE: {n_ok}/{len(verdicts_all)} -> {out}", flush=True)
    return out
