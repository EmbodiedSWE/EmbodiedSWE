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
    per rollout: reset(seed+rollout) [→ phase reset] → grader → recorder → solve
    grade every trajectory, write data/<batch>/ep_NNNN/{traj.npz, meta.json}
    finish with the batch meta.json: config, yield, per-episode verdicts

The stack around the solve:  solve(Recorder(env)).
Grading is generation's own job, no env wrapper: the cell's grader is
constructed at the entry state and its verdict() read from the final state.
States are recorded BEFORE each step (state_t, action_t pairs); the recorded action
is the solve's commanded (clean) one. Noise is SOLVE-AUTHORED: the solve may pass
`noise=` to step() (DART-style executed perturbation, its own phase knowledge
choosing where/how much), gated by run_batch's `noise_scale` master switch — the
Recorder guarantees structurally that labels never contain it (see Recorder).
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
from .noise import NoisyActionEnv

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


def _local_scene_cfg(scene_cls, preset_scene_cfg, overrides: dict | None):
    """THE scene-cfg construction — one path for generation AND replay.

    Always an instance of the LOCAL scene's own cfg class: preset-registered values
    migrate field-by-field where the names match, fields the local scene ADDED keep
    their local defaults, and `overrides` (replay's visual draw) win last — through
    the constructor, not setattr, so `__post_init__` derivations (a table preset
    filling its usd/height) see them.

    This is what makes a scene edit safe end-to-end: pen_holder's v28 wave died at
    every render because the preset registry handed the edited scene a cfg INSTANCE
    predating the edit — generation happened to build without one and passed, replay
    did not. There must be no build path on which the scene meets a cfg type other
    than its own.

    Only init=True fields migrate through the constructor: derived fields
    (dataclass init=False, filled by __post_init__ — coffee's cup_outer_r) re-derive
    from the migrated inputs, which is their meaning; passing them to the
    constructor is a TypeError (measured on the first Modal prelim, 2026-09-01).
    Overrides that name a non-init field (a visual band on a live attribute) are
    applied by setattr AFTER construction, once __post_init__ has run."""
    import dataclasses

    local_type = type(scene_cls().cfg)
    local_fields = {f.name: f for f in dataclasses.fields(local_type)}
    values: dict = {}
    if preset_scene_cfg is not None:
        if dataclasses.is_dataclass(preset_scene_cfg):
            src = {f.name for f in dataclasses.fields(preset_scene_cfg) if f.init}
        else:
            src = set(vars(preset_scene_cfg))
        values = {n: getattr(preset_scene_cfg, n) for n in src
                  if n in local_fields and local_fields[n].init}
    post: dict = {}
    for name, value in (overrides or {}).items():
        if name in local_fields and local_fields[name].init:
            values[name] = value
        else:
            post[name] = value
    cfg = local_type(**values)
    for name, value in post.items():
        setattr(cfg, name, value)
    return cfg


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

    The scene cfg is ALWAYS materialized as the local scene's own cfg type (see
    `_local_scene_cfg`); `scene_overrides` is replay's visual draw."""
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
    preset_cfg = ENVS.get(gen["preset"])()
    cfg = dataclasses.replace(preset_cfg, scene=scene_name)
    # env_spacing: None keeps the preset's grid; replay overrides it (recorded states
    # shift onto whatever grid the replay builds, so spacing is free there)
    extra = {} if env_spacing is None else {"env_spacing": env_spacing}
    extra["scene_cfg"] = _local_scene_cfg(scene_cls, preset_cfg.scene_cfg, scene_overrides)
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




def run_reset_builders(env, cond, num_envs: int) -> list[str]:
    """Run a phase reset file's builders on a freshly reset env, exactly as generation does —
    and exactly as replay must, so a phase entry is REBUILT from the seed rather than restored.

    A phase file holds reset_0(env), reset_1(env), … — ALL of them run, the batch's envs
    divided evenly among them: each builder shapes (and settles) the whole batch; its settled
    snapshot supplies its env-slice of the composed entry state, so a later builder's settle
    never disturbs an earlier builder's envs. Randomness inside uses the global RNGs, already
    seeded by env.reset(seed=seed+rollout). Returns the builder name per env."""
    names = sorted((n for n in vars(cond) if re.fullmatch(r"reset_\d+", n)),
                   key=lambda n: int(n[6:]))
    if len(names) == 1:
        getattr(cond, names[0])(env)
        return [names[0]] * num_envs
    # even split, remainder to the earliest; fewer envs than builders fills them in order
    # (later builders get none and are skipped)
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
    return fn_of_env


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
        # task-space gains AND the nullspace posture live on the instance, not the cfg; the
        # replay restores all three (bulb's solve points `_q_default` at the home pose)
        for name in ("_kp", "_kd", "_q_default"):
            v = getattr(c, name, None)
            if isinstance(v, torch.Tensor):
                d[name.lstrip("_")] = v.tolist()
        return d

    children = getattr(robot, "robots", None)
    if isinstance(children, dict):
        return {
            "class": type(robot).__name__,
            "action_slices": {
                name: [s.start, s.stop]
                for name, s in robot.action_slices.items()
            },
            "children": {
                name: _controller_info(child) for name, child in children.items()
            },
        }

    controller = getattr(robot, "controller", None)
    articulation = getattr(robot, "articulation", None)
    if controller is None or articulation is None:
        raise RuntimeError(
            f"cannot capture controller metadata for {type(robot).__name__}: "
            "expected either a robots mapping or a bound controller + articulation"
        )
    leaves = getattr(controller, "controllers", None) or [controller]
    data = articulation.data
    return {
        "class": type(robot).__name__,
        "leaves": [leaf(c) for c in leaves],
        "joint_names": list(articulation.joint_names),
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
    misdescribed — sampled every CTRL_CHECK steps, each change lands in ctrl_changes.

    Env-batch width is asserted at the FIRST recorded step: a scalar solve in a wide
    world must die in seconds, not after simulating the whole episode (a 512-env
    pen_holder probe once ran 3123 steps before the save-time check caught it).
    The save-time checks in run_batch remain as the backstop for later corruption.

    THE NOISE CHANNEL (DART/MimicGen-style, agent-authored): a solve may pass
    `noise=` to step(); the clean action is recorded FIRST and `action +
    noise_scale * noise` is what executes. Labels structurally never contain the
    noise — there is no code path from the perturbation into self.actions. The
    solve decides where and how much (its phase knowledge); the pipeline decides
    IF, through noise_scale (0 = probes/base set run noise-free even if the solve
    offers noise; the dynamics harvest enables it). Every perturbation is also recorded
    verbatim (self.noises) so downstream consumers can reconstruct the clean
    command in any derived space (e.g. joint targets, which the controller
    computes from the EXECUTED action).

    SUCCESS PROBING (tail trim): when a probe callback is given, the grader's
    per-env success is sampled every PROBE_EVERY latches; the save path derives
    each env's earliest SUSTAINED-success step so replay/export can drop the
    padded station-keeping tail that wide batches append to every finished env."""

    # 1, not 25: the replay re-applies each recorded change AT its stamped step, so a change
    # detected up to 24 steps late replayed 24 steps late — enough to break a solve that toggles
    # kp_null at every phase boundary (pc_motherboard: 126 toggles per episode). ~0.5 ms/check.
    CTRL_CHECK = 1
    PROBE_EVERY = 30  # latches between grader success samples (~2 s at 15 Hz control)

    def __init__(self, env, raw_env, num_envs: int, noise_scale: float = 0.0,
                 success_probe=None) -> None:
        self._env, self._raw, self._num_envs = env, raw_env, num_envs
        self._noise_scale = float(noise_scale)
        self._success_probe = success_probe
        self.states: list[dict] = []
        self.actions: list = []
        self.noises: list = []  # scaled executed perturbation per step (zeros when clean)
        self.success_samples: list[tuple[int, list[bool]]] = []
        self.joint_targets: list = []  # COMMANDED joint targets per tick (see step)
        self._intent_ok: bool | None = None  # all-position-mode leaves? resolved lazily
        self.ctrl_changes: list[dict] = []
        self._ctrl_ref: dict | None = None

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def success_steps(self, final_success: list[bool]) -> list[int | None]:
        """Per env: the earliest sampled step from which success held THROUGH the
        end (None = env failed, or success was never sampled). Conservative by
        construction: a transient mid-episode success that later regressed never
        shortens the episode."""
        out: list[int | None] = []
        for e in range(self._num_envs):
            if not final_success[e] or not self.success_samples:
                out.append(None)
                continue
            step = None
            for s, mask in reversed(self.success_samples):
                if not mask[e]:
                    break
                step = s
            out.append(step)
        return out

    def watch_controller(self, step: int | None = None) -> None:
        snap = _controller_info(self._raw.robot)
        if self._ctrl_ref is None:
            self._ctrl_ref = snap
        elif snap != self._ctrl_ref:
            self.ctrl_changes.append({"step": len(self.actions) if step is None else step,
                                      "changed": _ctrl_diff(self._ctrl_ref, snap)})
            self._ctrl_ref = snap

    def step(self, action, render: bool = False, noise=None):
        if len(self.actions) % self.CTRL_CHECK == 0:
            self.watch_controller()
        state = _flat(self._raw.get_states())
        if not self.states:
            for k, v in state.items():
                if v.ndim < 1 or v.shape[0] != self._num_envs:
                    raise RuntimeError(
                        f"state leaf {k!r} is not env-batched at the first step: "
                        f"shape={tuple(v.shape)}, expected axis 0 == num_envs "
                        f"({self._num_envs}). Fix the scene, robot, controller, or "
                        "solve that collapsed the environment dimension."
                    )
            if action.ndim < 1 or action.shape[0] != self._num_envs:
                raise RuntimeError(
                    "commanded action is not env-batched at the first step: "
                    f"shape={tuple(action.shape)}, expected axis 0 == num_envs "
                    f"({self._num_envs}). The solve must emit one action row per env."
                )
        # THE LABEL: recorded before any perturbation exists in this scope.
        self.states.append(state)
        self.actions.append(action.detach().cpu().clone())
        executed = action
        if noise is not None and self._noise_scale != 0.0:
            if tuple(noise.shape) != tuple(action.shape):
                raise RuntimeError(
                    f"noise shape {tuple(noise.shape)} must match the action shape "
                    f"{tuple(action.shape)} — one perturbation row per env."
                )
            scaled = (self._noise_scale * noise).detach()
            executed = action + scaled.to(action.device, action.dtype)
            self.noises.append(scaled.cpu().clone())
        else:
            self.noises.append(None)  # densified to zeros at save time
        if self._success_probe is not None and \
                len(self.actions) % self.PROBE_EVERY == 0:
            self.success_samples.append(
                (len(self.actions), self._success_probe())
            )
        ret = self._env.step(executed, render)
        # The scripted uniform wrapper (NoisyActionEnv) perturbs BELOW this recorder. Fold
        # the perturbation it executed into the same `action_noise` channel, so the replay
        # gate (which replays action + action_noise) and every consumer see what actually
        # ran; without this a --sigma batch can never pass the gate.
        w_clean = getattr(self._env, "last_clean", None)
        w_exec = getattr(self._env, "last_executed", None)
        if w_clean is not None and w_exec is not None and w_exec is not w_clean:
            extra = (w_exec - w_clean).detach().cpu().clone()
            if bool(extra.abs().sum() > 0):
                self.noises[-1] = extra if self.noises[-1] is None else self.noises[-1] + extra
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
              noise_scale: float = 0.0, noise: dict | None = None,
              device: str = "cuda:0",
              env_draw: int = 0, solve_draw: int = 0, nominal: bool = False,
              solo_draw: bool = False, phys_nominal: bool = False,
              solve_nominal: bool = False) -> Path:
    import numpy as np
    import torch

    from .sampler import sample, solve_bands

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
    # Delivered solutions are FOLDERS (init copies "solve.py plus its siblings"), and real
    # solves import those siblings bare (`from helpers import ...`) — the eval harness ran
    # them with the solution dir as sys.path[0]. Loading by file path skips that, so put the
    # strategy dir (and the phase cell dir, whose port may have its own siblings) on sys.path.
    for extra in ([strategy_dir / "phases" / phase] if phase else []) + [strategy_dir]:
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
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

    verdicts_all = []
    noise_rows = noise_rows_perturbed = 0
    noise_abs_sum, noise_abs_max, noise_elems = 0.0, 0.0, 0
    # one rollout per reset file (no phase = a single rollout) — every entry
    # covered once per batch; more episodes = more batches
    rollouts = max(1, len(conditions))
    for rnd in range(rollouts):
        env.reset(seed=seed + rnd)
        reset_name, fn_of_env = None, None
        if conditions:
            cond = conditions[rnd % len(conditions)]
            reset_name = cond.__name__.removeprefix("datagen_reset_")
            fn_of_env = run_reset_builders(env, cond, num_envs)
            entry = reset_name if has_port else None  # the file IS the phase
        grader = grader_cls(env)
        grader.setup()  # baselines captured at the entry state
        # VACUOUS-SUCCESS GUARD: an env the grader already judges successful AT
        # THE ENTRY STATE can never yield a valid episode — there is no
        # transition to learn. Without this, a scene whose success() holds at
        # reset plus a solve that exits immediately harvests unlimited "successes"
        # (pc_motherboard shipped 1-step score-1.0 episodes for two days).
        entry_success = [bool(v["success"]) for v in grader.verdict()]
        if any(entry_success):
            print(f"[batch {batch}] WARNING: {sum(entry_success)}/{num_envs} envs "
                  "satisfy the grader AT ENTRY — their episodes are voided as "
                  "vacuous. If this is every env, the scene/grader pair is "
                  "broken for data generation: fix the success predicate.",
                  flush=True)
        # Two independent, label-clean noise mechanisms:
        # - the solve-authored channel (env.step(action, noise=...)), scaled by
        #   noise_scale — the pipeline DEFAULT (0 = inert; the harvest runs it hot);
        # - the scripted uniform wrapper (NoisyActionEnv), opt-in through the
        #   --sigma/--prob/--duration/--dims/--gate-z flags. It perturbs BELOW
        #   the recorder, so recorded actions stay the clean commands either way.
        target = env
        if noise and noise.get("sigma", 0.0) > 0.0:
            dims = noise.get("dims")
            target = NoisyActionEnv(env, dims=slice(*dims) if dims else slice(0, 0),
                                    sigma=noise.get("sigma", 0.0),
                                    prob=noise.get("prob", 1.0),
                                    duration=noise.get("duration", 0.0),
                                    seed=seed + rnd,
                                    gate_z=noise.get("gate_z", 0.0))
        rec = Recorder(
            target, env, num_envs, noise_scale=noise_scale,
            success_probe=lambda: [bool(v["success"]) for v in grader.verdict()],
        )
        print(f"[batch {batch}] rollout {rnd + 1}/{rollouts}"
              + (f" ({reset_name})" if reset_name else "")
              + f": solve on {num_envs} envs"
              + (f", noise_scale={noise_scale}" if noise_scale else "") + " …",
              flush=True)
        solve(rec) if entry is None else solve(rec, entry=entry)

        verdicts = grader.verdict()
        for e, v in enumerate(verdicts):
            if entry_success[e] and v["success"]:
                v["success"] = False
                v["vacuous"] = True  # rides into ep + batch metas as provenance
        ctrl_info = _controller_info(env.robot)  # after the solve = overrides included
        T = len(rec.actions)
        rec.watch_controller(step=T)  # catch a change in the last <CTRL_CHECK latches
        if rec.ctrl_changes:
            print(f"[batch {batch}] WARNING: control law changed MID-SOLVE at steps "
                  f"{[c['step'] for c in rec.ctrl_changes]} — the stamped `controller` is the "
                  f"FINAL law; per-change diffs are in meta `controller_changes`", flush=True)
        # Every recorded state leaf must be env-batched (dim 1 = num_envs) for
        # per-env save/replay. Never replicate a malformed leaf: that can turn
        # one environment's controller or scene state into apparently valid but
        # incorrect data for every row.
        arrays = {}
        for k in rec.states[0]:
            v = np.stack([s[k].numpy() for s in rec.states])
            if v.ndim < 2 or v.shape[1] != num_envs:
                raise RuntimeError(
                    f"recorded state leaf {k!r} is not env-batched: shape={v.shape}, "
                    f"expected axis 1 == num_envs ({num_envs}). Fix the scene, robot, "
                    "controller, or solve that collapsed the environment dimension."
                )
            arrays[k] = v
        arrays["action"] = np.stack([a.numpy() for a in rec.actions])
        if arrays["action"].ndim < 2 or arrays["action"].shape[1] != num_envs:
            raise RuntimeError(
                "recorded action is not env-batched: "
                f"shape={arrays['action'].shape}, expected axis 1 == "
                f"num_envs ({num_envs}). The solve must emit one action row per env."
            )
        if len(rec.joint_targets) == T:  # commanded-target channel (see Recorder.step)
            arrays["robot/joint_target"] = np.stack([t.numpy() for t in rec.joint_targets])
        # The executed perturbation, verbatim (zeros on clean steps): action labels
        # stay clean by construction; every derived channel (joint targets, achieved
        # state) reflects the noisy execution, and this array is what lets any
        # consumer reconstruct the clean command in a derived space.
        if any(n is not None for n in rec.noises):
            zero = np.zeros_like(arrays["action"][0])
            arrays["action_noise"] = np.stack(
                [n.numpy() if n is not None else zero for n in rec.noises])
            row_norms = np.abs(arrays["action_noise"]).sum(
                axis=tuple(range(2, arrays["action_noise"].ndim)))
            noise_rows_perturbed += int((row_norms > 0).sum())
            noise_abs_sum += float(np.abs(arrays["action_noise"]).sum())
            noise_abs_max = max(noise_abs_max, float(np.abs(arrays["action_noise"]).max()))
            noise_elems += arrays["action_noise"].size
        noise_rows += T * num_envs
        success_steps = rec.success_steps([bool(v["success"]) for v in verdicts])
        for e in range(num_envs):
            ep = rnd * num_envs + e
            ep_dir = out / f"ep_{ep:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            # A finished env is parked while the batch's slowest env works, and every
            # row of that tail was recorded (pen_holder: success at step 1170 of a
            # 15,995-step episode). Keep one success-probe interval past the earliest
            # SUSTAINED-success sample (the grader still passes at the cut, and replay,
            # render and storage stop paying ~10x for a robot standing still).
            keep = T
            if success_steps[e] is not None:
                keep = min(T, int(success_steps[e]) + rec.PROBE_EVERY)
            np.savez_compressed(ep_dir / "traj.npz",
                                **{k: v[:keep, e] for k, v in arrays.items()})
            meta = {
                "episode": ep, "rollout": rnd, "env_index": e,
                "success": verdicts[e]["success"], "score": verdicts[e]["score"],
                # the draws this episode ran under ({} = nominal on that axis)
                "parameters": {"physical": slot_drawn[e] if slot_drawn else {},
                               "solve": solve_drawn},
                "reset": reset_name, "reset_fn": (fn_of_env[e] if fn_of_env else None),
                "entry": entry,
                "seed": seed + rnd, "steps": keep, "batch_steps": T,
                # earliest sustained-success step (None = failed / never sampled);
                # the saved trajectory ends PROBE_EVERY rows after it
                "success_step": success_steps[e],
                "sim_dt": env.dt, "decimation": env.robot.control_period,
                "controller": ctrl_info,
                "controller_changes": rec.ctrl_changes,
                # noise provenance: scale the pipeline enabled + whether THIS episode
                # actually saw perturbed steps (the verbatim vectors live in traj.npz
                # `action_noise`; action labels are clean by construction)
                "noise": ({"scale": noise_scale,
                           "perturbed": bool("action_noise" in arrays
                                             and np.abs(arrays["action_noise"][:, e]).sum() > 0)}
                          if (noise_scale or "action_noise" in arrays) else {}),
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
    from .contract import cell_fingerprint
    (out / "meta.json").write_text(json.dumps({
        "batch": batch, "cell": cell,
        # identity of the code that produced these episodes (see contract.cell_fingerprint)
        "cell_fingerprint": cell_fingerprint(real_root, cell),
        "preset": gen["preset"], "num_envs": num_envs, "seed": seed,
        # measured noise coverage — the orchestrator's mandatory-noise gate reads
        # this (an agent that ships zero effective noise cannot pass certification)
        "noise": ({"scale": noise_scale,
                   "perturbed_row_frac": round(noise_rows_perturbed / max(1, noise_rows), 4),
                   "mean_abs": round(noise_abs_sum / max(1, noise_elems), 6),
                   "max_abs": round(noise_abs_max, 6)}
                  if (noise_scale or noise_rows_perturbed) else {}),
        # the scripted uniform wrapper's config, when enabled (its perturbation
        # happens below the recorder, so it is provenance, not measured coverage)
        "uniform_noise": (noise if noise and noise.get("sigma", 0.0) > 0.0 else {}),
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
