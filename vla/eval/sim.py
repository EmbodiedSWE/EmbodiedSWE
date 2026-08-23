"""load_sim — build an eval sim from a preset name, a registered sim, or a bake stamp.

See DESIGN.md. Three sources, one build path: the robobench preset defines the
world; the spec (hand-written or derived from a dataset's meta/bake.json)
defines what an action means on top of it; the loader attaches the scene's and
robot's declared cameras so eval pixels come from the same views as training
renders. Returns an EvalSim facade: obs = {images, state, success} with
state = [arm q…, gripper closedness], the exact vla/convert layout.

Module imports are app-free; load_sim() needs a running AppLauncher.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, fields as _dc_fields, replace
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "data_engine"), str(_REPO / "vla" / "convert")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import episode as _convert  # vla/convert: FINGER_TRAVEL, _FINGER_MARKERS, _goal_sentence

_JOINT_SPACES = ("joint_pos", "joint_vel")


# ----- the spec + the local registry --------------------------------------------------------------
@dataclass
class SimSpec:
    """A fully explicit eval-sim setup. Registered sims write one by hand (bake-free);
    stamp_to_spec derives one from a dataset's bake.json."""

    preset: str                                   # world: the ENVS name
    control_space: str | None = None              # raw_cmd | joint_pos | joint_vel; None = preset controller
    control_freq_hz: float | None = None          # latch rate for the joint conventions
    finger_drives: tuple[float, float] | None = None   # (stiffness, damping) written onto the finger joints
    tracker_gains: tuple[float, float] | None = None   # arm PD override (joint conventions); None = preset's
    cams: tuple[str, ...] | None = None           # subset of declared views; None = all
    size: tuple[int, int] = (640, 480)            # (W, H), the render contract's default
    warmup: int = 12                              # hold-steps after any init (settle + denoiser flush)
    physical_params: dict | None = None           # a PHYSICAL_PARAMS draw to re-apply (episode meta's
                                                  # parameters.physical); None/{} = nominal
    grip_margin: float = 0.0                      # m of finger closure commanded BEYOND the closedness
                                                  # label when gripping — achieved-width labels carry no
                                                  # squeeze force (convert README caveat 1); this is the
                                                  # executor-side clamp-force restoration
    stamp: dict | None = field(default=None, repr=False)  # full bake dict when bake-derived


SIMS: dict[str, Callable[[], SimSpec]] = {}


def register_sim(name: str, factory: Callable[[], SimSpec]) -> str:
    if name in SIMS:
        raise KeyError(f"sim '{name}' already registered")
    SIMS[name] = factory
    return name


def stamp_to_spec(bake_path: str | Path) -> SimSpec:
    """Derive a SimSpec from a dataset's meta/bake.json — ALL settings from the one stamp."""
    stamp = json.loads(Path(bake_path).read_text())
    finger_drives = None
    ctrl = stamp.get("controller")
    if ctrl:
        names = ctrl["joint_names"]
        f_idx = [i for i, n in enumerate(names) if _is_finger(n)]
        if f_idx:
            ks = {round(float(ctrl["joint_stiffness"][i]), 6) for i in f_idx}
            kd = {round(float(ctrl["joint_damping"][i]), 6) for i in f_idx}
            if len(ks) > 1 or len(kd) > 1:
                raise SystemExit(f"{bake_path}: non-uniform finger drives {ks}/{kd} — extend SimSpec first")
            finger_drives = (ks.pop(), kd.pop())
    return SimSpec(
        preset=stamp["robot_type"],
        control_space=stamp["control_space"],
        control_freq_hz=float(stamp["control_freq_hz"]),
        finger_drives=finger_drives,
        stamp=stamp,
    )


def _is_finger(joint_name: str) -> bool:
    return any(m in joint_name.lower() for m in _convert._FINGER_MARKERS)


# ----- source resolution ---------------------------------------------------------------------------
def _resolve_spec(source: str | Path | SimSpec) -> SimSpec:
    if isinstance(source, SimSpec):
        return source
    import specs  # noqa: F401  (vla/eval/specs — importing registers the named sims)

    s = str(source)
    if s in SIMS:
        return SIMS[s]()
    if s.endswith(".json") or Path(s).is_file():
        if not Path(s).is_file():
            raise SystemExit(f"bake file not found: {s}")
        return stamp_to_spec(s)
    return SimSpec(preset=s)  # preset existence is checked against ENVS at build


def _joint_sibling(preset: str, envs) -> str:
    """The `.joint` variant of a preset (joint conventions need position-PD arm actuators)."""
    if preset.endswith(".joint"):
        return preset
    parts = preset.split(".")
    for cand in (".".join(parts[:-1] + ["joint"]), preset + ".joint"):
        if cand in envs.list():
            return cand
    raise SystemExit(f"no `.joint` sibling of '{preset}' registered (needed for joint_pos/joint_vel "
                     f"tracking). Known: {[n for n in envs.list() if n.startswith(parts[0])]}")


# ----- the build path ------------------------------------------------------------------------------
def load_sim(source: str | Path | SimSpec, *, num_envs: int = 1, device: str = "cuda:0",
             **overrides: Any) -> "EvalSim":
    """Build the eval sim. `source`: registered sim name | ENVS preset name | bake.json path |
    SimSpec. `overrides` patch SimSpec fields (overrides of bake-derived values are echoed)."""
    spec = _resolve_spec(source)
    if overrides:
        known = {f.name for f in _dc_fields(SimSpec)}
        bad = set(overrides) - known
        if bad:
            raise SystemExit(f"unknown SimSpec overrides {sorted(bad)}; known: {sorted(known)}")
        for k, v in overrides.items():
            old = getattr(spec, k)
            if spec.stamp is not None and old != v:
                print(f"[load_sim] OVERRIDING bake-derived {k}: {old!r} -> {v!r}", flush=True)
        spec = replace(spec, **overrides)
    if spec.control_space is None and spec.stamp is None:
        print("[load_sim] WARNING: control law = preset defaults, not calibrated to any dataset",
              flush=True)

    import robobench

    robobench.discover()
    from robobench.core.registries import ENVS, ROBOTS, SCENES

    name = spec.preset
    if spec.control_space in _JOINT_SPACES:
        if spec.control_freq_hz is None:
            raise SystemExit(f"control_space={spec.control_space} needs control_freq_hz")
        name = _joint_sibling(name, ENVS)
        if name != spec.preset:
            print(f"[load_sim] {spec.preset} -> {name} (position-PD arm for {spec.control_space})",
                  flush=True)
    if name not in ENVS.list():
        raise SystemExit(f"env '{name}' not registered. Known: {ENVS.list()}")

    env_cfg = ENVS.get(name)()
    scene_cls = SCENES.get(env_cfg.scene)
    robot_cls = ROBOTS.get(env_cfg.robot)
    if spec.physical_params:
        # generation's two-moment application: values onto the scene cfg BEFORE the build
        # (build-consumed knobs), the live subset re-applied after (apply_physical_params)
        if env_cfg.scene_cfg is None:
            env_cfg.scene_cfg = scene_cls().cfg
        for k, v in spec.physical_params.items():
            if not hasattr(env_cfg.scene_cfg, k):
                raise SystemExit(f"physical param '{k}' not a {type(env_cfg.scene_cfg).__name__} field")
            setattr(env_cfg.scene_cfg, k, v)
        print(f"[load_sim] physical params (episode draw): {spec.physical_params}", flush=True)
    env = _build_with_cameras(env_cfg, scene_cls, robot_cls, spec, num_envs, device)
    if spec.physical_params:
        env.scene.apply_physical_params(env, {k: [v] * num_envs
                                              for k, v in spec.physical_params.items()})
    views = env._eval_views  # stashed by _build_with_cameras
    sensors = {n: env.iscene.sensors[n] for n in views}

    arm_ids, arm_names, finger_ids, finger_names = _split_joints(env)
    _validate_against_stamp(spec, env, arm_names)
    rate = spec.control_freq_hz

    if spec.control_space in _JOINT_SPACES:
        _install_joint_tracker(env, arm_names, finger_names, rate)
        period = (1.0 / rate) / env.dt
        if abs(period - round(period)) > 1e-6 or round(period) < 1:
            raise SystemExit(f"control rate {rate} Hz is not an integer multiple of sim dt "
                             f"{env.dt} (period {period})")
        if env.robot.control_period != round(period):
            raise SystemExit(f"tracker latched at period {env.robot.control_period}, "
                             f"expected {round(period)}")
    elif spec.control_space == "raw_cmd":
        _apply_stamp_controller(env, (spec.stamp or {}).get("controller"))

    _write_drives(env, spec, arm_ids, finger_ids)
    grader_cls = _find_grader(name, env_cfg.scene)
    return EvalSim(env, spec, views, sensors, arm_ids, arm_names, finger_ids, finger_names,
                   grader_cls=grader_cls)


def _find_grader(env_name: str, scene: str):
    """The suite's grader class for this scene (success is grader-defined by design).
    None when the suite ships no grader for it — obs fall back to scene.success()."""
    import importlib

    suite = env_name.split(".")[0]
    try:
        graders = importlib.import_module(f"robobench.suites.{suite}.grader").GRADERS
        cls = graders.get(scene)
    except (ImportError, AttributeError):
        cls = None
    print(f"[load_sim] success check: "
          + (f"grader {cls.__name__}" if cls else "no suite grader — scene.success() fallback"),
          flush=True)
    return cls


def _build_with_cameras(env_cfg, scene_cls, robot_cls, spec: SimSpec, num_envs: int, device: str):
    """env_cfg.build with one TiledCamera per declared view injected before the build
    (external views on the scene's assets, ego views on the robot's — replay.py's recipe)."""
    import math as _math

    from engine.replay import _FAR_CLIP, _camera_cfg, resolve_views

    views = resolve_views(scene_cls, robot_cls, list(spec.cams) if spec.cams else None, None)
    scene_probe = scene_cls(env_cfg.scene_cfg) if env_cfg.scene_cfg is not None else scene_cls()
    surface_z = float(getattr(scene_probe.cfg, "surface_z", 0.0) or 0.0)
    cam_cfgs = {n: _camera_cfg(n, v, spec.size, surface_z) for n, v in views.items()}
    static_cfgs = {n: c for n, c in cam_cfgs.items() if not views[n].get("link")}
    ego_cfgs = {n: c for n, c in cam_cfgs.items() if views[n].get("link")}

    env_spacing = env_cfg.env_spacing
    if num_envs > 1:
        env_spacing = 50.0  # beyond the camera far clip: one standalone robot per frame
    ground_xy = (_math.ceil(_math.sqrt(num_envs)) - 1) * env_spacing + 2 * (_FAR_CLIP + 10.0)

    def assets_with_camera(self, _orig=scene_cls.assets):
        from isaaclab.sim import GroundPlaneCfg

        out = {**_orig(self), **static_cfgs}
        for asset in out.values():
            spawn = getattr(asset, "spawn", None)
            if isinstance(spawn, GroundPlaneCfg) and max(spawn.size) < ground_xy:
                spawn.size = (ground_xy, ground_xy)
        return out

    def assets_with_ego(self, _orig=robot_cls.assets):
        return {**_orig(self), **ego_cfgs}

    scene_cls.assets = assets_with_camera
    robot_cls.assets = assets_with_ego
    try:
        env = env_cfg.build(num_envs=num_envs, device=device, env_spacing=env_spacing)
    finally:
        scene_cls.assets = assets_with_camera.__defaults__[0]
        robot_cls.assets = assets_with_ego.__defaults__[0]
    env._eval_views = views
    print(f"[load_sim] views: " + ", ".join(
        f"{n} (ego on {v['link']})" if v.get("link") else n for n, v in views.items()), flush=True)
    return env


def _split_joints(env):
    names = list(env.robot.articulation.joint_names)
    finger_ids = [i for i, n in enumerate(names) if _is_finger(n)]
    arm_ids = [i for i in range(len(names)) if i not in finger_ids]
    for i in finger_ids:
        if names[i] not in _convert.FINGER_TRAVEL:
            raise SystemExit(f"gripper joint '{names[i]}' not in vla/convert episode.FINGER_TRAVEL")
    return arm_ids, [names[i] for i in arm_ids], finger_ids, [names[i] for i in finger_ids]


def _validate_against_stamp(spec: SimSpec, env, arm_names) -> None:
    stamp = spec.stamp
    if stamp is None:
        return
    ctrl = stamp.get("controller")
    live = list(env.robot.articulation.joint_names)
    if ctrl and ctrl.get("joint_names") and ctrl["joint_names"] != live:
        raise SystemExit(f"joint order drift: stamp {ctrl['joint_names']} vs env {live}")
    want_state = arm_names + ["gripper"]
    if stamp.get("state_names") and stamp["state_names"] != want_state:
        raise SystemExit(f"state layout drift: stamp {stamp['state_names']} vs env {want_state}")
    if spec.control_space in _JOINT_SPACES and stamp.get("action_names") and \
            len(stamp["action_names"]) != len(arm_names) + 1:
        raise SystemExit(f"action width drift: stamp {len(stamp['action_names'])} vs "
                         f"env {len(arm_names) + 1}")


def _install_joint_tracker(env, arm_names, finger_names, rate_hz: float) -> None:
    """One env.step = one position latch at the stamped rate: arm + fingers, both position."""
    from robobench.controllers import CompositeController, JointController, JointControllerCfg

    dt = 1.0 / rate_hz
    env.robot.set_controller(CompositeController([
        JointController(JointControllerCfg(tuple(arm_names), dt=dt), command_type="position"),
        JointController(JointControllerCfg(tuple(finger_names), dt=dt), command_type="position"),
    ]))


def _write_drives(env, spec: SimSpec, arm_ids, finger_ids) -> None:
    art = env.robot.articulation
    if spec.finger_drives is not None:
        stiff, damp = spec.finger_drives
        art.write_joint_stiffness_to_sim(float(stiff), joint_ids=finger_ids)
        art.write_joint_damping_to_sim(float(damp), joint_ids=finger_ids)
        print(f"[load_sim] finger drives: kp={stiff} kd={damp}", flush=True)
    if spec.tracker_gains is not None:
        if spec.control_space not in _JOINT_SPACES:
            raise SystemExit("tracker_gains only apply to joint_pos/joint_vel")
        kp, kd = spec.tracker_gains
        art.write_joint_stiffness_to_sim(float(kp), joint_ids=arm_ids)
        art.write_joint_damping_to_sim(float(kd), joint_ids=arm_ids)
        print(f"[load_sim] tracker gains (explicit): kp={kp} kd={kd}", flush=True)
    elif spec.control_space in _JOINT_SPACES and spec.stamp and spec.stamp.get("controller"):
        # data collected under joint control stamps REAL arm drives (the actual tracking
        # PD the demos ran under) — use them; torque-mode stamps record zeros -> keep the
        # .joint preset's PD (there were no arm position gains to record)
        import torch

        ctrl = spec.stamp["controller"]
        kp = torch.tensor([ctrl["joint_stiffness"][i] for i in arm_ids], dtype=torch.float32,
                          device=env.device)
        kd = torch.tensor([ctrl["joint_damping"][i] for i in arm_ids], dtype=torch.float32,
                          device=env.device)
        if bool((kp > 0).any()):
            art.write_joint_stiffness_to_sim(kp.repeat(env.num_envs, 1), joint_ids=arm_ids)
            art.write_joint_damping_to_sim(kd.repeat(env.num_envs, 1), joint_ids=arm_ids)
            print(f"[load_sim] tracker gains (stamped arm drives): kp={kp.tolist()} "
                  f"kd={kd.tolist()}", flush=True)


def _apply_stamp_controller(env, ctrl_block: dict | None) -> None:
    """raw_cmd: re-apply the stamped control law onto the preset-built controller.
    The stamped values were live solve-time overrides, not preset defaults. Leaf classes
    must match (a runtime-swapped custom controller needs its mode registered first).
    NOTE: certified by the replay check before use — cfg fields cached at construction
    (e.g. JointController.scale) are also re-set on the live object where present."""
    if not ctrl_block:
        raise SystemExit("raw_cmd needs a bake with a stamped controller block "
                         "(controller: null bakes predate stamping)")
    import torch

    ctrl = env.robot.controller
    live_leaves = getattr(ctrl, "controllers", [ctrl])
    stamped = ctrl_block["leaves"]
    live_cls = [type(l).__name__ for l in live_leaves]
    if live_cls != [s["class"] for s in stamped]:
        raise SystemExit(f"controller class drift: stamp {[s['class'] for s in stamped]} vs "
                         f"preset {live_cls} — register the mode that built this data")
    for leaf, st in zip(live_leaves, stamped):
        if st.get("control_period"):  # the recorded latch rate, not the preset's (campaigns retune it)
            if leaf.control_period != int(st["control_period"]):
                print(f"[load_sim] stamp {type(leaf).__name__}.control_period: "
                      f"{leaf.control_period} -> {st['control_period']}", flush=True)
            leaf._control_period = int(st["control_period"])
            if leaf.cfg is not None and hasattr(leaf.cfg, "dt"):
                # keep cfg.dt consistent with the applied period (dt is inert after
                # bind, but a stale value misleads readers and any future re-bind)
                leaf.cfg.dt = env.dt * leaf._control_period
        for k, v in (st.get("cfg") or {}).items():
            if v is None or k in ("dt", "joint_names", "arm_joint_names", "ee_body"):
                continue
            val = tuple(v) if isinstance(v, list) else v
            for holder in (leaf.cfg, leaf):  # cfg + any construction-time cache of it
                if holder is not None and hasattr(holder, k) and getattr(holder, k) != val:
                    print(f"[load_sim] stamp {type(leaf).__name__}.{k}: "
                          f"{getattr(holder, k)!r} -> {val!r}", flush=True)
                    setattr(holder, k, val)
        for gain, attr in (("kp", "_kp"), ("kd", "_kd")):  # instance tensors, built at bind
            if st.get(gain) is not None:
                cur = getattr(leaf, attr, None)
                if cur is None:
                    raise SystemExit(f"stamp carries {gain} for {type(leaf).__name__} "
                                     f"but the live leaf has no {attr} tensor")
                new = torch.as_tensor(st[gain], dtype=cur.dtype, device=cur.device)
                if not torch.equal(new.broadcast_to(cur.shape), cur):
                    print(f"[load_sim] stamp {type(leaf).__name__}.{attr}: "
                          f"{cur.flatten().tolist()} -> {st[gain]}", flush=True)
                setattr(leaf, attr, new.broadcast_to(cur.shape).clone())
        # bind-cached tensors that follow cfg fields we may just have changed
        cfg = leaf.cfg
        if cfg is not None and hasattr(leaf, "_q_default") and getattr(cfg, "nullspace_dof_pos", ()):
            leaf._q_default = torch.tensor(cfg.nullspace_dof_pos, device=leaf._q_default.device)
        if cfg is not None and getattr(cfg, "ema_factor", 1.0) < 1.0 and \
                getattr(leaf, "_prev_action", "absent") is None:
            raise SystemExit(f"stamp enables EMA on {type(leaf).__name__} but the preset built "
                             f"it without a smoothing buffer — rebuild the mode, don't patch")
    art = env.robot.articulation
    for key, writer in (("joint_stiffness", art.write_joint_stiffness_to_sim),
                        ("joint_damping", art.write_joint_damping_to_sim)):
        if ctrl_block.get(key):
            vals = torch.tensor(ctrl_block[key], dtype=torch.float32,
                                device=env.device).repeat(env.num_envs, 1)
            writer(vals)


# ----- the handle ----------------------------------------------------------------------------------
class EvalSim:
    """Facade over the built BaseEnv (raw env at .env). One step() = one control latch,
    rendered; obs mirror the bake layout exactly."""

    def __init__(self, env, spec: SimSpec, views: dict, sensors: dict,
                 arm_ids, arm_names, finger_ids, finger_names, grader_cls=None) -> None:
        import torch

        self.env, self.spec, self.views, self.sensors = env, spec, views, sensors
        self.arm_ids, self.arm_names = arm_ids, arm_names
        self.finger_ids, self.finger_names = finger_ids, finger_names
        self._grader_cls, self.grader = grader_cls, None
        self.travel = torch.tensor([_convert.FINGER_TRAVEL[n] for n in finger_names],
                                   dtype=torch.float32, device=env.device)
        self.rate_hz = spec.control_freq_hz or 1.0 / (env.dt * env.robot.control_period)
        self.task = _convert._goal_sentence(env.scene.describe())
        self.state_names = arm_names + ["gripper"]
        self._base_pos = None  # robot root anchor for episode-state shifting, set lazily

    # ----- obs ------------------------------------------------------------------------------------
    def _q(self):
        return self.env.robot.articulation.data.joint_pos

    def _closedness(self):
        q = self._q()
        frac = (q[:, self.finger_ids] / self.travel).mean(dim=1).clamp(0.0, 1.0)
        return 1.0 - frac

    def obs(self) -> dict:
        import torch

        q = self._q()
        state = torch.cat([q[:, self.arm_ids], self._closedness().unsqueeze(1)], dim=1)
        images = {}
        for n, cam in self.sensors.items():
            cam.update(0.0, force_recompute=True)
            images[n] = cam.data.output["rgb"].cpu().numpy()
        if self.grader is not None:
            success = self.grader.check_success().cpu().numpy().astype(bool)
            # the grader's weighted rubric progress (0..1) — THE per-episode score;
            # called once per obs = the grader's designed cadence
            progress = self.grader.progress().cpu().numpy().astype(np.float32)
        elif hasattr(self.env.scene, "success"):
            success = self.env.scene.success().cpu().numpy().astype(bool)
            progress = success.astype(np.float32)
        else:
            success = np.zeros(self.env.num_envs, dtype=bool)
            progress = np.zeros(self.env.num_envs, dtype=np.float32)
        return {"images": images,
                "state": state.float().cpu().numpy(),
                "success": success,
                "progress": progress}

    # ----- stepping -------------------------------------------------------------------------------
    def step(self, action) -> dict:
        """Policy-convention action, (E, A) or (A,). joint_pos: absolute arm targets + closedness.
        joint_vel: q_live + v/rate + closedness. raw_cmd/None: passed to env.step verbatim."""
        import torch

        a = torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        cs = self.spec.control_space
        if cs in _JOINT_SPACES:
            q_arm = a[:, :-1] if cs == "joint_pos" else \
                self._q()[:, self.arm_ids] + a[:, :-1] / self.rate_hz
            return self.step_targets(q_arm, a[:, -1])
        self.env.step(a, render=True)
        return self.obs()

    def step_targets(self, q_arm, closedness) -> dict:
        """The layer step() lands in: absolute arm targets + closedness -> one rendered latch.
        `spec.grip_margin` biases the finger target past the labeled width (squeeze force)."""
        import torch

        fingers = ((1.0 - torch.as_tensor(closedness, dtype=torch.float32,
                                          device=self.env.device).clamp(0.0, 1.0)
                    ).unsqueeze(-1) * self.travel - self.spec.grip_margin).clamp_min(0.0)
        self.env.step(torch.cat([torch.as_tensor(q_arm, dtype=torch.float32,
                                                 device=self.env.device), fingers], dim=1),
                      render=True)
        return self.obs()

    def hold_action(self) -> "np.ndarray":
        """The identity action: holds the current pose under the active control law."""
        import torch

        cs = self.spec.control_space
        q = self._q()
        # compensate grip_margin so holding is a fixed point, not a per-latch ratchet
        closed = (self._closedness() - self.spec.grip_margin / float(self.travel.mean())
                  ).clamp(0.0, 1.0).unsqueeze(1)
        if cs == "joint_pos":
            return torch.cat([q[:, self.arm_ids], closed], dim=1).cpu().numpy()
        if cs == "joint_vel":
            return torch.cat([torch.zeros_like(q[:, self.arm_ids]), closed], dim=1).cpu().numpy()
        # preset controller: per leaf, zeros for task-space deltas, current q for joint leaves
        from robobench.controllers import JointController

        ctrl = self.env.robot.controller
        parts = []
        for leaf in getattr(ctrl, "controllers", [ctrl]):
            if isinstance(leaf, JointController):
                parts.append(q[:, leaf.joint_ids])  # identity shaping assumed (scale 1, offset 0)
            else:
                parts.append(torch.zeros((self.env.num_envs, leaf.action_dim),
                                         device=self.env.device))
        return torch.cat(parts, dim=1).cpu().numpy()

    # ----- init -----------------------------------------------------------------------------------
    def _setup_grader(self) -> None:
        """Fresh grader per episode (once-milestones and baselines reset with it)."""
        if self._grader_cls is not None:
            self.grader = self._grader_cls(self.env)
            self.grader.setup()

    def _warmup(self) -> dict:
        obs = self.obs()
        for _ in range(max(0, self.spec.warmup)):
            obs = self.step(self.hold_action())
        return obs

    def reset(self, seed: int | None = None) -> dict:
        self.env.reset(seed=seed)
        if self._base_pos is None:
            self._base_pos = self.env.robot.articulation.data.root_pos_w.clone()
        self._setup_grader()
        return self._warmup()

    def init_from_episode(self, ep_dirs: str | Path | list, t0: int | list[int] = 0) -> dict:
        """Restore recorded episodes' states at row `t0` (default: the start), one per env
        slot (origin-shifted). A single dir fills every slot; a list assigns episode i ->
        slot i (unused slots repeat the last episode). `t0` may be per-slot. Mid-episode
        starts are exact: the traj records the full restorable state at every tick."""
        import torch

        from engine.replay import _unflatten

        dirs = [Path(p) for p in (ep_dirs if isinstance(ep_dirs, (list, tuple)) else [ep_dirs])]
        if len(dirs) > self.env.num_envs:
            raise SystemExit(f"{len(dirs)} episodes but only {self.env.num_envs} envs")
        dirs += [dirs[-1]] * (self.env.num_envs - len(dirs))
        t0s = list(t0) if isinstance(t0, (list, tuple)) else [t0] * len(dirs)
        t0s += [t0s[-1]] * (len(dirs) - len(t0s))
        if self._base_pos is None:
            self.env.reset(seed=0)
            self._base_pos = self.env.robot.articulation.data.root_pos_w.clone()
        datas = [np.load(d / "traj.npz") for d in dirs]
        keys = [k for k in datas[0].files if k != "action" and not k.startswith("robot/controller")]
        for d, dd in zip(dirs[1:], datas[1:]):
            if {k for k in dd.files if k != "action"} != {k for k in datas[0].files if k != "action"}:
                raise SystemExit(f"{d}: state keys differ from {dirs[0]} — mixed scenes?")
        flat = {}
        for k in keys:
            rows = []
            for e, dd in enumerate(datas):
                t = min(t0s[e], len(dd[k]) - 1)
                row = torch.as_tensor(dd[k][t], device=self.env.device)
                if row.shape[-1:] == (13,):  # world-frame root state: shift onto this slot's origin
                    row = row.clone()
                    shift = self._base_pos[e] - torch.as_tensor(dd["robot/root"][t, 0:3],
                                                                device=self.env.device)
                    row[..., 0:3] += shift
                rows.append(row)
            flat[k] = torch.stack(rows)
        self.env.set_states(_unflatten(flat))
        if self.env.robot.controller is not None:
            self.env.robot.controller.reset(None)
        self._setup_grader()
        return self._warmup()

    def close(self) -> None:
        self.env.close()
