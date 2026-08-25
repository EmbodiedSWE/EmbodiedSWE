"""BaseEnv — the open, GPU-batched env: Sim + BaseScene + BaseRobot.

Open object (not a sealed gym box): the agent reaches `.sim .scene .robot .iscene .stage .cfg`,
raw asset handles, `.data` tensors, and `pxr`, and may patch/replace parts in place.
`step()` dispatches through `self.scene` / `self.robot` (late binding), so patches/swaps take
effect immediately. Goal-agnostic: success is the hidden, harness-side grader's call.

Composition by injection: pass built `scene`/`robot` (+ a `sim_cfg`); switching embodiment = a new
env with a different robot. Constructing this needs `AppLauncher` running (isaaclab imported
lazily), so importing this module stays app-free. The Isaac build wiring is verified when the
first concrete suite is ported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from .robot import BaseRobot
    from .scene import BaseScene


def seed_rngs(seed: int) -> None:
    """Seed `random`, `np.random`, and torch (CPU + all CUDA devices)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class BaseEnv:
    def __init__(
        self,
        scene: BaseScene,
        robot: BaseRobot,
        sim_cfg: Any,
        *,
        num_envs: int = 1,
        env_spacing: float = 2.0,
        device: str = "cuda:0",
        seed: int | None = None,
    ) -> None:
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext  # lazy: requires AppLauncher

        self.cfg = sim_cfg
        self.num_envs, self.device, self.dt = num_envs, device, sim_cfg.dt
        self.scene, self.robot = scene, robot
        self.seed = seed
        if seed is not None:
            seed_rngs(seed)

        self.sim = SimulationContext(sim_cfg)
        iscene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        for name, asset in {**scene.assets(), **robot.assets()}.items():
            setattr(iscene_cfg, name, asset)
        self.iscene = InteractiveScene(iscene_cfg)

        self.sim.reset()
        self.scene.bind(self)
        self.robot.bind(self)
        self.reset()

    @property
    def stage(self):
        import omni.usd

        return omni.usd.get_context().get_stage()

    def _env_ids(self, env_ids: torch.Tensor | None):
        import torch

        return torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids

    def get_states(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        """All sim state = the scene's state + the robot's state. Restore with `set_states`."""
        env_ids = self._env_ids(env_ids)
        return {"scene": self.scene.get_state(env_ids), "robot": self.robot.get_state(env_ids)}

    def set_states(self, states: dict[str, Any], env_ids: torch.Tensor | None = None) -> None:
        """Restore the env to `states` (from `get_states`) — i.e. reset to any desired state."""
        env_ids = self._env_ids(env_ids)
        self.scene.set_state(states["scene"], env_ids)
        self.robot.set_state(states["robot"], env_ids)
        self.iscene.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None, *, seed: int | None = None) -> None:
        """Advance to the start state for `env_ids`. Read state with `get_states()`; observations
        (if any) come from a gym wrapper, not the raw env."""
        if seed is not None:
            seed_rngs(seed)
        env_ids = self._env_ids(env_ids)
        self.scene.reset(env_ids)
        self.robot.reset(env_ids)
        self.iscene.write_data_to_sim()

    def step(self, action: torch.Tensor, render: bool = False) -> None:
        """Apply `action` and advance one control step = `self.robot.control_period` physics substeps
        (decimation: controllers update on their own subdivision while physics runs at `self.dt`). No
        return — read state via `get_states()`. Dispatches through `self.robot`/`self.scene` (late
        binding). `post_step()` runs every substep, after `iscene.update`, so physics-coupled mechanics
        (e.g. auto-weld on seat) react at sim rate. `control_period == 1` -> exactly one physics step."""
        period = self.robot.control_period
        for k in range(period):
            self.robot.apply_action(action, substep=k)
            self.iscene.write_data_to_sim()
            self.sim.step(render=render and k == period - 1)  # render once, on the last substep
            self.iscene.update(self.dt)
            self.robot.post_step()
            self.scene.post_step()
        self._statelog_tick()

    def _statelog_tick(self) -> None:
        """REACHED-STATE LOG (grading source of record): when COSIGEN_STATELOG is set,
        every env — including ones the agent boots in its own scripts — periodically
        saves its full state, and ALWAYS saves on a score improvement. Runs are then
        graded on states actually reached in the run, no replay required. Off (and
        zero-cost) when the env var is unset."""
        import os
        if not hasattr(self, "_slog_dir"):
            d = os.environ.get("COSIGEN_STATELOG")
            self._slog_dir = None
            if d:
                from pathlib import Path
                self._slog_dir = Path(d) / f"pid{os.getpid()}"
                try:
                    self._slog_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    self._slog_dir = None
            self._slog_n = 0
            self._slog_seq = 0
            self._slog_best = float("-inf")
            self._slog_every = int(os.environ.get("COSIGEN_STATELOG_EVERY", "200"))
            self._slog_cap = int(os.environ.get("COSIGEN_STATELOG_CAP", "1500"))
        if self._slog_dir is None:
            return
        self._slog_n += 1
        score = None
        try:
            if hasattr(self.scene, "score"):
                s = self.scene.score()
                score = float(s.float().max() if hasattr(s, "max") else s) / 100.0
            elif hasattr(self.scene, "success"):
                s = self.scene.success()
                score = float(s.max() if hasattr(s, "max") else s)
        except Exception as exc:  # noqa: BLE001 -- scoring must never break stepping
            if not getattr(self, "_slog_score_warned", False):
                self._slog_score_warned = True
                import sys
                print(f"[statelog] score probe failed (reported once): {exc!r}", file=sys.stderr)
        improved = score is not None and score > self._slog_best + 1e-9
        if not improved and self._slog_n % self._slog_every != 0:
            return
        if not improved and self._slog_seq >= self._slog_cap:
            return  # cap periodic snapshots; improvements are always kept
        try:
            import time

            import torch
            tag = "peak" if improved else "tick"
            payload = {"states": self.get_states(), "score": score,
                       "t_wall": time.time(), "sim_step": self._slog_n, "tag": tag}
            try:
                succ = self.scene.success()
                payload["success"] = bool(succ.all() if hasattr(succ, "all") else succ)
            except Exception:  # noqa: BLE001
                pass
            # tmp+rename: a process killed mid-save must never leave a corrupt snapshot
            # that later poisons grading (readers still tolerate load failures)
            final = self._slog_dir / f"{self._slog_seq:06d}_{tag}.pt"
            tmp = self._slog_dir / f".{self._slog_seq:06d}_{tag}.tmp"
            torch.save(payload, tmp)
            tmp.rename(final)
            self._slog_seq += 1
            if improved:
                self._slog_best = score
        except Exception as exc:  # noqa: BLE001 -- logging must never break the run
            import sys
            print(f"[statelog] snapshot failed: {exc!r}", file=sys.stderr)

    def describe(self) -> str:
        """CURATED **natural-language** description for the agent: the scene's NL (objects + the
        goal — no separate task layer) followed by the robot's NL. This is the prompt-ready text.
        The low-level counterparts are `describe_stage()` (structured `list[dict]`) and
        `usd_text(self.stage)` (raw USD)."""
        return f"{self.scene.describe()}\n\n{self.robot.describe()}".strip()

    def describe_stage(self, root: str | None = None, raw: bool = False) -> list[dict[str, Any]] | str:
        """DETAILED, generic USD detail beneath the natural-language `describe()`, scoped to `root`
        (defaults to env 0). Two forms:
        - `raw=False` (default): structured `list[{path, type, pos, size}]` — already-extracted.
        - `raw=True`: the raw USD text (read it like XML)."""
        from .introspection import describe_stage as _describe_stage
        from .introspection import usd_text

        root = root or "/World/envs/env_0"
        return usd_text(self.stage, root) if raw else _describe_stage(self.stage, root)

    def close(self) -> None:
        self.sim.stop()
