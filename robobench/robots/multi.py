"""MultiRobot — several robots presented to the env as ONE robot (the robot-level `composite`).

Bimanual / multi-arm setups without authoring a monolithic dual-arm asset: each child robot keeps its
own articulation, its own controller stack and its own control mode; this wrapper only namespaces
their assets and fans the env-facing `BaseRobot` surface out over them. The env is untouched — it
sees a single robot whose action is the children's actions CONCATENATED in declaration order.

Usage — two Frankas flanking the work, bound like any single robot:

    from robobench.robots import FrankaRobotCfg, MultiRobotCfg

    cfg = MultiRobotCfg(robots={
        "left":  ("franka", FrankaRobotCfg(base_pos=(0.0,  0.35, 0.0))),
        "right": ("franka", FrankaRobotCfg(base_pos=(0.0, -0.35, 0.0))),
    })
    EnvCfg(scene="nut_thread", robot="multi", robot_cfg=cfg)   # or MultiRobot(cfg) directly

How the pieces map:
  - **Action**: `[left | right | ...]` in dict order, each child's slice sized by its own
    `action_dim` (two OSC Frankas -> 8 + 8 = 16). `action_slices` gives `{name: slice}` so an agent
    can address one arm: `action[:, robot.action_slices["right"]] = ...`.
  - **Control modes are per child**: set `control_mode` on each child cfg (left "osc" + right
    "joint" is fine). A non-empty `MultiRobotCfg.control_mode` — e.g. stamped by
    `EnvCfg.control_mode` — is propagated to ALL children instead, convenient when they are the
    same embodiment.
  - **Controllers need no changes**: each child builds + binds its OWN controller against its OWN
    articulation inside `child.bind(env)` (joint ids, sinks and limits are captured per child), so
    mixed command types / rates across children compose exactly like `CompositeController` leaves.
    The composite itself never owns a controller; to swap one at runtime, go through the child:
    `env.robot["left"].set_controller(MyController(...))`.
  - **Namespacing**: the dict key becomes the child's `cfg.name` -> its `env.iscene` key and prim
    (`{ENV_REGEX_NS}/Left`), so children never collide in the scene. Keys must be unique, non-empty
    identifiers; "robot" is reserved for the single-robot default.
  - **State / reset / describe**: fanned out per child; `get_state()` nests each child's dict under
    its name, `describe()` concatenates the children's NL with the action layout for the agent.
  - **Heterogeneous pairs** (franka + g1, ...) compose the same way — nothing here is arm-specific.

Import-light like the other robots (children defer their own heavy imports), so registration stays
app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robobench.core import ROBOTS, BaseRobot, BaseRobotCfg, info

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseController, BaseEnv


@dataclass
class MultiRobotCfg(BaseRobotCfg):
    """Config for `MultiRobot`: `robots` maps each child name to `(ROBOTS registry name, cfg)`,
    where cfg is a concrete `BaseRobotCfg` instance or None for that robot's default. Dict order is
    action order. The child name is stamped onto the child's `cfg.name` (its scene namespace).
    `control_mode` (inherited): "" -> each child keeps its own cfg's mode; non-empty -> stamped onto
    every child (each validates it against its own `control_modes`)."""

    #: {child_name: (ROBOTS name, child cfg | None)} — the children, in action-slice order.
    robots: dict[str, tuple[str, Any]] = info(factory=dict, doc="{name: (ROBOTS name, cfg|None)} children, in action order")


@ROBOTS.register("multi")
class MultiRobot(BaseRobot):
    """Named child robots acting as one env-facing robot. `action_dim` / `apply_action` /
    `control_period` / state / reset fan out over the children (see module docstring); index a child
    with `robot["left"]` (or iterate `robot.robots.items()`)."""

    control_modes: tuple[str, ...] = ()  # modes live on the children (see MultiRobotCfg)
    cfg: MultiRobotCfg

    def __init__(self, cfg: MultiRobotCfg | None = None) -> None:
        super().__init__(cfg or MultiRobotCfg())
        if not self.cfg.robots:
            raise ValueError(
                "MultiRobotCfg.robots is empty — give it children as {name: (ROBOTS name, cfg|None)}, "
                'e.g. {"left": ("franka", FrankaRobotCfg(base_pos=...)), "right": ("franka", ...)}.'
            )
        #: The children, by name, in action-slice order. Public: address one as `self["left"]`.
        self.robots: dict[str, BaseRobot] = {}
        for name, (kind, rcfg) in self.cfg.robots.items():
            if not name.isidentifier() or name == "robot":
                raise ValueError(f"child name {name!r} must be a unique identifier != 'robot' (it becomes the prim namespace)")
            child_cls = ROBOTS.get(kind)
            if rcfg is None:  # the robot's default cfg, so we can stamp name/mode onto it (as EnvCfg.build does)
                rcfg = getattr(child_cls(), "cfg", None)
            if rcfg is not None:
                rcfg.name = name  # namespace the child: its asset key + prim path
                if self.cfg.control_mode and hasattr(rcfg, "control_mode"):
                    rcfg.control_mode = self.cfg.control_mode
            self.robots[name] = child_cls(rcfg) if rcfg is not None else child_cls()

    def __getitem__(self, name: str) -> BaseRobot:
        """A child by name — `env.robot["left"].articulation`, `env.robot["right"].set_controller(...)`."""
        return self.robots[name]

    # ----- assets / lifecycle (fan out; each child binds itself + its own controller) -----------
    def assets(self) -> dict[str, Any]:
        """The union of the children's assets — disjoint by construction (each child's keys sit under
        its own `cfg.name` namespace)."""
        merged: dict[str, Any] = {}
        for r in self.robots.values():
            for key, asset in r.assets().items():
                if key in merged:
                    raise ValueError(f"asset key {key!r} appears in two children — child names must namespace disjointly")
                merged[key] = asset
        return merged

    def on_bind(self, env: BaseEnv) -> None:
        # Each child runs its FULL bind (grab handles -> build -> bind its own controller, against its
        # own articulation). The composite keeps controller=None; apply_action fans out instead.
        for r in self.robots.values():
            r.bind(env)

    def set_controller(self, controller: BaseController) -> None:
        raise RuntimeError(
            "MultiRobot has no single controller — set it on a child: env.robot['left'].set_controller(...)"
        )

    def actuator_sink(self, command_type: str):
        raise RuntimeError("MultiRobot has no single articulation — use a child's sink: robot['left'].actuator_sink(...)")

    def actuator_limits(self, joint_ids: Any) -> dict[str, Any]:
        raise RuntimeError("MultiRobot has no single articulation — use a child's limits: robot['left'].actuator_limits(...)")

    # ----- action (concatenated child slices, in declaration order) -----------------------------
    @property
    def action_dim(self) -> int:
        return sum(r.action_dim for r in self.robots.values())

    @property
    def action_slices(self) -> dict[str, slice]:
        """`{child: slice}` into the concatenated action, in declaration order — e.g.
        `action[:, robot.action_slices["right"]]` addresses the right arm alone."""
        out, i = {}, 0
        for name, r in self.robots.items():
            out[name] = slice(i, i + r.action_dim)
            i += r.action_dim
        return out

    @property
    def control_period(self) -> int:
        """The slowest child's period (the env loops this many substeps per action). Like
        `CompositeController`, each child still fires on its own subdivision inside the window."""
        return max(r.control_period for r in self.robots.values())

    def apply_action(self, action: torch.Tensor, substep: int = 0) -> None:
        for name, s in self.action_slices.items():
            self.robots[name].apply_action(action[:, s], substep)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        for r in self.robots.values():
            r.post_step(env_ids)

    # ----- state / reset (nested per child name) -------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        for r in self.robots.values():
            r.reset(env_ids)

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {name: r.get_state(env_ids) for name, r in self.robots.items()}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name, r in self.robots.items():
            r.set_state(state[name], env_ids)

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        slices = self.action_slices
        layout = ", ".join(f"'{n}' dims [{s.start}:{s.stop})" for n, s in slices.items())  # half-open, like the slices
        parts = [
            f"{len(self.robots)} robots operating in the same workspace as one composite embodiment. "
            f"The action vector is their actions concatenated in order ({layout}; total {self.action_dim})."
        ]
        parts += [f"[{n}] {r.describe()}" for n, r in self.robots.items()]
        return "\n".join(parts)
