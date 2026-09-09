"""The configuration layer — all of robobench's config types, in one app-free place.

Three cohesive pieces (import-light: stdlib only at module load; the few isaaclab/registry touches in
`EnvCfg.build` / `register_env` are lazy, so importing this — and listing/registering configs — never
needs AppLauncher):

  - `BaseCfg` — the shared base scene/robot cfgs subclass: plain dataclasses, plain fields
    (nothing is locked; a variant is a `replace(...)` / copy with a few fields changed).
  - `SimCfg` — the sim substrate (dt + PhysX), **declared by the scene** (its contact geometry drives
    the requirements), patchable per binding.
  - `EnvCfg` (+ `register_env`) — the one struct that **binds a runnable env** (scene + robot + control
    mode + sim) and is loaded by name from the `ENVS` registry. Mirrors CaP-X's `CodeExecEnvConfig`:
    orthogonal Sim/Scene/Robot/Grader knobs, copy-and-override derivations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .env import BaseEnv


# ----- BaseCfg ----------------------------------------------------------------------------------
@dataclass
class BaseCfg:
    """Shared base for scene/robot configs: a plain dataclass of plain fields. Nothing is locked —
    a curriculum/debug variant is just a copy with a few fields changed."""


# ----- SimCfg: the sim substrate (scene-declared) -----------------------------------------------
@dataclass
class SimCfg:
    """Sim-substrate params, **declared by the scene** (`BaseScene.sim_cfg()`): the scene's contact
    geometry is what drives the requirements. An `EnvCfg` may *patch* these for a specific embodiment (its
    `sim_overrides`). `physx` is kwargs splatted into isaaclab's `PhysxCfg` (so any PhysX knob works
    without growing this class). App-free; `to_isaaclab()` builds the real `SimulationCfg`."""

    dt: float = 0.01
    physx: dict[str, Any] = field(default_factory=lambda: {"solver_type": 1})
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    # kwargs splatted into isaaclab's `RenderCfg` (like `physx` -> `PhysxCfg`). Empty -> RTX defaults.
    # A scene with glass/translucent parts sets `{"enable_translucency": True}` or it renders invisible.
    render: dict[str, Any] = field(default_factory=dict)

    def to_isaaclab(self, device: str) -> Any:
        """Build the isaaclab `SimulationCfg` (needs AppLauncher running)."""
        import isaaclab.sim as sim_utils

        return sim_utils.SimulationCfg(
            device=device, dt=self.dt, gravity=self.gravity, physx=sim_utils.PhysxCfg(**self.physx),
            render=sim_utils.RenderCfg(**self.render),
        )


# ----- EnvCfg: the runnable-env binding ---------------------------------------------------------
@dataclass
class EnvCfg:
    """Binds scene + robot (by `SCENES` / `ROBOTS` name) + control mode + sim into one runnable env —
    the single thing a run or test loads (by name from `ENVS`).

    `scene_cfg` / `robot_cfg` are optional concrete cfg instances (a suite registering a config knows
    the cfg classes); leave them None for defaults. `control_mode` is applied to the robot's cfg. Sim
    defaults come from the scene; `sim_overrides` patch them. Curriculum/debug variants are cheap
    `dataclasses.replace(cfg, ...)`.
    """

    scene: str  # SCENES name
    # A cfg-only variant of the scene (another food, layout, ...) that deserves its own registered
    # name: "" -> none; "banana" -> the scene segment of the name becomes "<scene>_banana". The scene
    # class is untouched — the variant is carried entirely by `scene_cfg`.
    variant: str = ""
    robot: str = "null"  # ROBOTS name
    control_mode: str = ""  # applied to the robot cfg ("" -> the robot's first mode)
    scene_cfg: Any = None  # a scene BaseCfg instance, or None -> the scene's default
    robot_cfg: Any = None  # a BaseRobotCfg instance, or None -> the robot's default
    num_envs: int = 1
    env_spacing: float = 2.0
    device: str = "cuda:0"
    seed: int | None = 0  # None -> unseeded (RNGs left alone)
    # Sim defaults come from the SCENE (`scene.sim_cfg()` — its contact geometry drives the
    # requirements). These patch them for THIS binding (e.g. an embodiment that needs more solver
    # iters). Keys: "dt" / "gravity", and "physx" (merged into the scene's PhysX kwargs, not replaced).
    sim_overrides: dict[str, Any] = field(default_factory=dict)

    def qualified_name(self, suite: str) -> str:
        """The canonical `ENVS` name under `suite`, by the convention
        ``suite.scene[_variant][.robot[.control_mode]]`` — segments dropped from the RIGHT when
        default/absent (no robot, or the robot's default mode). So `assembly` + (scene=ikea_table,
        robot=null) -> ``assembly.ikea_table``; + (ikea_table, g1, joint) ->
        ``assembly.ikea_table.g1.joint``; a cfg-only variant joins the scene segment:
        (scene=slice, variant=banana, franka, osc) -> ``cutting.slice_banana.franka.osc``.
        General -> specific, so a sorted listing groups by suite -> scene -> robot."""
        parts = [suite, f"{self.scene}_{self.variant}" if self.variant else self.scene]
        if self.robot and self.robot != "null":
            parts.append(self.robot)
            if self.control_mode:
                parts.append(self.control_mode)
        return ".".join(parts)

    def describe(self) -> str:
        mode = self.control_mode or "(default)"
        var = f", variant='{self.variant}'" if self.variant else ""
        return f"EnvCfg(scene='{self.scene}'{var}, robot='{self.robot}', control_mode='{mode}', num_envs={self.num_envs})"

    def build(self, **overrides: Any) -> BaseEnv:
        """Construct the live `BaseEnv` (needs AppLauncher already running). `overrides` patch fields
        first (e.g. `build(num_envs=16, device="cuda:0", control_mode="pink_ik")`)."""
        cfg = replace(self, **overrides) if overrides else self

        from .env import BaseEnv
        from .registries import ROBOTS, SCENES

        scene_cls = SCENES.get(cfg.scene)
        scene = scene_cls(cfg.scene_cfg) if cfg.scene_cfg is not None else scene_cls()

        robot_cls = ROBOTS.get(cfg.robot)
        rcfg = cfg.robot_cfg
        if rcfg is None:  # default cfg, so we can stamp the control mode onto it
            rcfg = getattr(robot_cls(), "cfg", None)
        if rcfg is not None and cfg.control_mode and hasattr(rcfg, "control_mode"):
            rcfg.control_mode = cfg.control_mode
        robot = robot_cls(rcfg) if rcfg is not None else robot_cls()

        # Sim from the scene, patched by this binding's overrides (merge PhysX kwargs, don't replace).
        sim = scene.sim_cfg()
        ov = dict(cfg.sim_overrides)
        if "physx" in ov:
            sim = replace(sim, physx={**sim.physx, **ov.pop("physx")})
        if "render" in ov:
            sim = replace(sim, render={**sim.render, **ov.pop("render")})
        if ov:
            sim = replace(sim, **ov)
        sim_cfg = sim.to_isaaclab(cfg.device)
        return BaseEnv(
            scene,
            robot,
            sim_cfg,
            num_envs=cfg.num_envs,
            env_spacing=cfg.env_spacing,
            device=cfg.device,
            seed=cfg.seed,
        )


def register_env(suite: str, factory: Callable[[], EnvCfg]) -> str:
    """Register an `EnvCfg` `factory` in `ENVS` under its canonical `suite.scene[.robot[.mode]]` name
    (derived from the cfg by `EnvCfg.qualified_name`, so names never drift from the convention).
    Returns the name. Use it in a suite's `configs/envs.py`:

        register_env("assembly", lambda: EnvCfg(scene="ikea_table", robot="g1", control_mode="joint"))
        # -> registers "assembly.ikea_table.g1.joint"
    """
    from .registries import ENVS

    name = factory().qualified_name(suite)
    ENVS.register(name, factory)
    return name
