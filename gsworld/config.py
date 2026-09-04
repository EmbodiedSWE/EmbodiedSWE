"""One JSON per photoreal env (GSWorld's ``configs/*.json`` idea, extended to the whole env):

    {
      "env": {                                   # the robobench env (any registered scene / robot)
        "scene": "table", "robot": "franka_robotiq", "control_mode": "joint",
        "scene_cfg": {"layout": "gsworld_table"},          # dataclass fields of the scene cfg (optional)
        "robot_cfg": {"base_pos": [0, 0, 0]},              # dataclass fields of the robot cfg (optional)
        "env_spacing": 3.0, "sim_overrides": {}            # (optional)
      },
      "robot": {"ply": "../../models/franka_robotiq/franka_robotiq.ply",
                "poses": "../../models/franka_robotiq/franka_robotiq_poses.json"},      # or null
      "static":  [{"path": "room.ply", "crop": [[-0.1, -0.9, -1.0], [1.3, 1.1, 0.2]], "min_opacity": 0.05}],
      "objects": [{"name": "table", "ply": "../../models/table/table.ply",             # per-step posed from
                   "poses": "../../models/table/table_poses.json"}],                    #   env.iscene["table"]
      "camera":  {"eye": [1.9, -1.6, 1.1], "target": [0.45, 0.05, 0.15], "size": [960, 600]}
    }

Every model is ONE object: a PLY + ``_poses.json`` (see :mod:`gsworld.model`). ``robot`` and
``objects`` are re-posed from the sim every step (an articulation's bodies, or a rigid object's
root); ``static`` entries are plain PLYs fixed in the sim frame (backgrounds, room pieces). All PLYs are
expected in the metric sim frame already (preparation is external). Relative paths resolve
against the JSON's directory. ``env`` is optional when the JSON only decorates an env
built elsewhere (e.g. ``scripts/record_video.py --splat``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gsworld.splat import GaussianSet


def _abs(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


@dataclass
class SplatModelCfg:
    """A PLY already in the sim frame, fixed (backgrounds, room pieces); optional crop / opacity cut."""
    path: str
    crop: list | None = None
    min_opacity: float = 0.0

    def load(self, base: Path) -> GaussianSet:
        g = GaussianSet.from_ply(_abs(base, self.path))
        if self.min_opacity > 0:
            g = g.opacity_filter(self.min_opacity)
        if self.crop:
            g = g.crop_box(self.crop[0], self.crop[1])
        return g


@dataclass
class ModelRef:
    """A splat model (PLY + poses json); ``name`` = the env's ``iscene`` key that drives it."""
    ply: str
    poses: str
    name: str = ""


@dataclass
class EnvSpec:
    scene: str
    robot: str = "null"
    control_mode: str = ""
    scene_cfg: dict[str, Any] = field(default_factory=dict)
    robot_cfg: dict[str, Any] = field(default_factory=dict)
    num_envs: int = 1
    env_spacing: float = 3.0
    sim_overrides: dict[str, Any] = field(default_factory=dict)

    def build(self, device: str | None = None, seed: int = 0):
        """Build the robobench env: registry lookups by name, cfg dataclasses filled from the dicts."""
        import robobench
        from robobench.core import ROBOTS, SCENES, EnvCfg

        robobench.discover()
        scene_cfg = _make_cfg(SCENES.get(self.scene), self.scene_cfg)
        robot_cfg = _make_cfg(ROBOTS.get(self.robot), self.robot_cfg) if self.robot != "null" and self.robot_cfg else None
        cfg = EnvCfg(scene=self.scene, robot=self.robot, control_mode=self.control_mode, scene_cfg=scene_cfg,
                     robot_cfg=robot_cfg, num_envs=self.num_envs, env_spacing=self.env_spacing,
                     sim_overrides=dict(self.sim_overrides))
        kw = {"num_envs": self.num_envs, "seed": seed}
        if device:
            kw["device"] = device
        return cfg.build(**kw)


def _make_cfg(cls, fields: dict[str, Any]):
    """Instantiate a registered scene/robot's cfg dataclass (its ``cfg:`` annotation) from a dict.
    Tuples in dataclass defaults are accepted as JSON lists."""
    if not fields:
        return None
    import typing

    hints = typing.get_type_hints(cls)
    cfg_cls = hints.get("cfg")
    if cfg_cls is None:
        raise TypeError(f"{cls.__name__} declares no `cfg:` annotation; cannot build it from JSON")
    conv = {k: (tuple(v) if isinstance(v, list) else v) for k, v in fields.items()}
    return cfg_cls(**conv)


@dataclass
class CameraSpec:
    eye: tuple[float, float, float] = (1.9, -1.6, 1.1)
    target: tuple[float, float, float] = (0.45, 0.05, 0.15)
    size: tuple[int, int] = (960, 600)


@dataclass
class SplatSceneCfg:
    env: EnvSpec | None = None
    robot: ModelRef | None = None
    static: list[SplatModelCfg] = field(default_factory=list)
    objects: list[ModelRef] = field(default_factory=list)
    camera: CameraSpec | None = None
    base: Path = field(default_factory=Path)

    @classmethod
    def load(cls, path: str | Path) -> "SplatSceneCfg":
        path = Path(path)
        d = json.loads(path.read_text())
        return cls(
            env=EnvSpec(**d["env"]) if d.get("env") else None,
            robot=ModelRef(**d["robot"]) if d.get("robot") else None,
            static=[SplatModelCfg(**s) for s in d.get("static", [])],
            objects=[ModelRef(**o) for o in d.get("objects", [])],
            camera=CameraSpec(**{k: tuple(v) for k, v in d["camera"].items()}) if d.get("camera") else None,
            base=path.parent,
        )

    # ----------------------------------------------------------------- pieces
    def robot_splat(self, device: str):
        from gsworld.model import SplatModel

        if self.robot is None:
            return None
        return SplatModel(_abs(self.base, self.robot.ply), _abs(self.base, self.robot.poses), device=device)

    def load_static(self) -> list[GaussianSet]:
        return [s.load(self.base) for s in self.static]

    def load_objects(self, device: str = "cuda"):
        from gsworld.model import SplatModel

        return {o.name: SplatModel(_abs(self.base, o.ply), _abs(self.base, o.poses), device=device) for o in self.objects}

    # ----------------------------------------------------------------- the whole thing
    def build(self, device: str | None = None, seed: int = 0, attach_camera: bool = True):
        """Build the env from ``env`` and wrap it: ``SplatEnv`` with robot, static and object models,
        viewport camera attached when a ``camera`` block exists (or the scene's ``CAMERAS['front']``)."""
        from gsworld.wrapper import SplatEnv

        if self.env is None:
            raise ValueError("this scene config has no 'env' block; wrap an existing env with SplatEnv.from_config")
        env = self.env.build(device=device, seed=seed)
        senv = SplatEnv.from_config(env, self, device=device)
        if attach_camera:
            cam = self.camera or CameraSpec(**{k: tuple(v) for k, v in env.scene.CAMERAS.get("front", {}).items()
                                               if k in ("eye", "target")})
            senv.attach_viewport_camera(cam.eye, cam.target, width=cam.size[0], height=cam.size[1])
        return senv
