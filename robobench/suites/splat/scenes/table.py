"""TableScene — a kinematic table box, ground and dome light laid out like a splat capture.

World frame = the robot base frame (base at the origin, z-up, metres). The default layout is the
reference workcell capture the shipped robot model was made in: table top at z = 0, the box in
front of the base (x 0.14..1.03, y -0.72..0.86), ground at the box's feet, and a pedestal under the base so
the robot stands at table height like a real mount. No task objects: this scene is the physics
stand-in under a photoreal splat (see gsworld/README.md).

Heavy imports (isaaclab) are deferred so importing this module stays app-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg


@dataclass
class TableSceneCfg(BaseCfg):
    """`layout` picks a preset in `LAYOUTS`; `top_z` / `top_xy` / `height` override it when set."""

    layout: str = "gsworld_table"
    top_z: float | None = None                              # table-top height (m)
    top_xy: tuple[float, float, float, float] | None = None  # (x0, x1, y0, y1) of the top face (m)
    height: float | None = None                             # box height (m); ground sits at top_z - height
    pedestal: bool = True                                   # 0.30 m column under the base, down to the ground
    table_color: tuple[float, float, float] = (0.55, 0.55, 0.58)
    light_intensity: float = 2500.0

    #: Capture layouts (top face extents in the base frame). Add one per scanned workcell.
    LAYOUTS: ClassVar[dict[str, dict[str, Any]]] = {
        "gsworld_table": {"top_z": 0.0, "top_xy": (0.14, 1.03, -0.72, 0.86), "height": 0.9},
    }

    def __post_init__(self) -> None:
        preset = self.LAYOUTS[self.layout]
        self.top_z = preset["top_z"] if self.top_z is None else self.top_z
        self.top_xy = tuple(preset["top_xy"]) if self.top_xy is None else tuple(self.top_xy)
        self.height = preset["height"] if self.height is None else self.height

    @property
    def surface_z(self) -> float:
        """Work-surface height — read by scripts/record_video.py to anchor its camera."""
        return float(self.top_z)

    @property
    def size(self) -> tuple[float, float, float]:
        x0, x1, y0, y1 = self.top_xy
        return (x1 - x0, y1 - y0, float(self.height))

    @property
    def center(self) -> tuple[float, float, float]:
        x0, x1, y0, y1 = self.top_xy
        return ((x0 + x1) / 2, (y0 + y1) / 2, self.top_z - self.height / 2)

    @property
    def ground_z(self) -> float:
        return self.top_z - self.height


@SCENES.register("table")
class TableScene(BaseScene):
    cfg: TableSceneCfg

    #: Env-origin-relative views on the work surface (see BaseScene.CAMERAS).
    CAMERAS: ClassVar[dict[str, dict]] = {
        "front": {"eye": (1.9, -1.6, 1.1), "target": (0.45, 0.05, 0.15), "focal": 24.0},
    }

    def __init__(self, cfg: TableSceneCfg | None = None) -> None:
        super().__init__(cfg or TableSceneCfg())

    def sim_cfg(self) -> SimCfg:
        return SimCfg(dt=1.0 / 120.0)

    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(color=(0.35, 0.35, 0.38)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, c.ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.98, 0.98, 0.98)),
            ),
            # kinematic rigid body: never falls, but is addressable as env.iscene["table"] (root pose), so a
            # splat model can ride it and the layout can be moved through the sim like any object
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.table_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.center),
            ),
        }
        if c.pedestal:
            out["pedestal"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=(0.30, 0.30, c.height),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.38, 0.30)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.05, 0.0, c.ground_z + c.height / 2)),
            )
        return out

    def reset(self, env_ids=None) -> None:  # static scene
        pass

    def get_state(self, env_ids=None) -> dict:
        return {}

    def set_state(self, state, env_ids=None) -> None:
        pass

    def describe(self) -> str:
        x0, x1, y0, y1 = self.cfg.top_xy
        return (f"A bare table whose top is at z={self.cfg.top_z:.2f} m, spanning x [{x0:.2f}, {x1:.2f}] "
                f"and y [{y0:.2f}, {y1:.2f}] in front of the robot base at the origin; no objects.")
