"""push_cube_ref — reference port of the ManiSkill PushCube seed (format exemplar).

This is NOT a generated task: it is the seed itself, re-implemented against
``sim_gen.core`` so construction agents have one concrete example of the full task
contract (xml / reset_instance / success / score / describe + cfg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_gen.core import BaseScene, SceneCfg, register_scene

CUBE_HALF = 0.02


@dataclass
class PushCubeSceneCfg(SceneCfg):
    goal_radius: float = 0.10          # success: cube xy within this radius of the goal
    spawn_range: float = 0.10          # cube xy ~ U[-spawn_range, spawn_range]^2
    goal_dist: tuple[float, float] = (0.15, 0.25)  # goal distance from cube (annulus)
    settle_speed: float = 0.05         # |v| below which the cube counts as settled


@register_scene("push_cube_ref")
class PushCubeScene(BaseScene):
    cfg: PushCubeSceneCfg

    def __init__(self, cfg: PushCubeSceneCfg | None = None):
        super().__init__(cfg or PushCubeSceneCfg())
        self._goal_xy = np.zeros(2)
        self._d0 = 1.0  # initial cube->goal distance (score normalizer)

    def xml(self) -> str:
        return f"""
<mujoco model="push_cube_ref">
  <option timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.6 0.6 0.02"
          rgba="0.55 0.45 0.35 1" friction="1.0 0.005 0.0001"/>
    <body name="cube" pos="0 0 {CUBE_HALF}">
      <freejoint/>
      <geom name="cube_g" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
            rgba="0.9 0.15 0.15 1" mass="0.02" friction="1.0 0.005 0.0001"/>
    </body>
    <site name="goal" pos="0.2 0 0.0005" size="0.05 0.05 0.0004" type="ellipsoid"
          rgba="0.1 0.3 0.9 0.6"/>
  </worldbody>
</mujoco>
"""

    def reset_instance(self, rng: np.random.Generator) -> None:
        c = self.cfg
        cube_xy = rng.uniform(-c.spawn_range, c.spawn_range, size=2)
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(*c.goal_dist)
        self._goal_xy = cube_xy + dist * np.array([np.cos(ang), np.sin(ang)])
        self.set_body_pose("cube", (*cube_xy, CUBE_HALF))
        self.model.site("goal").pos[:2] = self._goal_xy
        self._d0 = float(np.linalg.norm(cube_xy - self._goal_xy))

    # ---- semantics -------------------------------------------------------------
    def _cube_goal_dist(self) -> float:
        return float(np.linalg.norm(self.body_pos("cube")[:2] - self._goal_xy))

    def _on_table_settled(self) -> bool:
        z_ok = abs(self.body_pos("cube")[2] - CUBE_HALF) < 0.005
        speed = float(np.linalg.norm(self.data.body("cube").cvel))
        return z_ok and speed < self.cfg.settle_speed

    def success(self) -> bool:
        return self._cube_goal_dist() < self.cfg.goal_radius and self._on_table_settled()

    def score(self) -> float:
        if self.success():
            return 1.0
        # graded progress toward the goal, capped below success level
        progress = max(0.0, 1.0 - self._cube_goal_dist() / max(self._d0, 1e-6))
        return min(0.95, progress)

    def describe(self) -> str:
        return (
            "A red cube rests on a table; a blue disc marks a goal region. Push the "
            f"cube so it comes to rest on the table with its center within "
            f"{self.cfg.goal_radius:.2f} m (xy) of the goal marker. The cube must end "
            "settled on the table surface — held in the air over the goal does not count."
        )
