"""push_cube_d1 — reorientation mutation of the ManiSkill PushCube seed.

The seed asks for planar transport: push a cube along the table into a goal region.
Here the cube is a die and the objective is pure orientation: it must come to rest
flat on the table with its BLUE face pointing up, anywhere on the table.  Sliding the
cube around (the seed's strategy) never changes what the criterion measures; the
solver must tip/roll the cube or lift and reorient it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim_gen.core import BaseScene, SceneCfg, register_scene

CUBE_HALF = 0.025
_SQ2 = float(np.sqrt(0.5))

# The five axis-aligned rest orientations whose blue face (+z body) is NOT up,
# keyed by which body axis ends up pointing at world +z.
BASE_QUATS: dict[str, tuple[float, float, float, float]] = {
    "-z": (0.0, 1.0, 0.0, 0.0),     # blue face down
    "+x": (_SQ2, 0.0, -_SQ2, 0.0),  # red face up
    "-x": (_SQ2, 0.0, _SQ2, 0.0),   # orange face up
    "+y": (_SQ2, _SQ2, 0.0, 0.0),   # green face up
    "-y": (_SQ2, -_SQ2, 0.0, 0.0),  # yellow face up
}


def compose_yaw(quat, yaw: float) -> np.ndarray:
    """World-frame yaw applied on top of a base orientation."""
    qy = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    out = np.empty(4)
    mujoco.mju_mulQuat(out, qy, np.asarray(quat, dtype=np.float64))
    return out


@dataclass
class FlipCubeSceneCfg(SceneCfg):
    up_tol_deg: float = 15.0      # success: blue normal within this angle of world +z
    spawn_range: float = 0.15     # cube xy ~ U[-spawn_range, spawn_range]^2
    settle_speed: float = 0.05    # |v| below which the cube counts as settled
    z_tol: float = 0.005          # resting-height tolerance ("flat on the table")
    camera_distance: float = 1.1
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.05)


@register_scene("push_cube_d1")
class FlipCubeScene(BaseScene):
    cfg: FlipCubeSceneCfg

    def __init__(self, cfg: FlipCubeSceneCfg | None = None):
        super().__init__(cfg or FlipCubeSceneCfg())

    def xml(self) -> str:
        h, d, t = CUBE_HALF, CUBE_HALF + 0.0016, 0.0015
        a = CUBE_HALF - 0.004  # face decals inset from the edges
        return f"""
<mujoco model="push_cube_d1">
  <option timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.6 0.6 0.02"
          rgba="0.55 0.45 0.35 1" friction="1.0 0.005 0.0001"/>
    <body name="cube" pos="0 0 {h}">
      <freejoint/>
      <geom name="cube_g" type="box" size="{h} {h} {h}" rgba="0.75 0.75 0.78 1"
            mass="0.05" friction="1.0 0.005 0.0001"/>
      <site name="face_blue"   type="box" size="{a} {a} {t}" pos="0 0 {d}"
            rgba="0.10 0.30 0.95 1"/>
      <site name="face_white"  type="box" size="{a} {a} {t}" pos="0 0 -{d}"
            rgba="0.95 0.95 0.95 1"/>
      <site name="face_red"    type="box" size="{a} {a} {t}" pos="{d} 0 0"
            zaxis="1 0 0" rgba="0.85 0.10 0.10 1"/>
      <site name="face_orange" type="box" size="{a} {a} {t}" pos="-{d} 0 0"
            zaxis="1 0 0" rgba="0.95 0.55 0.10 1"/>
      <site name="face_green"  type="box" size="{a} {a} {t}" pos="0 {d} 0"
            zaxis="0 1 0" rgba="0.10 0.70 0.20 1"/>
      <site name="face_yellow" type="box" size="{a} {a} {t}" pos="0 -{d} 0"
            zaxis="0 1 0" rgba="0.90 0.85 0.10 1"/>
    </body>
  </worldbody>
</mujoco>
"""

    def reset_instance(self, rng: np.random.Generator) -> None:
        c = self.cfg
        xy = rng.uniform(-c.spawn_range, c.spawn_range, size=2)
        face = sorted(BASE_QUATS)[rng.integers(len(BASE_QUATS))]
        quat = compose_yaw(BASE_QUATS[face], rng.uniform(0, 2 * np.pi))
        self.set_body_pose("cube", (*xy, CUBE_HALF + 0.0005), quat)

    # ---- semantics -------------------------------------------------------------
    def blue_up(self) -> float:
        """World z-component of the blue-face normal (body +z axis): 1 = up, -1 = down."""
        return float(self.data.body("cube").xmat.reshape(3, 3)[2, 2])

    def _flat_on_table(self) -> bool:
        z_ok = abs(self.body_pos("cube")[2] - CUBE_HALF) < self.cfg.z_tol
        speed = float(np.linalg.norm(self.data.body("cube").cvel))
        return z_ok and speed < self.cfg.settle_speed

    def success(self) -> bool:
        up_ok = self.blue_up() > np.cos(np.radians(self.cfg.up_tol_deg))
        return up_ok and self._flat_on_table()

    def score(self) -> float:
        if self.success():
            return 1.0
        # graded by how far the blue normal has rotated toward world up; sideways and
        # upside-down (every no-progress state, incl. any planar slide) clip to 0
        return 0.95 * float(np.clip(self.blue_up(), 0.0, 1.0))

    def describe(self) -> str:
        return (
            "A 5 cm die rests on a table: a gray cube with six colored face decals "
            "(blue, white, red, orange, green, yellow). Reorient the cube — tip it, "
            "roll it, or pick it up and turn it — so it comes to rest FLAT on the "
            f"table with its BLUE face pointing up (within {self.cfg.up_tol_deg:.0f} "
            "degrees of vertical). Where the cube ends up on the table does not "
            "matter; sliding it around without reorienting it accomplishes nothing, "
            "and holding it blue-face-up in the air does not count."
        )
