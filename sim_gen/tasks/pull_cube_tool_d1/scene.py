"""pull_cube_tool_d1 — bridge the trench: the seed's pull-tool becomes a load-bearing bridge.

Seed (ManiSkill PullCubeTool): hook an L-tool behind an out-of-reach cube and DRAG it
along a continuous table into reach.  Here the surface is deliberately discontinuous:
the cube starts on a far platform separated from the near platform by an open trench,
so the seed's drag plan drops the cube into the pit.  The tool is a plank that must be
laid ACROSS the trench, the cube must cross ON the bridge, be deposited in a walled
bin, and the plank must then be returned to its rest pad.

Ordered stage chain: ① span the trench with the plank → ② cross the cube over the
bridge → ③ deposit the cube in the bin → ④ return the plank to the rest pad.
Ordering is enforced physically (no crossing without support) and by the rubric
(delivery/return credit is gated on the crossing latch).

Mechanism notes:
- The "crossed the trench on the bridge" event cannot be read off the final state, so
  it is a LATCH updated every physics step: cube↔plank contact while the cube is over
  the trench AND the plank is genuinely spanning (both ends overhanging the platform
  edges, lying flat).  The spanning requirement is what makes a cantilevered plank —
  which tips under the cube anyway — count for nothing.
- Latches are cleared in reset_instance and appended to get_state()/set_state() so
  snapshot/restore round-trips keep earned credit (and state_hash covers them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_gen.core import BaseScene, SceneCfg, register_scene


@dataclass
class TrenchBridgeSceneCfg(SceneCfg):
    gap_half: float = 0.06                      # trench spans x in [-gap_half, +gap_half]
    cube_half: float = 0.02
    cube_mass: float = 0.05
    plank_half: tuple[float, float, float] = (0.11, 0.035, 0.006)
    plank_mass: float = 0.06                    # light vs. the cube so a cantilever tips (control E)
    span_overlap: float = 0.008                 # min plank-end overhang past each trench edge
    bin_x: float = -0.40                        # bin center x on the near platform (y randomized)
    bin_inner: float = 0.05                     # bin interior half-width
    bin_wall_h: float = 0.03
    bin_tol: float = 0.03                       # per-axis cube-center tolerance inside the bin
    rest_x: float = -0.46                       # rest pad center x (y randomized)
    rest_tol: float = 0.05                      # plank-center-to-pad-center xy tolerance
    settle_speed: float = 0.05                  # |cvel| below which a body counts as settled
    camera_distance: float = 1.5
    camera_azimuth: float = 115.0
    camera_elevation: float = -35.0
    camera_lookat: tuple[float, float, float] = (-0.05, 0.0, 0.0)


@register_scene("pull_cube_tool_d1")
class TrenchBridgeScene(BaseScene):
    cfg: TrenchBridgeSceneCfg

    _N_LATCH = 4  # bridged, crossed, delivered, best-approach — appended to the state vector

    def __init__(self, cfg: TrenchBridgeSceneCfg | None = None):
        super().__init__(cfg or TrenchBridgeSceneCfg())
        self._cube_bid = self.model.body("cube").id
        self._plank_bid = self.model.body("plank").id
        self._cube_gid = self.model.geom("cube_g").id
        self._plank_gid = self.model.geom("plank_g").id
        self._bin_xy = np.array([self.cfg.bin_x, 0.0])
        self._rest_xy = np.array([self.cfg.rest_x, 0.0])
        self._d0 = 1.0
        self._clear_latches()

    def xml(self) -> str:
        c = self.cfg
        g = c.gap_half
        near_c, near_h = -(0.55 + g) / 2, (0.55 - g) / 2
        far_c, far_h = (0.45 + g) / 2, (0.45 - g) / 2
        wall_t, wall_off = 0.008, c.bin_inner + 0.008
        wall_z = c.bin_wall_h / 2
        wall_len = c.bin_inner + 2 * wall_t  # long enough to close the corners
        return f"""
<mujoco model="pull_cube_tool_d1">
  <option timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1"/>
    <geom name="near_top" type="box" pos="{near_c} 0 -0.06" size="{near_h} 0.4 0.06"
          rgba="0.55 0.45 0.35 1" friction="1.0 0.005 0.0001"/>
    <geom name="far_top" type="box" pos="{far_c} 0 -0.06" size="{far_h} 0.4 0.06"
          rgba="0.50 0.42 0.33 1" friction="1.0 0.005 0.0001"/>
    <geom name="pit_floor" type="box" pos="0 0 -0.135" size="{g} 0.4 0.015"
          rgba="0.25 0.22 0.20 1" friction="1.0 0.005 0.0001"/>
    <body name="plank" pos="-0.27 0 {c.plank_half[2]}">
      <freejoint/>
      <geom name="plank_g" type="box" size="{c.plank_half[0]} {c.plank_half[1]} {c.plank_half[2]}"
            mass="{c.plank_mass}" rgba="0.85 0.65 0.25 1" friction="0.8 0.005 0.0001"/>
    </body>
    <body name="cube" pos="0.25 0 {c.cube_half}">
      <freejoint/>
      <geom name="cube_g" type="box" size="{c.cube_half} {c.cube_half} {c.cube_half}"
            mass="{c.cube_mass}" rgba="0.9 0.15 0.15 1" friction="1.0 0.005 0.0001"/>
    </body>
    <body name="bin" pos="{c.bin_x} 0.15 0">
      <geom name="bin_wall_xp" type="box" pos="{wall_off} 0 {wall_z}" size="{wall_t} {wall_len} {wall_z}"
            rgba="0.25 0.35 0.75 1"/>
      <geom name="bin_wall_xn" type="box" pos="-{wall_off} 0 {wall_z}" size="{wall_t} {wall_len} {wall_z}"
            rgba="0.25 0.35 0.75 1"/>
      <geom name="bin_wall_yp" type="box" pos="0 {wall_off} {wall_z}" size="{wall_len} {wall_t} {wall_z}"
            rgba="0.25 0.35 0.75 1"/>
      <geom name="bin_wall_yn" type="box" pos="0 -{wall_off} {wall_z}" size="{wall_len} {wall_t} {wall_z}"
            rgba="0.25 0.35 0.75 1"/>
      <site name="bin_floor" type="box" pos="0 0 0.0006" size="{c.bin_inner} {c.bin_inner} 0.0005"
            rgba="0.15 0.35 0.9 0.5"/>
    </body>
    <site name="rest" type="box" pos="{c.rest_x} -0.15 0.0006" size="0.05 0.13 0.0005"
          rgba="0.2 0.8 0.3 0.5"/>
  </worldbody>
</mujoco>
"""

    def reset_instance(self, rng: np.random.Generator) -> None:
        c = self.cfg
        self._clear_latches()
        bin_y = float(rng.uniform(-0.24, 0.24))
        while True:  # keep the rest pad clear of the bin footprint
            rest_y = float(rng.uniform(-0.24, 0.24))
            if abs(rest_y - bin_y) >= 0.20:
                break
        cube_xy = np.array([rng.uniform(0.16, 0.34), rng.uniform(-0.24, 0.24)])
        while True:  # plank spawn must not intersect the bin walls
            plank_xy = np.array([rng.uniform(-0.32, -0.22), rng.uniform(-0.20, 0.20)])
            if abs(plank_xy[1] - bin_y) >= 0.16:
                break
        yaw = float(rng.uniform(-np.pi, np.pi))
        self.model.body("bin").pos[1] = bin_y
        self.model.site("rest").pos[1] = rest_y
        self._bin_xy = np.array([c.bin_x, bin_y])
        self._rest_xy = np.array([c.rest_x, rest_y])
        self.set_body_pose("cube", (*cube_xy, c.cube_half))
        self.set_body_pose("plank", (*plank_xy, c.plank_half[2]),
                           (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
        self._d0 = float(np.linalg.norm(cube_xy - self._bin_xy))

    # ---- latches (updated every physics step) -----------------------------------
    def _clear_latches(self) -> None:
        self._bridged = False    # plank has been observed genuinely spanning the trench
        self._crossed = False    # cube touched the plank over the trench while it spanned
        self._delivered = False  # cube settled in the bin after a legitimate crossing
        self._approach = 0.0     # best progress toward the bin, gated on _crossed

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            super().step(1)
            self._update_latches()

    def _update_latches(self) -> None:
        spanning = self._plank_spanning()
        if spanning:
            self._bridged = True
        if spanning and abs(self.data.xpos[self._cube_bid][0]) < self.cfg.gap_half - 0.01 \
                and self._cube_plank_contact():
            self._crossed = True
        if self._crossed:
            d = float(np.linalg.norm(self.data.xpos[self._cube_bid][:2] - self._bin_xy))
            self._approach = max(self._approach, float(np.clip(1.0 - d / max(self._d0, 1e-6), 0.0, 1.0)))
            if self._cube_in_bin(settled=True):
                self._delivered = True

    def get_state(self) -> np.ndarray:
        extra = np.array([self._bridged, self._crossed, self._delivered, self._approach], dtype=np.float64)
        return np.concatenate([super().get_state(), extra])

    def set_state(self, buf: np.ndarray) -> None:
        super().set_state(buf[: buf.size - self._N_LATCH])
        b, x, d, a = buf[buf.size - self._N_LATCH:]
        self._bridged, self._crossed, self._delivered = bool(b), bool(x), bool(d)
        self._approach = float(a)

    # ---- geometry predicates -----------------------------------------------------
    def _plank_spanning(self) -> bool:
        """Plank lying flat with both ends overhanging the trench edges."""
        c = self.cfg
        pos = self.data.xpos[self._plank_bid]
        axis = self.data.xmat[self._plank_bid].reshape(3, 3)[:, 0]  # long axis in world
        e1 = pos + c.plank_half[0] * axis
        e2 = pos - c.plank_half[0] * axis
        need = c.gap_half + c.span_overlap
        flat = abs(axis[2]) < 0.3 and pos[2] < 0.025 and max(e1[2], e2[2]) < 0.03
        return flat and min(e1[0], e2[0]) < -need and max(e1[0], e2[0]) > need

    def _cube_plank_contact(self) -> bool:
        pair = {self._cube_gid, self._plank_gid}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if {int(con.geom1), int(con.geom2)} == pair:
                return True
        return False

    def _cube_in_bin(self, settled: bool = True) -> bool:
        c = self.cfg
        p = self.data.xpos[self._cube_bid]
        d = np.abs(p[:2] - self._bin_xy)
        ok = d[0] < c.bin_tol and d[1] < c.bin_tol and abs(p[2] - c.cube_half) < 0.012
        if settled:
            ok = ok and float(np.linalg.norm(self.data.cvel[self._cube_bid])) < c.settle_speed
        return bool(ok)

    def _plank_in_rest(self) -> bool:
        c = self.cfg
        p = self.data.xpos[self._plank_bid]
        return (float(np.linalg.norm(p[:2] - self._rest_xy)) < c.rest_tol
                and p[2] < 0.02
                and float(np.linalg.norm(self.data.cvel[self._plank_bid])) < c.settle_speed)

    # ---- semantics ---------------------------------------------------------------
    def success(self) -> bool:
        return self._crossed and self._cube_in_bin(settled=True) and self._plank_in_rest()

    def score(self) -> float:
        if self.success():
            return 1.0
        v = (0.20 * self._bridged
             + 0.25 * self._crossed
             + 0.25 * self._delivered
             + 0.10 * self._approach
             + 0.15 * (self._delivered and self._plank_in_rest()))
        return min(0.95, float(v))

    def describe(self) -> str:
        c = self.cfg
        return (
            "Two platforms are separated by an open trench "
            f"({2 * c.gap_half:.2f} m wide, along y). A red cube sits on the far platform; a "
            "plank, a walled bin, and a green rest pad are on the near platform. Dragging or "
            "sliding the cube straight across drops it into the pit. Required, in order: "
            "(1) lay the plank flat across the trench so BOTH ends rest past the platform "
            "edges; (2) move the cube across the trench ON the bridge — it must be in contact "
            "with the spanning plank while over the trench; (3) deposit the cube inside the "
            f"bin, settled with its center within {c.bin_tol:.2f} m (per axis) of the bin "
            "center; (4) return the plank so its center rests on the green pad (within "
            f"{c.rest_tol:.2f} m). Success requires all of: the bridge crossing happened, the "
            "cube is settled in the bin, and the plank is settled on the pad."
        )
