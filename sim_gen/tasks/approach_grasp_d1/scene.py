"""approach_grasp_d1 — the seed's grasp-and-hold becomes an ordered tower build.

Seed (pick_place.approach_grasp_simple): approach ONE cube, grasp it, lift it — success
is a stable held grasp, i.e. the goal state is "object in the air".  Here the goal is a
settled STRUCTURE: three blocks of different sizes must be stacked bottom-up on a raised
pedestal (large on the pedestal, medium on large, small on medium), each pair xy-aligned
within tolerance and the whole tower at rest.  Holding any block in the air — the seed's
entire plan — satisfies nothing: success requires correct heights, correct identities at
each level, and near-zero velocities, so blocks must be released and left to settle.

Ordering is enforced by identity + geometry, not convention: success names WHICH block
sits at each level, so a small-first tower fails no matter how neat (and physically a
larger block overhanging a smaller base is fragile anyway).

Mechanism notes:
- Stage events ("large was placed", "medium was stacked", "full tower stood") cannot all
  be read off an arbitrary later state, so they are LATCHES updated every physics step;
  score is built from latches so correct behavior never loses earned credit (e.g. the
  large-block credit survives the jostling of stacking the next block on it).
- Latches are cleared in reset_instance and appended to get_state()/set_state() so
  snapshot/restore round-trips keep earned credit (and state_hash covers them).
- Stack predicates use RELATIVE poses (top vs. its actual base) so a tower that is
  globally imperfect but pairwise-aligned still reads correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_gen.core import BaseScene, SceneCfg, register_scene


@dataclass
class StackTowerSceneCfg(SceneCfg):
    half_l: float = 0.035                 # large block half-edge
    half_m: float = 0.025                 # medium block half-edge
    half_s: float = 0.018                 # small block half-edge
    mass_l: float = 0.08
    mass_m: float = 0.04
    mass_s: float = 0.02
    ped_half_w: float = 0.055             # pedestal top half-width
    ped_half_h: float = 0.03              # pedestal half-height (top surface at 2*ped_half_h)
    align_tol: float = 0.025              # per-pair xy center alignment tolerance
    z_tol: float = 0.012                  # resting-height tolerance per level
    settle_speed: float = 0.05            # |cvel| below which a block counts as settled
    camera_distance: float = 1.5
    camera_azimuth: float = 120.0
    camera_elevation: float = -30.0
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.08)


@register_scene("approach_grasp_d1")
class StackTowerScene(BaseScene):
    cfg: StackTowerSceneCfg

    _N_LATCH = 4  # placed_l, placed_m, tower_seen, approach — appended to the state vector

    def __init__(self, cfg: StackTowerSceneCfg | None = None):
        super().__init__(cfg or StackTowerSceneCfg())
        self._l_bid = self.model.body("block_l").id
        self._m_bid = self.model.body("block_m").id
        self._s_bid = self.model.body("block_s").id
        self._ped_xy = np.array(self.model.body("pedestal").pos[:2])
        self._ped_top = 2 * self.cfg.ped_half_h
        self._d0 = 1.0
        self._clear_latches()

    def xml(self) -> str:
        c = self.cfg
        return f"""
<mujoco model="approach_grasp_d1">
  <option timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.6 0.6 0.02"
          rgba="0.55 0.45 0.35 1" friction="1.0 0.005 0.0001"/>
    <body name="pedestal" pos="-0.24 0 {c.ped_half_h}">
      <geom name="pedestal_g" type="box" size="{c.ped_half_w} {c.ped_half_w} {c.ped_half_h}"
            rgba="0.35 0.35 0.42 1" friction="1.0 0.005 0.0001"/>
      <site name="ped_top" type="box" pos="0 0 {c.ped_half_h + 0.0006}"
            size="{c.ped_half_w} {c.ped_half_w} 0.0005" rgba="0.15 0.35 0.9 0.6"/>
    </body>
    <body name="block_l" pos="0.12 -0.15 {c.half_l}">
      <freejoint/>
      <geom name="block_l_g" type="box" size="{c.half_l} {c.half_l} {c.half_l}"
            mass="{c.mass_l}" rgba="0.85 0.20 0.15 1" friction="1.0 0.005 0.0001"/>
    </body>
    <body name="block_m" pos="0.24 0.02 {c.half_m}">
      <freejoint/>
      <geom name="block_m_g" type="box" size="{c.half_m} {c.half_m} {c.half_m}"
            mass="{c.mass_m}" rgba="0.20 0.65 0.25 1" friction="1.0 0.005 0.0001"/>
    </body>
    <body name="block_s" pos="0.12 0.18 {c.half_s}">
      <freejoint/>
      <geom name="block_s_g" type="box" size="{c.half_s} {c.half_s} {c.half_s}"
            mass="{c.mass_s}" rgba="0.95 0.75 0.15 1" friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

    def reset_instance(self, rng: np.random.Generator) -> None:
        c = self.cfg
        self._clear_latches()
        ped_xy = np.array([rng.uniform(-0.36, -0.16), rng.uniform(-0.22, 0.22)])
        self.model.body("pedestal").pos[:2] = ped_xy
        self._ped_xy = ped_xy
        spots: list[np.ndarray] = []
        while len(spots) < 3:  # blocks spawn on the opposite table half, mutually clear
            cand = np.array([rng.uniform(0.06, 0.38), rng.uniform(-0.28, 0.28)])
            if all(np.linalg.norm(cand - s) >= 0.11 for s in spots):
                spots.append(cand)
        for body, half, xy in (("block_l", c.half_l, spots[0]),
                               ("block_m", c.half_m, spots[1]),
                               ("block_s", c.half_s, spots[2])):
            yaw = float(rng.uniform(-np.pi, np.pi))
            self.set_body_pose(body, (*xy, half), (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
        self._d0 = float(np.linalg.norm(spots[0] - ped_xy))

    # ---- latches (updated every physics step) -----------------------------------
    def _clear_latches(self) -> None:
        self._placed_l = False    # large observed settled+centered on the pedestal
        self._placed_m = False    # medium observed settled+stacked on a placed large
        self._tower_seen = False  # full success condition observed at least once
        self._approach = 0.0      # best progress of the large block toward the pedestal

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            super().step(1)
            self._update_latches()

    def _update_latches(self) -> None:
        c = self.cfg
        p1 = self._large_on_pedestal() and self._settled(self._l_bid)
        if p1:
            self._placed_l = True
        p2 = p1 and self._stacked_on(self._m_bid, self._l_bid, c.half_l + c.half_m) \
            and self._settled(self._m_bid)
        if p2:
            self._placed_m = True
        if p2 and self._stacked_on(self._s_bid, self._m_bid, c.half_m + c.half_s) \
                and self._settled(self._s_bid):
            self._tower_seen = True
        d = float(np.linalg.norm(self.data.xpos[self._l_bid][:2] - self._ped_xy))
        self._approach = max(self._approach, float(np.clip(1.0 - d / max(self._d0, 1e-6), 0.0, 1.0)))

    def get_state(self) -> np.ndarray:
        extra = np.array([self._placed_l, self._placed_m, self._tower_seen, self._approach],
                         dtype=np.float64)
        return np.concatenate([super().get_state(), extra])

    def set_state(self, buf: np.ndarray) -> None:
        super().set_state(buf[: buf.size - self._N_LATCH])
        pl, pm, ts, ap = buf[buf.size - self._N_LATCH:]
        self._placed_l, self._placed_m, self._tower_seen = bool(pl), bool(pm), bool(ts)
        self._approach = float(ap)

    # ---- geometry predicates -----------------------------------------------------
    def _settled(self, bid: int) -> bool:
        return float(np.linalg.norm(self.data.cvel[bid])) < self.cfg.settle_speed

    def _large_on_pedestal(self) -> bool:
        c = self.cfg
        p = self.data.xpos[self._l_bid]
        return (float(np.linalg.norm(p[:2] - self._ped_xy)) < c.align_tol
                and abs(p[2] - (self._ped_top + c.half_l)) < c.z_tol)

    def _stacked_on(self, top_bid: int, base_bid: int, gap: float) -> bool:
        c = self.cfg
        t, b = self.data.xpos[top_bid], self.data.xpos[base_bid]
        return (float(np.linalg.norm(t[:2] - b[:2])) < c.align_tol
                and abs((t[2] - b[2]) - gap) < c.z_tol)

    # ---- semantics ---------------------------------------------------------------
    def success(self) -> bool:
        c = self.cfg
        return (self._large_on_pedestal()
                and self._stacked_on(self._m_bid, self._l_bid, c.half_l + c.half_m)
                and self._stacked_on(self._s_bid, self._m_bid, c.half_m + c.half_s)
                and all(self._settled(b) for b in (self._l_bid, self._m_bid, self._s_bid)))

    def score(self) -> float:
        if self.success():
            return 1.0
        v = (0.15 * self._approach
             + 0.30 * self._placed_l
             + 0.30 * self._placed_m
             + 0.20 * self._tower_seen)
        return min(0.95, float(v))

    def describe(self) -> str:
        c = self.cfg
        return (
            "Three loose blocks — large red, medium green, small yellow — lie scattered on a "
            "table; a raised gray pedestal (top marked in blue) stands on the other side. Build "
            "a size-ordered tower ON the pedestal, bottom-up: first set the LARGE block down "
            f"centered on the pedestal top (xy within {c.align_tol:.3f} m), then stack the "
            "MEDIUM block on the large one, then the SMALL block on the medium one — each "
            f"block's center within {c.align_tol:.3f} m (xy) of the block beneath it and "
            "resting at the correct height. Success requires the finished tower to be at rest: "
            "every block released and settled (near-zero velocity). A block held in the air, a "
            "tower built on the table instead of the pedestal, or a tower in the wrong size "
            "order does not count."
        )
