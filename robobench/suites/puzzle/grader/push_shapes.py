"""Grader for the push_shapes scene: the T, X and L blocks each pushed into their own cutout.

Rubric stages (weights and modes live in RUBRIC). The three blocks are identical sub-goals,
so every stage is ONE fraction over the env's three blocks (k/3), never a stage per block:
    yaw_aligned  blocks whose orientation is inside the scene's coarse yaw band of their pad
                 (`yaw_matched`: cfg.yaw_stage_tolerance_deg) — the pivot push is done
    near_pad     blocks whose centre is within cfg.near_xy_tolerance of their own pad centre —
                 the translation push has arrived at the cutout
    seated       blocks seated in their recess AND settled — the scene's own per-piece success
                 gate (`seated() & settled()`: xy within cfg.xy_tolerance, orientation within
                 cfg.orientation_tolerance_deg, not lifted, dropped into the recess, at rest)
All three are LIVE: a block knocked back out of tolerance loses the credit, exactly as the
scene's own `success()` is a live conjunction. Every rung is implied by the next (7 mm inside
30 mm, 7 deg inside 10 deg), so a seated block earns all three. Success is the scene's own
`success()` — all three blocks seated and settled — read on the final state.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.puzzle.scenes.push_shapes import PIECES, PushShapesScene


class PushShapesGrader(BaseGrader):
    """All three blocks pushed into their own cutouts — aligned, dropped in, and at rest.

    Ladder (3 blocks, 3 equal rungs each, 1/9 per rung): per block yaw aligned +1/9 ·
    near its pad +1/9 · seated +1/9; 1 block seated 0.333 · 2 blocks seated 0.667 ·
    3 blocks seated 1.00.
    """

    SCENE = PushShapesScene
    # Equal rungs: each is a distinct physical event the scene measures with its own band
    # (yaw_stage_tolerance_deg, near_xy_tolerance, seated()); each stage is a fraction over
    # the three blocks, so one block contributes 1/3 of each rung.
    RUBRIC = (("yaw_aligned", 1), ("near_pad", 1), ("seated", 1))
    scene: PushShapesScene

    def setup(self) -> None:
        pass  # everything is measured against the pads, which never move after reset

    def check_success(self):
        return self.scene.success()  # (num_envs,) — all three seated AND settled, live

    # ---- rubric stages — each returns a (num_envs,) fraction over that env's blocks --------
    def yaw_aligned(self):
        import torch

        return torch.stack([self.scene.yaw_matched(k) for k in PIECES], dim=1).float().mean(dim=1)

    def near_pad(self):
        import torch

        tol = self.scene.cfg.near_xy_tolerance
        near = torch.stack([self.scene.position_error(k) <= tol for k in PIECES], dim=1)
        return near.float().mean(dim=1)

    def seated(self):
        import torch

        done = torch.stack([self.scene.seated(k) & self.scene.settled(k) for k in PIECES], dim=1)
        return done.float().mean(dim=1)
