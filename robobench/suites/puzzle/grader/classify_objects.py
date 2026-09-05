"""Grader for the classify_objects scene: every coloured block sorted into its matching zone.

Rubric stages (weights and modes live in RUBRIC):
    sorted_blocks  fraction of the env's blocks correctly sorted — the scene's own
                   `sorted_mask()`: centre within cfg.zone_half (xy) of ITS OWN category's
                   (jittered) zone centre, resting on the surface (cfg.z_tol) and settled
                   (cfg.settle_speed)
The blocks are identical pick-and-place sub-goals, so they share this ONE fraction stage
(k/B). It is LIVE: a block knocked off its zone loses the credit, which is what makes the
final state the only one that counts. No grasp/lift rung: the scene defines no lift
threshold, so it is dropped rather than invented. Success is the scene's own `success()`
— every block sorted — read on the final state.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.puzzle.scenes.classify_objects import ClassifyObjectsScene


class ClassifyObjectsGrader(BaseGrader):
    """Every block resting, settled, inside the zone of its own colour.

    Ladder (k / B blocks sorted; B = len(cfg.manifest), 3 by default): 1 block 0.333 ·
    2 blocks 0.667 · 3 blocks 1.00.
    """

    SCENE = ClassifyObjectsScene
    RUBRIC = (("sorted_blocks", 1),)
    scene: ClassifyObjectsScene

    def setup(self) -> None:
        pass  # zones are captured by the scene itself at reset (`_zones`); nothing to cache

    def check_success(self):
        return self.scene.success()  # (num_envs,) — every block of that env sorted

    # ---- rubric stages — each returns a (num_envs,) fraction over that env's blocks --------
    def sorted_blocks(self):
        return self.scene.n_sorted().float() / len(self.scene.cfg.manifest)
