"""Grader for the pen_holder scene: every present pen in the holder tip-up, holder set down upright.

Rubric stages (weights and modes live in RUBRIC):
    pens_in        fraction of the PRESENT pens the scene counts as in the holder — the
                   scene's own `counted()`: bottom within `xy_tol` of the holder axis,
                   deeper than `depth_min` below the rim, tip-up within `pen_align_max_deg`,
                   holder within `holder_tilt_max_deg` of up, pen and holder settled. All
                   pens are one identical family, so they share this one fraction stage;
                   the fraction is over the sampled subset (the present pen count is drawn
                   per env at reset, k ~ U{min_present..n} per family)
    holder_placed  the loaded holder standing upright on the work surface (the scene's
                   `holder_placed()`: within `placed_tilt_deg`, bottom within `placed_z_tol`
                   of the surface, still), gated on every present pen being in. The holder
                   spawns upright on the surface, so ungated it would be free credit at
                   reset; the order is fill, then set down
Both stages are live: a pen ejected or a holder knocked over loses its credit. Success is
the scene's own `success()` — `all_inserted() & holder_placed()` — on the final state.

Not measured: a grasped/lifted rung for the pens. The scene exposes no lift height or hold
criterion, and inventing one is forbidden, so that rung is dropped.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.packing.scenes.pen_holder import PenHolderScene


class PenHolderGrader(BaseGrader):
    """Every present pen in the holder tip-up, and the loaded holder standing upright on the surface.

    Ladder (N present pens): k pens in 0.90 * k / N · holder set down 1.00
        N = 4: 1 pen 0.225 · 2 pens 0.45 · 3 pens 0.675 · 4 pens 0.90 · set down 1.00
        N = 3: 1 pen 0.30 · 2 pens 0.60 · 3 pens 0.90 · set down 1.00
    """

    SCENE = PenHolderScene
    # 0.9 : 0.1 — the identical pens share one fraction stage; setting the filled holder
    # down is one short action after up to four insertions, so like the egg carton's lid
    # it is worth less than a pen.
    RUBRIC = (("pens_in", 0.9), ("holder_placed", 0.1))
    scene: PenHolderScene

    def setup(self) -> None:
        pass  # every quantity is read live from the scene's own predicates

    def check_success(self):
        return self.scene.success()  # (num_envs,) all present pens in AND holder set down

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------
    def pens_in(self):
        counted = self.scene.counted()  # (n, P) inserted & settled & present
        n_present = self.scene.present.sum(dim=1).clamp(min=1)
        return counted.sum(dim=1).float() / n_present.float()

    def holder_placed(self):
        return (self.scene.all_inserted() & self.scene.holder_placed()).float()
