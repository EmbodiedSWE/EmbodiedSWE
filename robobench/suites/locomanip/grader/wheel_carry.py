"""Grader for the wheel carry scene: the wheel picked up here, walked over there, and left in the basket.

Rubric stages (weights and modes live in RUBRIC) — the scene's own chain lifted -> transited ->
placed, one rung each:
    picked     the scene's `lifted` latch: the wheel's origin raised `cfg.lift_h` above the pick
               table top; a milestone, credit kept after it is put down
    carried    the scene's own `carried_fraction()` — the fraction of `cfg.carry_dx` the wheel
               has travelled along +x from the pick station, a linear ramp — reading 1 once the
               scene's `transited` latch has fired (its own "the transit is done" mark, which
               `success()` requires); a milestone, so a wheel that arrives and is released keeps
               the credit
    in_basket  the scene's own `placed()` — inside the basket's target bands, below the height
               cap, and settled; live, so it dips if the wheel leaves the basket
Success is the scene's own `success()` — `lifted & transited & placed()` — read on the final
state; the three rungs are exactly its three conjuncts, so progress reads 1.0 iff it holds.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.locomanip.scenes.wheel_carry import WheelCarryScene


class WheelCarryGrader(BaseGrader):
    """The steering wheel lifted off the near table, carried across, and resting inside the far basket.

    Ladder: picked 0.333 · carried 0.667 (linear in the distance closed) · in basket 1.00
    """

    SCENE = WheelCarryScene
    RUBRIC = (("picked", 1, "once"), ("carried", 1, "once"), ("in_basket", 1))
    scene: WheelCarryScene

    def setup(self) -> None:
        pass  # every reference is the scene's own (latches, cfg bands); nothing to capture

    def check_success(self):
        return self.scene.success()

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] -----------------------
    def picked(self):
        return self.scene._lifted.float()

    def carried(self):
        import torch

        s = self.scene
        return torch.where(s._transited, torch.ones_like(s._transited, dtype=torch.float32),
                           s.carried_fraction().clamp(0.0, 1.0))

    def in_basket(self):
        return self.scene.placed().float()
