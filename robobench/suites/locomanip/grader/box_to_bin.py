"""Grader for the box-to-bin scene: the box taken off the shelf, walked around the corner, and left in the bin.

Rubric stages (weights and modes live in RUBRIC) — the scene's own chain lifted -> transited ->
placed, one rung each:
    picked   the scene's `lifted` latch: the box got clear of its shelf board, either raised
             `cfg.lift_h` above its resting height or seen held up in the apron in front of the
             shelf face (both pick styles the open shelving affords); a milestone
    carried  the scene's own `carried_fraction()` — the fraction of the start-to-bin xy distance
             the box has closed, a linear ramp — reading 1 once the scene's `transited` latch has
             fired (its own "the transit is done" mark, which `success()` requires); a milestone,
             so a box that arrives and is released keeps the credit
    in_bin   the scene's own `placed()` — fully inside the bin's inset interior bands, below the
             rim-excluding height cap, and settled; live, so it dips if the box leaves the bin
Success is the scene's own `success()` — `lifted & transited & placed()` — read on the final
state; the three rungs are exactly its three conjuncts, so progress reads 1.0 iff it holds.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.locomanip.scenes.box_to_bin import BoxToBinScene


class BoxToBinGrader(BaseGrader):
    """The cardboard box picked off the shelf, carried around the corner, and resting inside the sorting bin.

    Ladder: picked 0.333 · carried 0.667 (linear in the distance closed) · in bin 1.00
    """

    SCENE = BoxToBinScene
    RUBRIC = (("picked", 1, "once"), ("carried", 1, "once"), ("in_bin", 1))
    scene: BoxToBinScene

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

    def in_bin(self):
        return self.scene.placed().float()
