"""Grader for the dice scene: the food diced into its target number of pieces, on the board.

Same rubric shape as the slice grader (knife_taken / planes_cut / on_board); `planes_cut`
counts released weld PAIRS (the dice gate is per pair), and success = connected components
of the live weld graph reach the target AND every piece is on the board.
"""

from __future__ import annotations

from robobench.suites.cutting.scenes.dice_food import DiceFoodScene

from .slice_food import SliceFoodGrader


class DiceFoodGrader(SliceFoodGrader):
    """Food diced into the target number of pieces, all still on the board."""

    SCENE = DiceFoodScene
    scene: DiceFoodScene

    def planes_cut(self):
        return self.scene.pair_cut.float().mean(dim=1)  # fraction of weld pairs released
