"""Grader for the slice scene: the food cut into its target number of pieces, all still on the board.

Rubric stages (weights and modes live in RUBRIC):
    knife_taken  the knife has been lifted clear of its rest — its root raised by at least
                 cfg.rest_rail_h (the rails' height) above where it lay at the graded reset, so
                 the edge is above the rail tops and the notches no longer hold it; a milestone
                 ("once"), credit kept after the knife is set down again
    planes_cut   fraction of the cuts the target needs that have been made — released planes
                 over (target - 1), a linear ramp from the welded food (0) to the target piece
                 count (1), target = cfg.target_pieces or every piece. The scored planes are
                 identical sub-goals: one stage, one fraction, each plane worth the same
    sliced       the scene's own done test: piece count (connected components of the live weld
                 chain, `pieces_count()`) reaches the target AND every piece is still on the
                 chopping board — inside the board's footprint (cfg.board_size) and above its
                 top. "On the board" is a GATE on this stage, not a stage of its own: it is
                 true at reset and only means anything once the food is cut
Success = `sliced` read when the delivery finishes.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.cutting.scenes.slice_food import SliceFoodScene


class SliceFoodGrader(BaseGrader):
    """Food sliced into the target number of pieces, every piece still on the board.

    Ladder (7 scored planes — the carrot / banana assets — target = all pieces): knife taken
    1/9 (0.111) · each plane cut +1/9 (7 planes -> 8/9 = 0.889) · sliced with every piece on
    the board 1.00.
    """

    SCENE = SliceFoodScene
    # One weight unit per step of the task: taking the knife (1), releasing each scored plane
    # (7 identical sub-goals -> one fraction stage carrying 7 units, so a single cut is worth
    # exactly as much as the pick), and the final done gate (1). The class default is the 7
    # planes of the carrot / banana assets; __init__ re-derives the unit count from the
    # graded scene's own target so the rule holds for any food or `target_pieces`.
    RUBRIC = (("knife_taken", 1, "once"), ("planes_cut", 7), ("sliced", 1))
    scene: SliceFoodScene

    def __init__(self, env) -> None:
        if isinstance(env.scene, self.SCENE):  # else leave the default; BaseGrader raises
            cuts = self._cuts_needed(env.scene)
            self.RUBRIC = (("knife_taken", 1, "once"), ("planes_cut", cuts), ("sliced", 1))
        super().__init__(env)

    @staticmethod
    def _cuts_needed(scene: SliceFoodScene) -> int:
        target = scene.cfg.target_pieces or len(scene.pieces)
        if target <= 1:
            raise ValueError(f"target_pieces={target} needs no cut — the planes ramp "
                             "(target - 1 cuts) is undefined")
        return target - 1

    def setup(self) -> None:
        self._knife_z0 = self.scene.knife.data.root_pos_w[:, 2].clone()  # resting on the rails
        self._target = self._cuts_needed(self.scene) + 1

    def check_success(self):
        return self._sliced()

    def _on_board(self):
        """(num_envs,) bool: every piece's root inside the board's footprint and above its top.
        The kinematic board is centred on the env origin (assets(): pos (0, 0)), so its
        half-extents are the test; a piece whose centre leaves the footprint tips off."""
        import torch

        s = self.scene
        hx, hy = s.cfg.board_size[0] / 2, s.cfg.board_size[1] / 2
        ok = torch.ones(s.env.num_envs, dtype=torch.bool, device=s.env.device)
        for pc in s.pieces:
            rel = pc.data.root_pos_w - s.env_origins
            ok &= (rel[:, 0].abs() < hx) & (rel[:, 1].abs() < hy) & (rel[:, 2] > s._board_top)
        return ok

    def _sliced(self):
        return (self.scene.pieces_count() >= self._target) & self._on_board()

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------
    def knife_taken(self):
        rise = self.scene.knife.data.root_pos_w[:, 2] - self._knife_z0
        return (rise >= self.scene.cfg.rest_rail_h).float()

    def planes_cut(self):
        made = (self.scene.pieces_count() - 1).float()
        return (made / (self._target - 1)).clamp(0.0, 1.0)

    def sliced(self):
        return self._sliced().float()
