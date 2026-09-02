"""Grader for the slice scene: the food cut into its target number of pieces, on the board.

Rubric stages (weights and modes live in RUBRIC):
    knife_taken   the knife has left its presentation stand — a milestone
    planes_cut    fraction of scored planes released
    on_board      every piece still on/near the chopping board (a piece flung off the
                  board is a failed cut, however many planes were released)
Every stage is per env. Success = piece count (connected components of the live weld
chain) reaches the target AND every piece is on the board, read when the delivery finishes.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.cutting.scenes.slice_food import SliceFoodScene


class SliceFoodGrader(BaseGrader):
    """Food sliced into the target number of pieces, all still on the board."""

    SCENE = SliceFoodScene
    RUBRIC = (("knife_taken", 0.1, "once"), ("planes_cut", 0.7), ("on_board", 0.2))
    taken_dist = 0.05  # m the knife root must move off its stand to count as taken
    board_margin = 0.05  # m beyond the board footprint still counted as "on the board"
    scene: SliceFoodScene

    def setup(self) -> None:
        self._knife_p0 = self.scene.knife.data.root_pos_w.clone()  # presented pose at reset

    def check_success(self):
        c = self.scene.cfg
        target = c.target_pieces or len(self.scene.pieces)
        return (self.scene.pieces_count() >= target) & self._on_board()

    def _on_board(self):
        import torch

        s = self.scene
        c = s.cfg
        hx, hy = c.board_size[0] / 2 + self.board_margin, c.board_size[1] / 2 + self.board_margin
        ok = torch.ones(s.env.num_envs, dtype=torch.bool, device=s.env.device)
        for pc in s.pieces:
            rel = pc.data.root_pos_w - s.env_origins
            ok &= (rel[:, 0].abs() < hx) & (rel[:, 1].abs() < hy)
            ok &= (rel[:, 2] > s._board_top - 0.02) & (rel[:, 2] < s._board_top + 0.3)
        return ok

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------
    def knife_taken(self):
        moved = (self.scene.knife.data.root_pos_w - self._knife_p0).norm(dim=-1)
        return (moved > self.taken_dist).float()

    def planes_cut(self):
        return self.scene.cut.float().mean(dim=1)

    def on_board(self):
        return self._on_board().float()
