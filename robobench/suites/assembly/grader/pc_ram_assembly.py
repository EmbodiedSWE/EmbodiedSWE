"""Grader for the pc_ram scene: both RAM sticks seated in their DIMM slots.

Rubric stages (weights and modes live in RUBRIC):
    removed   fraction of sticks taken out of their start spot — displaced
              >= `remove_dist` in xy from where they were at the graded reset
              (a holder-to-slot trip is ~0.3 m). Read from the CURRENT state
              on purpose, not accumulated: the data engine grades verdict-only
              (grader built at the entry state, verdict() at the end, no
              per-step record), where a stepping milestone can never score
    engaged   fraction of sticks whose blade is in its slot — within the
              align gate in xy and at/below the slot mouth
    inserted  mean insertion fraction over the 4.44 mm stroke (mouth -> channel
              floor), engaged sticks only; unengaged sticks count 0
Every stage is per env (fractions over that env's sticks). Success is the
scene's `seated()` per env — depth, xy, tilt AND heading gates — read when the
delivery finishes.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.pc_ram_assembly import PcRamAssemblyScene


class PcRamAssemblyGrader(BaseGrader):
    """Both sticks seated — pressed to depth, centred, upright, heading along the slot."""

    SCENE = PcRamAssemblyScene
    RUBRIC = (("removed", 0.2), ("engaged", 0.2), ("inserted", 0.6))
    remove_dist = 0.05  # m of xy displacement from the start spot to count as taken out
    scene: PcRamAssemblyScene

    def setup(self) -> None:
        import torch

        self._xy0 = torch.stack(  # start xy, captured at the graded reset
            [r.data.root_pos_w[:, 0:2] for r in self.scene.rams], dim=1)  # (n, S, 2)
        # full stroke: slot mouth -> channel floor (= the seated origin), per slot
        c = self.scene.cfg
        self._stroke = torch.tensor([c.slot_mouth_z - p[2] for p in c.seat_pos])

    def check_success(self):
        return self.scene.seated().all(dim=1)  # (num_envs,) — every stick of that env

    def _in_slot(self):
        """(engaged mask, insertion fraction), each (num_envs, num_slots)."""
        c = self.scene.cfg
        rel = self.scene._ram_offsets_in_case()  # (n, S, 3), zero at the seated poses — privileged
        depth = self.scene.engaged()  # (n, S) blade depth below the slot mouth (m)
        engaged = (rel[..., 0:2].norm(dim=-1) <= c.align_xy) & (depth >= 0.0)
        frac = (depth / self._stroke.to(depth.device)).clamp(0.0, 1.0)
        return engaged, frac

    # ---- rubric stages — each returns a (num_envs,) fraction over that
    # env's sticks -----------------------------------------------------------
    def removed(self):
        import torch

        xy = torch.stack([r.data.root_pos_w[:, 0:2] for r in self.scene.rams], dim=1)
        return ((xy - self._xy0).norm(dim=-1) >= self.remove_dist).float().mean(dim=1)

    def engaged(self):
        engaged, _ = self._in_slot()
        return engaged.float().mean(dim=1)

    def inserted(self):
        engaged, frac = self._in_slot()
        return (engaged * frac).mean(dim=1)
