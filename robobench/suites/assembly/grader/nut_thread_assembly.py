"""Grader for the nut_thread scene: every nut seated on a bolt.

Rubric stages (weights and modes live in RUBRIC), each a fraction over that env's nuts:
    lifted    nuts ever raised to the bolt-top rest height — origin at/above `bolt_height`
              in the nearest bolt's frame, the height the scene quotes for a nut resting on
              the bolt top and so the height a nut has to clear before it can be set on the
              bolt — or already aligned on a bolt (a nut on the bolt was necessarily lifted
              there); a milestone, latched per nut
    aligned   nuts aligned on a bolt — within `align_xy` of the bolt axis and tilted at most
              `align_axis_deg` off it, the scene's own two non-depth seat gates
    threaded  mean thread depth from the bolt-top rest height (`bolt_height`) down to the
              seat (`seat_z`), counted only while aligned — so a nut's value is 1.0 iff the
              scene calls it seated
The rungs nest (seated => aligned => lifted), so progress reaches 1.0 exactly when every
nut of the env is seated, and for a grader built after the pick as well as one that watched
it. Success is the scene's `seated()` per env, read when the delivery finishes.
"""

from __future__ import annotations

import math

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.nut_thread_assembly import NutThreadAssemblyScene


class NutThreadAssemblyGrader(BaseGrader):
    """Every nut seated — threaded down to seat depth, on the bolt axis, upright.

    Ladder: lifted 0.33 · aligned 0.67 · threaded 1.00 — three equal rungs for the three
    steps of the task (pick the nut up, set it on the bolt, screw it down); the last rung
    is a linear ramp of the nut's height from the bolt-top rest height (`bolt_height`) to
    the seat (`seat_z`). With N nuts every stage is the fraction of nuts.
    """

    SCENE = NutThreadAssemblyScene
    RUBRIC = (("lifted", 1, "once"), ("aligned", 1), ("threaded", 1))
    scene: NutThreadAssemblyScene

    def setup(self) -> None:
        import torch

        z, _, _ = self._on_bolt()
        self._lifted = torch.zeros_like(z, dtype=torch.bool)  # per-nut latch: ever lifted

    def check_success(self):
        return self.scene.seated().all(dim=1)  # (num_envs,) — every nut of that env

    def _on_bolt(self):
        """(height, aligned, thread), each (num_envs, num_nuts): the nut origin's height
        above its nearest bolt's origin, whether it passes the scene's xy + tilt seat gates,
        and its thread-depth fraction from the bolt-top rest height to the seat."""
        import torch

        c = self.scene.cfg
        off = self.scene._nut_offsets_in_bolt()  # (n, N_nut, B_bolt, 3) — privileged read
        near_dist, near = off[..., :2].norm(dim=-1).min(dim=-1)
        z = torch.gather(off[..., 2], 2, near.unsqueeze(-1)).squeeze(-1)
        aligned = (near_dist <= c.align_xy) & (
            self.scene._nut_axis_cos() >= math.cos(math.radians(c.align_axis_deg)))
        thread = ((c.bolt_height - z) / (c.bolt_height - c.seat_z)).clamp(0.0, 1.0)
        return z, aligned, thread

    # ---- rubric stages — each returns a (num_envs,) fraction over that
    # env's nuts -------------------------------------------------------------
    def lifted(self):
        z, aligned, _ = self._on_bolt()
        self._lifted |= (z >= self.scene.cfg.bolt_height) | aligned
        return self._lifted.float().mean(dim=1)

    def aligned(self):
        _, aligned, _ = self._on_bolt()
        return aligned.float().mean(dim=1)

    def threaded(self):
        _, aligned, thread = self._on_bolt()
        return (aligned * thread).mean(dim=1)
