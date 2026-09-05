"""Grader for the allen_bolt scene: every bolt driven home with the key.

Rubric stages (weights and modes live in RUBRIC):
    key_grasped  a key has been taken in the weld-on-closure grip (`scene.grasp_held`, any
                 key, either arm) — the scene's own grasp state; a milestone, latched. Also
                 true once every bolt is seated (a seated bolt was necessarily driven by
                 the key), so the rung nests under `driven`. Under an embodiment with no
                 gripper (the weld contract disabled, e.g. robot="null") the grasp state does
                 not exist and the rung reads from that implication alone
    driven       mean over bolts of the tip's depth from the start register to the seat —
                 `bolt_stage_depth` -> `seat_depth` when the bolts spawn hand-started
                 (`bolt_staged`, every robot binding), plate top (depth 0) -> `seat_depth`
                 for the lying spawn — counted only while the bolt is within `align_xy` of the
                 hole axis and `align_axis_deg` of upright, the scene's own non-depth seat
                 gates, so a bolt's value is 1.0 iff the scene calls it seated
Both stages are per env. The key's own "seated in the hex socket" step is NOT a rung: the
head/socket geometry is not in the scene cfg, so it cannot be read from state without an
invented tolerance. Success is the scene's `success()` per env (every bolt `seated()`),
read when the delivery finishes.
"""

from __future__ import annotations

import math

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.allen_bolt_assembly import AllenBoltAssemblyScene


class AllenBoltAssemblyGrader(BaseGrader):
    """Every bolt seated — driven down to seat depth with the key, upright on the hole axis.

    Ladder: key_grasped 0.50 · driven 1.00 — two equal rungs for the two measurable steps
    (take the key, drive the bolt home); the second is a linear ramp of tip depth from the
    bolt's start register (`bolt_stage_depth`, or the plate top for a lying spawn) to
    `seat_depth`. With N bolts `driven` is the mean over bolts.
    """

    SCENE = AllenBoltAssemblyScene
    RUBRIC = (("key_grasped", 1, "once"), ("driven", 1))
    scene: AllenBoltAssemblyScene

    def setup(self) -> None:
        import torch

        c = self.scene.cfg
        self._depth0 = c.bolt_stage_depth if c.bolt_staged else 0.0  # tip depth at the start
        # `grasp_held` exists only while the weld contract is live (a gripper on the stage).
        self._has_grasp = hasattr(self.scene, "grasp_held")
        self._grasped = torch.zeros(self.num_envs, dtype=torch.bool,
                                    device=self.scene.engaged().device)  # latch: a key ever held

    def check_success(self):
        return self.scene.success()  # (num_envs,) — every bolt of that env seated

    def _in_hole(self):
        """(aligned, drive), each (num_envs, num_bolts): whether the bolt passes the scene's
        xy + tilt seat gates in its nearest platform, and its tip-depth fraction from the
        start register to the seat."""
        c = self.scene.cfg
        off = self.scene._bolt_offsets_in_platform()  # (n, B, P, 3) — privileged read
        near_dist = off[..., :2].norm(dim=-1).min(dim=-1).values
        aligned = (near_dist <= c.align_xy) & (
            self.scene._bolt_axis_cos() >= math.cos(math.radians(c.align_axis_deg)))
        depth = self.scene.engaged()  # (n, B) tip depth below the plate top (m)
        drive = ((depth - self._depth0) / (c.seat_depth - self._depth0)).clamp(0.0, 1.0)
        return aligned, drive

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------
    def key_grasped(self):
        if self._has_grasp:
            self._grasped |= self.scene.grasp_held.any(dim=1)
        return (self._grasped | self.scene.success()).float()

    def driven(self):
        aligned, drive = self._in_hole()
        return (aligned * drive).mean(dim=1)
