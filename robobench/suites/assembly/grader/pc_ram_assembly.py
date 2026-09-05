"""Grader for the pc_ram scene: both RAM sticks seated in their DIMM slots.

The two sticks are identical sub-goals, so every rung is ONE stage whose value is the
fraction of sticks (k/2) that have reached it. Per stick the ladder is grasped -> aligned ->
seated, equal weights:
    grasped  sticks ever taken in the weld-on-closure grip (`scene.grasp_held`, one column
             per stick) — the scene's own grasp state; a milestone, latched per stick. Also
             true for a stick that is seated (a seated stick was necessarily carried there),
             so the rung nests under `seated`. Under an embodiment with no gripper (the weld
             contract disabled, e.g. robot="null") the grasp state does not exist and the
             rung reads from that implication alone
    aligned  sticks lined up on their slot — origin within `align_xy` of the seated point, up
             axis within `align_axis_deg` of the slot axis, length axis within `align_yaw_deg`
             of the slot heading: the scene's own three non-depth seat gates
    seated   mean over sticks of the blade depth below the slot mouth (`slot_mouth_z`, the
             scene's own `engaged()`) as a fraction of `seat_depth`, counted only while
             aligned — so a stick's value is 1.0 iff the scene calls it `seated()`
The rungs nest (seated => aligned => grasped), so progress reaches 1.0 exactly when both
sticks are seated, and for a grader built after the picks as well as one that watched them.
Success is the scene's own `success()` (every stick `seated()`), read when the delivery
finishes.
"""

from __future__ import annotations

import math

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.pc_ram_assembly import PcRamAssemblyScene


class PcRamAssemblyGrader(BaseGrader):
    """Both sticks seated — pressed to depth, centred, upright, heading along the slot.

    Ladder (per stick, each rung 1/6): grasped 1/6 · aligned 2/6 · seated 3/6;
    one stick done 0.50 · both sticks done 1.00 — three equal rungs for the three steps of
    each stick (take it, line it up on its slot, press it home); the last rung is a linear
    ramp of blade depth from the slot mouth (`slot_mouth_z`) to `seat_depth`.
    """

    SCENE = PcRamAssemblyScene
    # Equal weights: the three rungs of one stick's ladder; every stage is a k/2 fraction.
    RUBRIC = (("grasped", 1, "once"), ("aligned", 1), ("seated", 1))
    scene: PcRamAssemblyScene

    def setup(self) -> None:
        import torch

        # `grasp_held` exists only while the weld contract is live (a gripper on the stage).
        self._has_grasp = hasattr(self.scene, "grasp_held")
        depth = self.scene.engaged()  # (n, S)
        self._grasped = torch.zeros_like(depth, dtype=torch.bool)  # per-stick latch: ever held

    def check_success(self):
        return self.scene.success()  # (num_envs,) — every stick of that env seated

    def _aligned(self):
        """(num_envs, num_slots) bool: the scene's xy + tilt + heading seat gates, without depth."""
        c = self.scene.cfg
        rel = self.scene._ram_offsets_in_case()  # (n, S, 3), zero at the seated poses — privileged
        xy_ok = rel[..., 0:2].norm(dim=-1) <= c.align_xy
        up_ok = self.scene._axis_cos(2) >= math.cos(math.radians(c.align_axis_deg))
        yaw_ok = self.scene._axis_cos(1) >= math.cos(math.radians(c.align_yaw_deg))
        return xy_ok & up_ok & yaw_ok

    # ---- rubric stages — each returns a (num_envs,) fraction over that env's sticks --------
    def grasped(self):
        if self._has_grasp:
            self._grasped |= self.scene.grasp_held[:, : self.scene.cfg.num_slots]
        return (self._grasped | self.scene.seated()).float().mean(dim=1)

    def aligned(self):
        return self._aligned().float().mean(dim=1)

    def seated(self):
        depth = self.scene.engaged()  # (n, S) blade depth below the slot mouth (m)
        return (self._aligned() * (depth / self.scene.cfg.seat_depth).clamp(0.0, 1.0)).mean(dim=1)
