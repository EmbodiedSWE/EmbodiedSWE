"""Grader for the so101 scene: the elbow servo seated + tab-screwed into the upper arm, the forearm
clipped onto its horn + horn-screwed.

The scene defines no `success()`; the criterion below is read off its own assembly geometry
and fastening state (see `check_success`):
    every loose screw fastened — the four identical M2 tab screws in the four tab holes and the
    eight identical M3 horn screws in the eight horn-line holes (the scene's own `state == 2`,
    "fully fastened") — with the servo live in the arm pocket and the forearm live on the horn.

Rubric stages (weights and modes live in RUBRIC):
    motor_seated          the bare servo sits in the upper arm's pocket: its body frame on the
                          upper_arm link frame (identical by construction when seated) within the
                          scene's fastening-gate tolerances `motor_align_pos` / `motor_align_deg`
                          — the scene's own PARTS ALIGNED rule for the tab holes
    tab_screws_picked     fraction of the four identical M2 tab screws ever carried on the drill's
                          magnetic bit (`attached`, the scene's only "grasp" of a screw; a
                          fastened screw was necessarily driven off that bit) — a milestone
    tab_screws_fastened   fraction of the four M2s welded home in a tab hole (`fastened`)
    fork_seated           the forearm's clevis sits on the servo's output horn: the lower_arm at
                          the elbow-joint seat transform of the motor, within the same
                          `motor_align_pos` / `motor_align_deg` gate, angle-agnostic about the
                          horn axis (the assembled elbow may stand at any angle) — the scene's
                          PARTS ALIGNED rule for the horn holes
    horn_screws_picked    fraction of the eight identical M3 horn screws ever carried on the bit
    horn_screws_fastened  fraction of the eight M3s welded home in a horn-line hole
Every stage is per env. The two "picked" milestones are the grasped rung of the screws' ladder;
the servo and the forearm have no grasp the scene can measure (no grasp contract, no lift
threshold in cfg), so their ladders start at the seated rung. A screw's transient "in the hole,
not yet home" state is not a rung: the auto-driver takes it from engaged to fastened by itself.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.so101_assembly import SO101AssemblyScene


class SO101AssemblyGrader(BaseGrader):
    """The SO101 elbow assembled: servo seated and tab-screwed into the upper arm, forearm clipped onto the horn and horn-screwed.

    Ladder (14 units — one unit per part: servo 1, 4 M2 tab screws 4, forearm 1, 8 M3 horn
    screws 8; a screw's unit split equally over its two rungs, picked 0.5 + fastened 0.5):
        servo in pocket 1/14 = 0.071 · 4 M2 on the bit 3/14 = 0.214 · 4 M2 fastened 5/14 = 0.357 ·
        forearm on horn 6/14 = 0.429 · 8 M3 on the bit 10/14 = 0.714 · 8 M3 fastened 14/14 = 1.000
    The identical screws of each group share one k/N fraction stage weighted N/2 per rung.
    """

    SCENE = SO101AssemblyScene
    # one unit per part; a screw's unit split equally over its two rungs (picked / fastened),
    # so a group of N identical screws weighs N/2 on each rung
    RUBRIC = (
        ("motor_seated", 1),
        ("tab_screws_picked", 2, "once"),
        ("tab_screws_fastened", 2),
        ("fork_seated", 1),
        ("horn_screws_picked", 4, "once"),
        ("horn_screws_fastened", 4),
    )
    scene: SO101AssemblyScene

    def setup(self) -> None:
        import torch

        # per-screw latch: ever rode the magnetic bit (or was fastened, which requires it)
        self._ever_on_bit = torch.zeros(self.num_envs, self.scene.cfg.num_screws,
                                        dtype=torch.bool, device=self.env.device)

    def check_success(self):
        """Every loose screw fastened (the scene's `state == 2`) AND the servo live in the
        pocket AND the forearm live on the horn — the assembled elbow, on the final state."""
        all_fastened = (self.scene.fastened >= 0).all(dim=1)
        return all_fastened & self._motor_seated() & self._fork_seated()

    # ---- predicates (the scene's own PARTS ALIGNED gates) ------------------------------------
    def _motor_seated(self):
        import math

        from isaaclab.utils.math import quat_error_magnitude

        s, c = self.scene, self.scene.cfg
        ap, aq = s.upper_arm_pose()
        return (((s.motor.data.root_pos_w - ap).norm(dim=-1) < c.motor_align_pos)
                & (quat_error_magnitude(s.motor.data.root_quat_w, aq)
                   < math.radians(c.motor_align_deg)))

    def _fork_seated(self):
        import math

        s, c = self.scene, self.scene.cfg
        lp, lq = s.lower_arm_pose()
        seat_p, seat_q = s.lower_arm_seat_w()
        _, off_axis = s._elbow_angle_split(lq, seat_q)  # privileged: residual off the horn axis
        return (((lp - seat_p).norm(dim=-1) < c.motor_align_pos)
                & (off_axis < math.radians(c.motor_align_deg)))

    def _picked(self, lo: int, hi: int):
        s = self.scene
        self._ever_on_bit |= s.attached | (s.fastened >= 0)
        return self._ever_on_bit[:, lo:hi].float().mean(dim=1)

    def _fastened(self, lo: int, hi: int):
        return (self.scene.fastened[:, lo:hi] >= 0).float().mean(dim=1)

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] --------------------------
    def motor_seated(self):
        return self._motor_seated().float()

    def tab_screws_picked(self):
        return self._picked(0, self.scene.cfg.num_elbow_screws)

    def tab_screws_fastened(self):
        return self._fastened(0, self.scene.cfg.num_elbow_screws)

    def fork_seated(self):
        return self._fork_seated().float()

    def horn_screws_picked(self):
        return self._picked(self.scene.cfg.num_elbow_screws, self.scene.cfg.num_screws)

    def horn_screws_fastened(self):
        return self._fastened(self.scene.cfg.num_elbow_screws, self.scene.cfg.num_screws)
