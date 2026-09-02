"""Grader for the box-to-bin scene: the box taken off the shelf, walked around the corner, and left in the bin.

Rubric stages (weights and modes live in RUBRIC):
    picked   the box was really taken — clear of its shelf board AND quasi-static, sustained, so a
             box knocked flying never counts; a milestone, credit kept after it is put down.
             "Clear of the board" accepts both pick styles the open shelving affords: lifted
             straight up past the height gate, or seen held up in the apron just in front of the
             shelf face (the scene's `lifted` latch draws the same bounded line)
    carried  fraction of the horizontal distance from the box's start pose to the bin's centre
             that has been closed; a milestone, so a box that arrives and is released keeps the
             credit. The transit turns ~100 deg, so the fraction is radial, not per-axis
    in_bin   the scene's own `placed()` — fully inside the bin's walls, below rim height, at rest
    upright  the robot is still standing (see below)

Success is the scene's own `success()` — the journey latches AND `placed()` — AND `upright`. The
upright term is the half of the criterion the embodiment-agnostic scene cannot express: a robot
that topples against the low table could drop the box into the bin on the way down, and that is
not a solved pick-carry-place. With a robot that has no measurable base (e.g. NullRobot) the term
is vacuously satisfied and only the scene decides, which is what lets the scene smoke stage the
task by teleporting the box.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.locomanip.scenes.box_to_bin import BoxToBinScene


class BoxToBinGrader(BaseGrader):
    """The cardboard box carried around the corner into the sorting bin and released, the robot still on its feet."""

    SCENE = BoxToBinScene
    # Same weight split as the sibling wheel_carry grader (and the same rationale): `upright` is
    # live and true from the first step, so its 0.10 is the score a run earns by merely not
    # falling over — matching what "did nothing" scores across the suite.
    RUBRIC = (("picked", 0.20, "once"), ("carried", 0.25, "once"), ("in_bin", 0.45), ("upright", 0.10))
    # "picked" = clear of the board AND quasi-static, sustained. A box batted off the shelf can
    # pass the clearance gate for a moment, but in free flight its speed changes by ~2.5 m/s over
    # pick_hold_s, so it can never stay under pick_speed for that long continuously.
    pick_lift = 0.05  # m above the box's resting height on the board (the straight-lift path)
    pick_speed = 0.4  # m/s max speed while counted as held
    pick_hold_s = 0.25  # s both must hold, consecutively
    # Upright: the base link's height above the floor. A G1 walks with its pelvis at ~0.72 m and
    # the locomotion policy holds that to within a centimetre or two even in transit, so this
    # threshold is far below any healthy gait and only trips on an actual fall.
    upright_min_z = 0.45
    scene: BoxToBinScene

    def setup(self) -> None:
        import torch

        c = self.scene.cfg
        pos, _ = self.scene.box_pose()
        self._p0 = pos.clone()  # start pose, captured at the graded reset
        self._held_s = torch.zeros(self.num_envs, device=pos.device)
        self._bin_xy = torch.tensor(c.bin_pos, device=pos.device)
        self._span = (self._p0[:, :2] - self._bin_xy).norm(dim=-1).clamp(min=1e-6)
        # The upright term needs an embodiment with a floating base — same resolution as the
        # wheel_carry grader: None (vacuous) for NullRobot or any welded-root robot.
        robot = self.env.robot
        art = getattr(robot, "articulation", None)
        base = getattr(robot, "BASE_BODY", "pelvis")
        fixed = getattr(getattr(robot, "cfg", None), "fixed_base", True)
        self._base_idx = None
        if art is not None and not fixed and base in art.data.body_names:
            self._base_idx = art.data.body_names.index(base)

    def check_success(self):
        return self.scene.success() & (self.upright() > 0.5)

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------------
    def picked(self):
        import torch

        c = self.scene.cfg
        pos, _ = self.scene.box_pose()
        speed = self.scene.box.data.root_lin_vel_w.norm(dim=-1)
        clear = (pos[:, 2] > self._p0[:, 2] + self.pick_lift) | (
            (pos[:, 1] < c.shelf_front_y - 0.05) & (pos[:, 1] > c.shelf_front_y - c.apron_depth)
            & ((pos[:, 0] - c.box_init[0]).abs() < c.apron_half_x) & (pos[:, 2] > c.carry_min_z))
        held = clear & (speed < self.pick_speed)
        step_dt = self.env.dt * self.env.robot.control_period
        self._held_s = torch.where(held, self._held_s + step_dt, torch.zeros_like(self._held_s))
        return (self._held_s >= self.pick_hold_s).float()

    def carried(self):
        pos, _ = self.scene.box_pose()
        left = (pos[:, :2] - self._bin_xy).norm(dim=-1)
        return (1.0 - left / self._span).clamp(0.0, 1.0)

    def in_bin(self):
        return self.scene.placed().float()

    def upright(self):
        """1 where the robot's base is still at standing height, and 1 everywhere if this
        embodiment has no base that can fall (no articulation, or a pelvis welded to the world)."""
        import torch

        if self._base_idx is None:
            return torch.ones(self.num_envs)
        art = self.env.robot.articulation
        z = art.data.body_pos_w[:, self._base_idx, 2] - self.env.iscene.env_origins[:, 2]
        return (z > self.upright_min_z).float()
