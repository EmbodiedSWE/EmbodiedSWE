"""Grader for the wheel pick-and-place scene: the steering wheel resting in the basket, hands off.

Rubric stages (weights and modes live in RUBRIC):
    picked     the wheel was carried — lifted clear of the table AND quasi-static,
               sustained, so a wheel knocked into the air never counts; a milestone,
               credit kept after it is put down
    carried    fraction of the horizontal distance from the wheel's start pose to the
               basket's interior centre that has been closed; a milestone, so a wheel
               that arrives and is released keeps the credit
    in_basket  the scene's own `placed()` — inside the target box and come to rest
    retracted  the non-working arm is pulled back out of the way (see below)
Success is the scene's `placed()` AND the robot's right wrist retracted past
x < 0.26 in env-local coords — the half of the criterion the embodiment-agnostic
scene cannot express, since it is about the robot rather than the wheel. With a robot
that exposes no wrist (e.g. NullRobot) that term is vacuously satisfied and only
`placed()` decides.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.wheel_pick_place import WheelPickPlaceScene


class WheelPickPlaceGrader(BaseGrader):
    """The steering wheel placed in the basket and released, the working arm retracted."""

    SCENE = WheelPickPlaceScene
    RUBRIC = (("picked", 0.2, "once"), ("carried", 0.2, "once"), ("in_basket", 0.5), ("retracted", 0.1))
    # "picked" = lifted AND quasi-static, sustained. A wheel batted into the air can clear the
    # height gate, but in free flight its speed changes by ~2.5 m/s over pick_hold_s, so it can
    # never stay under pick_speed for that long continuously.
    pick_lift = 0.05  # m above the wheel's resting height on the table
    pick_speed = 0.4  # m/s max speed while counted as held
    pick_hold_s = 0.25  # s both must hold, consecutively
    right_wrist_max_x = 0.26  # env-local x the non-working wrist must be pulled back past
    scene: WheelPickPlaceScene

    def setup(self) -> None:
        import torch

        c = self.scene.cfg
        pos, _ = self.scene.wheel_pose()
        self._p0 = pos.clone()  # start pose, captured at the graded reset
        self._held_s = torch.zeros(self.num_envs, device=pos.device)
        wx, wy = c.workbench_pos
        self._basket_xy = torch.tensor(
            [wx + sum(c.BASKET_INNER_X) / 2, wy + sum(c.BASKET_INNER_Y) / 2], device=pos.device)
        self._span = (self._p0[:, :2] - self._basket_xy).norm(dim=-1).clamp(min=1e-6)
        # The right-wrist term needs an embodiment. Resolve the body index once; leave it None for a
        # robot with no wrist (NullRobot, a one-armed manipulator), which makes the term vacuous.
        robot = self.env.robot
        art = getattr(robot, "articulation", None)
        ee = getattr(robot, "EE_BODIES", ())
        self._wrist_idx = None
        if art is not None and len(ee) > 1 and ee[1] in art.data.body_names:
            self._wrist_idx = art.data.body_names.index(ee[1])

    def check_success(self):
        return self.scene.placed() & (self.retracted() > 0.5)

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------------
    def picked(self):
        import torch

        pos, _ = self.scene.wheel_pose()
        speed = self.scene.wheel.data.root_lin_vel_w.norm(dim=-1)
        held = (pos[:, 2] > self._p0[:, 2] + self.pick_lift) & (speed < self.pick_speed)
        step_dt = self.env.dt * self.env.robot.control_period
        self._held_s = torch.where(held, self._held_s + step_dt, torch.zeros_like(self._held_s))
        return (self._held_s >= self.pick_hold_s).float()

    def carried(self):
        pos, _ = self.scene.wheel_pose()
        left = (pos[:, :2] - self._basket_xy).norm(dim=-1)
        return (1.0 - left / self._span).clamp(0.0, 1.0)

    def in_basket(self):
        return self.scene.placed().float()

    def retracted(self):
        """1 where the robot's right wrist is pulled back past `right_wrist_max_x` (env-local x), and
        1 everywhere if this embodiment has no such wrist."""
        import torch

        if self._wrist_idx is None:
            return torch.ones(self.num_envs)
        art = self.env.robot.articulation
        x = art.data.body_pos_w[:, self._wrist_idx, 0] - self.env.iscene.env_origins[:, 0]
        return (x < self.right_wrist_max_x).float()
