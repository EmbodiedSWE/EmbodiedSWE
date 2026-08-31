"""Grader for the wheel carry scene: the wheel picked up here, walked over there, and left in the basket.

Rubric stages (weights and modes live in RUBRIC):
    picked     the wheel was really carried — lifted clear of the pick table AND quasi-static,
               sustained, so a wheel knocked into the air never counts; a milestone, credit kept
               after it is put down
    carried    fraction of the horizontal distance from the wheel's start pose to the basket's
               interior centre that has been closed; a milestone, so a wheel that arrives and is
               released keeps the credit. This is the term the transit lives in — on this scene it
               spans metres, not centimetres
    in_basket  the scene's own `placed()` — inside the target box and come to rest
    upright    the robot is still standing (see below)

Success is the scene's own `success()` — the journey latches AND `placed()` — AND `upright`. The
upright term is the half of the criterion the embodiment-agnostic scene cannot express, and on this
task it is not a formality: a robot that topples onto the far table could deposit the wheel in the
basket on the way down, and that is not a solved pick-carry-place. With a robot that has no
measurable base (e.g. NullRobot) the term is vacuously satisfied and only the scene decides, which
is what lets the scene smoke stage the task by teleporting the wheel.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.locomanip.scenes.wheel_carry import WheelCarryScene


class WheelCarryGrader(BaseGrader):
    """The steering wheel carried across to the far basket and released, the robot still on its feet."""

    SCENE = WheelCarryScene
    # `upright` is live and true from the first step, so its weight is the score a run earns by
    # merely not falling over — kept at 0.10, matching what the sibling wheel_pick_place grader
    # hands its vacuous-by-default `retracted` term, so "did nothing" scores the same on both.
    RUBRIC = (("picked", 0.20, "once"), ("carried", 0.25, "once"), ("in_basket", 0.45), ("upright", 0.10))
    # "picked" = lifted AND quasi-static, sustained. A wheel batted into the air can clear the height
    # gate, but in free flight its speed changes by ~2.5 m/s over pick_hold_s, so it can never stay
    # under pick_speed for that long continuously.
    pick_lift = 0.05  # m above the wheel's resting height on the table
    pick_speed = 0.4  # m/s max speed while counted as held
    pick_hold_s = 0.25  # s both must hold, consecutively
    # Upright: the base link's height above the floor. A G1 walks with its pelvis at ~0.72 m and the
    # locomotion policy holds that to within a centimetre or two even in transit, so this threshold
    # is far below any healthy gait and only trips on an actual fall.
    upright_min_z = 0.45
    scene: WheelCarryScene

    def setup(self) -> None:
        import torch

        c = self.scene.cfg
        pos, _ = self.scene.wheel_pose()
        self._p0 = pos.clone()  # start pose, captured at the graded reset
        self._held_s = torch.zeros(self.num_envs, device=pos.device)
        qx, qy = c.place_pos
        self._basket_xy = torch.tensor(
            [qx + sum(c.BASKET_INNER_X) / 2, qy + sum(c.BASKET_INNER_Y) / 2], device=pos.device)
        self._span = (self._p0[:, :2] - self._basket_xy).norm(dim=-1).clamp(min=1e-6)
        # The upright term needs an embodiment with a floating base. Resolve the base body index
        # once; leave it None for a robot with no articulation (NullRobot) or one whose root is
        # welded to the world, in which case "still standing" is not a question and the term is
        # vacuous. `BASE_BODY` lets an embodiment name its own root; the humanoids here use "pelvis".
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

    def upright(self):
        """1 where the robot's base is still at standing height, and 1 everywhere if this embodiment
        has no base that can fall (no articulation, or a pelvis welded to the world)."""
        import torch

        if self._base_idx is None:
            return torch.ones(self.num_envs)
        art = self.env.robot.articulation
        z = art.data.body_pos_w[:, self._base_idx, 2] - self.env.iscene.env_origins[:, 2]
        return (z > self.upright_min_z).float()
