"""Grader for the bulb scene: every bulb seated.

Rubric stages (weights and modes live in RUBRIC):
    picked    fraction of bulbs ever held aloft — lifted AND quasi-static,
              sustained, so a flung/falling bulb never counts; a milestone,
              credit kept after the bulb is placed
    engaged   fraction of bulbs on/in a socket bore
    threaded  mean thread depth from free-rest to seat (the scene's own
              glow band); unengaged bulbs count 0
Every stage is per env (fractions over that env's bulbs). Success is the
scene's `seated()` per env, read when the delivery finishes.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.bulb_assembly import BulbAssemblyScene


class BulbAssemblyGrader(BaseGrader):
    """Every bulb seated — threaded to depth, on the socket axis."""

    SCENE = BulbAssemblyScene
    RUBRIC = (("picked", 0.2, "once"), ("engaged", 0.2), ("threaded", 0.6))
    # "picked" = lifted AND quasi-static, sustained — a flung bulb can pass the
    # height gate, but in free flight speed changes by ~5 m/s per half-second,
    # so it can never stay below pick_speed for pick_hold_s continuously.
    pick_lift = 0.04  # m above the bulb's start height
    pick_speed = 0.3  # m/s max speed while held
    pick_hold_s = 0.25  # s both must hold, consecutively
    scene: BulbAssemblyScene

    def setup(self) -> None:
        import torch

        self._z0 = torch.stack(  # start heights, captured at the graded reset
            [b.data.root_pos_w[:, 2] for b in self.scene.bulbs], dim=1)
        self._held_s = torch.zeros_like(self._z0)  # per-bulb consecutive lifted-and-slow time

    def check_success(self):
        return self.scene.seated().all(dim=1)  # (num_envs,) — every bulb of that env

    def _in_socket(self):
        """(engaged mask, thread completion), each (num_envs, num_bulbs)."""
        import torch

        c = self.scene.cfg
        off = self.scene._bulb_offsets_in_socket()  # (n, N_bulb, B_socket, 3) — privileged read
        near_dist, near = off[..., :2].norm(dim=-1).min(dim=-1)
        depth = torch.gather(off[..., 2], 2, near.unsqueeze(-1)).squeeze(-1)
        engaged = (near_dist <= c.align_xy) & (depth <= c.socket_opening_z)
        thread = ((c.light_start_z - depth) / (c.light_start_z - c.seat_z)).clamp(0.0, 1.0)
        return engaged, thread

    # ---- rubric stages — each returns a (num_envs,) fraction over that
    # env's bulbs ------------------------------------------------------------
    def picked(self):
        import torch

        z = torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.bulbs], dim=1)
        speed = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.scene.bulbs], dim=1)
        held = (z > self._z0 + self.pick_lift) & (speed < self.pick_speed)
        step_dt = self.env.dt * self.env.robot.control_period
        self._held_s = torch.where(held, self._held_s + step_dt, torch.zeros_like(self._held_s))
        return (self._held_s >= self.pick_hold_s).float().mean(dim=1)

    def engaged(self):
        engaged, _ = self._in_socket()
        return engaged.float().mean(dim=1)

    def threaded(self):
        engaged, thread = self._in_socket()
        return (engaged * thread).mean(dim=1)
