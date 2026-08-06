"""Robot embodiments for sim_gen tasks (arm types, as in CoSiGen; humanoids skipped)."""

from .franka import FrankaArm, attach_franka, bind_franka

__all__ = ["FrankaArm", "attach_franka", "bind_franka"]
