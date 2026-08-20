"""Bulb-scene eval sims."""

from sim import SimSpec, register_sim

# The HF bulb dataset's law (CoSiGen/bulb_franka_osc_jointpos_60hz), as a named,
# dataset-independent setup: joint-PD tracking at the native 60 Hz latch rate.
register_sim("bulb_jointpd_60hz", lambda: SimSpec(
    preset="assembly.bulb.franka.osc",
    control_space="joint_pos",
    control_freq_hz=60.0,
    finger_drives=(8000.0, 100.0),
))
