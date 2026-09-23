"""Bulb-scene eval sims."""

from sim import SimSpec, register_sim

# The HF bulb dataset's law (EmbodiedSWE/bulb_franka_osc_jointpos_60hz), as a named,
# dataset-independent setup: joint-PD tracking at the native 60 Hz latch rate.
register_sim("bulb_jointpd_60hz", lambda: SimSpec(
    preset="assembly.bulb.franka.osc",
    control_space="joint_pos",
    control_freq_hz=60.0,
    finger_drives=(8000.0, 100.0),
))

# The OSC controller that generated the raw data in the HF bulb dataset
# (solve-time settings; everything unlisted = the preset default).
register_sim("bulb_osc_60hz", lambda: SimSpec(
    preset="assembly.bulb.franka.osc",
    control_space="raw_cmd",
    control_freq_hz=60.0,
    stamp={"controller": {
        "leaves": [
            {"class": "OperationalSpaceController", "control_period": 4,
             "cfg": {"rot_scale": 0.15},
             "kp": [150.0, 150.0, 150.0, 600.0, 600.0, 600.0],
             "kd": [24.494897427831781] * 3 + [48.989794855663558] * 3},
            {"class": "JointController", "control_period": 4, "cfg": {}},
        ],
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)]
                       + ["panda_finger_joint1", "panda_finger_joint2"],
        "joint_stiffness": [0.0] * 7 + [8000.0, 8000.0],
        "joint_damping": [0.0] * 7 + [100.0, 100.0],
    }},
))
