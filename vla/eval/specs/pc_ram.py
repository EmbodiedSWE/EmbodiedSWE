"""pc_ram-scene eval sims."""

from sim import SimSpec, register_sim

# The IK data-generation solve's law (experiments/2026-08-24_pc_ram_franka_ik): the
# `assembly.pc_ram.franka.joint` preset — arm joint-position targets under the stock PD
# (kp 400 / kd 80), latched every 5 physics substeps at dt 1/240 = 48 Hz — with the stock
# gripper drives (the solve writes no controller overrides). Joint-PD tracking of the
# bake's joint_pos labels at the native latch rate.
register_sim("pc_ram_jointpd_48hz", lambda: SimSpec(
    preset="assembly.pc_ram.franka.joint",
    control_space="joint_pos",
    control_freq_hz=48.0,
    finger_drives=(2000.0, 100.0),
))

# The same labels under a STIFF joint tracker. `joint_pos` labels are ACHIEVED positions —
# they already carry the recording arm's gravity sag — so re-tracking them through the stock
# kp 400 sags a second time (~2-4 mm at the hand, wider than the 1.2 mm DIMM funnel: the
# nominal_0 certification missed the first insertion). A real Panda's joint-position
# controller is far stiffer than the sim preset; this spec models that executor.
register_sim("pc_ram_jointpd_48hz_stiff", lambda: SimSpec(
    preset="assembly.pc_ram.franka.joint",
    control_space="joint_pos",
    control_freq_hz=48.0,
    finger_drives=(2000.0, 100.0),
    tracker_gains=(2000.0, 180.0),  # kd = 4*sqrt(kp), the preset's own ratio
))

# The joint_target convention through the STOCK kp-400 joint PD (no tracker override): the labels
# ARE the commanded targets our solve issued, so feeding them back to the same PD reproduces the
# demo (the matched-controller condition) — desired-target data needs no stiff tracker, unlike
# joint_pos (whose achieved-state labels re-sag; see pc_ram_jointpd_48hz_stiff).
register_sim("pc_ram_jointtarget_48hz", lambda: SimSpec(
    preset="assembly.pc_ram.franka.joint",
    control_space="joint_target",
    control_freq_hz=48.0,
    finger_drives=(2000.0, 100.0),
))

# Native 20 Hz twin of pc_ram_jointtarget_48hz: the `joint_20hz` preset (control_dt 0.05 ->
# latched every 12 substeps at dt 1/240). Certifies natively-recorded 20 Hz joint_target
# streams; also the executor spec for evaluating 20 Hz-trained policies.
register_sim("pc_ram_jointtarget_20hz", lambda: SimSpec(
    preset="assembly.pc_ram.franka.joint_20hz",
    control_space="joint_target",
    control_freq_hz=20.0,
    finger_drives=(2000.0, 100.0),
))

# Wide-funnel twin of pc_ram_jointtarget_48hz: the Stage-4 grounding data's own case variant
# (2.5 mm/side mouth), for evaluating policies trained on it in their training funnel condition.
register_sim("pc_ram_wide_jointtarget_48hz", lambda: SimSpec(
    preset="assembly.pc_ram_wide.franka.joint",
    control_space="joint_target",
    control_freq_hz=48.0,
    finger_drives=(2000.0, 100.0),
))
