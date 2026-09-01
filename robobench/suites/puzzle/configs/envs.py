"""Canonical runnable env configs for the puzzle suite — registered in `ENVS` by name.

Four scenes, scene-physics-only first (NullRobot smoke/oracle), embodiments after:
  - push_t:     simple G1-native planar pushing alignment task (port).
  - push_shapes: multi-stage G1-native shape sorting — three blocks (T/X/L) each pushed
                and reoriented onto its own matching pad.
  - coffee:     capsule coffee machine state machine (brew one capsule coffee and
                serve the filled mug on the tray).
  - syringe:    draw / triple-dose / re-park. Defining embodiment GR1-T2 (bimanual:
                one hand fixtures the barrel, the other meters the plunger); G1 for
                curriculum; franka / piper / wxai / bimanual_franka as arm bindings.
  - spatula:    non-prehensile tool payload control (port).
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import (
    BimanualFrankaCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    PiperRobotCfg,
    WxaiRobotCfg,
)
from robobench.suites.puzzle.scenes import (
    CoffeeServiceSceneCfg,
    PushShapesSceneCfg,
    PushTSceneCfg,
    SpatulaFlipServeSceneCfg,
    SyringeDosingSceneCfg,
)

SUITE = "puzzle"

_FRANKA_ROT = (0.7071068, 0.0, 0.0, 0.7071068)


# ================================ push_t =========================================
# Simple G1-native planar push: source visual assets and strict 7 mm / 7 degree
# alignment rubric. NullRobot exists for the recorded oracle; the two G1 modes are
# the only embodiment bindings because this contribution fills the humanoid lane.
register_env(SUITE, lambda: EnvCfg(scene="push_t", robot="null", env_spacing=3))

for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="push_t",
                scene_cfg=PushTSceneCfg(),
                robot="g1",
                control_mode=mode,
                # The packing-bench front face is near y=-0.45. Keep 15 cm of
                # clearance so G1's shins never initialize inside the chassis.
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.60, 0.75)),
                env_spacing=3,
            )
        ),
    )


# ============================== push_shapes ======================================
# Multi-stage successor to push_t: three shapes (T / X / L), three colour-keyed pads,
# each block yawed off its pad so reorientation is a goal rather than a disturbance.
# Same substrate as push_t — same bench, same 15 cm chassis clearance, same G1 modes.
register_env(SUITE, lambda: EnvCfg(scene="push_shapes", robot="null", env_spacing=3))

for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="push_shapes",
                scene_cfg=PushShapesSceneCfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.60, 0.75)),
                env_spacing=3,
            )
        ),
    )


# ================================ syringe ========================================
# Scene physics only (NullRobot). -> "puzzle.syringe"
register_env(SUITE, lambda: EnvCfg(scene="syringe", robot="null", env_spacing=3))


def _syringe_humanoid_cfg() -> SyringeDosingSceneCfg:
    """Cart-shelf layout pulled toward a fixed-base humanoid at -y (G1 reach ~0.55 m).
    NOT re-verified since the medical-cart rework (2026-08-09): the lying syringe,
    rack and beaker replaced the stand/plate layout this binding was tuned on."""
    return SyringeDosingSceneCfg(home_pos=(0.12, -0.10), rack_pos=(-0.12, 0.12),
                                 res_pos=(0.10, 0.10))


# -> "puzzle.syringe.{gr1t2,g1}.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="syringe",
                scene_cfg=_syringe_humanoid_cfg(),
                robot="gr1t2",
                control_mode=mode,
                # base y -0.50: the bench face is at -0.45; any closer spawns the
                # shins inside the bench and the contact solver kicks the robot over.
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.50, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="syringe",
                scene_cfg=_syringe_humanoid_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.52, 0.75)),
                env_spacing=3,
            )
        ),
    )

# -> "puzzle.syringe.franka.{osc,joint}" — ground-level layout in front of the base.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="syringe",
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.42, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

# -> "puzzle.syringe.{piper,wxai}.{osc,joint}" — the short arms (~0.6 m / ~0.5 m
# reach) are pedestal-mounted so the draw-top pose (hand z ~0.55 at the reservoir,
# 0.48 m out) stays inside their envelopes; bases centre both the stand and draw-top.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="syringe",
                robot="piper",
                control_mode=mode,
                robot_cfg=PiperRobotCfg(base_pos=(0.08, -0.26, 0.24), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="syringe",
                robot="wxai",
                control_mode=mode,
                robot_cfg=WxaiRobotCfg(base_pos=(0.0, -0.28, 0.35), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

# -> "puzzle.syringe.bimanual_franka.{osc,joint}" — both arms mount on the SIDE
# TABLE that runs along the cart's NORTH long side (top 0.71 m, 4 cm below the
# shelf), spread 0.56 m apart on its cart-side edge and facing south over the north
# guard rail: left arm west (rack/syringe side), right arm east.
# NOT yet reach-verified on the cart layout (2026-08-09).
_FACE_SOUTH = (0.7071068, 0.0, 0.0, -0.7071068)  # yaw -90: franka +x -> world -y
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="syringe",
                robot="bimanual_franka",
                control_mode=mode,
                robot_cfg=BimanualFrankaCfg(robots={
                    "left": ("franka", FrankaRobotCfg(base_pos=(-0.28, 0.42, 0.71),
                                                      base_rot=_FACE_SOUTH)),
                    "right": ("franka", FrankaRobotCfg(base_pos=(0.28, 0.42, 0.71),
                                                       base_rot=_FACE_SOUTH)),
                }),
                env_spacing=3,
            )
        ),
    )


# ================================ spatula ========================================
# port: spatula flip & serve (non-prehensile tool payload control). The defining
# embodiment is GR1-T2 (dexterous handle grip, matching the source's dex-hand lineage);
# G1 binds for curriculum; franka trivially holds the handle (the brief's ablation).
# No `multi` binding: one hand works the spatula, the other has nothing load-bearing
# to do. -> "puzzle.spatula"
register_env(SUITE, lambda: EnvCfg(scene="spatula", robot="null", env_spacing=3))


# Placements are STARTING guesses copied from the pen-holder measured
# reach values for the same embodiments at the same bench — re-verify with the
# per-binding stress smoke before any agent run (only the null smoke validates the
# scene). Board to the robot's left, plate to the right, spatula handle-first at the
# bench front.
def _spatula_gr1t2_cfg() -> SpatulaFlipServeSceneCfg:
    """GR1-T2 (longer arms, comfortable band farther out)."""
    return SpatulaFlipServeSceneCfg(
        surface_z=0.7,
        pan_pos=(-0.18, 0.16),
        plate_pos=(0.18, 0.18),
        spatula_pos=(0.02, -0.02),
    )


def _spatula_g1_cfg() -> SpatulaFlipServeSceneCfg:
    """G1 (short ~0.55 m arms): everything pulled toward the bench front."""
    return SpatulaFlipServeSceneCfg(
        surface_z=0.7,
        pan_pos=(-0.15, 0.10),
        plate_pos=(0.15, 0.12),
        spatula_pos=(0.02, -0.04),
    )


def _spatula_franka_cfg() -> SpatulaFlipServeSceneCfg:
    """Franka: ground-level work; the molded handle pinches under the 8 cm jaw.

    VERIFIED layout (the flip_serve reference solve, 2026-08-12): the spatula rest
    pulled to (0.02, -0.14) — at the earlier (0.02, -0.08) guess the grip band landed
    a 0.40 m side-reach where the OSC parked 11 mm off target, too coarse for the
    weld's centred-pinch window. Bread-size sampling off: the solution's verified
    claim is at the fixed demo size."""
    return SpatulaFlipServeSceneCfg(
        surface_z=0.0,  # table-mounted arm at ground level (the packing preset's
        # native 0.994 is for standalone/humanoid layouts)
        pan_pos=(-0.14, 0.08),
        plate_pos=(0.14, 0.10),
        spatula_pos=(0.02, -0.14),
        sample_size=False,
    )


# -> "puzzle.spatula.{gr1t2,g1}.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="spatula",
                scene_cfg=_spatula_gr1t2_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.50, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="spatula",
                scene_cfg=_spatula_g1_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.48, 0.75)),
                env_spacing=3,
            )
        ),
    )

# -> "puzzle.spatula.franka.{osc,joint}" — VERIFIED binding (the flip_serve reference
# solve, 2026-08-12): mounted WEST facing +x, every target 0.3-0.75 m straight ahead,
# no base yaw (the earlier side-mount guess at (0,-0.40) yaw 90 put the grasp in a
# side-reach where the OSC parked 11-21 mm off target). Gripper effort 25 N: a 120 N
# pinch punts the tool when the close lands imperfectly; nullspace () — the default
# posture winds the arm (the pen_holder lesson).
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="spatula",
                scene_cfg=_spatula_franka_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(
                    base_pos=(-0.60, 0.05, 0.0),
                    nullspace_dof_pos=(),
                    gripper_effort_limit=25.0,
                    gripper_stiffness=2000.0,
                ),
                env_spacing=3,
            )
        ),
    )


# ================================ coffee =========================================
# ---- capsule coffee service (appliance state machine; the microwave task's
# Franka-native successor, user directive 2026-08-09) ----
# Scene physics only (NullRobot oracle/smoke). -> "puzzle.coffee"
register_env(SUITE, lambda: EnvCfg(scene="coffee", robot="null", env_spacing=3))


# Robot bindings. The machine has NO swinging door — every embodiment stands
# FRONT-CENTER (the source-faithful stance the microwave's Franka ablation could
# never take; its east-side station put the task's west half at joint 1's stop and
# produced the 2026-08-05/06 contortion campaign). Humanoid placements are STARTING
# guesses; the Franka placement is reach-checked below.
def _coffee_g1_cfg() -> CoffeeServiceSceneCfg:
    """G1 (short ~0.55 m arms): bench work pulled close."""
    return CoffeeServiceSceneCfg(
        table="packing",
        surface_z=0.7,
        cm_pos=(0.0, 0.06),
        tray_pos=(0.30, -0.14),
        cup_slot=(0.24, -0.10),
        pod_slot=(0.36, -0.18),
    )


def _coffee_gr1t2_cfg() -> CoffeeServiceSceneCfg:
    """GR1-T2: same bench, wider layout."""
    return CoffeeServiceSceneCfg(
        table="packing",
        surface_z=0.7,
        cm_pos=(0.0, 0.08),
        tray_pos=(0.34, -0.12),
        cup_slot=(0.27, -0.08),
        pod_slot=(0.41, -0.16),
    )


def _coffee_franka_cfg() -> CoffeeServiceSceneCfg:
    """Franka (primary): work on the packing table at 0.55 m, base FRONT-CENTER.

    Base pushed to y -0.55 (was -0.42) after the 2026-08-09 cup-staging failure:
    the staged-cup pinch sat only 0.225 m from the base column — inside the
    cramped inner shell for a straight-down grip (wrist stack needs ~0.25 m above
    the pinch) — and the arm folded up with 233 deg of joint debt. The machine
    then moved north (cm_pos y 0.08 -> 0.16 -> settled at 0.12, user-approved
    2026-08-09): the LOW cup-placement descent stalled at every pinch distance
    0.26-0.29 m while converging at 0.37 m — precise straight-down work needs
    >= ~0.32 m — but at 0.16 the START key sat 0.77 m out and the pitched press
    became history-dependent (pressed from job_0012's clean rehome, stalled 9 cm
    short on the chain run). 0.12 keeps the placement pinch at 0.385 m (inside
    the proven zone) and pulls the key to 0.73 m. Reach check from base
    (0.0, -0.55): cup placement pinch 0.38 m, spout axis 0.48 m, cover ridge
    0.43 m closed / 0.35 m open, pod pocket 0.57 m, cup slot 0.55 m, pod slot
    0.57 m, START key 0.73 m — the key press rides a pitched hand (2.5 mm
    prismatic travel). Table deepened 1.14 -> 1.44 m (depth scale 1.9): machine
    rear y 0.44 (edge 0.72), tray dish x to 0.585 (edge 1.235), base south edge
    -0.64 (edge -0.72)."""
    return CoffeeServiceSceneCfg(
        surface_z=0.55,
        cm_pos=(0.0, 0.12),
        tray_pos=(0.38, -0.14),
        cup_slot=(0.31, -0.10),
        pod_slot=(0.44, -0.18),
        table_depth_scale=1.9,
        # the weld-on-closure grip contract (panda-hand pools; see the scene cfg).
        # Engage radius covers scan lips/flares (the microwave port's run-27 lesson).
        grasp_weld=True,
        grasp_weld_dist=0.022,
    )


# -> "puzzle.coffee.g1.{joint,pink_ik}" / ".gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="coffee",
                scene_cfg=_coffee_g1_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75)),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="coffee",
                scene_cfg=_coffee_gr1t2_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.48, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

# -> "puzzle.coffee.franka.{osc,joint}"
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="coffee",
                scene_cfg=_coffee_franka_cfg(),
                robot="franka",
                control_mode=mode,
                # FRONT-CENTER: no door arc exists to dodge (the machine's slider
                # moves toward the spout, over its own footprint). Base fully on
                # the deepened table with 8 cm south margin.
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.55, 0.55), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )
