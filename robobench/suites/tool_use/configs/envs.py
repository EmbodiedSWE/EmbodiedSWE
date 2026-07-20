"""Canonical runnable env configs for the tool_use suite — registered in `ENVS` by name.

Three scenes, scene-physics-only first (NullRobot smoke/oracle), embodiments after:
  - syringe:    draw / triple-dose / re-park. Defining embodiment GR1-T2 (bimanual:
                one hand fixtures the barrel, the other meters the plunger); G1 for
                curriculum; franka / piper / wxai / bimanual_franka as arm bindings.
  - whiteboard: marker + eraser work on the ink grid. Defining embodiment GR1-T2
                (pinch grasp on the chunky marker); per-embodiment board/tray layouts.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.core.registries import ROBOTS
from robobench.robots import (
    BimanualFrankaCfg,
    FrankaRobot,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    PiperRobotCfg,
    WxaiRobotCfg,
)
from robobench.suites.tool_use.scenes import SyringeDosingSceneCfg, WhiteboardWordSceneCfg

SUITE = "tool_use"

_FRANKA_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

# "franka_ped": the same FrankaRobot class under a second robot name, for
# pedestal-mounted bindings whose env names must not collide with ground bindings.
try:
    ROBOTS.register("franka_ped", FrankaRobot)
except KeyError:
    pass  # already registered (module re-import)


# ================================ syringe ========================================
# Scene physics only (NullRobot). -> "tool_use.syringe"
register_env(SUITE, lambda: EnvCfg(scene="syringe", robot="null", env_spacing=3))


def _syringe_humanoid_cfg() -> SyringeDosingSceneCfg:
    """Bench-height plate + stand in front of a fixed-base humanoid (robot at -y).
    Layout kept within 0.5 m of the base (G1 reach ~0.55 m)."""
    return SyringeDosingSceneCfg(surface_z=0.7, stand_pos=(0.12, -0.04),
                                 plate_pos=(-0.10, 0.16))


# -> "tool_use.syringe.{gr1t2,g1}.{joint,pink_ik}"
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

# -> "tool_use.syringe.franka.{osc,joint}" — ground-level layout in front of the base.
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

# -> "tool_use.syringe.{piper,wxai}.{osc,joint}" — the short arms (~0.6 m / ~0.5 m
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

# -> "tool_use.syringe.bimanual_franka.{osc,joint}" — LEFT arm at the proven solo-franka
# base (single-arm solutions run on it verbatim as arm 0); RIGHT arm parked far east as
# an idle helper the policy may recruit.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="syringe",
                robot="bimanual_franka",
                control_mode=mode,
                robot_cfg=BimanualFrankaCfg(robots={
                    "left": ("franka", FrankaRobotCfg(base_pos=(0.0, -0.42, 0.0),
                                                      base_rot=_FRANKA_ROT)),
                    "right": ("franka", FrankaRobotCfg(base_pos=(0.9, -0.42, 0.0),
                                                       base_rot=_FRANKA_ROT)),
                }),
                env_spacing=3,
            )
        ),
    )


# =============================== whiteboard ======================================
# Scene physics only (NullRobot). -> "tool_use.whiteboard"
register_env(SUITE, lambda: EnvCfg(scene="whiteboard", robot="null", env_spacing=3))


def _whiteboard_g1_cfg() -> WhiteboardWordSceneCfg:
    """Bench-height board for the G1 (robot at -y facing the board). board_y 0.10 keeps
    the writing plane a stroke-length out; tray_y 0.22 pulls the marker/eraser tray (the
    PICKUP point) to the bench front — at the default tray the marker sat a measured
    12 cm past the G1 fingertip."""
    return WhiteboardWordSceneCfg(surface_z=0.7, board_y=0.10, tray_y=0.22)


# -> "tool_use.whiteboard.{gr1t2,g1}.{joint,pink_ik}" — per-embodiment layouts: the
# GR1-T2's arms are ~0.2 m longer and its comfortable band sits farther out, so it gets
# a farther board (0.26) with a nearer tray (0.14) than the G1 layout.
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="whiteboard",
                scene_cfg=WhiteboardWordSceneCfg(surface_z=0.7, board_y=0.26,
                                                 tray_y=0.14),
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
                scene="whiteboard",
                scene_cfg=_whiteboard_g1_cfg(),
                robot="g1",
                control_mode=mode,
                # the legless G1 torso pole sits 4 cm closer than the GR1-T2 without
                # touching the bench; at -0.52 the tray was 13.7 cm past its fingertip.
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.48, 0.75)),
                env_spacing=3,
            )
        ),
    )

# -> "tool_use.whiteboard.{piper,wxai}.{osc,joint,impedance}" — raised-pedestal mounts
# put both the tray marker (z 0.09) and the writing area (z 0.16-0.38) in the small
# arms' sweet zone.
for _mode in ("osc", "joint", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="whiteboard",
                robot="piper",
                control_mode=mode,
                robot_cfg=PiperRobotCfg(base_pos=(0.0, -0.20, 0.30), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="whiteboard",
                robot="wxai",
                control_mode=mode,
                robot_cfg=WxaiRobotCfg(base_pos=(0.0, -0.22, 0.30), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

# -> "tool_use.whiteboard.franka_ped.{osc,joint}" — the ONLY franka whiteboard binding:
# a ground-mounted franka measurably cannot work this scene (the tray line sits in the
# board's shadow behind a joint-limit wall, and a dropped marker rolls against the
# robot's own base column), so the arm is pedestal-mounted at the measured-envelope
# placement: marker at r~0.35 (grasp) and the letter span at r<=0.47 for the
# horizontal-hand write fold.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="whiteboard",
                robot="franka_ped",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(-0.12, -0.16, 0.20), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )
