"""Canonical runnable env configs for the articulated suite — registered in `ENVS` by name.

Scene-physics-only first (NullRobot smoke/oracle), embodiments after the physics is
proven. Robot bindings raise the safe onto a bench and face its door (-y) toward the
robot, dial at comfortable fingertip height.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import (
    BimanualFrankaCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    MultiRobotCfg,
    PiperRobotCfg,
    WxaiRobotCfg,
)
from robobench.suites.articulated.scenes import (
    BalanceScaleSceneCfg,
    CombinationSafeSceneCfg,
    MicrowaveMealSceneCfg,
    PouringSceneCfg,
)

SUITE = "articulated"

# Scene physics only (NullRobot). -> "articulated.safe"
register_env(SUITE, lambda: EnvCfg(scene="safe", robot="null", env_spacing=3))


def _robot_scene_cfg() -> CombinationSafeSceneCfg:
    """Bench-height safe for fixed-base humanoids: dial ends up ~1.0 m high, door front
    (-y side) toward the robot."""
    return CombinationSafeSceneCfg(surface_z=0.7, safe_pos=(0.0, 0.22))


# Fixed-base G1. -> "articulated.safe.g1.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                scene_cfg=_robot_scene_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.55, 0.75)),
                env_spacing=3,
            )
        ),
    )

# GR1-T2 (fine dial work is its defining variant). -> "articulated.safe.gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                scene_cfg=_robot_scene_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.50, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )


# ---- balance scale (weigh-and-sort; the beam is the only instrument) ----
register_env(SUITE, lambda: EnvCfg(scene="scale", robot="null", env_spacing=3))


def _scale_robot_scene_cfg() -> BalanceScaleSceneCfg:
    """Bench-height scale for fixed-base humanoids (robot at -y; boxes row nearest)."""
    # bench shrunk to the crate-verified 0.9 depth (the default 1.1-deep bench put
    # its front edge at y=-0.55, INSIDE the humanoid bases at -0.50 -> boot chaos);
    # shelf pulled to 0.40 so it stays on the bench top.
    return BalanceScaleSceneCfg(surface_z=0.7, scale_pos=(0.0, 0.14),
                                shelf_pos=(0.0, 0.40), box_row_y=-0.16,
                                bench_size=(1.2, 0.9))


for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_robot_scene_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.50, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_robot_scene_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75)),
                env_spacing=3,
            )
        ),
    )


# ---- Franka bindings ----
# Table-mounted arm at ground level (scene surface_z=0, the null layout), base BEHIND
# the work facing +y (base_rot = +90 deg about z; the Franka's home faces +x).
_FRANKA_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

for _mode in ("osc", "joint", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.45, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )


def _scale_franka_scene_cfg() -> BalanceScaleSceneCfg:
    """Compact, gripper-sized layout: 6 cm boxes (the parallel gripper opens ~8 cm;
    the humanoid 9 cm boxes cannot be grasped), everything inside the ~0.75 m reach."""
    return BalanceScaleSceneCfg(scale_pos=(0.0, 0.10), shelf_pos=(0.0, 0.34),
                                box_row_y=-0.18, box_size=(0.06, 0.06, 0.05),
                                box_gap=0.11)


for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_franka_scene_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.45, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )


# ---- Small-arm + bimanual scale bindings ----
# PiPER (0.6 m reach) gets an even more compact, embodiment-fitted layout (the layout
# lives in the scene cfg; physics/rules untouched). Bimanual Franka reuses the franka
# layout with side-by-side bases so each arm owns its half of the bench (kills the
# far-slot reach margin problem).
def _scale_piper_scene_cfg() -> BalanceScaleSceneCfg:
    """PiPER-sized layout: 5 cm boxes, row pulled to y=-0.08, shelf at 0.28 with
    narrower slots — every target inside the ~0.6 m reach from base (0, -0.22)."""
    return BalanceScaleSceneCfg(scale_pos=(0.0, 0.12), shelf_pos=(0.0, 0.28),
                                box_row_y=-0.08, box_size=(0.05, 0.05, 0.045),
                                box_gap=0.09, shelf_slot_w=0.105)


for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_piper_scene_cfg(),
                robot="piper",
                control_mode=mode,
                robot_cfg=PiperRobotCfg(base_pos=(0.0, -0.22, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_franka_scene_cfg(),
                robot="bimanual_franka",
                control_mode=mode,
                robot_cfg=BimanualFrankaCfg(robots={
                    "left": ("franka", FrankaRobotCfg(base_pos=(-0.26, -0.50, 0.0), base_rot=_FRANKA_ROT)),
                    "right": ("franka", FrankaRobotCfg(base_pos=(0.26, -0.50, 0.0), base_rot=_FRANKA_ROT)),
                }),
                env_spacing=3,
            )
        ),
    )


# ---- Single-arm cobot bindings for the safe ----
# Base poses picked so (a) the door-swing arc
# (radius 0.42 around the hinge at (-0.21,-0.043)) clears the base column by >=5 cm and
# (b) the EE workspace covers the dial orbit, the handle and the prize-tumble press
# points (the EE never needs to reach the prize CENTER — the fingertip does).
for _mode in ("osc", "joint", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                robot="piper",
                control_mode=mode,
                robot_cfg=PiperRobotCfg(base_pos=(0.10, -0.40, 0.20), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

for _mode in ("osc", "joint", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                robot="wxai",
                control_mode=mode,
                robot_cfg=WxaiRobotCfg(base_pos=(0.14, -0.36, 0.20), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

# ---- robocasa microwave meal (appliance state machine + keypad) ----
# Scene physics only (NullRobot oracle/smoke). -> "articulated.microwave"
register_env(SUITE, lambda: EnvCfg(scene="microwave", robot="null", env_spacing=3))


# Robot bindings. Placements are STARTING guesses scaled from the safe/scale measured
# reach values at the same bench — re-verify with the per-binding stress smoke before
# any agent run (only the null smoke validates the scene itself). The microwave faces
# the robot (-y), bowls scatter front-left, the serving mat sits front-right; bowls are
# grasped by the ~8.5 mm rim (pinch) or palmed — both hand classes work.
def _mw_g1_cfg() -> MicrowaveMealSceneCfg:
    """G1 (short ~0.55 m arms): bench work pulled close; the deep reach to the
    turntable axis (~0.6 m) is the tight spot to verify."""
    return MicrowaveMealSceneCfg(
        table="packing",  # kinematic, sinks cleanly to humanoid work height (ikea pattern)
        surface_z=0.7,
        mw_pos=(0.0, 0.14),
        mat_pos=(0.30, -0.14),
        bowl_slots=((-0.26, -0.10), (-0.14, -0.20)),
    )


def _mw_gr1t2_cfg() -> MicrowaveMealSceneCfg:
    """GR1-T2 (primary embodiment: door + keypad + carry): same bench, wider layout."""
    return MicrowaveMealSceneCfg(
        table="packing",
        surface_z=0.7,
        mw_pos=(0.0, 0.20),
        mat_pos=(0.34, -0.10),
        bowl_slots=((-0.30, -0.06), (-0.16, -0.18)),
    )


def _mw_franka_cfg() -> MicrowaveMealSceneCfg:
    """Franka ablation: work on the packing table at 0.55 m (the base rides at table
    height, the repo's franka-binding convention), everything inside the ~0.75 m reach;
    bowls pinch-grasped by the rim (the 8 cm jaw cannot palm 115 mm). The station is
    RIGHT of the door's swing arc (see the binding's base_pos note): bowls spawn inside
    the arc (the task hazard — clear them before opening), the mat sits WELL right of
    the appliance, outside the arc, and every footprint stays fully on the deepened
    2.47 x 1.14 m packing top (y edges +/-0.57) with >=0.11 m margin."""
    return MicrowaveMealSceneCfg(
        surface_z=0.55,
        mw_pos=(0.0, 0.16),
        # Mat pushed to comfortable mid-reach, clear of the appliance (user layout
        # directive 2026-08-05): at (0.52,-0.06) the mat sat 6 cm off the appliance's
        # right face and its slots 0.28-0.33 m from the base, LOW — reaching them
        # folded the elbow around the base and wound the wrist trio ~140 deg (run 39
        # footage + per-joint debt), killing every later door grasp. At (0.70,-0.16)
        # the appliance-mat gap is 24 cm, slots sit 0.28-0.38 m out in FRONT-right,
        # and the mat still ends 0.385 m inside the table's right edge.
        mat_pos=(0.70, -0.16),
        # Slots front-left of the robot on the deepened table: visible to the camera
        # (the old between-base-and-appliance slots hid behind the arm), rim pinches
        # 0.33-0.46 m from the base (the position channel degrades past ~0.5 m low),
        # and INSIDE the door arc — the task's intended hazard (user-confirmed
        # 2026-08-05): the solve must stage them clear before opening the door.
        bowl_slots=((-0.02, -0.38), (0.11, -0.40)),
        # the weld-on-closure grip contract (panda-hand pools; see the scene cfg).
        # Engage radius covers the scan's rim FLARE: the pads stall on the flared lip
        # ~1.5-2 cm off the analytic rim circle (run 27: real grip, no weld at 12 mm).
        grasp_weld=True,
        grasp_weld_dist=0.022,
    )


# -> "articulated.microwave.g1.{joint,pink_ik}" / ".gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="microwave",
                scene_cfg=_mw_g1_cfg(),
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
                scene="microwave",
                scene_cfg=_mw_gr1t2_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.48, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

# -> "articulated.microwave.franka.{osc,joint}"
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="microwave",
                scene_cfg=_mw_franka_cfg(),
                robot="franka",
                control_mode=mode,
                # base RIGHT of the door's swing zone: the bar's opening arc (radius
                # 0.45 m about the hinge) passes within 0.14-0.19 m of any base parked
                # at x 0-0.28 in front — inside the arm's fold limit (pull stalls at
                # 23-32 deg, runs 16-17), and a fully open door would strike the robot.
                # (0.42, -0.30) put the CLOSED bar at 0.27 m — folded at the arc START,
                # latch never broke (run 18). From (0.44, -0.36) the arc spans
                # 0.32-0.51 m, keypad 0.27 m, turntable axis 0.65 m: all in-band, and
                # the base sits fully ON the deepened table (it overhung the 0.76 m
                # top's front edge; user layout note 2026-08-05).
                robot_cfg=FrankaRobotCfg(base_pos=(0.44, -0.36, 0.55), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )


# ---- dexmimicgen pouring (metered granular split-pour) ----
# Scene physics only (NullRobot oracle/smoke). -> "articulated.pouring"
register_env(SUITE, lambda: EnvCfg(scene="pouring", robot="null", env_spacing=3))


# Robot bindings. Placements are STARTING guesses scaled from the safe/scale/microwave
# measured reach values at the same bench — re-verify with the per-binding stress smoke
# before any agent run. The cup spawns nearest the robot (-y), the two pad-seated bowls
# behind it; the humanoid cup (outer dia 78 mm) is palmed by the dex hands, the franka
# binding shrinks it (outer dia 70 mm) under the 8 cm parallel jaw.
def _pour_g1_cfg() -> PouringSceneCfg:
    """G1 (short ~0.55 m arms): everything pulled close; the far bowl at ~0.6 m reach
    is the tight spot to verify."""
    return PouringSceneCfg(
        surface_z=0.7,
        cup_pos=(0.0, -0.16),
        bowl_slots=((-0.16, 0.06), (0.16, 0.06)),
        park_pos=(-0.34, -0.20),
    )


def _pour_gr1t2_cfg() -> PouringSceneCfg:
    """GR1-T2 (primary embodiment — the source runs this exact robot class): same
    bench, slightly wider layout."""
    return PouringSceneCfg(
        surface_z=0.7,
        cup_pos=(0.0, -0.16),
        bowl_slots=((-0.18, 0.10), (0.18, 0.10)),
        park_pos=(-0.38, -0.20),
    )


def _pour_franka_cfg() -> PouringSceneCfg:
    """Franka ablation: ground-level work inside the ~0.75 m reach; cup shrunk so the
    8 cm jaw can wrap it (outer dia 70 mm), bowls pinch-grasped by the 7 mm rim."""
    return PouringSceneCfg(
        cup_pos=(0.0, -0.14),
        bowl_slots=((-0.16, 0.10), (0.16, 0.10)),
        park_pos=(-0.34, -0.18),
        cup_inner_r=0.030,
        cup_h=0.10,
    )


def _pour_multi_cfg() -> PouringSceneCfg:
    """Dual Franka flanking the work (the honestly bimanual binding — the source is a
    two-handed pour): cup on the left arm's side, bowls centred between the bases."""
    return PouringSceneCfg(
        cup_pos=(-0.16, 0.0),
        bowl_slots=((0.12, 0.16), (0.12, -0.16)),
        park_pos=(-0.30, -0.24),
        cup_inner_r=0.030,
        cup_h=0.10,
        reserve_pos=(0.0, 1.05),
    )


# -> "articulated.pouring.g1.{joint,pink_ik}" / ".gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pouring",
                scene_cfg=_pour_g1_cfg(),
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
                scene="pouring",
                scene_cfg=_pour_gr1t2_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.48, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

# -> "articulated.pouring.franka.{osc,joint}"
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pouring",
                scene_cfg=_pour_franka_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.45, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

# -> "articulated.pouring.multi.{osc,joint}" — two Frankas facing each other across the
# work (the pen-holder dual-arm pattern): left at -x facing +x (default), right turned 180 deg.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pouring",
                scene_cfg=_pour_multi_cfg(),
                robot="multi",
                control_mode=mode,
                robot_cfg=MultiRobotCfg(robots={
                    "left": ("franka", FrankaRobotCfg(base_pos=(-0.55, 0.0, 0.0))),
                    "right": ("franka", FrankaRobotCfg(base_pos=(0.55, 0.0, 0.0),
                                                       base_rot=(0.0, 0.0, 0.0, 1.0))),
                }),
                env_spacing=3,
            )
        ),
    )
