"""Canonical runnable env configs for the packing suite — registered in `ENVS` by name.

Same convention as the assembly suite: `register_env` derives ``suite.scene[.robot
[.control_mode]]``. Scene-physics-only first (NullRobot smoke/oracle), embodiments after
the physics is proven.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import (
    BimanualFrankaCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    MultiRobotCfg,
)
from robobench.suites.packing.scenes import (
    EggCartonSceneCfg,
    PenHolderSceneCfg,
    ToolPackingSceneCfg,
)

SUITE = "packing"


# ---- RoboDojo fill-pen-holder (difficulty-floor tier, bimanual-friendly) ----------------------
# Scene physics only (NullRobot oracle/smoke). -> "packing.pen_holder"
register_env(SUITE, lambda: EnvCfg(scene="pen_holder", robot="null", env_spacing=3))


# ---- RoboDojo fill-egg-holder (long-horizon humanoid tier) ------------------------------------
# Scene physics only (NullRobot oracle/smoke). -> "packing.egg_carton"
register_env(SUITE, lambda: EnvCfg(scene="egg_carton", robot="null", env_spacing=3))


def _egg_carton_g1_cfg() -> EggCartonSceneCfg:
    """G1 bimanual layout: basket left, four-cell carton right, both in its measured band."""
    return EggCartonSceneCfg(
        surface_z=0.7,
        basket_pos=(-0.14, 0.12),
        carton_pos=(0.14, 0.12),
    )


# -> "packing.egg_carton.g1.{joint,pink_ik}".  Fixed-base G1 stands in front of the
# packing table; each hand can cover one side while the 2x2 carton remains at midline reach.
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="egg_carton",
                scene_cfg=_egg_carton_g1_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75)),
                env_spacing=3,
            )
        ),
    )


# Robot bindings. Placements are STARTING guesses copied from the stacking-toy / packing measured
# reach values for the same embodiments at the same bench — re-verify with the per-binding stress
# smoke before any agent run (only the null smoke validates the scene itself). The holder sits to
# the robot's right, pens scatter on a front arc to the left (the source's left/right split).
def _pen_holder_g1_cfg() -> PenHolderSceneCfg:
    """G1 (short ~0.55 m arms): work on a 0.7 m bench, holder right-front, pens on one front arc."""
    return PenHolderSceneCfg(
        surface_z=0.7,
        holder_pos=(0.18, 0.22),
        pens_center=(0.0, 0.16),
        spawn_radii=(0.26,),
        spawn_arc=(215.0, 325.0),
    )


def _pen_holder_gr1t2_cfg() -> PenHolderSceneCfg:
    """GR1-T2 (longer arms): same bench, slightly wider layout."""
    return PenHolderSceneCfg(
        surface_z=0.7,
        holder_pos=(0.20, 0.26),
        pens_center=(0.0, 0.18),
        spawn_radii=(0.30,),
        spawn_arc=(205.0, 335.0),
    )


def _pen_holder_franka_cfg() -> PenHolderSceneCfg:
    """Franka (single arm): table-level work in front of the base. One arm cannot hold AND
    fill, so the honest single-arm strategy is inserting into the STANDING holder — the rubric
    never requires holding it. The 12 mm pencils and the ~69 mm cup body (or its thin rim)
    both fit under the 8 cm jaw. The scatter arc faces away from the base (rotated after the
    base moved to y=-0.30 for the table's visual edge): every slot sits >= 0.42 m out — the
    old south slot landed 0.32 m from the new base, inside the weak close-in-grasp band.
    (Shifting the whole layout back instead put the cup by the table's frame rail, where a
    carry bump toppled it — watched on video.)"""
    return PenHolderSceneCfg(
        surface_z=0.55,  # packing table lowered to franka height (microwave convention)
        holder_pos=(0.12, 0.18),
        pens_center=(-0.10, 0.15),
        spawn_radii=(0.14,),
        spawn_arc=(60.0, 240.0),
    )


def _pen_holder_multi_cfg() -> PenHolderSceneCfg:
    """Dual Franka flanking the work (the genuinely bimanual binding, mapping the source's dual
    ARX X5): holder on the right arm's side, pens on the left arm's side."""
    return PenHolderSceneCfg(
        surface_z=0.55,  # packing table lowered to franka height (microwave convention)
        holder_pos=(0.14, 0.0),
        pens_center=(-0.14, 0.0),
        spawn_radii=(0.16,),
        spawn_arc=(100.0, 260.0),
    )


# ---- Tool packing (articulated toolbox: 2 doors + 3 drawers, real scanned assets) -------------
# Scene physics only (NullRobot oracle/smoke). -> "packing.tool_packing"
register_env(SUITE, lambda: EnvCfg(scene="tool_packing", robot="null", env_spacing=3))


# Franka binding: a STARTING-guess placement mirroring the pc_motherboard band analysis (base
# west of the work, drawers opening toward the arm) — re-verify with the per-binding stress
# smoke before any agent run. The box sits deep (+x) so the swinging doors clear the arm;
# items start on a front-left arc OUTSIDE the door sweep (x < box_front - door_len) yet
# inside the 0.36-0.60 m top-down band. Deterministic spawn for the smoke (jitter 0).
def _tool_packing_franka_cfg() -> ToolPackingSceneCfg:
    # The box sits OFF-AXIS (+y) from the base: straight-ahead low reaches put the wrist
    # in the outstretched singular plane and the pinch stalled 8-16 cm above every target
    # (measured via the driver's instrumented job); the pc_motherboard binding solved the
    # same problem by putting its bolt row off-axis. Items mirror to the -y side.
    return ToolPackingSceneCfg(
        surface_z=0.55,  # packing table lowered to franka height (microwave convention)
        box_pos=(0.10, 0.16),
        item_slots=((-0.29, -0.28), (-0.33, -0.36), (-0.27, -0.42)),
        reset_pos_jitter=0.0,
        reset_yaw_deg=0.0,
        box_pos_jitter=0.0,
        box_yaw_deg=0.0,
        shuffle_slots=False,
    )


# -> "packing.tool_packing.franka.{osc,joint}" — base riding ON the packing bench top
# (the repo's franka-binding convention), preserving the measured base->box offset of
# the validated lab-table layout (0.53 east, 0.16 north; box world (0.10, 0.16), base
# west of the work with the doors swinging clear).
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="tool_packing",
                scene_cfg=_tool_packing_franka_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(-0.43, 0.0, 0.55)),
                env_spacing=3,
            )
        ),
    )

# -> "packing.pen_holder.g1.{joint,pink_ik}" / ".gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pen_holder",
                scene_cfg=_pen_holder_g1_cfg(),
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
                scene="pen_holder",
                scene_cfg=_pen_holder_gr1t2_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.48, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

# -> "packing.pen_holder.franka.{osc,joint}"
# base rides ON the packing tabletop at surface height (the microwave franka
# convention); y -0.30 keeps it clear of the scatter with the whole plate on the
# deepened 1.14 m top
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pen_holder",
                scene_cfg=_pen_holder_franka_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.30, 0.55),
                                         base_rot=(0.7071068, 0.0, 0.0, 0.7071068)),
                env_spacing=3,
            )
        ),
    )

# -> "packing.pen_holder.multi.{osc,joint}" — two Frankas facing each other across the work
# (left at -x facing +x = the default facing; right at +x turned 180 deg). EnvCfg.control_mode
# propagates to both children.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pen_holder",
                scene_cfg=_pen_holder_multi_cfg(),
                robot="multi",
                control_mode=mode,
                robot_cfg=MultiRobotCfg(robots={
                    # +/-0.50, riding on the packing tabletop at surface height
                    "left": ("franka", FrankaRobotCfg(base_pos=(-0.50, 0.0, 0.55))),
                    "right": ("franka", FrankaRobotCfg(base_pos=(0.50, 0.0, 0.55),
                                                       base_rot=(0.0, 0.0, 0.0, 1.0))),
                }),
                env_spacing=3,
            )
        ),
    )
