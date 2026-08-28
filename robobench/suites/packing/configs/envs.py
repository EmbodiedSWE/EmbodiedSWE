"""Canonical runnable env configs for the packing suite — registered in `ENVS` by name.

Same convention as the assembly suite: `register_env` derives ``suite.scene[.robot
[.control_mode]]``. Scene-physics-only first (NullRobot smoke/oracle), embodiments after
the physics is proven.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import (
    AttachedArmRobotCfg,
    BimanualFrankaCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    MultiRobotCfg,
    XArm7RobotCfg,
)
from robobench.suites.packing.scenes import (
    ClearOrganicObjectsSceneCfg,
    PenHolderSceneCfg,
    ToolPackingSceneCfg,
)

SUITE = "packing"


# ---- Clear organic objects (RoboLab port: identify the produce, clear it into the bin) --------
# Scene physics only (NullRobot oracle/smoke, full 11-organic set). -> "packing.clear_organic_objects"
register_env(SUITE, lambda: EnvCfg(scene="clear_organic_objects", robot="null", env_spacing=3))


# Robot bindings. Placements are STARTING guesses on the shared packing bench — re-verify reach
# with `robot_binding_smoke` before trusting them (only the null smoke validates the scene). The
# bindings sample an organic SUBSET per episode (`subset_sample`, 4+ organics) so a graded floor
# of episodes stays solvable while the clutter (5 distractors) is always present.
def _clear_organic_objects_franka_cfg() -> ClearOrganicObjectsSceneCfg:
    """Franka (single arm, ~0.8 m reach): table-level work, base south of the bench. Bin front-
    right within easy reach; clutter grid in front, further out. One arm cannot sort AND hold, so
    the honest strategy is pick-from-table -> drop-into-standing-bin."""
    return ClearOrganicObjectsSceneCfg(
        surface_z=0.55,  # packing table lowered to franka height (microwave convention)
        # bin front-right, clear of the scatter; clutter grid pulled IN close to the base so
        # every top-down grasp sits in the 0.30-0.55 m band (far-low reaches go singular)
        bin_pos=(0.42, -0.05),
        # The grid must leave room for an OPEN PARALLEL JAW to descend between items: the
        # gripper spans 80 mm plus finger thickness, and a 6-column layout over 0.44 m put
        # items only 88 mm apart, so the descending fingers hit the NEIGHBOURS and the hand
        # stopped one fruit-height above the table (measured: fingertips floored at z=0.608
        # while a free-space reach probe reached the tabletop at 0.551 — so it was contact,
        # not reach). 4 columns over 0.44 m -> 147 mm pitch in x, and the jaw is aimed along
        # the roomy x direction by the solve's neighbour-aware azimuth choice.
        scatter_center=(0.0, 0.33),
        scatter_span=(0.44, 0.30),
        scatter_cols=4,
        # Leave out the items whose difficulty is INCIDENTAL rather than intended. A ball-like
        # fruit must be centred in the jaw to ~1 mm or first pad contact rolls it away, which
        # measures IK precision, not the identification + long-horizon sequencing this task
        # exists to test. Excluded, each for a measured reason (2026-08-28, full-set runs):
        #   red_onion (59x59x90) / avocado01 (61x61x92) — tall ellipsoids, never picked reliably
        #   orange_01 (72 mm tall) — the taller of the two oranges
        #   pumpkinlarge — the one item that failed in EVERY full-set run
        #   lime01_01 — the second lime; one lime keeps the shape in the mix at half the risk
        #   pomegranate01 — 64 mm, the largest remaining sphere; failed every full-set run even
        #     with per-attempt grasp diversity
        # FIVE organics remain — 2 lemons, a lime, an orange and a small pumpkin — each verified
        # to clear reliably, still four distinct produce shapes among the 5 non-food distractors,
        # so the identification and long-horizon sequencing the task measures are intact. The
        # NULL preset keeps the full 11, so the scene and its oracle still cover RoboLab's whole
        # named set; this is the ARM binding's solvable tier.
        exclude=("red_onion", "avocado01", "orange_01", "pumpkinlarge", "lime01_01",
                 "pomegranate01"),
        subset_sample=True,
        min_organics=4,
    )


# -> "packing.clear_organic_objects.franka.{osc,diff_ik,pink_ik,joint}"
for _mode in ("osc", "diff_ik", "pink_ik", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="clear_organic_objects",
                scene_cfg=_clear_organic_objects_franka_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.15, 0.55),
                                         base_rot=(0.7071068, 0.0, 0.0, 0.7071068),
                                         # the known-good single-arm pick-place config
                                         # (pen_holder lesson: the default nullspace posture
                                         # winds the arm on long lateral servos)
                                         nullspace_dof_pos=(),
                                         gripper_effort_limit=120.0,
                                         gripper_stiffness=4000.0),
                env_spacing=3,
            )
        ),
    )

# NOTE: clear_organic_objects is a TABLE-TOP manipulation task — arm bindings only, no humanoid.


# ---- RoboDojo fill-pen-holder (difficulty-floor tier, bimanual-friendly) ----------------------
# Scene physics only (NullRobot oracle/smoke). -> "packing.pen_holder"
register_env(SUITE, lambda: EnvCfg(scene="pen_holder", robot="null", env_spacing=3))


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
for _mode in ("osc", "diff_ik", "pink_ik", "joint"):
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

# The SAME tool-packing cell for the transfer suite: gen3n7_panda + xarm7 (panda-hand dial).
# gen3n7 (~0.9 m): the franka layout verbatim from the franka's bench mount; probe-verified
# ready pose (2026-08-18: box hover 6.3 cm — the march terminates at the box's near face —
# item picks <= 0.9 cm). xarm7 (~0.70 m): the franka layout's box<->items span (0.65 m)
# exceeds its usable annulus from ANY base (probe round 2026-08-18: items landed under the
# base column at an east mount, the box out of reach from the west), so its binding shrinks
# the cell — box pulled to (0.0, 0.10), items on a tighter arc — same task, per-embodiment
# placement dials (the humanoid-binding convention).
#   -> "packing.tool_packing.{gen3n7_panda,xarm7}.{osc,joint}"
def _tool_packing_xarm7_cfg() -> ToolPackingSceneCfg:
    return ToolPackingSceneCfg(
        surface_z=0.55,
        box_pos=(0.0, 0.10),
        item_slots=((-0.24, -0.22), (-0.28, -0.30), (-0.22, -0.36)),
        reset_pos_jitter=0.0,
        reset_yaw_deg=0.0,
        box_pos_jitter=0.0,
        box_yaw_deg=0.0,
        shuffle_slots=False,
    )


for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="tool_packing",
                scene_cfg=_tool_packing_franka_cfg(),
                robot="gen3n7_panda",
                control_mode=mode,
                robot_cfg=AttachedArmRobotCfg(base_pos=(-0.43, 0.0, 0.55),
                                              arm_effort_limit=120.0,
                                              gravity_compensation=True,
                                              # probe-verified ready pose (2026-08-18, drift-
                                              # weighted round: hold 1.4 cm over 60 steps, box
                                              # hover 4.8 cm, item picks <= 0.9 cm), held
                                              # actively by the nullspace
                                              default_dof_pos=(-0.9472, 0.5453, 1.0810, 1.9326,
                                                               0.2292, 0.2224, -2.0194),
                                              nullspace_dof_pos=(-0.9472, 0.5453, 1.0810, 1.9326,
                                                                 0.2292, 0.2224, -2.0194)),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="tool_packing",
                scene_cfg=_tool_packing_xarm7_cfg(),
                robot="xarm7",
                control_mode=mode,
                robot_cfg=XArm7RobotCfg(gripper="panda_hand",
                                        base_pos=(-0.43, 0.0, 0.55),
                                        arm_effort_limit=120.0,
                                        gravity_compensation=True,
                                        # probe-verified ready pose (2026-08-18, drift-weighted
                                        # round: hold 0.0 cm, box hover 4.8 cm, item picks
                                        # <= 3.3 cm at the shrunk cell), nullspace-held
                                        default_dof_pos=(-1.5625, -0.8566, 1.0458, 0.7819,
                                                         0.7351, 0.5271, 3.6916),
                                        nullspace_dof_pos=(-1.5625, -0.8566, 1.0458, 0.7819,
                                                           0.7351, 0.5271, 3.6916)),
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
for _mode in ("osc", "diff_ik", "pink_ik", "joint"):
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
