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

def _clear_organic_objects_g1_cfg() -> ClearOrganicObjectsSceneCfg:
    """G1 humanoid (fixed pelvis, ~0.51 m to the fingertips): the SAME task in a cell resized
    to a short-armed, torso-fixed embodiment reaching over a standing bench with a three-finger
    hand instead of a parallel jaw.

    Three dials carry the whole difference from the franka binding, each forced by measured
    geometry rather than taste:

    * `surface_z` 0.78 (vs the franka's 0.55). The pelvis is welded at 0.75 and the shoulders
      sit 0.25 m above it, so the arm's usable HORIZONTAL radius is
      `sqrt(0.51^2 - (shoulder_z - work_z)^2)` — every centimetre the work sits below the
      shoulder is spent going down instead of out. At the repo's usual G1 bench (0.7) that
      leaves 0.30 m of drop and only ~0.41 m of extreme (~0.30 m usable) radius, too little
      to hold a scatter grid AND a bin. At 0.78 the drop is 0.22 m and the radius ~0.46 m
      extreme / ~0.36 m usable — which the layout below fits. Scaling the G1's 1.0 m shoulder
      to a human's ~1.4 m, this is a ~1.1 m standing bench: high, but a real one.
    * a SMALLER bin — `bin_scale` (0.20, 0.22, 0.17) -> a ~23 x 18 cm crate 11 cm deep, a third
      of the franka binding's footprint. It has to fit beside the scatter inside one 0.36 m
      annulus, and it only ever holds the reduced organic set below (four fruits of 40-76 mm sit
      in one layer inside the ~21 x 16 cm counted interior). The shallow walls also matter: the
      rim lands 0.15 m under the shoulder, so releasing over it does not ask the arm to reach up
      and out at once.
    * a REDUCED item set (5 of 16). The G1 hand is a three-finger pinch — two fingers with
      45.8 mm phalanges opposing a thumb — not an 80 mm parallel jaw, so it has BOTH an upper and
      a LOWER size bound, and the organics have to sit between them:
        - upper: usable opposition width is ~86 mm (thumb parked), so the 62-65 mm spheres and the
          two tall ellipsoids the franka binding also drops are out;
        - lower: the pad-to-pad gap never closes below ~70 mm (measured), so anything much under
          ~50 mm cannot be pinched at all — it simply does not fill the hand. lemon_02, at 40 mm
          across its short axis, failed to lift in EVERY grasp configuration tried, in two
          different hand orientations. It is excluded as unpickable by construction rather than
          left in as false difficulty.
        - SHAPE, not just span: lime01 (76 x 61 x 60 mm) is also excluded, and its span is inside
          the bounds. The single-item study swept the full azimuth circle on it and it never
          lifted once, while lemon_01 (76 x 50 x 51 mm) lifted 262 mm at the same spot. The
          difference is that the lemon has a genuinely flat-ish 50 mm short axis for the pinch to
          seat on, and the near-spherical lime does not — it rolls out of a three-finger pinch the
          way it would out of a human's fingertips.
      That leaves two: lemon_01 (50 mm) and pumpkinsmall (55 mm), both verified liftable. Two
      organics is a thin board and a deliberate call — a G1 tier that reliably does two is worth
      more than one that unreliably attempts four.
      The clutter keeps the bottle and the pen holder and drops the three space hogs —
      the 160 mm serving bowl, the 171 mm utility jug and the milk jug. The jug goes for a second
      reason: at 172 mm it is the one item taller than the height a loaded hand traverses at, so it
      is the only thing on the table a carry could catch. Identification still means telling food
      from non-food, on a smaller board.

    Layout (table-relative; MEASURED — the right shoulder lands at world (0.100, -0.500, 1.042)
    with the pelvis at (0, -0.50, 0.75) facing +y, i.e. 0.262 m above this surface): scatter grid
    3 x 2 at (-0.02, -0.27) spanning 0.24 x 0.10 m, and the bin beside it at (0.29, -0.31).

    The reachable region is an ANNULUS about 0.20-0.34 m from the shoulder in the table plane,
    and it is smaller than it first looks. 0.438 m is the fingertip ceiling
    (`sqrt(0.51^2 - 0.262^2)`), but what has to arrive at the fruit is the PINCH ZONE, which sits
    ~66 mm short of the fingertips — so the working ceiling is ~0.34 m, and the inner bound is
    where the arm folds back on itself. Four limits pin the numbers, each one measured:

    * the OUTER bound is real: an earlier grid put its far-left slot 0.46 m out, and every
      azimuth tried there missed the hover pose by 215-292 mm.
    * the INNER bound is real too: a first guess (grid centre y -0.30) sat 0.12 m from the
      shoulder, where the arm must fold to reach straight down — the humanoid analogue of the
      franka binding's cramped inner shell.
    * the grid sits WEST of the bin, not spread across it. The hand approaches a pinch with its
      wrist offset ~0.14 m toward +x of the fruit (the pinch zone is out along the fingers, and
      every azimuth in the hand's usable band offsets the wrist the same way). A layout with the
      grid's east column at x 0.15, 60 mm from the bin wall, put the WRIST goal inside the
      crate's footprint at rim height and that column was unreachable (224-251 mm at every
      azimuth). Ending the grid at x 0.10 leaves the wrist clear above the rim.
    * the bin is pulled IN to 0.29: releasing over its centre puts the pinch zone 0.28 m out and
      0.07 m below the shoulder, comfortably inside the annulus.

    Grid pitch is 0.12 m in x and 0.10 m in y. 3 columns, not 4: four columns across the
    reachable width would space items 67 mm apart — less than the widest item — and a hand
    cannot descend between two touching items.
    """
    return ClearOrganicObjectsSceneCfg(
        surface_z=0.78,
        # Bin pushed OUT and the grid pulled IN, roughly doubling the clear space between the
        # produce and the crate wall (74 mm -> 154 mm). At the old spacing the wrist — which rides
        # ~0.14 m toward +x of whatever it grasps — landed over the crate for the grid's east
        # column, and a fruit near the wall left no room for the fingers descending on that side.
        # Dropping to four items is what buys the room: a 2 x 2 grid instead of 3 x 2.
        # 0.24, pulled in again for the PALM grasp: the fruit is now held only ~0.098 m from
        # the wrist instead of ~0.144 m, which puts the wrist correspondingly FURTHER out at
        # the drop pose. Measured carry residual with the deep aim at bin x 0.27 was 243 mm.
        bin_pos=(0.24, -0.31),
        bin_scale=(0.20, 0.22, 0.17),
        # The grid is placed on a MEASURED sweet band, not a guess. A 20-placement single-item
        # sweep (scripts/research_g1_grasp.py --phase pos) lifted from only two of them, and both
        # sat 120-140 mm clear of the crate wall and 0.27-0.34 m from the shoulder. Closer to the
        # crate than ~50 mm the descent residual jumps to 59-163 mm and nothing grasps; ~270 mm out
        # the arm reaches fine but the lift dies at 20-48 mm.
        # The two ORGANICS occupy the near row (grouped slot shuffling keeps produce in produce
        # slots), so the near row is what has to land in the band: these dials put it at
        # (-0.02, -0.26) and (0.06, -0.26) — 134-214 mm clear of the crate, 0.24-0.27 m out. The far
        # row holds only distractors, which are never grasped, so its reach costs nothing.
        # y -0.19 so the ORGANIC row lands at y -0.240 exactly. That is not a round number, it
        # is the spot where the single-item study completed the whole pick-and-place at five
        # consecutive azimuths; the row 20 mm further out (y -0.260) missed its staging pose by
        # 98-246 mm at the same azimuth and tilt. At a near-horizontal palm approach the
        # reachable set is that tight, so the grid is pinned to the verified spot.
        scatter_center=(-0.02, -0.19),
        scatter_span=(0.08, 0.10),
        scatter_cols=2,
        exclude=("lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01",
                 "pumpkinlarge", "red_onion", "avocado01", "serving_bowl", "utilityjug_a03",
                 "milkjug_a01"),
        subset_sample=False,   # only two organics remain; sampling a subset of two is not variety
    )


# -> "packing.clear_organic_objects.g1.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="clear_organic_objects",
                scene_cfg=_clear_organic_objects_g1_cfg(),
                robot="g1",
                control_mode=mode,
                # Actuator PD retuned off Isaac's G1_29DOF_CFG defaults for TABLE-TOP work.
                #
                # Arm (3000 / 10 -> 1500 / 90). The default pair is badly underdamped for precise
                # manipulation: critical damping for this link inertia is ~2*sqrt(k*m) ~ 35, so
                # damping 10 is about a third of critical and the arm RINGS. Measured
                # consequences, both of which vanish with a damped arm: the same wrist goal at the
                # same azimuth missed by 45 mm on one attempt and 427 mm on the next (an
                # oscillating arm, not an unreachable pose), and the ringing arm loaded up against
                # the table and then catapulted a fruit over a metre. Halving the stiffness also
                # makes contact with produce compliant rather than percussive.
                #
                # Hand (20 / 2 -> 200 / 10). The default is tuned for free-air finger poses; a
                # three-finger pinch on a 100 g fruit needs the phalanges to HOLD against contact.
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75),
                                     arm_stiffness=1500.0,
                                     arm_damping=90.0,
                                     # 400, not Isaac's 20: a three-finger pinch on a 100 g fruit
                                     # has to HOLD against contact, and at 200 the hand reached the
                                     # fruit accurately (descent residual 14-15 mm, seating
                                     # 33-48 mm) and still failed to lift it — closing without
                                     # gripping. Grip force in a position-controlled hand is
                                     # stiffness x (target - contact) error, so this and the
                                     # solve's over-driven close target are the same lever.
                                     hand_stiffness=400.0,
                                     hand_damping=16.0),
                env_spacing=3,
            )
        ),
    )


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
