"""Canonical runnable env configs for the packing suite — registered in `ENVS` by name.

Same convention as the assembly suite: `register_env` derives ``suite.scene[.robot
[.control_mode]]``. Scene-physics-only first (NullRobot smoke/oracle), embodiments after
the physics is proven.
"""

from __future__ import annotations

from pathlib import Path

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
    EggCartonSceneCfg,
    FruitsOnPlateSceneCfg,
    PenHolderSceneCfg,
    ToolPackingSceneCfg,
)

SUITE = "packing"
_EGG_CARTON_G1_USD = str(
    Path(__file__).resolve().parents[1] / "assets" / "egg_carton" / "g1_rubber_fingers.usda"
)


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
        exclude=("lime_a", "lime_b", "red_onion", "avocado01", "orange_01", "pumpkinlarge", "lime01_01",
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
          the bounds. Swept across the full circle of approach azimuths it is never liftable,
          while lemon_01 (76 x 50 x 51 mm) is, at the same spot. The difference is that the lemon has
          a genuinely flat-ish 50 mm short axis for a grasp to seat on and the near-spherical lime
          does not — it rolls out of a three-finger hand the way it would out of a human's
          fingertips.
      That leaves two: lemon_01 (50 mm) and pumpkinsmall (55 mm), both liftable. Two
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
        # (The superseded single-lemon deterministic tier's measured layout notes — grid
        # pocket, bin standoff, one-column geometry, yaw-presentation analysis — live in git
        # history at the previous revision; the fixed_layout below replaces the scatter grid
        # entirely, so those dials are gone rather than dead.)
        #
        # REWORKED 2026-09-01 into the two-destination SORT tableau (bussing the table): the
        # earlier single-lemon deterministic tier is superseded. Three structural changes,
        # each answering a measured defect of the shipped cell:
        #
        # * The robot STANDS BESIDE the table now: the old cell's packing bench (0.57 m half
        #   depth) had the pelvis 7 cm and the knees 10 cm inside its footprint (probed world
        #   AABBs, 2026-09-01). The RoboLab wooden table is only 0.35 m half-deep, so at the
        #   ORIGINAL measured pelvis (-0.50) the whole body clears its near edge by ~6 cm --
        #   every measured grasp pocket from the shipped solve stays valid -- and with the
        #   legs z-scaled to a 0.78 top the ground lands at z=0, so the feet stand naturally
        #   on the floor. No riser, no reach re-derivation.
        # * TWO destinations: blue crate (right) for produce, round tray (left) for the
        #   non-food items. The tray is deliberately flat — placement on it cannot wedge or
        #   penetrate the way a second crate's walls could.
        # * The FULL 16-object manifest is on the table (the RoboLab original's colourful
        #   richness): five in the front working row are the sort targets, the rest are a
        #   composed scenery spread outside the reach band, graded only via the
        #   no-distractor-in-bin rule.
        #
        # The old cell's measured grasp lessons (azimuth band 115-245, lemon narrow-axis
        # presentation, pumpkinsmall single-azimuth, lime unliftable) remain the basis for
        # the working row's item poses; per-item validation continues on the pod.
        surface_z=0.78,
        table="robolab",
        table_depth_scale=1.3,  # multiplies the asset's LONG axis (world x after the -90
        # rotation): a 1.3 m-wide tabletop, matching the original scene's proportions.
        # Destinations flank the working row at the reach annulus's edges; the DROP points
        # the solve uses are their robot-near quadrants (the centres themselves sit at
        # 0.36-0.41 m from the shoulder, past the 0.34 m pinch ceiling).
        # 0.38: at 0.35 the crate's west wall (x 0.234) sat in the descent path of the hand
        # picking the right-most item at torso yaw -25 deg (grip miss 115 mm, then the arm
        # pressed on the kinematic crate and the scene blew apart -- measured).
        bin_pos=(0.38, -0.16),
        bin_scale=(0.20, 0.22, 0.17),
        tray_pos=(-0.38, -0.16),
        tray_scale=0.7,
        # SORT THE CITRUS: lemons into the crate, oranges onto the tray. 2 + 2, chosen by
        # measured hand behaviour, not taste. The G1 palm grasp is validated on SMOOTH CONVEX
        # produce; every concave collider tried welded itself into the hand or blew up:
        #   - pumpkinsmall: grasped and lifted 196 mm, then stayed on the open hand through a
        #     palm-down roll and shaking (fingers hooked in its rib valleys), and the next
        #     approaches -- made with a pumpkin-sized lump in the hand -- shoved the lemon off
        #     the table; the pomegranate's crown is the same shape of problem;
        #   - milk jug (77 mm, 172 mm tall, 0.4 kg): toppled on the first approach; the next
        #     grasp caged the pumpkin against it and launched every item off the table;
        #   - red onion / avocado: explosive scanned colliders (ejected tens of metres).
        # The jugs, bowl, bottle and pen holder stay as CLUTTER: graded only through the
        # no-distractor-in-the-crate rule.
        # LEMONS TO THE CRATE, LIMES TO THE TRAY. Targets are the items that fit the hand: the
        # Dex3's two long fingers are stacked ~57 mm apart (about 37 mm of clear gap, ~77 mm
        # outer span), so a palm grasp only works on LYING produce <= ~50 mm tall. Measured
        # failures that fixed this list: a 63 mm orange caged between the fingers and the palm
        # blew up (9.5 m, then a numerical explosion); the 77 mm standing bottle took the upper
        # finger on its cap during the descent and blew up; the 30 mm pen holder had nothing to
        # squeeze; limes rolled onto a lumpy side and presented 67-71 mm. The clutter stays as
        # distractors, graded by the no-distractor-in-crate rule.
        sort_to_bin=("lemon_01", "lemon_02"),
        sort_to_tray=("lime_a", "lime_b"),
        # (the limes' green is baked into their own asset, lime_g; no spawn-time override)
        fixed_layout=(
            # WORKING ROW: a gentle arc, one item per torso yaw. The flat palm grasp's footprint
            # is ~14 x 26 cm (fingertips 13 cm past the palm centre, thumb 7 cm to the side,
            # wrist 2 cm above the table 8.5 cm on the robot side), so a second row 9 cm
            # behind the first gets pressed into the table by the descending fingers and PhysX
            # launches it metres (measured: pumpkin 11 m, lemon 4 m, jug 5 m in ONE descent).
            # One row it is -- and one arm's static reach holds only two items of it. The solve
            # therefore turns the WAIST toward each item, which moves the right arm's
            # calibrated sweet spot (0.245 m ahead of the shoulder) along an arc of radius
            # ~0.26 m about the pelvis (0, -0.50). These are the points of that arc at torso
            # yaw -25, 0, +25, +45 deg, 0.10-0.11 m apart; the destinations flank them.
            # arc of the sweet spot 0.27 m ahead of the shoulder, torso yaw -25/0/+25/+47 deg:
            # >= 6 cm from the table's near edge, ~12 cm apart
            # yaw = slot torso yaw + 90: the lemon's 50 mm short axis faces the palm of a hand
            # whose palm normal is torso -x (the fruits tier's measured presentation, 90 deg
            # centre in that cell), so the palm grasp closes across the narrow way.
            # Arc of fruit_delivery's verified stance (fruit 0.26 m ahead of and 0.10 m inboard
            # of the shoulder, i.e. 0.26 m dead ahead of the pelvis) at torso yaw -36/-12/+12/+36
            # (~0.109 m apart, >= 6 cm from the table's near edge; the thumb side of every
            # pick is already empty because picks go left to right). Container placement
            # is checked against the open fingertips (13 cm past the item along the
            # approach) and the thumb (8 cm to the robot's left) of every slot. The previous arc (0.27 m,
            # yaws -25/0/+25/+46) was measured on video to put the open fingertips of the
            # +46 slot onto the tray rim and the elbow of the 0 slot onto the crate's near
            # corner; the jammed arm then crushed the fruit into the table (launched).
            # BIMANUAL layout: the two lemons (-> crate, on the robot's right) on the right
            # arm's arc at torso yaw -36/-12, the two limes (-> tray, on the robot's left) on
            # the left arm's mirrored arc at +12/+36. Each arm works its own side; the crate
            # and tray are mirror images about the robot's midline.
            ("lemon_01", 0.153, -0.290, 54.0),
            ("lemon_02", 0.054, -0.246, 78.0),
            ("lime_a", -0.054, -0.246, 102.0),
            ("lime_b", -0.153, -0.290, 126.0),
            # Scenery spread: composed, colourful, OUT of the reach band, on a 1.3 x 0.7 m
            # top spanning x[-0.65, +0.65], y[-0.35, +0.35].
            ("milkjug_a01", -0.50, 0.16, 0.0),
            ("whitepackerbottle_a01", -0.47, -0.12, 0.0),
            ("lime01", 0.55, -0.20, 0.0),
            ("lime01_01", -0.10, 0.10, 70.0),
            ("orange_01", 0.36, 0.02, 0.0),
            ("orange_02", -0.38, 0.12, 0.0),
            ("crabbypenholder", 0.10, 0.02, 30.0),
            ("pomegranate01", -0.44, -0.04, 0.0),
            ("pumpkinsmall", 0.22, 0.06, 0.0),
            ("utilityjug_a03", -0.24, 0.10, 15.0),
            ("pumpkinlarge", 0.47, 0.14, 25.0),
            ("serving_bowl", 0.28, 0.22, 0.0),
        ),
        # The onion and avocado are BACK (they were excluded while their scanned colliders --
        # sliver-hull decompositions -- ejected them tens of metres from a clean spawn; those
        # colliders are now single convex hulls). 16 objects on the table.
        # avocado and onion: measured NOT graspable by the G1 hand even lying and correctly
        # gripped (seat 21-34 mm, lift <= 0 on every try) -- out, so nothing on the table
        # invites an impossible grasp.
        exclude=("red_onion", "avocado01"),
        subset_sample=False,
        shuffle_slots=False,
        # Small jitter so millimetre poses cannot be memorised; yaw stays authored until the
        # per-item grasps are re-validated at the new spots (the lemon lesson: yaw decides
        # whether the graspable axis presents inside the hand's azimuth band at all).
        reset_pos_jitter=0.006,
        reset_yaw_center_deg=0.0,
        reset_yaw_deg=0.0,
        # Spawn essentially AT REST (2 mm): the default 30 mm drop tumbles items and destroys
        # the authored presentation (measured on the old tier).
        drop_lift=0.002,
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
                #
                # base y -0.50 (the measured original): with the RoboLab wooden table's
                # 0.35 m half depth the body clears the near edge by ~6 cm at this pelvis,
                # and the ground plane sits at z=0, so the feet stand on the floor.
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75),
                                     # Reset ARM POSE, measured (A/B on the single-item probe,
                                     # 2026-09-01). Isaac's default rests both wrists at
                                     # (+/-0.15, -0.30, 0.85): in the working row, tangled with
                                     # crate, tray and items. Retracting the arms is NOT free:
                                     # the arm IK is a warm-started local solver and several
                                     # tucks (elbow-only 0.9; pitch -1.2 + elbow 1.4) parked
                                     # the hand below the table edge or in a bad basin and
                                     # the staging pose then missed by 265-366 mm (vs 31 mm
                                     # from the default). Right arm: pitch -1.05 + elbow 1.0
                                     # keeps the hand high (z ~0.92) over the table's middle
                                     # and the IK healthy (33 mm staging at -0.9). Left arm:
                                     # the solve's own LEFT_TUCK (roll 1.2, elbow 2.05), which
                                     # swings it clear of the table entirely and is what the
                                     # solve holds anyway.
                                     init_joint_overrides={
                                         "right_shoulder_pitch_joint": -1.05,
                                         "right_elbow_joint": 1.0,
                                         # left arm hangs naturally beside the hip (positive
                                         # pitch swings it back, clear of the table edge)
                                         "left_shoulder_pitch_joint": 0.45,
                                         "left_shoulder_roll_joint": 0.16,
                                         "left_elbow_joint": 0.90,
                                     },
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


# ---- Fruits on plate (RoboLab port: tell FRUIT from the rest, place each on the plate) --------
# Scene physics only (NullRobot oracle/smoke, the full 14-item set). -> "packing.fruits_on_plate"
register_env(SUITE, lambda: EnvCfg(scene="fruits_on_plate", robot="null", env_spacing=3))


def _fruits_on_plate_franka_cfg() -> FruitsOnPlateSceneCfg:
    """Franka (single arm, ~0.8 m reach): table-level work, base south of the bench. Plate
    front-right within easy reach; fruit grid in front, further out. One arm cannot sort AND hold,
    so the honest strategy is pick-from-table -> release over the standing plate.

    A STARTING-GUESS placement, per the suite convention — re-verify reach with
    `robot_binding_smoke` before trusting it; only the null smoke validates the scene.

    Item set reduced to what a parallel jaw can actually handle, reusing the sibling
    clear_organic_objects franka tier's measured exclusions (a ball-like fruit must be centred in
    the jaw to ~1 mm or first pad contact rolls it away, which measures IK precision rather than
    the identification and long-horizon sequencing this task exists to test):
      orange_01 (72 mm tall, the taller of the two oranges), pomegranate01 (64 mm, the largest
      sphere — failed every full-set run there even with per-attempt grasp diversity), and
      lime01_01 (the second lime; one lime keeps the shape in the mix at half the risk).
    FOUR fruits remain — two lemons, a lime and an orange.
    The non-fruit side keeps the three PRODUCE distractors, which are the ones that carry this
    task's identification (a pumpkin and an onion are food and still belong on the table), plus
    the storage box, and drops the three whose footprints do not fit a 0.48 x 0.36 m grid: the
    160 mm serving bowl (which fruit also rolls INSIDE, an unreachable degenerate pose rather
    than difficulty), the 309 mm wooden spoons and the 332 mm spatula.
    """
    return FruitsOnPlateSceneCfg(
        surface_z=0.55,  # packing table lowered to franka height (microwave convention)
        # Plate front-right, clear of the scatter and clear of the base: its near edge lands at
        # y = -0.09, 60 mm north of the base at y = -0.15, and the nearest grid slot's centre sits
        # 238 mm from the dish centre — 87 mm of clear table outside the rim, which an orange
        # (63 mm) still clears with the reset jitter (+/-25 mm) at its worst.
        plate_pos=(0.46, 0.06),
        # The grid must leave room for an OPEN PARALLEL JAW to descend between items — the
        # sibling tier measured a 6-column layout failing because the descending fingers hit the
        # NEIGHBOURS, not because of reach. 4 columns over 0.48 m -> 160 mm pitch in x.
        scatter_center=(0.0, 0.33),
        scatter_span=(0.48, 0.36),
        scatter_cols=4,
        exclude=("lime_a", "lime_b", "orange_01", "pomegranate01", "lime01_01",
                 "serving_bowl", "wooden_spoons", "spatula"),
        slot_override=(),  # the two long utensils are excluded, so nothing needs a fixed slot
        subset_sample=True,
        min_fruits=3,
    )


# -> "packing.fruits_on_plate.franka.{osc,diff_ik,pink_ik,joint}"
for _mode in ("osc", "diff_ik", "pink_ik", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="fruits_on_plate",
                scene_cfg=_fruits_on_plate_franka_cfg(),
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


def _fruits_on_plate_g1_cfg() -> FruitsOnPlateSceneCfg:
    """G1 humanoid (fixed pelvis, ~0.51 m to the fingertips): the SAME task in a cell resized to
    a short-armed, torso-fixed embodiment reaching over a standing bench with a three-finger hand
    instead of a parallel jaw.

    The geometry here is INHERITED, not re-derived: it is the measured cell of the sibling
    `clear_organic_objects` G1 tier with the crate swapped for the plate. Those measurements,
    which this layout depends on:

    * `surface_z` 0.78. The pelvis is welded at 0.75 and the shoulders sit 0.25 m above it, so
      the arm's usable HORIZONTAL radius is `sqrt(0.51^2 - (shoulder_z - work_z)^2)` — every
      centimetre the work sits below the shoulder is spent going down instead of out. At 0.78 the
      drop is 0.22 m and the radius ~0.46 m extreme / ~0.36 m usable. Scaling the G1's 1.0 m
      shoulder to a human's ~1.4 m, this is a ~1.1 m standing bench: high, but a real one.
    * the right shoulder lands at world (0.100, -0.500, 1.042) with the pelvis at (0, -0.50, 0.75)
      facing +y — 0.262 m above this surface.
    * the reachable region is an ANNULUS about 0.20-0.34 m from the shoulder in the table plane.
      0.438 m is the fingertip ceiling (`sqrt(0.51^2 - 0.262^2)`), but what has to arrive at the
      fruit is the PALM, which sits ~0.09 m back from the wrist, and the inner bound is where the
      arm folds back on itself.
    * within that annulus the palm grasp is holdable only at approach azimuths 115-245 deg, and
      the pocket where a near-horizontal palm approach both reaches AND lifts is only a few
      centimetres across — the measured spot is table-relative (0.000, -0.240), 0.279 m from the
      shoulder, and the same point 20 mm further out is 98-246 mm out of reach at the identical
      azimuth and tilt.

    THE PLATE IS THE EASY PART, and it is why this tier can be less reduced than the sibling's.
    A crate has to be cleared: the release pose must sit above a rim 0.11-0.17 m off the bench,
    and reaching down INTO it was measured out of the workspace. A plate has no walls — the dish
    rim is 47 mm and the floor 12 mm — so the release happens essentially at bench height, and
    the whole 266 mm-wide counted footprint is available to aim at instead of one crate interior.
    That freedom is spent on REACH: the release point is chosen along the shoulder-to-dish-centre
    line at whatever radius sits mid-annulus, rather than being pinned to the receptacle's centre.

    Layout: the plate's dish centre at (0.32, -0.26) — 0.326 m from the shoulder, so the segment
    of the shoulder-to-centre line lying inside the counted footprint spans 0.19-0.46 m from the
    shoulder and straddles the annulus; the plate's near edge stops 90 mm short of the pelvis
    plane, matching the crate's clearance in the sibling cell. The fruit sits on the measured spot
    at (0.000, -0.240), which is 0.320 m from the dish centre — 169 mm of clear table outside the
    rim, above the 120-140 mm the sibling tier measured as the band where a descent still works.

    ITEM SET: ONE fruit and ONE distractor, and the distractor choice is the point of this task.
    `pumpkinsmall` is produce, it is food, and it is NOT fruit — the sibling task counts it as a
    target and this one fails if it reaches the plate — so a one-distractor board still tests the
    distinction the task exists for. The fruit is `lemon_01`: of the produce it is the only one
    dependably graspable by this hand at this spot (measured in the sibling cell — lime01 settles
    tipped so its span runs 77-114 mm along every reachable azimuth, pumpkinsmall grasps at one
    azimuth of twelve, lemon_02 at 40 mm is below the hand's ~50 mm minimum, and the oranges and
    pomegranate are near-spherical at 63-64 mm).
    """
    # CELL SHIFT (robot clear of the bench). The original cell welded the pelvis at y=-0.50,
    # which is 71 mm INSIDE the bench footprint (measured slab: y +/-0.571): the tabletop
    # passed through the robot's waist. The pelvis now stands at y=-0.70 -- torso clear of
    # the bench edge like a person at a counter -- and every workpiece moved by the SAME
    # shoulder-relative offsets, so all the measured reach constants below still hold:
    # fruit spot stays 0.28 m from the shoulder, and the dish centre swings slightly
    # around the shoulder (0.346 m vs 0.326) so the plate stays fully ON the bench
    # (its rim now stops 11 mm short of the edge) while the fruit keeps 127 mm of clear
    # table outside the rim (>= the 120 mm descent band the sibling tier measured).
    return FruitsOnPlateSceneCfg(
        surface_z=0.78,
        plate_pos=(0.28, -0.405),
        plate_scale=1.0,  # RoboLab's own — a 301 mm dish. Kept at full size deliberately: the
        # plate is the one part of this cell the G1 finds EASY, and a big shallow target is what
        # makes a single release attempt land. Shrinking it would spend margin for nothing.
        # The fruit sits on the reachable spot at (0.000, -0.240) — the single grid slot — and the
        # distractor gets an explicit slot 180 mm beyond it. `_slot_xy` derives rows from the item
        # count, so with one grid item the slot IS `scatter_center`; changing the item count
        # re-derives this geometry, which is worth stating because it happens silently.
        scatter_center=(0.0, -0.44),
        scatter_span=(0.10, 0.03),
        scatter_cols=1,
        # ONE fruit type, and the exclusions are measured rather than assumed. Of the seven
        # RoboLab fruits this hand can only work the lemon:
        #   lemon_02 (40 mm across) is under the hand's ~50 mm minimum — the pad-to-pad gap
        #     bottoms out near 70 mm, so it does not fill the hand at all;
        #   orange_01/orange_02/pomegranate01 are near-spherical at 63-64 mm, and a sphere rolls
        #     out of a three-finger hand the way it would out of a person's fingertips;
        #   lime01/lime01_01 look admissible and are not — 62 mm narrowest against a 72 mm
        #     planning mouth — and the reason is SETTLING, not span. The lime comes to rest tipped,
        #     so its horizontal profile along the hand's reachable azimuths runs 72-97 mm however
        #     its yaw is commanded. Probed across eight reset draws: the lemon was liftable in every
        #     one, while the lime rolled out of the hand — no lift at a 72 mm span, and its
        #     in-hand seating 62-80 mm off afterwards. Excluded as unreliable by SHAPE, not span.
        exclude=("lime_a", "lime_b", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01",
                 "pumpkinlarge", "redonion", "serving_bowl", "storage_box", "wooden_spoons",
                 "spatula"),
        # The distractor sits 180 mm beyond the fruit, well clear of the hand's swept volume
        # (~+/-30 mm across the finger straddle at the near-horizontal grasp tilt) and off the
        # carry route, which goes up to traverse height before translating.
        slot_override=(("pumpkinsmall", 0.0, -0.26, 0.0),),
        # POSE IS RANDOMIZED, and that is where this tier differs from the sibling
        # `clear_organic_objects` G1 tier, which had to ship fully deterministic — same embodiment,
        # same bench, same hand. Two things bought the margin back:
        #   * the plate. Its counted footprint is a 266 mm disc rather than a 21 x 16 cm crate
        #     interior, so the release point can be chosen for REACH instead of being pinned above
        #     one spot. That removed the sibling's largest single error (a 150-240 mm place
        #     residual) and lets the pick-and-place complete on the FIRST attempt rather than on a
        #     retry sweep, which is what leaves room for the scene to vary.
        #   * the scene's item yaw is now applied correctly. The shared reset was writing
        #     (cos yaw, 0, 0, sin yaw) instead of the half-angle, silently DOUBLING every
        #     commanded orientation, so `reset_yaw_center_deg=90` actually presented items at
        #     180 deg. That put a lemon's 50 mm narrow axis at azimuth 90 — outside the hand's
        #     reachable 115-245 deg band — and forced every grasp onto its 76 mm profile instead.
        # RANDOMIZED: the fruit's xy within +/-15 mm and its yaw within +/-30 deg.
        # NOT RANDOMIZED: the yaw CENTRE stays at 90 deg, because that is what presents the fruit's
        # narrow axis at azimuth 180, dead centre of the hand's reachable band. Yaw is the one
        # dimension where this embodiment has no margin to give — a lemon's graspable narrow axis
        # lies 90 deg from its long axis, so the yaw decides whether that axis is presentable at
        # all: free +/-180 leaves only about half the draws favourable, and pinning it near 0 is
        # WORSE than free (it pins the narrow axis just outside the band).
        subset_sample=False,  # one fruit in the pool, so there is no subset to sample
        reset_pos_jitter=0.015,
        reset_yaw_center_deg=90.0,  # long axis along +y, so the narrow axis faces azimuth 180
        reset_yaw_deg=60.0,  # +/-30 deg about that centre
        shuffle_slots=False,  # one fruit, one distractor with a fixed slot: nothing to permute
        # Spawn essentially AT REST. The scene default drops items 30 mm and lets them settle,
        # which TUMBLES them: with yaw commanded to 90 deg the lemon still measured 65-84 mm along
        # the reachable azimuths rather than its 50 mm short axis, because a tipped box's oriented
        # bbox contributes its z extent to horizontal spans. 2 mm preserves the commanded
        # orientation, which is the entire point of a deterministic tier.
        drop_lift=0.002,
    )


# -> "packing.fruits_on_plate.g1.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="fruits_on_plate",
                scene_cfg=_fruits_on_plate_g1_cfg(),
                robot="g1",
                control_mode=mode,
                # Actuator PD retuned off Isaac's G1_29DOF_CFG defaults for TABLE-TOP work — the
                # same values the sibling G1 tier measured.
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
                # Hand (20 / 2 -> 400 / 16). Grip force in a position-controlled hand is
                # stiffness x (target - contact) error; at Isaac's default the hand reached the
                # fruit accurately and still closed without gripping it.
                #
                # base y -0.70, not -0.50: the bench slab spans y +/-0.571, and the old weld
                # stood the robot INSIDE it (tabletop through the waist). See the cell-shift
                # note on _fruits_on_plate_g1_cfg.
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.70, 0.75),
                                     arm_stiffness=1500.0,
                                     arm_damping=90.0,
                                     hand_stiffness=400.0,
                                     hand_damping=16.0),
                env_spacing=3,
            )
        ),
    )


# ---- RoboDojo fill-pen-holder (difficulty-floor tier, bimanual-friendly) ----------------------
# Scene physics only (NullRobot oracle/smoke). -> "packing.pen_holder"
register_env(SUITE, lambda: EnvCfg(scene="pen_holder", robot="null", env_spacing=3))


# ---- RoboDojo fill-egg-holder (long-horizon humanoid tier) ------------------------------------
# Scene physics only (NullRobot oracle/smoke). -> "packing.egg_carton"
register_env(SUITE, lambda: EnvCfg(scene="egg_carton", robot="null", env_spacing=3))


def _egg_carton_g1_cfg() -> EggCartonSceneCfg:
    """G1 bimanual layout: basket left, four-cell carton right, both in its measured band.

    The whole robot/fixture group sits 0.35 m closer to the table's front edge than the
    original calibration so the pelvis (y=-0.80) clears the 1.14 m-deep tabletop (front edge
    y=-0.57) instead of standing inside the bench.  The measured 0.40 m base-to-work offset —
    the only band where the 15-degree pre-grasp is reachable — is preserved exactly.  The
    fixtures sit close to the front edge on purpose: at a 0.23 m pull-back the ELBOW's
    down-forward swing zone landed exactly on the tabletop edge slab (the old inside-the-table
    base kept the elbow constrained above the top), and edge contact stalled every pocket
    descent 60--80 mm high; at 0.35 m the elbow swings in free air beyond the edge and only
    the wrist, well above surface height, crosses the edge plane.  ``ground_z=0`` pins the
    floor at the humanoid's feet and buries the lowered table's base instead of sinking the
    whole world by 0.29 m.
    """
    # Carton x=0.14 is the calibrated basin: shifting it to 0.08 to "help" the right column
    # re-rolled the whole contact chaos and broke the proven transfers.  The three-egg solver
    # fills back-right, front-left, then back-left (front-right is the spare).
    return EggCartonSceneCfg(
        surface_z=0.70,
        ground_z=0.0,
        basket_pos=(-0.14, -0.40),
        carton_pos=(0.14, -0.40),
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
                # Keep the humanoid outside the front of the packing table and
                # bring the movable work toward it instead.  RunPod calibration
                # showed that a 0.62 m base-to-work offset left 114--193 mm of
                # IK residual, while the 0.40 m band admits the 15-degree
                # pre-grasp.  Moving the fixtures, rather than the pelvis, also
                # avoids a visually implausible robot/table overlap.  Keep the
                # stock G1 base height: lowering it made the elevated pre-grasp
                # enter a shoulder singularity during RunPod calibration.
                # This task uses a thin rubber pad material on the existing finger colliders.
                # The source G1 hand is bare rigid plastic with its default physics material;
                # RunPod contact traces showed geometrically valid three-point grasps sliding
                # off a 41 g egg even during a 1 mm/substep lift.  The overlay changes friction
                # only on the hand collision shapes, leaving egg/basket/carton contacts honest.
                robot_cfg=G1RobotCfg(
                    base_pos=(0.0, -0.80, 0.75),
                    g1_usd=_EGG_CARTON_G1_USD,
                ),
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
