# approach_grasp_banana_i69 — Tilt Maze (gimballed labyrinth, beacon-branched, trap well)

**Scene** `tilt_maze` / env `simgen.tilt_maze` (NullRobot, scene-level physics).
**Seed** `pick_place/approach_grasp_banana` (RoboVerse
`roboverse_pack/tasks/pick_place/approach_grasp_banana.py`).

## Provenance and strategic difference

The seed is a **prehensile pick-and-place**: a Franka approaches a banana lying among
table clutter, closes its jaw around it, and carries it toward a basket; success is a
gripper-object relation (approach + grasp + transport + release), and the judged object
is exactly the thing held.

This task removes every element of that plan:

- **The judged payload can never be touched.** The orange ball (28 mm) lives inside a
  walled labyrinth tray whose corridor section is covered by a grate of 12 roof slats
  with **6 mm gaps** — narrower than a Franka fingertip (~7 mm) and far narrower than
  the ball. Past the open start bay the ball is sealed in; there is no approach, no
  grasp, no carry, no release.
- **Control is indirect gravity-steering through an anchored compliant mechanism.**
  The tray hangs on a 2-axis revolute gimbal (pitch + roll, ±12° hard stops) above a
  fixed pedestal, spring-centered back to level (κ = 0.5, damping 0.10). The only
  intended handles are four **yellow press tabs** on the outside rim; pressing one
  tilts the maze and gravity rolls the ball.
- **A per-episode branch decision.** The cross corridor ends in two 20 mm-deep wells.
  A green **beacon** post is re-posed each episode beside the goal well (side is a
  coin flip); the solver must read the beacon and steer the branch accordingly.
- **An irreversible failure mode.** Escaping a well needs a ~25.4° tilt
  (`atan(d/√(r²−d²))`, ball center 6 mm below the lip) but the gimbal stops at 12°
  (measured max under pinned max drive: 19.1° combined, still 6° short). Committing to
  the wrong branch loses the episode — the rubric caps it at 0.25.

Distinct from every corpus task read this session: the beam-balance family
(`approach_grasp_i29`, coke_task_i15, pick_and_lift_i16, pull_cube_i20) places weights
and reads equilibrium; `pull_cube_tool_i1` pushes carousel pegs directly; `track_bowl_i27`
(bell cage) and `hockey_i325` contact the ball's cage or the ball; `obstacle_i17` bowls a
free ball by direct contact; none has a spring-centered 2-DOF tilt plant, a grated
untouchable payload, a beacon-selected branch, or an in-scene irreversible trap.

## Geometry (tray local frame, origin at the gimbal pivot; pivot 0.115 m above ground)

Open start bay (|x| < 33 mm, y ∈ [0.02, 0.11], open sky) → roofed entry channel →
cross corridor (y ∈ [−0.10, −0.04], full width) → terminal wells at |x| ∈
[0.095, 0.155], floors 20 mm below the main floor. Rim walls flush with the slat tops;
fillers make the roofed region solid outside the channel; 4 press tabs (5 cm square,
world height ≈ 0.138 m) outboard of the rim at N/S/E/W. All colliders carry explicit
friction materials; tray mass/CoM(at pivot)/inertia authored explicitly. Honesty
asserts in `TiltMazeSceneCfg.__post_init__`: trap escape angle > stops + 8°, grate gap
beats finger and ball, roof clearance, corridor clearances, spring re-levels within the
success tolerance with the ball parked anywhere, drive can reach the stops, branch
saturation inside the goal well, per-substep travel ≪ the continuity gate, tabs inside
the documented reach, weights sum to 1.

## Rubric (latched, continuity-gated)

Every latch requires per-substep ball travel < 2 cm (`cont_max`) — teleports earn
nothing. `departed` (0.10): continuous passage through the roofed entry channel.
`crossed` (0.15): reaching the cross corridor after departing. `branch` (0.30, latched
max): progress toward the goal side, clamp(side·x/0.105, 0, 1) — the wrong direction
earns 0. `potted` (0.15): entering the goal well (z below the lip band) after crossing
with branch > 0.85. `success` (0.30, live): potted AND ball currently settled in the
goal well AND the tray released level (< 3°) and quiet. Score = exactly 1.0 iff
success(); null policy ~0; wrong well caps at 0.25; disturbing the tray after success
falls back to the latched 0.70 and success returns when re-released.

## Solution (solve.py) — no object teleport at all

The certificate never teleports the ball; everything flows through the scene's
`tilt_drive` plant buffer (bounded ≤ 0.30 N·m — the wrench of a fingertip pressing a
rim tab, an order of magnitude under the clamp only at ~0.05 N·m equivalents):

0. reset(seed), settle → `SIM_GEN_SCORE` ~0
1. SOUTH: ~6°-equivalent south tilt with mild x-centering rolls the ball out of the
   bay, under the grate, into the cross corridor → ~0.25
2. BRANCH: read `side` (the beacon), ~7°-equivalent tilt toward it (+2.5° south hold)
   until the ball falls into the **goal** well → 0.70
3. RELEASE: zero drive; the spring re-levels (residual ~1.8° < 3°), ball settles →
   1.00, success
4. PERSIST: 3.5 s untouched, success at every poll → `SIM_GEN_SOLVE: SUCCESS`

Verified on the forge: seeds 0 (side = west) and 1 (side = east), both `rc=0`,
score ladder 0 → 0.25/0.26 → 0.70 → 1.00, non-decreasing.

## Execution order (declared)

South leg first (bay → cross corridor), then the branch leg toward the beacon side,
then release. The branch choice is the episode's decision point: the wells are
terminal, so the wrong branch cannot be undone (by design and by measured physics).

## Embodiment argument (Franka)

Documented base at (0, 0). The four tabs sit 0.31–0.69 m from the base (farthest tab
outer edge 0.71 m < 0.72 m envelope), 5 cm square at height ~0.138 m — comfortable
single-finger or closed-jaw press targets. Required press wrench about the pivot is
≤ 0.3 N·m at ≥ 0.17 m lever arm ⇒ < 2 N of fingertip force. The channel gives 32 mm
of lateral slack around the ball, so coarse closed-loop pressing (watch the ball
through the open bay and the grate gaps, modulate the press) suffices; no precision
grasp anywhere. One plausible strategy: press the north tab to hold the tray level,
then lean on the south tab until the ball crosses, then the east/west tab named by the
beacon, then lift off.

## Checks

`smoke.py` — one linear recorded run, **15/15 PASS** on the forge: settle/finite,
randomization readback (side flips, spawn varies, beacon matches side), null ~0,
seed-strategy carry-and-drop rejected by the grate, teleport-into-well rejected
(continuity/latch), wrong-well capped 0.25 with verified real actuation,
trap irreversibility under full drive, oracle traverse (0.70 held), monotone trace,
exactness (release → success, score == 1.0), persistence, disturb/recover latch
(0.70 floor, success live), gimbal anchored (< 5 mm drift, tilt within stop envelope +
compliance, below escape), floor near-miss strictly below the potted plateau, ≥ 20
video frames saved (`frames.npz`).
