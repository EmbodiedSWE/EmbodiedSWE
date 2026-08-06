# primer_stove (libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i21)

**Env name:** `simgen.primer_stove` (scene `primer_stove`, registered scene-level with
robot `null`; `solve.py` builds its own Franka env).
**Tier:** medium — cyclic mechanism actuation with feedback (2-4 full pump strokes),
then a knob-handle carry onto the burner.
**Execution order:** NOT required — priming before or after placing the griddle both
succeed (success is the settled conjunction); declared in `describe()`.

## Seed provenance

Seed: `libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it.py`)
— turn the flat stove's knob past a threshold (`stove_joint_state > 0.5`, a ONE-SHOT
toggle) and put the chefmate frying pan on the burner, with a moka pot as distractor.

## What changed, and why it is strategically different

The stove keeps its identity (light the burner, put the pan on it) but the ignition
mechanism is replaced wholesale: there is **no knob anywhere in the scene**. This is a
pressure-primed camp stove — the burner lights only after the fuel line is pressurized
by **N full strokes of a spring-return pump plunger, N in {2, 3, 4} sampled per
episode**. A stroke is counted with **full hysteresis**: the red pump cap must be
pressed to >= 80 % of its 35 mm travel (DOWN edge) and then released back to <= 20 %
(UP edge) before it counts. Consequences, all tested in the smoke battery:

1. **The seed's plan — one actuation + placement — is measured-insufficient**: one
   perfect press-and-release plus a perfect griddle placement leaves the stove unlit
   (N >= 2 always), success rejected (smoke check 5).
2. **Press-and-hold does nothing** (no release edge — smoke check 7): the seed's
   "push the joint past a threshold and it stays on" mental model fails.
3. **Partial actuation does nothing**: shallow jiggles that never reach full depth,
   and press cycles that never fully release, both count 0 strokes (checks 8-9).

So a solver needs a different PLAN, not different parameters: a **repeated
press-release-verify loop with feedback** (the stroke lamps on the slab display both
the requirement and progress; the burner ring glows orange once primed — priming then
latches for the episode, like a real pressurized fuel line), followed by the placement.
The plan LENGTH itself is sampled per episode (2-4 strokes), which no single-toggle
policy can absorb. Claimed strategy axes: **cyclic repeated actuation with count
accumulation (pump-priming)** + **full-stroke hysteresis gating (edge detection, not
state holding)** + mechanism-state precondition with visual gauge feedback. Sibling
differentiation: weighbridge/i6 (weight holds a plate DOWN — a press is useless there,
mass is the point; here mass is useless and repetition is the point), oven_dials/i7
(precision rotary pinch-turn), sear_and_serve/i18 and v1-i21 short_order (dwell-time
windows), bayonet_switch/i35 (single compound stroke through a maze slot). No sibling
claims counted cyclic actuation.

The frying pan becomes a **cast-iron griddle disc with a steel center-knob handle**
(the parallel-jaw affordance: 24 mm knob-post pinch, CoM directly under the pinch),
and the seed's moka-pot distractor rides along as a procedural silver cylinder that
belongs nowhere.

Physics is honest everywhere: the stroke counter reads only the live cap depth of a
real prismatic spring mechanism (the weighbridge/microwave-button pattern: joint
authored per env, pair collision disabled, symmetric limits, post_step world-frame
spring with gravity disabled on the cap); success()/score() judge settled poses and the
latched counter. smoke's teleports/pins are rubric instrumentation, not a solution; the
real solution is `solve.py`.

## Solution outline (solve.py — the feasibility certificate)

Franka base pose: **(-0.50, 0, 0), identity rotation** (= scene cfg `stage_anchor`),
OSC control, `nullspace_dof_pos=()`, gripper effort 120 N / stiffness 4000. Layout is
base-polar by design: pump cap at ~0.56 m bearing +12 deg, burner at ~0.56 m bearing
-11 deg, griddle sampled at 0.42-0.50 m bearing -45 +/- 12 deg, moka pot mirrored on
the + side.

Phases (SIM_GEN_SCORE printed at every boundary; monotone along the real trajectory):
1. PUMP — gripper CLOSED, fingertips over the live cap center; press straight down
   chasing a target 12 mm below the live cap top until the scene's own `pump_depth()`
   crosses 30 mm; lift to hover until depth <= 4 mm (spring returns the cap — one full
   stroke). Repeat, polling `scene.strokes()/primed()`, until primed (score climbs
   0.35 * strokes/N, then latches 0.45 at primed).
2. PLACE — top-down 24 mm knob-post pinch of the griddle (grip 0.007, width gates
   15-34 mm), lift-verdict (rose > 4 cm + width band), closed-loop carry on the
   GRIDDLE xy to the live burner axis at 0.22 m, lower to the burner rest height until
   the scene's own on-burner clause holds, slow release, retreat (0.85 latches, then
   1.0 as the 90-substep sustained hold completes).
3. VERIFY — park high, 3 s more simulation, success must persist; monotone score trace
   asserted.

Arm-only manipulation: no task-object state writes, no external forces on task objects.
Verified on the forge: seeds 0, 1, 2 — `SIM_GEN_SOLVE: SUCCESS` each (34-72 s wall),
n_required 4/4/3, first-attempt pumping and grasping, score prints non-decreasing
(0.000 -> 0.087/0.117 ... -> 0.450 -> 1.000).

## Randomization (per episode)

Griddle + moka pot re-sampled on polar bands (bearing +/- 12 deg, radius 0.42-0.50 m,
free yaw) and the required stroke count re-sampled in {2, 3, 4} (the stroke-lamp row
displays it visually: dark-red = needed, amber = done, gray = unused). Verified by
readback in smoke check 2-3. The stove/boss/cap mechanism is deliberately FIXED at its
authored pose: on this PhysX stack a per-episode teleport of a jointed pair is
unreliable (the joint frame stays anchored at the authored pose), so the mechanism
never moves and the free bodies + stroke requirement randomize instead.

## Rubric (score in [0, 1], latched partial credit that never evaporates)

- 0.35 * strokes/N — the stroke counter is monotone by construction.
- 0.12 — latched once the griddle was lifted clear (> 6 cm).
- 0.25 — latched once the griddle rested on the burner (primed or not — this is the
  cap the seed strategy hits).
- 0.45 — latched once primed.
- 0.85 — latched once primed AND on-burner have coexisted.
- 1.0 — iff `success()`: primed AND the griddle resting centered on the burner
  (xy within 45 mm of the axis, at rest height +/- 12 mm, upright < 10 deg, settled),
  sustained 90 consecutive substeps.

## Checks (smoke: SIM_GEN_SMOKE: ALL PASS 18/18 on the forge, first run — rejection
tests only; solve.py is the acceptance evidence; item 5 is two named checks)

1. reset settles finite, cap at home, score 0
2. randomization is real (griddle/moka pose + yaw readback across seeds)
3. required stroke count is sampled (varies across resets)
4. null policy: 0 strokes, score ~0, no success
5. **seed strategy**: ONE full press-and-release + perfect placement -> not primed,
   no success, score <= 0.30 (+ terminality: waiting never lights it)
6. near-miss: n_required-1 strokes + perfect placement -> rejected
7. press-and-HOLD 150 substeps: 0 strokes while held (needs the full release)
8. 4 press cycles releasing only to 10 mm (> the 7 mm UP edge): 0 strokes during,
   exactly 1 after the eventual full release
9. shallow jiggling (12-22 mm, never the 28 mm DOWN edge): 0 strokes
10. wrong object: primed + moka pot on the burner -> no success
11. near-miss placement: primed + griddle settled 60 mm off the axis (tol 45) -> fails
12. wrong pose: griddle held 15 deg tilted at the rest pose -> upright gate rejects
13. wrong place: primed + griddle on the ground beside the stove -> primed plateau only
14. calibration: freed cap spring-returns home (<= 4 mm)
15. calibration: 3 clean strokes count exactly 3
16. score ladder: 0 < one-stroke credit < primed plateau 0.45, latched
17. video frames captured and saved (frames.npz)

Plus the solve gate: `SIM_GEN_SOLVE: SUCCESS` on seeds 0/1/2 with non-decreasing
SIM_GEN_SCORE prints and success persisting under 3 s of extra simulation.
