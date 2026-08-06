# plated_meal — gather all the food into ONE bowl, then present it on the plate (i2)

**Env name:** `simgen.plated_meal` (scene `plated_meal`, registered with robot `null`;
`solve.py` builds its own Franka binding).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate.py`)
- Seed plan: among three identical black bowls, select ONE by position ("the middle"),
  grasp it, carry it, set it down on the plate. Success = a single geometric relation
  (bowl xy within 6 cm of the plate, 0-3 cm above it), reached by one grasp-carry-place.

## What changed, and why it is strategically different

Same object vocabulary (three identical dark bowls + one plate, kitchen counter), but the
GOAL TYPE and the PLAN change:

1. **The deliverable is a CONSTRUCTED ASSEMBLY, not a spatial relation.** 2-3 bright food
   cubes (count sampled per episode) lie scattered on the counter. Success = every present
   cube rests INSIDE one bowl, that bowl stands centred on the plate floor, BOTH other
   bowls are kept clear of the plate, everything settled. The solver must manipulate two
   object classes with two different grasp strategies (bail-handle pinch for the open
   vessel, yaw-aligned face pinch for the cubes) and aggregate N loose items, not relocate
   one container.
2. **The seed's complete goal state is a tested, explicitly insufficient outcome.** A bare
   bowl teleported perfectly onto the plate centre latches the first rubric rung (0.15)
   and can never succeed — smoke check 4. Executing the seed's whole plan here leaves 85%
   of the task undone.
3. **Selection is inverted.** The seed keys success to one specific bowl chosen by
   position; here the bowls are interchangeable (any may serve) but the assembly and an
   EXCLUSIVITY clause must hold: food split across bowls, food lying on the plate outside
   the bowl, or a second bowl on the plate all fail (smoke checks 5-8).
4. **Count perception.** The food count (2 or 3) and a slot permutation are resampled per
   episode, with a whole-layout mirror (the plate side flips) — a memorized fixed motion
   sequence fails; the solver must count what it sees and finish only when ALL of it is
   served.

A solver that transfers the seed policy (grasp container -> place on target) scores 0.15;
the correct plan needs the aggregation loop and the exclusivity discipline on top.

## Embodiment note (designed for the single Franka + parallel jaw)

The bowls are 123 mm across — wider than the 80 mm jaw — so each carries a **bail handle**
(two rim posts + a 12 mm crossbar arching 35 mm over the mouth centre). Pinching the
crossbar's middle puts the grasp directly above the CoM: the bowl hangs level, the
corpus-proven can/cube pinch load case. Food is dropped through the two ~31 mm openings
beside the crossbar. This affordance was converged on empirically: wall pinches, inside
expansion grips, and rim-side tabs (flat and T-capped) all measurably fail on a
free-sliding 150 g bowl (tangential squirt / runaway / torque walk-off; see the solve run
history), while the bail grasp plated the bowl on the first attempt of every seed tested.

## Execution order

**NOT required.** Fill-then-carry and place-then-fill are both legal; the rubric judges
physical end states plus latched progress. (The reference solve places the empty bowl
first, then loads it.)

## Rubric (graded, latched in post_step; 1.0 iff success())

| score | state |
|-------|-------|
| 0.00  | nothing done |
| 0.15  | a bowl has been ON the plate at least once (the seed's whole goal = first rung) |
| 0.25  | a food cube has been INSIDE a bowl at least once |
| 0.50  | all present food gathered in ONE bowl at the same time |
| 0.75  | full assembly reached (all food in a bowl that is on the plate) |
| 1.00  | success(): assembly settled + both other bowls clear of the plate |

Honesty: all containment judged in body frames. A cube physically inside a bowl is at
most ~27 mm off-axis < food_xy_tol 30 mm (anything genuinely inside counts); a cube on a
bowl rim (~44 mm, above the depth gate) and a bowl resting on the plate rim (bottom-height
band) are rejected. The bowl-centering tolerance (36 mm) is a real precision demand: the
plate aperture leaves ~41-48 mm of physical play, so an uncentered set-down is a
constructible near-miss (smoke check 9 measures 40 mm settled and rejected).

## Solution outline (solve.py — the feasibility certificate)

- Franka base pose: **(-0.42, 0.0, 0.20)** (mounted on the counter slab), OSC mode,
  `nullspace_dof_pos=()`, gripper 80 N / 4000 stiffness.
- Phase 1 — serve the bowl: pick the most reachable bowl; jaw azimuth = crossbar heading
  + 90° wrapped mod 180° (wrist-limit safe); pre-narrowed 26 mm descent beside the bar;
  converging close chasing the bar's live XY with FROZEN z (live-z tracking is positive
  feedback and hoists the bowl); lift; closed-loop carry on the BOWL xy onto the live
  plate axis; lower until the bowl bottom meets the plate floor; release; retreat.
- Phase 2 — load the food, nearest cube first: top-down pinch with the jaw snapped to the
  cube's yaw (mod 90°; unaligned jaws meet edges and flick the cube — measured); carry
  over the handle (crossbar top + clearance); drop through the mouth opening beside the
  bar (22 mm perpendicular offset, jaw re-aligned along the bar); verify via the scene's
  own food_in_bowl readback; per-cube retries with grip-loss aborts.
- Phase 3 — park the arm, settle, read success()/score().
- Prints `SIM_GEN_SCORE` at every phase boundary (latched rubric -> non-decreasing:
  0 -> 0.15 -> 0.25 -> ... -> 1.0 observed on every passing run) and
  `SIM_GEN_SOLVE: SUCCESS`; hard exit behind a watchdog Timer.
- **Verified on the forge: seeds 0, 1 (mirrored layout), and 2 (three cubes) all reach
  success() — 3/3.**

## Scene / assets

Fully procedural, one rigid body per object: a shared octagonal-vessel compound spawner
(bottom disc + 8 wall boxes; explicit MassAPI, small contact offsets, 0.5 m/s
depenetration cap) builds the 3 bowls (cavity inradius 38 mm, depth 65 mm, bail handle)
and the plate (aperture inradius 95 mm, 20 mm rim); food = 30 mm colored cubes
(red/yellow/green, 40 g, zero restitution). Counter = kinematic slab (top at 0.20 m).
Randomized per episode: layout mirror side, plate/bowl/cube xy jitter + free yaw, food
count 2-3 + slot permutation; absent cubes park in an off-counter ground depot.

## Check list (smoke.py — rejection battery, ALL PASS 18/18 on the forge, run 1)

1. settle: reset states finite (no NaN)
2. rubric clean at reset (score 0)
3. null policy: score ~0, no success after 200 idle steps
4. randomization: layouts differ across resets incl. mirror side (readback)
5. randomization: both food counts {2,3} sampled (readback)
6. randomization: present cubes on counter slots, absent cubes parked (readback)
7. SEED STRATEGY (bare bowl centred on plate): success() False
8. SEED STRATEGY: score pinned at the 0.15 first rung
9. wrong place (all food in a bowl on the COUNTER): no success, score 0.50
10. wrong vessel (food directly on the plate, no bowl): no success, score ~0
11. exclusivity (2nd bowl on the plate): no success, score capped 0.75
12. clearing the distractor completes: success True, score 1.0
13. leftover (one cube on the plate beside the bowl): no success, score capped 0.25
14. near miss (assembly 40 mm off the plate axis, physically on the plate): no success
15. tolerance twin (assembly 25 mm off-axis): success True
16. drop funnel: the solve's drop point (beside the bar) lands in-bowl 2/2
17. drop funnel: a 60 mm-offset drop (over the wall) never counts
18. finite: all states finite at the end

Teleports in the smoke are instrumentation only (constructed settled states, real physics
steps between authoring and judging); the positive proof of feasibility is `solve.py`'s
arm-only trajectory.

## Sibling differentiation

Old-batch tasks on this seed family: `serve_ingredient` (old i2; pour the CONTENTS out of
a cup — extraction) and `dish_rack` (i16; reorientation into a rack). This task claims the
opposite axes: **aggregation INTO the manipulated container + composite nested goal
(food ∈ bowl ∈ plate) + exclusive presentation**; nothing is poured, nothing is
reoriented, and the container (not the contents) is the carried object. No pour, no
aperture keying, no dwell timing, no hidden state.
