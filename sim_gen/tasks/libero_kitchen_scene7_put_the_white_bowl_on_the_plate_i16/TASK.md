# cloche_service — uncover the plate, serve the white bowl, re-seat the cloche over it (i16)

**Env name:** `simgen.cloche_service` (scene `cloche_service`, registered with robot `null`;
`solve.py` builds its own Franka binding).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate.py`)
- Seed plan: the plate is an EXPOSED passive target; grasp the white bowl, carry it level,
  set it down on top. One grasp-carry-place; success = a single on-top xy/height relation;
  a microwave stands by as a distractor.

## What changed, and why it is strategically different

Kept: a white bowl that must end up on the plate, a kitchen-counter setting, a distractor.

Changed — the PLAN, not the numbers:

1. **The target is blocked and hidden at spawn.** The plate spawns COVERED by a
   brushed-steel serving dome (a cloche with a grip knob). The seed's single move is
   physically impossible until the scene is changed: the solver must first lift the dome
   off by its knob and set it down clear on the counter.
2. **Displace-and-restore on the same fixture.** The dome is not clutter to discard — it
   must come BACK. After the bowl is served, the solver re-grasps the dome, carries it to
   the plate and lowers it straight down so its ring lands inside the plate rim, enclosing
   the bowl. The same object is manipulated twice with opposite intents (remove / precisely
   reinstall), a loop with no analogue in the seed.
3. **The goal relation is ENCLOSURE, not on-top.** Success is a nested covered assembly —
   bowl on the plate floor AND dome seated over it (enclosure additionally judged in the
   dome's body frame) — plus exclusivity (the RED distractor bowl must stay clear of the
   plate). The seed's complete goal state (white bowl on the open plate) is an explicitly
   tested, insufficient outcome: latched 0.45, never success (smoke checks 7-8).
4. **Execution order is REQUIRED and enforced by physics, not by the rubric**: the plate is
   unreachable while covered, and the dome cannot enclose a bowl that is not there yet.
   Uncover -> serve -> cover is the only executable order (reset readback, smoke checks 2/6).
5. **Perception demands**: white-vs-red bowl discrimination with per-episode slot
   permutation, plus a whole-layout mirror — a memorized motion sequence fails.

A solver that transfers the seed policy cannot even start; executing the seed's whole plan
after uncovering leaves the task at 0.45.

## Embodiment note (designed for the single Franka + parallel jaw)

- The dome is carried by a dark 18 mm square KNOB post on its top — the corpus-proven
  knob-post pinch (CoM hangs directly under the grasp; the dome swings like a stable
  pendulum). Close cmd 6 mm, gripped width ~18 mm (band 14.5-27).
- The ramekin bowls are 61 mm across (outer flat-to-flat 56 mm) — the whole bowl fits in
  the 80 mm jaw, so the grasp is a SPANNING pinch on two opposite octagon flats (the proven
  can-pinch load case; jaw snapped to the bowl yaw mod 45°). Close cmd 24 mm, gripped ~56 mm.
- Clearances: dome ring vs plate aperture ~18 mm of funnel (solve lands 1-5 mm); bowl vs
  dome cavity ~40 mm of slack; carry heights clear the parked dome's knob (bowl carried
  with its bottom 170 mm above the counter).

## Execution order

**REQUIRED — and physically enforced** (see point 4). The rubric does not need to encode
order; the smoke proves the covered start at every reset.

## Rubric (graded, latched in post_step; 1.0 iff success())

| score | state |
|-------|-------|
| 0.00  | nothing done |
| 0.20  | the plate has been UNCOVERED at least once (dome lifted/carried off) |
| 0.45  | the WHITE bowl has been on the plate floor at least once (the seed's whole goal) |
| 0.70  | covered assembly reached (bowl on plate AND dome seated enclosing it) |
| 1.00  | success(): assembly settled + red bowl clear of the plate |

Honesty: all relations judged in body frames. The dome-seated clause is honest by
construction (physical play inside the rim ~18 mm < the 20 mm tolerance, so any genuinely
seated dome counts; a dome perched on the rim or on the bowl fails the base-height band).
The bowl-centering tolerance (22 mm) is a real precision demand (physical play ~72 mm);
smoke check 12 constructs a settled ~30 mm near-miss and asserts rejection.

## Solution outline (solve.py — the feasibility certificate)

- Franka base pose: **(-0.42, 0.0, 0.20)** (mounted on the counter slab), OSC mode,
  `nullspace_dof_pos=()`, gripper 80 N / 4000 stiffness.
- Phase 1 — UNCOVER: top-down knob pinch (jaw az = dome yaw mod 90°), lift the dome so its
  ring clears everything (base to counter+140 mm), closed-loop carry on the DOME xy to the
  park spot, lower to the counter, release, retreat. Latches 0.20.
- Phase 2 — SERVE: spanning pinch across the WHITE ramekin (jaw az = bowl yaw mod 45°,
  frozen-z converging close chasing the live bowl xy), lift high (clears the parked dome's
  knob), closed-loop carry the BOWL xy onto the live plate axis, lower until the bowl
  bottom meets the plate floor, release, retreat. Latches 0.45.
- Phase 3 — COVER: re-pinch the knob, lift, closed-loop carry the DOME axis onto the live
  plate axis (gate 5 mm), lower straight down — the ring drops inside the rim around the
  bowl — release, retreat. Latches 0.70.
- Phase 4 — park the arm, settle, read success()/score(). Prints `SIM_GEN_SCORE` at every
  phase boundary (latched rubric -> non-decreasing) and `SIM_GEN_SOLVE: SUCCESS`.
- Grip-loss aborts in every carry/lower; per-skill retries; dewind guard.
- **Verified on the forge: seeds 0, 1, and 2 all reach success() — 3/3, first attempt each
  (49-78 s wall), covering both mirror sides (seed 0: side=+1, seed 1: side=-1) and both
  bowl-slot permutations (swap=0 and swap=1); the printed SIM_GEN_SCORE sequence was
  0 -> 0.20 -> 0.45 -> 1.00 (non-decreasing) in every run.**

## Scene / assets

Fully procedural, one rigid body per object (octagonal-shell compound spawner: optional
bottom disc, optional top disc + knob; explicit MassAPI, 0.5 m/s depenetration cap):
pale platter (aperture inradius 102 mm, 16 mm rim, 0.40 kg), steel dome (cavity inradius
70 mm, 100 mm tall, closed top, 18 mm knob, 0.30 kg), white + red ramekins (56 mm
flat-to-flat, 42 mm tall, 90 g). Counter = kinematic slab (top at 0.20 m). Randomized per
episode: layout mirror side, white/red slot permutation, plate(+seated dome)/bowl xy jitter,
free yaw everywhere; dome always starts seated on the plate.

## Check list (smoke.py — rejection battery, ALL PASS 17/17 on the forge, run 1)

1. settle: reset states finite (no NaN)
2. reset: dome judged SEATED over the plate (ordering is physical)
3. rubric clean at reset (score 0)
4. null policy: score ~0, no success after 200 idle steps
5. randomization: layouts differ across resets incl. mirror side (readback)
6. randomization: both bowl-slot permutations sampled; dome seated at every reset
7. SEED STRATEGY (white bowl centred on the open plate): success() False
8. SEED STRATEGY: score pinned at the 0.45 rung
9. undo (dome re-seated on the EMPTY plate): no success, score 0.20
10. wrong object (RED bowl served under the dome): no success, score 0.20
11. wrong place (bowl+dome assembled on the COUNTER): no success, score 0.20
12. near miss (covered assembly, bowl ~30 mm off the plate axis): no success
13. tolerance twin (bowl ~10 mm off-axis, covered): success True, score 1.0
14. exclusivity (red bowl 155 mm from the plate axis): no success, score capped 0.70
15. clearing the red bowl completes: success True, score 1.0
16. seat funnel: dome dropped from 25 mm above the seat lands seated 2/2
17. finite: all states finite at the end

Teleports in the smoke are instrumentation only (constructed settled states, real physics
steps between authoring and judging); the positive proof of feasibility is `solve.py`'s
arm-only trajectory.

## Sibling differentiation

Same seed family, other batches: `dish_rack` (old i16; reorientation — plate slotted
vertically, vessels flipped upside-down) and `plated_meal` (v2 i2; aggregation of food into
a carried container + exclusivity). This task claims the opposite axes: **blocked/hidden
target + displace-and-restore of the same fixture + enclosure goal + physically-forced
execution order**. Nothing is reoriented (everything stays upright), nothing is aggregated,
nothing is poured.
