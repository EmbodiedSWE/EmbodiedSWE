# dish_rack — load the drying rack (plate vertical, bowl & cup upside-down)

**Env name:** `simgen.dish_rack` (scene `dish_rack`, robot `null`)

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate.py`)
- Seed plan: grasp the white bowl, keep it LEVEL, set it down ON the flat plate.
  One upright pick-and-place; success = an on-top xy/height relation; the plate is a
  passive target that never moves; a microwave is a distractor.

## What changed, and why it is strategically different

Kept: the white bowl and the plate as the central objects, a kitchen-counter setting.

Changed — the PLAN, not the numbers:

1. **Role swap.** The plate stops being the target and becomes a manipulated object.
   The target is a new fixture, a dish-drying rack (kinematic compound body: base slab,
   two slot rails with end stops, two drying pegs).
2. **Reorientation is the core skill.** The seed's plan is level carriage with the pose
   kept upright throughout. Here NO dish may stay in its spawn orientation: the plate
   must be rotated 90° (flat → edge-on) and inserted VERTICALLY between the slot rails;
   the bowl and a cup must each be flipped a full 180° and set OPENING-DOWN, capped over
   their drying pegs (bowl → thick peg, cup → thin peg), rims resting on the rack base.
3. **Relation change.** The seed's on-top stacking relation is replaced by slotted
   insertion (the plate is *held by* the rails — verticality is honest by construction:
   the gap/rail geometry bounds a slotted plate's lean to ~13°, judged with 25° margin)
   and capping (the peg is *inside* the inverted vessel's cavity — the xy tolerance is
   inner-radius − peg-radius, so any counted pose genuinely encloses the peg).
4. **The seed's own goal state is a tested failing control.** Setting the upright white
   bowl on the flat plate on the counter — a perfect execution of the seed task,
   sanity-asserted as a genuine seed-success in the smoke — scores ~0 here (negative A).
   The seed's on-top relation aimed at the rack (plate laid FLAT across the rail tops)
   is also a tested failing control (negative B).

A solver needs a different plan: pick per-item goal orientations that differ from the
spawn orientations by 90°/180°, sequence three placements onto one shared fixture, and
perform a precision edge-on insertion — none of which appears in the seed.

## Difficulty / stages / order

- **Tier: medium — 3 stages** (three placements, two distinct reorientation skills):
  (1) rotate the plate to vertical and slot it between the rails;
  (2) flip the bowl and cap it over the thick peg;
  (3) flip the cup and cap it over the thin peg.
- **Execution order: NOT required.** Any placement order reaches 1.0 (the smoke's
  item-by-item pass uses plate → bowl → cup; the oracle uses the same order, but
  nothing in the rubric encodes order).

## Rubric (score in [0,1])

- 0.1 latched once ANY dish has ever been lifted off the counter
  (transient-achievement latch, updated in `post_step`)
- +0.26 per correctly racked item (each clause: correct pose in the RACK BODY FRAME +
  settled), i.e. 0.36 / 0.62 with the latch for 1 / 2 items
- 1.0 **iff** `success()`: all three racked and settled. Null policy scores exactly 0.

Judging is final-state, physical, and computed in the rack's body frame (the rack pose
is randomized with free yaw, so memorized world-frame targets fail).

## Randomization

Rack xy + free yaw; per-dish xy jitter + free yaw; Bernoulli mirror of the whole dish
layout across the counter midline every reset. Verified by state readback in the smoke.

## Smoke checks (16)

1. reset settles finite with score 0
2. randomization is real (rack + plate move, rack yaw varies)
3. mirror randomization flips the dish layout side
4. null policy scores ~0 and no success
5. lift latch pays 0.1 after first lift, before any racking (oracle seed 0)
6. oracle solve reaches success() on seed 0
7. oracle solve reaches success() on seed 1
8. oracle solve reaches success() on seed 2
9. rubric strictly increases item-by-item
10. item-by-item loadout reaches 1.0 + success
11. negative A: seed strategy (upright bowl on the flat plate) scores ~0
    (with a sanity leg asserting the seed's own On(bowl, plate) relation IS met)
12. negative B: plate laid flat across the rail tops is not racked
13. near-miss: upright bowl on the peg (right place, wrong orientation) is not counted
14. near-miss: inverted bowl beside the peg (45 mm off — rim lands on the peg) is not counted
15. calibration: inverted-bowl capping offsets ≤ 20 mm all cap, 45 mm never does
16. calibration: plate released at 0/5° lean racks with settled lean ≤ 20°
    (the slot genuinely constrains the plate)

Oracle: per-dish gravity-compensated kinematic hold (teleport-oracle) — lift, reorient
in place, translate, lower, release; the final seating (plate leaning onto a rail,
vessels dropping over their pegs) is real physics, and all judging is on settled
physical state. Video recorded to `frames.npz` in the working directory.
