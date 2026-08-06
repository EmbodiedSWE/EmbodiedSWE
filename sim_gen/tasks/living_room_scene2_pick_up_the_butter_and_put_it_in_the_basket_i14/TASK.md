# butter_hopper — stage the basket, then dispense the butter into it

**Package:** `sim_gen/tasks_v2/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i14`
**Scene:** `simgen.butter_hopper` (robobench, procedural geometry only)

## Seed provenance

Seed: `libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket.py`).
The seed is a plain pick-and-place among distractors: grasp the butter off the table,
carry it, drop it into an open basket; success = butter inside the basket's
contain-region box.

## What changed, and why it is strategically different

The terminal relation is kept (butter resting inside the basket) but the seed's PLAN is
made physically impossible and every manipulation role is inverted:

1. **The butter can never be grasped.** It is a 96 x 96 x 36 mm slab — wider than the
   Franka's 80 mm jaw aperture in BOTH horizontal directions, and its 36 mm dimension
   is unreachable while it lies on a support. The seed's core skill (pick up the
   butter) is off the table; a solver transplanting the seed's plan gets nowhere.
2. **The butter starts confined in a dispenser.** It rests on the sliding tray that
   forms the only floor of an elevated, capped hopper shaft (bottom at z 0.35). The
   only way to move it is to actuate the mechanism: pull the tray out by its yellow
   handle fin and let the shaft wall scrape the slab off the receding tray — gravity,
   not the gripper, transports the target.
3. **The CONTAINER is the manipulated object.** The basket must be brought onto the
   green drop-zone square under the shaft; the seed never moves its basket.
4. **Execution order is REQUIRED and enforced twice.** Physically: a slab dispensed on
   the bare floor cannot be picked or pushed over the basket's 10 mm lip. In the
   rubric: a `_floored` latch (butter center at bare-floor height while not contained)
   permanently voids success — smoke found the one remaining hole (lowering the basket
   over the floored slab traps it inside) and the latch closes it, so the wrong order
   is irreversible, not just narrated as such.

A solver therefore needs a different plan (stage container -> actuate mechanism ->
verify), not different parameters.

## Solution outline (solve.py — the feasibility certificate)

Single Franka, OSC, base at **(-0.50, 0, 0)**, ground level (recorded in solve.py as
`BASE_POS`; the scene cfg's `stage_anchor`).

1. SETTLE, read the scene.
2. GRASP the basket by the rim of its most robot-facing wall (12 mm wall; cage 26 mm,
   quasi-static squeeze to 6 mm; width-band verdict).
3. DRAG the basket along the floor (never lifted — the ground carries the weight; a
   lifted rim-pinch carry pendulum-slipped on some spawns) to a staging spot 65 mm out
   from the shaft axis ALONG the live face normal of the wall to be pushed; slow
   release. Mid-drag loss is detected (jaw width + hand-basket distance) and recovered
   by regrasping at the basket's resting pose (up to 3 cycles).
4. PUSH the basket the last 65 mm onto the drop zone: fingertips low (tip z 0.045,
   the wrist stays far under the tray plane), pushing PERPENDICULAR to the chosen wall
   along a world-anchored tip rail that creeps at ~22 mm/s (a face-tracking goal let
   the basket coast 30+ mm past the target), with tangential steering; up to 3 rounds.
   If staging fails, the solver ABORTS without pulling (the drop is irreversible).
5. PULL the tray: pinch the handle fin across its 48 mm WIDTH (jaw axis parallel to
   the dispenser face — a jaw across the 12 mm thickness cannot descend beside the
   shaft wall/lid, measured), descend on a frozen fin snapshot (live re-chasing
   ratchets the tray), friction-pull the slide out to >= 165 mm. The butter falls
   ~0.32 m and settles in the basket.
6. VERIFY via the scene's own sustained-success counter, then 3 s persistence.

`SIM_GEN_SCORE` is printed at every phase boundary and never decreases along this
trajectory (all credit below 1.0 is latched).

## Rubric

- `success()`: butter contained in the basket (basket-frame xy within interior minus
  10 mm margin, center z in (4, 80) mm — the rim is at 105 mm, so a slab perched on the
  rim is rejected), butter and basket settled, basket upright (<= 20 deg), tray opening
  >= 140 mm, butter never floored — sustained 60 consecutive substeps. The tray-open
  clause rejects the seed-strategy end state (butter "already in the basket" with the
  dispenser sealed); the floor latch rejects the out-of-order recovery.
- `score()`: 0.30 latched for staging the basket on the drop zone; +0.30 scaled by the
  latched max tray opening (capped at the threshold); 0.85 latched once the butter has
  settled inside the basket without ever flooring; 1.0 iff success. Null policy
  scores 0.

## Execution order

REQUIRED: basket staged under the shaft BEFORE the tray is pulled. Enforced physically
(ungraspable slab, unpushable over the basket lip) and by the `_floored` rubric latch.
Declared here; rejected by smoke checks 7-8.

## Smoke battery (rejection-only; no probe ever reaches success — checked globally)

1. settle/no-NaN (butter seated in shaft, tray closed, score 0)
2. score ~0 at reset, no success
3. randomization is real (READBACK: basket pose + butter jitter across 4 seeds)
4. null policy scores ~0
5. seed-strategy end state (butter straight into basket, dispenser sealed) -> rejected
6. near-miss: tray 20 mm short of the open threshold -> rejected
7. wrong order A: dispense before staging (butter on bare floor) -> rejected
8. wrong order B: the out-of-order END state — basket then staged, capturing the
   floored slab inside it -> still rejected (floor latch)
9. tray honesty: released tray stays open (no hidden restoring force)
10. near-miss: butter perched on the rim topples outside -> not contained, rejected
11. flipped basket with butter on its upturned bottom -> rejected (upright clause)
12. staged partials monotone (0 < staged 0.30 < staged+part-open 0.45 < 1), never
    success
13. global never-success flag across the whole battery

## Status (forge, pod 2605:340:cd51:7700:d8b3:52a8:172:2fac)

- solve: `SIM_GEN_SOLVE: SUCCESS` on seeds **0, 1, 2, 3** (final scene re-verified on
  seeds 0 and 2 after the floor-latch rubric change; ~35-73 s sim each, monotone score
  prints 0.00 -> 0.31/0.36 -> 1.00)
- smoke: `SIM_GEN_SMOKE: ALL PASS 13/13`, frames.npz (428, 600, 960, 3) recorded
