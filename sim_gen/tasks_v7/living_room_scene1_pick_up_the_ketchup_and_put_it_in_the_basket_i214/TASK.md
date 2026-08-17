# sluice_hopper_catch — stage the basket, pull the sluice gate, catch the discharged ketchup

`sim_gen` task `living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i214`
— env `simgen.sluice_hopper_catch` (robot="null"; bodies driven by pose writes / wrenches).

## Seed provenance

Seed: `libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket`
(RoboVerse `roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket.py`).
The seed is a one-stage pick-and-place: grasp the visible standing ketchup bottle,
carry it over a passive open basket, release; success = ketchup CoM inside the
basket's `contain_region` bounding box. Kept from the seed: the red ketchup bottle,
a brown distractor sauce bottle, the green open basket, and "the ketchup inside the
basket" as the final containment predicate.

## What changed & why it is strategically different

**The target is never touched in the demonstrated solution.** The ketchup lies inside
an elevated, fully enclosed gravity HOPPER (tilted internal ramp, side walls, back
wall, roof) — it cannot be reached, let alone grasped. The only exit, the front
discharge lip, is sealed by a SLUICE GATE riding in vertical rails. The seed's entire
strategy — grasp the visible bottle, put it in the basket — executed here grasps the
only graspable floor bottle, the brown DECOY, and fails (smoke check 5). New
load-bearing skills:

1. **Operate an interlock mechanism**: grip the gate's yellow T-handle and pull the
   slab STRAIGHT UP out of its rail channel — a ~19 cm constrained vertical
   extraction against gravity, rail friction and the bottle's ramp-pressure on the
   gate's back face (contact physics; the gate is never pose-written while
   constrained). A 2.5 s forced shove of the bottle against the seated gate moves it
   ~1 mm and never breaches the lip (smoke check 6): the seal is real.
2. **Pre-position the catcher (perception-gated)**: the basket must be staged on the
   floor centered under the discharge lip BEFORE opening the gate — the hopper's
   side (Bernoulli left/right) and heading (±10° yaw) are randomized, so the landing
   spot must be read from the scene, not memorized.
3. **Cause-and-catch, not carry**: the delivery itself is a gravity discharge — the
   bottle rolls down the ramp, over the lip, free-falls ~8 cm and must be CAUGHT.
   Without the staged basket it lands on the open floor and rolls away ~1.5 m
   (smoke check 7) — the placement is ballistic, not quasi-static.
4. **Exclusion**: the brown bbq bottle in the basket (alone or additionally)
   forfeits success.

Distinct from the other examined tasks_v7 packages: **i164** (hang the RECEPTACLE on
a wall peg, then load it by hand — receptacle suspension, no mechanism, target is
hand-carried), **i292** (relocate a blocker, then force-roll a ball through a garage
— no sealed enclosure, no receptacle staging, the target is pushed directly), **i8**
(sliding-lid hamper: open → insert by hand → close articulation ordering), and
robobench `pen_holder` (fill a grounded cup by hand). Nothing else examined delivers
the target purely by releasing stored (gravitational) energy through a mechanism.

## Scene (procedural only)

- Kinematic HOPPER at x ≈ 0.50 ± 0.02 m, side = Bernoulli(½)·(y ≈ ±0.14 ± 0.03 m),
  yaw ≈ π ± 10°: pedestal (front face recessed 3 cm so the falling bottle clears),
  8°-pitched deck (lip at local (0.130, 0.165)), side walls (inner ±0.08 m), back
  wall, roof (top 0.282 m), and the gate rails — front bars (discharge corridor
  inner faces ±0.085 m), side stops, sill tabs. Rails span z 0.13–0.33.
- Dynamic GATE: 12 mm slab (0.20 × 0.16 m) + stem + yellow T-handle, mass 0.10 kg,
  origin/CoM at the slab centre, seated at hopper-local (0.140, 0, 0.225) with
  ~4 mm channel slack; free of the rails once its root passes local z ≈ 0.415.
  Heavy damping (a slab in a slack channel is otherwise a PhysX ring).
- Dynamic green BASKET 0.15 × 0.19 m outer, rim 0.085 m, mass 0.35 kg, origin/CoM at
  the floor bottom; spawns on the side OPPOSITE the hopper, xy jitter + free yaw.
- Two dynamic squeeze bottles (55 mm dia × 119 mm with cap): RED ketchup / white cap
  (target) LYING on the ramp at local x ∈ [−0.09, −0.03] — it rolls down against the
  gate while settling; dark BROWN bbq / black cap (decoy) standing on the open
  floor, opposite side, jittered. Low bottle damping: it must ROLL.

Randomization verified by READBACK in smoke checks 2–3 (side flips with basket and
decoy always opposite; continuous jitter in hopper x / yaw, bottle ramp position,
basket xy).

## Rubric (latched; anchored in the demonstrated solution)

Streaks (10 consecutive `post_step`s) latch `staged`/`extracted`; `released` latches
on first lip crossing:

| credit | condition |
|---|---|
| 0.15 | `staged_ever`: basket settled in the catch zone (hopper-local, within 6 cm of lip + 7 cm on the axis, upright, on the floor) |
| 0.25 | `extracted_ever`: gate clear of the opening (hopper-local z ≥ 0.35 or ≥ 0.25 m from the seat laterally) |
| 0.20 | `released_ever`: ketchup CoM past the lip plane (hopper-local x ≥ 0.15) |
| cap 0.60 | latched base |
| 1.00 | `success()` live: ketchup contained in the basket cavity (basket-frame, below rim − 10 mm) ∧ bbq NOT contained ∧ basket upright on the floor ∧ basket + ketchup still (lin < 0.05 m/s, ang < 0.7 rad/s) |

## Teleport solution (solve.py) — transport only; interactions are physics

- P0: reset, 2 s settle (the bottle rolls down the ramp against the gate — readback),
  layout readback (side, yaw, gate seat, ketchup enclosed). Score 0.000.
- P1: ONE pose write stages the basket under the lip in the hopper's CURRENT frame.
  Score 0.150.
- P2: velocity-capped vertical lift FORCE (2.2 N bang-bang at 0.30 m/s, stall
  escalation) extracts the gate up its rails — sliding contacts, growing gap, the
  bottle squeezing out under the rising slab are all contact dynamics. Only the
  measurably FREE gate (readback local z ≥ 0.46) is parked by a transport teleport.
  The discharge — roll, lip, free fall, impact, rattle-in — is untouched physics.
  Score 0.600 at the boundary, then 1.000 once settled.
- P3: ≥ 3.3 simulated seconds hands-off; success must persist. Score 1.000.

Measured on the forge: seeds 0, 2 (side = +1) and seed 1 (side = −1) all print
non-decreasing `SIM_GEN_SCORE` 0.0000 → 0.1500 → 0.6000 → 1.0000 → 1.0000 and
`SIM_GEN_SOLVE: SUCCESS` (~18 s wall each).

## Execution order

Stage-then-open is the demonstrated path and is declared in `describe()` as the
intended plan; it is NOT hard-required: if the bottle is discharged onto the open
floor, picking it up and placing it in the basket by hand is legal — the rubric
latches either way and `success()` judges only the final physical state (smoke
check 7 shows the missed discharge scores 0.45, not success, leaving the recovery
worth the remaining 0.55).

## Embodiment (single Franka, OSC, base at the origin)

- **Basket**: rim pinch (8 mm walls, 85 mm rim height, 68 mm of open interior under
  the grasp) at 0.13–0.20 m radius; carry 0.35 kg well under payload; set down at
  0.36–0.45 m radius under the lip (the lip overhang is at 0.165 m height — the
  85 mm basket passes under it with 80 mm clearance).
- **Gate**: grasp the 16 × 70 × 14 mm T-handle bar (fits the 80 mm parallel jaw)
  at ~0.36 m radius, 0.36 m height; pull STRAIGHT UP ~19 cm (weight ~1 N, rail
  friction ~1 N scale — trivial payload) to ~0.55 m height, still inside the
  workspace; set the freed gate down anywhere clear.
- **Ketchup**: never touched in the intended plan. In the floor-recovery fallback it
  is a 55 mm dia bottle lying on open floor < 80 mm jaw opening — a standard pinch.
- All grasps and goal poses lie in a 0.13–0.55 m radius annulus at 0–0.55 m height
  in front of the base: inside the Franka workspace with standard OSC gains; forces
  are ~2 N scale.

## Checks

- `solve.py`: 3 forge runs (seeds 0, 1, 2 — both hopper-side branches), each
  `SIM_GEN_SOLVE: SUCCESS` with non-decreasing scores and a 3.3 s hands-off hold.
- `smoke.py`: `SIM_GEN_SMOKE: ALL PASS 14/14` on the forge, frames.npz
  (500 × 600 × 960) recorded:
  1. reset settled, finite, gate seated, ketchup enclosed, score ≤ 0.02
  2. hopper SIDE flips across seeds; basket + decoy always opposite (readback)
  3. continuous jitter present (hopper x/yaw, bottle ramp x, basket xy) (readback)
  4. null policy 2.5 s → score ≤ 0.02, no success
  5. SEED strategy (the visible floor bottle → basket places the DECOY) → score
     ≤ 0.02, not success
  6. interlock: 2.5 s velocity-capped shove presses the bottle against the seated
     gate (probe motion asserted by readback) — no lip crossing, gate stays seated,
     no release credit
  7. the real extraction recipe WITHOUT staging → floor landing, 0.45 latch only,
     not success
  8. near-miss: ketchup settled against the basket's OUTER wall → not contained
  9. wrong place: ketchup resting on the hopper ROOF → no release credit, ~0
  10. success reproduction via the solve recipe → success True, score 1.0
  11. exclusion: decoy added INTO the loaded basket → success collapses, cap 0.60
  12. latch regression: ketchup yanked out → success collapses, 0.60 latch persists
  13. rejection audit: success never True at any rejection-battery judged point
  14. final no-NaN
