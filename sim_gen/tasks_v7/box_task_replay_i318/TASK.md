# weave_close_crate (box_task_replay_i318)

Pack two loose items into an open-topped crate, then close its two-flap
articulated lid in the one order the geometry admits: the short **tuck** flap
seats flat on its 0-degree stop first, and the long **main** flap swings over
the top and lands ON the seated tuck flap — the weave. Success is the current
physical state: crate upright and still, both items contained and settled,
both flaps in their closed bands, and the main plate measurably ABOVE the tuck
plate at the overlap (`weave_delta > 6 mm`).

## Seed provenance

Seed: `box_task/box_task_replay`
(`sim_gen/RoboVerse/roboverse_pack/tasks/box_task/box_task_replay.py`) — a
replayed pick-and-place that moves a box between poses; the manipulated object
is a rigid box, the goal is a pose match, and nothing articulates.

## Strategic difference

- **vs the seed**: the seed is single-object pose-matching by trajectory
  replay. Here the crate is a five-body articulated assembly (base + two
  hinged flaps on spawn-authored revolute joints), the goal is a *mechanism
  state* (a woven two-flap closure plus containment), not an object pose, and
  the load-bearing work is done by bounded hinge torques, not by carrying the
  goal object.
- **vs sibling `box_task_replay_i303` (carton flip-pack)**: i303 rights a
  fallen carton by rolling it with a torque plant and drops items into it —
  one free body plus items, no joints, and its lid never exists. i318's crate
  never moves; the whole task is the two-DOF lid linkage and its forced
  closing ORDER, which i303 has no analogue of.
- **vs `open_grill_i260` (serving shelf)**: i260 OPENS a single hinged hood to
  expose a surface and places items on top. i318 CLOSES two interleaving
  flaps over packed contents; the interesting physics is the flap-on-flap
  landing and the order interlock, the opposite motion and a two-joint
  coordination problem.

## Order enforcement (why the weave is forced)

- Tuck-first is stable: the tuck seats flat on its 0-degree stop; the closing
  main's tip then lands on the tuck plate and rests there (+3.3 deg,
  `weave_delta ≈ +14.5 mm`). The tuck's upper stop carries the main's weight,
  so the main cannot depress or undercut it.
- Main-first is rejected: a first-closed main parks on its +2.5-degree dip
  stop; a subsequently driven tuck rides ON the main plate, cocked ~13 deg —
  outside the +7-degree closed tolerance — with INVERTED (negative)
  `weave_delta`. Even a sustained press at the full working torque cap cannot
  seat it. The smoke battery proves both branches.

## Solution phases (solve.py, teleport = transport only)

0. reset(seed), settle; assert flaps at open rest (±170 deg), items standing,
   score ~0.
1. PACK can: teleport to hover ABOVE the crate mouth (crate-frame x −5 cm),
   release; falls in through the mouth and settles by contact — score 0.20.
2. PACK candle: same at x +5 cm — score 0.40.
3. TUCK: gravity-feedforward rate servo about the hinge (torque through the
   scene's `tuck_tau` plant buffer, cap 0.10 N·m); flap swings up from open
   rest, over vertical, lowers rate-limited onto its 0-degree stop; `tuck_set`
   latch — score 0.60.
4. MAIN: same servo (`main_tau`, cap 0.55 N·m); tip lands ON the seated tuck
   flap; `lid_set` + success — score 1.00.
5. PERSIST: ≥3.5 s of pure simulation with success at every poll →
   `SIM_GEN_SOLVE: SUCCESS`.

The flaps are never teleported; every packing impact, flap seating, and the
weave landing is resolved by contact dynamics. Verified on seeds 0 and 3.

## Execution order

Declared order: pack both items, then tuck flap, then main flap.
- tuck-before-main is REQUIRED and geometrically enforced (see above; smoke
  check `wrong_order_main_first` proves the reverse cannot reach the goal
  state).
- items-before-flaps is the declared order and the only one a real robot
  could follow to the goal (a closed lid blocks the mouth), but the rubric
  does not need to trap it: re-opening the main flap to insert an item
  revokes success (current-state) and the latch checks gate `lid_set` on both
  items already being inside.

## Embodiment argument

A single Franka arm with a parallel-jaw gripper, base at crate-frame
(0.0, −0.40) m, does every phase:

- **Reach**: crate xy jitter ±5 cm and item spawn radius ≤0.25 m keep every
  grasp/push point within ~0.66 m of the base (smoke asserts <0.75 m across
  seeds) at heights 0–0.30 m — inside the Franka's ~0.85 m envelope.
- **Pack**: the can is a Ø56 mm cylinder (fits the 80 mm jaw opening across
  the barrel); the candle is a 48 mm square prism (grasp across faces). Each
  is lifted from open floor, carried over the mouth, and released ~2 cm above
  the rim — exactly the transport the teleports stand in for.
- **Close**: each flap is flipped shut by a fingertip push on the outboard
  face of its plate, following the swing arc — the applied hinge torques are
  bounded by what such a push produces (tuck: 0.10 N·m over a 62 mm lever
  ≈ 1.6 N fingertip force; main: 0.55 N·m over 155 mm ≈ 3.5 N). The open
  flaps rest OUTSIDE the crate footprint at ±170 deg, so the approach is
  unobstructed, and the smoke reach check covers the flap tips.

## Rubric

`score() = 1.0` iff `success()` (current state), else `0.2 ×`
(`packed_can + packed_candle + tuck_set + lid_set`) — streak-gated (30
substeps) latched progress credit, so a disturbed final state falls back to
0.80, and partial progress is monotone: 0 → 0.2 → 0.4 → 0.6 → 1.0.

## Smoke battery (14 named checks)

1. `settle_open_rest` — reset settles with flaps at open rest, zero score.
2. `randomization_readback` — crate xy/yaw and item polar radius vary across
   seeds (readback, ≥3 distinct values each; both side-swaps seen).
3. `reach_envelope` — all interaction points <0.75 m from the declared base.
4. `null_policy` — 2.5 s untouched: score <0.02, no latch creep.
5. `sub_threshold_lift` — 61% of the tuck's gravity feedforward cannot move
   it off the open rest (non-vacuous force floor).
6. `near_miss_beside` — item against the OUTSIDE wall: no containment credit.
7. `near_miss_stacked` — candle perched on the in-crate can, above the
   containment ceiling: credit stays 0.20.
8. `seed_strategy_rejected` — the seed's endpoint (items placed, lid
   untouched): success False, score exactly 0.40.
9. `wrong_order_main_first` — main closed first parks on its dip stop but is
   NOT success; the tuck then stalls cocked (~13 deg > +7 tol) with inverted
   weave even under a hard press at full cap (press asserted to have MOVED
   the flap — non-vacuous).
10. `oracle_ladder` — full correct sequence through the same torque buffers:
    0 → 0.4 → 0.6 → 1.0.
11. `score_monotone` — the oracle trace never decreases.
12. `persistence` — success holds 2 s untouched.
13. `latch_fallback` — driving the main back open revokes success; score
    falls back to exactly 0.80 (latches retained).
14. `frames_recorded` — ≥20 rgb frames written to `frames.npz`.
