# fold_collar — wrap a hinged three-leaf screen around the blue column

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet`
— pull open the top drawer of a fixed kitchen cabinet; success is a prismatic
joint position crossing a threshold.

## What changed, and why it is strategically different

The seed's plan is: approach a **fixed, anchored** piece of furniture, grasp
one handle, and translate a **single prismatic DOF** past a threshold. Nothing
moves except the drawer along its rail.

This task inverts every load-bearing element of that plan:

- The articulated object is **free-standing and mobile**, not anchored. The
  robot must first TRANSPORT the whole three-leaf hinged chain (heavy base
  leaf + two wing leaves on vertical revolute hinges) across the floor to the
  target, keeping it upright and keeping the wings from flopping.
- The articulation is **two revolute DOFs on a free body**, not one prismatic
  DOF on a fixture, and the goal is not "a joint crosses a threshold" but a
  **topological** condition: fold both wings ≥ ~110° so the two amber free
  edges close behind the BLUE column with a gap of at most `gap_max = 0.045 m`
  — strictly less than the column diameter (0.06 m) — laterally **enclosing**
  the column inside the triangular pocket.
- There is a **discrimination** requirement (BLUE target column vs RED decoy,
  slots swapped and jittered per seed); the seed has none.
- Judging is a compound geometric/topological predicate (point-in-quad
  even-odd test + edge gap + uprightness + standing + settled), not a joint
  readout.

No surveyed corpus task manipulates a free articulated chain, and none closes
a loop around a target by folding. The nearest neighbors differ in plan:
lid/door/grill tasks articulate a joint that is anchored to the world;
transport tasks move rigid bodies; here the *same object* must be transported
AND articulated, and the articulation creates an enclosure rather than
covering or opening anything.

## Teleport solution (solve.py phases)

Teleports are used for transport only; every fold is driven through contact
dynamics (external torques on the wing rigid bodies, resisted by ground
friction, hinge limits, and inter-leaf collision).

- **P0 — settle & readback** (score ~0): 150 settle steps; read collar pose,
  wing fold angles (must be in the zigzag spawn window 2°–34°), blue/red slot
  positions; assert not success and score ≤ 0.03.
- **P1 — transport (teleport)** (score ≥ 0.15): one rigid teleport of the
  whole chain, joint-consistent (wing poses recomputed from the base pose and
  the *current* fold angles), parking the base flush against the BLUE column
  at stand-off `d_park = col_r + panel_t/2 + 8 mm`, pocket facing the column.
  Settle 90 steps; approach latch fires.
- **P2 — fold both wings (dynamics)** (score ≥ 0.70): simultaneous per-wing
  PD velocity-servo using pure world-z torques
  (`set_external_force_and_torque`, `is_global=True`), torque cap 0.15 N·m
  escalating to 0.45 N·m on stall, driving each fold to 112°. Hinge reactions
  on the base cancel by symmetry; the 1.6 kg high-friction base stays parked.
- **P3 — gap trim (dynamics)**: up to 3 rounds deepening the fold target by
  +5° until gap < `gap_max` and success() is True (score 1.0).
- **P4 — persistence**: ≥ 3.3 sim-seconds hands-off; success must persist;
  then `SIM_GEN_SOLVE: SUCCESS`, watchdog + `os._exit`.

## Execution-order declaration

**Transport MUST precede closure, and this order is physically enforced, not
scripted**: the success gap bound (0.045 m) is smaller than the column
diameter (0.06 m), so a chain folded first can never admit the column
laterally afterwards — the only way the column gets inside the pocket is to
bring the still-open chain to the column and then fold. (Smoke check 10
verifies the converse: folding closed at the spawn location scores ~0.)
The two wings may be folded in either order or simultaneously; the solve
folds them simultaneously so hinge reaction forces on the base cancel.

## Embodiment argument (single Franka, parallel-jaw gripper)

- **Base leaf transport**: the base leaf is 12 mm thick and 120 mm tall — a
  parallel-jaw gripper pinches its top edge (a standard door/board grasp) and
  drags/carries the 1.6 kg chain across the table. Wing angular damping (4.0)
  keeps the wings from swinging wildly during the carry, as verified by smoke
  check 6 (hinges hold within 12° over 2.5 s hands-off).
- **Folding the wings**: each wing is folded door-style by pushing on its
  outer face near the amber free-edge rail — a flat-palm/closed-gripper push,
  exactly how a person closes a folding screen. Required torque ≤ 0.45 N·m at
  a ~0.15 m lever arm is ~3 N of push force, trivial for a Franka.
- **Tolerances**: the terminal gap tolerance is 4.5 cm — generous relative to
  Franka repeatability (~0.1 mm); the fold target has a ~20° margin below the
  130° hinge limit.
- **Base pose**: all action happens inside a 0.7 m × 0.5 m patch (collar
  spawn near (0.14, 0), columns near (0.52, ±0.16)); a Franka based at
  ~(−0.35, 0) reaches everything within a 0.9 m radius.

## Rubric

Latched, calm-gated partial credit; monotone non-decreasing:
- 0.15 `w_appr` — chain brought within `approach_r` of the BLUE column
  (un-gated on calm so a carried chain counts; spawn-distance honesty is
  asserted in `__post_init__`: worst-case spawn separation 0.202 m > 0.18 m).
- +0.25 `w_half` — calm, base within `near_r` of blue, at least one wing
  folded past 90°.
- +0.30 `w_fold` — calm, blue column inside the pocket quad, both wings
  past 90°. Partial cap 0.70.
- 1.0 iff `success()`: blue inside quad ∧ gap < 0.045 ∧ all three leaves
  upright (≤15° tilt) ∧ standing on the ground ∧ settled ∧ finite.

## Smoke rejection battery (14 checks)

1. Settle / no-NaN; spawn folds in zigzag window; upright; score ≤ 0.01.
2. Randomization READBACK, seeds 101 vs 202: collar xy/yaw, fold angles, and
   blue column xy all differ measurably.
3. Slot swap: seeds 300–309 place blue in both slots.
4. Null policy, 240 idle steps: score ≤ 0.05.
5. **Seed-style negative (rigid transport, no articulation)** — the seed's
   drawer has no analog here, so the seed-strategy analog is "move something
   rigidly": the still-open collar is parked flush at blue with one rigid
   displacement. Column not inside; score ≤ 0.15.
6. Mechanism check: wings posed at 60° hold within 12° over 2.5 s hands-off.
7. Wrong object: chain wrapped (114°/114°) around the RED decoy — folds hold,
   red is inside, blue is not; score ≤ 0.15.
8. Near-miss gap: wrapped at 95°/95° — blue inside, but gap > gap_max; no
   success; score ≤ 0.70.
9. Near-miss one wing (112°/12°): score ≤ 0.40.
10. Fold-in-place at spawn (114°/114°, no transport): closed gap but score
    ≤ 0.01 — proves the physical ordering claim.
11. Tipped chain (rigid-rotated 90° onto the floor): finite, not upright, no
    success.
12. Rejection audit: success() never fired during checks 4–11.
13. Constructed-success sanity (audit closed): wrapped at blue 113°/113° →
    success True, score 1.0.
14. Video frames captured to `frames.npz` (> 10 frames).
