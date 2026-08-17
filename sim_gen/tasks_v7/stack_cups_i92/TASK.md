# stack_cups_i92 — shell covers: cap each die with its color-matched upside-down cup

Scene `cup_shells`, env `simgen.cup_shells` (robot="null").

## Seed provenance

Derived from **rlbench/stack_cups**: three cup meshes and a franka, no checker; the
implied strategy is transport + **NESTING** — pick each cup up rim-UP and lower it
INSIDE the target cup, success being cup-in-cup containment.

## Strategic difference

No cup ever goes into or onto another cup, and rim-up containment counts for
NOTHING. Three open cups (red/green/blue) stand rim-UP in one row; three dice (small
cubes in the same colors) sit in another row. The goal state is the shell-game end
state: every cup REORIENTED 180° (rim-DOWN), sealed flat on the floor, with its
SAME-COLORED die on the floor underneath it.

A solver needs a different plan and different code, not new constants:

- a **mandatory 180° reorientation** of every manipulated object (the seed never
  reorients anything);
- targets are **separate objects of a different family** (dice), matched by a
  per-episode **color correspondence** that must be read from the scene (the seed
  has a fixed nesting order onto one target cup);
- the placement is an **enclosure-over-target**: lower the inverted cup so the die
  passes through the open mouth and the rim seals on the floor around it —
  cup-over-die, the geometric opposite of the seed's cup-in-cup;
- the rubric is a per-pair predicate (inverted axis + rim-seal height + die-under-
  axis in the cup body frame + die-on-floor + sustained stillness), not any nesting
  test. The cups physically **cannot nest** (outer radius 32 mm > inner radius
  28 mm — asserted), so the seed's outcome is not even constructible here.

Also strategically different from the constructed tasks read while building it:
`packing/pen_holder` (insert pens tip-up INTO an upright cup — rim-up containment,
no reorientation of the container) and `put_umbrella_in_umbrella_stand_i72` (hook a
crook over a rail — suspension equilibrium, single object, no correspondence).

## Execution order

None required — the three colors may be capped in any order (declared in
describe()); solve.py happens to go red, green, blue.

## Teleport-solution outline (solve.py — teleports for transport only)

- **P0** settle + layout readback; assert score ≈ 0 (rubric-leak guard).
- Per color pair (×3):
  - **FLIP+CARRY (transport)**: teleport the cup from its rim-up row pose to a
    hover pose ALREADY INVERTED, mouth ~23 mm above the die top (nothing touches;
    asserted rim well off the floor). 12 regulated zero-velocity hold steps
    (gravity feed-forward + PD at the CoM) while the flip latch reads the pose.
  - **CAP (contact dynamics)**: a velocity-regulated vertical force (target
    −0.06 m/s, force at the CoM only — no torque, no orientation pinning) lowers
    the cup; the die passes through the open mouth (clearance inner_r − half-diag ≈
    12 mm) and the rim descends to the floor; xy PD keeps the cup centered over the
    die. The wrench is DROPPED at the seal; 60 hands-off steps settle the pair;
    `covered_i` asserted.
- **Final**: hands off; success(); ≥ 3.3 s persistence window checked every
  substep; `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing
(0 → 0.13 → 0.33 → 0.47 → 0.67 → 0.80 → 1.0). Passes on seeds 0 and 1 on the forge
(persistence clean on both).

## Rubric

`success()` = all three pairs `covered`; pair i covered ⇔

- **inverted**: cup axis within 20° of straight DOWN;
- **rim sealed**: mouth center < 8 mm above the floor (a cup cocked on its die
  lifts the mouth center ~die_s/2 ≈ 11 mm — margin asserted);
- **die under**: die center within `xy_tol` = 20 mm of the cup axis in the CUP BODY
  frame (honest by construction: any physically-covered die reads ≤ inner_r −
  die_s/2 = 17 mm; a die against the OUTSIDE wall reads ≥ outer_r + die_s/2 =
  43 mm — asserted), AND die center < 21 mm above the floor (a die on TOP of the
  inverted cup reads ~86 mm — asserted margin);
- **settled**: pair stillness SUSTAINED 20 consecutive substeps (counter in
  post_step).

Identity is pairwise — cup_i is only ever judged against the same-colored die_i.

`score()`: per pair, 1.0 if covered NOW, else a latched 0.4 once that cup has ever
been rim-down (within 40° of inverted); mean over pairs; 1.0 iff success(). Null
policy ≈ 0 (cups spawn rim-UP; nothing ever inverts them).

## Embodiment argument (single Franka, parallel jaw, OSC)

Only the three cups are ever manipulated (the dice are never touched):

- **cup**: outer Ø 64 mm — inside the ~80 mm Franka jaw span, side power-grasp
  around the body wall at mid-height (rigid decagonal shell, no deformation
  issues). Sequence per cup: side grasp, lift ~15 cm, rotate the wrist 180° (the
  cup extends ±37 mm from the grasp point — clears the floor with the hand at
  ≥ 15 cm), position over the die, lower until the rim rests on the floor
  (fingers end ~4 cm above the floor holding the cup mid-body — clear of both
  floor and rim), open jaws, retract vertically.
- **precision**: the drop needs the cup axis within inner_r − half-diag ≈ 12 mm of
  the die center — comfortably above closed-loop OSC noise; an off-center rim
  contact just nudges the die inward or the cup can be re-lifted.
- **die**: never grasped; its only interaction is being passively enclosed.
- **base pose**: Franka base at (−0.22, 0, 0) facing +x; both rows (x = 0.10 and
  0.30, slots y = ±0.16, 0) lie 0.32–0.58 m from the base at floor height — inside
  the comfortable envelope.

## Randomization (readback-verified in smoke)

Which row holds cups vs dice (side swap), an independent color-slot permutation for
each row, per-body ±30 mm xy jitter and free yaw.

## Check list (smoke.py — 15 checks, rejection only)

1. settle/no-NaN + layout sanity; 2. score ≈ 0 at reset; 3. side-swap readback over
8 seeds; 4. per-row color-permutation readback; 5. null policy ≈ 0; 6. seed-strategy
A — dice dropped INSIDE the rim-up cups (containment the nesting way) rejected;
7. seed-strategy B — cups stacked into a tower (bottom-on-rim; nesting impossible)
rejected; 8. wrong color — full derangement of genuinely sealed enclosures scores
exactly the flip credit, zero covered (identity); 9. near-miss — die against the
OUTER wall of its sealed cup rejected by radial; 10. perched — cup cocked on its die
rejected by rim seal/radial; 11. die ON TOP of the inverted cup rejected by the
floor gate; 12. settle gate — a covered pair with velocity injected is not covered
while moving; 13. latched flip credit survives teleport-away while covered credit
drops; 14. rejection audit (success never True in the battery); 15. final no-NaN.
