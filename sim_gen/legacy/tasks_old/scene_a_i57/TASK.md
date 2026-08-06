# scene_a_i57 — Tablecloth Pull (yank the runner out from under the trophy)

**Registered as:** `SCENES["tablecloth_pull"]`, env `simgen.tablecloth_pull`
(robot="null", scene-level). **Tier: easy — 2 stages** (1: extract the runner by a
dynamic yank; 2: stow it flat on the pad). **Execution order: physically forced** —
extraction necessarily precedes stowing (the runner cannot rest on the pad while it is
still under the trophy); a single motion may do both.

## Seed provenance

- Seed: `calvin/scene_A`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/calvin/scene_A.py`)
- Seed semantics: the CALVIN table — an articulated fixture (drawer / slider / button /
  switch DOFs) plus three colored blocks. Every task in the seed family is either
  "actuate a furniture DOF" or "grasp block → transport → release", and all of it is
  quasi-static: execution speed never matters, and the block named in the instruction is
  always the object the robot manipulates.

## What changed

A pink trophy block (50×50×70 mm, the CALVIN-pink nod) stands on a thin slick blue
runner sheet (300×140×6 mm, friction μ≈0.05 via `friction_combine_mode="min"`) lying on
the floor; a flat gray stow pad waits ~0.5 m away at a random bearing. The goal: get the
runner out from **under** the trophy and lay it flat on the pad, while the trophy keeps
standing exactly where it is. A permanent `disturbed` latch trips the moment the
trophy's center leaves a 6 cm sphere around its spawn (score capped 0.05 forever —
putting it back does not help), and the graded `extracted` latch only fires when the
trophy is *seen* standing on the ground, clear of the runner footprint, undisturbed.

## Why strategically different (a different PLAN, not different numbers)

- **The manipulated object and the graded object are swapped.** CALVIN manipulates the
  target block; here any grasp/lift/slide/knock of the graded block is the permanent
  failure (tested control A: lifting it latches `disturbed`; even finishing everything
  and teleporting it back to a geometrically perfect end state stays capped — the
  anti-teleport evidence). The thing the solver moves is the *substrate underneath*.
- **The required skill is dynamic, which the seed family does not contain anywhere.**
  The winning move is the tablecloth trick: snap the runner out at ~1.4 m/s so friction
  (μ≈0.05, acceleration bound μg≈0.5 m/s²) has ~0.1 s to act — the trophy moves ~5 mm.
  The *same trajectory* executed quasi-statically fails by physics, not by fiat: with no
  other horizontal force on it, the trophy rides the runner 1:1 and crosses the 60 mm
  latch radius long before the runner clears (tested control B; the calibration probe
  publishes the speed→displacement cliff on identical resets). A seed-style solver that
  treats this as slow pick-and-place fails on *manner*, not on target poses.
- **Honest by construction (no code-latch gymnastics needed for the alternatives):** the
  trophy's shortest travel OFF the runner footprint is ≥75 mm laterally (half-width 70 +
  half-diagonal 35.4 − max across-seat 30) and ≥150 mm axially — both above the 60 mm
  latch — so shoving the trophy off, or tilting the runner to dump it, displaces it past
  the latch by geometry. Removing the runner from under the *standing* trophy is the
  only scene-level solution. (A robot binding could also brace the trophy with a second
  finger while stripping the runner — a legitimate alternate plan that is still nothing
  like the seed's.)
- **Seed mechanism axis untouched:** there is no articulated DOF at all; the seed's
  button/switch/drawer skills have no analog here (declared N/A as a negative control —
  nothing to actuate).

Sibling-axis note (parallel batch, checked against the claimed-axes memory + sibling
TASK.md files): the **dynamic/impulsive-manipulation axis ("the quasi-static version of
the correct motion measurably fails") and substrate-extraction-from-under are claimed by
no sibling** — i50 claims the exact opposite manner axis (closing must be *gentle*);
i32 (same seed family) covers a marked cube with a vessel (no dynamics, containment
goal); i34 rotates in place (orientation goal); i25 topples; i48/i20 are timing/
interception. The preserve-the-bystander latch machinery itself follows the i32/i34/i25
precedents and is enforcement, not the claimed axis.

## Randomization (per episode)

Runner xy (±5 cm) + full yaw (the pull direction must be read from the scene); the
trophy's seat on the runner (±4 cm along, ±3 cm across, free yaw); pad bearing (full
circle), distance (0.48–0.62 m) and yaw. A memorized pull vector fails.

## Rubric (graded score, 1.0 iff success)

- ~0 for doing nothing (nothing latched, runner unmoved).
- Live shaping ≤0.30 ∝ runner withdrawal (gated on the trophy still standing in place).
- Latched milestones: `extracted` × (0.50 + 0.20·trophy-preserved-now +
  0.30·runner-stowed-now); exactly 1.0 iff `success()` (= extracted ∧ trophy upright /
  in-sphere / grounded / settled ∧ runner flat on the pad, centered, yaw-aligned mod
  180°, settled ∧ never disturbed).
- Permanent cap 0.05 once `disturbed` fires.

## Check list (smoke battery, 17 checks)

1. settle/no-NaN: states finite, trophy standing on the runner, at rest
2. reset score ~0, no success, nothing latched
3. randomization is real (readback: runner heading/xy, trophy seat, pad position)
4. null policy fails (240 idle steps → score ~0, no latches)
5–7. oracle reaches success() + score 1.0 on 3 seeds (yank → verify → stow)
8. monotonicity: 0 → partial withdrawal → extracted → stowed strictly increases
9. monotonicity: partials < 1.0; final is success at exactly 1.0
10. negative A (seed strategy): lifting the trophy trips the permanent latch
11. negative A cont.: perfect final geometry (trophy back home, runner stowed) stays
    capped ≤0.05 — latch permanence + `extracted` never latches while disturbed
12. negative B (manner): the same pull at 0.5 mm/substep drags the trophy past the
    latch radius — capped, no success
13. near-miss: yank stopped with the runner still under the trophy → no extraction
    latch, partial credit <0.5, no success
14. near-miss recovery: finishing the yank + stow reaches success
15. negative C: extracted + stowed but the trophy left tipped (in place) → no success,
    score ≤0.85 (preserved gate is current-state)
16. calibration probe: quasi-static end of the speed sweep drags the trophy >60 mm
17. calibration probe: yank end stays <30 mm and extracts cleanly
    (full 0.5/2/5/12 mm-per-substep displacement table published in the log)

Physics honesty: the oracle drives the runner kinematically but always with **matched
written velocity**, so PhysX friction sees the true sliding speed and the trophy's drag
(or ride-along) is real integrated contact physics; the trophy is fully dynamic
throughout, and every judged predicate reads settled physical poses.
