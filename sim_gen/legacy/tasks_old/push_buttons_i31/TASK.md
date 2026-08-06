# circuit_bridge (push_buttons_i31) — repair the three circuits in order

**Env name:** `simgen.circuit_bridge` (scene `circuit_bridge`, robot `null` — scene-level task)
**Tier:** medium — **3 stages** (one ordered fetch-and-span placement per circuit:
red bridge → green bridge → blue bridge).
**Execution order:** REQUIRED. The commanded order is fixed (red, then green, then blue —
stated in `describe()`); completing a circuit while an earlier one is still dead trips a
permanent interlock that voids success and caps the score. Randomization is spatial:
WHERE each color's fixture and bar sit changes every episode, so the fixed order maps to
a different motion sequence each time.

## Seed provenance

Seed: `rlbench/push_buttons`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/push_buttons.py`) — a Franka pushes a
sequence of colored articulated buttons in a commanded color order. The skills are
ordered reach-and-poke: nothing is transported, nothing persists after the fingertip
leaves, and each button is a self-contained momentary press.

## What changed, and why it is strategically different

The three colored buttons become three **broken circuits**: each is a pair of rigid
terminal posts (40 mm square, 60 mm tall) on a base strip, separated by a 70 mm open
trench, with a lamp that lights only while the circuit is live. The console is entirely
**kinematic — nothing is pressable**, so the seed's whole plan (ordered pokes) is
useless by construction: the smoke drives a sustained ordered press on every terminal
and it latches exactly nothing.

To energize a circuit the solver must instead **fetch the color-matched conductor bar**
(160×30×20 mm, scattered on the far side of the workspace) and **lay it across the two
posts** so that it genuinely spans the trench: resting **elevated on both post tops**,
centered over the gap, axis-aligned with the pair, level, settled, and *left there* —
and the three circuits must be completed in the commanded color order. The required
plan is therefore *transport + two-point-support precision placement under an ordering
constraint* — a different plan, not different parameters:

- **Actuation is replaced by construction.** Nothing on the fixture moves; the "press"
  is a free rigid body the solver must place so that physics holds it in a spanning
  pose. Success is a persistent physical state (three settled spans), not a transient
  contact event, and removing a bar un-succeeds the task.
- **Two-point support is the physically honest core.** The rubric's 35 mm along-axis
  window sits inside the geometry's ~40 mm both-post support window: a bar balanced
  level on a single post (correct height!), a bar dropped into the trench (falls 60 mm
  below the post tops), and a bar on the wrong-color pair are all rejected — each is a
  tested negative control.
- **Ordering is kept from the seed but re-based on construction latches**: first
  completions are latched from 0.25 s-sustained settled spans (anti-transient gate) and
  compared against the fixed rank order; an out-of-order completion trips a permanent
  interlock.

Sibling-axes check (per `simgen-batch-task-strategies` memory): the same-seed sibling
`push_button_i6` claimed *weight-activated spring mechanisms / heavy-object selection* —
here there are no joints, no springs, no masses to choose, and the fixtures cannot move.
Elevated **spanning/bridging (two-point support)** is claimed by no sibling: i11 builds a
vertical tower to a height, i25 topples pillars in order (gravity as actuator, nothing
transported), i8 arranges sticks into a flat closed frame by endpoint adjacency, i12/i14
key insertion on size/aperture. Matching here is color-only and secondary; the load-bearing
novelty is the elevated span + ordering interlock.

## Randomization (per episode)

- Circuit → station permutation (3 stations) + per-fixture xy jitter ±25 mm + free yaw.
- Bar → scatter-slot permutation + per-bar xy jitter ±40 mm + free yaw.

## Rubric (score in [0, 1], partial progress latched)

- 0.1 — latched once any bar is lifted clear of the floor.
- +0.25 — latched per circuit whose first completion happened in the commanded sequence
  (0.35 / 0.60 / 0.85 cumulative with the lift latch).
- Capped at 0.45 forever once the ordering interlock trips.
- 1.0 iff `success()`: all three circuits **currently** held bridged (each a settled
  span sustained ≥ 30 consecutive substeps) with no ordering violation.

## Smoke checks (14 — the acceptance battery)

1. settle: no NaN, nothing bridged, score 0
2. randomization is real (readback differs across seeded resets)
3. null policy: score ~0, no success
4.–6. oracle seeds 0/1/2: ordered fetch + span (genuine 8 mm drops onto the post
   tops), success + score 1.0
7. rubric monotone staged scores (0 → 0.1 lift → 0.35 red → 0.60 green → 1.0 blue)
8. **negative (seed strategy)**: sustained ordered presses on every terminal post —
   latches nothing, score pinned at the lift latch, no success (expressible, and fails)
9. negative (near-miss): bar balanced LEVEL at post-top height on a single post — no
   credit (elevation alone is not spanning)
10. negative (near-miss): bar dropped into the trench, resting below the post tops — no
   credit
11. negative (matching): red/green bars swapped, both physically spanning the wrong
   pair (height readback proves the spans are real) — no credit, no success
12. negative (ordering): all three genuinely bridged in reverse order — interlock
   trips, success rejected, score capped ≤ 0.45
13.–14. calibration sweep: aligned drop at growing along-axis offset (0/15/30/45/60 mm,
   3 seeds each); ≤ 15 mm must always bridge (6/6), ≥ 45 mm (one-post support) never
   (0/6); the 30 mm rate is published. Geometry note: both-post support physically ends
   at ~40–45 mm; the 35 mm rubric window is the honest inner bound.
