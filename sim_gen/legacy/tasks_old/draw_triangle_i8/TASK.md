# draw_triangle_i8 — Stick Triangle (assemble a closed triangle frame)

## Seed provenance

- Seed: `maniskill/draw_triangle`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/draw_triangle.py`)
- Seed semantics: a Franka holding a rigid stick **draws** a triangle — it traces the
  goal outline on a canvas with the tool tip, judged on the coverage of the traced path.
  The core skill is continuous, contact-maintained **path-following** of a prescribed
  curve with a held tool.

## What changed

The goal *shape* (a triangle) is kept; the *mechanism* is replaced wholesale. Nothing is
drawn and nothing traces. Three rigid sticks of **unequal lengths** (200 / 170 / 145 mm,
distinct colors) lie scattered on the floor around a randomly-placed dark square build
mat. The task is to **construct** a closed triangle frame: arrange the sticks flat on the
mat so their ends meet pairwise at three corners, each corner endpoint gap under 3.5 cm,
with each stick contributing both of its (different) ends and the enclosed area at least
half the ideal Heron area for the three lengths. Judged purely on the settled physical
poses read back from sim.

## Why strategically different

- **Discrete construction vs continuous tracing.** The seed's solution is one long
  guarded tool-tip sweep along a given outline; here a solver must *derive* the target
  configuration (a consistent triangle placement from three given side lengths — which
  stick spans which side, where each corner lands), then execute three independent
  place-and-orient operations and close per-corner tolerances. There is no path to
  follow and no tool to hold.
- **The seed's plan is executable here and fails**, which the smoke proves as a negative
  control: sweeping a stick, flat, along the exact triangle perimeter (the drawing
  motion) leaves no persistent structure — zero corners, score ≈ 0.05, no success.
- **Conversely** the seed's canvas has nothing to assemble, so this task's plan (endpoint
  matching of rigid parts) does not transfer back.
- The rubric judges a *persistent, settled arrangement* (with anti-cheese gates: no
  stacking, end-cycle closure against side-by-side bundles, area gate against degenerate
  chains), not a *trace history*.

## Difficulty tier: easy (declared stages: 2)

1. Reason out a consistent frame placement on the mat (which stick on which side).
2. Place the three sticks, closing the three corner tolerances (a single repeated
   place-and-orient skill).

**Execution order: NOT required** — the sticks may be placed in any order; the rubric is
symmetric in the corners (so no wrong-order negative control applies). Difficulty comes
from the 3.5 cm corner tolerances and consistency (all three corners must close
simultaneously with distinct ends), not from stage depth.

## Rubric

- +0.05 per stick placed flat + settled + fully on the mat (no stacking: resting height
  gate), +0.20 per closed corner; partial capped at 0.75 and **latched** (transient
  achievements don't evaporate while adjusting).
- Exactly **1.0 iff `success()`** holds on the current settled state: all 3 corners
  closed on the mat + end-cycle consistency + area gate.
- Null policy scores 0 by construction (sticks spawn on a ring outside the mat).

## Randomization (per episode, verified by readback)

- Mat centre: ±10 cm xy jitter.
- Sticks: ring slots around the mat ±25° arc jitter, ±5 cm xy jitter, free yaw.

## Smoke check list (`smoke.py`, teleport-oracle, NullRobot)

1. settle: finite state, sticks at rest after 120 steps
2. reset: rubric reads zero on the fresh scene
3. randomization: seeded resets differ by readback
4. state roundtrip: `set_state` restores `get_state`
5. null policy: 200 idle steps, score ~0, no success
6–8. oracle reaches `success()` on 3 seeds (teleport + settle, physics judges)
9. rubric monotone through oracle stages (0.05 → 0.30 → 1.0)
10. success persists 150 further steps (no flicker)
11. negative (seed strategy): tracing the outline with one stick fails
12. negative (near-miss): all corner gaps at joint_tol + 2 cm → no corner counts
13. negative (off-mat): gap-closed frame off the mat scores ~0
14. negative (bundle): side-by-side sticks fake ≤2 corners, never success
15. calibration: corner-gap knee sits at joint_tol (counted iff measured gap < 35 mm)
16. video frames captured (`frames.npz` in CWD)

Acceptance marker: `SIM_GEN_SMOKE: ALL PASS n/n`.
