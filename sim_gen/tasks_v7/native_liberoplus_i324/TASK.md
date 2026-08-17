# native_liberoplus_i324 — Block Magazine (bottom-dispense, classify, route)

Env: `simgen.block_magazine` · Scene: `block_magazine` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed: `libero/native_liberoplus`
(`RoboVerse/roboverse_pack/tasks/libero/native_liberoplus.py`) — LIBERO-plus
replays LIBERO pick-and-place scenes under nuisance perturbations (distractor
objects, layout shuffles, lighting, texture, language). Every seed task is
solved by the same invariant one-shot plan: **visually ground the named target,
ignore the perturbations, grasp it, transport it, release it over the goal.**
Distractors are only noise; the target is always directly graspable; the plan
has fixed length.

## Why this task is strategically different

The three pillars of the seed strategy are each inverted:

1. **The target cannot be grasped.** The red block sits inside a fully sealed
   vertical magazine (roofed, walled; only a 1.2 cm viewing slit). Nothing can
   leave except through a low side **port exactly one block high**, and the only
   actuator that reaches the stack is a captive **plunger rod** through the
   opposite wall. Grasp-hover-release on the target is physically void — the
   rubric latches port transit as a hard gate, so even a policy that somehow got
   the red block to the pad without passing the port scores ~0.
2. **The plan is a variable-length mechanism loop, not a one-shot transport:**
   push the plunger to its far stop (the bottom block is shoved out through the
   port), pull it back to its near stop (the stack drops one step), route the
   dispensed block by color, repeat. The red block's depth G ∈ {1,2,3} is
   randomized, so the episode — not memorization — decides how many cycles run.
   Bottom-first ordering is forced by geometry (roof denies top access, the port
   admits only the bottom block), not by a declared rule.
3. **Distractors carry an obligation instead of being noise.** Every gray block
   the mechanism forces out ahead of the red one must end up settled *inside*
   the discard bin; the red one must end up resting on the green pad. A gray on
   the floor, on the roof, or half-stuck in the port blocks success.

Differs from every examined neighbour: `native_liberoplus_i226` (hidden-mass
weighing instrument — mine has fully visible state and a dispense mechanism, no
instrument reading), `..._i332` (one object up a ramp through a doorway + a
gravity-bistable door — mine is an iterative multi-object loop with no bistable
element), `pen_holder` (fill a container by transport — mine forbids direct
transport of the target entirely).

## Solution outline (proven by solve.py on forge seeds 0, 1, 2)

1. Reset, settle, read back G and the stack order from state.
2. Per cycle: position-servo a force (error clamp 6 cm, cap 8 N ≈ 2.5× the
   measured push resistance) on the plunger toward its far hard stop until the
   bottom block clears the port; then press it back against the near stop
   (retracted tip parks 4 mm behind the shaft face so the dropping stack can
   never rest on it); wait for the stack to drop.
3. Identify the dispensed block; **transport-teleport** it to a hover and let
   gravity finish: gray → above a free spot in the bin interior; red → above
   the pad. (Teleports are transport-only: dispensing and the port transit are
   pure contact dynamics; containment/on-pad are reached by dropping.)
4. Repeat until the red block has emerged (its transit latch fires during the
   physical push), then hold hands-off ≥ 3.3 sim-seconds and verify success
   persists. `SIM_GEN_SCORE` printed at every phase boundary, non-decreasing.

## Franka embodiment argument (single arm, parallel jaw, base at world origin)

- **Plunger:** the 3.6 cm knob is graspable (< 8 cm jaw opening) and stands
  proud of the wall on the robot-facing side; pushing to the far stop and
  pulling to the near stop are straight-line ±y strokes of 8.8 cm at z ≈ 2.5 cm,
  with hard stops at both ends forgiving over-travel. Grip force needed ≈ 8 N
  push / pull — trivial for the Franka.
- **Blocks:** 5 cm cubes (top-grasp with 8 cm jaws) picked from the open floor
  outside the port and placed into a 13 cm bin / onto a 10 cm pad — standard
  pick-and-place with generous clearances (bin mouth is 2.6× the block width).
- **Reach:** everything task-relevant (knob at y ≈ −0.15…−0.23, port exit at
  y ≈ +0.05…+0.10 around x = 0.52; bin at (0.28, 0.24); pad at (0.26, −0.18))
  lies within a 0.60 m radius of a base at the origin, at heights 0–0.28 m.
- **Order:** the only ordering is mechanism-forced (bottom-first); the robot
  needs no declared-rule bookkeeping, just "pump, look at the color, route".

## Randomization (readback-verified in smoke)

- Red stack depth G ∈ {1, 2, 3} (uniform), gray permutation over the remaining
  slots.
- Bin and pad xy jitter ±2.5 cm each.
- The magazine itself is fixed (it anchors the plunger joint; jointed
  mechanisms are never teleported).

## Rubric (latched, gated)

0.15·(grays ever settled outside)/G + 0.15·(grays ever settled in bin)/G +
0.15·(red ever in the port passage) + 0.15·(red settled on pad, **gated on the
transit latch**), capped at 0.85; exactly 1.0 iff live success = red settled on
the pad having transited the port AND every gray either still inside the
magazine or settled in the bin.

## Checks (smoke.py, 15)

1. Settle/no-NaN: stack at slot heights, rod retracted, still.
2. Score ~0 at reset, no success.
3. Randomization: G takes ≥ 2 distinct values (readback).
4. Randomization: bin/pad jitter ranges > 1 cm (readback).
5. Null policy: 240 idle steps → score ~0.
6. Seed strategy: red dropped from above lands on the sealed roof — rejected.
7. Bypass gating: red teleported straight onto the pad, no transit → score ~0.
8. Pad near-miss (transit earned honestly): red just off the pad — rejected.
9. Red done but a gray beside the bin — rejected (non-vacuous: only the gray fails).
10. Red done but a gray on the magazine roof — rejected.
11. Red done but a gray straddling the port mouth — rejected.
12. Wrong destination: red into the bin — rejected.
13. Latched credit survives regression (gray binned then yanked out).
14. Rejection audit: success() never True at any judged point.
15. Final no-NaN.
