# lid_tray_stow — unload the lid-tray, stow the cargo, then cap the box

## Seed provenance

- Seed: `rlbench/close_box`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/close_box.py`)
- Seed semantics: an articulated box (`box_base`, fixed base) with a hinged lid stands
  open and EMPTY; the robot pushes the lid shut. Success is a `JointPosChecker` on the
  hinge (`box_joint <= -14 deg`). The entire plan is **one guarded push on an attached
  lid** — no objects, no sequencing, no placement tolerance, and the lid can never be
  anywhere but on its hinge.
- This task: `sim_gen.lid_tray_stow` (scene `lid_tray_stow`, robot `null`), fully
  procedural geometry (one compound kinematic open box + plain cuboid/cylinder
  primitives; no asset files, no articulation).

## What changed, and why it is strategically different

The box's closure is kept as the *end state*; the *plan* that produces it is replaced
wholesale. The lid is **detached**, lying on the floor, and **repurposed as a tray**:
the items that belong in the box sit ON the lid (a bottle + 1–2 cubes).

1. **Role reversal with a prerequisite chain.** The lid starts as the *support surface
   of the cargo*. "Close the lid" — the seed's entire plan, executed here by capping the
   box with the loaded lid, cargo riding on top — seals an EMPTY box and is a tested
   negative control (the lid seats, zero items stowed, score ≤ 0.25, no success). The
   solver must first unload the tray INTO the open box, which the seed never requires
   (its box is empty and its lid holds nothing).
2. **A reorientation stage forced by geometry.** The bottle (130 mm) is longer than the
   interior is deep (100 mm): stood upright it protrudes ~30 mm above the rim and the
   lid **physically cannot seat** (tested: the lid dropped on it tips off; also
   upright-inside earns no stow credit). It must be laid FLAT inside — a planned
   regrasp/reorientation, absent from the seed.
3. **Free-body placement tolerances instead of a joint threshold.** Closure is judged on
   the *pose of a free rigid lid*: centered within 15 mm, level within 8°, at rim height
   within 8 mm, and yaw-aligned with the box within 12° mod 90° (the coverage-honest
   limit is 15.9°: beyond it the square lid exposes the aperture corners — a 45°
   twisted lid at the exact seat pose is a tested control). The seed has no free-body
   goal at all — its lid cannot be offset, twisted, or dropped inside.
4. **Order is physically enforced, not scripted.** Capping first traps nothing (cargo
   ends on top of the lid, above the rim → not stowed); once capped, nothing can be
   inserted. The seed has no ordering to get wrong.

A solver therefore needs a different plan — *unload the tray → stow items (reorienting
the bottle flat) → retrieve the freed lid → precision flush-and-aligned capping* —
instead of *approach the hinged lid and push*. Same-strategy-different-numbers cannot
pass: the seed's push-the-lid-shut plan, expressed here as cap-immediately, is negative
control (a).

### Relation to sibling tasks

No sibling claims the container-lid role-reversal / unload-before-close axis: i4 pushes
into confinement (non-prehensile), i9 retrieves out of a cubby, i10 extracts + gently
perches on a convex support, i8 assembles a frame from sticks. The closest house
exemplar (pen_holder) fills an open cup and merely stands it upright — it has no lid, no
covering, no order constraint, and no reorientation-under-height-constraint.

## Difficulty tier / stages

**medium — 4 stages, execution order REQUIRED** (stow-before-cap is enforced by
physics; within the stow phase the item order is free):

1. stow the present cube(s) into the open box;
2. reorient the bottle (too long to stand under the closed lid) and lay it FLAT inside;
3. retrieve the freed lid from the floor;
4. seat the lid flush on the rim — centered, level, at height, yaw-aligned mod 90°.

## Rubric (graded score in [0, 1], judged on physical, settled state)

- `0.00` — nothing (null policy: cargo still on the grounded lid).
- `0.55 × k/K` — k of the K present items stowed (CURRENT state, settled): center inside
  the interior footprint (≤ 73 mm, admits any physically-inside pose by construction)
  and below the rim; the bottle additionally lying flat (axis ≤ 25° from horizontal).
- `+0.20` — latched (in `post_step`, sim-rate): the freed lid brought level over the box
  mouth. Transient achievement kept.
- `1.00` — **iff `success()`**: every present item stowed + the lid CURRENTLY seated
  flush/level/aligned on the rim + settled.

Honesty margins (dry-computed, asserted in `__post_init__`): items on the rim or riding
on the seated lid sit ≥ 22 mm above the 105 mm z-gate; a lid resting on in-box contents
sits ≥ 16 mm below (or, on the upright bottle, ≥ 24 mm above + tilted) the ±8 mm seat
window; the 15 mm xy tolerance sits inside the 20 mm physical capture zone of the lid
overhang; the 12° yaw gate is inside the 15.9° square-coverage limit.

## Randomization (verified by readback in the smoke)

- Box: xy jitter ±45 mm + full yaw (kinematic re-pose).
- Lid: xy jitter ±45 mm + full yaw; item slots ride the lid's frame with ±10 mm jitter,
  cubes get free yaw, the bottle ±15° about the lid axis.
- Cube-count subset sampling (1–2 cubes; the bottle is always present): success is
  judged on the sampled subset; absent cubes park in an off-camera depot.

## Check list (18, all must PASS; smoke prints `SIM_GEN_SMOKE: ALL PASS 18/18`)

1. settle/no-NaN — reset layout settles finite, cargo resting ON the lid, score 0.
2. randomization-is-real — box/lid positions AND yaws differ across seeded resets
   (readback).
3. subset sampling — present count varies across 10 resets.
4. null-policy-fails — 240 idle steps, score ≤ 0.05, no success.
5–7. oracle reaches success() on seeds 0/1/2 (seed 1 with subset sampling ON — proves
   judged-on-subset). Capping is a genuine 8 mm drop-and-settle, never a pose-set.
8. rubric stow transitions hit 0.55·k/K, strictly increasing.
9. rubric lid-over latch lifts score to ~0.75.
10. rubric monotone 0 → stows → 0.75 → 1.0.
11. negative (SEED STRATEGY) — cap immediately with the loaded lid: lid seats, zero
    stowed, score ≤ 0.25, no success.
12. negative (near-miss) — lid released 40 mm off-axis tips into the box: no seat,
    score pinned ~0.75, no success.
13. negative (incomplete) — one cube left outside, lid seated flush: no success.
14. negative (upright bottle, stow gate) — standing inside earns no stow credit.
15. negative (upright bottle, closure) — the lid physically cannot seat over it.
16. negative (tolerance) — 45° twisted lid at the exact seat pose rejected (yaw gate).
17. calibration — lid-release offset sweep: small offsets (≤ 10 mm) seat ≥ 8/9.
18. calibration — large offsets (≥ 20 mm) seat ≤ 1/6.

The seed's own strategy IS expressible here (control 11), so no N/A documentation is
needed. Oracle places objects kinematically (teleport-carry) but every judged outcome —
stowed items, the capped lid, all negative aftermaths — is real released physics.
