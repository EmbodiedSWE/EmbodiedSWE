# cliff_catch — stage the tray under the drop edge, then push the butter off so gravity loads it

**Env name:** `simgen.cliff_catch` (scene `cliff_catch`, robot `null`)

## Seed provenance

- Seed id: `libero_90/living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray.py`
- Seed plan: grasp the butter off the open table, carry it through the air among
  distractor groceries, and set it down inside a passive wooden tray that never moves.
  Success is a bounding-box containment test on the butter in the tray's contain region.
  The whole strategy is **transport the cargo by hand to a static container**.

## What changed, and why it is strategically different

Kept: the same two protagonists (a block of butter, an open tray) and the same *surface*
goal — the episode ends with the butter inside the tray.

Changed — the PLAN, not the numbers:

1. **The container is the manipulated object; gravity is the carrier.** The butter starts
   on top of a 20 cm ledge, boxed into a roofed pen that leaves only ~14 mm above it —
   it physically CANNOT be picked up, only slid. The pen's front opening is flush with
   the cliff's drop edge, so the only way the butter ever travels is a free fall. The
   solver's transport skill is applied to the TRAY: slide it across the floor into the
   marked drop zone at the base of the cliff (position + heading re-sampled per episode),
   i.e. *bring the container to the object*, the exact inverse of the seed's
   *bring the object to the container*.
2. **Arrival dynamics are part of the goal.** `caught` latches only when the butter
   crosses the tray's rim plane inside the aperture in genuine free fall (downward speed
   > 0.6 m/s at the crossing, consecutive-substep evidence). A slow, lowered placement —
   the seed's plan, executed by teleport in the smoke since the roof forbids it
   physically — produces the identical final pose and still scores ~0 (tested negative
   control A). Success additionally requires the butter physically at rest inside the
   upright tray on the floor, so it is judged on real physical outcome, not the latch
   alone.
3. **An irreversible hazard forces execution order.** The moment the butter lands
   anywhere that is not inside the tray, a permanent `dropped` latch caps the score at
   0.10 forever. Push-first-stage-later can never be repaired (tested), and near-miss
   staging (130 mm off) turns the drop into a permanent loss (tested), so "stage the
   catcher BEFORE releasing the cargo" is a hard ordering constraint, absent from the
   seed entirely.

A solver therefore needs a different plan shape: read the sampled drop zone → slide the
tray onto it (precision floor placement of a container) → actuate the drop (non-prehensile
push over an edge) → rely on gravity for the hand-off. No stage of the seed's plan
(grasp cargo, aerial carry, lowered place) survives.

### Differentiation from sibling tasks (same batch)

- `i14` (same butter family): orientation-selective slot insertion — still hand-carries
  the butter to the container; here the butter is never carried and the tray moves.
- `i20` runaway can: reactive interception of an object already moving at reset; here
  nothing moves until the solver triggers the release, and the skill is pre-positioning
  a passive catcher, not chasing.
- `i33` unjam drawer ("let the mechanism act"): environment performs the motion after a
  veto is removed; here the solver both stages the catcher and actuates the drop, and
  the "mechanism" is bare gravity plus an aiming problem.
- `i4`/`i9` low-clearance slide axes: the roof here is only the lift-block that keeps
  the butter push-only; the claimed axis is the gravity-fed catch + ordering hazard,
  not confined sliding.

## Difficulty tier: easy — 2 stages

1. Stage the tray in the drop zone (slide, floor-level placement, ±4.5 cm).
2. Push the butter off the edge; free-fall catch, settle.

**Execution order is REQUIRED** (stage before release), enforced by the permanent
`dropped` latch — this is the task's core constraint, not an incidental.

## Rubric

- 0.30 — latched: tray staged (settled, upright, on the floor, within 4.5 cm of the
  drop-zone center).
- +0.35 — latched: free-fall catch (`caught`).
- 1.0 — iff success(): caught, never dropped, butter resting inside the upright tray
  standing on the floor, everything settled.
- Cap 0.10 forever once `dropped` fires. Doing nothing scores exactly 0 (the tray parks
  ≥ 0.38 m from the drop zone).

## Check list (smoke battery, 16 named checks — forge: `SIM_GEN_SMOKE: ALL PASS 16/16`)

1. settle/no-NaN — reset settles finite; butter at rest in the pen, tray upright at its
   parking spot; score ~0, no latches (2 checks).
2. randomization-is-real — READBACK: fixture heading (full circle), drop-zone position,
   tray parking spot, butter start all move across seeds; tray never parks near the
   drop zone (1).
3. null-policy-fails — 240 idle steps → score ~0, no success, no latches (1).
4. oracle ×3 seeds — stage → push → catch → settle → success(), score 1.0 (3).
5. rubric monotonicity — 0 < 0.30 (staged) < ~0.65 (caught) < 1.0 (success); partials
   < 1.0 (2).
6. negative control A (seed strategy) — teleport carry-and-gentle-place into the tray:
   identical final pose, `caught` never latches, score ~0, no success (1).
7. negative control B (near-miss + permanence) — tray 130 mm off: not staged; butter
   drops, permanent cap ≤ 0.10; a later real drop into the tray cannot repair it (3).
8. negative control C (order violation) — push before staging: perfect staging
   afterwards stays capped ≤ 0.10 (1).
9. calibration probe — lateral staging-offset sweep 0/0/30/60/130 mm with published
   catch table: catches ≤ 30 mm, clean miss at 130 mm (2).

## Physics notes

- Fall to the rim ≈ 0.15 m → ≈ 1.7 m/s at the crossing, far above the 0.6 m/s catch
  gate; kinematic lowering (~0.14 m/s) and gentle robot placement sit far below it.
- Tray: 166 mm outer, 150 mm interior, 53 mm walls, 12 mm floor (impact-tunneling
  margin), `max_depenetration_velocity=0.5`.
- Drop zone center 95 mm out from the cliff face: the tray interior spans 20–170 mm from
  the face, bracketing the measured ~50–90 mm landing band of an edge-pushed butter.
- Oracle release adds a small outward exit velocity (0.15 m/s), the natural exit speed
  of a real push; everything after release is free physics.
