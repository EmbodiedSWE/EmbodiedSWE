# weigh_serve — find the ballasted bowl with a balance, then serve it

**Task id:** `libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate_i189`
**Seed task:** `libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate`

## Task

Three visually identical black bowls sit in a row on a kitchen counter. Exactly one
of them is secretly ballasted (0.40 kg; the other two are 0.06 kg) — nothing visual
distinguishes it. A two-pan balance beam stands on the counter: a revolute pivot with
±12° hard stops and its CoM 0.05 m below the pivot (a physical pendulum, so an empty
or equally-loaded beam actively returns to level). The goal: **identify the ballasted
bowl by performing one strictly pairwise weighing on the balance, then place that bowl
on the serving plate.**

The measurement is *binding*: at the instant the weighing latches, the rubric records
which bowl the physics readback names —

- |tilt| ≥ 8° (beam calm, one bowl per pan): the **lower pan's** bowl;
- |tilt| ≤ 5° level: the **left-out** bowl (two equal pans ⇒ the odd one is heavy);

— and from then on **only that bowl** may enter the plate's exclusion zone. Before any
weighing has latched, *no* bowl may enter it.

## Why this is strategically different from the seed (and the corpus)

- The seed is pure pick-and-place: the target bowl is *named by its position*
  ("at the front"), so perception + transport solves it. Here the target is defined by
  a **hidden physical property** (mass) that no camera can see; the *only* path to the
  goal runs through an **active physical sensing** procedure — load the instrument,
  wait for its dynamics to settle, read the answer out of the physics.
- The rubric contains an **information-gating** mechanism absent from the seed and the
  read corpus tasks: the plate is poisoned until a valid measurement exists, and the
  measurement's *verdict body* is latched at measurement time. You cannot serve first
  and weigh later, cannot serve a different bowl than your measurement named, and
  cannot rig a decisive-looking beam state without a strict one-per-pan comparison.
- Three qualitatively different verdict branches (−tilt / +tilt / LEVEL) arise from
  the reset randomization; the LEVEL branch requires *inference* (the heavy bowl is
  the one you did **not** weigh), not just readback.
- Seed-strategy check: teleporting the front-slot bowl straight to the plate — the
  literal seed plan — is spoiled while the bowl is still in the air, and the episode
  is unrecoverable (smoke checks 4–5).

## Scene (`scene.py`)

- Procedural compound spawners (no asset files): counter, plate, three bowls
  (open-top box bowls, 0.045 m outer radius, MassAPI-authored masses so CoM/inertia
  are correct), balance base (kinematic pedestal + post, post top 35 mm below the
  pivot so the bar can never wedge on it), balance beam (bar + pivot pin + struts +
  two 150 mm rimmed pans; authored diagonal inertia Ixx = 0.03 and angular damping
  8.0 so the beam *creeps* to its stop at ~2 rad/s instead of catapulting its load).
- Per-env revolute joint authored in `bind()` (Base→Beam, axis X, ±12° limits,
  joint-pair collision disabled).
- **Randomization (reset):** the ballasted bowl is shuffled over the three slots,
  per-slot xy jitter (±2 cm), the plate side mirrors (+y / −y) with ±2 cm jitter.
  Tunables: `slot_jitter`, `plate_jitter`, `shuffle_bowls`, `swap_plate_side`, all
  rubric thresholds.
- **Monitors/latches:** `_s_pan` (a bowl in a pan, 0.10), `_s_weigh` + `_verdict`
  (valid pairwise weighing latched after a 15-step decisive streak with the beam calm;
  0.25), `_s_near` (ballasted bowl within 0.15 m of the plate; 0.25), `_spoiled`
  (any bowl in the plate zone pre-weighing, or any non-verdict bowl post-weighing —
  permanent). `success()` = weighed ∧ ¬spoiled ∧ ballasted bowl served (within
  55 mm of the plate axis, on the plate top, upright, at rest). `score()`: staged
  0 → 0.10 → 0.35 → 0.60 cap, 1.0 on success; spoil zeroes everything.

## Solution outline (`solve.py`, teleports = transport only)

1. **P0 settle** — hands-off; beam levels itself (readback 0.00°).
2. **P1 load** — read slot positions, teleport the slot-0 bowl above the −y pan,
   300 hands-off steps, then the slot-1 bowl above the +y pan (sequential loading —
   simultaneous drops whip the beam and eject the light bowl). All motion of the
   beam under load is pure contact dynamics.
3. **P2 read** — wait for `_s_weigh`; read the tilt sign: −tilt ⇒ −y pan's bowl,
   +tilt ⇒ +y pan's bowl, level ⇒ the left-out slot-2 bowl. Oracle mass readback is
   cross-checked *after* the decision (assert only, never used to decide).
4. **P3 serve** — teleport the identified bowl 40 mm above the plate, drop, settle.
5. **P4 persist** — ≥3 s hands-off; `SIM_GEN_SOLVE: SUCCESS` only if success holds.

Verified on the forge on seeds 0, 1, 2, 7 (−tilt), 123 (+tilt), 42, 99 (LEVEL, 3.0–3.7°
readback vs the 5° gate), both plate sides; score ladder 0 → 0.10 → 0.35 → 1.0.

## Embodiment argument (Franka, 80 mm parallel jaw)

Every load-bearing interaction is a pick-and-place of a 90 mm-diameter, 44 mm-tall,
≤0.40 kg open-top bowl — squarely graspable rim-first by an 80 mm parallel jaw (rim
wall 6 mm thick). Pan wells are 150 mm squares with 24 mm rims at counter height
+ ~0.15 m, in free space on all sides — generous clearance for a top-down approach.
The balance is read, never actuated: the robot only places and removes bowls, and the
12°-stop beam keeps pan floors within easy reach. Placement tolerances (55 mm serve
radius, 50 mm pan half-width vs a 45 mm bowl) are far looser than Franka repeatability.

## Execution order declaration

1. Minimal goal + scene geometry; 2. working `solve.py` iterated on the forge
(pivot wedge-lock, beam whip, and rim-perch bugs found and fixed by geometry, not by
weakening checks); 3. rubric finalized (verdict-body latch closes the
"rig-a-fake-tilt, serve the unweighed heavy" loophole; level gate widened 4°→5° from
measured margins) and re-verified; 4. `smoke.py` rejection battery.

## Smoke checks (`smoke.py` — `SIM_GEN_SMOKE: ALL PASS 10/10`)

1. settle/no-NaN + mass-authoring readback (0.40/0.06/0.06) + beam level;
2. randomization readback: heavy slot ≥2 values, both plate sides, jitter differs;
3. null policy: 240 idle steps, nothing latches, score ≤0.01;
4. **seed strategy**: front-slot bowl straight to the plate, no weighing → spoiled
   mid-air, score 0 forever;
5. history unforgiven: honest weighing + perfect serve *after* the spoil → end state
   equals the goal arrangement, still score 0;
6. wrong bowl after a tilted weighing (left-out bowl served) → spoil, 0.35 → 0;
7. wrong bowl after a level weighing (a panned bowl served) → spoil, score 0;
8. near-miss: right bowl 120 mm off the plate axis → capped at 0.60, no success;
9. not-a-comparison: single-bowl slam and 2-vs-1 stacked load → weighing never
   latches;
10. frames.npz video captured and saved.
