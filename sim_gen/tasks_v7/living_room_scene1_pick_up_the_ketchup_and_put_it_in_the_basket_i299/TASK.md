# leaky_basket_seal — plug the basket's drain with the WIDER lid, then load the ketchup onto the seal

`sim_gen` task `living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i299`
— env `simgen.leaky_basket_seal` (robot="null"; bodies driven by pose writes).

## Seed provenance

Seed: `libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket`
(RoboVerse `roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket.py`).
The seed is a one-stage pick-and-place: grasp the standing red ketchup bottle, carry
it over a PASSIVE open basket resting on the surface, release; success = ketchup CoM
inside the basket's `contain_region` bbox. Kept from the seed: the red ketchup bottle,
a brown distractor sauce bottle, the green basket, and "the ketchup inside the basket"
as the final containment predicate.

## What changed & why it is strategically different

**The receptacle is DEFECTIVE and must be REPAIRED before it can hold anything.** The
green basket is an elevated hopper on legs whose floor is a steep 55° FUNNEL ending in
an open octagonal drain THROAT (70 mm across flats). The seed's entire plan — drop the
ketchup into the basket as found — executed faithfully here runs the bottle down the
funnel, DISCHARGES it through the throat, and lands it on the floor under the stand
(smoke #6): containment in the unsealed basket is physically impossible. New
load-bearing skills:

1. **Metric, physics-graded tool selection**: two knobbed steel lids lie on the floor,
   identical in build and color, differing ONLY IN SIZE; which side each is on swaps
   per episode (Bernoulli, readback-verified). Only the WIDE lid (90 mm > the 70 mm
   throat; disc radius 45 mm > the octagon's 37.9 mm circumradius) can seal — it can
   NEVER pass the throat in any orientation. The NARROW decoy (48 mm; circumscribed
   radius 26.8 mm < the 35 mm inradius) passes in EVERY orientation: released dead on
   the axis — the strongest possible attempt — it free-falls through and lands on the
   floor (smoke #7). The choice must be made by comparing widths against the opening.
2. **Repair the receptacle (funnel-seating contact)**: the wide lid must be released
   over the cavity and the funnel SELF-CENTRES it — the disc slides down the 55° cone
   and SEATS flat over the throat (readback: seat at local z = throat + 23 mm, the
   exact cone-carry height). The lid is never written into a seated pose; the seat is
   carried entirely by the cone contact.
3. **Load onto the live seal**: the ketchup then lands ON the seated lid and settles
   leaning in the cavity (the 133 mm bottle cannot lie flat in the 115 mm cavity and
   its 55 mm body would itself pass the open throat — the seal is load-bearing for the
   goal state, and the persistence window keeps it honest).
4. **Exclusions**: the brown bbq bottle must stay out, and the narrow lid must not be
   left in the cavity (dropping it onto the seated plug fakes nothing — smoke #12).

Distinct from the other examined tasks_v7 packages: **i164** (hang the intact
RECEPTACLE on a wall peg, then load it — suspension, nothing to repair), **i214**
(sluice-gate hopper: UNSEAL a mechanism to discharge the target and CATCH it — the
inverse operation: there the seal must be removed, here a seal must be CREATED from a
selected part), **i292** (relocate a blocker, force-roll a ball through a garage), and
robobench `pen_holder` (fill an intact grounded cup). Nothing else examined makes the
solver FIX the receptacle — turn a colander into a container by choosing and seating a
part — before the seed's own goal predicate can even be attempted.

## Scene (procedural only)

- KINEMATIC STAND at x ≈ 0.50 ± 0.03 m, y ± 0.05 m, FREE yaw: 4 corner legs (throat
  plane z = 0.16), an octagonal funnel of 8 slabs pitched 55° (inner faces from the
  35 mm-inradius throat out past the wall corners, gap-free), and 4 green walls
  (115 mm square interior, rim at ≈ 0.332 m). Anything in the cavity feeds the throat.
- DYNAMIC LIDS (steel gray, grip knob on top, spawn positions Bernoulli-swapped +
  jittered): PLUG disc 90 × 18 mm, knob 28 × 26 mm, 0.12 kg; DECOY disc 48 × 12 mm,
  knob 24 × 18 mm, 0.05 kg.
- DYNAMIC squeeze bottles (55 mm dia × 133 mm with cap, standing, slots
  Bernoulli-swapped + jittered, identify by COLOR): RED ketchup / white cap (0.30 kg,
  target); dark BROWN bbq / black cap (0.28 kg, excluded).

Randomization verified by READBACK in smoke checks 2–4 (wide-lid side flips with the
lids always opposite; bottle sides flip, always opposite; continuous jitter in stand
x/yaw and per-object xy).

## Rubric (latched; anchored in the demonstrated solution)

Streaks (8 consecutive `post_step`s, stillness required — a fly-through never latches):

| credit | condition |
|---|---|
| 0.30 | `sealed_ever`: wide lid SEATED over the throat (stand-local: centre within 14 mm of the axis, z in the seat band, level within 15°, still) |
| 0.15 | `in_ever`: ketchup resting contained in the cavity above the seal plane |
| 0.20 | `ret_ever`: sealed AND contained simultaneously |
| cap 0.65 | latched base |
| 1.00 | `success()` live: sealed ∧ ketchup contained ∧ bbq NOT contained ∧ decoy NOT in the cavity ∧ lid + ketchup still |

## Teleport solution (solve.py) — transport only; interactions are physics

- P0: reset, 0.5 s settle, layout readback (stand xy/yaw, both lids, both bottles,
  wide-lid side). Score 0.000.
- P1: ONE pose write stages the WIDE lid level in free air on the stand's CURRENT
  axis, ~8 cm above the seat (inside the cavity, touching nothing). Hands off: the
  drop, the funnel SELF-CENTRING and the flat SEAT over the throat are contact
  physics (seat readback local z = +0.183 = throat + 23 mm, the cone-carry height).
  Score 0.300.
- P2: ONE pose write stages the ketchup upright just above the rim, slightly off-axis
  so it does not balance on the grip knob. The drop, impact on the seated lid, lean
  against the wall and ring-down are contact physics. Score 1.000.
- P3: ≥ 3.3 simulated seconds hands-off; success must persist. Score 1.000.

Measured on the forge: seeds 0, 2 (wide lid on the right) and seed 1 (wide lid on the
left) all print non-decreasing `SIM_GEN_SCORE` 0.0000 → 0.3000 → 1.0000 → 1.0000 and
`SIM_GEN_SOLVE: SUCCESS` (~17 s wall each).

## Execution order

Seal-then-load is the demonstrated path and is declared in `describe()`; it is NOT
hard-required as an order: only the final settled state is judged (`describe()` says
so). A bottle dropped before sealing is simply discharged onto the floor — recovery
means sealing and then placing it again, so the seal physically gates the goal rather
than a scripted order check.

## Embodiment (single Franka, OSC, base at the origin)

- **Lids**: each is picked by its knob (24–28 mm dia × 18–26 mm tall — comfortably
  inside the 80 mm parallel-jaw stroke; masses 0.05–0.12 kg trivial) at 0.13–0.24 m
  radius. Width comparison is visual (90 vs 48 mm at 4× scale difference in area).
- **Seal**: carry the wide lid to above the open basket top (rim 0.332 m at
  0.42–0.58 m radius — inside the Franka envelope) and RELEASE over the cavity; the
  funnel does the fine placement, so no precision insertion is demanded of the arm.
- **Ketchup**: 55 mm dia standing bottle < 80 mm jaw — a standard side pinch at
  0.24–0.36 m radius; release above the rim, again no precision demanded.
- All grasps and releases lie in a 0.13–0.58 m radius annulus at 0–0.45 m height in
  front of the base with top-down or side approaches unobstructed (the basket is open
  on top; the lids and bottles stand on open floor).

## Checks

- `solve.py`: 3 forge runs (seeds 0, 1, 2 — both wide-lid-side branches), each
  `SIM_GEN_SOLVE: SUCCESS` with non-decreasing scores and a 3.3 s hands-off hold.
- `smoke.py`: `SIM_GEN_SMOKE: ALL PASS 17/17` on the forge, frames.npz
  (218 × 600 × 960) recorded:
  1. reset settled, finite, score ≤ 0.02
  2. wide-lid slot side flips across seeds; lids always opposite (readback)
  3. bottle slots flip; bottles always opposite (readback)
  4. continuous jitter present (stand x, yaw, lid xy, bottle xy) (readback)
  5. null policy 2.5 s → score ≤ 0.02, no success
  6. SEED strategy (drop the ketchup into the basket as found) → discharged through
     the throat to the floor (local z < 0.12), no containment, ≤ 0.02
  7. wrong lid: decoy released dead on the axis → falls through the throat to the
     floor, no seal credit
  8. seal alone → 0.30, not success
  9. wrong bottle: bbq resting contained on the seal → no containment credit, not
     success
  10. near-miss: ketchup standing on the floor UNDER the sealed basket, hugging the
      axis → not contained, no latch
  11. exclusion: ketchup AND bbq inside → capped 0.65, never success
  12. exclusion: decoy resting in the cavity on the seal + ketchup loaded → success
      collapses
  13. success reproduction via the solve recipe → success True, score 1.0
  14. latch regression: ketchup yanked out → success collapses, 0.65 latch persists
  15. latch regression: plug removed → seal broken, 0.65 latch persists
  16. rejection audit: success never True at any rejection-battery judged point
  17. final no-NaN
