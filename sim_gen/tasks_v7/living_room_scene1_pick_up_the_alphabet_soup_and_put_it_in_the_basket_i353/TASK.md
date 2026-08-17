# pantry_rack_order — slide three cans into a roofed one-ended rack so their depth order matches the tiles

`sim_gen` task `living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i353`
— env `simgen.pantry_rack_order` (robot="null"; bodies driven by pose writes and
applied forces).

## Seed provenance

Seed: `libero_90/living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket`
(RoboVerse `roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.py`).
The seed is a one-stage pick-and-place: grasp ONE target can (the alphabet soup)
among three distractors, carry it over a passive OPEN basket, release from above;
success = the soup's CoM inside the basket's `contain_region` bbox. Kept from the
seed: food cans on a living-room floor, a wooden container, and "get the can(s)
into the container" as the goal family.

## What changed & why it is strategically different

**Single containment became a SEQUENCING puzzle whose answer is physically
irreversible.** There are no distractors — ALL THREE cans are targets — and the
container is a PANTRY RACK: a low channel (66 mm wide, 80 mm tall inside, 185 mm
deep) with a slick floor, side walls, a closed back end and a ROOF, open only at
its front MOUTH. The judged predicate is not containment but the ARRANGEMENT: the
identity order of the cans from the closed end outward must equal a per-episode
random permutation, displayed by three colored ORDER TILES on the roof. New
load-bearing structure:

1. **The seed's plan is physically void**: releasing a can from above — the seed's
   entire motion — lands it ON THE ROOF; the rack has no top opening (smoke #5).
   Entry exists only by SLIDING through the mouth along the floor.
2. **Geometry turns insertion history into state**: the channel is too narrow for
   any two cans to pass inside (worst pair: max lateral centre separation
   13 + 8 = 21 mm << 45 mm sum of radii), and the roof forbids lifting a mis-placed
   can out. So the final depth order IS the insertion order, and a wrong FIRST push
   is unrecoverable — the rubric's stage 1 simply never latches (smoke #6).
3. **Plan-before-act**: the solver must read the tile permutation, invert it into a
   deepest-first insertion sequence, and only then touch a can. Each later can
   SHOVES the earlier ones deeper by contact — the mechanism that makes the puzzle
   honest also does the fine placement (no precision depth control is demanded:
   pushing "past the mouth" is enough, margins are guaranteed by can-to-can
   contact: contiguous ranks end >= 45 mm apart vs the 30 mm rubric margin).
4. **Identity mapping**: cans are told apart by color AND width (40/50/60 mm), the
   tiles by color; both the target permutation and the can->staging-slot assignment
   are independently shuffled per episode (readback-verified).

Distinct from the other examined tasks_v7 packages: **i299** (repair a DEFECTIVE
funnel-bottomed hopper by metric lid selection, then load one target — there the
receptacle is the puzzle, order is free), robobench **pen_holder** (fill an intact
cup, any order), and the seed itself (single pick-and-drop into an open basket).
Nothing else examined makes the solver DERIVE AND COMMIT TO A SEQUENCE whose first
wrong step is geometrically unrepairable — a FIFO-ordering constraint enforced by a
roof and a no-passing channel rather than by a scripted order check.

## Scene (procedural only)

- KINEMATIC RACK at x ≈ 0.44 ± 0.03 m, y ± 0.05 m, yaw ± 20° (mouth toward the
  base). Local frame: origin at the mouth centre on the ground, +x = depth. Slick
  floor strip (μ 0.10, 8 mm tall) spanning a 160 mm loading apron plus the channel;
  walls 10 mm thick, 80 mm tall; interior 66 mm wide × 185 mm deep; roof (top at
  96 mm); two 30°-flared mouth guides.
- DYNAMIC CANS (h 50 mm, upright, slots Bernoulli-shuffled + jittered ±3 cm over
  ground slots at (0.16,−0.20)/(0.13,0)/(0.16,0.20)): RED tomato Ø40 mm 0.12 kg,
  GREEN peas Ø50 mm 0.20 kg, BLUE soup Ø60 mm 0.30 kg (μ 0.25, restitution 0).
- KINEMATIC ORDER TILES (36×55×6 mm, one per can color) on the roof at local
  x = 150/95.5/41 mm, re-posed per episode so the tile nearest the CLOSED END names
  the can required deepest.

Randomization verified by READBACK in smoke #2–3: ≥3 distinct permutations over 8
seeds with tiles always at the matching rank slots; staging-slot shuffle (tomato
and soup y each span > 0.25 m); continuous jitter (rack x, yaw, per-can xy).

## Rubric (latched; anchored in the demonstrated solution)

Streaks (8 consecutive `post_step`s with ALL cans still — a fly-through never
latches). "Fully inside" = rack-local depth ≥ radius + 8 mm, between the walls, on
the channel floor (z band ±15 mm), upright (≤ 20°).

| credit | condition |
|---|---|
| 0.30 | `s1_ever`: the can required DEEPEST is fully inside and deeper (≥ 30 mm) than every other can inside |
| 0.30 | `s2_ever`: additionally the second-required can is fully inside at rank 2 (third absent or shallower) |
| cap 0.60 | latched base |
| 1.00 | `success()` live: all three fully inside, upright, exact target depth order (strict 30 mm margins), everything still |

## Teleport solution (solve.py) — transport only; interactions are physics

- P0: reset, 0.5 s settle, layout readback (rack xy/yaw, permutation, can slots).
  Score 0.000.
- P1–P3, one per rank in DEEPEST-FIRST order: ONE pose write stands the can on the
  loading apron on the rack's CURRENT axis, fully OUTSIDE the mouth (front edge
  45 mm short). Then a horizontal velocity-servo force (feedforward
  1.1·μ_pair·m_chain·g + 4·(0.12 − v), cap 2.2 N, plus a compensating torque that
  lowers the effective push point to ~8 mm above the floor — a fingertip pushing
  low) slides it through the mouth; forces are CLEARED the instant the can passes
  goal depth (radius + 22 mm); coast-out, settling and the shoving of earlier cans
  deeper are contact physics. Scores 0.300 / 0.600 / 1.000.
- P4: ≥ 3.3 simulated seconds hands-off; success must persist. Score 1.000.
- The can is NEVER written to a pose inside the rack; every interaction with the
  rack and between cans is contact dynamics.

Measured on the forge: seeds 0, 1, 2 draw three DIFFERENT permutations
(soup-peas-tomato, peas-tomato-soup, tomato-soup-peas) and all print non-decreasing
`SIM_GEN_SCORE` 0.0000 → 0.3000 → 0.6000 → 1.0000 → 1.0000 and
`SIM_GEN_SOLVE: SUCCESS` (~19 s wall each).

## Execution order

Deepest-first insertion is declared in `describe()` and is enforced PHYSICALLY,
not by a scripted order check: only the final settled arrangement is judged, but
the roof + no-passing channel make the arrangement equal the insertion history.
The mouth rank alone is repairable (pull the outermost can back out through the
mouth and re-insert — demonstrated in smoke #14); a wrong can at a deep rank can
never be extracted or bypassed, so those episodes are honestly failed (stage
latches keep whatever prefix was earned).

## Embodiment (single Franka, OSC, base at the origin)

- **Cans**: Ø40–60 mm × 50 mm uprights — under the 80 mm parallel-jaw stroke; side
  pinch at 0.10–0.25 m radius (staging slots), masses 0.12–0.30 kg. Transport =
  pick from the slot, set down on the open apron in front of the mouth (0.25–0.32 m
  radius, unobstructed from above).
- **Insertion**: the arm CANNOT reach inside the 66 mm-wide roofed channel and
  never needs to — it pushes the staged can low on its body straight through the
  mouth (fingertip-depth entry of ~1–2 cm at most; the 30° flares funnel small
  lateral errors, and later cans push earlier ones deeper by contact). All pushes
  are horizontal, at 0.28–0.46 m radius, approach direction along the rack axis
  (mouth always faces the base within ±20°).
- **Perception**: identity = color/width at 1.5–4× area differences; the order
  tiles are on top of the roof, unoccluded from above.
- Everything happens in a 0.10–0.58 m radius annulus at 0–0.15 m height with
  top-down or side approaches unobstructed.

## Checks

- `solve.py`: 3 forge runs (seeds 0, 1, 2 — three distinct permutations), each
  `SIM_GEN_SOLVE: SUCCESS` with non-decreasing scores and a 3.3 s hands-off hold.
- `smoke.py`: `SIM_GEN_SMOKE: ALL PASS 16/16` on the forge, frames.npz
  (237 × 600 × 960) recorded:
  1. reset settled, finite, score ≤ 0.02, no can inside
  2. permutation varies (5 distinct/8 seeds), tiles always at matching rank slots
     (readback)
  3. staging-slot shuffle + continuous jitter (rack x, yaw, per-can xy) (readback)
  4. null policy 2.5 s → score ≤ 0.02, no success
  5. SEED strategy (lower the can in from above) → lands ON the roof, never
     enters, ≤ 0.02
  6. wrong FIRST insertion (physical push, seats fully) → stage 1 never latches,
     ≤ 0.02
  7. correct-then-wrong (physical) → frozen at 0.30, stage 2 never latches
  8. completed-but-swapped (physical, all inside/upright/still, last two ranks
     swapped) → never success, 0.30
  9. near-miss: deepest can straddling the mouth plane → not "fully inside",
     nothing latches
  10. tipped: correct depth order, mouth can lying (z band passes) → upright gate
      rejects, no success
  11. cold wrong-order full arrangement (deepest two swapped) → ≤ 0.02, never
      success
  12. success reproduction via the solve recipe → success True, score 1.0
  13. latch regression: mouth can yanked out → success collapses, 0.60 latch
      persists
  14. recovery: mouth can physically re-inserted → success returns to 1.0
  15. rejection audit: success never True at any rejection-battery judged point
  16. final no-NaN
