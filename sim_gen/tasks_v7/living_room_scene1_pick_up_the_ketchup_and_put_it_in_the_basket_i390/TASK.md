# pinned_hatch_delivery — deliver the ketchup through the propped window, THEN pull the one-shot pin so the sash seals it in

`sim_gen` task `living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i390`
— env `simgen.pinned_hatch_delivery` (robot="null"; bodies driven by pose writes).

## Seed provenance

Seed: `libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket`
(RoboVerse `roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket.py`).
The seed is a one-stage pick-and-place: grasp the standing red ketchup bottle, lift
it over the rim of a PASSIVE open basket, release; success = ketchup CoM inside the
basket. Kept from the seed: the red ketchup bottle, a brown distractor sauce bottle,
the green basket, and "the ketchup in the basket" as the core containment predicate.

## What changed & why it is strategically different

**Access to the receptacle is CONTESTED, EXPIRING, and must be spent in the right
order.** The basket is intact but sits INSIDE a roofed cabinet: the only way in is a
front window guarded by a gravity-closing solid SASH, currently PROPPED open by a
steel pin — and the final goal state requires that window CLOSED (a negative
obligation the seed never has: seal the door behind your delivery). New load-bearing
skills:

1. **One-shot resource management (physics-forced ordering)**: the pin is the only
   thing holding the window open, and the sash plate is SOLID — once it falls, the
   guide bore faces bare plate and the pin can NEVER be re-inserted (smoke #9
   presses it back at up to 3.8 N: the tip advances 9 mm and stalls on the plate).
   Pull the pin first and the episode is unwinnable: the delivery attempt is walled
   by the closed sash (smoke #8, the bottle stalls at the plate after 85 mm of real
   pushing). Deliver-first-seal-second is never scripted — only the final settled
   state is judged (`describe()` says so) — the ORDER is enforced by geometry.
2. **Delivery by push-through-aperture, not lift-over-rim**: the roof kills the
   seed's whole motion. The bottle must be slid across the apron, THROUGH the 20 cm
   window under the propped sash (17 cm opening vs the 13.3 cm bottle), across a
   25 mm channel gap its 55 mm base bridges, over a 2 mm step-DOWN onto the sill,
   and over the inner edge so it free-falls into the basket. Containment is reached
   by a drop the solver never touches.
3. **Seal the cabinet as a goal condition**: success additionally requires the sash
   fully closed (≤ 8 mm of the bottom stop) AND the pin tip clear of the channel —
   the sash slam after the axial pin extraction is pure gravity + prismatic-joint
   contact physics, and a 0.15 credit is reserved for holding delivery and seal
   SIMULTANEOUSLY (smoke #7 shows sealed-but-empty tops out at 0.40).
4. **Exclusions**: the brown bbq bottle must stay out (a real push delivering it
   earns nothing — smoke #10 — and success collapses while it is inside — smoke
   #13).

Distinct from the other examined packages: **i299** (same seed — the basket itself
is DEFECTIVE and must be repaired by seating a lid; here the basket is fine and the
contested thing is ACCESS, which must be used and then deliberately destroyed),
**i164** (hang the receptacle on a peg — suspension, free access), **i214**
(sluice-gate: UNSEAL a hopper to discharge and catch — the inverse obligation; here
a seal must be lost forever at the right moment), **i292** (relocate a blocker,
force-roll a ball), **close_drawer_i386** (bistable bayonet lock — reversible
mechanism, no ordering constraint). Nothing else examined makes the solver SPEND an
irreversible resource, and none requires closing the only entry as part of the goal.

## Scene (procedural only)

Housing-local frame; the DYNAMIC 40 kg housing (teleports carry the sash joint
anchor) is one compound body:

- Cream APRON table (x −0.31..−0.155, top 0.182), then a 25 mm channel GAP, then
  the bored front WALL (−0.13..−0.11): window 20 cm wide, sill 0.18 (a 2 mm step
  DOWN from the apron — flush-seam trap), lintel 0.40. Slick mu 0.22.
- Roofed INTERIOR (x −0.11..0.10, roof 0.44..0.46, grip floor mu 0.75): entry is
  the window ONLY.
- Blue SASH: solid 12×280×260 mm plate, 0.25 kg, on a spawn-authored prismatic
  Z-runner (travel 0..0.20 m) in the channel; gravity-closing. Closed it covers the
  window AND the pin bore (cfg-asserted irreversibility).
- Steel PIN: 7 mm-radius × 105 mm shaft + red 32 mm knob, 0.06 kg, mu 0.12,
  inserted through the dark guide block; the sash bottom rests ON the shaft
  (propped opening 0.173 m). Insertion depth jittered per episode.
- Green BASKET (200×210×105 mm, 0.8 kg) seated snugly on the interior floor —
  slack 10/20 mm vs seat gates 30/45 mm, so it physically cannot leave the seat.
- Squeeze bottles (55 mm dia × 133 mm, 0.30 kg, LOW authored CoM z=0.042, identify
  by COLOR): RED ketchup / white cap (target); dark BROWN bbq / black cap
  (excluded). Slots Bernoulli-swapped + jittered.

Randomization verified by READBACK (smoke #2–3): housing yaw ±12° and xy ±3 cm;
bottle slots flip sides, always opposite; continuous per-bottle jitter; pin depth
jitter (tip spread 5.7 mm). ~30 cfg asserts pin the honesty invariants (opening
clears the bottle; closed sash walls window AND bore; bottle bridges the channel;
knob never passes the bore; rim far below the sill; roof slit passes nothing).

## Rubric (latched; anchored in the demonstrated solution)

Streaks (8 consecutive still `post_step`s — a fly-through or mid-slam frame never
latches, smoke #14):

| credit | condition |
|---|---|
| 0.25 | `in_ever`: ketchup ever RESTED contained in the basket (basket-frame box; any orientation — a dropped bottle topples and may roll to a wall) |
| 0.15 | `seal_ever`: cabinet ever sealed (sash ≤ 8 mm of the stop AND pin tip clear of the channel) |
| 0.15 | `full_ever`: delivered AND sealed simultaneously |
| cap 0.55 | latched base |
| 1.00 | `success()` live: contained ∧ bbq NOT contained ∧ sealed ∧ basket seated ∧ everything persistently still |

Null earns exactly 0 (the pin holds — smoke #4). The seed strategy (deliver, stop)
caps at 0.25 (smoke #5). Pin-first caps at 0.15 (smoke #8).

## Teleport solution (solve.py) — transport only; interactions are physics

- P0: reset, settle, authored-mass + layout readback (sash propped on the pin,
  bottle slots, pin depth). Score 0.0000.
- P1: ONE pose write stages the ketchup upright on the push lane; then a
  velocity-regulated force push (0.25 m/s, ≤1.6 N — under the 1.9 N tip threshold;
  fast enough to carry the CoM past the drop lip) drives it through the window;
  the topple over the inner edge and the drop into the basket are contact physics.
  Score 0.2500.
- P2: escalating velocity-capped axial pull on the pin (0.8→1.4 N sufficed); when
  the shaft clears the plate the sash free-falls and slams shut (readback q → 0.0000
  on the joint stop). Score 1.0000.
- P3/P4: hands-off ring-down + ≥3.3 simulated seconds persistence. Score 1.0000.

Measured on the forge: seeds 0, 1, 2 (both Bernoulli slot branches) all print
non-decreasing `SIM_GEN_SCORE` 0.0000 → 0.2500 → 1.0000 → 1.0000 → 1.0000 and
`SIM_GEN_SOLVE: SUCCESS` (~19.5 s wall each).

## Execution order

Deliver-then-seal is the demonstrated path and is declared in `describe()`; it is
NOT hard-required as a script: only the final settled state is judged. But the wrong
order is self-defeating BY PHYSICS: sealing first walls the only entry (smoke #8)
and the pin can never go back (smoke #9), so contained-and-sealed is reachable only
by delivering first. The order forcer is geometry, not a rule check.

## Embodiment (single Franka, OSC, base facing the apron)

- Every contact point lies on the OPEN FRONT side of the cabinet at 0.18–0.36 m
  height; nothing ever requires reaching into the roofed interior (that is the
  point of the design — delivery is a push, not an insertion).
- **Ketchup**: 55 mm dia standing bottle < 80 mm parallel-jaw stroke — side pinch
  or open-jaw push on the apron; slide it along the window axis and release before
  the inner edge; the drop does the placement, no precision demanded. Color tells
  it from the bbq bottle at a glance.
- **Pin**: the red 32 mm knob is a purpose-made handle (< 80 mm jaw, 0.06 kg);
  extraction is a straight axial pull in free front space with ~1.5 N of
  resistance — well inside Franka payload and force control.
- The sash needs no manipulation at all: gravity closes it.
- Order comes from a visual/causal judgment (the pin visibly props the sash over
  the only opening), not from dexterity.

## Checks

- `solve.py`: 3 forge runs (seeds 0, 1, 2 — both bottle-slot branches), each
  `SIM_GEN_SOLVE: SUCCESS` with non-decreasing scores and a 3.3 s hands-off hold.
- `smoke.py`: `SIM_GEN_SMOKE: ALL PASS 16/16` on the forge, frames.npz
  (283 × 600 × 960) recorded:
  1. reset settled/finite; the PROP is real (sash rests on the pin at height,
     readback), score ~0
  2. randomization A: housing yaw + xy jitter real (readback spreads)
  3. randomization B: Bernoulli slot swap (both sides, always opposite), bottle
     jitter, pin-depth jitter (readback)
  4. null policy 240 steps → sash still propped, score ~0
  5. SEED strategy done for real (solve's own push servo): delivered-only rests
     contained → 0.25 ONLY, not success
  6. latch regression: ketchup yanked back out → 0.25 latch persists, no success
  7. sealed-but-empty: real pin pull with the ketchup out → 0.40, simultaneity
     latch stays 0, not success
  8. pin-first trap: real pull, then a real 85 mm delivery push WALLED by the
     closed sash → ≤ 0.151, not success (order physically forced)
  9. no re-insertion: pin pressed back at the closed channel (held level,
     escalated to 3.8 N) → advances 9 mm, stalls on the plate, sash never lifts
  10. wrong bottle: bbq delivered for real → contained but earns 0, not success
  11. near-miss: ketchup standing IN the window on the sill → not contained
  12. near-miss: ketchup on the roof directly above the basket → z band rejects
  13. exclusion + cap: sealed cabinet with BOTH bottles in the basket → all
      latches fire, score pinned at 0.55, never success
  14. settle gate: goal pose written WHILE MOVING (real 0.43 m/s velocity) → all
      pose gates pass, not success; construct removed before ring-down
  15. rejection audit: success() never True at any judged point
  16. final no-NaN
