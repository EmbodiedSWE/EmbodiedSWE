# roll_in_garage — clear the doorway post, then roll the ketchup into the covered garage

**Package:** `sim_gen/tasks_v6/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i292`
**Env:** `simgen.roll_in_garage` (scene-level, `robot="null"`)

## Seed provenance

`libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket.py`):
grasp the standing ketchup bottle, carry it over the OPEN-topped basket, release, and a
bounding-box containment check ends the episode. One overhead pick-and-place, receptacle
passively available, no obstacle, no commitment point.

## What changed and why it is strategically different

The receptacle geometry **inverts the seed's plan** — vertical grasp-carry-drop is
replaced by declutter-the-aperture + a ground-level rolling insertion:

- The basket becomes a **roofed side-entry garage** whose only opening is a floor-level
  doorway. The roof covers the whole interior, so the seed's move — carry the bottle
  above the receptacle and release — leaves it resting **on the roof** and scores
  nothing (smoke #6). Nothing about this task is solvable from above.
- The doorway is **blocked by an orange post**. Metric interlock: max side gap next to
  the post is (200 − 130)/2 + 15 = **50 mm < the 55 mm bottle diameter**, in every
  randomized layout, in any bottle orientation — the post must be relocated before the
  bottle can pass (smoke #9 keeps it in place and fails).
- Just inside the doorway a **12° ramp rises to a ~13 mm crest with an overhanging
  drop-off face**: the bottle must be pushed/rolled along the floor, through the
  doorway, up the slope and over the crest — a physical commitment point. Short of the
  crest the slope rejects the bottle back out of the doorway (smoke #7 demonstrates
  this dynamically); past it, the overhang retains it (solve.py's hands-off persistence
  window demonstrates that).
- The target lies **on its side** (it is a roller to be pushed, not a stander to be
  grasped), and a same-shape **yellow mustard bottle** (slot-swapped each episode) adds
  a color-grounded identification clause plus an exclusion clause (smoke #10).

A solver therefore needs a different plan (relocate the obstacle, then a floor-plane
push-roll through a side aperture with a climb-and-commit interaction — the *obstacle*
is the only thing that gets picked and placed; the *target* never leaves the floor) and
different code structure (obstacle relocation + rolling push control instead of
grasp-lift-release). The seed's entire plan, executed here, produces the smoke #6
reject state.

## Teleport-solution outline (solve.py, phases; `SIM_GEN_SCORE` at each boundary)

- **P0** reset + settle; layout readback printed (garage offset/yaw, slot assignment).
- **P1 CLEAR (transport, teleport):** one pose write relocates the post from the
  doorway throat to an open floor patch. Free-space transport of a free object — its
  only role is to occupy the aperture; no interaction is bypassed.
- **P2 TRANSPORT (teleport):** one pose write stages the ketchup 14 cm OUTSIDE the
  doorway plane on the garage axis, lying with its rolling axis across the approach —
  asserted outside the garage; staging earns only doorway-approach credit.
- **P3 ROLL IN OVER THE CREST (contact dynamics — the load-bearing interaction):** a
  velocity-limited world-frame force at the CoM (≤ 6 N, v ≤ 0.22 m/s, with a small
  lateral centering term) rolls the bottle across the floor, through the doorway and up
  the 12° ramp. The force is CUT the moment the CoM passes the crest; gravity tips it
  over the drop-off into the landing bay where it settles hands-off. success() first
  turns True here. From force-cut to verdict nothing touches the bottle.
- **P4** hands-off persistence ≥ 3.3 s (retention by the overhanging crest face is
  exactly what this window certifies), then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds **0, 1 and 2** (all `SIM_GEN_SOLVE: SUCCESS`, scores monotone
0.00 → 0.14/0.15 → 0.20 → 1.00 → 1.00).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

Plausible base pose: **base at the world origin on the floor plane**; all required
contacts lie at radius 0.29–0.60 m, heights 0.0–0.16 m — inside the Franka envelope,
and nothing the arm must touch is ever under the roof deeper than ~8 cm past the
doorway plane.

- **Orange post:** the jaw grasps across its 55 mm depth (80 mm max opening) anywhere
  on its upper half — the post is 150 mm tall, taller than the roof top (132 mm), and
  stands 50 mm OUTSIDE the doorway plane, so a top-down pinch approaches in free air.
  Lift and set down anywhere clear (e.g. the open floor patch at (0.20, 0.40)).
- **Red ketchup bottle:** never grasped — pushed. Closed-fingertip contact on the
  trailing side of the lying 55 mm cylinder at ~28 mm height, rolling it along the
  floor. The doorway is 200 mm wide vs the 133 mm bottle; the flared wings recapture
  ±30° of heading error, so the required precision is far above OSC noise. The final
  push segment ends with the fingertip ~50 mm past the doorway plane under a 120 mm
  roof — fingertip-only intrusion, hand stays outside. The last 40 mm of travel
  (crest drop + roll-out) is gravity's, not the arm's.
- **Yellow mustard bottle:** requires no contact — it must merely be left alone.

## Execution order

Required and geometry-forced (declared in describe()): the post must leave the doorway
before the bottle can pass (50 mm max gap < 55 mm bottle — enforced by collision, not
by the rubric); the roll-in necessarily comes after. No rubric clause depends on
timestamps; the order emerges from the interlock.

## Rubric

`score()` = 0.10·cleared (post ever out of the blocking zone, latched) +
0.15·doorway-approach (gated on cleared, latched max) + 0.25·entered (ketchup ever past
the doorway plane under the roof, latched) + 0.35·landed (ketchup ever past the crest
in the bay, latched), capped at 0.85; exactly **1.0 iff `success()`**: ketchup CoM in
the landing bay (past crest + 15 mm, resting height, inside the walls) ∧ ketchup at
rest ∧ mustard NOT inside the garage. Null policy scores ~0 (the post starts blocking,
so every term is gated or latched off).

## Check list (smoke.py — rejection battery, forge: `SIM_GEN_SMOKE: ALL PASS 14/14`)

1. settle/no-NaN: post standing and blocking, bottles lying at slots, still, finite
2. score ~0 at reset, no success
3. randomization readback: ketchup/mustard slot assignment flips; always opposite slots
4. randomization readback: per-slot xy jitter (> 4 mm) and garage yaw spread (> 2°)
5. null policy (240 steps): score ~0, no success
6. **seed strategy**: released over the receptacle → rests ON THE ROOF → score ~0,
   no success
7. near-miss crest: abandoned on the ramp short of the crest → rejected back out,
   entered-only credit, NOT success, score ≤ 0.55
8. near-miss doorway: settled centred just outside the doorway plane → NOT success
9. wrong object: mustard in the bay, post still blocking, ketchup untouched → score ~0
10. exclusion clause: ketchup AND mustard both settled in the bay — every ketchup gate
    passes, the mustard clause alone rejects → NOT success, score ≤ 0.85
11. latched credit: regressing the ketchup out leaves the latched score unchanged
12. beside-wall: settled against the OUTSIDE of a garage wall at in-range depth →
    v gate rejects (garage-frame math, valid under yaw randomization)
13. rejection audit: success() never True anywhere in the battery
14. final no-NaN

frames.npz (222 × 600 × 960 × 3) recorded and saved in cwd by smoke.py.
