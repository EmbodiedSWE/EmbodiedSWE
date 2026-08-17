# plate_slot_swap — both color slots hold the WRONG plate; swap them through the one-plate transfer cradle

Package: `sim_gen/tasks_v7/put_plate_in_colored_dish_rack_i205`
Env: `simgen.plate_slot_swap` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/put_plate_in_colored_dish_rack`. In the seed, the robot lifts THE
plate off its stand and lowers it into the OPEN slot of a static dish rack whose slots
are color-named — one prehensile transport into a fixed, empty, sky-open target,
driven by recorded waypoints, with the color binding given for free by the variation
index.

## What changed, and why it is strategically different

1. **There is no empty target — the goal slots are the obstacles.** A two-slot
   edge-slot dish rack starts with BOTH pockets occupied by the WRONG plate (blue
   plate between the yellow fins, yellow plate between the blue fins), and each pocket
   physically holds exactly ONE plate: two plate thicknesses exceed the fin gap
   (`2t = 36 mm > gap = 32 mm`, asserted in `__post_init__`). The seed's entire
   strategy — carry the plate to its color slot and lower it in — is impossible as a
   first move, and smoke check 6 proves it physically: a plate dropped onto its
   occupied color slot topples off the occupant's rim and is rejected.

2. **The task is a combinatorial occupancy puzzle with a forced buffer stage.** The
   only other on-edge rest in the world is a free-standing gray TRANSFER CRADLE with
   one identical pocket. Solving requires THREE moves in a physics-enforced order:
   park either plate in the cradle, move the other into its freed color slot, recover
   the parked plate into the last slot. The seed has one move and no intermediate
   state; here the solver must plan through a state where a plate rests in a location
   that is worth nothing terminally (the cradle credit is a latched WAYPOINT, not a
   goal — a plate LEFT in the cradle blocks success).

3. **Every move is an edgewise extraction + reinsertion, twice pose-dependent.** The
   plates stand ON EDGE in 32 mm fin gaps; moving one means a straight vertical pull
   until the rim clears the 70 mm fins, then a re-entry threading the destination gap.
   The rack AND the cradle both take xy jitter + FULL ±180° yaw every episode, so the
   color→pose binding must be read from the scene (fin colors) and used twice per
   move (source and destination) — nothing is at a fixed world pose.

Unlike sibling `put_plate_in_colored_dish_rack_i86` (a carousel MECHANISM the solver
drives, with a non-prehensile shuffleboard insert), i205 has no driven mechanism and
no slide: the difficulty is the interlocked occupancy conflict + capacity-one buffer,
and all contact work is vertical edgewise extraction/insertion. No corpus task read
this session shares this plan shape.

## Solution outline (teleports = free-air TRANSPORT only — verified on forge)

`solve.py` uses teleports only to carry a plate that is already hanging clear in free
air to a hover directly above the destination pocket (velocities zeroed); every
load-bearing interaction happens through contact dynamics:

- **P0** settle + readbacks; assert the swapped arrangement, score ~0.
- **P1 BUFFER** — velocity-servoed vertical FORCE lift extracts the yellow plate from
  the blue slot through fin friction until its rim clears the fins (readback
  z_loc ≥ 0.157); free-air teleport to a hover centered over the cradle pocket;
  RELEASE — gravity threads it down the fin gap onto the pocket floor (the fins and
  end stops do the fine alignment by contact). Cradle seat latch (30 consecutive
  slow seated steps) fires. Score → 0.20.
- **P2 MATCH 1** — same extraction/transport/gravity-insertion for the blue plate:
  out of the yellow slot, into the now-free BLUE slot. Match latch. Score → 0.55.
- **P3 MATCH 2** — recover the yellow plate from the cradle into the YELLOW slot;
  wait for `success()` to hold 120 consecutive steps. Score → 1.00.
- **P4** hands-off ≥ 3.3 simulated seconds; assert success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

The pod's external-force frame semantics are measurably UNSTABLE (the same world-z
arg arrives world-vertical for one plate and tilted for another, differing between
runs), so every forced lift runs under an escape/regression monitor with per-plate
encoding self-correction (recover by hover + gravity re-drop, toggle the encoding,
retry; a certified lift keeps its encoding). Insertions apply no force at all — a
centered free release is encoding-proof.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, scores monotone
0.00 → 0.20 → 0.55 → 1.00, ~19 s wall clock each; seed 2 exercised the full
self-correction loop (escaped world-lift caught by the monitor, body-mode stall
caught by gain-escalation watchdog, clean retry).**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(0.55, −0.08, 0), between the rack (nominal (0, 0.10)) and the cradle
(nominal (0, −0.26)): every task point is within ~0.65 m — inside Franka's ~0.85 m
envelope even at extreme jitter/yaw. **Grasping:** every manipulation is the same
top-rim pinch of a standing plate — a seated plate's top rim stands
`2r − fin_h = 80 mm` proud of the fins (asserted), and the rim is 18 mm thick vs the
80 mm jaw span; the disc faces are free on both sides above the fins. **Extraction:**
a straight ~100 mm vertical lift (demonstrated by the force servo) — no lateral
clearance needed, the gap passes the plate with 14 mm slack. **Transport:** a free
carry between hovers. **Insertion:** center the plate over the destination gap
(±7 mm lateral slack, generous ±30 mm along the slot) and lower until the rim enters,
then release — the demonstrated release-and-drop shows the pocket funnels the last
80 mm by contact alone, so gripper accuracy of a few mm suffices. **Perception:**
the fin colors and apron tabs name the slots; the plates are color-matched — direct
visual binding. Both plates and both fixtures are always upright-accessible from the
sky (nothing is covered).

## Execution order

The three moves are REQUIRED and their order is enforced by PHYSICS (slot capacity),
not by the rubric: no matching move exists until a slot is freed, and freeing a slot
requires the cradle (the only other on-edge rest). Within that: EITHER plate may make
the buffer move (the solve parks yellow; parking blue and mirroring is equally
valid), and pausing between moves is harmless. All of this is stated in `describe()`.

## Rubric

- +0.20 once EITHER plate has EVER been seated in the transfer cradle for 30
  consecutive slow steps (the unavoidable buffer stage; the cradle starts empty, so
  a null policy can never earn it), latched.
- +0.35 once EITHER plate has EVER been seated in its OWN color slot for 30
  consecutive slow steps (impossible at reset — both slots start wrong), latched.
- Partial credit capped at 0.55; `score = 1.0` iff `success()` (blue plate seated in
  the BLUE slot ∧ yellow plate in the YELLOW slot ∧ persistently still, live).
  A seat = rack-frame x within the slot band ∧ y inside the end stops ∧ center z in
  the on-edge rest window ∧ plate ON EDGE (|axis·ẑ| ≤ 0.35). `settled()` is a
  stillness counter-latch (30 consecutive steps under the velocity gates), so a
  flying/jolted arrangement judged mid-motion never counts. Null policy scores ~0.

Scene `__post_init__` carries honesty asserts: capacity one (2t ≥ gap + 3 mm); the
gap passes one plate with margin over both contact offsets; the x band covers every
physically-seated lean yet the two slot bands stay disjoint; the z window accepts
max-lean and upright rest but rejects fin-top perches, center-block perches and
ground rests; the axis window accepts max lean and rejects flat poses; the end stops
confine the rim; the seated top rim stands ≥ 60 mm proud (rim pinch); reset jitter is
penetration-free.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 13/13`)

1. Reset settles: states finite, plates seated ON EDGE in each other's color slots
   (the swapped start, by readback), NEITHER matched (color binding), settled.
2. Fresh reset: score ~0, no success.
3. Randomization: rack yaw varies across 8 seeded resets (wide spread, both signs),
   live quat agrees with the cached sample, xy jitter real (readback).
4. Randomization: cradle yaw/xy vary; each plate's in-slot x/y jitter is real.
5. Null policy 240 steps: score ~0, no success, still swapped.
6. Capacity (seed-analog): the blue plate dropped straight onto its OCCUPIED color
   slot topples off the occupant's rim — NOT seated, NOT matched, score ~0, the
   occupant still seated. A pocket holds exactly one plate.
7. Flat fin-top perch: with the pocket emptied, a plate laid FLAT across the fin
   tops sits INSIDE the z window yet the on-edge axis test alone rejects it.
8. Buffer ≠ goal: yellow plate seated in the cradle latches EXACTLY +0.20, is not
   matched, not success.
9. Half done: blue plate then seated in its freed BLUE slot → score capped at 0.55,
   not success (yellow still parked in the cradle).
10. Latched credit: removing the matched blue plate to the floor keeps the latched
    0.55 while live matched drops — score never decreases.
11. Settle gate: both plates written into their matched slots and judged
    immediately — matched LIVE but stillness has not persisted → NOT success; the
    probe is dismantled before the stillness latch can complete.
12. Rejection audit: `success()` never fired at any judged point.
13. Final state no-NaN.

Smoke also records `frames.npz` (600 × 960 × 3 frames) from a perspective camera.
