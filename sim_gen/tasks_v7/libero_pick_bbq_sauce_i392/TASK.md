# tare_lift — declutter a spring-scale freight car until it rises level with the deck

**Task id:** `libero_pick_bbq_sauce_i392` · **Env:** `simgen.tare_lift` · **Robot:** `null` (scene-level)

## Seed provenance

Seed: `libero/libero_pick_bbq_sauce` — *"pick up the bbq sauce and place it in the basket"*.
Kept from the seed: the two protagonist objects (a bbq-sauce bottle, a basket) and the
tabletop pick-and-place vocabulary. Everything else is new.

## Strategic difference

The seed is a single free-space pick-and-place: grasp the bottle, drop it in a basket.
Here the bottle **starts already seated in the basket and is never handled**; the judged
object is the *basket itself*, and the core mechanic is a **weight gate** the seed has no
analogue of:

- The basket rides a spring-preloaded freight car (vertical prismatic joint, k = 250 N/m,
  constant 14.1 N preload at the top stop) sunk in a shrouded pit. 2–4 heavy ballast cans
  (850 g each) ride in the basket; **any one can sinks the car ≥ 19 mm below the doorway
  sill**, burying the basket's leading face behind a wall of static geometry — no push can
  extract it, and the car's roof strips (a few mm of headroom) cap any free rise and deny
  a top grasp of the basket rims.
- The **only path** the mechanism affords is to *unload*: pinch each can out through the
  open roof lane, drop it in the discard bin, and the spring **itself** lifts basket +
  bottle to deck level (car floor 3 mm proud of the sill). Then the basket slides out
  through the doorway onto a randomized goal pad, bottle still seated.

So the solved episode *contains* the seed's motion vocabulary (top-down pinches, a
horizontal place) but inverts its logic: instead of putting a thing into the basket, you
must take things **out** of it — and the reason is a live force balance, not a rule. The
scene is also strategically distinct from every other package in `tasks_v7`: no other
task uses a spring scale as an ordering mechanism (the closest, `pin_latch_dumbwaiter`,
uses a spring car as an elevator with a friction pin, not a tare/weight gate; the
counterpoise and weigh-beam tasks use free levers, not preloaded prismatic springs).

The rubric's negative space mirrors this: the seed family's own strategy — transport the
bottle to the goal — scores ≈ 0 (smoke check 5).

## Teleport-solution outline (solve.py, verified on forge, seeds 0 and 1)

Teleports are transport only; every load-bearing interaction is contact dynamics.

- **P0** settle 120 steps; layout readback; asserts: ≥ 2 cans aboard, car at its bottom
  stop (q ≤ −(travel − 4 mm)), bottle seated, score ≤ 0.03, no success.
- **P1 unload** — each aboard can is *transported* to a 50 mm hover over a free bin park
  and **dropped** (gravity + contact does the binning); `SIM_GEN_SCORE` after each can
  (0.1625 → 0.225 → 0.2875 on seed 0), monotone-asserted.
- **P2 rise** — hands off: the **spring alone** raises car + basket + bottle to the top
  stop (readback q ≥ −4 mm, settled). Score latch `risen` → 0.50. The 1-can depression
  readback matched the spring arithmetic to < 1 mm (predicted 22.1 mm).
- **P3 egress** — a velocity-regulated horizontal CoM force servo (v_des 0.08 m/s,
  stiction floor 1.0 → 4.5 N escalation, ±8 N clamp, cross-track damping, force-frame
  mode probing) pushes the basket over the sill past the egress line (x > 206 mm).
  Score latch `egressed` → 0.70.
- **P4 deliver** — same servo steers the basket onto the goal pad (12 mm tolerance
  circle judged at 55 mm); requires `basket_on_pad ∧ bottle_seated ∧ cans_binned ∧
  settled` → success, score 1.0.
- **P5 persistence** — 400 hands-off steps re-checking success, then
  `SIM_GEN_SOLVE: SUCCESS`, hard exit.

## Embodiment argument (single Franka, 80 mm parallel jaw, OSC)

Base the Franka on the deck side facing the doorway, ~0.5 m from the pit centre; every
waypoint below is inside a 0.85 m reach envelope.

- **Can extraction:** cans are 36 mm square × 50 mm tall — an easy pinch for an 80 mm
  jaw. They stand in the basket poking up through the car's open roof **lane** (158 mm
  wide between the strips, no roof above it), so a straight top-down pinch-and-lift at
  each can's readback pose clears the shroud (top 230 mm; wrist approach from above the
  open doorway side). The bottle's neck also pokes through the lane but the cans are
  offset ±55 mm from it — 19 mm clearance to the closed jaw, comfortable for OSC.
- **Discard:** carry each can over the bin (walls only 35 mm tall) and release; the
  solve proves a 50 mm free drop bins reliably.
- **Rise:** no actuation at all — the robot simply withdraws; the preload does the lift.
- **Egress + delivery:** the solve's push is a horizontal ~1–8 N CoM-height force — a
  closed-fingertip push on the basket's back wall (70 mm tall, standing proud of the car
  strips once risen; the doorway is open on this side). The pad tolerance (55 mm) is
  coarse; the servo held ±12 mm, so fingertip nudging with OSC impedance is ample.
  Nothing requires grasping the basket (the strips deny it while in the pit; outside the
  pit a rim grasp is *allowed* but unnecessary) and the bottle is never touched.
- **Forces:** max solve force 8 N horizontal, well inside Franka payload/wrench limits.

## Execution-order declaration

The order **all cans binned → basket egress** is forced by physics, not latches: while
any can is aboard, the basket is buried ≥ 19 mm below the sill (no pushed/resting
egress — smoke check 6 presses the sill with the solve's own escalating servo and jams),
and a sustained 25 N lift only drags the car up through its roof strips a few mm, never
out (smoke check 7, with car-follow readback). The residual idealized-wrench carry
window is excluded by embodiment (the strips deny any top grasp of the rims; a wedged
carry of a loaded 2.4 kg basket through a few-mm headroom slot is not a Franka pinch).
The score latches are additionally order-aware only in that `egressed` requires the live
risen state, which requires zero cans aboard at that instant.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 16/16`)

1. settle/no-NaN: seeded reset finite; car at bottom stop, bottle seated, cans
   consistent; score 0, no success
2. randomization by readback (3-seed max-pairwise): pad xy > 5 mm, pad yaw > 1°,
   bottle yaw > 5°, can arrangement > 5 mm
3. ballast coverage: aboard count spans ≥ 2 distinct values in [2, 4] over 8 resets
4. null policy: 300 idle steps, score ≤ 0.03, no success
5. seed-family strategy (bottle alone to the goal pad): score ≈ 0, no success
6. **flagship** loaded egress denied: 1-can depression matches spring arithmetic
   (< 6 mm error); an escalated 10 N push (the jam is geometric, so the probe wrench
   may exceed the solve's) presses the sill — moved 11 mm, non-vacuous — yet jams
   < 60 mm, far short of the 206 mm egress line
7. loaded free-rise denied: a full-weight 25 N-capped lift only presses the basket
   into the roof strips and drags the whole car to its top stop (readback: q −23.1 →
   +0.5 mm), strip-capped, never out of the pit; released, falls back loaded
8. gate opens: last can binned → spring alone raises the car; the same servo with the
   same 10 N budget slides the basket past the egress line (score latches at the
   0.70 cap, still no success) — identical wrench: loaded jams, unloaded exits
9. near-miss off-pad (95 mm): no success, score ≤ 0.70
10. can still aboard on the pad: cans-binned clause refuses
11. missing bottle: seated clause refuses
12. tipped bottle inside the basket: uprightness refuses
13. can on the deck instead of the bin: refuses
14. settle gate: exact success construct sliding at 0.45 m/s refused while moving
15. rejection audit: success() never True at any step of the battery
16. frames.npz captured in CWD

Execution: forge-only (`forge_client.py submit` / `run`), solve verified
`SIM_GEN_SOLVE: SUCCESS` on `--seed 0` (4 cans) and `--seed 1` (3 cans).
