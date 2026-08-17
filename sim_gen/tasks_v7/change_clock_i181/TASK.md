# change_clock_i181 — set the card clock by exchanging its hour card

## Seed provenance

- **Seed:** `rlbench/change_clock` (RoboVerse
  `roboverse_pack/tasks/rlbench/change_clock.py`): a Franka rotates the crown
  knob of an articulated clock — the whole task is continuous regulation of a
  single revolute DOF (turn until the hands show the target time), driven by a
  recorded trajectory.
- **Kept from the seed:** the goal *semantics* — "make the clock display a
  different, specified time".
- **Changed:** everything about how time is represented and how it is changed.
  There is **no rotary DOF anywhere** in this scene. The displayed time is a
  free rigid **card** seated in a snug display slot, and changing the clock is a
  discrete **select / extract / discard / insert** exchange against a symbolic
  target read off an indicator tile.

## Strategic difference (vs the seed and the corpus)

- **vs the seed:** the seed's plan is *angle regulation of an articulated
  joint* (grasp knob, rotate, stop at the target angle); its code is a
  trajectory/knob controller and its rubric is a joint-angle window. This
  task's plan is *symbolic identification + object exchange*: read the target
  hour from the tile's pip rows, pick the one matching card among three
  distractors (identical except pip count), extract the wrong card from a
  one-card slot, discard it into a container, and insert/seat the right one.
  Different plan, different primitives (pull-out / place-in-container /
  insert-into-slot instead of turn-knob), different rubric (identity-gated
  seating + containment instead of an angle window). The seed's strategy is
  physically a no-op here: rotating the displayed card in place leaves the same
  pip count showing (smoke check 5 proves it scores 0).
- **vs the exemplars/corpus I read:** `pen_holder` (robobench) is many-objects
  → one cup insertion with no identity choice; `hit_ball_with_queue_i77`
  (skyway) is structural bridging plus a passive gravity payload. This task is
  neither: its core is **identity selection under distractors** plus a
  **geometry-forced one-out-one-in exchange** through a snug slot.

## The scene

One kinematic compound **console** (procedural USD, all boxes; visual-only pip
cylinders) carries: a crimson **display stand** with a vertical card slot
(interior 72 x 16 mm for a 60 x 10 x 100 mm card, floor at z 0.060, mouth at
0.110, 45° funnel plates above the mouth), a gray **3-slot rack** (spares), a
blue walled **discard tray**, and a green **indicator pedestal** whose lipped
recess holds the target-hour tile. Four dynamic **cards** are identical ivory
plates except their pip code (rows of three pips near the top, on both faces:
1 row = 3 o'clock … 4 rows = 12); a dynamic **tile** shows the target hour the
same way on its top face.

**Randomization (readback-verified in smoke):** whole-console yaw (±25°) + xy
offset (±5 cm); which card is displayed; which hour is the target (never the
displayed one — the tile is placed on the indicator, unused tiles parked
off-console); the rack permutation.

**Rubric (latched, monotone):** 0.15 old card ever clear of the display stand
+ 0.20 old card ever settled inside the tray + 0.45 target card ever seated in
the display slot (pose within seat tolerances, **upright with pips up**,
still); non-success cap 0.80; score 1.0 iff `success()` = target seated NOW
AND old card inside the tray NOW, both still.

## Execution-order declaration

The order is **geometry-forced**, not rubric-forced: the display slot's 16 mm
gap admits exactly one 10 mm card (`__post_init__` asserts both "two do not
fit" and "one fits with clearance"), so the wrong card must be extracted
before the right one can be seated. Extract-old → discard → fetch-target →
insert. Discard and fetch commute with each other; both must precede nothing
(the tray half and the seat half are independently judged), but insertion
cannot precede extraction.

## Teleport solution (solve.py) — teleports are transport only

- **P0** settle, layout readback (console pose, displayed/target identity, rack
  occupancy), authored-mass assertion, score ~0.
- **P1 extract (contact):** velocity-limited vertical force (0.6 N, 0.25 m/s
  cap, stall escalation) at the old card's CoM slides it up along the slot
  walls until readback shows it clear of the mouth and the display zone.
- **P2 discard (transport + contact):** the airborne card is moved over the
  tray, laid flat, released; it falls, hits the tray floor and settles;
  containment judged on the settled pose.
- **P3 fetch (contact):** same pull slides the target card out of its rack slot.
- **P4 insert (contact):** transport to a hover above the slot mouth with a
  deliberate 3 mm lateral offset + 3° tilt (asserted NOT to read as seated),
  then a gentle velocity-limited down-press; the funnel and slot walls align
  the card under contact until the scene's own `_seated_display` readback
  confirms seating.
- **P5/P6** settle, require `success()`, hold hands-off 3.3 simulated seconds,
  `SIM_GEN_SOLVE: SUCCESS`. `SIM_GEN_SCORE` printed at every phase boundary,
  non-decreasing (latched credit). Verified on forge: **seed 0 and seed 1**.

## Franka embodiment argument

Place a Franka base at console-local **(0.0, −0.50, 0)** (facing the console
front). Reaches: rack slots (−0.11/0/+0.11, −0.12) ≈ 0.40 m; display slot
(0, +0.12) ≈ 0.62 m; tray (−0.26, +0.12) ≈ 0.68 m; indicator is look-at only.
All within Franka's ~0.85 m envelope at working heights 0.02–0.20 m. Every
manipulated object is pinch-graspable: cards are 10 mm thick and protrude
**50 mm** above the display slot mouth and **52 mm** above the rack mouths
(top at 0.160 / 0.138 vs mouths 0.110 / 0.078), leaving a clear two-finger
pinch on the protruding top; rack slots are 110 mm apart so the gripper body
fits between neighbors. Insertion tolerance is realistic: the funnel adds
~3 mm/side capture over the 16 mm gap and the slot admits ~±11° yaw error;
the solve deliberately inserts with 3 mm/3° error and lets the funnel align.
The tile is never manipulated.

## Checks (smoke.py — rejection battery, 14 checks, recorded)

1. settle/no-NaN: reset layout settles (displayed seated, spares upright in
   rack), score ~0, no success.
2. randomization readback: console yaw + xy spreads real over 8 seeds.
3. randomization readback: displayed varies, target varies and never equals
   displayed, target tile really on the indicator (others parked), rack
   permutation varies.
4. null policy: 240 idle steps → score ~0.
5. seed-strategy proxy: displayed card spun 180° in place — rotation cannot
   change the pip count; score ~0 (the seed's plan is the losing move).
6. wrong card: a distractor constructed seated physically seats but earns no
   seat credit (identity is load-bearing).
7. near-miss seat: target lying flat across the slot mouth — z + upright gates
   reject.
8. inverted: target seated pips-down — pips-up gate rejects.
9. discard missing: target seated + old on the base plate → 0.60, no success.
10. latched credit: regressing the seated target leaves 0.60 latched, still no
    success (cap holds).
11. wrong discard: distractor in the tray earns no tray credit.
12. tray near-miss: old card against the tray's outer wall — outside ≠ inside.
13. rejection audit: success() never True anywhere in the battery.
14. final no-NaN.
