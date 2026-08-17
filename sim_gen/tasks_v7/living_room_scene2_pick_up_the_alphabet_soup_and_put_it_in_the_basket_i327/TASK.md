# packed_tote — make room in an occupied tote, then stand the can on its floor

`living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i327` · scene `packed_tote` · env `simgen.packed_tote`

## Seed provenance

Seed task: `roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.py`
(RoboVerse / LIBERO-90). The seed is a pick-and-place: lift the alphabet-soup can off a
table crowded with six other grocery items and release it inside an open basket;
`_terminated` is a bounding-box containment readout on the basket's `contain_region`.

Kept from the seed: the protagonist (a soup-can-sized cylinder, 66 mm x 105 mm), the
container-loading surface story, and clutter that must be understood rather than ignored.

Changed: the clutter moved from AROUND the target to INSIDE the container — the basket
became a high-walled tote whose floor is already occupied by two resident blocks; the
one-way "transport and release" strategy became a **re-packing puzzle** (rearrange the
container's cargo to open a can-sized patch of floor, without evicting anything); the
containment readout became a floor-standing pose window plus a keep-in clause over the
residents; the tabletop scene became a floor scene with a randomized free-yaw tote.

## Strategic difference argument

- **vs the seed:** the seed's whole strategy is one gravity drop — any release inside
  the rim succeeds, and the basket is empty. Here that exact move is constructed in
  smoke check 8 and REJECTED: the parked blocks nearly span the tote's short axis, so
  a dropped can settles PERCHED on top of them (center 113 mm above the floor; the
  success window tops out at 62 mm) with every free floor gap narrower than the can.
  Success requires a strategy with no counterpart in the seed: operate on OTHER
  objects first (slide the pair along the tote to an end wall, or stack one block on
  the other) to open a 72+ mm floor span, and only then lower the can — under the
  constraint that both residents must END inside the tote, so "clear the junk out
  over the wall" also fails.
- **vs the corpus (tasks_v7 siblings surveyed, incl. the other five children of this
  seed):** i38 loads a spring plunger, i177 dumps a lever hopper, i275 threads a drop
  chute, i178 rights a fallen bottle, i149 hangs a hook — every one manipulates the
  TARGET against a mechanism or fixture. **No corpus task makes the container's own
  resident cargo the obstacle**: here the load-bearing interaction is a horizontal
  friction-budget shove of two free bodies inside walls that admit no crosswise
  rotation (interior 116 mm vs block length 105 mm), a pure occupancy/rearrangement
  problem with zero joints, and the goal predicate quantifies over the residents
  (both stowed) as well as the target. Nearest neighbours are the roofed-alcove and
  roofed-dock push tasks; their pushed object IS the target and their containers are
  empty — nothing there has the "make room first, and keep what you displaced" step.

## The scene (numbers)

- Tote: one dynamic 3 kg compound (floor slab 12 mm + four 15 mm walls, 130 mm tall);
  interior 116 x 215 mm; floor μ 0.45/0.40, walls μ 0.25/0.20; zero sleep thresholds.
- Blocks: red 105 x 66 x 60 mm, 0.40 kg. Length 105 vs interior width 116 —
  IN_X − BLK_L = 11 mm, so a block can never turn crosswise; blocks only rearrange
  along the long axis or by stacking. Parked side by side mid-tote (pair offset
  ±12 mm), the three free floor gaps are 28–53 mm — all narrower than the 66 mm can.
- Can: blue cylinder r=33 mm, h=105 mm, 0.35 kg, spawns upright on the open floor at
  0.40±0.03 m from the tote, full-circle arc.
- Shove budget: μ·(m_blk)·g ≈ 3.5 N per block (both ≈ 7 N when pushing the pair) —
  inside a Franka's envelope; ground friction pins the 3.75 kg loaded tote (~17.7 N).
- ~20 `__post_init__` asserts pin the geometry story: gaps blocked at spawn, floor
  window excludes ground-standing / perched / lying cans, consolidated span clears
  the gate with margin, stacked block center inside the stow window, spawn clearance.

## Rubric

`success()`: blue can upright (axis within 15°) with base ON THE TOTE FLOOR (tote-frame
x/y/z windows; z window 44–62 mm excludes both the perch at 113 mm and lying at 33 mm)
AND both red blocks stowed inside the tote (loose windows incl. stacking) AND can,
blocks and tote settled AND the tote upright.

`score()` (latched monotone): 0.15 once the can has been inside the tote with the tote
upright (L1) + 0.25 once a ≥72 mm floor span is open with both blocks stowed and
settled (L2, conservative oriented-box support-function span readback), capped at
0.40; 1.0 iff success. Null policy ~0 (can spawns outside; span blocked by
construction).

## Solution (solve.py) — teleport for transport only

- P0 settle + layout readback; assert span blocked, score 0.
- P1 TRANSPORT: teleport the can once to a hover pose over the tote (upright, zero
  velocity) — what a pick-and-carry delivers. Score 0.
- P2 hands-off drop: gravity lands it PERCHED on the blocks (the naive seed strategy;
  asserted NOT success, not in the floor window). Score 0.15 (L1 enter credit).
- P3 set-aside: the can is transported back out to the floor. Score 0.15.
- P4 MAKE ROOM by applied force: feedforward + velocity-servo shove
  (F = clamp(3.5 + 30·(0.06 − v), 0, 8) N, ff escalating to 6 N on stall) drives the
  trailing block into the leading one and the pair into the far end wall; a runtime
  force-frame probe (measured progress at step ~180, rollback + mode switch) guards
  against pods that rotate wrenches with the body. Span readback exits at ≥76 mm.
  Score 0.40 (L2).
- P5 seat: free-interval readback locates the cleared slot center; the can is
  released upright 6 mm above the floor inside the open top — gravity seats it.
  success() True, then 3.3 s hands-off persistence before `SIM_GEN_SOLVE: SUCCESS`.

Scores printed at phase boundaries are non-decreasing: 0 -> 0 -> 0.15 -> 0.15 -> 0.40 -> 1.0.

## Embodiment argument (single Franka, parallel jaw)

- Grasp: the can is 66 mm across — inside the ~80 mm jaw span; it spawns upright on
  open floor with clearance on all sides (arc 0.37–0.43 m from the tote).
- Make room: the blocks sit inside a 130 mm-deep open-top tote; the natural motion is
  to reach in from above with the closed gripper and push a block face horizontally —
  3.5–7 N, well inside the arm's envelope; the 11 mm side clearance means the block
  cannot jam crosswise, and the walls guide the slide. Stacking one block on the
  other (grasp its 66 mm width, lift 60 mm) is an equally valid stow the rubric
  accepts.
- Place: the cleared span is ≥72 mm vs the 66 mm can, and the tote's open top is
  116 mm across — the gripper can lower the held can to just above the floor before
  opening, then retract straight up past the walls.
- Base pose: tote jitter ±5 cm with free yaw plus the 0.40 m can arc keep everything
  inside a ~0.5 m disc; a Franka based ~0.5 m from the tote reaches the can arc, the
  tote interior, and the shove stroke from above at elbow-up configurations. All
  interactions are from the open top; nothing reaches under an overhang.

## Execution order

1. `scene.py` (registers `packed_tote` + `simgen.packed_tote`)
2. `python -m simgen_tasks.<task>.solve --headless [--seed N]` — prints
   `SIM_GEN_SCORE` at phase boundaries and `SIM_GEN_SOLVE: SUCCESS`.
3. `python -m simgen_tasks.<task>.smoke --headless` — rejection battery, saves
   `frames.npz`, prints `SIM_GEN_SMOKE: ALL PASS <n>/<n>`.

## Smoke battery (smoke.py)

1. settle/no-NaN: blocks parked flat mid-tote, can upright outside, span BLOCKED
2. baseline score ~0, no success
3. mass readback: tote/block/can masses match the cfg (custom-spawner guard)
4. randomization readback: tote xy + free yaw vary across seeds
5. randomization readback: pair parking offset + can position vary
6. null policy: ~0 after 300 idle steps
7. carried-not-placed: can posed at the hover pose — no credit
8. SEED STRATEGY: drop into the container -> PERCHED ON THE BLOCKS (113 mm above
   the floor, span still blocked), enter credit only, no success
9. side-gap drop: released over the WIDEST free gap (readback < 66 mm) — cannot
   reach the floor -> rejected
10. room-readout real: blocks teleport-consolidated to one end -> span readback
    crosses the 72 mm gate, L2 fires, 0.25 credit, NOT success
11. lying-in-slot: can laid FLAT on the genuinely cleared floor -> rejected
12. acceptance construct: can RELEASED upright just above the cleared floor —
    gravity alone seats it -> success TRUE
13. keep-in clause: one block teleported OUT of the tote while the can stays
    seated -> success flips FALSE
14. block returned STACKED on its partner (stacking is a legal stow) -> success
    returns TRUE
15. settle gate: the seated can kicked and judged immediately -> NOT success
16. wrong object: a RED BLOCK stood on end in the slot windows (same height band
    as a standing can), blue far outside -> rejected
17. rejection audit: success never True outside the two acceptance probes
18. final no-NaN

## Verification record

- solve: forge (RTX-4090 pod), seeds 0 / 1 / 2 — all `SIM_GEN_SOLVE: SUCCESS`, rc=0.
  Scores non-decreasing 0 -> 0 -> 0.15 -> 0.15 -> 0.40 -> 1.0; seeds 0/1 shove +y,
  seed 2 shove −y (both directions exercised); cleared span 76.1–76.5 mm; seated can
  base on the floor (center 52–53 mm), 3.3 s persistence.
- smoke: forge — `SIM_GEN_SMOKE: ALL PASS 18/18`, rc=0, 227 frames saved to
  `frames.npz`. Seed-strategy drop perches at 113 mm (floor window tops at 62 mm);
  widest side gap 45.3 mm < 66 mm can; consolidated span 76.0 mm ≥ 72 mm gate;
  keep-in clause flips success both ways; wrong-object red block rejected.
