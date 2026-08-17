# put_bottle_in_fridge_i320 — ChestSwapScene (`simgen.chest_swap`)

**Seed:** `rlbench/put_bottle_in_fridge` (RoboVerse
`roboverse_pack/tasks/rlbench/put_bottle_in_fridge.py`): a Franka opens a fridge's
revolute front door and stands a bottle upright inside an EMPTY interior. One
grasp-carry-place; nothing else has to move; nothing is re-closed.

## What changed and why it is strategically different

**Slide open the chest cooler's captive top lid, evict the stale red can from the
single bottle well into the discard bin, stand the amber bottle into the vacated
well, then slide the lid fully shut.**

1. **Access mechanism: captive sliding TOP lid, not a revolute door.** The chest is
   top-loading; its blue lid rides on the wall tops inside an overhead rail channel
   (risers + inward retaining lips, 3 mm lateral / 4 mm vertical play, shelf beams +
   posts carrying it where it overhangs, end fences both ways). It translates along
   one axis only and can never be lifted off or swung. Jointless — the channel
   geometry is the mechanism, so the randomized chest pose carries it with no stale
   joint anchors.
2. **The goal slot starts OCCUPIED — the task is a swap, not a put.** The deck inside
   the chest has ONE square bottle well (84 mm) and a stale red can (66 mm) already
   stands in it. The well physically holds one cylinder: max in-well centre
   separation ≈ 30 mm < 63 mm = r_can + r_bottle (asserted in smoke), so the bottle
   can NEVER seat until the can is evicted. The seed's fridge is empty; its "red
   item" siblings use the can as a pick-the-wrong-object distractor — here the can is
   a mandatory sub-goal with its own disposal target (the green bin).
3. **The episode must be CLOSED UP again.** Success requires the lid slid back shut
   over the seated bottle. The seed (and both sibling tasks) end the moment the
   bottle is placed; neither ever re-actuates the access mechanism.
4. **Vs. sibling i179 (hang bottle by cap flange on a slotted rail, slide rearward)
   and i311 (lay bottle into a captive sliding rack, push the rack through a
   letterbox mouth):** both siblings transport the payload *aboard/along* a fixture
   and judge a lying/hanging pose. Here the fixture (lid) never carries the payload —
   it *gates* access twice (open first, close last), the bottle ends UPRIGHT like the
   seed but down inside a well, and the core novelty is the forced evict-then-insert
   swap with two placements (bottle→well, can→bin) that must both hold at once.

**Execution order is forced by geometry, not decreed:** closed lid ⇒ no access to the
well (from anywhere); occupied well ⇒ bottle cannot seat; so any successful episode
is `open → evict → (bin) → seat → close`. The rubric latches are chained in exactly
that order (`evicted` counts only after `opened`, `binned`/`seated` only after
`evicted`), so teleport shortcuts that skip a stage earn nothing downstream.

## Scene

- **chest** (kinematic compound, randomized yaw ±9° + xy ±2 cm): 244 mm square,
  walls to 230 mm; base plate top = well floor (10 mm); deck ring at 80 mm with the
  central 84 mm square well; overhead rail channel for the lid (+y = slide-open
  direction, stroke u ∈ [−0.004, 0.271], well fully exposed past u = 0.177).
- **lid** (dynamic compound, 0.40 kg): 228 × 270 × 12 mm plate + yellow handle bar
  (160 × 22 × 22 mm) near the leading edge. Starts closed (u = 0).
- **bottle** (dynamic, 0.35 kg): amber body r30 × 150 + neck r13 × 40 (190 mm
  standing); starts upright on the ground in a randomized band.
- **can** (dynamic, 0.25 kg): red r33 × 130; starts standing IN the well (60 mm
  proud of the deck), xy-jittered.
- **bin** (kinematic, randomized band + yaw ±30°): green open-top box, inner
  160 mm square, rim 150 mm.

## Rubric

`score = 0.15·opened + 0.10·evicted + 0.20·binned + 0.30·seated` (all latched,
order-chained, capped 0.85), and exactly **1.0 iff `success()`**: bottle seated
upright in the well (planar ≤ 28 mm, base in [4, 30] mm, tilt ≤ 10°) AND can inside
the bin (bin frame, orientation-free, z band rejects rim-perching) AND lid closed
(|u| ≤ 10 mm, sane in channel) AND everything still. Null policy ≈ 0.

## Teleport-solution outline (solve.py)

P1 **open** — bang-bang world-frame force at the lid CoM along the chest +y axis
(4→12 N stall escalation, v ≈ 0.10 m/s) to u ≥ 0.24. P2 **evict** — vertical hoist
force ≈ 1.5 mg (vz-gated, lateral damping) lifts the can straight up out of the well
to z ≥ 0.30 — possible only through the open aperture. P3 **bin transport** — ONE
pose write, free air (above chest) → free air 28 cm above the bin (outside the bin
band, earns nothing); gravity drops it in. P4 **seat transport** — ONE pose write
carries the bottle from its ground spawn to a hover 95 mm above the well floor
(outside the seat band); gravity drops it 85 mm; it self-centres and settles
upright. P5 **close** — same force drive back to |u| ≤ 6 mm (+ trim). Then ≥ 3.3 s
hands-off persistence before `SIM_GEN_SOLVE: SUCCESS`. Scores print at each phase
boundary and are asserted non-decreasing (0 → 0.15 → 0.25 → 0.45 → 0.75 → 1.0 → 1.0).
Teleports are transport-only; the lid never receives a pose write.

## Embodiment argument (single Franka, parallel-jaw gripper)

One base pose at the world origin serves the whole episode; every contact point lies
0.33–0.62 m from the base, within the ~0.855 m reach envelope.

- **Lid:** grasp the 22 mm handle bar from above (fingers straddle it in y; the bar
  spans x ∈ ±80 mm, well clear of the lips at |x| ≥ 107 mm) and drag horizontally —
  a 1-DOF pull needing ~2–4 N against sliding friction. Handle height 242–264 mm.
- **Can:** it pokes 60 mm above the deck; pinch its upper barrel (66 mm < 80 mm jaw
  span) through the 220 mm top opening — the hand descends ~90 mm below the wall-top
  plane into a 220 × 215 mm aperture, ample for the ~90 × 60 mm hand — then lift
  straight up and carry to the bin (an open drop).
- **Bottle:** pinch the 26 mm neck; lower it upright through the same aperture until
  the base rests on the well floor; the 12 mm radial well clearance self-centres it.
  At release the hand is ~40–80 mm below the wall-top plane — clear on all sides.
- **Order as above** — the geometry admits no alternative.

## Checks (smoke.py — 16)

1–2 settle/finite + score ≈ 0; 3–4 randomization readback (chest/bin/bottle/can
jitter, spawn separations); 5 null policy; 6 seed-strategy A (stand it on the deck
"inside" → only open credit; deck base statically outside the seat band); 7
seed-strategy B (carry-drop onto the CLOSED chest → lands on the lid, nothing); 8
occupied-well rejection + static exclusion math; 9 no-open teleport shortcut (can in
bin but chain latches nothing); 10 reversed swap (bottle in bin → nothing); 11 wrong
disposal (can beside the bin → 0.55, no success); 12 clear-path force close over the
seated bottle (actuator-moved readback; still no success on the unbinned can alone);
13 near-miss lid 30 mm short (0.75, no success); 14 lid-path knockover (deck-standing
bottle knocked over by the real closing drive — can't be laundered into success); 15
rejection audit (success never True in the battery); 16 final no-NaN.
