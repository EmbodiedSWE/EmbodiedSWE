# libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i2 — BowlDecantScene (`simgen.bowl_decant`)

Empty the loaded white bowl into the teal rimmed dish — a gravity POUR, not a
placement — then park the emptied bowl UPSIDE-DOWN inside the brown drying tray on
the other side. Pour-then-park order is encoded in physics: inverting first dumps the
balls where they do not belong.

## Seed provenance

- **Seed task**: `libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate`
  (RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate.py`)
  — "put the white bowl on the plate". A kitchen scene with a white bowl, a plate and
  a microwave distractor; the plan is ONE rigid pick-and-place, judged purely by the
  bowl's own resting pose: `xy_distance(bowl, plate) < 0.06 and 0 < z_bowl - z_plate
  < 0.03`.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| payload | the bowl itself — rigid, empty | the bowl's CONTENTS: 2–4 dynamic balls; the bowl is a tool (container) whose final pose is a *separate* sub-goal at a *different* fixture |
| core act | set the bowl down upright on the plate | POUR: tip the bowl over the dish and let **gravity** carry every ball over the lip into the basin — the hand never touches a ball, the last centimeters of every transfer are ballistic |
| final bowl pose | upright on the plate | INVERTED (rim-down) inside the drying tray on the opposite side — the seed's orientation is a failure here |
| ordering | none — one placement | **total order, physically encoded**: invert-before-pour spills the balls on the ground (smoke 10 shows parking first earns nothing); a spilled ball cannot be un-spilled |
| the seed's end state | = success | expressible and scored ~0: the loaded bowl set ON the dish leaves every ball "in the bowl", and the **containment-exclusion clause** says a ball inside the bowl is never "in the dish" (smoke 6) |
| perception | fixed layout | dish and tray **swap sides at random**, xy jitter on all three movables, tray/bowl yaw, **ball count 2–4 sampled per episode** — count what you see |
| assets | LIBERO USD kitchen assets | 100 % procedural (compound spawners): dynamic octagonal-cup bowl, kinematic 12-gon rimmed dish, kinematic lipped tray, 4 dynamic spheres |
| judging | one bbox-style pose check on the bowl | latched partial credit (lift / first-ball / all-balls / park-after-pour) + live success: every present ball settled in the dish AND the bowl at rest inverted in the tray |

Code structure shares nothing with the seed: `@SCENES.register` BaseScene with three
compound spawners, containment-exclusion predicate, order-aware latches in
`post_step` (`_park |= _mall & bowl_parked()`), `register_env(..., robot="null")`.

## Why strategically different

The seed's entire skill is *grasp one rigid object, set it down near a target pose* —
tolerance-loose, orderless, judged by the transported object's own resting position.
Here that plan is worth nothing: the object the seed transports (the bowl) must end
up in a DIFFERENT place, in the OPPOSITE orientation, and only AFTER its contents
have been transferred. Smoke check 6 constructs exactly the seed's end state — the
white bowl set neatly on the dish — and the rubric scores it ~0, because success is
about the *contents*, and contents inside the bowl are excluded from "in the dish" by
construction. What the solver must bring instead is (1) **a pour**: the transfer is
performed by gravity through a tilted container, not by per-object placement — no
gripper ever touches a ball; (2) **an irreversible execution order** encoded in
physics, not rubric fiat — inverting the bowl for parking while it is still loaded
dumps the balls on the open ground (smoke 10), and the dish's 30 mm rim means a
correctly poured ball stays poured (smoke 5: a 3×-weight shove cannot push a settled
ball out); (3) **perception**: which side the dish is on, and how many balls exist,
change every episode (smoke 2, 3).

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; readback layout (dish side ±, dish/tray/bowl xy, present-ball
   mask); baseline score 0, no success.
2. **P1** (teleport = transport only): "hold" the loaded bowl by writing its root pose
   in ≤2 mm / ≤1.5° per-step increments — lift straight up, carry to the hold point
   over the dish. Balls are NEVER written: they ride inside on real contacts. Lift
   latch → score 0.10.
3. **P2** POUR: ramp the held tilt to 135°; gravity takes every ball over the lip into
   the basin, the rim catches them (a wrist-wiggle 120°↔150° is armed for hang-ups).
   First-ball + all-balls latches → score 0.55.
4. **P3** PARK: carry the emptied bowl to the tray while rotating 135°→180°, descend
   to 8 mm above the pad, RELEASE — gravity seats the rim on the pad inside the lips.
   Park latch + live success → score 1.0.
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.0000 → 0.10 → 0.55 → 1.0 → 1.0.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m). Bowl
start ≈ 0.16 m ahead; dish/tray stations at (0.36, ±0.16) — everything inside a
comfortable dexterous shell; pour hold height 0.155 m.

- **Grasping the bowl** (Ø ~118 mm cup, 60 mm tall, 6 mm walls, 150 g): a rim pinch —
  fingers straddle one 6 mm wall from above (opening 80 mm ≫ wall + clearance). The
  octagon gives eight flat wall segments, so any approach yaw within ±22.5° of a facet
  normal works; bowl yaw is randomized but the facets make the grasp yaw-tolerant.
- **The pour**: hold the rim-pinched bowl over the dish (mouth Ø 160 mm — a generous
  target for a Ø ~96 mm bowl interior) and rotate the wrist to ~135°. Joint 7 is
  continuous ±2.9 rad: the tilt is a pure wrist roll about the pinch axis, no regrasp.
  The balls (Ø 24 mm, 10 g) are never touched — gravity and the dish rim do all fine
  positioning; there is no insertion and no precision placement.
- **The park**: keep rotating to 180° while carrying to the tray (total 180° is within
  a single wrist revolution from the pinch), lower to ~8 mm above the pad, open the
  gripper. The 175 mm tray vs the ~118 mm bowl leaves ~28 mm per side; the 6 mm lips
  only need to stop residual sliding, which the release height keeps negligible.
- **Payload/forces**: 150 g bowl + ≤40 g balls — trivial. The only contact-rich moment
  is the rim pinch; the dish and tray are kinematic (heavy fixtures), so incidental
  contact cannot move a goal frame.

## Execution order (declared)

`pour the balls into the dish → park the bowl inverted in the tray`. The order is
enforced by physics, not rubric fiat: inverting the loaded bowl anywhere but over the
dish dumps the balls on the ground (smoke 10 — park-first with grounded balls earns
score ~0 and the park latch refuses because `_park` requires `_mall` first), and a
poured ball is locked in by the 30 mm rim (smoke 5).

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0`: SUCCESS, scores 0.0000/0.1000/0.5500/1.0000/1.0000 (33 s).
- `solve --seed 1` (different sides/jitter/ball count): SUCCESS, same monotone scores.
- `solve --seed 2` (3 balls present): SUCCESS — three seeds, wiggle never needed.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 11/11**, frames.npz saved:
  1. settle/no-NaN; every present ball in the bowl, none in the dish; score 0
  2. randomization readback: dish/tray/bowl xy differ AND the dish/tray side swaps
  3. ball-count subset: ≥ 2 distinct present counts in {2,3,4} over 10 resets
  4. null policy: 240 idle steps, score ~0, no success
  5. RIM INTERLOCK: settled ball shoved outward at 3× weight for 1.5 s, stays in
  6. SEED STRATEGY: loaded bowl set upright ON the dish → containment-exclusion
     voids every "in the dish" claim → score ~0, no success
  7. wrong orientation: balls poured, bowl parked UPRIGHT in the tray → no success
  8. near-miss pour: bowl parked, one ball on the ground just outside the rim → no
     success
  9. near-miss park: balls in the dish, bowl inverted on the ground beside the
     tray → no success
  10. park-before-pour: bowl inverted in the tray, balls on the ground → park latch
      stays unset, score ~0, no success
  11. video frames.npz saved

## Files

- `scene.py` — BowlDecantScene + bowl/dish/tray compound spawners + rubric; registers
  `simgen.bowl_decant`.
- `solve.py` — teleport-transport + gravity-pour certificate (`--seed N`).
- `smoke.py` — 11-check rejection battery + video.
