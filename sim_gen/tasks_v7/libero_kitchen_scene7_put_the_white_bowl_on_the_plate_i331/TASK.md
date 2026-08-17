# libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i331 — BalanceServeScene (`simgen.balance_serve`)

Serve the loaded white bowl LEVEL: seat it (cubes and all) on the plate bolted to
one end of a 440 mm balance board, then COUNT the copper cubes it carries (0–3,
sampled per episode) and place the 500 g steel weight at the matching tick mark
on the weighing half — radius = 0.36 × load mass ∈ {54, 90, 126, 162} mm — so the
spring-centered board settles inside its ±3.5° level band. The seed's whole plan
(set the bowl on the plate) is expressible, latches 0.30, and leaves the board
grounded on its deck stop at −5.75°: failure.

## Seed provenance

- **Seed task**: `libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate`
  (RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate.py`)
  — "put the white bowl on the plate". A kitchen scene with a white bowl, a plate
  and a microwave distractor; the plan is ONE rigid pick-and-place, judged purely
  by the transported object's resting position: `xy_distance(bowl, plate) < 0.06
  and 0 < z_bowl - z_plate < 0.03` — the plate is a passive surface, the goal a
  pose predicate on the moved object itself.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| the plate | a passive surface on a fixed table | bolted to one end of a **1-DOF balance board** (free Y-revolute pivot + gentle centering spring, 2.2 N·m/rad); putting the bowl on it **tips the board onto its deck stop** — the seed's success pose *causes* this task's failure state |
| goal predicate | bbox pose of the moved object | a **live force equilibrium**: bowl seated AND every cube accounted for AND the board at rest inside ±3.5° — an emergent state of the whole mechanism, judged LIVE (success is LOST if the counterweight is later removed; smoke 10) |
| reasoning | none — one target pose | **count-then-compute**: the bowl carries 0–3 cubes sampled per episode; the correct counterweight radius is a *function of the perceived count* (r = 0.36·m_load ∈ {54, 90, 126, 162} mm, tick marks every 30 mm). One cube of miscount = 36 mm = 4.6° residual, OUTSIDE the 3.5° band (smoke 6: 50 mm off ⇒ grounded) |
| payload | the bowl itself, rigid, empty | the bowl is a **loaded container**: its cubes ride on real contacts through the whole carry (never pose-written) and must ALL still be inside at the end — plus a **conservation clause**: absent cubes must stay untouched in the far rack |
| distractor | passive microwave | a **40 g pale dummy block** that is *torque-insufficient by physics*: at its best radius it musters 0.075 N·m against a ≥0.26 N·m imbalance — the board stays grounded (smoke 7, −5.74°) |
| cheat surface | none | propping the plank with a cube on the deck **physically levels the board** (+2.45°, inside the band — smoke 9) and is refused by the cube-accounting clause, not by geometry fiat |
| perception | fixed layout | weight/dummy **swap sides** at random, xy jitter + yaw on both, bowl xy jitter + yaw, **cube count 0–3** re-sampled per episode — count what you see, find which block is the steel one |
| assets | LIBERO USD kitchen assets | 100 % procedural: kinematic pedestal (deck = tip stop), plank with spawn-authored revolute joint + spring drive (jointed-pair collision re-enabled so the stop is real), octagonal cup bowl, 3 copper cubes, 2 blocks |
| judging | one pose check | latched partial credit (place 0.25 / seat 0.30, cap 0.55) + live success = 1.0 |

Code structure shares nothing with the seed: `@SCENES.register` BaseScene with three
compound spawners, beam-frame predicates (`tilt()` from the plank's +x world z,
`weight_on_half()`, `bowl_seated()`), a `cubes_ok()` conservation predicate
(`where(present, in_bowl, in_rack).all()`), latched credit in `post_step`,
`register_env(..., robot="null")`.

## Why strategically different

The seed's entire skill is *grasp one rigid object, set it down near a target
pose* — the goal is a pose predicate on the transported object, the plate inert.
Here that exact plan is smoke check 5: the loaded bowl set neatly on the plate,
nothing else — the un-counterweighted board tips onto its deck stop at −5.75°,
score caps at 0.30 latched, no success. What the solver must bring instead:
(1) **counting → parameter binding**: the episode's cube count (0–3, randomized)
determines the only correct counterweight radius out of four tick-marked
candidates; the mapping (torque balance: r·m_w = 0.18·m_load) must be *computed*,
and the level band is tight enough that one miscounted cube lands outside it;
(2) **acting on a compliant 1-DOF mechanism**: both releases (weight, then bowl)
load a live plank whose angle is produced by gravity, spring, and the deck stop —
the judged tilt is never written, and success is judged LIVE: it drops if the
equilibrium is disturbed (smoke 10); (3) **conservation**: success requires every
cube accounted for — present cubes in the bowl (they must survive the carry),
absent cubes untouched in the rack — which also closes the one physically-working
cheat (a cube propping the plank levels it, +2.45°, and is refused); (4) a
**physics-rejected decoy**: the pale 40 g dummy cannot level the board from any
radius. Against siblings: i2 transfers *contents by gravity pour* into a passive
dish; i3 does *rotation under contact* into a bayonet mechanism; i331's core is
*counting-conditioned force balance on a compliant mechanism with live judging* —
none of the three shares a goal mechanism, core act, or judged quantity.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.25 s; readback: weight/dummy sides, bowl xy, present-cube mask
   → k, r_bal = 0.18·(0.15 + 0.1·k)/0.5. Calibration assert: empty board rests
   level. Baseline score 0.
2. **P1 COUNTERWEIGHT** (teleport = transport only): carry the steel weight over
   the weighing half and release it 6 mm above radius r_bal — gravity loads the
   board, which tips onto the deck stop (+5.75°, out of the band: partial credit
   only latches `place` 0.25).
3. **P2 SERVE**: lift the loaded bowl (the cubes are NEVER written — they ride
   the octagonal cup on real contacts), traverse high, then descend tracking the
   LIVE plate pose tilt-matched to the tipped board, release 6 mm up. The board
   swings back and — because the torques match — settles LEVEL. Seat latch + live
   success → score 1.0 (tilt −0.60° on the hardest seed).
4. **P3 TRIM** (safety loop, unused on all three seeds): if the residual tilt
   were off, re-carry the weight by dr = k_spring·tilt/(m_w·g), ≤3 times.
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

The board's angle is never written after reset; every judged fact (tilt, seating,
containment, stillness) is an outcome of the two releases. Monotone
`SIM_GEN_SCORE`: 0.0000 → 0.2500 → 1.0000 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m).
Bowl starts ≈ 0.10 m ahead; block stations at (0.28, ±0.22); the board's pivot at
(0.42, 0), plate center at 0.60 m, weighing half spans 0.20–0.36 m — everything
inside the dexterous shell, all manipulation below 0.20 m height.

- **Counting**: a top-down glance into the ~104 mm bowl; the copper cubes are
  30 mm and high-contrast — count, then pick the tick mark (marks every 30 mm on
  the dark-red strip, generous ±27 mm placement window per the spring math).
- **Grasping the weight** (50×50×70 mm, 500 g): a side pinch across the 50 mm
  faces (opening 80 mm > 50 mm); 500 g is trivial payload. The dummy is
  distinguished by color (pale vs dark steel) and size.
- **Grasping the loaded bowl** (octagonal cup, 6 mm walls, total ≤450 g): a rim
  pinch — fingers straddle one wall facet from above; the octagon's eight facets
  make the grasp yaw-tolerant under the randomized bowl yaw. Carried gently
  (the demonstrated carry is slow and smooth for exactly the same reason a robot
  would be: the cubes are loose cargo).
- **Placements**: the weight is released 6 mm above the strip — a drop onto a
  30 mm-pitch tick grid with a ±27 mm tolerance band; the bowl is released 6 mm
  above the plate, whose retaining rim leaves ~8 mm annular clearance around the
  96 mm bowl base. Both are coarse, insertion-free drops; the board under them is
  compliant (spring + stop), which OSC handles well.
- **No step requires touching the board itself**, and the pedestal is kinematic —
  incidental contact cannot move the goal frame.

## Execution order (declared)

`count the cubes → place the steel weight at the matching tick → serve the bowl
onto the plate` (as demonstrated). The physics also admits serving the bowl first
and counterweighting second — what is load-bearing is not the path but the LIVE
end state: the rubric's success is an equilibrium predicate re-evaluated every
step (smoke 10 constructs it, then breaks it by removing the weight — success
drops, only the 0.55 latch remains). Partial credit is order-free by design;
there is no rubric-fiat sequencing to game.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (k=0, r=54 mm): SUCCESS, scores 0.0000/0.2500/1.0000/1.0000/
  1.0000, final tilt −0.27° (21 s).
- `solve --seed 1` (k=1, r=90 mm, weight starts on the other side): SUCCESS, same
  monotone scores, final tilt −0.02°.
- `solve --seed 2` (k=3, r=162 mm, the full 2+1 cube pile rides the carry):
  SUCCESS, final tilt −0.60° — three seeds, the trim loop never needed.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 11/11**, frames.npz (386 × 600 × 960) saved:
  1. settle/no-NaN; EMPTY board rests level (+0.00°, spring calibration);
     present cubes in the bowl; score 0
  2. randomization readback: bowl xy Δ 63.9 mm, weight xy Δ 474.2 mm (3-seed
     max-pairwise), weight/dummy sides seen {−1, +1}
  3. cube-count subset: counts [0, 1, 2, 3] all observed over 10 resets
  4. null policy: 240 idle steps, score 0, no success
  5. SEED STRATEGY: loaded bowl set on the plate, nothing else → board grounds
     at −5.75° (band ±3.5°), seat latches 0.30, no success
  6. wrong radius: weight 50 mm off (k=2: 126→76 mm) → grounded −5.73°; both
     latches, score caps 0.55, no success
  7. dummy decoy: 40 g dummy at its best radius (190 mm, 0.075 N·m vs 0.618 N·m
     load) → board stays −5.74°, no success
  8. missing cube: weight tuned to the REDUCED load → board IS level (+0.01°,
     the check is non-vacuous) but one present cube on the ground → accounting
     refuses success
  9. CUBE-PROP cheat: a present cube stood on the deck props the board to
     +2.45° — physically INSIDE the band — and success still refuses (that cube
     is neither in the bowl nor in the rack); score 0
  10. LIVE judging: constructed success (−0.02°) accepted, then the weight is
      teleported away → board falls to −5.75°, success DROPS, latched 0.55
      remains
  11. video frames.npz saved (386 × 600 × 960)

## Files

- `scene.py` — BalanceServeScene + stand/beam/bowl compound spawners + rubric;
  registers `simgen.balance_serve`.
- `solve.py` — teleport-transport + two-release force-balance certificate
  (`--seed N`).
- `smoke.py` — 11-check rejection battery + video.
