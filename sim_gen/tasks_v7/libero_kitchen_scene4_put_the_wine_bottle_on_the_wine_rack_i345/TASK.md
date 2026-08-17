# libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_i345 — Repair the Rack, then Rack the Bottle (scene `missing_rail_rack`)

A floor-standing DISPLAY RACK is delivered BROKEN: two side towers carry ONE fixed
far rail (with two dark chock blocks marking the display saddle) — but the NEAR
rail is MISSING. On each tower top, at the near-rail position, an empty open-top
BRACKET SLOT (45 mm wide, between two short guide walls) waits for it. The missing
CROSSBAR (25 × 380 × 25 mm wooden bar) lies on the floor on one side; a green
wine bottle (55 mm body, 20 mm neck) stands on the floor on the other. Success =
the crossbar SEATED in both bracket slots (spanning the towers, top level with the
fixed rail) AND the bottle resting HORIZONTAL bridging the two rails, centred on
the chock saddle, settled. The bottle's goal x-window lies entirely on the near
side of the fixed rail: laid there before the repair, it is supported on ONE side
only and tips through the missing-rail gap to the floor — the repair must come
first, and the rubric additionally makes it a conjunct of success.

## Provenance

- **Seed:** `libero_90/libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack.py`)
  — pick THE wine bottle and rest it ON TOP of THE rack; success is a bbox test
  over the finished rack's top region. The rack is complete scenery; one object,
  one placement, support surface given.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene
  `missing_rail_rack`, env `simgen.missing_rail_rack`, robot `"null"`),
  `solve.py` (teleport solution), `smoke.py` (rejection battery), all procedural
  geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the goal SUPPORT SURFACE DOES NOT EXIST at reset. The seed's
  plan — carry the bottle to the rack and lower it on — is a single placement onto
  given scenery; here that exact plan, transplanted verbatim, is what smoke check
  6 physically constructs (bottle released at the EXACT goal pose, bar still on
  the floor) and gravity itself rejects: one-sided support, the bottle tips
  through the gap and lands on the floor (score capped 0.105). The solver must
  first FETCH AND INSTALL a rack component (drop the crossbar into two bracket
  slots) to CREATE the surface, then execute a two-rail bridge placement into a
  chock saddle — a two-object, order-forced construction plan with a conjunct
  rubric (`bar_seated AND bottle_racked AND settled`), versus the seed's
  single-object bbox test. Bridging the bottle onto a bar resting anywhere else
  on the rack also fails: the repair is load-bearing in the rubric, not just as
  scaffolding (smoke check 7).
- **vs the corpus tasks read this session:** `i90` (same seed family) inverts the
  support relation — its bottle ends SUSPENDED by its lip from rails that exist
  from the start, the challenge being threading kinematics plus color binding;
  here the bottle ends supported from BELOW (like the seed), but the support has
  to be BUILT first — no threading, no color binding, and a second manipulated
  object that becomes rack structure. `stack_wine_i48` is epistemic (hidden mass
  probing, trivial placement); everything here is visible and the challenge is
  construction order. No task read this session requires installing a structural
  component of the goal fixture before the goal placement is physically possible.
- **Execution order (declared):** repair-then-rack is geometrically forced (the
  goal x-window |x| ≤ 25 mm lies entirely on the near side of the fixed rail at
  x = −70 mm, asserted in cfg), but the rubric hard-codes no step order — any
  trajectory that ends settled with the bar seated and the bottle bridging the
  saddle wins. There is no hidden second ordering constraint.

## Randomization (per episode, verified by READBACK in smoke)

Rack xy jitter ±3 cm + yaw 180°±12°; Bernoulli side sign: the bar spawns on one
side of the centreline and the bottle on the OTHER (always opposite, verified),
each with xy jitter ±2.5/±3 cm; bar yaw jitter ±25°; bottle free yaw.

## Rubric

`success()` iff, judged live on physical settled poses in the RACK frame:

- BAR seated: |x − 70 mm| ≤ 14 mm, |y| ≤ 40 mm, |z − seat| ≤ 8 mm, long axis
  within 8° of the rack y-axis (rejects the floor, the tower tops between the
  rails, and half-seated poses);
- BOTTLE racked: |x| ≤ 25 mm, |y| ≤ 30 mm, |z − rest| ≤ 12 mm (rest = rail top
  + body radius — rejects floor stands 78 mm below and anything not on the rail
  plane), axis within 15° of horizontal and 25° of square-across-the-rails;
- settled: both bodies |v| < 0.05 m/s, |ω| < 0.60 rad/s.

`score()` (latched every physics substep in `post_step`): `0.10 ×` bar lift (bar
origin ever above 0.08 m) `+ 0.30 ×` bar seat (ever seated) `+ 0.10 ×` bottle
lift (origin ever above 0.155 m; stand is 0.095) `+ 0.10 ×` cradle (bottle ever
in loose saddle bands WHILE the bar is seated — bridging a mis-laid bar earns
nothing), capped at 0.60; exactly 1.0 iff `success()`. Doing nothing scores ~0;
the seed strategy (bottle at the goal spot without the repair) is capped at 0.105.

## Teleport solution (`solve.py`) — transport only, both rests earned by contact

- **P1 — repair:** ONE pose write stages the crossbar level above the two bracket
  slots (bottom just above the guide-wall tops, a deliberate 4 mm lateral offset
  and 1.5° yaw error), zero velocity. Released, it falls ~4 cm between each
  slot's guide walls onto the two tower tops and settles SEATED — drop-in, wall
  guidance and the rest on the slot floors are pure contact (a square retry
  exists but was never needed on the forge).
- **P2 — rack:** ONE pose write stages the bottle horizontal over the saddle
  (axis along rack +x, +6 mm y offset, body underside 20 mm above the rail
  plane), zero velocity. Released, the body drops onto the fixed rail AND the
  freshly installed crossbar simultaneously and settles bridging both, arrested
  by the chock saddle — the rest is earned through contact with the rail the
  solver itself installed.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing (0.00 → 0.40
  → 1.00), ≥ 3.3 simulated seconds hands-off persistence, then `SIM_GEN_SOLVE:
  SUCCESS`. **Verified on the forge: seeds 0, 1 and 2, all SUCCESS, provably
  distinct by stdout readback** — seed 0: rack (0.591, +0.016) yaw 191.3°, bar
  side − at (0.258, −0.163), bottle at (0.290, +0.155); seed 1: bar side +
  (rack-local y +0.171), bottle at −0.174; seed 2: bar side − (rack-local
  −0.142), bottle at +0.194.

## Embodiment sanity (single-arm Franka feasibility)

Base at the origin: bar spawn at ~0.31 m radius, bottle spawn at ~0.33 m, the
bracket slots at ~0.53 m and the saddle at ~0.60 m, all working heights ≤ 0.21 m
— inside a Franka reach envelope, and the empty brackets are on the NEAR side of
the rack by construction. Per-object contact strategy: (1) the 25 mm square bar
grasped across its section at the centre with the 80 mm jaw (55 mm margin,
0.25 kg), carried level over the rack, lowered until the ends enter the two
open-TOP bracket slots (10 mm of x-play per side, ±3.8° yaw play — the solve
seats it with a worse 4 mm + 1.5° error), released just above the wall tops: the
guide walls funnel the drop and the hand stays ABOVE the slots in free air the
whole time. (2) The bottle side-grasped on the 55 mm body (25 mm margin,
0.45 kg), wrist-rotated to horizontal, lowered onto the saddle from the open sky
above the rails (±11.5 mm of y-play between the chocks, ±25 mm of x-play across
the rails); release is a plain jaw-open with the body already resting on both
rails, fingers clearing in free air above the rail plane. The rack is kinematic
and cannot be disturbed; nothing overhangs either approach path.

## Checks (`smoke.py` — rejection battery, 13 named checks, ALL PASS on the forge)

1. settle: states finite; bar lying on the floor, bottle standing, on OPPOSITE
   sides of the centreline; all still.
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization A: bar/bottle side sign flips across 8 seeded resets (always
   opposite each other, readback), bar spawn yaw spread > 5° (measured 40°).
4. randomization B: rack yaw spread > 2° and rack xy jitter > 4 mm (readback).
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: bottle released at the EXACT goal pose with the crossbar still
   on the floor → one-sided support, TIPS THROUGH the missing-rail gap, lands on
   the floor (z 0.027) → NOT success, score ≤ 0.105 (the order is physical).
7. bar conjunct: bar dropped onto the open tower tops BETWEEN the rail positions
   (resting at seat HEIGHT but not in its brackets) + bottle laid level bridging
   fixed rail + mis-laid bar INSIDE the goal window → `bottle_racked` reads True
   yet NOT success, score ≤ 0.205 (the repair is a conjunct, not scaffolding).
8. near-miss saddle: bar GENUINELY drop-seated (alone: not success, score 0.40)
   + bottle resting across BOTH rails at height but OFF the chock saddle
   (|y| ≈ 95 mm) → NOT success, score ≤ 0.505.
9. latched credit: teleporting that bottle back to the floor leaves the latched
   score unchanged (0.500), still no success.
10. under the cradle: bottle standing on the FLOOR directly below the saddle →
    score ~0, no success (right (x, y), wrong relation).
11. transient motion: bar seated + bottle in the success pose but MOVING
    (0.5 m/s along the rails, judged one step after injection) → the settle gate
    rejects; probe removed unsettled.
12. rejection audit: success() never True at any judged point in the battery.
13. final no-NaN. Plus `frames.npz` recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest:
the goal x-window lies entirely on the near side of the fixed rail (seed strategy
tips through by construction), the body bridges both rails across the whole
window, a single-rail balance is outside the window, the slot admits the bar with
real play and the walls retain it, the bar spans both towers even at the y-window
edge, the chock saddle admits the body with play while a rest outside the chocks
fails the y-window, the racked neck clears the chocks and the guide walls, both
graspable bodies fit the 80 mm Franka jaw, and the lift latches sit strictly
between the floor states and the goal states.
