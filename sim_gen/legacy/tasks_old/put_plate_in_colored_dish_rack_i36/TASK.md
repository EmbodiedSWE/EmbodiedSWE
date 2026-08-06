# plate_carousel (`put_plate_in_colored_dish_rack_i36`)

**Env id:** `simgen.plate_carousel` (scene-level, `robot="null"`)
**Seed:** `rlbench/put_plate_in_colored_dish_rack`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_plate_in_colored_dish_rack.py`)
**Tier:** hard — ~6 dependent stages (rotate → insert, three times, on a shared mechanism)

## What changed

The seed is one free placement: a plate on a stand, a static, fully exposed dish rack,
and an instruction naming the colored slot — grasp, carry, insert. Every goal slot is
reachable at every moment; the whole task is a single transport with a color lookup.

Here the colored dish fixture becomes a **machine that time-multiplexes its own goal
slots**:

- The rack is a **carousel**: a free-spinning turntable deck riding a low-friction drum
  (a *physical* bearing — the deck's under-skirt is captured laterally around the drum,
  so it spins freely about one axis but cannot be dragged off; no joints, all contact),
  carrying three color-coded single-plate pockets (walled corrals) at 120° spacing.
- A fixed **shroud** (roof + outer wall) covers the carousel except one 120° **loading
  window** at a per-episode azimuth. Only the pocket currently rotated under the window
  can receive a plate; everywhere else the roof and wall physically block insertion
  (roof underside 35 mm above the deck, seated plates ride beneath it).
- Three colored plates start flat on shuffled floor stands. Goal: each seated flat in
  the pocket of its own color, settled.

## Why strategically different (a different PLAN, not different numbers)

- **Interleaved mechanism actuation vs one-shot transport.** The seed's plan skeleton
  (approach → grasp → carry → insert into the visible colored slot) cannot be executed
  even once for two of the three plates: their pockets are covered. The solver must
  loop *actuate → place → actuate → place → actuate → place*, re-indexing the carousel
  between placements. Nothing in the seed requires touching the fixture at all.
- **The goal set is accessibility-scheduled, not spatially fixed.** Which color can be
  loaded *right now* is a function of mechanism state the solver itself controls, and
  it changes every episode (random deck yaw + random window azimuth + shuffled stands
  ⇒ different index sequence and rotation amounts each episode).
- **Already-achieved subgoals ride the mechanism.** Loaded plates rotate away under the
  roof with the deck; the rubric judges seating **in the deck's body frame** and the
  smoke's carry probe spins the fully loaded carousel 180° with success persisting —
  the seed's world-frame "plate in rack slot" predicate has no analogue of this.
- The seed's own strategy is a tested failing control (see check list): executing
  "carry each plate straight to its colored slot" with the mechanism untouched leaves
  the plates on the shroud roof (score ≤ 0.25, no rotation credit, never success).

Differentiation from claimed sibling axes: i7 claims *precision rotary positioning as
the terminal goal* (knob-to-angle with detents) — here rotation is coarse (±50° window)
and purely instrumental, the goal is still placement; i22 claims a *self-closing
transient-access mechanism needing a prop*; here access never decays, it is spatially
multiplexed and must be re-scheduled per subgoal; i14 claims orientation-selective
aperture insertion — the window here is orientation-agnostic and selects *which goal*
is open, not *how* the object must pass; i15's drawer claims open-then-restore — no
restore stage exists here and the mechanism is a continuous free DOF, not a bounded
prismatic with a terminal state.

## Execution order

Partially ordered, order-of-colors free: the three colors may be loaded in any order
(the oracle uses red→green→blue), but rotation and placement **must strictly
alternate** — no two plates can be loaded without re-indexing the deck in between, and
a placement attempted without indexing fails physically. Declared: order required at
the *stage-interleaving* level, free at the *goal-selection* level.

## Scoring rubric

- `engaged` (latched): ≥ ~29° of cumulative *smooth* deck rotation (per-substep yaw
  jumps don't accumulate) → **0.10**
- **+0.25 per plate** currently seated in its own-color pocket (deck-frame xy ≤ 15 mm —
  honest by construction, corral walls cap the physical offset at 8 mm — z band ±6 mm,
  flat ≤ 15°), settled (plate and deck), **and** `loaded` — a latched through-window
  ENTRY EVENT (seated flips true with per-substep displacement < 5 cm while its pocket
  is within 50° of the window centre). Teleporting a plate into a covered pocket makes
  it seated-by-geometry but never loaded.
- **1.0 iff success()** = all three plates counted. Ladder: 0 → 0.10 → 0.35 → 0.60 → 1.0.

## Smoke check list (18)

1. reset-sane: finite states, deck resting at bearing height, score exactly 0
2. reset: no success
3. randomization is real: deck yaw + window azimuth vary across 8 seeded resets (readback)
4. randomization: plate stands permute/jitter (readback)
5. null policy: 2 s idle → score 0, no latches, no success
6. rubric ladder (a): first smooth indexing rotation → `engaged`, score 0.10
7. oracle seed 0 → success(), score 1.0
8. rubric ladder (b): milestones 0.35 → 0.60 → 1.0, monotone
9. oracle seed 1 → success()
10. oracle seed 2 → success()
11. carry probe: loaded carousel driven +180°, plates ride under the roof, success persists
12. **negative A (seed's own strategy)**: direct placement over each colored pocket with
    the mechanism untouched → covered pockets blocked by the roof, at most one plate
    loads, no rotation credit, score ≤ 0.25, no success
13. **negative B (anti-teleport)**: plate written kinematically into a covered pocket is
    seated by geometry but never counts (no entry event), score 0
14. **negative C (wrong color / near-miss semantics)**: plate dropped through the window
    into another color's pocket seats physically, earns nothing beyond rotation credit
15. recovery: oracle extracts the misplaced plate via the window and finishes → success
16. near-miss tolerance: 45 mm off-axis drop rests on/against the corral wall, never counts
17. calibration probe: drop-capture sweep through the window — centred and 5 mm (funnel
    edge = 8 mm) seat 3/3; middle offsets published
18. calibration probe: 40 mm offset never seats; fresh reset afterwards reads clean

(Bullets 1/2 and 17/18 each fold two printed checks; the smoke prints 18 named
PASS/FAIL lines and `SIM_GEN_SMOKE: ALL PASS n/n` on success.)

## Physics notes

Fully procedural (primitives + compound `clone()` spawners). Honest outcomes: the
bearing is real contact (low-friction drum top, `friction_combine_mode="min"`; capture
skirt play 8 mm > combined 4 mm contact offsets), plates are carried during rotation by
real friction + corral walls (oracle drives the deck smoothly about its free axis with
matched angular velocity), all seating judged on settled poses. Roof is 12 mm thick vs
~6 mm/substep impact — no tunneling. The oracle stages transports kinematically but
every scoring entry is a genuine 20 mm drop through the window.
