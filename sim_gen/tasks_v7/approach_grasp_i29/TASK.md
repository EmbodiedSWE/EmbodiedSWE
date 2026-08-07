# approach_grasp_i29 — Ridge Poise (scene `ridge_poise`)

Poise two OFF-CENTER-LOADED tray bars into a free-standing two-tier cross-stack on a
24 mm-wide ridge: the red bar balanced across the fin, the blue bar balanced crosswise
on the red bar's rails, every bar end airborne (env `simgen.ridge_poise`, robot
`null` · files: `scene.py`, `solve.py`, `smoke.py`).

## Seed provenance

Seed: `pick_place/approach_grasp` (`RoboVerse/roboverse_pack/tasks/pick_place/
approach_grasp.py`) — a Franka approaches a red cube and closes its jaw around it;
success is a gripper-object distance below 2 cm held a few frames, then a small
lift. The seed's plan is "reach one known object and acquire it"; its rubric reads
the gripper-object relation, and the episode ends *holding* the object.

## What changed, and why it is strategically different

The seed's object-acquisition plan is discarded entirely. Nothing here is judged
held — holding is worthless; the judged outcome is a **built equilibrium** that must
stand on its own after the hand (teleport) lets go.

The scene: a kinematic **ridge** (vertical fin, flat top 24 × 300 mm at height
140 mm) and two dynamic open **tray bars** (340 × 40 × 16 mm slab, side rails
forming a flat 40 mm-wide deck, eight dividers forming four pockets at ±90/±145 mm).
Each bar carries a **tungsten slug** (24 × 24 × 12 mm, 0.12 kg — 43 % of the loaded
0.28 kg system) seated in one *per-episode-random* pocket, shifting the loaded
balance point 38.6 or 62.1 mm off the bar's middle — beyond 2.5× the fin half-width
(12 mm) and 1.5× the rail-deck half-span (20 mm), asserted in `__post_init__`.
Goal: lay RED across the fin poised at its loaded balance point, then lay BLUE
crosswise on red's rails, ITS balance point over the stack, all four bar ends
airborne, both slugs still seated.

Strategic differences from the seed and from every other tasks_v7 task:

1. **Anti-seed end state.** The seed ends with the object in the gripper; here an
   episode that ends with a bar "acquired" (or delivered anywhere but poised) scores
   ~0 — smoke check 6 builds the *correct* cross-stack on the floor beside the ridge
   and it is rejected. No grasp, distance, or held-pose relation is ever read.
2. **The equilibrium is built, not operated.** The three balance-flavored corpus
   tasks — `coke_task_i15` (counterweigh a two-pan beam), `pick_and_lift_i16`
   (ballast a seesaw lever), `pull_cube_i20` (overload a beam scale) — all ADD
   weights to a PRE-BUILT pivoting mechanism and read the mechanism's settled angle.
   Here **nothing pivots and nothing is added to a mechanism**: the transported
   payloads themselves must be poised, two tiers high, on a fixed fin. The rubric
   reads the *payloads'* settled level/height bands, and tier 2's support is tier 1
   — a dependency none of those tasks has.
3. **Per-episode plan from a hidden physical parameter.** The slug pocket (4 choices
   per bar, independent) moves each bar's true balance point across four positions on
   both sides of center. The solver must *read* the pockets (readback or vision),
   *infer* the loaded CoM, and place each bar by a point that is never marked on the
   bar. A geometric-center placement — correct for an empty bar — tips off (smoke
   check 4). The rubric never computes a CoM; gravity grades the inference.
4. **Wrong-but-standing states are rejected by identity, not physics alone.** The
   smoke battery constructs five states that genuinely STAND (slugless stack, floor
   stack, blue-beside-red, inverted red-on-blue, parallel stack) and each is
   rejected by exactly one live clause — proving every success clause is
   load-bearing, and distinguishing this task from "make anything balance".

## Scene (engineered numbers, asserted in `__post_init__`)

- Kinematic ridge: ground plate + fin, top 24 mm wide × 300 mm long at z 0.140;
  nominal (0.35, 0), yaw ±30°, xy jitter ±30 mm.
- Dynamic bars ×2 (0.16 kg, explicit mass/CoM/inertia): slab 340 × 40 × 16 mm, rails
  6 × 14 mm at y ±17 mm (deck top at local z 0.022), dividers making four 28 mm
  pockets at ±90/±145 mm. Rail deck rides 2 mm proud of a seated slug (top 0.020),
  so the upper bar rests on rails only.
- Slugs ×2: 24 × 24 × 12 mm, 0.12 kg (tungsten ~17.4 g/cm³), seated in one pocket
  at reset. Loaded balance offset = 0.4286 × slug_x = ±38.6 or ±62.1 mm.
- Height bands (half-width z_tol 6 mm; all bands ≥ 2.5× z_tol apart): floor bar root
  0.008, red poised 0.148, blue-on-red 0.178 (red-frame 0.030); level gate 3°,
  crosswise gate 12°.
- Statics asserts: balance offset > 2.5× fin half-width and > 1.5× deck half-span
  (geometric-center placement MUST tip; correct placement is flat-on-flat stable);
  pocket clearances; slug-below-deck; jaw-fit; band separations.
- Randomization (readback-verified): both slug pockets (4 × 4 independent), ridge
  yaw/xy, bar floor spawns in a 0.18–0.60 m annulus with ridge keep-out 0.36 m,
  bar-bar separation 0.40 m, free yaw.

## Rubric (`success` / `score`)

`success` requires ALL of, live and settled (never bookkept):
- `red_poised` — red level (< 3° both axes), settled, ridge-frame |x| < 0.10,
  |y| < 0.13, root z within 6 mm of 0.148: resting ON the fin, the only support at
  that height, both ends airborne;
- `blue_on_red` — blue level, settled, long axis within 12° of perpendicular to
  red's, red-frame |x| < 0.15, |y| < 0.10, z within 6 mm of 0.030: resting on red's
  rails, supported by red alone;
- `slug_red`, `slug_blue` — each slug seated in a pocket of its OWN bar (bar-frame
  pocket bands: |x − cell| < 10 mm, |y| < 8 mm, z ∈ (8, 20) mm);
- all states finite.

`score` is latched (credit never evaporates): 0.30 × red-ever-poised-with-slug +
0.30 × full-stack-ever-standing, cap 0.60; exactly 1.0 iff `success()` live. Null
policy scores 0 (bars start on the floor).

## Teleport solution (`solve.py`) — transport only

- **P0** — settle; READBACK-assert both slugs seated, both bars on the floor,
  score ≤ 0.03, no success.
- **P1** — read red's slug pocket from its bar-frame position, compute the measured
  balance x_bal = 0.4286 × slug_x, teleport red (slug co-carried, relative pose
  preserved) to an 8 mm hover with −x_bal over the fin, release; the set-down, tip
  or poise is pure contact dynamics. Up to 5 retries; asserts `red_poised` latched,
  score ≥ 0.30. `SIM_GEN_SCORE` printed at every phase boundary, never decreasing.
- **P2** — teleport blue (slug co-carried) to a hover crosswise over the fin line at
  red's y minus blue's x_bal, z 0.186, release; closed-loop success wait; up to 6
  retries with a red-rebuild branch if the drop knocks red off. Asserts score
  ≥ 0.999.
- **Persistence** — 400 steps (≥ 3.3 sim-seconds) fully hands-off with success
  re-sampled 10×, all required, then `SIM_GEN_SOLVE: SUCCESS`. Watchdog hard-exit.

Teleports move bars through free space only; every load-bearing interaction — bar
on fin, bar on rails, slug in pocket, every tip and every poise — is contact
dynamics.

## Embodiment argument

A Franka based at the origin (0, 0, 0) covers the workspace: ridge nominal (0.35, 0)
± 3 cm and the whole 0.18–0.60 m spawn annulus lie inside its ~0.85 m reach, and the
highest set-down (blue at 0.186 m) is mid-workspace. Per object:
- **Bars**: grasp across the 40 mm width (< 80 mm jaw) anywhere along the slab —
  ends and inter-divider spans leave > 60 mm of clear slab; loaded mass 0.28 kg ≪
  payload. Carrying a bar level keeps the slug pocketed (rails + dividers cage it;
  solve's co-carry mirrors this). Set-down accuracy required is the ±12 mm CoM
  window — the solver hits it open-loop from the measured pocket; an arm can also
  close the loop on visible tipping (level gate 3°).
- **Slugs**: never require direct manipulation (they ride their pockets); if
  re-seating were ever needed, a 24 mm cube at 0.12 kg is a trivial pinch.
- **Ridge**: kinematic fixture, never manipulated. No moving mechanism threatens
  the arm; the tallest structure is 0.19 m.

## Execution order (declaration)

1. Designed the geometry/statics and a *minimal* `success()` first.
2. Wrote `solve.py` and iterated it on the forge until the stack was reached
   robustly — **seeds 0 and 1 SUCCESS**, monotone `SIM_GEN_SCORE`
   0.00 → 0.30 → 1.00, first-attempt placements on both.
3. Only then finalized the rubric (latches, score shape) and the `__post_init__`
   statics asserts.
4. Wrote `smoke.py` last, as an adversarial battery against the final rubric.
5. Re-ran solve (both seeds) and smoke clean on the final submitted code.

## Checks

`solve.py` (per run): P0 readback asserts (slugs seated, bars on floor, score
≤ 0.03, no success), per-phase latch + score asserts, final success + score
≥ 0.999, 3.3 s hands-off persistence sampled 10×. Passed on seeds 0 and 1.

`smoke.py` — `SIM_GEN_SMOKE: ALL PASS 12/12`:
1. Settle / no-NaN: bars flat on the floor, slugs seated, score 0, no success.
2. Randomization readback over 6 resets: ridge yaw spread 31.3°, ridge xy 56 mm,
   red spawn 1028 mm, ≥ 3 distinct slug-pocket pairs (read from settled bar-frame
   slug positions).
3. Null policy 300 steps → score ~0, no success.
4. GEOMETRIC CENTER: red set down on the fin by its middle, slug aboard — tips off
   (ends off-level, out of the poised band), score ~0.
5. SLUGLESS cheat: slugs dumped, both EMPTY bars stacked centered — STANDS in the
   exact judged bands, rejected by the slug-seated clauses, score ~0.
6. FLOOR STACK (seed strategy): the correct cross-stack built on the floor beside
   the ridge — `blue_on_red` holds, `red_poised` does not, score ~0.
7. BLUE-BESIDE-RED: both bars poised on the fin side by side — a genuine second
   balance, rejected because blue must ride red's rails (30 mm higher band); score
   stays at the 0.30 partial.
8. INVERTED stack: blue poised on the fin, red crosswise on top (built with the
   mirrored balance math, stands at the correct bands) — rejected purely by bar
   identity; score ~0.
9. PARALLEL stack: blue on red's deck aligned with red, standing level at the
   correct height — rejected by the crosswise clause; score stays at 0.30.
10. REMOVE-SLUG: full success constructed live (score 1.0), blue slug teleported
    away → gravity dislodges the blue bar, success collapses, latched score 0.60.
11. Rejection audit: success() never fired during any negative probe (4–9).
12. frames.npz recorded (504 × 600 × 960 × 3).
