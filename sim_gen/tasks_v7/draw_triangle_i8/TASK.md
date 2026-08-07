# bar_triangle (`draw_triangle_i8`)

**Seed:** `maniskill/draw_triangle`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/draw_triangle.py`) — a Franka holding
a rigid stick **draws** a triangle: it traces the goal outline on a canvas with the tool
tip, judged on the coverage of the traced path. The core skill is continuous,
contact-maintained **path-following** of a prescribed curve with a held tool.

**Scene:** `simgen.bar_triangle` (robot="null", scene-level; solve.py and smoke.py build
this same env).

## What changed

| | seed | this task |
|---|---|---|
| goal artifact | a transient trace history on a canvas | a persistent, settled STRUCTURE: a closed triangle frame of three rigid bars |
| mechanism | one long guarded tool-tip sweep along a given outline | three discrete pick / orient / place operations to derived target poses |
| what the solver must derive | nothing (the outline is given) | a consistent placement: WHICH unequal bar (red 180 / green 155 / blue 130 mm) spans WHICH side, corner retractions that close three gaps at once |
| verification | traced-path coverage | settled physical poses: per-corner endpoint gaps < 4 cm, flat-at-ground at BOTH ends, on-mat, distinct-ends cycle, Heron-anchored area gate |

Geometry: three square-section bars (18 mm, they cannot roll) spawn lying flat in
randomized staging spots on a polar fan about (−0.52, 0) — the documented arm-base
anchor — clear of a 36 cm dark build mat whose centre jitters ±4 cm. Randomized per
episode (verified by readback in smoke check 3): mat centre, per-spot bearing/radius/xy
jitters, spot-3 side, bar yaw, and a random bar→spot permutation.

## Why strategically different

- **Discrete construction vs continuous tracing.** The seed's solution is one
  path-following sweep with a held tool. Here a solver must *derive* a consistent
  target configuration from the three given side lengths (which bar spans which side,
  where each corner lands, retractions that close the gaps without colliding the bars),
  then execute three independent transport-and-place operations. There is no path to
  follow and no tool to hold — a different plan and code structure, not different
  numbers.
- **The seed's plan produces nothing here.** Tracing a triangle outline over the mat
  (with a bar or the empty gripper) leaves no persistent structure — the end state of
  the seed's strategy is the null state, score ~0 (smoke checks 4 and 7 cover the null
  and the "right shape, wrong site" variants). Conversely the seed's canvas has nothing
  to assemble, so this task's plan does not transfer back.
- **The rubric judges a persistent, settled arrangement**, not a trace history — with
  anti-cheese gates for stacking/weaving (endpoint-height at both ends), side-by-side
  bundles and hub fans (distinct-ends cycle), and near-degenerate layouts
  (area ≥ 0.5 × Heron(bar lengths)).

## Solution (solve.py — the teleport-solution legitimacy certificate)

Teleports are TRANSPORT ONLY; the load-bearing interaction — placement — runs through
contact dynamics:

1. **PLAN:** read the settled mat centre; `scene.frame_layout` fixed-point-solves the
   ideal frame (uniform ~24 mm corner gaps → 16 mm margin under the 40 mm tolerance,
   ~6 mm edge clearance between neighbouring bars at the corners).
2. **PLACE ×3 (transport teleport + dynamics):** each bar is carried across free space
   to hover 30 mm ABOVE its planned slot (planned yaw, zero velocity) and released —
   never written into a seated pose. It falls, impacts the mat, and settles under
   gravity/contact; only the settled pose the sim reads back can pass the flat /
   ground-height / settled gates and close the corner gaps. `SIM_GEN_SCORE` climbs
   0 → 0.05 → 0.30 → 1.00 as bars validate and corners close (scene-latched,
   non-decreasing).
3. **REPAIR (bounded, rarely triggered):** while success() is False (≤ 2 rounds),
   re-hover + re-drop the placed bar deviating most from plan — again transport +
   physical landing. Then require success() and score == 1.0 exactly.
4. **PERSIST:** ≥ 3.5 s of pure simulation with zero intervention, success re-verified
   at every poll → `SIM_GEN_SOLVE: SUCCESS`.

Passes seeds 0 and 1 on the forge (see acceptance runs).

## Embodiment argument (single Franka, parallel jaw, OSC)

Plausible base pose: **(−0.52, 0, 0)** on the floor facing +x — exactly the scene's
`stage_anchor`. The staging fan puts every bar 0.42–0.60 m from this base and the build
plan keeps every grasp/place target inside the 0.35–0.65 m comfort band over all
randomization extremes (verified numerically at design time). This is not hypothetical:
a real Franka OSC solve of this same scene (earlier campaign on this rig) passed seeds
0 and 1 first run using exactly the strategy below, landing bars 1–3 mm / < 1° from
plan against the 40 mm / few-degree budgets.

- **Bars (the only manipulated objects) — pinch grasp:** 18 mm square cross-section,
  ground-level side pinch across the bar at its midpoint: fingertips descend to ~4 mm
  above ground so the pads cover the bar's lower half, cage at 30 mm, quasi-static
  close ramp; lift verdict = bar rose ≥ 40 mm with a plausible jaw width. Square
  sections cannot roll, so staged bars wait indefinitely and placements persist.
- **Place:** closed-loop carry onto the planned centre with a yaw servo mapping the
  bar's live long axis onto the planned side direction (mod π); lower to ~3 mm, wait
  for stillness, slow release, vertical retreat. Required precision (40 mm endpoint
  gaps ≈ 12 mm centre budget) is an order of magnitude above OSC noise as measured.
- **Clearances:** everything happens on open floor — no overhangs, no apertures; the
  only near-ground contact is the deliberate ground-level pinch, which the measured
  runs performed repeatedly; bar-bar spawn separation ≥ bar width + 9 mm over all
  jitter draws, and the planned frame keeps ~6 mm edge clearance at the corners.

## Execution order: NOT required

Bars may be placed in any order; the rubric is symmetric in the corners (no
out-of-order negative control applies — documented per the brief). Difficulty comes
from deriving a consistent placement and closing three corner tolerances at once.

## Rubric

- `success()`: all three corners simultaneously closed ON the mat (closest endpoint
  pair < `joint_tol` = 40 mm, both bars flat within 8° with centre AND both ends within
  12 mm of rest height, settled < 0.05 m/s, centre + both ends on the mat, corner point
  on the mat) AND a distinct-ends cycle (each bar contributes its two DIFFERENT ends —
  rejects bundles/hub fans) AND enclosed corner area ≥ 0.5 × Heron(180, 155, 130 mm).
- `score()` = 0.05 per valid bar + 0.20 per closed corner, capped at 0.75, latched
  every physics substep (NaN-scrubbed); exactly **1.0 iff success() now**. Null policy
  scores 0 by construction (bars spawn clear of the mat).

## Checks (smoke.py — rubric rejection battery, 15 named checks; never constructs success)

1. settle/no-NaN: fresh reset settles finite, bars at rest
2. reset-zero: score ~0, nothing valid, no corners
3. randomization by READBACK (mat centre, every bar spot + yaw move; bars clear of the
   mat every seed)
4. null policy 2 s → score ~0, no success
5. `set_state(get_state)` roundtrip restores poses
6. near-miss: ideal frame ON the mat, every gap at joint_tol + 15 mm → 0 corners
7. seed-shape-wrong-site: gap-closed frame built beside the mat → rejected, score ~0
8. side-by-side bundle → ≤ 2 fake corners, never success
9. hub fan (all three pair gaps close!) → distinct-ends cycle + area gates reject
10. woven/propped bar (end resting on a neighbour) → flat/ground-height gates reject
11. calibration: 28 mm two-bar corner COUNTS (knee below joint_tol)
12. calibration: 55 mm two-bar corner does NOT count
13. latch: removing a placed bar keeps latched credit, success stays False
14. fresh reset clears the latch
15. video frames recorded (frames.npz in CWD)

The seed's end state (a traced outline, no persistent structure) is the null state —
covered by checks 4 and 7; the tool-tracing half is N/A by construction (nothing to
draw with or on).
