# draw_triangle_i8 — Bar Triangle (build a closed triangle frame; Franka-solved)

## Seed provenance

- Seed: `maniskill/draw_triangle`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/draw_triangle.py`)
- Seed semantics: a Franka holding a rigid stick **draws** a triangle — it traces the goal
  outline on a canvas with the tool tip, judged on the coverage of the traced path. The
  core skill is continuous, contact-maintained **path-following** of a prescribed curve
  with a held tool.

## What changed

The goal *shape* (a triangle) is kept; the *mechanism* is replaced wholesale. Nothing is
drawn and nothing traces. Three rigid square-section bars of **unequal lengths**
(red 180 / green 155 / blue 130 mm, 18 mm thick) wait in randomized staging spots on the
floor beside a randomly-placed dark square build mat (36 cm). The task is to **construct**
a closed triangle frame: lay the bars flat on the mat so their ends meet pairwise at three
corners, each corner endpoint gap under 4 cm, every bar flat at ground level and entirely
on the mat, each bar contributing both of its (different) ends. Judged purely on the
settled physical poses read back from sim.

## Why strategically different

- **Discrete construction vs continuous tracing.** The seed's solution is one long guarded
  tool-tip sweep along a given outline. Here a solver must *derive* a consistent target
  configuration (which bar spans which side, where each corner lands, corner retractions
  that close the gaps without colliding the bars), then execute three independent
  pick / re-orient / place operations. There is no path to follow and no tool to hold.
- **The seed's plan produces nothing here.** Tracing the triangle outline with a bar (or
  the empty gripper) leaves no persistent structure — the end state of the seed's strategy
  is the null state, which scores ~0 (smoke probes 4 and 7 cover the null and the
  "right shape, wrong site" variants). Conversely the seed's canvas has nothing to
  assemble, so this task's plan does not transfer back.
- **The rubric judges a persistent, settled arrangement** (corner gaps + flat-at-ground +
  on-mat + end-cycle consistency + area gate), not a trace history — with anti-cheese
  gates for stacking/weaving (endpoint-height), side-by-side bundles and hub fans
  (end-cycle), and near-degenerate layouts (Heron-anchored area gate).

## Embodiment and solution (solve.py — the feasibility certificate)

Single Franka, robobench OSC, **base at (-0.52, 0, 0)** facing +x — the scene's
`stage_anchor`. The staging spots fan out at 0.42-0.60 m from this anchor and the build
plan keeps every grasp/place target in the 0.35-0.65 m comfort band (verified
numerically over all randomization extremes at design time).

Phases (per bar, farthest planned pose first):

1. **PLAN** — read the settled mat centre; `scene.frame_layout` computes the ideal frame
   (uniform ~24 mm corner gaps — 16 mm margin under the 40 mm tolerance, 6 mm edge
   clearance between bars so nothing collides), centred 3 cm base-ward of the mat centre,
   rotation chosen to minimize the worst endpoint reach.
2. **PICK** — hover over the bar midpoint, jaw across the bar (branch nearest the wrist
   home azimuth, base-ward tilt when close), fingertips to 4 mm above ground so the pads
   cover the bar's lower half, cage (30 mm) then quasi-static close ramp; verdict = the
   bar rose ≥ 40 mm with a plausible jaw width.
3. **PLACE** — closed-loop carry on the live bar centre onto the planned centre while a
   yaw servo maps the bar's live long axis onto the planned side direction (mod pi);
   lower to 3 mm above rest, wait for stillness, slow release, vertical retreat. A bar
   that lands off-plan (>12 mm / >4°) is re-picked with a compensated aim.
4. **REPAIR** — any unclosed corner: re-place the adjacent bar that deviates most from
   plan, compensated aim, up to 2 rounds.

Square cross-sections are deliberate: released bars cannot roll, so placements persist.
solve.py never writes task-object state and applies no external forces; it prints the
scene's `corner_report()` and `SIM_GEN_SCORE` at every phase boundary (scene-latched, so
the printed series never decreases) and `SIM_GEN_SOLVE: SUCCESS` on the settled end state.
Passes seeds 0 and 1 (see acceptance runs).

## Execution order: NOT required

Bars may be placed in any order; the rubric is symmetric in the corners (no out-of-order
negative control applies — documented here per the brief). Difficulty comes from deriving
a consistent placement and closing three corner tolerances at once, not from stage depth.

## Rubric

- 0.05 per bar placed flat + settled + entirely on the mat, 0.20 per closed corner;
  partial capped at 0.75 and **latched** every physics substep (transient achievements
  don't evaporate while adjusting or re-picking).
- Exactly **1.0 iff `success()`** on the current settled state: all 3 corners closed on
  the mat + distinct-ends cycle + area ≥ 0.5 × Heron(bar lengths).
- Null policy scores 0 by construction (bars spawn clear of the mat).

## Randomization (per episode, verified by readback in smoke check 3)

- Mat centre: ±4 cm xy.
- Staging spots: polar fan about the anchor — bearing ±12° jitter per spot, radial
  ±4 cm, spot-3 side random, residual xy jitter, near-tangential yaw ±20°; bars dealt to
  spots by a random permutation. (Worst-case spawn separation between bars verified ≥
  bar width + 9 mm over the full jitter ranges at design time.)

## Smoke check list (`smoke.py` — rejection battery; NO teleport solution, never
constructs success; solve.py is the acceptance proof)

1. settle/no-NaN; 2. reset scores ~0; 3. randomization by READBACK; 4. null policy ~0;
5. `set_state(get_state)` roundtrip; 6. near-miss: ideal frame with every gap at
joint_tol + 15 mm → 0 corners; 7. gap-closed frame built beside the mat → rejected
(site is load-bearing); 8. side-by-side bundle → ≤ 2 fake corners, never success;
9. hub fan (all three gaps close!) → distinct-ends cycle + area gates reject;
10. woven/propped bar (rests across a neighbour) → flat/ground-height gates reject;
11-12. calibration: two-bar corner counts at 28 mm, not at 55 mm (knee at joint_tol);
13. latched credit survives removing a placed bar (no evaporation), success stays False;
14. fresh reset clears the latch; 15. frames.npz saved.

Acceptance markers: `SIM_GEN_SMOKE: ALL PASS n/n`, `SIM_GEN_SOLVE: SUCCESS`.
