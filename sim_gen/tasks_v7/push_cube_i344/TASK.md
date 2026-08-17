# balance_triage — weigh two identical cubes, then sort heavy/light into bins

`simgen.balance_triage` · derived from seed **maniskill/push_cube** (`push_cube_i344`)

## Seed provenance

`RoboVerse/roboverse_pack/tasks/maniskill/push_cube.py`: one 4 cm cube, one visible
goal region on the plane, one straight planar push (`DetectedChecker` +
`Relative2DSphereDetector`, radius 0.15). Everything about the seed's episode is
observable at t=0; the optimal plan is a single geometric motion.

## What changed and WHY it is strategically different

The seed keeps only its atoms — small cubes on a tabletop-scale floor, delivered to
goal regions. The task-defining structure is inverted around **hidden state**:

- **Hidden state instead of full observability.** TWO cubes, visually IDENTICAL
  (same 45 mm size, same brushed-steel color); one is ~3x heavier
  (0.38–0.46 kg vs 0.11–0.17 kg), and WHICH one is heavy is a per-episode coin
  flip written straight into the physics engine (masses + scaled inertias). No
  camera and no straight push can reveal it.
- **An instrument-as-SENSOR instead of a goal region.** A passive balance scale —
  heavy dynamic pedestal carrying a free beam on a spawn-authored revolute hinge
  (±10°), a walled pan at each end, beam CoM below the hinge so the empty beam
  self-levels. The solver must CREATE information: load both cubes onto OPPOSITE
  pans simultaneously, let go, and let gravity tip the beam toward the heavier
  side. The corpus uses weight as an *actuator* (weigh-press wedge, weigh-beam
  drop-in counterweights); here the beam's settled tilt direction *is the answer
  to a question*, and the rubric latches the physically verified comparison event
  (`weighed`: opposite pans + calm + beam ≥6° toward the TRUE heavy side, held 25
  consecutive steps).
- **Verdict-conditional delivery instead of a fixed goal.** Heavy → RED bin,
  light → BLUE bin. The delivery latches and success are GATED on the weighing:
  the seed's own strategy — just move the cubes into the goal regions — settles
  into the exact success placement and still scores ~0 with no success (flagship
  smoke check). A lucky blind guess never counts. One-pan tilts (single cube, or
  both piled on one pan) are constructed and rejected: they are not weighings.
- Also distinct from the sibling `push_cube_i30` (Silo Scoop: tool-mediated
  extraction) and the sieve/riddle sorters (which classify by SIZE with passive
  geometry): here the classification variable is invisible and must be measured
  by an explicit comparison experiment before delivery counts.

## Solution outline (solve.py — teleports are transport only)

1. **P0** reset + settle: beam level, cubes at floor spots, score 0. Oracle mass
   readback sanity-checks the hidden state.
2. **P1 weigh**: hover-drop the heavy cube ~12 mm above pan A (gravity seats it;
   the beam tips to the stop under contact), then hover-drop the light cube onto
   pan B. Beam holds its verdict; the `weighed` streak latch fires. The tilt
   direction is cross-checked against the oracle masses. Score 0.45.
3. **P2** hover-drop the light cube into the BLUE bin (beam stays pinned by the
   heavy cube). Score 0.60.
4. **P3** hover-drop the heavy cube into the RED bin; the unloaded beam
   self-levels on its own. success() holds. Score 1.00.
5. **P4** ≥3.3 s hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Nothing is ever teleported INTO a rubric-satisfying pose: the verdict tilt, every
seating and all settling are pure gravity + contact. Passes seeds 0 and 1.

## Embodiment argument (single Franka, base at the origin)

- The 45 mm cubes fit the ~80 mm Franka jaw with standard top grasps; every
  manipulation is a pick, a carry, and a low release — no fine tolerances
  (pan inner 88 mm and bin inner 130 mm vs 45 mm cube).
- Reach: pans at ~0.15 m height at (0.46, ±0.16); bins at (0.22, ±0.30); spawn
  spots at (0.20, ±0.09). Worst-case radius <0.68 m (asserted in cfg) — a
  comfortable single-arm workspace from a base at (0,0).
- The scale is operated exactly the way a robot would: place a cube in each pan
  and release; the beam does the measuring. Re-weighing and re-binning are
  allowed at any time.

## Execution-order declaration

No fixed motion order is imposed. The only ordering that matters is **causal**:
bin credit and success only count AFTER a completed two-pan weighing (the latched
comparison event). Cubes may be weighed, re-weighed, and delivered in any order;
only the weighing event and the final resting bins matter.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 12/12`)

1. settle/no-NaN: beam level, cubes at spots, score 0;
2. randomization readback (3-seed max-pairwise): hidden masses, heavy-spawn coin
   flip, bin xy+yaw, pedestal yaw, spawn spots;
3. null policy 240 steps: score ~0;
4. **BLIND SORT (seed strategy)**: both cubes settled in the CORRECT bins with no
   weighing — live placement clause fully satisfied, no success, score ~0;
5. single-cube tilt held at the stop ≫ streak: `weighed` stays False;
6. both cubes piled on ONE pan: beam at the stop, `weighed` stays False;
7. wrong bins after a REAL weighing: swapped delivery settles, score stays 0.45;
8. near-miss: heavy resting just OUTSIDE the red bin wall (light correct): 0.60,
   no success;
9. settle gate: exact success placement with the heavy cube sliding 0.35 m/s is
   refused (probe dismantled before it could settle genuine);
10. rejection audit: success() never True at any step of the battery;
11. score-cap audit: ≤0.70 everywhere;
12. frames.npz recorded.

## Verification

- `run --module solve` (seed 0 and seed 1): `SIM_GEN_SOLVE: SUCCESS`, scores
  0.00 → 0.45 → 0.60 → 1.00, non-decreasing.
- `run --module smoke`: `SIM_GEN_SMOKE: ALL PASS 12/12`, frames saved.
