# weighbridge (push_button_i6)

**Env name:** `simgen.weighbridge` (scene `weighbridge`, registered scene-level with
robot `null`; `solve.py` builds its own Franka env).
**Tier:** easy-medium — pick the one heavy block among decoys, place it centred on a
compliant sprung target, leave it resting.
**Execution order:** none beyond the natural pick-then-place; declared in `describe()`.

## Seed provenance

Seed: `rlbench/push_button`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/push_button.py`) — a Franka reaches a
small articulated button and gives it a momentary fingertip press. One skill: reach +
poke. No object interaction, no lasting physical requirement.

## What changed, and why it is strategically different

The "button" is inverted into a **weight-activated weighbridge plate**: a bright-red
10 cm plate riding 35 mm of vertical prismatic travel on a pedestal, with a spring
return (`f = k*(home − z) − c*v`, k = 80 N/m — the jointed-button mechanism proven in
robobench's microwave scene; gravity disabled on the plate so home is the exact rest
pose). A pedestal lamp glows green while the plate is held fully down.

The seed's entire plan — reach and press — is **useless here by construction**:

1. A momentary press springs straight back (nothing latches on touch).
2. Even an **infinitely patient sustained press fails**: success requires the plate held
   at ≥ 80 % travel by **enough settled resting mass** — the rubric sums the masses of
   the settled blocks geometrically resting on the plate and requires that sum to reach
   the spring's trigger mass (k·travel/g ≈ 285 g). A bare end-effector press has no
   mass to show, and a press **through a 60 g foam block** shows only 60 g (smoke
   checks 4 and 6 construct both and assert rejection).

The solver instead needs a different plan: **select the one sufficiently heavy object**
(dark-red 0.50 kg, 5.5 cm load cube; the two pale 4.5 cm foam cubes are decoys — one
sags the plate ~7 mm, both stacked ~15 mm, still far under the 28 mm press threshold),
**pick it up, place it centred on the 10 cm plate, and walk away** with the weight
keeping the plate bottomed. That is object selection by weight-relevant identity plus
pick-and-place onto a compliant, moving target — not a poke.

Physics is honest everywhere: success()/score() judge only settled poses, live plate
depth, and real spring compression under real block weight. smoke's teleports/pins are
rubric instrumentation, not a solution; the real solution is `solve.py`.

## Solution outline (solve.py — the feasibility certificate)

Franka base pose: **(-0.48, 0, 0), identity rotation** (facing +x), OSC control,
`nullspace_dof_pos=()`, gripper effort 120 N / stiffness 4000. The scene layout was
designed around this pose: pedestal fixed at (0.15, 0) (~0.63 m ahead), blocks scattered
on a lateral arc 0.34–0.45 m from the base.

Phases (SIM_GEN_SCORE printed at every boundary; monotone along the real trajectory):
1. SETTLE — let the scatter rest; read the scene (score 0.0).
2. PICK — top-down pinch of the load cube: hover over its live xy, descend fingertips
   to pad height, quasi-static close ramp to 44 mm, lift + verdict (block rose > 5 cm
   AND jaw width in the 55 mm band). (score 0.2 latches during transit.)
3. CARRY — closed-loop on the BLOCK xy onto the live plate axis at transit height.
4. LOWER — press-down tracking the block onto the plate until the plate is measured
   ≥ 30 mm deep (the spring bottoms under the gripped block; score ~0.83 gripped).
5. RELEASE — slow open, retreat straight up, park clear (score 1.0 once the scene's
   sustained-press counter latches on the resting block alone).
6. VERIFY — success() polled, then 3 s more simulation: success persists.

Arm-only manipulation: no task-object state writes, no external forces on task objects.
Verified on the forge on seeds 0, 1, 2 — `SIM_GEN_SOLVE: SUCCESS` each, first-attempt
grasps, ~20 s sim / ~33 s wall each, score prints non-decreasing (0.00 → 0.20 → 0.83 →
1.00).

## Randomization (per episode)

Block-to-slot permutation over the scatter arc + per-block xy jitter (±3 cm) + free yaw
(verified by readback in smoke check 2). The pedestal/plate pair is deliberately FIXED:
on this PhysX stack a per-episode teleport of a jointed pair is unreliable (the joint
frame stays anchored at the authored pose — measured here: the plate stayed at the
authored xy while the pedestal moved), so the mechanism never moves and only the free
bodies randomize.

## Rubric (score in [0, 1], partial progress latched)

- 0.2 — latched once any block is lifted clear of the surface (bottom > 8 cm).
- 0.5 — latched once a block reaches the plate top.
- 0.5–0.9 — live, a block on the plate, scaling with depth toward the press threshold.
- 1.0 — iff `success()`: plate ≥ 80 % travel (28 mm), settled on-plate mass ≥ trigger
  (285 g), plate settled, sustained 90 consecutive substeps (the anti-poke gate).
  Success is live physical state: remove the load and it reverts.

## Checks (smoke: SIM_GEN_SMOKE: ALL PASS 13/13 on the forge; rejection tests only —
solve.py is the acceptance evidence)

1. settle/no-NaN: plate at home on its axis, score 0
2. randomization is real (block slots/jitter/yaw readback across seeded resets)
3. null policy: score ~0, no success
4. **seed strategy**: sustained bare press (plate pinned at full depth, 150 substeps)
   → never success, score ~0 (expressible, and it fails)
5. spring-back: releasing the press returns the plate home, no credit
6. **insufficient-mass press**: patient press THROUGH a 60 g foam pinned at depth
   → never success (the mass clause)
7. near-miss: one resting foam → ~7 mm sag, partial credit only, never success
8. near-miss: both foams stacked → ~15 mm, still under the 28 mm threshold, no success
9. wrong place: load settled on the ground beside the pedestal → score ~0
10. wrong place: load on the pedestal rim beside the plate → no press, no success
11. near-miss: load released just outside the on-plate radius (straddling the plate
    edge) → never success (it slides clear — physically self-rejecting)
12. staged partials monotone: reset 0 < lifted 0.2 < foam-on-plate 0.61 < 1.0

Plus the solve gate: `SIM_GEN_SOLVE: SUCCESS` on seeds 0/1/2 with non-decreasing
SIM_GEN_SCORE prints and success persisting under 3 s of extra simulation.
