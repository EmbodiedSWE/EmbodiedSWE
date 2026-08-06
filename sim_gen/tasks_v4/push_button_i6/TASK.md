# weighbridge (push_button_i6) — hold the plate down with enough resting MASS

**Env name:** `simgen.weighbridge` (scene `weighbridge`, robot `null`; both `solve.py`
and `smoke.py` build the same scene-level env).
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
return (`f = k*(home − z) − c*v`, k = 80 N/m; gravity disabled on the plate so home is
the exact rest pose). A pedestal lamp glows green while the plate is held fully down.

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
pick-and-place onto a compliant, moving target — not a poke. A solver needs different
code structure too: no press primitive at all; a grasp, a carry onto a live compliant
surface, a release, and a hands-off verify.

Physics is honest everywhere: success()/score() judge only settled poses, live plate
depth, and real spring compression under real block weight. smoke's teleports/pins are
rubric instrumentation, not a solution; the acceptance evidence is `solve.py`.

## Teleport-solution outline (solve.py — the task's legitimacy certificate)

Scene-level env, robot `null`. Teleports do TRANSPORT only; the release point is
deliberately chosen OUTSIDE the on-plate scoring band (asserted in code), so no
placement credit can latch at a teleport instant — every latch past "lifted" fires only
after contact dynamics:

- **PHASE 1** — teleport the 0.50 kg load cube from its floor scatter slot to a hover
  pose centred on the LIVE plate axis, cube bottom **28 mm above the plate top**
  (on-plate band is |bottom − plate_top| < 20 mm, so the teleport instant latches no
  placement; asserted, plus loaded_mass readback = 0). The lift latch (0.2) fires —
  honest transport credit. `SIM_GEN_SCORE 0.000 → 0.200`.
- **PHASE 2** — hands off. Gravity drops the cube the last 28 mm; the impact, the
  spring compression to the joint stop, and the 90-substep sustained press under the
  resting 0.50 kg are all pure contact dynamics — nothing pinned, no velocity clamps,
  no applied forces on task objects. success() = live depth ≥ 28 mm AND settled
  on-plate mass ≥ 285 g AND plate still, sustained 90 consecutive substeps.
  `SIM_GEN_SCORE → 1.000`.
- **PHASE 3** — hands off for 420 physics steps (3.5 simulated seconds); success() here
  is LIVE state (a bounce-off or slide-off reverts it), so persistence rejects
  fly-through outcomes. Only then `SIM_GEN_SOLVE: SUCCESS`. Watchdog Timer +
  `os._exit` guard teardown.

The printed `SIM_GEN_SCORE` sequence is monotone (asserted in code). Verified on the
forge on seeds 0 and 1.

## Embodiment argument (single Franka + parallel jaw, OSC)

This exact plan was previously EXECUTED end-to-end by a real Franka arm in this exact
scene (an earlier arm-driven run of the same task package): base at
**(-0.48, 0, 0), identity rotation** (facing +x), OSC (kp 220/600), gripper effort
120 N / stiffness 4000 — `SIM_GEN_SOLVE: SUCCESS` on seeds 0/1/2, first-attempt grasps,
~20 s sim each, score prints 0.00 → 0.20 → 0.83 → 1.00. Per-object contact strategy:

- **Load cube (0.50 kg, 55 mm edge)**: top-down pinch across two opposite faces — the
  55 mm edge fits the 80 mm jaw with room to approach from above on an open floor
  scatter (no overhangs, no walls); 0.50 kg is inside the measured 0.62 kg carry
  ceiling at grip 18 mm/effort 120. Carried to the plate axis, pressed down onto the
  plate (the spring bottoms under the gripped block), slow release, straight-up
  retreat.
- **Placement tolerance**: on-plate radius 50 mm on a 100 mm plate for a 55 mm cube —
  ±2 cm of lateral slack, far above OSC control noise; the plate is only 35 mm of
  travel below its home top at ~0.13 m height, nowhere near the floor.
- **Foam decoys**: never need to be touched.
- **Reach**: pedestal at (0.15, 0) → ~0.63 m from the base; scatter arc 0.34–0.45 m —
  all inside the Franka 0.45–0.71 m comfortable envelope (close blocks are still
  reachable top-down at their 0.02 m height).

## Randomization (per episode)

Block-to-slot permutation over the scatter arc + per-block xy jitter (±3 cm) + free yaw
(verified by readback in smoke check 3). The pedestal/plate pair is deliberately FIXED:
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
- Null policy ≈ 0. Seed strategy ≈ 0. Latched credit never evaporates (the
  `SIM_GEN_SCORE` prints along the solve trajectory are monotone, asserted in code).

## Check list (smoke.py — 13 checks; rejection tests only — solve.py is the acceptance
evidence; teleported/pinned probes are instrumentation, not a solution)

1. settle/no-NaN: plate at home on its axis, all states finite
2. settle: score ~0 at reset
3. randomization is real (block slots/jitter/yaw READBACK across seeded resets)
4. null policy: 240 idle steps → score ~0, no success
5. **seed strategy**: sustained bare press (plate pinned at full depth, 150 substeps,
   nothing resting) → never success, score ~0 (expressible, and it fails)
6. spring-back: releasing the press returns the plate home, no credit
7. **insufficient-mass press**: patient press THROUGH a 60 g foam pinned at depth
   → never success (the mass clause)
8. near-miss: one resting foam → ~7 mm sag, partial credit only, never success
9. near-miss: both foams stacked → ~15 mm, still under the 28 mm threshold, no success
10. wrong place: load settled on the ground beside the pedestal → score ~0
11. wrong place: load on the pedestal rim beside the plate → no press, no success
12. near-miss: load released just outside the on-plate radius (straddling the plate
    edge) → never success
13. staged partials monotone: reset 0 < lifted < foam-on-plate < 1.0

Out-of-order end state: N/A — no execution order is declared (single pick-then-place).
Video: smoke saves `frames.npz` in the current working directory.

Plus the solve gate: `SIM_GEN_SOLVE: SUCCESS` on ≥ 2 seeds with non-decreasing
SIM_GEN_SCORE prints and success persisting under 3.5 s of extra hands-off simulation.
