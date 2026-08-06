# crate_turnover — evict the milk, stow the clutter, cap the crate

## Seed provenance

- Seed id: `libero/libero_pick_milk`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero/libero_pick_milk.py`
  ("Pick the milk and place it in the basket": grasp the milk carton among 5 food
  distractors, transport it, drop it inside the basket; success = a relative-bbox
  containment check of the milk against the basket; the distractors exist only to be
  ignored).

## What changed, and why it is strategically different

The seed is one prehensile transport INTO a container. Here the seed's terminal
relation — *milk inside the container* — is the **initial state**, and every role in
the scene is inverted:

1. **Goal inversion (target OUT).** The milk carton starts standing inside the open
   crate and must be EVICTED: stood upright on a blue delivery pad on the ground.
   A solver executing the seed's plan ("put the milk in the basket") performs a no-op
   — that end state is the null layout, scores 0, and is the tested seed-strategy
   negative control.
2. **Distractor role inversion (clutter IN).** The seed's ignorable distractors (red
   cube, yellow can; subset-sampled 1–2 per episode) are now graded objects: the
   crate's final content must be exactly the clutter.
3. **Closure terminal stage with physically-enforced ordering.** The crate must be
   capped with a detached free lid (graspable knob post). While the 140 mm carton
   stands inside the 70 mm-walled crate, the lid physically cannot seat (it rests
   ~80 mm high, tilted, on the carton top — tested); once seated, nothing can be
   inserted. So the lid is forced LAST, and the seed's plan is geometrically
   self-blocking, not just un-credited.

A solver needs a different **plan** (multi-object rearrangement with an inverted
containment goal + closure), not different parameters: 4 pick-and-places with
role-swapped sources/destinations vs the seed's single one.

(Also distinct from prior-campaign tasks_old designs: the old i3 for this seed was a
carton *reorientation* task — nothing shared; unload-before-close appeared in old
close_box_i13 but there the lid was the cargo tray and closure the headline; here the
headline axis is the eviction/role swap, the lid is the ordering interlock.)

## Difficulty tier / stages

- **Tier: medium. Declared stage count: 3–4** (milk out; 1–2 items in; lid on).
- **Execution order: partially required** — the lid must go on LAST (physically
  enforced by the carton height vs. rim, and by the seated lid covering the mouth);
  milk-out and items-in may be interleaved freely.

## Judging (physical outcomes only)

`success()` (all current-state, settled): milk standing upright (≤ 10°) on the pad
(xy ≤ 40 mm of pad center, base within 12 mm of the slab top) · every PRESENT item
inside the crate interior (crate-frame |xy| ≤ 65 mm, center below the rim) · lid
seated (crate-frame xy ≤ 15 mm, center z = 75 ± 10 mm, level ≤ 8°, yaw aligned mod
90° ≤ 15°) · milk/items/lid settled.

`score()` (graded, 1.0 iff success): 0.10 latched extraction (milk lifted clear of
the rim or taken outside the footprint) + 0.30 delivered (current) + 0.30 × stowed
fraction (current) + 0.15 lid seated (current). Null policy = 0.

Honesty margins (asserted in `scene.py` `__post_init__`, verified on the forge):
any item pose physically inside the crate earns stow credit (max offset 51 mm < 65 mm
gate); an item on the seated lid is above the z gate; lid-on-carton rests ≥ 145 mm
(seat gate 75 ± 10); lid coverage holds at combined xy+yaw tolerance
((75+15)·√2·sin(60°) = 110 mm ≤ 115 mm lid half-side); pad is provably out of reach
of any in/on-crate milk pose. Lid xy calibration cliff measured on the forge:
≤ 10 mm offsets close 6/6, ≥ 25 mm close 0/6 (25 mm rests level but outside the
gate; 45 mm physically topples into the mouth).

Randomization (verified by readback): crate xy ± 30 mm + yaw ± 180°, pad xy ± 30 mm,
milk offset-in-crate ± 20 mm (crate frame) + free yaw, lid xy ± 20 mm + yaw ± 180°,
item xy ± 20 mm (+ yaw for the cube), item subset 1–2. All assets procedural
(compound spawners / primitives); crate and pad are kinematic fixtures.

## Solution (solve.py — the feasibility certificate)

- **Base pose: `(-0.42, 0.0, 0.0)`**, identity rotation (recorded in solve.py as
  `BASE`); OSC control, gains kp=(220,220,220,600,600,600), `nullspace_dof_pos=()`.
- Phases (each a closed-loop hover → descend → ramped close → lift-verdict → carry on
  the OBJECT xy → lower → slow release → settle-predicate, with per-phase retries):
  1. **MILK OUT** — top-down pinch of the carton's upper half (it pokes 70 mm above
     the rim; fingertips never go below the rim plane), jaw azimuth = carton yaw mod
     90°; carry to the pad; lower until the base is ~4 mm off the slab; release →
     stands upright. (`SIM_GEN_SCORE` 0.10→0.40 latched+current)
  2. **ITEMS IN** — per present item: pinch at center height (cube jaw aligned to its
     yaw), carry over a crate-frame drop slot, release ~12 mm above rest → drops in.
  3. **LID ON** — pinch the gray knob, lift, carry over the crate while rotating the
     wrist to square the lid with the crate yaw (mod 90), lower to ~4 mm above the
     rim, slow release → seats flush.
- No task-object state writes, no external forces; arm + gripper joints only.
- Forge results: `SIM_GEN_SOLVE: SUCCESS` on seeds **0, 1, 2, 3, 4, 5** (31–40 s
  each; seed 5 is a 1-item subset episode, present=[1,0]), per-seed layouts verified
  different by readback, score trace 0.000 → 0.400 → 0.550 → 0.700 → 1.000 monotone
  (SIM_GEN_SCORE prints), success persists ~3 s post-goal.
- Solve robustness lessons baked in: seeding via `env.reset(seed=...)` (a plain
  `torch.manual_seed` before build is stomped); item slots kept 0.62–0.71 m from the
  base (a can at 0.34 m wound the wrist and got punted/rolled away — the one observed
  failure, fixed by moving the slot); lying-can grasp fallback (azimuth across the
  rolled axis).

## Smoke check list — REJECTION battery (final forge run: `SIM_GEN_SMOKE: ALL PASS 14/14`, 37 s)

solve.py is the acceptance proof; smoke constructs only WRONG outcomes as settled
states (teleport probes are instrumentation, none reaches success()):

1. settle/no-NaN (milk starts IN the crate, score 0) · 2. randomization readback
(crate pos+yaw / pad / milk / lid all differ) · 3. subset sampling {1,2} · 4. null
policy ~0 (the seed's terminal relation IS the start state — doing nothing earns 0) ·
5. SEED-STRATEGY control (milk kept in the crate, items stowed, lid laid on: it rests
high on the carton, never seats; score ≤ 0.35) · 6. milk upright 110 mm off the pad →
rejected (xy gate) · 7. milk lying on the pad → rejected (upright gate) · 8. lid
45 mm off-axis → topples into the mouth, never closed · 9. lid twisted 30° at the
exact seat → rejected (mod-90 yaw gate) · 10. items ON the seated lid → no stow
credit (z gate) · 11. one item left on the floor, all else right → no success ·
12–13. lid-offset calibration sweep on an unloaded crate (≤10 mm seats 6/6, ≥25 mm
0/6 — the 15 mm gate is physically reachable and the cliff is real) · 14. video
frames.npz saved to CWD.
