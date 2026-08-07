# libero_kitchen_scene3_put_the_frying_pan_on_the_stove_i75 — hang the frying pan on the wall hook

## Provenance

Seed: `libero_90/libero_kitchen_scene3_put_the_frying_pan_on_the_stove` — grasp
the frying pan and put it ON the flat stove (a place-onto-a-horizontal-surface
outcome, judged while the pan rests on the burner).

## Strategic difference from the seed

The seed's GOAL state is this task's START state: the pan begins lying flat on
the LIT burner, and the goal is to take it OFF the heat and hang it up. The
outcome class is inverted — the pan must end **suspended in the air**, supported
only by hook-through-loop contact, not resting on any surface:

1. **Lift** the pan clear of the glowing burner.
2. **Reorient** it ~90° from lying flat to vertical (handle up, face parallel to
   the wall) — an in-hand reorientation the seed never needs.
3. **Thread** the wall hook's peg (14 mm square, tilted 12° up, capped by a red
   knob) through the 36 mm hanging loop at the end of the pan's handle — an
   aperture-over-peg alignment with 11 mm radial clearance, past the knob.
4. **Release** so the pan hangs freely under gravity, handle up, every body
   point airborne (clear of deck, burner and ground).

A solver needs a different plan (reorientation → axis alignment → translation
along a horizontal axis → free-hanging release) and different code structure
(there is no "place on surface" anywhere in the goal). It is also distinct from
every other task in this corpus: no other task threads an aperture onto a wall
fixture or certifies a free-hanging suspension outcome.

## Scene

Fully procedural (native PhysX box colliders): kinematic counter **deck**
(720×600×24 mm), kinematic **wall** (face at x = 0), kinematic lit **burner**
compound (dark base + glowing hot plate), kinematic **hook** compound (plate +
130 mm peg + red knob), dynamic 300 g **pan** compound (140 mm body, 30 mm rim,
150 mm stick handle ending in a loop: outer 52 mm, aperture 36×36 mm).

Randomization (all readback-verified in smoke): burner xy jitter, pan free yaw +
xy jitter on the burner, hook height 405..470 mm and lateral ±140 mm on the wall.

## Rubric

Latched partial credit, capped at 0.60 unless success holds live:

| credit | clause |
|---|---|
| 0.15 | pan ever clear of the burner (latched) |
| 0.15 | pan ever vertical (handle-up) while clear (latched) |
| 0.30 | hook peg ever through the loop's aperture, calm (latched) |
| 1.00 | `success()` live |

`success()` = peg geometrically through the loop aperture **retained for 60
consecutive free physics substeps** (`_thr_streak`, updated only in
`post_step`) AND suspended (all 12 pan sample points > 30 mm above the deck)
AND handle-up (< 35° from vertical) AND no violent motion AND finite.

**Retention-based "hanging" (measured on the forge):** a ring hanging on a box
peg sustains a solver-driven residual swing (~1 cm, 0.05–0.13 m/s) that heavy
damping, extra velocity iterations and per-iteration external forces do NOT
kill — the contact solver re-injects energy each rock cycle. "Hanging" is
therefore certified by retention (a falling, bouncing or thrown-past pan cannot
stay threaded for 60 free substeps) while the velocity gates (0.50 m/s,
2.0 rad/s) only reject genuinely violent motion. Honesty asserts in
`SceneCfg.__post_init__` guarantee the threaded() acceptance span covers the
whole peg incl. knob, and that a pan on the lowest randomized hook still clears
the deck by > 5 cm.

## Solution outline (solve.py — teleport-transport contract)

All load-bearing interaction is contact dynamics or applied wrenches:

- **P0** settle; assert on-burner baseline, score ≈ 0.
- **P1** teleport-lift the pan straight up off the burner (transport) → 0.15.
- **P2** teleport to a hover pose in front of the hook, vertical, loop aligned
  with the peg axis (threaded() still False) → 0.30.
- **P3** PD wrench servo (force ≤ 12 N, torque ≤ 1 N·m, persistent-wrench
  zeroed afterwards) translates the pan along the peg axis, threading the loop
  over the knob onto the peg; retry loop backs the seat point off 10 mm per
  attempt → threaded.
- **P4** release (zero wrench), hands-off settle: the pan drops ~1 cm onto the
  peg and hangs; streak certifies retention → score 1.0.
- **P5** ≥ 3.3 s fully hands-off persistence (10 × 40 substeps, success every
  checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified on
the forge for seeds 0 and 1 (`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

The pan's 150 mm stick handle (22×12 mm cross-section) is a canonical Franka
pinch-grasp feature. Required motions: lift ~12 cm, a ~90° wrist reorientation,
a straight ~15 cm insertion along a horizontal axis with 11 mm radial clearance,
and a release — all within a 72×60 cm counter footprint with the hook 405..470
mm high, comfortably inside the Franka workspace. The 36 mm aperture vs 14 mm
peg (plus 20 mm knob to pass) is a forgiving insertion tolerance by
peg-in-hole standards.

## Execution order

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the task and the
   monotone score trace.
2. `smoke.py` — rejection battery, 9 checks:
   1. settle/no-NaN (start = seed's goal state, score 0);
   2. randomization readback (burner xy, pan yaw, hook y, hook z differ across
      seeds);
   3. hook spread over 10 resets (> 30 mm spans, in range);
   4. null policy 240 steps (score ≤ 0.05, no success);
   5. deck set-down (clear only, score ≤ 0.16);
   6. wall lean handle-up (touches deck, not threaded, score ≤ 0.31);
   7. cavity hang — pan hung by its BODY on the peg, airborne (threaded()
      False, score ≤ 0.31);
   8. grounded thread — genuinely hung by the loop on a lowered hook, lowest
      point inside the 30 mm clearance band (suspended() refuses, score capped
      at 0.60);
   9. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS <n>/<n>`) and hard-exit; a top-level try/except prints
a FAIL verdict on any exception so the forge watchdog never hangs.
