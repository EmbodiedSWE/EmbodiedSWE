# libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i141 — weigh the moka pot on a two-pan balance

## Provenance

Seed: `libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove` — grasp the
moka pot and set it down ON the flat stove: a single-object pick-and-place whose
success is a pose predicate (pot xy near the burner, z in a band), satisfiable
the instant the pot rests there.

## Strategic difference from the seed

The seed's GOAL state is this task's START state: the pot begins standing on the
(unlit) stove. The goal is not a pose at all — it is a **measurement** carried
out with a passive mechanism:

1. Move the pot from the stove into the **RED** hanging pan of a balance scale.
   Placing it there visibly *disturbs* the scene — the loaded beam slams to its
   −12° stop — instead of finishing the task.
2. The pot's mass is **hidden**: randomized per episode over
   {150, 200, 250, 300, 350} g through the physics view, with identical
   appearance. Select **labeled weights** from six on the counter (2×50 g
   silver, 2×100 g brass, 2×200 g copper — sizes/colors are the labels) and add
   them to the **BLUE** pan until their sum equals the pot's mass. The required
   subset differs every episode (a small subset-sum instance solved by
   iterating against the beam's response).
3. Let the mechanism answer: the beam's only restoring torque is its CoM 20 mm
   below the pivot, so the free beam returns **level** only under an exact
   counterweight (~2 g already tilts it visibly; a 50 g error slams it to the
   stop). Success is judged with everything at rest, hands off.

A solver needs a different plan (an iterative weigh–observe–adjust loop over
MULTIPLE objects, terminated by a mechanism's equilibrium, not by an object
pose) and different code structure (nothing is "move X to pose Y"; the terminal
predicate lives on the balance). Distinct from every other task in this corpus:
no other task measures a hidden randomized physical quantity with a passive
instrument, and none has a discrete combinatorial (subset-sum) decision at its
core. The sibling i75 (same LIBERO scene family) threads an aperture onto a
wall hook — different mechanism, different outcome class, different plan.

## Scene

Fully procedural (native PhysX box colliders), on a 900×750×24 mm kinematic
counter **deck**:

- **stand** (kinematic): base + column + bearing prongs; knife-edge pivot
  300 mm above the deck.
- **beam** (dynamic): 440 mm bar + needle, 0.40 kg, CoM 20 mm below the pivot,
  on an authored revolute X joint limited to ±12°; angular damping 2.0
  (ζ≈0.76 at the balance's slow mode, period ≈4.8 s).
- **basket_r / basket_b** (dynamic): red/blue hanging pans (crossbar hanger,
  masts, 150×140 mm fenced platform) on free revolute joints at beam
  y = ±200 mm — pans hang plumb at any tilt, so the payload lever arm is fixed
  at 200 mm regardless of where cargo sits on the pan.
- **pot** (dynamic): octagonal moka pot (63 mm across, 85 mm tall, side
  handle); its engine mass is the hidden quantity (set + read back through the
  physics view, inertia scaled with it).
- **six weights** (dynamic): octagonal pucks on a 2×3 grid whose assignment is
  permuted per reset; **frypan** (dynamic distractor, 230 g — provably ≠ any
  legal weight combination); unlit **stove** (kinematic) the pot starts on.

Randomization (all readback-verified in smoke): pot mass (5 choices), stove xy,
pot xy+free yaw, weight-grid permutation + jitter + yaw, frypan jitter + yaw.

Honesty-by-construction asserts in `SceneCfg.__post_init__`: a free level beam
certifies < imb_tol of imbalance and any imbalance ≥ imb_tol slams a free beam
to its stop (statics: tanθ·m_beam·d/L); every pot-mass choice is exactly
weighable; the frypan matches no combination by > 1.5× tolerance; the pot fits
the pan and passes under the hanger crossbar; grid weights can never spawn in
contact; tipped pans clear the deck; stove/weights spawn clear of the pans.

## Rubric

Latched partial credit, capped at 0.60 unless success holds live:

| credit | clause |
|---|---|
| 0.20 | pot ever at rest in the red pan (latched) |
| 0.20 | a labeled weight ever at rest in the blue pan (latched) |
| 0.20 | balanced: pot in red + weight(s) in blue + beam level (2.5°) and quasi-static + cached-mass imbalance ≤ 10 g (latched) |
| 1.00 | `success()` live |

`success()` (all live) = pot in the red pan AND ≥1 labeled weight in the blue
pan AND nothing misplaced (no weight in red, no pot/frypan in blue, nothing
parked on the beam) AND the free beam level (≤2.5°) and calm (≤0.15 rad/s) AND
every payload and pan at rest AND |Σm_red − Σm_blue| ≤ 10 g from masses **read
back from the physics engine at reset** AND finite.

The mass clause is the anti-pinning backstop: an external wrench (or gripper)
holding the beam level with the wrong weights passes every geometric clause but
not this one; conversely any honest level beam passes it automatically, because
a free beam holding level physically certifies < ~1.8 g of imbalance
(tan 2.5° · 0.40 kg · 0.02 m / 0.20 m — asserted in `__post_init__`). A beam
*swinging through* level is rejected by the calm/settled gates plus a
240-substep continuous-success requirement in the solve before its P3 boundary.

## Solution outline (solve.py — teleport-transport contract)

Teleports carry ONE object at a time through free space to a hover a few mm
above the destination pan's platform (inside its open top, below the hanger
crossbar); everything load-bearing afterwards is contact dynamics:

- **P0** settle 120 steps; assert pot on the stove, empty beam level, score ≈ 0.
- **P1** transport the pot into the red pan; it falls, seats, the beam slams to
  its −12° stop → pot latch, 0.20.
- **P2** oracle (solver-only): read the pot's randomized mass from the reset
  readback cache and pick the exact weight subset directly; transport the
  weights one by one into the blue pan (bounced releases retried — the final
  configuration is still 100 % contact-made) → weight latch, 0.40.
- **P3** hands off: the free beam swings back and settles level; the phase
  boundary requires success to hold for 240 consecutive substeps (2 s) so it
  marks a settled beam, not a mid-swing pass → 1.0.
- **P4** ≥ 3.3 s fully hands-off persistence (10 × 40 substeps, success at
  every checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified on
the forge for seeds 0 (350 g pot) and 1 (200 g pot)
(`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

Every required interaction is a tabletop pick-and-place of hand-sized rigid
objects inside a 0.90×0.75 m counter: the pot (63 mm octagonal body, 36×12 mm
side handle — a canonical pinch-grasp feature) and 26–44 mm octagonal pucks.
Drops into the pans are vertical placements into a 134×124 mm open top with
≥17 mm xy clearance for the pot (far looser for pucks), release height a few
mm — no insertion, no reorientation. The pan platforms ride ~190 mm above the
deck and the hanger crossbar leaves 130 mm of headroom over the platform, so a
top-down approach reaches in freely with the 85 mm pot. With the base at the
counter's +x edge, all targets (stand pans ~0.54 m, stove ~0.45 m, weight grid
0.25–0.40 m) sit inside the Franka's 0.855 m reach. The perceptual loop a
policy needs — read the beam/needle tilt, add or swap weights, re-observe — is
sized for the balance's deliberately visible response (2 g tilts it, 50 g slams
it).

## Execution order

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the task and the
   monotone score trace.
2. `smoke.py` — rejection battery, 12 checks:
   1. settle/no-NaN (start = the seed's goal state, beam level, score 0);
   2. randomization readback (stove xy, pot yaw, weight permutation differ);
   3. pot-mass spread (10 resets: ≥3 distinct engine-readback masses, all
      legal);
   4. null policy 240 steps (score ≤ 0.05, no success);
   5. deck set-down (pot off the stove onto the bare deck: score ≈ 0);
   6. pot-in-red only (pot latch 0.2, beam slammed to its stop);
   7. mirrored pans (pot in BLUE + exact weights in RED: genuinely level, yet
      score ≈ 0 — the colored placement clauses are load-bearing);
   8. near-miss 50 g (latches 0.4, beam at its stop, no balanced latch);
   9. frypan cheat (frypan engine mass set EQUAL to the pot's — instrumentation
      — and used as counterweight: genuinely level, score ≤ 0.205);
   10. shared red pan (pot + 50 g in red vs exact sum in blue: level, latches
       cap at 0.60, misplacement refuses success);
   11. anti-pinning (wrong weights + external PD torque wrench holds the beam
       level and calm — hold verified non-vacuous — cached-mass clause refuses:
       score ≤ 0.405);
   12. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS 12/12`) and hard-exit; a top-level try/except prints a
FAIL verdict on any exception so the forge watchdog never hangs.
