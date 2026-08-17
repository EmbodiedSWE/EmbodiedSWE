# open_wine_bottle_i99 — swing-top bottle: snap the bail, free the stopper, stand it on the coaster

Env: `simgen.swing_top_bottle` · Scene: `swing_top_bottle` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed task: **rlbench/open_wine_bottle** — a Franka grips the cap of a wine bottle and
pulls it straight off. One object, one grasp, one vertical motion; success is "cap is
off the bottle".

## Strategic difference

The seed's entire plan is *pull the cap off*. Here that exact plan is **physically
impossible**: the stopper is held captive by an over-center **swing bail** (Grolsch-style
flip-top lever on a side hinge, crossbar authored 8° *past* top-dead-centre over the
cap). Pulling the stopper up presses its cap into the crossbar, and the past-TDC
geometry converts the pull into *closing* torque on the hinge — pulling harder jams
harder (smoke check 4 shows 3× the solve's force rises 7.6 mm to the crossbar and
stalls; clearing the bore needs 52 mm). The required plan is a forced three-stage
sequence whose first action is not a grasp at all:

1. **Unlatch** — a sideways fingertip *push* on the crossbar through the hinge
   dead-centre (~51° apex). The mechanism is bistable under gravity: cut the push past
   the apex and the bail snaps the rest of the way to its 125° stop by itself, and
   stays there at rest.
2. **Extract** — only now does the seed's vertical pull work.
3. **Place** — a positive placement objective the seed lacks: stand the stopper on a
   randomized coaster (it self-rights on its plug end: explicit CoM at the plug bottom).

Plus a **color-matched decoy**: a second identical bottle (amber vs green) that must
stay sealed, closed and upright — goal assignment, restraint, and free per-bottle yaw
(the robot must read each bail's heading; the push direction is cued in `describe()`
by the counterweight collars).

Vs siblings looked at: `close_grill_i8` (lid slam-vs-carry manner test), `open_oven_i7`
(dial turning), `close_fridge_i89` (carry-close retention of a loose egg), pen-holder
style pick-and-insert — none has a bistable over-center latch driven through its apex,
a contact interlock that *rejects force* rather than requiring it, or an
unlatch→extract order forced by joint-level geometry. Code structure is also distinct:
per-env authored revolute joints (bottle→bail), a scene-owned wrench plant
(`bail_drive` / `stopper_pull`) as the only actuation channel, and mechanism honesty
asserts in `__post_init__` (apex between rubric thresholds, swing arc clears the cap,
bail parts statically clear the hinge lugs, gravity-bistable at both limits).

## Solution outline (solve.py — the legitimacy certificate)

- **P0** settle 1.5 s; layout readback (slot swap, yaw, coaster); baseline score ≈ 0.
- **P1 unlatch (applied torque):** 0.009 N·m about the live hinge axis (~2× the static
  gravity torque, ~0.17 N at the crossbar), CUT at 65° — past the 51° apex — so the
  snap to the 125° rest is the mechanism's own dynamics. `open` latch requires the
  bail at rest past 105°. Retry escalates torque ×1.6.
- **P2 extract (applied force):** 0.55 N vertical world pull (~1.9× stopper weight)
  lifts the plug out of the bore; `out` credit is *gated on the open latch*.
- **P3 place (teleport = transport only):** one root-state write carries the airborne
  stopper to a 20 mm hover above the coaster — the carry a gripper performs. Landing,
  self-righting and rest are gravity + contact; `placed`/success require it *resting*
  on the coaster.
- **P4** 3.3 s hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Non-decreasing `SIM_GEN_SCORE` prints at each boundary: 0.00 → 0.35 → 0.60 → 1.00.
Passed on the forge for seeds 0 and 1 (opposite slot swaps, yaws −173° and −91°).

Rubric: latched 0.10 moved (>25°) + 0.25 open (rest >105°) + 0.25 out (gated) + 0.10
placed, capped at 0.70; exactly 1.0 iff live success (bail open, stopper on coaster,
decoy sealed + closed, both bottles upright, all settled, finite).

## Embodiment argument (single Franka, parallel-jaw gripper, OSC)

Base at the world origin: bottles at (0.28–0.33, ±0.17) m, 21 cm tall; coaster at
(0.55, 0) ± 4 cm — everything inside a comfortable 0.25–0.7 m reach annulus below
0.25 m height.

- **Unlatch — closed-jaw fingertip push:** the crossbar is a 6 mm bar spanning 6 cm at
  z ≈ 0.24 m, exposed above the bottle on all sides at any yaw. The needed force is
  ~0.2–0.5 N horizontally at the bar (0.009 N·m over the 5.3 cm lever) — an OSC
  cartesian push with closed fingertips, direction read from the scene (push away from
  the dark collars, toward the crossbar's offset side). Past half-travel the arm
  retreats; the mechanism finishes itself.
- **Extract — parallel-jaw grasp:** the freed cap is a 2.6 cm cylinder standing proud
  of the rim — a canonical top grasp well inside the jaw span; the 0.55 N ≈ 56 g
  equivalent lift is a trivial payload; 6 cm straight-up stroke clears the bore.
- **Place:** carry ~25 cm, set down on the 9 cm coaster inside the 3.7 cm tolerance,
  open jaws, retreat. The stopper stands on its flat plug end (CoM at the plug bottom).
- **Restraint:** 26 cm slot separation and free space around both bottles let the
  planner keep clear of the amber decoy throughout.

## Execution order

`scene.py` (registers `swing_top_bottle` + `simgen.swing_top_bottle`) is imported by
both entry points; then `python -m simgen_tasks.open_wine_bottle_i99.solve --headless`
(optionally `--seed N`); then `python -m simgen_tasks.open_wine_bottle_i99.smoke
--headless` (writes `frames.npz` in the CWD).

## Checks

`solve.py` asserted milestones: finite reset, bails settle closed, stoppers seated,
bottles upright, baseline ≈ 0, no premature success; P1 rest-open latch and score
≥ 0.35; P2 gated `out` and score ≥ 0.60; P3 success and 1.0; P4 persistence ≥ 3 s.
Forge: SUCCESS on seeds 0 and 1.

`smoke.py` — 14 named checks, forge `ALL PASS 14/14`:
1. settle — clean reset (finite, closed, seated, upright, score ~0)
2. random — swap/xy/yaw/coaster draws differ across 6 seeds; both slots seen
3. null — 2 s of nothing: score < 0.05, no success
4. interlock A — the SEED's plan (closed-bail pull, up to 3×) rises to the crossbar
   and jams; vacuity-guarded (the stopper must actually move), re-seats on release
5. interlock B — those pull attempts earn zero credit
6. accept A — the solve's finger-scale torque snaps the bail to rest past 105°
7. accept B — the *identical* pull clears the bore once the bail is open (the gate is
   mechanism state, not force budget)
8. negative — stopper stood on the floor beside the coaster: rejected, latched 0.60
9. exactness — stopper on the coaster: success() and score == 1.0
10. revocation — re-closing the bail revokes success; latched 0.70 cap remains
11. negative — decoy stopper out of its bottle: rejected, ≤ 0.70
12. negative — decoy bail driven open: rejected
13. negative — whole decoy linkage laid on its side (not upright): rejected
14. frames — 280 video frames recorded to frames.npz
