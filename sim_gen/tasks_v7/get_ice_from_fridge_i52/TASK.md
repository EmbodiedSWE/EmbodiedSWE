# get_ice_from_fridge_i52 — single-ball dosing ice dispenser (`simgen.ice_doser`)

## Provenance

- **Seed:** `rlbench/get_ice_from_fridge`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/get_ice_from_fridge.py`) — a Franka
  holds a cup against a fridge's ice-dispenser lever. Trajectory playback, one grasp,
  one continuous press-and-hold; the amount of ice is never modeled or judged.
- **This task:** the dispenser is rebuilt as a real machine — a VOLUMETRIC DOSING
  VALVE (candy-machine / ice-maker chute mechanism). A capped single-file hopper of
  5 ice balls (32 mm) stands over a horizontal shuttle riding in an enclosed tunnel.
  The shuttle carries a one-ball through-pocket and has an orange knob plate at each
  end; the knobs are the push handles AND the travel hard stops. At the CLOSED stop
  the pocket sits under the hopper throat (one ball loaded, outlet sealed); pushed to
  the OPEN stop, the pocket crosses the outlet in the tunnel floor — its one ball
  falls — while the shuttle body seals the throat behind it. One ball per full
  close→open→close cycle, **by geometry, not timing** (pocket depth == ball
  diameter, so the next ball shears off cleanly at the shuttle's top plane).
- **Goal:** stage the BLUE cup under the outlet (green marks), pump the shuttle
  through exactly **2** full cycles, push it back to its CLOSED stop. Red decoy cup
  stays empty; the other 3 balls stay sealed inside the machine.

## Strategic difference (vs the seed and vs neighboring tasks)

- Seed skill: *hold* a cup at a lever — one grasp + one sustained press; quantity
  irrelevant. Here the objective is a **discrete COUNT with failure on both sides**
  (1 ball fails, 3 balls fail), reached by a **reciprocating pump cycle** operated an
  exact number of times, then STOPPED. Smoke check 5 runs the seed's strategy
  literally (hold the valve open under force for 4 s): exactly ONE ball ever comes
  out and the score caps at 0.40 — the seed's plan cannot solve this task.
- A required **staging prerequisite** precedes any dispensing: the machine drops
  balls onto open ground if no cup is staged, and a ground ball is unrecoverable
  (sealed hopper, no way to reinsert) — physically enforced ordering, checked by
  smoke 6.
- Against tasks read during this session: push_button_i35 (rotate a cam wheel to
  depress a button), coke_task_i15 (beam balance), pour_water_i7 (ramp chock),
  peg_insertion_side_i1, pen_holder exemplar — none involve count-metering, a
  two-hard-stop reciprocating mechanism, or an irreversible-overfill objective.

## Solution outline (solve.py — passes seeds 0 and 1 on the forge)

1. **P0 read/settle** (180 steps): housing yaw ±180° + xy jitter, cup bearings, and
   shuttle stroke state read back live; baseline score ~0 asserted.
2. **P1 stage** (the ONLY teleport, transport-only): blue cup written upright onto
   the drop line under the outlet. Hands-off settle → `staged` latch (0.15).
3. **P2/P3 pump ×2** (real contact): a horizontal force ≤ 8 N — a fingertip on a
   knob plate — drives the shuttle under a velocity-limited PD law (≤ 0.08 m/s).
   Each stroke aims PAST the hard stop so knob-vs-housing contact parks it. Ball
   falls pocket→outlet→cup by gravity. Force RELEASED at every phase boundary;
   scores print hands-off (0.40 after cycle 1, 1.0 after cycle 2 with the shuttle
   re-closed).
   - Force-frame trap: `set_external_force_and_torque` applies the given vector in
     a rotating frame (body frame on this pod). The solve body-encodes the world
     push with `quat_apply_inverse(q_now, ·)` and carries a runtime probe that
     flips encoding if the shuttle measurably moves away from the stroke target
     (pod-dependent semantics; seed 1's yaw +140° exposed it).
4. **P4 persistence**: 3.3 s hands-off, success() still true → `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, parallel-jaw gripper)

- **Base pose:** on the cantilever (outlet) side, ~0.4–0.5 m from the housing — the
  cup slots (0.26–0.34 m ring) and both knobs are inside a Franka's ~0.85 m reach.
- **Stage the cup:** rim-pinch or two-finger cage on the 75 mm-tall, 84 mm-wide cup
  (μ 0.6), carried at ground level and released on the drop line — the tunnel floor
  is 100 mm up and the outlet side cantilevers over open ground, so there is 25 mm
  of free clearance above the cup rim and open approach from three sides.
- **Pump:** each stroke is a push-to-hard-stop primitive — closed fingertips press
  the protruding knob plate (90 × 47 mm face, 8 N, 60 mm travel) straight in until
  it stops; no grasp, no precision endpoint, both endpoints are mechanical stops.
  The two knobs alternate sides; both faces are open to the arm (knob bottoms ride
  above the pedestal top; the hopper shaft is between the two push lines but 170 mm
  up). No simultaneous contacts, no regrasping mid-stroke, no bimanual action.
- **Watch/stop:** the count is observable (balls visibly drop into the open cup) and
  the finish state is the rest state of the machine.

## Execution order (declared)

The task has a physically-enforced order: **the cup must be staged before the first
cycle** (a ball dispensed early lands on open ground and can never be recovered —
the hopper is capped and the tunnel enclosed, so the machine cannot be refilled).
Overfilling (a 3rd cycle) is likewise terminal: balls cannot be pushed back in.
Solve demonstrates the canonical order: stage → cycle ×2 → park closed → hands-off.

## Rubric

`score()` = latched stages, capped at 0.70; exactly 1.0 iff `success()` live:

| credit | stage |
|---|---|
| 0.15 | blue cup ever staged under the outlet (settled, upright, on the ground) |
| 0.25 | first ball in the blue cup, red empty |
| 0.30 | target count (2) in the blue cup, red empty |
| == 1.0 | success: blue EXACTLY 2, red 0, 3 balls retained in the machine, shuttle at its CLOSED stop, settled, finite |

Null policy ~0; seed strategy ≤ 0.40; unstaged dispensing ~0 and terminal;
overfill keeps earned stage credit but success is impossible.

## Verification (forge, RTX 4090 pod, Isaac Sim 5.1 / isaaclab 0.54.2)

- `solve --seed 0`: SUCCESS (score 0 → 0.15 → 0.40 → 1.0, 3.3 s persistence).
- `solve --seed 1`: SUCCESS (yaw +140.5° — the layout that catches force-frame bugs).
- `smoke`: **ALL PASS 15/15**, `frames.npz` (387, 600, 960, 3) recorded:
  1. settle/no-NaN — closed, 5 sealed, cups empty, score ~0
  2. randomization A — housing yaw span 250°, xy std 0.017
  3. randomization B — blue bearing span > 90°, separations ≥ 55°, radii in band
  4. null policy — nothing dispensed, score ~0
  5. SEED strategy — valve held open 4 s → EXACTLY ONE ball, score 0.40, no success
  6. unstaged dispense — ball lost on the ground, no latches, score ~0
  7. overfill — 3 balls constructed in blue → success refuses, score ≤ 0.70
  8. red-cup poison — blue 2 + red 1 constructed → all latches refuse, score ~0
  9. wrong cup — red staged, two REAL cycles pumped → both in decoy, score ~0
  10. shuttle left open — perfect fill but mid-stroke park → CLOSED clause refuses
  11. rim lean — ball outside the cup wall (58 mm off-axis) not counted
  12. tipped cup — upright gate refuses staged() and in_cup()
  13. rejection audit — success() never fired during the battery
  14. final no-NaN
  15. video — >10 frames recorded

*(near-miss-on-rim is N/A by construction: a 32 mm ball cannot rest on a 6 mm wall
edge; the closest analogues are checks 11/12.)*
