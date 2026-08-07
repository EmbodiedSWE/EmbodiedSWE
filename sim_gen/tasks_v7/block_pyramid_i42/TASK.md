# block_pyramid_i42 — Tilt Labyrinth (`simgen.tilt_labyrinth`)

## Seed provenance

Derived from **rlbench/block_pyramid** (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/block_pyramid.py`):
stack six green 37.5 mm cubes into a 3-2-1 pyramid among six red distractors. The seed's
strategy is *repeated color-selective pick-and-place*: every judged quantity is a block
pose, and the plan is "grasp a green cube, place it, repeat".

## What changed, and why it is strategically different

Kept from the seed only its abstract germ — *color tells you which target counts* (green
= goal, red = forbidden) and *a multi-stage buildup toward a goal configuration*.
Everything about the strategy is different:

- **Nothing is picked, placed, or stacked — the payload must never be touched.** The
  only free object is a 30 mm ball riding a maze tray on a two-axis gimbal; the robot's
  only useful contact is pressing the tray's amber rim tabs to TILT it. The task is
  indirect, closed-loop control of the gravity vector, not pose-to-pose transport.
- **Continuous feedback replaces discrete pick/place cycles.** The tray is bottom-heavy
  and self-levels, so nothing can be "set and forgotten": steering the ball along the
  serpentine (east leg → gap → middle leg → gap → last leg) requires sustained,
  reactive two-axis plate control, including actively *skirting* the red-bordered trap
  hole mid-route.
- **The goal state is a containment event under the tray** (ball settled in the green
  catch box after falling through the green-bordered hole), not an assembled structure
  on top of a surface.
- **Versus the read corpus:** no existing task controls an object by reorienting its
  supporting surface. `carousel_airlock` sweeps a ball with a driven rotor vane;
  `gap_ferry`/`tile_shunt` push the payload directly; the balance tasks
  (`counterweight_scale` i40, beams) judge equilibrium angles, not path-following. The
  gimbal-plate control loop, the hazard-avoidance path constraint, and the
  pathway-gated containment rubric appear nowhere else in tasks_v4–v7.

## Apparatus (fully procedural)

Heavy dynamic BASE (slab, journal posts, sealing guard skirt + rim cover, green goal
catch box under the goal hole, red trap box under the trap hole) → FRAME on a revolute
X-joint (pitch) → maze TRAY on a revolute Y-joint (roll), both limited to ±7°. The tray
has an authored CoM 60 mm below the pivot (self-levelling pendulum, ~1.5 N m/rad), a
serpentine corridor of three legs (two interior walls, 70 mm gaps at alternating ends),
a 40 mm red-bordered TRAP hole in the middle leg, a 70 mm green-bordered GOAL hole at
the end of the last leg, and four amber press tabs on the rim. The skirt + rim cover
seal the under-tray volume: the only way into a catch box is through its hole in the
tray (verified by a smoke drop test).

**Randomization (readback-verified):** whole-apparatus yaw ±180° + xy jitter ±3 cm; ball
start position sampled along the start leg. Memorized world-frame tilt sequences fail.

## Rubric

- `success()` = ball settled inside the GREEN catch box **and** all three corridor-stage
  latches were earned while rolling on the maze floor (a ball that appears in the box
  without traversing the serpentine is refused — smoke checks 6 and 7).
- `score()` (monotonic, latched): 0.20 middle corridor, 0.50 north corridor, 0.70 goal
  approach, 1.0 iff success. Latches exclude the hole openings, so falling through the
  trap earns nothing; credit never evaporates during the airborne goal drop.

## Solution outline (solve.py — the legitimacy certificate)

**No teleports at all**: the ball spawns on the tray at reset and is never written
again. The only actuation is a bounded external torque on the tray body (≤ 0.40 N m —
the moment of a ≤ 1.8 N fingertip press at the 0.228 m tabs), through a nested
controller: outer loop maps ball-position error (tray frame) to a desired downhill
direction (≤ 5°); inner loop torques the tray's up-vector onto it against the
self-levelling moment. Phases: P0 settle/readback → P1 start leg + first gap (0.20) →
P2 wall-hugging trap pass + second gap (0.50) → P3 goal approach, torque cut at the
drop, gravity landing (1.0) → P4 hands-off persistence 3.4 s → `SIM_GEN_SOLVE: SUCCESS`.
Verified on the forge for seeds **0, 7, 13** (19–24 s each), scores non-decreasing
0 → 0.20 → 0.50 → 1.0.

## Embodiment argument (Franka, one base pose)

Base at ~(0, −0.55, 0) facing the apparatus (tray pivot at z 0.24, tabs at z ≈ 0.226 on
a 0.228 m radius — comfortably inside a Franka's reach envelope from one pose, apparatus
footprint 0.70 m). Per-object contact strategy:

- **Amber tabs (the only intended contact):** closed-gripper knuckle press straight
  down on a tab; ≤ 1.8 N produces the full working torque (0.40 N m). Two tabs at
  adjacent side midpoints (e.g. south + west from the stated base pose) span both tilt
  axes; press-and-ease modulation gives proportional control, and releasing lets the
  bottom-heavy tray self-level — so single-arm sequential pressing suffices.
- **Ball:** never touched (explicit task rule; solve.py touches it with nothing but
  gravity and the maze floor).
- **Base/skirt/boxes:** static furniture, no contact needed; the sealed underside means
  the robot cannot (and need not) reach into the boxes.

## Execution-order declaration

No discrete execution order is declared: the serpentine geometry itself enforces the
route (walls + the sealed underside make the corridor the only path to the goal box),
and the stage latches simply record that enforced traversal.

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 14/14` on forge)

1. Settle/no-NaN: ball at rolling height in the start leg, tray self-levelled (readback).
2. Score ~0 at reset, no success.
3. Randomization readback: apparatus xy + yaw vary across 6 seeds.
4. Randomization readback: ball tray-local start varies.
5. Null policy (300 steps): ball still in the start leg, score 0.
6. **Seed-strategy analog** (the seed's literal stack-a-pyramid end state has no analog
   here — documented N/A — so its *strategy*, pick-and-place to the target zone, is
   probed): ball set down beside the goal hole; it even rolled in, and was still
   refused (mid-corridor latch unearned), score ≤ 0.70.
7. Direct-to-box teleport: physically in the green box and settled → refused, score 0.
8. Wrong place: teleported into the red trap box → refused, score 0.
9. Trap drop-through: falls through the red hole; hole-opening exclusion keeps all
   corridor latches False → score 0.
10. Sealed underside: ball dropped onto the tray-rim gap rests ON the rim cover, enters
    neither box.
11. Latched credit survives the ball being removed.
12. Monotonicity: deeper floor placements latch strictly more credit (0.20 < 0.50 < 0.70).
13. Rejection audit: success() never True anywhere in the battery.
14. Final no-NaN.
