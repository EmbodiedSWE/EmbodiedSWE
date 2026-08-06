# chute_dispatch (`pull_cube_tool_i1`)

**Seed:** `maniskill/pull_cube_tool`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/pull_cube_tool.py`) — an L-shaped
tool lies within reach; the solver must grasp it and use it to PULL an out-of-reach
cube inward, until the cube is within `reach_distance` of the arm base.

**Scene:** `simgen.chute_dispatch` (robot="null", scene-level; `solve.py` builds its
own Franka env).

## What changed

| | seed | this task |
|---|---|---|
| object | cube out of reach | cube within easy reach |
| goal | near the robot base | a walled pen far BEYOND reach |
| means | grasped L-tool extends reach; quasi-static drag | gravity transport down a fixed slick chute |
| choice | none | two identical chutes; the GREEN floor mat marks the target pen, gray is a dead-end decoy (side sampled per episode) |
| commitment | drag is correctable | release is irreversible (pens/lower chutes unreachable) |

Geometry: two open-top slick channels (95 mm wide, mouths raised 105 mm, 13.5°
slope, μ≈0.06 combine-min) descend from within reach (mouths ~0.6 m from base) into
two fully-walled pens (interior 0.70–0.92 m in x — the whole pen is ≥ 1.0 m from the
recorded base, beyond a Franka's ~0.85 m envelope). The only physical way into a pen
is down its chute. Randomization: green-target side (left/right) + cube spawn x, y,
yaw (verified by readback in smoke).

## Why strategically different

The seed's plan is *tool-mediated reach extension toward the base*: fetch tool, hook
behind the cube, drag inward. Here that plan direction is exactly wrong (the goal is
away, not toward — smoke's seed-strategy control parks the cube near the base and
scores 0), there is no tool, and no contact the arm can make moves the cube to the
goal directly: the solver must (1) READ the scene (which side is green), (2) do a
short in-reach pick, and (3) COMMIT the cube to environmental gravity transport down
the correct guideway, accepting irreversibility (wrong chute = unrecoverable decoy
pen, tested). Skills: goal selection + committed gravity delivery through a fixture,
vs. the seed's reach-extension dragging. Proximity earns nothing (tested near-miss
controls); teleport/airdrop arrivals are rejected by the per-substep in-channel
transit latch (tested).

## Solution (solve.py — the feasibility certificate)

Franka base at **(-0.32, 0, 0)** on the ground, facing +x (recorded here; chosen in
`solve.py`). OSC task-space control, arm + gripper commands only; no task-object
state writes. Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):

1. read `target_side` (green-mat side) and live cube pose;
2. HOVER/DESCEND over the cube (jaw azimuth snapped to cube yaw), slow-ramp CLOSE
   (19 mm target), LIFT to 0.26 m carry height, lift verdict (rise > 50 mm + jaw
   width band 40–52 mm) with up to 4 re-pick attempts → score 0.15 (lift latch);
3. CARRY closed-loop on the cube xy to the drop point (0.36, ±0.17), LOWER to
   0.165 m (fingers stay above the channel wall tops), open, retreat;
4. WATCH: the cube falls ~60 mm into the channel, slides down, crosses the gate
   plane (transit latch → 0.5), rests inside the green pen → success, score 1.0;
   hold 2 s and re-verify (persistence) before printing `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge on seeds 0, 2 (left lane) and 1 (right lane), 3/3 SUCCESS,
scores monotone 0 → 0.15 → 1.0, ~15.5 s sim time each.

## Execution order

None required: the task is a single delivery; the only "ordering" is physical
(reading the mat before committing is the solver's problem, not a rubric clause).

## Rubric

- `success()`: cube settled (|v| < 0.04 m/s, |ω| < 0.5 rad/s) resting on the floor
  (z within 12 mm of rest height) fully inside the green pen (cube half-diagonal
  containment margin), AND that pen's chute transit latch fired this episode
  (per-substep gate-plane crossing inside the channel with dx ∈ (0, 5 cm) — a
  kinematic teleport never latches).
- `score()`: 0.15·lifted (continuous-rise latch, z > 0.12 with per-step dz ∈
  (0.5 mm, 3 cm)) + 0.35·correct-transit; exactly 1.0 iff success. Null ≈ 0;
  wrong-chute delivery keeps at most the lift credit; latched credit never
  evaporates on the correct trajectory.

## Checks (smoke.py — rubric rejection battery, ALL PASS 19/19 on the forge)

1. settle/no-NaN + score ~0 at rest (2)
2. randomization readback: cube spawn moves; green side varies; mats mirror
   `target_side` (3)
3. null policy 240 steps → score ~0, no success (1)
4. lift latch: one-write teleport up does NOT latch; continuous rise does (2)
5. latched credit survives set-down; full ladder 0 → 0.15 → 0.15 → 1.0 (2)
6. physical chute ride accepted on 3 seeds with ±15 mm mouth offsets; success
   persists 240 further steps with no flicker (4)
7. anti-cheat: teleport straight into the green pen → in-pen + settled but NOT
   success, score 0 (1)
8. airdrop over the pen walls (the carry-over-and-drop cheat) → NOT success (1)
9. wrong (decoy) chute: real ride, rests in gray pen → NOT success, score ~0 (1)
10. near-misses: settled against the pen's outer wall / beside the chute exit →
    no success, score ~0 (1)
11. seed-strategy control: cube brought toward the base → score ~0, no success (1)

Seed-strategy end state is expressible (control #11); no tool exists here, so the
tool-specific half of the seed plan is N/A by construction.
