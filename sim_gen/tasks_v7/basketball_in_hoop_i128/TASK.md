# basketball_in_hoop_i128 — `crater_run`

Roll the ungraspable 90 mm orange ball from the floor-level foot bay up a walled
switchback ramp — lower incline, flat 180° turn pocket, upper incline, summit
landing — and push it over the low red lip so it drops into the raised sunken
crater basin and rests inside.

- **Env name:** `simgen.crater_run` (robot `"null"`, `env_spacing=6.0`)
- **Package:** `scene.py` (scene + rubric), `solve.py` (legitimacy certificate),
  `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `rlbench/basketball_in_hoop`
(`RoboVerse/roboverse_pack/tasks/rlbench/basketball_in_hoop.py`): a Franka grasps a
ball off its stand, carries it above a hoop, releases; a `DetectedChecker` with a
`RelativeBboxDetector` under the hoop judges the drop. One grasp, one transport, one
release — **gravity is the delivery mechanism**.

## What changed and why it is strategically different

| Axis | Seed | This task |
|---|---|---|
| Contact strategy | grasp–carry–release | **push-only**: the ball (Ø90 mm) exceeds the Franka jaw span (80 mm, asserted in the cfg) — no grasp exists |
| Role of gravity | delivery (the drop scores) | **adversary**: the goal is ~0.18 m *up* a ramp; a released ball mid-incline rolls straight back to the bay (proved in smoke) |
| Plan shape | one open-loop pick/place/release | **sustained closed-loop conveyance** along a constrained 3D path with a 180° direction reversal, ending in a controlled drop over a lip |
| Scene reading | fixed hoop pose | **chirality randomization**: the whole ramp mirrors at random per episode (two complete kinematic twin structures, one swapped into the workspace, one parked in a depot), so the route must be read, not memorized |
| Judged region | box below the release point | sunken basin behind rim walls: containment window + settle gates, with lip/rim/landing/floor near-misses all geometrically rejected |

Distinct from the corpus ball tasks: `roll_ball_i81` (arch_quarry — demolition then
lift into an elevated quarry, grasping allowed), `hockey_i325` (gate_goal — barrier
extraction then a floor-level push through a gate), `hit_ball_with_queue_i77`
(skyway_bridge — build infrastructure, then passive descent delivers the ball).
None require sustained anti-gravity conveyance of an ungraspable object along a
switchback with a direction reversal, nor mirrored-route randomization.

## Rubric

- `success()`: ball center inside the basin containment window (canonical frame:
  `|x−basin_cx| ≤ 0.045`, `|y−laneB_y| ≤ 0.045`, `z ∈ [0.155, 0.205]`) **and**
  settled (`|v| < 0.05`, `|ω| < 1.5`). `__post_init__` asserts the window accepts
  every physically-possible in-basin rest and rejects landing / lip-top / rim-perch /
  floor rests, that the lip and walls physically retain the ball, and that the
  channel passes it.
- `score()`: latched stages, 0.15 each — upper half of incline A, turn pocket, upper
  half of incline B, summit landing — capped at 0.60; 1.0 iff `success()`. Credit
  never evaporates (verified in smoke); null policy scores ~0 (ball spawns in the
  flat foot bay).

## Teleport-solution outline (`solve.py`)

Uses **no teleports at all** — the ball starts in the workspace and every meter is
contact dynamics (the stand-in for pushing with the closed gripper):

1. **P1 climb A** — horizontal velocity-servo force `F = kv(v_des·dir − v_xy)`
   (kv=6, v_des=0.18 m/s, cap 4 N; world-frame, re-encoded into the tumbling ball's
   body frame every step) conveys the ball up the lower incline to the pocket entry.
2. **P2 turn** — same servo steers it around the 180° pocket (v_des=0.15).
3. **P3 climb B + landing** — up the upper incline onto the summit landing.
4. **P4 lip drop** — a faster shove (v_des=0.30) carries it over the 8 mm lip; the
   wrench is cleared the instant the basin window reads True; gravity + contact
   finish. The success state is never spawned.
5. **P5 persistence** — ≥3.5 s hands-off, success must hold.

Canonical waypoints are mapped through the scene's `canon_to_world`, so one list
serves both chiralities. Gains respect the one-substep wrench delay
(`kv·dt/m = 0.20 ≪ 1`); stalls escalate the gain, never the cap. Passes seeds
0, 1 (mirrored), 2 — `SIM_GEN_SCORE` 0.00 → 0.30 → 0.30 → 0.60 → 1.00 → 1.00,
`SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka + parallel jaw, OSC)

Base at structure-frame ≈ (−0.05, −0.42) — facing the lane-A side, mirror-symmetric,
so one base pose serves both chiralities (the mirrored structure presents its lane-A
side the same way after the robot walks around / the scene is mirrored about the
robot's sagittal plane).

- **Ball (only manipuland):** closed-fist / fingertip pushing. Contact heights
  0.075–0.235 m, all push points within ~0.66 m reach. The grey channel walls top
  out ≤ 0.26 m — the arm reaches over them from outside the channel; the switchback
  brings the ball back past the robot, so the far pocket (x ≈ 0.37) is the maximum
  excursion. The final lip shove is a straight push along lane B.
- **Structures (kinematic):** never manipulated; they are the track. No other
  objects exist.

## Execution order

1. `scene.py` written first (geometry derived from cfg constants; honesty asserted
   in `__post_init__`).
2. `solve.py` iterated on the forge until `SIM_GEN_SOLVE: SUCCESS` on seeds 0/1/2.
3. `smoke.py` rejection battery (19 checks) on the forge.
4. `TASK.md` + final clean runs.

## Check list (smoke, 19/19)

1. settle/no-NaN + layout sanity (active at workspace, twin parked, ball in bay)
2. score ~0 at reset, no success
3. randomization readback: both chiralities, yaw/xy jitter real (8 seeds)
4. randomization readback: ball start varies, tracks its structure
5. null policy: 240 idle steps → score ~0
6. seed strategy (grasp-carry-drop) documented N/A: Ø90 mm > 80 mm jaw span
7. gravity adversary: released mid-incline ball rolls back to the bay
8. landing near-miss (against the lip, landing side) rejected by x and z windows
9. latched credit survives teleporting the ball away
10. lip-top perch rejected by the height window
11. rim perch rejected by the height window
12. floor ball (knocked off) rejected below/outside the window
13. settle gate: in-basin but moving is not success
14. all four latches + parked elsewhere → score == 0.60 cap, not success
15. wrong-chirality (mirror-image) basin position is empty space → rejected
16. wall retention: quasi-static 3 N push moves the ball 165 mm, never over a wall
17. rejection audit: success() never True at any judged point
18. final no-NaN
19. camera ≥ 20 rgb frames → `frames.npz`
