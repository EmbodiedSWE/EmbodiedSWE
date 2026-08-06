# coke_task_i20 — Runaway Can Rescue (`simgen.runaway_can`)

## Seed provenance

- **Seed id:** `simpler_env/coke_task`
- **Seed source:** `sim_gen/RoboVerse/roboverse_pack/tasks/simpler_env/_metasim/coke_task.py`
- The seed is the SimplerEnv pick-coke-can family: a **static** coke can on a table
  (orientation randomized), success the instant the can is **grasped and lifted** a few
  centimetres (`PickSuccessEvaluator` on grasp + Δz). One stage, no placement target, no
  time pressure — the world waits for the robot.

## What changed

The can is kept (330 ml cylinder, 33 mm radius × 115 mm), everything else is inverted.
At reset the can is **already rolling on its side** across a raised counter toward the
open far edge (spawn pose, roll speed 0.30–0.45 m/s, and heading all sampled; launched
with the matched no-slip angular velocity so the roll is clean). Below the edge is the
floor: a can that falls is ruined — a **`lost` latch** (set every substep by real fall
physics) zeroes the episode irreversibly. A flat green coaster pad (position sampled)
lies flush on the counter behind the spawn.

**Stages (declared: 3, order REQUIRED — difficulty tier: medium):**

1. **Intercept** — arrest the rolling can on the counter before the edge. The sampled
   roll speed sets a hard reaction deadline of ~1.0–1.7 s (measured by the smoke's
   calibration probe; worst case asserted ≥ 0.9 s).
2. **Right** — stand the lying cylinder upright (either end up, ≤ 15° of vertical).
3. **Park** — place it standing on the coaster pad (≤ 45 mm of pad centre, base at pad
   height) and leave it settled.

The order is physically forced: nothing can be righted or parked after the can is lost,
and the can is lost unless intercepted first.

## Why strategically different (not same-strategy-different-numbers)

- **The world acts first.** The seed's scene is static and patient; here the episode
  opens mid-crisis with a moving object and a countdown. The core new skills —
  perceiving motion, meeting a reaction deadline, arresting momentum — do not exist in
  the seed at all.
- **The seed's plan is a tested failing control.** "Grasp the can and hoist it" (the
  seed's entire success condition) is expressed in the smoke as a kinematic
  grab-and-hold 20 cm above the counter: it never succeeds and earns at most the 0.2
  intercept credit. Lifting, the seed's goal, is worth nothing here by itself.
- **Success judges placement + orientation + history, not grasp.** The seed checks a
  grasp flag and a height delta; here success is a settled upright pose on a sampled
  pad **plus a clean episode history** (the can never fell) — an irreversible-loss
  latch the seed has no analogue of.
- **Sibling differentiation:** `pull_cube_tool_i1` claims *sending* an object away with
  a calibrated impulse (momentum production); this task is the arrest of momentum the
  environment produced — no impulse calibration, no distance control, and the deadline
  + righting + irreversibility axes are claimed by no sibling (checked against
  `simgen-batch-task-strategies` and sibling TASK.md files).

## Rubric (graded, latching)

- `score() = 0` for doing nothing (a rolling can carries no credit) and **0 forever
  once lost**; else `0.2 ·` intercepted-latch (can came to rest on the counter) `+
  0.3 ·` righted-latch (upright + settled on the counter); **exactly 1.0 iff
  success()**. Max non-success score 0.5.
- `success()`: current-state physical predicate — upright (≤ 15°), on the pad
  (≤ 45 mm, base at pad height), settled (|v| < 0.04 m/s, |ω| < 0.5 rad/s) — gated by
  `~lost`.
- Physics honesty: the smoke's oracle places the can kinematically (teleport-oracle),
  but every judged outcome (rest, standing stability, the fall) is settled real
  physics; the pad tolerance (45 mm) is under the fully-on-pad geometric limit
  (80 − 33 = 47 mm).

## Smoke check list (20 checks)

1. Reset state finite AND the can is genuinely rolling (speed readback in band).
2. Score exactly 0 at reset (null slate).
3. Randomization is real by READBACK (spawn y, roll speed, pad x/y all move).
4. Null policy: the unattended can rolls off the edge and is LOST (threat is real).
5. Null policy: lost episode scores 0, no success.
6.–8. Oracle (intercept → right → park) reaches `success()` + score 1.0 on 3 seeds.
9. Ladder: intercepted → score 0.2.
10. Ladder: righted off-pad → score 0.5, not success.
11. Ladder: parked → success, 1.0.
12. Ladder: non-decreasing, partials < 1.0 (rubric monotonicity).
13. **Negative A (seed strategy):** grab-and-hoist, held 20 cm up — never success,
    score ≤ 0.2.
14. **Negative B (anti-cheat):** can falls, then is teleported upright onto the pad —
    lost is irreversible: no success, score 0.
15. **Negative C (near-miss):** upright 10 cm from pad centre → 0.5 only, no success.
16. Negative C recovery: moved onto the pad → success (tolerance honest both ways).
17. **Negative D:** can lying on its side ON the pad — no righting credit, no success.
18. Calibration: every forced roll speed reaches the edge.
19. Calibration: time-to-edge monotone decreasing in speed.
20. Calibration: worst-case reaction budget ≥ 0.9 s (interceptable, not a coin flip).

Video frames are saved to `frames.npz` in the working directory.

## Files

- `scene.py` — `RunawayCanSceneCfg` + `RunawayCanScene` (`SCENES.register("runaway_can")`,
  `register_env("simgen", …, robot="null")`); fully procedural primitives (kinematic
  counter + flush pad, one dynamic cylinder), no external assets.
- `smoke.py` — standalone forge battery (AppLauncher `--headless`, `enable_cameras`,
  RTX driver-check override); exports `oracle_solution(scene_or_env)`; prints
  `SIM_GEN_SMOKE: ALL PASS n/n` and hard-exits via watchdog.
