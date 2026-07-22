# push_cube_ref — reference task (format exemplar)

| Field | Value |
|---|---|
| Seed | `RoboVerse/roboverse_pack/tasks/maniskill/push_cube.py` (ManiSkill PushCube) |
| Status | **Reference port, not a generated task** — this is the seed itself, re-implemented against `sim_gen.core` to demonstrate the task contract. |
| Strategy | Move the cube along the table into the goal disc; it must end settled on the surface. |

## Semantics

- **Instance distribution**: cube xy ~ U[-0.10, 0.10]²; goal placed at a uniformly
  random angle, 0.15–0.25 m from the cube.
- **Success**: cube center within `goal_radius` (0.10 m, xy) of the goal marker AND
  settled on the table (correct height, near-zero velocity) — hovering doesn't count.
- **Score**: graded progress `1 - dist/dist_initial`, capped at 0.95; exactly 1.0 on
  success.

## Checks (smoke.py)

settle/no-drift · no-NaN · determinism · randomization-is-real · null-policy fails ·
oracle succeeds on 3 seeds · rubric monotone on 3 seeds · near-miss outside radius
fails · hover-over-goal fails · offset sweep (knee at the goal radius).
