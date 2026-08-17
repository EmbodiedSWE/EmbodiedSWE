# stack_chairs_i110 — tuck each chair under the desk into the one bay it fits

## Provenance

Seed task: `rlbench/stack_chairs` — three chairs must be picked up and stacked
vertically on top of a target chair.

## What this task is

A kneehole desk (six kinematic cuboids: top slab, two outer walls, two inner
divider walls, all re-posed every reset) stands with its open side facing the
robot. Its underside is divided into **three bays of different widths — 116, 90
and 62 mm — whose left-to-right order is permuted every episode**. In front of
the desk, three rigid compound chairs (4 legs + seat + backrest) of widths
**100, 72 and 48 mm** (red / green / blue) are scattered with randomized
positions and yaws.

Goal: slide each chair **horizontally** under the desk's 80 mm-tall overhang
into the **only bay it fits through**, front-first, until it is genuinely tucked
(seat ≥ 20 mm behind the desk's front edge, laterally inside its bay, origin
height in the [30, 70] mm band, upright ≤ 15°, facing into the desk ≤ 35°, and
**sustained still for 20 substeps**). All three bays covered by exactly one
chair each ⇒ success. Judged entirely in the desk's (randomized) frame.

Assignment is **forced by physics, not by rubric identity checks**: the 100 mm
chair only passes the 116 mm bay, the 72 mm chair only the 90 mm bay, the 48 mm
chair fits anywhere but is the only one that passes 62 mm — so any complete
solution is the unique width-matching. The rubric never names a chair-to-bay
map; wrong assignments simply cannot reach the depth gate (smoke checks 7–8).

## Why strategically different

- **From the seed (`rlbench/stack_chairs`)**: the seed is vertical stacking —
  lift chair, place on chair, repeat; verticality and pile stability are the
  whole game. Here chairs must **never** be piled: a solver has to plan
  *horizontal under-overhang insertions* with a size-to-aperture matching
  subproblem. The seed's plan (grasp → lift → align above → lower) is useless;
  smoke check 6 constructs an honest settled chair-on-chair tower and shows it
  scores ~0. The failure modes are also disjoint: yaw wedging in a bay mouth,
  backrest colliding with the desk edge, stalling short of the depth gate —
  none exist in vertical stacking.
- **From other tasks in this corpus**: `stack_cups_i92` is flip-and-cap
  nesting of open containers; `pen_holder`-style tasks drop items *into* an
  upward-open container. This task's container is **side-open with a low roof**
  (80 mm clearance vs the 120 mm backrest ⇒ the chair only fits moving
  horizontally, upright, front-first — backward insertion is physically blocked
  by the backrest, smoke check 9), and the matching constraint is
  width-to-aperture, not size-ordering of nested objects. Drawer/slide tasks
  move a mounted prismatic body; here the manipulated bodies are free chairs
  and the "channel" is formed by scene geometry the chairs must thread.

## Solution outline (solve.py)

Transport is teleport-only; every load-bearing interaction is contact dynamics
through `set_external_force_and_torque`.

For each chair i (any order; the code goes red → green → blue):

1. **STAGE (teleport, transport only)**: place chair i upright, facing the
   desk, in front of its width-matched bay, seat front ~80 mm before the desk
   edge, then let it settle.
2. **TUCK (forces only)**: velocity-servo push along the desk's inward axis
   (feed-forward ≈ μmg plus 4.0·(v_des − v), v_des = 55 mm/s), lateral PD onto
   the bay centreline (±0.5 N), and a **steering torque** (pull-point lever
   0.06 m · f_lat + yaw PD, clamp ±0.02 N·m) because a CoM push on a sliding
   chair is yaw-unstable and a 106 mm diagonal wedges in a 90 mm mouth. The
   force frame is **probed at runtime** on the first chair (apply a small push,
   compare commanded vs measured displacement direction, flip encode mode if
   off) since this pod's `set_external_force_and_torque` frame handling is
   configuration-dependent. Stall ⇒ active retreat backoff (reverse servo 90
   steps, bump feed-forward, retry, ≤ 5 attempts). Dock when depth > 34 mm (or
   > 26 mm and stalled-still), drop the wrench, settle 60 steps, assert the
   bay reads `covered`.
3. `SIM_GEN_SCORE` printed at each phase boundary: 0.000 → 0.333 → 0.667 →
   1.000 (monotone; each tucked chair latches 0.4 tuck-credit and covered bays
   dominate the mean).
4. **Persistence**: ≥ 400 substeps (3.3 sim-seconds) fully hands-off; success
   and score re-asserted; then `SIM_GEN_SOLVE: SUCCESS`.

Passes forge seeds 0, 1, 2.

## Execution order

Any chair order works — the width matching makes assignments independent
(bays are separated by divider walls; a tucked chair never blocks another
bay's mouth). Declared order: red, green, blue.

## Embodiment argument (single Franka)

A single Franka with base at ≈ (−0.10, 0), facing the desk at x ≈ 0.45 m,
covers the whole workspace (chairs scatter at x ≈ 0.08, desk edge at
x ≈ 0.38). The chairs are 48–100 mm wide and 150 g — graspable across the
backrest (8 mm panel, well within the gripper's 80 mm span) for the transport
phase, and the tuck phase is a planar push the Franka performs with its
closed-gripper fingertips against the backrest panel at ~50 mm height,
exactly where the solve applies its wrench: the arm reaches under nothing (the
push point stays in front of the desk edge until the final 20 mm, and the
backrest remains proud of the overhang at dock, sticking out in the mouth of
the bay). Forces used (≤ 1.5 N push, ±0.02 N·m steering via off-centre
contact) are far inside Franka capability.

## Physically excluded wrong outcomes (verified in smoke)

- Misfit bay: 100 mm chair pushed hard at the 90 mm bay jams on the walls
  short of the depth gate (check 8, non-vacuous: it moves 74 mm first).
- Backward insertion: backrest (120 mm) taller than the 80 mm clearance —
  desk edge stops it at dep ≈ −81 mm (check 9).
- Two chairs per bay / piling: bays fit exactly one matched chair; a genuine
  settled 3-chair tower scores ~0 (check 6).
- On the desk top / in the mouth / still moving / non-upright: rejected by the
  z-band, depth gate, stillness gate, uprightness gate (checks 10–12).

## Checks (smoke.py): 15

1. Settle + layout sanity (physical readback: slab on walls, chairs upright).
2. Score ~0, no success at reset.
3. Bay-order randomization via **physical wall-gap readback** — 5 distinct
   left-to-right width orders across 8 seeds.
4. Desk yaw/x spread, chair scatter y/yaw spread (all physical readback).
5. Null policy 240 steps ⇒ score ~0.
6. Seed-strategy rejection: real settled chair-on-chair tower ⇒ ~0.
7. Dead-end: green in wide + blue in mid (both genuinely covered) ⇒ 2/3,
   never success.
8. Misfit jam (non-vacuous push probe: frame-probed velocity servo).
9. Backward insertion blocked (non-vacuous push probe).
10. Near-miss: 12 mm depth < 20 mm gate ⇒ not covered.
11. On top of the desk ⇒ rejected by height band alone.
12. Settle gate: covered chair kicked to 0.23 m/s ⇒ not covered while moving.
13. Latched tuck credit survives teleport-away; covered credit drops.
14. Audit: success() never True anywhere in the battery.
15. No NaN in any task-object state.

`SIM_GEN_SMOKE: ALL PASS 15/15`; frames.npz recorded (187 frames).
