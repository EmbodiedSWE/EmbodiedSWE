# latch_drawer — press-to-open latch drawer, then retrieve the stashed cube

**Package:** `sim_gen/tasks_v2/libero_kitchen_scene1_open_top_drawer_i15`
**Scene/env:** `SCENES: latch_drawer`, `ENVS: simgen.latch_drawer` (registered `robot="null"`;
solve.py builds its own Franka env).

## Seed provenance

`libero_90/libero_kitchen_scene1_open_top_drawer`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_top_drawer.py`):
open the top drawer of a wooden cabinet. The whole seed task is one contact strategy —
**grasp the drawer handle and PULL** the prismatic drawer out — and the goal is the drawer
joint itself (`JointPosChecker` past a threshold). Bowl and plate are inert set dressing.

## What changed, and why it is strategically different

A solver needs a different PLAN, not different numbers:

1. **The seed's contact strategy is removed by construction.** The drawer front is a flush
   plate inside the cabinet face — no handle, no knob, no rim. There is nothing to hook or
   pinch, so grasp-and-pull cannot even start. The latch spring anchors the drawer shut:
   smoke proves a drawer state-teleported to 11 cm open snaps straight back closed.
2. **The opening motion is INVERTED (push-to-open).** The only way in is to press the
   plate INWARD (~14–20 mm, quasi-static), then withdraw: the latch clicks off and an
   ejection spring drives the drawer out on its own. The robot pushes with a closed
   gripper — no grasp anywhere in phase 1 — and never moves the drawer in the opening
   direction; the mechanism does the opening.
3. **The goal moved off the mechanism.** Drawer-open is NOT success (smoke's click probe
   ends with the drawer fully open and success() False). The drawer is transient access:
   sealed inside rides a red cube, and success is that cube settled on a green floor pad
   beside the cabinet. A same-size blue decoy on the opposite flank punishes
   wrong-object shortcuts.

## Execution order

Required and **enforced by physics**, not by code: the cube is enclosed (carcass top +
flush plate) until the drawer ejects, so press → eject → pick → place is the only
physically realizable order. No code-level ordering constraint exists.

## Mechanism (honest physics)

Cabinet carcass kinematic and never teleported; drawer = one compound rigid body on a
per-env authored X-prismatic joint (pair collision disabled, symmetric ±16 cm limits as
the GPU sign-convention hedge). post_step force law along the slide:

- **latched:** capped spring anchored at closed (k=240 N/m, cap 6 N → ~3.4 N at the 14 mm
  arming depth; snap-back damping 30 N·s/m so a yanked-open drawer rebounds without
  self-arming). Arming additionally requires |v| < 0.03 m/s (quasi-static press).
- **armed → withdrawn past 4 mm:** released; capped spring toward the 13 cm open stop
  (cap 2.5 N → ~4 m/s² ejection, under the cube's friction slip threshold, so the cargo
  rides the drawer out and stays in the front half of the cavity — exposed under open sky).
- One-sided stiff end-stop springs at +20 mm (press stop) and −130 mm (open stop).

## Solution outline (solve.py — the feasibility certificate)

Franka base **(-0.28, 0, 0)**, OSC, franka_session substrate vendored. Phases:

1. **PRESS** — hand pitched 30° forward (probed: a vertical finger column cannot cross
   the carcass top-panel front edge; 45° rides the q5 limit, 60° hits the q6 limit),
   jaw at a 12 mm gap, both fingertips on the plate center (z≈0.078); quasi-static lead
   ramp presses to ~16 mm (armed, `SIM_GEN_SCORE 0.15`); withdraw back-and-up; the
   spring ejection parks the drawer open (`0.30`).
2. **RETRIEVE** — top-down cage (64 mm) descent into the open cavity at the cube's live
   pose, squeeze to 19 mm, lift + width-band verdict (`0.50`).
3. **PLACE** — closed-loop carry to the pad center, lower to rest, slow release, retreat,
   settle until `success()` (`1.00`, `SIM_GEN_SOLVE: SUCCESS`).

Arm-only manipulation: no task-object writes, no external forces from solve.py.
Passed on seeds 0, 1 and 2 on the forge (~43 s each, all phases first-attempt).

## Rubric

`success()`: red cube center within 5 cm of the pad center, z in the pad-top rest band
(0.026–0.042 m — floor rest and stacking rejected), settled. `score()`: latched ladder
0.15 armed + 0.15 ejected (requires the released latch — a yanked drawer earns nothing)
+ 0.20 cube lifted + 0.25 cube on pad; exactly 1.0 iff success; null ≈ 0.

## Check list (smoke.py — rejection battery, no probe reaches success)

1. settle/no-NaN, drawer closed, score ~0, no latches
2. cube starts sealed (order-enforcer)
3. randomization readback across 6 seeds (pad polar + both flanks, cube offset, decoy)
4. null policy: score ~0, drawer closed
5. **seed strategy**: drawer teleported 11 cm open w/o press → snaps shut, zero credit
6. shallow 8 mm poke → springs back, not armed
7. mechanism click: 17 mm quasi-static press → armed 0.15 → ejects ≥9 cm, cube exposed,
   drawer-open-alone is NOT success
8. persistence: latched credit survives 240 steps, drawer stays open
9. near-miss: cube just outside pad tolerance → no success
10. wrong object: blue decoy on the pad → no success
11. stacked: red cube on the decoy above the pad → no success (z rest band)
12. wrong place: cube on the cabinet top → no success (lift latch = 0.50 rung)
13. monotonicity: 0 < 0.15 < 0.30 < 0.50 < 1.0
14. calibration: click probe ejects + exposes the cube on 2 more seeds
