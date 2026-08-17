# plug_charger_i230 — shutter-gated charging garage

Scene `shutter_garage`, env id `simgen.shutter_garage`.

## Seed provenance

Seed: `maniskill/plug_charger`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/plug_charger.py`).
The seed is a terminal INSERTION alignment task: a free two-prong charger lies on the
table, a wall-mounted receptacle is fixed and always exposed, and the goal is one
pose — prongs aligned with the holes, charger pushed in. Nothing gates access to the
receptacle, and the episode ends the moment the inserted pose is reached.

## What changed and why it is strategically different

The receptacle is buried inside a **roofed charging garage whose only entrance is a
sliding shutter**, and the shutter's state is itself part of the goal — the task is an
airlock cycle, not an insertion:

| | seed | this task |
|---|---|---|
| receptacle access | always exposed | behind a **closed sliding shutter** in a roofed bay; the doorway (95×75 mm) is the only way in (roof forbids top-loading, facade is solid elsewhere) |
| goal | one inserted pose | **cycle**: slide the shutter fully clear → drive the brick through the doorway prongs-first until the copper plate physically stops it → **slide the shutter closed again** |
| judged scope | charger pose only | settled brick seated on the plate (prongs-in, upright, on-floor, laterally centred) **AND** shutter back in its rails within 12 mm of centre |
| ordering | none | **mechanism-enforced both ways**: a closed shutter blocks entry at floor level (smoke drives 3 N into it and the tip never crosses the facade), and the brick must be fully inside before the shutter can close over the doorway |
| naive strategy | align and push | the seed's whole strategy — get the charger inserted and stop — caps at **0.70** here because the re-close is judged; doing nothing scores 0 |

Also different from every other task read this session:
- `plug_charger_in_power_supply_i21` (keyhole unplug): the seed's goal state is that
  task's reset state and the motion is unlock–EXTRACT–stow of a captive plug, with
  nothing inserted at all. Here the brick starts free and OUTSIDE, is inserted
  through a guarded doorway, and the mechanism (shutter) must be operated twice —
  opposite direction of travel, and the mechanism state is a goal predicate rather
  than a one-way geometric latch.
- `peg_insertion_side_i1` (ram-rod ejection): a free tool ejects a payload; here
  there is no tool and nothing is ejected.
- `peg_insertion_side_i2` (create-passage-then-fasten): builds a passage by seating a
  bridge, then threads a pin in; here the passage is a permanent doorway guarded by a
  movable shutter, and the final act is closing the guard, not fastening the payload.
- `robobench/suites/packing/scenes/pen_holder.py`: repeated open-top container
  insertion; here top-loading is physically impossible (roof) and the single
  insertion is bracketed by two mechanism operations.

## Scene

Fully procedural (axis-aligned `UsdGeom.Cube` parts, explicit 1 mm contact offsets).
Dock frame: origin at the facade's outer face centre at floor level, +y inward.

- **garage** — KINEMATIC compound: facade 390×12×100 mm pierced by a 95 mm-wide ×
  75 mm-tall doorway; side walls, back wall, roof (z 0.100–0.110 — no top access);
  copper plate (goal pad) on the bay floor at plate face y = 0.117; shutter rail
  channel on the outer face: soffit + outer lip (channel depth 30 mm) + a floor
  ridge **interrupted at the doorway** (segments flank the gap; the 130 mm slab
  always overlaps at least one segment — asserted in `__post_init__`), so the
  shutter is captive in its rails but the brick can pass at floor level.
- **shutter** — DYNAMIC 130×12×90 mm slab (0.40 kg) with a 36 mm face knob, riding
  the channel; at reset it is centred (doorway covered). Sliding ~112 mm to either
  side clears the doorway.
- **brick** — DYNAMIC charger: 60×90×50 mm body with two 8 mm brass prongs (total
  length 106 mm), 0.30 kg, spawned on the apron OUTSIDE the garage.

Randomization (verified by readback in smoke): garage xy ±40 mm + free yaw; brick
apron position (dock-frame xy) and relative yaw. All predicates are dock-frame.

`success()` iff, with brick AND shutter settled (< 0.05 m/s): brick docked (prong-tip
y in the 14 mm seat band ending at the plate face, prong axis within 30° of +y,
upright, on the floor, |x| < 30 mm) AND shutter closed (|x| < 12 mm, in rails).
`score()` = 0.20·door-open latch (gated in-rails, dead-banded) + 0.35·entry progress
latch (tip-depth fraction, gated to the doorway lane) + 0.15·dock latch, capped at
0.85; exactly 1.0 iff success. Insert-and-stop (the seed strategy) = 0.70.

## Solution outline (solve.py, the legitimacy certificate)

Transport-only teleportation: exactly ONE pose write, moving the free brick from its
random apron spawn to a staging pose on the apron centreline (tip still 47 mm
OUTSIDE the facade). Everything load-bearing is contact dynamics via
`set_external_force_and_torque`:

- **P1 open slide** — velocity-regulated push (≤ 6 N, friction-feedforward with
  stall escalation) on the shutter along the channel toward the side away from the
  brick, with a lateral PD holding the channel line and a yaw/upright steering
  torque (Kp = 0.5 N·m/rad, ≤ 0.25 N·m) that keeps the 130 mm slab from wedging in
  the 30 mm channel; slides until the doorway is fully clear (+3 mm margin).
- **P2 transport** — the single teleport to the staging pose outside the doorway.
- **P3 doorway push** — velocity-regulated floor-level push (≤ 4 N) driving the
  brick prongs-first through the doorway until the **copper plate physically stops
  it** (goal tip-y set 0.5 mm past the plate face; a stall under forward press with
  the tip inside the seat band is accepted as plate contact — the plate, not the
  controller, defines the stop; rest tip_y = +0.1166/+0.1167 vs plate face 0.117);
  release and settle → docked.
- **P4 close slide** — same regulated slide back to centre; settle → success.
- **P5 persistence** — ≥ 3.3 simulated seconds hands-off; SUCCESS only if
  `success()` holds throughout and the phase-boundary `SIM_GEN_SCORE` prints never
  decreased.

Execution order is mechanism-enforced: P3 cannot precede P1 (closed shutter blocks
the doorway — proven by smoke's 3 N contact probe), and P4 closing over an empty bay
leaves the brick outside forever (the doorway is the only entrance). The scored
subgoals are reachable only as open → enter → dock → close.

## Embodiment argument (single Franka + parallel jaw, recorded not simulated)

- Base pose ≈ (0.0, −0.50) on the floor facing +y; garage (±4 cm around (0, 0.05))
  and the brick's apron spawn are within ~0.70 m reach; the whole task happens
  between floor level and ~15 cm height.
- Shutter: the 36 mm face knob is a purpose-built pinch target for the 80 mm jaw
  stroke, always proud of the facade; slide forces 1–6 N over ~112 mm of straight
  travel, and the steering torque (≤ 0.25 N·m) is what a rigid grasp provides for
  free.
- Brick: the 60 mm-wide body takes a top or side pinch well inside the jaw stroke;
  the push phase is a floor-level guided slide — the gripper only needs to reach
  ~20 mm past the facade plane through the 95×75 mm doorway before the brick's own
  momentum-free contact guidance (doorway jambs + floor) takes over, or
  equivalently a fingertip push on the brick's back face from outside.
- Forces: 2–6 N slides and a ≤ 4 N push, far inside Franka limits; tolerances the
  gripper must actually meet are the 95 mm doorway vs 60 mm body (17 mm/side) and
  the 12 mm shutter-close window over a 130 mm slab — low-precision by design; the
  mm-scale rubric bands are enforced by the mechanism (plate stop, rails), not by
  the hand.

## Checks (smoke.py — 16/16 on forge, recorded to frames.npz)

1. settle/no-NaN: shutter read back closed, brick outside, both settled.
2. SEED strategy: shutter opened + brick perfectly docked, then stop → dock latches
   but NOT success, score ≤ 0.705.
3. randomization readback: garage xy + yaw vary over 8 seeds.
4. randomization readback: brick dock-frame xy + relative yaw vary.
5. null policy 240 steps → score ≤ 0.02, no success.
6. blocked entry CONTACT probe: 3 N floor push at the CLOSED shutter for 240 steps —
   non-vacuous (brick advanced > 10 mm to the wall) yet the tip never crosses the
   facade, entry latch stays 0, shutter stays closed, score ≤ 0.02.
7. door-open-only → 0.18 ≤ score ≤ 0.22, no success.
8. near-miss transit: door open, brick in the doorway short of the seat band →
   0.40 ≤ score ≤ 0.62, dock latch 0, no success.
9. sideways parking: brick inside but yawed 90° (prong axis across the bay),
   shutter closed → not docked, no success.
10. backwards dock: brick inside facing outward, shutter closed → not docked.
11. roof percher: brick on the garage roof → entry latch 0, score ≤ 0.02.
12. latched door credit survives re-closing the shutter.
13. entry progress is monotone in depth (two-depth comparison).
14. latched entry credit survives pulling the brick back out to the apron.
15. rejection audit: success() never True anywhere in the battery.
16. final no-NaN.

## Verification (forge server, RTX 4090)

- `solve --seed 0`: SIM_GEN_SCORE 0.0000 → 0.2000 → 0.2000 → 0.7000 → 1.0000 →
  1.0000, `SIM_GEN_SOLVE: SUCCESS` (rc=0).
- `solve --seed 1`: same trace, rest tip_y +0.1167, `SIM_GEN_SOLVE: SUCCESS` (rc=0).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 16/16` (rc=0), frames.npz saved.
