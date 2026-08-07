# plug_charger_in_power_supply_i21 — keyhole unplug and stow

Scene `keyhole_unplug`, env id `simgen.keyhole_unplug`.

## Seed provenance

Seed: `rlbench/plug_charger_in_power_supply`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/plug_charger_in_power_supply.py`).
The seed is a terminal INSERTION task: a free charger lies on the table, a wall-mounted
power supply is fixed, and the (trajectory-file) goal is to pick the charger up, align
its prongs, and push it INTO the socket. The episode is about reaching one inserted
pose; nothing else in the scene is judged.

## What changed and why it is strategically different

The plan is inverted and a mechanism is added — the goal state of the seed is the
**reset state** here, and insertion appears nowhere in this task's goal:

| | seed | this task |
|---|---|---|
| judged plug at reset | free on the table | **captive inside the supply** |
| goal interaction | insert (align prongs, push in) | **unlock–extract–relocate** (slide to the keyhole opening, lift out, lay in a tray) |
| mechanism | plain socket holes | **keyhole latch**: 24 mm foot captive under a 20 mm lid slot; the only exit is the 31 mm square opening at the slot's far end |
| judged scope | one pose | settled outcome + **preservation constraints** (black decoy plug still seated; free-standing strip within 50 mm / 20° / upright of its reset pose) |
| naive strategy | push harder | a straight upward yank physically hoists the whole 14.7 N strip (the foot bears on the lid underside) and is rejected by the strip-drift constraint; smoke check 6 pulls with 6 N for 2 s and proves the plug stays captive |

A solver therefore needs a different plan (read the slide direction off the scene from
the wide opening, slide-unlock, lift out, transport, stow, disturb nothing) and a
different code structure (slide-progress latch gated on being inside the channel, an
extraction latch, a stow predicate, and preservation predicates on two untouched
bodies) — not "align a peg with a hole and push".

Also different from every other task read this session:
- `peg_insertion_side_i1` (ram-rod ejection): uses a free TOOL rammed through a
  side-open sleeve to eject a payload — here there is no tool and no ram; the judged
  object itself is unlocked by a geometric keyhole and extracted vertically.
- `peg_insertion_side_i2` (create-passage-then-fasten): builds an aligned passage by
  seating a bridge, then threads a pin IN (terminal insertion with an ordering latch) —
  here nothing is inserted at all, and the ordering (slide before lift) is enforced by
  rigid geometry, not by a rubric latch.
- `robobench/suites/packing/scenes/pen_holder.py` (repeated tip-up container
  insertion): put N pens into a holder — here one object comes OUT of a mechanism and
  is laid flat into an open tray, with decoy/strip preservation constraints.

## Scene

Fully procedural (axis-aligned `UsdGeom.Cube` parts, explicit 1 mm contact offsets —
the ~2 cm default would eat the 3 mm working clearances):

- **strip** — DYNAMIC, free-standing, 220×110×30 mm, 1.5 kg. Two identical keyhole
  sockets at local x = ±45 mm: an internal channel (30 mm wide, 62 mm long, 16 mm
  tall) under a 6 mm lid whose opening is a 20 mm slot widening into a 31 mm square
  opening at the channel's +y end. Nothing anchors the strip.
- **plugs** — DYNAMIC, blue + black, identical geometry: 24 mm × 10 mm foot (captive:
  24 > 20), 14 mm neck (rides the slot), 34 mm × 45 mm head (grasp knob; 34 > 31, so
  the head can enter nothing — no burying loopholes). Which socket holds the blue
  plug is randomized.
- **tray** — KINEMATIC open square tray, 110 mm interior, 30 mm walls.

Randomization (verified by readback in smoke): strip xy ±30 mm + free yaw (the slide
direction must be read from the scene), tray xy ±40 mm + free yaw with keep-out from
the strip, blue/black socket swap. Geometry honesty is asserted in
`KeyholeUnplugSceneCfg.__post_init__`.

`success()` iff, with strip AND both plugs settled (< 0.05 m/s): blue plug inside the
tray interior (tray frame, per-axis 50 mm, height band 4–45 mm — rim perchers and
floor-droppers fail), black plug still at its lock seat (strip frame), strip within
50 mm / 20° / upright of its reset pose. `score()` = 0.25·latched slide progress
(gated on the foot being inside its own channel) + 0.30·extracted + 0.30·stowed,
capped at 0.85; exactly 1.0 iff success. Doing nothing — the seed's whole strategy —
scores ~0.

## Solution outline (solve.py, the legitimacy certificate)

Transport-only teleportation: exactly ONE pose write, performed when the plug is
already a free body in mid-air. All load-bearing interaction is contact dynamics via
`set_external_force_and_torque`:

- **P1 unlock slide** — velocity-regulated push (≤ 2 N) along the strip-local slot
  direction + lateral PD centring + upright/yaw-steadying torque; the foot drags the
  real channel floor guided by the real walls and slot rails, from the lock seat to
  under the wide opening. The strip is never held (~1–2 N push vs its ~7 N friction
  footprint; `strip_ok` verified in the readback).
- **P2 lift-out** — velocity-regulated vertical pull hard-capped at plug weight
  + 2.5 N (an order of magnitude below the 14.7 N strip weight: a locked plug cannot
  be freed or the strip hoisted this way), foot rising through the wide opening under
  contact guidance until clear of the lid.
- **P3 stow** — the single transport teleport to a lying hover 9 mm above the tray
  floor, then release: a real gravity set-down and settle.
- **P4 persistence** — ≥ 3.4 simulated seconds hands-off; SUCCESS only if `success()`
  holds throughout and the phase-boundary `SIM_GEN_SCORE` prints never decreased.

Execution order is **mechanism-enforced**: the lift cannot precede the slide (the foot
bears on the lid anywhere except under the opening — demonstrated by the capped-force
lift and by smoke's 6 N yank probe), and the stow cannot precede the lift. No rubric
ordering latch is needed; the scored subgoals are reached in the only physically
possible order: slide → extract → stow.

## Embodiment argument (single Franka + parallel jaw, recorded not simulated)

- Base pose ≈ (0.0, −0.55) on the floor, facing the workspace; strip (±7 cm around
  (−0.04, 0.10)) and tray (±4 cm around (0.10, −0.20)) are both within ~0.75 m reach,
  and the whole task happens between floor level and ~15 cm height.
- Grasp: the 34 mm square head is a purpose-built knob for the 80 mm Franka jaw
  stroke, always riding ≥ 6 mm clear above the lid; a top or side pinch works at any
  strip yaw.
- The solve controller is a floating-hand proxy: ~1–2 N lateral slide over 34 mm of
  travel, ≤ 3.3 N lift over ~25 mm, forces and moments far inside Franka limits; the
  steadying torque (≤ 0.06 N·m) is what a rigid grasp provides for free.
- Transport: a ~0.35 m free-space carry at ≤ 15 cm height, then lowering the plug
  ~1 cm into a 110 mm tray with a 50 mm per-axis tolerance — a low-precision place.
- Clearances are gripper-realistic: 6 mm neck-to-slot, 6 mm foot-to-channel, 7 mm
  foot-to-opening; the 3 mm-scale rubric tolerances apply only to states the
  mechanism itself enforces.

## Checks (smoke.py — 17/17 on forge, recorded to frames.npz)

1. settle/no-NaN: both plugs verified at their lock seats by strip-frame readback.
2. SEED strategy: the seed's goal state IS this reset state → score ≤ 0.02, no success.
3. randomization readback: strip xy+yaw, tray xy+yaw all vary over 8 seeds.
4. randomization readback: socket assignment takes both values.
5. null policy 240 steps → score ~0, no success.
6. yank probe (contact): 2 s of 6 N straight-up pull on the locked plug — still
   captive, extraction never latches, strip not hoisted/dragged.
7. near-miss slide: at the opening but not lifted → 0.15 ≤ score ≤ 0.30, no success.
8. near-miss stow: on the floor beside the tray → not in tray, ≤ 0.60, no success.
9. rim percher: on the tray wall top → above the interior band, no success.
10. left on the strip: standing on the lid, below the extraction latch height → ~0.
11. wrong object: BLACK plug in the tray → ≤ 0.02, no success.
12. constraint: strip dragged 80 mm with blue properly stowed → no success, ≤ 0.85.
13. constraint: black unseated with blue properly stowed → no success, ≤ 0.85.
14. latched credit survives sliding the plug back to the lock seat.
15. deeper slide latches strictly more credit.
16. rejection audit: success() never True anywhere in the battery.
17. final no-NaN.

## Verification (forge server, RTX 4090)

- `solve --seed 0`: SIM_GEN_SCORE 0.0000 → 0.2428 → 0.5500 → 1.0000 → 1.0000,
  `SIM_GEN_SOLVE: SUCCESS` (rc=0).
- `solve --seed 1`: SIM_GEN_SCORE 0.0000 → 0.2500 → 0.5500 → 1.0000 → 1.0000,
  `SIM_GEN_SOLVE: SUCCESS` (rc=0).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 17/17` (rc=0), frames (172, 600, 960, 3) saved.
