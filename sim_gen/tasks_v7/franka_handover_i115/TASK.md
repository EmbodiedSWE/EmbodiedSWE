# franka_handover_i115 — `transfer_carousel`

Hand the tile-named parcel across a sealed counter through the station's rotating
pass-through carousel.

## Seed provenance

- **Seed task:** `bimanual/franka_handover`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/bimanual/franka_handover.py`): two Franka
  arms at y = ±0.45 replay a mirrored, coordinated joint-space reach toward a shared
  5 cm cube; one arm grasps and passes it to the other **directly through free
  space**. The entire skill is an unobstructed arm-to-arm transfer.

## What changed, and why it is strategically different

| | seed | this task |
|---|---|---|
| transfer path | free space between two grippers | a **sealed boundary** — no free-space path exists |
| transfer agent | the second arm | the station's **pass-through carousel** (rotating drum with one open-top bay) |
| skill | mirrored reach + grip handoff | **perception** (colour tile names 1 of 3 parcels), **place** into the bay on the open side, **mechanism actuation** (push the drum's pegs half a turn), distractor discipline |
| ordering | none (one motion) | **physically forced**: load, THEN rotate |

The seed's plan — carry the object to the destination — is not merely unrewarded
here, it is **physically impossible**: the delivery alcove is enclosed by an arc
wall, side pillars and a canopy roof; a parcel carried over the top just lands on
the roof (smoke check 5 proves this with a settled construct). The only way across
is the drum's bay, and the canopy clears the rotating drum by a 14 mm slot, so the
bay can only be loaded while it faces the open side (smoke check 6: the empty bay
really rotated into the alcove by drive torque cannot be loaded there). Rotation
credit is aboard-gated, so spinning the empty drum earns ~0 — the seed's "move
toward the other side" instinct pays nothing without the load step.

## Teleport solution (solve.py) — phases

1. **Settle + perception** — read back station pose, the pedestal tile's colour
   (target index), the parcel-slot permutation and the drum angle θ0. `SIM_GEN_SCORE
   ~0.00`.
2. **LOAD** — transport-only teleport of the target parcel from its apron slot to
   free air 5 cm above the open bay (open sky above both endpoints), release;
   gravity drops it in, the bay floor/walls seat it by contact. `SIM_GEN_SCORE
   ~0.35`.
3. **ROTATE** — cascaded velocity servo on the scene-sanctioned `drum_drive` torque
   (the fingertip-push stand-in, plant-clamped at 1.2 N·m; outer loop 1.5·err →
   ±1.2 rad/s, inner loop 0.8·(w_des − FD rate) → ±1.1 N·m, K·dt/I ≈ 0.42). The
   parcel **rides the bay** through the canopy slot into the alcove — pure contact
   dynamics. Drive cut only when slow and near centre (coast ≈ 0.01 rad).
   `SIM_GEN_SCORE 1.00` once `success()` holds.
4. **VERIFY** — hands off ≥ 3.3 more simulated seconds; success must persist →
   `SIM_GEN_SOLVE: SUCCESS`. Passes on seeds 0 and 1.

## Embodiment argument (single Franka arm, parallel-jaw gripper)

Base pose: on the open side of the counter, station-local ≈ (+0.55 m, 0), gripper
above the slab (slab top is 41 mm off the ground; a table-mounted Franka reaches the
whole open half, r ≤ 0.36 m from its base).

- **Parcels (55 mm cubes, 0.12 kg):** within the Franka gripper's 80 mm jaw span
  with 12 mm margin per face; top-down grasp in every slot (slots at r = 0.24, all
  in the open half, open sky above); lift-over into the bay needs ≤ 0.15 m of free
  vertical travel — available (nothing above the open half).
- **Bay (84 mm square interior vs 55 mm parcel):** ±14 mm placement slack — well
  inside arm repeatability; release height 5 cm keeps the drop centred.
- **Pegs (⌀22 mm, 55 mm tall, at r = 0.132):** at least one peg is always in the
  open half (pegs every 90°), graspable or pushable side-on by a closed gripper;
  the plant clamp (1.2 N·m ≙ ≤ 9 N tangential at the peg radius) is fingertip-scale.
  The ±20° sector tolerance is a 4.6 cm arc at the peg radius — coarse, repeated
  nudges suffice; the drum can be re-gripped every 90° of travel.
- **Colour tile (70 mm, on the 95 mm pedestal at the open corner):** perception
  only — never touched.

## Execution order

**Required, and physically enforced:** load first, rotate second. The 14 mm canopy
slot forbids loading in the alcove; the sealed perimeter forbids bypassing the
carousel. (Rotate-then-load is dead on arrival; smoke check 6 demonstrates it.)

## Rubric (scene.py)

- 0.35 — target parcel ever seated in the bay (latched)
- 0.40 — latched max rotation progress toward the alcove, **aboard-gated**
- capped at 0.75; exactly 1.0 iff live `success()`: target in bay ∧ bay within
  ±0.35 rad of alcove centre ∧ no distractor in bay ∧ drum FD-rate settled ∧
  parcels/station settled ∧ finite.

## Checks

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on seeds 0 and 1 (forge), monotone
  `SIM_GEN_SCORE` prints at phase boundaries, ≥ 3.3 s hands-off persistence.
- `smoke.py`: 13 rejection-only checks — settle/no-NaN, randomization A/B
  (yaw/xy/θ0 spans; target colour, slot permutation, pedestal-tile match and
  parcel-slot READBACK), null policy, seed-strategy carry (lands on the roof),
  load interlock (real-drive empty rotation + rejected drop), wrong parcel,
  angle near-miss (26° short), containment near-miss (on platter beside bay),
  latched-credit survival, rejection audit, final no-NaN, frames.npz video.
