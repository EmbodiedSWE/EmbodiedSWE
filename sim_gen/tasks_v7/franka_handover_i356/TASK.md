# franka_handover_i356 — `relay_cascade`

Relay the tile-named cube through a sealed two-stage counterweighted see-saw
cascade into the tower's bottom bin.

## Seed provenance

- **Seed task:** `bimanual/franka_handover`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/bimanual/franka_handover.py`): two Franka
  arms at y = ±0.45 replay a mirrored, coordinated joint-space reach toward a shared
  5 cm cube; one arm grasps it and passes it to the other **directly through free
  space**. The entire skill is an unobstructed arm-to-arm transfer.

## What changed, and why it is strategically different

| | seed | this task |
|---|---|---|
| transfer path | free space between two grippers | a **sealed tower** — every gap/slot in the shell is thinner than the cube |
| transfer agent | the second arm | the tower's own **bucket brigade**: two counterweighted see-saw trays |
| skill | mirrored reach + grip handoff | **perception** (colour tile names 1 of 2 cubes), **drop** through the roof intake, **two lever presses** (momentary fingertip torque, gravity-return), distractor discipline |
| ordering | none (one motion) | **physically forced**: intake → L1 → L2 (the intake feeds only tray 1; L2 early tips an empty tray) |

The seed's plan — carry the object to the destination — is not merely unrewarded, it
is **physically impossible**: the delivery bin is the floor of a fully enclosed
cabinet (walls, roof, internal seal wall, 6 mm channel gaps, lever slots 36–38 mm
< the 50 mm cube). A cube carried to the tower just rests on the roof plate (smoke
check 5 proves this with a settled construct). The only opening is the roof intake
mouth, which feeds tray 1 only; each subsequent stage requires actuating that tray's
lever rod — a protruding pin through a narrow wall slot — with the sanctioned
plant-clamped torque (0.5 N·m ≈ 7 N at the rod). Stage credit is **aboard-gated and
target-only**: pressing levers with nothing aboard, or relaying the decoy, earns ~0
(smoke checks 6–8). This differs from the sibling tasks examined: i115 rotates a
carousel drum, i124 weighs, i339 clamps; here the mechanism is a *chained* pair of
gravity-returning see-saws whose stage order is topological.

## Teleport solution (solve.py) — phases

1. **Settle + perception** — read back the housing pose, WHICH colour the pedestal
   tile names, that cube's slot, and both tray rest angles (−8°/+8°, joint-stop
   readback). `SIM_GEN_SCORE ~0.00`.
2. **DROP** — transport-only teleport of the target cube from its apron slot to free
   air 3 cm above the intake mouth (open sky above both endpoints), release; gravity
   drops it through the mouth onto tray 1; the deck and retainer cradle it at the
   rest tilt by contact. `SIM_GEN_SCORE 0.20`.
3. **PRESS L1** — bang-bang `lever_drive[:, 0] = +0.5 N·m` drives tray 1 to its +30°
   stop (peak-angle asserted); the cube slides off the deck — pure contact dynamics
   (tan 30° > μ_pair + 0.10) — and lands on tray 2. Release: the ballast
   gravity-returns the tray (asserted). `SIM_GEN_SCORE 0.45`.
4. **PRESS L2** — same on `lever_drive[:, 1]` to the −35° stop; the cube discharges
   over the partition into the sealed bin (ballistic flight passes under the seal
   wall — asserted on paper in `__post_init__`). `SIM_GEN_SCORE 0.75 → 1.00` once
   `success()` holds.
5. **VERIFY** — hands off ≥ 3.3 more simulated seconds; success must persist →
   `SIM_GEN_SOLVE: SUCCESS`. Passes on seeds 0 and 1.

## Embodiment argument (single Franka arm, parallel-jaw gripper)

Base pose: in front of the tower (housing-local ≈ (+0.55 m, 0)); the apron, intake
mouth and both lever rods all face this side.

- **Cubes (50 mm, 0.10 kg):** within the Franka's 80 mm jaw span with 15 mm margin
  per face; top-down grasp in both apron slots (slot tops at 0.125 m, open sky
  above); the drop point over the mouth is at 0.48 m height and r ≈ 0.35 m from the
  base — inside a table-mounted Franka's envelope.
- **Intake mouth (100 mm along y × 120 mm interior x vs the 50 mm cube):** ±25 mm
  placement slack for the release — well inside arm repeatability.
- **Lever rods (⌀8 mm, protruding 45 mm proud of the front wall at heights 0.35 m
  and 0.22 m):** pressable side-on by a closed gripper; the plant clamp (0.5 N·m ≈
  7 N at the rod) is fingertip-scale; the press is bang-bang against a joint stop —
  no fine force control needed, and released rods gravity-return on their own.
- **Colour tile (70 mm, on the apron pedestal):** perception only — never touched.

## Execution order

**Required, and physically enforced:** drop, then L1, then L2. The intake feeds only
tray 1; L2 with an empty tray 2 moves nothing (smoke check 6: real-torque L2-first,
actuation verified, leaves the cube on tray 1 and pays nothing new); the sealed
shell forbids every bypass (smoke checks 5 and 9).

## Rubric (scene.py)

- 0.20 — target cube ever aboard tray 1 (latched)
- 0.25 — target cube ever aboard tray 2 (latched; reachable only via a tray-1 discharge)
- 0.30 — target cube ever in the bin box (latched)
- capped at 0.75; exactly 1.0 iff live `success()`: target settled in the bin ∧ no
  decoy in the bin ∧ tray FD rates still ∧ everything settled and finite.

## Checks

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on seeds 0 and 1 (forge), monotone
  `SIM_GEN_SCORE` prints at phase boundaries (0.00 → 0.20 → 0.45 → 1.00), ≥ 3.3 s
  hands-off persistence.
- `smoke.py`: 13 rejection-only checks — settle/no-NaN (tray rest-stop readback),
  randomization A/B (yaw/xy spans; target colour, slot permutation, pedestal-tile
  match and cube-slot READBACK), null policy, seed-strategy carry (rests on the
  roof), order interlock (real-torque L2-first, actuation verified), empty presses,
  wrong cube (full decoy relay refuses), containment near-miss (settled behind the
  partition), latched-credit survival, rejection audit, final no-NaN, frames.npz
  video. `SIM_GEN_SMOKE: ALL PASS 13/13` on the forge.
