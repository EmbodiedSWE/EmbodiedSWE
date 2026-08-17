# close_laptop_lid_i152 — BallastLidChest

**Scene name:** `ballast_lid_chest` · **Env:** `simgen.ballast_lid_chest` (robot="null")

## Seed provenance

Seed task: `rlbench/close_laptop_lid` — "close the laptop lid": a laptop screen
stands raised on its hinge; the robot pushes it down through its arc until it lies
shut. One pushing contact on a hinged panel, one unordered act.

## Strategic difference

The seed's entire plan — push the hinged panel shut — is **physically voided
here**, not re-parameterized. The chest lid carries a dense counterweight bar
*behind* its hinge that biases it open with ~1.4 N·m to spare: a lid pushed
closed and released swings right back up to its 46° stop (smoke check 4
constructs exactly the seed's end state and watches the physics reject it).
Closing is instead achieved by **ballast**: the lid's top face carries a shallow
walled tray near its front edge, and each 0.40 kg steel block laid in the tray
adds ~0.10 kg·m of closing moment. One block is never enough (≥ 0.35 N·m short —
smoke check 5); two always overcome the counterweight (≥ 0.27 N·m of net closing
torque at every hinge angle), so gravity swings the lid shut and holds it shut
(~0.54 N·m). The manipulation is therefore pick-and-place of removable masses
into a receptacle *on the moving panel* — the panel is never pushed — and the
final state is self-maintaining only because the ballast stays in the tray. The
full static torque budget is asserted in `BallastLidChestSceneCfg.__post_init__`.

Differs from every other tasks_v7 package: no other task closes a hinged panel by
loading ballast into a receptacle carried on the panel itself (close_microwave_i4
drops a gate into slots after an extraction; i33 opens a one-way flap; the vault /
pantry / bottle tasks are latch or pour mechanisms).

## Assets (fully procedural)

- **chest** — heavy DYNAMIC open-top box (360×300 mm footprint, 200 mm walls,
  12 mm thick, 25 kg). Dynamic, not kinematic: on this stack a joint anchored to
  a kinematic body0 stays world-fixed when the body is teleported at reset.
- **lid** — DYNAMIC compound, body origin ON the hinge axis (chest back top edge,
  axis = chest-local y, spawn-authored `UsdPhysics.RevoluteJoint`, limits
  [−46°, +2°], joint-pair collision explicitly ON — the closed rest *is*
  slab-on-rim contact). Forward of the hinge: 335×300×12 mm slab carrying the
  tray (interior 70×200 mm, 50 mm walls). Behind the hinge: link bar + 2.50 kg
  counterweight. All masses via per-child density (root `mass_props` on a custom
  spawner is silently ignored on this stack); readback-asserted in solve
  (3.275 kg vs ref 3.28, CoM x = −44 mm, behind the hinge).
- **blocks** — three 50×50×36 mm, 0.40 kg steel blocks scattered on the ground in
  front of the chest.

Randomization (readback-verified, smoke check 2): chest yaw ±20° + xy ±40 mm;
per-block scatter ±30 mm + free yaw.

## Rubric

- 0.25 `ballast1` — ever ≥ 1 block settled in the tray (latched, guarded by a
  6-step consecutive-still counter so a block skipping through cannot latch)
- 0.25 `ballast2` — ever ≥ 2 blocks settled in the tray (same guard)
- 0.20 `closing` — lid ever < 20° while ≥ 2 blocks ride in the tray (pushing the
  empty lid down latches nothing)
- non-success capped at 0.70; **1.0 iff success()**: opening ≤ 3°, ≥ 2 blocks in
  the tray, everything settled and finite — all judged live.

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset, 180-step settle: lid at its 46° stop, tray empty, score 0; lid
  mass/CoM readback asserts.
- **P1 (transport)** one root-state write puts block 0 in free space *inside* the
  tray airspace (lid-local x = 0.26 centred between the walls, ~17 mm above the
  tray floor, flush with the lid tilt, zero velocity — the release a gripper
  performs after lowering the block between the walls; releasing above the wall
  tops is unusable because a world-vertical fall in the 46°-tilted lid frame
  drifts forward by tan 46° per unit drop and carries the block over the front
  lip — observed). Gravity seats it against the hinge-side wall. Asserted: lid
  still at its stop (one block must NOT close it), score 0.25.
- **P2** same transport for block 1 → the torque balance tips, the lid swings
  shut on its hinge under gravity, slab lands on the chest rim (−0.16°), both
  blocks riding in the tray. Score 1.0.
- **P3** ≥ 3.3 simulated seconds hands-off; success persists → `SIM_GEN_SOLVE:
  SUCCESS`.

Teleports are transport-only: every rubric fact (blocks seated in the tray, lid
angle, settled) is produced by gravity and contact. The lid is never pushed,
teleported after reset, or touched. Verified on forge, seeds 0 and 1 (rc=0,
~19 s each).

## Execution order

NOT strictly ordered: any two of the three blocks, in either order. The third
block is never needed. Declared order-free; the smoke battery accordingly
contains no ordering trap.

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: at the world origin facing the chest, whose front face
stands ~0.26 m away (chest centre nominal (0.44, 0), front toward the robot).
Each act is a standard 50 mm-wide block pick from the ground (top grasp, jaw
travel > 50 mm) followed by a place into the tray: the tray mouth is open to the
sky at the lid's 46° stop, world height of the tray centre ≈ 0.40 m at reach
≈ 0.45 m — comfortably inside the Franka envelope. The 200 mm-wide tray accepts
the 50 mm block with 75 mm clearance per side for the fingers; the solver lowers
the block between the walls and opens the gripper ~17 mm above the tray floor —
exactly the release the solve certifies. After the second place the robot only
retracts; the lid closes itself away from the arm (hinge at the far side). No
step needs a second arm, regrasping in flight, or pushing.

## Files

- `scene.py` — cfg (+ torque-budget/clearance asserts), spawners (chest, lid +
  hinge), scene (rubric, latches in `post_step`), `register_env`.
- `solve.py` — phased teleport solution; watchdog + hard exit.
- `smoke.py` — 8-check rejection battery (settle, randomization readback,
  null-policy, SEED-strategy rejection, one-block near-miss, wrong-place cavity,
  ajar near-miss at the 3° gate, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 19.3 s)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 19.3 s)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 8/8`
