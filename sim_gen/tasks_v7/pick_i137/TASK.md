# pick_i137 — Gravity Vault (unlock the interlock, deliver by gravity)

## Seed provenance

- **Seed**: `mujoco_playground/pick` —
  `sim_gen/RoboVerse/roboverse_pack/tasks/mujoco_playground/pick.py`.
- The seed is "bring the box to the target": a Franka grasps a 4 cm cube and carries it
  through free space to a floating target pose. The target is an arbitrary *reachable*
  point; the entire plan is one grasp + one guided carry + one place, and the reward
  shapes gripper→box then box→target directly.

## What changed and WHY it is strategically different

The kept DNA: one small object must end up at a designated target region, positions
randomized per episode. Everything about *how* is inverted:

1. **The target region is SEALED.** The goal is the interior of a closed, roofed
   chamber. No pose the arm can reach places, pushes, or drops the ball in directly —
   the only opening is a funnel mouth 0.42 m up, at the top of a roofed feed tower
   (smoke proves the seal: a ball pressed onto the roof at 2× weight stays out). The
   seed's carry-and-place primitive, aimed at the target, scores zero.
2. **Delivery is by GRAVITY, not by the hand.** The last 0.4 m of transport is done by
   the environment: release over the mouth, and the funnel + tower channel do the
   guidance. The manipulation skill is choosing *where to let go*, not servoing to the
   goal pose (the seed shapes position+orientation error to zero; here the ball's final
   pose is not commanded at all, only judged).
3. **A REMOVABLE INTERLOCK guards the channel.** A free-riding gate tongue spans the
   tower through slots midway down; a ball dropped early just parks on it (smoke check).
   The load-bearing skill the seed never needs: a contact-rich captive-slide extraction
   — grip a knob and draw the tongue out through its slots, supporting its weight so it
   doesn't bind. The interlock is one-way: pushed inward, the knob jams on the tower
   wall (smoke check), so the knob direction must be read from the scene.
4. **Object identity matters.** A same-size blue decoy pays nothing (smoke check); the
   seed has a single object and no discrimination.

Also distinct from every other task read this session: `pick_up_cup_i76` is confined
planar maneuvering with no mechanism; `pick_cube_i125` is in-place reorientation by
edge pivots; `pick_and_lift_i16` is counterweight statics; here the core is a one-way
sliding interlock plus an environment-actuated (gravity) delivery into a sealed region.

## Scene (procedural, no external assets)

One kinematic compound (vault: chamber + tower + funnel, with honesty-asserted slit /
slot / side-gap widths all smaller than the ball), one dynamic compound (gate tongue +
knob, 0.12 kg), two 50 mm spheres (red target, blue decoy, 0.10 kg). Randomized per
episode: vault xy (±35 mm) and yaw (±30°), gate insertion side (mirrored) and depth,
ball slots swapped and jittered. All draws `torch.rand`; smoke verifies by readback.

## Rubric (latched partial credit, anchored in the solve trajectory)

- 0.30 `gate_out` — the tongue ever fully clear of the tower channel (latched)
- 0.25 `entered` — the red ball ever inside the tower channel (latched)
- 0.20 `chamber` — the red ball ever inside the chamber interior (latched)
- 1.0 iff `success()`: red ball in the chamber, settled, finite. Non-success capped at
  0.75 (smoke drives the cap and checks it exactly). Null policy ~0.

## Teleport solution (solve.py) — phases

- **P0** settle + layout/mass readback (gate spans, balls outside, score ~0).
- **P1** *(force, no teleport)* velocity-servo pull at the gate CoM along the knob
  direction (~0.10 m/s, 5 N cap, 0.9·m·g vertical support, gain-escalating stall
  watch) against real slot contact until the scene's `pin_clear()` latches → 0.30.
  Then the free gate is parked on the floor (transport only).
- **P2** *(transport + release)* the red ball is teleported to a hover 3 cm above the
  funnel mouth — the pose an arm reaches by carrying a grasped ball — and released
  with zero velocity. Gravity + funnel + tower deliver it: `entered` (0.55) then
  `chamber` (0.75) latch during the fall.
- **P3** settle → live success → `SIM_GEN_SCORE 1.0000`.
- **P4** hands-off ≥ 3.3 simulated seconds → `SIM_GEN_SOLVE: SUCCESS`.

No teleport crosses a barrier: with the gate in place the same release parks on the
gate (smoke check 5), so the P2 transport only shortcuts free-space carrying.

## Embodiment argument (Franka, base at world origin; vault at (0.45, 0) ± jitter)

- **Gate knob**: a 32 × 22 mm tab standing 50 mm proud of the tongue at z ≈ 0.26–0.31 —
  a standard Franka two-finger pinch across the 32 mm face; ≥ 45 mm asserted clearance
  between knob and tower wall for the fingers. The pull is a ~0.15 m horizontal
  straight-line drag at ~1 N — trivial for the arm from a side approach.
- **Red ball**: 50 mm sphere on open floor, within the 80 mm gripper span. The release
  is a hover at z ≈ 0.45 over a 170 mm mouth: ±60 mm of lateral tolerance, no
  orientation requirement.
- All interaction points lie within 0.30–0.65 m of the base at z ≤ 0.45 — inside the
  Franka workspace with margin.

## Execution order

Declared: NO required order (describe() says so). A gate-first plan is demonstrated;
ball-first merely parks the ball on the gate (no chamber credit until the gate is
pulled — and pulling it then feeds the ball in, which is a legitimate alternative).
The one-way interlock does force *pull-by-the-knob* rather than push-through.

## Checks

- solve: `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1 and 2 (forge; seed 2 draws the
  mirrored gate side), scores non-decreasing 0.00 → 0.30 → 0.55 → 0.75 → 1.00.
- smoke: `SIM_GEN_SMOKE: ALL PASS 11/11` (forge) — settle/no-NaN, randomization
  readback, null policy, sealed-roof press, gate-blocks drop (+ wedge-drop guard),
  latched credit, one-way interlock jam, blue-decoy rejection, settle gate + exact
  0.75 cap, rejection audit (success never True in the battery), frames.npz video.
