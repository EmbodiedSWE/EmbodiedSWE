# pick_cube_v1_i170 — Weight Airlock

## Seed provenance

Seed: `maniskill/pick_cube_v1` (RoboVerse
`roboverse_pack/tasks/maniskill/pick_cube_v1.py`): a 4 cm red cube on a bare
table must be grasped and lifted to a floating goal point (DetectedChecker,
2.5 cm radius). One object, one plan: grasp, lift, hold at a point in the air.

## Strategic difference

The red cube is kept as the *cargo*, but the seed's single grasp-and-lift plan
is deliberately defeated and replaced by a three-stage mechanism puzzle:

- **The seed's plan fails here.** The goal region is *inside a roofed vault*.
  Lifting the red cube and dropping it from above bounces off the roof
  (smoke check `seed-plan-drop`). The only way in is *through the front
  doorway*, sliding on the table — a push, not a lift.
- **The doorway is gated.** A spring-loaded sliding gate rests closed. It is
  driven open by a weight-interlock: a pressure plate on a pedestal *outside*
  the vault must be depressed. Only the heavy blue cube (0.80 kg) exceeds the
  spring preload margin (~0.19 kg-equivalent); the red cargo (0.06 kg) and the
  look-alike gray decoy (0.08 kg) physically cannot depress it — wrong-object
  rejection is contact dynamics, not rubric fiat.
- **Physically forced execution order** (the seed has no ordering at all):
  1. weigh the plate with the blue cube → interlock slides the gate open;
  2. push the red cargo through the open doorway into the vault;
  3. remove the weight → the spring returns the plate and the interlock
     drives the gate *shut*, locking the cargo in. Success additionally
     requires the closed/locked end state, so leaving the weight on
     (out-of-order end state) is not success (checks `weight-left-on`,
     `revocation`).
- Versus the sibling tasks read this session: `pick_cube_i125` is a die
  *orientation* puzzle (no orientation goal here — containment + mechanism
  state); `press_switch_i161` is continuous setpoint regulation (here the
  mechanism states are discrete open/closed driven by a weight interlock);
  the `pen_holder` exemplar is multi-item container filling (here a single
  cargo through an interlocked airlock with a lock-in end state).

## Scene (env-local, table top z0 = 0.40)

- **Vault**: roofed box centred (0, 0.16); interior x∈±0.10, y∈[0.06, 0.26],
  height 0.12; a full-height doorway x∈±0.06 in the front wall.
- **Gate**: 13.5×1.5×12 cm slab on a D6 prismatic joint (transX ∈ [0, 0.14],
  anchored to the back wall), gravity-neutral horizontal slide, lifted 4 mm
  off the table; ±2.0 N interlock force toward open iff the plate is
  depressed, else toward closed.
- **Pressure plate**: 9×9×1.2 cm plate on a D6 vertical joint
  (transZ ∈ [−0.02, 0], anchored to a pedestal at (0.22, −0.07), top
  z0+0.05), constant 2.5 N spring up. Empty margin ≈ 1.9 N.
- **Cubes** (masses via authored MassAPI, asserted by readback):
  red cargo 5 cm / 0.06 kg; blue weight 6 cm / 0.80 kg (7.85 N ≫ margin);
  gray decoy 6 cm / 0.08 kg (0.78 N < margin).
- **Randomization** (verified by readback in smoke): the three cubes are
  dealt a random permutation of three staging slots, ±4 cm xy jitter,
  ±180° yaw.

## Rubric

Latched partial credit (running-max semantics, asserted non-decreasing in
solve): `latch_open` (plate depressed & gate at open stop) → 0.25;
`latch_inside` (red fully inside & slow) → +0.35; success — red inside AND
gate closed AND plate up AND settled — → 1.0.

## Teleport solution (transport only; all load-bearing interaction is contact)

1. Teleport the blue weight to hover 8 mm above the plate and release —
   gravity + contact depress the plate; the interlock slides the gate open.
2. Teleport the red cargo to free table space in front of the doorway, then
   push it through with a capped-force velocity servo on the scene's `push_f`
   buffer (K·dt/m = 0.42 < 1 against the one-substep wrench delay; stalls
   escalate gain, never the 2 N cap).
3. Teleport the blue weight off to a parking spot — the spring raises the
   plate and the interlock closes the gate over the cargo.
   Then ≥3.5 simulated seconds hands-off (buffers asserted zero) before the
   verdict.

## Franka embodiment argument (per object)

Base at ~(0.0, −0.45, z0), facing +y:

- **Blue weight (6 cm, 0.80 kg)**: fits the ~8 cm Franka jaw; 0.8 kg is well
  inside payload. Staging slots (reach ~0.5 m) → plate top at (0.22, −0.07,
  z0+0.06), reach ~0.48 m: a standard grasp, carry, set-down. Removal in
  phase 3 is the same grasp in reverse.
- **Red cargo (5 cm, 0.06 kg)**: graspable for the *staging* carry across
  free table; the through-doorway transit is a fingertip/knuckle *push* of a
   5 cm cube through a 12 cm doorway (~3.5 cm lateral slack) at push depth
  ~0.62 m — within reach with the arm extended low over the table. The 2 N
  capped push force is fingertip-scale.
- **Gray decoy**: never needs to be touched; it only needs to *exist* as the
  wrong answer.
- No simultaneous two-hand requirement anywhere: the weight holds the gate
  open statically while the arm goes and pushes the cargo.

## Execution order (REQUIRED, physically enforced)

weigh → push through → unweigh. The gate physically blocks the doorway when
closed (check `closed-gate-push`), and success requires the gate re-closed
with the plate up, so the weight must come off *after* the cargo is in.

## Checks (smoke.py, 13)

1. `clean-reset` — slot readback <5 mm, gate closed, plate up, score <0.05.
2. `mass-readback` — authored masses via `get_masses()`; red/decoy < 0.7×
   margin, blue > 3× margin.
3. `randomization` — seeds 11–18: ≥2 distinct slots for red and blue, ≥5
   distinct jitters and yaws, physical readback.
4. `null-policy` — 2 s hands-off: score <0.05, gate closed.
5. `seed-plan-drop` — the seed's lift-and-drop from above: roof rejects it.
6. `closed-gate-push` — solve-scale push at the closed gate: gate holds,
   cargo stopped, no inside latch.
7. `decoy-cannot-weigh` — decoy on the plate: plate stays up, gate shut.
8. `wrong-cargo` — full interlock but the *decoy* pushed inside: no success,
   score stays in the open-latch band.
9. `straddle-close` — weight removed while the cargo straddles the doorway:
   shoved out or stalled, never counted inside.
10. `weight-left-on` — cargo inside but weight still on the plate (gate
    open): no success, score in the inside-latch band.
11. `exactness` — completing phase 3 → success, |score−1| <1e-3, stable.
12. `revocation` — re-weighing after success reopens the gate: success
    revoked while open.
13. `frames` — ≥20 RGB frames recorded to frames.npz.

Run (forge): `python -u -m simgen_tasks.pick_cube_v1_i170.solve --headless`
and `python -u -m simgen_tasks.pick_cube_v1_i170.smoke --headless`.
