# lift_peg_upright_i116 — HoodPropScene (`simgen.hood_prop`)

Prop a chest's falling lid half-open by standing the long red peg in a socket under
it — the peg is a load-bearing tool, not the judged outcome.

## Provenance

- **Seed task**: `roboverse_pack/tasks/maniskill/lift_peg_upright.py`
  (maniskill/lift_peg_upright): grasp a 24 cm peg lying flat, pitch it ~90°, stand it
  on the table. Success is a pose readout on the manipulated peg itself
  (|peg_z − 0.12| < 0.05 AND peg axis within ~10° of world-up) — one terminal
  reorientation, episode over the instant the peg stands.
- **This task**: fully procedural robobench scene (no external assets), scene
  `hood_prop`, env `simgen.hood_prop`, robot `"null"`.

## Strategic difference

**Vs the seed**: an upright peg is never the goal here — it is a PROP inside a
three-stage, mechanically ordered mechanism, and the judged outcome is the resting
angle of a *different* body (the lid) that the peg holds up. The seed's entire
strategy executed perfectly — red peg stood upright free on the ground, correct
height band, settled — is this task's null outcome: smoke check 5 constructs exactly
that end state and asserts score ~0, no success.

**Vs the corpus tasks read this session**:
- `pen_holder` (packing): repeated tip-up insertion into a container — here nothing
  is contained; the single peg insertion is shallow (14 mm collar) and worthless
  until the lid is *lowered onto* the peg.
- `beam_scale` (pull_cube_i20): mass accumulation tips a beam the robot never
  touches, no ordering. Here the judged body (lid) is DIRECTLY driven through a
  bistable contact hinge, and geometry forces a strict order.
- `hasp_pin_link` (peg_insertion_side_i2): align-then-thread fastening of a linkage —
  nothing is propped, nothing falls. Here the goal band is gravitationally unstable
  and exists only through a support contact.
- `latch_vault` (screw_nail_i59): unlock-uncover-retrieve disassembly. Here the chain
  is *constructive* (open → emplace prop → lower onto prop) and ends with a body at
  rest on the emplaced part.

## The mechanism (why the order is forced)

- The lid (0.5 kg, 30 cm) rides a **contact hinge**: a capsule axle captive in
  capped, slick (μ=0.02) tower pockets — no USD joint, so the chest can be
  teleport-randomized at reset without the kinematic-anchor trap. It can swing
  0…~106° but cannot be lifted off (cap 3 mm above the axle).
- **Bistable**: below vertical the lid falls SHUT (CoM 15.2 cm ahead of the hinge);
  past vertical it rests on a back-stop bar at ~106° (CoM behind the hinge) — the
  "temporary hold" a single gripper needs while it fetches the peg.
- The socket collar (42 mm cavity, 14 mm tall, on the sill 15 cm from the hinge) is
  **covered by the closed lid** (collar top 26 mm < closed underside 52 mm sweep…
  asserted in cfg), so the socket is unreachable until the lid is parked open.
- Long red peg (24 cm) props the lid at **~57°** — inside the success band
  [40°, 72°] with ≥8° margin each side. Short blue peg (12 cm) is a decoy: props at
  **~31°**, below band_min. Back-stop rest ~106°, ≥20° above band_max. All three
  angles are solved by bisection in `scene.py` and asserted against the band in
  `__post_init__`.

## Required execution order (declared)

1. **Swing the lid past vertical** onto the back-stop (skip if the episode starts
   parked open — 35% do; readable from the scene and from `describe()`).
2. **Stand the red peg** upright with its foot in the green socket collar.
3. **Pull the lid off the stop and lower it** until it rests on the peg inside
   [40°, 72°]; hands off.

Step 2 cannot precede step 1 (closed lid covers the socket). Step 3 cannot precede
step 2 (the band is gravitationally unstable — smoke proves a 55°-posed lid with no
peg falls shut, and a 100°-posed lid falls back onto the stop).

## Rubric

- `success()`: lid riding its hinge (origin within 2.5 cm of the hinge axis), at
  rest inside [band_min, band_max], long peg standing (≤15° tilt) with its foot
  inside the socket (16 mm xy tol), everything settled.
- `score()`: latched stage credit — 0.15 · (lid ever swung past 88° **and** the
  episode started closed; open starts get no free credit) + 0.25 · peg socketed
  + 0.20 · band occupied with peg in place, capped 0.60; **1.0 iff success**.
  Null policy ~0 in both start states.

## Randomization (readback-verified in smoke)

Chest xy jitter ±5 cm + free yaw ±180°; both pegs scattered lying flat on a jittered
ring (r ≈ 0.53–0.63 m, min 35° angular separation, free yaw); lid start state
closed / parked-open (p = 0.35).

## Solution outline (solve.py — teleports are TRANSPORT ONLY)

- **P1 open** (closed starts): external torque about the hinge axis —
  gravity feedforward m·g·com_x·cos θ + velocity servo on a *finite-difference*
  hinge rate (ang-vel readback is phantom under wrenches), K·dt/I ≈ 0.27, cut at
  96°; the lid falls onto the back-stop by gravity. Hinge-axis torque is invariant
  under the pod's rotation-since-reset wrench drag; a runtime progress probe guards
  the frame mode anyway.
- **P2 place**: the peg is teleported once — foot 3 mm above the collar, zero
  velocity (what a pick-and-carry delivers) — and DROPPED; it seats under gravity.
- **P3 lower**: closing servo (−0.8 → −0.2 rad/s profile); the feedforward carries
  most of the weight so the underside meets the peg at ~0.2 rad/s; contact detected
  from the FD rate; feedforward then RAMPED to zero over 0.75 s (quasi-static weight
  hand-off) and the wrench cleared. The propped rest is pure contact statics.
- ≥3.3 s hands-off persistence after success, `SIM_GEN_SCORE` at every boundary
  (non-decreasing, asserted).

**Verified on the forge**: seeds 0 (open start), 1 (open start), 2 (closed start)
all `SIM_GEN_SOLVE: SUCCESS` (27 s / 39 s / 40 s wall). Score traces
0 → 0.25 → 1.0 (open starts) and 0 → 0.15 → 0.40 → 1.0 (closed start). Lid rests at
55.4° vs the 57.0° geometric prediction (corner compliance), well inside the band.

## Embodiment argument (single Franka, parallel jaw)

Plausible base pose: in front of the chest, ~0.55 m from the hinge line.

- **Lid**: the 16 mm plate front edge is a natural jaw lip; the swing is a 30 cm
  radius arc — graspable edge throughout, and the back-stop holds the lid open
  hands-free, which is exactly why the task is single-arm feasible (no need to hold
  the lid while placing the peg). Lowering can be done grasping the front edge or
  supporting the underside with a fingertip.
- **Pegs**: 32 mm square section fits the 80 mm jaw with wide margin; lying flat on
  open floor at ~0.55 m ring radius — clear side-grasp approach; the socket has 5 mm
  per-side insertion play and the collar funnels the foot; peg top is below the
  parked lid's clearance (lid front edge at z ≈ 0.29 m when parked).
- **Torques**: lid weight ~0.75 N·m about the hinge = ~2.5 N at the front edge —
  trivially within Franka payload.

## Smoke battery (15/15 PASS on the forge, frames.npz recorded)

1. closed-start settle: finite, lid rests shut (−0.4°), score ~0, no success
2. randomization readback: chest xy/yaw + peg positions vary (12 seeds)
3. both lid start states occur (9 closed / 3 open in 12)
4. null policy, closed start: ~0
5. null policy, open start: pre-parked lid earns NO opened-credit, ~0
6. **seed-strategy end state**: red peg stood upright free on the ground → ~0
7. red peg upright ON TOP of the closed lid → no credit
8. near-miss xy: peg upright on the sill 9 cm from the socket → no credit
9. out-of-order: peg socketed but lid left on the back-stop → 0.25, no success
10. hinge_ok: lid posed at a band angle 8 cm OFF the hinge, peg socketed, zero vel →
    rejected by the hinge clause alone
11. **short-peg decoy**: blue peg genuinely props the lid — at 30.5°, below
    band_min → no success, no peg credit
12. band unstable downward: 55°-posed lid with no peg falls SHUT
13. band unstable upward: 100°-posed lid falls back onto the stop (106°)
14. rejection audit: success() never True anywhere in the battery
15. final no-NaN

## Files

- `scene.py` — cfg (tunables + honesty asserts), compound spawners (chest, lid),
  rubric, registration
- `solve.py` — teleport+wrench solution (`SIM_GEN_SOLVE: SUCCESS`, seeds 0/1/2)
- `smoke.py` — rejection battery (`SIM_GEN_SMOKE: ALL PASS 15/15`)
