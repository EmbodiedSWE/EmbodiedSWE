# libero_kitchen_scene8_turn_off_the_stove_i363 — `clutch_valve_stove`

Turn OFF a lit stove whose gas valve hides inside a sealed housing: the only
handle is a spring-less **pull-and-turn safety knob** (a dog clutch). The knob
free-spins uselessly at rest; you must PULL it up ~12 mm to engage its posts
under the valve rotor's vanes, wind the rotor clockwise from its lit angle
(68–88°) down to the OFF window (≤ 8°), and then LET GO — success requires the
knob re-seated at the bottom and everything calm, with the flame out.

## Provenance

Seed: `libero_90/libero_kitchen_scene8_turn_off_the_stove` (LIBERO kitchen
scene 8). The seed's success predicate is a direct knob-joint readback —
`stove_joint_pos[:, 0] < 0.1` — i.e. "rotate the stove knob joint below a
threshold", solvable by grabbing the always-engaged knob and twisting it.
Objects kept: a stove with a flame indicator, a turn-to-off valve, moveable
pots on the counter. Everything about *how* the valve is reached is new.

## What changed (strategic difference)

- **The seed's winning move is the decoy here.** The knob is on a D6 joint
  (free yaw + 16 mm heave; pitch/roll/slide locked). At rest its drive posts
  hang 6 mm BELOW the rotor's vane skirt: grab-and-twist — the seed strategy —
  spins the knob forever and moves the valve zero degrees. Smoke check 5
  executes exactly that strategy (≥ 180° of measured knob sweep) and asserts
  the dial does not move and the score stays ≈ 0.
- **State lives in a sealed mechanism, not a bare joint.** The valve rotor is
  a separate dynamic body on a revolute-Z joint inside a walled, roofed
  housing; the only opening is a 44 mm square roof hole occupied by the knob
  shaft (cfg-asserted finger-proof: hole ≤ 26 mm half-diagonal margins). The
  rotor can ONLY be moved through the clutch.
- **Execution order is REQUIRED and physically inherent** (see below): engage
  before you can turn; release before success can register. The seed task has
  no ordering at all.
- **Different code structure**: generic D6 `UsdPhysics.Joint` with per-axis
  `LimitAPI` + revolute rotor authored at bind time; all interaction in the
  solve is external-wrench force control (`set_external_force_and_torque`,
  body frame, re-set every step with
  `enable_external_forces_every_iteration`); there are **NO teleports** in the
  solve — every load-bearing interaction is contact dynamics or applied
  force. The rubric latches engagement and the running-min dial angle in
  `post_step`.

Vs siblings read while building this corpus (same seed and neighbours):
- `i13` (burner snuff, same seed): smothers the flame with a lid — no
  mechanism, no clutch, no ordering.
- `i280` (weigh-beam stove, scene3 seed): statics/counterweight puzzle solved
  by transport+drop teleports; here nothing is teleported and the puzzle is a
  2-DOF engage/turn/release mechanism.
- `i193` (fueling), `i338` (3-dial combination), `i36` (bridge): different
  mechanisms, different rubrics; none has a free-spin decoy DOF or a
  release-to-finish clause.

## Scene

Stove-local origin = valve axis on the deck top, world (0.12, 0.08, 0.14) on
a 0.80×0.64 counter. Kinematic stove: floor plate, 4 walls (interior half
75 mm, height 60 mm), roof strips (z 70–78 mm) leaving the 44 mm central
hole; burner at (−0.24, 0) with a flame tile recolored orange/charcoal from
the latched valve state. Dynamic bodies:
- **rotor** (0.25 kg, revolute-Z, limits [0°, 90°], ang-damp 4.0): plate
  annulus riding z 56–64 mm under the roof + 4 hanging vanes (radial 36–46 mm,
  z 38–56 mm). Dial angle is readable through a wall slit visual; 0° = OFF.
- **knob** (0.35 kg, D6: yaw free, heave [0, 16] mm): hub + 4 arms + 4 drive
  posts (top at 32 mm rest → 6 mm air gap below the vanes; at lift ≥ 12 mm
  the posts overlap the vanes ≥ 6 mm — cfg-asserted, as is "lifted posts
  never touch the rotor plate"), shaft through the roof hole, disc + red wing
  bar (110×24×18 mm) riding ≥ 10 mm above the roof at rest.
- **2 pots** (0.40 kg) shuffled across 3 counter slots with jitter + free yaw
  (decoys; never touched).

Randomization (READBACK-verified in smoke): initial dial angle U[68°, 88°],
knob yaw U[0, 2π), pot slot permutation + xy jitter.

## Rubric

- `success()` = dial ≤ 8° (OFF) AND knob released (lift ≤ 4 mm) AND knob calm
  (lin < 0.12, ang < 1.0) AND rotor calm (< 0.6) AND finite.
- `score()` = 0.15·(engaged latch) + 0.55·progress((θ0 − min-dial)/(θ0 − 8°)),
  capped 0.70 while not success; exactly 1.0 on success. Non-decreasing along
  the solve (engaged latches; min-dial only falls).

## Execution-order declaration (REQUIRED)

1. **ENGAGE before TURN** — physically forced: with the knob seated, posts and
   vanes share no z-interval (6 mm gap ≥ speculative-contact sum, asserted),
   so no torque on the knob can move the rotor.
2. **TURN to OFF** — only possible while holding the lift (posts drop out of
   the vane band below 12 mm of lift).
3. **RELEASE after OFF** — success explicitly requires lift ≤ 4 mm and calm;
   the solve asserts `not success()` while still holding at the stop, then
   releases and lets the knob fall 16 mm onto its bottom stop.

Out-of-order attempts are smoke-tested: turn-without-engage (check 5), wrong
direction (check 7, pins at the 90° stop), held-at-stop-not-released
(check 9: success never fires while held).

## Solution outline (solve.py, no teleports)

- P0: settle 120 steps; mass/pose/score readbacks.
- P1 ENGAGE (press-release-rephase): press straight up (mg+2 N, NO twist —
  a pressed twist would friction-drag the dial through the vane-bottom
  thrust contact); if a post parks under a vane, release fully (air gap →
  zero coupling), re-phase the knob ~24° while down (the free-spin decoy
  motion), press again, until a 20-step streak holds lift ≥ engage_z+0.5 mm.
  Latch check; 0.15 ≤ score ≤ 0.35 and dial provably un-wound (asserted).
- P2 TURN: full press (10 N, into the D6 top stop — friction-free, posts
  cfg-asserted clear of the plate) + velocity servo −1.2 rad/s to dial ≤ 1.5°,
  then brake. Asserts: valve OFF, still held, **`not success()` on record**,
  score ≈ 0.70.
- P3 RELEASE: zero wrench, 180 steps; knob falls and re-seats; success; 1.0.
- P4: ≥ 3.3 sim-seconds hands-off persistence; `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at each phase boundary; non-decreasing. Watchdog +
hard exit. Passes seeds 0 and 1.

## Embodiment argument (single Franka, parallel jaw, OSC)

- **Knob**: the red wing bar (110×24×18 mm) rides ≥ 10 mm above the roof at
  all times — an 80 mm parallel jaw grasps it from above across the 24 mm
  width. Pull up 16 mm (≈ 4.6 N ≪ payload), then wrist-roll clockwise while
  OSC holds the upward force; total twist ≤ 90° + ~35° backlash + slip ≈ 150°,
  within one wrist-roll stroke (±166°). Open the jaw to release. The solve's
  wrench (≤ 10 N, ≤ 0.4 N·m) is well inside Franka wrench limits.
- **Pots** (decoys, never touched): 75 mm square bodies with lid knobs — a
  standard top grasp exists, so their *irrelevance* is a choice the policy
  must make, not an impossibility.
- **Base pose**: (0.12, −0.55, 0) at the counter front, facing +y — the knob
  at (0.12, 0.08) is ~0.63 m out, inside the Franka envelope; the burner and
  pot row are visible for the flame/dial observations.

## Checks

Smoke battery (12): settle/no-NaN · 3-seed randomization readback ·
pot-slot permutation · null policy ≈ 0 · seed-strategy free-spin rejection ·
engage-only partial credit + latch persistence · wrong-direction rejection ·
near-miss (released at ~28°) rejection · held-at-stop success-never-fires ·
wrong-object (pot onto burner) rejection · per-step success audit clean ·
frames.npz recorded. Verdict `SIM_GEN_SMOKE: ALL PASS 12/12`.
