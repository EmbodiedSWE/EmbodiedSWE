# stack_cube_i382 — Swing Pump (`simgen.swing_pump`)

## Seed provenance

Derived from **maniskill/stack_cube** (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/stack_cube.py`):
grasp the red cube and set it statically on the blue cube so their centres align; the
judged quantity is the relative pose of two free cubes on a tabletop. The seed's
strategy is *a single grasp–transport–align–release cycle*, and its goal state is a
*static cube-on-cube stack*.

## What changed, and why it is strategically different

Kept from the seed only its protagonist — *the red 50 mm cube*. Everything about what
the cube is, what can be done to it, and what is judged is inverted:

- **The red cube can never be grasped, carried, or placed.** It is WELDED to a steel
  rod as the bob of a rigid pendulum on a 45 kg stand. There is no free-object
  transport anywhere in the task, no second cube to align to, and no pose target: the
  judged coordinate is the *pendulum hinge angle's trajectory*.
- **Energy accumulation replaces pose accuracy.** The bob is only touchable inside a
  ground-level access window (exposed below the enclosure plates' lower edge,
  |θ| ≲ 50°), and the declared 2 N push cap is asserted in `__post_init__` to be
  *incapable* of the goal in one stroke: it holds statically only to ~32° and a full
  window transit at the cap injects ~0.92 J < the 1.08 J climb to the flap. The ONLY
  route is resonant pumping — many timed strokes synchronized with the swing phase,
  banking kinetic energy across cycles. The seed has zero dynamics; here the entire
  task is dynamics.
- **The goal is a one-way capture event, not an arrangement.** At arc 100° a
  gravity-returned check flap lets a fast bob shove through outward and then falls
  shut (a mechanical ratchet, race-condition asserted: blade re-close beats the bob's
  ballistic dwell). Success = the pendulum resting captured in the pocket beyond the
  flap, with the flap re-closed — *behind* a mechanism, reached only by a controlled
  ballistic release at the right stored energy (too little: bounces off the pass
  line; too much: the ±140° stop slams it back out).
- **Versus the read corpus:** no existing task judges resonance or timed periodic
  forcing. Sibling `stack_cube_i183` (Ballast Hatch) banks *weight* quasi-statically
  to hold a door for a separate pushed payload; here nothing is accumulated but
  *energy in the goal object itself*, and the flap is passive one-way, never held.
  `carousel_airlock` velocity-servos a free rotor; `die_quarter_roll` tips statically
  edge over edge; pocket/detent tasks (`rolling_disc`) coast a free body in once.
  None have a phase-locked pump, an amplitude rubric, or a check-valve capture of a
  jointed goal body.

## Apparatus (fully procedural)

A 45 kg dynamic STAND: two masts carry a Y-axis revolute pendulum at 0.55 m — steel
rod down to the RED 50 mm cube bob at radius 0.30 m (0.33 kg, authored CoM 0.285 m
below the hinge, travel ±140°). Two vertical enclosure PLATES sandwich the swing
plane (62 mm slot for the 50 mm bob), lower edge at z = 0.357: the bob is EXPOSED
(reachable) only below that edge, |θ| ≲ 50° — everything higher swings inside the
sealed sandwich (end columns + top beam close the other faces). On the +x side at arc
100° a CHECK FLAP rides its own frame-mounted revolute: a 105 mm hinge-weighted blade
(20 g) whose closed tip sits just inside the bob's face radius (asserted 0.267 <
tip_r = 0.270 < 0.275 m), open stop at 70° — 30° short of over-center, so gravity
always re-closes it. Beyond it the pocket band [103°, 139°] ends at the swing stop.
Nine `__post_init__` asserts pin the honesty: cap can't hold at the window edge,
single transit can't pay the climb, coast target clears flap but not the stop, blade
covers/clears the swept band closed/open, ratchet race dwell > 1.3× blade fall.

**Randomization (readback-verified):** whole-apparatus yaw ±180° + xy jitter ±3 cm
(stand, pendulum, flap written consistently), initial pendulum angle ±20°. Memorized
world-frame push directions fail; the solve re-reads the frame quaternion every step.

## Rubric

- `success()` = pendulum resting in the pocket (θ ∈ [103°, 139°]) **and** still
  (pose-FD rate < 0.25 rad/s held 0.5 s — phantom-velocity-proof) **and** flap
  re-closed (|φ| < 8°) **and** both trajectory latches earned:
  `lat_amp` (a continuous up-crossing of 60°) and `lat_pass` (a continuous crossing
  of the 106° pass line *while* the flap is ≥ 25° open — a written-open flap plus a
  teleported bob earns nothing, because crossings are guarded by a per-substep
  continuity bound of 0.05 rad that teleports and slams violate).
- `score()` (monotonic, latched): 0.30 amplitude, 0.60 flap transit, 1.0 iff
  success. Credit survives the swing decaying, the bob being yanked back out, and
  the flap re-closing.

## Solution outline (solve.py — the legitimacy certificate)

**Zero teleports** — this solve is pure bounded-force contact dynamics end to end:

- **P0** settle 0.5 s, frame/theta readback (seed-dependent yaw and start angle).
- **P1+P2** `pump_and_release()` — a release-decision controller, all forces ≤ the
  declared 2 N cap, applied only while the bob is exposed in the window (body-frame
  wrench re-encoded from the frame quaternion each step):
  - − descents are pumped toward a servoed energy reference `e_ref`;
  - + ascents are pumped low but BRAKED above e_hold = PE(92°), so + peaks stay
    below the flap until the controller *chooses* to release;
  - − turning points (pose-exact, ω = 0) are judged against the release band
    [128°, 137.5°] (PE 1.48–1.60 J: clears the 106° pass line with margin, stays
    under PE(139°) so the stop can't slam it back). Off-band peaks correct `e_ref`
    by the exact PE error (clamped ≤ 1.65 J to avoid stop slams);
  - an in-band − peak ARMS the controller: force-free coast through bottom, up the
    +x side, shoves the flap open, crosses the pass line (Δθ/step ≈ 0.04 < the
    0.05 continuity bound — a genuine slow crossing), the blade falls shut behind
    (readback: shut ~0.15 s after transit) — scores 0.30 then 0.60 en route.
- **P3** capture monitor: wait for pocket + still; if the bob escapes below 55°
  (never observed in the final runs), raise the release band 2° and re-pump (≤ 3
  attempts) — score 1.0 on capture.
- **P4** hands-off persistence ≥ 3.4 simulated seconds, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge for seeds **0, 7**; scores non-decreasing 0 → 0.30 → 0.60 → 1.0
(seed 0: release at −135.7°, first-attempt capture; seed 7: release at −128.8°,
arrested at +105.5° on the shut blade).

## Embodiment argument (Franka, one base pose)

Base at ~(0.6, 0, 0) m from the stand centre in the frame yaw, gripper closed,
facing the swing plane — exactly a person pushing a playground swing. Per-object
contact strategy:

- **Red bob (pushed, never grasped):** at the hang the bob centre is at z = 0.25 m,
  fully below the plate edge; the whole ±50° exposed arc (a ~0.46 m ground-level
  span) is inside a Franka's reach envelope from the one base pose. The solve's
  strokes are ≤ 2 N horizontal fingertip taps timed to the swing phase — precisely
  the contact a closed-gripper knuckle delivers, and the release decision maps to
  simply *stopping*. No grasp is possible (welded bob) and none is needed.
- **Flap / pocket / plates:** never touched. The flap and the entire pocket band
  live above z = 0.357 inside the 62 mm plate sandwich, sealed by end columns and
  the top beam — nothing need (or can) reach them; the flap is actuated only by the
  transiting bob.

## Execution-order declaration

No discrete order is declared beyond what the physics enforces: the check flap's
closed blade covers the bob's swept band (asserted + smoke check 7: the 2 N cap
cannot push back through), so amplitude build-up necessarily precedes the transit and
the transit necessarily precedes the captured rest; the two trajectory latches simply
record that enforced order, and `lat_pass` demanding a flap ≥ 25° open at a
continuity-guarded crossing makes staging the end state worthless.

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 15/15` on forge)

1. Settle/no-NaN: flap closed (readback), pendulum in its start band, bob exposed,
   score ≈ 0.
2. Randomization readback: apparatus xy + yaw (max-pairwise circular) vary across 6
   seeds.
3. Randomization readback: initial pendulum angle varies.
4. Null policy (400 steps): small-swing band, flap closed, no latches, score ≈ 0.
5. Exposure window is real (readback): bob exposed at the hang, NOT exposed at 80°.
6. **Seed-strategy analog:** the pendulum written statically into the pocket (the
   seed's precise-placement move) settles in-pocket + still + flap closed, yet is
   REFUSED — no trajectory latches, score ≈ 0.
7. One-way gate: from rest on the blade, the declared 2 N cap pressed back toward
   the window for 3.3 s — min θ stays > 98°; the gravity ratchet physically holds.
8. Fake transit: bob written into the pocket WITH the flap written wide open — the
   staged mid-pass picture earns no latches.
9. Amplitude-only: a genuine 80°-peak swing earns exactly the 0.30 stage.
10. Flap near-miss: a genuine sub-pass-energy swing verifiably reaches and deflects
    the blade (max flap 56°, readback) but never crosses; flap re-closes; score
    still 0.30.
11. Genuine pass: a high swing earns the transit latch (0.60); the bob is yanked
    back out before settling — latched credit survives regression, no success.
12. Self-reclosing: the shoved-open flap falls shut on its own (readback), 0.60
    survives the mechanism resetting.
13. Monotonicity: null 0 < amplitude 0.30 < transit 0.60, strictly; success never.
14. Rejection audit: success() never True at any judged point in the battery.
15. Final no-NaN.
