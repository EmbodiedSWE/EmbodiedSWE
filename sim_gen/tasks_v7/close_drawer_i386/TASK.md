# twistlock_drawer_i386 — bayonet-lock the self-opening drawer of a nose-down cabinet

Package: `sim_gen/tasks_v7/close_drawer_i386`
Env: `simgen.twistlock_drawer_i386` (scene-level, `robot="null"`; the scene name is
corpus-unique — "bayonet_drawer" was already taken by
`libero_kitchen_scene1_open_bottom_drawer_i88`)

## Seed provenance

Seed task: `rlbench/close_drawer`. In the seed, the robot pushes a plain, level
prismatic drawer shut, driven by recorded waypoints. One guided push IS the goal:
the drawer stays wherever it is left, so "closed" is just "was pushed once."

## What changed, and why it is strategically different

1. **The drawer cannot simply "be closed."** The whole cabinet is pitched 7.5°
   NOSE-DOWN on a world-flat plinth and the drawer rides near-frictionless rails —
   a dead-man mechanism with gravity as the ADVERSARY: any un-locked drawer glides
   fully open on its own within a second (proved physically in smoke checks 1, 5
   and 6). The seed's entire strategy — push shut, let go — is executed for real
   in smoke with the solve's own force servo and scores 0.29 with `success()`
   False: the drawer glides right back out.

2. **Closure must be MANUFACTURED via a two-DoF bayonet lock.** The drawer's front
   face carries a revolute ROTOR (red T-handle → steel lug bar 100×12 mm); the
   cabinet carries a fixed slotted BEZEL PLATE (slot 112×34 mm) standing proud of
   the drawer. Pre-computed lock geometry, all asserted in `__post_init__`:
   - the bar passes the slot only within ~12.9° of horizontal (pass angle solved
     by bisection at cfg time); squared it clears by 6 mm (y) / 11 mm (z);
   - pressed home, the bar sits BEHIND the plate; a ≥ 70° twist (90° natural)
     projects 98 mm across the 34 mm slot — deep, unmissable retention;
   - released, the drawer creeps until the bar bears on the plate's back:
     q_rest = 7 mm, INSIDE closed_tol = 12 mm with ≥ 4 mm margin;
   - a rotor twisted while the drawer is OPEN becomes a TRAP: the vertical bar
     jams on the plate's FRONT at q_trap = 25 mm ≥ closed_tol + slack + 8 mm, so
     the pre-twisted press can never even reach the locked-latch gate.

3. **A pre-condition with per-episode variety.** The rotor spawns misaligned by
   ±[8°, 22°] (random sign) — STRADDLING the 12.9° pass angle (asserted), so some
   episodes require squaring the handle before the press and some merely reward it.

4. **Physics is the judge.** The locked latch fires only for twist ≥ lock_min
   WHILE the drawer is inside the closed band (q-gated, so the trap and the
   locked-open decoy earn nothing); `success()` additionally demands a 30-step
   persistent-stillness streak — hands-off self-retention is exactly what the
   bayonet provides and exactly what every non-locked state cannot.

## Solution outline (zero teleports — verified on forge)

No teleports at all: the drawer and rotor are driven by regulated external
wrenches only (velocity-regulated force on the drawer along its body-frame rail
axis with gravity feedforward −mg·sin(pitch) ≈ −1.93 N; PD torque on the rotor
about its body-frame hinge axis; gains sized KV·dt/m ≈ KD·dt/I ≈ 0.17).

- **P0** settle 120 steps + honesty readbacks: authored masses via
  `root_physx_view.get_masses()`, drawer GLIDED fully open on its own
  (q = 0.1300 = stroke), rotor misaligned ≥ 3°, score < 0.02 → `SIM_GEN_SCORE 0.00`.
- **P1** square the rotor (|θ| < 0.8°, 20-streak) — the bar can now pass the slot.
- **P2** press the drawer home holding the rotor square: q → 0.003; pressed latch
  fires → `SIM_GEN_SCORE 0.29`.
- **P3** twist to 92° while maintaining the press: the bar turns vertical behind
  the plate; locked latch fires; success() already holds under the press →
  `SIM_GEN_SCORE 1.00`.
- **P4** RELEASE (all wrenches off): the drawer creeps to q = 0.0071 ≈ q_rest
  0.0070 and the bar catches the plate's back; ring down until `success()` holds
  120 consecutive steps.
- **P5** hands-off ≥ 3.3 simulated seconds; success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, scores
monotone 0.00 → 0.29 → 1.00, zero persistence flickers, released rest pose
matching the predicted q_rest to 0.1 mm on all three seeds.**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(0.55, 0, 0), facing the cabinet front (bezel plane at x ≈ 0.03,
cabinet ±0.10 m of xy jitter). EVERY interaction happens on ONE handle: the red
T-handle crossbar (90 × 16 × 16 mm, jaw-graspable across its 16 mm bar) stands
24 mm PROUD of the bezel plate (asserted ≥ 20 mm finger clearance), 0.36–0.49 m
from the base at height ~0.15 m — comfortably inside Franka's envelope, approach
along +x with the wrist axis roughly aligned with the rotor shaft. A single grasp
suffices for the whole task: twist the wrist to square the handle (±22° → 0,
≤ 0.05 N·m against the free hinge), push 12 cm along the rail against
1.93 N + damping to press the drawer home, twist the wrist a quarter turn
(bar + handle inertia is tiny; wrist roll range ±166° covers 90° easily), open
the jaw and retract. Forces and travels are trivial for the arm; no regrasp, no
second hand, no whole-drawer grasp is ever needed (1.5 kg drawer is driven
through the handle).

## Execution order

The mechanically-forced order is square → press → twist → release, and
`describe()` states it. Geometry enforces it both ways: twist-before-press jams
at q_trap = 25 mm (smoke 7); press-with-misaligned-rotor jams the same way at
spawn angles beyond the pass angle; twist-after-release is impossible (the drawer
has already glided away). Within that, the handle may be squared and the drawer
pressed any number of times — nothing is consumable.

## Rubric

- +0.15 × max closure fraction over the LAST 5 cm of travel (latched; spawns
  start at ≥ 7.15 cm open and glide AWAY, so the null policy earns exactly 0).
- +0.15 once the drawer has EVER been pressed inside closed_tol = 12 mm (latched).
- +0.30 once the rotor has EVER been ≥ 70° WHILE the drawer was inside
  closed_tol + 2 mm (latched; the q-gate is what kills the trap and the decoy).
- Partial credit capped at 0.60; `score = 1.0` iff `success()` = q ≤ 12 mm ∧
  θ ≥ 70° ∧ settled (30-step stillness counter-latch) ∧ finite.
- Null policy scores exactly 0.

`__post_init__` honesty asserts (13): squared bar passes the slot with margin /
locked bar cannot; q_rest inside the closed band, q_trap far outside the
locked-latch gate; spawn misalignment straddles the pass angle both ways;
wrong-direction stop (−25°) far below lock_min; 90° lock comfortably inside the
stops and clear of the ±180° revolute wrap; shaft clears the slot; bar swing
circle clears the bezel arms; T-handle proud of the bezel for a jaw grasp; spawn
band beyond the closure-credit ramp; pitch actually drives the glide; jaw fit.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 13/13`)

1. Reset settles finite AND the dead-man is real: the drawer spawned part-open
   glided fully open on its own (readback q = 0.1300 = the open stop).
2. Fresh reset: score ~0, no success.
3. Randomization A: station yaw (spread 2.64 rad) and xy jitter (46/93 mm) real.
4. Randomization B: immediate post-reset readback — q0 varies (37 mm spread),
   θ0 varies in magnitude AND sign across 6 seeds.
5. Null policy 240 steps: drawer back on the open stop, score ~0.
6. SEED strategy for real (flagship): rotor squared, drawer genuinely pressed to
   q = 0.0034 with the solve's own servo, released → glides fully back open;
   NOT success, score 0.29, latched credit survives the re-opening.
7. Pre-twist trap: rotor twisted to ~90° while open, then a REAL press — the
   vertical bar jams at q = 0.0247 ≈ q_trap (moved 105 mm, non-vacuous); the
   q-gated locked latch never fires; score 0.076.
8. Under-twist snag: pressed home, twisted only ~50° (> pass angle, < lock_min),
   released — mechanically RETAINED at q = 0.0071 yet REFUSED: not success, 0.29.
9. Wrong-direction twist (−24°): also mechanically retained, also refused.
10. Locked-open decoy: rotor ≥ 70° while the drawer sits open on its stop — the
    q-gated locked latch never fires, score ~0.
11. Settle gate: the linkage placed in the locked pose WHILE MOVING (real 0.12 m/s
    rail velocity written with the pose — a zero-velocity teleport is vacuous, the
    stillness counter keeps running): pose passes both gates but NOT success;
    removed before ring-down (the battery never succeeds).
12. Rejection audit: `success()` never fired at any judged point.
13. Final state no-NaN.

Smoke also records `frames.npz` (190 × 600 × 960 × 3) from a perspective camera.
