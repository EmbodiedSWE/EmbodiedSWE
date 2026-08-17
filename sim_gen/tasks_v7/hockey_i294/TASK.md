# hockey_i294 — CageCapture: slide the roofed cage around the stationary ball, then bar the mouth

`simgen.cage_capture` — single-arm tabletop-scale manipulation, NullRobot physics
package (scene + rubric + teleport solve + rejection smoke).

## Seed provenance

Derived from **rlbench/hockey**: *grasp the hockey stick and strike the ball into
the goal* — propel a free ball across open floor into a fixed, passively waiting,
open-mouthed goal.

## Strategic difference

The seed's plan is "the ball is the thing you move; the goal just waits". Both
halves are inverted here:

- **The ball must NOT move.** Success requires the white ball to sit within
  `anchor_tol` (4.5 cm) of the exact spot where it settled at reset, and EVERY
  partial-credit latch is gated on that same anchor at latch time. The seed's
  strategy — propelling the ball toward the receptacle — forfeits everything by
  construction (smoke check 6 fires the ball at the open cage mouth; it rolls
  ~0.5 m off its anchor, typically ends up physically "in the goal", and scores
  0.00 with no latch).
- **The goal is the thing you move.** The receptacle is a free, floorless,
  roofed CAGE (two side walls + back wall + roof, open front mouth 16 x 11 cm,
  open bottom) standing elsewhere on the floor. At 3.2 kg it exceeds the arm's
  ~3 kg payload — it cannot be lifted over the ball, only SLID on the floor. The
  roof means nothing can be dropped in from above; the walls and back mean the
  only way in is the mouth; the anchor clause means the ball cannot be brought
  to the cage. The one remaining route is the inverse of the seed's: steer the
  heavy receptacle mouth-first ACROSS the floor so it passes AROUND the
  stationary ball without touching it.
- **The final act is a placement, not a strike:** a yellow CHOCK BAR (21 cm,
  jaw-sized) must be laid flat on the floor squarely across the mouth (cage-frame
  window + alignment cone) to enclose the ball. A black decoy ball of identical
  size must remain OUTSIDE the cage (wrong-object interlock).
- **Different code structure:** cage-frame servo on the BALL's relative position
  + a precision stop + a place-from-above replaces grasp-stick + swing; nothing
  is ever struck, and the target object is never touched at all.

Within tasks_v7 the other hockey derivatives are i185 (load/tip/release a hinged
hopper feeding a sealed box through an elevated window) and i325 (gate extraction
+ floor-level push into an open goal mouth). Here there is no mechanism joint and
no ball propulsion of any kind — the mover/moved inversion (receptacle travels,
ball anchored) plus the mouth-barring placement is shared with neither.

## Scene

Fully procedural compound spawners (no external assets):

- **cage** (dynamic, 3.2 kg): side walls + back wall + roof; NO floor, NO front
  wall. Origin at the interior floor centre, mouth plane at local +x = hx
  (interior 21 x 16 x 11 cm). Authored MassAPI mass/CoM/inertia (custom spawners
  ignore cfg mass_props; low CoM z=0.035 so a 60 N push cannot tip it) and an
  authored slick material (mu 0.35 → pair-averaged 0.425 on the ground, ~13 N
  slide force — inside the arm's range, per the pair-averaged-friction rule).
- **chock** (dynamic, 0.18 kg): yellow 4 x 4 x 21 cm bar, long axis local y.
- **ball / decoy** (dynamic, 60 g): 6 cm white / black spheres, restitution 0,
  solver velocity iterations 4 (GPU sphere-creep fix — the anchor clause needs a
  truly still ball; smoke check 5 holds 300 idle steps at 0.0 mm drift).

**Randomization (readback-verified in smoke 3-4):** white ball xy band (the spot
it settles IS the anchor), decoy on a Bernoulli LEFT/RIGHT flank (torch.rand
comparison after burning the degenerate first draw), cage start pose (xy band +
yaw = facing-the-ball heading ± 25°), chock parked flat on the flank OPPOSITE
the decoy with xy + yaw jitter.

## Rubric

Latched partial credit (transient achievements keep credit), every latch gated
on the ball anchor at latch time, non-success cap 0.70:

| credit | clause |
|---|---|
| 0.15 | mouth centre ever within 9 cm of the anchored ball (approach) |
| 0.20 | anchored ball ever past the mouth plane (cage-frame readback) |
| 0.25 | anchored ball ever fully inside (≥ 4 cm behind the mouth plane) |
| 0.10 | chock ever seated across the mouth while the ball is captured |
| 1.00 | iff success(): ball fully inside AND on its anchor, chock seated (cage-frame window + 25° alignment cone), decoy outside, cage upright + seated, everything at rest |

## Teleport solution (solve.py)

Teleports are transport-only; the load-bearing interaction is contact dynamics:

- **P0** settle + layout readback; asserts score ~0 and the ball on its anchor.
- **P1 STAGE** (teleport): the cage is parked 22 cm short of the ball, mouth
  facing it — the arm's push-and-pivot repositioning of a receptacle it cannot
  lift. The mouth lands 11.5 cm from the ball, OUTSIDE the 9 cm approach gate:
  the staging write earns nothing (asserted).
- **P2 CAPTURE SLIDE** (wrench-driven): a horizontal external force at the cage
  CoM velocity-servos the cage at 0.10 m/s, steered on the ball's CAGE-FRAME y
  so the open mouth passes around the stationary ball, with a yaw-hold PD torque
  using a finite-difference yaw rate and a stall probe that escalates the force
  cap (pod-side wrench under-application guard). Stop at ball x_loc ≤ 0.050
  (10 mm inside the capture bound, 143 mm short of the back wall); friction
  brakes the 3.2 kg cage in ~1 mm. The ball is never touched: anchor error
  0.0 mm at every phase boundary on both tested seeds.
- **P3 CHOCK** (teleport + gravity): one pose write holds the bar flat, centred,
  aligned, 4 mm above its seat just outside the walls (6 mm clear of the wall
  fronts, 31 mm clear of the ball) — the arm's place-from-above of a 0.18 kg
  bar; gravity sets it down. Hands-off from the set-down to the verdict.
- **P4** persistence: ≥ 3.3 simulated seconds hands-off after success(), then
  `SIM_GEN_SOLVE: SUCCESS`. Passed on seeds 0 and 1 (score staircase
  0.00 → 0.00 → 0.60 → 1.00, asserted non-decreasing).

## Embodiment argument (single Franka, parallel jaw, OSC, one base pose)

Base at the world origin facing +x; everything actionable lies 0.20-0.72 m out
at floor level, inside the comfortable frontal workspace:

- **Slide the cage:** the cage cannot be lifted (3.2 kg > payload) but slides at
  ~13 N (mu_pair 0.425); the solve's stall-probe ceiling of 60 N is never
  reached — even 30 N is a light two-finger-knuckle push on the BACK WALL
  (11 cm tall, always facing the robot), applied at wall mid-height 5.5 cm
  ≈ CoM height, so the push cannot tip it (tip moment ≤ 2.1 N·m vs 3.3 N·m
  restoring). Steering = repositioning the pushing contact along the back wall;
  the ±25° start yaw is corrected the same way. Total travel ≤ ~0.45 m along a
  corridor the decoy (≥ 0.28 m to the side) and chock (≥ 0.30 m opposite) never
  intrude on.
- **Never touch the ball:** the mouth is 16 cm wide vs the 6 cm ball — ±5 cm of
  lateral slack; the solve's closed-loop steering error was ≤ 5 mm.
- **Place the chock:** 4 cm square bar < 8 cm jaw opening, lying in the open
  with clearance on all sides; place-from-above at ~0.65 m reach, target window
  ±2 cm x ±4 cm x ±25° — coarse by pick-and-place standards, and the drop is
  4 mm onto flat floor next to (not onto) any other body.
- No bimanual step, no regrasp under load, no lift of the heavy object, no
  reach outside the frontal workspace.

## Execution order is physics-forced, not rubric-timestamped

The cage must reach the ball before the chock can matter (the chock latch
requires the ball captured at latch time — barring the empty cage earns nothing,
smoke 11). The slide must happen mouth-first around the anchored ball (roof
blocks drop-in, walls block side entry, anchor gate blocks moving the ball —
smoke 6). The chock must come last (it seats just outside the mouth; sliding the
cage after seating would plough it away). No rubric clause consults time.

## Checks (smoke.py rejection battery, 16 checks)

1. settle/no-NaN + MASS READBACK (cage 3.2 / ball 0.06 / chock 0.18 via
   root_physx_view.get_masses); 2. reset score ~0; 3-4. randomization readback
over 8 seeds (ball band, decoy side flips, cage pose+yaw spreads, chock always
opposite the decoy); 5. null policy 300 steps ~0 with 0 mm anchor creep;
6. SEED-STRATEGY end state: ball fired at the open mouth travels 0.49 m off its
anchor and scores 0.00 — no latch, even though it ends inside the cage;
7. approach-only = exactly the 0.15 band; 8. partial near-miss (2 cm past the
mouth plane, short of the 4 cm margin) = 0.35 band; 9. capture with the mouth
left open = 0.60 band, NOT success; 10. chock dropped beside the side wall —
cage-frame window rejects, still 0.60; 11. chock-first on the EMPTY cage —
physically seated (readback) but capture-gated: score 0.00; 12. wrong object:
cage slid over the DECOY + chock seated — score 0.00; 13. exclusion: white
captured + chock seated + decoy teleported inside too (decoy FIRST, so no
construction prefix satisfies the goal) — only the decoy clause rejects, latched
0.70 cap, NOT success; 14. latched credit unchanged after the white ball is
removed, no success; 15. never-success audit; 16. final no-NaN. Frames recorded
to frames.npz (212 frames, 960x600).

## Files

- `scene.py` — CageCaptureScene, `SCENES.register("cage_capture")`,
  `register_env` (robot="null").
- `solve.py` — `python -m simgen_tasks.hockey_i294.solve --headless [--seed N]`.
- `smoke.py` — `python -m simgen_tasks.hockey_i294.smoke --headless`.
