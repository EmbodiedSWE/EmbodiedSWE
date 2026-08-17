# registry_i373 — Totem Tunnel (topple, thread the bore, re-erect)

## Seed provenance

Seed: `simpler_env/registry`
(`RoboVerse/roboverse_pack/tasks/simpler_env/_metasim/registry.py`) — the SimplerEnv
task registry: pick-coke-can (standing / laid / upright variants), move-near,
open/close-drawer, put-in-closed-drawer, put-spoon-on-towel / carrot-on-plate /
eggplant-in-basket, stack-cube. Every entry is ONE grasp-lift-place through free
space (or one drawer-joint push); the object's orientation never matters beyond
"however it was grasped", and the route to the goal is unconstrained air.

## What changed and why it is strategically different

The route itself becomes the task, and orientation is load-bearing twice:

- A tall terracotta TOTEM (40 x 40 x 110 mm) stands inside an open-top walled PEN
  (260 x 260 mm inside, 120 mm walls). The one permitted exit is a low roofed
  TUNNEL through the front wall — a 66 mm wide x 56 mm tall aperture under a
  lintel. Honest-by-construction selectivity (asserted in the cfg): the totem
  cannot pass standing (110 > 56), cannot pass crosswise (110 > 66), and passes
  ONLY lying down, long axis along the bore (40 < 44 and 40 < 46 clearance).
- The plan is topple -> slide lengthwise through the bore -> stand back upright on
  a goal disc outside. No seed task (and no other task in this corpus) requires
  knocking an object down as a means, threading it through a roofed aperture, and
  restoring its original pose at the goal.
- The seed's own strategy is explicitly rejected: an order-chained trajectory
  latch (entered at the mouth inside the pen -> mid-bore, the body spanning the
  whole roofed aperture, a pose unreachable from above -> through, clear of the
  wall) must fire in order. Carrying the totem over the open walls and standing it
  perfectly on the disc reproduces every live success clause — and still scores
  ~0 (smoke check 5). The rule "it must leave through the tunnel" is declared in
  describe()/instruction() (like "put it in the CLOSED drawer" declares the
  drawer cycle) and enforced physically by the latch chain.

## Solution phases (solve.py)

- P0 settle + layout readback; baseline score ~0 asserted.
- P1 TOPPLE (contact): carry the totem (teleport = transport, zero velocity) to a
  launch spot on the tunnel axis, then push at its CoM with 0.78 N — above the
  tipping threshold m·g·a/h = 0.54 N, below the static-friction slide threshold
  0.88 N — so it PIVOTS over its base edge and falls flat, aligned. Latched
  `lying` credit -> score 0.15.
- P2+P3 THREAD (contact): body-frame wrench servo (0.9 N feedforward just above
  sliding friction 0.74 N + velocity loop + lateral centring + small yaw torque)
  slides it lengthwise through the bore. entered -> mid -> through fire in order
  -> score 0.55 (0.15 + 0.30 progress + 0.10 through).
- P4 ERECT (transport + gravity): carry it upright to a 10 mm hover over the goal
  disc, release; it seats and settles under contact. near latch + success ->
  score 1.0.
- P5 hands-off persistence >= 3.3 sim-seconds, then `SIM_GEN_SOLVE: SUCCESS`.

Execution-order declaration: topple must precede threading (only a lying totem
enters the bore) and threading must precede the erection at the disc (the transit
chain is a success clause); the order is physically inherent and latch-enforced.

## Embodiment argument (single Franka, parallel-jaw gripper, one base pose)

Base ~0.45 m from the pen centre, facing the tunnel face. (1) Topple: push the
totem's upper face with closed fingertips — reach over the 120 mm wall, a 0.5-1 N
horizontal nudge. (2) Thread: push the trailing end with a fingertip over the wall
until the nose exits, then finish from outside by pushing the last stretch through
the 66 x 56 mm aperture (fingertip clearance) or pulling the protruding nose.
(3) Erect: the 40 mm square section fits the ~80 mm jaw span; grasp the lying
midsection, lift, rotate the wrist 90 deg, set the base on the disc. All poses are
within a 0.85 m reach envelope; the disc (radius 60 mm, tolerance 45 mm) is a
forgiving target.

## Checks

- solve: `SIM_GEN_SOLVE: SUCCESS` on >= 2 seeds; monotone `SIM_GEN_SCORE` at
  every phase boundary; per-phase asserts (baseline ~0, lying latch, all three
  transit latches in order, no-success-until-erected, persistence).
- smoke: 10 checks — settle/no-NaN; 3-seed max-pairwise randomization readback
  (pen yaw/xy, totem local xy, disc local xy); null policy ~0; CROSSWISE CANNOT
  PASS (physical push probe, non-vacuous: moved but walled); SEED-analog
  over-the-wall carry (end state identical to success, refused, ~0); transit
  granted + lying at disc (upright clause); transit granted + 30 deg sloppy
  release (falls over); transit granted + upright 15 cm off-disc (placement
  clauses); mid-bore cold teleport (order chain needs its first link); frames.npz
  video. `SIM_GEN_SMOKE: ALL PASS 10/10`.
