# pick_cube_i295 — Catwalk Bridge

## Seed provenance

Seed: `maniskill/pick_cube` (RoboVerse
`roboverse_pack/tasks/maniskill/pick_cube.py`): a 4 cm red cube
(`PrimitiveCubeCfg`, 0.02 half-size) on a bare table must be grasped by a
Franka and lifted 0.1 m (`PositionShiftChecker`, axis z). One object, one
plan: grasp, lift, hold.

## Strategic difference

The 4 cm red cube is kept as the *cargo*, but the seed's one-grasp lift plan
is deliberately defeated and replaced by a build-the-path-then-traverse-it
puzzle:

- **The seed's plan fails here.** The goal is the floor of a low, roofed
  ISLAND deck on the far side of an open TRENCH (no floor between the decks).
  The roof leaves 7.2 cm of interior headroom, is walled on both sides and at
  the back, and its only opening is the tunnel mouth at the bench edge —
  every carry/lift/drop strategy lands the cube ON the roof and earns nothing
  (smoke check `seed-drop` reproduces exactly the seed's plan and rejects it).
- **The path must be BUILT before it can be used.** The trench (14 cm) can
  only be crossed on the yellow plank (24 cm — a ~5 cm seat on each sill).
  The plank's *half-length (12 cm) is shorter than the gap*, so shoving it
  lengthwise along the deck tips it into the trench before its nose can reach
  the far sill (smoke check `plank-shove` proves the decisive shove fails —
  contact dynamics, not rubric fiat). It must be carried in *level* through
  the mouth and set down spanning — a placement problem, not a push.
- **The traverse is itself a manipulation.** The cube must then be pushed
  along the plank, over the open trench, off the step-down at the plank's far
  end, onto the island floor — deep inside the tunnel, beyond fingertip
  reach. The blue rod (35 cm) is the poling tool the embodiment uses for it.
- **Physically forced execution order** (the seed has no ordering at all):
  bridge first, crossing second. Without the bridge, the *same solving push*
  drops the cube into the trench (check `no-bridge`).
- Versus the sibling tasks read this session: `pick_cube_i125` is a die
  *orientation* puzzle (no orientation goal here — the goal is position via a
  built structure); `pick_cube_v1_i170` is a weight-interlock airlock (joints,
  springs, a mechanism — here the plant is *pure passive rigid bodies*; the
  "mechanism" is a structure the solver must assemble); the `pen_holder`
  exemplar is multi-item container filling.

## Scene (env-local; ground z = 0, deck tops z0 = 0.08)

- **Bench** (start deck): x∈[−0.46, −0.08], y∈±0.16, open to the sky.
- **Island** deck: x∈[0.06, 0.26], same width. **Trench**: the floorless gap
  x∈(−0.08, 0.06) between them, dropping 8 cm to the ground.
- **Tunnel**: from the bench's rear edge (x=−0.08, the mouth) to behind the
  island, side walls + back wall + a flat roof at z 0.152–0.182 → 7.2 cm of
  headroom over the decks. Only opening: the mouth.
- **Bodies** (masses authored, asserted by readback): yellow plank
  24×10×1.2 cm / 0.15 kg; red cube 4 cm / 0.05 kg (the seed's cargo); blue
  rod 35×2×2 cm / 0.08 kg (poling tool, staged on the ground beside the
  bench). Friction 0.30 everywhere, restitution 0.
- **Randomization** (verified by readback in smoke): plank and cube are dealt
  to the two bench slots by a sampled swap (position never identifies a
  body), ±2.5 cm xy jitter, ±180° yaw; rod slot ±4 cm jitter, ±30° yaw.

## Rubric

Latched partial credit (asserted non-decreasing in solve): `latch_bridge`
(plank seen *spanning*: level, centre in the sill-height z band, x-extent
covering both sill edges, still) → 0.25; `latch_transit` (cube seen *riding
the spanning plank over the trench*: on-plank z band, past mid-gap, slower
than 0.5 m/s, while the bridge holds) → +0.35; success — cube at rest ON THE
ISLAND FLOOR (deck-height z band 0.093–0.107; the on-plank rest sits at
0.112 and is excluded, the trench at 0.02, the roof at 0.20), well inside the
island, everything settled — → 1.0. Success is live physical state: the only
floor between the decks is the one the solver built.

## Teleport solution (transport only; all load-bearing interaction is contact)

1. **Build the bridge**: teleport the plank to hover 1 cm above its spanning
   pose (the pose the arm reaches by inserting it level through the mouth)
   and release — gravity + the two sills do the seating. Nothing ever writes
   a seated pose; if the seats were wrong the plank would tip into the trench.
2. **Cross the trench**: teleport the cube onto the plank's back seat, still
   outside the mouth (arm-reachable free space), then push it along the plank
   with a capped-force velocity servo on the scene's `push_f` buffer
   (K·dt/m = 0.067 ≪ 1 against the one-substep wrench delay; cap 0.40 N is
   under the cube's mg = 0.49 N tipping bound; stalls escalate the friction
   feedforward, never the cap). Riding the bridge, the step-down at the
   plank's far end, and the landing are all contact dynamics.
3. Then ≥3.5 simulated seconds hands-off (push buffers asserted zero) before
   the verdict.

## Franka embodiment argument (per object)

Base on the ground at ~(−0.55, 0.0), facing +x; the bench top (z 0.08) and
the mouth (x=−0.08) are within a ~0.5 m reach envelope.

- **Yellow plank (24×10×1.2 cm, 0.15 kg)**: the 1.2 cm edge fits the Franka
  jaw. Grasp the plank near its *back end* across the thickness, carry it
  level, and insert it nose-first through the mouth low over the deck — the
  wrist stays outside the tunnel; only the plank (1.2 cm thick, under the
  7.2 cm headroom with room to spare) goes in. Set down when the nose rests
  on the island sill; release. The spanning pose keeps ~11 cm of plank on
  the bench side of the mouth — exactly where the gripper holds it.
- **Red cube (4 cm, 0.05 kg)**: the seed's own pick — grasp on the bench,
  set down on the plank's back seat at x≈−0.105, *outside* the mouth.
- **Blue rod (35×2×2 cm, 0.08 kg)**: grasped mid-shaft like a pen; the free
  tip pushes the cube ahead of it at cube mid-height. With the tip at the
  island edge the hand is still ~0.2 m outside the mouth — the rod is what
  makes the deep-tunnel push reachable, and its 2 cm section fits the
  headroom trivially. The 0.40 N capped push is far below fingertip scale.
- No simultaneous two-hand requirement anywhere: the bridge is a static
  structure once seated; the arm then re-grips cube, then rod, sequentially.

## Execution order (REQUIRED, physically enforced)

bridge → crossing. There is no floor between the decks until the plank is
seated (check `no-bridge` drops the cargo into the trench under the same
solving push), and the plank cannot be seated by pushing (check
`plank-shove`), so the build must be a deliberate level insertion completed
*before* the cube ever leaves the bench.

## Checks (smoke.py, 13)

1. `settle` — clean reset: finite, all three bodies at their sampled slots by
   readback (<5 mm), no bridge, score <0.05.
2. `plant` — authored masses via `get_masses()`; plank half-length < gap <
   plank length; a riding cube fits under the roof.
3. `randomization` — seeds 11–18: both swap deals seen, ≥5 distinct jitters
   and yaws, physical readback each seed.
4. `null` — 2 s hands-off: score <0.05, no latches.
5. `seed-drop` — the seed's carry-and-drop over the target: lands ON the roof
   (physical roof readback), never inside, ~0.
6. `no-bridge` — the solving push without a bridge: cube falls INTO the
   trench, no transit latch.
7. `plank-shove` — decisive lengthwise shove (0.9 N cap, ≫ sliding friction):
   the plank demonstrably moves, then tips — bridge latch never fires.
8. `island-rest` — near miss: plank level, still, at sill height but fully on
   the island (covers neither sill): `bridge_ok` false, no latch.
9. `rod-span` — wrong object: the rod seated spanning the trench (readback)
   earns nothing — the rubric reads the plank.
10. `stop-short` — real bridge, cube pushed across then STOPPED on the plank:
    both latches fire, score ~0.60, no success (on-plank z excluded).
11. `exactness` — continuing the push lands the cube on the island floor:
    success, |score−1| <1e-3, stable 1 s later.
12. `revocation` — cube teleported back to the bench: success revoked (live
    state), the latched 0.60 remains.
13. `frames` — ≥20 RGB frames recorded to frames.npz.

Run (forge): `python -u -m simgen_tasks.pick_cube_i295.solve --headless`
and `python -u -m simgen_tasks.pick_cube_i295.smoke --headless`.
