# pin_lock_drawer (`pull_cube_i245`)

Disassemble a geometry-enforced interlock in its forced order: **lift-unlatch** and
slide out the yellow GUARD pin, slide out the green LOCK pin (free only once the
guard's arm is gone), then pull the freed captive drawer to its hard stop with the red
cargo cube riding in its open-top bay. The cube is never touched.

- Env id: `simgen.pin_lock_drawer` (NullRobot scene-level env)
- Files: `scene.py` (scene + rubric + force plant), `solve.py` (teleport solution
  certificate), `smoke.py` (18-check rejection battery, recorded), this file.

## Seed provenance

| | seed | this task |
|---|---|---|
| source | `maniskill/pull_cube` (RoboVerse `roboverse_pack/tasks/maniskill/pull_cube.py`) | new scene, procedural geometry only |
| setup | one free cube on an open floor | cube CAPTIVE from the start: riding a drawer bay under a roof, drawer pinned shut by two interlocked cross-pins, the guard pin latched behind catch posts |
| plan | one planar act: hook the cube and **drag** it a few cm | four-stage ORDERED disassembly: lift-unlatch + extract the guard pin, extract the lock pin, pull the drawer, cargo rides along |
| goal | xy-proximity of the moved cube to a painted floor region | the MECHANISM's end configuration: drawer at its out-stop, both pins fully outside the channel volume, cube still aboard, all settled |

## Why strategically different

- **vs the seed**: the seed's manipulated object IS the goal object and the goal is a
  floor region readout on it. Here the cube is never the manipulated object at all —
  there is **no floor goal region anywhere** (smoke check 6 constructs the seed's
  outcome, cube dragged to a floor spot, and it scores ~0). The goal predicate reads
  the CONFIGURATION OF A MECHANISM, and the plan is a multi-stage ordered disassembly
  where every stage is a guided sliding extraction under contact constraint.
- **vs `pull_cube_i20` (beam_scale)**: that task banks mass to tip an untouched beam —
  a continuous torque-threshold mechanism with interchangeable cargo. Here the
  mechanism is a discrete INTERLOCK: binary pin-in/pin-out states, a forced order, and
  a latch (lift-then-slide) that makes one extraction a two-DoF move.
- **vs `pull_cube_tool_i186` (shelf_drop_dispatch)**: that task is a one-shot
  stored-energy RELEASE with passive gravity transport and a terminal precision place.
  Here nothing is spring-loaded or irreversible-by-energy; every stage is quasi-static,
  the order is enforced by interference geometry, and transport is the drawer carrying
  its own cargo under direct pull.

## Mechanics

NO joints — every constraint is contact geometry, all bodies plain rigid compounds. A
kinematic housing (plinth, two side rails with one square pin hole per station, roof
strips, an out-stop lintel, outboard support ledges, two catch posts) captures:

- a dynamic **drawer** (plate + open-top bay + rear guide rails with pin notches +
  handle + top lug) that slides along housing −x until its lug hits the lintel
  (travel stop 0.16 m);
- a yellow **GUARD pin** (square shaft + flat L-arm) and a green **LOCK pin** (plain
  shaft), lying across the channel through rail holes AND drawer-rail notches at
  stations x = 0.095 / 0.035 — the drawer is pinned shut (~5 mm rattle) until both
  are out;
- the red **cube** (3.5 cm), riding the drawer's 7×7 cm bay under the roof.

The interlock (all clearances asserted in `SceneCfg.__post_init__`):

1. The guard's arm **curtains the lock pin's exit hole top-to-bottom** (arm z
   0.084–0.120 vs hole 0.080–0.116; 4 mm slit ≪ 18 mm pin): the lock pin jams into it
   after ~11 mm at ANY achievable lift.
2. Two **catch posts** on the ledge dead-stop the arm after ~4 mm of +y travel at rest
   height — so shoving the lock pin, which daisy-chains +y force through the arm,
   cannot expel the guard (smoke check 9), and sliding the guard flat jams too (check
   8). The guard's hole is 36 mm tall (2× the pin): the un-latch is **lift to the hole
   ceiling** (arm clears the posts by 10 mm), slide ~4 cm raised, drop, slide free.
   The lock pin's own corridor passes BETWEEN the posts with 3 mm/side at max drift.
3. Pin extraction rides a flush hole-bottom/ledge plane (both z = 0.080); the drawer
   cannot be lifted over the pins (16 mm roof headroom < 26 mm needed).

A `post_step` force plant consumes `scene.drive` (world-frame forces at the CoM of
pin1/pin2/drawer — the push a fingertip or pinch grasp exerts, ≤ 8 N); the plant
pre-encodes world → body per substep (`quat_apply_inverse`, validated recipe with
`enable_external_forces_every_iteration`), so every caller — solve AND smoke probes —
is frame-correct by contract under the housing yaw.

## Rubric (stateless, anchored in solve.py)

- `pin_clear(k)`: all 9 sample points along pin k's shaft OUTSIDE the channel volume
  (housing frame x ∈ [−0.28, 0.13], |y| ≤ 0.0545, z ∈ [0.06, 0.145]). Seated pins are
  inside by construction; so is a pin dumped back into the channel or the open bay.
- `success()`: drawer at its out-stop (d ≥ 0.145, centered, upright, seated), cube
  riding inside the bay (drawer frame), BOTH pins clear, everything settled.
- `score()` = 0.15·clear1 + 0.15·clear2 + 0.45·clamp((d − 0.02)/0.125, 0, 1); 1.0 iff
  success. Null ~0: the locked drawer's ~5 mm rattle sits inside the 0.02 dead-zone.
  A drawer teleported open with the pins seated camps at 0.45, never success.

**Execution order is REQUIRED and physically enforced**: the drawer cannot leave the
score dead-zone while any pin is seated (notch/pin interference, smoke check 7); the
lock pin dead-ends into the guard's arm (check 9); the guard dead-ends into the catch
posts unless lifted first (check 8). No latches — order comes from interference
geometry, so the rubric cannot be gamed by state-machine tricks.

## Randomization (verified by READBACK in smoke)

Housing xy ± 3 cm and yaw ± 20°; pin seat y ± 2 mm; drawer closed-position jitter
0–2.5 mm; cube xy ± 8 mm + free yaw in the bay.

## Solution outline (solve.py, teleport = transport only)

0. reset(seed), settle — drawer closed, pins seated (not clear), cube aboard; score ~0.
1. **GUARD pin** (contact dynamics): constant 2.5 N ceiling press (hard-stop lift,
   ~15 mm — the un-latch), +y velocity servo (F = K(v_des − v), ≤ 5 N, gain-escalating
   on stall) to 4 cm while pressed, drop the press, servo flat to 15 cm → clear;
   teleport the freed resting pin to a floor depot (TRANSPORT ONLY — the in-reach
   pick-and-carry of a pin lying fully outside the mechanism) → score 0.15.
2. **LOCK pin**: same flat servo (its path is free only because the arm is gone);
   park at the depot → score 0.30.
3. **DRAWER**: −x velocity servo (≤ 8 N) on the handle line until the lug hits the
   lintel; release; drawer + riding cube settle by contact → score 1.00.
4. **PERSIST**: ≥ 3.3 s pure simulation, success at every poll → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (0 → 0.15 → 0.30 → 1.0).
Passes on forge seeds 0 and 1 (25 s each, no retries).

## Franka embodiment (base at (−0.45, 0.35), facing the mechanism)

- **Yellow guard pin**: 25 mm+ of square 1.8 cm shaft protrudes past the outer rail
  face at world y ≈ 0.15–0.16, x ≈ 0.095 ± 3 cm — about 0.55–0.62 m from the base,
  top at 10–13 cm height: a standard side pinch. The un-latch is lift ~1.5 cm then
  slide ~15 cm sideways along the open ledge — a straight in-air Cartesian stroke at
  fingertip force (≤ 5 N; extraction friction is ~0.3 N), nothing overhead blocks the
  raised stroke (the ledge and catch posts are below, open air above).
- **Green lock pin**: identical grasp geometry at x ≈ 0.035, a flat 15 cm slide
  between the catch posts (corridor 3 mm/side is the PIN's guided clearance in its
  holes — the gripper holds the protruding stub, never enters the corridor).
- **Blue drawer handle**: 6 × 4.5 cm tab at world x ≈ −0.13 → −0.29, |y| ≤ 5 cm,
  z 6.5–11 cm — 0.35–0.55 m from the base; hook or pinch and pull 16 cm along the
  rails (≤ 8 N; the rails guide, no precision needed).
- **Red cube**: never touched. At the end it sits in the open bay 0.2–0.3 m from the
  base — comfortably inspectable, deliberately not part of the motion plan.
- Nothing requires reaching past 0.65 m, more than 8 N, or entering any cavity: both
  pins are grasped OUTSIDE the mechanism and the drawer by its external handle.

## Checks

- **solve.py**: phase gates (P0 null asserts: drawer closed, pins not clear, cube
  aboard, score ≤ 0.02; P1a lift readback ≥ 12.6 mm; servo goal asserts per phase;
  P1/P2 clear asserts; success + score exactly 1.0), non-decreasing `SIM_GEN_SCORE`
  ladder, ≥ 3.3 s hands-off persistence polled every step. Passes seeds 0 and 1.
- **smoke.py**: 18 named checks — settle/no-NaN, reset state, housing randomization
  readback, jitter randomization readback, null policy, seed-strategy floor drop
  (~0), locked-drawer 8 N rattle probe (peak-sampled, non-vacuous), guard flat-slide
  latch jam, out-of-order lock-pin daisy-chain dead-stop (both pins stay in), oracle
  guard lift-unlatch extraction (0.15), oracle lock extraction with the jam force
  (0.30), near-miss partial pull, cargo-ejected full open (0.75, no success), freed
  pin dumped back into the bay (credit revoked), teleport-open cheat (camps at 0.45),
  settle gate (moving success layout rejected, then destroyed), rejection audit
  (success never True), final no-NaN — frames recorded (`frames.npz` in CWD).
  Verdict line: `SIM_GEN_SMOKE: ALL PASS 18/18`.
