# pick_and_lift_small_i350 — pin-latched spring dumbwaiter

Scene: `pin_latch_dumbwaiter` · Env: `simgen.pin_latch_dumbwaiter` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed task: **`rlbench/pick_and_lift_small`** — "pick up the \<shape\> and lift it up to
the target": a Franka grasps a small cube among shape distractors and raises it to a
floating red sphere marker. Kept from the seed: a **small target cube must end up
HIGH**, and it must be the **right** cube — a same-size red decoy preserves the
seed's identity-discrimination pressure. Everything about *how* the cube gets high
is replaced.

## The task

A 41 cm dumbwaiter tower stands on a base plate: a vertical shaft whose lower front
is an open loading window and whose upper section is a sealed **penthouse** (front
wall + roof; every gap around the goal volume is smaller than the cube). Inside
rides a yellow open-top elevator **car** on a vertical slide (310 mm stroke), pushed
up by a lift spring stronger than car + cargo. At spawn the car idles low, pressed
up against a slick steel **latch pin** that runs through slots in both side walls;
the pin's black pinch knob protrudes on a per-episode random side. A blue cube and a
red decoy lie on the ground in front of the window.

Goal: the blue cube delivered to the penthouse — car resting at the top stop with
the blue cube contained in its cargo box, the decoy not, everything settled.

1. **LOAD** — drop the blue cube through the window into the car's open top (a
   straight drop corridor past the pin is asserted in cfg);
2. **RELEASE** — pinch the knob and pull the pin axially out of both slots, against
   the sliding friction of the spring preload (~1.4 N, finger-scale);
3. **RIDE** — hands off: the spring hoists car + cube into the penthouse and holds
   them pressed on the top stop (the delivered state is self-holding).

## REQUIRED execution order (declared)

**Load BEFORE release — the wrong order is irreversible.** Pulling the pin while
the car is empty sends the empty car up into the sealed penthouse, where the spring
(stronger than the empty car's weight, asserted) holds it and nothing can reach it:
the cube can then never be loaded and the episode is unsolvable. The rubric's
`pin_out` and `risen` latches only fire with the cube aboard an already-loaded car,
so the doomed empty release also earns zero credit (smoke check 8 demonstrates the
trap physically).

## Why strategically different

- **vs the seed** (`pick_and_lift_small`): the seed is grasp-and-raise to a floating
  height marker — pure transport by the arm. Here the robot *cannot* raise the cube
  to the goal: the penthouse geometry (asserted) denies every direct path (window
  ends 110 mm below the risen cargo, roof-to-rim gap 8 mm ≪ 40 mm cube). The lift
  is performed by a stored-energy machine; the robot's contributions are loading
  cargo into a latched carrier and extracting a friction-held latch pin — plus an
  irreversible ordering constraint the seed does not have.
- **vs the corpus**: no other tasks_v7 task is a spring-driven captive elevator
  gated by an extractable pin. The nearest mechanism families differ in kind:
  `handover_i339`-style courier/shuttle tasks move cargo *laterally* with the robot
  driving the carrier; counterweight/lever tasks (e.g. ballast-lever i16) transmit
  robot-supplied energy through a mechanism *while the robot acts*; hook-den
  retrieval (i55) and bend-gallery extraction (i76) are about getting an object
  *out* through constrained geometry. Here the delivery stroke is pure stored
  spring energy released by removing a mechanical interlock, the goal volume is
  sealed against every non-machine path, and the wrong action order permanently
  destroys solvability.

## Franka embodiment (single arm, parallel-jaw gripper, OSC)

Base pose: on the ground at world ≈ **(0.95, 0.0)**, facing −x toward the tower's
loading window (tower origin at (0.32, 0.0) ± 3 cm, yaw ± 20°) — every manipuland
is then within a 0.25–0.65 m reach envelope at heights ≤ 0.42 m.

- **Blue cube / red decoy** (40 mm, 90 g): on open ground, top-down or angled
  parallel-jaw grasp; carried over the window sill and released ≥ 15 cm above the
  car mouth — the asserted drop corridor (62 mm wide vs the 40 mm cube) makes the
  drop tolerant. The solve's hover-release at zero velocity is exactly what a
  gripper release reproduces.
- **Latch pin** (12 mm square rod, 50 g): its 24×20×24 mm knob protrudes ≥ 60 mm
  outside the tower wall at height ≈ 0.10 m — a canonical parallel-jaw pinch. The
  extraction is a straight axial pull of ~1.4 N (escalating cap in the solve tops
  at 12 N), well inside Franka payload/force limits; the pull axis is horizontal
  and unobstructed on the knob side.
- **Car / tower**: never need to be touched; the ride is hands-off.

All interaction heights are ≤ 0.42 m and forces ≤ 12 N: comfortably a
single-Franka task.

## Rubric (latched, anchored in the demonstrated solve)

- 0.20 `loaded` — cube ever contained in the car while the car is low (pinned band)
- 0.25 `pin_out` — pin fully clear of both walls while loaded & cube aboard
- 0.20 `risen` — cube aboard past half stroke
- non-success cap 0.70; **1.0 iff `success()`**: car at top stop (q ≥ stroke − 2 cm),
  blue cube in the car-frame cargo box, decoy not, tower upright, all settled+finite.

Solve prints `SIM_GEN_SCORE` 0.00 → 0.20 → 0.45 → 0.65 → 1.00 (seeds 0/1/2 all
`SIM_GEN_SOLVE: SUCCESS`, with a 3.3 s hands-off persistence window).

## Checks

- `scene.py` `__post_init__` asserts (~25): pin-gate geometry (pinned car floats at
  q_pin = 12 mm, pin over the rims, above cargo head-room), drop corridor, car/shaft
  clearances, penthouse denial (window, roof gap), spring margins (loaded ride, empty
  trap, no top-slam cargo pop), extraction force band, slot play, rubric band
  separations, spawn lane feasibility.
- `smoke.py` battery: **14 checks** — settle/no-NaN + layout sanity; score ~0 at
  reset; randomization readback over 10 seeds (tower pose; cube lane + side swap +
  pin handle flip); null policy; seed-strategy rejection (cube hoisted to goal
  height earns nothing); wrong-object rejection (decoy riding at the top); doomed
  empty-release demonstration (+ roof-sealed re-load denial); loaded partial credit
  ≈ 0.20 only; physical pin-gate probe (2× preload extra push does not pass the
  pin, paired non-vacuity with the free ascent); latched credit survives removal;
  settle gate on the exact goal geometry in motion; rejection audit (success never
  True); final no-NaN. Records `frames.npz`.
