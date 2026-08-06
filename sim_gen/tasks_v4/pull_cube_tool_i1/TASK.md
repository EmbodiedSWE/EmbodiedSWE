# carousel_ferry (`pull_cube_tool_i1`)

**Seed:** `maniskill/pull_cube_tool`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/pull_cube_tool.py`) — an L-shaped
tool lies within reach; the solver grasps it, hooks it behind an out-of-reach cube, and
DRAGS the cube inward until it is within `reach_distance` of the arm base.

**Scene:** `simgen.carousel_ferry` (robot="null", scene-level; solve.py and smoke.py
build this same env).

## What changed

| | seed | this task |
|---|---|---|
| means of reaching the far object | a portable hand-held tool extends the arm | an ANCHORED mechanism (free-spinning carousel on a fixed axle) carries the object |
| interaction | one grasp + one quasi-static drag | cyclic capstan work: push a peg through the near quadrant, release, re-engage the next peg (~155 deg of rotation), then a terminal pick-and-place |
| goal | proximity disc around the base | cube RESTING INSIDE a walled tray with a green floor mat (position sampled per episode) |
| verification | cube xy near base | physical ferry latch (signed carried rotation while riding) + settled containment |

Geometry: platter r=0.34 m on a pedestal (top at 0.114 m), axle at (0.68, 0); four brass
pegs (r=16 mm, h=11 cm) at radius 0.26 every 90 deg; red 4.5 cm cube spawns on the FAR
rim (azimuth ±25 deg about +x, radius 0.23–0.27, ~1.0 m from the base — beyond reach);
tray (15 cm inner square, 5 cm walls, green mat) near the base at (0.22, −0.30) ± 4 cm.
Randomized per episode (verified by readback): cube azimuth/radius/yaw, platter yaw
(peg pattern), tray position.

## Why strategically different

The seed's plan is *tool-mediated reach extension*: fetch a portable implement, hook,
drag inward — the tool moves with the hand and the cube's whole trajectory is under
direct quasi-static control. Here there is NO portable tool, and no contact the arm can
make moves the cube directly (it spawns ~1.0 m out; smoke's seed-strategy control parks
the cube near the base and scores 0). The solver must (1) actuate a fixed rotary
mechanism through CYCLIC re-engagement — a peg can only be worked through the near
quadrant, so the ~155 deg ferry takes several push-release-re-engage strokes on
successive pegs, (2) let environmental friction carry the cargo (the cube rides the
platter; the arm never touches it until it arrives), and (3) finish with a precision
pick off the moving fixture and a place into a walled, per-episode-located tray.
Skills: mechanism actuation + handoff pick-and-place vs the seed's reach-extension
dragging. A different plan and code structure, not different numbers.

## Solution (solve.py — the teleport-solution legitimacy certificate)

Teleports are TRANSPORT ONLY; both load-bearing interactions run through contact
dynamics:

1. **FERRY (dynamics):** closed-loop torque on the platter's free axle via the scene's
   `platter_drive` plant buffer (|τ| ≤ 1.2 N·m — what a ~5 N fingertip push on a peg at
   the 0.26 m circle exerts), PD on the cube's azimuth. The cube is carried around by
   real platter-top friction; the rubric's `ferried` latch fires only from this carried
   transport. → `SIM_GEN_SCORE` 0.30.
2. **DELIVER (transport teleport + dynamics):** the cube is teleported across free
   space to hover 2.5 cm ABOVE the open tray mouth (never seated), then falls, impacts
   the mat, and settles between the walls under gravity/contact. → success,
   `SIM_GEN_SCORE` 1.00.
3. **PERSIST:** ≥ 3.5 s of pure simulation, success re-verified at every poll →
   `SIM_GEN_SOLVE: SUCCESS`.

Scores print at every phase boundary and are non-decreasing (0 → 0.30 → 1.00 → 1.00).
Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka, parallel jaw, OSC)

Plausible base pose: **(−0.10, 0, 0)** on the floor, facing +x (recorded in
`cfg.base_pos`; all numbers below relative to it).

- **Pegs (the only objects pushed):** vertical 32 mm cylinders, tops at 0.23 m height —
  a comfortable side-push or wrap-grasp for the jaw with open approach from above; a
  peg transiting the near quadrant is 0.35–0.62 m from the base (inside the 0.45–0.71 m
  comfortable envelope). Working stroke: engage the peg nearest the near direction,
  sweep ~45–90 deg, release, wait for the next peg (90 deg pitch guarantees one is
  always entering the quadrant). Required precision: none beyond touching a 32 mm
  cylinder — the rubric never scores the pegs.
- **Cube (picked once, after the ferry):** 4.5 cm cube at 0.42–0.53 m from the base,
  top face at 0.137 m — a standard top-down pinch (jaw opens to 8 cm), same class as
  the chute_dispatch pick that ran 4/4 on this rig. The platter is settled (axle
  viscous friction stops it in ~0.5 s) before the pick.
- **Tray place:** open-top 15 cm square at ~0.44 m from the base; release the cube
  centred 3–5 cm above the wall tops (walls only 5 cm high, fingers never enter the
  tray). Tolerance ±5 cm — far above OSC noise.
- Clearances: cube-to-nearest-peg arc distance ≥ 0.20 m at spawn (pegs authored 45 deg
  off the cube); nothing required within 2 cm of the ground except the tray release,
  which happens from above.

## Execution order

Physically forced, not rubric-declared: the ferry must precede the pick (the cube is
unreachable until carried around); no ordering clause exists in the rubric beyond the
`ferried`-before-placement latch chain that this implies.

## Rubric

- `success()`: `ferried` latch AND the cube currently rests settled inside the tray
  (per-axis |Δxy| < 5 cm of the mat centre, z within 12 mm of rest height,
  |v| < 0.04 m/s, |ω| < 0.6 rad/s).
- `ferried` (latch): cube riding the platter (z band, inside rim, platter-frame slip
  < 8 mm/substep, platter rate < 3 deg/substep) inside the near sector (±55 deg of the
  base-facing direction) with |signed carried rotation| ≥ 80 deg (spawn geometry forces
  ≥ ~100 deg). Kinematic teleports never accumulate carried rotation; back-and-forth
  rocking cancels in the SIGNED sum.
- `score()` = 0.3·ferry_prog (latched |net sweep|/100 deg, forced 1.0 on `ferried`) +
  0.2·placed (latched: cube enters the tray footprint below wall top after the ferry) +
  0.5·success. Exactly 1.0 iff success; null ≈ 0; a knock-out after success falls back
  to 0.5 (latched credit does not evaporate).

## Checks (smoke.py — rubric rejection battery, 15 named checks)

1. clean reset: finite, cube riding, score ~0
2. randomization by readback (cube spawn, platter yaw, tray centre differ across seeds)
3. cube spawns > 0.90 m from the documented base (every seed)
4. null policy 2 s → score ~0, no success
5. seed-strategy control: cube parked near the base on open floor → rejected, score ~0
6. anti-cheat: teleport straight into the tray, verified physically settled inside →
   success False, score ~0 (no ferry latch)
7. anti-cheat: rocking the platter (real riding, no net transport) → no latch,
   score < 0.15
8. oracle ferry: torque-driven carried rotation → ferried, score ~0.30
9. score monotone (never decreases) along the ferry
10. axle anchored: platter centre moves < 5 mm under drive
11. near-miss: settled just outside the tray wall after a real ferry → ferry credit
    only, no success
12. exactness: ferry + physical drop → success and score == 1.0 exactly (ladder
    0 → 0.30 → 1.00 monotone)
13. persistence: success holds 2 further seconds, no flicker
14. achievement latch: knock-out revokes success, score falls to exactly 0.50
15. video frames recorded (frames.npz in CWD)

The seed's end state is expressible and tested (check 5); the seed's tool-wielding half
is N/A by construction (no portable tool exists in this scene).
