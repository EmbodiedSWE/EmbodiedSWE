# approach_grasp_banana_i372 — Crank Ejector (scotch-yoke ram magazine, side decision, foul trap)

**Scene** `crank_ejector` / env `simgen.crank_ejector` (NullRobot, scene-level physics).
**Seed** `pick_place/approach_grasp_banana` (RoboVerse
`roboverse_pack/tasks/pick_place/approach_grasp_banana.py`).

## Provenance and strategic difference

The seed is a **prehensile pick-and-place**: a Franka approaches a banana lying in the
open, closes its jaw around it, carries it, and releases it into a basket; success is a
gripper-object relation (approach + grasp + transport + release), and the judged object
is exactly the thing held.

This task inverts every element of that plan:

- **The judged payload can never be grasped.** The red cargo cube (42 mm) rests deep
  inside a roofed magazine tunnel whose mouths leave only **6 mm of side clearance**
  (< a ~7 mm Franka fingertip — no jaw can straddle the cube) and whose roof slot is
  **30 mm wide** (< the cube — it cannot leave upward; verified live by a 2×-weight
  lift probe). There is no approach-grasp, no carry, no release.
- **Transport is machine-mediated through a rotary-to-linear mechanism.** A side-mounted
  crank wheel (revolute, ±92° stops) carries an inner drive peg caged by the tall fork
  of a double-ended ram (prismatic, along the tunnel) — a **scotch yoke**
  (`x_ram = R·sinθ`, R = 0.13 m). The only intended handle is the crank's outer
  **yellow handle peg**; the robot's hand turns the crank, the peg-fork contact drives
  the ram, the ram plows the cube, and gravity drops it over the lip into a sunk catch
  pocket. The hand never does the transport — the mechanism does.
- **A per-episode direction decision.** Which end holds the red cargo (vs the grey
  decoy) is a coin flip each episode; the solver must read the side and crank the
  matching direction.
- **An irreversible failure mode.** Cranking the wrong way ejects the grey decoy into
  the opposite pocket. Nothing can bring a pocketed cube back (it cannot be grasped and
  the ram cannot pull), so the rubric latches a permanent foul that caps the score at
  0.20 — even a subsequent real, correct delivery of the cargo stays capped (verified
  in smoke).

Distinct from the seed and from every corpus task read while building it:
`approach_grasp_banana_i69` (this seed's sibling) is a spring-centered 2-DOF
**gravity-steering tilt maze** — here the plant is a 1-DOF crank with a kinematic
rotary→linear conversion and no spring; the pen-holder exemplar is direct prehensile
insertion; the scotch-yoke ferry (`native_libero_i365`) carries cargo INSIDE a moving
cage through a sealed tunnel loaded at dead center — here the cubes are never inside
any carrier: a double-ended ram plows free cubes OUT of a static magazine, with a
binary direction decision and a decoy-foul trap that i365 has nothing like.

## Geometry (env-local frame, rig plan center at the origin; tunnel along x)

Plinth + spine carry a roofed tunnel: floor top 0.100 m, interior 54 mm wide × 50 mm
tall, mouths at x = ±0.150; roof slot |y| < 15 mm for the ram's fork. Below each mouth
a sunk teal catch pocket: interior x ∈ ±[0.145, 0.265], half-width 45 mm, floor top
0.030 m, wall tops 0.090 m (below the drop lip — no perch rim on the ejecta path). The
blue ram (110 × 42 × 46 mm body, two fork plates rising through the slot, 15 mm inner
gap) rides a frictionless prismatic joint; the crank (axis at y = 0.053, z = 0.200)
carries a **round** 12 mm drive pin at R = 0.13 between the fork plates (a square pin
wedges the fork at θ ≈ 20.5° — found and fixed on the forge) and the yellow handle peg
at r = 0.10 on the outboard face. Joint pairs are collision-filtered; crank↔ram couple
ONLY through the real peg-fork contact. All colliders carry explicit friction
materials; ram and crank have authored mass/CoM/inertia (crank CoM at the axis —
gravity-neutral). ~20 honesty asserts in `CrankEjectorSceneCfg.__post_init__`: mouth
clearance beats the fingertip, slot beats the cube, full-stroke CoM carry ≥ 45 mm past
the lip, peg clears roof and bearing block over the whole sweep, fork backlash 2–4 mm
and full-travel span, spawn bands off the walls and ≥ 30 mm inside the mouths, pocket
seat z band separates floor rest from wall-top perch, pocket walls below the lip,
drive-torque margin ≥ 5× the worst resisting load, one free-fall substep far under the
continuity gate, handle sweep inside the documented reach, weights sum to 1.

## Rubric (latched, continuity-gated)

Object latches require per-substep cargo travel < 3 cm (`cont_max`), ram latches
< 2 cm — teleports earn nothing. `prog` (0.20, latched max): correct-direction ram
displacement, clamp(side·x_ram/0.075, 0, 1). `ejected` (0.25): cargo beyond
side·x > 0.155 and below z 0.095 (off the floor, over the lip) with prog > 0.92 — the
crank-driven path is required. `seated` (0.25): ejected AND inside the goal pocket
(x/y window + z band 0.036–0.080 that accepts a floor rest and rejects a wall-top
perch); seating snaps prog → 1. `fouled` (permanent): the decoy beyond its mouth or
below the lip caps the score at 0.20. `success` (0.30, live): seated AND not fouled
AND cargo currently settled in the pocket AND decoy still housed in the tunnel AND the
mechanism quiet (cargo < 0.10 m/s, |crank rate| < 0.6 rad/s, |ram vx| < 0.05 m/s).
Score = exactly 1.0 iff success(); null ~0; disturbing the crank after success falls
back to the latched 0.70, never lower.

## Solution (solve.py) — no teleport of any object, ever

Everything flows through the scene's `crank_drive` plant buffer (clamped ±0.60 N·m —
the wrench of a hand on the yellow handle; at the 0.10 m handle arm that is ≤ 6 N of
fingertip force):

0. reset(seed), settle; assert both cubes rest in the tunnel → `SIM_GEN_SCORE` ~0
1. CRANK: read the episode's side, velocity-servo the crank toward that side's stop
   (τ = side·ff + k(side·ω_des − ω_FD), speed guard, stall-escalating ff/k); the peg
   drives the fork, the ram plows the cargo over the drop lip → `ejected`, ~0.45
2. FINISH: keep the stroke until the cargo lands `seated` in the pocket → 0.70
3. RELEASE: zero drive; crank settles at the stop, cargo rests, decoy never moved →
   success, 1.00
4. PERSIST: 3.5 s untouched, success at every poll → `SIM_GEN_SOLVE: SUCCESS`

Verified on the forge: seeds 0 (side = east) and 1 (side = west), both `rc=0`,
score ladder 0 → 0.45 → 0.70 → 1.00, non-decreasing, ~18.5 s each.

## Execution order (declared)

The crank **direction** is the episode's only decision point, and it is order-critical
in the strong sense: the first full stroke in the wrong direction ejects the decoy,
which latches a permanent foul that no later correct stroke can repair (by design and
by measured physics — smoke drives the wrong stroke for real, then a real correct
stroke that genuinely seats the cargo, and the score stays at 0.20 with success False).
Within the correct direction the order is forced by the mechanism: crank → eject →
seat → release.

## Embodiment argument (Franka)

Documented base at (0, 0.42), facing the crank side of the rig. The yellow handle peg
sweeps a circle of radius 0.10 m about the axle at (0, 0.053, 0.200); the farthest
point of the sweep is 0.478 m from the base — well inside the 0.72 m envelope, at
comfortable manipulation heights (0.10–0.30 m). Driving needs ≤ 0.60 N·m about the
axle = ≤ 6 N at the handle — light fingertip work for a hand curling around a
13 × 45 × 13 mm peg. The cubes themselves are unreachable as grasp targets: 6 mm mouth
side clearance beats any fingertip, the 30 mm roof slot beats the 42 mm cube, and the
pockets are open — reading the cargo side (red vs grey through either mouth or the
slot) is a perception task, not a reach task. One plausible strategy: look through a
mouth, grip the yellow peg, crank hand-over-hand toward the red end until the cube
drops, let go.

## Checks

`smoke.py` — one linear recorded run, **15/15 PASS** on the forge: settle/finite,
randomization readback (side flips across 8 seeds, crank angle/gap poses vary, cube
positions match the side), null ~0, seed-strategy direct transport rejected (cargo
force-dragged into the pocket without the crank turning — physically delivered, score
~0), teleport-into-pocket rejected, wrong-direction real crank fouls (decoy ejected,
cap 0.20), foul irreversibility (real correct stroke seats the cargo, score stays
capped), slot lift-out denied (2×-weight lift moves the cube but the roof contains
it), oracle crank (ejected + seated, 0.70 held live), monotone trace, exactness
(release → success, score == 1.0), persistence, disturb/recover latch (moving crank
revokes success, 0.70 floor, brake restores), seat z band rejects a perch-height
cargo, ≥ 20 video frames saved (`frames.npz`, 280 frames).
