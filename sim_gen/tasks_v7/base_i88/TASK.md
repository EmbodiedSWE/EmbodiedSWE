# base_i88 — BallastLiftScene (`simgen.ballast_lift`)

Tip a see-saw with ballast cubes to raise an ungraspable ball to shelf height, then
push the ball sideways off the raised tray into a walled dock on the shelf top.

## Seed provenance

Derived from **`pick_place/base`**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/base.py`): a direct
grasp-and-carry task — close the parallel jaw on a 4 cm cube and translate it through
free-space waypoints to a goal pose. The seed's core loop is *one grasp, one guided
transport*, and its judging is waypoint/goal-pose tracking on the held object.

## Strategic difference

**vs the seed (`pick_place/base`)** — different plan AND different code structure:

- *Plan*: the cargo here is a **90 mm ball that cannot be grasped** (wider than the
  Franka's 80 mm jaw span) and the goal is a shelf top with **no path up from below**.
  The solver must actuate **indirectly**: accumulate mass (three loose 40 mm cubes,
  dropped one at a time into a chimney hopper) on the far end of a hinged see-saw
  until the counterweight out-moments the ball and the beam tips, raising the tray to
  shelf height; only then push the ball laterally — along the hinge axis, which stays
  level at every beam angle — over the tray's 12 mm lip, across a 10 mm gap, into the
  dock. The target object is **never held**; the pick-and-place motion is applied to
  the *ballast*, whose placement target (the hopper mouth) *moves* as the beam tips.
- *Code structure*: judging is a **mechanism-state predicate stack** (beam tilt angle
  from the hinge quaternion, per-cube hopper containment in the beam's local frame,
  a dock containment window + settle gate on the ball) with **latched partial credit**
  (`torch.maximum` in `post_step` — credit never evaporates), not gripper-distance /
  waypoint tracking. The physics legitimacy is *asserted from the cfg's own numbers*
  in `__post_init__` (moment ledger: 1 cube can NEVER tip, 3 cubes ALWAYS tip), and
  the beam is authored with MassAPI CoM at the hinge so the plank contributes zero
  gravity moment and the ballast-vs-ball ledger is the whole story.

**vs `screw_nail_i59`** (the other corpus package read during construction): that task
is a *disassembly* chain — unscrew/withdraw fasteners, remove a lid, extract a dish —
i.e. sequenced removal of constraints on pre-assembled bodies. base_i88 is the
opposite direction: *accumulation* (adding ballast to a lever mechanism) followed by a
push-only lateral transfer of a never-held object. No shared mechanism, predicate
style, or solve primitive beyond the house conventions.

## Scene summary

Fully procedural (compound-box spawners + sphere; no external assets), env frame,
ground z = 0:

- **See-saw**: kinematic pylon at (0.45, 0), authored per-env `UsdPhysics.RevoluteJoint`
  (axis X, limits ±15°, hinge height 0.15 m), beam plank 0.13×0.56×0.032 m with a
  fenced **tray** (12 mm lip) on the +y cargo end and a chimney **hopper**
  (62×62 mm cavity, walls to local z 0.156) on the −y end. Beam mass 1.0 kg, CoM at
  the hinge.
- **Ball**: r = 45 mm, 0.25 kg, spawns resting in the tray (beam starts cargo-end-down
  on its lower stop).
- **Ballast**: three 40 mm / 0.15 kg cubes scattered on the ground at permuted slots.
- **Shelf**: pedestal at (0.60, 0.17), top 0.225 m, near face 10 mm from the raised
  beam edge; **dock** = far wall + two side walls + a low **sill ridge** on the open
  side (blocks the sphere-on-box creep artifact; the push servo crosses it easily).
- **Randomization** (readback-verified in smoke): cube slots permuted per env with
  ±25 mm xy jitter and free yaw; ball ±18 mm jitter along the hinge axis.

`success()`: ball center inside the dock window x∈[0.545,0.645], y∈[0.115,0.225],
z∈[0.245,0.300] AND settled (|v| ≤ 0.05). Identity enforced by name — a cube in the
dock counts for nothing. `score()`: latched 0.10/cube ever in hopper + 0.25 beam ever
raised + 0.10 ball ever crossed onto the shelf (cap 0.65) + 1.0 iff success.

## Solution phases (solve.py — teleport = transport only)

- **P0** settle + readback; assert beam starts lowered, score ≤ 0.02.
- **P1a–c** for each cube: teleport it to **free space** 55 mm above the hopper
  chimney's open mouth (computed from the beam's *current* pose — the mouth moves),
  release at zero velocity, and let **gravity + chimney-wall contact** make the
  insertion. Assert containment after settling.
- **P1d** hands-off: the accumulated ballast out-moments the ball and the see-saw
  tips (pure hinge dynamics; empirically the beam tips after cube 2). Assert raised.
- **P2** push the ball (never teleported at any point) with a high-gain **clamped
  velocity servo** — f = 60·(0.11 − vₓ) clamped to [0, 4] N, re-set every physics
  step in the ball's current link frame: stall force = the 4 N clamp > the ~2.3 N
  the 12 mm lip demands, while cruise speed stays low (a constant 4 N would launch
  the ball over the dock walls; a low-gain servo would stall at the lip). Over the
  lip, across the gap, into the dock, then release.
- **P3** hands-off settle to `success()`.
- **P4** persistence: ≥ 3.3 simulated seconds hands-off with success still true.

`SIM_GEN_SCORE` printed at every phase boundary (observed staircase, seeds 0 and 1:
0 → 0.10 → 0.45 → 0.55 → 0.65 → 1.0, non-decreasing), then `SIM_GEN_SOLVE: SUCCESS`.
Passes on seeds 0 and 1.

## Franka embodiment argument

Single Franka arm, parallel jaw, OSC. Base pose ≈ **(0.0, −0.05, 0)** facing +x: every
manipulation point is then within ~0.62 m reach with margin.

- **Ballast cubes** (the only grasped objects): 40 mm edges — dead-center of the
  parallel jaw's span. Ground slots lie at radius 0.20–0.36 m from the base; standard
  top grasp. The drop target is the hopper mouth, whose center sits at
  (0.45, −0.28 ± 0.03) and height ≈ 0.41 m beam-down (lower as the beam tips) —
  radius ≤ 0.51 m from the base, a free-space hover-and-release directly above an
  open chimney; the chimney's funnel geometry tolerates centimeter-level drop error
  (exactly what solve's gravity-drop primitive demonstrates).
- **Ball**: 90 mm diameter — *ungraspable by design* (jaw span 80 mm). The required
  interaction is a lateral push at the ball's equator height ≈ 0.26 m (beam raised)
  along +x with the jaw closed — a knuckle/fingertip push at radius ≈ 0.55–0.65 m,
  inside reach, with the approach corridor above the beam free of obstructions. The
  ~2.3 N lip force and ~4 N ceiling are far below the arm's capability, and the dock
  fences forgive lateral aim error.
- **Ordering is mechanically forced**: pushing first (beam down) wedges the ball
  against the tilted lip/fence corner below shelf height (smoke check 6 demonstrates
  it — the gate probe drives the ball with the same servo and it never reaches the
  shelf); the tray can only be emptied after the hopper is ballasted.

**Execution order declaration**: cube → hopper (×3, any cube order) strictly before
ball push; within P1 the cubes are interchangeable; P2–P4 are a single forced
sequence.

## Checks (smoke.py — 15)

1. Settle + no-NaN, 2. null score ≈ 0 at rest, 3. randomization readback (slot
permutations, xy spread, yaw spread across seeds), 4. ball-jitter readback, 5. null
policy 240 steps scores ≤ 0.02, 6. gate probe — the solve's own push servo with the
beam DOWN never gets the ball to the shelf (non-vacuity asserted: peak speed, lip
climb, displacement), 7. one cube can NEVER tip the beam (settled probe), 8. three
cubes ALWAYS tip it (and partial score lands in [0.50, 0.66] without success),
9. raised-tray ball pose rejected (x window), 10. ball on the floor beside the
pedestal rejected (z window), 11. identity: a cube settled in the dock is not
success, 12. latched credit survives removing the cubes, 13. settle gate: a moving
ball inside the window is rejected, 14. rejection audit (success never true during
any rejection construction), 15. final no-NaN. Camera frames recorded to
`frames.npz`. Prints `SIM_GEN_SMOKE: ALL PASS 15/15`.
