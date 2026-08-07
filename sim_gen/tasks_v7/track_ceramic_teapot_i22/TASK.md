# Task `track_ceramic_teapot_i22` — Hang the Color-Matched Mugs

## Seed provenance

Derived from **`pick_place/track_ceramic_teapot`** (RoboVerse pack:
`roboverse_pack/tasks/pick_place/track_ceramic_teapot.py`). The seed: a ceramic teapot
starts already grasped; the job is to carry it along a prescribed free-space waypoint
trajectory and set it down near a plate. Its judgment is pose-tracking of the carried
object plus a place check — the terminal state is an object **resting** on a support
surface, and the whole skill is transport fidelity.

## What changed, and why it is strategically different

Here nothing is judged on a resting pose and no trajectory is prescribed. The goal
state is **suspension with identity binding**:

- A gray **rack** (kinematic compound: base plate, post, crossbar) carries two hook
  rods (Ø14 × 84 mm, tilted 15° up, each tipped with a Ø20 mm retaining knob) on the
  crossbar's front face. One hook is painted **RED**, the other **BLUE** — paint is the
  only difference.
- Three dynamic **mugs** (Ø64 × 95 mm body + open 3-bar C handle enclosing a
  34 × 65 mm aperture) stand upright on the floor in front of the rack: RED, BLUE, and
  a **GREEN decoy**. Which mug starts where is shuffled per episode, and each mug's
  handle points in a random direction.
- **success()** = the red mug **hanging** on the RED hook AND the blue mug hanging on
  the BLUE hook — per mug: the hook segment, transformed into the mug's body frame,
  crosses the handle plane INSIDE the aperture rectangle (real loop-over-rod
  containment), the mug is suspended (center above 0.16 m — above every rest surface),
  near its hook, and settled. The green decoy is judged nowhere.

| | seed | this task |
|---|---|---|
| terminal state | object resting near a goal | objects **hanging in the air** from hooks |
| judged skill | trajectory tracking while carrying | aperture-over-rod threading + weight handover |
| what is prescribed | the path | nothing — only the suspended end state |
| object identity | one teapot | 2 targets + 1 decoy, color-bound to hooks |
| geometry that matters | waypoint distance | handle-aperture containment in the mug's body frame |

The seed's entire strategy — carry the object to the goal area and set it down — is
constructed and rejected in smoke #6 (mugs settled at the rack's foot and on its base
plate → score ≤ 0.02): a set-down mug satisfies nothing here. Code structure is also
disjoint: custom compound spawners (kinematic hook rack; C-handled mugs), a body-frame
segment-through-aperture predicate, and latched per-mug progress in `post_step`.

## Randomization (all verified by readback in smoke #3/#4)

- Rack: xy ± 4 cm, yaw ± 25° around nominal 180° — hook roots and axes must be read
  from the scene, not hard-coded.
- Mug-to-slot **permutation** (which mug starts where is shuffled), per-mug xy jitter
  ± 3 cm, free yaw ± 180° (the handle direction must be read).

## Rubric

Latched every physics substep in `post_step` (credit never evaporates), targets only
(red→RED hook, blue→BLUE hook):

`score() = 0.10·lift + 0.25·correct-hook-thread` per target mug (cap 0.70), floor
**0.60** once either target has ever hung on its own hook, **0.85** once both have,
exactly **1.0** iff `success()` holds now.

- **lift**: mug center has been above `hang_z_min` (0.16 m) — carry credit.
- **thread**: the mug's aperture has been genuinely over its own color hook's rod.
- Wrong hook and the green decoy earn nothing (smoke #8/#9); doing nothing scores ~0
  (smoke #2/#5).

## Teleport solution (`solve.py`)

- **P0** — reset(seed), 60-step settle, layout readback printed (rack xy/yaw, both hook
  roots→ends, all mug poses/yaws).
- **P1 TRANSPORT (teleport, free space only)** — per target mug, one pose write carries
  it from the floor to a hover pose on its hook's axis: handle up, handle plane normal
  to the hook, aperture center **15 mm OUTSIDE** the retaining knob. Exactly the pose
  an arm reaches after pick + reorient + align; the hook is not through the handle, so
  no threading gate is satisfied (only the honest lift latch fires, +0.10).
- **P2 THREAD + HANDOVER (contact dynamics)** — a floating-hand PD controller
  (gravity-compensating position hold of the aperture center + orientation hold,
  clamped at 4 N / 0.05 N·m) slides the mug along the hook axis so rod and knob pass
  through the aperture under real clearances (~4 mm effective), then the hold is
  **ramped to zero over 2 s** so the mug's weight transfers onto the hook through
  handle-rod contact; forces cut, 4 s settle. The mug ends hanging (leaning on the rack
  is allowed). The mug is never teleported onto the hook; threading and the weight
  handover are physical.
- **P3** — repeat for the blue mug. **P4 PERSISTENCE** — ≥ 3.3 simulated seconds
  hands-off; `SIM_GEN_SOLVE: SUCCESS` only if `success()` still holds.

`SIM_GEN_SCORE` printed at each phase boundary; non-decrease asserted
(P0 0.000 → 0.100 → 0.600 → 0.600 → 1.000 → 1.000 on seeds 0 and 1). Watchdog +
`os._exit` guard the known Kit-teardown hang.

## Execution-order declaration

No ordering is imposed: either mug may be hung first (the rubric is symmetric in the
two targets, and `describe()` says so). Within one mug the ordering is physical, not
rubric-imposed: the aperture cannot be over the rod (thread credit) without the mug
having been lifted and aligned first, and the hang state cannot hold without the
weight actually resting on the hook after release.

## Embodiment argument (single Franka arm, parallel jaw, OSC)

One plausible base pose serves the whole episode: **base at (−0.35, 0, 0)** facing +x.
The rack base center drifts ± 4 cm around (0.30, 0); with nominal yaw 180° ± 25° the
hooks point back toward the robot, and all working points (mug spawns at x ≈ 0.0–0.35,
hook tips at z ≈ 0.32) lie 0.35–0.70 m from the base — inside Franka's comfortable
reach annulus.

1. **Pick a mug**: body Ø64 mm ≪ 80 mm jaw span — a top-down or slightly tilted side
   grasp of the upright cylinder anywhere the handle isn't (handle yaw is read from
   RGB/state). Mass 0.10 kg.
2. **Reorient + align**: rotate the wrist so the handle points up and its plane is
   normal to the hook axis (read from the rack pose), and bring the aperture to a
   point 15 mm outside the knob at z ≈ 0.33 — a free-space pose, exactly solve's P1.
3. **Thread**: a single straight-line ~90 mm push along the hook axis. The aperture
   clears the knob by ≈ 7 mm per side (≈ 4 mm after contact offsets) — the spherical
   knob self-funnels small lateral errors, and solve's PD (equivalent to a compliant
   OSC hold) threads it reliably. The gripper holds the mug body ≥ 60 mm below the
   rod, so the hand never approaches the hook itself.
4. **Hand over + release**: lower the commanded hold force / open the jaws; the mug
   drops ~1 cm until the rod meets the top of the aperture and swings ≤ 15° to its
   hanging equilibrium (solve ramps its hold out over 2 s — the same weight transfer).
   Retreat is free space.

Repeat for the second mug; the first hangs 19 cm away and is not disturbed (verified in
solve: red's pose is unchanged through blue's insertion). No bimanual need, no regrasp
(one grasp covers reorient + thread + release), forces ~1 N (0.98 N mug weight).

## Constructibility notes (smoke design)

- The **seed-strategy end state** (mugs carried to the goal area and set down) is
  constructible everywhere on the floor and on the rack's base plate; both variants are
  settled and rejected in smoke #6.
- A mug **perched on the crossbar top** directly above its hook is settled, elevated,
  and near — but not threaded; constructed and rejected in smoke #7.
- **Wrong-hook** and **decoy** hangs are genuinely constructible (the aperture fits
  either hook) — constructed, verified threaded by readback, and rejected (#8/#9).

## Checks

`smoke.py` runs a 13-check rejection battery: (1) settle/finite + spawn readback (all
three mugs upright at rest height, settled); (2) initial score ≤ 0.02, no success;
(3) mug randomization spread by readback over 6 seeds (xy, yaw, slot-shuffle
arrangement); (4) rack xy + yaw spread; (5) null policy 240 steps ≤ 0.02;
(6) SEED-STRATEGY end state (mugs set down at the rack) → rejected ≤ 0.02; (7) perch
on the crossbar → elevated + settled but NOT threaded → ≤ 0.15; (8) wrong hook: red
mug hung on the BLUE hook (threaded readback true) → ≤ 0.20, NOT success; (9) wrong
object: GREEN decoy hung on the RED hook → ≤ 0.02; (10) one-mug partial: red hung on
RED only → NOT success, exactly the 0.60 floor; (11) latched credit: hung mug
teleported back to the floor → score unchanged; (12) `ever_success` audit False across
all constructed non-successes; (13) final no-NaN. Records `frames.npz` (RGB) in CWD.
Prints `SIM_GEN_SMOKE: ALL PASS 13/13`.

## Run

```
python -m simgen_tasks.track_ceramic_teapot_i22.solve --headless [--seed N]
python -m simgen_tasks.track_ceramic_teapot_i22.smoke --headless
```
