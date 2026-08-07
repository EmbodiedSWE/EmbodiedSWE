# latch_canister (close_jar_i54)

Seal the RED ball inside a square canister: drop it through the open mouth, lay
the square lid into the lip-ring recess, then swing the two orange turn-tabs a
quarter turn inward so their bars cross over the lid and hold it down. The BLUE
ball must stay outside.

## Seed provenance

Seed: `rlbench/close_jar` — pick up the lid and screw it onto the jar of the
target colour.

What was kept from the seed:

- a container with an open mouth, a separate lid, and a colour-cued
  target-vs-distractor choice;
- "the jar is closed" as the headline outcome.

What was changed:

- **The lid alone closes nothing.** In the seed, placing/screwing the lid IS
  the task. Here the canister must first be *loaded* (red ball dropped through
  the mouth), the lid must then be *seated* into a recess, and finally two
  independent **turn-tabs** (80×16×8 mm bars on vertical-axis revolute joints
  at the east/west posts) must each be swung ~90° inward so the bars physically
  overhang the lid. Success is a 4-clause conjunction with a mandatory order.
- **The colour cue moved from the container to the contents**: the choice is
  which ball goes in (red, not blue), not which jar to close.
- **Screwing (helical, one body) became latching (two extra articulated
  bodies)** — the closure state is carried by joint angles of parts of the
  fixture itself, not by the lid's pose alone.

## Strategic difference vs seed and corpus

- vs the seed (`close_jar`): the seed's own winning strategy — put the lid on
  the container — scores **0** here (smoke check 5): credit is order-coupled,
  a lid on an *empty* canister earns nothing, and the tabs must still be
  turned. The seed has no loading step, no recess seating, and no fixture
  articulation.
- vs `close_box_i26` (trap-crate): i26's closure is a *falling lid* driven by
  removing a prop; here the lid is a free body that must be *placed* into a
  recess and the closure is completed by *turning two latches* — the moving
  DOFs are on the fixture, not the lid, and there are two of them.
- vs `pen_holder` / drop-in tasks (`i4` drop-gate): dropping the object in is
  only worth 0.25 here; the task's centre of mass is the seat-then-latch
  sequence after the drop.
- vs `i5` (bayonet twist): i5 twists the *lid itself* about the vertical axis;
  here the lid never rotates — instead two *separate* fixture-mounted bars
  rotate over it, and each is an independent joint with its own randomized
  start angle.
- vs `i33` (flap chute) / other hinge tasks: those hinges are horizontal-axis
  gravity-loaded flaps; these are **vertical-axis** turn-tabs (gravity exerts
  no torque about the axis — parking is by angular damping, and the tab can
  rest at any angle), swung by a lateral push rather than lifted.

Mechanical interlocks make the order load-bearing (all probe-verified in
smoke): tabs turned first leave the dead-centre lid resting ON the bars at
z≈113.5 mm, ~15 mm above the 88.5–99.5 mm seat window (check 7); a mis-seated
(45°-yawed) lid perches on the lip rails and physically blocks the tab's swing
short of the −70° locked window (check 9); and the seated lid seals the mouth
so a late ball cannot enter (check 12).

## Scene

Procedural geometry only (UsdGeom cubes/spheres under compound rigid roots):

- **canister** (12 kg dynamic compound, origin at footprint centre on the
  ground): 150×150×10 mm floor, walls to z 90 mm (interior 134×134), seat
  ledges (top z 90, reaching in to x/y 55 mm), lip ring (inner ±75 mm, top
  z 100 mm; full rails on ±y, corner stubs on ±x leaving 60 mm swing gaps),
  and two posts at x=±98 mm carrying the tab pivots at z 105.5 mm.
- **tab_e / tab_w**: 80×16×8 mm bars, one vertical-axis revolute joint each
  (limits −95°…+15°; θ=0 open/outboard, θ=−90 locked/inboard). Held by strong
  angular damping (4.0 /s) — PhysX `jointFriction` is inert on
  non-articulation joints (warning verified in the log).
- **lid**: 140×140×8 mm plate + 22 mm knob; seats at canister-frame z 94 mm.
- **red / blue balls**: r 18 mm, in two start slots (which ball is where is
  randomized).

Randomization (seed-dependent, verified by readback in smoke check 2): canister
xy + yaw, both tab start angles (−18°…+8°), ball slot assignment + jitter, lid
start position around the canister.

## Rubric

`success()` (all live): `ball_in & lid_seated & tabs_locked.all & decoy_out &
still & finite`, where

- `lid_seated`: lid centre within 12 mm xy / 88.5–99.5 mm z of the seat, tilt
  < 6°, yaw within 10° of a square-symmetry alignment;
- `tabs_locked` (per tab): θ ∈ (−115°, −70°) **and** the tab tip physically
  over the lid footprint at bar height (pose-judged, so a blocked or merely
  commanded tab does not count);
- `still`: 60-consecutive-substep stillness counter (not instantaneous).

`score()`: latched, order-coupled — 0.25·ball_ever + 0.20·lid_ever (needs ball
first) + 0.15·tab_ever (needs lid first), capped at 0.60; exactly 1.0 iff
`success()` holds live. Latches are included in get_state/set_state.

## Solution outline (solve.py)

Teleports for TRANSPORT ONLY; every load-bearing interaction is contact
dynamics or an applied wrench:

1. settle + layout asserts (score ≤ 0.03, no success);
2. red ball teleported to a hover 150 mm above the canister floor (above the
   cavity ceiling) → **gravity drop** through the open mouth → 0.25;
3. lid teleported to a yaw-aligned hover at canister z 112 mm (above the seat
   window) → **gravity seats** it into the recess → 0.45;
4. each tab driven by a velocity-regulated **pure world-z torque** (frame-drag
   immune) — gain 0.004 sized for stability ((k/I)·dt < 2), cap 0.02 N·m
   escalating on stall, released at θ ≤ −80°, damping parks it ~−94.5°;
   both locked → success, score 1.0;
5. persistence: ≥3 s (400 steps) fully hands-off, success must hold every
   sample → `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge for seeds 0, 1, 2: non-decreasing SIM_GEN_SCORE
0.00 → 0.25 → 0.45 → 0.60 → 1.00, tabs drift ≤ 0.1° over the hold.

## Embodiment argument (Franka, single arm)

All bodies are tabletop-scale and reachable from one base pose at the south
side of the canister (workspace ~0.6 m across, everything below z 0.16 m):

- **red ball** (r 18 mm): standard two-finger pinch (36 mm < 80 mm gripper
  span), release above the mouth — the same gravity drop the solver uses;
- **lid** (140 mm plate): grasp the 22 mm **knob** (pinchable) from above,
  hover over the recess, lower and release — seating is gravity + recess
  self-alignment, tolerant to the 12 mm xy window;
- **tabs**: each bar tip sweeps a 60 mm gap in the lip ring; a closed-finger
  side push at the bar tip (lever arm 70 mm, needed force ≈ 0.3–1 N ≪ Franka
  payload) sweeps it inward; the vertical axis means no gravity load and the
  damping parks it wherever released — no precision hold required;
- ordering is enforced by the scene's own interlocks, matching the solver's
  phase order.

## Execution order declaration

scene → solve (verified) → rubric finalized → smoke. Tested exclusively
through the forge client; final artifacts re-verified after the last edit.

## Checks

- solve: forge seeds 0/1/2 → `SIM_GEN_SOLVE: SUCCESS`, non-decreasing scores,
  ≥3 s hands-off hold.
- smoke: forge → `SIM_GEN_SMOKE: ALL PASS 16/16` — settle/no-NaN,
  randomization readback, slot swap, null policy, SEED-strategy (lid on empty
  canister ⇒ 0), wrong object (blue sealed ⇒ 0), INTERLOCK tab-first,
  near-miss seat (20 mm offset perch), INTERLOCK mis-seat blocks tab,
  near-miss tab (−45° short of window), restraint (decoy clause
  load-bearing), MECHANISM seal (late ball stays out), settle gate (rolling
  blue defeats stillness), latched credit after dismantle (0.60, no success),
  whole-battery success()-never-True audit, frames.npz.
