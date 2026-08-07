# draw_svg_i49 — `swing_arrest`

Two workshop pendulums were left swinging. Arrest each one through timed inelastic
contact with a heavy arrestor block, then take the blocks away so both pendulums
hang plumb, dead still, and unsupported — with the gantries untouched at their
spawn poses.

- Scene id: `swing_arrest` · Env id: `simgen.swing_arrest` · Robot: `null`
- Files: `scene.py` (procedural geometry only), `solve.py`, `smoke.py`, this file.

## Seed provenance and why this is strategically different

Seed: **maniskill/draw_svg** — the robot drags a red marker cube along a
prescribed 2-D SVG path; the arm's own continuous motion is the product, the scene
is passive until touched, and the judged quantity is the position trace the robot
itself generated.

This task inverts that relationship on every axis that matters for a solver:

| | seed (draw_svg) | this task (swing_arrest) |
|---|---|---|
| scene at reset | at rest, passive | **moving** — two pendulums released at 50–75° swing autonomously from frame 0 |
| robot's job | inject motion (drag a trace) | **remove energy** (inelastic arrest) |
| judged product | a path of robot-generated positions | **stillness** — a sustained rest state the scene cannot reach alone (pivot decay τ ≈ 50 s ≫ episode) |
| plan structure | waypoint tracking along a known curve | **phase-aware timing**: read each pendulum's live phase, stage an interceptor in the swing path while the bob is on the far side, let impact kill the swing, extract without re-exciting |
| the red cube | dragged marker | returns as the pendulum **bob** — the thing that must *stop* being moved |

Against the tasks_v7 corpus (surveyed before design): every existing task starts
at rest and *adds* or routes energy (stacking, insertion, pouring, transport,
button-pressing, chute-routing...). None starts with stored kinetic energy that
must be dissipated, and none judges arrest of an autonomously moving mechanism.

## Success criteria (all judged live, simultaneously)

- **A. PLUMB** — each rod within `plumb_tol_deg = 6°` of vertical.
- **B. STILL (sustained)** — both rods' angular speed < 0.12 rad/s **and** both
  blocks' linear speed < 0.06 m/s for `still_steps = 60` consecutive sim steps
  (0.5 s). A swinging rod is momentarily slow at every turning point; the
  consecutive-step latch is what makes "judged only at rest" real.
- **C. CLEAR** — each arrestor block's xy ≥ `exclusion_r = 0.36 m` from *each*
  gantry centre. Rejects leaving a block propping the bob at plumb; asserted in
  `__post_init__` that even a block lying on its side with the post tip extended
  cannot reach a within-tolerance bob from beyond the radius.
- **D. GANTRIES HOME** — each gantry within 5 cm / 10° of its spawn pose.
  Rejects knocking a gantry over or dragging it to stop its pendulum.

Given C and D, physics itself certifies the result: a plumb, sustained-still rod
with nothing within reach is hanging freely in equilibrium — the swing energy
truly left through contact.

**Score (latched, non-decreasing):** 0.20 per pendulum ever *arrested* (rod
sustained-still and plumb) + 0.15 per pendulum ever *clean* (arrested while both
blocks are clear of that gantry), capped at 0.70; exactly 1.0 iff `success()`
holds now. Null policy ≈ 0: the pendulums keep swinging for minutes and the
sustained-still gate never opens (verified in smoke check 5).

## Per-seed randomization (readback-verifiable via `describe()`)

Release side and amplitude per pendulum (± U[50°, 75°]), long/short gantry side
swap, gantry xy (±2.5 cm) + yaw (±8°) jitter, arrestor staging-slot permutation +
xy jitter (±2 cm) + free yaw (±180°).

## Teleport-solution outline (`solve.py`)

Teleportation is transport only; every load-bearing interaction is contact
dynamics. Per pendulum, in any order:

1. Wait until the bob is far out on one side S (|θ| > 18°).
2. Teleport an arrestor block onto empty bench on side −S, strike face a few mm
   past the bob's plumb position, aligned to the gantry's live yaw. Press it down
   (45 N hold-down = the hand steadying the block against impact recoil).
3. The bob swings back through plumb and strikes the post: restitution 0 plus
   slab-bench friction (0.95/0.90) stop bob and block together within mm; the bob
   rests leaning on the post ≈ asin(strike_gap/ln) ≈ 0.6–1.3° past plumb.
4. Release the hold-down, teleport the block to a parking spot far outside
   `exclusion_r`. The residual sub-degree swing is below the stillness gates *by
   design* (asserted in `__post_init__`: peak residual ω × 1.2 < 0.12 rad/s) and
   keeps shrinking under rod damping. Retry on arrest-latch readback if a strike
   goes wrong.

Then wait for `success()` and hold ≥ 3.3 simulated seconds hands-off, printing
`SIM_GEN_SCORE` at each phase boundary (non-decreasing) and
`SIM_GEN_SOLVE: SUCCESS` only if success still holds at the end.

## Franka embodiment argument

Every manipulation in the plan is Franka-feasible:

- The arrestor block is **2.4 kg** (< 3 kg payload) with a **50 mm** square strike
  post — a purpose-built grasp feature under the 80 mm parallel-jaw opening; the
  block is carried or slid across a flat bench, never lifted high.
- Placing the block at the strike gap is a coarse tabletop place (±few mm
  tolerance across the slab's 0.14 m footprint); the hold-down force is exactly
  what a hand pressing on the slab does during the strike.
- The pendulums need only be *watched*, not touched; timing comes from vision of
  a 55 mm red bob on an amber rod — high-contrast, slow (period ~1.2 s).
- Everything sits on a 1.10 × 1.00 m bench with top at 0.10 m; gantry pivots at
  0.42–0.55 m — inside Franka's 0.85 m reach from a front-edge base.

**One plausible base pose:** base mounted at the bench front edge, env-local
`(0.0, -0.62, 0.10)`, facing +y — both staging slots (±0.12, −0.34), both strike
zones (gantry feet at (±0.26, 0.10)), and both parking spots (±0.45, −0.35) fall
in a 0.25–0.75 m fan in front of the base.

## Execution order

**No execution order is required.** The pendulums may be arrested in either
order, and the two arrest-extract cycles are independent (blocks and gantries are
interchangeable up to which slot/side randomization assigned them). `solve.py`
happens to go long-then-short; the rubric never inspects order.

## Verification

- `smoke.py`: **13 checks**, rejection-only battery — (1) premise: finite state,
  both pendulums genuinely swinging > 25°, blocks clear, gantries home; (2) score
  ≈ 0 / no success / still-gate closed at reset; (3) release randomization across
  10 seeds by readback (spread + both signs + side swap); (4) gantry/slot/yaw
  randomization by readback + invariants; (5) null policy: still swinging, score
  ≤ 0.02 after 3 s; (6) turning points never open the sustained-still gate; (7)
  wedged-off-plumb bob (blocked at 15°) → arrest refused; (8) block parked too
  close (inside exclusion radius) → success refused; (9) block left propping the
  bob at plumb → still+plumb+home but success refused (clause C); (10) gantry
  dragged from home with rods plumb → success refused (clause D); (11) score
  latch: credit persists after re-exciting a rod, never reaches 1.0 without
  success; (12) success never observed during the battery; (13) no NaN/Inf, and
  frames.npz recorded. Prints `SIM_GEN_SMOKE: ALL PASS 13/13`.
- `solve.py` on the forge, seeds 0/1/2: `SIM_GEN_SOLVE: SUCCESS`, score
  trajectory 0 → 0.20 → 0.35 → 0.55 → 0.70 → 1.0, non-decreasing, ≥ 3.3 s
  hands-off persistence.

## Design notes (physics traps avoided)

- Rod CoM authored explicitly from collider volumes (`CreateCenterOfMassAttr`) —
  MassAPI-only mass on a compound root leaves the CoM at the body origin (= the
  pivot), which silently kills the restoring torque.
- Gantries are heavy **dynamic** bodies (28 kg): a joint anchored to a teleported
  kinematic body0 stays world-fixed at spawn pose on this stack.
- `jointFriction` is inert for non-articulation joints here; the sole decay
  device is rod angular damping (0.02, τ ≈ 50 s), which keeps the null policy
  honest *and* lets the post-arrest residual die out.
- Stillness is a consecutive-step counter latched in `post_step`, never an
  instantaneous velocity test (turning-point trap).
- The gantry foot is narrow (`foot_y = 0.05`) and the strike post sits at the
  slab's front edge, so a block standing at the 4 mm strike gap touches nothing
  (asserted).
