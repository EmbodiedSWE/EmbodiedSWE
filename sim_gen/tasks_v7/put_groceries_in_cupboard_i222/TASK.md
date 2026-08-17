# put_groceries_in_cupboard_i222 — GravityFeedRack

**Scene name:** `gravity_feed_rack` · **Env:** `simgen.gravity_feed_rack` (robot="null")

## Seed provenance

Seed task: `rlbench/put_groceries_in_cupboard` — "put the groceries in the
cupboard": named grocery items among lookalike distractors stand on a table;
the robot grasps the named one and SETS IT DOWN on a passive open cupboard
shelf. Any resting pose on the shelf surface, reached by lowering from above,
is terminal; the cupboard itself is inert scenery.

## Strategic difference

The seed's entire plan — pick the grocery, lower it onto a shelf, let go — is
**insufficient here by construction**: there is no shelf. The "cupboard" is a
store-style **gravity-feed FIFO can rack**, fully enclosed (14° inclined ramp
behind side walls, front stop wall, ramp-parallel roof). Its only opening at
storage height is a 24 mm viewing slot — less than half a can diameter, passes
nothing (asserted). The only entry is a **loading port at the TOP-REAR** of
the roof line, ringed by slick 45° funnel flares, and placement is done by the
**mechanism**: a can dropped cross-lane through the port lands on the ramp,
rolls downhill under the roof, and queues against the stop wall (or the can
already stored). The roof gap (78 mm) admits exactly one rolling can — no
stacking, no leapfrog, no standing under the roof (all asserted numerically in
`__post_init__`). Because the port is *behind* the queue and the roof pins it
down, **first in = frontmost, forever**.

The goal is a facing arrangement — RED at the window, GREEN behind it, BLUE at
the rear — so the load-bearing decision is not *where to set the can down*
(gravity decides that) but **WHICH CAN TO FEED NEXT**. The seed's plan carries
zero information about this; the required plan is an ordered loop of
pick → deliver-to-port → hands-off-wait → verify episodes whose sequencing
(red, then green, then blue) is irreversibly frozen into the outcome by the
physics of the queue. Wrong order is not recoverable and is rejected as a
fully *stored* end state (smoke check 8); the seed's own end state — the can
set down on the enclosure from above — rests on the roof and earns nothing
(check 5).

Differs from the sibling same-seed packages: `put_groceries_in_cupboard_i169`
(crank-indexed **carousel** — align bay, insert through window, rotate to
stow) and `put_groceries_in_cupboard_i329` (spring-loaded facing-lane
**pusher** — press items into a sprung lane). Here there is no actuated
mechanism at all: the "mechanism" is gravity on an enclosed incline, and the
strategy variable is the **FIFO insertion order** of three items. No other
tasks_v7 package makes ordering-into-a-passive-dispenser the load-bearing
decision.

## Assets (fully procedural)

- **rack** — KINEMATIC compound (no joints), re-posed each reset: 14° ramp
  (interior x ∈ [−0.135, +0.135], surface z 0.050 at the front edge), lane
  width 0.098 (can + 4 mm play per side), side walls (t 0.015, h 0.20), front
  stop wall with a 24 mm **viewing slot** (z 0.098–0.122, split lo/hi boxes),
  ramp-parallel **roof** (underside 0.078 above the ramp) whose leading edge
  at x = −0.0525 leaves the top-rear **loading port** (downhill slope-length
  82.5 mm < can length 90 mm — a can cannot be posted lying along the slope),
  four slick 45° funnel **flares** (μ 0.10) around the port. Defined friction
  μ 0.60 and restitution 0 authored via a bound physics material (custom
  spawners apply no cfg schemas).
- **cans** — red / green / blue plain cylinders r 0.030 × l 0.090, mass 0.25,
  μ 0.60, restitution 0, `solver_velocity_iteration_count=4` (GPU capsule/
  cylinder creep hygiene), standing on three floor pickup slots in front of
  the rack.

Randomization (readback-verified, smoke checks 3–4): rack xy ±30 mm + yaw
±15°; can-to-slot **permutation** over the three floor slots + per-slot xy
jitter ±25 mm.

Geometry interlocks asserted in `__post_init__`: port slope-length band
(one cross-lane can passes, an along-slope can cannot), roof gap band
(2r + 14 mm ≤ gap ≤ 3r: rolls under, cannot stack or stand), lane play band,
viewing slot < can radius, queue-slot non-aliasing vs x_tol, incline ≥ 8°
with μ > tan(incline) + 0.15 (cans roll, nothing slides), standing-can
stability in the port shaft.

## Rubric

- 0.10 `lifted` — red can ever raised above 0.15 m (latched)
- 0.10 `entered` — red can ever inside the lane under the roof (latched)
- 0.20 `q1` — red can ever seated at the front window slot (latched, 6-step
  streak, updated only in `post_step`)
- 0.20 `q2` — red AND green ever seated at slots 1+2 together (latched, streak)
- non-success capped at 0.60; **1.0 iff success()**: red/green/blue seated at
  queue slots 1/2/3 (rack-frame x/y/z windows + cross-lane axis cosine ≥ 0.95),
  all settled, all finite — judged LIVE. Null policy ~0 (the rack does not
  stock itself; cans start on the floor slots).

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset (seed AFTER build), 60-step settle, layout readback (rack pose,
  per-can floor slot); score ~0 asserted.
- Per can, in the forced order red → green → blue:
  - **PICK (applied wrench)**: PD carry wrench (gravity compensation + z hold
    + xy centring + attitude damping) lifts the can straight off its floor
    slot to 0.25 m; the `lifted` latch is earned by this force lift.
  - **TRANSPORT (teleport)**: ONE pose write stages the can hovering centred
    over the loading port, axis cross-lane, 2 cm above the funnel mouth —
    free air, on no queue slot, outside the lane volume (asserted credit-free).
  - **FEED (gravity + contact, never teleported)**: all wrenches cut. The can
    falls through the port onto the ramp, rolls downhill under the roof and
    queues against the stop wall / the previous can. The solver only WAITS
    for a 40-step seated+settled streak and verifies the seat readback.
- Scores at the phase boundaries: 0.00 → 0.40 (red fed) → 0.60 (green fed) →
  1.00 (blue fed), asserted non-decreasing with per-phase minima.
- **P4** ≥ 3.3 simulated seconds hands-off; success persists →
  `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport-only: lifting is forced, and every queue position is
produced by gravity, rolling friction and can-on-can contact.

## Execution order

STRICTLY ordered: red, then green, then blue. The order is enforced by the
mechanism, not by bookkeeping — whatever drops first rolls to the window and
is frontmost forever (the port is behind the queue, the roof gap forbids
overtaking or stacking, the slot passes nothing). A wrong order produces a
fully stored but non-goal facing and is rejected (smoke check 8); a correct
prefix earns exactly the latched cap 0.60 (check 9).

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing the rack (base centre nominal
(0.48, 0), downhill face toward the robot). The three pickup slots stand
0.42–0.55 m out at graspable height — a side pinch on a Ø60 mm can (< 80 mm
jaw span). The loading port is at the top-rear of the rack, its funnel mouth
at z ≈ 0.20 and 0.54–0.62 m out — within Franka's ~0.85 m envelope from
above; the flares mean a coarse release centred over the port suffices (the
funnel and the ramp do the rest), which is exactly the hands-off feed the
solve certifies. Each can is a separate pick-carry-release stroke; no step
needs a second arm, a regrasp in flight, or simultaneous contacts, and the
stored cans are never touched again.

## Files

- `scene.py` — cfg (+ port/roof/slot/lane/incline interlock asserts), compound
  rack spawner (raw pxr, bound friction materials), cylinder cans, scene
  (rubric, latches only in `post_step`, LIVE success), `register_env`.
- `solve.py` — phased solution (force lift, one staged hover write, hands-off
  gravity feed, per-phase score asserts); watchdog + hard exit.
- `smoke.py` — 15-check rejection battery (settle, null policy, permutation +
  jitter readback, seed-strategy roof rest, quasi-static viewing-slot force
  probe, along-slope port drop, wrong order, prefix cap, latch regression,
  standing-in-port, axis near-miss, positive control + live-success drop,
  rejection audit, finite; frames.npz). The slot probe commands its wrench in
  the can's BODY frame (a 90°-rotated body makes world-frame commands
  scramble y/z on this stack) and samples only while engaged.

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 18.2 s; scores
  0.00 → 0.40 → 0.60 → 1.00, non-decreasing, 3.3 s hands-off persistence;
  queue readback red (+0.105, z 0.088), green (+0.047, 0.103),
  blue (−0.011, 0.117) — the analytic rest slots)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 18.2 s; different rack
  pose (+0.490, −0.016, quat_z 0.9948 vs +0.508, +0.017, 0.9997) and floor
  permutation)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, 40.8 s; slot press
  advanced 48 mm and was arrested at x 0.177 ∈ (0.172, 0.192) outside, never
  entered; wrong-order end state fully stored yet score 0.20; prefix cap
  exactly 0.600; standing-in-port stable (axis 1.00) and refused; 257 frames
  saved)
