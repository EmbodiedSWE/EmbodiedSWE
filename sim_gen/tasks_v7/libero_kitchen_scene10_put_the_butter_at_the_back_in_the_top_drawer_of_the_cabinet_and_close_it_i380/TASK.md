# butter_hatch — close the drawer FIRST, then post the butter through the countertop hatch

## Seed provenance

- **Seed task**: `libero_90/libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it`
  (RoboVerse, `roboverse_pack/tasks/libero_90/libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.py`).
- **Seed semantics**: pick `butter_2` off the table, place it into the TOP drawer of the wooden cabinet
  (drawer-region containment check), then close the drawer (`joint_pos > -0.01`). Distractors: a black
  bowl, a chocolate pudding, and a second identical butter. Success = butter in the drawer region AND
  the drawer closed.
- **This package** is fully procedural (no external assets) and implemented as a robobench scene-level
  task (`SCENES.register("butter_hatch")`, `register_env("simgen", ...)` with `robot="null"`).

## What changed and why it is strategically different

The seed's plan is **open → insert from the top → close**: the drawer cavity is top-accessible whenever
the drawer is pulled out, insertion is a plain top-down place, and closing is an independent finishing
move that can, in principle, happen any time after the place. Here that plan is INVERTED and the order
is **geometrically forced**, not merely rubric-rewarded:

1. **The cavity is NEVER top-accessible.** A fixed countertop extends forward as a canopy that roofs the
   drawer over its whole 160 mm travel, leaving a 6 mm gap over the drawer walls — a 26 mm-thick butter
   cannot enter from above or the side at ANY drawer position (import-time assert + smoke check 6: a
   butter released above the open cavity lands ON the canopy).
2. **The only way in is a fixed hatch that aligns only at CLOSED.** A 100 mm square hole through the
   countertop (ringed by an orange collar chute) sits over the drawer cavity ONLY when the drawer is
   fully closed (>= 10 mm containment margin, asserted). While the drawer is open, the hatch column is
   fully clear of the drawer body (>= 20 mm, asserted) and drops past the drawer plane into a **reject
   cellar**: a deposit made before closing is not just unrewarded — it is physically LOST below the
   drawer and cannot be recovered by closing afterwards (smoke check 7).
3. **Closing is therefore the FIRST move and the alignment-creating act**, the exact reverse of the
   seed, where closing is the last move and destroys top access. The insert itself is a ballistic
   gravity deposit through a chute, not a place into an open receptacle.
4. **Color discrimination is load-bearing**: a white paraffin block of identical shape shuffles spawn
   slots with the yellow butter every episode; posting the paraffin fails, and the paraffin must stay
   out of the drawer (success clause + smoke checks 9/10).

Versus its same-seed siblings: **i307** (butter_cellar) is a spring dumbwaiter with a one-way pawl
ratchet driven by an ingot gravity press — its mechanism story is *press-cycle + irreversible latch*.
**i62** (rammer_gallery) is a rammer-trolley stroke plus a transverse gate seal — *horizontal ram +
separate closure*. This task's story is *order-forcing by alignment*: a passive drawer on a plain slide
whose closed pose is the only pose that connects the world to its interior, with an irreversible
punishment path (the cellar) for doing the seed's order. No spring, no ratchet, no ram, no second
closure member — the countertop geometry is the entire mechanism.

## Teleport-solution outline (solve.py)

- **P0** settle + layout readback: drawer open at random q0 in [-0.14, -0.06], blocks on the floor,
  score ~0 asserted.
- **P1 CLOSE (force dynamics, no drawer pose writes)**: a velocity-servoed external push
  `F_x = 8 * (0.06 - vx)` N drives the drawer along its real prismatic joint until it seats on the
  q = 0 hard stop; force zeroed; settles closed and still. `SIM_GEN_SCORE` (close latch, ~0.20+).
- **P2 DEPOSIT (transport teleport + gravity contact)**: one pose write carries the butter to a hover
  above the collar mouth — asserted OUTSIDE both scoring bands, so nothing latches at the write — then
  it falls through collar → hatch hole → lands inside the closed drawer through real contact.
  `SIM_GEN_SCORE` (chute + inside latches; success → 1.0).
- **P3 persistence**: >= 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS` only if success()
  holds throughout. Scores are printed at each phase boundary and are non-decreasing (latched credit).

## Embodiment argument (single Franka arm, parallel jaw + OSC)

Base at the world origin, facing +x; the cabinet bay centre is at (0.58, 0), blocks spawn at
(0.24, ±0.30). All interaction points lie within ~0.70 m of the base and above z = 0.16 m.

- **Close the drawer**: the drawer carries a dark 24 mm-square knob bar protruding from its face at
  z ≈ 0.16; at full open the knob tip sits ≈ 39 mm in FRONT of the canopy's front edge (asserted), so
  the closed gripper can press the knob face square-on and push straight in (+x, ~0.10–0.16 m stroke,
  a planar push at fixed height) until the drawer stops — exactly what the solve's regulated force
  reproduces. No grasp is required for this stroke.
- **Pick the butter**: 58 x 32 x 26 mm block on open floor; top-down pinch of the two 32 mm side faces
  (fits the Franka jaw span), lift vertically.
- **Deposit**: carry to (0.62, 0.00) and release ~0.32 m up, directly above the 74 mm-wide orange
  collar mouth — an open-top drop with >= 25 mm clearance around the butter's largest horizontal
  diagonal (asserted); gravity does the insertion, the arm never enters the collar.
- **Nothing requires a second arm**: the drawer stays closed by its hard stop (deposit impacts push it
  toward closed), the blocks are independent, and no step needs simultaneous contacts.

## Execution-order declaration

The required order is DECLARED in `describe()`/`instruction()` and FORCED by geometry: close first,
deposit second. Both branches of the wrong order are physically punished — top insertion is denied by
the 6 mm roof gap (any drawer position), and hatch deposit while open loses the butter in the reject
cellar (unrecoverable; smoke-verified). The rubric's chute/inside latches carry `drawer_closed` as a
conjunct, so no credit for those phases can be earned out of order.

## Checks

- `scene.py` `__post_init__`: ~10 import-time geometry asserts (roof-gap denial, hatch-in-cavity at
  closed, hatch-clear-of-drawer at open, canopy coverage, hole admission margin, cellar drop clearance,
  knob protrusion at open, lateral sweep clearances, q0 band clear of the joint stops).
- `solve.py`: per-phase asserts (open at reset, baseline ~0, closed stop reached + latch, hover outside
  scoring bands, chute latch fired, landed in cavity, drawer still closed, paraffin out), non-decreasing
  `SIM_GEN_SCORE` at each boundary, 3.3 s hands-off persistence, watchdog hard-exit. Passes on
  seeds 0 and 1.
- `smoke.py`: 14 named checks — settle/no-NaN, score ~0 at rest, q0/slot/jitter/yaw randomization by
  READBACK, null policy ~0, seed-strategy denial (canopy), out-of-order cellar loss (unrecoverable),
  near-miss ajar, wrong object, paraffin-exclusion, goal-adjacent park, latched-credit persistence,
  never-success audit, final no-NaN; frames.npz recorded.
