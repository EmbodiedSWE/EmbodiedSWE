# tunnel_relay — feed four blue blocks through a tunnel so the column ejects the unreachable red cube into the pen

**Package:** `sim_gen/tasks_v7/slide_block_to_target_i352`
**Env:** `simgen.tunnel_relay` (scene-level, `robot="null"`)

## Seed provenance

`rlbench/slide_block_to_target`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/slide_block_to_target.py`): push a
red 5.6 cm cube across the floor until it sits on a flat target marker — ONE direct
planar push on the goal object itself; success is position-only and the pusher may
contact the goal object freely.

## What changed and why it is strategically different

The red cube and its journey to a target survive, but the seed's core affordance —
touch the goal object — is removed BY CONSTRUCTION, and the target stops being a
marker you can slide onto:

- The red cube (4 cm) starts INSIDE a square bore (4.8 cm wide x 5.2 cm tall)
  running along the top of a 12 cm-tall deck, both cube faces >= 5 cm from either
  opening. No gripper, finger, or tool fits beside or above it — the cube can only
  be moved BY PROXY: blue feeder blocks (5.5 x 4 x 4 cm) slid in through the
  tunnel mouth form a column that shunts it forward.
- The target is a walled GREEN PEN on the floor, one deck-height below, enclosed
  on all four sides at floor level (three walls + the deck's own cliff face). It
  can only be entered FROM ABOVE: the cube must be driven off the cliff at the
  tunnel's far end and drop in. A cube slid to any floor location outside the pen
  scores ~0 (smoke #6 constructs the seed-strategy end state and watches it fail).
- The geometry encodes an exact RESOURCE ARITHMETIC (asserted in the scene cfg at
  import time): three feeders pushed flush to the mouth reach cargo-centre 0.135 m
  — 15 mm short of the 0.15 m cliff line (smoke #14 demonstrates this with real
  contact pushes) — while four reach 0.19 m, ejecting the cube when the last
  feeder is still 40 mm proud of the mouth. The solver must fetch and relay ALL
  FOUR scattered blocks through the same mouth, and never needs (nor is able) to
  reach inside the bore.

A solver therefore needs a different PLAN (collect four proxy objects and sequence
them through a shared aperture; the goal object is never contacted) and different
CODE STRUCTURE (a repeated fetch/stage/insert cycle with column-depth readback and
an eject-triggered stop, instead of a single position servo on the goal object).
The seed's plan, executed here, is the smoke #6 reject state.

## Teleport-solution outline (solve.py, phases; `SIM_GEN_SCORE` at each boundary)

- **P0** reset + settle; layout readback printed (rig xy/yaw, cargo depth, feeder
  scatter); asserts cargo in-bore and score ~0.
- **P1–P4 FEED CYCLES (x4):** each cycle
  1. **TRANSPORT (teleport):** one pose write stages the next feeder on the open
     apron behind the mouth, long side leading — the pick-and-place of a
     graspable 4 cm-wide block. Asserted outside the bore by readback. The CARGO
     is NEVER teleported and NEVER directly forced anywhere in solve.py.
  2. **INSERT (contact dynamics):** a velocity-capped horizontal force along the
     bore axis (1.5 N start, v <= 0.05 m/s, stall escalation to 12 N then a flip
     to body-frame wrench encoding for pod-dependent frame drag) slides the
     feeder into the mouth; it shunts the column, which shunts the cargo — every
     millimetre of cargo motion is transmitted through block-to-block contact.
     Cycles 1–3 stop with the feeder rear flush at the mouth plane (the fingertip
     never enters the bore); cycle 4 cuts the force the instant the cargo
     readback crosses the cliff line — tip-over and drop are hands-off gravity.
- **P5** hands-off settle; success() verified live (cargo settled on the pen
  floor).
- **P6** hands-off persistence 3.5 s, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds **0, 1, 2** — all `SIM_GEN_SOLVE: SUCCESS` first run
(~19 s each); scores monotone 0 → 0.03–0.13 → 0.27–0.32 → 1.0; cargo landing
0.216–0.220 m local (pen centre) on every seed.

## Embodiment argument (single Franka arm, parallel jaw, OSC)

Plausible base pose: **base on the floor 0.55 m from the rig origin on the mouth
side** (rig yaw is limited to +/-45 deg, so the mouth quadrant always faces the
base); all required contacts lie at radius 0.15–0.65 m, heights 0–0.16 m — inside
the Franka envelope.

- **The feeders are the manipulanda:** 4 cm wide — comfortably inside the 8 cm
  jaw span — grasped from the open floor and set down on the apron (deck top at
  0.12 m, a natural working height), long side leading. Each insertion is a
  closed-fingertip push on the feeder's rear face; cycles 1–3 end with the rear
  flush at the mouth plane, so the fingertip never crosses into the bore, and the
  final feeder ends 40 mm proud — still fully graspable. The velocity-capped
  force in solve.py is exactly this quasi-static push.
- **The cargo is intentionally untouchable** — the bore leaves 4 mm lateral and
  12 mm vertical clearance around it, both faces >= 5 cm deep: no jaw, finger, or
  tool reaches it, which is why this is a proxy-manipulation task. The roof's
  1 cm sight slot keeps it visible for planning.
- **The pen** never requires contact: the final approach is a gravity drop off
  the cliff.

## Execution order

NOT constrained (declared in describe()): only the final settled state of the red
cube is judged — any fetch order of the four blocks, any pacing, any staging is
accepted. Physics itself forces the blocks to pass sequentially through the one
mouth.

## Rubric

`score()` = 0.45·prog (latched running max of the cargo's normalized advance from
its spawn depth toward the 0.17 m ejection line; gated to the bore lane / past-
cliff floor region so off-lane placements latch nothing) + 0.25·eject_ever (cargo
ever past the cliff plane AND below deck height, latched), capped at 0.70; exactly
**1.0 iff `success()`**: cargo centre inside the pen x/y bands, at floor resting
height (z-band rejects perch/stack states), linear AND angular velocity below the
settle gates — live physical state, no latches. Null policy scores ~0.

## Check list (smoke.py — rejection battery, forge: `SIM_GEN_SMOKE: ALL PASS 16/16`)

1. settle/no-NaN: cargo riding the deck top inside the bore, feeders at rest on
   the floor, all still
2. score ~0 at reset, no success
3. randomization readback: rig xy spread > 10 mm, rig yaw spread > 5 deg, cargo
   depth spread > 6 mm inside its band, across 6 seeded resets
4. randomization readback: feeder scatter > 8 mm and free yaw spread > 10 deg
5. null policy (240 steps): score ~0, no success
6. **seed strategy**: cargo slid on the floor flush against the pen's outer wall
   → NOT success, score ~0 (the pen is entered only from above)
7. wrong object: a FEEDER dropped into the pen, settled on its floor → NOT
   success, score ~0
8. stacked: cargo resting ON TOP of a feeder inside the pen (centre above the
   resting band) → NOT success
9. partial extraction: cargo settled inside the bore just short of the cliff →
   NOT success, score <= 0.42
10. backward removal: cargo settled on the open apron behind the mouth → NOT
    success, score ~0
11. overshoot: cargo settled on the floor behind the pen's back wall → NOT
    success
12. airborne: cargo in free fall directly over the pen, judged mid-air → NOT
    success (resting-height gate; relocated before landing)
13. latched credit: returning the cargo to its spawn depth leaves the latched
    score unchanged, still no success
14. **THREE FEEDERS SHORT**: three feeders contact-pushed to the mouth plane
    exactly like solve.py → cargo advances but stalls inside the bore short of
    the cliff, NOT success, score <= 0.46 (the fourth block is load-bearing)
15. rejection audit: success() never True anywhere in the battery
16. final no-NaN

frames.npz recorded and saved in cwd by smoke.py.
