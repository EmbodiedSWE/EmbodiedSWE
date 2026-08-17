# stack_pyramid_i218 — Pyramid Freight

Scene: `pyramid_freight` · Env: `simgen.pyramid_freight` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed: `maniskill/stack_pyramid` (`RoboVerse/roboverse_pack/tasks/maniskill/stack_pyramid.py`):
three 0.04 m cubes; "pick up the red cube, place it next to the green cube, stack the
blue cube on top of both". Success is a `DetectedChecker` with two relative-bbox
detectors — blue on top of red AND green. The entire task is repeated free-space
pick-and-place AT the goal location: each placement is judged where it is made, every
intermediate is independently stable, and nothing already built ever moves again.

## What changed and why it is strategically different

The seed's plan is inverted from build-in-place to **BUILD-THEN-DELIVER**. The goal
location is now the pocket of a freight depot whose orange roof leaves ~17 mm of
headroom over the finished pyramid (roof underside 115 mm; deck-top + two cube layers
= 98 mm): *no gripper — and no falling cube — can bring a block down onto a stack in
there*. Smoke certifies this physically: a cube released above the pocket lands ON the
roof (check 10). So the seed's whole strategy is impossible at the goal, and the
pyramid must be assembled OUTSIDE, in the shallow two-cell tray of a yellow cart
parked on guide rails, and then the whole **fragile assembly transported**: pushed by
the cart's black post along the rails, through the door, until the cart rests against
the back wall. Transport is stability-limited honest dynamics — only the two base
cubes are confined by the 10 mm tray lips; the bridging blue cube rides free, so a
12 N slam throws it off (smoke's shove certificate, ~30 m/s² ≫ the μg ≈ 8.8 m/s² slip
threshold) while a ≤ 2 N push keeps the stack together. The signature skills —
*forced assemble-before-transport ordering* and *moving an already-built structure
gently enough that it survives* — appear nowhere in the seed (its placements commute
and nothing built ever moves), and are distinct from the corpus read this session: no
falsework (i71's tent), no lever/pry, no tilt-maze, no counterweights, no insertion,
no pouring, no toppling, no captive-gate extraction.

The seed's own strategy is a settled reject state: a real pyramid built on the ground
inside the pocket scores ~0 (smoke check 5 — every rubric clause judges cubes ON the
cart, in the cart body frame).

## Teleport-solution outline (solve.py)

1. **P0** settle + layout readback; assert score ≈ 0, no success.
2. **P1 seat the base pair (contact)**: red and green teleported to release poses —
   square to the cart, 4 mm above the two tray cells (tray_cx ± 21 mm) — and dropped;
   the lips seat them. Settled pair latches `seated` (0.20).
3. **P2 bridge (contact)**: blue teleported 3 mm above the ACTUAL settled base pair's
   midpoint (read back in the cart body frame) and dropped; it bridges the seam (the
   tray bounds the gap to ≤ 4 mm, so no wedge). Latches `built` (0.25).
4. **P3 delivery push (applied force)**: horizontal force servo on the cart — the
   hand on the black post: `F = M(μg + 8(v_des − v_along))` along the door direction,
   capped at 2.0 N (peak ~5 m/s², under the 8.8 m/s² blue-slip threshold — gentle by
   construction), plus a soft lateral hold to the channel centreline and a weak
   yaw-uprighting torque (the grip). Cruise 0.08 m/s, final approach 0.03 m/s so the
   back-stop jolt slides blue < 1 mm (v²/2μg). The cart stalls on the back stop;
   forces cleared; 2 s hands-off settle → `deliv` latches (0.40), success.
   All teleports are transport-only; every rubric fact (seated pair, bridge, arrival)
   is produced by gravity/contact/friction.
5. **P4** hands-off persistence 10×40 steps (3.33 s at 120 Hz); `SIM_GEN_SOLVE:
   SUCCESS` only if success still holds. `SIM_GEN_SCORE` printed at every boundary,
   non-decreasing (latched credit).

## Embodiment argument (Franka, base near origin)

Everything sits within ~0.75 m of a base at the origin (depot at ~(−0.10, 0), cube
scatter ~0.5–0.65 m out, cart start ~0.36 m from the depot). Per-object contact
strategy:

- **Cubes** (40 mm, 50 g): the seed's own objects — standard top-down parallel-jaw
  grasps (40 mm ≪ the ~80 mm Franka jaw span), free-space carries, releases a few mm
  above the tray cells / the base pair. Exactly the writes the solve performs. The
  tray's 10 mm lips leave the upper 30 mm of a seated cube proud for regrasping.
- **Cart** (0.25 kg loaded to 0.40 kg): pushed by the **black post** (22 mm square,
  100 mm tall, at the rear — fingers close on free post, no cargo contact; the post
  stays OUTSIDE the roof line even at full insertion, depot-x 0.132 vs roof edge
  0.115, so the hand is never under the roof). The push is a straight horizontal
  shove ≤ 2 N (≪ Franka payload), guided by the rails; no reorientation.
- No bimanual holds, no in-hand manipulation beyond square set-downs; the roof forces
  ordering, not dexterity.

## Execution order declared

Built in this order: (1) minimal goal predicate + scene, (2) working solve, (3) final
rubric (latched weights 0.20/0.25/0.40, cap 0.85), (4) smoke battery.

## Checks (smoke.py)

12 checks: settle/baseline · randomization readback (depot xy+yaw, cart depot-x, red
xy) · slot-permutation coverage (≥3 perms / 8 resets) · null-policy · SEED-strategy
ground-pyramid-in-pocket reject · built-not-delivered (score ~0.45, no success) ·
empty-cart-delivered reject (delivery latch conditioned on cargo) · wrong-cube-on-top
reject · near-miss span (blue on ONE base cube) reject · ROOF certificate (cube
released above the pocket lands on the roof — build-inside impossible) · SHOVE
certificate (12 N slam: cart arrives, pyramid doesn't; actuation asserted non-vacuous)
· frames.npz saved.
