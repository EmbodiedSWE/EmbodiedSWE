# screw_nail_i309 — `leaning_chain` (frozen domino cascade)

Build the frozen end state of a fallen domino run: three flat tiles must end
LEANING toward a raised anvil inside a marked lane, each tile's foot on the
floor and its head resting ON the next body in the chain — BLUE's head on the
ANVIL top, GREEN's head on blue, RED's head on green. The chain can only be
built from the anvil backwards (blue, then green, then red), because a tile
leaned onto empty lane simply falls flat: each tile needs the next one already
leaning to hold it up.

Env: `simgen.leaning_chain` (robot="null"), scene `leaning_chain` in
`scene.py`.

## Provenance

Seed task: `rlbench/screw_nail`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/screw_nail.py`) — a Franka
picks up a screwdriver and drives a nail by pressing down and rotating:
tool-mediated rotary fastening of a single fastener into a fixed base.

## Strategic difference

The seed's strategy is *tool-mediated rotary insertion*: acquire one tool,
mate it to one fastener, and convert wrist rotation plus downward force into
screw advance. Success is a single body reaching a depth along one axis.

This task keeps the "assemble parts against a fixed base" family but the
strategy shares nothing with it:

- **No tool and no rotation-to-advance conversion.** Nothing is screwed,
  pressed, or twisted. Smoke check 6 applies the seed family's exact move
  (press down + twist) to a tile and asserts it achieves nothing.
- **Multi-body statics instead of single-fastener kinematics.** The goal state
  is a mutually loaded chain of three leaning bodies at *intermediate* tilt
  (15–65° from horizontal) — each tile is in equilibrium only because the next
  body carries its head load. The judged quantity is a self-supporting
  arrangement, not a depth.
- **Mechanically forced build ORDER, back-to-front.** The seed has no ordering
  structure at all. Here the physics itself serializes the plan: leaning a
  tile onto empty lane drops it flat (asserted in smoke check 7's collapsed-
  cascade probe and enforced by `__post_init__` geometry asserts — a flat or
  rail-leaning tile can never reach the head-height gate). The rubric
  additionally latches stage credit only while the deeper stages are seated
  (order-aware gated streaks).

It is also distinct from the sibling `screw_nail_i59` ("latch_vault": an
unlock→uncover→retrieve *disassembly* chain on prismatic sliders) — this is a
free-body *construction* task with no joints at all — and from container-
insertion tasks like `pen_holder`.

## Solution outline (solve.py, teleport = transport only)

1. Read the randomized layout (lane origin, lane yaw, tile→slot permutation)
   from state readbacks.
2. For each stage in the forced order blue → green → red:
   teleport the tile STANDING in the lane at a chosen foot offset from its
   support (transport only), then topple it toward the anvil with a
   rate-regulated torque about the lane-cross axis (τ = clamp(k·(ω_des−ω)))
   until 30° from vertical, release, and let gravity + contact seat the head
   on its support. Retry ×4 with adaptive foot spacing using the tilt
   readback (fell flat → closer; too steep → farther).
3. Every seat is created by contact dynamics; the streak latch (25 substeps of
   seated stillness, gated on the deeper stages) converts seats into score.
   `SIM_GEN_SCORE` prints at P0 (0.00) → P1 blue (0.25) → P2 green (0.50) →
   P3 red (1.00) → P4 settled → P5 persistence (≥3 s hands-off), never
   decreasing, then `SIM_GEN_SOLVE: SUCCESS`. Verified on seeds 0 and 1.

## Embodiment argument (single Franka + parallel-jaw gripper)

Every tile is a 120 × 48 × 16 mm slab of 60 g: while standing, its 16 mm
thickness is a natural parallel-jaw grasp far below the ~80 mm jaw span, from
a top-down or lateral approach with nothing overhead. The lane center sits at
~0.45 m from the env origin with ±0.06 m position jitter and ±30° yaw, and all
spawn slots and lane stations lie within a 0.65 m radius — inside a
base-mounted Franka's comfortable dexterous workspace at these low heights
(everything below 0.12 m). The robot's motion mirrors the solve: grasp a
standing tile, carry it into the lane, set it down standing at the spacing
offset, then tip it over with a fingertip push — a low-force (sub-newton)
quasi-static interaction. No bimanual coordination, no tool, no
re-orientation in the air is required.

## Execution order

Declared and mechanically forced: **blue first (onto the anvil), then green
(onto blue), then red (onto green)**. Any other order fails physically (no
support yet → tile falls flat, rejected by tilt/head gates) and is also
rejected by the rubric's identity adjacency (each head must rest past the
NEXT tile's actual foot; green needs blue seated, red needs both) and its
order-gated streak latching.

## Checks (smoke.py — 15)

1. Reset settles finite; three tiles standing beside the lane; score ~0.
2. Layout sanity (rails/anvil/marker poses; standing tile heights).
3. Randomization readback: lane position + yaw spreads over 6 seeds.
4. Color→slot permutation varies; per-tile spawn poses sane.
5. Null policy (240 idle steps) → score ~0, no success.
6. Seed-strategy analog: press-down + twist on a tile → no seat, score ~0.
7. Collapsed cascade: all three flat/shingled in-lane in color order, every
   tile below the tilt band and head gate → zero credit.
8. All three standing vertical in-lane → tilt band rejects, no success.
9. Rail lean (shallow, cross-lane) → rejected.
10. Wrong color order (green on anvil, blue mid, red last; a REAL settled
    leaning chain) → identity predicates reject every tile.
11. Skip-chain: blue honestly seated (0.25 credit allowed) but red leaning
    directly on blue with green parked away → red rejected, no success.
12. Settle gate: perfect in-band pose but moving → not seated.
13. Latched credit: demolishing seated blue leaves latched score unchanged
    while `seated_now` drops.
14. Audit: success() never True at any judged point of the battery.
15. Final no-NaN over all task objects.
