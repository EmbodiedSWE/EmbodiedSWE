# gimbal_service — seat, nest and serve on a tilted slick shelf

**Task id:** `libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_i377`
**Scene name:** `gimbal_service` (env `simgen.gimbal_service`)

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl`
— pick one black bowl off a flat kitchen table and stack it on the other black bowl.

Kept from the seed: two black bowls and the act of nesting one inside the other, plus a
small golden payload (the LIBERO kitchen prop scale), on a tabletop workspace.

## What the task is

A kitchen service station stands on the bench: a squat column carrying a **slick
tilted shelf** (pitch randomized 16–20°, yaw and center randomized per episode) with a
single grippy **ring curb** bolted to its face. Three free bodies start parked on the
flat bench, sides mirrored per episode:

* the **socket bowl** (black, flat-bottomed, with an inner step),
* the **gimbal bowl** (black, with a ball-shaped underside and heavy below-pivot
  ballast — a self-leveling cup),
* the **golden cube** (the payload to serve).

Goal, in declared order:

1. **Ring** — seat the socket bowl inside the ring curb on the tilted shelf. The bare
   slab is slick (bowl-base pair μ ≈ 0.03, cube pair μ ≈ 0.21, both « tan 16° ≈ 0.287):
   anything placed on it outside the ring slides off. Only the curb arrests and holds
   the bowl, face-aligned with the shelf.
2. **Nest** — nest the gimbal bowl into the seated socket bowl. Its polished ball
   bottom drops through the socket's mouth and wedges on the socket's polished seat;
   because the contact is a sphere, leveling is a rotation about the sphere centre
   that slides at the wedged contacts — the ball/seat pairing is deliberately slick
   (pair μ 0.04, friction stick angle ~3.6°) and the ballast sits 34 mm below the
   pivot, so the cup **self-levels to world-level (≤ 8°) while the socket under it
   stays tilted 16–20°**. A bowl that simply rode the shelf rigidly could never pass
   the level gate.
3. **Serve** — place the cube inside the gimbal cup, resting level (≤ 10°). The cube
   put directly into the *tilted* socket instead rests face-flat on the tilted floor,
   contained by the socket's lower wall, so its tilt equals the shelf pitch and it
   fails the gate by geometry — the level gimbal cup is the only valid serving
   surface.

Structure clause (declared in `describe()`/`instruction()` and enforced by the latch
chain): nesting only counts **while the socket bowl is ringed**, and serving only
counts **while the full stack (ringed + nested) is intact**. Dropping the gimbal bowl
straight into the ring curb — physically stable and level — latches nothing, because
`nested` demands the socket-bowl body-frame bands.

Score: 0.20 (ringed) + 0.30 (nested, order-gated) + 0.30 (served, order-gated),
overridden to 1.00 iff the full live conjunction holds and the scene is settled.

## Why this is strategically different

* **vs the seed:** the seed is a flat-table pick-and-place proximity pose. Here the
  stack is a **passive articulation machine on a randomized tilted fixture**: the
  slick incline actively rejects placements, the ring curb is the only anchor, and the
  middle element of the stack is a ball-and-socket **gimbal that must self-level by
  contact physics** — the success predicates are about emergent orientation
  (world-level cup on a 16–20° base), not about where objects were put down.
* **vs i7 (push-to-mark + board bridge):** no pushing, no bridge-building, no
  planar transport puzzle; the challenge is orientation decoupling on an incline.
* **vs i146 (counterweight gravity gate):** no joints anywhere — the station is a
  pose-randomized kinematic compound; the "mechanism" is a free-body spherical
  contact, and the gate is a friction/geometry shed condition, not a lever law.
* **vs the corpus recipes:** no recipe uses a **ball-bottomed self-leveling vessel**
  whose righting is the load-bearing check, nor a slick-shelf-plus-curb anchor where
  the same fixture both rejects (bare slab sheds) and accepts (curb arrests) the
  same object.

## Solution outline (mirrors solve.py phases)

Teleports do TRANSPORT ONLY; every load-bearing interaction is contact + gravity:

0. Reset, settle at the parks, layout readback → score 0.000 (hard-fail if nonzero).
1. **Socket bowl:** hover-release 22 mm above the ring seat, face-aligned to the
   shelf, zero velocity. It falls, slides a few mm downhill on the slick coat and is
   arrested by the curb (the writer never holds it) → 0.200.
2. **Gimbal bowl:** hover-release **world-level** 115 mm above the socket, identity
   orientation, never re-oriented afterwards. The ball falls through the mouth, seats
   on the socket floor, and the ballast rights the cup on its own → 0.500.
3. **Cube:** hover-release 130 mm above the (self-leveled) cup; it falls in and rests
   level by itself → 0.800.
4. Settle to full success → 1.000.
5. Hands-off persistence ≥ 3.3 simulated seconds, success holds → `SIM_GEN_SOLVE: SUCCESS`.

Failed drops re-park the body and retry with millimetre nudges (≤ 4 attempts each).
`SIM_GEN_SCORE` is printed at every phase boundary and hard-fails on any decrease.

## Single-Franka embodiment argument

A single Franka at base (−0.42, 0, 0.20) on the bench solves this without teleports:

* **Reach:** parks sit 0.33–0.61 m from the base, the station center at most ~0.58 m
  — all inside the ~0.85 m reach envelope.
* **Socket bowl:** rim grasp — the 8 mm-thick upper wall is a standard antipodal
  pinch for the Franka gripper (max aperture 80 mm); lower it into the ring curb
  (6.4 mm radial play) roughly face-aligned and release; gravity and the curb finish
  the seating, exactly as the certificate's hover-release does.
* **Gimbal bowl:** rim grasp on the 8 mm cup wall; the mouth of the socket leaves
  ~10 mm radial insertion clearance for the 84 mm ball crown, so a hover over the
  seated socket and release suffices — self-leveling needs no wrist dexterity.
* **Cube:** 30 mm cube, trivial pinch; drop into the ~80 mm level cup mouth.
* No phase needs more than one object in hand, no re-grasp under load, no
  simultaneous contacts — strictly sequential pick, hover, release.

**Execution order (declared):** socket bowl → gimbal bowl → cube. The latch chain
makes any other order score-dead: nesting before ringing latches nothing, serving
before nesting latches nothing.

## Checks (smoke.py, 15)

1. Settle & finite state, pitch in [16°, 20°], score 0 / no latches at reset.
2. Rubric clean at reset on a second seed.
3. Determinism: same seed twice → identical layout readback (< 1e-5).
4. Randomization by READBACK: pitch spread > 1°, yaw spread > 0.3 rad, center
   spread > 1 cm across seeds.
5. Park mirroring: both sides occur across seeds; park jitter > 5 mm.
6. Null policy: 240 steps hands-off → score stays 0.
7. Bare-slab shed: cube placed on the slick slab (off-ring lane) slides off
   (Δx > 5 cm) — the incline really rejects.
8. Cube-in-tilted-socket rejection: real ringed socket, cube placed inside it —
   rests face-flat on the tilted floor at shelf pitch, fails the 10° level gate,
   score pinned at 0.20.
9. **Seed-strategy end state:** the full bowls+cube stack assembled flat on the
   bench — geometric predicates TRUE, but no latches, score 0 (the seed's flat-table
   stack is worthless here).
10. Declared clause: gimbal bowl straight into the ring curb + cube in it — cube is
    in the cup, but score 0 (nest requires the socket bowl).
11. Inverted socket bowl in the ring: `a_in_ring` alignment gate rejects, score 0.
12. Uphill arrest honesty: socket bowl released below the ring slides downhill and is
    NOT captured (never ringed) — the curb only works at the seat.
13. Real stages 1+2 → score 0.50 with cup self-leveled.
14. Latch survival: after 0.50, gimbal bowl teleported away → additive credit
    survives (score stays 0.50, success false).
15. All state finite after the full battery.

frames.npz (camera record) is written to the CWD.
