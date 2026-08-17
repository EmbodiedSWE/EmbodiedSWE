# stack_cube_i183 — Ballast Hatch (`simgen.ballast_hatch`)

## Seed provenance

Derived from **maniskill/stack_cube** (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/stack_cube.py`):
grasp cube A and place it statically on top of cube B; the judged quantity is the
relative pose of the two cubes (aligned within half a cube edge, resting, ungrasped).
The seed's strategy is *a single grasp-transport-align-release cycle*, and its goal
state is *a cube-on-cube stack*.

## What changed, and why it is strategically different

Kept from the seed only its abstract germ — *small graspable cubes on a tabletop are
the working material*. Everything about the strategy and the judged outcome is
different:

- **No cube is ever stacked on, aligned to, or judged against another cube.** The four
  steel ballast cubes are consumed as *counterweights*: dropped one after another into
  a hopper cage on a bell-crank, where their accumulated WEIGHT (contact force on the
  cage floor) swings a self-closing hatch flap open. The judged quantity is the
  *mechanism's hinge angle*, not any cube pose — the exact inversion of the seed, where
  cube pose is everything and no mechanism exists.
- **Mass accumulation replaces pose accuracy.** One cube (settles ~30°, forge readback)
  is physically useless — below the ~37° transit minimum; two or more are needed (two
  in the far cells reach the ~62° limit stop), and the solution banks all four for
  margin. Success depends on *how much* steel is banked, not *where precisely* anything
  sits — a strategy with no analog in the seed.
- **The goal object is not a cube and is never lifted.** A separate light blue parcel
  must transit the held-open doorway by a nonprehensile ground PUSH (bounded force,
  below its tipping bound) and settle inside a sealed chamber. The goal state is a
  *containment event behind a mechanism*, not a stack on an open surface.
- **Versus the read corpus:** no existing task actuates a mechanism by accumulating
  object weight. `counterweight_scale` (i40) judges an equilibrium *angle* reached by
  placing masses on a balance — but there the angle IS the goal; here the angle is a
  latched *means* that gates a separate payload-delivery goal, and the mechanism is a
  self-closing hatch, not a scale. `block_pyramid_i42` steers a ball by tilting its
  supporting surface; `carousel_airlock` drives a rotor vane; `gap_ferry`/`tile_shunt`
  are pure push-to-zone tasks. Here pushing is only the trailing phase — the core is
  weight-banking mechanism actuation with a physically enforced ordering (the door is
  shut until the ballast holds it open).

## Apparatus (fully procedural)

Heavy dynamic VAULT (40 kg compound: back/side walls, doorway jambs + lintel forming a
75 x 66 mm opening, sealed roof, two hinge towers) with a floorless interior chamber on
the ground plane. On the towers' Y-axle rides the GATE, one rigid bell-crank body
(0.55 kg, authored CoM 11 cm below the hinge): an amber flap that hangs in the doorway,
two 45° up-and-back arms, and a steel-gray hopper CAGE — a 2×2 grid of snug cube cells
(48 mm cells, 68 mm deep, for the 42 mm cubes), pre-tilted 32° toward the approach side
so the cells are droppable when closed and retentive when open, and so each banked
cube's lever arm is LOCKED (an open box would let the cubes slide hinge-ward and sag
the gate). Joint limits: +0.8° (closed stop — the flap covers the doorway) to −62°
(open stop). The gate is *self-closing*: its hanging CoM restores it
shut (~0.032 N m preload), so only sustained weight in the cage keeps it open. Free
objects: four 42 mm / 0.18 kg steel ballast cubes on an approach-side arc, and one
48 mm / 0.08 kg blue parcel in the approach lane.

**Randomization (readback-verified):** whole-apparatus yaw ±180° + xy jitter ±3 cm
(vault and gate written consistently); ballast cubes on polar slots with radius and
angle jitter; parcel lane position jitter. Memorized world-frame poses and push
directions fail; the solve reads the vault frame back every phase.

## Rubric

- `success()` = parcel inside the chamber (vault-frame box, below the lintel sill)
  **and** settled **and** both latches earned: `lat_open` (gate HELD above 35° open for
  0.75 s continuously — streak-gated, so a drop-impact overswing that flicks past the
  threshold earns nothing) and `lat_door` (parcel seen inside the doorway aperture
  *while* `lat_open` — pathway gating, so a parcel that appears in the chamber without
  transiting the held-open doorway is refused).
- `score()` (monotonic, latched): 0.30 gate held open past 35°, 0.60 parcel transited
  the doorway, 1.0 iff success. Latches never un-earn; the gate re-closing after
  delivery does not evaporate credit.

## Solution outline (solve.py — the legitimacy certificate)

Teleports are TRANSPORT ONLY; every load-bearing interaction is contact dynamics:

- **P0** settle 0.5 s, layout readback (seed-dependent), gate closed at reset (0.0).
- **P1** for each of the four ballast cubes: teleport to hover 3 cm above its hopper
  CELL's CURRENT opening (gate-pose readback — the cells move as the gate opens),
  attitude-aligned with the tilted cell, release, and let it FALL in; its weight on the
  cage floor is what swings the bell-crank. The gate angle is never written and no
  wrench ever touches the gate. Far cell pair first; cubes 1→4 read back
  30° → 62° → 62° → 62° (limit stop) (score 0.30).
- **P2** transport teleport of the parcel between free ground poses in the open
  approach lane (spawn band → aligned staging spot) (score still 0.30).
- **P3** nonprehensile doorway push: bounded external force on the parcel body
  (velocity servo, |F| ≤ 0.7 N < m g ≈ 0.78 N tipping bound), applied in the parcel's
  BODY frame recomputed each step, direction and centering read from the vault frame.
  Force CUT once inside; friction settles it (score 1.0).
- **P4** hands-off persistence ≥ 3.4 simulated seconds, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge for seeds **0, 7**; scores non-decreasing 0 → 0.30 → 0.60 → 1.0.

## Embodiment argument (Franka, one base pose)

Base at ~(0.55, 0, 0) in the vault frame, facing the doorway side — the approach lane,
ballast arc (r 0.28–0.32 m), and hopper mouth (z ≈ 0.28 m when closed, tilted toward
this side) are all within a Franka's reach envelope from one pose. Per-object contact
strategy:

- **Ballast cubes (grasped):** 42 mm steel cubes fit a parallel-jaw gripper (opening
  ~80 mm); pick from the arc, carry ~30 cm, hold above the tilted mouth, open the
  jaws — exactly the hover-and-release the solve performs. No precision beyond the
  64 mm mouth aperture (vs 42 mm cube) is needed, and the cage lips funnel the drop.
- **Parcel (pushed, never lifted):** closed-gripper fingertip push at mid-height of the
  48 mm face; the solve's 0.7 N cap is a light fingertip force, and the servo's lateral
  centering maps to nudge corrections. Pushing (not carrying) avoids any need to reach
  through the doorway: the arm releases before the parcel crosses the sill and the
  parcel coasts/settles inside on friction.
- **Gate / cage / vault:** never touched by the robot; the gate is actuated only by
  ballast weight, and the sealed roof + walls mean nothing need (or can) reach inside.

## Execution-order declaration

No discrete execution order is declared beyond what the physics enforces: the
self-closing flap blocks the doorway until sufficient ballast weight holds it open
(closed-gate pushes stall — smoke check 8), so hopper loading necessarily precedes the
transit; the pathway-gated `lat_door` latch simply records that enforced order.

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 15/15` on forge)

1. Settle/no-NaN: gate closed (readback |angle| < 3°), parcel in the lane, score ≈ 0.
2. Randomization readback: apparatus xy + yaw vary across 6 seeds.
3. Randomization readback: parcel lane pose and ballast polar slots vary.
4. Null policy (300 steps): nothing moves meaningfully, score 0.
5. **Seed-strategy analog:** a parcel stacked on top of a ballast cube (the seed's
   literal goal state, readback-verified dz > 0.035) earns score ≈ 0, no success.
6. Sealed roof: parcel dropped on the vault rests ON the roof, never enters, no credit.
7. Direct-to-chamber teleport: physically inside and settled → refused (no latches).
8. Closed-gate push: solve-strength push moves the parcel (readback) but it stalls
   outside the shut flap; gate barely moves — the door really blocks.
9. One-cube near-miss: gate responds (6°–34°, actuation proven) but stays below the
   35° latch; the push still cannot get through.
10. Remaining cubes in: gate ≥ 50°, score exactly 0.30, still no success.
11. Doorway near-miss: parcel placed in the aperture (gate open) → lat_door, score
    0.60, not success; pulled back out → latched credit survives.
12. Ballast removed: gate re-closes (< 8° within 5 s — the self-closing return is
    real), score unchanged (latches hold).
13. Monotonicity: null < gate-open < doorway scores, strictly.
14. Rejection audit: success() never True anywhere in the battery.
15. Final no-NaN.
