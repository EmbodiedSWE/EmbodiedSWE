# counterpoise_rack — "Load the balance so it still reads level"

## Seed provenance

Seed: `rlbench/place_shape_in_shape_sorter` — pick up a shape and insert it
through the matching cutout in the top of a shape-sorter box.

## Strategic difference

The seed's strategy is **selection by silhouette**: the piece that FITS the
hole is the right piece, and threading it through is the whole task. This
task keeps the surface form (loose pieces, a fixture with sockets, seat the
pieces in the sockets) and inverts the selection principle:

1. **Every piece fits every socket.** The pockets are all 50 mm square and
   both cargo cubes are 45 mm: geometric insertion — the seed's entire
   strategy — always succeeds mechanically and proves nothing. The flagship
   smoke check drops the cubes into wrong pockets exactly the way the solve
   drops them into right ones: both seat cleanly (both latches fire), and
   the task still fails.
2. **The pairing is ARITHMETIC, not geometric.** The sockets ride a gravity-
   centred balance beam and the two cubes have stated, color-coded masses in
   a 2:1 ratio (orange 0.32 kg, blue 0.16 kg — printed in `describe()`,
   nothing hidden). Success requires the loaded beam to read LEVEL, which by
   the lever law admits only assignments with orange at radius r and blue at
   radius 2r on the opposite arm. The verifier is the beam itself: any wrong
   assignment leaves ≥ 0.094 N·m and the beam settles ≥ ~10.7° tilted or on
   a ±14° stop, versus a 4.5° level tolerance (≥ 2× above worst in-pocket
   play, ≥ 2× below the mildest wrong pairing).
3. **A per-episode occupancy constraint replaces the per-piece cutout.** One
   RED locking plug occupies one pocket on each arm (6 legal configurations,
   left radius ≠ right radius), its mass factory-matched so the plug torques
   cancel (mass × radius = 0.0192 kg·m both sides). The plugs prune which of
   the lever-law assignments survive — the solver must read the occupancy
   and re-derive the answer each episode — and "do not dislodge the plugs"
   is physical, not decorative: remove one and the beam falls to its stop.

Difference from the other packages examined:
- `place_shape_in_shape_sorter_i180` (feed_line_cull, same seed): its core
  is selection-by-removal plus mechanism-and-gravity conveyance (pull a pin,
  the queue delivers itself). Here nothing is culled and nothing conveys
  itself: both pieces are deliberately placed, and the novelty is that the
  placement is judged by static torque arithmetic.
- `push_cube_i344` (balance_triage): also has a beam scale, but there the
  masses are HIDDEN, the beam is an instrument for a weighing EXPERIMENT,
  and the goal state is cubes delivered to bins with the beam empty and
  self-levelled. Here masses are public, nothing is measured, the puzzle is
  a discrete constrained assignment, and the goal state is the cargo ON the
  beam holding it level — the beam is the goal, not the instrument.
- `weigh_beam_drop_in` / `weigh_press_wedge` recipes: both use weight as an
  actuator (a lever arm doing work on a mechanism). Here weight does no
  work in the goal state — the task is a torque-cancellation equilibrium.
- pick-and-place family (`lift_numbered_block_i153` etc.): end with an
  object held or placed at a location; here location per se is worthless —
  five of six seatable placements of each cube are failures, and which are
  correct changes with the plug episode.

## The scene

A 14 kg steel GANTRY (200×240 mm slab, two uprights outboard of the beam's
swing at y = ±55 mm, crossbar) carries a Y-axis revolute hinge 165 mm up.
From it hangs a 0.90 kg BEAM: a 600×62×16 mm bar whose centre rides 45 mm
below the hinge (all on-beam CoMs below the axis → gravity is the centering
spring, k_eff ≈ 0.5 N·m/rad; PhysX angular damping 2.5 gives ζ ≈ 0.4, the
verdict settles in ~3 s), joint hard stops at ±14°. Each arm carries three
open-top pockets (50 mm square inner, 5 mm walls, 16 mm tall) at radii 60,
120, 240 mm. One RED plug (47×47×30 mm) per arm sits in a sampled pocket;
its mass is written to PhysX at reset as 0.0192/r kg (readback-verified).
Two cargo cubes (45 mm) rest on the ground: ORANGE 0.32 kg, BLUE 0.16 kg.

Randomization (readback-checked, 4 seeds max-pairwise in smoke): gantry xy
±40 mm + yaw ±20°; plug pocket pair (6 configs; masses follow the pockets);
both cube spawn xy ±50 mm + free yaw.

## Success

All of, live and settled: orange seated AND blue seated AND both plugs
seated (beam-frame: |x| within 16 mm of a slot radius, |y| < 12 mm, cube
z ∈ [20, 42] mm — excludes wall-perch at 46.5 and cube-on-plug at 60.5 —
plug z ∈ [14, 32] mm), beam level (|tilt| ≤ 4.5°), gantry upright,
everything still (60-substep counter) and finite.

Score: 0.30·orange_seated_ever + 0.30·blue_seated_ever (latched, cap 0.60);
1.0 iff success. Null policy ≈ 0.

## Solve phases (teleports = transport only; verdicts by gravity + contact)

- **P0** reset + settle 90; audits: plug mass readback on the ledger
  (m·r = 0.0192 both arms), plugs seated, beam level, cubes unseated,
  score 0. Derive the torque-cancelling assignment from the LIVE plug
  seats (enumerate (side, r): r free on side, 2r free opposite). Score 0.
- **P1 load**: hover both cubes over their assigned pockets at beam-frame
  z = 50 mm — ABOVE the seated z-band, asserted at the write, so the seat
  latches can only fire from physics — release one step apart; each falls
  ~20 mm into its pocket before the beam can build tilt. Retry ladder: on a
  miss both cubes go back to the ground and the beam re-levels itself (the
  plug ledger). Score 0.60.
- **P2 verdict**: hands-off; the damped pendulum settles level (correct
  pairing → residual ≤ ~2.2° from in-pocket play ≪ 4.5°). success() holds.
  Score 1.00.
- **P3 persistence**: ≥ 3.3 sim-seconds untouched, success holds
  throughout, then `SIM_GEN_SOLVE: SUCCESS`. Passes on seeds 0 and 1.

Execution order: none enforced and none needed — cube placements commute
(drop order is irrelevant; a wrong placement is recoverable by lifting the
cube back out of its open-top pocket, and the beam re-levels itself
whenever the remaining load balances). No wrong order destroys feasibility.

## Franka embodiment (per object)

Plausible base pose: rig-local (0.0, −0.50, 0), facing +y toward the beam's
broad side. Every pocket centre lies within 0.66 m of that point at z ≈
0.14–0.16, and the cube spawns (nominal (0.15, −0.32) and (0.55, −0.32)
rig-local, i.e. ≤ 0.45 m from base) are nominal tabletop picks.

- **Cargo cubes (the only manipulated objects)**: 45 mm — inside the Franka
  gripper's 80 mm span. Ground pick is a standard top pinch. The drop into
  a pocket is top-down into an OPEN-TOP 50 mm socket whose walls are only
  16 mm tall with 2.5 mm clearance per side; the solve's release (cube
  bottom 3.5 mm above the wall tops, centred on the pocket) is exactly a
  gripper release at clearance height. Approach is unobstructed from above:
  the uprights are outboard at y = ±55 mm and the crossbar sits over the
  hinge, 25+ mm above the wall tops of the innermost pockets and far from
  the outer ones.
- **Beam**: never grasped; it swings freely below hand height and its
  ±14° sweep stays inside the gantry footprint — the arm works above it.
- **Plugs**: must NOT be touched — the task is to leave them seated.
- Forces: zero insertion force (gravity seats the cube); nothing exceeds
  the payload.

## Checks (smoke.py, 12) — success() audited never-True across the battery

1. Settle/no-NaN: plugs seated, plug-mass ledger readback, beam level,
   cubes unseated, score 0.
2. Randomization readback, 4 seeds max-pairwise: rig xy + yaw, both cube
   spawns, plug pocket pair all vary.
3. Null policy 240 steps: score ≈ 0, never success.
4. FLAGSHIP (seed strategy): cubes swapped (orange at 2r, blue at r) —
   insertion succeeds mechanically (both latches fire, score 0.60 cap) but
   the beam settles on its stop (−14.0°): not level, no success.
5. Same radius on both arms (the one plug-free radius): equal arms, 2:1
   masses → stop (−14.0°), no success.
6. Both cubes on one arm → stop (+14.0°), no success.
7. Half-load latch: orange alone pins the beam and latches 0.30; removed
   again, the beam re-levels itself and the 0.30 survives; never success.
8. Plug dislodged: removing one plug breaks the cancelling ledger — the
   beam falls to its stop, plugs_seated False, no success.
9. Balanced-but-unseated: blue perched ON a plug at 2r′ with orange seated
   at r′ opposite — torques cancel EXACTLY, the beam holds dead level and
   still, yet the cube z-band refuses the perch: no success, score 0.30.
   (Config-dependent; the battery seed-searches a qualifying episode.)
10. Bare-bar rest: blue on the un-walled bar near a balancing abscissa
    (x = 0.18) — no slot radius there, never seated; the beam also tips
    (+10.35°, matching the ~10.7° margin-ledger prediction); no success.
11. Rejection audit: success() never True at any step of the battery.
12. frames.npz written (170 frames).
