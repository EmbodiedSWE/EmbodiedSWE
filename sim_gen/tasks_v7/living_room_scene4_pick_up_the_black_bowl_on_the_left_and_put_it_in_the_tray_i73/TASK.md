# living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_i73 — Roller Freight

Scene: `roller_freight` · Env: `simgen.roller_freight` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed: `libero_90/living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray`
("pick up the black bowl on the left and put it in the tray"): two identical black bowls
disambiguated only by position; the solver grasps the left one and drops it in a tray.
Success is a bounding-box containment readout, the bowl weighs grams, and every motion is
a direct free-space grasp-carry-release — the object goes wherever the gripper takes it.

## What changed and why it is strategically different

The delivery problem is inverted into a FORCE-BUDGET problem. The black bowl is demoted to
loose cargo riding on the actual freight: a 7 kg, 30 × 22 × 6 cm stone slab that is
**un-liftable** (far beyond an ~80 mm jaw and an arm-scale payload) and **un-slidable** —
gritty slab on gritty channel floor needs μ_eff·m·g ≈ 1.15·7·9.81 ≈ 82 N, about twice the
40 N fingertip push budget the scene declares (`push_cap`, asserted ≥ 1.8× margin in
`__post_init__` and demonstrated in smoke #7–8: the full-budget shove launches off the
slick plinth, then jams on the grit short of mid-channel with the force pinned at the
cap). *Direct manipulation — the seed's entire strategy — cannot move the freight.* The
task forces the ancient logistics answer: **interposed rolling elements**. Three free
steel rollers (30 mm dia capsules) wait chocked in a side rack; the channel floor sits
exactly one roller diameter (+1 mm) below the slick launch plinth, so rollers staged ahead
of the plinth edge engage flush under the advancing slab nose. On rollers the same capped
push walks the slab 0.42 m to the yellow end stop. The support **migrates**: each engaged
roller translates at half the slab's speed and drifts rearward through the slab frame, so
placement must anticipate the support schedule — the rust cargo block biases the slab CoM
4 cm forward (authored explicitly; MassAPI CoM) so the migrating bed straddles the CoM
through the plinth hand-off and at the dock. Success = slab nose within 3.5 cm of the end
stop, aligned, **still riding its roller bed**, black bowl still upright aboard, all at
rest.

Distinct from the corpus (survey of every tasks_v7 card): no existing task interposes
rolling elements between cargo and ground, and none is decided by a friction/force
economics budget. The nearest neighbours all SLIDE on fixed paths (cellar-tow i43 tows a
slick-bottomed sled, trolley i62 and tunnel-shuttle ride captive channels, hockey herds a
puck); none makes the support itself migrate under the payload as a kinematic consequence
of moving it, and none makes the same push succeed or fail purely on what is interposed
under the load. The seed's own strategy is a settled reject state: the bowl alone carried
to the dock scores ~0 (smoke #9 — the bowl is cargo, not freight).

## Teleport-solution outline (solve.py)

1. **P0** settle + layout readback (fixture yaw ±25°, xy jitter, slab depth, bowl spot);
   assert bowl aboard, no roller staged, score ≈ 0.
2. **P1 stage (transport only)**: each roller carried from its chock and released with
   zero velocity 1.5 mm above the channel floor at fixture-local x = 0.00/0.10/0.20,
   axis across the channel; landing verified (a free capsule can roll) with up to 3
   re-carries of the still-free body. Gravity seats the bed → `staged` latches (0.20).
3. **P2 push (applied force)**: a horizontal velocity-servo force on the slab
   (`F = KP·(0.08 − v)` clamped [0, 39 N] < `push_cap`, aimed along the channel axis, no
   torque — the walls do the aligning), routed through `encode_force` with a runtime
   force-frame probe (mode 1 first; no launch off the *slick* plinth within 2 s ⇒ roll
   back to the staged snapshot and lock mode 0). The slab launches at ~4 N, its nose
   drops one roller diameter onto the bed, and the rollers carry it — the SAME capped
   push smoke proves jams without them. `riding` (0.45) and `half` (0.70) latch on the
   way. If the migrating bed starves (slab at rest, force at cap), a roller that has
   fully EXITED behind the slab rear is re-fed ahead of the nose (single unverified
   drop — transport of a free body; at most 2 refeeds).
4. **P3** forces cleared; the freight coasts against the end stop and settles → success,
   `SIM_GEN_SCORE 1.0000`. All teleports are transport-only; every rubric fact (staged
   bed, riding height, dock contact) is produced by gravity/contact/the capped push.
5. **P4** hands-off persistence 400 steps (3.33 s at 120 Hz); `SIM_GEN_SOLVE: SUCCESS`
   only if success still holds. `SIM_GEN_SCORE` printed at every phase boundary,
   non-decreasing (latched credit, serialized through get/set_state).

## Embodiment argument (Franka, base at ~(0, −0.45), facing the yard)

Everything lies within ~0.75 m of the base: rack slots at (±0.15, 0.275), slab rear face
between x = −0.35 and +0.07 on the channel centerline, dock at x = 0.37. Per-object
contact strategy:

- **Rollers** (30 mm dia, 24 cm, 0.25 kg): parallel-jaw pinch across the diameter — 30 mm
  is well inside an ~80 mm jaw span, and the 10 mm chock battens leave finger clearance
  under the exposed mid-section. Lift out of the open-top rack, free-space carry over the
  wall (5 cm tall), release just above the channel floor — exactly the write the solve
  performs (zero-velocity release, 1.5 mm drop).
- **Slab** (7 kg, 22 cm wide): never grasped. A fingertip/knuckle push on the rear face at
  CoM height, one DoF along the channel, capped at 40 N — inside an arm-scale sustained
  push and matching the solve's torque-free force servo. The channel walls, not the hand,
  keep it aligned.
- **Bowl** (black, 9 cm, 0.15 kg): never touched — it rides the deck; keeping it aboard
  and upright is a constraint on how gently the slab is accelerated (the 0.08 m/s servo).
- No bimanual holds, no force beyond 40 N, no reorientation of anything but the carried
  roller (held horizontal throughout).

## Execution order declared

Built in this order: (1) minimal success() predicate + scene, (2) working solve iterated
on the forge (capsule rollers for smooth rolling, kinematic fixture, roller damping as
rolling resistance, drop verification), (3) final rubric (latched weights
0.20/0.25/0.25, 1.0 iff success), (4) smoke battery.

## Checks (smoke.py)

17 checks: settle/no-NaN baseline · randomization readback 8 seeds (fixture xy+yaw, slab
depth + roller slot jitter, bowl spot) · null-policy ~0 · force-economics pair (the capped
shove MOVES on the slick plinth — live-actuator proof — then JAMS on the grit short of
mid-channel at the full 40 N) · SEED-strategy bowl-alone-to-dock reject · dock-by-fiat
slab-on-floor reject (riding/under gates) · near-miss riding bed parked short of the dock window = exactly the
0.70 partial credit · acceptance construct (riding bed in the dock window → success TRUE)
· cargo-clause flip pair (bowl off → FALSE, restored → TRUE) · settle-gate kick ·
rejection audit · final no-NaN · frames.npz saved.
