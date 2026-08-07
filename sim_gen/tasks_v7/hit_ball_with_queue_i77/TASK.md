# skyway_bridge (`hit_ball_with_queue_i77`)

Seat a bridge span across the broken elevated track, then gently release the captive
ball so gravity carries it the whole way down into the walled catch basin.

## Seed provenance

- **Seed:** `rlbench/hit_ball_with_queue`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/hit_ball_with_queue.py`) — grasp a
  cue stick and STRIKE a free ball across an open surface into a pocket.
- **Seed strategy:** tool-mediated impulse transfer. One ballistic contact; aim and
  momentum decide the outcome; the surface between ball and pocket is continuous and
  passive.

## What changed and why it is strategically different

The seed's core verb — *propel the payload at the goal* — is inverted into
*construct the environment so the payload propels itself*:

1. **The robot never carries or propels the payload to the goal.** The ball's entire
   journey (slope 1 → bridge → slope 2 → flight → basin) is passive gravity + contact.
   The robot's only ball contact is a slow, velocity-limited nudge over an 8 mm detent
   ridge at the very start.
2. **Striking is the losing move.** The track is BROKEN mid-span; a struck ball
   cannot jump the 18 cm break (that would need > 6 m/s) and falls through the gap to
   the floor, where it is **unrecoverable**: the 90 mm ball is wider than the 80 mm
   Franka jaw span (asserted in `SkywayBridgeSceneCfg.__post_init__`) and the basin's
   ≥ 12 cm vertical walls cannot be rolled up. smoke check 6 executes exactly the
   seed's plan and shows it scores ~0.
3. **The real work is infrastructure with a selection problem:** pick the correct
   yellow span by LENGTH (16 cm bridge vs 9 cm decoy — the decoy is shorter than the
   11 cm tab-tip opening and bears on NEITHER tab), carry it by its 12 mm gantry
   handle bar, and SEAT it across the gap onto two recessed support tabs between
   lateral guide walls: a real contact placement under tolerances (±20 mm
   longitudinal, guide-slot-capped lateral, bearing height, upright + axis-aligned).
4. **Execution order is geometry-forced, not rubric-decreed:** success() judges only
   the physical outcome (ball at rest inside the basin), but a ball released before
   the bridge is seated is physically unrecoverable — so bridge-first is the only
   order that can ever succeed.

Distinct from the corpus siblings read this session:

- **hockey_i325** (gate extraction + floor push into a roofed goal): there the robot
  pushes the payload the whole way to the goal after removing a barrier. Here the
  robot *adds* a structure and the payload travels hands-off; the manipulation target
  is the infrastructure, not the payload.
- **track_bowl_i27** (bell cage drag/pocket delivery): payload is dragged/carried by
  the robot throughout. Here carrying the payload is impossible by construction.
- **obstacle_i17** (ballistic launch to topple a pin): that IS aim-and-impulse — the
  strategy this task makes the explicit losing move (smoke check 6).

Loophole audit: using the bridge as a ramp to roll the fallen ball up into the basin
is impractical (the 16 cm span against the 12 cm near wall is a ~48° ramp, and the
ball cannot be grasped to be placed on it); dropping the ball into the basin directly
is impossible (it can never be lifted). The decoy fits through the gap and can never
substitute for the bridge.

## Teleport solution outline (solve.py)

1. **P0** reset + settle; layout readback printed (seed-provable); score ~0 asserted.
2. **P1 (teleport = transport only):** one pose write stages the bridge HOVERING
   40 mm above its seat, axis-aligned, with a deliberate 6 mm lateral offset —
   asserted NOT seated.
3. **P2 (contact dynamics):** release → the span falls onto both tabs under gravity,
   rocks, settles; the scene's `_bridge_seated` readback (pose + axes + stillness)
   confirms → `SIM_GEN_SCORE 0.30`.
4. **P3 (contact dynamics):** velocity-limited horizontal force at the ball CoM
   (fingertip-nudge emulation, 2.5 N base, v ≤ 0.3 m/s) walks it over the ridge; the
   force is CUT the moment it clears. A runtime probe toggles the wrench pre-encoding
   (`R_ref·R_now^T`) if the pod's external-force API drags the frame — both stall- and
   backward-motion-triggered (the ball rolls ~70° reaching the ridge).
5. **P4 (hands-off):** nothing is touched; the ball rolls, crosses the seated bridge
   (crossing latch requires the bridge seated UNDER the moving ball), flies off the
   lip over the low near wall, lands in the basin, settles → `SIM_GEN_SCORE 1.0`.
6. **P5:** ≥ 3.3 further simulated seconds hands-off; success must persist →
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge on **seeds 0 and 1** (rc=0; distinct layouts in the readback;
frame-drag toggle exercised on both).

## Embodiment argument (single Franka, OSC, parallel jaw)

- **Base pose (assembly frame):** (u, v) = (−0.15, −0.45) — all contacts below are
  within ~0.75 m reach: ground slots at v=−0.38, the gap at (0,0) with seat height
  0.22 m, the deck at u∈[−0.54,−0.36] height 0.26–0.31 m.
- **Bridge:** the 12 mm gantry handle bar sits 0.115 m above the span floor — a clean
  top-down parallel-jaw pinch (12 mm ≪ 80 mm opening), with the bar clear of the
  rails and above the ball's crown so the channel stays open. Lift, carry, lower
  between the guide walls (a ±10 mm lateral funnel that forgives OSC placement
  error), release when the tabs take the load.
- **Ball:** 90 mm > 80 mm jaw span — never grasped. The only contact is a fingertip
  push at deck height (0.26–0.31 m, comfortable Franka workspace) with ~1–2 N to
  climb the 8 mm ridge (quasi-static F > m·g·d_h/(r−h) ≈ 1.0 N); the closed fist /
  fingertip pushes horizontally at CoM height, exactly what the solve's
  velocity-limited CoM force emulates.
- **Decoy:** never needs touching.

## Execution order (declared)

REQUIRED order: (1) seat the bridge, (2) release the ball. The order is enforced by
irreversibility (see above), not by the rubric; the rubric's crossing latch
additionally credits only a ride over a seated bridge.

## Rubric

- `0.30 seated` (latched) — the BRIDGE ever seated on both tabs (centred/upright/
  aligned/at height/still).
- `0.35 crossed` (latched) — the ball ever over the gap at track height WHILE the
  bridge is seated under it.
- `1.0 iff success()` — ball at rest inside the basin. Non-success cap 0.65. Null
  policy ~0.

## Checks (smoke.py — rejection-only battery, recorded to frames.npz)

1. settle/no-NaN: reset layout settles finite, ball on deck, spans on ground, still
2. reset score ~0, no success
3. randomization readback: whole-assembly yaw + xy offset real
4. randomization readback: bridge/decoy slot swap flips; per-span + ball jitter real
5. null policy (240 steps) → score ~0
6. SEED STRATEGY: ball struck at 1.5 m/s with no bridge → falls through the gap to
   the floor → score ~0 (striking loses)
7. out-of-order: gentle release before bridging → through the gap, unrecoverable →
   score ~0 (order is forced)
8. near-miss seat (longitudinal, 30 mm > 20 mm tol): span tips into the gap → NOT
   seated
9. wrong object: decoy dropped centred over the gap falls through → no credit
10. near-miss seat (skew 30°): cannot enter the guide slot, rests high/canted → NOT
    seated
11. cross gate: ball over the gap with NO seated bridge → crossed latch never fires
12. beside-wall: ball on the ground beside the basin → frame/z math rejects
13. latched credit: probe-seated bridge earns exactly 0.30, survives bridge removal,
    never success (cap holds)
14. rejection audit: success() never True anywhere in the battery
15. final no-NaN

`SIM_GEN_SMOKE: ALL PASS 15/15` expected; solve.py separately proves acceptance.
