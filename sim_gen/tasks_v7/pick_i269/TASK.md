# pick_i269 — Relay Tunnel (drive the red cube home with the blue cube as a tool)

## Seed provenance

- **Seed**: `mujoco_playground/pick` —
  `sim_gen/RoboVerse/roboverse_pack/tasks/mujoco_playground/pick.py`.
- The seed is "bring the box to the target": a Franka grasps a 4 cm cube and carries it
  through free space to a floating target pose. The target is an arbitrary *reachable*
  point; the entire plan is one grasp + one guided carry + one place, and the reward
  shapes gripper→box then box→target directly.

## What changed and WHY it is strategically different

The kept DNA: one small cube must end up at a designated target region, positions
randomized per episode. Everything about *how* is inverted:

1. **The target region is UNREACHABLE by the hand.** The goal is a sunken well (16 mm
   step down) at the dead end of a low, fully roofed, single-lane tunnel. The roof
   (66 mm interior height over a 50 mm cube) covers the entire channel *and the well* —
   no top-down place or drop anywhere past the mouth (smoke presses a cube onto the
   roof at 2× weight to prove the seal). The well starts 90 mm past the mouth plane;
   a closed parallel-jaw prong reaches ~60 mm. The seed's carry-and-place primitive,
   aimed at the target, scores zero.
2. **The last leg is OBJECT-AS-TOOL force transmission.** The unreachable distance is
   covered by pushing the BLUE cube into the channel behind the red one: blue becomes
   a rigid extension of the finger, transmitting the push through block-on-block
   contact until red overhangs the lip and drops into the well. The seed never routes
   force through a second object; here the relay is load-bearing (the rubric's `deep`
   band and the well both lie beyond fingertip reach).
3. **The well CAPTURES — delivery finishes by gravity, an anti-goal exists.** The step
   makes seating discriminative (seated z ≈ 25 mm vs on-slab 41 mm vs lip-straddle
   ~32 mm) and one-way against horizontal pushes. The well admits exactly ONE cube
   and the lane admits no passing, so pushing BLUE in first is an exclusive,
   unrecoverable wrong outcome (smoke constructs it: score stays at entered-credit).
   Object identity and *ordering* matter; the seed has one object and no ordering.

Also distinct from every other task read this session: `pick_i137` (Gravity Vault) is
a captive-slide interlock extraction plus a vertical gravity drop through a funnel —
no object-as-tool contact chain and no confined horizontal pushing; here the core is
single-lane relay pushing under a sealed roof into a step-capture well.

## Scene (procedural, no external assets)

One kinematic compound (the "garage": apron slab, slick ±y walls, back wall, full
roof; the well is the slab-free span between `well_x0=0.090` and `well_x1=0.175`) and
two identical-physics 50 mm / 0.08 kg dynamic cubes (red = target, blue = tool).
Honesty asserts in `__post_init__`: roof covers the whole channel + well; interior
height passes one cube but forbids riding over another; well shorter than two cubes;
seated z-window separates slab / lip-straddle / seated; a tumbling cube's corner
reach (with 12 mm over-push allowance) clears the back wall; spawn slots clear the
garage by a real margin. Randomized per episode (all `torch.rand`, smoke reads back):
garage xy ±30 mm, yaw ±20°; cube spawn slots jittered ±30 mm, yaw ±180°, sides
swapped. Walls are slick (μ 0.06) so pushed cubes square up instead of wedging.

## Rubric (latched partial credit, anchored in the solve trajectory)

- 0.25 `entered` — red ever inside the roofed channel (latched)
- 0.25 `deep` — red ever past half-depth `x > 0.080` (beyond a direct fingertip
  push; only reachable through the relay) (latched)
- 0.25 `seated` — red ever seated in the sunken well (latched)
- 1.0 iff `success()`: red seated in the well, settled, finite. Latches are chained
  (seated ⇒ deep ⇒ entered); non-success capped at 0.75 (smoke drives the cap and
  checks it to float32 exactness). Null policy ~0.

## Teleport solution (solve.py) — phases

- **P0** settle + layout/mass readback (cubes on the open floor, score ~0).
- **P1** *(transport, then force)* red is teleported to the open apron at the mouth
  (staging a pick-and-place the arm does trivially; `stage()` asserts every teleport
  target is apron-side of the mouth plane), then a velocity-servo push (≤3 N,
  fingertip-frame) drives it inside to the documented fingertip reach x ≈ 0.060 →
  `entered`, 0.25.
- **P2** *(transport, then force)* blue is staged on the apron behind red, then
  pushed (≤4 N, 0.045 m/s velocity servo, gain-escalating stall watch, per-step
  live-quat body-frame force encoding with an encoding-jam-gated world/body toggle).
  All red progress comes through blue-on-red contact: `deep` latches at 0.50 (P2a),
  then red overhangs the lip and drops into the well — `seated`, 0.75 (P2b).
- **P3** settle → live `success()` → `SIM_GEN_SCORE 1.0000`.
- **P4** hands-off ≥ 3.3 simulated seconds → `SIM_GEN_SOLVE: SUCCESS`.

No teleport crosses a barrier: both stagings land on the OPEN apron outside the
mouth plane, poses an arm reaches by an ordinary grasp-carry-place of a 50 mm cube.

## Embodiment argument (Franka, base at world origin; garage at (0.42, 0) ± jitter)

- **Cubes**: 50 mm on open floor — within the 80 mm jaw span; staging poses are
  flat places on the apron at z ≈ 41 mm, 0.2–0.4 m from the base.
- **Finger push (P1)**: the mouth is 68 mm wide × 66 mm tall; a closed Franka jaw
  prong (~20 mm wide, ~60 mm usable reach past the knuckle before the hand body hits
  the mouth frame) pushes the cube ~45 mm inside at ≤3 N — well under the arm's
  force envelope.
- **Relay push (P2)**: pushing blue's exposed apron-side face needs only fingertip
  contact at the mouth plane; force ≤4 N along the channel axis.
- All interaction points lie within 0.25–0.60 m of the base at z ≤ 0.10 m — inside
  the Franka workspace with margin.

## Execution order

Declared: FORCED order, red first (describe()/instruction() say so). The lane is
single-width and the well single-occupancy: blue-first seats the wrong cube
irreversibly (smoke check 8 constructs it and asserts no success and no seated
credit for red).

## Checks

- solve: `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1 and 2 (forge; seed 2 draws the
  swapped spawn sides), scores non-decreasing 0.00 → 0.25 → 0.50 → 0.75 → 1.00.
- smoke: `SIM_GEN_SMOKE: ALL PASS 11/11` (forge) — settle/no-NaN + mass readback,
  randomization readback (+ side-swap coverage over seeds 0..7), null policy,
  roof-seal press over the well, near-miss entered (0.25 only), near-miss deep
  (on-slab z rejected at 0.50), lip-straddle rejected (the held analytic
  straddle pose has x in the well span but z above the seated window; released it
  topples flat into the well — physically adjacent to seating), blue-seated wrong
  outcome (no success, red credit
  stays 0.25), sliding transit + exact 0.75 non-success cap (float32), rejection
  audit (success never True in the battery), frames.npz video.
