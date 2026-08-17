# pawl_ladder_i431 — meter the gravity cart down to the commanded station

`sim_gen` task `libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i431`
(scene `pawl_ladder_i431`, env `simgen.pawl_ladder_i431`).

## Provenance

Seed: `libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet` —
"open the top drawer of the cabinet". One prismatic DOF; the checker is a
`JointPosChecker` on the drawer joint, mode `le`, threshold −0.1: grab the
sliding body and haul it past a one-sided position threshold. Any pull that
overshoots still succeeds — more displacement is strictly better.

## What changed, and why it is strategically different

The prismatic DOF is kept; everything about how you are allowed to move it is
inverted:

- **The DOF is self-driving, not hauled.** The amber cart sits on an 8°-pitched
  rail and is pulled downhill by gravity the whole episode. Nothing ever needs
  to (or may usefully) push the cart: the seed's entire strategy — grasp the
  slider, pull along the joint axis — is physically blocked by a **two-way
  detent**: a gravity-seated pawl blade standing in a rack notch on the cart.
  Smoke check 6 hauls the cart with 3.2× the ramp-gravity load in BOTH
  directions and the cart only rattles within the notch free play.
- **The goal is a position BAND at a sampled interior station, not a one-sided
  threshold.** The scene samples k ∈ {2,3,4} each reset and plants a GREEN post
  on the fin-alignment line of station k and a RED post at station k+1. Stopping
  short (k−1) fails; passing the red line (> q_target + 30 mm) latches a
  **permanent overshoot foul** that zeroes the score forever — the seed's
  "overshoot is free" property is exactly negated.
- **The only productive actuation is on a *different* body than the goal DOF.**
  You operate the pawl's red T-handle (lift ≈ 20 mm, release), k times. Each
  lift lets the cart run; releasing early drops the blade onto the passing land
  so it rides into the NEXT notch and arrests the cart one station further —
  release-timing metering, with an asserted 1.67× timing margin (catch window
  vs blade drop time).
- **A fragile rider raises the stakes.** A loose ball rides an open tray on the
  cart; success requires it still aboard, so slamming/jerking strategies shed
  the rider and fail (smoke check 10).

Differentiation from the tasks_v7 corpus reviewed while building this (stated
in the scene docstring as well): i34 `gumball_meter` counts *dispensed objects*
through an airlock; i283 props a sash with a *one-way* ratchet; i307 uses a
one-way pawl as a gravity press; i279 tilts a hopper; i74 folds a screen; i386
is a press-and-twist bayonet where the actuation is *on the sliding body
itself*. Here the sliding body may never be touched productively: the task is
**closed-loop position regulation of a continuously falling load by operating
its brake**, with an irreversible overshoot foul and a count sampled per
episode.

## Teleport solution (solve.py)

Zero teleports, zero forces on the cart. The only actuation is a
velocity-regulated vertical force on the pawl (body-frame z = the guide axis;
gains respect the one-substep wrench delay, KV·dt/m ≈ 0.17), re-set every step
and zeroed before judging:

- **P0** settle + readback: the spawned cart glides a few mm and is ARRESTED by
  the seated pawl (the dead-man demo); read k; assert score ≈ 0.
- **P1..Pk** metering cycles, one notch each: lift the pawl ~22 mm; when the
  cart has run ~16 mm, release; the blade rides the land and drops into the
  next notch. A same-notch recatch is detected by station readback and retried
  (the latches make score monotone). `SIM_GEN_SCORE` printed at each arrival,
  asserted non-decreasing and ≥ the latched arrival credit.
- **P(k+1)** ring-down: fin at the green post, pawl seated, ball aboard,
  settled → success.
- **P(k+2)** hands-off persistence ≥ 3.3 simulated s, then
  `SIM_GEN_SOLVE: SUCCESS`.

Passed on forge, seeds 0 and 1 (k=2 both; station pose differs; k spread over
{2,3,4} verified by readback in smoke check 4).

## Embodiment argument (single Franka, parallel-jaw gripper)

- **Pawl T-handle (the only required contact):** a 16 mm square-section red
  crossbar, 90 mm wide, at ≈ 0.34 m above the table, faces clear on all sides —
  a canonical top-down or side-on parallel-jaw grasp. The entire task is: grasp
  the crossbar, lift ~20–25 mm along the (near-vertical) guide, hold ~0.1–0.3 s,
  lower/release. Repeat k ≤ 4 times. Forces are ~2 N — trivial for the arm.
  The gripper may equally hook *under* the crossbar with a closed jaw and lift.
- **Nothing else needs touching.** The cart is driven by gravity; the ball
  never needs handling; the posts are kinematic markers.
- **Base pose:** ~0.5 m from the rail on the +y side (the gantry/handle side),
  centred on the pawl x-station. From there the handle is inside the sweet
  ~0.35–0.75 m reach annulus at all times (the handle is station-fixed — it
  does not travel with the cart), and the marker posts on the −y side stay in
  camera view.
- **Why the foul is honest for one arm:** recovering from an overshoot would
  require pushing the cart uphill *while* holding the pawl lifted — two
  simultaneous contacts a single arm cannot make; the scene therefore latches
  overshoot as a permanent foul instead of leaving an un-doable dead end
  ambiguous.

## Execution order

The order is physically forced, not conventioned: stations can only be reached
downhill, one notch per lift-release cycle (the blade cannot skip a land — the
asserted timing race guarantees the drop wins), and the foul latch makes the
sequence 0 → 1 → … → k the only legal trajectory. Latches (`lift_latch`,
`arrive_latch`, `foul_latch`) are order-aware: arrival credit only accrues
un-fouled, seated, and slow.

## Score

- 0 while nothing has happened; **0 forever once fouled** (checked first).
- +0.10 first genuine pawl lift (`lift_latch`).
- +0.50·(highest un-fouled arrival 1..k)/k (`arrive_latch`, latched → monotone).
- 1.0 iff `success()`: un-fouled ∧ |q − q_target| ≤ 12 mm ∧ pawl seated
  (≤ 4 mm) ∧ ball in tray ∧ settled ≥ 30 steps ∧ all states finite.

## Smoke battery (15 checks)

1. settle/dead-man (cart arrested at the station-0 rest by the seated pawl);
2. fresh score ≈ 0, no success;
3. randomization A: yaw/xy spreads by readback over 6 seeds;
4. randomization B: k varies in {2,3,4} AND both posts stand on fin-alignment
   lines recomputed from the station pose (≤ 3 mm);
5. null policy 240 steps — still at station 0, score ≈ 0;
6. SEED strategy blocked: ±3.5 N haul (3.2× ramp load) both ways — cart rattles
   only within notch free play, no foul, no arrivals, re-arrested at station 0;
7. one-cycle near-miss with the solve's own servo — score exactly 0.10+0.50/k,
   not success;
8. FLAGSHIP overshoot foul: free-run past the red line latches the foul; the
   exact success end-state is then rebuilt — success STILL False, score 0;
9. pawl-held-up: cart pinned at the target with the blade lifted — every clause
   passes except pawl-seated; success never fires;
10. ball ejected — seated at the green post but rider gone: not success;
11. wrong station (k−1): not success;
12. settle gate: goal pose written WITH real velocity — not success;
13. rejection audit: success() never True at any judged point above;
14. positive control (audit off): cleanly built goal state IS accepted
    (success True, score 1.0) — the rejections are non-vacuous;
15. no-NaN finiteness.
