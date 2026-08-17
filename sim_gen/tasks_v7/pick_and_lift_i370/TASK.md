# Task `pick_and_lift_i370` — Roller-Relay Freight

**Env id:** `simgen.roller_relay_freight` · **Robot:** null (scripted physics solution)

## Seed provenance

Seed task: `rlbench/pick_and_lift` — pick up a small block and lift it to a hover
target. The seed's whole strategy is *grasp the goal object and move it through
free space*.

## Why this is strategically different

Here the goal object — a 7 kg freight slab (0.260 × 0.160 × 0.050 m) — is far too
heavy and too wide to pick up, and the rubric never asks for a lift. Instead the
task inverts the seed's relationship between "the thing you grasp" and "the thing
the task is about":

- **What gets picked and carried** are three light *tool* rollers (Ø36 mm,
  0.14 m long, 0.18 kg — comfortable single-jaw grasps), which are never judged.
- **What is judged** — the slab — must be moved 0.625 m down a rubber channel it
  cannot slide on (μ ≈ 1.2 rubber vs a ≤ 14 N push against a ~55 N breakaway
  requirement: sliding is force-infeasible, asserted in cfg). The only way to
  move it is to *build a rolling road under it*: lay the rollers across the
  channel, push the slab onto them from the slick start dock, and let it ride.
- Rollers translate at **half** the slab's speed, so the roller road is consumed
  as it is used — spent rollers emerge behind the slab and must be relayed ahead
  of it (or retired). The channel length, roller count, and spacing are tuned
  (and cfg-asserted) so the support-span timeline closes exactly: the lead
  roller is still on the mat, ahead of the slab's centre of mass, when the
  slab's nose reaches the goal dock.
- Delivery is a *step down*: the goal dock sits 2 mm below the ride plane, and a
  pit between mat and dock swallows the last rollers below both surfaces, so the
  slab lands flat on the slick dock and coasts to the bumper. Success = all four
  slab corners over the goal dock, flat, at dock height, settled.

No corpus task uses a consumable rolling-support relay; the closest relatives
(plank-bridge span, gauge-adapter chain) build *static* infrastructure once,
whereas here the infrastructure moves with — and is consumed by — the payload.

## Teleport solution (`solve.py`) — the legitimacy certificate

Teleports transport the **tool rollers only**, always ending in free space
(hover ~9 mm above the mat, or back on the open staging floor). The judged slab
is **never** teleported or posed; all of its motion is contact dynamics under a
regulated horizontal push (velocity servo, ≤ 14 N at the slab centre, plus a
light lateral/yaw keeper), cut entirely for parking, settling, and persistence.

- **P0** — reset (seeded), settle, layout readback, baseline score.
- **P1** — lay the three rollers across the rubber channel at x = 0.035 / 0.125 /
  0.215 (drop 9 mm, seat by gravity). Score: staging credit.
- **P2** — servo-push the slab off the slick start dock onto the rollers
  (flush boarding: dock top == roller crown, asserted). Score: aboard credit.
- **P3** — ride the relay: spent rollers that emerge *clear behind* the slab are
  picked from free space and relaid ahead of it while the channel still needs
  support, then carried back to staging once the slab's nose is over the goal
  dock. Slab lands on the dock, push is cut, it settles flat against the bumper.
  Score: advance credit → success.
- **P4** — hands-off persistence ≥ 3.3 simulated seconds, then
  `SIM_GEN_SOLVE: SUCCESS` only if `success()` still holds.

`SIM_GEN_SCORE` is printed at each phase boundary and asserted non-decreasing
(the scene latches credit).

## Embodiment argument (single Franka, parallel jaw ≤ 80 mm)

- **Rollers**: Ø36 mm cylinders (jaw closes at 36 mm ≪ 80 mm), 0.18 kg, resting
  on an open staging floor at y = ±0.27 with 0.09 m pitch — top-down grasps with
  full clearance. Laying = place-and-release 9 mm above the mat. Spent rollers
  are retrieved from *free space behind* the slab (the relay only touches
  rollers that have emerged clear of the slab footprint) — again open-top
  grasps.
- **Slab**: never grasped. The Franka presses on the slab's rear 160 × 50 mm
  face; the required push is 4–14 N horizontal — trivially inside the arm's
  wrench capability, and the velocity-servo profile is exactly what an
  impedance-controlled end-effector push produces.
- **Base pose**: one plausible fixed base at (x ≈ 0.30, y ≈ −0.42), facing the
  channel. From there the reach envelope covers the staging floor (y = −0.27
  side; the scene's per-seed `side` flip mirrors staging to whichever side is
  sampled — mirror the base accordingly), the whole 0.79 m track for pushes,
  and the behind-slab pickup zone.

## Execution-order declaration

The order is forced, not stylistic: rollers **must** be staged before the slab
leaves the start dock. Once the slab is on the rubber with no rollers under it,
it is force-infeasible to move (≤ 14 N vs ~55 N breakaway) — an irreversible
dead end, which `smoke.py` constructs and verifies (check 8). Relaying must
happen while spent rollers are behind the slab and the nose has not yet reached
the dock; afterwards they are retired to staging.

## Rubric (score / success)

- 0.10 — rollers staged across the channel (per-roller latch, axis across the
  track, slow).
- 0.15 — slab aboard the rollers (flat, bottom at ride height, in-channel).
- 0.55 — advance fraction toward the goal-dock centre (latched running max,
  gated on riding/landing posture).
- Cap 0.80 without success; exactly 1.0 on success (all four corners over the
  goal dock, flat within 6°, bottom within 8 mm of dock top, settled).

## Checks

`smoke.py` — 15 rejection/validity checks: settle & no-NaN; reset score ~0;
randomization readback (slab jitter/yaw, roller jitter, side flip) over 8 seeds;
null policy ~0; seed strategy (1.5× weight hoist + hold: lifts the slab, earns
≤ 0.02, no success — picking-and-lifting the goal object is worthless here);
post-release score still ~0; channel dead-end force-infeasibility (35 N moves
the slab < 3 cm on rubber) with a paired probe-validity check (same push moves
it > 5 cm on the slick dock); near-miss xy (corners short of the dock);
near-miss z (slab perched on rollers *on* the dock — right place, wrong height,
not flat-on-dock); wrong object (rollers delivered instead of slab); latched
staging credit survives roller removal; rejection audit (no probe ever tripped
success); final no-NaN. Prints `SIM_GEN_SMOKE: ALL PASS 15/15`.
