# Chimney Catch — `approach_grasp_bowl_i412`

Scene name: `chimney_catch` (registered as `simgen.chimney_catch`, robot `"null"`).

## Provenance

Seed task: `pick_place/approach_grasp_bowl` — approach a bowl, grasp it, lift it.
The bowl and the "get the payload into the bowl" motif are kept; everything else is
replaced by an order-forced, irreversible catch-and-deliver mechanism.

## What the task is

A raised deck (z = 0.10) has **two open floor slots** (75 mm square, at
(±0.22, 0.10)); underneath each slot is a **sealed plenum** — anything that falls
through a slot is unrecoverable. Above **one** of the slots (the *live* side,
randomized per seed) stands a chimney tower whose chute holds a 30 mm ball. The
ball rests on a **sliding gate blade** protruding from the tower; the blade rides
on ribs between gantry posts and has an outboard knob. Pulling the blade outboard
(+y in blade frame) past `blade_open_y` releases the ball straight down — through
the deck slot into the plenum if the slot is uncovered, or into whatever covers
the slot. A free **bowl** (flat base, ring wall, inner r = 65 mm) starts on the
deck; a red **dock disc** on the deck (position randomized) is the delivery zone.

Goal: **cover the live slot with the bowl, pull the gate to drop the ball into the
bowl, then slide the loaded bowl onto the dock** and let it settle.

## Why this is strategically different

- vs. the seed (`approach_grasp_bowl`): the seed is a single grasp-and-lift of the
  bowl. Here the bowl is never lifted — it is a *tool* (a slot cover / catcher)
  that must be slid into three distinct places in a forced order, and the payload
  (the ball) is initially untouchable inside the tower.
- vs. `i133` ("Cliff Sweep"): i133 is a sweep-off-an-edge herding task with a
  cliff hazard on the way to a bin; there is no trigger mechanism, no captive
  payload, no cover-then-release ordering, and its loss branch is spatial (over
  the cliff) rather than a gated one-way drain under the work surface.
- vs. `i285` ("Wedge Press"): i285 is a force-threshold pressing task (drive a
  wedge to close a drawer); no free-rolling payload, no irreversible loss latch,
  no multi-station transport.
- vs. the `pen_holder` exemplar: that is insert-into-container; here the
  container itself is the mobile actor and the insertion happens *by gravity,
  through the machine*, only if the container was staged correctly first.
- Distinct mechanism class: **order-forcing via an irreversible drain**. The only
  path to the payload runs through a one-shot gate above a hazard; triggering
  before covering permanently caps the score (0.25). This makes the execution
  order physically load-bearing, not just declared.

## Declared execution order (physically forced)

1. **Cover** — push the bowl across the deck until it covers the live slot
   (`covered` latch, +0.25). Pushing is square-on with a staging waypoint so the
   bowl approaches the slot along one axis.
2. **Trigger** — push the blade knob outboard past `blade_open_y`; the ball drops
   into the bowl (`caught` latch, +0.35 → score 0.60). Triggering first instead
   drops the ball through the open slot into the plenum: irreversible `lost`
   latch, score capped at 0.25 forever.
3. **Deliver** — slide the loaded bowl off the slot and onto the dock disc;
   success = ball in bowl AND bowl on dock AND everything settled AND not lost
   (score 1.0).

The dead tower's blade is present but its chute is empty — pulling it does
nothing (checked in smoke); the agent must identify the live side (the ball is
visible in the live chute).

## Solution outline (solve.py)

All interaction is via external forces at the CoM (`is_global=True`), never
teleports:

- `push_bowl(target_xy)`: distance-proportional velocity servo with a
  **dry-friction feed-forward** (`0.55·m·g`, tapered near the target) plus an
  **escalating breakaway bias** (+3 N per 60 stalled steps, reset when the bowl
  moves > 0.06 m/s) — sized for the observed in-situ μ_eff ≈ 0.5. A 60-step brake
  finishes each leg. Staging waypoint first, then square-on final approach.
- Blade pull: small +y feed-forward (0.35 N) + velocity servo on the knob body
  until blade-frame y ≥ `blade_open_y + 3 mm`; forces cleared; 2 s free fall/settle.
- Delivery: exit push to mid-deck, then dock push (tol 8 mm), settle, then a
  10×40-step hands-off persistence hold before `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge for seeds 0 and 1 (live side +x and −x), scores strictly
staged 0 → 0.25 → 0.60 → 1.0.

## Franka embodiment argument

A Franka based at ≈ (0, −0.48, deck level) on the open −y side reaches the whole
work area: deck span |x| ≤ 0.35, y ∈ [−0.30, 0.25], all interaction heights
0.10–0.21 m. Required actions are planar pushes at bowl-wall height (~0.13 m,
well within payload — push forces ≤ 10 N) and a single outboard knob push at
0.198 m directed away from the robot (+y of the live tower faces −y world on the
blade table side, knob is outboard and unobstructed by the gantry posts). No
grasping, no lifting, no bimanual coordination is required; the end-effector can
do everything with a closed-fist push.

## Checks (smoke.py) — 17 named checks, rejection-only (forge: ALL PASS 17/17)

1. `settle_no_nan` — initial rest poses (ball on blade z≈0.198, bowl z≈0.100), score ≤ 0.02.
2. `randomization_readback` — seeds 21–28: both live sides occur, ball follows the
   live tower, physical dock body matches internal dock_xy (< 2 mm) and stays in zone.
3. `randomization_spread` — dock/bowl position + yaw spreads over seeds; latches zeroed on reset.
4. `null_policy` — 300 steps hands-off: everything stays put, score ≤ 0.02.
5. `trigger_first_lost` — PHYSICAL blade pull (asserts blade moved > 50 mm) with the
   slot uncovered → ball falls below `lost_z`, `lost` latched, score ≤ 0.02.
6. `lost_then_docked` — after loss, docking the bowl gives NOT success, score ≤ 0.25.
7. `dead_gate_noop` — opening the dead tower's blade leaves the ball on the live blade.
8. `wrong_slot_cover` — bowl over the DEAD slot: `covered` stays False.
9. `caught_not_docked` — legit cover + trigger: score ∈ [0.599, 0.60], NOT success.
10. `caught_latch_holds` — removing the ball from the bowl keeps the 0.60 latch, not lost.
11. `lost_cap_dominates` — ball placed in the plenum: score ≤ 0.25 despite other latches.
12. `docked_empty` — empty bowl on dock: docked True, score ≤ 0.02.
13. `ball_beside_bowl` — ball leaning outside the bowl at the dock: NOT in_bowl.
14. `dock_near_miss` — bowl just outside dock_tol with ball inside: in_bowl True,
    docked False, score ≤ 0.36.
15. `blade_hard_stop` — PHYSICAL inboard push (−0.8 N, 240 steps): blade travel
    caught by the end stop (4–20 mm), ball undisturbed, never lost.
16. `rejection_audit` — `ever_success` never fired across all constructs.
17. `final_no_nan` — all task-object states finite at battery end.

frames.npz recorded throughout (rgb, every 8 steps).
