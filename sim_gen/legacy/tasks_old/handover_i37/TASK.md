# handover_i37 — Rail Ferry (`simgen.rail_ferry`)

## Seed provenance

- Seed id: `mujoco_playground/handover`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/mujoco_playground/handover.py`
- Seed strategy: a bimanual ALOHA handover — ONE box is picked by the left arm, transferred
  **gripper-to-gripper in mid-air** at a fixed handover point on the workspace boundary, and
  placed by the right arm on a floating target. The entire task is a *direct aerial transfer
  of the object across a workspace divide*, reward-shaped on gripper–box proximity and the
  handover-point approach.

## What changed and WHY it is strategically different

The derived task keeps the seed's *situation* (cargo starts on side A, its goal is on side B,
something must carry it across a divide) but **inverts the seed's plan into the tested failure
mode**. A solid wall splits the workspace; the only legitimate way across is a **rail-bound
ferry shuttle** riding an overpass bridge channel:

1. load a parcel into the shuttle basin at a bridge-end station,
2. drive the shuttle along its rail channel across the divider,
3. lift the parcel out and place it on the delivery pad,
4. send the empty shuttle back and repeat — the one-parcel basin forces **round trips with
   empty return legs** for the 2–3 parcels.

A permanent per-parcel `violated` latch trips the moment a parcel crosses the divider plane
**not riding the seated shuttle**. That kills, and the smoke explicitly tests:

- the seed's own strategy — carrying the box **through the air over the boundary** and setting
  it perfectly on the target (negative control A: perfect end pose, score ≤ 0.05, permanent);
- the reduction of this task back to the seed — treating the **loaded shuttle as one big box
  and handing IT across** through the air (negative control C: crossing while unseated);
- the ground detour around the wall end (negative control B: the rubric judges the plane, not
  the wall).

A solver therefore needs a qualitatively different plan: **vehicle logistics** (fetch the
ferry if it starts at the far station, schedule load → ride → unload cycles through a
single-capacity carrier), not a mid-air object transfer. Same-strategy-different-numbers is
impossible: executing the seed's plan with any parameters scores ~0 here.

Claimed strategy axes (checked against sibling TASK.md files / batch notes, none claimed):
vehicle-mediated transfer (cargo must *ride* a carrier through the crossing), ride-gated
crossing latch, single-carrier round-trip scheduling.

## Difficulty tier: hard

Declared stage count: **8–11 ordered stages** depending on the sampled parcel count k ∈ {2,3}
and shuttle start side — (optional ferry fetch) + per parcel {load, ride across, unload+place}
+ (k−1) empty return legs. **Execution order is required**: the single-capacity shuttle
serializes the parcel pipelines, each parcel's load must precede its crossing, and its clean
crossing must precede its delivery (anti-teleport: `delivered` requires the latched clean
crossing). Irreversibility: any off-shuttle crossing spoils that parcel permanently.

## Randomization (per episode)

- parcel count k ∈ {2, 3} (subset-sampled; absent parcels park in an off-camera depot),
- parcel spawn poses (slot + xy jitter + free yaw),
- delivery pad position on side B (x band + both lateral signs),
- shuttle start station (near or far end — far start prepends a ferry-fetch stage).

## Rubric

Per present parcel: 0.15 once loaded in the seated shuttle (latched), 0.45 once ridden
cleanly across (latched), 1.0 while resting settled on the pad (physical, current state);
capped at 0.03 forever once violated. Score = mean over present parcels, capped at 0.95
unless `success()` (every present parcel delivered) → exactly 1.0. Null policy scores 0.
Success and delivery judge PHYSICAL outcomes (settled poses on the pad, real ridden
crossings detected per substep); the oracle stages objects kinematically but every drop,
ride, and set-down settles under real physics.

## Check list (smoke battery, 18 checks)

1. settle/no-NaN: reset layout finite, shuttle seated, parcels at rest (1) + score ~0 (2)
2. randomization-is-real by READBACK: parcel scatter, pad pose, shuttle side, parcel count (3)
3. null-policy-fails: 240 idle steps → score ~0, no success (4)
4. oracle reaches success() + score 1.0 on 3 seeds (5, 7, 8)
5. rubric monotonicity: staged score strictly increases across every load/cross/deliver
   milestone to 1.0 (6)
6. negative A (the seed's own strategy): air-carried handover of every parcel → all violated
   (9), perfect pad pose yet score ≤ 0.05 and no success (10), latch permanence after a clean
   re-ferry (11)
7. negative B: ground detour around the wall end → violated, capped (12)
8. negative C: loaded-latch partial credit (13), then carrying the loaded shuttle through the
   air → violated (14)
9. near-miss/tolerance: clean ferry, parcel set down beside the pad → crossed credit only
   (15); moved onto the pad → delivery credit recovered (16)
10. calibration probe A: basin drop-offset funnel (0–60 mm), published table + cliff asserts
    (17)
11. calibration probe B: traverse-speed ride retention (0.4/1.2/2.4 m/s), published table +
    gentle-ride assert (18)
