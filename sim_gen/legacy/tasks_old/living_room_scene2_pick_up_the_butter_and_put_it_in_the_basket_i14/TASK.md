# slot_deposit — post the butter through the deposit box's mail slot

**Env name:** `simgen.slot_deposit` (scene `slot_deposit`, robot `null`)
**Tier:** medium — **4 stages**. **Execution order: REQUIRED** (forced by geometry, see below).

## Seed provenance

- Seed: `libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket.py`)
- Seed plan: identify the butter among 6 grocery distractors, grasp it, carry it over an
  **open-top basket**, release **in any orientation** — a bounding-box containment check
  on the basket's `contain_region` fires. The container is a pure catch-all: gravity and
  a generous open aperture do all the accepting.

## What changed, and why it is strategically different

Kept: the butter block as the target, grocery distractors (milk carton, soup can), a
container as the goal, the identify-the-target flavor.

Changed — the container's *acceptance mechanism* is inverted, which invalidates the
seed's entire plan rather than its parameters:

1. **The container is CLOSED.** The deposit box (procedural kinematic compound: floor,
   4 walls, 4 lid boards) has no open top. Its only entrance is an **80 x 46 mm mail
   slot** in the lid, at a **randomized heading** (the box yaw randomizes ±180°).
2. **The slot is orientation-selective by construction** (asserted in the config): of the
   butter's three face-first cross-sections (62x32, 95x32, 95x62 mm), only the **end-first
   62x32** one fits, and only within **~14° of yaw error** (62·sinθ+32·cosθ ≤ 46). The
   seed never cares about orientation; here reorientation *is* the task.
3. **The seed's own plan is a tested negative control:** carrying the butter over the
   container and releasing it — flat, any yaw — lands it on the closed lid and scores
   only the 0.10 lift latch. Expressible, executed in the smoke, fails.
4. **Fit, not just identity, rejects the distractors:** the milk carton (70 mm min
   dimension) and the soup can (66 mm diameter) cannot pass the slot in **any**
   orientation (asserted; the can is physically test-dropped as a negative control).
   The three objects also **permute their spawn slots** per episode, so the butter must
   be identified by appearance, not location.
5. **Anti-teleport transit latch** (the tunnel_shuffle rubric device): success requires
   that the butter's centre once crossed the slot mid-plane *downward*, *inside the slot
   aperture*, moving **< 4 cm per step**. Writing the butter straight into the box
   interior leaves it inside-and-settled yet unsuccessful (tested negative control) —
   the judged outcome is a *physical passage*, honest under the teleport-oracle.

A solver needs a different plan: **select → reorient end-first vertical → yaw-align to a
randomized heading → thread through a tight aperture** (a "shape-sorter" insertion),
instead of the seed's *select → carry → drop anywhere above*.

### Differentiation from sibling generated tasks (claimed-axes check)

- i4 `pan_lowshelf_slide` / i9 `pan_cubby_retrieval`: horizontal low-*clearance* alcoves
  forcing planar sliding in/out. Here the constraint is an **aperture in the lid**, the
  approach is vertical, the core skill is **in-air reorientation + precision insertion**,
  and nothing is slid.
- i5 `empty_the_bowl`: reorients the *container* to pour contents out. Here the container
  is fixed and closed; the *manipulated object* is reoriented to get contents **in**.
- i10 `tee_up`: extraction from a container + low-energy perch on a convex support. No
  extraction and no perch here; the target is still interior containment, but gated by
  shape/orientation selection.

## Stages (order required)

1. **Identify + pick** the butter among permuted distractors (lift latch, +0.10).
2. **Reorient + align**: long axis vertical, yaw matched to the slot heading, centred
   over the slot (aligned-hover latch, +0.15 → 0.25). Must precede insertion — every
   other attitude is geometrically blocked.
3. **Insert through the slot** — physical transit past the lid plane (transit latch,
   → 0.70). Must precede stage 4: the interior is unreachable any other way.
4. **Release and settle inside** → success, score 1.0.

## Rubric

`score() = 0.10·lifted + 0.15·aligned` latches; `0.70` once transited; `1.0` iff
`success() = transited ∧ inside ∧ settled`. Null policy scores exactly 0; teleporting
into the box caps at 0.25.

## Smoke check list (17)

1. reset settles finite with score 0
2. randomization is real (box pose + slot heading, by readback)
3. butter spawn slot permutes across resets
4. null policy scores ~0, no success
5. oracle milestone: lift latch pays 0.10
6. oracle milestone: aligned hover pays 0.25
7. oracle milestone: slot transit reads 0.70 mid-insertion
8–10. oracle reaches success() on seeds 0 / 1 / 2
11. rubric strictly increases through the stage sequence [0, .10, .25, .70, 1.0]
12. negative A: the seed's own strategy (drop over the container) fails on the closed lid
13. negative B: teleport into the box (no transit) is inside yet capped ≤ 0.25, no success
14. negative C: the soup can, dropped end-first on the slot, is blocked (wrong object)
15. near-miss: correct attitude/position, 90° yaw — rejected by the slot
16. calibration: free-drop orientation-selectivity probe (0° passes, 30°/90° blocked;
    geometric yaw limit ~14° printed)
17. video frames recorded (frames.npz saved to CWD)

## Physics honesty notes

- The oracle inserts kinematically (gravity-compensated hold, sub-mm descent steps), but
  the final stretch is a **free fall** through the slot and the judged state is the
  settled physical pose; the calibration probe additionally proves an **unassisted**
  aligned drop passes the slot while misaligned drops bounce off.
- Slot clearance (7–9 mm/side) is protected by tiny lid/wall contact offsets (1.5 mm);
  the box floor is thick (16 mm) with a generous speculative offset (5 mm) to capture
  ~15 mm/step end-on impacts (GPU PhysX has no CCD).
