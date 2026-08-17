# sounding_wells — find the deep well by probing, plant the capped rod flush

`sim_gen` task `living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i401`
— env `simgen.sounding_wells`, scene `sounding_wells` (robot slot `"null"`; bodies
driven through scene handles).

## Seed provenance

Derived from **libero_90 / living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket**:
grasp ONE known object (the butter), carry it, drop it into ONE open receptacle (the
basket), judged by a position bounding box. One object, one known destination, one
unordered pick-and-place.

## What changed and why it is strategically different

The seed's plan needs **no information gathering**: the goal object and the goal
receptacle are both identified at reset, and the skill is pure transport. Here the
**correct receptacle is hidden state** that no amount of looking-from-the-side
reveals — it must be discovered by ACTING, and the acting IS the manipulation:

- THREE visually identical square wells stand in a row (inner bore 38 x 38 mm, rim
  78 mm high). TWO are secretly half-filled by squat captive filler slugs resting on
  their floors (34 x 34 x 40 mm — too squat to tip inside the bore, top 30 mm below
  the rim, unreachable); ONE — a different one each episode — is empty and 70 mm
  deep.
- The manipulandum is a **capped PROBE ROD** (30 mm square shaft, 66 mm long, under
  a 50 mm square cap that cannot enter the bore). The rod is **its own depth
  gauge**: lowered into a filled well the shaft stalls on the slug and the cap
  hangs **~36 mm proud** of the rim; only in the deep well does the cap settle
  **flush** on the rim (success tolerance 6 mm — a 30 mm discrimination margin).
- Success: the rod standing upright (<= 15°) in the TRUE well, cap flush on the rim,
  everything settled. An uncapped tan DUMMY BAR of the same stock is an identity
  distractor: it has no cap to seat, and it always protrudes >= 25 mm (retrievable).

So where the seed is *pick -> carry -> drop into the known container*, this task is
a **sense-then-commit loop**: probe -> read the rod's own rest height -> withdraw ->
repeat -> commit. Depositing the object into a receptacle without sensing — the
seed's entire strategy — fails 2 times out of 3, and the failure is enforced by a
hidden physical obstruction, not by a rubric fiat (smoke check 5). Nothing in the
scene is an open container that accepts a dropped object as success. This also does
not repeat any of my other derived tasks: the goal is not transport past an
obstruction, not construction, not a mechanism to actuate — it is **interactive
discrimination of hidden state with the manipulated object as the measuring
instrument**.

## Teleport solution (solve.py) — phases

1. **P0** settle + layout/hidden-state readback (slugs verified seated inside the
   two decoy wells); baseline score ~0, not success.
2. **Probe loop (canonical order 0, 1, 2)** — per well: TRANSPORT (teleport) the rod
   to a free-space hover over the mouth (upright, yaw-aligned, tip 10 mm above the
   rim, zero velocity), then hands-off gravity insertion; what stops the rod is real
   contact (slug or rim). The plan branches ONLY on the physically measured cap rest
   height (stall ~+36 mm vs flush <16 mm); the scene's hidden `true_idx` is used
   after the fact for verification only. A stalled probe's withdrawal is the next
   hover teleport (the real cap sits proud and graspable). Score latches 0.15
   (lift) + 0.35 (sound) = 0.50 on the first probe; 1.0 on the flush plant.
3. **Persistence**: >= 3.3 simulated seconds hands-off; success must still hold;
   `SIM_GEN_SOLVE: SUCCESS` printed only then.

Verified on forge: seeds 0–5 and 7 all `SUCCESS` (rc=0), covering the deep well in
every position (seeds 1,3 -> well 0 with 1 probe; 0,2,4,5 -> well 1 with 2 probes;
7 -> well 2 with 3 probes). Measured stall +36.0 mm, flush ±0.0 mm, monotone
scores 0.00 -> 0.50 -> 1.00.

## Embodiment argument (single Franka arm, parallel-jaw gripper)

One base pose serves the whole task: base at ~(0.05, 0.0) facing +x, wells at
0.31–0.62 m reach, rod/dummy spawns at 0.12–0.22 m — all inside a Franka's ~0.85 m
envelope, tallest interaction height 154 mm (rod hover over a rim).

- **Probe rod (the only object that must be touched):** it spawns lying on the
  ground; the 50 mm cap holds that end up, so the 30 mm square shaft is presented
  at a graspable height with finger clearance beneath it — side pinch across the
  shaft (30 mm << 80 mm max opening). Insertion is upright plunge with 4 mm radial
  clearance per side, well within closed-loop precision; the well mouth chamfers
  nothing, but the solve shows a plain vertical drop with mm-level alignment seats
  reliably. Reading the outcome is trivial for wrist F/T or even open-loop height:
  the cap stops 36 mm higher in a filled well. To withdraw a stalled probe the
  gripper re-pinches the proud cap (50 mm < 80 mm opening; 36 mm of clear side
  face). At the flush plant the fingers release the cap top face and retract
  vertically.
- **Wells, slugs:** never touched (wells are fixtures; slugs are physically
  unreachable past the rim — that is the point).
- **Dummy bar:** never touched (restraint); if mis-planted it protrudes 25 mm and
  can be pinched back out.

## Execution-order declaration

**No required order.** The wells may be probed in any order (the instruction says
so); a lucky first plant with zero prior probes is a legitimate success. The only
physically-forced structure is *sense-before-commit in expectation*: an agent that
never reads its probe outcome succeeds only 1/3 of the time.

## Rubric

- 0.15 — `lift` (latched): rod ever carried above 0.12 m.
- 0.35 — `sound` (latched): rod tip ever >= 15 mm below some rim, inside that bore,
  near-upright — a probe actually made. Non-success total capped at 0.50.
- 1.0 iff `success()`: rod upright in the TRUE well, cap gap < 6 mm, all bodies
  settled, states finite.

Honesty-by-construction asserts in the cfg `__post_init__`: shaft/bore clearance,
cap > bore (the gauge), stall margin >> flush tolerance, tip clear of the floor at
flush, slug captivity (fit, cannot tip, below rim), dummy always retrievable.

## Smoke battery (smoke.py) — forge result `SIM_GEN_SMOKE: ALL PASS 8/8`

1. settle/no-NaN + slugs verified seated inside the two decoy wells; score 0.
2. randomization readback (3-seed max-pairwise): well xy 286 mm, terrace yaw 176°,
   rod xy 70 mm.
3. hidden-state coverage: true-well histogram over 9 resets = {0:4, 1:3, 2:2}.
4. null policy: 240 idle steps, score ~0, no success.
5. **SEED-strategy analog**: rod deposited into a decoy well exactly as into the
   true one — stalls on the hidden slug at +36.0 mm proud (in the asserted
   25..50 mm slug-rest band, so it is slug contact, not a mouth jam); sound credit
   only (score = 0.50 cap), **no success**.
6. near-miss (delivered, not planted): rod parked CAP-DOWN on the TRUE well's rim —
   stays up on the rim, but inverted, tip in no bore: no credit, no success.
7. wrong object: dummy bar planted in the TRUE well — protrudes +25.0 mm
   (retrievability is physics), no success.
8. video frames.npz (44 frames, 960 x 600).
