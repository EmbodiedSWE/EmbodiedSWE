# pour_from_cup_to_cup_i406 — RationTicketScene (`simgen.ration_ticket_station`)

Right two mouth-down cups onto their painted goal pads and load each with **exactly
the number of balls its pad's printed dot ticket shows** (1–3 dots, per-episode),
drawing from a bin **bolted to the floor** that holds more balls than the two tickets
need — every surplus ball must be **left in the bin**.

## Seed provenance

Seed: `rlbench/pour_from_cup_to_cup`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/pour_from_cup_to_cup.py`) — "pour the
liquid (balls) from the source cup into the target cup", with look-alike distractor
cups. Its plan is **grasp → carry → wrist-tilt → bulk dump**: transport plus
reorientation of a held source vessel, discharging *everything* in one pour; the only
decision is which cup is the target.

## What changed and why it is strategically different

The theme kept: loose contents move from a source container into target vessels. **The
transfer model is replaced wholesale:**

| | seed | this task |
|---|---|---|
| source | free cup, grasped, carried, tilted | bin **bolted to the floor** (kinematic): nothing to pick up or pour — content leaves only as individually carried balls |
| amount | everything, in one bulk dump | **EXACTLY the per-pad ticket count**, read visually from 1–3 printed dots; the bin holds 1–2 balls MORE than the tickets sum to, so "move everything" over-fills and fails |
| targets | one target among look-alike distractors | two targets, **each with its own independent quota** (usually different) — a per-target allocation, not a target-identification |
| stopping rule | stop when the source is empty | **stop while balls remain**: the surplus must be abandoned in the bin — an act of *not* doing |
| ordering | none (cup already usable) | **physically forced first step**: both cups spawn mouth-down off-pad; nothing can enter a mouth-down cup, so right-and-place precedes any delivery — geometry enforces the order, not the rubric |

A solver needs a different plan (perceive two counts, erect two containers, run a
counted per-item loop with a stopping rule) and different code (quota perception,
allocation bookkeeping), not different numbers on the seed's grasp-carry-tilt-pour
plan.

Also distinct from the sibling task from the same seed viewed this session:
`pour_from_cup_to_cup_i100` (mechanism actuation — press-and-hold a pedal on a
trunnion-mounted hopper; no counting, no allocation, balls never touched), and from
the exemplar `pen_holder` (insert *all* pens into one holder — no exactness, no
surplus clause, no forced ordering).

## Scene

Fully procedural (compound spawners, authored MassAPI CoM + inertia, friction
materials bound to every collider). KINEMATIC bin at the origin (interior 26 × 18 cm,
30 mm walls, bolt-head studs as the visual "fixed" statement); two DYNAMIC octagonal
cups (inner flat radius 31 mm, 85 mm tall, 150 g, CoM authored low); two collider-less
painted pads at y = ±0.30 (thin plates would edge-catch a settling cup); per-pad
ticket rows of collider-less black dots between pad and bin; 8 orange balls (r 14 mm,
30 g, velocity iters 4 against the GPU sphere-creep artifact, restitution 0).

**Randomization (readback-verified in smoke):** per-pad quota 1–3 (independent),
surplus 1–2, which ball bodies are present (permuted jittered bin slot grid), pad xy
jitter, cup offset ring 10–15 cm × free direction band × free yaw. Absent balls park
in a ground depot; shown-dot count *is* the quota (asserted by readback).

`__post_init__` asserts every geometric claim: a 3-ball ticket fits far below the rim,
two balls fit side-by-side on the cup floor, the in-cup radial test is conservative
w.r.t. the octagon corners, the cup fits the 80 mm Franka jaw, pads can never merge or
double-occupy under jitter, a spawned cup can never start inside pad tolerance, the
ticket row sits clear of both the goal footprint and the bin.

## Rubric (anchored in the demonstrated solve)

- `0.15` per pad — **placed latch**: an upright, settled cup once stood centred on
  that pad (latched; both pads → 0.30);
- `0.45 · fill` — ticket-fulfilment fraction `Σ_j min(count_j, quota_j) / Σ_j quota_j`
  counting only balls inside an **upright cup standing on a pad** (live; overfill
  earns nothing);
- `1.0` **iff `success()`**: each pad carries exactly one upright settled cup holding
  EXACTLY its quota, every remaining present ball rests inside the bin, everything
  settled and finite (all live physical clauses); non-success caps at 0.75.

Null policy ≈ 0 (cups mouth-down off-pad: no latch, no countable containment —
verified in smoke). Latched placement credit cannot evaporate; fill credit is live but
balls resting inside a standing cup stay there.

## Teleport-contract solution (solve.py)

Teleports are TRANSPORT ONLY — every load-bearing interaction ends in a gravity drop
through live contact:

- **P0 settle** — 1.0 s hands-off; layout readback (quotas, surplus, present count,
  pad poses); assert cups mouth-down, score ≈ 0, no success.
- **P1 place** — per pad: teleport its cup **upright, 20 mm above the pad**, zero
  velocity, and drop; wait for the scene's own `on_pad()` (upright + centred + bottom
  on floor + still). The placed latch arms physically. `SIM_GEN_SCORE` 0.15 → 0.30.
- **P2 fill** — per pad: exactly `quota_j` present bin balls, one at a time, each
  teleported **25 mm above the standing cup's mouth** (small xy scatter) and dropped;
  each must settle contained before the next. The surplus balls are **never touched**.
- **P3 confirm** — hands-off settle until the scene's own `success()` holds live.
  `SIM_GEN_SCORE = 1.0`.
- **P4 persist** — ≥ 3.5 more simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS`
  only if `success()` still holds live.

Printed score sequence is asserted non-decreasing. No pose is ever written into a
seated/contained configuration.

## Embodiment argument (single Franka)

Base pose: **(−0.45, 0.0, 0)**, facing the station. Bin centre 0.45 m, pads
≈ 0.55 m, cup spawn ring ≤ 0.60 m — all inside the Franka envelope.

- **Right a cup**: the octagonal body is 72 mm across the flats — inside the 80 mm
  parallel-jaw span; pinch-grasp the mouth-down cup across its flats near the (upper)
  floor end, lift, rotate the wrist 180°, set it down on the pad centre, release.
  Pads are paint (no lip to catch); 3 cm centring tolerance is generous for a placed
  object.
- **Count the ticket**: 14 mm black dots on a bare floor, high-contrast — a wrist- or
  scene-camera perception task, no interaction needed.
- **Deliver a ball**: the 28 mm balls lie on a spaced slot grid in an open 30 mm-walled
  tray — top-down pinch grasps with clearance on every side; release each ball just
  above the 62 mm cup mouth (ball-to-wall clearance 17 mm on radius). 30 g payloads.
- **Stop**: the surplus balls are simply never picked — no interaction required.

**Execution order**: only the causal chain right-cup-before-fill is required (and it
is enforced by geometry — nothing enters a mouth-down cup); pad order and ball order
are free. No arbitrary ordering constraints are imposed.

## Checks (smoke.py — 10)

1. settle/no-NaN (present balls in the bin, absent parked, cups mouth-down, SHOWN
   dot count == quota readback, score ~0);
2. randomization readback differs across seeds (3-seed max-pairwise: pad, cup, ball);
3. coverage: ≥ 2 distinct quota pairs, an asymmetric pair, ≥ 2 present counts over 10
   resets;
4. null policy: 240 idle steps → score ~0, no success;
5. **BULK-DUMP (the seed strategy's end state)**: both cups placed, then *every*
   bin ball dropped into cup A — verified overfilled; success refuses, score capped
   at the quota's own fill credit (overfill earns nothing);
6. **OFF-BY-ONE allocation**: right total delivered, one ball on the wrong pad —
   leftovers fine, cups placed, yet no success; fill credit exactly one ball short;
7. **OFF-PAD near-miss**: cup A upright and correctly filled but centred past
   pad_tol — latch never arms, its balls count for nothing;
8. **MOUTH-DOWN COVER cheat**: cup A upside-down ON its pad covering exactly its
   quota (balls verified geometrically inside the inverted interior — only the
   upright gate rejects); count reads 0, no latch, no success;
9. **LEFTOVER LOOSE**: both pads exactly right but one surplus ball abandoned on the
   open floor — base credit full (0.75) yet success refuses (the
   leave-the-rest-in-the-bin clause is load-bearing);
10. frames.npz video.

Check 5 is the **seed-strategy end state**: "move everything into the target" is the
literal outcome of the seed's bulk pour, and it fails here by design.

## Verification

- `solve.py` on the forge, seeds 0, 1 and 2 (quotas [2,1], [2,1], [3,2]; surplus 1–2;
  present 4/5/7): `SIM_GEN_SOLVE: SUCCESS`, monotone scores 0.000 → 0.150 → 0.300 →
  fill → 1.000, 3.5 s persistence.
- `smoke.py` on the forge: `SIM_GEN_SMOKE: ALL PASS 10/10`, frames.npz saved
  (113 frames, 960×600).
