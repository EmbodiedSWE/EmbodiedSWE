# pile_driver — drive the red post flush by dropping a slug on it

Task directory: `sim_gen/tasks_v7/libero_pick_milk_i51`
Scene name: `pile_driver` · Env id: `simgen.pile_driver` (robot `"null"`)

## Provenance

Seed task: `libero/libero_pick_milk` — pick the milk carton out of a clutter of
distractor objects and place it in a basket. What survives from the seed:

- **Pick-among-distractors**: the RED post must be acted on while the visually
  identical WHITE post beside it must NOT be (color is the only discriminator,
  as milk-vs-ketchup was in the seed).
- **A carried object with a designated grasp feature**: the slug's narrow red
  grip stud is the "carton handle" — the one thing the gripper holds.
- **A place-back container**: the yellow walled holster tray plays the basket;
  the episode ends with the slug standing inside it.

## Strategic difference

The seed is one quasi-static transport: grasp, carry, release. This task is
built so that **quasi-static transport accomplishes nothing** — the goal state
can only be reached through a *repeated impulsive* interaction:

- Each post runs through a spring-loaded friction clamp (~199 N static
  force-closure, forge-calibrated). Steady pushing — even a full-arm 90 N
  lean, measured — moves the post 0.0 mm. Only a collision's force spike
  breaks the clamp, and each blow buys a finite advance (4–10 mm coarse,
  ~2.3 mm in the fine band) before friction re-locks. Progress is a
  physical ratchet: the joint's upper stop means a post can never rise back.
- So the manipulation primitive is **lift–hold–DROP, repeated ~15 times with
  monitoring** (read the proud height, decide coarse vs fine drop height,
  stop at flush), not a single pick-and-place.
- The distractor is not just clutter to route around: striking the WHITE post
  is **irreversible failure**, because posts cannot be raised. The seed's
  distractors were passive; here the distractor is a trap.
- Overdrive is prevented by geometry, not by policy care: the slug's 72 mm
  face is wider than the 46 mm deck aperture, so the slug bottoms on the deck
  exactly at flush depth.

Versus the nearest tasks_v7 neighbors: `cam_press` (continuous cam rotation →
press), `kerf_chop` (single committed cut), `ramrod` (quasi-static push
through a bore), `spring_bay` (compress-and-latch statics) — none is an
iterated impact ratchet against a friction brake, and none has an
irreversible same-mechanism decoy.

## Scene (all procedural)

Pile-driver rig: heavy base, DECK plate at 0.26 m, two 40 mm square posts
(red +Y side, white −Y side) through deck apertures. Each post is a plain
rigid body on a USD **prismatic Z joint** (limits: 98 mm down-travel, +0.5 mm
up — the ratchet), gripped by a spring-loaded brake pad (horizontal prismatic
+ linear drive, k = 22 kN/m, squeezed to ~0.0145 m → ~319 N normal). Slug:
72 mm × 60 mm steel block, 1.2 kg, with a 30 mm red grip stud. Holster:
yellow walled tray on the floor at 0.33 m radius from the rig.

Randomized per episode (all readback-verified in smoke): rig xy (±4 cm
jitter) and **free yaw**, both posts' initial proud heights (60–78 mm sampled;
72–90 mm after settle), holster arc position.

### Calibration notes (forge-measured, do not "fix")

- `physxJoint:jointFriction` is **silently ignored** for joints outside
  articulations — the brake is pure pad contact friction. PhysX patch
  friction saturates **sublinearly in normal force** (effective mu ~0.24 of
  nominal at ~319 N) but stays **linear in mu** — verified by doubling mu
  0.65→1.30, which doubled kinetic capacity (~92→~184 N; drop advance
  17.4→7.4 mm as 1/brake predicts). The design derates by `clamp_eff = 0.24`
  and buys margin with brake-pad-grade mu (1.30/1.20).
- **Settle-lift artifact**: any root-state write on a clamped post re-forms
  the pad squeeze and lifts the post ~13 mm before the clamp latches. The
  scene absorbs it: a 60-step warmup window re-latches `proud0` from
  readback (rubric references *settled* heights), `settle_lift = info(0.012)`
  documents it, `describe()` prints the settled range, and smoke constructs
  post poses with an iterative `place_post` (write → settle → correct).
- TGS velocity iterations: > 4 triggers changed-behavior and broke end-state
  settling; posts/pads use `vel_iters = 4` (the safe max).

## Rubric (latched, monotone, warmup-gated)

`post_step` latches with `live = ~warm`: `ever_near` (slug within 0.22 m of
the rig, 0.08), `best_drive` (max fraction of red drive completed, 0.52),
`flush_latch` (red flush AND white still ok, 0.15). Score sums to 0.75;
**1.0 iff `success()`**: red within 12 mm of deck ∧ white within 12 mm of its
settled start ∧ slug parked in holster ∧ everything settled. Latches survive
`get_state`/`set_state`. Score can never decrease; wrong-post damage kills
`flush_latch` permanently.

## Solution outline (solve.py)

Teleport = **transport only** (carry the slug between poses at zero
velocity); every load-bearing interaction is contact dynamics.

- P0: settle, read layout (rig pose/yaw, both prouds, holster) from state.
- P1: teleport-hover the slug over the red post (score 0.08).
- P2: drop loop — release from 14 cm (coarse) or 8 cm (fine, inside the
  30 mm band); gravity does the striking, hands off from release to rest.
  ~15 blows, 5.6–7.3 mm coarse / ~2.3 mm fine advances; stall guard aborts
  after 6 consecutive blows under 1.5 mm. Flush latches at 0.75.
- P3: teleport the slug back over the holster, drop it in (3 attempts).
- P4: success asserted, then **780 steps (3.25 s) hands-off persistence**,
  then `SIM_GEN_SOLVE: SUCCESS`. Watchdog Timer + `os._exit` guards hangs.

Verified on the final submitted files: **seed 0** (15 blows, 45.1 s) and
**seed 1** (15 blows, 44.8 s), both `SIM_GEN_SOLVE: SUCCESS`, monotone
scores 0 → 0.08 → … → 0.75 → 1.0.

## Embodiment argument (single Franka + parallel-jaw)

Every action is within one arm's envelope: the grip stud (30 mm square) is a
canonical parallel-jaw pinch; carrying a 1.2 kg slug is well inside Franka
payload; "hold 10–20 cm above the post and open the gripper" is a release,
not a dexterous skill; the drop itself is gravity's job, so the arm never
needs to deliver the impact force (~199 N would exceed the arm — which is
exactly why the task forbids pushing and the clamp is sized to defeat a
90 N arm-lean). Re-grasping the slug off the deck top after each blow is a
top-down pinch of the same stud. No bimanual, no in-hand reorientation, no
force beyond payload.

## Execution order used

1. Minimal goal predicate (`success()` on flush/white/park/settled) first.
2. Working solve demonstrated against it (forge, seeds 0/1).
3. Rubric weights and thresholds anchored in the demonstrated solution
   (drop advances, near radius, flush band, persistence).
4. Smoke battery written last, against the locked behavior.

## Smoke battery (SIM_GEN_SMOKE: ALL PASS 20/20)

1. Reset sanity: posts held by clamps at latched `proud0` (readback).
2. Null policy ~0: 480 idle steps, score stays 0, no drift.
3. Steady push rejected: 90 N quasi-static shove advances the post 0.0 mm.
4. Resting slug rejected: slug parked on the red head, no advance, no score.
5. One drop advances ≥ 4 mm (impulse beats statics).
6. Randomization spread across seeds: rig dx/dy/dyaw, both prouds, holster
   (readback-verified deltas, e.g. dyaw = 3.36 rad, dred = 9.3 mm).
7. get_state/set_state round-trip preserves poses, latches, warmup.
8. Score monotonicity under latching.
9. White-post ratchet: driven white cannot recover — `white_ok` stays False.
10. Wrong-outcome constructs rejected: red flush but slug not parked; parked
    but not flush; near-miss (red at 22 mm, outside band) — via iterative
    `place_post`; white driven 30 mm (irreversibility clause).
11. Acceptance construct: red flush + white ok + slug parked → success True,
    score exactly 1.000.
12. Frames: `frames.npz` written in CWD (163, 600, 960, 3).

Final line: `SIM_GEN_SMOKE: ALL PASS 20/20` (forge run, 62.3 s).
