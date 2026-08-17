# approach_grasp_bowl_i133 — Cliff Sweep (scene `cliff_sweep`)

Herd three loose 40 mm orange balls off a raised stage into the catch basin below the
red-striped cliff edge — WITHOUT ever lifting one. The tool is a blue SCOOP DOME (a
walled disk with a 90° open mouth and a handle): set it down over the pack so the balls
are caged inside its walls, slide it mouth-first toward the cliff to plow them across
the stage, and brake short of the edge so the pack coasts out of the mouth, over the
cliff, and drops into the basin (a curling-style plow-and-release delivery). The balls
are trivially graspable — on purpose: any ball raised above `lift_z` (190 mm — about
one ball over another) SPOILS the episode irreversibly, so the seed's verb is the trap.
Finish with all three balls resting in the basin and the scoop parked back on the stage.

## Provenance

- **Seed:** `pick_place/approach_grasp_bowl`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_bowl.py`) —
  approach ONE bowl lying in tabletop clutter, close the parallel jaw on its rim, lift
  it and carry it along marked waypoints: one grasp affordance plus free-space
  transport of a rigidly held object.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `cliff_sweep`, env
  `simgen.cliff_sweep`, robot `"null"`), `solve.py` (teleport solution), `smoke.py`
  (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed is grasp → lift → carry of a single judged object, and
  height GAIN is its load-bearing resource. Here everything inverts: there are THREE
  judged objects (a pack, not an object), the goal is BELOW the start (gravity does the
  final transport leg, off a cliff), and lifting — the seed's entire strategy — is not
  merely unrewarded but IRREVERSIBLY PENALIZED by a latched spoil gate (`lift_z`,
  asserted in cfg `__post_init__` to sit above anything floor herding can reach and
  below any genuine carry over the guard rails). The balls are deliberately graspable
  (40 mm ≪ 80 mm jaw, asserted): the constraint is behavioral, not geometric. The
  load-bearing mechanic — caging rolling bodies under an open-mouthed dome, plowing the
  cage, then braking so the pack RELEASES through the mouth and falls — is a
  tool-mediated, multi-body, momentum-and-gravity delivery the seed never touches, and
  the delivery direction (which stage end is the cliff) flips per episode.
- **vs the five sibling tasks read this session:** `approach_grasp_spoon_i12` (die tip)
  is non-prehensile reorientation of ONE ungraspable body — no tool, no multi-body
  pack, no drop; `approach_grasp_knife_i25` (underpin swap) is support-substitution
  statics; `approach_grasp_banana_i69` (tilt maze) steers by tilting the WORLD, the
  object is never touched by a tool; `approach_grasp_ceramic_teapot_i109` (mug-rack
  hang) is prehensile placement onto a hook; `approach_grasp_i29` (ridge poise) is a
  single-body balance task. None combines a herding tool, a 3-body pack, a
  plow-and-release over a drop, and a "never lift" invariant.
- **Execution order (declared):** cage → plow → release → repeat for stragglers → park.
  The capture credit (0.10) gates on the scoop being DOWN on the stage, the delivery
  credit (0.25/ball) latches only in the basin, and the park clause is judged at the
  end — there is no ordering loophole: delivering balls first and dropping the scoop
  into the basin afterwards fails the park clause (smoke check 10), and lifting balls
  at ANY time spoils (checked transiently every physics substep).

## Randomization (per episode, verified by readback in smoke)

Cliff SIDE s ∈ {+x, −x} (coin flip via `torch.rand`, not first-`randint`): the
kinematic basin teleports to x = s·0.58 (yawed 180° on −x so its far wall stays
outboard) and the kinematic end rail closes the opposite stage end at x = −s·0.374.
Ball cluster centre (mirrored-x ±50 mm, y ±90 mm) with per-ball disc offsets
(radius ≤ 45 mm) batch-resampled (64 rounds) to pairwise gaps ≥ 45 mm, with a
deterministic fallback — an equilateral triangle at random rotation whose side
(≈ 58 mm) exceeds the gap by construction (cfg-asserted) — so the gap guarantee is
unconditional even when rejection sampling fails; scoop spawn near the closed
end (mirrored-x −0.27..−0.24, y ±0.12) with free yaw.

## Rubric

`success()` iff, all judged live on physical poses:

- all three ball centres inside the basin interior — mirrored-x ∈ (0.395, 0.755),
  |y| < 0.272, centre z < 55 mm (one floor-resting ball passes at 28 mm; a two-ball
  stack at 68 mm fails the low gate) — and slow (|v| < 0.05 m/s);
- the scoop resting on the stage: |x| < 0.37, |y| < 0.25, origin z in (0.14, 0.22) —
  rest is 0.1475; a scoop in the basin (~0.036) or riding on balls fails — and settled
  (|v| < 0.05 m/s, |ω| < 0.50 rad/s);
- NOT spoiled: no ball centre ever exceeded `lift_z` = 190 mm (latched every substep).

`score()` (latched in `post_step`): `0.10 ×` captured (some ball within
`inner_r − ball_r − 3 mm` ≈ 55 mm of the scoop axis while the scoop is down)
`+ 0.25` per ball delivered (in the basin and |v| < 0.30), capped at 0.85; a spoiled
episode is capped at 0.20 regardless; exactly 1.0 iff `success()`. Doing nothing ~0.

## Teleport solution (`solve.py`) — transport only, every write ends in FREE SPACE

- **Transport:** the scoop (the tool — never a judged object) is teleported only to
  HOVER poses ~5 cm up, wall bottoms verifiably above every ball top, mouth yawed
  toward the cliff, velocities zeroed; GRAVITY then lowers it over the balls through
  real contact. The judged balls are never teleported.
- **P1 cage:** hover-drop centred on the approximate min-enclosing-circle centre of the
  settled pack (worst ball offset ≤ 45 mm + drift, cage inner radius 78 mm — clearance
  by construction, printed as readback).
- **P2 plow + release:** a velocity-servoed horizontal force (F = m·K·(v_des − v),
  K·dt = 0.5, cap 6 N) applied at wall-bottom height via the equivalent COM force plus
  the r × F counter-torque (r_z = −wall_h/2), with a yaw-hold torque on the mouth
  heading; slow (0.14 m/s) to mirrored-x 0.12, fast (0.34 m/s) to 0.285, then a hard
  brake — the scoop stops with its nose short of the edge and the caged pack coasts out
  through the mouth, over the cliff, into the basin.
- **P3 cleanup:** any straggler still on the stage is re-covered (hover-drop at
  ball + 40 mm along the plow line, xy clamped so the descent path clears the rails and
  the cliff) and re-plowed, ≤ 6 attempts.
- **P4 park + P5 persist:** hover-drop the scoop onto the open closed-end floor, then
  ≥ 3.4 simulated seconds hands-off with `success()` held.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing (all credit is
  latched), then `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1,
  both SUCCESS**, provably distinct by stdout readback — seed 0: side=+1, scoop spawn
  (−0.252, +0.082) yaw +0.81, trace 0.00 → 0.10 (cover, caged dists 0.028/0.036/0.035
  < 0.055) → 0.60 (first plow: all three over the cliff, one still rolling) → 0.85
  (cleanup settle) → 1.00 (park) → 1.00 (persist); seed 1: side=−1 (mirrored cliff),
  scoop spawn (+0.242, −0.046) yaw +2.85, trace 0.00 → 0.10 → 0.85 (all three
  delivered and settled in the first plow) → 0.85 → 1.00 → 1.00. Smoke battery:
  `SIM_GEN_SMOKE: ALL PASS 14/14`, frames.npz (239, 600, 960, 3).

## Embodiment sanity (single-arm Franka feasibility)

One base pose serves the whole episode family: base at (0, ∓0.75) facing the stage's
long side — every contact point lives on the stage top (x ∈ ±0.38, y ∈ ±0.26,
z ≈ 0.12–0.21), inside a Franka's ~0.85 m envelope from either long side; the basin is
only ever a passive receiver, never a contact target. Per-object contact strategy: the
SCOOP is worked by its yellow HANDLE stub (22 mm square — well inside the 80 mm jaw
stroke) or simply pushed at wall height with the closed fist; the solve's force profile
is exactly a fist-push at the wall bottoms (the r × F term) with modest magnitudes
(≤ 6 N) and speeds (≤ 0.34 m/s) that OSC tracks comfortably. The BALLS are never
touched by the gripper at all — the dome is the only thing that touches them, which is
what the spoil latch demands. Raising the whole dome by its handle to reposition it
(the hover-drop) is an ordinary pick of a 0.30 kg tool. No contact is required below
stage-top height, so the cliff and basin never enter the reachable-workspace argument.

## Checks (`smoke.py` — rejection battery, 14 named checks)

1. settle: states finite; balls at sphere height on the stage, scoop at dome resting
   height (readback).
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: cliff side flips both ways across 8 seeded resets AND the
   kinematic basin/end-rail teleports follow the side (sign-correlated readback).
4. randomization readback: ball cluster xy, scoop xy and yaw vary; pairwise spawn gaps
   ≥ `ball_gap`.
5. null policy: 300 idle steps → score ~0, no success, and max ball drift < 30 mm
   (GPU sphere-creep regression).
6. SEED strategy: each ball lifted above `lift_z` and carried into the basin — final
   tableau visually complete (all in basin, scoop parked) yet spoiled → NOT success,
   score ≤ 0.20. THE key check: the seed's verb produces the goal picture and still
   fails.
7. beside-basin miss: balls settled on the ground OUTSIDE the basin side wall → no
   delivery credit, NOT success.
8. cliff-edge miss: balls settled on the STAGE at the lip → no delivery credit, NOT
   success.
9. captured-only: scoop set down over the pack → capture latch fires (0.08 ≤ score
   ≤ 0.25), NOT success.
10. scoop-in-basin: all three balls delivered but the scoop in the basin too →
    0.70 ≤ score ≤ 0.85, NOT success (the tool must end parked on the stage).
11. unsettled: balls IN the basin but still moving fast (velocity written into the
    root state; scoop pre-moved off-stage so the audit stays honest) → NOT success at
    the judged instant.
12. latched delivery: removing a delivered ball back onto the stage leaves the latched
    credit unchanged, success stays gone.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest: the
balls are trivially graspable (the latch, not size, forbids the seed strategy); the
spoil gate sits above floor herding's reach and below a rail-clearing carry; the
delivery window sits strictly inside the basin walls and the one-ball/two-stack z gate
brackets correctly; the basin interior is wider than the stage (cliff drops cannot
miss); the scoop spawn clears the end rail and side rails while its reach covers every
possible ball spawn; a covered cluster fits inside the cage with clearance; and the
park z band brackets exactly the dome's resting height.
