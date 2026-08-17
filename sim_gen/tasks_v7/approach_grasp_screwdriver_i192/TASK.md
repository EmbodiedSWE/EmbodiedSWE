# approach_grasp_screwdriver_i192 — Bridge the Moat, Roll the Ball Home (scene `moat_bridge`)

Two raised platforms face each other across an open MOAT — a sheer 150–200 mm gap
whose floor lies 120 mm below the tops. A blue CHANNEL PLANK (300 × 160 × 12 mm deck
with 8 mm side rails and 35° end ramps) lies loose on the near platform, and an
orange 100 mm BALL rests somewhere behind it. The goal: lay the plank across the moat
so its deck rests on BOTH rims in line with the green walled DOCK on the far
platform, then roll the ball over the bridge and into the dock until it rests against
the walls. The ball is STRICTLY wider than the Franka jaw (100 mm vs 80 mm), so it
can never be grasped or carried — and the moat is irreversible: a ball that drops in
is lost forever. The graspable object (the plank) is a TOOL whose placement is
infrastructure, not the goal; the judged object can only ever be rolled.

## Provenance

- **Seed:** `pick_place/approach_grasp_screwdriver`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_screwdriver.py`)
  — approach a small screwdriver on a tabletop, close the parallel jaw on its handle,
  and carry it: one grasp affordance plus free-space transport of a rigidly held
  object.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `moat_bridge`, env
  `simgen.moat_bridge`, robot `"null"`), `solve.py` (teleport solution), `smoke.py`
  (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs the sibling corpus)

- **vs the seed:** the seed's entire strategy is prehensile transport of the target —
  grasp the graspable thing, carry it where it belongs. Here that strategy applied to
  the judged object is PHYSICALLY IMPOSSIBLE (100 mm ball vs 80 mm jaw), and applied
  to the graspable object it earns exactly nothing: carrying the PLANK to the dock —
  the seed's plan verbatim — is smoke check 6 and scores ~0. The task inverts the
  seed's structure twice over: (1) the grasp affordance belongs to a TOOL whose
  correct placement (a two-support bridge found by settling physics, judged level, at
  rim height, overlapping both rims) is a PREREQUISITE, not the goal; (2) the goal
  object moves only by non-prehensile ROLLING along the plank's rail channel, across
  a gap where the floor itself is a terminal failure state. This forces a strict
  EXECUTION ORDER the seed never has: bridge first, ball second — rolling the ball
  toward the gap before the bridge exists loses the episode irreversibly.
- **vs the siblings:** no sibling task makes tool-placement-as-infrastructure the
  load-bearing mechanic. `approach_grasp_spoon_i12` is non-prehensile reorientation
  (tipping a die) with no tool and no irreversibility; `close_box_i26` and the
  push/slide family move the judged object directly; the LIBERO bowl/drawer tasks are
  prehensile pick-and-place; hinge/actuation tasks (`get_ice_from_fridge_i52`,
  `libero_kitchen_scene3_turn_on_the_stove_i193`, etc.) drive mechanisms, not
  constructed spans. None combines (a) building a bridge by settling a free rigid
  body onto two supports, (b) rolling an ungraspable sphere across it, and (c) an
  irreversible hazard that dictates execution order.
- **Execution order (declared):** BRIDGE REQUIRED BEFORE BALL. The dock is on the far
  platform; the only path is over the moat; the ball cannot be lifted (jaw too
  narrow) and cannot climb out of the 120 mm moat. The plank must therefore be laid
  and verified spanning before the ball ever approaches the gap. There is no
  alternative ordering.

## Randomization (per episode, verified by readback in smoke)

Moat width sampled in [150, 200] mm (the far platform AND dock are re-posed each
episode — far-platform x readback varies); dock lateral offset ±220 mm along the far
rim; ball spawn box x ∈ [−0.44, −0.20], y ± 0.28 on the near platform; plank spawn
box x ∈ [−0.30, −0.21], y ± 0.16 with FREE YAW (|q_z| readback varies 0.17–0.99) and
a 170 mm ball keep-out (batched resampling + deterministic corner fallback). The
progress baseline `x_spawn` and mouth plane `mouth_x` are captured per episode.

## Rubric

`success()` iff, all judged live on physical settled poses:

- ball centre inside the dock interior (past the mouth plane with margin, between
  the walls);
- ball at FAR-PLATFORM rest height (centre z within 20 mm of top + ball_r — kills
  fly-through, hover, and under-the-dock-in-the-moat states);
- settled (lin < 0.06 m/s, ang < 1.5 rad/s).

`score()` (latched every physics substep in `post_step`): `0.10 ×` best ball
x-progress toward the dock mouth, HEIGHT-GATED (z > plat_h − 20 mm, so moat
wandering earns nothing) `+ 0.30 ×` BRIDGED (plank seen resting level within 12° at
rim height ± 12 mm with its deck overlapping BOTH rims by > 12 mm, near-still)
`+ 0.30 ×` CROSSED (ball seen past the far rim at platform height ± 30 mm) `+ 0.15 ×`
DOCKED (ball inside the dock, slow). Base capped at 0.85; exactly 1.0 iff
`success()`. Doing nothing scores ~0.

## Teleport solution (`solve.py`) — transport only, every write ends in free space

- **P1 — bridge:** the plank (graspable: the jaw pinches its 8 mm side rail) is
  teleported to a HOVER pose over the moat — long axis along x, centred on the
  dock's lateral line, span position overlapping both rims with the far ramp clear
  of the dock walls — velocities zeroed. It FALLS and SETTLES onto the rims through
  real contact; the scene's `bridged` predicate judges the settled result, not the
  write.
- **P2 — roll:** the ball (ungraspable — a robot would cradle it) is staged resting
  on the deck's NEAR END in the rail channel, 2 mm proud. Everything load-bearing
  then happens through contact dynamics: a capped (5 N) velocity-servo force at the
  ball's centre — the push of a fingertip at mid-height — rolls it along the
  channel, across the moat, down the far ramp and through the dock mouth; the force
  is CUT once the centre passes the accept plane and the back wall + friction stop
  the ball. The ball is never teleported past the moat, into the dock, or over any
  wall. **Build quirk handled:** on this forge build `is_global=True` is silently
  ignored (deprecated wrench path) and forces apply in the BODY frame — catastrophic
  for a rolling sphere whose frame revolves every π·d = 157 mm. The solve rotates
  the desired world force into the body frame each step
  (`quat_apply_inverse(root_quat_w, f_world)`).
- `SIM_GEN_SCORE` printed at each phase boundary is non-decreasing (latched credit),
  ≥ 3.3 simulated seconds hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.
  **Verified on the forge: seeds 0 and 1, both SUCCESS, provably distinct by stdout
  readback** — seed 0: gap 170 mm, dock y +0.208; seed 1: gap ~195 mm, dock y
  +0.076; both reach score 1.000 with persistence.

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (−0.75, 0.0), beside the near platform's outer edge: the plank spawn
box (x ∈ [−0.30, −0.21]) and the bridge hover line (x ≈ 0.0–0.1) sit within a
Franka's ~0.85 m envelope, and the far dock (x ≈ 0.43–0.51) is reached only by the
BALL, never by the hand. Per-object strategy: the PLANK is pinch-grasped from above
on one 8 mm side rail (8 mm ≪ 80 mm jaw stroke; 0.35 kg well under payload), carried
over the moat, aligned by wrist yaw with the dock's lateral line, and lowered until
the deck rests on both rims — exactly solve.py's hover-and-settle. The BALL is
worked with the fingertip/closed fist: a push at mid-height (the applied CoM force in
solve.py) rolls it along the rail channel, which self-centres the roll line, so the
robot only tracks from behind; it never needs to reach past the near rim — the
bridge and gravity (far ramp descends into the dock mouth line) deliver the ball.
The guard rails on the platforms' outer edges are 30 mm — no barrier to a top-down
rail pinch or a mid-height push.

## Checks (`smoke.py` — rejection battery, 14 named checks, ALL PASS on the forge)

1. settle: states finite; ball at rest height on the near platform, plank flat on
   the near platform (readback heights), settled.
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: far-platform x (moat width), dock x AND y vary across 6
   seeded resets.
4. randomization readback: ball xy, plank xy, plank yaw (|q_z|) vary; ball always
   spawns on the NEAR side.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: the graspable PLANK carried to the dock and settled (verified at
   the dock by readback), ball untouched → NOT success, score ≤ 0.02 (identity: only
   the ball counts, and the plank is a tool, not the goal).
7. moat loss: ball released over the gap settles on the moat floor (z < 0.07) →
   NOT success, score ≤ 0.02, crossing/docking latches stay 0 (height gate).
8. cantilever: plank level at rim height but short of the far rim → `bridged_now`
   False, bridged latch stays 0 (two-support overlap is required).
9. hover loophole: ball in the air over the dock centre → NOT success, docked and
   crossed latches stay 0 (z-band gates both).
10. bridge + ball resting mid-deck: bridged latches 1.0, yet NOT success, score
    ≤ 0.60 (observed 0.368 — no crossing, no docking).
11. mouth near-miss: ball settled inside the dock mouth but short of the accept
    plane → crossed latched, `in_dock` False, NOT success, score ≤ 0.85 (observed
    0.700).
12. latched credit: removing the ball back to the near platform leaves the latched
    score unchanged (0.700 → 0.700), success stays gone.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN across all five bodies. Plus `frames.npz` (125 × 600 × 960 × 3)
    recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest:
the ball is strictly wider than the jaw (+15 mm margin), the moat is deeper than the
ball (falling in is irreversible), the plank deck spans `gap_max` with real overlap
on both rims, the rail channel and dock interior admit the ball with margin, the
plank rail is pinchable by the jaw, the plank fits on the near platform, the dock
stays inside the far platform's guard rails at any sampled offset, the ball always
spawns on the near side, and the accept window sits strictly inside the dock mouth.
