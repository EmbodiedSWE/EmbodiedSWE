# two_robot_pick_cube_i315 — `wedge_jack`

Load an orange crate onto a green elevator platform and JACK the platform up to the
yellow beam's delivery level by driving a self-locking 15° wedge ram under it — the
elevation must be produced and SUSTAINED by the machine, never by holding.

## Seed provenance

- **Seed task:** `maniskill/two_robot_pick_cube`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/two_robot_pick_cube.py`): two
  Panda arms face each other across a table; the left arm picks a 4 cm cube, lifts
  it to a midpoint, the right arm takes the handover and carries the cube to a
  floating elevated goal pose, where it is **held**. The whole skill is a two-agent
  free-space relay to a HELD elevated pose — success is a pose match sustained by a
  gripper.

## What changed, and why it is strategically different

| | seed | this task |
|---|---|---|
| elevation | the goal pose floats in the air; the cube is **held** there by arm B | produced by a **force-amplifying machine**: a 15° wedge ram fed under the platform's matching foot (≈ 186 mm of push → 50 mm of lift); friction **self-locks** the raised state (μ 0.60 = 2.24 × tan 15°), so nothing is held |
| what sustains the goal | a gripper, forever | passive mechanics: the rubric's `supported()` clause demands the platform's height be CONSISTENT with the ram insertion — a hoisted platform is refused |
| skill | grasp, lift, handover, carry to pose | **tool use / mechanism operation**: place cargo, then a long regulated push on the ram's end plate, transforming horizontal effort into sustained vertical lift |
| success criterion | cube at the floating pose (held) | settled configuration: crate seated on the deck ∧ deck at/above the delivery level ∧ platform resting on the driven ram ∧ everything at rest |
| failure structure | miss the pose | altitude-by-holding (crate or platform) refused by aboard-z / supported clauses; empty jacking earns exactly 0 (aboard-gated); stopping 14 mm low refused by the level clause alone |

The seed's plan — carry the cargo to the elevated goal and hold it — executed here
earns ~0 while held (aboard's z clause refuses altitude) and at most the 0.30 aboard
latch after release (smoke check 5); hoisting the loaded platform by hand earns the
same 0.30 + a ≤ 0.05 support-tolerance crumb and collapses on release (check 6).
Success demands what the seed never asks: operating a passive force-amplifying
mechanism whose self-locking friction, not an agent, owns the goal state. It also
differs from every sibling read during construction: `two_robot_pick_cube_i165`
(size-graded discs, physically forced insertion ORDER — here one cargo, no ordering:
load-then-jack and jack-then-load both work, stated in `describe()`),
`franka_handover_i115` (sealed boundary crossed by a rotating carousel — here no
boundary, no rotation; the mechanism AMPLIFIES force and holds altitude),
`handover_i124` (mass arithmetic on a balance) and robobench `pen_holder`
(container filling).

## Teleport solution (solve.py) — phases

1. **Settle + perception** — station pose, sampled ram start (`xw()` readback
   cross-checked against the recorded sample) and crate slot all read from episode
   state. `SIM_GEN_SCORE 0.00`.
2. **LOAD** — crate teleported (transport only) to free air 4 cm above the deck
   centre and released; gravity + deck/rim contact seat it. `SIM_GEN_SCORE 0.30`.
3. **JACK** — feedforward + velocity servo on the scene's sanctioned `ram_drive`
   force plant (the stand-in for a sustained push on the pale end plate, 25 N
   clamp): `drive = clamp(8 + 60·(v_des − ins_rate), 0, 24)`, v_des tapered to a
   2 mm overshoot target past the 50 mm requirement; effective velocity gain
   (60+6)/(120·1.2) = 0.46 < 1 (one-substep-late wrenches). The platform + crate
   ride up on pure contact. Stall detector escalates the feedforward ×1.5 (never
   fired in practice).
4. **VERIFY** — drive cut to zero; the self-locking wedge must hold the lift.
   `success()` + `SIM_GEN_SCORE 1.00`, then 400 more hands-off steps (≥ 3.3 s);
   success must persist → `SIM_GEN_SOLVE: SUCCESS`. Passes on forge seeds 0, 1, 2
   (ram starts 138 / 111 / 107 mm — the sampled band exercised).

## Embodiment argument (single Franka arm, parallel-jaw gripper)

Base pose: ground-mounted at station-local ≈ (0.55 m, 0), facing −x (the open apron
side; the gantry and beam are on the far −x side, never between the arm and its
work). Everything the task touches lies within 0.66 m horizontal reach at ≤ 0.20 m
height — inside the Franka's 0.855 m envelope.

- **Crate (50 mm cube, 120 g):** plain top-down grasp on the apron
  (x 0.24–0.38, |y| = 0.14, open sky above), well inside the 80 mm jaw span; carried
  ≤ 0.35 m to the deck centre and released 4 cm above it — exactly the demonstrated
  move. The deck is open from above at EVERY lift height (pillars/beam are beyond
  −0.095 in x, the deck ends at ±0.075), so load-first and jack-first both remain
  reachable.
- **Ram drive (the only actuated interaction):** a sustained horizontal push on the
  pale end plate — a 100 × 96 mm vertical face at z 0.008–0.104, riding from
  station-x ≈ 0.36 in to ≈ 0.10 at full insertion, always at the arm-facing end of
  the rail with free approach from +x. Closed-jaw pushing at ≤ 7 N (analytic
  insertion load 6.8 N ≪ the arm's ≥ 30 N continuous wrench); ~200–250 mm of
  travel at 15 cm height. No pulling, no in-mechanism reach: the wedge slides UNDER
  the platform, the hand never does.
- **Self-locking = forgiving pacing:** the ram holds every intermediate insertion
  when the push pauses (μ = 2.24 × tan θ, verified hands-off in solve and smoke), so
  the arm can re-grip or re-position mid-push without losing progress.
- **Perception:** 50 mm orange cube vs green deck vs blue ram vs yellow beam —
  colour classes at wrist-camera range; the goal level is marked by the beam's
  underside, adjacent (±0.02 m in x) to the deck it must reach.

## Execution order

**Not constrained — and `describe()` says so.** Load-then-jack (demonstrated) and
jack-then-load both succeed; the deck stays open from above at every height. The
task's difficulty is mechanism operation and the no-holding contract, not
sequencing.

## Rubric (scene.py)

- 0.30 — crate EVER seated on the deck inside the rim (car-frame box, latched).
- 0.45 × latched lift fraction toward the 50 mm delivery level, gated **aboard ∧
  supported** — jacking the empty platform earns nothing, hoisting the loaded
  platform by hand earns nothing.
- Capped at 0.75; exactly 1.0 iff live `success()`: aboard ∧ deck at/above the
  delivery level ∧ platform resting on the driven ram (`q ≤ q_from_ram + 6 mm`) ∧
  everything settled ∧ finite.

## Checks

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on forge seeds 0/1/2, monotone
  `SIM_GEN_SCORE` prints at phase boundaries (0.00 → 0.30 → 1.00 → 1.00), ≥ 3.3 s
  hands-off persistence after the drive cut.
- `smoke.py`: 12 rejection-only checks — settle/no-NaN (positive ram gap, q = 0);
  randomization A (station yaw span > 90°, xy jitter, ram-start span with wedge-pose
  READBACK); randomization B (crate side both signs, slot-x spread, slot readback);
  null policy; SEED strategy (crate held at the delivery level earns ~0 for the
  whole hold, seats on the DOWN deck on release — aboard latch only); hoist cheat
  (car + crate held at q = 54 mm with the ram out: `supported()` False throughout,
  lift latch never pays, falls back on release); empty jack (REAL drive to full
  insertion, q ≥ 50 mm readback, aboard-gated ~0, self-lock holds < 2 mm drift over
  2 s hands-off); low near-miss (aboard + supported + settled at 36 mm — the level
  clause alone refuses, credit under the 0.75 cap); latched credit survives theft;
  rejection audit (success never observed); final no-NaN; frames.npz video
  (305 frames). `SIM_GEN_SMOKE: ALL PASS 12/12`.
