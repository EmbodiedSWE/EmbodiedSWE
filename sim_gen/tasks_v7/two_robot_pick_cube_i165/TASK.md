# two_robot_pick_cube_i165 — `sorting_tower`

Gauge-sort three graded GREEN discs down a keyed tower shaft: drop them through the
one top mouth in ASCENDING size order so each seats flat at its own keyed depth, and
leave the RED distractor disc out of the shaft.

## Seed provenance

- **Seed task:** `maniskill/two_robot_pick_cube`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/two_robot_pick_cube.py`): two
  Panda arms face each other across a table; the left arm picks a 4 cm cube, lifts
  it to a midpoint, the right arm takes the handover and carries the cube to a
  floating elevated goal pose. The whole skill is a **two-agent relay to a held
  elevated pose** — success is a pose match, and the only ordering (A before B)
  lives in the agents.

## What changed, and why it is strategically different

| | seed | this task |
|---|---|---|
| elevation | the goal pose floats in the air and the cube is **held** there | each disc's elevated rest height is produced by **size-keyed seating** on internal funnel bevels — nothing is held, the plant sustains the pose |
| ordering | implicit in the two agents (A hands to B) | **physically forced by passive geometry**: a seated disc wedges the shaft, so only smallest-to-largest works; any larger-first order strands discs on top (proved by real drops in smoke) |
| skill | grasp, lift, handover, carry to pose | **size perception → sequencing** (grade three diameters, insert in ascending order), one shared insertion point (the mouth), distractor discipline (red disc of episode-varying size must stay out) |
| success criterion | cube at the floating pose | a settled configuration of THREE bodies at three different keyed depths inside a shaft, plus a negative clause (no red aboard) |
| failure structure | miss the pose | wrong order ⇒ perched discs ~2–6× the z tolerance off their bands; same-size red swap ⇒ level-proof rejection by colour/cleanliness; off-axis rests rejected by xy alone |

The seed's plan — carry each object to its destination pose, order-blind — executed
here with real drops seats only the first (largest) disc and strands the other two
on top of it: 0.22, never success (smoke check 5). Success demands what the seed
never asks: reading disc size as a sequence key and respecting a passive mechanical
interlock. It is also different from every sibling task read during construction:
`franka_handover_i115` (sealed boundary crossed by an actuated carousel — here there
is no boundary and no mechanism, the tower is fully passive), `handover_i124`
(size→mass arithmetic on a balance with NO required order — here size→sequence with
a hard forced order and no equilibrium plant), and robobench `pen_holder` (any-order
container filling — here order is the content of the task).

## Teleport solution (solve.py) — phases

1. **Settle + perception** — slot permutation, which size the red disc is (left
   untouched), tower pose: all read back from episode state. `SIM_GEN_SCORE ~0.00`.
2. **INSERT ×3 (ascending)** — each green disc teleported (transport only) to free
   air 3 cm above the mouth rim, axis vertical, and released; gravity + the mouth
   funnel + internal funnels/tubes — real contacts — guide it to its keyed seat.
   `SIM_GEN_SCORE 0.22 / 0.44 / 0.66` as the latches pay.
3. **VERIFY** — hands off until `success()` (all seated ∧ no red inside ∧ settled ∧
   finite), `SIM_GEN_SCORE 1.00`, then ≥ 3.3 more simulated seconds hands-off;
   success must persist → `SIM_GEN_SOLVE: SUCCESS`. Passes on forge seeds 0, 1
   and 2 (red distractor sizes 2, 0 and 1 — all sampled sizes seen and left out).

## Embodiment argument (single Franka arm, parallel-jaw gripper)

Base pose: ground-mounted at tower-local ≈ (0, −0.55 m), facing +y. The slab top is
40 mm off the ground; the mouth rim is at 191 mm; all four slab slots lie on a
0.21 m circle — everything within ≤ 0.62 m horizontal reach at ≤ 0.20 m height,
inside the Franka's 0.855 m envelope.

- **Discs (28/48/68 mm dia, 16–20 mm tall, 50–180 g):** every diameter fits the
  80 mm jaw span with ≥ 12 mm margin (asserted in `__post_init__`); top-down rim
  grasp on the slab (open sky above every slot; adjacent slots are ≥ 0.30 m apart,
  discs never touch). 180 g ≪ the 3 kg payload.
- **Insertion:** hold the disc flat over the golden mouth (116 mm square opening —
  ±48 mm centring slop for even the largest disc) and open the jaws; the 45° mouth
  funnel and internal funnels do all fine alignment. The demonstrated release
  (3 cm above the rim, axis vertical) is exactly this move. No in-shaft
  manipulation is ever needed — the shaft interior is reached only by falling
  cargo.
- **The tower is passive and kinematic** — nothing to actuate, nothing to hold;
  disc–funnel friction (μ ≈ 0.4 < tan 45°) guarantees released discs slide to
  their seats, so partial progress is never undone.
- **Perception:** size classes are separated by 20 mm — trivially resolvable by a
  wrist camera; the distractor differs by COLOUR only (green vs red), a colour
  read at slab range.

## Execution order

**Required, and physically enforced:** smallest → mid → largest. The order is not a
rubric convention — a seated disc wedges the shaft shut (annular gap smaller than
any disc, audited), so any other order strands discs above their bands and caps the
score at one seat. `describe()` states the order and why; smoke proves both wrong
orders fail with real drops.

## Rubric (scene.py)

- 0.22 per green disc EVER seated at its own band — flat (≤ 15° tilt), centred
  (radial < 12 mm), at its keyed height (|Δz| < 8 mm), tower frame — latched,
  capped at 3 × 0.22 = 0.66.
- Exactly 1.0 iff live `success()`: all three seated ∧ NO red disc inside the
  shaft ∧ everything settled ∧ finite.

## Checks

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on forge seeds 0/1/2, monotone `SIM_GEN_SCORE`
  prints at phase boundaries (0.00 → 0.22 → 0.44 → 0.66 → 1.00), ≥ 3.3 s hands-off
  persistence.
- `smoke.py`: 13 rejection-only checks — settle/no-NaN (shaft empty), tower
  yaw/xy randomization, slot-permutation + red-size randomization with position
  READBACK (greens on named slots, active red on the fourth, parked reds in the
  depot), null policy, SEED strategy (descending-order real drops → one seat,
  0.22 cap), order interlock (mid first really seats, small then perches;
  seated=(F,T,F)), cleanliness (all three seated at the 0.66 cap yet refused by
  the red-inside clause alone), red poison (red really seats at its band; the
  same-size green perches one disc height off — tightest z near-miss), wrong-place
  rest (seat height + flat on the slab — xy clause alone refuses), latched credit
  survives theft, rejection audit (success never observed), final no-NaN,
  frames.npz video.
