# living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_i194 — Pagoda Relay

Scene: `pagoda_relay` · Env: `simgen.pagoda_relay` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed: `libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray`
("stack the left bowl on the right bowl and place them in the tray"): stack one free bowl
on another (an xy/dz readout) and park the pair inside a tray's bounding box. Two
free-space pick-and-place moves, in whatever order the solver likes — nothing constrains
WHICH move is legal WHEN, and the finished stack can even be carried as a unit.

## What changed and why it is strategically different

The stack-then-deliver goal is kept, but move ORDER becomes the task — a physical
Tower-of-Hanoi relay. The three "bowls" become stepped pagoda tiers (crimson large /
amber mid / teal small, strictly nested sizes and masses), each with a 34 mm square
center hole THREADED over a slender steel post (22 mm dia, tip 13 mm proud of the full
stack). Threading makes captivity physics, not rules: only the TOP tier of any pile can
be extracted — smoke #8 shoves the loaded base tier sideways with a sustained 8 N and it
just rattles across the 6 mm threading slack, pinned. Three posts stand on the table:
the red start pedestal (pagoda spawns assembled there), the blue buffer pedestal, and
the GOAL post rising from the floor of a walled wooden tray. Two rules are enforced by
the scene itself, as per-substep debounced latches in `post_step`: (1) at most ONE tier
off a post at any instant — parking a tier on the table is legal but freezes the relay
(moving anything else trips the latch), and (2) never a bigger tier above a smaller one
on any post. A tripped latch VOIDS the episode: score pinned 0, success impossible until
reset. With one buffer, the shortest legal plan is the classic 7-move relay — the small
tier alone must be parked and re-picked three times. The seed's own strategy (carry the
finished stack in one go) lands geometrically PERFECT on the tray post and still scores
0 (smoke #9): sequencing, not placement, decides the episode.

Distinct from the corpus (survey of every tasks_v7 card): no existing task constrains
the ORDER of individually-trivial moves via physical captivity, and none has a
rules-of-the-game layer enforced by continuous latches. The corpus is decided by force
budgets, mechanisms, pouring, toppling, balance, or geometry — all their subgoals can be
attempted in any order; the nearest stack-things tasks (bowl stacking, tower builds)
accept any construction order and never void an episode for a wrong move.

## Teleport-solution outline (solve.py)

1. **P0** settle + layout readback (station xy/yaw under randomization); assert all
   three tiers seated on the start post at levels 0/1/2, score ≈ 0.
2. **P1–P7 the seven legal moves** (transport only): before each move, `assert_top`
   proves by readback the tier is the TOP of its pile — exactly what captivity would let
   a gripper take. Each move hover-teleports the tier with ZERO velocity 12 mm above the
   destination post tip (3 mm lateral offset); gravity threads the square hole down the
   round post; seating, level height, and settle are verified by readback (≤3
   re-carries of the still-in-flight tier). No tier is ever written into a seated pose.
   Sequence: S→tray, M→B, S→B (onto M — legal), L→tray (0.40), S→A, M→tray (0.70),
   S→tray. `SIM_GEN_SCORE` printed at every boundary, latched non-decreasing
   0 → 0.15 → 0.40 → 0.70 → 1.0; rule latches asserted clean after every move.
3. **P8** final settle → success TRUE; **P-persist** hands-off 400 substeps (3.33 s at
   120 Hz) → `SIM_GEN_SOLVE: SUCCESS` only if success still holds. Verified on seeds 0
   and 3.

## Embodiment argument (Franka, base at ~(0, −0.55), facing the table)

The station pattern spans ~0.46 m; everything lies within ~0.65 m of the base.
Per-object contact strategy:

- **Tiers** (flanges 15/12/9 cm square, 12 mm thick, 0.45/0.30/0.18 kg): every tier is a
  stepped plate — a narrow base pedestal under a wide grasp FLANGE, leaving a ≥15 mm
  overhang ring with 22 mm of finger clearance beneath (`__post_init__` asserts all of
  this). A parallel jaw pinches the 12 mm flange edge top-to-bottom, lifts straight up
  along the post until the hole clears the tip (≤0.65 N·m wrist moment at the worst
  grasp), free-space carries, and lowers over the destination post — the post/hole 6 mm
  slack self-funnels the drop, exactly the zero-velocity release the solve performs.
  The post tips at 13.3 cm are below any wrist-collision height.
- **Posts / pedestals / tray** (kinematic, 20 kg): never manipulated — they are the
  fixture that enforces captivity.
- No bimanual holds, no force beyond a 6 N lift, no reorientation (tiers stay
  horizontal; yaw is irrelevant — the holes are square but the posts are round with
  6 mm clearance per face).

## Execution order declared

Built in this order: (1) minimal success() predicate + scene, (2) working solve iterated
on the forge (base relief hole to kill an edge-on-edge contact limit cycle, settle gate
raised above the GPU phantom-velocity readback with position windows + persistence doing
the real at-rest verification), (3) final rubric (latched weights 0.15/0.25/0.30, 1.0
iff success, violated ⇒ 0), (4) smoke battery.

## Checks (smoke.py)

20 checks: settle/no-NaN baseline (pagoda assembled at levels 0/1/2) · fresh-reset score
~0 · randomization readback 8 seeds (start pedestal xy+yaw, tray xy+yaw, tier headings)
· null policy 300 steps · captivity-is-top-only live-force probe (4 N lifts the top tier
clear of the post, re-drop re-seats; single legal carry) · lateral-captivity probe (8 N
sustained shove rattles the pinned base tier 12 mm across the slack but cannot extract
it) · SEED-strategy reject (whole stack carried at once lands geometrically perfect yet
violated, score 0) · parking-is-legal (one tier on the table: not violated) · parked-
tier-freezes-relay pair (second carry trips the latch) · wrong-order reject (large
threaded above mid: order latch, score 0) · near-miss 6-of-7 legal moves = exactly the
0.70 partial credit · beside-the-post rest (in the station, not threaded ⇒ not seated)
· acceptance construct (7th move → success TRUE, 1.0) · latch-flip pair (identical
geometry, violated latch on → score 0 / exact state restored → success TRUE) ·
settle-gate kick · rejection audit · final no-NaN · frames.npz saved.
