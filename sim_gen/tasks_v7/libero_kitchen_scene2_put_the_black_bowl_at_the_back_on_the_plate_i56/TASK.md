# hanoi_plates — relay the three-plate tower to the marked post (Hanoi rules)

Task id: `libero_kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate_i56`
Scene: `simgen.hanoi_plates` (`scene.py`, robot="null")

## Seed provenance

Seed: `libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate`
(RoboVerse `roboverse_pack/tasks/libero_90/...put_the_black_bowl_at_the_back_on_the_plate.py`).
The seed asks for ONE act: identify the back bowl among three identical black bowls and
set it on the plate — a single unordered pick-and-place judged by a final-state xy/z
window.

## What this task is

A wooden rack carries three vertical posts (150 mm apart, 115 mm tall). A tower of
three square ring-plates — RED 100 mm, YELLOW 80 mm, BLUE 60 mm, each a picture-frame
with a through aperture and a raised boss ring — starts threaded on a randomized START
post, biggest at the bottom. A green pad marks the randomized TARGET post. Goal:
rebuild the tower on the target post (red under yellow under blue, every plate
threaded), obeying three rules monitored at EVERY physics step and latched permanently:

1. move only ONE plate at a time (never two plates off the posts at once);
2. a plate may only be set down threaded on a post — never parked on the plank, the
   ground, or anywhere else;
3. never rest a larger plate on a smaller one, on any post.

Randomization (readback-verified in smoke): rack xy jitter ±30 mm, yaw 90°±20°, and
the (start, target) post pair sampled over all 6 ordered pairs.

## Why it is strategically different

- **From the seed**: the seed is one unordered pick-and-place judged on the final
  state. Here the final arrangement is necessary but NOT sufficient — the rubric is a
  trajectory contract. The three latched rules make the canonical 7-move Hanoi
  recursion (through the spare post, with the marked post as the goal) the ONLY legal
  plan; a bulk carry of the stack — the seed's strategy scaled up — is latched as a
  rule-1 violation while the stack is still in the air, and a physically perfect final
  tower built after any violation is worth exactly 0 (both proved in smoke).
- **From every other task read**: pen_holder is container filling; close_microwave_i4
  is extract-then-release with a physical interlock. Neither has a constraint-governed
  move *sequence*, a buffer resource, or rules-over-the-whole-trajectory as the core
  difficulty. No task in the corpus read uses a Tower-of-Hanoi relay (checked that
  `robobench/suites/articulated/scenes/balance_scale.py` exists and deliberately
  avoided that design).

## Solution outline (solve.py — the legitimacy certificate)

Teleports are TRANSPORT ONLY: each of the 7 moves is one root-state write carrying a
single plate to ~6 mm above the destination post tip (flat, zero velocity, outside the
threaded z-span). Threading, sliding 60–110 mm down the post, landing on the plank /
pad / the boss of the plate below, tilting and settling are pure gravity + contact —
every fact `success()` checks is produced by dynamics, never written. The canonical
sequence (S = start, T = target, P = spare):

`blue→T, yellow→P, blue→P, red→T, blue→S, yellow→T, blue→T`

with latched stage credit 0 → 0.15 → 0.50 → 0.75 → 1.0 asserted non-decreasing at
every phase boundary, then a hands-off 3.33 s persistence window before
`SIM_GEN_SOLVE: SUCCESS`.

## Execution order: REQUIRED

Order is load-bearing, enforced by the rules themselves: red can reach the target
bottom only after blue and yellow have been relayed to the spare post (rule 3 blocks
any larger-on-smaller shortcut, rule 2 blocks laying plates aside, rule 1 blocks
carrying two at once). The 7-move order above is forced up to the mirror-free choice
of buffer. smoke.py proves the wrong orders die: bulk carry (rule 1), red-onto-blue
(rule 3), parking (rule 2), and a legal relay to the wrong post (no success, credit
stuck at 0.15).

## Franka embodiment argument (base at world origin, on the ground, facing +x)

- **Reach**: rack centre at (0.42±0.03, ±0.03) m, posts at ±150 mm along the rack's
  long axis (yaw 90°±20° → posts spread roughly along world y). Every grasp/drop point
  lies 0.27–0.60 m from the base at heights 0.03–0.20 m; lifting a plate clear of a
  post needs ≤ 0.16 m of lift. All well inside a Franka's ~0.85 m envelope with the
  wrist above the work.
- **Per-object graspability** (Franka gripper opens 80 mm):
  - *Plates* (the only objects moved): pinch top-down across the 16 mm frame
    thickness on the frame margin outside the boss ring (margin widths red 24 mm /
    yellow 14 mm / blue 4 mm+boss shoulder), or across a whole frame side (60–100 mm
    is too wide for red/yellow — use the top-down pinch; blue's 60 mm side also fits
    the 80 mm opening). When plates are stacked, the 12 mm boss ring holds a clear
    12 mm finger gap between plates, so the top plate's frame margin is always
    pinchable without touching the plate below.
  - *Threading tolerance*: apertures leave 13/11/9 mm of radial slack around the
    14 mm post — a plate released roughly centred above a post tip self-threads under
    gravity (exactly what solve.py's drops demonstrate).
  - *Rack, posts, pad*: kinematic fixtures, never grasped.

## Files

- `scene.py` — cfg + spawners + `HanoiPlatesScene` (describe/instruction/reset/
  get_state/set_state/threaded/at_rest/success/score, trajectory monitors in
  `post_step`), registered as `simgen.hanoi_plates` with robot="null".
- `solve.py` — the 7-move teleport-drop relay; monotone `SIM_GEN_SCORE`, 3.33 s
  persistence, `SIM_GEN_SOLVE: SUCCESS`.
- `smoke.py` — 9-check rejection battery, `SIM_GEN_SMOKE: ALL PASS 9/9`, saves
  `frames.npz`.

## Verification record (forge, RTX 4090)

- solve `--seed 0`: rack (+0.448,−0.029) yaw 86.0°, start 1 → target 2 —
  scores 0 / .15 / .15 / .15 / .50 / .50 / .75 / 1.0 — `SIM_GEN_SOLVE: SUCCESS`.
- solve `--seed 1`: rack (+0.430,−0.015) yaw 105.6°, start 0 → target 1 — same
  monotone ladder — `SIM_GEN_SOLVE: SUCCESS`.
- smoke: `SIM_GEN_SMOKE: ALL PASS 9/9` —
  1. settle/no-NaN + baseline score 0;
  2. randomization readback (yaw Δ16.1°, xy Δ29 mm, 5 distinct (start,target) pairs
     over 10 resets);
  3. null policy: 240 idle steps, score 0, no violation;
  4. SEED STRATEGY bulk carry: rule 1 latches mid-fall, score 0 forever;
  5. perfect tower built AFTER a violation: arrangement clauses all True, success
     False, score 0;
  6. red dropped onto blue: rule 3, score 0;
  7. near-miss: plate on the plank 70 mm off-axis is not threaded (gate 30 mm), and
     rule 2 wipes a 0.75 score to 0;
  8. legal 7-move relay to the WRONG post: no violation, perfect tower, no success,
     score 0.15;
  9. frames.npz saved (105 frames, 960×600).
