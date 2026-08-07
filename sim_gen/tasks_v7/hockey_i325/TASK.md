# gate_goal — extract the captive gate board, then roll the white ball into the covered goal

**Package:** `sim_gen/tasks_v6/hockey_i325`
**Env:** `simgen.gate_goal` (scene-level, `robot="null"`)

## Seed provenance

`rlbench/hockey` (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/hockey.py`): grasp a
hockey stick and STRIKE the ball across open floor into an open-mouthed goal — one
ballistic tool swing at a passively available receptacle; success is a bbox check
relative to the goal.

## What changed and why it is strategically different

The goal is SEALED and ROOFED, and the stick is gone — the seed's entire plan (tool
strike at an open mouth) is dead on arrival:

- The goal becomes a **roofed chamber** whose only ball-sized opening is a floor-level
  mouth. The roof covers the whole interior, so a lobbed/dropped ball rests **on the
  roof** and scores nothing (smoke #7). Nothing about this task is solvable from above.
- The mouth is sealed by a **portcullis**: a yellow gate board standing in a vertical
  channel (20 mm post pairs fore/aft with ±6 mm play, end caps sideways with ±5 mm
  play). The board is CAPTIVE — it cannot tip, cannot slide sideways, cannot be pushed
  through; the **only exit is a ~13 cm straight vertical extraction** by its exposed
  top edge (40 mm proud of the posts). The seed's move — fire the ball at the goal —
  bounces off the closed gate and scores nothing (smoke #6 fires the shot and watches).
- Even with the gate out, the open strip above the channel is **50 mm — narrower than
  the 60 mm ball** (smoke #8 drops the ball on it), so the ball must travel the floor
  through the mouth. There is no stick; a controlled hand push is the intended entry.
- A same-size **black distractor ball** (slot-swapped with the white target every
  episode) adds a color-grounded identification clause (smoke #11) and an exclusion
  clause (smoke #12).

A solver therefore needs a different PLAN (constrained vertical extraction of a captive
barrier → park it → controlled floor-level conveyance through a low aperture; no tool,
no strike, two ordered phases instead of one swing) and different CODE STRUCTURE
(channel-extraction control + push control instead of stick grasp + swing trajectory).
The seed's plan, executed here, produces the smoke #6 reject state.

## Teleport-solution outline (solve.py, phases; `SIM_GEN_SCORE` at each boundary)

- **P0** reset + settle; layout readback printed (goal offset/yaw, slot assignment).
- **P1 EXTRACT THE GATE (contact dynamics — the first load-bearing interaction):** a
  velocity-limited vertical CoM force (≈ weight + 1.6 N, v ≤ 0.30 m/s — the wrench
  emulation of the arm's pinch-lift on the top edge) slides the captive board UP
  THROUGH its channel under contact. Only after readback shows the bottom edge above
  the posts (CoM > post_h + board_h/2) is the board a free body in open air; one pose
  write then parks it flat on the floor — free-space transport of an already-freed
  object. The channel was exited through the channel.
- **P2 TRANSPORT (teleport):** one pose write stages the white ball 16 cm OUTSIDE the
  mouth plane on the goal axis — asserted outside; staging earns only mouth-approach
  credit.
- **P3 ROLL IN THROUGH THE MOUTH (contact dynamics):** a velocity-limited world-frame
  CoM force (≤ 4 N, v ≤ 0.22 m/s, small lateral centering term) rolls the ball along
  the floor through the mouth; the force is CUT at the inside line (mouth + r + 15 mm);
  the ball coasts and settles hands-off. success() first turns True here.
- **P4** hands-off persistence ≥ 3.3 s, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds **0 and 1**, both `SIM_GEN_SOLVE: SUCCESS` first run (scores
monotone 0 → 0.21/0.22 → 0.24 → 1.0 → 1.0; white ball settles at u = +0.130 against
the back wall).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

Plausible base pose: **base at the world origin on the floor plane**; all required
contacts lie at radius 0.28–0.58 m, heights 0.03–0.31 m — inside the Franka envelope.

- **Yellow gate board:** top-down pinch across its 8 mm thickness anywhere on the
  exposed top edge — the edge spans the full 160 mm central mouth width at height
  0.13–0.17 m, in free air (the roof top is at 0.122 m and sits ≥ 20 mm behind the
  board plane; the channel posts stop at 0.13 m). Extraction is a pure vertical
  13 cm stroke ending at 0.30 m — comfortable OSC motion; the ±6/±5 mm channel play
  is an order of magnitude above OSC noise, and the channel itself funnels the stroke.
  Set the board down anywhere clear (describe() suggests the free patch (0.16, 0.42)).
- **White ball:** never grasped — pushed. Closed-fingertip contact on the trailing
  side of the 60 mm ball at ~30 mm height, rolling it along the floor. The mouth is
  160 mm wide × 110 mm tall vs the 60 mm ball: ±50 mm lateral tolerance, far above
  control noise; the final push segment needs at most fingertip intrusion barely past
  the mouth plane (the inside line is only 15 mm + r past it, and the ball may coast
  the last stretch). No contact ever happens under the roof deeper than ~6 cm.
- **Black ball:** requires no contact — it must merely be left alone.

## Execution order

Required and geometry-forced (declared in describe()): the seated board covers the
full mouth with no ball-sized gap (200 mm board vs 160 mm mouth, bottom on the floor),
so the ball cannot enter until the board has left the channel — enforced by collision,
not by rubric timestamps. The roll-in necessarily comes second.

## Rubric

`score()` = 0.15·gate_out (board ever out of the gate zone, latched — the captive
channel means only a real extraction, or a smoke probe, ever moves it) +
0.15·mouth-approach (gated on gate_out, latched running max) + 0.45·inside (white ball
ever inside the chamber past the inside line, latched), capped at 0.75; exactly
**1.0 iff `success()`**: white ball inside (u ∈ [0.045, 0.155], |v| ≤ 0.075, resting
height under the roof, goal-frame math) ∧ at rest ∧ black ball NOT inside. Null policy
scores ~0 (the board starts seated, so every term is gated or latched off).

## Check list (smoke.py — rejection battery, forge: `SIM_GEN_SMOKE: ALL PASS 16/16`)

1. settle/no-NaN: board seated in the gate (mouth barred), balls at slots, still, finite
2. score ~0 at reset, no success
3. randomization readback: white/black slot assignment flips; always opposite slots
4. randomization readback: per-slot xy jitter (> 4 mm) and goal yaw spread (> 2°)
5. null policy (240 steps): score ~0, no success
6. **seed strategy**: ball FIRED at the sealed mouth → stops outside against the closed
   gate, board stays seated → score ~0, no success
7. drop-in (roof): released over the box → rests on the roof → score ~0, no success
8. drop-in (slot): gate out, dropped over the open 50 mm channel strip → lands outside
   the inside line (slot narrower than the ball) → NOT success
9. near-miss mouth: gate out, settled just OUTSIDE the mouth plane → NOT success
10. near-miss line: settled past the mouth plane but short of the inside line
    (u = 0.02 < 0.045) → NOT success
11. wrong object: BLACK ball inside, white untouched, board seated → score ~0
12. exclusion clause: white AND black both settled inside — every white gate passes,
    the black clause alone rejects → NOT success, score ≤ 0.75
13. latched credit: regressing the white ball out leaves the latched score unchanged
14. beside-wall: settled against the OUTSIDE of a chamber wall at in-range depth →
    v gate rejects (goal-frame math, valid under yaw randomization)
15. rejection audit: success() never True anywhere in the battery
16. final no-NaN

frames.npz (273 × 600 × 960 × 3) recorded and saved in cwd by smoke.py.
