# pick_single_egad_i4 — Tag Hangers

Hang two colored tags on the pegs of their SAME-colored stands (scene `tag_hangers`).

## Seed provenance

Seed: `maniskill/pick_single_egad` (RoboVerse
`roboverse_pack/tasks/maniskill/pick_single_egad.py`) — pick one EGAD object off the
table and raise it 7.5 cm (`PositionShiftChecker(distance=0.075, axis="z")`). The seed
is a bare pick-and-lift: grasp affordance + a vertical position delta, no placement, no
relation between objects.

## What the task is

Three free-standing kinematic stands (RED, BLUE, GRAY decoy) stand in a shuffled,
jittered row. Each carries one square peg (16 mm across, 100 mm long, pitched 8° UP)
jutting horizontally from its post at 26 cm height; per-episode randomization permutes
the stand order, jitters positions, and swings each stand's yaw (peg heading) by ±25°
around "facing the tags". Two dynamic tags (RED, BLUE, 80 g) lie flat on the ground:
each is a square washer frame with a 32 mm square aperture plus a 50 × 60 mm handle
plate hanging below the washer.

Goal: for each tag, thread the matching stand's peg through the tag's aperture from the
peg's free tip and release it so it hangs freely — washer's inner top bar resting on the
peg, handle plate dangling, settled. Both tags must hang simultaneously. Wrong-colored
stand, the gray decoy, resting on top of a peg, or leaning anywhere counts for nothing.

## Strategic difference

- **vs. the seed** (free-space pick-lift): the object here is a *tool-like carrier* that
  must be re-oriented (ground-flat → aperture-axis-horizontal, aligned with a per-episode
  peg heading), *threaded* over a 70 mm peg run with 8 mm radial clearance, and
  *released into a stable suspension*. Success is a relational suspension predicate
  (ring below peg axis, hanging settled), not a position delta of the grasped object.
  A pick-and-lift end state (tag held in the air) scores only the small transport latch
  and can never reach success — smoke check 6 proves the seed strategy is rejected.
- **vs. corpus tasks read**: no gravity-gate door closure (`close_microwave_i4`), no
  pour-then-invert-park (`libero...i2`), no tool-as-ramrod ejection
  (`peg_insertion_side_i1`), no pin-unlock + covered tunnel slide into a bin
  (`pick_single_egad_i3`), no ordered gravity drop-stack (`setup_checkers_i2`). The
  core mechanic here — *suspension established by threading then releasing onto a
  horizontal peg* — appears in none of them.
- **vs. robobench house suites**: `pen_holder` inserts pens *downward into a cup*
  (gravity-assisted containment); this task threads *horizontally onto a peg* and is
  judged by hanging, the opposite support relation. `balance_scale`, `combination_safe`,
  `syringe_dosing`, packing/pouring suites share no mechanic. (An earlier balance-scale
  design for this slot was discarded exactly because `balance_scale.py` exists.)

## Solution outline (the teleport solution = legitimacy certificate)

Per tag (red then blue; order is NOT required by the rubric):

1. **Transport (teleport)** — the tag is teleported from the ground to a free-air pose
   just off the MATCHING stand's peg *tip* (outside the peg span; scores only the
   transport-legal lift/approach latches), aperture axis aligned with the peg heading.
2. **Threading (contact)** — a gravity-compensated hold (the arm's grip) plus a
   velocity-regulated axial push (0.8 N, escalating on stall) and weak centering springs
   drive the tag tip-first along the peg under real collision: ~70 mm of peg pass
   through the 32 mm aperture; misalignment scrapes washer bars on the peg.
3. **Release (contact)** — all external forces cleared mid-peg. The tag falls ~8 mm
   until the washer's inner top bar lands on the peg, swings as a pendulum, and rings
   down to a settled hang. The judged suspension is established by gravity + contact,
   never written.

Then ≥ 3.3 simulated seconds hands-off persistence before `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and is non-decreasing (transport
latches are monotone; hang credit only adds).

## Rubric

- 0 → 0.60: latched shaping per tag — lift off the ground (0.04), approach the matching
  peg (0.10), aperture threaded onto the matching peg (0.16); latches never decay.
- 0.90: both tags simultaneously hanging (geometry: peg through aperture, ring below
  the peg axis in the hang window, on the SAME-colored stand).
- 1.00: success — both hanging AND settled (velocity thresholds).

Unfakeable from the ground: the tallest ground-supported pose puts the washer ring at
≈ 0.174 m, far below the 0.26 m peg axis (asserted in `__post_init__`), so no
resting/leaning configuration can enter the hang window.

## Embodiment argument (Franka, parallel-jaw)

- The handle plate is 16 × 50 × 60 mm: a Franka gripper (~80 mm stroke) pinches it
  across its 50 mm width with full finger-pad contact; the washer then sticks out
  beyond the fingers, so the aperture is never occluded by the grasp.
- Threading clearance is 8 mm radial (32 mm aperture vs 16 mm peg) over a 100 mm peg —
  comfortably inside closed-loop visual-servo accuracy for a wrist-mounted or external
  camera; the 8° upward peg pitch means small axial pushes seat the tag and gravity
  holds it against walk-off after release.
- Base pose: place the Franka base at ≈ (−0.45, 0, 0) facing +x. Stands sit at
  x ≈ 0.29–0.35, |y| ≤ 0.27, pegs at 0.26 m height pointing generally back toward the
  robot (yaw = 180° ± 25°) — every peg tip and both tag spawn zones (|y| ≈ 0.07–0.15)
  are inside a 0.85 m reach envelope at comfortable heights (ground pick at 0.01 m,
  hang at 0.26 m).
- Execution order: NONE required — tags may be hung in either order (the solve does
  red-then-blue only as one valid demonstration).

## Checks (smoke.py, rejection-only battery — 15)

1. settle + no-NaN after reset; 2. baseline score ≤ 0.02; 3–4. randomization readback
across seeds (stand y/yaw spread, tag y spread); 5. null policy 2 s ≈ 0 score;
6. SEED-strategy rejection: lifting the red tag > 0.15 m in free air never succeeds and
scores ≤ 0.15; 7. wrong color: red tag physically hung on the BLUE peg — real
suspension, yet not success and thread latch stays 0; 8. gray decoy hang rejected;
9. suspension honesty: the wrong-color hang survives a 0.8 N lateral shove — a draped
fake would be swept off, a threaded tag is retained by the peg through the hole (real
hang, rejected only by color); 10. resting flat on top of the peg (no through-hole)
never counts;
11. tag leaning at a stand's foot not success; 12. one correct hang alone < 0.9, not
success; 13. latched credit survives teleporting the tag back to the ground (no decay);
14. rejection audit: success never fired during checks 5–12; 15. final no-NaN.
frames.npz (rgb) saved from a replicator camera.

Run (forge):
`python -u -m simgen_tasks.pick_single_egad_i4.solve --headless [--seed N]`
`python -u -m simgen_tasks.pick_single_egad_i4.smoke --headless`
