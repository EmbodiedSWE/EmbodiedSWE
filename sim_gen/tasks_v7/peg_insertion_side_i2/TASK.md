# peg_insertion_side_i2 — Pin the Latch (scene `hasp_pin_link`)

Fasten a free green bar to an elevated platform: first lay the bar flat on the
platform's top plate so the 26 mm square hole in the bar's eye stacks over the plate's
26 mm square through-bore, then take the blue headed pin, hold it vertical, and lower
it down through BOTH aligned holes until its 34 mm head seats on the bar's eye and its
tip hangs clear through into the open gap under the plate. A red pin with a 30 mm shaft
is a decoy that fits neither hole.

## Provenance

- **Seed:** `maniskill/peg_insertion_side`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/peg_insertion_side.py`) — grasp a
  peg lying on the table and insert it sideways into a pre-existing hole in ONE fixed
  block; the checker is pure relative-bbox containment of the peg in the block's frame.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `hasp_pin_link`,
  env `simgen.hasp_pin_link`, robot `"null"`), `solve.py` (teleport solution),
  `smoke.py` (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed is a single TERMINAL insertion into a passage that already
  exists; the episode ends the instant the peg's pose is deep enough. Here the passage
  the fastener needs DOES NOT EXIST at reset — the solver must first CREATE it by
  seating the bar's eye over the platform bore (a plate-on-plate alignment sub-goal
  with its own 10 mm tolerance), and only then can the pin thread the two stacked
  holes. The judged outcome is a fastened two-body LINKAGE (bar seated + pin through
  BOTH holes + head carried by the eye), not a peg-in-block bbox. The seed's whole
  strategy — "put the peg into the fixed block's hole, done" — is exactly this task's
  out-of-order failure state and is explicitly rejected (smoke check 6).
- **vs `peg_insertion_side_i1` (read this session):** i1 is tool-use — ram a rod
  through a tube to EJECT a judged third body into a basin; the rod is instrumental
  and the judged object is never touched. Here there is no ejection and no third body:
  both manipulated bodies are themselves the judged outcome, coupled by an ordered
  fastening.
- **vs `robobench/suites/packing/scenes/pen_holder.py` (read this session):** that is
  repeated container insertion (pens tip-up into a holder). Here nothing is placed in
  a container; the two sub-goals are heterogeneous (surface alignment, then vertical
  threading) and strictly ordered.
- **Execution order is REQUIRED and physically enforced** (declared): the pin head
  (34 mm) is wider than both holes (26 mm), so a pin dropped into the bare bore first
  leaves a captive head sticking up that the bar's eye can never pass over. Bar
  BEFORE pin, always.

## Randomization (per episode, verified by readback in smoke)

Stand xy jitter + free yaw (the bore axis and hole orientation move — the solver must
read them from the scene), bar xy + free yaw, pin xy + free yaw (lying flat), decoy
xy + free yaw, with batched keep-out resampling so nothing spawns intersecting. The
approach baseline `d0` is captured per episode.

## Rubric

`success()` iff, settled (bar AND pin |v| < 0.05 m/s):

- bar SEATED: eye centre within 10 mm of the bore axis per axis (STAND frame),
  resting flat on the plate top (±6 mm), upright (≤15°);
- pin LINKED: near-vertical (≤25°), tip inside the bore cross-section (STAND frame)
  and below the plate underside by ≥8 mm (and above the ground — the head carries the
  pin), shaft passing through the bar's eye (axis point at the eye plane, BAR frame),
  and the HEAD seated at the eye top (kills the axis-line loophole of a pin standing
  on the ground in the gap under the plate).

`score()` (latched every physics substep in `post_step`): `0.15 ×` best bar approach
toward the bore axis (normalized by the episode's own spawn distance) `+ 0.25 ×` bar
seated (while settled) `+ 0.45 ×` best pin-through depth — gated on the bar being
seated AND the tip actually inside the eye hole, so pin credit is impossible before
the passage exists (the order latch). Base capped at 0.85; exactly 1.0 iff
`success()`. Doing nothing scores ~0.

## Teleport solution (`solve.py`) — transport only, two writes, both ending in free space

- **P1 — bar:** carried from its floor spawn to a hover 8 mm ABOVE the plate top, eye
  over the bore, yaw matched to the stand. Seating is CONTACT DYNAMICS: an 8 mm
  gravity drop onto the real plate (the release an arm performs at set-down).
- **P2 — pin:** carried to a vertical hover with its tip 10 mm ABOVE the seated eye —
  outside both holes. Fastening is CONTACT DYNAMICS: a floating-hand force controller
  (velocity-regulated descent at ~0.12 m/s, lateral PD toward the bore axis,
  upright-steadying torque, stall escalation) threads the shaft through the eye and
  the bore under real contact until the head lands on the eye and carries the pin;
  forces are cut and everything settles.
- The pin is never teleported into either hole; the bar is never teleported onto the
  plate. `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing
  (0.000 → 0.400 → 1.000 → 1.000), ≥3.3 simulated seconds hands-off persistence, then
  `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1, both SUCCESS,
  provably distinct layouts by stdout readback** (stand yaw +170.0° vs +61.9°, all
  spawns moved).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (−0.42, 0.00), facing the workspace: every judged point is within
~0.55 m reach. Per-object contact strategy: the bar is grasped at the raised
34 × 26 × 28 mm grip block mid-handle (top face 44 mm above the floor — clean
parallel-jaw target, 26 mm width < Franka's 80 mm stroke), carried up and laid on the
80 mm-high plate, slid until the eye lines up (10 mm tolerance ≫ arm repeatability);
the block sits mid-handle so the seated bar's COM stays over the plate and it cannot
tip while released. The pin is grasped around its 18 mm shaft (lying on the floor,
axis 13 mm up — head radius 17 mm keeps the shaft clear of the ground on one side),
reoriented vertical in a regrasp-free wrist rotation, centred over the eye and lowered
— 4 mm radial clearance per side at 26 mm hole width is generous for guarded descent;
the wide head both terminates the insertion mechanically and keeps the pin captive.
The 140 mm plate leaves ≥38 mm of free plate around the bore for the eye plate, and
the open front/back of the stand leave the approach corridor clear.

## Checks (`smoke.py` — rejection battery, 14 named checks, ALL PASS on the forge)

1. settle: states finite; bar flat on the floor, pin lying on its side (readback).
2. settle: score ~0 at reset, no success.
3. randomization readback: stand xy + yaw vary across 6 seeded resets.
4. randomization readback: bar xy + yaw, pin xy, d0 vary.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy / out-of-order: pin dropped into the BARE bore (head captive on the
   plate, verified in-bore by readback) → NOT linked, NOT success, score ≤ 0.05.
7. near-miss: bar ON the plate but 14 mm off the axis (tol 10 mm) → NOT seated,
   score ≤ 0.20.
8. pin lying HORIZONTALLY on top of the correctly seated bar → NOT linked,
   score ≤ 0.45.
9. wrong object: RED decoy offered to the aligned holes cannot enter → NOT linked,
   NOT success (identity control).
10. axis-line loophole: pin standing on the GROUND under the plate, axis through both
    holes → head-seated clause rejects it.
11. latched credit: removing the seated bar leaves the latched score unchanged,
    success gone.
12. monotonicity: closer bar placement latches strictly more approach credit.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` (163 × 600 × 960 × 3) recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest: pin
threads with real clearance, decoy cannot enter, head cannot pass (order enforcement +
captivity), seated tip hangs free of the ground, seat tolerance within the physically
linkable window.
