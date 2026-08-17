# peg_insertion_side_i251 — Quarter-Turn Keyed Dial (scene `quarter_latch_drum`)

Operate a recessed rotary mechanism THROUGH a tool: pass the blue key's flat blade
through an octagonal porthole into the keyway slot of an orange dial riding free in a
bearing 50 mm behind the faceplate, TWIST a quarter turn until the dial latches on its
far stop (it is bistable — an internal counterweight snaps it back below ~45° and
completes it past ~45°), then WITHDRAW the key fully and set it down clear. A red key
whose 56 mm blade exceeds the porthole's maximal chord is a decoy.

## Provenance

- **Seed:** `maniskill/peg_insertion_side`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/peg_insertion_side.py`) — grasp a
  peg lying on the table and insert it sideways into a pre-existing hole in ONE fixed
  block; the checker is pure relative-bbox containment of the peg in the block's frame.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene
  `quarter_latch_drum`, env `simgen.quarter_latch_drum`, robot `"null"`), `solve.py`
  (teleport solution), `smoke.py` (rejection battery), all procedural geometry — no
  external assets, no authored joints (the dial is a free rigid body captured in a
  kinematic pocket, so reset can write the whole "linkage" consistently).

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed is a single TERMINAL insertion — the episode ends the
  instant the peg's pose is deep enough, and the peg itself is the judged body. Here
  insertion is INSTRUMENTAL and judged NOWHERE on its own: the judged outcome is the
  ANGULAR STATE of a second body the hand never touches (the recessed dial), plus the
  key being REMOVED again. The seed's whole strategy — "get the peg deep into the
  hole, done" — is exactly this task's incomplete state: smoke check 6 constructs the
  deepest possible insertion and asserts it tops out at the engage credit (≤ 0.45),
  and a key left inserted even with the dial latched caps at 0.95 (smoke check 8).
  The plan is insert → rotate (transmit torque through the tool) → RETRACT, i.e. the
  inverse of the seed's single translate-in.
- **vs `peg_insertion_side_i1` (read this session):** i1 is tool-use by RAMMING — a
  rod pushed straight through a tube to EJECT a judged third body into a basin; the
  motion is one translation and the tool ends wherever it likes. Here the tool
  transmits TORQUE about its own long axis (a wrench, not a ram), the mechanism is
  bistable with an over-center point, and the tool's final state is itself judged
  (must be fully withdrawn and clear).
- **vs `peg_insertion_side_i2` (read this session):** i2 fastens two free bodies into
  a static linkage (bar seated, headed pin threaded through two stacked holes) — both
  manipulated bodies REMAIN in the mechanism and are the judged outcome. Here the
  judged mechanism body is never touched directly, the manipulated tool must LEAVE
  the mechanism, and the sub-goals are insertion, rotation, and extraction rather
  than align-then-thread.
- **vs `robobench/suites/packing/scenes/pen_holder.py` (read this session):** that is
  repeated container insertion (pens tip-up into a holder). Nothing here is placed in
  a container; the porthole is a passage, and what matters is what the blade DOES
  while inside (quarter-turn torque transmission) and that it comes back out.
- **Execution order is REQUIRED and physically enforced** (declared): the dial can
  only be turned by something reaching through the porthole (it is recessed 50 mm
  behind a 46 mm opening), the blade can only engage the slot after passing the
  porthole at a compatible roll, and the quarter turn must be completed past the
  45° over-center BEFORE withdrawal — a dial released short of halfway snaps back to
  the start stop (smoke check 7), so withdraw-early fails by mechanism, not by fiat.

## Randomization (per episode, verified by readback in smoke)

Housing xy jitter + FREE yaw (the bore axis direction moves — the solver must read it
from the scene), key xy + free yaw lying flat, decoy xy + free yaw, with batched
keep-out resampling so nothing spawns intersecting; the dial is teleported WITH the
housing (whole-linkage write) and settles onto its 0-stop. The approach baseline `d0`
is captured per episode.

## Rubric

`success()` iff, settled (key |v| < 0.05 m/s; dial gates sit above the measured
kinematic-pocket phantom-velocity band): dial angle θ ∈ [80°, 110°] about the bore
axis (hard stop at ~90°, readback ~95°), AND the BLUE key fully clear of the porthole
keep-out zone (tip, midpoint and butt all outside {housing-local x < 45 mm AND radial
< 30 mm}). The decoy is judged nowhere.

`score()` (latched every physics substep in `post_step`): `0.10 ×` best key-tip
approach toward the collar mouth (normalized by the episode's own spawn distance)
`+ 0.30 ×` best insertion depth — gated on the tip actually INSIDE the bore, so waving
the key outside earns nothing — `+ 0.45 ×` best dial angle above a 4° deadband
`+ 0.10 ×` dial latched at the stop (at-stop AND settled). Base capped at 0.95;
exactly 1.0 iff `success()`. Doing nothing ~0; the seed strategy tops out ~0.40;
everything-but-withdrawal tops out at 0.95.

## Teleport solution (`solve.py`) — transport only, two writes, both ending in free space

- **P1 — key transport:** carried from its floor spawn to a hover with the blade tip
  15 mm OUTSIDE the collar mouth, axis on the bore line, blade rolled vertical to
  match the slot — free space.
- **P2 — contact insertion:** a floating-hand force controller (axial velocity servo
  at ~6 cm/s, TIP-based lateral PD onto the bore line, blade-alignment torque, stall
  bias escalation) pushes the blade through the porthole into the slot to 12 mm depth
  under real contact (readback: axis_dot +1.000). The key is held at all times inside
  the mechanism — an unsupported key pitches and the slick slot cams it back out.
- **P3 — quarter-turn twist:** torque about the bore axis ramps 0.02→0.15 N·m with
  two safety gates (no twist unless the blade is engaged past the dial face; no twist
  while the key's roll rate exceeds 2.5 rad/s — twisting a disengaged key's tiny roll
  inertia causes runaway spin) until θ ≥ 84°, then 180 held steps while the
  counterweight completes the over-center latch (θ → ~95°).
- **P4 — withdrawal under contact:** stage A grinds the blade out of the slot
  (stick-slip against the slick cheek; velocity-triggered bias reset prevents a held
  bias from yanking the freed blade); then a retreat-square-retry loop clears
  drawer-jams (a yawed blade locks diagonally across the aperture — both edges on
  opposite jambs): square up under the alignment PD, re-pull, escalating the bias cap
  0.4 → 0.9 → 1.5 N across attempts, backing in between tries. The key is NEVER
  teleported while any part of it is inside the mechanism — a hard fail-fast aborts
  before P5 if the tip has not cleared the porthole.
- **P5 — park:** the free key (in open air outside the porthole) is carried to a
  clear floor spot ≥ 16 cm from the decoy, set down with a 0.5 mm gap (no set-down
  jolt), and success is streak-gated (30 consecutive true steps).
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing
  (0.000 → 0.096 → 0.400 → 0.950 → 1.000 → 1.000), ≥3.3 simulated seconds hands-off
  persistence, then `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge on the
  identical package: seeds 0 and 1, both SUCCESS, provably distinct layouts by
  stdout readback** (housing (−0.006, +0.121) yaw +170.0° d0 0.355 vs
  (+0.023, +0.092) yaw +61.9° d0 0.399; key/decoy spawns all moved).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly 0.45 m in front of the porthole, facing the faceplate: the porthole
sits 160 mm up, every judged point within ~0.55 m reach. Per-object contact strategy:
the key is grasped around its 16 mm shaft (lying flat, axis 8 mm up — within Franka's
80 mm stroke), lifted, pitched horizontal with the blade rolled vertical, and guided
in — 3 mm radial clearance at the 46 mm porthole and 2 mm/side in the 10 mm slot are
generous for guarded insertion, and the collar itself funnels the blade. The quarter
turn is a pure wrist roll (Franka joint 7 spans ±166°, > 90° + margin) with the shaft
still in the jaws; the dial takes ~0.07 N·m — trivial for the wrist. Withdrawal is
the reverse guarded pull, and set-down is a floor place anywhere clear. The dial is
NEVER touched by the gripper — it is recessed 50 mm behind a 46 mm opening that no
parallel-jaw finger pair can usefully enter, which is exactly why the key is
load-bearing. The decoy never needs to be moved.

## Checks (`smoke.py` — rejection battery, 13 named checks, ALL PASS on the forge)

1. settle: states finite; key and decoy flat on the floor, dial resting on its
   0-stop at θ ≈ −4° (readback).
2. settle: score ~0 at reset, no success.
3. randomization readback: housing xy + yaw vary across 6 seeded resets.
4. randomization readback: key xy + yaw, decoy xy, d0 vary.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: key seated 12 mm deep in the slot (verified by readback + the
   gated engage latch), never twisted → dial still at the 0-stop, NOT success,
   score ≤ 0.45.
7. under-rotation: dial released at ~40° (below over-center, readback) snaps BACK
   to the 0-stop → NOT at_stop, NOT success (bistability, refusal side).
8. key left inserted: dial released at ~60° completes OVER-CENTER to the far stop
   (mechanism honesty, readback ~95°) while the key hangs in the porthole →
   at_stop TRUE yet NOT success, score ≤ 0.95 (withdrawal is required).
9. wrong object: the RED decoy pressed at the porthole by a real force servo in the
   blue key's exact approach pose reaches the collar mouth (readback: the push was
   real) and never passes it → no credit, NOT success (identity control).
10. latched credit: removing the deeply inserted key to the floor leaves the latched
    approach + engage score unchanged, success still absent.
11. monotonicity: closer key placement latches strictly more approach credit.
12. rejection audit: success() never True at any judged point in the battery.
13. final no-NaN. Plus `frames.npz` (155 × 600 × 960 × 3) recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest: the
blade passes the porthole AND can rotate a full turn inside it, the decoy blade
exceeds the octagon's maximal chord (cannot enter at any roll), the blade fits the
slot with real clearance, the stop pair spans a true quarter turn with the stop
inside the success window, the counterweight's toggle torque dominates the
slick-bearing friction scale, and full engagement keeps the tip off the slot bottom.
