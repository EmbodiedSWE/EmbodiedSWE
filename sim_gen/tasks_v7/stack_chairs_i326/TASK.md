# stack_chairs_i326 — fold each folding chair flat, then slot it into the storage rack

## Provenance

Seed task: `rlbench/stack_chairs` — three chairs must be picked up and stacked
vertically on top of a target chair.

## What this task is

A steel storage rack (7 kinematic cuboids, re-posed every reset: floor plate
260 × 420 mm, two side walls, four upright fins forming **three identical
top-open slots, each 115 mm wide**, fins 142 mm tall) stands at a randomized
position and heading. In front of it, three identical **FOLDING chairs** (red /
green / blue) are scattered with randomized slot-order, jitter and free yaw.
Each chair is an ARTICULATED two-body mechanism: a base frame (foot plate +
tall backrest, 0.5 kg) and a 130 × 130 mm seat panel on a **real spawn-authored
revolute hinge** (axis Y, hard stops at 0° deployed / −100° folded). The seat
CoM sits ~5° past vertical at the folded stop, so the fold is **gravity-
bistable**: deployed (0°) and folded (100°) are both pressed-on rest states and
nothing between is stable — a genuine mechanical click.

Deployed, a chair is 197 mm deep × 140 mm wide — bigger than the 115 mm slot
gap in **every** horizontal orientation. Folded, it shrinks to 95 mm deep and
passes the gap with 20 mm to spare. Goal: for each chair, **fold the seat up
past its balance point** so it clicks flat against the backrest, then **lower
the folded chair down through a slot's top opening** until it stands upright on
the rack floor between the fins — one chair per slot, all three slots filled,
everything still. Judged in the rack's randomized frame: fold angle ≥ 80° (read
from the RELATIVE pose of the two chair bodies), origin inside the slot's
x/y band, in the on-rack-floor height band [38, 68] mm, upright ≤ 15°,
stillness sustained 20 substeps; each slot must hold **exactly one** such
chair.

The fold gate is **forced by physics, not rubric fiat** (asserted numerically
in `__post_init__` and probed in smoke): a deployed chair can never pass fully
below the fin tops in any yaw — at most its 95 mm foot drops in while a fin
**catches the protruding seat and props it well below the fold gate** (a
geometric cap) — and two folded chairs cannot share a slot (2 × 95 mm > gap;
both in-band would overlap).

## Why strategically different

- **From the seed (`rlbench/stack_chairs`)**: the seed is pile-making with
  rigid chairs — grasp, lift, place on top, verticality and pile stability are
  the whole game. Here piling counts for NOTHING (smoke check 6 builds a real
  settled chair-on-chair stack — as tall as these top-heavy folding chairs can
  physically stand free — and shows it scores ~0). The core subproblem does not
  exist in the seed: each object must first have an **internal DOF operated**
  (swing the seat through its gravity balance onto the far stop) because only
  the folded configuration passes the container aperture. "Transport harder"
  can never substitute for the fold. Code structure differs too: a hinge-angle
  gate from the relative pose of two bodies of one object + exactly-one slot
  occupancy, instead of any on-top-of / pile-height predicate.
- **From the sibling `stack_chairs_i110`**: that task threads rigid chairs
  HORIZONTALLY under a kneehole desk's overhang, with a width-to-bay matching
  subproblem (chairs differ, bays differ). Here the chairs and slots are all
  identical — there is no assignment problem at all; insertion is **VERTICAL**
  through a top opening, and the gate is the object's **own articulation
  state**. Failure modes are disjoint: seat not past the balance point,
  jolt-unfolding during transport/landing, seat propped on a fin top.
- **From other tasks in this corpus**: cup/bowl tasks nest rigid open
  containers; drawer/window tasks actuate a mounted prismatic/revolute body
  that stays part of the fixture. Here the articulated mechanism IS the
  manipulated free object — the hinge travels with the chair, and its state
  gates the container aperture.

## Solution outline (solve.py)

Teleport = transport only; both load-bearing interactions are contact dynamics
via `set_external_force_and_torque`. Per chair (red → green → blue; slots are
identical, chair i takes slot i — see Execution order):

1. **FOLD (dynamics, in place at the scatter pose)**: a hinge-axis torque servo
   on the seat body — gravity feed-forward mgd·cos(θ−5°) (mgd = 0.041 N·m)
   plus rate PD 0.015·(2.0 − ω), clamped [−0.03, 0.09] N·m — swings the seat
   through the ~95° balance onto the −100° stop, presses briefly, then drops
   the torque; the fold is verified **hands-off** by relative-pose readback (a
   gravity-stable state, not a held one). The wrench frame/sign convention is
   pod-dependent, so it is probed at runtime on the first chair (4 combos; a
   wrong combo presses into a hard stop and moves nothing).
2. **HOVER (teleport, transport only)**: the folded chair — both bodies, exact
   relative pose preserved, velocities zeroed — is placed in free air above its
   slot (foot 5 mm above the fin tops, chair depth axis across the gap).
3. **INSERT (dynamics)**: a vertical velocity servo (−0.22 m/s, descent gated
   on xy alignment < 10 mm) lowers the chair through the opening with an xy PD
   onto the slot center, a yaw PD, and an **uprighting attitude PD**
   (0.40·(ẑ_b × ẑ_w) − 0.020·ω — the lift force acts at the frame CoM while
   the combined CoM is ~14 mm higher, an inverted pendulum). The wrench is
   dropped 12 mm above the rack floor; the chair free-falls, lands and
   settles where contact left it. Wedge-stall abort + restage-behind-the-rack
   + refold recovery path (unused on seeds 0/1/2 — all inserts land first
   try).
4. `SIM_GEN_SCORE` at each phase boundary: 0.000 → 0.117 → 0.333 → 0.450 →
   0.667 → 0.783 → 1.000 (monotone; fold credit 0.35 latches, racked credit is
   physical).
5. **Persistence**: 400 substeps (3.3 s) fully hands-off; success and score
   re-asserted; then `SIM_GEN_SOLVE: SUCCESS`.

Passes forge seeds 0, 1, 2 (~20 s each).

## Execution order

Any chair order and any chair-to-slot assignment works — the chairs and slots
are identical and a racked chair never blocks another slot's opening. Declared
order: red, green, blue into slots 0, 1, 2.

## Embodiment argument (single Franka)

A single Franka with base at ≈ (0.85, 0), facing the rack from behind at
x ≈ 0.47 m, covers the workspace (chairs scatter at x ≈ 0.05, rack top opening
at z ≈ 0.15). Per interaction:

- **Fold**: the seat panel is a 130 × 130 × 12 mm plate at 60–190 mm height —
  the gripper closes across it (12 mm ≪ 80 mm span) near its free edge and
  swings it up in an arc about the hinge; the needed torque peaks at
  0.041 N·m ≈ 0.3 N at the 134 mm lever arm, trivially inside Franka wrist
  capability, and the bistable click means the arm releases after passing 95°
  and gravity finishes the seat onto the stop.
- **Transport + insert**: the folded chair weighs 0.56 kg; the backrest panel
  (14 mm thick) is a natural top-down grasp bar (grasped across its top edge,
  which stays proud of the fin tops until the last 30 mm of descent — at
  release height the backrest top is at 202 mm, 60 mm above the fins, so the
  gripper never enters the slot). The insertion forces used by the solve
  (≤ 1.5 N lateral, ≤ 11 N vertical incl. weight, ≤ 0.15 N·m attitude) are far
  inside Franka limits, and the 10 mm-per-side aperture clearance is generous
  vs. its sub-mm repeatability.

## Physically excluded wrong outcomes (verified in smoke)

- Piling (the seed strategy): a real settled chair-on-chair stack (a third
  free-standing level is statically impossible — the pile CoM walks past the
  50 mm foot edge) scores ~0 (check 6).
- Deployed insertion: a deployed chair released over a slot drops its foot in,
  but the fin catches the seat and props it at ~53° — geometrically capped
  well below the 80° fold gate — so even standing on the rack floor it never
  covers; pressed down at 3× its weight for 2 s the seat never passes the
  gate (fold sampled every substep) and the slot never covers (checks 7–8,
  non-vacuous: the wrench probe moves a chair, the drop is read back, and the
  same controller inserts a folded chair in check 11).
- In-slot with the seat not folded: the fin top props a deploying seat at
  ~45–60° < 80° — position/height/uprightness all fine, the fold gate alone
  rejects (check 9).
- Folded but perched ON the fin tops (bridging two fins): height band alone
  rejects (check 10).
- Two chairs per slot: a second folded chair pressed down onto the occupied
  slot at 1.5× its weight stalls riding the first chair's folded seat
  > 100 mm above the band (2 × 95 mm > gap); the slot counts exactly one
  (check 11).
- Still moving / abandoned: kicked chair not covered while moving (check 12);
  teleported-away chair drops the covered credit, keeping only the latched
  fold credit (check 13).

## Checks (smoke.py): 15

1. Settle + layout sanity (physical readback: plate/walls/fins at pose, fin
   row matches the bookkeeping yaw, chairs deployed upright on the floor).
2. Score ~0, no success at reset.
3. Rack randomization via physical fin readback — yaw spread > 8°, x spread
   > 15 mm over 8 seeds, every reset sane.
4. Chair scatter randomization — left-to-right chair order permutes (≥ 3
   distinct orders), chair yaw spread > 60°.
5. Null policy 240 steps ⇒ score ~0.
6. Seed-strategy rejection: real settled chair-on-chair stack ⇒ ~0.
7. Deployed drop: the foot falls into the slot but the fin props the seat well
   below the fold gate ⇒ not covered (non-vacuous: drop read back).
8. Deployed press at 3× weight: the seat never passes the fold gate (sampled
   every substep), the slot never covers.
9. Fold-gate near-miss: in-slot seat propped on the fin top ⇒ fold gate alone
   rejects.
10. Height band: folded chair standing on the fin tops ⇒ rejected by z alone.
11. Occupied slot: wrench-controller insertion covers a slot (the folded
    contrast for 8); a second folded chair pressed onto it stalls far above
    the band ⇒ exactly one counts, the first stays covered.
12. Settle gate: covered chair kicked to 0.25 m/s ⇒ not covered while moving,
    re-covers on settling.
13. Latched fold credit survives teleport-away; covered credit drops.
14. Audit: success() never True anywhere in the battery.
15. No NaN in any task-object state.

`SIM_GEN_SMOKE: ALL PASS 15/15`; frames.npz recorded.
