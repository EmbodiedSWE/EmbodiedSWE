# pick_single_egad_i3 — gate-pin + tunnel-slide shuttle puzzle (`tunnel_shuttle`)

## Seed provenance

Derived from **`maniskill/pick_single_egad`**
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/pick_single_egad.py`): a single
loose EGAD object rests in the open; the task is one grasp + one vertical lift, and the
checker is a pure z-position shift (`PositionShiftChecker(distance=0.075, axis="z")`).
One object, one motion, free space everywhere.

## What changed, and why it is strategically different

| | seed | this task |
|---|---|---|
| plan | grasp nearest loose object, lift 7.5 cm | extract a gate pin vertically, then a constrained horizontal slide under a cover, then lift + place into a bin |
| target at start | free, liftable | **unliftable** — trapped under a cover with ~8 mm headroom; only a knob pokes through a slot |
| obstruction | none | a seated gate pin blocks the slide path; a closed rear wall leaves one route |
| success predicate | z-shift of the object | containment: shuttle settled **inside** a bin (rim-perch rejected) |
| code structure | one checker on one body | ordered mechanism: latched gate-extraction credit + latched slide-progress along a randomized fixture axis + latched approach credit + containment gate |
| distractor | — | a loose RED cube — exactly the seed's kind of object — scores ~0 |

The seed's plan is *expressible* here and *rejected*: lifting the loose red decoy (or
binning it) earns nothing, and the smoke battery proves the shuttle itself cannot rise
out of the tunnel under a direct upward force. Nothing about the seed's single-pick
strategy transfers; the solver must discover an ordered unlock-slide-place plan.

## Scene summary

Fully procedural, self-contained geometry (compound single-body spawners):

- **fixture** (KINEMATIC): floor slab, two side walls, closed rear end wall, four
  ORANGE cover strips forming a 24 mm slot along the channel axis, interrupted by a
  20 mm cross-gap at `gate_x` where the gate pin drops through. The last ~60 mm of the
  channel (beyond `cover_end`) are uncovered — the open exit.
- **shuttle**: BLUE 40 mm cube + 16 mm square knob post rising through the slot
  (~32 mm proud of the cover). 8 mm headroom under the cover.
- **gate**: YELLOW blade 16 × 46 × 86 mm filling the channel cross-section, seated
  through the cover gap; a T-head (32 × 16 × 20 mm) stands above the cover.
- **bin** (KINEMATIC): GREEN open box, 120 mm interior, 30 mm walls, at a random
  bearing/distance off the channel exit.
- **decoy**: loose RED 40 mm cube in the open.

Contact offsets are explicit (1 mm) so the 2–8 mm working clearances are physically
real. Per-episode randomization (readback-verified in smoke): fixture xy jitter
(±3 cm) + yaw (±25°), shuttle start depth along the tunnel (slide length varies), bin
bearing (±45°) + free yaw, decoy pose.

## Rubric (anchored in the demonstrated solution)

`score()` — latched, never decreases:

- 0.15 × best gate-pin extraction fraction (gated near its slot),
- 0.35 × best slide progress `(x_l − start_x)/(exit_x − start_x)` (gated in-channel,
  normalized by the episode's own randomized start),
- 0.15 × best approach to the bin (gated once clear of the tunnel),
- capped at 0.65; **0.9** once the shuttle is in the bin; **1.0** iff `success()`.

`success()` = shuttle settled inside the bin, in the bin's body frame
(|xy| ≤ 45 mm, root height within (4 mm, 42 mm] — a rim-perched shuttle reads
~56 mm and is rejected).

## Solution phases (solve.py, teleport = TRANSPORT only)

- **P0** settle + layout readback (seed-dependent numbers printed).
- **P1 — gate extraction (contact)**: velocity-regulated upward force on the gate
  (grav-comp bang-bang, lateral centring, stall escalation) until the blade bottom
  clears the cover top; then one pose write **parks** the free pin off to the side
  (open-space transport only — the pin is already fully out of its slot).
- **P2 — tunnel slide (contact)**: fixture-frame forward force on the shuttle
  (velocity-regulated, y-centring, stall escalation) drags it along the slot, through
  the vacated gate station, until it emerges past the exit. The cover, slot, walls and
  floor constrain it the whole way — this is the load-bearing interaction.
- **P3 — transport + drop**: one pose write carries the now-free shuttle across open
  space to a hover 8 mm above the bin floor opening; gravity + contact seat it (no
  state is ever written into a scoring region — the hover pose satisfies no gate).
- **P4 — persistence**: ≥ 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS`
  only if `success()` still holds. `SIM_GEN_SCORE` printed at every phase boundary and
  asserted non-decreasing.

Verified on the forge on **seeds 0 and 1** (distinct layouts by readback), both
`SIM_GEN_SOLVE: SUCCESS`, scores 0.00 → 0.15 → 0.50 → 1.00 → 1.00.

## Execution-order declaration

Order is enforced by **geometry**, not rubric fiat:

1. gate before slide — the seated blade fills the channel cross-section (smoke check 8:
   a forward force stalls the shuttle at the gate, and the gate holds);
2. slide before lift — 8 mm headroom under the cover makes vertical extraction
   impossible anywhere in the tunnel (smoke check 7: 2.5 N up-force peaks below the
   wall top); the closed rear wall leaves exactly one path;
3. the rubric itself only reads physical outcomes (poses, containment, settledness).

## Embodiment argument (single Franka, parallel jaw, OSC)

Base pose: fixture side, ≈ (0.00, −0.42) in the fixture frame, gripper down-facing;
every waypoint below is within ~0.6 m of the base and above z = 0.02 m.

- **Gate pin**: jaw closes across the T-head's two end faces (32 mm span — inside the
  Franka's 80 mm stroke); the blade below is only 16 mm thin along that axis, so the
  fingers pass beside it. Straight vertical lift of ~60 mm frees the blade from the
  cover gap; place anywhere aside. Pure top-down grasp + z-motion — trivially within
  OSC.
- **Shuttle**: the 16 mm square knob protrudes ~32 mm above the cover — a canonical
  top-down jaw grasp. The arm then drags the knob horizontally along the slot (the
  channel guides the part; the arm only supplies tangential force, exactly what the
  solve's regulated force emulates), and once past the uncovered section lifts the
  shuttle straight up out of the open channel and places it in the bin (single
  free-space pick-and-place, ≤ 25 cm travel).
- **No other object needs contact**; the decoy is never touched. Required forces
  (≈1–5 N, ≈0.05 N·m equivalents) are far inside Franka payload limits.

## Smoke battery (smoke.py) — 15 checks

1. settle, no NaNs; 2. reset score ≤ 0.02; 3. randomization readback A (fixture
xy/yaw + start-depth spread over 6 seeds); 4. randomization readback B (bin bearing +
decoy spread); 5. null policy 240 steps → score ≤ 0.02; 6. **seed-strategy control**:
red decoy settled in the bin → score ≤ 0.05, not success; 7. **cover mechanism**:
2.5 N up-force on the trapped shuttle → peak below wall top, no success; 8. **gate
mechanism**: forward force with gate seated → shuttle stalls at the gate, gate holds;
9. wrong object: gate pin in the bin → score ≤ 0.05; 10. near-miss: shuttle on the
ground just outside the bin wall → not in_bin, score < 0.9; 11. wrong place: shuttle
resting on the cover top → not success, score < 0.9; 12. monotonicity: deeper slide
probe strictly out-scores shallower; 13. latched credit survives regression to start;
14. rejection audit (success never True in any wrong/partial state); 15. final no-NaN.
Frames recorded to `frames.npz`. Prints `SIM_GEN_SMOKE: ALL PASS 15/15`.

## Run (forge)

```
python -u -m simgen_tasks.pick_single_egad_i3.solve --headless [--seed N]
python -u -m simgen_tasks.pick_single_egad_i3.smoke --headless
```
