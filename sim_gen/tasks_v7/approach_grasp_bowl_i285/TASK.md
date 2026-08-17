# approach_grasp_bowl_i285 — Wedge Press Splitter (scene `wedge_press`)

Split a sealed pair of captive cargo sleds apart WITHOUT ever touching them: two boxes
(one crimson, one teal) rest nose-to-nose inside a low roofed tunnel whose roof is open
only over their seam. The only sanctioned actuator is an amber WEDGE KEY lying on a
spawn pad beside the housing — pick it up by its yellow stem (the task's one legitimate
grasp), stand it apex-down over the roof slot in line with the tunnel, and let GRAVITY
drive it into the seam: its 45° faces convert the vertical stroke into symmetric
horizontal thrust, plowing BOTH sleds outward AT ONCE, in OPPOSITE directions, onto the
green dock stripes, until the tip bottoms out on the slick floor strip (end caps arrest
whatever coasts). Moving the cargo by hand is forbidden in both ways a hand could try:
lifting a sled above `lift_z` spoils irreversibly, and so does any sled displacement
while the wedge is NOT presented in the seam window (an unattended-motion provenance
latch) — so finger-dragging the sleds fails exactly like carrying them. Finish with
both sleds past the dock lines, the wedge left SEATED in the seam, everything at rest.

## Provenance

- **Seed:** `pick_place/approach_grasp_bowl`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_bowl.py`) —
  approach ONE bowl lying in tabletop clutter, close the parallel jaw on its rim, lift
  it and carry it along marked waypoints: one grasp affordance plus free-space
  transport of a rigidly held object; the cargo moves exactly where the hand moves.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `wedge_press`, env
  `simgen.wedge_press`, robot `"null"`), `solve.py` (teleport solution), `smoke.py`
  (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's whole content is hold-and-carry — the judged object is
  rigidly attached to the hand and its trajectory IS the hand's trajectory. Here the
  judged objects may never be held OR displaced by the hand at all (dual spoil
  latches: lift AND unattended-motion provenance), there are TWO of them, and they
  must move in OPPOSITE directions SIMULTANEOUSLY — something no single rigid carry
  can do even in principle. The one grasp in the task is on the TOOL, and the hand's
  productive motion (vertical drop) is orthogonal to both cargo motions (horizontal,
  opposed): a force-DIRECTION conversion through a wedge mechanism, where the seed
  has a force-free transport. The load-bearing resource is the wedge's own weight
  falling through the stroke — the hand only positions and releases.
- **vs the sibling tasks read this session:** `approach_grasp_bowl_i133` (cliff sweep)
  herds free balls with a pushed cage — cargo still moves WITH the tool, one
  direction, no mechanism; `approach_grasp_spoon_i12` (die tip) is non-prehensile
  reorientation of one body; `approach_grasp_knife_i25` (underpin swap) is
  support-substitution statics; `approach_grasp_banana_i69` (tilt maze) tilts the
  world; `approach_grasp_ceramic_teapot_i109` (mug-rack hang) is prehensile placement
  onto a hook; `approach_grasp_i29` (ridge poise) is single-body balance. None drives
  a mechanism that splits a multi-body pack in opposite directions off one stroke.
- **vs the wider corpus:** `libero_pick_milk_i51` (pile driver) drops a slug onto a
  post — a momentum hammer, no direction conversion, one driven body; here the falling
  body IS the tool, the drive is quasi-geometric (bottom-out sets the spread
  deterministically), and the outputs are two opposed lateral deliveries.
  `native_liberoplus_i226` (balance verdict) and the ram/chute tasks share no element
  with a seam-splitting wedge. No corpus task centers a wedge.
- **Execution order (declared):** retrieve the wedge from its pad → align apex-down
  over the slot (yaw = tunnel heading) → release and let gravity drive the stroke →
  re-strike if it stalls short of the seat → leave the wedge installed and settle.
  There is no ordering loophole: delivery credit is GATED on engagement (hand-placing
  sleds at the docks with the wedge absent scores 0 — smoke check 9), the wedge must
  be seated NOW at the judged instant, and both spoil latches are checked every
  physics substep and are irreversible (smoke checks 6–8).

## Randomization (per episode, verified by readback in smoke)

Housing xy ±0.05 m and FREE yaw (the whole rig teleports; all judging is done in the
rig frame); spawn-pad side (rig +y / −y, coin flip via `torch.rand`, kinematic pad
teleport follows the latched side); seam gap between the sled noses ∈ [12, 18] mm;
seam centre offset ±6 mm; which colored sled starts on which end (coin flip); wedge
resting yaw on the pad (free). Cfg `__post_init__` asserts the honesty of the family
dead-man style: tip enters the minimum gap; the deterministic bottom-out spread clears
the dock line with margin for every seam offset; the dock line is unreachable by
tolerance-band drift; wedge thrust out-margins sled friction ≥ 5× (45° far from
self-locking); the lift gate brackets (above legit sliding, below through-slot
extraction); engagement latches strictly before any possible sled contact (no false
spoil on a legit stroke); the slot passes the widest wedge section; end caps arrest
coasting sleds but sit strictly beyond the bottom-out excursion; the stem is trivially
jaw-graspable.

## Rubric

`success()` iff, all judged live on physical poses: both sleds beyond the dock line
(rig-frame |x| > 76 mm, opposite signs by construction) and slow (|v| < 0.05 m/s); the
wedge seated NOW (tip-origin z < 22 mm inside the seam window) and settled; NOT
spoiled (no sled centre ever above `lift_z` = 90 mm; no sled |x| growth beyond the
latched engaged-reach + 8 mm while the wedge was not presented — both latched every
substep). `score()` = 0.10 × engaged (wedge ever presented in the window) + 0.30 per
delivered sled (past the line and slow, gated on engagement) + 0.15 × seated (ever
bottomed out), capped at 0.85; spoiled episodes capped at 0.20 regardless; exactly 1.0
iff `success()`. Doing nothing ~0.

## Teleport solution (`solve.py`) — transport only, every write ends in FREE SPACE

- **Transport:** the wedge (the tool — never a judged object) is teleported once to a
  hover pose apex-down over the roof slot: tip flat centred on the seam readback
  (midpoint of the sled inner noses; tip 4 mm in a ≥ 12 mm gap — ±4 mm tolerance by
  construction), yaw aligned with the housing, tip bottom at 95 mm — above the roof
  plane, overlapping nothing — velocities zeroed. The judged sleds are never teleported.
- **Strike:** hands-off. Gravity drops the wedge ~4 cm, the tip enters the seam, the
  45° faces plow both sleds outward through real contact, the end caps arrest the
  coasting sleds, and the tip bottoms out on the slick strip (origin z ≈ 12 mm <
  seat gate 22 mm). A re-strike loop (lift to a 150 mm free-space hover over the
  fresh seam readback, re-drop, ≤ 5 attempts) covers stalls — unused on both seeds.
- **Persist:** ≥ 3.4 simulated seconds hands-off with `success()` held.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing (all credit is
  latched), then `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1,
  both SUCCESS**, provably distinct by stdout readback — seed 0: rig (−0.010, +0.047)
  yaw +0.12, pad side −1, gap 14.7 mm, trace 0.00 → 1.00 on the first strike (sleds
  ±0.110, wedge seated z = 0.012) → 1.00 (settle) → 1.00 (persist); seed 1: rig
  (+0.039, +0.017) yaw +1.55, gap 13.0 mm, mirrored color ends (sleds_x
  [−0.047, +0.046]), same single-strike trace. Smoke battery:
  `SIM_GEN_SMOKE: ALL PASS 13/13`, frames.npz (195, 600, 960, 3).

## Embodiment sanity (single-arm Franka feasibility)

One base pose serves the whole episode family: base ~0.55 m from the housing on the
spawn-pad side — every contact point (pad at 0.28 m from the rig centre, slot over the
rig centre) lies inside a Franka's ~0.85 m envelope. Per-object contact strategy: the
WEDGE is grasped by its yellow 28 mm square stem (well inside the 80 mm jaw stroke,
asserted), an ordinary pick of a 1.2 kg tool; reorienting it from lying-on-side to
apex-down is a wrist rotation (or one regrasp) in free space; the "release over the
slot" is exactly what the solve's zero-velocity hover encodes — open the jaw at ~95 mm
tip height and retract. All manipulation happens between z ≈ 0.03 and 0.30 m over a
~0.35 m span. The SLEDS are never touched by the gripper at all — the falling wedge is
the only thing that ever contacts them, which is precisely what both spoil latches
demand. No contact is required inside the tunnel or below the roof; the housing
interior never enters the reachable-workspace argument.

## Checks (`smoke.py` — rejection battery, 13 named checks)

Because ANY clean wedge drop into a live unspoiled seam genuinely solves the task,
every smoke construct that ends with delivered sleds is deliberately SPOILED FIRST —
the battery can never reach `success()`, and check 12 audits exactly that.

1. settle: states finite; sleds at box rest height on the strip, wedge at pad height
   (readback).
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: rig xy and FREE yaw vary across 8 seeded resets; the spawn
   pad flips to both rig sides AND its kinematic teleport follows the latched side
   (sign-correlated readback).
4. randomization readback: seam gap and seam centre vary within their bands; the
   color-end assignment flips both ways.
5. null policy: 300 idle steps → score ~0, no success, no spoil, sled drift < 5 mm.
6. SEED strategy: both sleds raised above `lift_z` and carried to the docks, wedge
   then genuinely dropped and seated — final tableau visually complete yet spoiled →
   NOT success, score ≤ 0.20. THE key check: the seed's verb produces the goal
   picture and still fails.
7. finger-drag: a sled slid 30 mm outward at floor height with the wedge on its pad →
   the unattended-motion provenance latch spoils it (the seed's floor-level variant
   fails too). KEY check 2.
8. spoiled-then-perfect: continuing ep. 7, a flawless strike physically delivers both
   sleds and seats the wedge — still ≤ 0.20, NOT success (spoil is irreversible).
9. tableau without the mechanism: sleds hand-placed at the docks, wedge never
   presented → delivered latch stays 0 (gated on engagement), score ~0.
10. engaged-only: wedge presented in the window 2 substeps (no sled contact possible,
    tip-z readback proves it) then withdrawn → exactly the 0.10 credit, latch
    persists after withdrawal, no spoil, NOT success.
11. wedge parked on the roof slab → never presented → no credit, no spoil.
12. rejection audit: success() never True at any judged point in the battery.
13. final no-NaN. Plus `frames.npz` recorded and saved in CWD.
