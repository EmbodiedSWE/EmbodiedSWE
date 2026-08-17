# sauce_carousel — crank the roofed carousel until the sauce can reaches the hatch

Env id: `simgen.sauce_carousel` · Scene: `sauce_carousel` · Robot: `null` (scripted
physics solution)

## Seed provenance

Derived from `libero/libero_pick_tomato_sauce` — *"pick up the tomato sauce and place
it in the basket"*. What the seed tests: identify the named can among grocery clutter,
prehensile pick, carry, and a bounding-box containment drop into an open basket.

What is kept from the seed:

- A target **tomato-sauce can** that must be identified against a same-shape
  distractor (RED sauce vs. WHITE decoy, on randomly swapped sides).
- A **containment goal**: the episode ends with the can resting inside an ordinary
  open basket, distractor excluded.

## What changed, and why it is strategically different

In the seed (and in every sibling read for this construction) the target is graspable
from the first frame — the task IS the grasp and the carry. Here the can is **captive
on a rotary stage**: it rides a free-spinning carousel platter inside a walled, roofed
ROTUNDA. A 16-segment fence ring and 16 radial roof slabs close 265–285° of the
circumference; the only opening is a **75° hatch sector** (95° of open roof) guarded
by a 15 cm sill, and both cans spawn **65–145° away from the hatch bearing**, deep
under a roof that clears their tops by 4 cm. Reaching in from above or from the side
is geometrically impossible (constructed and rejected in smoke: a 1.3 m/s radial kick
cannot leave the ring).

The only access path is to **operate the machine**: the platter's shaft rises through
a hole in the roof to a green **crank bar** spinning freely above the rotunda. The
solver must read the RED can's angular position, choose the shorter spin direction,
drive the carousel by the crank, and **stop it inside the hatch tolerance** — a
park-to-tolerance problem where overshoot puts the can back under the roof — then lift
the can over the sill through the open sector and drop it into the basket, leaving the
WHITE decoy still riding.

Strategy vs. the corpus neighbours read for this construction: `cellar_tow` (i43)
builds a temporary peg-in-eye tool coupling between free bodies and tows;
`rocker_lock` (i104) is a press-and-hold see-saw that gravity-ferries the bottle
through a gated letterbox; `flat_pack_crate` (i151) assembles the receptacle itself
with a lid-last order; `pen_holder` is multi-insert into an open cup. None involve
**angular planning on a rotary stage**: here the mechanism is a free revolute platter
driven from outside the sealed volume, the core skill is
choose-direction/spin/stop-without-overshoot, and the cans are never touched during
the transport phase — the platter's top friction alone carries them. The receptacle is
deliberately ordinary; the difficulty is upstream of the place, in *creating access*.

## Scene (procedural geometry only)

- **Rotunda** (45 kg dynamic fixture, teleported coherently at reset): plinth disc
  (56 cm), slick steel center hub (the thrust bearing), 16 fence chords (window ±37.5°
  about local +x), a 3-segment sill plugging the window below 15 cm, and 16 radial
  roof slabs (underside 26 cm, gap ±47.5°, center hole for the shaft).
- **Platter** (1.2 kg, free — no joints anywhere): grippy amber disc (42 cm), a
  **slick bearing pad** protruding 2 mm below the disc (so the thrust contact is
  slick-on-slick; PhysX pair-averages materials, and a grippy disc on the slick hub
  would re-create a ~0.2 N·m brake), an 8-segment slick skirt collar journaling the
  hub (3 mm play), a steel shaft through the roof hole, and the green crank bar
  (18 cm × 16 mm square) 31.5 cm up, above the roof.
- **Cans**: RED sauce (target) and WHITE decoy, 5.5 × 10.5 cm cylinders riding the
  platter 14 cm from center.
- **Basket** (0.7 kg, free): tan open box, 15 cm square interior, 8 cm walls, on the
  open floor.
- Per-seed randomization (verified by readback in smoke): rotunda xy jitter ±4 cm and
  **free yaw** (the hatch faces anywhere), platter free yaw, cans on random OPPOSITE
  sides at random 65–145° bearings (spin direction and amount both vary), free can
  yaws, basket on a random world ring 0.43–0.55 m from the rotunda.

## Teleport solution (solve.py) — teleport is transport only

- **P0** settle + layout readback; assert both cans riding ≥ 60° off the hatch, not
  aligned, score ≈ 0.
- **P1** spin: a pure world-z torque on the platter (the crank-hand proxy; a z-torque
  is yaw-invariant, so the external-wrench frame quirk cannot misdirect it) under a
  two-speed rate servo (0.5 rad/s far out, 0.15 rad/s inside 40°) drives the carousel
  the SHORT way; inside 10° it switches to an active brake (servo to zero rate). Up to
  4 park attempts with cap+gain escalation on stall (none needed on either seed). The
  cans ride by friction alone. Parked latch fires → score 0.40.
- **P2** transport: the parked can sits in the hatch sector under open sky (the roof
  gap clears its lift corridor by ≥ 3°, asserted numerically in the cfg with a
  41-point roof-slab corner sweep). ONE teleport of the can to 6 mm above the basket
  floor, zero velocity — exactly what a pick-over-the-15 cm-sill carry delivers.
  Extraction latch fires → score 0.60.
- **P3** gravity landing + settle → success, score 1.0; hands-off persistence 400
  steps (3.3 s) with success held, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds 0 and 1 (opposite spin distances: 139.8° and 84.9°; different
rotunda yaws +170°/+62° and basket rings) — score trace 0.00 → 0.40 → 0.60 → 1.00,
monotone, success persists ≥ 3.3 s hands-off.

## Rubric (anchored in the demonstrated solve)

Latched milestones (updated only in `score()`, non-decreasing, serialized in
`get_state`/`set_state`): 0.25 × spin progress (running max of the fractional angular
approach from the spawn bearing toward the hatch, **while riding the platter**) →
+0.15 parked within the 20° hatch tolerance → +0.20 extracted from the rotunda,
**gated on the parked latch** (an extraction that never parked scores nothing) → 1.0
iff `success()`. `success()` requires ALL of: the ordered latches fired
(parked-then-extracted), the RED can resting inside the basket volume (basket-frame
walls + height band — a lying can is accepted, seed-faithful containment), sauce /
basket / platter settled, the WHITE decoy still upright and riding the platter, all
states finite. Non-success states cap at 0.60.

## Embodiment argument (Franka, 80 mm parallel jaw)

- **Crank bar**: 16 mm square section — a natural full-closure pinch. It rides 31.5 cm
  above the ground, above the roof, reachable from above anywhere on its 18 cm span;
  the spin is a sequence of short horizontal arc strokes (grasp, sweep ~60°, release,
  regrasp — a valve-wheel motion), each ≤ 9 cm radius at a fixed height. The measured
  drive torque is ≤ 0.4 N·m → ≤ 4.5 N tangential at the bar tip, trivial for the arm.
- **Sauce can**: 5.5 cm diameter < 80 mm jaw span — a plain top-down cylinder wrap in
  the hatch sector, where the roof gap gives open sky (asserted ≥ 3° clear of the lift
  corridor) and the fence window passes the aligned can with ≥ 4° margin; lift 4 cm
  over the sill and carry at ≤ 26 cm height.
- **Basket drop**: an open 15 cm interior from above with > 4 cm xy clearance to the
  walls when centred — the solve's 6 mm free release is exactly a gripper release.
- **One base pose**: post the Franka ~0.6 m from the rotunda centre on the hatch
  bearing, offset toward the basket ring; crank (needs ≤ 0.3 m reach past the roof
  edge at 0.32 m height), hatch mouth, and basket (0.43–0.55 m ring) all fall inside
  a ~0.85 m reach disc. No step needs force closure beyond pinch, no bimanual
  coordination, no forced regrasp of the can.

## Execution order is physically forced

The cans spawn ≥ 65° from the hatch under a roof 4 cm above their tops, behind a fence
that meets the roof (both asserted in `__post_init__`); the platter-to-fence moat is
asserted narrower than a can. So the can cannot be reached, lifted, or thrown out
before the spin — smoke kicks it at 1.3 m/s and it stays caged. The rubric's ordered
latches (`parked` gates `extracted`) additionally reject fiat states that skip the
machine: a can written directly into the basket scores ~0 (smoke check 8). Success
further requires the decoy still riding, so "dump everything out" fails.

## Files

- `scene.py` — `SauceCarouselScene`, registered as `sauce_carousel`, env
  `simgen.sauce_carousel` (robot="null").
- `solve.py` — teleport solution,
  `python -m simgen_tasks.libero_pick_tomato_sauce_i235.solve --headless [--seed N]`.
- `smoke.py` — 20-check rejection battery (below), records `frames.npz`.
- `TASK.md` — this file.

## Checks (smoke.py, 20)

1 settle/no-NaN + layout (cans riding ≥ 60° off the hatch, score ~0) · 2–4
randomization real via readback (rotunda xy + free yaw; sauce bearing varies AND both
sides occur with the decoy always opposite; basket ring varies) · 5 null policy 300
steps ≈ 0 · 6 mechanism real (crank z-torque spins the platter ≥ 20° and BOTH cans
ride by friction, stopped outside tolerance, no success) · 7 rotunda cages a 1.3 m/s
radial kick (readback: still inside the ring) · 8 seed strategy / anti-teleport (can
written into the basket with no spin → ordered latches give ~0, no success) · 9 wrong
object (WHITE decoy in the basket) rejected · 10 acceptance (legal mini-solve: spin
parks at 0.40 — judged NOT success — then lift-out + drop → success, 1.0) · 11/12 can
removed to the open floor → success off and score = latched 0.60 cap / set_state
restore → success on · 13/14 decoy dumped off the platter → off / restore → on ·
15 near miss: standing against the basket's OUTER wall · 16 near miss: over the mouth
above the height band (z gate isolated, xy inside) · 17 settle gate (kicked in the
basket, judged moving → rejected; resettled → accepted) · 18 rejection audit (success
never True except 10/12/14/17b) · 19 final finite-state audit · 20 video: > 10 frames
to `frames.npz`.
