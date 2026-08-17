# robosuite_env_i408 — Polish-Probe QC Chute

**Env:** `simgen.qc_chute` (scene `qc_chute`, robot `"null"`)
**Seed:** `robosuite/robosuite_env` (RoboVerse `roboverse_pack/tasks/robosuite/robosuite_env.py`)

## What the task is

A quality-control rig (position + heading sampled per episode) carries three
staging PADS, a 12°-pitched CHUTE, and a sealed CATCH CUP past the chute's
exit lip. Three visually IDENTICAL gray discs (Ø40 × 22 mm, same authored
mass) sit on the pads. Exactly one disc is POLISHED (near-frictionless
surface, μ ≈ 0.04–0.05); the other two are ROUGH (μ ≈ 0.65–0.70). **Which pad
holds the polished disc is a hidden per-episode permutation** — no color, no
size, no mass difference; the label lives only in the contact physics.

Goal: get the polished disc — and ONLY the polished disc — to rest inside the
sealed catch cup. The chute's upper test section is OPEN (a gripper can set a
disc there and take it back); its lower half is a covered TUNNEL far too deep
and low (12 mm headroom) for any gripper, ending at an exit lip over a sill
into the cup, which is walled and roofed shut. The only way anything enters
the cup is ballistically, off the lip, under gravity.

The intended interaction is a physical EXPERIMENT: place a disc on the open
test section and let go. The ramp pitch (tan 12° = 0.213) is engineered
between the two friction coefficients — a rough disc holds still (3.3×
static margin); the polished disc accelerates away on its own (4.2× sliding
margin), shoots through the tunnel, flies off the lip and is captured. A disc
that held still is carried back to its pad and the next pad is probed.

## Seed provenance and why this is strategically different

The seed wraps the five robosuite oracle tasks (Lift, Stack, Door,
PickPlaceCan, NutAssemblySquare). In every one, the target object is KNOWN at
reset (the can, the square nut, the cube …) and the plan is a single known
transport: grasp the identified free object, carry it through free space to
the goal pose. This task inverts the epistemics and breaks the transport:

- **The target is not identifiable by looking.** The three discs render
  identically and weigh the same; the discriminating property (surface
  friction) is invisible. The core skill is INTERACTIVE PERCEPTION: design a
  contact experiment (release on a calibrated incline), read the outcome
  (slid vs held), and branch on it. No seed task requires learning anything
  at runtime — all seed goals are fully observable at t=0.
- **The final transport is physically impossible for the gripper.** The cup
  is sealed (walls + roof, verified by probes) and the tunnel has 12 mm of
  headroom over the disc; the only path in is the disc's own ballistic
  flight off the lip. The seed's one skill — carry the object to the goal
  pose — cannot finish this task; the agent must instead ARM the physics
  (release the right disc on the ramp) and let gravity execute delivery.
- **A wrong hypothesis is punished by the goal predicate itself.** Shoving a
  rough disc down the chute delivers it into the cup and FOULS it: success
  requires no rough disc inside (smoke check 6). Guessing (⅓ per pad) is a
  strictly losing strategy; probing is forced.

Also deliberately unlike the sibling corpus tasks from this seed and the
exemplars examined: i175 (same seed) is discrete MASS decomposition read out
by a balance's equilibrium angle — here masses are identical and there is no
mechanism readout, the hidden bit is a MATERIAL property revealed only by a
release experiment; i223 (same seed) is movable-obstacle routing where
nothing can be picked up — here everything on the pads is trivially
graspable and there are no obstructions; the challenge is knowing WHICH to
send, and the delivery leg is ballistic, not dragged. pen_holder / close_box
are free-object containment with fully visible goals.

## Mechanics

One KINEMATIC compound rig (level slabs: pedestal, sill, far wall, cup side
walls, cup roof, three pads; a pitched child Xform carrying deck, side
walls, tunnel roof and top cap — ~14 authored box slabs, all procedural)
plus three dynamic cylinder discs with authored mass and authored friction
materials using `friction_combine_mode="multiply"` against unit-friction rig
faces, so the contact pair coefficient IS the disc's own μ (readback-
asserted at reset). Margin ledger (import-time asserts in cfg): tan 12° =
0.213 vs μ_rough 0.70 (3.29× hold) and μ_polished_d 0.04 (4.25× slide);
tunnel headroom 12 mm over the 22 mm disc; slowest possible transit (release
at the tunnel mouth) exits the lip at 0.81 m/s and clears the entry sill by
18.8 mm (projectile math in the asserts); flight stays under the cup roof;
both computed rest poses (against the far wall, or slid back to the sill
foot) lie strictly inside the success box; a disc perched ON the sill is
outside it.

## Rubric (score 0..1, anchored in the solve trajectory)

- **0.30 · track** — latched: the polished disc has been on the chute deck
  band (correct height/width/along-slope span, in any episode pose).
- **0.30 · tunnel** — latched: the polished disc reached the covered tunnel
  section (deck band ∧ s past the roof start) — the point of no return.
- **1.0 iff success()** — the polished disc rests inside the cup box AND no
  rough disc is inside the cup AND all three discs are settled
  (< 0.10 m/s) AND every pose is finite. Non-success capped at 0.60;
  success is live (removing the disc from the cup drops the score back to
  the latched 0.60 — smoke check 8; a fouled cup denies success forever —
  smoke check 6).

## Solution outline (solve.py — teleports are TRANSPORT ONLY)

Every teleport does exactly what a gripper does: pick a disc off its pad,
set it down 4 mm above the OPEN upper test section (zero velocity, aligned
with the deck pitch), release; or carry a still disc back to its pad. All
load-bearing events are watched contact dynamics. Probe pads in order: set
the next pad's disc on the test section, hands off ~1.2 s; rough discs move
< 2 mm and are carried home; the polished disc slides away on its own
(measured ds = +431 mm during the probe window), enters the tunnel (track +
tunnel latch → 0.60), flies off the lip, and settles in the cup — the solve
just waits and verifies. Probe verdicts are cross-checked against the hidden
permutation readback (`scene._slot`) and `get_material_properties`, so a
wrong verdict dies loudly. Verified on forge seeds 0/1/2 (distinct
permutations and rig poses): `SIM_GEN_SOLVE: SUCCESS` on all three, ~18 s
wall clock each; final rest pose (rig-local +0.326, 0.000, 0.034) matches
the cfg-assert prediction exactly; score staircase non-decreasing to 1.000
with a 3.3 s hands-off persistence hold.

## Embodiment argument (single Franka, parallel jaw)

- **Base pose:** on the ground at the origin, facing the rig centre 0.40 m
  away (+x). Everything the arm ever touches — three pads at rig-local
  (∓0.15|0, −0.20) and the open test section (rig-local x ≈ 0.15–0.30,
  z ≤ 0.16) — lies in a 0.25–0.55 m radius annulus at z ≤ 0.16, well inside
  the ~0.85 m Franka envelope, under open sky (the rig's tallest static
  above the open region is the 34 mm chute wall).
- **Grasps:** each disc is a free-standing Ø40 mm cylinder ≪ the 80 mm jaw;
  top-down side-pinch on the barrel with 8+ mm of wall-free clearance on
  the pads and on the open test section.
- **The moves:** pick, place on the ramp, RELEASE, watch, re-pick if still —
  a set-down with zero velocity at ±centimetre accuracy suffices (the band
  is 64 mm wide and the verdict is binary). The delivery leg needs no reach
  at all: tunnel, lip, flight and cup are gravity's job; the cup is sealed
  precisely so no arm ever needs (or is able) to go there.
- **Perception:** the probe verdict is a gross, binary motion cue (a disc
  either stays within millimetres or vanishes down the chute) — readable
  from any camera; no force sensing needed.

## Execution order

No enforced sequence: pads may be probed in any order; a lucky first probe
of the polished disc finishes immediately. The only hard gate is physical —
nothing but a ballistic entry can put a disc in the sealed cup, and a fouled
cup (any rough disc inside) permanently denies success.

## Randomization (readback-verified)

Per episode: rig xy jitter ±40 mm, yaw ±25°; hidden pad permutation via
`torch.rand(...).argsort` (uniform over the 6 permutations; first-randint
degeneracy on this stack avoided); per-pad placement jitter ±20 mm.

## Checks (smoke.py — 10, frames.npz recorded)

1. **settle** — seeded reset settles finite; three discs on three distinct
   pads; authored materials (0.05/0.70) and mass took (physx readbacks);
   score ~0; no success.
2. **randomization** — 8 seeds: polished disc appears on ≥ 2 distinct pads,
   every draw a valid permutation, rig yaw / xy / pad-jitter spread.
3. **null** — 2.5 s of nothing: score ≤ 0.02, no success, discs stay put
   (< 5 mm drift).
4. **DECOY STALL** — a rough disc released on the test section exactly as
   the solve releases the polished one: moves < 5 mm in 2 s, stays on-deck
   (height readback), no credit. Paired with checks 6/8 (same release point
   sends other discs flying), so the stall is physics, not vacuity.
5. **TUNNEL ROOF** — a rough disc pulled straight up at 4× its weight from
   inside the tunnel: rises measurably, is stopped below the roof
   underside, drops back when released; no credit. The no-gripper-in-the-
   tunnel claim is physics, not fiat.
6. **FOULED CUP** — a rough disc servo-rammed down the chute enters the cup
   (traversal + in_cup readback); then the polished disc released on the
   test section slides in on top of the foul: latches reach 0.60 but
   success is DENIED — the contaminated cup can never satisfy the goal.
7. **near-miss gates** — the polished disc ON the cup roof (press-down at
   3× weight held: the roof holds), BESIDE the cup, and BEYOND the far
   wall: all refused by the success box.
8. **success + revocation** — polished disc released on the test section:
   full autonomous slide-flight-capture, score exactly 1.0, still true 1 s
   later; plucking it out revokes success LIVE and the score falls to the
   latched 0.60.
9. **wrong object** — a rough disc constructed at rest inside the (empty)
   cup: in_cup readback true, yet no success and no credit (the predicate
   keys on the polished disc).
10. **frames** — ≥ 20 video frames saved to `frames.npz`.

## Files

- `scene.py` — cfg + scene + registration (`qc_chute`, `simgen.qc_chute`)
- `solve.py` — `python -u -m simgen_tasks.robosuite_env_i408.solve --headless [--seed N]`
- `smoke.py` — `python -u -m simgen_tasks.robosuite_env_i408.smoke --headless`
