# setup_chess_i302 — Capture the King

Push the **black king** off the fenced board through the one rim **opening** so gravity
carries it down the **capture chute** into the **capture box** on the floor, then stand the
**white king** upright on the vacated **gold dais** (the throne square). Env id:
`simgen.capture_arena` (scene-level, `robot="null"`).

## Provenance

- **Seed task:** `rlbench/setup_chess` — "set up the chess board": 32 chess pieces are
  picked off the table one by one and placed on their marked board squares
  (`RoboVerse/roboverse_pack/tasks/rlbench/setup_chess.py`; USD board + free-standing
  pieces, pure pick-and-place, no ordering, no checker in the ported config).
- **Kept from the seed:** the chess dressing (two kings, a checkerboard playing surface,
  a marked goal square) and the placement-tolerance flavor of the final predicate.
- **Replaced:** the plan itself. The seed only ever ADDS pieces to free squares; here the
  goal square starts OCCUPIED, and the first half of the task is a REMOVAL — a capture.

## Why it is strategically different

| axis | seed `setup_chess` | this task |
|---|---|---|
| direction | 32 pieces are put ONTO the board | the critical move takes a piece OFF the board |
| removal mechanic | none | **ejection through a randomized aperture**: three rim sides are walled, one has a 16 cm opening feeding a gravity chute; capture = real containment in a floor-standing box (sill 3 cm below its wall rims) |
| ordering | none — 32 independent moves | **occupancy-enforced strict order**: the throne square starts held by the black king; two Ø44 mm bases cannot both centre within the 25 mm tolerance, so the white king physically cannot be seated until the capture is done (smoke-proved: the solve's own release executed early topples off the occupier) |
| target state | squares are all free | the ONE goal square starts blocked |
| failure modes | drop/misplace | wrong exit side (rim/blocker retain), captured king left on the floor beside the box, coronation attempted before the capture, wrong piece in the box |

Also differs from the sibling read while working, `setup_chess_i121` (captive-slide
channel routing: pieces trapped under lip rails, push-to-hard-stop, beacon decoy) and
from the `pen_holder` exemplar (insertion): nothing here is captive, slid to a stop, or
inserted — the mechanics are ejection-through-an-aperture, a gravity delivery, and an
occupancy-gated placement.

## Scene (fully procedural)

- **Platform** (kinematic compound): 0.46×0.46×0.12 m slab, top painted as a visual-only
  4×4 checkerboard (no tile colliders — no proud edge can park a slide), fenced by a
  35 mm rim wall; BOTH ±y rims have a central 16 cm gap.
- **Blocker** (kinematic): wall segment plugging the inactive gap each episode, so exactly
  one opening exists.
- **Chute** (kinematic compound) at the active gap: slick 24° ramp (entry recessed 4 mm
  below board top; tan 24° = 0.445 ≫ pair-averaged μ ≈ 0.21, nothing can rest on it),
  guard walls, and an open-top capture box on the floor (grippy 0.65 μ floor, interior
  23×23 cm, walls to 8.9 cm; near wall stays 7 mm under the ramp exit).
- **Dais** (kinematic): 0.10×0.10×0.006 m gold square, position sampled on the board.
- **Black king / white king** (dynamic compounds): flanged base Ø44×14 mm, shaft
  Ø24×75 mm, ball crown; m = 0.15 kg, authored base-weighted CoM (z = 12 mm) + diagonal
  inertia. Black is near-black with a crimson crown; white ivory with a gold crown.
- Friction materials bound per-collider; restitution 0; solver iters 16/4; dt = 1/120.

**Randomization (readback-verified):** platform yaw ±20° + xy jitter ±4 cm; active gap
side ±y (chute AND blocker follow); dais position x ∈ [−0.06, 0.06], y ∈ [−0.05, 0.05];
white start x ∈ [−0.13, 0.13] on the walled side opposite the opening.

## Rubric (latched partial credit; success judged live)

- **0.25** `l1` — black king ejected: past the gap line and descending in the chute frame.
- **0.30** `l2` — black king contained in the capture box (root below the sill, 3 cm
  under the wall rims — a piece on the ramp, on a rim, or on the floor outside never passes).
- **0.20** `l3` — with `l2` set, white king upright within 10 cm of the dais.
- **1.0 iff `success()`** (else capped at 0.75): black king inside the box AND white king
  standing upright (≤12°) with its base centred on the dais (±2.5 cm, resting in the
  seated z-window), both settled (<0.04 m/s), states finite.

Tolerances are honest by construction: a base centred within 2.5 cm of the dais centre is
fully supported (edge ≤ 4.7 cm < 5 cm half-width), while double occupancy is impossible
(two Ø44 mm bases cannot both centre within 2.5 cm).

## Solution outline (`solve.py`, demonstrated on forge)

The capture — the load-bearing interaction — is pure contact dynamics: a horizontal
velocity-servoed CoM push (F = m·K·(v_des − v) + friction feedforward, K·dt = 0.25, cap
~1.5× piece weight, v ≤ 0.12 m/s, world forces pre-encoded against external-force frame
drag) drives the black king off the dais, across the board and through the opening; the
chute and gravity do the rest — no teleport ever touches the black king.

P0 settle + layout/mass readback (score 0) → P1 push the black king to the gap, then
through it until the ejection latch fires (`l1`, 0.25) → P2 hands-off gravity delivery
into the box (`l2`, 0.55) → P3 the one transport teleport: park the white king 25 mm
ABOVE the vacant dais, release; the seating is a real contact settle (`l3` en route) →
success, 1.0 → P4 ≥3.4 s hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (Franka, one base pose)

Base at ≈ (−0.55, 0, 0) board-frame (the always-walled −x side), facing +x: every board
cell (|x|,|y| ≤ 0.22, z = 0.12) and the release point over the dais lie within a
0.33–0.80 m reach disc at a comfortable working height; the arm never needs to reach the
capture box — gravity delivers the captured king. Per-object contact strategy: both kings
expose a Ø24 mm shaft standing 18–89 mm above the 35 mm rim, so a two-finger gripper
fingertip-pushes the shaft low (the base-weighted CoM keeps a shaft push from tipping —
the solve's ≤2.2–5 N horizontal push is exactly that wrench) to eject the black king, then
side-grasps the white king's shaft, lifts over the rim and lowers it above the dais —
the solve's transport-teleport-and-release. The platform, chute, blocker and dais are
never manipulated. All forces are far inside Franka payload.

## Declared execution order

1. Capture the black king (push through the opening → chute → settled inside the box).
2. Enthrone the white king on the gold dais.

The order is physical, not narrated: the dais seats exactly one piece, and smoke check 9
executes the solve's own coronation release BEFORE the capture — the white king can only
land on the occupier's crown and topple off (no enthronement, score 0).

## Validation evidence (all on the forge, Isaac Lab / RTX 4090)

- `solve` seeds 0 (−y gap side) and 1 (+y gap side): `SIM_GEN_SOLVE: SUCCESS`, scores
  monotone 0 → 0.25 → 0.55 → 1.0, ~18 s each; both gap sides exercised.
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 11/11`, frames.npz (137, 600, 960, 3) saved:
  1. settle/no-NaN (throne occupied, both upright, score 0);
  2. randomization readback, 3 seeds max-pairwise (Δyaw 11.0°, Δplatform-xy 56 mm,
     Δdais 104 mm, Δwhite-x 69 mm; dais body matches the throne target to 0 mm);
  3. gap-side coverage over 10 resets + chute/blocker follow the draw every time;
  4. null policy ~0;
  5. **rim retention** with the solve's own live actuator: a quasi-static push at the
     BLOCKED side travels 234 mm and is retained on the board, no ejection latch;
  6. **seed strategy rejected**: white placed perfectly on a plain square, black
     untouched → score 0;
  7. near-miss capture: black on the floor BESIDE the box, white enthroned → captured
     False, success False, score 0.25;
  8. near-miss throne: black captured, white 8 cm off the dais → all latches fire,
     score exactly 0.75, success False;
  9. **order is occupancy**: the P3 release executed before the capture topples off the
     black king → no enthronement, score 0;
  10. swapped identities (WHITE king in the box, black on the throne) → score 0;
  11. video saved.
