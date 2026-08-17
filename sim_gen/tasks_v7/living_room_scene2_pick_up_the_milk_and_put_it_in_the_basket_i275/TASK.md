# living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i275 — Drop Chute (scene `drop_chute`)

Get the WHITE milk carton inside the container — but the container is a SEALED drop-bin:
a full roof denies the seed's whole delivery (nothing can be dropped in from above), and
the only entrance is a low letter-slot in the front wall guarded by a BLUE gravity-closed
swing flap that opens INWARD only. A carton standing up is taller than the slot; it fits
only lying down. The plan is a horizontal push-through insertion: lay the carton on its
side on the YELLOW loading shelf (top flush with the slot sill), push it horizontally
against the self-closing flap until the door yields and the carton tips over the inner
sill edge and drops to the bin floor, then let the flap swing shut behind it on its own.
The RED juice decoy must stay out of and off the bin.

**Scene** `drop_chute` / env `simgen.drop_chute` (NullRobot, scene-level physics).
**Seed** `libero_90/living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket`.

## Provenance and strategic difference

The seed is a **grasp-carry-release-from-above** of the milk into an open-top basket:
approach vertically, close the jaw, carry OVER the container, open the jaw, gravity does
the last 10 cm downward. Every load-bearing element of that plan is denied or reversed:

- **The seed's means is physically removed.** The bin has a FULL roof: smoke check 6
  drops the carton from directly above the bin mouth and it lands ON the roof (readback
  z = 350 mm ≥ 300 mm), earns nothing, and never enters. There is no top-drop.
- **The insertion is horizontal, against a self-closing door.** The only way in is the
  120 × 200 mm letter slot at sill height 120 mm, sealed by a passive one-way swing flap
  (real `UsdPhysics.RevoluteJoint`, limits −1°…80°, authored below-hinge CoM so gravity
  returns it shut — smoke 9 opens it to 80.6° with 0.08 N·m and reads −1.01° after
  release; smoke 10 shows the same torque OUTWARD leaves it at the −1° stop). The
  load-bearing interaction is a sustained horizontal contact push (≈ 0.9–1.8 N servo
  force) that makes the cargo itself shoulder the door open — the seed has no door, no
  push, no horizontal approach, and its release is a jaw opening, not a tip-over-the-sill.
- **Orientation is a hard constraint, not a nuance.** A standing carton (160 mm) cannot
  pass the 120 mm slot; a lying one (60 mm) passes with ≥ 40 mm margin (both asserted).
  The seed judges bounding-box containment with no orientation gating on the way in.
- **The container is also a one-way trap.** The flap cannot swing outward, so delivery
  is irreversible (smoke 13 shoves the delivered carton outward with 4 N: it moves
  126 mm, the sealed front wall keeps it at peak x = 96 mm, success survives).

"Seed end state rejected" is N/A by design: the seed's end state IS this task's goal;
what is strategically different is that the seed's MEANS is unavailable (probe 6) and
the required plan — stage lying on a shelf, horizontal push-through past a one-way flap,
hands-off gravity close — shares no step with approach-grasp-carry-drop. Also distinct
from the packages read while building this: sibling `i177 dump_hopper` cages the ITEM
(untouchable milk), transports the CONTAINER, and requires a HELD lever through a
gravity discharge — here the milk is directly manipulated throughout, the container
never moves, and no mechanism is ever held (the flap is passive and yields to the cargo
itself); the `pen_holder` exemplar is many-object tip-up insertion into a carriable
open cup with no mechanism at all.

## Mechanism numbers (bin frame; bin yaw randomized ±180°)

- Bin: 360 × 360 mm outer, 320 mm tall, 10 mm shell, one 30 kg DYNAMIC compound (never
  kinematic, so the flap hinge anchor follows reset teleports). Front wall = sill panel
  up to 120 mm + pillars + lintel from 240 mm, leaving the 120 mm-tall × 200 mm-wide
  slot. Interior floor top z = 10 mm, grippy (μ 0.45/0.40).
- Shelf: YELLOW slick slab (μ 0.15/0.12), x ∈ [0.180, 0.380], half-width 120 mm, top
  exactly at the 0.120 m sill, with a support leg. Long/wide enough for a yawed lying
  carton (asserted).
- Flap: BLUE 190 × 127 × 6 mm plate, 60 g, hinge at (0.160, ·, 0.238) just under the
  lintel, axis body-y; limits [−1°, +80°]; authored CoM (0, 0, −FLAP_LEN/2) + diagonal
  inertia; angular damping 1.5. Closed bottom edge z = 0.111 seals below the sill
  (asserted); at +80° it rides OVER a lying carton on the sill (asserted); gravity-close
  bias torque in [0.010, 0.200] N·m (asserted) — cargo pushes it open at fingertip force
  (computed push ≤ 5 N, asserted).
- Cartons: milk WHITE, juice RED, both 60 × 60 × 160 mm boxes (0.35 / 0.30 kg), spawned
  STANDING on mirrored jittered arcs (bearing 35–75° off bin +x, radius 0.50–0.62 m,
  free yaw) — beyond the shelf tip plus carton diagonal, and well clear of the bin sides
  (both asserted), so the null policy scores 0.
- Randomization: bin xy ± 50 mm and yaw ± 180°; both cartons' side/bearing/radius/yaw.
  Smoke 3–4 read all of these back across 8 seeded resets (Δbin = 69/77 mm, Δyaw
  3.36 rad; Δmilk 1.07 m, Δjuice 1.14 m).

## Rubric

`success()` = milk **inside below the aperture** (bin-local |x| ≤ 160 mm, |y| ≤ 145 mm,
z ∈ [13, 95] mm — the z ceiling sits ≥ 20 mm BELOW the sill and ≥ 45 mm below any
slot-straddling pose, so a carton stuck in the doorway can never pass; both lying and
standing rest heights are in-window, asserted) AND **flap hanging closed** (< 15°, which
also rejects a carton propping the door) AND **milk settled** (lin ≤ 0.05 m/s, ang ≤
1.0 rad/s) AND **decoy out** (no part of the juice inside, in the doorway, or against
the bin — violation box |x| ≤ 185, |y| ≤ 175 mm, z ∈ [−20, 330] mm).

`score()` (stateless monotone ladder, `max` of): 0.20 carton lying on the loading shelf
· 0.55 carton entered past the doorway plane · 0.80 carton deep inside on the bin floor
· 1.0 iff `success()`.

## Solution outline (`solve.py`, transport-only teleports)

- **P0** settle + layout readback (flap < 3°, cartons on the floor, score ≤ 0.03).
- **P1** stage the carton: teleport (free transport of a graspable 60 mm-square carton)
  to the shelf at bin-local (0.290, 0, 0.154), lying with its long axis at the slot;
  assert `on_shelf`; score 0.20.
- **P2** push it through: velocity-servo horizontal force built in the BIN frame each
  step and applied in the CARTON body frame (`f_body = R_milkᵀ R_bin f_bin`; the pod's
  world-frame wrench drag reference is captured at the FIRST application per body and
  goes stale across resets, so world-frame forces mis-aim after a new bin yaw). Ladder
  (v 0.12/kp 15/cap 2 N → 0.20/25/3 → 0.30/40/5) with `set_state` snapshot rollback,
  plus a "deep re-push" with tighter cut planes when the carton commits past the doorway
  but parks straddling the sill. The flap yields (readbacks +22° → +35° → +57°), the
  carton tips over the inner sill edge and drops in; assert `milk_inside`; score 0.80.
- **P3** let the door fall shut: if the delivered carton lands STANDING against the flap
  sweep it props the door open (live failure, see execution order) — up to 4 gentle
  clear-nudges (1.1–1.3 N fingertip pushes through the still-open doorway) slide it
  clear; the flap then gravity-closes (readbacks +59° → +0.9° → −0.3°). Wait for a
  60-consecutive-step success streak (single instantaneous readings can land on velocity
  turning points), assert the flap is shut, score 1.0.
- **Persistence**: 400 hands-off steps (3.3 s) with success re-checked every step, then
  `SIM_GEN_SOLVE: SUCCESS`. Watchdog `threading.Timer` (daemon) + `os._exit` hard exit.

The milk is teleported exactly once (the P1 transport onto the shelf); every entry into
the bin is contact dynamics — push, door, tip, gravity. The flap is never actuated
directly by anything: no wrench is ever applied to it in solve.py.

## Embodiment (single Franka, parallel jaw, one base pose)

Base ≈ 0.6–0.7 m out along the bin's +x (shelf) axis reaches everything: the carton
spawn arcs (≤ 0.62 m from the bin, standing 60 mm cross-section < jaw stroke), the
shelf set-down at 0.29 m out and 0.15 m up, and the push — a 1–2 N horizontal fingertip
press at z ≈ 0.15 m sustained over a 0.20 m stroke, comfortably inside the arm's wrench
envelope. The clear-nudges reach THROUGH the open doorway: while the carton props the
flap the slot is a clear 120 × 200 mm window and the carton's near face sits ≤ 100 mm
behind the wall plane — within a finger's reach, and the nudge force (≈ 1.3 N) is a
fingertip touch. Reaching in any deeper is never needed; the roof denies all top-down
work by geometry, not fiat.

## Execution order

`scene.py` (geometry + ~20 honesty asserts in `__post_init__`) → `solve.py` iterated on
the forge until robust → rubric finalized (streak/settle gates) → `smoke.py` rejection
battery. No check was ever weakened to make a run pass; the one live failure — the
carton landing STANDING inside and propping the flap at ≈ 57°, failing the flap-shut
assert — was fixed in the solver (the P3 clear-nudge phase), not by relaxing
`flap_closed_deg`.

## Smoke battery (19 checks, rejection-driven)

1. reset settles: finite, flap shut, milk outside · 2. baseline score ≈ 0 · 3. bin
xy + yaw randomization readback (8 seeds) · 4. milk/juice poses vary · 5. null policy:
300 idle steps, score ≈ 0 · 6. roof denial: carton dropped from above lands ON the roof
(z = 350 mm), no credit · 7. staged-only → score 0.20, no success · 8. doorstep: carton
nosed against the CLOSED flap settles, flap holds (< 5°), ≤ 0.21 · 9. flap self-closes:
0.08 N·m inward → 80.6° (non-vacuous), released → −1.01° · 10. one-way: same torque
OUTWARD → min −1.00° (holds the stop) · 11. doorway mid-transit judged immediately →
0.55 cap, not inside · 12. acceptance: carton lying settled on the bin floor, flap shut,
decoy away → success TRUE · 13. retention: 4 N outward shove moves it 126 mm but the
sealed wall keeps it (peak x = 96 mm), success survives · 14. decoy posed in the doorway
→ success flips FALSE · 15. decoy removed → TRUE again · 16. settle gate: delivered
carton kicked (0.28 m/s) and judged immediately → rejected · 17. wrong object: JUICE
inside, milk on the floor → score 0.00 · 18. audit: success() never True at any judged
point except the constructed acceptance probes · 19. final no-NaN. Frames (231,
600 × 960) → `frames.npz`.

## Verification record (forge, 2026-08-09)

- `solve --seed 0`: rc=0, `SIM_GEN_SOLVE: SUCCESS`, 20.8 s. Scores 0.000 → 0.200 →
  0.800 → 1.000 → 1.000 (non-decreasing). Bin (−0.010, +0.002) yaw +170°; delivered at
  bin-local (+0.092, 0, +0.090) standing; 3 clear-nudges → (−0.011, 0, +0.090); flap
  +0.16° shut.
- `solve --seed 1`: rc=0, SUCCESS, 21.3 s. Bin yaw +62°; ladder rung 0 stalled, rung 1
  parked straddling → deep re-push delivered; 3 clear-nudges → (−0.021); flap +0.28°.
- `solve --seed 2`: rc=0, SUCCESS, 20.7 s. Bin yaw −153°; rung 0 delivered directly;
  3 clear-nudges → (−0.012); flap +0.07°.
- `smoke`: rc=0, `SIM_GEN_SMOKE: ALL PASS 19/19`, 39.5 s, 231 frames saved.
