# put_umbrella_in_umbrella_stand_i220 — pull the umbrella OUT of the stand, lay it in the wall cradle

Scene `umbrella_unrack_cradle`, env `simgen.umbrella_unrack_cradle` (robot="null").

## Seed provenance

Derived from **rlbench/put_umbrella_in_umbrella_stand**: an umbrella and a floor
stand (mesh visuals, franka, no checker); the implied strategy is transport +
**vertical insertion** — grab the free-lying umbrella, point its tip DOWN, and
plunge it into the open tube of the stand so it ends up standing in a receptacle.

## Strategic difference

The seed's GOAL state is this task's INITIAL state, and the motion is inverted end
to end. At reset the umbrella is already **seated tip-down inside a deep socket**
(0.30 m tube, 12 mm wall, mouth at 0.33 m). The job is:

1. **constrained EXTRACTION** — draw the umbrella ~0.35 m straight UP the socket
   axis until the tip clears the mouth (the seed never extracts anything);
2. **reorientation to HORIZONTAL** — the seed's insertion keeps the shaft vertical
   throughout; here the goal orientation is axis within 10° of the horizontal
   plane, the opposite attitude;
3. **open two-point rest** — lay the bare shaft into TWO 90° V-notches atop a
   pair of cradle pillars (14 cm high, 0.20 m apart). No aperture, no
   containment, no suspension: the end state is a balance carried by two line
   contacts, nothing wrapped around anything.

Also different from sibling **i72** (crook-on-rail hang): i72 ends in a hanging
suspension judged by hook containment + ground clearance with the shaft
near-vertical; i220 ends in a supported horizontal rest judged by seat-point
distances + horizontality, starts from the seed's success state, and its umbrella
has no crook at all.

A decoy CANE (straight stick + ball knob) stands in the SECOND socket of the same
stand (socket sides randomly swapped); success and score are judged on the
umbrella by identity.

## Solution outline (solve.py — teleports for transport only)

- **P0** settle + layout readback; assert score ≤ 0.02 (rubric-leak guard).
- **P1 EXTRACT (contact dynamics)**: a velocity-regulated vertical force
  (+0.25 m/s target, force at the CoM, clamped ≤ 2× weight) with a lateral PD
  keeping the shaft on the socket axis draws the umbrella up the tube; the socket
  walls guide the whole travel. Latch fires when the low end clears
  `MOUTH_Z + 0.03` = 0.36 m; 20 hold steps; assert score ≥ 0.30.
- **P2 CARRY (transport)**: teleport to a hover pose above the cradle — shaft
  horizontal along the notch line, 50 mm above the seats; 10 regulated
  zero-velocity hold steps. Asserted the seats are NOT yet engaged.
- **P3 LAY (contact dynamics)**: regulated slow descent (−0.04 m/s) to a 1.4 mm
  gap, then a clamped position-servo **press** closes the air gap under full
  support, then a staged quasi-static load transfer (0.60/0.35/0.15 mg with pure
  velocity damping) hands the weight to the notches; wrench cut; long hands-off
  settle. The final balance is pure contact statics.
- **P4 persistence**: ≥ 3.3 more simulated seconds hands-off with success()
  checked every substep, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing
(0 → 0.30 → 0.30 → 1.0 → 1.0). Passes on seeds 0, 1, 2 on the forge.

Iteration note: the balance failed deterministically for 3 runs (canopy-end
pitched down out of every two-point rest, even quasi-static). Root cause: only
MassAPI *mass* was authored on the compound root, so PhysX volume-weighted the
colliders and put the CoM ~15 cm inside the bulky canopy capsule — outside the
support span. Fix: author `centerOfMass` explicitly at the body origin, midway
between the two seats. After the fix the rest is rock-still at seat_d =
(12.7 mm, 12.7 mm), rest axis-z = 0.153 m.

## Rubric

`success()` = seated ∧ horizontal ∧ settled, judged on the umbrella:

- **seated**: BOTH V-notch seat points within `seat_tol` = 30 mm of the shaft
  axis SEGMENT. The demonstrated rest reads exactly 12.7 mm on both (nested
  distance R_REST = shaft_r·√2); a floor rest reads (381, 336) mm; a crosswise
  one-notch balance engages one seat (23 mm) but leaves the other at 201 mm.
- **horizontal**: |axis·ẑ| ≤ sin 10°. A diagonal lean propped on one notch
  (~14°) fails this clause.
- **settled**: stillness (lin < 0.02 m/s, ang < 0.30 rad/s) SUSTAINED for 30
  consecutive substeps — thresholds set BELOW the solve's regulated descent
  speed so a wrench-held touchdown can never read settled (observed live in run
  2, then fixed).

`score()`: latched 0.30 (low end ever above 0.36 m — extraction) + 0.30 (ever
seated ∧ horizontal), cap 0.60; 1.0 iff success(). Null policy ≈ 0: a seated
umbrella's low end sits at 0.033 m, an order of magnitude under the extract
latch line.

## Embodiment argument (Franka)

The seated umbrella exposes ~35 cm of bare 18 mm shaft above the socket mouth —
a natural power grasp. From a base at the origin facing +x (stand at ~0.55 m,
cradle seats at 0.45–0.65 m reach, both inside Franka's envelope), the arm:
grasps the exposed shaft, pulls straight up ~0.35 m (a pure vertical Cartesian
move, the socket itself guides the part), reorients to horizontal with a wrist
rotation while carrying, aligns the shaft over the two notches, lowers the last
few cm, and opens the gripper. The load transfer needs no force control — the
V-notches self-center a cylinder. The difficulty is perception (which object is
the umbrella, socket/cradle poses under randomization) and the two-point
alignment, not dexterity.

## Execution order

1. `scene.py` written first (geometry + minimal rubric).
2. `solve.py` iterated on the forge: runs 1–4 exposed a set-down jolt, a rubric
   settle-threshold hole, and finally the volume-weighted-CoM root cause (fixed
   by authoring `centerOfMass` on the compound root). SUCCESS on seeds 0, 1, 2.
3. Rubric finalized against the demonstrated rest (12.7 mm seats, axis-z 0.153,
   extract clearance margins) with `__post_init__` asserts.
4. `smoke.py` rejection battery: **ALL PASS 14/14** on the forge, frames.npz
   saved.

## Smoke checks (14)

1. settle/no-NaN + layout sanity; 2. score ≈ 0 at reset; 3. stand + cradle pose
randomization readback; 4. socket side-swap occurs and stays sane; 5. null
policy ≈ 0 (low end never nears the extract line); 6. seed-strategy end state —
umbrella re-inserted into the socket — scores ~0; 7. floor rest (horizontal on
the ground) rejected by seating; 8. crosswise one-notch balance rejected (far
seat unengaged); 9. diagonal one-notch lean rejected by horizontality; 10. cane
laid in the cradle counts for nothing (identity); 11. settle gate — seated +
horizontal but still moving is not success (and anchors seat_d < tol); 12.
latched credit survives teleport-away (0.30 kept, seated drops); 13. rejection
audit (success never True in the battery); 14. final no-NaN.
