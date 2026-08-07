# milk_carousel — rotate the covered dispenser, extract the milk, stand it on the pad

## Seed provenance

- Seed id: `libero/libero_pick_milk`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero/libero_pick_milk.py`
  ("Pick the milk and place it in the basket": the milk carton stands FREE on the
  table among food distractors that exist only to be ignored; the plan is one
  prehensile transport — grasp, carry, drop inside the basket; success = a
  relative-bbox containment check against the basket).

## What changed, and why it is strategically different

The seed's whole plan is a single free-space pick-and-place into an open container.
Here every element of that plan is invalidated:

1. **The target is not free to grasp — it rides a mechanism.** The milk carton sits
   in a walled bay on a rotary carousel (a real revolute joint) parked under a fixed
   ring roof. The ceiling gap over a covered bay is 30 mm; clearing the bay walls
   takes a 55 mm rise. Extraction from a covered bay is therefore PHYSICALLY
   impossible (smoke check 9 proves a 3x-gravity lift cannot escape), not merely
   un-credited. The first and dominant skill is **continuous rotary actuation with
   feedback**: push the crank peg (which orbits above the roof) tangentially,
   re-approaching as it moves, until the milk's bay reaches the single open roof
   sector — there is nothing analogous anywhere in the seed.
2. **Containment is the WRONG answer, not the goal.** The terminal relation is
   inverted in kind: the carton must end standing UPRIGHT on a flat open pad within
   45 mm / 10 deg, and the open gray bin sitting nearby — the literal analogue of the
   seed's basket — is a decoy: a carton settled inside it is a tested, rejected
   outcome (smoke check 5). A solver replaying the seed's plan fails twice over: it
   cannot grasp the milk without rotating first, and its preferred destination earns
   nothing.
3. **Distractors act, not just distract.** The juice carton and the can ride the
   same carousel through the same contact dynamics — they are moved BY the solver's
   rotation and must simply not be delivered (wrong-object rejection, smoke check 6).

A solver needs a different plan (mechanism actuation -> constrained extraction ->
precision free placement) and different code structure (a closed-loop rotation servo
on a measured azimuth error, then a pick), not different parameters of the seed's
plan. (Also distinct from this seed's earlier campaign derivatives: v0 was carton
reorientation, v1/`tasks/` was crate load-inversion with a free lid — no carousel, no
covered-access mechanism, no decoy container in either.)

## Difficulty tier / stages

- **Tier: medium. Declared stage count: 3** (rotate to align; extract through the
  sector; deliver upright on the pad).
- **Execution order: REQUIRED and physically enforced** — rotation must precede
  extraction (the roof interlock; smoke checks 9/10 measure both sides of it).
  Extraction necessarily precedes delivery.

## Teleport-solution outline (solve.py — the legitimacy certificate)

1. settle; assert score 0; `SIM_GEN_SCORE 0.0`;
2. ROTATE (contact dynamics): torque about the carousel spindle against its viscous
   friction — the same external-wrench channel a fingertip on the crank peg
   exercises — with the three bay items carried by their bay walls through real
   contact; closed loop on the measured milk azimuth until |error| < 8 deg and the
   rotor is at rest. No pose of the rotor or of any bay item is written after reset.
   `SIM_GEN_SCORE 0.25` (aligned latch);
3. LIFT (contact dynamics): a 1.6x-gravity vertical force at the carton CoM raises
   it out of the bay and through the open sector past the roof plane (the covered
   counterfactual is the smoke's interlock check). `SIM_GEN_SCORE 0.55`;
4. PLACE (transport only): with the carton verified airborne above the roof plane,
   ONE teleport to 25 mm above the pad; the landing is a real gravity drop that must
   settle upright inside the tolerances. `SIM_GEN_SCORE 1.0`;
5. persistence: 3.3 simulated seconds hands-off; success() must still hold; only
   then `SIM_GEN_SOLVE: SUCCESS`. Passed on seeds 0 and 1 (see forge logs).

## Embodiment argument (single Franka + parallel jaw, OSC)

Plausible base pose: **(-0.45, 0.00, 0)** facing +x (carousel axis at (0.12, 0), i.e.
0.57 m; the open sector faces the robot at azimuth 180 deg +/- 12).

- **Crank peg (the only mechanism contact):** a 22 mm red cylinder, top at 0.35 m,
  orbiting at radius 0.13 ABOVE the roof plane — an obstacle-free horizontal push
  target. Strategy: closed fingertips push the peg tangentially in strokes
  (0.44-0.70 m from the base at an easy height; re-approach between strokes; either
  direction works and the carousel coasts to rest on viscous friction, so strokes
  compose). Pinch-and-drag arcs also work (oven_dials precedent: pinch-turn cribs).
  Required precision: stop the milk bay within +/-25 deg of the sector center —
  bay-vs-gate slack is ~3x the arm's demonstrated rotary-stop accuracy on dials.
- **Milk carton (the only grasped object):** 60 mm square, 140 mm tall, mass 0.25 kg;
  top-down pinch on its upper half (grip gate 60 mm << 80 mm jaw; the proven
  pen_holder-family carton pinch). At the open sector the carton center is ~0.45 m
  from the base; the grasp band (z 0.14-0.20) is 87+ mm above the 123 mm bay-wall
  tops, and the open sector is 110 deg wide (the carton subtends < 57 deg with all
  slack) — free vertical extraction with > 25 mm lateral clearance everywhere.
- **Placement:** pad center 0.46 m from the base, ground level; stand-upright within
  45 mm / 10 deg — the same tolerance family the crate_turnover / pen_holder Franka
  runs landed at 1-12 mm.
- **Juice carton / can:** never need contact; they ride the carousel.
- Nothing requires reaching under the roof: the interlock forbids exactly the
  contacts the arm cannot make, and rewards the ones it can.

## Judging (physical outcomes only)

- `success()` (current state): milk upright (<= 10 deg) on the pad (center within
  45 mm, bottom at pad top +/- 12 mm), settled (< 0.05 m/s).
- `score()`: latched 0.25 aligned (milk bay ever within 25 deg of the sector while
  in the dispenser) + latched 0.30 freed (ever above the roof plane or outside the
  ring) + latched 0.15 transported (ever over the pad); exactly 1.0 iff success().
  Latches are NaN-guarded and never decrease; null policy scores exactly 0 (the
  start offset is sampled >= 65 deg from the gate).

## Check list (smoke.py — rejection battery; solve.py is the acceptance proof)

1. settle/no-NaN; milk starts covered; score 0
2. randomization readback (rotor yaw, milk error, sector azimuth, pad)
3. bay permutation + both offset signs across 10 resets
4. null policy ~0
5. seed strategy (milk in the bin) rejected
6. wrong object (juice on the pad) rejected
7. near-miss pad xy (60 mm vs 45 mm gate) rejected
8. near-miss upright (lying on the pad) rejected
9. interlock: covered bay defeats a 3x-gravity lift (never freed)
10. interlock: aligned bay releases the same lift (opening is real)
11. latch persistence: earned 0.55 does not evaporate, still no success
12. aligned-gate honesty: 40 deg rejected / 10 deg accepted
13. frames.npz saved

Self-contained: all geometry procedural (boxes + cylinders); no external assets.
