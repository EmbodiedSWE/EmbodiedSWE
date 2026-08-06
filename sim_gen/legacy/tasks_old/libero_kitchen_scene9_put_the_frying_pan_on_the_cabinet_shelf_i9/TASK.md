# pan_cubby_retrieval — slide the frying pan OUT of the cabinet, set it on the burner

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf.py`)
- Seed plan: pick the frying pan up from the open counter, lift it, place it **onto** the
  wooden two-layer shelf (free vertical pick-and-place into an open region; the flat stove
  is a distractor). Success = pan inside the shelf's `top_region` bounding box.

## What changed / why strategically different

The spatial relationship and the required plan are **inverted**, not re-parameterized:

- The pan starts **inside** a closed-back cabinet cubby whose roof leaves only ~29 mm of
  head room above the 36 mm pan. A vertical lift — the seed's opening move — is
  **geometrically impossible**: the pan physically jams on the roof (verified by a
  velocity-kick probe in the smoke). The only feasible plan is to **slide the pan
  horizontally** out through the single open face, and only then lift it.
- The seed's goal fixture and its distractor **swap roles**: the shelf/cabinet is now the
  obstacle the pan must escape, and the stove burner (a distractor in the seed) is the
  placement target. Placing the pan **on top of the cabinet** — the seed's literal goal —
  is a tested negative control and scores < 0.3 with no success.
- Success is judged on physical outcome only: pan settled flat ON the burner pedestal
  (centre within 5 cm of the axis, base at the burner top ±2 cm, upright within 15°,
  |v| < 5 cm/s) and fully outside the cubby footprint.

A solver therefore needs a different plan (horizontal extraction under a ceiling
constraint, then a placement on a raised pedestal), not different numbers.

### Relation to sibling tasks from the same LIBERO scene

Sibling `..._put_the_frying_pan_under_the_cabinet_shelf_i4` (scene `pan_lowshelf_slide`)
also uses a low-clearance alcove, but with the **opposite goal topology and a different
skill**: there the pan starts in the open and must be pushed non-prehensilely INTO the
confined space, which is the terminal state (1 stage, insertion). Here the confinement is
only the *starting* obstacle: the pan must come OUT of it (retrieval), and the deliverable
is a two-stage retrieve-then-place onto a separate raised target with placement tolerances
(centring, height, uprightness, settling) — plus the goal/distractor role swap of the
seed's fixtures. The plans do not transfer: i4's push-in policy never leaves the cabinet
region and never touches a placement target; i9's policy must exit it and finish a
precision set-down elsewhere.

## Difficulty tier / stages

**easy — 2 stages**, execution order **REQUIRED** and topologically forced:
1. slide the pan out through the cubby mouth (drag on its base; no lift possible inside);
2. lift and set it down flat on the burner.

The order cannot be swapped: the pan cannot reach the burner without first leaving the
cubby through the mouth (back and sides closed, roof too low to lift over).

## Scene / rubric

Procedural assets only (compound spawners, one rigid body each): pan = base disc + 8 rim
boxes + handle box; cubby = kinematic floor/roof/sides/back, open front; burner =
kinematic cylinder. Randomized per episode (verified by readback): cubby yaw + xy, pan
depth/lateral/handle-yaw inside the compartment, burner xy over a 36 cm band.

Graded score in [0, 1], transient achievements latched (running maxima):
- 0.35 · extraction progress (pan travel toward/through the mouth, cubby body frame)
- 0.15 · extracted (pan centre fully outside the cubby footprint; the roof is *inside*)
- 0.35 · approach progress toward the burner (gated on extracted)
- capped at 0.85 unless `success()` holds live → exactly 1.0. Doing nothing ≈ 0.

## Smoke checks (20)

1. settle: states finite, pan at rest after 60 steps
2. clean slate: score ~0, no success at reset
3. randomization real (readback): pan spawn depth spread > 2 cm
4. randomization real (readback): burner y spread > 4 cm, cubby yaw spread > 1.5°
5. randomization real (readback): handle yaw spread > 2°
6–8. teleport-oracle reaches `success()` with score 1.0 on seeds 0/1/2
9. rubric monotone along the oracle
10. milestones strictly increase (start < mid-drag < extracted < approach < placed)
11. null policy: score ~0, no success after 150 idle steps
12. **seed-strategy control**: pan on the cabinet TOP does not count as extracted
13. **seed-strategy control**: no success, score < 0.3 on the cabinet top
14. near-miss control: 9 cm off the burner axis (tol 5 cm) is not success
15. near-miss control: partial credit only (0.5 ≤ score < 1.0)
16. flipped-pan control: upside-down on the burner rejected (upright gate)
17. calibration probe: roof physically blocks lifting (velocity kick, max rise < 50 mm)
18. get/set_state roundtrip restores the pose
19. calibration sweep: placements at 0/2/4 cm all succeed
20. calibration sweep: placements at 6/8/10 cm all fail (clean edge around the 5 cm tol)

Negative controls: the seed's own strategy **is expressible** here (put the pan on top of
the cabinet) and is tested (checks 12–13); near-miss/tolerance controls are checks 14–15
and the sweep (19–20).
