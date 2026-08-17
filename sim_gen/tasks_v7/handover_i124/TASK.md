# handover_i124 — `weigh_station`

Hand the green parcel over to a counterweighed balance scale IN BALANCE: weigh it by
its size, seat exactly that many unit counterweights in the opposite rack, and leave
the beam level.

## Seed provenance

- **Seed task:** `mujoco_playground/handover`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/mujoco_playground/handover.py`): two ALOHA
  arms; one gripper picks a 4 cm box, hands it to the other gripper at a fixed
  handover point, which carries it to a floating target sphere. The entire skill is a
  **direct free-space transfer to a destination pose** — success is a pose match
  reached by carrying.

## What changed, and why it is strategically different

| | seed | this task |
|---|---|---|
| what "delivering" means | reaching a floating target pose | the destination is the cradle of a **pivoting balance beam** — merely reaching it makes the scale keel over onto its ±20° stop |
| skill | grasp, hand over, carry | **perception-driven mass arithmetic** (parcel size ⇒ mass in units, one shared density), counting (seat exactly k unit weights), **equilibrium management**, distractor discipline (spares stay off the scale) |
| success criterion | object at pose | a settled **physical equilibrium**: beam level within 6° with the right cargo in the right receptacles and nothing else aboard |
| failure structure | miss the pose | one unit too few / too many reads ~14–20° (≥ 2× tolerance); wrong-receptacle and unclean placements are level-proof rejected |
| ordering | pick → handover → carry | **none required** — ballast-first and parcel-first both reach equilibrium; only the settled end state is judged |

The seed's plan — carry the object to the destination — executed verbatim here latches
0.30 and **keels the scale onto its stop** (smoke check 5 proves it with a settled
construct). Success demands what the seed never asks: reading the parcel's size as a
mass label (46/58/66 mm = 1/2/3 units at one shared density), seating exactly that
many 0.15 kg counterweights in the rack at the same 0.25 m radius, and leaving a
*pendulum-stabilized* beam (authored CoM 95 mm below the pivot, restoring stiffness
K = Mg·|z_com| ≈ 1.49 N·m/rad) at rest inside a 6° window that a one-unit error
misses by design (imbalance torque 0.368 N·m ⇒ ≈ 14°, audited in `__post_init__`).
It is also different from every sibling task read during construction (e.g.
`franka_handover_i115`'s sealed-boundary carousel with forced ordering and mechanism
actuation): here there is no sealed boundary, no actuated mechanism, no forced order —
the difficulty is metrology on a passive plant.

## Teleport solution (solve.py) — phases

1. **Settle + perception** — read back the station pose and WHICH parcel stands on
   the green pad (station-local box test); its size class IS k. Asserted against the
   episode's k. `SIM_GEN_SCORE ~0.00`.
2. **DELIVER** — transport-only teleport of the shipment to free air 3 cm above the
   cradle rest pose, released with the beam's orientation; gravity + cradle walls
   seat it. The scale keels to its stop — expected. `SIM_GEN_SCORE 0.30`.
3. **BALLAST** — k unit weights, one per pocket: each released 12 mm above the LIVE
   pocket-centre world position (a world-vertical fall lands centred in the tilted
   pocket; entry offset h·sin20° ≈ 4 mm inside the ±8 mm slop). The beam swings back
   as the arithmetic closes (overdamped, slow pole K/b ≈ 0.5 /s — waited out, never
   forced). `SIM_GEN_SCORE 0.60`.
4. **VERIFY** — hands off until `success()` (level ∧ exact k ∧ cradled ∧ clean ∧
   settled), `SIM_GEN_SCORE 1.00`, then ≥ 3.3 more simulated seconds hands-off;
   success must persist → `SIM_GEN_SOLVE: SUCCESS`. Passes on forge seeds 0, 1
   (k = 3) and 2, 3 (k = 1) — both ballast-loop depths exercised.

## Embodiment argument (single Franka arm, parallel-jaw gripper)

Base pose: ground-mounted at station-local ≈ (0, −0.50 m), facing +y. The slab top is
31 mm off the ground; every graspable object and both receptacles lie within 0.56 m
horizontal reach at ≤ 0.31 m height — inside the Franka's 0.855 m envelope.

- **Counterweights (40 mm cubes, 0.15 kg):** 80 mm jaw span leaves 40 mm of margin;
  top-down grasp at every apron slot (open sky above); pocket entry slop ±8 mm with
  8 mm funnel walls — inside arm repeatability; release 1–2 cm above the pocket
  mouth reproduces the demonstrated centred drop.
- **Parcels (46/58/66 mm, 0.15–0.45 kg):** all within the jaw span (≥ 14 mm margin);
  0.45 kg ≪ the 3 kg payload; top-down grasp on the pad; the cradle accepts the
  largest parcel with ±5 mm slop and the release-above-rest strategy is demonstrated.
- **The beam is passive** — the robot never needs to touch it: all interaction is
  placing objects into open-top receptacles from above. Loading tilts the beam at
  most onto its ±20° stops; pocket friction (μ ≈ 0.55, arctan ≈ 29° > 20°) retains
  seated cargo, so intermediate keeling never undoes work.
- **Perception:** size classes are separated by 12 and 8 mm — resolvable by wrist
  camera at apron range; receptacles are colour-coded (RED cradle, BLUE rack, GREEN
  pad/parcels).

## Execution order

**Not required.** Ballast-first, parcel-first, or interleaved all reach the same
settled equilibrium; the rubric judges only the end state. (`describe()` says so
explicitly; the solve uses parcel-first, the "handover" reading of the seed.)

## Rubric (scene.py)

- 0.30 — latched max fraction of the REQUIRED ballast ever seated in the rack
  (min(seated, k)/k — over-ballasting earns nothing extra)
- 0.30 — shipment ever seated in the cradle (latched)
- capped at 0.60; exactly 1.0 iff live `success()`: cradled ∧ |tilt| < 6° ∧ exactly
  k weights in the rack ∧ no weight riding the beam outside the rack ∧ no depot
  parcel aboard ∧ settled (FD tilt rate) ∧ finite.

## Checks

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on forge seeds 0/1/2/3 (k = 3 and k = 1),
  monotone `SIM_GEN_SCORE` prints at phase boundaries, ≥ 3.3 s hands-off persistence.
- `smoke.py`: 14 rejection-only checks — settle/no-NaN (empty beam level), yaw/xy/k
  randomization spans, shipment-on-pad + depot + slot READBACK, null policy, SEED
  strategy (carry with no weighing → keeled, 0.30 cap), under-ballast (k−1, +18.5°),
  over-ballast (k+1, −15.5°, exact-count refusal, 0.60 cap), mirrored placement
  (level yet ~0 — wrong receptacles), unclean rider (level, exact k, cradled, yet
  refused by cleanliness alone), delivery miss (ballast-only, 0.30 cap), latched
  credit survives theft, rejection audit (success never observed), final no-NaN,
  frames.npz video.
