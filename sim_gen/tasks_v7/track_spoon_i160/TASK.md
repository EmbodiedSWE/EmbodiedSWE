# track_spoon_i160 — Utensil Balance: weigh the hidden counterweight (`simgen.utensil_balance`)

## Seed provenance

Derived from **pick_place/track_spoon**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_spoon.py`): a Stage-3
trajectory-tracking task — the spoon starts ALREADY RIGIDLY GRASPED in the Franka's
closed gripper, and reward is dense per-step position+rotation tracking of a
prescribed free-space waypoint path toward a basket. The seed's whole strategy is
*transport fidelity of a held payload along given waypoints*; there is nothing to
discover, no mechanism in the loop, and no failure mode beyond deviating from the
path.

## What changed, and why it is strategically different

Kept only the protagonist family (kitchen utensils that must end up somewhere
specific). The judged skill is inverted from transport fidelity to **interactive
mass identification through a compliant mechanism readout**:

- **Nothing is prescribed and nothing is tracked.** There is no waypoint list and no
  per-step reward. A two-pan balance scale holds a hidden counterweight in one
  hanging pan; the three candidate weights are visually IDENTICAL (25/45/75 g steel
  cylinders — mass is invisible to any camera), and exactly one of three utensils
  (brass fork 25 g, silver spoon 45 g, dark ladle 75 g) matches it.
- **The payload's placement is only the INPUT to a mechanism; the mechanism's settled
  RESPONSE is what is judged.** The beam is level (within 6°, ≈ 3 g) only under
  equal pan loads; every wrong pairing (≥ 20 g gap) out-torques the authored keel
  and pins the beam at its ±28° stop with a 1.25× margin (asserted in cfg). The
  solver must USE the scale — load the empty pan, read the beam, swap — a
  hypothesis-test loop, not a trajectory.
- **Information asymmetry forces interaction.** Which mass is active is drawn per
  episode and cannot be perceived; the only channel that reveals it is the physical
  experiment itself. The seed (and every corpus task) judges geometry a camera could
  verify; here the goal predicate depends on a hidden physical property (mass).
- **Versus the read corpus:** no existing task uses a passive compliant mechanism as
  a measurement instrument. `track_banana_i79` actuates a spring clamp to *release*
  objects; `track_bowl_i27` drags an enclosure as a *capture tool*; the drawer /
  microwave / faucet tasks articulate containers; `pen_holder` is containment.
  None have a hidden physical quantity the robot must measure, and none judge a
  mechanism's equilibrium response to a chosen load.

## Apparatus (fully procedural, no meshes)

Heavy DYNAMIC stand (25 kg — never kinematic: spawn-authored joint anchors on a
teleported kinematic body0 stay world-fixed on this stack; reset teleports the
stand): base slab + column + clevis. Beam (0.36 m, 0.40 kg) on a spawn-authored
revolute joint (axis Y, ±28° hard stops) with its authored CoM 10 mm BELOW the pivot
— a keel, drawn as a red pointer fin, that makes the empty beam stable-level and maps
mass error to tilt via tanθ = Δm·L/(M·d): the 6° level band ↔ < 3 g, the smallest
wrong pairing (20 g) pins the stop. Two HANGING pans (walled square trays on yoke
rods under revolute joints at the beam tips, CoM 105 mm below the pivot): hanging
pans make beam torque depend only on total pan load, not placement luck — the
readout measures MASS. Three utensils (fork/spoon/ladle: distinct sizes/colours,
flat heads with rims, handles attached at head-top level so a lying handle floats
~6 mm for a parallel-jaw pinch; each fits fully inside a tray). Three
identical-looking weights whose authored masses mirror the utensils; one is the
ACTIVE counterweight seated in a random pan at reset, two are parked ~2 m away.
Masses are authored via MassAPI inside the custom spawners (cfg mass_props are
ignored on this stack) and read back in smoke check 3. Config honesty is asserted in
`__post_init__` (stop-pinning margins, level-band separation ≥ 4×, tray fit, handle
float + jaw fit, pan clearance of column and slab at the stops, floor slots outside
the clear_r clause).

**Randomization (readback-verified):** stand xy ±3 cm + yaw ±15°, the hidden match
identity (3 values), which pan holds the weight (both sides), utensil floor slots
(permutation + xy jitter ±2 cm + free yaw).

## Rubric

Judged on the settled state, in body frames (beam pitch relative to the live stand;
tray containment in the pan's frame — valid mid-tilt since pans hang plumb):

- `success()` = active weight resting in one pan ∧ the MATCHING utensil resting in
  the OTHER pan ∧ beam level (|pitch| ≤ 6°) ∧ both rejected utensils flat on the
  FLOOR (not the base slab) ≥ 0.24 m from the stand ∧ both unused weights ≥ 0.50 m
  away ∧ everything at rest.
- `score()` (monotonic, latched every substep): 0.20 · a utensil ever loaded
  opposite the seated weight + 0.30 · the match-vs-weight arrangement ever level and
  quiet + 0.20 · success() ever held, capped at 0.70; exactly 1.0 iff success() now.
  Null policy ~0 (the weight alone visibly pins the beam); the seed's strategy
  (carry the spoon somewhere and set it down) ~0.

## Solution outline (solve.py — the legitimacy certificate)

Teleport = TRANSPORT ONLY, always ending in a non-contact hover 30 mm above the open
tray airspace (pans hang plumb — no tilted-wall drift) or over open floor; every
load-bearing interaction is gravity + contact through the passive mechanism:

- **P0 SETTLE** — the counterweight tips the beam to its stop (~5 s); assert
  |pitch₀| ≥ 20° with the drawn side's sign. The reset state is a loaded scale.
- **P1 PROBE (discovery demo)** — a deliberately WRONG utensil is dropped into the
  empty pan; the load transfers through contact and the beam settles pinned at the
  OTHER stop (readback −28.00°). The teleport bypasses nothing: the judged quantity
  is the beam's settled response to the utensil's true mass.
- **P2 SWAP** — the wrong utensil is lifted out (dropped back to its floor slot) and
  the MATCH dropped in; equal masses leave the keel as the only net torque and the
  beam settles LEVEL (+2.3° → +0.1°).
- **P3 PERSISTENCE** — hands off ≥ 3.3 simulated seconds after success() first
  holds; verdict only if it still holds.

`SIM_GEN_SCORE` printed at each phase boundary, non-decreasing 0 → 0.20 → 1.0 → 1.0.
Verified on the forge for seeds **0**, **1**, and **2** (rc=0, ~36 s each) —
covering two match identities (fork, spoon) and BOTH pan sides — with the physics
matching the cfg math exactly (stop = ±28.00°, level = |0.07°|).

## Embodiment argument (Franka, one base pose)

Base on the floor at the world origin facing +x; the stand centre is at
(0.45 ± 0.03, ±0.03) so the pans (stand-local x = ±0.16, tray floors ~0.19 m up)
sit 0.28–0.65 m from the base — inside the 0.855 m reach envelope at comfortable
heights. Per-object contact strategy:

- **Utensils:** lying handles are 15 mm wide and float ~6 mm above the floor (both
  asserted) — a standard parallel-jaw pinch (80 mm stroke closes to 15 mm). In a
  pan, the handle stays ≥ 8 mm above the tray floor and the 22 mm walls leave the
  hand approach open from above; each utensil fits fully inside the 145 mm tray, so
  placing = hover-and-release, exactly the motion solve certifies. Heaviest is
  75 g — trivial payload.
- **The scale:** never needs to be touched. Reading the beam is visual (the red
  pointer fin swings 56° lever-arm); loading a pan is a release into the open tray.
  All-force interactions in solve are ≤ the weight of a 75 g utensil.
- **Weights:** never need to be touched (the active one is pre-seated; the parked
  ones must merely be left alone, 2 m away).

## Execution-order declaration

NO required execution order. The rubric judges only the settled terminal state
(plus latched partial credit); any probe sequence — including placing the match
first by luck — counts. In practice a solver that cannot perceive mass must
iterate load-read-swap, but the rubric imposes no such sequence.

## Checks (smoke.py — rejection battery, recorded, `SIM_GEN_SMOKE: ALL PASS 16/16`)

1. Settle/no-NaN: the counterweight pins the beam toward its own pan (sign
   readback); utensils flat on the floor.
2. Score ~0 at reset, no success.
3. Authored-mass readback: utensils and weights = 25/45/75 g via get_masses().
4. Randomization readback: stand xy + yaw and utensil slots vary (8 seeds).
5. Randomization readback: match takes ≥ 2 values, the weight's pan takes both
   sides, and the active weight rests in its declared pan every episode.
6. Null policy (240 steps): score ~0, no success.
7. Seed-strategy analog: the spoon carried and SET DOWN by the scale (on its base
   slab) — mechanism never engaged, score ~0.
8. Wrong utensil in the free pan: the beam answers by pinning at the stop;
   engagement latches its 0.20 and nothing more.
9. Level readout positive control (with a permanent floor-clause violation): the
   match levels the beam, `balanced` latches (0.50), success still refused.
10. Same-pan stuffing: the match beside the weight — pinned, no engagement latch,
    ~0.
11. No weight: the keel returns the empty beam level — level alone is not the task.
12. Parked-weight clause: otherwise-perfect terminal state with a spare weight at
    0.30 m — parked_far alone rejects success.
13. Settle gate: mid-swing the state is refused (velocity readback), settled after.
14. Latched credit survives removing the match from its pan.
15. Rejection audit: success() never True at any judged point in the battery.
16. Final no-NaN; frames.npz saved.
