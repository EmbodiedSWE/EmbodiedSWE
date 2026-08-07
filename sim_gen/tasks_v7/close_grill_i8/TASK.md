# close_grill_i8 — Build the Fire Crib (scene `fire_crib`)

Erect a two-layer log-cabin FIRE CRIB on a marked hearth plate from four loose wooden
kindling splits — two laid parallel with an open 7.8–12.8 cm air gap, two laid across
them at right angles — then prove the structure by resting a steel griddle plate flat
and LEVEL on top (its centre ends ~49 mm above the hearth, a height reachable only
through the completed two layers). A 48 mm OFFCUT of the same stock is scrap: it is
shorter than the smallest legal air gap (dropped across the gap it falls straight
through) and must be left OFF the hearth plate.

## Provenance

- **Seed:** `rlbench/close_grill`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/close_grill.py`) — "close the
  grill": a barbecue-grill USD with one revolute lid, robot=franka. The entire plan
  is ONE unordered pushing contact on a panel already attached to the fixture,
  rotating it about its hinge until shut; judged by a door joint, no other object.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `fire_crib`, env
  `simgen.fire_crib`, robot `"null"`), `solve.py` (teleport solution), `smoke.py`
  (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed manipulates a body that is already ATTACHED to the goal
  fixture — one push, one rotation, the appliance itself guides the motion, and
  nothing else exists in the scene. Here NOTHING is hinged and nothing is pushed
  shut: the solver must CREATE a structure from five free bodies where none exists,
  and every placement after the first rests on objects the solver itself placed —
  the scene's only fixture (the flat hearth plate) guides nothing. The seed's whole
  skill has no analog here; its closest naive translation — "cover the appliance:
  lay the griddle flat on the hearth" — is constructed verbatim in smoke check 6 and
  scores ~0 with no success.
- **vs `close_microwave_i4`/`i5`, `libero_i3` (read this session):** those close an
  aperture by mechanism release (drop-gate) or bayonet twist-lock — a fixture with
  internal DoF does the guiding. Here there is no aperture, no mechanism, and no
  fixture DoF anywhere: the judged outcome is an emergent free-standing STRUCTURE.
- **vs `setup_checkers_i2` (read this session):** that is ordered insertion into a
  guiding silo channel that holds whatever it is given. Here nothing guides or
  captures — stability is the solver's own doing, and layers are heterogeneous
  (on-plate pair → orthogonal spanners → level load), not a repeated drop.
- **vs `peg_insertion_side_i2` (bar+pin fastening), `peg_i1` (ramrod eject),
  `egad_i3/i4`, `libero_i2`, `light_bulb_i5`, `pen_holder`:** no insertion, no
  threading, no pouring, no lever, no container — none of their mechanics appear.
- **Length-identity perception:** five bars of identical cross-section and colour
  family are permuted over five scatter slots each episode; only LENGTH separates
  members from scrap, and the scrap physically cannot serve (smoke 5).
- **Execution order is REQUIRED and physically inherent** (declared): bottom pair,
  then spanners, then griddle — each stage RESTS ON the one below, so no later stage
  can exist before its support does; the griddle success band (≥41 mm above the
  hearth) is unreachable from a single layer (cfg assert + smoke 7).

## Randomization (per episode, verified by readback in smoke)

Hearth plate xy jitter + full yaw (the build frame moves — the solver must read it),
the five bars permuted over five scatter slots with xy jitter + free yaw (batched
overlap-rejected resampling), griddle xy jitter + free yaw.

## Rubric

`success()` iff, settled (every dynamic body |v| < 0.05 m/s) and finite:

- CRIB: the four splits partition into a legal bottom pair (both flat ON the hearth
  within its footprint, parallel ≤20°, centreline gap 78–128 mm, height band
  4–20 mm above the hearth top) and a legal top pair (height band 26–42 mm,
  parallel, same gap window), every top member crossing BOTH bottom members
  (orthogonal ≤20° tol, 2D intersection ≥8 mm inside both bars' ends);
- GRIDDLE: centre 41–60 mm above the hearth top, face level (≤12°), xy within 55 mm
  of the four-split centroid;
- RESTRAINT: offcut centre outside the hearth footprint and at ground level.

`score()` (latched in `post_step`, splits calm for structure latches): `0.10 ×`
approach (a split ever within 0.20 m of the hearth centre) `+ 0.25 ×` base pair ever
formed `+ 0.20 ×` first spanner ever rested across both rails `+ 0.20 ×` full crib
ever complete. Capped at 0.75; exactly 1.0 iff `success()`. Doing nothing scores ~0.

Cfg `__post_init__` asserts the honesty geometry: offcut < smallest legal gap
(cannot span), splits and griddle reach across the widest legal gap, the griddle
success band is unreachable from a single layer, layer bands disjoint.

## Teleport solution (`solve.py`) — transport only, five writes, all ending in free space

Each body is teleported ONLY to a free-space HOVER 12 mm above its final rest —
over the hearth for the bottom pair, over the placed bottom pair for the spanners,
over the completed crib for the griddle — with zero velocity (the carry a gripper
performs). Every seating is GRAVITY + CONTACT: the body falls and lands on the real
plate / the real lower splits; no structural fact is ever written. Order: bottom
pair (hearth-frame ±47.5 mm, gap 95 mm) → two orthogonal spanners → griddle. The
offcut is never touched. `SIM_GEN_SCORE` at every phase boundary is non-decreasing
(0.000 → 0.350 → 0.550 → 0.750 → 1.000 → 1.000), ≥3.3 simulated seconds hands-off
persistence, then `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1,
both SUCCESS, provably distinct layouts by stdout readback** (hearth yaw −173.0° vs
−90.9°, hearth/griddle xy moved, slot permutation [1,2,4,0,3] vs [3,1,0,4,2]).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0.08, −0.05), facing the workspace: hearth centre ~0.40 m, scatter
slots 0.28–0.42 m, griddle spawn ~0.43 m — all within a comfortable dexterous shell,
and the tallest judged point (griddle knob top, ~90 mm) is trivially low. Per-object
contact strategy: each split (170 × 22 × 22 mm, 60 g) is pinched TOP-DOWN across its
22 mm width (≪ 80 mm jaw stroke) anywhere along its free length, carried, and laid
down — bottom-layer tolerance (gap window 50 mm wide, pad footprint margin 10 mm)
and top-layer tolerance (crossing margin leaves ±77 mm of legal contact per rail)
are far beyond arm repeatability, and set-down is a release from a few mm, not a
precision insertion. The griddle (150 mm square, 220 g) is grasped by its dedicated
24 mm square knob (top-down pinch, the plate hangs level below the wrist) and
lowered onto the top pair; 55 mm centring tolerance and 12° level tolerance make
this a compliant set-down. Nothing needs to be reached under or through anything:
grasp approach is a clear vertical. The offcut requires no action at all.

## Checks (`smoke.py` — rejection battery, 12 named checks, ALL PASS on the forge)

1. settle: states finite, splits flat on the ground, offcut off the hearth; score 0.
2. randomization readback: hearth yaw Δ175.5°, hearth xy Δ32 mm, split_0 xy Δ266 mm
   across seeded resets.
3. slot permutation: offcut occupies all 5 scatter slots, 10/10 distinct
   permutations over 10 resets.
4. null policy: 240 idle steps → score 0, no success.
5. OFFCUT CANNOT SPAN (identity claim is physics): dropped across the narrowest
   legal pair it falls straight through to the hearth (settled 11 mm vs 44 mm rail
   height).
6. SEED-analog naive: griddle laid flat on the bare hearth ("cover the appliance")
   → score 0, no success.
7. single-layer cheat: four splits flat side by side + griddle on them → griddle at
   27 mm, below the 41 mm band; no crib; score ≤ 0.36.
8. parallel-tower cheat: top pair stacked PARALLEL on the bottom pair, griddle at
   the correct height (griddle clause even true) → crossing clauses refuse the
   crib, no success.
9. near-miss gap: full crossed crib + griddle with the bottom pair at 60 mm
   (< 78 mm window) → no base/crib credit, score ≤ 0.15, no success.
10. near-miss spanner: second spanner shifted along its axis, resting on only ONE
    rail → span1 only, no crib, no success.
11. restraint load-bearing: FULL correct crib + griddle with the offcut parked ON
    the hearth (built offcut-first so success is never True at any judged instant)
    → every structural clause holds, restraint alone refuses; score = 0.75 cap.
12. frames.npz (80 × 600 × 960 × 3) recorded and saved in CWD.
