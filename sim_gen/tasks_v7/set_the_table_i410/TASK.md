# set_the_table_i410 — Wobbly Bistro: shore the short leg, THEN set the table

## Seed provenance

Derived from **`rlbench/set_the_table`**
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/set_the_table.py`): take five
utensils (fork/knife/spoon/plate/glass) out of a holder and arrange them at
designated flat spots on a table. The seed is N independent pick-and-places onto
a passive, perfect surface — the table contributes nothing but a plane.

## What changed and why it is strategically different

The **table itself is the mechanism**, and the plan is inverted:
repair-then-place instead of arrange-N-items.

- The bistro table is a free **dynamic** 4-leg body with ONE leg **25 mm
  short**. A small CoM bias makes the empty table rest **level** on its three
  long legs, so the fault is invisible in the tabletop — the only tell is the
  short leg **hovering ~25 mm** off the floor.
- Both marked seats (red ring = plate, pale-blue disc = cup) sit on the
  short-leg half, OUTSIDE the three grounded feet's support hull (the rocking
  pivot is the hull EDGE through the two adjacent foot-pad corners, ~21 mm past
  the leg-centre diagonal). Setting EITHER dish down on its seat pushes the
  combined table+dish CoM ≥ 1.25× past that edge (asserted): the table rocks
  until the short leg lands and the top pitches to **~7.2°**.
- The top is **polished** (slick material, pair-averaged with the equally slick
  dish bases): tan 7.2° ≥ 2.2× the friction coefficient (asserted at import), so
  a dish placed on the unrepaired table **accelerates off its seat and off the
  table**. Place-first is the losing move — the physics forces **SHIM FIRST,
  PLACE SECOND**.
- The repair: an orange **ramp wedge** (0.11 × 0.06 m, ramp 10 → 30 mm) lies on
  the floor. Slid **tip-first** into the gap under the hovering leg — a
  floor-level push, no grasp — its ramp takes the leg's load; the slope
  self-locks under load (tan 10.3° < μ budget, asserted). The table becomes a
  4-point stance and stays level (≤1°) under load.
- Success is **physical state only**: top level within 1.0°, plate settled on
  the red-ring seat and cup on the blue-disc seat (both judged in the TABLE body
  frame, xy AND resting z), everything still. No wedge clause is needed: with
  both seats loaded, a level top is only reachable with the leg shored (a wedge
  parked on the table as a counterweight is asserted insufficient, and the smoke
  battery probes it).

A seed-style solver (drop each item at its labeled spot) ends near score 0. A
correct solver needs a different plan (diagnose the hover gap, shim from the
floor, then place) and different code structure (a wedge push with a
level/height readback loop plus two placements, instead of a target list).

## Execution order (declared)

1. **Shim** — push the wedge tip-first under the hovering short leg until its
   ramp bears at the leg tip (0.30 latched).
2. **Plate** — set the plate on the red ring seat (0.20 latched).
3. **Cup** — set the cup on the pale-blue disc seat (0.20 latched).
4. Steps 2 and 3 may be swapped; step 1 must be first — the unshored table sheds
   any dish (enforced by physics, probed by smoke).

## Teleport-solution phases (solve.py)

Teleports are transport only; every load-bearing interaction is contact
dynamics under applied forces.

- **P0** settle 150 steps; read back layout (level top, 15–35 mm hover gap,
  score ≤ 0.02).
- **P1** transport: teleport the wedge onto the FLOOR in front of the hovering
  foot, tip aimed inward along the table-frame (+1,+1)/√2 diagonal (works at
  any random yaw). Assert not shored.
- **P2** contact: bang-bang **force** push (0.8 N escalating to 2.0 N cap,
  velocity servo at 0.05 m/s, frame-drag-aware encoding) slides the wedge into
  the gap. The cut target is **edge-based**: the 30 mm foot bears on its
  heel-side EDGE, so the stop is where the ramp height under that edge reaches
  stub − 2 mm; a bearing stall past x ≥ 0 is accepted (PhysX carries the load
  within the contact-offset zone). Settle, assert `shored()` and score ≥ 0.30.
  Deliberately does NOT lift the foot: lifting needs ~3.7 N vertical and the
  horizontal equivalent would skid the whole table (feet budget ~2.1 N).
- **P3/P4** transport + gravity: teleport plate (then cup) to hover 20 mm above
  its own seat **in the table frame** and let it drop; wait for the 30-step
  latch (cup phase also requires `success()`).
- **P5** hands-off persistence ≥ 3 sim-seconds (10 × 40 steps), re-asserting
  success and score 1.0, then print `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and never decreases
(0 → 0.30 → 0.50 → 1.0). Watchdog `threading.Timer` + `os._exit` bounds the
run. Verified on the forge for seeds 0 and 1.

## Embodiment argument (Franka)

A Franka on a fixed pedestal base at ~(0.55, 0.0, 0) facing the table cluster
reaches everything (all spawns within ~0.75 m, work height 0–0.17 m):

- **Wedge**: no grasp needed — a fingertip/knuckle **floor-level push** on the
  60 mm-wide heel wall slides it (≤2.0 N, well under Franka's capability);
  pushing is exactly what solve.py's force servo emulates. Alternatively the
  36 mm-tall heel fits the 80 mm jaw span for a pinch-lift.
- **Plate**: D60 × 14 mm, 0.30 kg stoneware — a top-down rim pinch
  (60 mm < 80 mm span); set down from a ~20 mm hover exactly as solve.py does.
- **Cup**: D45 × 70 mm, 0.25 kg upright ceramic tumbler — side pinch across the
  diameter (45 mm < 80 mm); at the fault angle it slides rather than tips
  (r/h_com = tan 32.7° ≫ tan 7.2°), same regime as the solve.
- Nothing requires exceeding ~4 N of applied force or reaching under the slab;
  the shim gap is approached from open floor.

## Checks (smoke.py rejection battery)

15 named checks, `SIM_GEN_SMOKE: ALL PASS 15/15`:

1. Initial settle: no NaN, hover gap 15–35 mm, top level, all still.
2. Score ≈ 0 / no success at spawn.
3. **Mass readback**: PhysX `get_masses()` matches every authored explicit
   mass within 2% (custom spawners silently fall back to density mass, which
   would void all the moment-margin statics).
4. Table randomization READBACK across seeds (xy spread > 8 mm, yaw > 40°).
5. Item-slot randomization READBACK (xy spreads > 10 mm, wedge yaw > 40°).
6. Null policy 240 steps: score stays ≈ 0.
7. **Seed-strategy end state**: both dishes dropped on their marked seats with
   NO shim — the table rocks > 3° (7.3° observed), the slick top sheds both
   dishes, no dish latch, score ≤ 0.02.
8. Wedge parked in FRONT of the gap (not inserted) + dishes → tips, no shore
   credit.
9. Wedge under the WRONG (long) leg + dishes → tips, no shore credit.
10. **Ballast cheat**: wedge parked on the far tabletop corner as a
    counterweight + dishes → still tips (moment margin is real).
11. Shim + **SWAPPED** dishes (plate on cup seat, cup on plate seat): level
    holds but no dish latch; score ≈ 0.30 only.
12. Shim + plate 50 mm OFF its seat (toward the centre): no plate credit.
13. **Load-bearing proof**: shim + plate seated = exactly 0.50, then yank the
    wedge back to the floor (table and plate woken by identity state re-writes
    — a settled stack sleeps and moving the wedge does not wake it) → table
    tips > 3° (7.2° observed — the wedge was carrying load) and sheds the
    plate; the latched 0.50 survives (latches are earned), success stays false.
14. Rejection audit: `success()` was never true at any step of any probe.
15. Final no-NaN.

Plus per-check settle discipline, camera frames recorded to `frames.npz`.

## Import-time feasibility asserts (scene.py)

10 assert groups pin the physics story: EACH dish alone pushes the combined
CoM ≥ 1.25× past the support-hull edge (single placement rocks the unshored
table); both dishes out-tip even the wedge-as-ballast ≥ 1.5×; tan(fault angle)
≥ 2.2× pair-averaged friction (tipped table sheds); tan(level tol) ≤ 0.5×
friction (level table keeps); wedge ramp self-locks under load; wedge tip
enters the gap; wedge too short to prop the slab; full insertion stays near
level; seats on-top, distinct, simultaneous-fit; plate/cup within the Franka
jaw span.
