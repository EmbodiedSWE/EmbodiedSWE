# bowl_letterbox — turn the bowl on edge and post it through the coin slot

`sim_gen` task `libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i212`
— env `simgen.bowl_letterbox`, scene `bowl_letterbox`, robot slot `null` (scene-level task).

## Seed provenance

Derived from RoboVerse `libero_90/kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet`
(`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.py`):
open the cabinet's sliding bottom drawer, pick the akita black bowl off the table, and
lower it in from above; `_terminated` is a single bowl-centre-inside-the-drawer OBB
test (the drawer bbox displaced by the drawer joint), with a wine bottle and wine rack
as inert distractors.

## What changed, and why it is strategically different

The scene keeps the seed's core relation — the black bowl must end up inside the
cabinet's bottom compartment — but **removes the mechanism entirely and replaces it
with a shape-selective aperture**. The receptacle can never be opened: both stacked
compartments are permanently sealed boxes whose ONLY access is a narrow **vertical
70 mm slot** in the front face, like an oversized coin slot. The black bowl is a flat
octagonal dish 130 mm across but only 48 mm thick: carried flat (the seed's carry
pose — and the smoke battery drives exactly this pose at the slot with the solve's
own bounded force to prove it jams) it cannot enter in principle; **turned on edge
like a coin** it passes with ~2 cm of clearance. The plan the seed's demo encodes
(actuate drawer → top-down place) is physically unavailable; the required plan is
pick → **reorient 90°** → stage on the apron shelf → **slide/roll through the
aperture** → it topples flat inside.

Structural differences from the seed (and from every task I read):

- **No articulation at all**: the cabinet is one kinematic compound; there is no
  joint to actuate, unlike the seed's prismatic drawer, i134's force-opened slide,
  i237's bistable hinged bin, or i102's drawer re-installation. The interlock is
  purely **metric** (slot 70 mm vs dish 130 flat / 48 on edge vs bottle 88 mm),
  asserted at cfg build time (honesty by construction: containment ⇒ passage
  through the slot, since every other face is sealed).
- **Judged relation**: containment reached only via a **reorientation + guided
  passage** — the rubric's middle milestone latches "on edge near the mouth", a pose
  requirement no sibling has; the insertion term is a gated cabinet-frame progress
  ramp through the aperture band, not a bbox drop test.
- **Decoys**: an identical dark-framed slot into the sealed TOP compartment (wrong
  floor — full containment there scores no insertion credit), and a fat red bottle
  (88 mm > slot in every orientation) as the wrong object.
- Not a variant of the sibling tasks: no contents transfer / pouring (i134
  `drawer_bead_pour`), no 6-DOF re-mating of a freed drawer (i102 `drawer_refit`),
  no bistable closure goal (i237 `tilt_bin_stow`), and the pen-holder exemplar fills
  a container top-down — here the container is entered *sideways through a wall*.

All assets are procedural compound spawners (no external files): kinematic shell
(base block, cantilevered apron, side/back walls, divider, roof, per-cell front
strips leaving the centred vertical slots — white frame below, dark above), dynamic
octagonal black dish (8 wall boxes + floor disc, sleep/stabilization zeroed,
solver-vel iterations 4), dynamic red bottle (cylinder + neck).

**Randomization** (readback-verified in smoke): cabinet yaw ±12° + lateral jitter
±5 cm, Bernoulli LEFT/RIGHT bowl side swap + ±3 cm xy jitter + free bowl yaw, bottle
mirrored on the opposite side. All rubric geometry is computed in the cabinet's body
frame, so the yawed cabinet judges identically.

## Teleport-solution outline (solve.py)

Teleports are transport-only; the load-bearing interaction is contact dynamics:

1. **Pick + reorient** — the bowl is lifted and rotated 90° about the cabinet's
   x-axis (per-step pose writes with path-consistent velocities; the wrist-roll
   emulation), from flat to on-edge.
2. **Stage** — carried around the front and lowered onto the apron shelf centred on
   the lower slot, then RELEASED: it stands on an octagon flat on its own; no pose
   write ever touches the bowl again.
3. **Post through the slot** — a bounded (≤ 10 N) horizontal servo force along the
   cabinet's +x axis (the applied-wrench emulation of pushing the trailing edge;
   pre-encoded per step into the body frame — the wrench API's default frame — so
   the push stays world-correct even while the bowl rolls) slides the standing dish
   across the apron, through the 70 mm slot, into the bottom compartment; the drive
   is cut and the bowl settles inside under gravity/contacts alone.
4. **Persistence** — success() first turns True on the settled state; the solve then
   holds hands-off ≥ 3.3 sim-seconds before `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` prints at every phase boundary and is asserted non-decreasing
(latched credit). Verified on the forge: seeds 0 and 1, `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, base at the world origin)

Every load-bearing interaction is a standard Franka contact strategy, all within a
0.25–0.60 m reach annulus of one base pose at the origin:

- **Pick + reorient**: the dish's 8 mm octagonal wall is a parallel-jaw pinch target
  (opening ≥ 8 mm + clearance); turning it on edge is a single 90° wrist roll with
  the arm nearly static — the classic reorientation primitive. Dish mass 0.15 kg.
- **Stage**: placing the on-edge dish on the apron (top at 12.2 cm, front edge at
  x ≈ 0.25) is a plain place at ~0.3–0.4 m reach; the octagon flat (54 × 48 mm
  footprint) makes the on-edge stand statically stable, so the gripper can let go.
- **Push through**: the final stroke is a straight horizontal push of ≤ 10 N over
  ~25 cm at 19 cm height — the closed-fingertip pair (~20 mm wide) fits the 70 mm
  slot with room to spare, so the last centimetres can be pushed *through* the slot
  mouth, or the dish can simply be given a firm shove and rolled in. Well inside
  Franka payload and workspace; no interaction needs a second gripper or exotic
  contact.

## Execution order

1. `scene.py` — scene, spawners, sealed-slot geometry, rubric (written first).
2. `solve.py` — iterated on the forge until SUCCESS (seeds 0, 1).
3. `smoke.py` — rejection battery, iterated on the forge until ALL PASS.
4. `TASK.md` — this file.

## Check list (smoke.py, 13 checks)

1. Settle: states finite, bowl FLAT on the floor in front, bottle upright, all still.
2. Score ~0 at reset, no success.
3. Randomization readback: bowl side flips AND cabinet yaw varies across seeds.
4. Randomization readback: per-slot xy jitter > 4 mm (cabinet-frame).
5. Null policy (240 idle steps): score ~0, no success.
6. **Seed-pose physical probe**: the FLAT bowl driven at the lower slot with the
   bounded push advances, JAMS at the front wall, never enters — insertion latch ~0,
   NOT success, score ≤ 0.5 (the reorientation requirement is load-bearing
   physically; the probe asserts the actuator really moved the bowl).
7. Top-cell decoy: bowl settled fully inside the UPPER compartment — zero insertion
   credit (z-gate), NOT success.
8. Straddle near-miss: on-edge bowl resting half-through the slot — strictly partial
   insertion credit, NOT success, score ≤ 0.85.
9. Latched credit: teleporting the straddling bowl back out leaves the score
   unchanged.
10. Roof: bowl flat on top of the cabinet — score ~0, no success.
11. Wrong object: red bottle parked on the apron against the slot — score ~0.
12. Rejection audit: success() never True at any judged point in the battery.
13. Final no-NaN on all task objects.
