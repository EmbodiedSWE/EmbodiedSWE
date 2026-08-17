# track_knife_i402 — Build the Knife Stand (`simgen.knife_stand`)

## Seed provenance

Derived from **pick_place/track_knife**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_knife.py`): a Stage-3
trajectory-tracking task — the knife starts ALREADY RIGIDLY GRASPED in the Franka's
closed gripper, and reward is dense tracking of a prescribed free-space waypoint path
(position + orientation gates along the path). The seed's entire strategy is *moving a
held knife through given waypoints to a destination that already exists*; there is no
construction, no contact event, no ordering constraint.

## What changed, and why it is strategically different

Kept only the protagonist (a knife that must end up resting somewhere specific). The
seed's strategy is inverted: **the knife's destination does not exist at reset — it
must be BUILT first.**

- **The goal is a construction, not a location.** Two 160 × 8 × 120 mm panels form a
  cross-lap knife stand: panel A stands upright (top-down half-slot), panel B lies flat
  on a rack. B must be picked up, rotated 90° in yaw, aligned over A, and **pressed
  down through the blind 16 mm cross-slot** until the two half-slots fully mesh (root
  z < 5 mm) — a contact-rich press-fit insertion with ±4 mm play, the opposite of
  free-space waypoint tracking.
- **The knife has no legal rest until the stand exists.** The rubric's knife gate
  demands the blade LEVEL (±8°), inside a z band [100, 117] mm, centred on the meshed
  cross and yaw-aligned with the notch diagonal. Only the two 44 mm V-notches of a
  *meshed* stand provide that support: a lone upright panel offers an 8 mm knife-edge
  (smoke 7 shows the blade does not latch there), the rim diagonal has no notches
  (smoke 11), and the floor/rack are 100 mm too low. The knife-credit latch is
  additionally **gated on live `meshed()`** — physics AND rubric both force
  build-before-place.
- **Waypoint transport of the knife earns nothing.** Smoke check 6 *constructs* the
  seed's strategy — carries the knife level through a 9-waypoint sinusoidal arc with
  per-waypoint judging — and scores ≤ 0.02.
- **Versus the read corpus:** `track_banana_i79` frees a captive object by actuating a
  spring mechanism above a staged catcher — release/catch, no assembly.
  `repair_first_missing_rail` installs a crossbar into an *existing* fixture;
  `plank_bridge` lays a plank across a gap. No task in tasks_v4–v7 mates two free
  bodies by a blind press-fit joint (complementary half-slots meshing under load) and
  then uses the assembly as the goal surface. The press-fit servo, the
  live-meshed-gated knife latch, and the notch-diagonal seat are new.

## Apparatus (fully procedural, no meshes)

- **Panel A** (1.2 kg, upright): solid lower half + upper half with a 16 mm top-down
  centre half-slot and two 44 mm-wide, 12 mm-deep V-notches at local x = +45 mm; two
  160 × 110 × 10 mm foot pads with a 24 mm centre channel (the seated cap slab passes
  through — asserted `foot_gap/2 > slot_w/2 + 2 mm`). Slab faces carry a slick
  material (0.06/0.05/0) so meshing is geometric, not frictional; feet keep default
  friction so the stand does not skate.
- **Panel B** (0.28 kg, the cap): mirror piece — 16 mm bottom-up half-slot + notched
  solid top; starts lying FLAT on a kinematic rack 300 mm away. Aligned descent is
  contact-free until full seat (complementary volumes verified analytically in
  `__post_init__`).
- **Knife** (70 g): 140 × 20 × 5 mm blade + offset grip, lying on a kinematic pedestal
  block on the opposite side. Seated rest z = 110.5 mm (inside the judged band,
  asserted).
- Panels carry an authored CoM 20 mm up from the bottom edge: with the CoM on the
  slab boundary no flat rest is statically stable (the support hull cannot extend
  past the CoM edge) — the inset puts the racked cap's CoM 10 mm inside the rail
  hull while keeping the press's discrete damping stable (c·dt/I = 0.79 < 1).
- Config honesty asserted in `__post_init__`: slot play, notch-vs-blade crossing at
  worst yaw, foot-channel clearance, rest heights, judged band brackets the analytic
  seat, lying-CoM-inside-the-rail-hull, slab spans both rails.

**Randomization (readback-verified):** panel A xy ±3.5 cm + free yaw; rack xy ±3 cm +
yaw ±15° with B slid ±1 cm along it; pedestal xy ±3 cm; knife yaw ±0.25 rad + jitter.

## Rubric

- `success()` = half-slots fully meshed (B root z < 5 mm, xy ≤ 12 mm, relative yaw
  90° ± 20°, both upright ≤ 12°) ∧ knife seated (level ± 8°, z ∈ [100, 117] mm,
  centred ≤ 15 mm, yaw on the notch diagonal ± 20°) ∧ everything still.
- `score()` (monotonic, latched): 0.20 engage (B aligned in the slot mouth, z < 55 mm)
  + 0.25 mesh + 0.30 knife-seated — the knife latch fires only while `meshed()` is
  LIVE-true — capped at 0.75; exactly 1.0 iff success().

## Solution outline (solve.py — the legitimacy certificate)

Teleport = TRANSPORT ONLY: pose writes only carry a body across open floor/air to a
hover that satisfies no rubric gate (asserted at each). Everything load-bearing is a
body-frame PD wrench servo (`held_move`) with live targets, symmetric fz clamp,
discrete-damping-stable gains (c·dt/I < 1 on every controlled axis), stall-gated
wiggle, and early aborts:

- **P0** settle; **P1** carry B to hover 132 mm over A's live xy at ayaw+90°
  (not engaged — asserted).
- **P2** press-fit: servo B straight down through the blind cross-slot against
  contact (press ≤ 2–4 N, 3 gain-escalating attempts with release/re-place retry)
  until root z < 5 mm → engage 0.20, mesh 0.45.
- **P3** carry the knife to a hover over the live notch midpoint (yaw = live notch
  diagonal, nearest mod-π branch); **P4** lower it onto the V-notches (k_up = 0 —
  knife roll inertia makes any explicit roll damping discrete-unstable) until it
  rests at z ≈ 110 mm → 1.0.
- **P5** hands-off persistence 3.4 s → `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge for seeds **0** (11.3 s) and **1** (11.2 s) against the final
submitted files, scores non-decreasing 0 → 0.45 → 1.0; the press descends
dead-straight (up_z = 1.000, exy = 0.0 mm throughout) and meshes in 782 steps on
both seeds; the knife seats at z ≈ 0.110 (analytic rest 0.1105).

## Embodiment argument (Franka, one base pose)

Base at the origin facing +x. Farthest work point is panel A's slot mouth with worst
jitter (~0.53 m at z 0.13) — inside the 0.855 m reach envelope; rack and pedestal sit
at 0.30–0.35 m. Per-object contact strategy:

- **Panel B (the press):** lies on the raised rack, long edge proud — the 8 mm slab
  thickness is a native parallel-jaw pinch (80 mm stroke). The press needs ≤ 4 N
  straight down (demonstrated), far under the arm's payload; grip on the solid top
  half keeps fingers clear of the slot region.
- **Knife:** 5 mm blade lies on the raised pedestal with the grip overhanging — a
  standard top pinch. Seating force ≤ 0.8 N press.
- **Panel A:** furniture during the solve — never needs to be moved (feet keep it
  planted; the press is vertical so lateral shove is minimal, and A's pose is tracked
  LIVE anyway).
- **Rack/pedestal:** kinematic furniture; no contact required.

## Execution-order declaration

The physics and rubric enforce the only order that matters: the stand MUST be meshed
before the knife can rest at judged height (a lone panel offers only an 8 mm edge —
smoke 7; the knife latch is gated on live `meshed()`), and B must engage the slot
mouth before it can mesh. The knife may be staged nearby at any time; solve.py builds
first, then seats.

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 16/16` on forge)

1. Settle/no-NaN: A upright, B flat on the rack, knife flat on the pedestal, still.
2. Score ~0 at reset, no success.
3. Randomization readback: panel A xy + free yaw vary within the declared band
   (3-seed max-pairwise).
4. Randomization readback: rack/B xy, rack yaw, pedestal xy, knife yaw all vary.
5. Null policy (240 steps): score ~0, no success.
6. **Seed-strategy analog:** the knife pose-carried level through a 9-waypoint
   sinusoidal arc, judged at every waypoint, then set down — score ≤ 0.02.
7. Lone-panel perch: the knife laid level at seat height over the UNMESHED panel A's
   notch — not seated (the live notch line needs the cap; the cap is still racked),
   no latch.
8. Beside-park: B upright 6 cm beside A — not engaged, not meshed.
9. Half-press: B aligned in the slot at z = 40 mm — engage latches (0.20), mesh
   refused.
10. Built stand without the knife: mesh latches (0.45), empty stand is not success.
11. Rim-perch: knife on the un-notched rim diagonal at rim height — not seated,
    score stays 0.45.
12. Skew half-perch at 10.7° pitch — the flat clause refuses.
13. z band + settle gate: rim-height knife refused by the band; the same knife at
    seat height sliding 0.4 m/s latches seat credit (0.75) but success is refused
    by the settle gate.
14. Latched credit survives scattering the knife to the floor and yanking B out.
15. Rejection audit: success() never True at any judged point in the battery.
16. Final no-NaN; frames.npz saved.
