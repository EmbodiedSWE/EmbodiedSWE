# base_i257 — CargoTramScene (`simgen.cargo_tram`)

Load a red cargo cube into an open-top cart on the flat loading zone, then push the
loaded cart up a slick 14° ramp, over a crest drop, into a roofed summit pocket and
seat it against the back wall. The roof's canopy leaves only a 31 mm slit over the
parked cart — a parked cart can never be loaded, so LOAD-then-CLIMB is the only
order physics admits.

## Seed provenance

Derived from **`pick_place/base`**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/base.py`): a direct
grasp-and-carry task — close the parallel jaw on a 4 cm cube and translate it through
free-space waypoints to a goal pose. The seed's core loop is *one grasp, one guided
transport*, and its judging is waypoint/goal-pose tracking on the held object.

## Strategic difference

**vs the seed (`pick_place/base`)** — different plan AND different code structure:

- *Plan*: the cube is never carried to the goal. It is dropped into a **vehicle**
  (an open-top cart) at the loading zone, and it is the *loaded cart* that is pushed
  — never lifted, never grasped — up a ramp too steep and slick to rest on
  (tan 14.04° = 0.25 vs pair friction 0.12), over an 8 mm crest drop, into a summit
  pocket whose **canopy** roofs the goal region. The goal pose for the cargo is only
  reachable *indirectly*, riding inside the cart; the free-space transport that IS
  the seed's whole task is physically blocked at the goal (canopy slit 31 mm <
  40 mm cube). Order is forced by geometry, not by the rubric: smoke check 6 parks
  the empty cart with the solve's own servo (parking needs no cargo), and check 7
  shows the parked cart cannot be loaded from anywhere above.
- *Code structure*: judging is a containment + vehicle-pose predicate stack — cargo
  containment in the **cart's body frame** (rides with the cart at any pose/tilt),
  cart park windows + upright cone in the env frame, settle gates on both bodies —
  with **latched partial credit** (`torch.maximum` in `post_step`: load / climb /
  dock latches, credit never evaporates), not gripper-distance or waypoint tracking.
  Physics legitimacy is asserted from the cfg's own numbers in `__post_init__`
  (canopy slit ∈ [10 mm, cube − 7.5 mm]; diagonal aperture at the canopy lip
  ≤ cube − 5 mm; crest-transit headroom; tan θ ≥ μ + 0.08; pocket/channel slack
  bands; park windows sit between the seated rest and the crest-perch rest).

**vs `base_i88` (BallastLiftScene)**: that task actuates a *lever mechanism*
(accumulate ballast until a hinged see-saw out-moments a ball, then push the
never-held ball off the raised tray). base_i257 has **no articulation at all** — the
mechanism is a fixed guideway (ramp + crest + roofed pocket) and the manipulated
"tool" is a free rigid **vehicle** the cargo rides in. There the target is never
contained by anything; here containment-while-riding IS the goal predicate, judged
in a moving body frame. No shared mechanism, predicate style, or ordering device
(there: moment ledger; here: aperture geometry).

**vs `pen_holder`** (exemplar read during construction): that task is insertion of
grasped objects into a static receptacle with pose-window judging. Here the
receptacle itself is the mobile object, the insertion is a gravity drop, and the
scored motion is pushing the receptacle through terrain it cannot rest on.

## Scene summary

Fully procedural (compound-box spawners; no external assets), env frame, ground
z = 0, channel along +x at y = 0:

- **Station** (one static compound): flat slab x ∈ [0.14, 0.445], top z = 0.020;
  ramp rise 0.08 / run 0.32 (θ = 14.04°, tan θ = 0.25) to the crest at
  (0.765, z = 0.100); summit **pocket** slab x ∈ [0.745, 0.885], floor z = 0.092
  (8 mm crest drop-in step); back wall at x = 0.863; side walls (inner faces
  y = ±0.054, top z = 0.213) flanking the whole channel; **canopy** roof
  x ∈ [0.788, 0.885] at z ∈ [0.213, 0.233], full channel width. Station surface
  slick: μ = 0.12.
- **Cart**: yellow open-top bucket, outer 92×92×90 mm, walls 8 mm, floor 10 mm
  (interior 76 mm square, cavity 14–90 mm above origin), mass 0.25 kg, CoM at the
  bottom center, μ = 0.12 (pair μ with the station 0.12 — cannot rest on the ramp;
  pair μ with a cube 0.36 — a cube CAN park on the ramp, so the slickness is the
  cart's, not the world's).
- **Cargo**: red 40 mm cube, 0.06 kg, μ = 0.6. **Decoy**: identical blue cube.
- **Pocket** is 6 mm longer than the cart: seated rest x = 0.817; canopy slit over
  the parked cart = 0.213 − (0.092 + 0.090) = **31 mm < 40 mm cube**.
- **Randomization** (readback-verified in smoke): cart start x ∼ U(0.21, 0.33);
  cargo and decoy on two distinct slots of a 4-slot grid (ordered pair permuted per
  seed) with ±25 mm xy jitter and free yaw.

`success()`: cargo inside the cart's bucket **in the cart's body frame**
(|x|,|y| < 0.034, z ∈ (0.014, 0.078)) AND decoy NOT in the bucket AND cart parked
(x ∈ [0.8125, 0.8235], |y| ≤ 0.02, z ∈ [0.084, 0.108], upright within 10°) AND cart
and cargo settled (|v| < 0.08). `score()`: latched 0.20 load (cargo ever in bucket)
+ 0.20 climb (loaded cart ever past x = 0.60 on the ramp) + 0.20 dock (loaded cart
ever in the pocket), cap 0.60 + 1.0 iff success.

## Solution phases (solve.py — teleport = transport only)

- **P0** settle 90 steps + layout readback; assert cart on the flat, score ≤ 0.02.
- **P1** LOAD: teleport the cargo to **free space** 5 mm above the cart's open
  bucket mouth (computed from the cart's *current* pose), release at zero velocity;
  **gravity + bucket-wall contact** make the insertion. Assert containment.
- **P2** CLIMB + SEAT (cart never teleported): velocity-regulated +x push,
  f = 80·(0.10 − vₓ) clamped to [0, 6] N, re-set every physics step in the cart's
  current link frame — stall force = the 6 N clamp ≫ the ~1.1 N the ramp demands,
  while cruise speed stays low (a constant 6 N would slam the back wall and bounce
  the cargo out; a low-gain servo would stall at the ramp foot). Across the flat, up
  the ramp (stop pushing mid-ramp and it slides back), over the crest, 8 mm drop
  into the pocket; then a gentle 1.5 N constant press seats the cart against the
  back wall (contact-terminated, self-aligning end stop) and releases.
- **P3** hands-off settle to `success()`.
- **P4** persistence: ≥ 3.3 simulated seconds hands-off with success still true.

`SIM_GEN_SCORE` printed at every phase boundary (observed staircase, seeds 0 and 1:
0 → 0.20 → 0.40 → 0.60 → 1.0 → 1.0 → 1.0, non-decreasing), then
`SIM_GEN_SOLVE: SUCCESS`. Passes on seeds 0 and 1.

## Franka embodiment argument

Single Franka arm, parallel jaw, OSC. Base pose ≈ **(0.50, −0.45, 0)** facing the
channel (+y-ish): every manipulation point lies at radius 0.25–0.55 m with margin.

- **Cargo cube** (the only grasped object): 40 mm edges — dead-center of the
  parallel jaw's 80 mm span. Spawn slots lie at radius 0.15–0.30 m from the base;
  standard top grasp on the open floor. The drop target is the cart's open bucket
  mouth at the loading zone — a 76 mm square aperture at height ≈ 0.11 m, radius
  ≤ 0.50 m, sky above it completely clear (the canopy is 0.35 m away) — a hover-
  and-release whose funnel tolerance is exactly what solve's gravity-drop
  demonstrates.
- **Cart**: pushed, never lifted. The push contact is the rear (−x) wall exterior,
  a 92×90 mm flat face at height 0.02–0.16 m as the cart travels x ∈ [0.26, 0.82] —
  a knuckle/closed-jaw push at radius 0.30–0.55 m along +x, with the corridor above
  the channel open (the canopy only covers the final 97 mm; during the seat-press
  the cart's rear wall is still 25 mm proud of the canopy lip at fingertip reach).
  The ~1.1 N ramp demand and 6 N ceiling are far below the arm's capability, and
  the channel's side walls (12 mm total slack) forgive lateral aim error — the
  guideway steers.
- **Decoy**: requires NO interaction — success demands only that it stay out of the
  bucket, and it spawns on the open floor.
- **Ordering is mechanically forced**: parking first roofs the bucket behind a
  31 mm slit that a 40 mm cube cannot pass in ANY orientation (smoke check 7 drops
  it from above — it rests ON the canopy — and at the canopy's front lip — it is
  refused); carrying the cube to the summit by hand scores nothing (check 8:
  containment is judged in the cart's body frame).

**Execution order declaration**: LOAD (cargo → bucket) strictly before CLIMB/SEAT
(cart → pocket); the decoy is never touched; P2–P4 are a single forced sequence.

## Checks (smoke.py — 16)

1. Settle + no-NaN + layout sane, 2. null score ≈ 0 at rest, 3. randomization
readback (4 distinct ordered (cargo, decoy) slot pairs across seeds, xy jitter
spreads), 4. free-yaw spread + cart start-x spread, 5. null policy 240 steps scores
≤ 0.02, 6. order gate 1 — the solve's own servo parks the EMPTY cart (non-vacuous:
driven 60 cm, seated readback) yet earns nothing, 7. order gate 2 — with the cart
parked, cargo dropped from above rests ON the canopy and dropped at the canopy lip
is refused: a parked cart cannot be loaded, 8. seed-strategy reject — cargo carried
straight to the summit pocket floor counts for nothing, 9. roof dump reject,
10. bridge near-miss — the LOADED cart perched tail-on-crest (tilt 5°) passes the z
window and upright cone but the park x window rejects it, 11. identity — the blue
decoy inside the parked cart's bucket counts for nothing, 12. exclusivity — cargo
stacked on top of the in-bucket decoy earns every partial latch (0.60) but success
stays False, 13. latched credit survives removing both cubes, 14. settle gate — all
position windows pass at 0.24 m/s but success is False, 15. rejection audit
(success never true at any judged point), 16. final no-NaN. Camera frames recorded
to `frames.npz`. Prints `SIM_GEN_SMOKE: ALL PASS 16/16`.
