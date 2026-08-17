# pick_i330 — Chock the Ramp (a rest pose that must be BUILT before it exists)

## Seed provenance

- **Seed**: `mujoco_playground/pick` —
  `sim_gen/RoboVerse/roboverse_pack/tasks/mujoco_playground/pick.py`.
- The seed is "bring the box to the target": a Franka grasps a 4 cm cube and
  carries it through free space to a floating target pose. The target pose is
  trivially STABLE — wherever the hand releases the box, it stays; the whole
  plan is grasp + guided carry + place, with reward shaping gripper→box then
  box→target.

## What changed and WHY it is strategically different

The kept DNA: one small cube must end up at a designated, per-episode-randomized
target region. Everything about *how* is inverted:

1. **The goal pose DOES NOT EXIST until the solver builds it.** The target band
   lies on a slick 25° ramp face where nothing rests unsupported: every
   object/face friction pair has a friction angle ≥ 6° below the slope
   (asserted in cfg). The seed's one and only primitive — carry to the target
   and let go — executed literally at the goal pose, ends with the cube on the
   floor past the ramp foot (smoke check 4 runs exactly this). The task is to
   **construct the support first**: key the blue CHOCK's underside tab into the
   recessed pocket just downhill of the band, and only then does "place near
   the target" mean anything.
2. **The last leg is delivered BY THE ENVIRONMENT, not by the hand.** The cube
   is laid on the face *outside* the band (uphill) and released; the slick
   slope transports it and the seated chock's plate arrests it *inside* the
   band. In the seed the hand guides the object all the way; here the terminal
   approach is gravity + a fixture the solver installed. (solve.py's P2 applies
   *zero* force to the cube — release is the entire action.)
3. **Execution order is forced by PHYSICS, and a spatial binding decides
   credit.** Cube-first fails by gravity (nothing to rest on); and there are
   THREE identical pockets 90 mm apart, only one of which — the one just
   downhill of the randomized yellow band — parks the cube in band. A
   physically stable, visually identical park at a wrong station scores ~0
   (smoke check 6 constructs it in full). The seed has no ordering and no
   station choice.

Distinct from every other task read this session: `pick_i137` (Gravity Vault)
is interlock-extraction + a vertical funnel drop into a sealed vault;
`pick_i269` (Relay Tunnel) is object-as-tool relay pushing under a sealed roof
into a sunken well; `pick_and_lift_i16` (Ballast the Lever) is a counterweight
seesaw with mass decoys. Here the core is *fixture installation on a surface
with no unsupported rest states* — a keyed anti-slide chock — followed by a
gravity-transported capture; no roofed pushing, no funnels, no levers, no
second-object force relay (the chock transmits nothing; it statically arrests).

## Scene (procedural, no external assets)

- **Ramp** (one KINEMATIC compound; SLOPE frame: origin = face centre, +x
  upslope, +z face normal; world pose ≈ (0.46, 0) at 25°, jittered): slick
  0.48 × 0.24 m face (μ 0.10) made of a 12 mm top layer over a dark base, with
  three pocket cutouts (24 × 52 mm, 12 mm deep) on the centreline at
  x ∈ {−0.075, 0.015, 0.105}; cosmetic pedestal underneath.
- **Chock** (dynamic compound, custom spawner, 0.15 kg, μ 0.40; origin =
  plate-bottom/tab-top centre): upright plate 14 × 60 × 45 mm, downhill foot
  30 × 60 × 8 mm, underside tab 14 × 40 × 11 mm. Tab keys into any pocket with
  real play (10 mm along, 12 mm across); seated, the plate lands flush
  (tab 11 mm < pocket 12 mm), so the seated z-window (−4..+5 mm) cleanly
  rejects the ~11 mm-proud unkeyed rest.
- **Cube** (RED, 50 mm, 0.10 kg, μ 0.40, restitution 0) and a **yellow band
  marker** (thin plate, kinematic, **collision disabled** — visual only,
  re-posed per episode onto the starred band).
- Bands: b_k = pocket_x + 0.032 (= plate_t/2 + cube_s/2, asserted), half-width
  30 mm; stations 90 mm apart so wrong-station rests are far out of band.
- Honesty asserts in `__post_init__`: slick-face friction angles; tab/pocket
  play and flush seat; cube cannot enter a pocket; ≥ 2× anti-tip moment margin;
  ≥ 2× impact-capture energy margin (pocket wall blocks pitch-out until the tab
  lifts a full tab_h); plate faces most of the cube; band separation; release
  span on-face; spawn slots clear the foot.
- Randomized per episode (all `torch.rand`/`torch.randint`, burn draws first;
  smoke reads all of it back): ramp xy ±30 mm + yaw ±15°; starred station
  k* ∈ {0,1,2} (marker follows); chock/cube floor slots (ramp-yaw frame
  (−0.32, ±0.18)) side-swapped, ±30 mm jitter, free yaw.

## Rubric (latched partial credit, anchored in the solve trajectory)

- 0.375 `seated` — chock EVER keyed in the STARRED pocket (flush z-window,
  aligned, slow) (latched)
- 0.375 `parked` — cube EVER braced against the seated chock inside the band,
  slow (latched; parked ⇒ seated by chaining)
- 1.0 iff `success()`: braced in band, finite, AND still for ≥ 12 consecutive
  physics steps (stillness STREAK ticked only in `post_step` — smoke constructs
  braced-but-moving states and dismantles them before the streak completes).
  Non-success capped at 0.75 (0.375 + 0.375, float32-exact; smoke asserts the
  cap to 1e-6). Null policy ~0.

## Teleport solution (solve.py) — phases

- **P0** settle + readback (ramp pose/yaw, k*, marker-on-band, masses via
  `get_masses()`, score ~0).
- **P1** *(transport, then gravity + a fingertip press)* the chock is lowered
  tab-into-the-open-socket over the starred pocket (plate bottom 4 mm above the
  face — the arm's ordinary insertion approach; the socket is open from above,
  no barrier crossed) and RELEASED: gravity drops it flush and slides it until
  the tab keys against the pocket's downhill wall; a 2 N press along the
  downhill axis confirms the key. `seated` → SCORE 0.375.
- **P2** *(transport, then pure gravity)* the cube is laid on the face 6 mm
  OUTSIDE the band's uphill edge (asserted) and released; it slides down and is
  arrested by the plate inside the band. No force ever touches the cube.
  `parked` → SCORE 0.75.
- **P3** settle → live `success()` (streak ≥ 12) → `SIM_GEN_SCORE 1.0000`.
- **P4** hands-off ≥ 3.3 simulated seconds → `SIM_GEN_SOLVE: SUCCESS`.

No teleport creates the goal state: the chock staging hovers above the face
with the tab in the socket *mouth* (seating happens by contact dynamics), and
the cube staging is outside the band on a face it cannot rest on.

## Embodiment argument (Franka, base at world origin; ramp face centre at (0.46, 0) ± jitter)

- **Chock**: grasped across the 14 mm plate (jaw span 80 mm) from its lying
  pose on the open floor 0.25–0.40 m from the base; the socket has 10–12 mm of
  play, so the insertion is a compliant lower-and-release at z ≤ 0.10 m —
  well inside positioning tolerance; the confirming press is 2 N, fingertip
  scale.
- **Cube**: 50 mm pinch grasp; the release pose is a flat set-down on the face
  at slope-local z = 29 mm — for the highest band that is ~0.62 m from the
  base at world z ≈ 0.19 m, inside the Franka workspace with margin.
- Nothing requires reaching under, through, or into anything: the face and all
  three sockets are open from above; all interaction points lie 0.25–0.65 m
  from the base at world z ≤ 0.22 m.

## Execution order

Declared: FORCED order, chock first (describe()/instruction() say so). The
order is enforced by gravity, not the rubric: with no seated chock the face has
no rest states, so any cube-first attempt ends on the floor (smoke check 4);
and the chock must go to the *starred* station — the same park built at either
other station scores ~0 (smoke check 6).

## Checks

- solve: `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1 and 2 (forge), scores
  non-decreasing 0.00 → 0.375 → 0.75 → 1.00.
- smoke: `SIM_GEN_SMOKE: ALL PASS 10/10` (forge) — settle/no-NaN + mass +
  floor-clear readback; randomization readback (ramp/spawns pairwise over seeds
  0–2; k* ≥ 2 values + marker-tracks-band + slot swap over seeds 0–7); null
  policy; slick-face/seed-strategy rejection (cube set ON the goal band slides
  to the floor, marker arrests nothing); unkeyed-chock rejection (proud
  z-readback, slides away); wrong-station full park (physically stable,
  readback-verified, scores ~0); downhill release (seated credit stays exactly
  0.375, no leak); settle-gate + exact 0.75 cap (braced-but-moving latches,
  success refuses, dismantled before the streak); rejection audit (success
  never True in the battery); frames.npz video.
