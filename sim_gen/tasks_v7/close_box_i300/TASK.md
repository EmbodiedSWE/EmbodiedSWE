# close_box_i300 — Groove-Lid Chest

**Scene name:** `groove_lid_chest` · **Env:** `simgen.groove_lid_chest` (robot="null")

## Seed provenance

Seed task: `rlbench/close_box` — "close the box": an articulated box USD stands with
its hinged lid open; the robot pushes the lid shut about its BUILT hinge, judged by
the lid joint angle (`JointPosChecker`, mode "le"). One pushing contact on a hinged
panel that is already part of the box, ending in a joint readout.

## Strategic difference

The seed's closure model is replaced wholesale: **there is no joint anywhere in the
scene** — no hinge, no built lid, no joint angle to read (the seed's literal end
state is inexpressible here; its naive transfer — bring a lid DOWN over the mouth —
is demonstrated dead in smoke check 5). The chest's lid is a **separate free body**
(a thin blue plate with a red grasp knob) lying elsewhere on the floor, and the chest
closes like a pencil box: two C-shaped side grooves (shelf below, overhanging flange
above) plus an open loading **porch** with flared guide walls at one end. Closure
must be **manufactured in three physically forced stages**: (1) fetch the plate and
set it down on the porch — the only entry, because the flanges make the grooves
unreachable from above (the plate dropped flat over the mouth rests ON the flanges,
~22 mm above the channel plane — asserted geometrically in `__post_init__` and
demonstrated in smoke); (2) push it so its leading edge threads into BOTH grooves;
(3) slide it captive down the channel to the far stop so it covers the mouth, with
the gold cube (the contents) still inside underneath. Once home, the lid is held
captive by the flanges (a 3× weight upward raid lifts it < 12 mm — smoke check 9).

Differs from every corpus task read: `close_box_i26` *inverts a free crate over a
block* and slides the trap (closure = capture by inversion; lid IS the box);
`open_box_i184` *removes* a gravity-seated roller via a threshold roll (no free lid,
no grooves); `close_microwave_i4` releases a prop so gravity shuts a hinged gate;
`close_laptop_lid_i152` loads ballast onto a hinged panel; the libero/rlbench
pick-place families transport freely graspable objects to open goals. **No corpus
task manufactures a closure by threading a separate free lid through a
geometry-gated side channel (porch → grooves → stop) with contents retained under
it** — and none has a flange-captive sliding lid at all.

## Assets (fully procedural)

- **chest** — KINEMATIC compound (184×160 mm footprint, 92 mm at the flanges, 102 mm
  end stop): floor plate, −x full-height end STOP, +x entry wall flush with the shelf
  plane, and per side a stacked C-groove (low wall → shelf bar protruding inward to
  y=58 mm, top 70 mm → slot outer wall (inner face 70 mm) → flange bar overhanging
  back to 58 mm, z 84..92 mm). The PORCH extends the channel to x=210 mm: two rail
  columns continuing the shelf plane + two FLARED guide walls (rotated slabs,
  70→82 mm half-opening) that funnel a drop into the channel. Slick (μ≈0.15).
- **lid plate** — DYNAMIC compound: 190×132×8 mm blue slab + 24×24×25 mm red knob
  offset toward the trailing edge (rides the open centre strip between the flanges,
  clears the stop in both orientations — asserted). 0.18 kg, slick.
- **cube** — DYNAMIC 40 mm gold cube, 0.05 kg, spawns inside the cavity.

Randomization (readback-verified, smoke check 2): chest yaw 180°±60° + xy ±20 mm,
plate side swap (±y, both sides hit over 10 resets — check 3) + xy ±20 mm + free
yaw, cube chest-frame xy ±12 mm.

Honesty geometry asserted in `__post_init__`: slot admits the plate with real
clearance yet the plate can never pass between the flanges from above (worst-case
overlap stays positive); covered window reachable from the hard stop and far smaller
than the cube; porch supports the dropped plate past its CoM; guides admit a
misplaced drop; knob clears flanges and stop; spawn separation under all jitters.

## Rubric

Latched partial credit (never evaporates), finite-guarded:

- 0.15 `seated` — plate ever RESTS in the channel plane (porch rails or shelves:
  centred, shelf height, aligned, still) — a floor plate lies ~70 mm below it
- 0.20 `engaged` — leading edge ever ≥ 25 mm past the slot mouths while in the
  channel bands
- 0.25 `deep` — engagement ever ≥ 100 mm (past halfway)
- non-success capped at 0.75; **1.0 iff success()**: plate in-channel and slid home
  (uncovered strip ≤ 6 mm), gold cube inside the cavity BELOW the lid plane,
  everything settled and finite — all judged live.

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset, 120-step settle: plate on the floor, chest open, cube inside,
  score 0.
- **P1 (seat — gravity)** one root-state write carries the plate to a free-space
  hover 30 mm above the porch (aligned, knob up); it FALLS between the guide walls
  onto the porch rails and settles. The seat is made by ballistics + contact, never
  written. Score 0.15.
- **P2 (thread + slide — applied force + contact)** a velocity-regulated horizontal
  force at the plate CoM (chest-local −x, ~0.10 m/s, cap 4 N, stiction floor with
  stall escalation, force-frame mode probed from progress) threads the leading edge
  into both grooves and drives the plate captive to the end stop; a 1.2 N press
  squares it, forces cleared, hands-off settle. Score 1.0.
- **P3** ≥ 3.3 simulated seconds hands-off; success persists → `SIM_GEN_SOLVE:
  SUCCESS`.

Teleports are transport-only; every rubric fact (seated/engaged/deep/covered) is
produced by gravity, contact and the pushed slide.

## Execution order

Seat-then-thread-then-slide is **physically forced**, not declared: the flanges make
the grooves unreachable from above (assert + smoke check 5), so the porch seat must
precede engagement, and `covered` lives at the end of the only channel. Within that,
which plate end leads and when the push pauses are free; the cube is simply never
touched.

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing +x; chest at (0.50, 0) with the porch
nominally toward the robot (yaw 180°±60°), plate at (0.28, ±0.32), everything at
reach 0.28–0.55 m.

- **Fetch (pinch the knob):** the plate is an 8 mm-thin slab — ungraspable flat on
  the floor — but carries a 24×24×25 mm knob: a standard top pinch (Franka jaw opens
  80 mm ≫ 24 mm) at ~0.42 m reach. Free yaw at spawn only changes wrist roll.
- **Seat (place on the porch):** the porch drop window between the guide tips is
  164 mm for the 132 mm plate (±14 mm lateral slack, funnelled tighter toward the
  mouth); set-down height 74 mm. The knob-up carry is the natural pinch attitude.
- **Thread + slide (push the knob):** the knob rides the open centre strip between
  the flanges for the whole travel, so the closed fingertips push its face
  horizontally at ~103 mm height, ≤ 4–5 N at quasi-static speed — trivially inside
  the Franka force envelope; the flared guides and groove walls do the aiming. The
  final press is the same contact.
- No step needs a second arm, regrasping through the flanges, or any contact with
  the gold cube.

## Files

- `scene.py` — cfg (+ flange-seal/coverage/porch-support/knob-clearance asserts in
  `__post_init__`), compound chest + lid spawners, scene (latched rubric in
  `post_step`), `register_env`.
- `solve.py` — phased teleport solution; watchdog + hard exit.
- `smoke.py` — 14-check rejection battery (settle, randomization readback, side
  swap, null policy, seed-strategy top drop, mid-channel near-miss, porch seat,
  contents-removed closure, captive raid, cube-on-lid, settle gate,
  success-never-True audit, score-cap audit, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 20.6 s), score trace
  0.00 → 0.15 → 1.00 → 1.00
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 20.5 s), same trace
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 14/14` (rc=0, 46.5 s)
