# put_groceries_in_cupboard_i169 — CarouselCupboard

**Scene name:** `carousel_cupboard` · **Env:** `simgen.carousel_cupboard` (robot="null")

## Seed provenance

Seed task: `rlbench/put_groceries_in_cupboard` — "put the groceries in the
cupboard": named grocery items among lookalike distractors stand on a table; the
robot grasps the named one and SETS IT DOWN on a passive open cupboard shelf.
Any resting pose on the shelf surface, reached by lowering from above, is
terminal; the cupboard itself is inert scenery.

## Strategic difference

The seed's entire plan — pick the grocery and set it down on a shelf — is
**insufficient here by construction**, not re-parameterized: there is no passive
shelf. The cupboard is a fully roofed and walled round cabinet with ONE front
window, and its only storage is a **mechanism**: a free-pivot 3-bay CAROUSEL
indexed from outside by a crank KNOB on the roof. Two bays already hold decoy
cans; the EMPTY bay starts rotated 75–170° away from the window. Storing the
red carton demands an interleaved plan the seed never needs:

1. **index** the carousel by the knob until the empty bay faces the window,
2. **insert** the carton horizontally through the window onto the carousel
   floor of that bay (the seed's vertical set-down is geometrically impossible:
   the roof is closed, the axle hole passes nothing, wall/divider gaps are
   sub-carton — all asserted numerically in `__post_init__`),
3. **index again** ≥ 100° so the loaded bay is hidden behind the wall, the
   carton riding the turntable on friction, upright, decoys still seated.

The placement is *sandwiched between two mechanism actuations*, and the order
is physically forced (walls/roof block any insertion while unaligned — smoke
check 6 presses the carton at an unaligned bay with a real 6 N force and the
wall arrests it; stowing can only move a carton that is already aboard). The
seed's own end state — carton set down on the cupboard (its roof) or left at
the window mouth — is constructed and rejected in smoke check 5. A carton
stocked but left facing the window (a no-stow policy's natural end) is rejected
in smoke check 8; the stow threshold boundary in check 10; a wrong (occupied)
bay in check 7; a toppled carton in check 11; a displaced decoy in check 12.

Differs from the sibling same-seed package `put_groceries_in_cupboard_i329`
(spring-loaded facing-lane pusher: press items into a sprung lane against a
pusher plate) — here the storage is a **rotary indexing mechanism** operated
remotely via a crank, with a hide-the-item stow rotation and riders that must
survive it. No other tasks_v7 package indexes a carousel.

## Assets (fully procedural)

- **cabinet** — heavy DYNAMIC compound (50 kg root MassAPI; dynamic because a
  joint anchored to a kinematic body0 stays world-fixed when teleported at
  reset): round plinth (r 0.24, top z 0.10), a 260° polygonal wall band (inner
  face r 0.21, z 0.10–0.32) leaving a ±50° front WINDOW, and a roof (z
  0.32–0.34) with a 60 mm square axle hole (free gap beside the axle ~24 mm —
  passes nothing).
- **turntable** — DYNAMIC compound on a spawn-authored free vertical
  `UsdPhysics.RevoluteJoint` to the cabinet (no limits, angular damping 0.8,
  joint-pair collision explicitly ON, ≥ 7 mm authored clearances at every
  angle; per-child density 500 so the pivot inertia is real, ~2.5 kg): floor
  disc r 0.19 (top z 0.127), hub, three radial dividers every 120° (bay
  centres at turntable-local 0/120/240°), axle through the roof hole, crank
  arm + vertical KNOB (Ø24 × 60 mm, z 0.42–0.48 at 155 mm radius) — the
  graspable index handle, reachable at every angle.
- **carton** — the grocery: red box 55×55×110 mm, density 750 (0.25 kg),
  starting upright on the stand.
- **cans** — blue + green decoys Ø60×95 mm, density 600, seated at slot radius
  0.115 in two bays (which bays, and which can where, shuffled per episode).
- **stand** — kinematic grey pickup block 140×140×120 mm in front.

Randomization (readback-verified, smoke checks 2–3): cabinet yaw ±18° + xy
±40 mm; empty-bay index (3 values); initial empty-bay bearing ±U(75°, 170°);
decoy-to-bay shuffle; stand bearing ±35° at U(0.42, 0.52) m + free carton yaw.

## Rubric

- 0.15 `lifted` — carton ever raised above 0.27 m (latched)
- 0.25 `aligned` — empty-bay bearing ever within 30° of the window (latched;
  the initial bearing ≥ 75° makes this real work, asserted)
- 0.30 `loaded` — carton ever upright inside the empty bay (turntable-frame
  radial/angular/height bands; latched, 3-step persistence)
- non-success capped at 0.70; **1.0 iff success()**: carton upright and settled
  on the carousel floor of the formerly-empty bay, that bay ≥ 100° away from
  the window (fully behind the wall, asserted), both decoys seated in their own
  bays, everything finite — all judged live. Null policy ~0 (the damped pivot
  holds its angle; the carton starts on the stand).

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset, 180-step settle; layout + mass readbacks (cabinet via MassAPI,
  turntable/carton/cans via per-child density — custom spawners apply no cfg
  mass schemas); score ~0.
- **P1 (torque servo, real dynamics)** PD torque about the pivot
  (`set_external_force_and_torque`, pure z — what a hand dragging the knob
  applies; Kp 0.6 N·m/rad, Kd 0.25, clamp 0.45 N·m, sized for the ~0.045 kg·m²
  pivot inertia and the 1-substep wrench delay) indexes the empty bay to the
  window; the cans ride their bays. `aligned` latches from the pivot dynamics.
  Score 0.25.
- **P2a (applied force)** PD force + gravity feedforward + righting torque (a
  firm force-limited grasp) lifts the carton off its stand; `lifted` latches
  during the force lift. Score 0.40.
- **P2b (transport)** ONE root-state write carries the held carton to a hover
  OUTSIDE the window (cabinet-local x 0.32, z 0.20), upright, zero velocity —
  free air, outside every bay; nothing judged is satisfied by the write
  (asserted: not in-bay).
- **P2c (guided carry)** the same force-limited carry brings the carton in
  through the window and lowers it onto the carousel floor at the bay's slot
  radius; a light PD hold on the turntable (a hand steadying the knob) keeps
  the bay centred and is released before judging. `loaded` latches from
  contact. Score 0.70. Asserted: loaded-but-facing-the-window is NOT success.
- **P3 (torque servo)** the knob servo indexes the loaded bay to ~150°
  (clamp 0.30 N·m → tangential/centripetal accelerations at the slot radius
  far below the friction cone; the carton and cans ride). Wrenches zeroed.
  Score 1.0.
- **P4** ≥ 3.3 simulated seconds hands-off; success persists →
  `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport-only: alignment, loading and stowing are produced by
torque, force, friction, gravity and contact.

## Execution order

STRICTLY ordered: align, then insert, then stow. Alignment must precede
insertion (the wall arrests a forced carton at any unaligned bay — smoke
check 6, real force probe that demonstrably moves before arresting); stowing
must follow insertion (a rotation can only hide a carton that is already in
the bay; a stow with the carton anywhere else leaves `in_bay` false — smoke
checks 5/7/11). Geometry clauses asserted in `__post_init__`.

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing the cupboard (axis nominal
(0.50, 0), window facing the robot). Indexing = grasp or fingertip-drag of the
Ø24 mm knob (jaw span 80 mm > 24 mm, asserted) that stands proud above the
roof at 0.42–0.48 m height on a 155 mm-radius circle centred 0.50 m out —
worst-case reach ~0.66 m, within Franka's ~0.85 m envelope; large rotations
are a few re-grip strokes, and the free damped pivot holds its angle between
strokes (exactly what the solve's release-and-resume torque phases certify).
The carton pick is a side pinch on the 55 mm square body (< 80 mm jaw span,
asserted) at 0.12–0.23 m height on the stand 0.42–0.52 m out. Insertion is a
horizontal reach through the 100°-arc window (chord ~0.32 m, asserted > carton
diagonal + 0.15 m) to the bay centre 0.115 m past the axis — the wrist enters
above the disc with ≥ 60 mm of finger headroom under the wall top, then a
straight lower and jaw-open, the release the solve certifies. The decoy cans
are never touched. No step needs a second arm, a regrasp in flight, or
simultaneous contacts.

## Files

- `scene.py` — cfg (+ window/roof/stow/bay/topple/clearance asserts), spawners
  (cabinet, turntable + pivot joint, carton, cans, stand), scene (rubric,
  latches in `post_step`), `register_env`.
- `solve.py` — phased solution (knob torque-servo align, force-lift, transport,
  window carry, knob torque-servo stow); watchdog + hard exit.
- `smoke.py` — 13-check rejection battery (settle, randomization readback, bay
  shuffle readback, null-policy, SEED-strategy roof/ground rejection,
  wall-load-bearing force probe, wrong bay, loaded-not-stowed, latch
  regression, stow boundary 85°, toppled, displaced decoy, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 20.3 s; scores
  0.00 → 0.25 → 0.40 → 0.70 → 1.00, non-decreasing, 3.3 s hands-off
  persistence; bearing −117.8° → 0.0° → +151.1°)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 20.2 s; bearing +90.2°,
  opposite decoy shuffle, +13.9° cabinet yaw delta)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 13/13` (rc=0, 39.5 s; wall probe moved
  22 mm then arrested at r 0.278 > wall face 0.23; roof rest at z 0.395
  rejected; latched credit 0.55 preserved through carton removal without
  success; 99 frames saved)
