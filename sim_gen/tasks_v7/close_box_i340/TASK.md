# close_box_i340 — Slam-Shut Courier

**Scene name:** `slam_shut_courier` · **Env:** `simgen.slam_shut_courier` (robot="null")

## Seed provenance

Seed task: `rlbench/close_box` — "close the box": an articulated box USD stands with
its hinged lid open; the robot pushes DOWN on the lid, rotating it shut about its
hinge, judged by the lid joint angle (`JointPosChecker`, mode "le"). The seed's
whole interaction is one direct quasi-static contact ON the lid panel itself, at a
stationary box.

## Strategic difference

The lid here is still a real spawn-authored revolute joint on the box — but **the
seed's strategy (touch the lid, rotate it shut) is geometrically impossible for the
entire useful workspace, and the goal is compound**: the box is a courier package
that must first be DELIVERED down a roofed alley to a bumper dock, and the lid can
only be shut by **impact dynamics** — the box is launched down a slick alley, hits
the dock bumper, arrests dead, and the open lid's own angular momentum about the
suddenly-stopped hinge carries it over its over-center balance point so gravity
slams it shut over the cargo.

Concretely, three interlocking mechanisms replace the seed's push:

- **Over-center bistable hinge** (limits −120°..0°): at its open rest the lid leans
  past vertical, so gravity HOLDS it open; a quasi-static push must drive it through
  the balance apex — but
- **the low roof (z 0.2485) sits BELOW the closing sweep's apex (z 0.259)** for the
  whole launch/coast region: any attempt to rotate the lid shut in place jams on the
  roof at 111° open (predicted analytically, measured −111.1° live in smoke checks 4
  and 6 with the box held), while the open lid's tail (z 0.2382) rides freely
  beneath it. The closure window (high canopy, z 0.270) opens only within ~15 mm of
  full delivery — where the box only ever arrives at speed. And pressing the lid
  *without* holding the box just drags the free box down the slick alley into the
  bumper — i.e. the seed's action, applied here, degenerates into the launch
  strategy itself.
- **Arrest-flip energetics**: closure needs an arrest speed v ≥ 0.602 m/s
  (rod-arrest model: ω = 3·v·sinΘ/(2L) must beat the (4gL/3)(1−sinΘ) energy hill),
  so a *gentle* delivery that parks the box at the dock leaves the lid open — smoke
  check 5 delivers quasi-statically to the exact success x and the lid stays at
  −120°, score capped at 0.40. Delivery and closure cannot be satisfied separately:
  only the slam produces both.

Differs from every corpus task read: the seed and `close_laptop_lid_i152` rotate a
hinged panel by direct/loaded contact; `close_box_i26` inverts a free crate over a
block (lid IS the box); `close_box_i300` threads a separate free lid through side
grooves (no joint at all); `open_box_i184` rolls a gravity-seated roller off;
`close_microwave_i4` releases a prop so gravity shuts a gate that was already past
its balance point. **No corpus task closes a real hinged lid by arresting the moving
box it is mounted on — impact-transfer closure, with the direct strategy jammed out
by a roof for the whole approach** — and none couples "deliver the container" and
"close the container" into one launch.

## Assets (fully procedural)

- **fixture** — KINEMATIC alley compound: slick floor slab (μ≈0.05/0.04,
  x −0.16..0.66), side walls (inner faces ±0.088), a LOW ROOF over the launch and
  coast region (bottom z 0.2485, ending x 0.369), a HIGH CANOPY over the dock
  (bottom z 0.270, x 0.369..0.612), a full-height red BUMPER (inner face x 0.60,
  restitution 0), and a one-way entry SILL (top z 0.018) behind the launch pad.
- **box** — DYNAMIC open-top courier crate, 160×160×100 mm, wall/floor 8 mm,
  0.6 kg, authored CoM + diagonal inertia, slick bottom (pair-averaged with the
  slick floor → μ≈0.04 coast), teal.
- **lid** — DYNAMIC thin plate (155×150×8 mm, 0.08 kg) whose root sits ON the hinge
  line; a spawn-authored `UsdPhysics.RevoluteJoint` (axis Y, limits −120°..0°,
  collision-filtered to the box) makes it a true articulated lid. Explicit CoM
  (L/2, 0, t/2) and inertia authored.
- **cargo** — DYNAMIC 30 mm gold cube, 0.05 kg, spawns inside the box cavity.

Randomization (readback-verified, smoke check 2): fixture yaw ±20° + xy ±30 mm,
box launch-pad x ±12 mm, cargo box-frame xy ±20 mm.

Honesty geometry asserted in `__post_init__` (23 asserts): over-center rest; open
tail under the roof with margin while the closing apex is above it; jam angle far
short of the shut latch; closure window opens only 5–25 mm before full delivery;
the flip clears the roof's end face; closed lid covers the mouth and clears bumper
and walls; sill is one-way for the box; coast room and v_launch sanity; cargo fits
with headroom and a cargo ON the closed lid fails the contents band; thresholds
ordered.

## Rubric

Latched partial credit (never evaporates), finite-guarded, upright/in-alley gated:

- 0.20 `entered` — box ever past the sill line (x ≥ 0.20) upright in the alley
- 0.20 `arrived` — box ever within 40 mm of full delivery (x ≥ 0.48)
- 0.25 `shut` — lid ever swung past −45° (only reachable via the slam: the roof
  jams in-place pushes at ≈−111°)
- non-success capped at 0.70; **1.0 iff success()**: box delivered (x ≥ 0.495,
  upright, in the alley), lid closed (≥ −10°), cargo inside the cavity BELOW the
  lid plane, everything settled and finite — all judged live.

## Teleport solution (solve.py) — the legitimacy certificate

**Zero teleports after reset** — the whole solution is one honest shove:

- **P0** reset, 90-step settle: box on the launch pad under the low roof, lid at
  its −120° open rest, cargo inside; score 0.
- **P1 LAUNCH (applied force)** a velocity-regulated CoM force along the alley
  (force-frame mode probed from progress, stiction floor with stall escalation)
  accelerates the box to v_launch = √(2.2·v_req² + 2μg·d_coast) ≈ 1.00 m/s and cuts
  at the release line x 0.26 — well under the roof, 0.32 m short of the bumper.
  Score 0.20.
- **P2 COAST + SLAM (hands off)** the box slides free, hits the bumper, arrests
  dead (restitution 0); the lid's angular momentum carries it over the apex and
  gravity slams it shut over the cargo; settle. Score 1.00.
- **P3** ≥ 3.3 simulated seconds hands-off; success persists → `SIM_GEN_SOLVE:
  SUCCESS`.

Every rubric fact (entered/arrived/shut/delivered/closed) is produced by the
applied launch force, sliding contact, and arrest dynamics.

## Execution order

Deliver-then-close is **physically forced, and fused**: the roof makes closure
impossible until the box is within ~15 mm of full delivery (assert + smoke checks
4/6), and the over-center hinge plus arrest energetics make the slam the only
closer once there (smoke check 5: gentle delivery leaves the lid open). The single
launch action necessarily produces enter → arrive → shut in that order; the cargo
is never touched.

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing +x; fixture at (0.30, 0), launch pad at
x≈0.39 world — the box's rear face sits at reach ≈0.30 m through the open alley
entry (the sill is only 18 mm tall and the roof starts at x −0.11 fixture-frame,
leaving the rear face approachable from behind).

- **Launch:** closed fingertips push the box's rear face horizontally at CoM height
  (~30–60 mm). The servo peaks at 8 N and typically runs 2–4 N over a 0.17 m
  stroke ending at 1.0 m/s — a trivial flick well inside the Franka's force and
  speed envelope, executed entirely OUTSIDE the roofed section (release at fixture
  x 0.26 is under the roof, but the pushing contact is on the rear face, trailing
  at x ≈ 0.18, still short of the roof mouth; the arm never needs to enter the
  245 mm-wide, 249 mm-low duct).
- **No other contact exists in the solution:** coast, arrest, flip, and settle are
  all hands-off. The dock sits 0.43 m inside the roofed alley — unreachable for a
  direct lid push even if it weren't jammed by the roof, which is precisely why the
  impact strategy is the intended (and only) one.
- No step needs a second arm, regrasping, or any contact with the lid or cargo.

## Files

- `scene.py` — cfg (+23 honesty asserts in `__post_init__`), compound
  fixture/box/lid spawners (spawn-authored revolute joint), scene (latched rubric
  in `post_step`), `register_env`.
- `solve.py` — force-driven solution (no teleports after reset); watchdog + hard
  exit.
- `smoke.py` — 12-check rejection battery (settle, 3-seed randomization readback,
  null policy, seed-strategy lid press with the box held — jams on the roof at
  −111.1°, gentle-delivery mechanism probe, near-miss park + still-jammed press,
  closed-but-undelivered, cargo-outside, settle gate with moving success pose,
  success-never-True audit, score-cap audit, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 18.0 s), score trace
  0.00 → 0.20 → 1.00 → 1.00
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 18.1 s), same trace
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 12/12` (rc=0, 54.6 s); both lid presses
  jam at −111.1° (predicted −111), gentle delivery caps at score 0.40, battery-wide
  max score 0.65
