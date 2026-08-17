# libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i290 — Moka Carousel

## Provenance

Seed: `RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove.py`
(LIBERO-90 kitchen scene 3: moka pot, chefmate frypan distractor, flat stove; success =
the pot's xy within 6 cm of the burner site with 0 < dz < 5 cm — a direct
pick-and-place pose predicate).

Kept from the seed: the moka pot as the manipuland, the frying pan as the distractor,
a kitchen-counter setting, and "get the pot onto the warming target" as the goal
statement. Everything else — the target's geometry, the transport mechanism, the goal
predicate, and the required plan — is new. All assets are procedural (native PhysX box
colliders); no external files.

## The task

A free-spinning 3-bay serving CAROUSEL (turntable on a revolute-Z joint, brass push
posts) stands on the counter. The warming station is one azimuth stop (world +x of the
carousel axis, marked by an orange lamp) covered by a LOW KINEMATIC CANOPY: roof
underside 100 mm above the turntable. One bay is pre-occupied by the frying pan; the
turntable starts at a uniformly random angle; the pot starts on the counter in front
with xy jitter and free yaw.

Goal: pot upright inside a bay, that bay's azimuth within 15° of the station,
turntable stopped, everything at rest (judged live on the settled state).

## Why this is strategically different

**vs the seed** — the seed is one grasp-carry-release to an open support surface; its
predicate is a pot pose. Here the pot's goal pose is GEOMETRICALLY UNREACHABLE by any
carry: entering a bay requires crossing a 35 mm fence, so the pot (76 mm) must pass
with its top at ≥ 111 mm — above the 100 mm canopy underside. The pot can only reach
the station riding IN a bay (headroom 100 − 76 = 24 mm). The plan therefore becomes:
stage an empty bay to the open front (the frypan may occupy the convenient one),
load the pot, rotate the carousel by its posts under closed-loop control to an azimuth
window, and stop it — transport is mediated by a mechanism and the terminal predicate
lives on the mechanism's angle. Nothing in the code is "move pot to pose X".

**vs the same-seed sibling i141 (balance scale)** — i141 is a measurement/subset-sum
task: pick the counterweight combination for a hidden randomized mass, and the beam's
free tilt is the read-out. Here there is no hidden quantity and no measurement; the
core is indirect delivery through a rotating carrier plus a geometric barrier, with a
stop-inside-a-window control requirement. Different mechanism (continuous revolute
carousel vs limited-tilt beam), different plan (stage/load/rotate/stop vs choose
subset/weigh), different predicate (mechanism azimuth + at-rest vs level-beam +
exact-mass).

**vs the pen_holder exemplar** — that is container filling with pose predicates on the
contents; no mechanism, no barrier, no ordering constraint.

## Teleport solution (solve.py, proven on the forge)

- P0: settle; assert pot on the counter, frypan riding a bay, score ~0.
- P1: pick the empty bay nearest (by rotation) to the loading azimuth π and rotate it
  there with a cascaded velocity servo — torque about the carousel's own hinge axis
  (drag-invariant), ω_des = clamp(−2·err, ±0.8 rad/s), τ = 0.6·(ω_des − ω_fd),
  |τ| ≤ 0.5 N·m, hinge rate finite-differenced (ang-vel readback is phantom under
  wrenches). Audit: k·dt/I = 0.14 ≪ 1; post-cut coast ω/damp ≤ 2.3° ≪ 15° window.
- P2: ONE teleport carries the pot across free space to a hover 15 mm above the staged
  bay's floor (open sky at azimuth π); gravity + contact seat it; seat latch fires
  (score 0.25). Bounces are retried from the same hover.
- P3: servo the loaded bay to azimuth 0, brake, cut the wrench; wait for success to
  hold 240 consecutive substeps (2 s). Score 1.0.
- P4: 3.33 s hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport only; the delivery itself (the pot appearing at the station)
is 100 % mechanism + contact: the write that placed the pot happened at azimuth π,
outside the canopy. `SIM_GEN_SCORE` is printed at each boundary and is non-decreasing
(0.0000 → 0.0000 → 0.2510 → 1.0000 → 1.0000 on both seeds 0 and 3).

## Embodiment argument (Franka, no robot in this scene)

Base at (−0.35, 0, 0) facing +x, everything within ~0.75 m reach. Pot: power grasp
across the 54 mm octagonal flats or pinch on the black side handle; lift over the
8 mm fence of a front bay (open sky there) and set down — a standard top-down place.
Rotation: the three 16 mm-square, 80 mm-tall brass posts ride at r = 19 cm; at any
turntable angle at least one post is in the open front half, so the closed fingertip
can push it tangentially (ratchet pushes, re-gripping the next post as it comes
around) — exactly what the applied hinge torque emulates. Stopping inside the 15°
window needs ~3 cm accuracy at the post radius — coarse by manipulation standards.
No motion requires entering the canopy volume: loading happens at the front, and the
final stop is judged with hands off.

## Execution order

Loading must precede delivery (the rubric's progress term is gated on the seat
latch); either rotation direction is allowed; no other ordering constraint. If the
randomly convenient front bay is occupied by the frypan, the solver must rotate
first — the staging phase covers this uniformly.

## Rubric

- 0.25 — seat latch: pot ever at rest upright inside any bay (latched).
- 0.35 — progress: latched running-MIN of the loaded bay's azimuth error, scaled
  (1 − err/π), gated on the seat latch.
- Non-success capped at 0.60; exactly 1.0 iff live success (upright-in-bay + window +
  pot at rest + turntable stopped + finite).

Honesty is asserted in `MokaCarouselSceneCfg.__post_init__`: fence + pot > canopy gap
(no lift-over under the roof), 20 mm ride headroom, bays/posts on slab material
through the crossed-square star profile, pillars clear the swept star, pot/pan fit a
bay at any yaw, the whole success window lies deep under the canopy, and the pot
spawn zone clears the swept star.

## Verification (forge, RTX-4090 pod)

- solve: `SIM_GEN_SOLVE: SUCCESS`, seeds 0 and 3 (rc=0, ~22 s each), scores
  monotone 0 → 0.251 → 1.0.
- smoke: 13 checks — settle/no-NaN; randomization readback (turntable yaw, pot
  xy/yaw); frypan-bay spread (all 3 bays, position-readback-consistent); null policy;
  seed-strategy drop intercepted by the canopy roof; pot-on-open-disc rejected;
  seated-only partial credit without success; 28° azimuth near-miss rejected; frypan
  parked at the station rejected; lying pot in the station bay rejected (upright
  clause); spinning fly-through of the window rejected (at-rest clauses) while the
  window is provably crossed; latched credit persists after removal without granting
  success; frames.npz saved. `SIM_GEN_SMOKE: ALL PASS 13/13`.
