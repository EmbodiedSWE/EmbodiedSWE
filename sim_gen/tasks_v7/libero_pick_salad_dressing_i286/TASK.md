# rail_hanger — hang the salad dressing from the overhead rail rack

## Seed provenance

Seed: `libero/libero_pick_salad_dressing` — pick the amber salad-dressing bottle
out of a clutter of grocery distractors and place it into a basket that already
exists on the table (one pick-and-place judged by a containment bbox).

## What changed and why it is strategically different

The seed and every sibling variant examined (i43 cellar tow, i51 pile driver,
i104 rocker lock, i119 silo tip-pour, i18 wedge hopper, i46 pudding sift, i129
ballast vault, i52 ice doser, and i151 flat-pack crate) all end with the target
**contained**: the object comes to rest inside or on some receptacle — whether
the receptacle pre-exists (seed, most variants) or must first be assembled
around the object (i151). The strategy space is "how does the object travel
into its container".

This task abolishes containment entirely: **the goal state is the bottle
dangling in mid-air**, touching nothing but two overhead rails. A gallows rack
carries two parallel rails forming a narrow slot — open at a flared mouth,
closed at a column. The bottle's geometry gates the mechanism (all honest by
construction): neck dia 20 mm < rail gap 32 mm < cap-flange dia 52 mm < body
dia 56 mm, and the hanging body top rides 20 mm below the rail undersides. So
the bottle can be supported *only* by its cap flange on the rail tops, the cap
cannot pass down through the gap, the body cannot pass up, and the closed-end
seat is reachable *only* by threading the neck in through the open mouth (cap
above the rail plane, body below) and sliding the hanging bottle 0.35 m along
the slot until its body stops against the column. Instead of a vertical drop
into a receptacle, the load-bearing motion is a constrained **horizontal**
insertion + travel; release direction is irrelevant because gravity is carried
by the flange, not a floor. The seed's own receptacle survives as a **decoy**:
an open basket sits on the ground and dropping the bottle into it earns
exactly nothing (smoke proves it), as does standing the bottle on the rack's
slab under the slot. A red decoy bottle of identical shape (station-swapped
with the target each episode) punishes hanging the wrong object.

## Scene summary

- **Rack** (25 kg dynamic compound, free on the ground): 380×240 mm base slab,
  full-height column, two rails (tops at z = 0.32, inner faces ±16 mm) spanning
  x ∈ [−0.13, +0.13], flare beams at ±30° widening the mouth capture to
  ~40 mm. Rails/flares/column are slick (μ 0.06, min-combine); the slab keeps
  default friction so the rack stays planted.
- **Bottles** (amber target / red decoy, 0.25 kg compounds): body cyl r 28 ×
  140 mm, neck r 10 × 40 mm, cap flange r 26 × 12 mm (cap child slick).
  Hanging rest: root z = rail_top − cap_under = 0.210 (confirmed by readback).
- **Basket** (1.0 kg): 160 mm open box — the seed's receptacle, pure decoy.
- **Randomization** (readback-verified): rack yaw ±20° + xy ±30 mm; bottle
  stations swap; every loose piece gets ±30 mm jitter + free yaw.

## Teleport solution (solve.py)

Teleport = transport only; every load-bearing interaction is applied wrench
(the grasp force a wrist exerts) + contact through the rail/flange geometry.

1. **P0 settle** — 180 steps; assert score ≤ 0.03, no success, layout readback.
2. **P1 thread** — one root-state write to free air beside the mouth
   (rack-local x = +0.25 > flare tips +0.178, touching nothing); then a PD /
   velocity servo (gravity-feedforward lift, centreline PD, −0.10 m/s slot-axis
   velocity, uprighting torque; all gains bounded by K·dt/m ≪ 1) drives the
   neck in through the flared mouth. Retries re-transport at z ± 5 mm (none
   needed on either seed). Then lift is cut to 0.7·m·g: the cap sinks onto the
   rail tops and the rails carry 0.3·m·g — `hooked` latches on a genuinely
   supported state. Score 0.25.
3. **P2 slide** — same servo, cap riding the slick rails, until the bottle
   body stops against the column (root x = −0.102, measured); gain escalates
   ×1.6 on stall (never triggered after the stop was calibrated). A short
   40-step press (< the 60-step still window) latches `seated`. Score 0.60.
4. **P3 release** — wrench zeroed while slow; the rails carry the full weight;
   success() after the still window. Score 1.0.
5. **P4 persistence** — 400 further steps hands-off, success holds, then
   `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge with `--seed 0` and `--seed 1` (both rc=0, ~21 s, monotone
score trace 0.00 → 0.25 → 0.60 → 1.00).

## Rubric (written after the solution worked)

Latched credits, each gated on "slow (< 0.08 m/s) ∧ finite": `0.25` hooked
(hanging anywhere in the slot: on-centreline ±14 mm, root z in the hang band
(0.195, 0.222) — excludes standing anywhere and lying on top of the rails —
within the slot span, tilt < 15°) + `0.35` seated (hanging at x < −0.095;
physical stop at −0.102), clamped at **0.60** for any non-success state.
Score = **1.0** iff live `success()`: amber bottle seated ∧ red decoy not in
the rack exclusion volume ∧ settled (60-substep consecutive-still counter) ∧
all states finite.

## EMBODIMENT ARGUMENT (single Franka, one base pose)

Place the Franka base ~0.55 m from the rack centre on the mouth-side diagonal
(rack-local ≈ (+0.35, −0.40)): the 0.8 m workspace disc covers both bottle
stations, the basket, the mouth approach corridor and the full slot span, at
working heights ≤ 0.35 m; the ±20° rack-yaw band keeps all of these in reach.

- **Grasp**: cylindrical wrap around the 56 mm body barrel (< 80 mm stroke),
  0.25 kg — trivial load. One grasp for the whole task; no regrasp, no
  bimanual step.
- **Thread**: hold the bottle upright, cap at ~0.33 m, and move horizontally
  along the slot axis. The insertion window is generous: 20 mm of vertical
  clearance (cap above rails / body below them), ±6 mm lateral neck clearance
  widened to ~±20 mm capture by the flares, 15° tilt tolerance — all well
  inside Franka repeatability; the flares chamfer the final alignment.
- **Slide**: a horizontal guarded move of 0.35 m at fixed height. Once the cap
  rides the rails the fixture itself provides vertical compliance; resisting
  friction is ~0.05 N (slick-on-slick, 2.5 N weight) — far below wrist limits.
  The demonstrated servo uses exactly this wrench profile (≤ 3 N lateral,
  ≤ 2·m·g lift).
- **Release**: open the gripper at the seat and retract sideways; fingers wrap
  the barrel ~100 mm below the rail undersides, so the retract path is free
  air. Gravity is already carried by the flange (proven: hands-off persistence).

## Files

- `scene.py` — `RailHangerScene`, registered as `rail_hanger`,
  env `simgen.rail_hanger` (robot="null").
- `solve.py` — teleport + wrench solution,
  `python -m simgen_tasks.libero_pick_salad_dressing_i286.solve --headless [--seed N]`.
- `smoke.py` — 12-check rejection battery (below), records `frames.npz`.
- `TASK.md` — this file.

## Smoke checks (12) — forge result: `SIM_GEN_SMOKE: ALL PASS 12/12`

1. Settle/no-NaN: layout settles finite; both bottles upright on the ground,
   nothing hanging, decoy off the rack; score ~0, no success.
2. Randomization-is-real: rack yaw/xy, dressing xy, basket xy READBACK all
   differ across seeds (15.6°, 30 mm, 217 mm, 88 mm measured).
3. Bottle swap: the amber bottle occupies BOTH scatter stations over 10 resets.
4. Null policy: 240 idle steps, nothing hangs, score ~0, no success.
5. Negative (SEED strategy): the amber bottle dropped INTO the basket (probe
   verified fallen and staying at the basket) earns ZERO credit, no success.
6. Under-rack stand: the bottle standing on the rack slab directly under the
   seat earns nothing (hang credit is a height band, not proximity).
7. Hang mechanism: the bottle dropped into the slot near the mouth stays
   SUSPENDED in mid-air by cap-on-rails contact (root settles in the hang
   band), latches hooked (0.25), and does NOT self-slide to the seat: partial
   credit only, no success.
8. Off-slot drop: released at hang height but outside the slot, the bottle
   falls to the ground — the hang band without the rails earns nothing.
9. Lying across rails: the bottle dropped horizontal onto the rail tops (probe
   verified fallen; rests in the rail trough at z ≈ 0.343) never hangs: no
   credit.
10. Negative (wrong object): the RED decoy hung at the seat (it does hang —
    the mechanism is object-agnostic) → decoy-exclusion fails: no success, no
    credit for the untouched amber bottle.
11. Live + cap: a seat-hang constructed BY DROP succeeds only after the still
    window (first fired at post-drop step 66 ≥ 60); lifting the bottle off →
    success OFF, score = latched cap 0.60 (float-safe bounds); re-hanging by
    drop → success returns.
12. Video: >10 frames captured and saved to frames.npz (148 frames).
