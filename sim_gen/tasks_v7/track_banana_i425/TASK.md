# glazing_bench — clear the seat, bed the pane, slide the keeper home

**Task dir:** `sim_gen/tasks_v7/track_banana_i425/` · **Scene name:** `glazing_bench` ·
**Env key:** `simgen.glazing_bench` (robot=`null`)

## Seed provenance

Derived from `pick_place/track_banana`
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_banana.py`). The seed
task's core mechanic is *trajectory tracking of a held object*: a banana is
carried through a sequence of free-space waypoints and success is a function of
the held object's pose history. Nothing in the seed ever has to interact with
anything — the world is a prop.

## What changed, and why it is strategically different

The glazing bench inverts the seed's premise: **no free-space carry is ever
judged; every scored state change must be made through contact.**

The task is a three-stage workbench assembly with **interference-geometry order
forcing** (no scripted latches):

1. **Clear** — 2–4 glass offcuts ("pebbles", 18 mm) foul a recessed seat sunk
   into a bench bay. Each pebble is taller than the recess is deep, so while any
   pebble remains, a dropped pane genuinely **rests proud on it** (physics, not a
   rule, denies the seat credit).
2. **Seat** — the 110 mm square pane must lie flat inside the 120 mm recess,
   knob-up, within a 8 mm-tall flush band.
3. **Lock** — the keeper (an 80 mm tongue with a 30 mm-tall head) must be slid
   along the bench top so its tongue passes under a 12 mm housing slot and its
   tip crosses the pillar line, overhanging the seated pane by ≥ 15 mm. The slot
   admits the 10 mm tongue but jams the 30 mm head — the pillar face is a hard
   stop, so "slide until it stops" is the honest strategy.

Order is forced both ways by geometry alone:
- **Seat before lock:** with the keeper locked, its 15 mm tongue overhang
  shrinks the 120 mm opening to 105 mm < the 110 mm pane — the natural flat
  drop is physically blocked (asserted in cfg, verified by the flagship smoke
  probe). *Honest caveat:* only the flat drop is strictly excluded; a tilted
  sneak-in is theoretically conceivable but has no natural execution path.
- **Clear before seat:** any remaining pebble props the pane above the flush
  band (propped lower bound 0.0454 > band top 0.041, asserted in cfg).

Strategic distance from neighbours examined: not a count-fill (pen_holder), not
a ram-feed push (i240), not a clamp-release catch (i79), not an armed cascade
(poke_cube_i83). The signature move — a *forced assembly order encoded purely
in interference margins*, finished by a friction slide into a hard stop — is in
none of them, and the seed's own strategy (waypoint carry of a held object) is
explicitly constructed in smoke and scores ~0.

## Rubric

- `pebs_out_frac()` — fraction of *present* pebbles outside the bay footprint
  (bench frame, present-masked).
- `pane_seated()` — bench-frame |xy| ≤ 10 mm, z in [0.033, 0.041], up·z ≥ 0.995.
- `keeper_locked()` — tongue-tip x ≤ 0.052 (past the pillar line), |y| ≤ 15 mm,
  z in band, signed body-axis alignment ≥ 0.99 (rejects a backwards keeper),
  settle-gated (rejects fly-through writes).
- `success()` = all pebbles out ∧ pane seated ∧ keeper locked.
- `score()` = clamp(0.15·frac + 0.35·seated, ≤ 0.50); 1.0 iff success.

All thresholds live in the bench (fixture) frame — anchor jitter (±3 cm) and
yaw (±20°) randomization never move them. ~25 `__post_init__` asserts pin every
interference margin (slot admission 10 < 12 < 30 mm; locked opening 105 < 110;
propped-pane bound; decoy 155 mm > recess 120 mm; Franka-jaw admission).

## Teleport solution (solve.py) — forge-verified

Transport-only teleports; the one load-bearing interaction (the lock slide) is
a genuine external-force push.

- **P0** settle 120 steps, layout readback, score 0.
- **P1** teleport each present pebble to a depot on the apron → frac = 1,
  score 0.15.
- **P2** hover-drop ladder (8.0 / 6.6 / 6.0 cm) releases the pane over the
  cleared recess; it beds itself under gravity (measured z = 0.0370,
  up·z = 1.000) → score 0.50.
- **P3** stage the keeper at the drop zone; **calibrate** the wrench frame (4
  candidate encodings probed at 0.55 N from a saved state, restored via
  `set_state`; best displacement wins), then a velocity-capped (0.12 m/s)
  0.7 N push drives the tongue under the housing until the head jams on the
  pillar (tip settles at +0.0450, exactly the hard stop) → score 1.00.
- **P4** hold ≥ 3 simulated seconds; success stable → `SIM_GEN_SOLVE: SUCCESS`.

Forge results: seeds 0 and 1 both SUCCESS first try (20.5 s / 18.1 s), score
non-decreasing 0.00 → 0.15 → 0.50 → 1.00.

## Embodiment argument (Franka)

Every manipulated object admits a standard two-jaw grasp (jaw span 78 mm):
pebbles are 18 mm cubes (top pinch); the pane is lifted by its 24 mm knob
(knob-pinch, keeps fingers out of the recess); the keeper is pushed by
flat-palm/fingertip contact on its 30 mm-tall, 32 mm-long head — the lock
stroke is a straight, horizontal, obstruction-free push along the bench top.
One plausible fixed base pose serves the whole task: base at ≈ (−0.55, 0)
facing +x reaches the ground slots (x ≈ −0.26/−0.34), the bay (x ∈ [−0.12,
0.26]) and the apron drop zone (x ≈ 0.175), all within ~0.85 m.

## Declared execution order

`clear → seat → lock`, physically forced by the interference geometry above
(with the stated flat-drop caveat). The out-of-order end state — keeper
genuinely force-slid home first, pane then dropped — is constructed in smoke
and yields no success.

## Checks (smoke.py — 17)

1–2 settle/no-NaN + score ~0 · 3–4 randomization readback (12 seeds: pebble
count 2–4, slot shuffle, anchor/yaw spread) · 5 null policy (300 steps) ·
6 seed-strategy construct (5-waypoint held carry → set-down, score ~0) ·
7 drop-without-clearing rests proud · 8 straddle near miss (off-center/tilted)
· 9 upside-down pane (knob props it) · 10 wrong object (decoy spans the
opening) · 11 out-of-order FLAGSHIP (genuine lock-first slide, pane blocked) ·
12 partial slide (genuine push stops short of the overlap threshold, score
capped 0.50) · 13 backwards keeper (signed alignment) · 14 fly-through (settle
gate) · 15 geometry + mass audit (bench-frame margins + `get_masses` readback)
· 16 success-never-True audit · 17 final no-NaN. Frames recorded to
`frames.npz`.
