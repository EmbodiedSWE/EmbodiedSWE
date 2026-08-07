# balance_scale — load the three weights so the pendulum balance settles level

Package: `sim_gen/tasks_v7/place_cups_i82`
Env: `simgen.balance_scale` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/place_cups`. In the seed, the robot picks up 1–3 **identical**
mugs from a table and hangs each on its own peg of a mug tree — N independent,
interchangeable prehensile transports onto N independent static targets, judged by
per-object proximity sensors, driven by recorded waypoint trajectories.

## What changed, and why it is strategically different

1. **A mass-partition puzzle instead of repeated identical placements.** The three
   weights are deliberately NOT interchangeable: a green 42 mm, blue 56 mm, and red
   70 mm cylinder with the invariant `m_big = m_small + m_mid` (exact masses
   resampled every episode). There is exactly one balancing split — the widest
   weight alone versus the other two together — so the task is *which weights go
   where*, not *repeat the same skill N times*. No corpus task poses a reasoning
   constraint enforced by statics.

2. **The target is one shared, compliant, moving mechanism** — two rimmed pans
   hanging from a beam that pivots on a spawn-authored revolute joint (±14° hard
   stops, pendulum bob restores level). Every placement changes the pose of the
   target for the next placement (the solve reads the LIVE beam pose to aim each
   drop). The seed's mug-tree pegs are static and independent.

3. **Physics is the judge.** Success does not check "each object near its sensor":
   it demands all three weights riding pan FLOORS (beam-frame membership window
   rejects rim perches and stacking) *and* the beam itself settling LEVEL within
   6°. A wrong split physically heels the beam to ~9°+ (measured −9.08° on forge)
   — the mechanism, not a distance threshold, discriminates correct from wrong.

## Teleport-solution outline (transport only — verified on forge)

Teleportation is used ONLY to transport a weight to ~25 mm above its target pan
floor (zero velocity, beam-aligned orientation, target computed from the live beam
pose so it rides the heeled beam). Everything load-bearing is contact dynamics:
gravity seats the weight, rims + friction keep it aboard, the beam heels onto its
stop under one-sided load and swings back as counter-loads arrive, and the final
LEVEL state is the mechanism's own ring-down.

- **P0** settle + readbacks (masses, layout); assert empty beam rests level; score ~0.
- **P1** drop `big` into pan +y; assert it rides the pan AND the beam heels past
  `tilt_max` (mechanism reality: −13.4…−14.0° observed).
- **P2** drop `mid` into pan −y at x = −0.026 (partial counterweight; −4.5…−8.9°).
- **P3** drop `small` beside it at x = +0.030; wait for `success()` to hold 120
  consecutive steps as the beam rings down to |tilt| < 0.3°.
- **P4** hands-off ≥ 3.3 simulated seconds; assert success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, scores
monotone 0.00 → 0.10 → 0.20 → 1.00, zero persistence flickers.**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(−0.45, 0, 0) facing +x. Reachability: the staging row sits 0.37–0.46 m
from the base at grasp height ~0.035 m; the pans hang at radius ~0.63 m, floors at
~0.12–0.15 m height — all inside Franka's ~0.85 m envelope. Graspability: the
weights are solid cylinders 42/56/70 mm in diameter × 70 mm tall — a side or top
pinch well within the ~80 mm jaw opening, and their differing widths/colors make
them visually and haptically identifiable. Deposits: each pan pocket (112 mm
inside, 24 mm rims) is open to the sky — the crossbar ends at ±0.24 m before the
pan interiors begin at ±y 0.244–0.356 m, so a vertical approach is unobstructed;
even the widest cylinder leaves ±21 mm of centering tolerance. A gripper release
~25 mm above the pan floor reproduces exactly the solve's drop. The beam moving
under load is handled the same way the solve handles it: aim relative to the
current beam pose.

## Execution order

NONE required, and none is enforced: either pan may take the big weight, the three
drops may happen in any order, and the beam resting on its hard stop while partly
loaded is expected and harmless. All of this is stated in `describe()`.

## Rubric

- +0.10 per weight that has EVER ridden a pan floor (beam-frame window: |x| and
  |y−±pan_y| within 55 mm, z within [−0.145, −0.095]; rest is −0.117, a stacked
  weight reads −0.047 → rejected), latched.
- +0.20 once all three ride simultaneously, latched.
- +0.40 once loaded ∧ level (|tilt| ≤ 6°) ∧ settled, latched. `settled()` is a
  stillness **counter-latch**: all bodies under the velocity gates for ≥ 36
  consecutive steps, so a pendulum's instantaneously-still turning points don't
  count.
- Partial credit capped at 0.90; `score = 1.0` iff `success()` (loaded ∧ level ∧
  settled, live). Null policy scores ~0.

Scene `__post_init__` carries honesty asserts on the statics: the worst-case wrong
split must heel ≥ 1.4× `tilt_max` (9.17° > 8.4°) and the worst rim-parked correct
split must stay level (3.49° < 4.5°), so the geometry cannot silently drift into a
rubric that a wrong answer passes.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 15/15`)

1. Reset settles: finite state, post upright, empty beam level, settled.
2. Fresh reset: score ~0, no success.
3. Mass randomization: per-seed spreads 27.8/25.2 g across seeds 21–26, invariant
   |m_big − (m_small+m_mid)| ≤ 3e-8 kg every reset, view-vs-cache agreement.
4. Layout randomization: staging slot permutation varies (4 distinct in 6 seeds),
   x jitter observed up to 30 mm.
5. Null policy 240 steps: score stays ~0.
6. Seed-analog rejection (one weight per pan, third left on floor): loaded False,
   not success, score ≤ 0.25.
7. Wrong partition {big,small}|{mid}: beam heels −9.08° (past 6°), not success.
8. All three in one pan: beam pinned at −14.0°, not success.
9. Weight parked on the crossbar at the pivot: beam level, yet in_pan False, ~0.
10. Stacking small ON mid (correct split!): beam genuinely level (+1.77°) yet the
    stacked weight reads z −0.047 outside the floor window → rejected.
11. Settle gate: correct full load judged mid-swing (ω 0.52 > gate 0.10) → not
    success until the ring-down completes.
12. Latched credit: removing a weight keeps latched score (0.50 → 0.50) while
    live `loaded()` drops — score never decreases.
13. Mechanism reality: big alone heels the beam to −14.0°; removing it restores
    +0.43° — the joint and pendulum genuinely work.
14. Rejection audit: `success()` never fired during any rejection scenario.
15. Final state no-NaN.

Smoke also records `frames.npz` (500 × 600 × 960 × 3) from a perspective camera.
