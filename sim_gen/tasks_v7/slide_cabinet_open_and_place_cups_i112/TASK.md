# shutter_cabinet — sort the cups through a bypass shutter that never uncovers both openings

Package: `sim_gen/tasks_v7/slide_cabinet_open_and_place_cups_i112`
Env: `simgen.shutter_cabinet` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/slide_cabinet_open_and_place_cups`. In the seed, the robot slides
one cabinet door OPEN — getting the panel out of the way is the whole articulation
act, done once — and then places interchangeable cups into the single revealed
volume, driven by a recorded waypoint trajectory (the checker is a TODO).

## What changed, and why it is strategically different

1. **The sliding panel can NEVER be gotten out of the way.** It is a captive BYPASS
   SHUTTER (26 cm plate on a spawn-authored prismatic track, hard stops ±55 mm) over
   TWO top-loading openings (13 cm each, centers ±11.5 cm). Geometry proved in
   `__post_init__` (221-position sweep): at NO track position does a cup-passable gap
   (≥ 60 mm) exist at both openings — uncovering one seals the other. The seed's
   "open, then fill" plan is impossible; there is no state where the cabinet is
   simply "open". Access is a mutually exclusive resource that must be MULTIPLEXED.

2. **Color-sorted deposits interleaved with shutter shuttling.** Two RED cups and one
   BLUE cup (identical size — color is identity) must each end upright on the floor
   of the bay matching its color. Every deposit into one bay requires sealing the
   other, so the solver alternates mechanism actuation with transport — N identical
   independent placements (seed) become a schedule around a shared resource.

3. **The end state must restore the mechanism:** success additionally requires the
   shutter parked FULLY COVERING the blue opening. A cup dropped on a covered
   opening rests ON the shutter (rejected by the membership z band), so the blue
   deposit is physically forced to precede the final parking — an ordering
   constraint enforced by the mechanism, not by fiat.

4. **Physics is the judge**, not proximity-to-recorded-waypoints: cabinet-frame
   membership windows whose z band accepts only floor-rest height (rejecting cups on
   the top plate at +0.033, on the shutter at +0.046, or stacked at −0.022 vs floor
   rest −0.080), an uprightness cone (a lying cup PASSES the z band — uprightness is
   load-bearing, asserted in `__post_init__`), a shutter-interval coverage
   predicate, and a stillness counter-latch.

**Free-shutter honesty note** (discovered on forge, kept deliberately): the shutter
is a free body on its track, so a cup dropped INTO a partial gap can wedge against
the shutter edge, shove it aside, and fall in. This does not break the invariant —
the displaced shutter then SEALS the other opening (verified physically: the wedge
drop drove the shutter to a stop and the opposite drop was denied), and the ordering
constraint survives (wedging blue's opening uncovers it, i.e. moves the shutter OFF
the required final pose). Smoke check 8 asserts the true invariant: at most one side
ever admits, and any admission coincides with a large shutter displacement.

## Teleport-solution outline (transport only — verified on forge)

Teleportation is used ONLY to release a cup (zero velocity, upright) ~4 cm ABOVE the
currently-uncovered strip of its bay's opening — gravity carries it through the
opening and seats it; nothing is teleported into the rubric state (floor-rest height
is only reachable by falling through an open aperture). The SHUTTER is never
teleported: it is driven by a velocity-regulated external force (body frame = world
frame for a non-rotating prismatic body; gains respect the one-substep wrench delay,
K·dt/m ≈ 0.14) along its real track into its real hard stops.

- **P0** settle + readbacks (shutter start c₀, cup slots); assert score ~0.
- **P1** push the shutter to the −stop → blue opening fully exposed.
- **P2** drop the BLUE cup through the exposed blue strip (center +0.128) → seats.
- **P3** push the shutter to the +stop → blue sealed (the required end pose), red
  fully exposed.
- **P4/P5** drop RED cup A (x −0.031) and RED cup B (x +0.031) through the red
  strip; ring down until `success()` holds 120 consecutive steps.
- **P6** hands-off ≥ 3.3 simulated seconds; assert success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, scores monotone
0.00 → 0.15 → 0.30 → 1.00, zero persistence flickers.**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(0.55, 0, 0) facing the cabinet (cabinet at the origin, 48×23 cm footprint,
16 cm tall; staging row at x ≈ 0.30). Reachability: cup slots lie 0.25–0.45 m from
the base at grasp height ~0.03 m; the openings are at radius 0.55–0.70 m and height
0.16 m — inside Franka's ~0.85 m envelope with a top-down approach. Graspability:
cups are 56 mm cylinders (< 80 mm jaw opening), grasped by a side or top pinch; the
yellow shutter handle (22×32 mm post, 5 cm tall) is an ideal pinch or hook target,
and the required push force (≈ 0.7 N against 4 N·s/m track damping at 12 cm/s) is
trivial for the arm. Deposits: each opening is 13×14 cm and open to the sky when
uncovered — a vertical lower-and-release 2–4 cm above the plate reproduces exactly
the solve's drop, with ±37 mm x-tolerance for the single blue cup and ±11 mm at the
offset red drop points. The 16 cm-deep bays admit the gripper itself if a gentler
place is preferred. The shutter stroke (11 cm) is a single short horizontal push.

## Execution order

Only the mechanically-forced constraint exists, and `describe()` states it: the blue
cup must be deposited before its opening is finally covered. Everything else is free
— red-first or blue-first, either red cup first, and the shutter may shuttle any
number of times. The solve's order (blue first, then reds, ending at the +stop which
IS the required park) is one convenient schedule, not a rubric requirement.

## Rubric

- +0.15 per cup that has EVER been seated in its own color bay (cabinet-frame
  window: |x| ≤ 0.075, side·y ∈ [0.034, 0.200], |z − (−0.080)| ≤ 0.021, uprightness
  within 30°), latched.
- +0.15 once all three are seated simultaneously, latched.
- +0.15 once all-seated coincides with the shutter covering blue (c ≥ 0.048 —
  residual slit ≤ 2 mm ≪ cup; GATED on all-seated so a shutter that merely starts
  there, or covers an empty cabinet, earns nothing), latched.
- Partial credit capped at 0.85; `score = 1.0` iff `success()` = all seated ∧
  covers-blue ∧ settled (stillness counter-latch, 30 consecutive quiet steps).
- Null policy scores ~0.

`__post_init__` honesty asserts: the 221-position mutual-exclusion sweep; full blue
coverage at the +stop with ≥ 4 mm slack; the cover-band edge sealed vs a cup; the z
band accepts pad AND bare-floor rest but rejects plate-top/shutter-top/stacked
heights; a lying cup passes the z band (uprightness is load-bearing); the two red
drop points clear the opening and each other; two cups fit one bay; jaw fit; reset
never writes the shutter into a stop.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 17/17`)

1. Reset settles: states finite, cups upright on the floor, shutter on its track.
2. Fresh reset: score ~0, no success.
3. Randomization: shutter start position varies (readback spread 75 mm across seeds
   21–26) and stays on the track.
4. Randomization: cup slot permutation varies (4 distinct / 6) + xy jitter (28 mm).
5. Null policy 240 steps: score ~0.
6. Coverage gating: EMPTY cabinet with the shutter parked over blue → covers_blue
   True yet score ~0 (coverage credit demands all-seated).
7. Sealed opening: a cup dropped over the COVERED blue opening rides ON the shutter
   (z +0.046 readback vs floor rest −0.080) — the cover physically denies access.
8. Mutual exclusion under free-shutter dynamics: drops over the most-open strip of
   EACH opening at mid-track → at most one admits (blue wedged in, driving the
   shutter to the −stop), the other is denied and rides high (+0.046).
9. Mechanism reality: a regulated push aimed 45 mm BEYOND the +stop travels 107 mm
   and the hard stop CLAMPS it (max c = +0.0550 ≤ stroke + 4 mm); released, it
   parks covering blue.
10. Seed-analog (open one door, everything into the revealed volume): all three
    through the exposed red opening → blue in the wrong bay, not success, ≤ 0.35.
11. Lying cup on the blue pad: z band PASSES (readback) yet uprightness rejects.
12. z band: a cup stacked on a seated cup (−0.022) and a cup on the solid top plate
    inside the bay's y window (+0.046) both rejected.
13. Parking near-miss: all three seated but the shutter 18 mm short of the cover
    band → not success, latched score 0.60.
14. Latched credit: removing the blue cup keeps the score (0.60 → 0.60) while
    all_seated() drops.
15. Settle gate: the completed arrangement judged while the last cup is still
    falling is NOT success (cup removed before ring-down — battery never succeeds).
16. Rejection audit: `success()` never fired at any judged point.
17. Final state no-NaN.

Smoke also records `frames.npz` (486 × 600 × 960 × 3) from a perspective camera.
