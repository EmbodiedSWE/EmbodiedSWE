# beat_the_buzz_i263 — Warded spindle (keyed descent down a post)

**Scene:** `simgen.warded_spindle` (`WardedSpindleScene`, robot="null")
**Seed:** `rlbench/beat_the_buzz`
(`RoboVerse/roboverse_pack/tasks/rlbench/beat_the_buzz.py`)

## Seed provenance and what changed

The RLBench seed is the buzz-wire game: grasp a wand and guide its loop along a
bent wire end-to-end while NEVER touching it. Its whole plan is a **keep-out
traversal** — one continuous guarded sweep where the skill is clearance
maximization along a curve and any contact is instant failure.

This task keeps the seed's one recognizable surface — *a captive ring travelling
along a fixed rod* — and inverts every piece of its content:

1. **Contact is the normal state, not failure.** The blue sleeve (a closed
   octagonal hub with one yellow fin) is threaded on a vertical steel post and
   RESTS on the upper of two ring-shaped ward shelves. There is no buzzer, no
   keep-out band; the fin lying on a shelf is a stable, expected configuration,
   not a fault.
2. **Avoidance and clearance control are worthless.** The sleeve cannot descend
   past a ward at all — however slowly or carefully — unless its fin is rotated
   into that ward's single 60° notch. Smoke check 5 runs the solve's *own*
   gentle descent servo while misaligned for 3 s: z does not move a millimetre.
   The seed's entire skill transfers zero.
3. **The skill is discrete keyed engagement, twice.** Each ward's notch yaw is
   sampled independently per episode (th1 anywhere on the circle; th2 offset by
   55–180° either side). Progress = read the notch, rotate the collar about the
   post axis (~±10° tolerance), lower through, land on the next shelf, repeat
   with a *different* answer, then seat on the base plinth.
4. **The execution order is forced by topology, not convention.** The hub is a
   closed loop around the post: there is no lateral path, so the seat is
   reachable only DOWN through ward 1 then ward 2, in that order. The only
   bypass the topology leaves — lifting the sleeve off the *top* of the post
   and putting it down next to the plinth — is physically constructed in smoke
   check 7 and earns exactly 0: every latch and success() requires the
   `threaded()` readback (hub actually around the post).

So the plan skeleton changes from *"guide one object along a curve, maximizing
clearance, contact = fail"* to *"decode two randomized notch angles and execute
two align-then-lower keyed engagements in a topology-forced order, contact-borne
throughout"*.

## Why strategically different from the examined sibling

Sibling examined in tasks_v7: **beat_the_buzz_i182** (pressure-plate heist).
i182's decisive content is a **load invariant** (keep force on a plate above a
spring threshold at every instant), a **mass discrimination** (granite vs foam),
a **temporal order enforced by statics** (counterweight before removal) and a
**permanent alarm latch** that zeroes all credit. Here there is no load to
maintain, no mass inference, no substitution, and no alarm — nothing is
permanent-failure. The decisive content is instead **geometric keying**: two
independently sampled rotation answers, an alignment servo about a fixed axis,
and an order that comes from *topology* (a closed loop on a post between two
apertured shelves), not from statics. Conversely i182 has no captive kinematics,
no rotation-keyed apertures and no multi-stage traversal of the same mechanism.
The two tasks share only the seed.

## Solution outline (solve.py — no teleports at all)

The sleeve is captive on the post, so the whole solve is wrench control through
the scene's `drive_f`/`drive_t` buffers (the stand-in for the Franka's grasp on
the fin, below), applied by `post_step`, which also latches the rubric every
step. Wrenches act one substep late, so gains obey K·dt/m « 1 (vertical
KV·dt/m = 0.22; yaw Kd·dt/I ≈ 0.4 at dt = 1/120 s).

- **Phase 0 (reset):** settle 0.5 s; read th1/th2/fin readbacks and the sleeve
  mass (0.150 kg); drives asserted zero; `SIM_GEN_SCORE` 0.000.
- **Phase 1 (upper ward):** yaw PD torque (Kp 0.03 escalating ×1.5 to 0.12 on a
  2 s timer, Kd 0.010, cap 0.06 N·m) rotates the fin toward th1 while the
  sleeve rests on the shelf. The fin corner-hooks on the notch edge and rattles
  (|wz| never low — a velocity-gated stall counter never fires), so escalation
  is time-based and the escalated gain carries across retries. The drop is
  detected by the z readback (−12 mm), then a velocity-servo descent
  (fz = m·g + 4·(−0.12 − vz), clamp [−2, 4] N) with a soft yaw hold lowers it
  onto ward 2; release, streak-gated settle. `SIM_GEN_SCORE` 0.300.
- **Phase 2 (lower ward + seat):** same align/descend to th2, landing seated on
  the plinth; release, settle, wait for `success()`. `SIM_GEN_SCORE` 1.000.
- **Phase 3 (persistence):** drives asserted zero, 3.5 simulated seconds
  hands-off; `success()` is live state. Only then `SIM_GEN_SOLVE: SUCCESS`.
- Recovery path (exercised in early runs, not needed by the final gains): if a
  descent wedges, lift back above the shelf, re-align at the escalated gain,
  retry (up to 4 attempts).

Verified on the forge: seeds 0, 1 and 2 all print the monotone sequence
0.000 → 0.300 → 1.000 → 1.000 and `SIM_GEN_SOLVE: SUCCESS` (~21 s each,
single-attempt on every ward).

**Declared abstraction:** the wards are kinematic collars (12 tangential boxes
covering 300°) and the tower is re-posed kinematically per reset so the notch
angles and tower xy are genuinely randomized; the sleeve is one dynamic compound
rigid body (8-box closed hub + fin) with authored mass/CoM, CCD, and low-friction
materials. The keying is real rigid-body geometry — misaligned, the fin lands on
the shelf ring and 2× body-weight presses do not pass it (smoke 6); aligned, it
falls through under gravity alone.

## Required execution order (declared)

Upper ward (th1) BEFORE lower ward (th2) BEFORE seat: the hub is a closed loop
around the post, so the seat is reachable only by descending through both wards
in top-down order. This order is geometric/topological, not conventional; the
only alternative route (off the post top) is unthreaded by construction and is
physically built and rejected in smoke check 7.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `p1_latch` (0.30): sleeve centre has been below the upper-ward pass plane
  while THREADED on the post.
- `p2_latch` (0.30): same for the lower-ward pass plane.
- `seat_latch` (0.20): sleeve has rested seated on the plinth (threaded,
  upright, settled).
- current success (→ 1.0): sleeve currently seated, threaded, upright, settled.
- Latches are NaN-safe transient-achievement credit (torch.maximum); they
  survive correct-behavior regression (smoke 9) and success is live state
  (smoke 11). Every term requires `threaded()`, so the off-post bypass earns
  nothing. Null policy ≈ 0.

## Embodiment argument (single Franka, parallel jaw, OSC)

Base at ≈ (−0.42, 0.0) on the bench top (z = 0.40), facing +x: the post axis is
0.49–0.55 m away (tower jitter ±30 mm), the fin tip orbits at that distance ±
0.105 m, work heights 0.44–0.62 m — everything inside a 0.75 m reach disc at
elbow height, approached from the open side of the bench.

- **Fin grasp:** the yellow fin (14 × 16 mm cross-section) extends 30 mm beyond
  the ward rim (fin tip r = 105 mm vs shelf r = 75 mm), a textbook top-down or
  side pinch for the 80 mm jaw with the wrist axis vertical; 0.15 kg total.
- **Rotation about the post axis:** grasp the fin tip and orbit the EE on the
  r ≈ 0.10 m circle centred on the post (readback/vision gives the notch gap —
  it is a visible 60° missing sector). The ±10° alignment tolerance is coarse;
  for offsets > ~120° regrasp once at mid-arc instead of running the wrist to
  its limit. A compliant alternative — pushing the fin's flat side faces
  tangentially — also works and needs no closure at all.
- **Lower-through:** with the fin in the notch, a straight −z Cartesian move at
  ≤ 0.12 m/s — exactly the vertical CoM servo the drive buffers stand in for —
  lowers the sleeve 80 mm to the next shelf; the post itself guides the hub
  (3 mm radial clearance), so lateral precision is not load-bearing.
- **Order constraint:** satisfied by sequencing alone; one arm, one object, no
  simultaneous actions, no forces beyond lifting 0.15 kg.

## Checks (smoke.py — rejection battery, 12 checks)

1. Clean reset: finite state, sleeve resting ON ward1 at the readback height,
   threaded, upright, score ~0.
2. Readback: ward yaws equal sampled th1/th2 (from the ward quats), fin equals
   th_s0, misalignment bands (≥ 55°) hold, sleeve mass 0.150 kg, tower xy in
   its jitter band.
3. Randomization is real (10 seeds): th1, the signed notch offset, the signed
   fin offset and the tower xy all vary; both offset signs drawn (pose
   readback).
4. Null policy: 2 s of no action — still resting on ward1, no latch,
   score < 0.05.
5. **Seed-strategy family (careful lowering):** the solve's own gentle descent
   servo, misaligned — the sleeve does not pass (z unchanged, no latch).
6. Near miss: fin rotated to ~30° off the notch (actuator provably moved; the
   path never crosses the notch) plus a 2× body-weight press — no pass, no
   latch.
7. **Bypass:** the sleeve servo-lifted clean off the post top, transported and
   released beside the plinth — settles at seat height, upright, but
   UNTHREADED: zero credit, no success.
8. Partial credit: real phase-1 to a rest on ward2 — score == 0.30 exactly, no
   success.
9. Latch retention: sleeve lifted back UP through ward1 (beyond the latch
   plane) and parked misaligned on ward1 — p1 survives at 0.30, no success.
10. Exactness: full correct strategy → success() and |score − 1.0| < 1e−3.
11. Live success: the seated sleeve lifted back off the plinth and parked on
    ward2 — success revoked, score falls back to the latched 0.80.
12. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
