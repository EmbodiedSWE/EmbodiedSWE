# close_fridge_i360 — Spring-loaded door, slide-bolt keeper latch

**Scene:** `simgen.spring_latch_fridge` (`SpringLatchFridgeScene`, robot="null")
**Seed:** `rlbench/close_fridge`
(`RoboVerse/roboverse_pack/tasks/rlbench/close_fridge.py`)

## Seed provenance and what changed

The RLBench seed is a fridge with an open hinged door and one goal predicate: door
angle ≈ 0. Its plan is **one uncontrolled push** — any shove that lands the panel on
its stop scores, and the closed door *stays* closed for free because nothing tries
to reopen it.

This task keeps the closing-a-fridge-door surface but removes the seed's silent
load-bearing assumption — *"a closed door stays closed"* — and replaces the one-DOF
push with a **two-DOF serial mechanism problem**:

1. **The door fights back.** The hinge carries a software return spring with an
   OPEN equilibrium (θ_eq sampled 42–72° per episode, k = 1.0–1.6 N·m/rad, plus
   viscous damping). An unbolted door parked at 0° is *unstable*: release it and
   the spring swings it back open. The seed's own strategy — push the panel shut
   and leave — is physically self-defeating; no push, however hard or precise,
   produces a lasting closed state (smoke checks 5 and 6 demonstrate both a slam
   and a perfectly servo-landed flush press reopening past 25° on release).
2. **A second DOF secures the first.** A spring-free slide bolt rides a prismatic
   joint on the door's free edge (50 mm travel along door-local y). Opposite it,
   on the static cabinet wall, a keeper housing with a blind bore (4 mm play
   around the 20 mm bolt tip). Success = bolt tip ≥ 18 mm into the bore AND door
   within 3° of its stop AND settled — the bolt carries the spring load, not a
   hand.
3. **Execution order is forced by collision geometry, twice over.**
   - *Retract before press:* the bolt spawns extended (ext₀ = 32–48 mm, sampled).
     The keeper housing's front face is proud of the door's swing corridor: an
     extended bolt tip rams it and **jams the door ≈ 5–7° short of flush**
     (measured 6.7° on the forge — outside both the 3° success band and the 2°
     close-latch band). The door physically cannot close until the bolt is pulled
     back.
   - *Press before throw:* the bore's 4 mm play at a 0.49 m/rad swing-misalignment
     lever means the bolt can only enter the bore while the door is held flush
     (≲ 0.5°). Throwing the bolt with the door even 6° ajar just parks the tip on
     the housing face (smoke check 7).
   So the plan skeleton is *retract → press-and-hold → throw → release*, a serial
   chain where the articulation being closed is not the goal — it is the thing the
   goal mechanism must capture.

So the plan changes from *"push panel"* to *"operate a two-DOF latch mechanism
against a live restoring load, with the hand-off from finger to mechanism as the
actual deliverable."*

## Why strategically different from every examined sibling

Corpus tasks examined (tasks_v7), including the same-seed sibling:

- **close_fridge_i89 (same seed):** a *payload-retention* task — an egg rides the
  closing door and a speed bound emerges from ejection physics. The door there is
  passive (springless, parks wherever left); closing slowly with one push
  suffices, and nothing must be *secured*. Here there is no payload and no speed
  bound; the hard part is that the closed state is unstable until a second
  mechanism DOF is thrown, and the finger must not leave until the bolt carries
  the load. Different failure mode, different plan skeleton, different physics.
- *open_oven_i7 / oven dials:* doors are goal objects; nothing re-opens them and
  no second DOF secures them.
- *pin + self-closing tray:* the spring there *helps* (self-closing) and the pin
  is removed, not engaged; order is one-way. Here the spring *opposes* the goal
  and the bolt must be retracted first and thrown last — a retract→re-engage
  round trip on the same DOF.
- *bayonet twist-lock / lid + turn-tab latch:* engagement by rotation of the goal
  object itself, no restoring load fighting the closure while it is secured.
- Drop-gate release, log-cabin build, crate inversion, shuttle metering,
  letterbox rack, dice tumbling, detent dialing, rammer gallery: no
  spring-loaded closure member, no engage-a-bolt-under-load mechanism.

Same-strategy-different-numbers this is not: the seed has one DOF, no restoring
spring, no jam geometry, and no notion of "secured vs merely closed".

## Solution outline (solve.py — NOTHING teleported; force-only through the plant)

All phases act through one buffer: `bolt_force`, a BODY-frame force on the bolt's
knob — i.e. a single fingertip contact (body frame == door frame: +x presses the
door closed through the prismatic joint, ±y slides the bolt).

- **Phase 0 (reset):** settle 0.5 s at θ_eq; assert both drive buffers zero;
  `SIM_GEN_SCORE` ≈ 0.000.
- **Phase 1 (retract):** 2.5 N pull along −y until ext ≤ 5 mm (against the bolt's
  own stop). The swing corridor is now clear. `SIM_GEN_SCORE` 0.250.
- **Phase 2 (press):** velocity-servo fingertip press, +x force =
  clamp(−τ_des/L, 0, 8 N) with τ_des = k·(θ−θ_eq) − 0.4 + 3.0·(ω_des−ω),
  ω_des = −min(0.5, 1.5·θ): cancel the spring, close at ≤ 0.5 rad/s, land softly
  at ≤ 0.3°. A −0.8 N y-bias rides along (diagonal push pinning the free-sliding
  bolt against its retract stop — centrifugal ω²r otherwise creeps it outward and
  re-jams). 8 N at the 0.39 m lever = 3.1 N·m authority vs ≤ 2.0 N·m spring
  torque: a one-finger push. `SIM_GEN_SCORE` 0.600 (door HELD, not secure).
- **Phase 3 (throw):** the press servo KEEPS running (release now and the spring
  reopens the door) while +3 N on the knob slides the bolt into the bore. At
  tip ≥ 18 mm past the mouth, ALL forces zero at once: the spring shoves the door
  a fraction of a degree back onto the bore wall — the bolt now holds the door
  (measured rest: 0.5° with residual rate ≈ 0.03 rad/s). `SIM_GEN_SCORE` 1.000.
- **Phase 4 (persistence):** both buffers asserted zero, ≥ 3.5 simulated seconds
  hands-off; success() is live state. Only then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 (θ_eq 71.2°, k 1.47, ext₀ 39.5 mm) and 1
(θ_eq 62.2°, k 1.14, ext₀ 43.5 mm) both print the monotone score sequence
0.000 → 0.250 → 0.600 → 1.000 → 1.000 and `SIM_GEN_SOLVE: SUCCESS`.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `retract_latch` (0.25): the bolt has ever been fully retracted (ext ≤ 10 mm).
- `close_latch` (0.35): the door has ever been within 2° of its stop.
- current success (0.40): bolt tip engaged ≥ 18 mm into the bore AND door within
  3° AND door + bolt settled. Latches are forced full whenever success holds, so
  the constructed goal scores exactly 1.0; a bolt pulled back out revokes the
  0.40 and the score falls to the latched 0.60. Null policy scores ~0 (door
  parked open at θ_eq).

## Embodiment argument (single Franka, parallel jaw, OSC)

Base at ≈ (−0.55, 0.15, 0), facing +x toward the fridge front; door edge, knob
and keeper all lie within a 0.85 m reach disc across the whole swing.

Every solve-phase force is one fingertip on the **bolt knob** (a 20 mm tab on the
bolt, protruding from the door's front face — reachable at any door angle):

- *Retract:* 2.5 N hook-pull along the door face (−y door-local) — fingertip in
  front of the knob, wrist following the door frame.
- *Press-and-hold:* 8 N cap normal push on the knob face plus 0.8 N tangential
  bias — a single diagonal fingertip press, ≈ 0.14 m/s max tip speed at the
  0.39 m lever under the 0.5 rad/s rate cap. Trivial OSC tracking.
- *Throw:* +3 N tangential slide with the same fingertip while the press holds —
  the classic thumb-throws-the-bolt motion.
- *Release:* lift the finger; the mechanism holds.

No regrasp, no second arm, no tool: the entire task is one finger operating a
door-mounted slide bolt. Execution order is forced by geometry (jam + bore
play), not by instruction sequencing.

## Checks (smoke.py — rejection battery, 12 checks)

1. Clean reset: finite state, door settled at θ_eq, bolt at ext₀, score < 0.05.
2. Randomization by readback, 3 seeds: θ_eq, ext₀, k all differ (max-pairwise).
3. Null policy: 2 s of no action, door stays at θ_eq, score < 0.05.
4. Spring is real: door driven to 20° and released returns to within 6° of θ_eq.
5. **Seed-strategy rejection (slam):** −4 N·m shove with the bolt extended jams
   at 3.5–12° (measured ≈ 6.7°, outside both rubric bands); on release the
   spring reopens > 25°; score stays < 0.05.
6. **Push-only rejection:** retract + perfect servo-landed flush press, held,
   then released — the door reopens > 25°, no success, score ∈ [0.55, 0.625]
   (latched credit only — a pusher cannot reach 1.0).
7. Ajar throw: door held 6° open, bolt thrown hard — full extension but NOT
   engaged (tip parks on the housing face short of the bore); door reopens.
8. Under-engaged near miss: flush press + bolt servo-fed to 28 mm ext (tip
   ~13 mm past the mouth < 18 mm engage_min), released — not engaged, no
   success, score ≤ 0.625.
9. Exactness: constructed goal state (door 0.4°, ext 44 mm) yields success and
   |score − 1.0| < 1e−3.
10. Latch semantics: pulling the bolt back out of the engaged goal revokes
    success — door reopens > 25°, score falls to the latched 0.60.
11. Full bolt extension with the door wide open scores < 0.05 (no credit for
    waving the bolt around).
12. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
