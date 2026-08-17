# close_fridge_i89 — Egg-on-the-door-shelf gentle close

**Scene:** `simgen.egg_shelf_fridge` (`EggShelfFridgeScene`, robot="null")
**Seed:** `rlbench/close_fridge`
(`RoboVerse/roboverse_pack/tasks/rlbench/close_fridge.py`)

## Seed provenance and what changed

The RLBench seed is the plainest articulation task there is: a fridge with an open
hinged door, and the plan is **one uncontrolled gross-motion push** — swing the panel
shut, done. There is no payload, no rubric on the manner of the motion, and any
arrival speed is as good as any other.

This task keeps the closing-a-fridge-door surface but changes the **plan skeleton**,
not the numbers:

1. **The door becomes a vehicle.** A fresh egg (42 mm sphere, 55 g) starts on a
   counter beside the fridge. It must first be seated into one of two shallow egg
   cups on the **inner shelf of the open door** — the cup whose rim color matches
   the beacon block shown on the pedestal (BLUE = near-hinge cup, ORANGE = near-edge
   cup; the target is sampled fresh every episode and the non-matching beacon parks
   out of view).
2. **The manner of the close is load-bearing.** The egg is held in its cup by
   nothing but gravity and an 8 mm rim. The seed's own strategy — slam the door —
   is *physically self-defeating*: when a slammed door slaps its end stop the egg
   keeps its tangential velocity, and the arrest impulse at the low rim corner
   (13 mm below the egg's centre) redirects ≈ half of it upward — the egg vaults
   the rim and is gone. The wall height is a deliberately calibrated boundary:
   16 mm walls were *measured* to retain even a 1.7 m/s slam (the corner impulse
   is then nearly horizontal), so they were lowered to 8 mm, where the corner-vault
   threshold is ≈ 0.8 m/s of cup speed. A 4 N·m shove (≈ 10 N at the handle — an
   easy uncontrolled arm push) arrives at ≥ 6 rad/s ⇒ cup speeds ≈ 0.9–2.3 m/s,
   over threshold for **both** cups (verified for both in smoke check 6, on the
   settled aftermath). A careful 0.35 rad/s close keeps cup speeds ≤ 0.10 m/s,
   ≈ 8× under escape. Nothing in the rubric says "slowly"; the *ejection physics*
   enforces it.
3. **Execution order is forced by geometry.** The cups are on the door's inner
   face: when the door is closed they are sealed inside the cabinet behind a ≤ 8 mm
   slit (vs. a 42 mm egg), with ≥ 20 mm panel overlap on every edge. Load-then-close
   is the only reachable order; close-then-load is rejected **by unreachability**
   (demonstrated physically in smoke check 11), so the task needs no temporal
   bookkeeping in the rubric.

So the plan skeleton changes from *"push panel"* to *"load a loose payload onto the
closure member, then execute a speed-bounded transport with the articulation"* —
a dynamic-retention ("waiter's tray") constraint riding on an articulated close.

## Why strategically different from every examined sibling

Corpus tasks examined (tasks_v7): drop-gate release, bayonet twist-lock, log-cabin
stacking build, pin-removal + self-closing tray, crate inversion, lid + turn-tab
latch, shuttle count-metering (×2), letterbox rack loading, dice tumbling, detent
knob dialing, rammer gallery, oven door/dials. None of them contains this task's
core mechanism: **a passive loose payload that must ride the moving closure member,
with the payload's retention imposing a speed bound on the articulation**. The
closest neighbours differ structurally:

- *open_oven_i7* (same authored-hinge machinery): the door there is the **goal
  object itself**; nothing rides on it and speed is irrelevant.
- *letterbox rack loading / pen-holder style placement*: placement into a static
  receptacle; here the receptacle **moves after loading** and the placement can be
  retroactively undone by the closing motion.
- *pin + self-closing tray*: order is enforced by a latch mechanism; here order is
  enforced by pure containment geometry and the hard part is *after* the ordering.

Same-strategy-different-numbers this is not: the seed has no payload, no target
selection, no speed constraint, and no failure mode other than "door not closed".

## Solution outline (solve.py, teleport = transport only)

- **Phase 0 (reset):** settle 0.5 s; assert drive buffer zero; `SIM_GEN_SCORE`
  ≈ 0.000.
- **Phase 1 (load):** the egg is teleported from the counter to a hover pose ~3 cm
  **above the rim** of the sampled target cup (door-frame pose converted through
  the door's live world pose), zero velocity, then released — the drop, impact,
  rattle and settling inside the 8 mm walls are real contact dynamics. The egg is
  never spawned seated. `SIM_GEN_SCORE` 0.300 (seat latch).
- **Phase 2 (carry-close):** the load-bearing interaction runs entirely through
  the live plant. The solver writes the scene's `door_drive` hinge-torque buffer,
  clamped to **TAU_MAX = 1.2 N·m** (≈ 3 N tangential at the handle's 0.41 m lever
  arm — a single-fingertip OSC push). A velocity servo tracks
  ω_des = −min(0.35, 1.2·θ) rad/s, so cup speeds stay ≤ 0.10 m/s and the egg rides
  the door by nothing but friction + cup walls. The drive is **released at 1.5°**
  and viscous hinge friction parks the door on its stop, hands-off.
  `SIM_GEN_SCORE` 1.000.
- **Phase 3 (persistence):** with the drive buffer asserted zero, ≥ 3.5 simulated
  seconds hands-off; `success()` is live state (an egg popping out or a door
  drifting open would revert it). Only then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 (ORANGE cup, θ₀ = 98.8°) and 1 (BLUE cup,
θ₀ = 85.2°) both print the monotone score sequence 0.000 → 0.300 → 1.000 → 1.000
and `SIM_GEN_SOLVE: SUCCESS`.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `seat_latch` (0.30): the egg has ever rested settled inside the target cup.
- `carry_latch` (0.30): best closure progress (1 − θ/θ₀) reached **while** the egg
  was in the target cup (forced to 1.0 when closed with the egg aboard). An
  ejection en route keeps only the credit earned up to the eject angle.
- current success (0.40): egg in the target cup AND door within 3° of the stop AND
  egg + door settled. Null policy scores ~0 (egg on the counter, door ≥ 55° open).

## Embodiment argument (single Franka, parallel jaw, OSC)

Base at ≈ (−0.15, 0.0), facing +x; everything below is within a 0.85 m reach disc.

- **Egg (42 mm sphere, 55 g):** a textbook top grasp for the 80 mm parallel jaw
  from the counter at z ≈ 0.32; place is a free-space carry to the open door's
  shelf (door ≥ 55° open puts the cups at x ≈ 0.33–0.45, y ≈ −0.4…−0.2, z ≈ 0.30,
  approached from above with the wrist vertical — no cabinet interference while
  the aperture is fully open) and a 2–3 cm gravity drop into the cup. This is
  exactly the transport the solve's teleport-and-release stands in for.
- **Door close:** a fingertip/knuckle push on the handle bar or outer face near
  the free edge, tracking ≤ 0.15 m/s tip speed — the 1.2 N·m hinge-torque cap in
  solve.py is ≈ 3 N at the 0.41 m handle arm, comfortably single-finger OSC
  authority, and the 0.35 rad/s rate cap is ≈ 0.14 m/s at the handle. The Franka
  can trivially go faster; the *task* is what demands it doesn't.
- **Execution order:** load, then close — forced by geometry (sealed cups), not by
  instruction sequencing.

## Checks (smoke.py — rejection battery, 15 checks)

1. Clean reset: finite state, egg on the counter, door at θ₀, score < 0.05.
2. Randomization by readback: target cup, θ₀, egg xy all differ across seeds.
3. Beacon logic: matching beacon on the pedestal (< 5 mm), other parked off-view.
4. Null policy: 2 s of no action, score stays < 0.05.
5. Calibration: the demonstrated gentle carry-close keeps the egg seated.
6. **Seed-strategy rejection (slam), two checks:** (a) seat the egg and drive
   4 N·m to the stop, for **both** cups — the settled aftermath shows the egg
   ejected (peak rate > 2 rad/s and cup speeds reported); (b) with the ejected egg
   parked on the cabinet floor and the door closed, no success and score ≤ 0.62
   (partial latched credit only — a slammer cannot reach 1.0).
7. Near miss: egg seated, door servo-landed ~7–9° ajar (near-zero rate, so the
   springless hinge keeps it there) — no success, score < 0.90.
8. Wrong cup: egg seated in the non-matching cup, door closed — no success, < 0.10.
9. Egg loose on the cabinet floor, door closed — no success, score < 0.10.
10. Egg on the shelf strip **between** the cups, door closed — no success, < 0.10.
11. Order forcing, physically: close the empty door, drop the egg at the doorway —
    it cannot enter (readback: egg stays outside the aperture plane).
12. Exactness: the constructed goal state yields success and |score − 1.0| < 1e−3.
13. Latch semantics: reopening the door past 35° revokes success; score falls to
    the latched 0.60, not 1.0.
14. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
