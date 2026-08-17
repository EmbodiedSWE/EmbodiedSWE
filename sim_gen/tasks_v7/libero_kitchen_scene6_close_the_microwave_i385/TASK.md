# libero_kitchen_scene6_close_the_microwave_i385 — Geneva-indexed rotary-airlock vault feeder

**Scene:** `simgen.geneva_vault_feeder` (`GenevaVaultFeederScene`, robot="null")
**Seed:** `libero_90/libero_kitchen_scene6_close_the_microwave`
(`RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene6_close_the_microwave.py`)

## Seed provenance and what changed

The LIBERO seed is a kitchen microwave with an open hinged door and one goal
predicate: door joint angle ≈ closed. Its plan is **one uncontrolled push** on a
1-DOF free hinge — any shove that lands the panel on its stop scores.

This task keeps the seed's kernel — *make a rotating panel arrive at a prescribed
station* — but replaces the free 1-DOF hinge with a **4-slot Geneva intermittent-
motion transmission**, and makes the panel's arrival not the goal but the *transport
mechanism* for cargo:

1. **The goal DOF cannot be touched.** The paddle wheel (a drum with four radial
   vanes dividing the narrow ball corridor between the drum and a blue inner guide
   ring into four compartments) is sealed under a roof, inside a 12-gon shroud and
   skirts. The only openings are the yellow roof port (balls in), the deck drop
   hole (balls out, into the enclosed vault under the deck) and the underdeck pin
   slit — none passes a finger to a rotating face, and the pin slit sits above
   ball-top height so nothing escapes through it. The seed's strategy — push the
   panel — is physically unavailable.
2. **The only drive is quantized.** The external red crank turns a shaft whose
   underdeck pin engages radial slots on the wheel's slot plate: a classic 4-slot
   Geneva. One full crank revolution advances the wheel **exactly one quarter
   turn** and parks it (wheel velocity is kinematically zero at pin entry and
   exit; peak ratio 2.41 mid-engagement). You cannot half-index: outside the
   90° engagement arc the crank spins freely and the wheel does not move (smoke
   check 4 rocks the crank ±15° in the disengaged arc — the wheel holds).
3. **The plan is a routing program, not a push.** Each present ball (k ∈ {1,2},
   sampled per episode) must be dropped through the roof port into the compartment
   parked beneath it (azimuth 90°), then indexed **twice** — port → mid station
   (180°) → drop hole (270°) — where it falls into the vault. With k=2 the loads
   and indexes interleave (load, index, load, index, index): the port only feeds
   the station parked under it, so execution order is forced by the mechanism's
   geometry, not by instruction sequencing.

So the plan changes from *"push panel onto its stop"* to *"operate a quantized
intermittent transmission to route cargo through a sealed rotary airlock."*

## Why strategically different from every examined sibling

Corpus tasks examined (tasks_v7), including the same-seed sibling:

- **microwave_ballast_door_i114 (same seed):** the door there is still directly
  pushed shut; the twist is a counterweight that must first be ballasted with
  mugs. The goal DOF is hand-accessible and continuous. Here the goal rotor is
  sealed away and only reachable through a motion-quantizing transmission, and
  the rotor's angle is not even the goal — cargo delivery is.
- **microwave_carousel_i106:** a free platter rotated directly by friction-coupled
  pushes — continuous, hand-on-the-rotor, no transmission, no quantization.
- **stone_door_vault_i306:** a millstone *rolled along the ground* to cover an
  opening — transport of the door itself, no gearing.
- **drop_gate_oven_i4 / bayonet_canister_i5:** gate release and twist-lock lid —
  direct manipulation of the goal member.
- **crank_ejector (scotch-yoke) / scotch_yoke_ferry:** crank transmissions exist
  in the corpus, but both convert rotation to *continuous reciprocation*. The
  Geneva is the opposite phenomenon: intermittent motion with kinematic dwell —
  discrete quarter-turn indexing with the output parked and locked between
  engagements, which is what makes the load/index interleave meaningful.
- **carousel ride-delivery / detent dial / dog-clutch:** free rotor + canopy,
  friction detents, and clutch engagement respectively — none has a pin-and-slot
  external gear stage, none quantizes input revolutions into exact output
  quarter-turns, none forces a multi-cycle routing program.

No examined corpus task contains an intermittent-motion transmission or a
sealed rotary-airlock cargo router. Same-strategy-different-numbers this is not:
the seed has a free hinge, direct hand access, and a one-shot angle goal.

## Solution outline (solve.py — transport-only teleports; all work through contact)

Teleports only *stage* balls above the roof port (the same pick-and-drop a
gripper would do); every joule that moves the wheel comes from the crank torque
buffer through the pin-slot contact.

- **Phase 0 (reset):** settle; assert both drive buffers zero; `SIM_GEN_SCORE`
  ≈ 0.000.
- **LOAD i:** teleport ball i to (0, 0.245, 0.47) — over the port, on the ball-
  corridor centreline — and let it fall into the parked compartment. Loading
  alone caps the score at 0.9·0.28/k (`SIM_GEN_SCORE` 0.252 at k=1).
- **INDEX:** PI velocity servo on the crank (KV=1.0, KI=1.5 anti-windup,
  τ ≤ 2.5 N·m), cruise 1.2 rad/s in the disengaged arc, slowed to 0.35 rad/s
  within ±70° of the engagement centre — the slow-zone plus wheel damping kills
  the slot-clearance exit coast so the wheel parks on station (measured park
  errors ≤ 0.29°). Exactly one crank revolution (unwrap-tracked), then brake to
  rest and zero the buffer. Each index is asserted: wheel advanced 90° ± 12° and
  parked within 6° of a station.
- **Program:** for k=1: load, index, index. For k=2: load, index, load, index,
  index — the second ball rides one station behind the first.
- **Persistence:** after success() (every present ball in the vault AND settled),
  420 steps (≥ 3.5 s) hands-off with both buffers asserted zero, success
  re-checked live, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seed 0 (k=1) prints 0.000 → 0.252 → 0.450 → 1.000;
seed 1 (k=2, interleaved) prints 0.000 → 0.126 → 0.225 → 0.351 → 0.675 → 1.000;
both monotone, both `SIM_GEN_SOLVE: SUCCESS`.

## Rubric (score 0..1, latched; score == 1.0 iff success())

Per present ball: p_i = 0.28·loaded_latch + 0.22·mid_latch + 0.50·vault_latch
(latches forced by current vault state, so the constructed goal scores exactly
1.0). Score = 0.90·mean over present balls + 0.10·success(). success() is live:
every present ball inside the vault (z < 0.20, |x|,|y| < 0.37) and settled.
Null policy ~0 (balls never leave the staging ground). A ball removed from the
vault revokes the 0.10 success term (score falls to the latched 0.90). Loading
without any cranking caps at 0.252.

## Embodiment argument (single Franka, parallel jaw, OSC)

Base at ≈ (0.62, 0, 0), facing −x toward the rig; the staging slots
(0.58, ±0.28), the roof port (0, 0.21..0.28, z 0.42) and the crank knob orbit
(centre (0.30, 0), r = 0.11, z 0.51..0.60) all lie within a 0.85 m reach disc.

- *Load:* pick a 25 mm, 60 g ball from its staging slot (standard sphere pinch),
  carry, release over the yellow port collar — a plain pick-and-drop; the collar
  funnels the drop.
- *Index:* grasp the 16 mm crank knob and drive it around its 0.11 m orbit.
  τ ≤ 2.5 N·m at r = 0.11 m ⇒ ≤ 23 N tangential — comfortably inside Franka
  payload, and the two-speed profile (1.2 / 0.35 rad/s ⇒ tip speed ≤ 0.14 m/s)
  is trivial OSC circle tracking. The knob stays at constant height; no regrasp
  is needed within a revolution.
- No second arm, no tool: the entire task is pick-drop-crank cycles. Execution
  order (load before index, one ball per compartment, two indexes per ball) is
  forced by the sealed geometry and the Geneva's quantization.

## Checks (smoke.py — rejection battery, 12 checks)

1. Clean reset: finite state, balls staged on the ground (x > 0.45, z < 0.06),
   wheel parked within jitter, score < 0.05.
2. Randomization by readback, 3 seeds (max-pairwise): crank phase > 5°, ball
   staging xy > 5 mm, wheel park jitter > 0.2°.
3. Null policy: 240 steps of no action, ball drift < 2 cm, score < 0.05.
4. Geneva dwell is real: crank rocked ±15° in the disengaged arc (position
   servo) swings > 10° while the wheel moves < 2° — the output is parked and
   locked between engagements.
5. Load-only cap: dropping a ball through the port without cranking scores
   0.9·0.28/k ± 0.03 — loading is not delivering.
6. **Sealed compartment (seed-strategy rejection):** a 0.3 N world-frame probe
   pushes the loaded ball tangentially (capped at 8° of wheel back-drive — the
   dwell is damped, not rigidly locked) — the ball moves ≥ 15 mm but stays
   captive between its two vanes (compartment-relative azimuth inside 40°..140°):
   no mid credit, no vault. You cannot shove cargo through the machine; only
   indexing moves it on.
7. **One index is one quarter-turn, and is not enough:** one full crank
   revolution advances the wheel 90° ± 12° and parks within 3°; the ball earns
   the mid latch (0.9·0.50/k ± 0.03) but is not in the vault, no success.
8. Exactness: constructed goal (present balls settled in the vault) yields
   success and |score − 1.0| < 1e−3.
9. Latch semantics: lifting a delivered ball back out of the vault revokes
   success — score falls to the latched 0.90 ± 0.02.
10. Pin slit does not leak: a vault ball pushed 0.6 N against the east skirt
    reaches the inner face (x_max > 0.333) yet stays in the vault — the slit is
    above ball-top height.
11. Roof is sealed: a ball dropped outside the port footprint lands ON the roof,
    never enters lane or vault, score < 0.05.
12. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
