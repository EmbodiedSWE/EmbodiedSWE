# get_ice_from_fridge_i397 — Sealed ice caddy, probe-operated poppet valve, dock-and-drain

Env id: `simgen.ice_caddy_dock` (robot="null", scene-level). Files: `scene.py`,
`solve.py`, `smoke.py`. Forge-verified: `SIM_GEN_SOLVE: SUCCESS` on seeds 0 and 1,
`SIM_GEN_SMOKE: ALL PASS 13/13`.

## Seed provenance and what changed

Seed: `rlbench/get_ice_from_fridge` — hold a cup against a fridge's ice-dispenser
lever; ice falls into the cup. The seed's whole plan is one sustained press at a
fixed machine: the container is a passive receiver and the machine is the source.

Here that relationship is inverted and given state. The ICE SOURCE ITSELF is the
carried object: a closed-top caddy (2-4 ice balls, count sampled per episode) whose
only opening is a bottom port sealed by a spring-loaded poppet plug. Nothing done to
the caddy in free space discharges it — the roof is solid and the spring preload
out-pulls the plug's weight in every orientation. The one discharge path is
mechanical: seat the caddy on the blue dispensing DOCK, whose fixed probe post rises
through the bottom port and pushes the valve open against its spring under the
caddy's own weight. An identical-looking DECOY basin (no probe) seats the caddy but
opens nothing. After gravity drains every ball into the dock basin, lifting the
caddy lets the spring re-seal it, and it must finish parked upright on the green pad.

## Why strategically different from every examined sibling

- The seed (`get_ice_from_fridge`): memoryless single press at a fixed dispenser; no
  transported container state, no ordering, no discrimination between fixtures.
- Sibling `get_ice_from_fridge_i52` ("ice_doser"): a FIXED machine with a
  reciprocating shuttle and an exact-count objective. Here there is no machine-side
  actuator and no counting at all — the carried container carries the mechanism, the
  valve is operated PASSIVELY by fixture geometry, and a decoy forces recognition.
- The plan skeleton is a forced serial program with a physical interlock (lift →
  seat on the CORRECT dock → dwell → lift off → park), not a rubric-encoded order:
  balls physically cannot leave except through the probe-opened seat, and success
  requires the re-sealed caddy AWAY from the dock, so draining must precede parking.

## Mechanics (scene.py)

- Caddy: ONE dynamic compound body (0.50 kg) — roof root, walls, base flange with a
  72 mm square port, four slick 30° funnel plates, bridge handle (14 mm bar, 30 mm
  knuckle clearance). Balls feed by gravity: pair-averaged friction on the slick
  funnel is below tan 30°.
- Plug: second dynamic body (40 g, slick) — 88 mm head (out-spans the 72 mm throat,
  so it can never pass through), stem recessed 2 mm above the base plane (no floor
  or finger reaches it while the caddy stands), 45° ridge tent sheds balls off the
  head. Rides an authored prismatic joint (caddy z, travel 0-34 mm, 0 = sealed) with
  a linear DriveAPI return spring whose target sits below the lower stop (preload).
- Dock and decoy: kinematic look-alike basins; rim at 100 mm seats the caddy's
  out-spanning flange; a slick registration collar (±5 mm play) self-centres the
  descending flange — without it the caddy tips and skates off the slick probe pin.
  Only the dock has the probe post (top = rim + 30 mm). Seating geometry: probe top
  minus seated stem tip = 28 mm lift, and caddy weight ≥ 1.5× the spring force at
  that lift (asserted).
- 12 geometry/force-margin assert groups run at import (`__post_init__`): seal
  overlap, feed gap ≥ ball_d + 6 mm, throat clearances, seating force margin,
  any-orientation seal margin, stem recess, flange-out-spans-basin, stroke band,
  hover clearance, ridge ground clearance, collar play/overlap, collar entry.
- Randomization (seed-driven, physically read back): dock side ±y and xy jitter,
  mirrored decoy, pad xy, caddy xy + free yaw, ball count 2-4.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- 0.12 — lift latch: caddy raised above 0.16 m.
- 0.18 — dock-open latch: valve ext > 15 mm while the caddy is at the dock (only
  the probe can do this with the caddy seated somewhere).
- 0.40 — delivered fraction: per-ball latch, ball settled inside the dock basin.
- 0.30 — success(), live state: every present ball in the basin AND caddy parked
  upright on the pad AND valve re-sealed (ext ≤ 6 mm) AND everything settled.
  Success forces all latches full; removing a basin ball afterwards revokes it.

## Solution outline (solve.py — teleports are TRANSPORT ONLY)

Transports relocate the caddy+plug(+balls) through free space preserving their
current caddy-frame offsets, zero velocity — exactly what a carrying hand does; the
valve state is never written. All load-bearing interaction is live contact dynamics:

1. SEAT — transport to 35 mm above the dock, DROP. The probe meets the stem through
   the port and pushes the valve open against its spring under the falling caddy's
   weight; a 2.5 N hold-down (half the caddy's weight) steadies it.
2. DRAIN — dwell, hold-down on. Gravity feeds the balls down the funnel, under the
   lifted head rim, past the probe into the basin (3 Hz, ≤1.5 N wiggle if one
   dawdles). Waits for every present ball's delivery latch.
3. PARK — 8 N straight-up pull unseats the caddy; in free flight the spring
   re-seals the valve unscripted (printed telemetry); transport to 30 mm above the
   pad (balls STAY in the basin), drop, settle to success().
4. PERSIST — all force buffers asserted zero, 3.5 simulated seconds hands-off,
   success() and score == 1.0 re-checked live, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary; asserted non-decreasing.

## Embodiment argument (single Franka, parallel jaw)

Base at ≈ (−0.25, 0, 0) facing +x: caddy spawn (0.13±0.02, ±0.30±0.02), dock
(0.38±0.02, ±0.22±0.02), pad (0.12±0.02, ±0.03) all inside a 0.72 m reach disc.

- Caddy — the only object the robot touches. Grasp: the 14 mm bridge crossbar
  (fingers span 80 mm; 30 mm knuckle clearance below the bar; grippy material), bar
  at z 0.148 standing, 0.248 seated. Loads: 0.5 kg carry, 2.5 N hold-down, 8 N
  lift, 1.5 N wiggle — one-hand scale, far under Franka payload. The registration
  collar gives ±5 mm docking tolerance, well within arm repeatability.
- Plug — never commanded. Operated passively by the dock probe; the spring does
  every re-seal. (The solve's plug force buffer is smoke instrumentation only.)
- Ice balls — never touched; gravity feeds them. Feed gaps carry ≥ 6 mm margin
  over the ball diameter.
- Dock / decoy / pad — static fixtures, never moved.

## Execution order (declared)

lift caddy → seat on the REAL dock (decoy is a dead end) → dwell until drained →
lift off (spring re-seals in flight) → park upright on the pad → hands off.
The order is physics-forced: no valve opening away from the probe reaches the
basin, and success requires the caddy sealed, parked, and elsewhere.

## Checks (smoke.py — rejection battery, 13 checks, frames.npz recorded)

1. Clean reset: finite state, caddy standing sealed, every present ball inside it.
2. Randomization: dock/decoy/pad/caddy poses + yaw + ball count physically read
   back across seeds 20/21/22; max-pairwise feature distance; decoy mirrors dock.
3. Null policy: 2 s of nothing — caddy stays put, sealed, score < 0.02.
4. Sealed under abuse: held inverted and shaken 2.5 s (30 mm, 2.5 Hz) — valve
   never opens a ball-passable gap (needs ext ≥ 22 mm), no ball escapes, spring
   re-seals in-hand, zero credit.
5. Decoy: seats identically but the valve NEVER opens (ext peak < 6 mm), nothing
   drains, no dock-open credit.
6. Force-open away from the dock (emptied caddy, instrumentation force): the valve
   DOES open fully (actuator moved, non-vacuous) — zero credit, spring re-seals.
7. All balls dumped in the DECOY basin, caddy parked: score ~0.
8. Partial delivery (one ball left inside the parked caddy): no success, score =
   delivered fraction exactly.
9. Flagship physics: seat on the REAL dock, then HANDS-OFF — the probe holds the
   valve open and every ball drains by gravity alone; caddy left on the dock caps
   at 0.70, no success.
10. Tipped: all delivered but the caddy dropped on its side on the pad — not
    parked, no success.
11. Exactness: constructed goal state → success() and score == 1.0.
12. Revocation: plucking one ball back out of the basin revokes success (1.0 →
    latched 0.70).
13. ≥ 20 camera frames recorded to frames.npz.
