# light_bulb_in_i120 — Deliver the bulb through the lantern's rotary transfer turnstile

**Scene:** `simgen.lamp_turnstile` (`LampTurnstileScene`, robot="null")
**Seed:** `rlbench/light_bulb_in`
(`RoboVerse/roboverse_pack/tasks/rlbench/light_bulb_in.py`)

## Seed provenance and what changed

The RLBench seed is a direct install: two bulbs on stands, a lamp with an open
socket, and the plan is **grasp the bulb, carry it to the lamp, insert it into
the socket and screw it down** — one grasp, one free-space transfer, one
continuous wrist rotation ON the held object, judged by the bulb seated in a
fixed, always-reachable socket.

This task keeps the goal predicate ("the bulb ends up seated in the lamp's
socket") but replaces the entire plan skeleton:

1. **The socket is unreachable.** The lamp is a sealed cabinet (walls, roof,
   and a front service window permanently blocked by the diametral vane of a
   rotary transfer turnstile — a bank-hatch mechanism). The socket cradle is
   mounted ON the turnstile's far half, inside. Carrying the bulb to the lamp
   and pushing achieves nothing: smoke check 6 pushes the bulb straight at the
   closed window with 3× its weight and it jams on the vane by real collision;
   check 7 drops it from above and it lands on the roof. The seed's
   carry-and-insert schema transfers zero.
2. **The mechanism, not the arm, transports the bulb.** The only way in is to
   operate the turnstile: half a turn brings the empty cradle OUT through the
   window, the bulb is set into the open well, and half a turn back carries the
   loaded cradle INSIDE. The ordering (turn out → load → turn in) is physically
   forced by the seal — the cradle cannot be loaded while inside, and the bulb
   cannot enter except riding the cradle.
3. **Nothing is screwed.** The seed's signature verb (continuous screwing of
   the held object) is absent; the rotation here is a *gross transport of the
   carrier mechanism* (two free half-turns of a turnstile, pushed at handle
   pegs), not a fine wrist rotation of the grasped object, and the bulb itself
   is placed by a simple release into an open well.
4. **Live mechanism, not scripted state.** The turnstile is a plain dynamic
   body on a spawn-authored free vertical revolute joint (no motor, no spring,
   no detent flag); angular damping parks it wherever it is left, and the lit /
   service indices are read from its live pose. The seal is real geometry:
   every gap around the closed vane (vane–jamb 4.8 mm, vane–header 4 mm,
   sill–platter 5 mm) is far smaller than the 48 mm bulb — asserted in
   `__post_init__`, force-probed in smoke.

So the plan skeleton changes from *"grasp, carry, insert, screw"* to *"operate
a carrier mechanism out through a sealed boundary, load it, operate it back
in"* — indirect, mechanism-mediated delivery with a physically forced
three-stage order, instead of a direct manipulation of the goal object into a
fixed receptacle.

## Why strategically different from every examined sibling

- **Seed (`rlbench/light_bulb_in`):** see above — the seed's whole strategy
  (bring the bulb to the socket, insert, screw) is physically rejected by the
  sealed cabinet (smoke checks 6–7); the delivery is by mechanism transport.
- **`lamp_off_i101` (twistlock unplug):** there rotation is a small keying
  twist that *gates a translation* of the grasped plug, with cord-trace
  identification and a keep-alive decoy; extraction OUT of a mechanism. Here
  there is no identification, no decoy, no interlock keying: rotation is the
  *gross transport stroke of a carrier* (two half-turns), the object is never
  pulled through a mechanism, and the goal is delivery INTO a sealed volume
  with a forced out–load–in sequence.
- **`screw_nail_i59` (latch vault):** there the structure is unlock-and-remove
  — slide two latches, lift a lid OFF, extract a block and relocate it to an
  open dish; every stage removes constraints. Here nothing unlocks and nothing
  is disassembled: the boundary stays sealed the whole time, the moving part is
  the *delivery vehicle itself*, and success requires the mechanism returned to
  a specific configuration (lit index) with the object riding it.
- **`robobench/packing/pen_holder` (exemplar, not corpus):** direct drop of
  free objects into an open fixed receptacle — exactly the schema this task's
  sealed cabinet removes.

## Solution outline (solve.py — teleport = TRANSPORT ONLY)

External-wrench plant recipe (all three, or the servo stalls/rings):
`enable_external_forces_every_iteration: True` in the scene's physx cfg; hinge
rate finite-differenced from yaw (`root_ang_vel_w` is phantom under external
wrenches); carousel diagonal inertia authored (Iz = 0.0026 → kw·dt/Iz ≈ 0.26 <
1, discretely stable).

- **Phase 0 (reset):** settle 0.5 s, read back yaw / tray / bulb, assert score
  ≤ 0.02. `SIM_GEN_SCORE` ≈ 0.000.
- **Phase 1 (turn OUT, dynamics):** rate-cascade torque servo on the
  turnstile's own free DOF (the stand-in for pushing a red handle peg along its
  arc): outer loop ω_des = clamp(3·err, ±1.2 rad/s), inner τ = clamp(0.08·
  (ω_des − ω_fd), ±0.25 N·m); GAIN escalates ×1.5 on stall (never the cap);
  release only near target AND slow (|ω| < 0.05 → damping-1.0 coast ≈ 3°,
  inside both index tolerances). Parks at the service index.
  `SIM_GEN_SCORE` 0.200.
- **Phase 2 (LOAD, transport + drop):** the bulb is teleported across free
  space to 55 mm above the now-exposed cradle well — outside the cabinet,
  above an open well — zero velocity, released. Gravity + brass/platter
  contact seat it; the seating is never spawned. `SIM_GEN_SCORE` 0.500.
- **Phase 3 (turn IN, dynamics):** the same servo (rate cap 0.9) rotates the
  LOADED turnstile back to the lit index — the mechanism carries the bulb
  through the window into the cabinet. `SIM_GEN_SCORE` 1.000.
- **Phase 4–5 (persistence):** wrenches zero, ≥ 3.5 simulated seconds
  hands-off; `success()` is live state. Only then `SIM_GEN_SOLVE: SUCCESS`.

**Declared abstraction:** the turnstile rides a spawn-authored free vertical
revolute joint (kinematic housing anchor, pair collision disabled — the
clearances are asserted; the joint constrains the DOF). The vertical axis makes
gravity neutral along the DOF for any load, so no motor/spring/detent is needed
and the loaded turnstile behaves exactly like the empty one.

**Execution order is REQUIRED and physically forced:** load-before-turn-out is
impossible (the cradle is inside the sealed cabinet — smoke 6/7 prove force
cannot substitute), and turn-in-before-load just closes an empty cradle (no
seat credit, no success). The rubric's latches mirror exactly this order.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `out_latch` (0.20): turnstile ever parked at the service index (cradle out).
- `seat_latch` (0.30): bulb ever seated in the cradle well (carousel-frame
  containment window — accepts every physically-in-well resting pose, rejects
  beside-the-wall and perched poses; asserted in `__post_init__`).
- `carry_latch` (0.20): bulb ever seated AND past the window plane (riding the
  cradle inside).
- current success (0.30): bulb in the cradle AND turnstile at the lit index
  AND bulb + turnstile settled.
- Latches are transient-achievement credit (torch.maximum); cap 0.70 without
  success (float32 boundary honoured with +eps in smoke). Null policy ≈ 0
  (the turnstile spawns closed with the cradle inside, the bulb on its tray).

## Embodiment argument (single Franka, parallel jaw 80 mm, OSC)

Base at (0, 0) on the floor, facing +x: the tray slots (0.15–0.26 m) and the
turnstile's outer half (axle at x = 0.50, platter edge at 0.39, pegs sweeping
r ≈ 0.03–0.10 around it) are all within a 0.75 m reach disc; nothing behind
the window plane ever needs touching.

- **Turnstile operation:** each red handle peg (14 mm square section,
  protruding 45 mm from the vane faces at z ≈ 0.20) is pushed SIDEWAYS along
  its arc with a fingertip or closed-jaw poke — two 180° strokes, re-gripping
  the opposite peg at half-stroke if preferred. The 0.25 N·m servo cap
  corresponds to ≤ 2.8 N at the 0.09 m arm — a light fingertip push; the free
  joint + damping mean no precision is needed beyond parking inside ±25°/±18°
  windows.
- **Bulb pick:** the 48 mm frosted globe in an open 68 mm tray recess is a
  textbook equator pinch for the 80 mm jaw, approached from above.
- **Bulb place:** a drop from ~5 cm above the exposed 66 mm well (bulb Ø48 →
  9 mm per-side slack) — release accuracy of a couple of centimetres suffices;
  exactly what solve.py's release stands in for.
- **No blind reach:** every episode-variable quantity (turnstile angle, tray
  slot, bulb pose) is visible from outside; the cradle only ever needs
  touching while it is outside the cabinet.

## Checks (smoke.py — rejection battery, 16 checks)

1. Clean reset: states finite; turnstile closed near lit, bulb in tray, score ~0.
2. Score ~0 / no success at rest.
3. Randomization (6 seeds): turnstile start yaw physically posed and varying.
4. Randomization: tray slot (≥ 2 of 3) / xy / yaw vary; bulb tracks its tray.
5. Null policy: 240 idle steps, score ≤ 0.02.
6. **Seed-strategy family (straight insert):** regulated push (≤ 1.5 N ~ 3×
   bulb weight) at the closed window moves the bulb ≥ 15 mm (non-vacuous) then
   jams it on the vane by real collision — never past the window plane, ~0.
7. **Seed-strategy family (from above):** bulb dropped over the cradle's
   inside position lands on the ROOF and never reaches the cradle, ~0.
8. Loose inside: bulb constructed loose on the cabinet floor — inside but not
   in the cradle — earns nothing.
9. Near-miss: bulb settled on the platter beside the cradle wall at service —
   rejected by the xy window; only the out-index credit (≤ 0.20 + eps).
10. Wall perch: bulb balanced on a cradle wall top reads above the height
    window — rejected, removed before it topples.
11. Part-way + cap: out + seated + carried all latched, loaded cradle parked
    45° short of lit → score == 0.70 cap (+eps), NOT success.
12. Settle gate: bulb in the cradle at lit but moving — not success; removed.
13. Latched credit: bulb teleported away — latched score unchanged, in_ring
    drops.
14. Rejection audit: success() never True at any judged point.
15. Final no-NaN.
16. Camera: ≥ 20 rgb frames captured → `frames.npz`.
