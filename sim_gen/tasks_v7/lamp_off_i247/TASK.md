# lamp_off_i247 — LampIsolator

**Scene name:** `lamp_isolator` · **Env:** `simgen.lamp_isolator` (robot="null")

## Seed provenance

Seed task: `rlbench/lamp_off` — "turn off the lamp": a lamp with a push button
sits on a table; the robot presses the button down and the lamp is off. One
pressing contact on the judged fixture, one act, judged by a button/joint state
directly under the fingertip.

## Strategic difference

The seed's entire plan — reach the lamp, press its switch — is **insufficient
here by construction**. The floor lamp in this scene has NO switch of its own;
its power is cut at a separate wall-mounted **isolator panel**, and the panel's
overcenter hub is recessed 24 mm deep between two cheek plates that physically
exclude any finger or palm (smoke check 6 drives a Franka-finger-sized probe at
the hub with 8 N for 2 s: it arrests on the plate rims above the funnel, throws
nothing, scores nothing). Doing anything at all to the lamp itself is judged
worthless: smoke check 5 shoves the lamp 25 N until it has visibly moved or
toppled — hub unmoved, score ~0.

Instead the hub must be actuated **through a tool the robot has to install
first**: a removable steel HANDLE BAR rests in a saddle stand out on the floor.
The bar's ball end fits through the 24 mm slot; success requires (1) fetching
the bar, (2) a blind peg-in-hole insertion — threading the Ø8 mm shaft down
through the slot into the hub's 12 mm square socket (2 mm/side slop, funnel
mouth) — and (3) levering the hub from its gravity-held ON stop (−15°) through
the gravity dead centre (0°, bore straight up) to past +100° toward the OFF
stop (+110°). The mechanism is **bistable**: released before dead centre it
falls back to ON (smoke check 8 places it at −4° and watches it return), past
dead centre it commits to OFF by gravity alone (smoke check 9 releases at +25°
and success arrives hands-off). Insertion itself cannot cheat the throw — the
bore axis passes through the axle, so insertion pushes exert ~zero throw torque
(asserted in the solve: after seating, hub within 8° of ON).

Differs from the sibling `lamp_off_i101` (chain-pull actuation on the lamp
itself) and from `close_door_i162` (rod dropped through two static channels to
bar a door — the rod is the *outcome* there; here the bar is a *tool* and the
outcome is a hub angle across a bistable overcenter throw). No other examined
tasks_v7 package installs a removable handle into a recessed hub and works the
hub through a dead centre.

## Assets (fully procedural)

- **panel** — DYNAMIC 40 kg (root MassAPI; dynamic because a joint anchored to
  a kinematic body0 stays world-fixed when the body is teleported at reset):
  plinth 0.36×0.30×0.06, two cheek plates 0.28×0.018×0.28 at y ±0.021 forming a
  24 mm slot (x ±0.14, z 0.06–0.34), visual axle stub.
- **hub** — DYNAMIC rotor on a spawn-authored `UsdPhysics.RevoluteJoint` at
  panel-local (0,0,0.20), axis Y, limits −15°/+110°, joint pair
  collision-filtered (hub never touches the panel). Body: square bore (12 mm
  inner, walls z 0.016–0.062 above the axle in hub frame), backstop, boss
  cylinder, asymmetric funnel mouth z 0.062→0.075 (x half 0.012 / y half 0.009
  — the y funnel must live inside the 24 mm slot). Density 4000, angular
  damping 1.0. Gravity-loaded: the boss/bore mass (Σmr ≈ 0.018 kg·m) makes
  both stops attractors and 0° a repeller.
- **bar** — steel shaft Ø8×195 mm + ball Ø19 at the far end, origin at the TIP;
  ball > bore (19 > 16 mm outer) so it cannot fall through; seated ball stands
  proud of the plate rims (grip is never inside the recess).
- **saddle** — kinematic two-block stand with 18 mm ridge slots, holding the
  bar horizontal at rest z 0.058, standing U(0.42,0.58) m out on a random
  bearing; ball end faces either way (flag).
- **lamp** — floor lamp (base Ø0.26, 1 m pole, shade), 4.1 kg, on a random
  side bearing U(80°,140°) at U(0.70,0.90) m. Judged only negatively.

Randomization (readback-verified, smoke checks 2–3): panel yaw ±22° + xy
±50 mm; saddle bearing ±45° + radius + yaw jitter ±25°; bar ball-direction
flip; lamp side flip + bearing + radius. All geometry/clearance claims
(ball-corner swing clearance, finger exclusion margins, funnel-in-slot, bore
slop 1.5–4 mm, ≤175° travel, saddle CoM between supports, worst-case asset
separations) are asserted numerically in `LampIsolatorSceneCfg.__post_init__`.

## Rubric

- 0.15 `fetch` — bar tip ever above 0.16 m (latched)
- 0.25 `insert` — bar shaft ever seated in the hub's socket (latched, 3-step
  persistence counter)
- 0.30 `throw` — latched max normalized hub travel from ON toward 100°
- non-success capped at 0.70; **1.0 iff success()**: hub angle ≥ 100°, hub
  angular velocity < 0.5 rad/s, panel settled, all finite — judged live. The
  bar is NOT gated (a removable handle may stay cantilevered in the socket by
  friction or be withdrawn; the judged outcome is the isolator state). Null
  policy ~0 (hub held at ON by gravity; bar starts in the saddle).

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset, 180-step settle; layout + mass readbacks (bar/hub via
  per-child density, panel via MassAPI — custom spawners apply no cfg mass
  schemas); hub within 2° of ON; score ~0.
- **P1 (applied force)** PD force + gravity feedforward (kp=6, kd=2, clamp 4 N
  — gains sized to the wrench-delay bound kp·dt/m < 1 for the 0.105 kg bar)
  lifts the bar out of the saddle (P1a, near-free orientation), then rights it
  vertical in free air (P1b, righting torque, transverse-only angular
  damping — never damp the axial spin of a thin rod). `fetch` latches during
  the force lift. Score ≈ 0.15.
- **P2 (transport)** ONE root-state write carries the held bar to a hover
  30 mm above the funnel mouth, tip down, zero velocity — open sky, nothing
  judged is satisfied by the write (asserted: not in socket).
- **P3 (guided blind insertion)** the same force-limited carry lowers the bar;
  the funnel and the 2 mm/side bore slop steer the shaft; the tip lands on the
  backstop. Asserts: `insert` latched AND the hub still within 8° of ON —
  insertion produced no throw. Score ≈ 0.40.
- **P4 (velocity-servo throw)** a hand-on-the-handle force at the ball:
  tangential velocity servo (ω_des = clamp(3·err, ±1.6 rad/s), kv escalated
  ×1.4 on stall — escalate gain, not the 5 N clamp) plus a light seat press
  along the bore axis sweeps the hub from −15° through dead centre; release at
  ≥102° with the hub slow; gravity carries it onto the OFF stop. All wrenches
  zeroed, 150-step settle; success live. Score 1.00.
- **P5** ≥ 3.3 simulated seconds hands-off; success persists → `SIM_GEN_SOLVE:
  SUCCESS`.

Teleports are transport-only: the lift, the insertion and the entire throw are
produced by applied forces, gravity, the funnel/bore contacts and the hinge
dynamics.

## Execution order

STRICTLY ordered: fetch → insert → throw. The hub is recessed beyond finger
reach (throw without the bar is physically excluded — smoke check 6), and the
bar cannot be inserted without first being lifted from the saddle. Insertion
cannot pre-satisfy the throw (bore axis through the axle, zero moment arm).

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, panel nominal at (0.70, ±0.35 band) facing
the robot, saddle 0.42–0.58 m out, lamp 0.7–0.9 m to the side. The fetch is a
pinch of the Ø8 mm shaft at mid-span (86 mm free span between saddle blocks,
~52 mm ground clearance — jaw feasibility asserted in cfg). Insertion is a
vertical-ish lower of the held bar so the tip enters the funnel mouth at world
height ~0.47 m; the funnel plus 2 mm/side slop forgive several mm of error —
exactly the tolerance the solve certifies. The throw is an arc sweep of the
grasped ball end, radius ~0.2 m through ~117°; the wrist never enters the
24 mm slot (grip is on the shaft/ball, always proud of the plate rims —
asserted), and the robot may release any time past dead centre (smoke check 9
certifies gravity completes the throw from +25°). No step needs a second arm,
a regrasp in flight, or simultaneous contacts.

## Files

- `scene.py` — cfg (+ exclusion/clearance/bistability geometry asserts),
  spawners (panel, hub + revolute joint, bar, saddle, lamp, probe), scene
  (rubric, latches in `post_step`), `register_env`.
- `solve.py` — phased solution (force fetch + right, transport, guided blind
  insertion, velocity-servo overcenter throw); watchdog + hard exit.
- `smoke.py` — 11-check rejection battery (settle, randomization readback,
  flag coverage, null policy, SEED-strategy rejection on the lamp, finger
  exclusion probe, slot near-miss drop, dead-centre fallback, overcenter
  commit, bar-revocation cap, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 19.0 s; scores
  0.00 → 0.15 → 0.40 → 1.00, non-decreasing; insertion left the hub at
  −15.08° — no throw credit from seating; throw released at +102.1°, gravity
  finished onto the +110° stop; 3.3 s hands-off persistence)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 19.1 s; opposite ball
  direction, opposite lamp side, +16.9° panel yaw; released at +102.2°)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 11/11` (rc=0, 44.4 s; lamp shoved
  618 mm for zero credit; finger probe arrested 16 mm above the funnel mouth
  with the hub unmoved; dead-centre release at −4° fell back to −15.07°
  (score 0.429, no success); overcenter release at +25° completed hands-off
  to +110.01°; revocation reads the latched 0.70 cap; 143 frames saved)
