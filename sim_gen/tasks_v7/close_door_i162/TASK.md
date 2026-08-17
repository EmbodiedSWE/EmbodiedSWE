# close_door_i162 — BarredGate

**Scene name:** `barred_gate` · **Env:** `simgen.barred_gate` (robot="null")

## Seed provenance

Seed task: `rlbench/close_door` — "close the door": a hinged door panel stands
open on its frame; the robot pushes it through its arc until the joint reads
closed. One pushing contact on the judged panel, one unordered act, judged by a
joint angle.

## Strategic difference

The seed's entire plan — push the hinged panel shut, done — is **insufficient
here by construction**, not re-parameterized. Closing the gate door is only the
easy half: the judged outcome is a **LOCKED state built out of two bodies**. The
door's free edge carries a steel hasp EYELET (open vertical channel), and the
frame carries a matching STAPLE channel on a pedestal directly below the eyelet's
closed pose. Success requires the long BLUE lock rod fetched from a caddy and
dropped down through the eyelet into the staple so its shaft threads BOTH
channels — only then is the door barred. A closed-but-unbarred door is judged
unlocked (smoke check 5 constructs exactly the seed's end state, shows no
success, then reopens the door with a real 1.5 N·m torque), while the completed
bar arrests a 2 N·m opening pull within the closed tolerance (smoke check 10).

The plan skeleton also acquires a **physically forced order** (the seed has
none): the rod cannot ride in the eyelet while the door closes (its shaft
protrudes ~95 mm below the eyelet and side-strikes the staple pedestal — smoke
check 6 arrests the solve's own closing servo), and it cannot wait in the staple
either (the exposed shaft stands across the closing eyelet's arrival band —
smoke check 7). Close FIRST, then insert vertically. A RED short decoy stub in
the caddy's other well adds a perception trap: fully dropped it bottoms 35+ mm
short of the staple and locks nothing (smoke check 8). All lock/ordering
geometry clauses are asserted numerically in `BarredGateSceneCfg.__post_init__`.

Differs from every other tasks_v7 close_* package: i58 clears an obstruction so
a spring closes a drawer; i89 carries a payload and closes gently; i4 extracts a
prop and drops a gravity gate; i8 builds a structure; i152 ballasts a
counterweighted lid. None closes a panel by hand AND builds a two-channel
rod-lock through the moving panel; here the robot actively closes and then
mechanically bars the door, and the rubric's success is the lock, not the angle.

## Assets (fully procedural)

- **frame** — heavy DYNAMIC gate frame (posts + header + sill + feet, 60 kg via
  root MassAPI; dynamic because a joint anchored to a kinematic body0 stays
  world-fixed when the body is teleported at reset). Front (+x) side carries the
  staple pedestal: column to z 0.30, then a 28 mm square channel z 0.30–0.36
  with a 45° funnel mouth to 64 mm.
- **door** — DYNAMIC panel (350×460×30 mm, wood density 600) on a spawn-authored
  vertical `UsdPhysics.RevoluteJoint` (limits: open stop 100°, closed stop
  +0.5°; joint-pair collision explicitly ON; per-child density so the hinge
  inertia is real). A steel hasp ARM cantilevers past the free edge carrying the
  eyelet: 34 mm square channel z 0.39–0.45 with its own funnel mouth; at the
  closed stop it stands coaxially 30 mm above the staple. The panel edge stops
  short of the pedestal's swept-disc radius (asserted — a longer panel arrests
  on the pedestal at ~17.6°, observed); only the hasp, above the funnel top,
  overhangs it. The hinge is neutral and damped — the door stays where it is
  left (like the seed's door).
- **lock rod** — steel shaft Ø18×160 mm + BLUE head Ø36×20 (origin at the TIP);
  seated, its tip rests on the staple floor and the shaft fills both channels.
  **decoy** — same shaft Ø18 but 55 mm, RED head.
- **caddy** — kinematic two-well stand (40 mm square wells) holding both rods
  upright, standing 0.60 m out on a random bearing in front of the gate.

Randomization (readback-verified, smoke checks 2–3): frame yaw ±25° + xy
±50 mm; initial door angle U(55°, 95°); caddy bearing U(10°, 65°) + free yaw;
rod/decoy well swap.

## Rubric

- 0.15 `lifted` — lock rod ever raised clear of the caddy (tip above 0.17 m,
  latched)
- 0.30 `closure` — latched max closure fraction of the initial opening a0
- 0.25 `threaded` — lock-rod shaft ever inside the door's eyelet channel
  (latched, 3-step persistence counter)
- non-success capped at 0.70; **1.0 iff success()**: door ≤ 3° on its stop, rod
  tip seated in the staple channel, shaft threading the eyelet, everything
  settled and finite — all judged live. Null policy ~0 (the door does not move
  by itself; the rods start in the caddy).

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset, 180-step settle; layout + mass readbacks (rod/door via per-child
  density, frame via MassAPI — custom spawners apply no cfg mass schemas);
  score ~0.
- **P1 (torque servo, real dynamics)** PD torque about the hinge axis
  (`set_external_force_and_torque`, pure z — what a hand on the panel applies;
  Kp 8 N·m/rad, Kd 2, clamp 4 N·m) swings the door onto its closed stop and
  presses. Closure credit comes from the hinge dynamics. Asserts
  closed-but-unbarred is NOT success. Score ≈ 0.30.
- **P2a (applied force)** PD force + gravity feedforward + righting torque (a
  firm force-limited grasp) lifts the rod straight out of its well; `lifted`
  latches during the force lift. Score ≈ 0.45.
- **P2b (transport)** ONE root-state write carries the held rod to a hover 32 mm
  above the eyelet funnel mouth, upright, zero velocity — open sky, nothing
  judged is satisfied by the write (asserted: not seated).
- **P2c (guided descent)** the same force-limited carry lowers the rod; eyelet
  funnel → eyelet → staple funnel → staple steer the shaft; the tip lands on
  the staple floor. A light 0.6 N·m press holds the door meanwhile (a hand on
  the panel), then ALL wrenches are zeroed. Score 1.0.
- **P3** ≥ 3.3 simulated seconds hands-off; success persists → `SIM_GEN_SOLVE:
  SUCCESS`.

Teleports are transport-only: closure, threading and seating are produced by
torque, force, gravity and contact.

## Execution order

STRICTLY ordered: close, then insert. Both wrong orders are physically arrested
(geometry asserted in `__post_init__`, demonstrated under the solve's own servo
in smoke checks 6–7).

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing the gate (frame nominal (0.35, 0),
door opening toward the robot). Closing = a palm-side push on the panel's free
edge through a 55–95° arc at heights ~0.3–0.5 m and reach ≤ 0.85 m — the
canonical seed motion. The rod pick is a top grasp on the Ø36 mm head (< 60 mm
jaw span, asserted) standing proud of the caddy at ~0.23 m height, 0.60 m from
the gate. Insertion is a vertical lower of the held rod into the eyelet funnel
mouth at world height ~0.47 m directly over the pedestal, followed by an open of
the jaws; both funnels forgive several mm of lateral error — exactly the
release the solve certifies. No step needs a second arm, regrasp in flight, or
simultaneous contacts (the damped door stays shut on its stop by itself while
the arm fetches the rod).

## Files

- `scene.py` — cfg (+ lock/order/decoy/clearance asserts), spawners (frame,
  door + hinge, rods, caddy), scene (rubric, latches in `post_step`),
  `register_env`.
- `solve.py` — phased solution (torque-close, force-lift, transport, guided
  insert); watchdog + hard exit.
- `smoke.py` — 11-check rejection battery (settle, randomization readback, well
  swap, null-policy, SEED-strategy rejection + reopen, wrong order A/B arrest
  under the closing servo, decoy near-miss + reopen, seated-but-ajar near-miss,
  lock-reality under 2 N·m pull, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 30.2 s; scores
  0.00 → 0.30 → 0.45 → 1.00, non-decreasing, 3.3 s hands-off persistence)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 29.4 s; opposite well,
  +19.4° yaw)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 11/11` (rc=0, 108 s; wrong-order
  arrests at 9.4° / 6.7°, seed-strategy and decoy doors reopened past 84°,
  barred door max 2.36° under a 2 N·m pull, 225 frames saved)
