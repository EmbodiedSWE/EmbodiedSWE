# close_door_i274 — SaggingGate

**Scene name:** `sagging_gate` · **Env:** `simgen.sagging_gate` (robot="null")

## Seed provenance

Seed task: `rlbench/close_door` — "close the door": a hinged door panel stands
open on its frame; the robot pushes it through its arc until the joint reads
closed. One pushing contact on the judged panel, one unordered act, judged by a
joint angle.

## Strategic difference

The seed's entire plan — push the hinged panel through its arc, done — is
**voided by the hinge itself**, not re-parameterized. The gate hangs on a WORN
hinge with a second degree of freedom: 50 mm of vertical play (a generic D6
joint, rotZ swing + transZ heave, all other axes locked), and gravity keeps it
sagging at the BOTTOM. A raised threshold APRON (75 mm) crosses the closing
sweep, so the sagging gate's free-edge steel SHOE rides 35 mm below the apron
top: ANY flat push — the solve's own closing servo (smoke check 4) or a 5 N·m
slam at 4× its clamp (smoke check 5) — arrests face-on against the apron's
vertical outer wall at ~53°, far from closed. The hinge has no pitch/roll
freedom and the wall is vertical, so there is no wedge channel that converts
swing momentum into lift (the slam's heave readback is 0.0 mm).

Closing demands an internally coordinated, genuinely different plan: **LIFT**
the whole gate up its hinge slack by the blue T-knob, **CARRY** it shut held
high so the shoe passes 15 mm over the apron, and **RELEASE** it over the
green-rimmed SOCKET WELL so the shoe drops in and the gate seats — and the
judged outcome is the settled physical SEAT (frame-local shoe xy + depth),
not the angle: closed-but-held-high fails (asserted mid-solve AND smoke check
9), dropped-proud-on-the-apron fails (smoke checks 6–7). The seat is captive:
it arrests a 2 N·m reopening pull at 4.2° (smoke check 10). There is only ONE
moving body — the gate itself is the payload — so unlike the sibling packages
there is no second object and no inter-object execution order; the difficulty
is the coordination of the two hinge DOFs against the threshold geometry.

Differs from every other tasks_v7 close_* package: i162 closes by hand then
bars the door with a separate rod through two channels; i58 clears an
obstruction so a spring closes a drawer; i89 carries a payload and closes
gently; i4 extracts a prop and drops a gravity gate; i8 builds a structure;
i152 ballasts a counterweighted lid. None lifts the JUDGED PANEL ITSELF up a
worn-hinge DOF to defeat a threshold and seats it into a socket; here closing
and depositing are the same act on the same body. All threshold/seat geometry
clauses are asserted numerically in `SaggingGateSceneCfg.__post_init__`.

## Assets (fully procedural)

- **frame** — heavy DYNAMIC assembly (hinge post + latch post + feet, 60 kg via
  root MassAPI; dynamic because a joint anchored to a kinematic body0 stays
  world-fixed when the body is teleported at reset). Its front (+x) sill is the
  raised dark THRESHOLD APRON (x −0.055..0.215, top z 0.075, exact boxes —
  nothing protrudes above the top, so the lifted clearance is honest) with the
  SOCKET WELL sunk at the closed shoe pose: 62 mm square opening, GREEN rim
  walls z 0.015..0.075, dark floor.
- **gate** — DYNAMIC panel (300×420×26 mm, wood density 550) whose
  origin sits ON the hinge axis at ground level, on a spawn-authored generic
  `UsdPhysics.Joint` (D6) = the WORN HINGE: rotZ ∈ [−105°, +0.5°] (swing),
  transZ ∈ [0, 50 mm] (the play), transX/transY/rotX/rotY locked (low > high);
  joint-pair collision explicitly ON (the arrest and the seat ARE gate–frame
  contacts); per-child density so the hinge inertia is real (mass readback
  2.126 kg). Free edge carries the steel SHOE (36 mm square, bottom 40 mm up
  when sagging) and the top rail the BLUE T-knob (18 mm stem, 36 mm cap — a
  parallel-jaw lift handle). The hinge is damped — the gate stays where left.

Randomization (readback-verified, smoke check 2): frame yaw ±25° + xy ±50 mm;
initial gate angle U(62°, 95°).

## Rubric

- 0.20 `lifted` — gate ever raised past 32 mm of its 50 mm slack while still
  open (> 12°) (latched)
- 0.35 `closure` — latched max closure fraction of the initial opening a0
- 0.15 `carried` — gate ever inside the apron span (< 40°) while raised > 30 mm
  (latched; unreachable without the lift — the push arrest is at ~53°)
- non-success capped at 0.70; **1.0 iff success()**: shoe seated INSIDE the
  socket well (frame-local |xy| < 16 mm of the well axis AND centre below the
  seat depth), gate ≤ 6° open, everything settled and finite — all judged
  live. Null policy ~0 (the gate hangs still at its random angle; smoke 3).

## Solution (solve.py) — the legitimacy certificate

NO teleports at all — the gate never leaves its hinge; every phase is applied
wrenches through the joint + contact dynamics (`set_external_force_and_torque`
on the gate root, recomputed per step, world→body converted):

- **P0** reset, 180-step settle; layout + mass readbacks (gate 2.126 kg via
  per-child density, frame 60 kg via MassAPI — custom spawners apply no cfg
  mass schemas); asserts the worn hinge is live and limited (gate holds a0,
  heave 0); score ~0.
- **P1 (force lift)** PD heave servo + gravity feedforward (the grip on the
  knob; kp·dt/m ≈ 0.39, under the wrench-delay bound) raises the gate to 48 mm.
  `lifted` latches during the force lift. Score 0.20.
- **P2 (carry shut)** heave hold + PD yaw torque about the hinge axis (kp 1.6
  N·m/rad, kd 0.5, clamp 1.2 — kp·dt/I ≈ 0.13 for the ~0.10 kg·m² hinge
  inertia) swings the gate to 0.6°, the raised shoe passing over the apron.
  Asserts closed-but-held-high is NOT success. Score 0.70.
- **P3 (guided release)** the heave target ramps to the bottom while a light
  yaw hold (0.5 N·m clamp) keeps the shoe over the well; wrenches zeroed; the
  shoe drops into the rim walls and the gate seats at 1.8°. Score 1.0.
- **P4** ≥ 3.3 simulated seconds hands-off; success persists → `SIM_GEN_SOLVE:
  SUCCESS`.

## Execution order

No inter-object order (single moving body). The internal coordination is
strictly forced: swing before lift is physically arrested (smoke checks 4–5),
release outside the socket leaves the shoe proud and unseated (checks 6–7),
lift without the swing scores 0.20 (check 8), and the closed angle without the
seat is not success (check 9 and mid-solve assert).

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing the gate (frame nominal (0.35, 0),
opening toward the robot). The one grasp is a parallel-jaw closure under the
blue T-knob cap (Ø36 mm < 60 mm jaw span, asserted) at world height ~0.58 m,
reach ≤ 0.75 m across the whole randomization band. The entire task is then a
single-arm guarded motion of that grasp: lift ~21 N (2.13 kg gate, < 30 N
Franka payload) 50 mm straight up, a horizontal arc of radius ~0.29 m at
constant height (the damped hinge follows the hand; forces stay ~2 kg·g), and
a 48 mm vertical lower + release over the socket — the well's 13 mm/side
clearance forgives the release, exactly what the solve's servo certifies. No
step needs a second arm, a regrasp, or simultaneous contacts; the hinge
carries the gate's weight direction changes, the hand only guides.

## Files

- `scene.py` — cfg (+ threshold/seat/clearance/mass asserts), spawners (frame
  + apron + socket, gate + D6 worn hinge), scene (rubric, latches in
  `post_step`), `register_env`.
- `solve.py` — wrench-only phased solution (lift, carry-shut, guided release);
  watchdog + hard exit.
- `smoke.py` — 11-check rejection battery (settle, randomization readback,
  null-policy, SEED-strategy push arrest, 5 N·m slam no-vault, proud-rest and
  early-release near-misses, lift-only, held-high, seat lock-reality under a
  2 N·m pull, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 19.2 s; scores
  0.00 → 0.20 → 0.70 → 1.00, non-decreasing, 3.3 s hands-off persistence)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 19.1 s; +19.4° frame
  yaw, a0 87.0°)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 11/11` (rc=0, 47.3 s; push arrest at 53.3° vs
  geometric band 51.0–53.3°, slam min 52.6° with 0.0 mm heave kick, seated
  gate max 4.16° under a 2 N·m reopening pull, 151 frames saved)
