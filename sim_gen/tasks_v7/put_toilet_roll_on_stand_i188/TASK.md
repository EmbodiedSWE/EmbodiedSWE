# put_toilet_roll_on_stand_i188 — thread the spindle through the roll, hang it in the rack

Scene `spindle_roll`, env `simgen.spindle_roll` (robot="null").

## Seed provenance

Derived from **rlbench/put_toilet_roll_on_stand**: a toilet roll and a wall-mounted
stand with a fixed horizontal peg (mesh visuals, franka, proximity checker); the
implied strategy is transport + **sliding onto a cantilever** — grab the roll,
align its core with the stand's fixed peg, and push it sideways onto the peg. The
peg is part of the fixture; the roll is the only moving part.

## Strategic difference

The peg is not part of the fixture — it is a free rigid body the solver must first
**assemble with the roll**. The scene has two free items (a hollow 12-gon ROLL with
a real 40 mm bore, and a ball-ended SPINDLE rod) and two fixtures (a V-CRADLE with
a notched backstop, and a RACK with two open-top slotted plates). The goal is a
two-stage construction the seed never needs:

- **Stage 1 (assembly)**: brace the roll in the cradle, then thread the spindle
  through the roll's hollow core — a real peg-in-hole insertion where the HOLE is
  in the payload, driven through a notch in the backstop so the push has a reaction
  surface. In the seed the "threading" is degenerate: the peg is world-fixed and
  the roll slides on.
- **Stage 2 (suspension)**: carry the threaded assembly to the rack and lower it so
  the rod's two exposed shaft ends drop into the open-top slots; the ball ends are
  wider than the slots (axial retention by geometry), and the roll ends up hanging
  UNDER the rod, suspended between the plates, touching nothing else. The goal
  state is a two-body suspension equilibrium — roll-on-rod-on-seats — not a roll
  resting on a fixed peg.
- The rubric judges the **relative** geometry of two free bodies (threaded:
  rod-axis-to-core radial distance + protrusion on both faces) plus rod-in-both-
  slots seating plus the roll hanging in a band strictly below the rod, all
  settled. A roll placed anywhere on the rack without the rod through it is the
  canonical FAILURE (the seed's success state scores ~0 here).

A solver needs a different plan and different code: an in-fixture insertion phase
with force control along the core axis, a carry that preserves a two-body relative
pose, and a slot-drop that lands two shaft ends simultaneously.

## Solution outline (solve.py — teleports for transport only)

- **P0** settle + layout readback; assert score ≈ 0 (rubric-leak guard).
- **P1 CRADLE (transport + drop)**: teleport the roll to a hover above the cradle
  V and let gravity seat it against the backstop. Score 0.20.
- **P2 THREAD (contact dynamics)**: teleport the rod coaxial with the core, ball
  nose 6 mm short of the near face, then drive it through with a velocity-regulated
  axial force (target +0.10 m/s in the cradle frame), y/z gravity-feed-forward PD
  at the CoM, and an attitude PD (τ = K·(â×x̂) − D·ω_perp; ω_perp only — the rod's
  axial inertia is ~3e-6 kg·m², so damping full ω violates the wrench-delay bound
  and bang-bangs). The z-target drops 9 mm once the leading ball is inside so the
  ball rides the core bottom through the backstop notch. Asserted not-threaded
  before the push; threaded (radial 13.1–13.4 mm, both protrusions > 60 mm) after.
  Score 0.45.
- **P3 CARRY (transport)**: relative-pose-preserving teleport of rod+roll to a
  hover above the rack slots; the rod alone is then held by a regulated wrench
  (gravity feed-forward scaled by a load-transfer indicator σ) while the roll falls
  ~26 mm and forms the hang IN MID-AIR under gravity — the assembly's internal
  state is pure contact. Asserted threaded ∧ not seated ∧ roll below rod. Score
  0.60.
- **P4 MOUNT (contact dynamics + hands off)**: regulated descent (−0.08 m/s) until
  the rod is 12 mm above the seats, then the wrench is CUT and the last drop into
  both V-groove bearings is free. Hands-off settle (chunked, with velocity
  telemetry) + 2 s extra margin. Score 1.0.
- **P5 persistence**: ≥ 3.3 more simulated seconds hands-off with success()
  checked every substep, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (0 → 0.20 → 0.45
→ 0.60 → 1.0 → 1.0). Passes on seeds 0, 1, 2 on the forge.

## Rubric

`success()` = threaded ∧ seated ∧ hanging ∧ clear ∧ settled:

- **threaded**: rod-axis-to-roll-center radial distance < 16 mm (a true threading
  reads 13.1–13.4 mm = CORE_R − ROD_R; a rod lying ON the roll reads ~39 mm),
  axes within 15°, AND the shaft protrudes ≥ 10 mm beyond BOTH faces (a
  partial nose-in reads negative far-side protrusion — rejected).
- **seated**: both rod shaft ends within 8 mm (y,z) of their V-seat centers at
  SEAT_Z = 0.085, axis within 10° of the slot line, |x| centered within 20 mm.
- **hanging**: roll center within a 9 mm band around hang_z = 0.072 (i.e. hanging
  strictly BELOW the rod axis, suspended), |x| within 25 mm, |y| within 20 mm of
  the rack center.
- **clear**: the roll's lowest corner > 8 mm above the slab top — the anti-seed
  clause: a roll resting on any rack surface is rejected.
- **settled**: stillness SUSTAINED 30 consecutive substeps (counter in post_step)
  — an instantaneous threshold fires at pendulum turning points; the rotisserie
  limit cycle (capsule-on-flat phantom axle spin friction-driving the roll
  pendulum) was observed live and fixed with V-groove seats + engine angular
  damping, not by weakening the gate.

`score()`: latched 0.20 (roll ever braced in the cradle) + 0.25 (rod ever threaded
through the core) + 0.15 (assembly ever threaded-and-airborne together near the
rack), cap 0.60; 1.0 iff success(). Null policy ≈ 0 (asserted).

## Embodiment argument (Franka)

Every manipulated object affords a Franka power grasp: the rod's exposed shaft is
14 mm diameter (ball ends 26 mm — both under the 80 mm gripper stroke), the roll
is 90 mm wide with a 12-gon outer surface graspable across its width. From a base
at the origin facing the workspace (cradle at ~(0.32, −0.30), rack at ~(0.44,
0.30) — both inside Franka's envelope), the arm: picks the roll and sets it in
the cradle V (the funnel walls forgive ~35° of approach error); grasps the rod by
its trailing half and pushes it axially through the core — the backstop takes the
reaction force, so this is a stiffness-controlled insertion with a 13 mm radial
clearance, well within force-guided peg-in-hole practice; regrasps the protruding
ball end, lifts the assembly (the roll self-centers by gravity), carries it over
the rack, and lowers until both shaft ends drop into the funnel-topped slots —
the 35° funnels and the ball-end axial retention make the final alignment
tolerant. The hard parts are perception (core axis pose, slot line) and two-point
alignment, not dexterity or force beyond Franka's payload (roll ~0.12 kg, rod
~0.09 kg).

## Execution order

1. `scene.py` written first (geometry + rubric).
2. `solve.py` iterated on the forge: runs 1–3 exposed a constant-velocity limit
   cycle after mounting (phantom axle spin on flat seats, ω regrowing to ~3 rad/s
   after genuine stillness); fixed in the SCENE with V-groove bearing seats +
   angular damping 2.0 on both free bodies. Run 4 exposed an axial-damping
   instability in the carry torque (D·dt/I ≈ 13 about the rod axis → 36 rad/s
   bang-bang); fixed by damping ω_perp only. SUCCESS on seeds 0, 1, 2.
3. Rubric finalized against the demonstrated assembly (13.4 mm radial, rod z =
   SEAT_Z ± 1 mm, roll z = hang_z ± 2 mm across seeds).
4. `smoke.py` rejection battery: **ALL PASS 15/15** on the forge, frames.npz
   saved. (One iteration: the settle-gate probe first sampled the swing velocity
   AFTER the roll had already rammed the rod and decayed below settle_lin —
   fixed by sampling the peak across the first substeps, not by weakening the
   gate.)

## Smoke checks (15)

1. settle/no-NaN + layout sanity; 2. score ≈ 0 at reset; 3. fixture randomization
readback (cradle/rack x, y, yaw spreads); 4. item spawn + side-swap readback (both
sides occur); 5. null policy ≈ 0 after 240 idle steps; 6. seed-strategy analog —
bare roll placed ON the rack (no rod through it) rejected, score ≈ 0; 7. rod-only
mount — rod seated in both slots without the roll reads seated() yet scores ≈ 0
(the roll is the payload); 8. out-of-order physical probe — the roll force-pushed
onto the ALREADY-MOUNTED spindle moved 55 mm (non-vacuous) but is blocked by the
plate and never threads: threading after mounting is impossible; 9. rod resting ON
TOP of the cradled roll (radial ≈ 39 mm) not threaded; 10. partial threading (nose
20 mm in, negative far-side protrusion) rejected; 11. threaded assembly resting on
the SLAB (not the slots) → threaded True but seated/hanging False, score ≈ 0.25;
12. settle gate — the exact goal geometry with an injected 0.29 m/s swing is not
success (peak-sampled, judged every substep); 13. latched credit survives
teleport-away (score keeps 0.25, seated/hanging drop, no success); 14. rejection
audit (success never True in the battery); 15. final no-NaN.
