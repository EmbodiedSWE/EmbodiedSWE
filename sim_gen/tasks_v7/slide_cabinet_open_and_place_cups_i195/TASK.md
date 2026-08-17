# wedge_gate_cabinet — jack a handle-less gravity gate open with a ramp wedge, slide the cups through, then let it fall closed

Package: `sim_gen/tasks_v7/slide_cabinet_open_and_place_cups_i195`
Env: `simgen.wedge_gate_cabinet` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/slide_cabinet_open_and_place_cups`. In the seed, the robot slides
one cabinet door OPEN — the panel has a handle, moves horizontally, and STAYS where
it is put — and then places interchangeable cups into the revealed volume, driven by
a recorded waypoint trajectory (the checker is a TODO). Opening is a one-shot state
change; nothing ever needs to be closed again.

## What changed, and why it is strategically different

1. **The door cannot be "opened" at all — it can only be PROPPED.** The cabinet's
   doorway is guarded by a vertical prismatic GATE with no handle, gravity-returned
   to its closed stop (spawn-authored joint, stroke 94 mm, closed slit 12 mm — cup
   height 60 mm, so nothing passes). Any direct lift is a dead end: release the gate
   and it falls shut. The seed's persistent "open" state does not exist; openness is
   a resource that must be actively MAINTAINED by a mechanism.

2. **A dedicated TOOL is the only way to hold it open.** A ramp WEDGE (6 mm tip →
   80 mm plateau) fits the floor slit tip-first; pushing it under the gate converts
   horizontal insertion into vertical lift (the incline jacks the gate) until the
   gate rides the flat plateau — propped hands-free at ~66 mm. This is tool USE with
   a mechanical advantage, not articulation of the target itself: the seed has no
   tool, and sibling `i112` (bypass shutter) multiplexes access but never employs an
   intermediate object. Orientation matters: the wedge inserted TAIL-first (80 mm
   face forward) just rams the closed gate (smoke check 8).

3. **Transport is THROUGH the propped gap, on the floor.** Two cups must be slid
   along the floor through the doorway while the wedge holds the gate up — the cabinet
   interior is roofed, so there is no top-loading route (a cup on the roof is rejected
   by the z band). Every cup transit is causally dependent on the prop being in place.

4. **The end state must RESTORE the mechanism:** success additionally requires the
   wedge extracted clear of the cabinet and the gate back on its closed stop. The
   ordering is enforced physically, not by fiat — with the wedge out, the slit is
   12 mm and no cup can pass, so both deposits are forced to precede extraction, and
   a wedge left inside (or even just its tip still in the slit, holding the gate
   1–2 mm proud) is caught by the `wedge_clear` / `gate_closed` predicates.

5. **Physics is the judge**, not proximity-to-recorded-waypoints: cabinet-frame
   membership windows whose z band accepts only floor-rest height (rejecting stacked
   cups at +0.04 and roof-top cups at +0.10 vs rest 0.030), an uprightness cone (a
   lying cup PASSES the z band — uprightness is load-bearing, asserted in
   `__post_init__`), a joint-readback gate-closed predicate, a yaw-proof wedge-clear
   radius (clear_y_min 0.115 > max reach 0.098 of any wedge point at any yaw), and a
   stillness counter-latch.

## Teleport-solution outline (transport only — verified on forge)

Teleportation is used ONLY to stage bodies at free-space start poses (zero velocity,
outside every rubric region, overlap-checked against all randomized layouts). Every
load-bearing interaction is real contact dynamics driven by a PI velocity servo
(outer position loop → v_des, inner force loop with a slow integral term that
supplies break-away force; integrator dumped near target; world→body force frame via
`quat_apply_inverse`; gains respect the one-substep wrench delay, KV·dt/m ≤ 0.37).
The gate is NEVER touched by the solver — it moves only because the wedge jacks it.
The cups' push force is capped at 0.50 N, below their 0.62 N tipping threshold.

- **P0** settle + readbacks (gate on its closed stop); assert score < 0.05.
- **P1** park both cups clear of the work corridor → stage the wedge on the
  approach lane → servo-push it tip-first into the slit and under the gate to
  y −0.075 (plateau fully spans the gate footprint, 14 mm margin) → hands-off
  240-step hold, re-assert gate_j ≥ 0.058 with zero applied force (a true static
  prop, not a force-assisted hover).
- **P2/P3** stage each cup at the doorway lane mouth and servo-slide it along the
  floor through the propped gap to y −0.130; assert `cup_inside` on readback.
- **P4** servo-extract the wedge to y +0.170; assert `wedge_clear` ∧ `gate_closed`
  (the gate falls shut on its own, j −0.0020).
- **P5** ring-down until `success()` holds 120 consecutive steps.
- **P6** hands-off ≥ 3.3 simulated seconds; assert success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, scores monotone
0.00 → 0.30 → 0.475 → 0.65 → 1.00, zero persistence flickers.**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(0, 0.60, 0) facing the doorway (cabinet at the origin, 34×28 cm footprint,
18 cm tall; staging row at y ≈ 0.15–0.32). Reachability: every manipulated pose lies
0.25–0.60 m from the base at heights 0.00–0.08 m — inside Franka's ~0.85 m envelope
with ordinary top-down or lateral approaches; the arm never needs to enter the
cabinet (cups are released at the lane mouth and pushed, or simply pushed from
behind the whole way). Graspability: the cups are 54 mm cylinders (< 80 mm jaw
opening), side-pinched; the wedge's tail block is a 50 mm-wide, 8 mm-tall plate —
pinchable across its width, or simply pushed on its 80 mm-tall rear face, which is
exactly what the solve's servo does. Forces: insertion peaks near 6–8 N (jacking a
0.30 kg gate against gravity via the incline) and extraction ~4 N — trivial for the
arm; the cup pushes are 0.5 N. Every motion in the solve is a straight horizontal
push at fixed height, the easiest primitive a manipulator has; the hands-free
plateau prop means the arm is completely FREE during cup transport — no
dual-arm-like "hold the door while placing" conflict ever arises.

## Execution order

`describe()` states the two mechanically-forced constraints: cups can only pass
while the wedge props the gate (closed slit 12 mm ≪ cup 60 mm), and the wedge must
be extracted AFTER both cups are inside (extraction re-seals the doorway). Free
choices: which cup goes first, either interior lane, how deep the wedge is driven
past the minimum prop point, and any number of prop/extract cycles. The solve's
order (wedge in → cup A → cup B → wedge out) is one convenient schedule; wedge in →
cup B → cup A → wedge out is equally valid.

## Rubric

- +0.15 the first time the gate has EVER been jacked to j ≥ 0.036 (lift latch).
- +0.15 the first time j ≥ 0.058 (deep prop latch — enough headroom to pass a cup).
- +0.175 per cup that has EVER been inside its membership window (cabinet frame:
  |x| ≤ 0.135, y ∈ [−0.256, −0.048], |z − 0.030| ≤ 0.014, uprightness within 30°),
  latched.
- +0.10 once both-inside ∧ gate-closed ∧ wedge-clear have held simultaneously
  (done latch).
- Partial credit capped at 0.75; `score = 1.0` iff `success()` = both inside ∧ gate
  closed (j ≤ 0.012) ∧ wedge clear (every-point yaw-proof radius) ∧ settled
  (stillness counter-latch, 30 consecutive quiet steps).
- Null policy scores ~0.

`__post_init__` honesty asserts (~15): the closed slit passes the wedge tip but not
a cup; the plateau props above deep_j_min; the gate stroke clears cup height; the
lying-cup z-band pass (uprightness load-bearing); clear_y_min exceeds the wedge's
max yaw reach; lane/slot/park geometry all overlap-free under worst-case
randomization; reset never writes the gate off its track.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 18/18`)

1. Reset settles: states finite, cups upright, gate on its closed stop.
2. Fresh reset: score ~0, no success.
3. Randomization: cup slot assignment varies (≥3 distinct / 6 across seeds 21–26),
   never shared, x jitter > 10 mm.
4. Randomization: wedge home position spread > 10 mm, yaw spread > 30°.
5. Null policy 240 steps: score ~0.
6. Gravity return is real: a regulated vertical force servo lifts the gate past the
   deep-prop threshold; the moment force is removed it FALLS back to its closed
   stop (probe asserts the actuator moved — non-vacuous).
7. Sealed doorway: a cup pushed at the CLOSED gate (real 0.5 N-capped servo) stalls
   at the gate face, stays upright, and never budges the gate.
8. Orientation matters: the wedge inserted TAIL-first (flat 80 mm face forward)
   stalls against the gate; the gate never reaches the lift latch.
9. Seed-analog ("open, then place, walk away"): a REAL wedge insertion props the
   gate, cups appear inside, wedge left in place → gate not closed, wedge not
   clear, NOT success, latched score ≈ 0.65.
10. Tip-in-slit near-miss: wedge withdrawn until only free of the gate but its body
    at y +0.085 < clear radius → gate closed ∧ both cups in ∧ NOT clear → NOT
    success.
11. Lying cup inside: z band PASSES (readback) yet uprightness rejects.
12. z band: a cup stacked on a seated cup and a cup on the cabinet roof both
    rejected.
13. Doorway loiterer: a cup at the inside face of the gate (y −0.044, outside the
    membership y window) rejected.
14. Wedge smuggled inside the cabinet with cups in and gate closed: NOT clear,
    score ≤ 0.36 at that point.
15. Latched credit: removing an inside cup keeps the score while the live predicate
    drops.
16. Settle gate: the completed arrangement judged while a cup is still falling is
    NOT success (cup removed before ring-down — battery never succeeds).
17. Rejection audit: `success()` never fired at any judged point.
18. Final state no-NaN.

Smoke also records `frames.npz` (≤ 500 × 600 × 960 × 3) from a perspective camera.
