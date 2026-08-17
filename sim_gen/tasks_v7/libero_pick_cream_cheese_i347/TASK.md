# balance_verdict — weigh two identical cartons, then deliver only the FULL one

`libero_pick_cream_cheese_i347`
Env: `simgen.balance_verdict` · robot slot: `null` (scene-level task) · procedural assets only.

## Seed provenance

Seed: `roboverse_pack/tasks/libero/libero_pick_cream_cheese.py` — "pick up the cream cheese
and put it in the basket": grasp the visually distinctive cream-cheese box among distractors,
carry it through free air, release it over an open basket; judged by a containment bounding
box. One unordered pick-and-place; the target is identified by sight.

## What changed, and why it is strategically different

The seed's perceptual premise is destroyed and an **epistemic step** is inserted before the
manipulation. There are TWO cream-cheese cartons, **visually identical** (same 60×60×55 mm
size, same color): one FULL (150 g), one an EMPTY replica (40 g), the assignment shuffled per
episode with **no visual cue**. The goal is still "put the cream cheese in the basket" — but
*which carton that is* can only be learned by **running a physical experiment**: a heavy stand
carries a beam BALANCE (bar on a central revolute hinge, limits ±12°, an open weighing pan at
each end, authored-CoM keel 60 mm below the hinge so the empty beam self-levels). Put one
carton on each pan and gravity renders the verdict: the sinking pan holds the full carton
(the imbalance out-torques the keel ~3× — a decisive slam to the stop). The weighing is
**mandatory, not instrumental**: the episode declares a legality rule — any carton entering
the basket before a completed weighing latches a permanent FOUL that freezes all credit, even
if the "right" carton was chosen by luck.

Strategic distance, verified in physics on the forge:

- **The seed's entire plan is impossible twice over.** The target cannot be identified by
  perception (identical cartons, per-episode shuffle), and executing the seed's pick-and-place
  anyway — even with the CORRECT carton — is a permanent foul scoring ~0 (smoke #9, including
  a post-hoc weighing that cannot atone).
- **Information-gathering before delivery, new to the corpus:** the solver must *create* the
  distinguishing observation (load both pans, read which side sank) and then *condition its
  plan on the instrument's output*. No tasks_v7 task makes the deliverable's identity a hidden
  state recoverable only through a measurement. i119 (same seed family) is mechanism actuation
  + receptacle staging; the weigh-beam drop-in task uses a beam as a delivery lever, not a
  comparator; the riddle-tray sorts by geometry, which is visible — here mass is invisible.
- **The instrument cannot be forged:** the `weighed` latch carries a truth clause (lower pan
  must actually hold the heavy carton). An external press holding the empty side down displays
  a false verdict for 2 s and latches nothing; released, gravity restores the truthful verdict
  which then latches (smoke #6/#7). A single carton tips the beam to the stop (the balance is
  sensitive) yet is not a weighing (smoke #5).
- Rule + foul is enforced physics-side in `post_step`, order-aware: credit latches before the
  foul each step, and every credit latch is `& ~foul`.

## Teleport solution (solve.py — passes seeds 0 and 1)

Teleports are transport-only; the weighing, the verdict, the delivery drop and all settling
are contact dynamics on the free hinge:

- **P0** reset + settle 150 steps: keel self-levels the beam (|angle| < 0.1°), cartons on
  their ground spots, basket empty; mass readback beam 0.500 / cartons 0.150/0.040; score 0.
- **P1** identity-agnostic loading: each carton teleported to 15 mm above the pan on its own
  start side (live beam-pose readback, oriented with the beam), one at a time; each falls in
  and the beam swings to its stop by gravity (±12.0°). `loaded` + `weighed` latch from
  physics → score 0.40.
- **P2** the solver reads `down_side()` (a world-z comparison of the two pan centres — camera
  information) and selects the carton on the lower pan; asserted against the hidden truth.
  One write carries it to 145 mm above the basket centre (clearing the 100 mm walls); it
  falls ~110 mm into the mouth and settles; the freed beam swings to the other stop on its
  own → success, score 1.0.
- **P3** persistence: 10 × 40 hands-off steps (3.33 s) with success asserted each block →
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` prints at every phase boundary: 0.0000 → 0.4000 → 1.0000 → 1.0000
(non-decreasing; credit is latched in `post_step`).

## Embodiment argument (Franka, base at (−0.25, 0, 0) facing +x)

Stand nominal (0.45, 0) yaw 180° — the pans and carton spots face the base. Everything the
arm touches lies at radius 0.42–0.73 m, heights 0–0.28 m: inside the envelope.

- **Carton pick (the only grasp):** 60 mm cubes, top pinch (jaw 80 mm > 60 mm), 40–150 g.
  From the ground spots (~0.44 m from base); onto a pan: the pans are OPEN-TOPPED, inner
  100 mm vs carton 60 mm leaves 20 mm of fingertip clearance per side, walls only 26 mm tall,
  pan top surface at ~0.20–0.28 m height depending on tilt — a set-down into a shallow tray.
  Lifting the verdict carton back off a 12°-tilted pan is the same top pinch.
- **Reading the balance is visual** (a 12° tilt over a 500 mm bar displaces the pan tips by
  ±40 mm vertically); the arm never needs to touch the beam, and pressing it cannot forge the
  verdict (smoke #6).
- **Delivery:** release the carton above the 180 mm open basket mouth (~0.43 m from base).
- The stand is 25 kg: incidental contact cannot move the goal frame, and every predicate is
  stand/beam/basket-frame relative regardless.

## Execution-order declaration

`set one carton on each pan → let the beam settle (the sinking pan holds the full carton) →
carry the full carton into the basket`. Weigh-first is REQUIRED and rubric-enforced by a
permanent physics-side foul latch: any carton inside the basket before the `weighed` latch is
a frozen ~0 (smoke #9). Loading order of the two pans is free; after the weighing all moves
are free (smoke #12).

## Rubric

- 0.15 — loaded: both cartons ever rest simultaneously on OPPOSITE pans (latched, velocity-gated)
- 0.25 — weighed: loaded ∧ beam tipped ≥ 8° ∧ the lower pan actually holds the heavy carton
  (the truth clause; latched when the beam is settled)
- 0.20 — delivered: heavy carton inside the upright grounded basket after the weighing (latched)
- non-success cap 0.60; **1.0 iff** success(): weighed ∧ no foul ∧ heavy in basket ∧ light NOT
  in basket ∧ basket upright on ground ∧ 60-step stillness counter ∧ finite.
- Permanent foul: any carton inside the basket before `weighed`; freezes all credit. Null
  policy: 0.000 (smoke #2).

Anti-flake measures baked in (memory-verified forge traps): explicit MassAPI CoM + diagonal
inertia for the beam (mass-only leaves the CoM at the hinge — no keel), heavy dynamic stand
as joint body0 (kinematic anchors stay world-fixed after reset teleports), hidden identity via
two fixed-mass bodies + per-episode start-spot swap (no runtime set_masses), imbalance/keel
margins sized on paper (0.20 N·m vs 0.061 N·m; even the 40 g carton alone reaches the stop,
so `weighed` demands BOTH pans loaded), consecutive-still counter instead of instantaneous
velocity gates, credit-then-foul latch order inside one `post_step`, containment z-window
(0.015–0.085) that rejects rim perches and cartons stacked on a contaminant (observed live in
smoke #11's first construction), burn the first post-seed rand draw, float32 slack on the
0.60 cap assert.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 14/14` on the forge)

1. settle/no-NaN: finite; empty beam LEVEL, cartons on spots, basket upright; score ~0
2. null policy: 240 idle steps → beam level, score ~0, no success
3. randomization readback (3-seed max-pairwise): stand yaw Δ8.0°, stand xy Δ42 mm, basket
   Δ132 mm / Δ174°, carton Δ290 mm
4. side swap: heavy carton starts on BOTH spots over 10 resets, readback matches the draw
5. single-carton near-miss: one carton tips the beam to the stop (it moved) yet
   loaded/weighed stay False — a one-sided load is not a weighing
6. forged verdict: a 0.6 N·m press holds the EMPTY side down with both cartons loaded
   (display false ≥ 8° for 2 s) → `weighed` never latches
7. instrument self-truth: press released → gravity alone swings the beam to the heavy side,
   truthful verdict latches (0.40)
8. keel self-levels: cartons lifted off → beam returns level (|angle| 0.1°) on its own
9. **seed strategy**: the CORRECT carton teleported straight into the basket unweighed →
   permanent foul, score ~0; a full weighing after the foul earns nothing, re-delivery never
   succeeds
10. wrong carton: after a legal weighing the EMPTY carton is delivered → no success, delivery
    credit never latches (score stays 0.40)
11. contamination live: both cartons side-by-side inside → no success; empty one removed →
    success returns
12. containment near-miss: full carton on the ground against the basket's OUTER wall → not
    inside (score = latched 0.60 cap); re-dropped in → success again
13. legal run (fresh seed): load → verdict points at the heavy carton → deliver → success;
    success never fired while the carton was still moving fast
14. video: frames.npz (321 × 600 × 960) captured and saved

Verification: forge pod (Isaac Sim 5.1, RTX 4090). solve seeds 0 & 1 → `SIM_GEN_SOLVE:
SUCCESS` (rc=0, 21.2 s / 21.3 s); smoke → `SIM_GEN_SMOKE: ALL PASS 14/14` (rc=0, 82.2 s).
