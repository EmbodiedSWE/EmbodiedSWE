# scoop_with_spatula_i387 — Quarry: excavate the buried gold cube, keep the rubble contained

## Seed provenance

Seed task: `rlbench/scoop_with_spatula` (RoboVerse
`roboverse_pack/tasks/rlbench/scoop_with_spatula.py`): "scoop up the cube and lift it
with the spatula". One 20 mm cube sitting free on the table, one spatula tool (USD
asset), a Franka, and a trajectory file; the whole skill is TOOL MEDIATION — grasp the
thin blade, slide it under the unobstructed cube, lift, and be judged on the cube
riding aloft on the blade.

## What changed and why it is strategically different

The manipulation model is replaced wholesale, not perturbed:

| | seed | this task |
|---|---|---|
| plan | tool-mediated scooping of a FREE cube | OCCLUSION-DRIVEN EXCAVATION under a containment invariant |
| objects | 1 cube + 1 spatula tool | 1 buried gold prize + 6 gray rubble stones + walled pit + pocketed pedestal |
| access | the cube is graspable from step one | the prize starts physically UNREACHABLE — a capping stone rests on it, a ring hems it in |
| reasoning | none beyond the scoop | multi-object clutter clearing where every removed obstacle must itself be RE-PLACED inside the pit (relocation, not disposal), plus goal-object identity (only gold belongs in the pocket) |
| judged on | cube carried aloft on a hand-supported blade (transient) | the settled END STATE of 7 free bodies: prize seated in the pedestal pocket AND all six stones inside the pit walls |

There is no tool and nothing is scooped or held aloft. The load is inverted relative
to the seed: instead of getting one object onto a support, the work is removing a
pile of obstructing objects off the goal object — and the containment clause makes the
obvious shortcut (fling the rubble away) a guaranteed failure, so clearing is a
placement task in its own right. It is also different from every other package read
while building it: `..._i97` (mechanism actuation by accumulated load / mass
identity), `pen_holder` (containment fill of a single receptacle), `close_grill_i8`
(structure building to a height band). Here the core is UNCOVER → RELOCATE-WITH-KEEP-IN
→ EXTRACT → SEAT, none of which those tasks contain.

The seed's own end state — a cube aloft on a hand-supported blade — is not settleable
under a hands-off judge; its nearest constructed analog in the smoke battery is the
near-miss place (check 7), rejected.

## The scene

Fully procedural (`scene.py`, scene `simgen.quarry`, robot `"null"`). Static geometry
(no rigid bodies, immovable): the ground, a rectangular QUARRY PIT at (0.28, 0) —
40 × 26 cm floor, four 12 mm thick, 55 mm tall dark-brown walls, open top — and a BLUE
PEDESTAL at (0.28, 0.30): 9 × 9 cm, 12 cm tall, topped by a shallow square pocket
(52 mm inner span, 14 mm light-blue rim walls; seat plane z = 0.1375). Free bodies:
the gold PRIZE cube (35 mm, 0.10 kg) and six gray RUBBLE stones (40 mm, 0.12 kg each).

Per-episode randomization (readback-verified in smoke checks 2–3): the prize spawns in
the west end of the pit, x ∈ [0.17, 0.23], y ∈ ±0.042, free yaw; a random permutation
(`pile_slot`, stored for readback) assigns one stone as the CAPPING stone directly on
top of the prize (±5 mm jitter) and the other five to a surrounding ring (radius
56–62 mm, ±3° angular jitter). After settling, `covered()` is True: the cap's weight is
carried by the prize. A first post-seed draw is burned (degenerate-first-draw quirk).

Geometry honesty is asserted in `QuarrySceneCfg.__post_init__` (verified standalone):
the pocket admits the prize with 8.5 mm/side slack and max in-pocket offset
(0.0085 m) < `seat_tol` (0.020 m), so a seated read is a physically seated cube; a
rim-perched cube fails BOTH the xy and the z clause; the burial ring fits inside the
walls at worst yaw; the pile footprint (≤ 0.314 m) never reaches the east drop slots
(≥ 0.316 m); every drop slot is > `clear_r` + 3 mm from any possible prize spawn.

## Rubric

Latched partial credit (booleans latched in `post_step`, survive transients), success
judged live on the settled state. **Every latch is gated on the containment invariant**
(all six stones inside the pit, calm), so ejecting rubble earns nothing at any stage:

- 0.15 `clear` — ≥ 3 stones calm inside the pit, > 0.10 m from the prize spawn;
- 0.20 `expose` — prize UNCOVERED (no stone near-above it), all stones in-pit and calm;
- 0.20 `lift` — exposed prize carried above z = 0.10 (containment still live);
- 0.15 `near` — lifted prize within 0.10 m of the pocket seat;
- non-success capped at 0.70; exactly 1.0 iff `success()`: prize seated in the pocket
  (xy < 20 mm, |z − seat| < 8 mm, still) AND all six stones inside the pit AND
  everything settled and finite.

Success is end-state: rubble that leaves the pit and RETURNS still counts (smoke
check 8 shows the legitimate version scores < 1 only because the prize is back on the
floor). A gray stone in the pocket is worth nothing (check 6).

## Teleport solution (`solve.py`)

Teleports are TRANSPORT ONLY; every load-bearing interaction is contact/gravity:

- **P0** reset + settle 240 steps: the cap seats ON the prize (covered=True asserted),
  ring on the floor, all-in-pit asserted. `SIM_GEN_SCORE 0.00`.
- **P1** clear the overburden TOP-DOWN in current-height order (= gripper
  accessibility order). Each stone: one root-state write to a zero-velocity HOVER
  above its own empty drop slot in the EAST half of the pit, then hands-off fall +
  settle; the settled pose is contact-made, never written. Containment is kept
  throughout. Mid-print after 3 stones (0.35), then cleared: prize uncovered,
  clear+expose latched. `SIM_GEN_SCORE 0.35`.
- **P2** only after `covered()` reads False: carry the prize to a hover 55 mm above
  the open pocket; it DROPS and seats against the pocket floor/walls by contact.
  `SIM_GEN_SCORE 1.00`.
- **P3** hands-off persistence 3.33 simulated seconds; success must still hold.
  `SIM_GEN_SOLVE: SUCCESS`.

Scores are monotone by construction (latches). Verified on the forge for seeds 0
and 1 (~19 s each).

## Smoke battery (`smoke.py`, 9 checks)

1. settle/no-NaN baseline: finite, prize on the pit floor and COVERED, all-in-pit,
   score ~0, no success; 2. randomization readback: prize spawn moved > 5 mm AND the
   burial permutation differs across seeds; 3. burial robustness: covered after
   settling in 8/8 seeded resets, ≥ 6 distinct prize positions; 4. null policy: 240
   idle steps, score ~0, nothing moves; 5. KEEP-IN GATE: prize seated PERFECTLY but
   one stone left outside the pit — seated True yet success False and ALL latches dark
   (score ~0); 6. WRONG OBJECT: a gray stone seated in the pocket — no success, only
   legitimate clear+expose credit (≤ 0.36); 7. NEAR-MISS PLACE: prize settled on the
   ground beside the pedestal — not seated, score < 1; 8. BACK-IN-PIT: exposed prize
   returned to the pit floor — no success, score < 1; 9. frames.npz video (149 frames
   captured across the probes).

## Embodiment argument (single Franka + parallel jaw, OSC)

- Every manipulated object is a 35–40 mm rigid cube ≤ 0.12 kg — canonical parallel-jaw
  picks (Franka jaw opening 80 mm, payload 3 kg).
- The pit is open on top with only 55 mm walls: every stone and the prize are plain
  top-down picks/places over the wall; the 40 mm stones stand proud enough of their
  neighbours for a top grasp, and the solve's top-down clearing order is exactly the
  accessibility order a gripper faces on a pile.
- The pocket has 8.5 mm of slack per side and an open top at 0.12 m — a straight
  top-down place; the solve's release-from-hover mirrors a gripper release a few cm
  above the pocket.
- Workspace: pit spans x ∈ [0.08, 0.48], y ∈ ±0.13; pocket at (0.28, 0.30, 0.14). A
  base at ≈ (−0.15, 0.10, 0) puts every pick and both place zones within a 0.35–0.65 m
  reach annulus with nothing between them.
- Forces: nothing is pressed or actuated — gravity seats every placement; the arm only
  transports. The "extraction only after uncovering" order is enforced at arm level by
  the physical occlusion itself (the cap rests ON the prize).

## Execution order

The physics forces a partial order — the prize cannot be extracted until the capping
stone (and enough of the ring) is off it, and each buried layer must come off
top-down — but the RUBRIC judges only the end state: any clearing order works, stones
may be placed anywhere inside the walls (not just the east slots), and rubble that
temporarily leaves the pit may be returned. No order-latch exists; check 5 shows the
containment clause, not an ordering rule, is what gates credit.

## Checks

- forge `solve` seed 0: `SIM_GEN_SOLVE: SUCCESS` (scores 0.00 → 0.35 → 1.00 → 1.00);
- forge `solve` seed 1: `SIM_GEN_SOLVE: SUCCESS` (same monotone score trace);
- forge `smoke`: `SIM_GEN_SMOKE: ALL PASS 9/9` + `frames.npz` (149 frames).
