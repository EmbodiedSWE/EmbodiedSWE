# libero_pick_orange_juice_i129 — Ballast-vault tool-use insertion

**Scene:** `simgen.ballast_vault` (`BallastVaultScene`, robot="null")
**Seed:** `libero/libero_pick_orange_juice`
(`RoboVerse/roboverse_pack/tasks/libero/libero_pick_orange_juice.py`)

## Seed provenance and what changed

The LIBERO seed is the plainest pick-and-place there is: identify the orange juice
carton among six grocery distractors, grasp it, carry it through free space, and
release it over an **always-open static basket** — one prehensile transport, judged
by a bbox around the basket. No mechanism, no ordering, no tool.

This task keeps the surface (get the juice carton into the receptacle, among
distractors) but replaces the **plan skeleton**:

1. **The receptacle fights back.** The basket becomes a green PANTRY VAULT whose
   only opening is covered by a counterweighted hinged lid that **rests closed**
   (closing imbalance ≈ 0.097 N·m about the hinge). The seed's whole plan — carry
   the item to the receptacle and let go — is physically impossible, and the smoke
   battery demonstrates it: a carton released over the vault lands on the closed
   lid, which the impact only presses harder into its stop (smoke check 5).
2. **The opener is a tool, and it must be the right one.** Past the hinge, a short
   lever carries an open yellow ballast cup. Loading the **dark iron block**
   (450 g, hinge torque ≈ 0.5 N·m ≈ 5× the imbalance) swings the lid to its 65°
   stop and *holds* it there — a one-object "power-hold" the arm doesn't have to
   maintain. The **white foam block**, identical in shape at 12 g (≈ 0.02 N·m,
   4–5× *under* the imbalance), physically cannot: choosing the wrong tool fails
   through the plant, not through a rubric clause (smoke check 4 vs 6). This keeps
   the seed's identify-the-right-object element, but the discrimination is now
   *functional* (mass), not visual labeling alone.
3. **The plan is a three-stage tool cycle with a physically forced order and a
   forced cleanup.** Load the cup → insert the carton through the timed 65°
   aperture (the un-hinged half of the mouth is fully uncovered; the covered half
   is not) → **unload** the cup and set the block aside so the lid swings shut on
   its own and seals the carton in. Success demands the cup empty and both blocks
   resting *off* the vault below rim height, so the tool must be returned, not
   abandoned — the mirror image of the load. A single arm gets no shortcut: holding
   the lid open by hand costs the only gripper, so nothing can be inserted while
   holding (that end state is smoke check 7's dilemma rejection).

So the plan skeleton changes from *"grasp target, place into open receptacle"* to
*"acquire a counterweight tool, use it to hold a self-closing closure open, insert
the payload through the timed aperture, then unload the tool so the closure seals
itself"* — tool-use with a forced load/unload cycle instead of one transport.

## Why strategically different from every examined sibling

- **The seed** (`libero_pick_orange_juice`): open static receptacle, one grasp-carry-
  release; here the release-over-receptacle plan is demonstrated (in smoke) to
  bounce off. Distractors there are visual; here the decoy is *mechanical*.
- ***libero_pick_orange_juice_i6*** (same seed, sibling): non-prehensile ledge push
  into a pre-staged mobile basket — receptacle is open the whole time, the trick is
  staging + a ballistic drop. No mechanism, no tool, no ordering.
- ***close_fridge_i89*** (nearest hinge-machinery neighbour): the payload *rides*
  the moving door and the constraint is a speed bound on the arm's own push. Here
  the arm never drives the lid at all — the lid is driven by a **counterweight the
  solver loads and unloads**, and the payload never touches the moving member.
- ***close_grill_i8***: placement of a lid as the goal object itself; nothing is
  sealed in, nothing holds anything open.
- ***pen-holder / letterbox-style insertions***: static receptacles; here the
  aperture exists only while the mechanism is held open by ballast.

Same-strategy-different-numbers this is not: the seed has no self-closing closure,
no functional tool selection, no unload obligation, and no failure mode beyond
"item not in bbox".

## Solution outline (solve.py, teleport = transport only)

The lid is **never touched** by the solver (`lid_drive` stays zero, asserted).
Every teleport is a free-space transport a Franka performs; every load-bearing
interaction runs through live contact + hinge dynamics.

- **Phase 0 (reset):** settle 0.5 s; lid readback ~0° (rests closed);
  `SIM_GEN_SCORE` ≈ 0.000.
- **Phase 1 (load):** iron block teleported from its staging slot to a hover 60 mm
  above the cup floor along the cup's tilted normal (lid-frame pose through the
  live lid world pose), zero velocity, released. The drop, the seating in the
  48 mm-walled cup, and the lid swinging open to its 65° stop are all plant
  dynamics. `SIM_GEN_SCORE` 0.350 (approach + open latches).
- **Phase 2 (insert):** carton teleported to a hover over the open half of the
  mouth (vault-local y = −0.045, z = 0.33, yaw-aligned), released; it free-falls
  ~25 cm through the aperture and seats on the vault floor by contact
  (dt = 1/120 s, 24 mm floor ⇒ no tunneling). `SIM_GEN_SCORE` 0.650.
- **Phase 3 (unload):** iron block teleported out of the cup back to the carton's
  vacated staging slot (grasp knob, lift, carry, set down). The lid swings shut on
  its own — imbalance vs. viscous damping — and parks on the 0° stop, hands-off.
  `SIM_GEN_SCORE` 1.000.
- **Phase 4 (persistence):** ≥ 3.5 simulated seconds hands-off; `success()` is
  live state (a lid creeping open or a block slumping back would revert it). Only
  then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0, 1 (slots [0,1,2]) and 2 (slots [1,2,0]) all print
the monotone score sequence 0.000 → 0.350 → 0.650 → 1.000 → 1.000 and
`SIM_GEN_SOLVE: SUCCESS` (~20–25 s wall each).

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `appr_latch` (0.10): the iron block has ever been within 0.20 m of the cup.
- `open_latch` (0.25): the lid has ever swung past 50° (reachable by loading the
  cup — or by an arm holding the lid, which is real progress that then dead-ends).
- `insert_latch` (0.30): the carton has ever been fully inside (8-corner test) and
  calm.
- Success (→ 1.0): carton fully inside, lid within 3° of its stop, cup empty and
  both blocks resting off the vault below rim height, vault upright, everything
  settled and finite. Non-success capped at 0.65. Null policy scores ~0.

## Embodiment argument (single Franka, parallel jaw)

Base at ≈ (0, 0), facing +x; vault at ≈ (0.52, 0.06), staging table at
(0.22, −0.16) — all contacts within a 0.75 m reach disc, all approaches from above.

- **Iron / foam block (60 mm puck + 20 mm knob):** the knob is a purpose-built
  top grasp for the parallel jaw (20 mm across, 40 mm tall, fits the standard
  Franka fingers); pick from the table at z ≈ 0.12, carry, hover over the cup
  (cup centre at z ≈ 0.23 with the lid closed), open the gripper — exactly the
  solve's teleport-and-release. Unloading is the same grasp on the same knob,
  which stays up-facing: the cup's +30° pre-tilt means the cup floor is at −35°…
  +30° through the swing, and the block seats with its knob exposed above the
  44 mm-deep cup walls.
- **Carton (6×6×12 cm, 350 g):** side grasp on the 60 mm width (within the 80 mm
  jaw span), carry to a hover over the open half of the mouth at z ≈ 0.35, release
  — a 25 cm gravity drop, the same transport the teleport stands in for. The open
  strip at the 65° stop is ≈ 12 cm wide for a 6 cm carton.
- **No two-handed shortcut exists:** holding the lid open (a push on the plate)
  occupies the only arm, and then nothing can pick the carton — the iron block IS
  the second hand. The lid needs no arm contact at any point in the intended plan.
- **Execution order (REQUIRED and physically enforced):** load → insert → unload.
  Insert-before-load bounces off the closed lid (smoke 5); never-unload leaves the
  lid held open and the cup occupied (smoke 7); the lid can never be "left closed
  with the carton in" except by exactly this cycle.

## Checks (smoke.py — rejection battery, 14 checks)

1. Clean reset: finite state, lid resting closed (readback < 2°), all three items
   physically on their staging slots, PhysX mass readback matches the authored
   masses (vault 6.0 / iron 0.450 / foam 0.012 kg — guards the mechanism against
   the custom-spawner mass_props-ignored trap), score < 0.05.
2. Randomization by readback: vault xy/yaw and slot permutation differ across
   seeds; each item's world pose matches its assigned slot.
3. Null policy: 2 s of no action, score < 0.05, no success.
4. Decoy physics: the foam block, given exactly the iron's transport, lands in the
   cup and the lid **stays closed** (< 10°, open latch never arms) — the probe is
   asserted non-vacuous (foam actually in cup).
5. Seed-strategy rejection: carton released over the CLOSED vault never enters —
   no success, score < 0.05.
6. Accept side: the iron block's identical transport swings the lid to > 58° and
   holds it there 2 s.
7. Dilemma rejection: carton inside but iron still in the cup (lid held open) —
   no success, score ≤ 0.66.
8. Perched rejection: full goal except the iron parked ON the closed lid (above
   rim height) — no success (not "set aside"), score ≤ 0.66.
9. Transient guard: a mid-fall carton inside the vault footprint arms neither the
   insert latch nor success.
10. Wrong object: the FOAM sealed inside instead of the carton — no success,
    score ≤ 0.36.
11. Wrong place: the IRON sealed inside alongside the carton (lid held open by the
    drive probe, then released to seal) — no success, score ≤ 0.66.
12. Exactness: the constructed full goal state yields success and
    |score − 1.0| < 1e−3.
13. Latch semantics: prying the sealed lid back open (drive probe holds it > 45°)
    revokes success while held; score falls to the latched 0.65, not 1.0.
14. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
