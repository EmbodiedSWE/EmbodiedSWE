# cloche_service — serve the cake under the black dome cover

`libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221`

## Seed provenance

Seed: `libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate`
(`RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate.py`).
The seed is a single blind pick-and-place: a Franka carries a known **black bowl** onto a
passive, empty **plate**; `_terminated` checks bowl-center xy within 0.06 of the plate and a
0–3 cm height gap. One object moves once; the target surface starts free; success is
"A on top of B".

## What changed, and why it is strategically different

The same two protagonists return with inverted roles:

- The **black bowl** becomes a black **dome cover** (a cloche): the bowl upside-down with a
  pinch-graspable knob on top. It starts **seated on the plate** — the goal surface begins
  OCCUPIED by the very object the task must end with.
- The **plate** is no longer the terminal drop zone but a build site: a small pale **cake**
  (5.6 cm cylinder) waits on the floor beside it.

Goal: the cake standing upright in the middle of the plate, **hidden under the re-seated
cover** — cover knob-up, its open rim resting flat on the plate, the cake fully inside the
covered space, everything settled.

Strategic differences from the seed (and from the sibling i187 beam-balance task, which is a
hidden-parameter *measurement* task with a lever mechanism):

1. **The seed's plan is impossible as stated.** "Carry A onto B" fails because B starts
   blocked. The blocker is not trash — it is the reused final component. The plan becomes
   uncover → serve → re-cover, with the same object handled twice (removed, then re-placed
   in a different relational role: obstacle → enclosure).
2. **The ordering is forced by geometry, not rubric fiat** (asserted in
   `ClocheServiceSceneCfg.__post_init__`): a seated rim touches the plate all around (zero
   aperture), and the free annulus between the cover wall and the plate rim inner face
   (≈30 mm) is narrower than the cake diameter (56 mm) — the cake physically cannot reach
   the plate top while the cover is seated. Smoke check 15 drops a cake onto the covered
   plate and verifies it deflects off.
3. **Success is ENCLOSURE, not on-top.** Cover-on-plate is necessary but not sufficient;
   the cake must be on the plate *and* inside the seated cover. The seed's own literal end
   state — the bowl in open-bowl pose on the plate with the cake inside it — is explicitly
   rejected by the upright + rim-height clauses (smoke check 10).
4. Novel failure surface: the re-cover drop must funnel the rim *around* the standing cake
   (perched-on-cake is a rejected negative, check 13), and the final state must survive a
   3.5 s hands-off persistence window.

## Scene

Everything procedural, no external assets, no joints, robot="null", one env, ground plane.

- **plate**: custom compound spawner — white disc (r 0.115, t 0.014) + 12 rim box segments
  (h 0.012); free rigid, 1.5 kg (MassAPI authored in the spawn func; smoke asserts
  `get_masses()` readback).
- **cloche**: custom compound spawner — 10 wall box segments (inner r 0.062, wall 8 mm,
  h 0.075, open rim at local −wall_h/2) + roof disc + knob cylinder (r 0.025, h 0.022);
  0.30 kg. Body +z = knob direction.
- **cake**: plain cylinder r 0.028 × h 0.036, 0.10 kg.

Randomization (verified by readback in smoke check 4): plate base (0.34, 0) ± 0.03 xy +
free yaw; cover seated on the plate with ±6 mm xy jitter + free yaw (dropped the last
4 mm); cake side of the plate **sampled** (y = ±0.24, via `torch.rand` comparison — first
`randint` after seeding is degenerate on this stack), ±0.03 xy jitter + free yaw.

Rubric: `score() = 0.15·uncover_ever + 0.25·serve_ever` (latched in `post_step` every
step; clamp 0.90), exactly 1.0 iff `success() = cake_on_plate & seated & enclosed &
settled`. Enclosure tolerance uses the wall's inner CORNER reach (10-gon,
0.062/cos(π/10) ≈ 0.0652 − cake_r), so a cake genuinely resting against a wall corner
counts. Null policy scores 0.

## Teleport solution (solve.py)

Teleports are TRANSPORT ONLY; every release point is asserted OUTSIDE the credit bands, and
every placement finishes through gravity + contact:

1. **UNCOVER** — transport the resting cover to a park spot 0.32 m away (> uncover_dist
   0.20), released 6 mm up; it drops onto its rim and settles standing. → score 0.150
2. **SERVE** — transport the cake to 50 mm above its on-plate rest height (asserted above
   the `cake_z_hi` band; asserted no on-plate credit at the release instant); it falls and
   settles standing on the plate. → score 0.400
3. **RE-COVER** — transport the cover to just above the plate, deliberately off-center
   (+10, −7 mm), rim released above the cake top and asserted above the `seat_z_hi` band
   (no seated credit at the release instant); the falling rim must clear the standing cake
   and land flat on the plate — seating is pure contact dynamics. → score 1.000
4. **PERSIST** — ≥ 3.5 simulated seconds hands-off; `success()` must still hold (a cover
   perched on the cake or still rocking fails here). Then `SIM_GEN_SOLVE: SUCCESS`.

Also: mass readback asserts for all three bodies at reset; `SIM_GEN_SCORE` printed at each
phase boundary with a monotonicity assert; daemon watchdog + `os._exit` hard exit.
Passes seeds 0 and 1 on the forge.

## Embodiment argument (single Franka arm, parallel-jaw gripper, OSC)

Base at ≈ (−0.10, 0, 0) facing +x: the plate center (x ≈ 0.34 ± 0.03) and both cake slots
(0.12, ±0.24) all sit within a 0.30–0.55 m reach disc at floor height — comfortable for a
floor-mounted Franka.

- **Cover**: pinch-grasp the knob (Ø 50 mm ≤ 70 mm jaw opening, asserted in cfg) from
  above; lift vertically ≥ 12 cm (clears the rim), place at the park spot, release. The
  same grasp re-seats it: hover over the plate, lower until the rim is ~2 cm above the
  cake top, release and let it seat — exactly the solve's release, so the contact
  dynamics the teleport solution exercises are the ones the arm would produce.
- **Cake**: side pinch on the 56 mm cylinder (≤ 70 mm jaw) or top pinch across the
  36 mm height; lift, hover over the plate center, release a few cm up — again the
  solve's release condition.
- No step needs more than one object held at a time, no bimanual squeeze, no in-hand
  re-orientation: the cover is carried in the same knob-up pose it starts and ends in.

## Execution-order declaration

The rubric judges the END STATE plus two latched progress bits; the uncover → serve →
re-cover order is **physically forced** by the zero-aperture seated rim and the
too-narrow plate annulus (see above, asserted in cfg, probed in smoke check 15). No
rubric clause depends on trajectory shape.

## Smoke checks (15)

1. Reset settles finite; cover seated; cake on the floor off-plate.
2. Fresh-reset score 0, no latches, no success.
3. Mass readback (plate/cloche/cake) matches authored masses.
4. Randomization by READBACK over seeds 21–26: plate xy spread, cake xy spread, cake
   SIDE flips (both signs seen), cover yaw spread.
5. Null policy: 240 steps hands-off → score stays 0, no success.
6. Oracle seed 0 → success, score 1.0, persists 240 steps.
7. Oracle seed 1 → success, score 1.0.
8. Score monotonicity ladder: 0 → 0.15 (uncover) → 0.40 (serve) → cake knocked off →
   still 0.40 (latched) → 1.0 (full oracle); partials < 1.0.
9. Partial-credit caps: uncover-only and serve-only both < 1.0 (0.15 / 0.40).
10. **Seed-strategy end state rejected**: cover flipped into open-bowl pose on the plate
    with the cake dropped inside it → not seated, cake not at plate-top height, no success.
11. Served-but-uncovered → no success (score latched 0.40).
12. Covered-but-empty (cake elsewhere on the floor) → no success.
13. Cover perched on an off-center cake (rim on cake top, not the plate) → no success.
14. Cake dropped on the cover roof/knob → not on plate, no success.
15. Forced order probed: cake dropped onto the COVERED plate → deflects off (moves
    off-axis, ends below the roof line), cover stays seated, not enclosed, no success.

Camera frames recorded to `frames.npz` in CWD. Prints `SIM_GEN_SMOKE: ALL PASS 15/15`.
