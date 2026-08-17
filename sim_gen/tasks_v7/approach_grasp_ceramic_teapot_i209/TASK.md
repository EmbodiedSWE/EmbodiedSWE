# tea_ball_transfer (`approach_grasp_ceramic_teapot_i209`)

Unseal the capped tea canister, move the steel infuser ball into the OTHER (open)
canister, and seat the stopper lid in THAT canister's funnel mouth.

## Seed provenance

Derived from **`pick_place/approach_grasp_ceramic_teapot`**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_ceramic_teapot.py`):
a Franka approaches a ceramic teapot among table clutter and closes its jaw around
it; success is a gripper–object distance relation held for a few frames plus a small
joint lift — the episode ends HOLDING the object, and the rubric reads the gripper.

## What changed and why it is strategically different

The seed's plan (approach → grasp → hold) is discarded entirely; only the theme of
"squat ceramic tea vessels on a surface" survives:

- **Judged state, not judged grip.** No gripper relation is ever read; holding or
  possessing an object is worth exactly 0 (smoke check 7 constructs the seed's
  grasp-and-hold end state and gets only incidental credit, no success). The verdict
  is a settled physical END state: ball resting inside the destination canister AND
  the stopper wedge-seated level at bearing depth in that canister's mouth,
  everything at rest, hands off.
- **Ordered multi-goal pipeline vs a single grasp.** The payload starts physically
  CAGED: sealed under the seated stopper (resting ball top 44 mm < seated plug
  bottom 52.5 mm, walls all round). Smoke check 6 shoves it with a velocity-capped
  lateral force and it measurably rattles but cannot leave — so the order
  unseal → extract → transfer → re-seal is geometry-forced, not rubric-suggested.
- **A precision insertion the seed lacks.** The finish is a wedge seat: a 124 mm
  plug lowered into a 58→76 mm funnel collar (~±14 mm capture, contact dynamics
  self-centre it down the flats to a level bearing rest). Askew-on-the-rim fails
  (smoke check 10).
- **Target selection under randomization.** Which canister starts capped/loaded
  flips per episode (Bernoulli), both canisters get xy jitter + free yaw — a
  memorised fixed sequence targets the wrong canister half the time, and re-capping
  the ORIGINAL canister is an explicit failure end state (smoke check 8).

Also strategically distinct from the other artifacts examined: `pen_holder`
(fill an open cup — no seal, no caging, no ordering), sibling
`approach_grasp_ceramic_teapot_i109` (hang a mug on a peg by its handle window —
hanging mechanics, no containment/seal), and `hit_ball_with_queue_i77` (bridge
seating + gravity ball run — construction + dynamics, no unseal/transfer/re-seal).

## Assets (fully procedural compound spawners)

- **canister ×2** (KINEMATIC; terracotta / slate): base disc r 78 mm, 8 octagon
  wall boxes (inner inradius 65 mm, sill at 48 mm), 8 outward-tilted collar boxes
  forming a funnel mouth (inner inradius 58 mm at the sill → ~76 mm at the rim,
  rim ~68 mm). A resting ball sits fully below the aperture.
- **lid** (DYNAMIC 0.15 kg, cream): plug disc r 62 mm × 14 mm (origin/CoM), cap
  disc r 82 mm × 8 mm, dark square knob 20×20×26 mm. Analytic seat plane 52.5 mm,
  lid origin seat ~59.5 mm (measured on GPU convex hulls: 61–64 mm, inside the
  ±8 mm depth band).
- **ball** (DYNAMIC 60 g): steel-grey sphere r 18 mm.

## Randomization (readback-verified in smoke)

Per-canister xy jitter ±40 mm + free yaw ±180°; Bernoulli role swap (`src_is_a`);
ball xy jitter ±12 mm inside the capped canister.

## Rubric (latched; anchored in the demonstrated solution)

- `0.15 * opened` — lid ever clearly off the source mouth (latched)
- `0.25 * transferred` — ball ever resting inside the DESTINATION (latched)
- non-success cap **0.40**; **1.0 iff success()**: ball in destination (below the
  sill, any interior floor rest incl. octagon corners) & still, AND lid seated in
  the destination mouth (axis ≤ 12 mm, depth ±8 mm, tilt ≤ 8°, still).
- Null policy scores 0 (the lid starts seated; nothing fires by itself).

## Teleport-solution outline (solve.py, passes seeds 0 and 1)

1. **P1 UNSEAL (contact):** velocity-servoed vertical force at the lid CoM lifts
   the stopper straight out of the funnel (frame-drag probe + stall escalation).
2. **P2 park (transport):** lid teleported to a hover over open floor; falls and
   rests there (never written into any goal-relevant state).
3. **P3 EXTRACT (contact):** vertical velocity-servo force lifts the ball out
   through the open mouth to clear air above the rim.
4. **P4 DEPOSIT (transport + contact):** ball staged hovering over the destination
   mouth, released; falls through the funnel, settles inside → score 0.40.
5. **P5 SEAT (transport + contact):** lid staged level over the destination with a
   deliberate 6 mm lateral offset (inside the ±14 mm capture), released; the plug
   self-centres down the collar and wedges at bearing depth → success, score 1.0.
6. **P6:** ≥3.3 s hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary; asserted non-decreasing
(0.00 → 0.15 → 0.15 → 0.15 → 0.40 → 1.00 → 1.00).

## Embodiment argument (single Franka, base at the origin facing +x)

Both canisters stand at ~0.43–0.49 m reach, 68 mm tall — comfortable workspace.

- **Lid:** the 20 mm square knob is a clean parallel-jaw pinch (80 mm jaw span,
  knob proud on top of the cap). Lift is a straight vertical retract; placement
  needs only ~±14 mm / roughly-level accuracy — the funnel does the rest, and the
  solve demonstrates a 6 mm-offset release still seats.
- **Ball:** 36 mm sphere < jaw span with 44 mm margin. Top-down grasp through the
  open mouth: the top opening is ~152 mm across and the ball's equator sits
  ~42 mm below the rim, well within Franka finger length (~54 mm) plus wrist
  approach; in-pot jitter is ±12 mm around the axis.
- **Transport:** all moves are short free-space arcs between two floor canisters
  0.38 m apart; one base pose (origin, facing +x) reaches both.

## Execution-order declaration

`unseal source → park lid → extract ball → deposit ball in destination → seat lid
in destination`. The first step is geometry-forced (caged payload, smoke check 6);
the seal must come last on the destination (its mouth must be open to admit the
ball); re-capping the source instead is a rejected end state.

## Checks (smoke.py — SIM_GEN_SMOKE: ALL PASS 14/14)

1. settle/no-NaN: lid seated on source, ball inside, all still
2. reset score ~0, no success
3. randomization readback: per-canister xy jitter + free yaw real
4. randomization readback: Bernoulli role flip + ball in-pot jitter real
5. null policy: 240 idle steps → score ~0, no success
6. caged ball: velocity-capped lateral shove moves the sealed ball (>8 mm,
   non-vacuous) but it never passes the sill / leaves / latches transfer
7. seed strategy (grasp-and-hold end state): only opening credit 0.15, no success
8. undo / wrong pot: lid re-seated on ORIGINAL with ball inside → 0.15, no success
9. missing seal: ball transferred but lid re-seated on SOURCE → exactly the 0.40
   cap, no success
10. near-miss seat: lid dropped 32 mm off-axis + 18° tilt ends askew on the rim →
    not seated, no success
11. near-miss payload: lid perfectly seated in DESTINATION, ball on floor beside →
    0.15, no success
12. latched credit: probe transfer earns exactly 0.25; teleporting the ball back
    out leaves it latched, still no success
13. rejection audit: success() never True at any judged point
14. final no-NaN
