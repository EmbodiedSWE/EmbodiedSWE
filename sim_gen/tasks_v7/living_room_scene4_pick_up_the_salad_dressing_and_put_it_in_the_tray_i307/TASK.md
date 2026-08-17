# crate_dock — clear the blocker, slide the amber crate into the roofed dock

`living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i307`

## Seed provenance

Derived from `libero_90/living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray`
(RoboVerse `roboverse_pack/tasks/libero_90/...`): grasp the salad-dressing bottle among
distractors, carry it through free space, and drop it inside the wooden tray's open
containment region. The kept skeleton is "move the designated payload into the designated
container while other objects are in the way". Everything else is replaced.

## Why this is strategically different

The seed's plan is **prehensile transport**: grasp → carry → release over an open-top
container. Both halves of that plan are removed *by design*, and a third element (ordering)
is added — enforced by geometry, not by rubric fiat:

1. **No grasp.** Both crates have a 110 mm square footprint (156 mm diagonal) — wider in
   every horizontal direction than an 80 mm parallel jaw's stroke. They cannot be grasped
   or lifted, only pushed on their side faces (asserted in `CrateDockSceneCfg.__post_init__`).
2. **No drop.** The goal container (the dock) is **roofed**: its underside is 112 mm over
   the ground, 72 mm over the deck. Its only entrance is the corridor mouth at deck level.
   Nothing can be released into it from above; the payload must be slid in along the floor.
   The rubric's **mouth pathway latch** makes this load-bearing: success requires the
   payload to have been *observed* crossing the mouth window (smoke check 4 constructs the
   seed's end state — payload materialized inside the dock, genuinely settled and
   contained — and it is refused).
3. **Geometry-enforced ordering.** A taller gray blocker crate (95 mm tall; top at 135 mm)
   stands in the junction. It exceeds the 112 mm roof underside by 23 mm, so it can never
   be pushed forward through the mouth — it jams on the roof's proud front edge (10 mm
   ahead of the mouth plane). It must first be pushed **sideways** into the siding branch;
   only then can the payload cross the junction. Pushing the payload first merely rams the
   blocker into the roof edge, with the payload center pinned well short of the junction
   band (smoke check 6).

So a solver needs a different plan (route two ungraspable bodies through a corridor
topology in the correct order, purely non-prehensilely) and a different code structure
(rig-frame push-axis control on two different axes with jam recovery, instead of a
grasp-carry-release routine). It is also different from every other task read while
building it (in particular sibling `..._i18`, which actuates a wedge-jack mechanism to
tilt a hopper: single tool, mechanism actuation, drain-by-gravity — no corridor routing,
no ordering, no non-prehensile payload transport).

## Scene

Fully procedural (boxes only). A kinematic rig — randomized yaw ±30° and xy jitter each
episode, so the push axes must be read from the scene — carries a slick deck (top at
40 mm) with low walls forming: a 140 mm-wide **entry lane** (+x), a **junction**, a
**siding** branch (+y), and a roofed **dock** (mouth plane x = 0.160, roof underside
0.112, back wall 0.360). The amber payload (110×110×55 mm, 0.40 kg) starts in the entry
lane (randomized slot ±35 mm, lateral ±10 mm, yaw ±6°); the gray blocker (110×110×95 mm,
0.70 kg) starts in the junction (±8/10 mm jitter). Lane slack is 30 mm but the crate
diagonal exceeds the lane width, so crates cannot spin and wedge (asserted).

**success()**: payload settled at deck-rest height, center inside the dock containment
window (rig-frame x ∈ [0.225, 0.315], |y| < 0.060) **and** the mouth pathway latch is set.
**score()**: latched credit — blocker cleared into the siding 0.15, junction band crossed
0.15, mouth window 0.20, ever in the dock 0.25 (cap 0.75); exactly 1.0 iff success. The
junction credit is a 60 mm crossing *band*, uncrossable in one 120 Hz step at the push
speed caps, so teleport-past earns nothing.

## Solution outline (solve.py — zero teleports)

All state changes are contact dynamics; poses are never written:

- **P0** settle 1 s, layout readback, baseline asserts (score ≈ 0, no success).
- **P1** clear the blocker: speed-capped (0.08 m/s), stall-escalated CoM force along the
  rig **+y** axis until the blocker center passes y = 0.20, inside the siding
  (1.5 N base, 5 N cap — under the 0.70 kg crate's tipping bound F·h_com < mgw/2).
- **P2** dock the payload: same push style along rig **+x** (1.2 N base, 6 N cap) down
  the lane, across the junction band, through the mouth, under the roof, to rest near the
  back wall; jam recovery = one reverse un-wedge bout per hard stall.
- **P3** hands-off settle; success() must hold live.
- **P4** persistence: 3.3 simulated seconds hands-off; success still true →
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at each phase boundary: 0.0000 → 0.1500 → 1.0000 → 1.0000 →
1.0000 (non-decreasing; latched). Verified on the forge for **seed 0 and seed 3**
(both `rc=0`, ~19 s each).

## Franka embodiment argument

A single Franka with an 80 mm parallel jaw, base at rig-frame ≈ (−0.45, −0.35) facing
the deck (all interaction sites within ~0.75 m reach), closes its jaw and pushes with
the closed fingertips / knuckle side:

- **Blocker (P1)**: its south face has a 40 mm exposed strip above the 55 mm walls
  (wall top 95 mm, blocker top 135 mm); a horizontal fingertip push at ~100–130 mm
  height along +y matches the applied CoM force (≤ 5 N, ~0.5 kg-force — trivial for the
  arm; below the crate's tipping bound even applied above the CoM, since the solve's
  speed cap keeps it quasi-static and the 23 mm roof interference is not in this path).
- **Payload (P2)**: the entry lane is open-topped along its whole run (walls top out at
  95 mm; the roof only covers x > 0.150), and the payload's top is flush with the wall
  tops (40 mm deck + 55 mm crate = 95 mm), so its back face is reachable from above and
  behind at 45–90 mm height, straight down the open lane. The last
  110 mm under the roof are covered by pushing on the back face while the wrist stays
  behind the mouth plane — the payload is 110 mm deep, the dock window starts 65 mm past
  the mouth, and the fingertips (~20 mm thick) fit the 72 mm opening behind it.
- Forces stay ≤ 6 N horizontal at 40–135 mm heights — well inside Franka payload and
  workspace envelopes; no grasp, no lift, no regrasp is ever needed.

## Ordering declaration

The order is **geometry-enforced**: blocker-to-siding strictly before payload-to-dock.
The rubric mirrors it (junction/mouth credit is unreachable while the blocker plugs the
junction — smoke check 6), but the constraint itself is physical: the blocker is 23 mm
too tall for the mouth and there is no side-by-side room in the 140 mm lane.

## Smoke battery (smoke.py)

1. settle/no-NaN — seeded reset settles finite, layout as declared, score 0, no success.
2. randomization-is-real — 3-seed **max-pairwise** readback: rig yaw, rig xy, payload
   lane slot, blocker world position all differ.
3. null-policy — 240 idle steps: score ≈ 0, no success.
4. **SEED strategy / bypass** — payload materialized inside the dock window (the
   carry-and-drop end state, impossible under the roof), genuinely settled: refused for
   skipping the mouth; score ≤ 0.26.
5. cleared-only — blocker moved into the siding, payload untouched: exactly ~0.15.
6. **wrong order** — payload pushed with the junction plugged: blocker jams at the roof
   edge, payload never crosses the junction band; score ≤ 0.01.
7. mouth straddle — payload settled in the doorway: pathway credit only, not success.
8. **blocker-into-dock grind** — blocker advances freely on the open lane (probe is
   live), then jams at the roof front edge; never enters, never pops over.
9. rejection audit — success() never fired during checks 4–8.
10. frames.npz — video captured and saved to the CWD.
