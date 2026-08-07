# libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i3 — BayonetLidScene (`simgen.lid_bayonet`)

Seal the stock pot with its keyed BAYONET lid: align the lid's three lugs with the
three entry gaps in the pot's brass collar ring, lower it through them onto the
internal ledge, then TWIST it CLOCKWISE under contact until the lugs hit the end
stops. A lid merely SET DOWN on the pot — the seed's whole plan — physically rests
on the tab ring 26 mm too high and is rejected.

## Seed provenance

- **Seed task**: `libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate`
  (RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate.py`)
  — "put the white bowl on the plate". A kitchen scene with a white bowl, a plate
  and a microwave distractor; the plan is ONE rigid pick-and-place, judged purely by
  the transported object's resting position: `xy_distance(bowl, plate) < 0.06 and
  0 < z_bowl - z_plate < 0.03` — no orientation clause, no mechanism, no contact
  beyond "set it down".

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| goal mechanism | none — open support surface | a **bayonet coupling**: keyed entry gaps, overhanging tabs, directional end stops — the goal state is *inside* a mechanism |
| core act | set the bowl down near a target pose | **rotation under contact**: with the lid seated, twist it clockwise so the lugs slide under the tabs until the stops arrest them; the final angle is produced by a hard-stop contact, not by a placement |
| orientation | irrelevant | load-bearing twice: yaw must match the gaps (mod 120°) to enter, and the final engagement angle (16–40°) *is* the goal |
| ordering | none — one placement | **insert-before-twist, enforced by geometry**: the locked window is reachable only by sweeping the lugs through the tab channels; twisting the wrong way (CCW) jams at δ≈106° (smoke 7) |
| the seed's end state | = success | expressible and scored ≤0.30: the lid released on the pot rests ON the tab ring at ~86 mm — 16+ mm above the 50–70 mm seat band (smoke 4) |
| distractor | passive microwave | a **decoy lid** (blue knob) that is *physically keyed out*: its 78 mm lugs overshoot the 72 mm collar wall, so it cannot enter at any yaw (smoke 8) |
| perception | fixed layout | pot xy jitter + **full pot yaw** (the gap pattern rotates — read it back, don't memorize), lids **swap start sides**, lid xy jitter + free yaw |
| assets | LIBERO USD kitchen assets | 100 % procedural (compound spawners): kinematic pot with 12-segment collar, ledge ring, 3 tab sectors, 3 end stops; two dynamic lug lids |
| judging | one bbox-style pose check | latched partial credit (lift / over-collar / seated) + live success: seated in the 50–70 mm band AND engagement in the locked window AND settled |

Code structure shares nothing with the seed: `@SCENES.register` BaseScene with three
compound spawners, an engagement-angle predicate (`(lid_yaw − pot_yaw) mod 120°`),
latched credit in `post_step`, `register_env(..., robot="null")`.

## Why strategically different

The seed's entire skill is *grasp one rigid object, set it down near a target pose*
— tolerance-loose, orientation-blind, judged by the object's own resting position on
an open surface. Here that plan is worth almost nothing: smoke 4 constructs exactly
the seed's end state — the lid released squarely on top of the pot — and the lid
comes to rest on the brass tab ring at ~86 mm, out of the 50–70 mm seat band, score
≤0.30, no success. What the solver must bring instead is (1) **keyed insertion**:
the pot's heading is randomized, so the solver must *perceive* where the three entry
gaps are and match the lid's yaw mod 120° (±16°) before lowering — a pose match to a
*mechanism*, not a surface; (2) **rotation under contact**, the skill the corpus
does not have (i2 pours by gravity, other tasks push/drop/stack): the seated lid
must be twisted ~65° clockwise while the lugs slide inside the tab channels, and the
goal angle is *produced by the end-stop contact*; (3) **directionality**: the same
twist counter-clockwise jams almost immediately (smoke 7) — the coupling encodes an
irreversible choice of direction; and (4) a **physically keyed decoy**: the wrong
lid is rejected by geometry, not by rubric fiat. The lock is real retention, not
bookkeeping: an engaged lid survives a 3×-weight straight pull under the tabs while
a disengaged one pulls straight out at 1.5× weight (smoke 9).

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1 s; readback layout (pot xy + yaw, red-lid side/pos/yaw); baseline
   score 0, no success.
2. **P1** (teleport = transport only): "hold" the red lid by writing its root pose in
   small per-step increments — lift, carry high over the pot while rotating in FREE
   SPACE to the entry yaw (`pot_yaw + 90°` mod 120°, nearest branch), descend to a
   hover with the lid bottom 12 mm above the rim. Lift + over latches → score 0.25.
3. **P2** KEYED INSERTION: release; the aligned lid FALLS through the three entry
   gaps and seats on the internal ledge at 60 mm (retry loop re-grabs and re-drops if
   a bounce misaligns it — never needed in three seeds). Seat latch → score 0.75.
4. **P3** CLOCKWISE TWIST: speed-governed torque about −z (0.12 N·m pulses,
   escalating if stalled); the lugs sweep from δ=90° down the tab channels and the
   END STOPS arrest them at δ≈25–28°; two extra press pulses seat the lugs firmly
   against the stops; torque cleared, everything settles → success, score 1.0.
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

The lid's pose is never written after the release: seat height, engagement angle and
settledness are all outcomes of gravity, contact, and the applied torque. Monotone
`SIM_GEN_SCORE` prints: 0.0000 → 0.25 → 0.75 → 1.0 → 1.0.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m). Lids
start ≈ 0.22 m ahead at (0.12, ±0.18); the pot at (0.40, 0) — everything inside a
comfortable dexterous shell; all manipulation below 0.25 m.

- **Grasping the lid** (88 mm disc, 250 g, central knob Ø 24 mm × 50 mm tall): a
  knob pinch — the parallel jaws close on the cylindrical grip knob from above
  (opening 80 mm ≫ 24 mm). The knob is on the lid's axis, so the grasp is
  yaw-symmetric: any approach yaw works regardless of the lid's randomized heading.
- **Keyed insertion**: with the knob pinched, the required yaw alignment (±16° about
  a 120°-periodic target) is a pure wrist roll — joint 7 spans ±2.9 rad, far more
  than the ≤60° worst-case correction. Lateral entry tolerance is ~6–10 mm (lug
  outer 62 mm vs wall inner 72 mm) at a 12 mm drop — coarse by insertion-task
  standards, and the solver may release just above the rim exactly as solve.py does,
  letting gravity finish the entry.
- **The twist**: a wrist roll of ~65° clockwise about the pinch axis (one motion,
  no regrasp — the knob is round so the fingers can also slip-regrasp if desired).
  Required torque ≈ 0.1–0.3 N·m about the lid axis, transmitted as a friction couple
  across the 24 mm knob — trivial for Franka's wrist (max joint-7 torque 12 N·m).
  The hard stops make the endpoint *tactile*: the arm twists until the wrench spikes,
  no angle estimation needed.
- **Payload/forces**: 250 g lid; the pot is kinematic (a heavy fixture), so
  incidental contact cannot move the goal frame. The only contact-rich moments —
  lug-on-ledge seating and lug-on-stop arrest — are exactly the compliant-contact
  events OSC handles well.
- **The decoy** requires nothing: it is identified visually (blue vs red knob) and
  rejected physically if tried.

## Execution order (declared)

`align yaw to the gaps → lower through the gaps onto the ledge → twist clockwise to
the stops`. The order is enforced by geometry, not rubric fiat: the locked window
[16°, 40°] is unreachable from above (the tabs at 78–86 mm block descent everywhere
except the gaps at δ≈90°, and a lid on the tab ring rests out of the seat band —
smoke 4), so the only path into the goal set sweeps the lugs from the gaps through
the tab channels to the stops; twisting counter-clockwise instead jams at δ≈106°
(smoke 7), and the seat band + upright clause reject every other resting place
(smoke 5, 6, 10).

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0`: SUCCESS, scores 0.0000/0.2500/0.7500/1.0000/1.0000, lock
  δ=26.7° (40 s).
- `solve --seed 1` (different pot yaw/jitter): SUCCESS, same monotone scores, lock
  δ=27.1°.
- `solve --seed 2` (lids swapped sides): SUCCESS, lock δ=27.8° — three seeds, the
  re-drop retry never needed.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 11/11**, frames.npz (134 frames) saved:
  1. settle/no-NaN; both lids flat on the ground; score 0, no success
  2. randomization readback: pot xy (max Δ 57 mm) AND pot yaw (max Δ 160°) differ,
     red lid's start side swaps across seeds
  3. null policy: 240 idle steps, score 0, no success
  4. SEED STRATEGY: lid released ON the pot (lugs over the tabs) rests on the
     brass ring at exactly 86.0 mm, out of the 50–70 mm seat band → no success
  5. no twist: lid dropped through the gaps, seated at δ=90° → no success, 0.75
  6. near-miss: partial twist to δ=48°, short of the stops → no success
  7. CCW JAM: counter-clockwise torque jams at δ=104.7°, never enters the locked
     window → no success
  8. decoy keyed out: blue lid dropped aligned cannot enter the collar, rests at
     86.0 mm → no success
  9. RETENTION interlock: engaged lid (δ=48°) rises only to 66 mm under a
     3×-weight straight pull (tabs catch the lugs) and re-seats; disengaged lid
     (δ=90°) pulls straight out at 1.5× weight (z > 205 mm)
  10. flipped lid: upside-down in the collar rests at 72 mm with +z inverted →
      not seated, no success
  11. video frames.npz saved (134 × 600 × 960)

## Files

- `scene.py` — BayonetLidScene + pot/lid compound spawners + rubric; registers
  `simgen.lid_bayonet`.
- `solve.py` — teleport-transport + gravity insertion + torque twist certificate
  (`--seed N`).
- `smoke.py` — 11-check rejection battery + video.
