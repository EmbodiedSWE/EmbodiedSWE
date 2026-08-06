# facing_lane — stock the red tin into a spring-loaded shelf-pusher lane

**Package:** `sim_gen/tasks_v6/put_groceries_in_cupboard_i329`
**Scene:** `simgen.facing_lane` (robot="null", scene-level)

## Seed provenance

- Seed: `rlbench/put_groceries_in_cupboard`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_groceries_in_cupboard.py`)
- Seed strategy: identify a named grocery among lookalike distractors, grasp it,
  carry it up, and SET IT DOWN on a passive open cupboard shelf. The receptacle is
  inert: any pose on the shelf surface, reached by lowering from above, is terminal.

## What changed and why it is strategically different

The cupboard keeps the seed's theme (stock a grocery into a bay, among decoys) but
the receptacle now fights back — it is a supermarket **facing lane** with a
spring-loaded pusher plate:

1. **The storage cavity does not exist at rest.** The orange pusher plate is sprung
   toward the front and parks with its face 3 mm behind the sill. There is no shelf
   surface to set anything on and no slot to drop anything into. The seed's
   put-it-down move terminates on the roof (smoke #6) and scores nothing.
2. **The cavity must be created by the tin itself.** The only way in is through the
   front window: press the tin horizontally against the plate, drive the plate back
   against its spring (~2 N, ~63 mm of sustained opposed contact), keep the spring
   compressed while lowering the tin behind the sill, then release. One
   parallel-jaw hand cannot hold the plate and place the tin separately — the tin is
   the tool that opens its own slot.
3. **Success is a read-back CLAMP, not a region.** Plate compression ≥ 45 mm AND tin
   front face within 12 mm of the sill inner wall AND upright AND still — the
   spring-held "faced" state a store lane produces. A tin loose in the lane behind a
   returned plate is seated but NOT successful (smoke #9).
4. **Metric interlocks:** the roof covers the rear of the lane; the only roofless
   patch (the 25 mm front strip) is narrower than the 55 mm tin (smoke #7); the
   plate-top-to-roof gap (40 mm) is also narrower than the tin, so nothing rides in
   over the plate.

A solver therefore needs a different PLAN (press-against-a-spring insertion through a
side window with held compression and timed release — not lift-carry-lower onto a
passive surface) and different CODE STRUCTURE (opposed-force control + compression
readback + release timing instead of pick-and-place waypoints).

Same-strategy-different-numbers? No: no parameter change to the seed reproduces "the
shelf does not exist until you press it open".

## Teleport-solution outline (solve.py — phases)

- P0 reset + settle; assert score ~0, plate at home.
- P1 TRANSPORT (teleport): one pose write stages the red tin hovering in the front
  window, 47 mm outside the sill, bottom 4 mm above the sill top. Asserted
  credit-free (no entry latch, zero compression, score unchanged).
- P2 PRESS (contact dynamics): grasp-emulation wrench (gravity compensation + z hold
  + attitude hold, i.e. the hand) plus a velocity-limited horizontal push drives the
  tin's face into the plate; the spring is compressed THROUGH the tin from 0 to
  ~63 mm, verified by joint-side readback. Every millimetre of the cavity is opened
  under opposed contact.
- P3 LOWER + RELEASE (contact dynamics): holding the live spring force through the
  tin, the z hold ramps the tin down behind the sill onto the lane floor; then all
  applied wrenches are cut at once. The returning spring closes the gap and clamps
  the tin against the sill — the success state is produced hands-off by the scene's
  own mechanism.
- P4 persistence ≥ 3.3 simulated seconds, no intervention; `SIM_GEN_SOLVE: SUCCESS`
  only if success() still holds. `SIM_GEN_SCORE` printed at every phase boundary
  (non-decreasing, latched credit).

Teleports move the tin across free space only; the spring compression, the descent
into the pressed-open slot, and the final clamp are all contact dynamics. The decoy
tins are never touched.

## Embodiment argument (single Franka arm, parallel jaw, OSC)

Plausible base pose: **(0, 0, 0), facing +x** (bay sill at 0.48 m, tins at
0.27–0.33 m — all inside the 0.30–0.55 m comfort envelope; working heights
0.04–0.20 m).

Per manipulated object:

- **Red tin (the only object the task requires moving).** Contact strategy: top-down
  side pinch across the 55 mm width (jaw opens to 80 mm) on the tin's upper half at
  its floor slot; lift to sill height + 4 mm. Insertion: wrist keeps the tin square,
  hand slides it horizontally into the front window; the tin's front face presses
  the plate back (~2 N peak — far below arm limits). Crucially the hand never has to
  enter under the roof: at full insertion the tin's REAR third (where the fingers
  pinch) is still under the 25 mm open strip / window zone, so the fingers descend
  in open sky while the tin's nose is under the roof. Lower 34 mm, open the jaw
  ~10 mm (fingers at ±36 mm < ±40 mm lane half-width — clears the walls), retract
  up through the strip. Precision required: ±10 mm lateral (12.5 mm slack per side),
  release height tolerance ~10 mm — comfortably above OSC noise.
- **Pusher plate:** never touched directly by the intended strategy (pressed via the
  tin). A direct fingertip press is possible (front zone is open above) but earns
  only the 0.15 press credit — one hand cannot hold it and stock the tin.
- **Green/blue decoys:** must NOT be moved into the lane; no contact required.

## Execution order

No declared order beyond what the mechanism physically forces: the spring must be
compressed (by the tin) before the tin can descend, and the clamp forms only after
release. The rubric encodes no timestamps; collision geometry enforces everything.

## Check list (smoke.py — 15 named checks)

1. settle/no-NaN (plate home, lane closed, tins at slots)
2. reset score ~0
3. randomization: slot permutation varies (readback)
4. randomization: xy jitter + yaw jitter real (readback)
5. null policy ~0
6. seed strategy (lower from above) → rests on roof → reject
7. 25 mm strip drop → tin cannot fit → reject
8. bare hand-press, released → spring returns, score ≤ 0.16 → reject
9. tin behind the pusher (seated, unclamped) → reject (load-bearing tolerance)
10. side-lying clamped tin → reject (upright clause)
11. latched credit survives regression
12. green decoy clamped instead → reject (identity)
13. approach-only → ~0
14. rejection audit (success never True in the battery)
15. final no-NaN

N/A note: the seed's literal end state (tin resting on an open shelf) is expressed
as #6 (roof rest — the only "shelf surface" the bay offers). Both-tins-in-lane is
geometrically unconstructible as a settled state (single-file lane); the decoy
clause is defense in depth, and #12 is what identity actually protects.

## Results (forge)

- solve: `SIM_GEN_SOLVE: SUCCESS` — seeds 0, 1 (see run logs)
- smoke: `SIM_GEN_SMOKE: ALL PASS 15/15`, frames.npz saved
