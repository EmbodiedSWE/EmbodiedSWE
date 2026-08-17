# rack_tray_serve — fetch the captive tray, seat it in the well, serve the front bowl

`simgen.rack_tray_serve` · `sim_gen/tasks_v7/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate_i357`

## Seed provenance

- **Seed**: `libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate`
  (RoboVerse `roboverse_pack/tasks/libero_90/...`): three identical akita black bowls in a
  row on a kitchen counter, a plate sitting ready beside them; goal = pick the bowl **at the
  front** of the row and set it on the plate. `_terminated` is a single xy/z window of the
  front bowl relative to the plate — one unordered pick-and-place onto a waiting surface.
- **Kept from the seed**: the cast (three identical black bowls, of which the FRONT one is
  the target, plus a flat white "plate" as the serving surface) and the final relational
  goal (front bowl resting on that surface).

## What changed, and why it is strategically different

The seed's entire strategy is *transport the target to a goal that is already prepared*.
Here the goal surface itself is the problem:

1. **The "plate" (a square white TRAY) starts CAPTIVE.** It is stowed **on edge** inside a
   tall dish rack: a 16 mm channel between 200 mm walls, with a roof rail leaving ~4–6 mm of
   headroom over the stowed tray's top edge. The seed's universal primitive — *lift the
   thing* — is geometrically impossible (smoke check 4: a pull at 3× the tray's weight rises
   6 mm and jams). The tray's only escape is its one open DoF: a ~140 mm **constrained
   slide** out through the rack's mouth onto a low-fenced apron (validated as the real free
   DoF by smoke check 5).
2. **The goal fixture is an empty recess, not a surface.** Success requires the tray
   **seated flat on the floor of a shallow square well** (190 mm inner for the 170 mm tray;
   entry demands yaw alignment within ~8°). A rim-perched, tilted, or 45°-misyawed tray
   rests ≥ 8–12 mm high and is rejected by the ±4 mm seat band (z-stratification: seat
   15 mm vs rim 27 mm vs tray-on-bowl 63 mm above the counter).
3. **Execution order is forced by physics, not by fiat.** The literal seed plan — put the
   front bowl at the goal — drops it into the bare well, where it becomes an obstruction: a
   tray dropped on top rests 48 mm high and can never seat (smoke checks 6–7). The bowl
   must come out again; *prepare the destination first* is the only viable plan. The task
   is reversible, so this is a physical order-forcer with no spoil latch.

So the solver needs a different plan (extract → reorient 90° → install → then place), and
different code structure (an in-channel force slide under a captivity constraint, a
recess-seating with yaw alignment, then a relational stacking placement judged **relative
to the tray**, plus a decoy-exclusion clause) instead of the seed's single
grasp-carry-set-down. No sibling in `tasks_v7` uses a *captive goal-surface extraction +
recess install + serve* structure (nearest neighbours: i227 slides captive tiles as a
lock, i296 twist-locks the transported object itself, i189 senses hidden mass).

## Scene (procedural, no asset files)

- Counter (static, top at 0.40 m); kinematic **rack** (floor strip, two 200 mm slot walls
  16 mm apart, back wall, roof rail over the rear, two 35 mm apron fences), origin at the
  mouth, re-posed each reset (xy ±15 mm, yaw ±8°); kinematic **stand** (base slab + 12 mm
  rim walls forming a 190 mm square well), xy ±20 mm + **free yaw**; dynamic **tray**
  (170×170×10 mm, 0.15 kg, stowed on edge at a jittered depth); three dynamic identical
  black **bowls** (96 mm, 48 mm tall, 0.10 kg), dealt over the three row slots by a fresh
  permutation; the TARGET is whichever body landed on the FRONT slot (largest x — the end
  nearest the counter's front edge, the side the rack mouth faces), latched like the seed's
  "bowl at the front".
- **Rubric** (monotone; running-max + latched stages): 0–0.30 extraction progress →
  0.35 tray out of the rack → 0.60 tray seated → 0.80 seated + target bowl on it →
  1.0 iff `success()`: seated ∧ target on tray (55 mm / dz (−3,20) mm / upright) ∧ both
  decoys off the tray ∧ settled. All geometric clauses are live readback; only the target
  identity is a latched episode fact.

## Teleport-solution outline (solve.py, both interactions honest)

- **P0** settle 120 steps; readback masses, rack/stand poses, stow depth; target chosen
  **by position** (body nearest the front slot) and cross-checked against the scene latch;
  baseline score ≈ 0.
- **P1 extraction (applied force)**: bang-bang horizontal force along the rack's +x from
  live yaw readback (0.9 N, ×1.5 on 120-step stall, cap 4 N; velocity cap 0.12 m/s), paired
  with the torque of the same force acting 80 mm below the CoM (a low fingertip push point)
  so escalation cannot pitch the on-edge board; slides until rack-frame x > 105 mm → 0.35.
- **P2 transport + gravity seat**: one root-state write to 30 mm above the seat, flat, yaw
  = stand yaw readback (airborne: satisfies no clause); the drop through the 10 mm-per-side
  slack seats it → 0.60.
- **P3 serve**: teleport the target bowl 25 mm above the seated tray's center; gravity
  places it → success → 1.0.
- **P4 persistence**: 3.33 s hands-off, success still true → `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, parallel jaw, OSC)

One plausible base pose: **on the counter's front-right edge, base at (0.62, 0.03) on the
counter plane, facing −x** — every manipulated pose (rack mouth 0.51 m, apron 0.38 m,
stand 0.52 m, front bowl 0.48 m, all ≤ 0.20 m above the counter) is inside a 0.65 m
comfortable reach envelope, and the rack mouth faces the robot.

- **Tray, extraction**: the stowed tray's top edge is exposed ahead of the roof rail (the
  rail ends 35 mm behind the mouth); pinch the exposed 10 mm top edge with the fingertips
  and drag along the channel — exactly the low-force, velocity-capped slide the solve
  applies (≤ 4 N ≪ Franka payload), with the fences guiding the board. The solve's wrench
  emulates this fingertip contact (force + low-contact-point torque).
- **Tray, install**: once on the apron the board stands proud of the 35 mm fences; pinch
  the top edge, lift, rotate the wrist 90° to flat, hover over the well aligned to the
  stand's heading (both visible), and release ~2–3 cm above the seat — the validated
  drop-seating in solve P2 is precisely this release, and the ±10 mm slack + 55 mm bowl
  tolerance absorb release scatter.
- **Bowl**: 96 mm outer width exceeds the 80 mm jaw span, but the bowls are OPEN boxes
  with 8 mm walls — the canonical Franka rim pinch (one finger inside, one outside a
  48 mm-tall wall) carries it; hover over the tray center and release low. Decoys never
  need to be touched on the golden path.
- **Order**: the gripper cannot make the tray seat over an occupied well any more than the
  teleport drop can (contact geometry, not controller, refuses it).

**Execution order is REQUIRED**: extract before install (captivity), install before serve
(a bowl in the bare well physically blocks seating and earns nothing). Bowl-first is a
recoverable dead end, not a spoil.

## Verification (forge, RTX 4090, Isaac Sim 4.5)

- `solve --seed 0` and `--seed 1`: `SIM_GEN_SCORE` 0.0000 → 0.3500 → 0.6000 → 1.0000
  (monotone), `SIM_GEN_SOLVE: SUCCESS`, rc=0 (~19 s each).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 13/13`, rc=0 (~28 s), frames.npz (29×600×960×3):
  1. settle/no-NaN + authored-mass + stow readback, score ~0
  2. randomization real (8 seeds: 8 stand yaws, 3 targets, 410 mm jitter spread, 10.8 mm stow spread)
  3. null policy: no latches, score ~0
  4. captivity: 4.5 N up-pull (3× weight) rises 6 mm, tray stays racked
  5. free-DoF contrast: same 4.5 N along the channel slides 60 mm
  6. SEED STRATEGY: front bowl into the bare well → success False, score 0.000
  7. blocked install: tray onto occupied well rests +48 mm, never seated
  8. wrong place: tray on counter + bowl on it → on_tray True, no seat/load credit
  9. rim-perch: 45° misyaw rests +12 mm, not seated
  10. offset near-miss: bowl 65 mm off-center → no load credit (cap 0.60)
  11. wrong bowl centered on seated tray → decoys_clear False, no success (cap 0.60)
  12. contamination: correct target + intruding decoy → loaded latches (0.80) but success refused
  13. video frames saved
