# libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i118 — Pudding carousel vault

**Scene:** `simgen.pudding_carousel` (`PuddingCarouselScene`, robot="null")
**Seed:** `libero_90/libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it`
(`RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it.py`)

## Seed provenance and what changed

The LIBERO seed is *put payload in receptacle, then close it*: pick the chocolate
pudding box, **drop it from above** into the already-open top **prismatic** drawer,
then push the drawer shut. The receptacle is passive (it neither moves the payload
nor constrains when it can be loaded), top access is free, and the two goal clauses
are essentially order-free — you could even close the drawer partway, reopen, load,
and close again; nothing in the geometry cares.

This task keeps the payload-sealed-in-a-receptacle goal but replaces the plan
skeleton:

1. **The receptacle is the closure member.** The drawer becomes a **revolute
   carousel drum** inside a fixed square shroud: a wooden turntable with ONE
   side-opening cargo bay, a full-disc roof, a sealing flank wall, and a red
   overhead lever, hanging on a Z-axis axle with hard stops 92° apart. At the 0°
   stop the bay faces the shroud's only window (LOAD); at the 92° stop the drum's
   flank wall fills the window and the bay faces the solid back (SEALED). The same
   articulation both *carries the payload* and *closes the vault* — the seed's
   "place, then independently close" decomposition does not exist here.
2. **Loading is a side-load through a fixed aperture, not a drop.** The drum roof
   kills the seed's own move: a box dropped from above lands ON the roof and never
   enters (smoke check 4). The only way in is horizontal — through the 110 mm-wide
   front window, across a deep sill whose top is flush with the bay floor (3 mm rim
   gap), into the bay.
3. **Execution order is forced by geometry, both ways.** The drum starts ≥ 48° from
   aligned; the bay/window half-angles (≈ 22.4° + 21.4°) guarantee zero angular
   overlap, so the window sees only the curved rim and wall edges — the same
   fingertip push that loads in solve.py wedges dead and does not even rotate the
   drum toward alignment (smoke check 5: 49.2° → 49.0° after 5 s of pushing).
   Sealed-first is equally dead: the flank wall turns into the window and the box
   stops at the window plane, ~10 cm from containment (smoke check 6). ALIGN → LOAD
   → SEAL is the only executable order, with no temporal bookkeeping in the rubric.
4. **The seal is a carry.** Rotating the loaded drum transports the box ~90° around
   the axle held by nothing but friction and the bay walls — payload retention rides
   on the closing articulation itself.

So the plan skeleton changes from *"pick, drop in, push shut"* to *"rotate the
mechanism to expose its single loading window, thread the payload through sideways,
then rotate the loaded mechanism to its sealed stop"*.

## Why strategically different from every examined sibling

Sibling i332 (same seed) was examined and deliberately avoided: it is a **ramp push
into a static roofed vault + a gravity-assisted drawbridge door past vertical** — its
receptacle is static and its closure member is a separate door. Here the receptacle
itself is the articulated member, it must be *aligned before loading* (i332 has no
pre-loading articulation step), and it *carries the payload during closure*. Other
examined corpus tasks (egg-on-fridge-door gentle close i89, oven dials, drop-gate,
bayonet twist-lock, tray/pin, letterbox rack, rammer gallery, …) share machinery
(authored hinges, drive buffers) but none contains this task's core mechanism: **a
rotary receptacle whose single aperture must be indexed to a fixed window twice —
once to admit the payload, once (at the opposite stop) to seal it**. The closest,
i89, loads a payload onto a moving door but the door is never a *container* and
nothing gates when loading is possible.

## Solution outline (solve.py, teleport = transport only)

- **Phase 0 (reset):** settle 0.5 s, assert both drive buffers zero,
  `SIM_GEN_SCORE` ≈ 0.000.
- **Phase 1 (ALIGN, joint dynamics):** velocity-servoed torque on the axle via the
  scene's `drum_drive` buffer, |τ| ≤ 0.8 N·m, ω ≤ 0.45 rad/s; drive cut at 2° and
  viscous axle friction parks the drum on its 0° stop hands-off. `SIM_GEN_SCORE`
  0.100.
- **Phase 2 (LOAD, transport + contact push):** the ONLY teleport of the target —
  one pose write to a hover 3 mm above the sill shelf, outside the window, ~5 cm
  outside the drum footprint (free space; nothing bypassed). Gravity lands it; a
  velocity-servoed WORLD force via `box_drive` (≤ 2.5 N on the 100 g box, 0.12 m/s,
  with lateral centering) slides it through the window over the 3 mm rim gap; the
  force is CUT once the centre passes drum-local x = −0.050 and friction seats it.
  During the push a −0.15 N·m bias on `drum_drive` holds the drum against its
  one-sided load stop (disclosed below). `SIM_GEN_SCORE` 0.450.
- **Phase 3 (SEAL, joint dynamics):** same axle servo to the 92° stop with the box
  riding inside; drive cut at 89°, ~1–2° coast onto the stop. `SIM_GEN_SCORE` 1.000.
- **Phase 4 (persistence):** drive buffers asserted zero, 420 steps (3.5 s)
  hands-off; `success()` is live state. Only then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge with the final geometry: seed 0 (θ₀ = 58.8°, pudding at slot
A) and seed 1 (θ₀ = 72.0°, pudding at slot B — the slot swap is exercised) both
print the monotone sequence 0.000 → 0.100 → 0.450 → 1.000 → 1.000 and
`SIM_GEN_SOLVE: SUCCESS`.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `align_latch` (0.10): drum has ever been at the aligned stop (≤ 6°); also implied
  by a successful load (loading proves alignment happened).
- `load_latch` (0.35): the pudding has ever rested inside the bay (drum-frame
  containment band, honest by construction: a protruding box reads outside it).
- `seal_latch` (0.30): best sealing progress (θ/θ_closed) reached **while** the box
  was in the bay; forced to 1.0 when sealed with the box aboard. Cargo lost en
  route keeps only the credit earned up to the loss angle.
- current success (0.25): pudding in the bay AND drum within 5° of the 92° stop AND
  box + drum settled. Null policy ~0 (drum ≥ 48° misaligned, boxes on the floor).

## Embodiment argument (single Franka, parallel jaw, OSC)

Base at ≈ (−0.10, 0.0, 0), facing +x. Everything task-relevant lies at
x ∈ [0.15, 0.66], |y| ≤ 0.29, z ≤ 0.21 — inside a 0.855 m reach envelope with
generous margin, and the shroud is only 0.15 m tall so the wrist works above it
freely.

- **Drum rotation (align and seal):** a fingertip push on the RED lever bar, which
  sweeps ABOVE the shroud rim at z ≈ 0.19, radius 0.19–0.28 m from the axle —
  always exposed, never inside the shroud. The solve's 0.8 N·m torque cap is
  ≈ 2.9–4.2 N tangential at that arm — single-finger authority — and the
  0.45 rad/s rate cap is ≤ 0.13 m/s tip speed. The lever also *indicates* the bay
  direction (it points along the bay opening), so the two stops are visually
  distinguishable (at LOAD it points at the robot, at SEALED toward −y).
- **Box transport (the teleport's stand-in):** the 55 mm, 100 g cube sits on open
  floor at x ≈ 0.19 — a textbook top pinch for the 80 mm jaw; the place is a
  free-space carry to the sill shelf in front of the window (top of sill z = 0.040,
  drop from ~3 mm). No clutter within a jaw's width of either floor slot.
- **Window push:** the box then needs a ~15 cm horizontal slide at 0.12 m/s. The
  window aperture is 110 mm wide × 85 mm tall (z 0.040–0.125); pushing the box's
  rear face keeps the fingertip at the aperture mouth, never deeper than the wall
  plane, at z ≈ 0.05–0.09 — a straight-line fingertip slide, 2.5 N ≫ the ~0.5 N
  friction the 100 g box needs.
- **HOLD_TAU disclosure:** during the push, solve.py holds −0.15 N·m on the axle so
  wall friction from the entering box cannot walk the drum off its one-sided stop.
  The arm equivalent is bracing the lever with the second fingertip (or the side of
  the hand) while the first pushes the box — both contact points are simultaneously
  reachable (lever tip at ≈ (0.22–0.31, 0, 0.19), push point at ≈ (0.30, 0, 0.07),
  0.15 m apart) and 0.15 N·m is ≈ 0.6 N at the lever. A detent at the stop would
  make the bias unnecessary; it is kept as an honest, disclosed regulation torque.
- **Execution order:** align → load → seal, forced by geometry (rim/flank block the
  window, roof blocks the top), not by instruction sequencing.

## Checks (smoke.py — rejection battery, 12 checks, forge: `ALL PASS 12/12`)

1. Clean reset: finite state, drum at authored θ₀ ∈ [48°, 75°] (readback), both
   boxes at their floor slots, score < 0.05.
2. Randomization by readback: θ₀ / pudding-vs-decoy slot swap / box xy differ
   across 4 seeds (both slot assignments observed).
3. Null policy: 2 s of no action — score < 0.05, no success.
4. **Seed-strategy rejection:** the box dropped from above lands ON the drum roof
   (z readback ≈ 0.17), never enters; sealing afterwards yields no credit
   (score < 0.05). Top access is physically killed.
5. Order forcing A: with the drum misaligned, solve's own fingertip push cannot
   enter the bay and does not rotate the drum toward alignment (49.2° → 49.0°
   after 5 s); no load credit.
6. Order forcing B: sealed first, the flank wall fills the window — the pushed box
   stops at the window plane (x < 0.378), bay inaccessible, no credit.
7. Near miss (under-rotation): loaded, drum servo-LANDED at ≈ 79–80° with ~zero
   rate (parked, viscous-only friction) — short of the 87° sealed band: no
   success, partial credit 0.60 < score < 0.90.
8. Near miss (protrusion): push cut early, box settles with its centre past the
   disc rim (drum-local x ≈ −0.095 < −0.088) — `in_bay` False, no load credit,
   score ≈ 0.10 (align only).
9. Wrong object: the WHITE decoy sealed inside (drum fully at the sealed stop) —
   rejected, score ≤ 0.12.
10. Exactness: full gentle construction (solve's own phases) → success and
    |score − 1.0| < 1e−3.
11. Latch semantics: rotating the drum back open revokes success; score falls to
    the latched 0.75, not 1.0 (the box still rides in the bay — only the live
    sealed term is lost).
12. Camera: ≥ 20 rgb frames across all checks → `frames.npz` (280 frames saved).
