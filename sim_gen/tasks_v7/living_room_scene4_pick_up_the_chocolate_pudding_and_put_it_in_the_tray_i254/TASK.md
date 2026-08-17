# pudding_dock_dispense — dock the tray, then dispense the pudding into it

**Env**: `simgen.pudding_dock_dispense` · **Files**: `scene.py` (scene + rubric),
`solve.py` (teleport-solution certificate), `smoke.py` (rejection battery).

## Seed provenance

Seed task: `libero_90/living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray`
("pick up the chocolate pudding and put it in the tray"): grasp one free box among
distractors (two bowls, a salad-dressing bottle), carry it through free air, release
it over a passive open tray; judged by a single containment bbox. One unordered
pick-and-place.

## What changed, and why it is strategically different

The delivery problem is inverted into an **ordered logistics protocol with an
irreversible hand-off**:

1. **The payload is ungraspable.** The pudding box waits inside a roofed chute
   (60×60 mm channel, entry sill behind it) on a dispenser station's raised deck.
   No grasp is possible under the roof — the only affordance is a fingertip **push
   deeper into the chute**, which ends over a **drop hole** through the deck.
2. **The receiver is mobile and must be docked first.** The deck is the roof of a
   ground-level covered **dock bay** (a garage, open only through its front mouth).
   The tray must be slid through the mouth, handle trailing, until it seats against
   the back-wall stop — only then does the drop hole sit over the tray interior.
3. **The ordering is enforced by physics, not rules.** A box dispensed before the
   tray is docked lands in the bare covered bay where nothing can retrieve it (the
   roof blocks any reach-in, and a tray pushed in afterwards just perches on the
   lost box). Dock-then-dispense is the only order that can ever succeed.
4. **Object identity is load-bearing.** An identically-sized bright-red gelatin box
   waits in the mirror chute; which chute holds the brown pudding is sampled per
   episode. Dispensing the red box into the tray is a violation
   (`~decoy_in_tray` is a success clause).

Versus the seed: no grasp, no carry, no free-air release; the tray is an actor, not
a passive target; success is *pudding in the tray **while docked***, a state
physically unreachable by direct placement (the bay is covered). Versus the corpus
tasks read for differentiation: i18 `wedge_hopper` and i236 `airlock_transfer` have
**fixed** receivers and gate/wedge mechanisms — here the *receiver itself* is the
manipulated vehicle and must be pose-accurately parked; i73 `roller_freight` is
about force budgets under a slab; i139 `beam_balance_tray` is a measurement task;
i194 `pagoda_relay` orders stacking moves — here the ordering is a single
irreversible hand-off through a covered volume, and the discriminator (lost box in
a bare bay) is spatial, not sequential-stack.

## Scene summary

Procedural compounds only. Station: 60 kg dynamic compound (garage walls + back
wall, deck slab with two drop holes, two roofed chutes with sills, center divider,
lane stop). Tray: 0.5 kg compound (floor, four 45 mm walls, tall yellow handle tab
on one short side — the handle never passes under the deck, so the arm can drive
the tray to full depth). Boxes: brown pudding + red decoy, 60×45×34 mm, 80 g.
Randomization (verified by readback in smoke): station yaw ±20° + xy ±40 mm, tray
scattered with free yaw in a zone in front of the mouth, pudding lane ∈ {+y, −y},
box jitter ±3 mm.

Success (all live physical readouts): pudding inside the tray interior AND tray
inside the docked window (|dx| ≤ 30 mm, |dy| ≤ 18 mm, |dyaw| ≤ 6°, z on the
ground) AND red decoy not in the tray AND 0.5 s sustained stillness AND finite.
Score: 0.30·docked + 0.25·chute-progress + 0.25·landed (latched, cap 0.80);
1.0 iff success. Null policy ≈ 0.

## Teleport-solution phases (solve.py, passes seeds 0 and 1 on the forge)

- **P0** settle; assert baseline ≈ 0, boxes seated per `pud_lane` readback,
  authored masses read back (custom-spawner honesty).
- **P1** TRANSPORT teleport: tray to the open floor in front of the mouth (23 mm
  clear of the mouth plane — 0.37 m from the docked window). Then a
  velocity-capped 6 N horizontal push (hand-on-handle stand-in) with small
  lateral/yaw PD slides it through the mouth to the back-wall stop. Docked credit
  earned purely by sliding contact. Assert: docked-but-empty is NOT success (and
  the seed's own plan — pudding in an undocked tray — never can be).
- **P2** lane readback (color-cue stand-in, asserted against box positions), then
  a velocity-capped 0.6 N fingertip push along the chute; the force is cut the
  instant the box tips into the hole; gravity and contact deliver it into the
  docked tray. Success turns True by settling.
- **P3** ≥ 3.3 s hands-off persistence; `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport-only, always into free space outside every scoring volume;
no force ever exceeds ~6 N (station ground friction is ~290 N); every push is
velocity-capped so nothing scores by being slammed.

## Embodiment argument (Franka, one base pose)

Base at station-local ≈ (−0.50, −0.30), facing the mouth: from there the approach
floor, the chute mouths (deck top 0.15 m, 0.115 m from the mouth plane) and the
tray scatter zone are all inside a 0.85 m reach envelope at convenient heights.

- **Tray**: the 12 mm yellow handle tab is a standard parallel-jaw grasp (or a
  two-finger hook-pull / palm-push). The handle stays outside the covered zone for
  the entire insertion (24 mm clearance from the deck edge at full depth), so the
  gripper drives the tray to the stop without ever reaching under the deck. The
  15 mm-per-side bay clearance funnels the last alignment; the stop sets dock
  depth. Reversed insertion is impossible, not just wrong: the 150 mm-tall handle
  jams on the 138 mm deck edge (smoke check 8).
- **Pudding/decoy**: a closed-gripper fingertip enters the 60×60 mm chute mouth
  and pushes the box ≤ 43 mm — less than a Franka finger length — until it tips
  through the hole. No grasp is needed at any point after that: gravity delivers.
- No interaction requires reaching into the covered bay, lifting the station, or
  exceeding gentle push forces (≥ 2.6× static friction margin on both pushes).

## Execution order (declared)

**Dock the tray first, then dispense the pudding.** The reverse order is
unrecoverable by construction (covered bay), which smoke verifies as physics.

## Checks (smoke.py — rejection battery, 13 checks)

1. settle/no-NaN + seating readback + authored masses + baseline ≈ 0
2. randomization is real (station yaw/xy, tray xy/yaw readback across seeds)
3. lane assignment takes both values over 12 resets, readback-consistent
4. null policy: 240 idle steps → score ≈ 0, no success
5. SEED strategy: pudding dropped into the scattered undocked tray → score ≈ 0
6. out-of-order (a): dispense with no tray → box lost on the bay floor, no success
7. out-of-order (b): tray lowered onto the dock over the lost box → perches,
   unrecoverable, no success
8. reversed (handle-first) insertion jams on the deck edge → not docked
9. near-miss: tray 60 mm short of the window + pudding delivered into it →
   contained but undocked, no landed credit, no success
10. deck perch: pudding on the roof directly above the docked tray's interior
    footprint → z-gate refuses containment
11. wrong object: red decoy dispensed into the docked tray → violation, no success
12. both boxes in the docked tray → score capped at 0.80, success still refused
13. frames.npz video captured

`solve.py`: forge SUCCESS on seed 0 (lane −y, yaw −4.0°) and seed 1 (lane +y,
yaw +15.5°), score trace 0.00 → 0.30 → 1.00 → 1.00, non-decreasing.
