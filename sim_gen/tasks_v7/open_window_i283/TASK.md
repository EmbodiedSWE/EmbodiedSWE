# sash_vent — prop the dead-man sash on its one-way pawl, then deliver through the gap

Package: `sim_gen/tasks_v7/open_window_i283`
Env: `simgen.sash_vent` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/open_window`. In the seed, the robot rotates a handle and swings
a hinged casement panel open, driven by a recorded waypoint trajectory (the checker
is a TODO). The articulation act IS the goal: the panel stays wherever it is swung,
and nothing happens after it.

## What changed, and why it is strategically different

1. **The window cannot simply "be opened."** The casement is now a DOUBLE-HUNG
   VERTICAL SASH that is gravity-loaded shut — a dead-man mechanism: released
   un-propped at ANY height it slides fully closed on its own (prismatic track,
   gravity is the return spring; verified physically in smoke check 6). The seed's
   plan — actuate the panel, walk away — leaves nothing behind.

2. **Openness must be MANUFACTURED via a passive one-way ratchet.** The frame
   carries an orange gravity PAWL (revolute tab, hard stop at horizontal, free to
   swing up). The only way to a persistent aperture: lift the sash PAST the pawl
   (the rising yellow lift rail cams the tab aside — no second hand needed), wait
   for the tab to gravity-return underneath, then SET THE SASH DOWN ON the tab.
   Sweep geometry proved in `__post_init__`: the tab's return sweep apex (0.309)
   clears the held lip bottom (0.331), and the pawl rest gives q_prop = 0.1265
   while the tallest wedgeable substitute prop (a 6 cm cube under the sash) yields
   only q = 0.0585 ≪ q_open_min = 0.11 — improvised props are rejected by geometry.

3. **Opening is only a MEANS.** The goal is a through-the-wall delivery: the RED
   parcel must end at rest on the green fenced tray on the FAR side of the wall,
   reachable only through the propped gap (the fixed upper pane and the wall seal
   everything else). A BLUE distractor of identical size must stay behind — color
   is identity. The seed has no post-articulation payload at all.

4. **Physics is the judge**, not proximity-to-recorded-waypoints: a transit-slab
   latch (the parcel's center must cross the wall slab BELOW the fixed pane while
   the sash is open — an over-the-wall lob crosses the slab only above the pane
   band, an around-the-wall carry only outside the jambs; neither fires), a
   tray-membership window whose z band rejects stacked / fence / ground rests, an
   open-streak latch (a teleported-open sash free-falls and breaks the stillness
   streak within ~2 steps), and a stillness counter-latch on the judged end state.

## Teleport-solution outline (transport only — verified on forge)

Teleportation is used ONLY to stage the parcel on the INSIDE half of the sill
(x = +0.09, the solver's own side of the wall — outside every rubric window; the
tray lies beyond the transit slab, so no teleport can shortcut the delivery). The
SASH is driven by a gravity-feedforward velocity-regulated vertical force along its
real track (KV·dt/m ≈ 0.17), the PARCEL by a horizontal push servo whose
world-frame force is rotated into the body frame every step (the cube can tip while
crossing), and the PAWL is never touched — the mechanism cams and returns itself.

- **P0** settle + readbacks (sash closed, tab horizontal, cube slots); score ~0.
- **P1** lift the sash to q = 0.21 — the lift rail cams the pawl aside en route.
- **P2** hold until the tab gravity-returns under the lip (angle readback streak).
- **P3** lower to q_prop + 12 mm, cut the wrench — the sash free-falls onto the tab
  and rests propped (q = 0.1244 ≈ q_prop; the dead-man moment: wrench OFF, stays
  open). Open latch fires → SIM_GEN_SCORE 0.30.
- **P4** teleport the parcel to the inside sill staging spot; assert NOT through.
- **P5** push the parcel −x through the propped gap (breakaway feedforward beats
  static friction; stall point aimed past the break threshold): it crosses the
  transit slab and settles mid-tray (x = −0.096); ring down until `success()`
  holds 120 consecutive steps → SIM_GEN_SCORE 1.00.
- **P6** hands-off ≥ 3.3 simulated seconds; success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, scores
monotone 0.00 → 0.30 → 1.00, zero persistence flickers.**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(0.55, 0, 0.0), facing the wall (wall inside face at x = 0). Reachability:
the cube slots lie 0.42–0.48 m from the base at grasp height 0.03 m; the lift rail
spans z 0.12–0.36 m at x ≈ 0.05–0.08, and the sill staging spot is 0.46 m out —
all inside Franka's ~0.85 m envelope. Graspability: the YELLOW lift rail is a
30 mm-thick bar (< 80 mm jaw) grasped from the front anywhere along its 30 cm
span at y < 0.07 (clear of the pawl's y range); the 60 mm cubes take a top or side
pinch. The lift is a single 21 cm vertical stroke against 11.8 N of sash weight —
trivial for the arm — and CRUCIALLY one arm suffices: while the hand holds the
sash up, the pawl resets ITSELF under the lip (passive gravity return, ~0.3 s), so
the hand simply lowers the sash 7 cm and lets go. The delivery is a horizontal
push of the staged cube along the sill: ~0.7 N against friction, 19 cm of travel,
with the 12.8 cm × 42 cm propped aperture passing the forearm comfortably; a
release just past the wall plane also works — the fenced tray catches the slide.

## Execution order

Only the mechanically-forced order exists, and `describe()` states it: prop first,
deliver second (the closed sash seals the aperture — smoke check 9 shows the
solve's own push servo jamming against it). Within that, everything is free: the
sash may be lifted/lowered any number of times, the parcel may be staged anywhere
inside, and the distractor may be moved or ignored.

## Rubric

- +0.30 once the sash has EVER been open (q ≥ 0.11) AND still for a 20-step
  streak (latched) — only the pawl prop reaches this band.
- +0.25 once the parcel has EVER crossed the transit slab (x ∈ (−0.045, 0.015),
  |y| < 0.21, 0.05 < z < 0.35) while the sash was open, latched.
- +0.30 once the parcel has EVER been at rest in the tray window (x ∈
  [−0.155, −0.055], |y| ≤ 0.19, |z − 0.1505| ≤ 0.020) while open — GATED on the
  through-latch (a parcel that never transited earns no tray credit), latched.
- Partial credit capped at 0.85; `score = 1.0` iff `success()` = in-tray ∧
  through-latch ∧ sash open ∧ settled (stillness counter-latch, 30 steps).
- Null policy scores ~0 (the sash starts closed).

`__post_init__` honesty asserts: the open band is only reachable propped (pawl
rest vs wedged-cube ceiling); the pawl's return sweep clears the held lip
(single-arm feasibility); the propped gap passes the parcel with ≥ 45 mm margin
while the closed sash seals it; the transit slab brackets the wall and the tray
band lies beyond it; the tray z band rejects stacked/fence/ground rests; the
raised sash stays inside the opening; jaw fits for rail and cube.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 17/17`)

1. Reset settles: states finite, sash on its closed stop, tab horizontal, cubes
   upright on the floor.
2. Fresh reset: score ~0, no success.
3. Randomization: cube slot assignment varies (readback 4 distinct / 6 seeds) +
   xy jitter (25 mm spread).
4. Randomization: free yaw varies (readback spread 2.88 rad).
5. Null policy 240 steps: score ~0.
6. Dead-man: the sash teleported open to q = 0.070 falls closed on its own
   (readback → q ≈ 0); the open latch never fires un-propped.
7. One-way pawl: a hinge-axis torque swings the tab to the −85° hard stop
   (clamped at −1.484 rad); torque off, gravity returns it to horizontal.
8. Seed-analog ("the window is open" — the seed's whole goal): sash released just
   above the pawl LANDS on it and holds hands-off 240 steps at q ≈ q_prop —
   genuine 0.30, NOT success, ≤ 0.35.
9. Closed-window denial: the solve's own push servo with the window closed — the
   parcel moves then jams at the sash face (x = +0.108), never transits, score ~0.
10. Over-the-wall cheat: window propped, parcel teleported DIRECTLY onto the tray
    — in_tray True but the through-latch is 0 → tray credit denied, not success.
11. Near-miss: parcel settled 7 mm short of the tray x band → not in_tray.
12. Wrong object: the BLUE distractor on the tray earns nothing (score stays 0.30).
13. z band: parcel STACKED on the distractor in the tray (z +0.2105 vs rest
    +0.1505) rejected.
14. z band: parcel on the GROUND beyond the tray (z +0.030) rejected.
15. Settle gate: parcel genuinely pushed into the tray but judged mid-slide —
    in_tray ∧ through ∧ open all True yet NOT success; removed before ring-down
    (the battery never succeeds).
16. Rejection audit: `success()` never fired at any judged point.
17. Final state no-NaN.

Smoke also records `frames.npz` (323 × 600 × 960 × 3) from a perspective camera.
