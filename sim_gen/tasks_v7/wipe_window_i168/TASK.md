# frost_scrape (wipe_window_i168)

Knock every frost chip off the tilted glass pane so the trough at its base catches it.

## Seed provenance

- Seed: `pick_place/wipe_window`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/wipe_window.py`)
- The seed: grasp a light wiper bar, then TRACE a fixed Z-pattern of six floating
  waypoints across a vertical window pane. The rubric is trajectory tracking
  (`tracking_approach` / `tracking_progress` against the marker chain); the window is
  scenery — no object's final state matters, and the episode's success is a motion
  pattern, not a world outcome.

## What changed, and why it is strategically different

- **Different goal type.** The seed judges a MOTION (did the tool visit the waypoints
  in order); here nothing about the motion is judged — only a physical OUTCOME: every
  frost chip must end up settled INSIDE the collection trough. There are no waypoints,
  no fixed pattern, and a solver that reproduces the seed's plan (rub the glass in a
  Z-pattern, pressing INTO the pane) achieves exactly nothing (smoke check 4
  constructs it: the chips stay racked, score ~0).
- **Different plan.** The solver must count the sampled chips (2-4, random slots,
  random row height), and per chip: push it SIDEWAYS along the glass until its weight
  tips it off a narrow ledge, then rely on gravity down the 75-deg face and on the
  trough's capture. The plan is per-object dislodge-and-capture with a funnel
  mechanism, not path following; the code structure is containment predicates and
  per-chip latches, not waypoint distance.
- **Different failure modes.** Overshooting a push can throw a chip past the trough
  onto the table — a real, permanent-ish mistake the rubric rejects (partial credit
  for the dislodge, nothing for containment). The seed has no equivalent state.
- Compared to the other corpus task read while building this one
  (`press_switch_i161`, dial setpoint regulation): no dials, no setpoint stopping
  problem, no joints at all — this is dislodge-and-capture of free bodies.

## Scene

- Kinematic: table; glass pane (0.68 x 0.55 m, leaning 15 deg away from the robot);
  green collection trough at the pane's base (front wall 10 cm high + end caps; the
  pane itself is the back wall); one narrow gray ledge per chip (32 mm wide under a
  70 mm chip).
- Dynamic: 2-4 white frost chips (7 x 6 x 1.2 cm, 30 g), each racked leaning against
  the glass with its bottom edge on its own ledge.
- Randomized per episode (readback-verified): chip COUNT, WHICH of the 4 slots are
  occupied, the row height on the pane (0.22-0.44 m along the face), per-slot lateral
  jitter. Absent chips + ledges park in an off-scene depot.

## Teleport-solution outline (solve.py)

No teleports at all — reset() racks the chips, and everything after is contact
dynamics driven through the scene's `drive_f` buffer (fingertip-scale, 1 N cap):

1. Reset, settle, read back the sampled instance. `SIM_GEN_SCORE` ~0.
2. Per chip (leftmost first — order is a solver choice, not a task rule): velocity-
   cascade lateral push (KV*dt/m ~ 0.7 < 1) until readback shows the chip tipping
   (centre 3 cm below its rack) or fully clear of the ledge; release; the chip
   tumbles down the glass; wait for the trough to catch it and for the slow-gate
   containment streak to mature. `SIM_GEN_SCORE` after each chip (monotone).
3. All chips in: success() holds; 3.5 s hands-off persistence; `SIM_GEN_SOLVE:
   SUCCESS`.

## Rubric

- Per present chip: 0.25 latched dislodge (centre dropped > 8 cm below its rack) +
  0.55 latched containment (inside the trough interior, slower than 0.25 m/s, for 12
  consecutive substeps — fast transits latch nothing). Mean over present chips.
- +0.20 success(): every present chip inside the trough interior NOW (between wall
  and pane, between caps, centre below rim - 2 cm) and all settled. score == 1.0 iff
  success; null policy ~0; latches never evaporate.

## Embodiment argument (single Franka, OSC)

- Plausible base pose: on the table at (0.0, 0.0, 0.40), facing +x. Chips sit at
  x ~ 0.58-0.63, world z 0.62-0.86 — comfortable reach; the trough (x ~ 0.42) is
  below and in front and never needs to be touched.
- **Frost chip (the only object the robot moves):** contact strategy is a fingertip
  PUSH on the chip's side edge — closed jaw, fingertip against the 6 cm tall side
  face, push 2-4 cm sideways along the glass. No grasp is needed anywhere. Chip
  protrudes 18 mm off the glass, so a fingertip (~15 mm) engages it without touching
  the pane; the pane leans AWAY 15 deg, giving the knuckles clearance; adjacent chips
  are >= 5 cm apart. Required precision ~1 cm on a 2-4 cm stroke (the racked-stable
  window is CoM within +-16 mm of ledge centre; anything past that tips off — wide
  margins on both sides, and an under-push is simply retried). The fall and capture
  are passive — the arm never operates near the ground or through an aperture.
- Execution order: NOT required (any chip order, either push direction).

## Checks (smoke.py, 11)

1. clean reset: finite, chips racked at sampled slots (readback), score ~0.
2. randomization real across 8 seeds (count / row height / jitter; readback).
3. null policy ~0.
4. seed strategy (wiping press INTO the glass) does nothing.
5. near miss: 8 mm push (inside the ledge's stable window) leaves the chip racked, ~0.
6. dislodged but missed (settled on the table in front of the wall): partial credit
   only, no success.
7. wrong place (beside the trough, past the end cap): rejected.
8. wrong side (behind the pane): rejected.
9. exactness: full correct strategy -> success, score == 1.0, stable.
10. plucking a captured chip back out revokes success; latched credit remains.
11. video frames recorded (frames.npz).
