# glaze_window (wipe_window_i301)

Uncap the empty window frame, seat the glass pane in its sill groove, put the cap back.

## Seed provenance

- Seed: `pick_place/wipe_window`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/wipe_window.py`)
- The seed: grasp a light wiper bar, then TRACE a fixed Z-pattern of six floating
  waypoints across an already-installed vertical window pane. The rubric is trajectory
  tracking (`tracking_approach` / `tracking_progress` against the marker chain); the
  window is scenery — no object's final state matters, and the episode's success is a
  motion pattern, not a world outcome.

## What changed, and why it is strategically different

- **Different goal type.** The seed judges a MOTION over an installed window; here the
  window starts UNINSTALLED and nothing about the motion is judged — only the final
  CONFIGURATION: the pane seated in the sill groove and the cap lid back in its tray,
  simultaneously, at rest. There are no waypoints and no pattern; a solver that
  reproduces the seed's plan (press the glass flat-on and rub a pattern over it)
  achieves exactly nothing (smoke check 4 constructs it: no latch, score ~0).
- **Different plan.** The plan is an ORDERED three-step assembly with a real insertion:
  (1) remove the cap lid that roofs the frame's only insertion corridor, (2) thread a
  12 mm plate down between the posts into a 22 mm groove under contact, (3) replace the
  lid. The order is enforced by GEOMETRY, not by fiat: with the cap on, the top is
  closed, and the two fixed mullion bars at mid-height close the front/back
  "letterbox" — threading from the side needs a lean steep enough to duck the cap
  plane (> ~53 deg) yet shallow enough to clear a mullion (< ~10 deg), a
  contradiction. Smoke check 5 runs the solve's own insertion servo with the cap on
  and shows the stall. The rubric is order-aware in the other direction too: the
  capped latch requires the pane seated SIMULTANEOUSLY, so recapping an empty frame
  latches nothing (smoke check 7).
- **Different failure modes.** Stalling on a rail top, missing the groove, leaving the
  frame unroofed, recapping in the wrong order or 90 deg misoriented — all
  configuration errors with no seed equivalent (the seed cannot "fail" the world, only
  the trace).
- Compared to the sibling task built from the same seed (`wipe_window_i168`,
  frost_scrape: dislodge free chips into a catch trough): no free-body herding and no
  passive capture — this is a guided INSERTION plus an ordered lid interlock, with an
  explicit two-way order gate. Compared to the packing exemplar (`pen_holder`): a
  single plate into a snug groove with an uncap/recap protocol, not many bodies into
  one open cup.

## Scene

- Kinematic: table; window FRAME re-posed per episode — thick sill carrying a 22 mm
  groove between two rails, two posts (inner faces +-95 mm), two mullion bars closing
  the side letterbox at mid-height, and a nub tray (4 x-stops + 2 y-stops) on the post
  tops; separately, a slotted pane STAND.
- Dynamic (the only two): the light-blue glass PANE (17 x 20 cm, 12 mm, 150 g)
  standing upright in the stand slot, and the dark-brown CAP lid (16 x 30 cm plate,
  120 g) resting in the tray, roofing the frame.
- Randomized per episode (readback-verified): frame table position AND yaw, stand
  table position AND yaw — every transport target and the insertion axis move.

## Teleport-solution outline (solve.py)

Teleports for TRANSPORT ONLY (airborne body across free space); every load-bearing
interaction is contact dynamics through the scene's `drive_f` buffer (CoM forces,
4 N cap, gravity-feedforward velocity cascades with K*dt/m ~ 0.11):

1. Reset, settle, `SIM_GEN_SCORE` ~0.
2. UNCAP — force-lift the cap out of its nub tray, teleport-hover over an empty patch
   of table, free drop, settle. cap_off latch: score 0.20.
3. GLAZE — force-lift the pane out of the stand slot, teleport-hover above the frame's
   open top, then a contact DESCENT (rate servo down + 0.3 N frame-local centering on
   the pane's bottom point) threads it past the mullions into the groove; 0.6 N seat
   press; release; slow-gate streak matures. Stall on a rail -> lift out and retry.
   Score 0.60.
4. RECAP — force-lift the parked cap, teleport-hover centered above the tray, guided
   descent between the nubs, release, settle. Order-aware capped latch + success:
   score 1.0. Then 3.5 s hands-off persistence; `SIM_GEN_SOLVE: SUCCESS`.

## Rubric

- 0.20 cap_off latch (cap carried clear of the frame top; granted retroactively once
  the pane is seated — a seated pane physically implies the corridor was opened);
- 0.40 seated latch: pane bottom in-groove (frame-local |x| < 8 mm, |y| < 30 mm,
  floor height +-8 mm, near-upright) slower than 0.10 m/s for 12 consecutive
  substeps;
- 0.25 capped latch: cap level in its tray WHILE the pane is seated (same streak
  gate) — recapping an empty frame earns nothing;
- 0.15 success() live: pane seated AND cap seated AND both settled. score == 1.0 iff
  success; null policy ~0; latched credit never evaporates.

## Embodiment argument (single Franka, OSC)

- Plausible base pose: on the table at (-0.15, 0.0, 0.40), facing +x. Frame at
  x 0.42-0.52 (groove floor at world z 0.45, tray at z 0.68-0.70), stand at
  x 0.14-0.22 — all inside a comfortable reach envelope, all interactions from above
  or the front, never near the ground.
- **Cap (twice):** pinch-grasp the 15 mm-thick plate by its front edge — the cap
  overhangs its 4 corner nubs by construction (the nubs sit at the cap's corners;
  the mid-edge is free), so a fingertip pair closes on bare plate. Lift 6 cm
  (clearing the 20 mm nubs), carry, set down. Precision required ~5 mm against the
  5-8 mm tray slack on the way back in — the nubs themselves funnel the final seat.
- **Pane:** pinch-grasp the top edge of the 12 mm plate — 17 cm of pane stands proud
  of the 30 mm stand rails, free on both faces. Lift 10 cm out of the slot, carry
  upright, lower through the top opening (posts admit +-10 mm around the pane; the
  fingers stay ABOVE the frame the whole way — the pane hangs below the grip, so the
  insertion needs no in-frame clearance for the hand). Required precision ~5 mm
  lateral on the 22 mm groove (5 mm slack each side of the 12 mm plate) at a
  vertical-only final move — the rails themselves guide the last 40 mm.
- Forces are fingertip-scale throughout (<= 4 N on 120-150 g bodies).
- Execution order: REQUIRED (uncap -> glaze -> recap), and it is enforced by the
  geometry + the order-aware latch, not by the solver's choice.

## Checks (smoke.py, 12)

1. clean reset: finite, pane racked in the stand + cap seated in the tray (readback
   vs cfg formulas), no latch pre-fired, score ~0.
2. randomization real across 8 seeds (frame x/y/yaw, stand x/y/yaw; spawn readback).
3. null policy ~0 — and the cap resting in its tray latches nothing (order gate).
4. seed strategy (flat-on wiping press + rub against the pane) seats nothing.
5. corridor closed: with the cap ON, solve.py's own descent servo stalls on the lid
   far above the groove; cap undisturbed, no credit.
6. near miss: pane standing on the table 10 cm in front of the frame — rejected.
7. wrong order: honest uncap + geometrically perfect recap of the EMPTY frame —
   capped latch stays cold, only cap_off (0.20) earned.
8. partial: honest uncap + honest seating, cap left parked — exactly 0.60, no
   success.
9. exactness: full ordered strategy -> success, score == 1.0, stable hands-off.
10. plucking the cap back off revokes success; latched 0.85 remains.
11. cap dropped back 90 deg rotated never finds the tray (its narrow axis slips
    between the posts and it jams tilted ~30 mm off-height) — never seats.
12. video frames recorded (frames.npz).
