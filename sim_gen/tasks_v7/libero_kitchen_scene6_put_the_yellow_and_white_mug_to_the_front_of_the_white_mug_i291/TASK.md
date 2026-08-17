# Task: mug_alcove — push the yellow-and-white mug through the alcove doorway to the front of the white mug

## Provenance

Seed task: `roboverse_pack/tasks/libero_90/libero_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.py`
(LIBERO-90, kitchen scene 6). The seed is a tabletop Franka pick-and-place: grasp the
yellow-and-white mug, lift it, and set it down so that it sits in a relative-position box in
front of the porcelain (white) mug — `x_diff ∈ (0, 0.2)`, `|y_diff| < 0.05` in the white mug's
neighborhood. The judge is purely a relative-pose box between two free mugs on an open table.

## What changed, and why it is strategically different

The nominal goal keeps the seed's words — *put the yellow-and-white mug to the front of the
white mug* — but the goal zone is now inside a low **roofed alcove** (a three-walled bay with a
115 mm awning over the doorway strip). The white reference mug lives at the back of the alcove;
"in front of" means the floor strip between the doorway sill and the white mug, under the roof.

- **Vs the seed (pick & place):** top-down placement is geometrically impossible. The roof
  underside is 115 mm above the floor and the mug is 90 mm tall — 25 mm of headroom, far less
  than any lower-then-release or drop trajectory needs, and the doorway lintel blocks a carried
  mug from entering at any height above ~25 mm. The only way to deliver the mug is
  **nonprehensile**: slide it across the floor, through the 240 mm doorway, and stop it inside a
  33–48 mm deep target window without ramming the dynamic white mug (a ≤3 cm anchor-disturbance
  gate protects it). The manipulation problem changes from grasp/transport/release to a
  precision floor push with a stopping-distance constraint.
- **Vs `put_mug_i173` (hang the mug on a wall peg):** no suspension, no handle-threading; the
  mug never leaves the floor.
- **Vs `put_mug_i228` (pull a silo gate to dispense balls into the mug):** no articulated
  mechanism, no dispensing; the mug itself is the manipulated payload.
- **Vs `libero_kitchen_scene6_close_the_microwave_i114` (counterweight-ballast door):** no door,
  no ballast; static geometry plus two free rigid mugs only.
- **Vs `pen_holder` (drop pens into a holder):** no insertion from above; the roof forbids
  exactly that family of solutions.

The white mug is a *dynamic* reference: shoving it (or ramming it with the yellow mug) moves the
anchor and fails the `white_intact` gate, which also kills the "move the white mug to the yellow
mug" cheat family that the seed's relative-position judge admits.

## Scene (all procedural, self-contained)

- **Alcove** (kinematic compound): floor slab flush with ground, two side walls 160 mm tall,
  a back wall, and a roof plate spanning the first 150 mm past the doorway at z = 115 mm.
  Interior half-width 120 mm (240 mm doorway), interior depth 300 mm. Pose randomized per env:
  xy jitter ±4 cm around (0.45, 0), yaw ±12°.
- **White mug** (dynamic, all-white): spawned at the back of the alcove, depth 180–210 mm past
  the sill, lateral jitter ±3 cm. Its world xy at reset is recorded as the anchor.
- **Yellow-and-white mug** (dynamic payload, yellow lower band / white upper band, bar handle):
  spawned on the open floor 280–450 mm *outside* the doorway, lateral band ±12 cm, yaw ±180°.
  Body Ø80 × 90 mm, mass 250 g, authored bottom-heavy CoM (z −25 mm) so a CoM-level push
  slides rather than tips.

## Success (judged live on settled physical state)

All of, simultaneously:
1. Yellow mug alcove-local `x ∈ [62 mm, min(roof_len − r, white_x − 85 mm)]` (inside past the
   sill, under the roof, ≥85 mm gap to the white mug), `|y − white_y| ≤ 50 mm`;
2. upright (tilt ≤ 15°) and resting on the alcove floor (|z − h/2| ≤ 15 mm — rejects roof
   parking);
3. white mug intact: within 3 cm of its recorded anchor, upright;
4. settled: 45-step stillness streak on BOTH mugs plus a pose-jump guard (2 cm/step) — a
   teleported-in end state must survive ≥45 real contact steps before it can count.

Score: latched partials 0.12 (approached the doorway) + 0.25 (entered past the sill upright on
the floor) + 0.30 (inside the target window), capped at 0.67 without success; 1.0 iff success.

## Demonstrated solution (solve.py) — honest teleport convention

- **P0** reset, settle 60 steps, mass/layout readback asserts (white in band, yellow outside,
  not success).
- **P1** single **transport-only teleport** of the yellow mug to a free-space staging pose on
  the open floor 150 mm *outside* the sill, aligned with the white mug's lateral position,
  handle trailing. Assert: still not success; the approach latch fires, the entered latch does
  not.
- **P2** the load-bearing interaction, entirely contact dynamics: a velocity-regulated bang-bang
  horizontal force (1.5 N at CoM when v_along < 0.07 m/s, small lateral correction) slides the
  mug across the sill, through the doorway, to the midpoint of the target window
  (x_target ≈ 96–116 mm past the sill). Runtime frame-encode probe after 240 no-progress steps;
  stall escalation +0.75 N to a 4 N cap. Force cleared at the stop point.
- **P3** settle until success() reads true (streak completes under gravity/contact only).
- **P4** hands-off persistence: 10 × 42 steps (>3.5 sim-seconds), success asserted every probe,
  then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing.

## Embodiment argument (single Franka, parallel-jaw gripper, OSC)

Base pose ≈ (0.05, 0.40, 0), facing the alcove. Per-object contact strategy:

- **Yellow mug:** closed-gripper nonprehensile push at the mug's waist. The staging pose and the
  full spawn band lie on open floor within a 0.28–0.75 m reach annulus of that base. The push
  into the alcove needs the fingertip to follow the mug at floor level to at most ~60 mm past
  the sill (the mug then coasts ≤ 40 mm into the window); the wrist stays outside the doorway
  and below the 115 mm lintel only over the last ~6 cm, which a horizontal, low-wrist OSC posture
  reaches. Alternative for the pre-staging transport: a low sideways handle-hook drag along the
  floor (the 26 mm handle bar admits two 10 mm fingers), never lifting above 25 mm.
- **White mug:** never touched; every strategy keeps the end-effector out of the back half of
  the alcove (the target window ends ≥ 85 mm short of it).
- No required execution order between sub-goals: there is a single payload; approach/enter/inside
  latches are strictly nested along the same push.

Randomization (alcove pose/yaw, both mug positions, payload yaw) is proven by readback in smoke
check 3–4 and only shifts the push line and stop distance; it never changes the strategy class.

## Execution-order declaration

Single-object goal; no order constraints. The latched partials (approach → entered → inside) are
milestones of one push and are naturally monotone; no alternative order exists.

## Smoke rubric battery (16 checks, rejection-only, written after the solve passed)

1. Reset sanity: finite state, layout readback in bands, both mugs at rest.
2. Fresh reset judges: score ≤ 0.02, no success.
3. Alcove randomization spread over seeds (xy and yaw) by readback.
4. Mug randomization spread; white always in its band, yellow always outside the alcove.
5. Null policy 240 steps: score ≤ 0.02, never success.
6. **Seed-strategy rejection:** yellow mug placed just *outside* the sill, exactly "in front of
   the white mug" by the seed's relative-position box — no success, score ≤ 0.13.
7. Short near-miss: 17 mm short of the sill-side window edge — no success, ≤ 0.40.
8. Crowding: settled 82.5 mm from the white mug (gap readback < 85 mm) — no success.
9. Off-axis: correct depth, Δy = 70 mm — no success.
10. Toppled at goal depth (on its side) — upright gate rejects, ≤ 0.13.
11. Anchor gate: white mug displaced 6 cm, yellow placed perfectly in front of its *new*
    position — placement predicate true, `white_intact` false, no success.
12. Roof parking: mug on top of the roof plate over the window — no success, ≤ 0.13.
13. Fly-through: exact goal pose judged cold on arrival — placement true, success false
    (streak not yet earned).
14. Yank-out: 12 real frames at goal (below the 45-step streak, so success never fires), then
    teleported back outside — success stays false, latched score stays in [0.66, 0.68]
    (partials never evaporate, cap holds).
15. Rejection audit: success never fired across all probes.
16. Final state no-NaN.

Renders 16+ frames to `frames.npz` (CWD), prints named PASS lines and
`SIM_GEN_SMOKE: ALL PASS 16/16`.
