# open_washing_machine_i393 — Coin-op washer: pay with the brass token to unlock the door, then get the laundry ball out and into the basket

Env: `simgen.coinop_washer` (scene `coinop_washer`, robot `"null"`).

## Seed provenance

Seed task: `rlbench/open_washing_machine`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/open_washing_machine.py`) — a Franka
grabs the washing machine door handle and swings the hinged door open. The entire
plan is ONE pull on ONE articulation, the articulation is free to move from step
zero, and its end state (door open) IS the goal.

## What changed and why it is strategically different

| | seed | this task |
|---|---|---|
| articulations | one free revolute door | TWO coupled ones: a gravity ROCKER pay-lever (revolute) whose tab LOCKS a sliding door (prismatic) |
| can the door move at t=0? | yes — just pull | no — the lock tab blocks it after ~5 mm; shoving HARDER presses the tab INTO its stop (torque-sign geometry, smoke-verified) |
| role of the articulation | its end pose IS the goal | both are instrumental GATES; the scored object is a free ball sealed behind them |
| discrimination | none | two same-sized coins; only the heavy BRASS token (~3x the rocker's holding torque) tips the lever — the light slug (~0.4x) physically cannot. Selection is by MASS, invisible in shape |
| plan | one pull | pay (selective deposit into a moving lever) -> open -> extract through the doorway -> deliver to a separate container |
| goal predicate | door angle | ball resting in the basket AND the token still riding the paid-down tray |

So the inversion is: the seed rewards moving an articulation; here moving the
articulation is impossible until a physics puzzle (torque balance on a lever) is
solved by depositing the right payload, and the articulation's motion is never
scored — only the payload chain it releases. It also differs from the sibling
built from the same seed (`open_washing_machine_i252`: open-deposit-close cycle
through a payload-carrying drawer — no lock, no discrimination-by-mass, the
articulation must RETURN to start) and from the other packages inspected
(`close_grill_i8`: structure building; `robobench/suites/packing/pen_holder`:
multi-object insertion): none contains a pay-to-unlock mechanism where a lever's
torque balance gates an extraction.

## Scene

- Housing: ONE kinematic compound on a table — raised chamber floor, back/side
  walls, front wall split around a 110 x 90 mm doorway, lintel, roof (no top
  extraction), and the free-standing pay-post column. Never teleported: safe
  anchor for both spawn-authored joints.
- Door: dynamic compound (blue plate 130 x 100 x 10 mm + black handle bar) on a
  spawn-authored `UsdPhysics.PrismaticJoint` (housing -> door, local +y, limits
  `[0, 0.08]`), linear damping 4/s parks it where released; mass 0.40 kg.
- Rocker: dynamic compound (beam along y, red lock-tab finger at the -y arm,
  yellow coin-tray pocket at the +y arm) on a spawn-authored `RevoluteJoint`
  (axis x, limits `[-20 deg (paid), +8 deg (locked)]`). MassAPI authors mass
  0.10 kg, CoM offset -15 mm y (gravity holds it LOCKED, tau0 ~ 0.015 N m) and a
  diagonal inertia. Tab centre sits 24 mm BELOW the pivot, so door contact
  torques the rocker INTO the locked stop; the tab clears the door top by
  ~14 mm at the paid stop and overlaps it by ~20 mm locked (asserted in cfg).
- Token (brass, 60 g) and slug (grey, 8 g): identical Ø30 x 8 mm cylinders.
  Token tray torque ~3.0x tau0, slug ~0.43x (asserted).
- Ball: Ø45 mm, sealed in the chamber; Basket: dynamic open box (inner 90 mm),
  teleported at reset for randomization.
- `post_step` owns the wrench slots (`door_f`, `ball_f`, `rocker_tau`) and
  latches rubric progress through slow-speed gates. `ball_f` is WORLD-frame by
  contract: post_step re-encodes it per step with `quat_apply_inverse` because a
  rolling sphere's body frame revolves once per pi*d of travel (the raw call is
  body-frame — unencoded, the OUT servo stalls at seed-dependent "walls").

Per-episode randomization (readback-verified in smoke check 1/2): token, slug
and basket table poses, ball pose inside the chamber, door crack 0–2 mm (far
below the ~4 mm tab gap and the 60 mm open latch).

## Rubric (anchored in the demonstrated solve trajectory: 0 -> 0.30 -> 0.45 -> 0.60 -> 1.0)

- `success()` = token resting in the coin tray (judged in the ROCKER'S BODY
  FRAME, so it rides the tipping lever; z-window rejects rim perching) AND
  rocker at the paid stop AND ball resting inside the basket (BASKET body frame,
  z-window rejects rim perching) AND everything settled.
- `score()` = 1.0 iff `success()`; else `0.15*token_latch` + `0.15*paid_latch`
  (paid stop reached WITH the token aboard — a hand-press on the empty lever
  earns nothing, smoke check 10) + `0.15*open_latch` (door past 60 mm — geometry
  says this needs the tab lifted) + `0.15*out_latch` (ball at rest outside the
  chamber) + `0.15*in_basket` (live). Null policy ~0; latched credit never
  evaporates under correct behavior.

## Solution outline (solve.py, robot="null", teleport = transport only)

1. PAY — the token is teleported ONLY to free air ~30 mm above the live coin
   tray (readback of the rocker pose; asserted clear of everything), then FALLS
   in; its weight tips the rocker to the paid stop by gravity alone -> 0.30.
2. OPEN — velocity-cascade force servo on the door body (`door_f` slot,
   KV*dt/m = 0.25 against the one-substep wrench delay, F <= 8 N) slides the
   live prismatic joint to 8 mm short of the stroke; released slow -> 0.45.
3. OUT — velocity servo on the ball (`ball_f`, F <= 1.2 N) rolls it through the
   exposed doorway window and off the sill onto the table; released slow -> 0.60.
4. DELIVER — the ball (now out in the open) is teleported to free air above the
   basket (readback, asserted clear of the housing), falls in, settles -> 1.0.
5. PERSIST — 3.3 simulated seconds hands-off (wrench slots asserted zero),
   success re-checked, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 and 1, both `SIM_GEN_SOLVE: SUCCESS` with the
monotone score trace 0.000 / 0.300 / 0.450 / 0.600 / 1.000 / 1.000.

## Embodiment argument (a real Franka could do this)

Base on the table at ~(0.55, 0.05, 0.40), facing -x toward the washer front
(~0.5 m to the housing face — mid-workspace). Per object:

- Token PICK: a Ø30 x 8 mm coin is thin for a table-top pinch, so the natural
  Franka strategy is the standard edge-slide: drag it to the table edge or use
  a fingertip side-pinch on the 8 mm rim (the same grasp both coins afford —
  which is the point: only MASS tells them apart, and a gripper feels that at
  lift). Carry ~0.25 m and release ~30 mm above the open 58 x 58 mm tray pocket
  — the solve's free-air drop + gravity tip mirrors this; generous tolerance.
- Door OPEN: the handle is a 60 x 34 x 16 mm bar standing proud of the plate —
  canonical two-finger wrap; slide it +y along the rail, <= 8 N against a
  0.4 kg door with 4/s damping (an easy one-hand pull). The capped-force,
  capped-velocity servo is exactly a compliant gripper pull.
- Ball OUT: the chamber roof forbids top grasps; the Ø45 mm ball is reached
  THROUGH the 110 x 90 mm doorway with the fingertips and rolled/raked out over
  the flush sill — the solve's low-force rolling push mirrors a fingertip rake.
  Once on the open table it affords a standard top grasp.
- DELIVER: carry and release over the 90 mm-inner basket — generous.

No phase needs more than one hand, exotic wrenches, or poses outside the reach
envelope; every contact surface is a plain face >= 8 mm across.

## Execution order — declared and physically enforced

Declared order: PAY before OPEN before OUT (DELIVER last). Enforced by
geometry/mechanics, not rubric fiat: the tab blocks the door after <= 5 mm of
travel and door shoving torques the rocker INTO its stop (check 4), so OPEN
physically requires PAY (with the token — the slug cannot, check 5); the chamber
is sealed (roof, walls, 15 mm under-door gap vs the 45 mm ball) so OUT requires
OPEN (checks 4/5 imply the ball cannot leave); and stopping after any prefix
(paid+open but ball inside — check 9) is not success. The rubric additionally
demands the payment STAND at the end (token still in the tray, lever still
down): removing the token revokes success (check 11).

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 12/12` on the forge)

1. settle — clean reset; door AT its sampled crack, rocker AT the locked stop,
   coins/ball/basket AT their sampled poses by READBACK; score ~0.
2. random — 3 seeds: all sampled quantities vary (max-pairwise deltas).
3. null — 2 s of nothing: score < 0.05, no success.
4. lock holds — 8 N door shove: moves > 1 mm (tab engaged, not fused) but
   blocked < 15 mm; rocker stays at its locked stop.
5. slug can't pay — slug dropped in the tray: rocker stays locked, the same
   shove still blocked, zero pay credit.
6. token pays — token dropped in: rocker tips to the paid stop (readback angle)
   and the SAME 8 N shove opens the door past 60 mm — the mechanism differential.
7. no-pay delivery — ball constructed resting in the basket, nothing paid (the
   seed-analog "just deal with the door/ball" breach): no success, score < 0.5.
8. slug + delivery — slug in tray on top of check 7: still locked, no success.
9. prefix stop — paid + door opened, ball left inside: no success, score < 0.55.
10. press cheat — a probe torque presses the EMPTY lever: door opens (the gate
    is the angle) but `paid_latch` stays 0, the lever falls back locked once the
    door is returned shut and the press released (an open door plate sits under
    the tab's return path), and a delivered ball still is not success.
11. exactness + revocation — full correct end state: success and score == 1.0,
    stable 1 s; removing the token revokes success, latched credit remains.
12. frames — rgb frames recorded and saved as frames.npz.
