# Queue dispenser: sight the queue, purge the rejects, stage the tray, serve the ketchup

Task id: `living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i207`
Env id: `simgen.queue_dispenser` (scene `simgen.queue_dispenser`, robot `"null"`)

## Seed provenance

Seed: `roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray.py`
(LIBERO-90 `living_room_scene3`, "pick up the ketchup and put it in the tray").
The seed is a passive pick-and-place: a free-standing ketchup bottle on an open table,
a static open `wooden_tray` next to it, and `_terminated` fires the instant the
ketchup's position enters the tray's `contain_region` box. Grasp, carry through free
air, drop — nothing is ordered, nothing is at risk, nothing meters or hides anything.

## What changed and why it is strategically different

| Axis | Seed | This task |
| --- | --- | --- |
| The target object | Free-standing on a table, graspable from any direction | **Never directly manipulable.** The ketchup carton is sealed inside a roofed 610 mm DISPENSER COLUMN, stacked in a 3-carton vertical queue on an 11° feed ramp. Smoke 5: a 10 N upward yank lifts a queued carton 8+ mm but it stays captive in the magazine; an 8 N rearward pull through the push slot cannot extract it |
| How objects move to the goal | Carried by the agent | **Only via the machine's own outlet**: the bottom carton rests against a 16 mm RETENTION LIP; a firm push on its back face through the 44 mm rear push slot vaults it over the lip; gravity takes it down a slick launch tongue and off the tip. One push ejects exactly one carton — the queue drops and re-seats behind the lip (smoke 7); a 1.5 N nudge for 300 steps does NOT defeat the lip (smoke 6) |
| Ordering | None | **Physically forced bottom-first**: every carton below the ketchup must be dispensed before it. There is no other route out of the column |
| Risk | None — the bottle can be re-grasped forever | The tongue tip overhangs a 200 mm deep REJECT PIT. Anything dispensed with nothing staged below is **irrecoverably lost** (pit too deep/narrow to retrieve; smoke 8: ketchup dispensed before staging the tray → pit → score capped, no success ever) |
| The tray | Static scenery | A loose 0.25 kg catch tray the agent must **stage on the pit rim** (seat window 20 mm xy) so it covers the mouth and catches the drop — staging is a sub-goal with its own credit, and the tray must still be seated at the end |
| Perception | Named asset | 3 same-shape cartons (ketchup red / butter yellow / cream white) **shuffled over the 3 queue levels** per episode; the order is readable only through a 16 mm sight slot. Wrong carton in the tray at the end scores no success (smoke 9) |
| Success | Instantaneous containment | Settled end state: ketchup at rest inside the tray AND tray seated on the rim AND **no other carton in or over the tray** (dirty gate covers the tray's full column of space, smoke 11) AND everything finite/settled |

Same-seed-family siblings share no mechanism with this task:
`..._i158` (shuttle-hatch kiosk) is about **operating a captive drawer through a
stroke** so a fixed roof aperture and the moving tray interior line up, then undoing
the motion — the target bottle is free-standing on a table the whole time.
`..._cream_cheese_..._i33` (flap-chute pantry) forces entry through a compliant
one-way flap; `..._butter_..._i139` (beam balance) is mass measurement. This task's
mechanism is a **metered gravity-feed queue with a point of no return**: the target is
never touchable, dispensing is one-at-a-time and bottom-first by contact mechanics
(lip energy barrier + shielded follower drop), and a staging error destroys the
episode irreversibly. No joints anywhere — the column is a rigid kinematic compound;
the metering, captivity, and loss are all honest contact dynamics.

Honest geometry budget (station-local, dispense = +x): ramp surface z = 0.315 −
tan 11°·x; the queue rests on an 8 mm grippy FEED BED (μ = 0.14) whose front edge
carries a lip 16 mm proud of the bed. Follower drop-slide energy after a dispense
(~0.04 J/kg: 4 mm gap drop + bed slide) is under the lip pivot barrier (~0.09 J/kg),
and the front wall's lower edge (outlet top z = 0.420) shields ~80 of the follower's
88 mm drop against forward topple — so one push moves exactly one carton. Past the
lip the tongue is slick (μ = 0.06 < tan 11°), so a vaulted carton accelerates to the
tip (x = 0.117) and drops into the pit column (mouth x ∈ (0.085, 0.235), rim z =
0.200). The tray seated on the rim (origin z ∈ (0.190, 0.215), xy tol 20 mm) covers
the mouth; its 40 mm walls leave 32 mm of clearance under the tongue tip for sliding
it in level from the front.

## Solution outline (solve.py, teleport = transport only)

- **P0** settle 120; mass readback (tray 0.250 kg, ketchup 0.150 kg via
  `root_physx_view.get_masses()`); queue order + layout readback; asserts (all three
  cartons in the magazine, tray unseated, score ≤ 0.03). `SIM_GEN_SCORE 0.0000`.
- **P1** for each level below the ketchup: bang-bang push (4 N base, stall
  escalation ≤ 12 N, velocity caps 0.15/0.30 m/s, frame-drag probe every 45 steps)
  on the bottom carton's back face through the push slot, release once past the lip
  (x > 0.095), hands-off until it rests in the pit, wait for the queue to re-seat.
  `SIM_GEN_SCORE 0.1500`.
- **P2** teleport the tray (pure transport of the free accessory across open ground)
  to 5 mm above its seat, asserting `not tray_seated` and `not _staged` at the write;
  180 hands-off steps to settle onto the rim. `SIM_GEN_SCORE 0.3000`.
- **P3** same push routine on the ketchup → it vaults the lip, slides the tongue,
  drops off the tip into the seated tray; hands-off until live success.
  `SIM_GEN_SCORE 1.0000`.
- **P4** hands-off persistence 400 steps (~3.3 sim s); success holds.
  `SIM_GEN_SCORE 1.0000`, then `SIM_GEN_SOLVE: SUCCESS`.

Scores are non-decreasing (latched credit: rejects-done 0.15 + tray-staged 0.15 +
served-into-seated-tray 0.35, non-success cap 0.65; 1.0 iff live success). No force
is ever applied to the ketchup except the in-fiction push on its back face through
the slot; the only teleport moves the tray — a free object — across open ground to
free air above its seat, never into any rubric volume.

## Franka feasibility

Single Franka based at (−0.30, 0, 0) behind the column, or equivalently at the
side — every working point is inside a 0.75 m envelope:

- **Perception**: the sight slot is a vertical 16 mm window at camera-friendly
  heights (0.30–0.55 m); a wrist camera glance reads the three colors.
- **Pushes**: the rear push slot is a 44 mm-wide opening low on the back face
  (sill at the ramp surface); a closed-gripper knuckle or a held rod pushes the
  bottom carton's exposed back face horizontally at 4–12 N — well within arm limits,
  no reaching around geometry, and the slot walls guide the tool laterally.
- **Tray staging**: the tray is an open-top 180 × 150 mm box with 40 mm walls on
  open ground — a trivial top grasp of a wall (8 mm thick fits the jaw), then a
  place onto the flat 200 mm-high rim; the 20 mm seat tolerance is generous, and the
  32 mm clearance under the tongue admits the tray slid in level from the front.
- Nothing needs two hands or force control beyond a gentle push servo; the column
  is a heavy kinematic station that incidental contact cannot displace, and all
  predicates are station-frame relative regardless.

## Randomization

Per episode: station yaw ±20° plus ±4 cm base translation; the three cartons
permuted over the three queue levels (`torch.rand(m,3).argsort(dim=1)` — the
ketchup's level decides how many rejects the episode needs: 0, 1, or 2); tray start
pose uniform in x ∈ (0.31, 0.41), y ∈ (−0.15, 0.15), free yaw ±180°. Smoke 2
verifies station yaw/xy and tray xy/yaw all differ across seeds by readback; smoke 3
verifies the ketchup occupies ≥ 2 distinct queue levels over 10 resets.

## Smoke battery (12 rejection-only checks)

1. Settle / no-NaN; score ≈ 0; force-frame probe (5 N station-+x on the grounded
   tray) pins the external-force frame mode.
2. Randomization readback: station yaw/xy, tray xy/yaw all differ across seeds.
3. Queue shuffle: ketchup occupies ≥ 2 levels over 10 resets.
4. Null policy 240 steps → tray drifts < 8 mm, score ≈ 0, no success.
5. **Seed strategy dead / captive queue**: 10 N upward yank on a queued carton →
   rises but stays in the magazine; 8 N rearward pull through the push slot →
   cannot extract it. The target cannot be grasped out.
6. Gentle 1.5 N nudge for 300 steps does NOT defeat the retention lip (metering is
   an energy barrier, not decoration).
7. **One-per-push metering**: firm push ejects exactly the bottom carton into the
   pit; the other two remain in the magazine.
8. **Order violation is fatal**: ketchup dispensed with no tray staged → falls into
   the pit; staging the tray afterwards recovers nothing — no success, score capped.
9. Wrong object: butter in the seated tray → no success; ketchup then placed on
   top of the pile → still no success (dirty gate).
10. Unseated serve: ketchup inside the tray but tray on the ground → partial credit
    only, no success.
11. Follower-pile regression: ketchup correctly in the seated tray but butter
    resting on top of it (the historical follower-escape failure) → dirty gate
    holds, no success.
12. `frames.npz` rendered and saved to CWD.

Prints `SIM_GEN_SMOKE: ALL PASS 12/12`.

## Execution order declaration

The protocol `read the queue → eject every carton below the ketchup into the pit →
seat the tray on the rim → eject the ketchup` is REQUIRED and physically forced:
the column only dispenses bottom-first (rigid geometry), the pit is unrecoverable
(smoke 8), and staging the tray after a premature serve recovers nothing. The
rubric judges only the settled end state (ketchup in the seated, clean tray) plus
latched non-decreasing partial credit — no step is rewarded by fiat.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

`solve` run on ten consecutive seeds — **10/10 `SIM_GEN_SOLVE: SUCCESS`**, ~21 s
each, covering every reject count the shuffle can produce:

- **0 rejects** (ketchup at the bottom): seeds 3, 4 — e.g. seed 3: station
  (−0.019, +0.021) yaw +12.7°, queue [ketchup, cream_cheese, butter]; scores
  0.0000 → 0.0000 → 0.1500 → 1.0000 → 1.0000 (rest tray-local (+0.003, −0.014, +0.033)).
- **1 reject**: seeds 0, 1, 2, 5, 7, 8 — e.g. seed 2: station (−0.034, +0.003)
  yaw −3.0°, queue [butter, ketchup, cream_cheese], tray start (+0.328, +0.136);
  scores 0.0000 → 0.1500 → 0.3000 → 1.0000 → 1.0000 (rest tray-local
  (+0.024, +0.007, +0.048)); butter verified at rest on the pit floor (z = 0.037).
- **2 rejects** (ketchup on top): seeds 6, 9 — e.g. seed 6: station
  (−0.034, −0.038) yaw −10.3°, queue [cream_cheese, butter, ketchup].

All runs: scores non-decreasing; the push escalation typically settles at 8.5 N to
vault the lip; the ketchup lands and rests inside the seated tray.

`smoke`: **SIM_GEN_SMOKE: ALL PASS 12/12** (59.7 s), frames.npz saved.

## Files

- `scene.py` — QueueDispenserScene + kinematic column/pit compound + tray + carton
  spawners + rubric; registers `simgen.queue_dispenser`.
- `solve.py` — slot pushes (bang-bang force, stall escalation, frame-drag guard) +
  tray staging teleport + persistence certificate (`--seed N`).
- `smoke.py` — 12-check rejection battery + video.
