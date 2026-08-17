# Shuttle-hatch kiosk: operate the transfer drawer to serve the ketchup

Task id: `living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i158`
Env id: `simgen.shuttle_hatch` (scene `simgen.shuttle_hatch`, robot `"null"`)

## Seed provenance

Seed: `roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray.py`
(LIBERO-90 `living_room_scene3`, "pick up the ketchup and put it in the tray").
The seed is a passive pick-and-place: a ketchup bottle sits on a table among distractor
items next to a static open `wooden_tray`; `_terminated` fires the moment the ketchup's
position enters the tray's `contain_region` bounding box. The tray never moves, nothing
is sealed, and a single carry-through-free-air-and-drop is the entire task.

## What changed and why it is strategically different

| Axis | Seed | This task |
| --- | --- | --- |
| Tray | Static open-top box in free air; approachable from any direction | The tray is a **captive shuttle drawer** riding in a walled channel inside a **fully roofed, sealed kiosk**; it can only translate between a back stop (LOAD) and a front stop bar (SERVE) and can never be lifted out (smoke 11: an 8 N upward-and-outward yank leaves it captive and it reseats at serve) |
| Access to the tray interior | Everywhere (open top) | **Position-dependent**: the only opening in the kiosk is a roof **deposit chimney over the BACK of the channel**. The chimney's drop zone overlaps the tray interior ONLY when the tray is parked at the LOAD stop (34 mm margins); at SERVE the tray sits fully under solid roof |
| Serve window | n/a | The front aperture the tray protrudes through at SERVE is sealed to a **12 mm** gap over the tray walls — less than the bottle's 36 mm minimum dimension, so nothing can be inserted from the front (smoke 6: a bottle flung at the window at 0.7 m/s bounces off) |
| Success | Geometric containment, instantaneous | Ketchup at rest **inside the tray** AND tray parked back **at SERVE** AND everything settled/finite — after loading, the drawer must be pulled back out |
| Ordering | None | **Physically forced 3-step protocol**: push tray IN to the load stop → drop the bottle through the chimney → pull the tray back OUT to serve. Doing the drop first strands the bottle on the channel floor behind the tray, where it **jams the tray short of the load gate** (smoke 7) |
| Perception | Named ketchup asset | 3 same-size bottles (ketchup red / mustard yellow / mayo white) **shuffled over 3 table slots** per episode — color is the only cue; mustard in the tray at serve scores 0 (smoke 10) |
| Seed's plan | Sufficient | **Impossible**: lowering the bottle from above onto the tray-at-serve puts it on the closed kiosk roof (smoke 5, score 0); the front window rejects insertion (smoke 6) |

Same-seed-family siblings share no mechanism with this task:
`..._cream_cheese_..._i33` (flap-chute pantry) is about **forcing entry through a
compliant one-way barrier** — a push against a yielding flap that gravity-seals behind
the object; there is no moving container and no position-dependent aperture.
`..._butter_..._i139` (beam balance) is about **mass measurement and restoring a
two-sided equilibrium** — nothing is sealed and nothing shuttles. This task's mechanism
is a **captive transfer drawer**: the agent must operate a constrained container through
a stroke so that a fixed opening and the moving interior line up, then undo the motion.
No joint is used anywhere — the drawer is captive purely by rigid geometry (channel
walls, back wall, front stop bar, roof), so the mechanics are honest contact dynamics.

Honest geometry budget (station-local, front = +x): roof hole x ∈ [−0.156, −0.068];
tray interior at LOAD x ∈ [−0.190, −0.034] (34 mm margins both sides), at SERVE the
interior starts at −0.033 > −0.068 → zero overlap. Roof underside 0.204 vs tray wall
top 0.192 → 12 mm crack < 36 mm bottle. A bottle dropped out of order comes to rest
between the back wall inner face (−0.200) and the tray's back wall, capping the tray's
reachable x at ≥ −0.076, short of the −0.092 load gate — measured, not asserted by fiat
(smoke 7 pushes at 5 N for 300 steps and the tray stalls at −0.077).

## Solution outline (solve.py, teleport = transport only)

- **P0** settle 120 steps; mass readback (tray 0.450 kg, station 25.0 kg via
  `root_physx_view.get_masses()`); assert tray starts at SERVE, ketchup outside, score
  ≈ 0. `SIM_GEN_SCORE 0.0000`.
- **P1** push the tray IN by a bang-bang horizontal force on the exposed grip tab
  (2.5 N base, stall-escalation ≤ 8 N, frame-drag probe every 45 steps), release, settle
  90 → tray at the LOAD stop, `loaded` latch. `SIM_GEN_SCORE 0.1500`.
- **P2** teleport the ketchup to **free air above the chimney mouth** (station-local
  (−0.112, 0, 0.359), 83 mm above the mouth — outside every rubric volume; asserted
  `not ketchup_in_tray` at the write), zero velocity; gravity threads it down the
  chimney into the tray. Hands-off until in-tray + settled; assert **no success** (tray
  still at load). `SIM_GEN_SCORE 0.4000`.
- **P3** pull the tray back OUT (same bang-bang, opposite sign) and seat it at SERVE →
  live success. `SIM_GEN_SCORE 1.0000`.
- **P4** hands-off persistence 400 steps (~3.3 simulated s); success holds.
  `SIM_GEN_SCORE 1.0000`, then `SIM_GEN_SOLVE: SUCCESS`.

Scores are non-decreasing (latched partial credit: loaded 0.15 + in-tray 0.25, cap
0.40; 1.0 iff live success). No force is ever applied to the bottle; the teleport never
places anything inside a rubric volume — the bottle enters the tray only by falling
through the chimney.

## Franka feasibility

Single Franka based at **(−0.26, 0, 0), facing +x**. The three bottle slots sit
0.34–0.44 m from the base on the open table; the kiosk front is ~0.5 m out with the
grip tab toward the robot; the chimney mouth is ≈ 0.73 m away at z = 0.276 — all inside
a 0.85 m reach envelope.

- **Pick**: bottles are free-standing 36 × 52 × 150 mm boxes with ≥ 70 mm clearance —
  a side or top pinch across the 36 mm face fits the 80 mm parallel jaw trivially.
- **Drawer strokes**: the tray's grip tab (50 × 60 mm plate on a handle bar) stays
  OUTSIDE the kiosk at every tray position (station-local x up to 0.345), so both
  strokes are plain horizontal push/pull on an exposed handle at ~2.5 N — no reaching
  into the mechanism, no force control beyond a gentle servo.
- **Deposit**: release the bottle above the 88 × 88 mm chimney mouth; gravity does the
  insertion, exactly as the teleport solution demonstrates. Clearance to the mouth
  walls is ≥ 18 mm per side around the 36 × 52 mm bottle footprint.
- Nothing requires two hands, regrasping inside an enclosure, or torque beyond the
  arm's capability; the kiosk is a 25 kg damped body that incidental contact cannot
  displace (and all predicates are station-frame relative regardless).

## Randomization

Per episode: kiosk yaw 180° ± 25° plus ±4 cm base translation; the three bottles
permuted over the three table slots (`torch.rand(m,3).argsort` — slot of the ketchup is
the episode's perception problem) with ±2 cm jitter and free yaw; tray start position
jittered along its stroke. Smoke check 2 verifies pose deltas by readback across seeds
(station yaw Δ19.9°, station xy Δ38.8 mm, tray x Δ7.7 mm, ketchup xy Δ184.7 mm, yaw
Δ87.9°); check 3 verifies the ketchup occupies ≥ 2 distinct slots over 10 resets
(observed all 3).

## Smoke battery (12 rejection-only checks)

1. Settle / no-NaN; tray at serve; score ≈ 0.
2. Randomization readback: station yaw/xy, tray x, ketchup xy/yaw all differ across seeds.
3. Slot shuffle: ketchup occupies ≥ 2 slots over 10 resets (saw [0, 1, 2]).
4. Null policy 240 steps → tray moves < 8 mm, score ≈ 0, no success.
5. **Seed strategy**: bottle lowered from above onto the tray-at-serve → lands on the
   closed ROOF (z = 291 mm > roof top 216 mm), score 0, NOT success.
6. Serve window sealed: bottle flung at the front window at 0.7 m/s → bounces off
   (min approach 177 mm vs wall face 133 mm), nothing enters, score ≈ 0.
7. **Order violation**: bottle dropped down the chimney with the tray still at SERVE →
   strands on the channel floor behind the tray; a 5 N push for 300 steps then stalls
   the tray at −77 mm, short of the −92 mm load gate → score 0. Out-of-order is
   physically punished, not just unrewarded.
8. Serve near-miss: bottle inside the tray but tray 8 mm short of seated → in-tray
   credit only (0.25), no success.
9. Abandoned at load: tray at the load stop with the bottle inside → 0.40 exactly
   (float32-safe band), no success.
10. Wrong object: mustard in the tray at serve → score 0, no success.
11. Captive drawer: tray at load, 8 N up-and-out yank for 180 steps → max x 56 mm
    (stop at 45 mm), tray stays captive and reseats at SERVE.
12. `frames.npz` rendered and saved to CWD (116 frames, 600 × 960).

Prints `SIM_GEN_SMOKE: ALL PASS 12/12`.

## Execution order declaration

The protocol `push tray to LOAD → drop bottle through the chimney → pull tray to
SERVE` is REQUIRED and physically forced: the chimney only lines up with the tray
interior at the load stop, the serve window is sealed, and an out-of-order drop jams
the load stroke (smoke 7). The rubric itself judges only the settled end state
(ketchup in tray, tray at serve, settled) plus latched non-decreasing partial credit —
no step is rewarded by fiat.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0`: station (+0.398, −0.038) yaw +175.0°, ketchup slot 2 — SUCCESS.
- `solve --seed 1`: station (+0.374, −0.020) yaw +199.4°, slot 1 — SUCCESS.
- `solve --seed 2`: station (+0.326, +0.003) yaw +176.3°, slot 0 — SUCCESS.
  All three: scores 0.0000 → 0.1500 → 0.4000 → 1.0000 → 1.0000, ~19 s each; the bottle
  lands standing dead-center in the tray (tray-local ≈ (−0.002, −0.001, +0.087));
  tray load x −0.109, serve x +0.039.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 12/12** (40.1 s), frames.npz (116 × 600 × 960)
  saved.

## Files

- `scene.py` — ShuttleHatchScene + kiosk/tray compound spawners + bottles + rubric;
  registers `simgen.shuttle_hatch`.
- `solve.py` — tab-push drawer strokes (bang-bang force with frame-drag guard) +
  chimney gravity deposit + persistence certificate (`--seed N`).
- `smoke.py` — 12-check rejection battery + video.
