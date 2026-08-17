# living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i426 — Ballast Gate (scene `ballast_gate`)

Get the WHITE milk carton into the green crib inside a roofed vault — but the vault's
only doorway is sealed by a spring-loaded ORANGE drop-gate that is too stiff to yield to
pressed cargo. The gate carries a walled loading tray on its outer face: the RED juice
carton (0.45 kg, the "decoy" of the seed) is the REQUIRED ballast — laid in the tray its
weight overpowers the closing spring and sinks the gate, opening the doorway hands-free.
The milk is then carried horizontally through the open doorway and lowered into the crib.
The task is not done until the ballast is REMOVED again: with the tray emptied the spring
re-seals the doorway on its own, and success demands milk-in-crib AND gate-shut AND the
juice clear of the vault. The seed's decoy is this task's key; the seed's one-way "drop
it in" becomes open → deliver → un-do.

**Scene** `ballast_gate` / env `simgen.ballast_gate` (NullRobot, scene-level physics).
**Seed** `libero_90/living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket`.

## Provenance and strategic difference

The seed is a **grasp-carry-release-from-above** into an open-top basket. Every
load-bearing element of that plan is denied or inverted:

- **Top entry is physically removed.** The vault has a FULL roof: smoke check 7 drops
  the carton from directly above the crib and it lands ON the roof (z ≥ 0.540), earning
  nothing. The only boundary opening is the front doorway.
- **The doorway does not yield to cargo.** The gate is spring-preloaded shut at 2.04×
  its own weight; smoke check 8 presses a held lying carton against it with 2 N
  horizontal force for 2 s — the gate does not budge (min q ≥ −0.02) and the carton
  never passes the wall plane. This is the exact anti-recipe of sibling `i275
  drop_chute`, where the load-bearing act is the cargo shouldering a passive flap open.
  Here pushing is USELESS; the door is opened by a remote counterweight.
- **The seed's decoy becomes required equipment.** The juice (0.45 kg vs milk 0.35 kg)
  must be picked up and laid into the gate's tray: 0.57 kg on the gate makes
  (0.12+0.45)·g = 5.59 N ≥ 1.33× the spring's max closing force (4.20 N), sinking the
  gate to its lower stop and holding it there with no hands (smoke 11: laid juice sinks
  q ≤ −0.20 and holds for 300 hands-free steps). The seed never touches the decoy.
- **The end state requires UN-doing the enabling step.** Success gates on the gate
  being re-sealed (q ≥ −0.006) AND the juice being fully out of/off the vault. Leaving
  the ballast in place after delivery is the natural failure (smoke 13: milk delivered,
  juice still on tray → success FALSE, score capped at 0.65). Removing it lets the
  spring re-close on its own — an active reversal with no analogue in the seed.
- **Distinct from every package read while building this.** Sibling `i177 dump_hopper`:
  a HELD lever throughout discharge, the container transported, the milk untouchable —
  here NO mechanism is ever held (the ballast holds the gate, the arm is free), the
  vault never moves, and the milk is directly carried. Sibling `i275 drop_chute`: cargo
  pushes a passive one-way flap open — here the door is weight-LOCKED against exactly
  that plan (smoke 8) and insertion is a carried lay-in, not a push-through; i275 also
  ends when gravity closes its flap with the decoy merely kept away, while here the
  operator must actively strip the ballast. The `pen_holder` exemplar is many-object
  tip-up insertion into a carriable open cup with no mechanism at all.

The hands-free ballast is also the single-arm forcing function: one Franka cannot hold
the gate down AND carry the milk through the doorway at the same time, so the
counterweight plan is not a convenience but the only embodied route.

## Mechanism numbers (vault frame, +x out the doorway; vault yaw randomized ±180°)

- Vault: 0.400 × 0.400 m outer, walls 12 mm, one 30 kg DYNAMIC compound (zero
  sleep/stabilization thresholds so the joint anchor follows reset teleports). Solid
  base slab up to the 0.260 m sill = interior floor. Full roof 0.560–0.572 m. Front
  wall = two pillars |y| ∈ [0.120, 0.200] (slick outer faces) leaving a 0.240 m-wide
  doorway; corridor headroom under the open gate 0.340–0.560 m (≥ milk width + 100 mm,
  asserted).
- Gate: ORANGE 270 × 320 × 12 mm panel at plane x = 0.210 (4 mm running gap to the
  vault face, 0.003 ≤ gap ≤ 0.020 asserted vs contact offsets), 0.12 kg, on a real
  `UsdPhysics.PrismaticJoint` (axis Z, limits [−0.225, 0]); closed spans z 0.245–0.565
  (seals the doorway, asserted); open bottom edge = 0.340 curb. `DriveAPI` "linear"
  spring: stiffness 8 N/m, damping 10, target +0.300 above the upper stop → 2.4 N
  preload = 2.04× gate weight (holds shut); max closing force at full travel 4.2 N.
- Tray: walled (35 mm walls, inner 116 × 176 mm) on the gate's outer face, top at
  z = 0.535 when closed; center x = 0.280, y = +0.235 — outboard of the doorway, under
  open sky. A lying juice carton fits with margin and is retained through the 0.225 m
  drop (asserted).
- Ballast arithmetic (all asserted with margins): preload 2.40 N vs gate 1.177 N
  (2.04×); loaded (0.57 kg)g = 5.59 N vs 4.20 N max spring (sinks, 1.33×); empty
  re-close 4.20/1.177 = 3.57×; the MILK too would sink it (1.05×) — using the milk as
  ballast is possible but then nothing is left to deliver, an arithmetic dead end, not
  a rule. Probe pair non-vacuity: 0.8 N on the gate must NOT open it, 3.0 N must
  (smoke 9/10).
- Crib: green pad (top 0.266) + 4 walls (rim 0.316) centered at x = 0.030, inner
  0.280 × 0.220 m. Curb (0.340) clears the rim by ≥ 15 mm (asserted) so the carried
  carton passes over.
- Cartons: milk WHITE 0.35 kg, juice RED 0.45 kg, both 60 × 60 × 160 mm, spawned
  STANDING on mirrored jittered arcs (bearing 30–70° off vault +x, radius 0.60–0.72 m,
  free yaw) — beyond tray reach plus half-diagonal (0.590 < 0.60, asserted), so the
  null policy scores 0.
- Randomization: vault xy ± 50 mm, yaw ± 180° (gate re-written consistently closed in
  the new frame); both cartons side/bearing/radius/yaw. Smoke 3–4 read all of these
  back across 8 seeded resets.

## Rubric

`success()` = **milk in crib** (vault-local |x − 0.030| ≤ 0.105, |y| ≤ 0.075,
z ∈ [0.268, 0.362] — accepts lying and standing rests on the pad, rejects rim-perches
and floor-beside-crib, asserted) AND **gate sealed** (q ≥ −0.006) AND **both cartons
settled** (lin ≤ 0.05 m/s, ang ≤ 1.0 rad/s) AND **juice out** (no part inside the box
|x| ≤ 0.26, |y| ≤ 0.26, z ∈ [0.20, 0.80] — covers the interior, doorway, roof-top;
the tray center sits at x = 0.28 deliberately outside it because a loaded tray already
fails via the gate clause).

`score()` (stateless monotone ladder, `max` of): 0.25 gate ballasted fully open
(q ≤ −0.200) · 0.65 milk inside the crib window · 1.0 iff `success()`. Order is forced
by geometry, not bookkeeping: the roof and the sealed doorway are the only boundary,
so milk-in-crib is unreachable while the gate is shut.

## Solution outline (`solve.py`, transport-only teleports)

- **P0** settle + layout readback (gate q > −0.01, cartons standing outside,
  score ≤ 0.03).
- **P1** ballast: teleport the juice (free transport of a graspable carton) lying
  `_qx(π/2)` onto the tray at vault-local (0.280, 0.235, 0.571); the spring loses to
  the weight and the gate sinks to its stop hands-free (readback q = −0.2250); assert
  `gate_open_deep`; score 0.25.
- **P2** carry the milk through: hover-place lying at (0.460, 0, 0.390), then a
  force-carried traverse — fz = weight-feedforward + PD height servo (clamped
  [0, 2 mg]), fx velocity-servo inbound then position-spring to the crib center,
  fy centering, small body-frame angular damping. All forces are built per-step in the
  VAULT frame and converted to the CARTON BODY frame (`f_body = qinv(q_milk)·q_vault·
  f_vault`) — the pod's world-frame wrench drag reference is captured at first
  application and goes stale across resets. Retry ladder (3 gain/cap rungs) with
  `set_state` snapshot rollback; all forge runs delivered on rung 0. Lower at 0.15 m/s
  to z = 0.310, release inside the window; assert `milk_in_crib`; score 0.65.
- **P3** strip the ballast: teleport the juice out to (0.75, −0.45) standing; the
  spring re-seals the doorway on its own (readback q = −0.0000); assert the milk was
  not disturbed; 60-consecutive-step success streak (instantaneous readings can land
  on velocity turning points); score 1.0.
- **Persistence**: 400 hands-off steps (3.3 s) with success re-checked every step,
  then `SIM_GEN_SOLVE: SUCCESS`. Watchdog `threading.Timer` (daemon) + `os._exit`.

The cartons are teleported only as transport (juice onto the tray / off to the floor,
milk to the doorway hover); the gate is NEVER teleported, wrenched, or held — every
gate motion in the entire solve is spring-vs-ballast physics, and the delivery traverse
is a closed-loop force carry.

## Embodiment (single Franka, parallel jaw, one base pose)

Base ≈ 0.55–0.65 m out along the vault's +x (doorway) axis reaches everything: both
carton spawn arcs (≤ 0.72 m, standing 60 mm cross-section < jaw stroke), the tray
load at (0.280, 0.235, 0.535) and the strip-off — both outboard of the doorway under
open sky — and the delivery: a lying carton carried at z ≈ 0.39 through the 0.24 m-wide
doorway with 140 mm of headroom under the roof edge, lowered 80 mm to the pad, jaw
opened, retracted. Reach past the wall plane is ≤ 0.21 m (crib center at x = 0.030),
within a Franka's envelope at that height. The single arm CANNOT hold the gate and
carry simultaneously — the hands-free ballast is the only embodied route, by
construction. The roof denies all top-down work by geometry, not fiat.

## Execution order

`scene.py` (geometry + ~25 honesty asserts in `__post_init__`, including the full
spring/ballast force arithmetic with margins) → `solve.py` iterated on the forge →
rubric finalized (streak/settle gates) → `smoke.py` rejection battery. No check was
ever weakened to make a run pass; all three solve seeds and the smoke battery passed
on their first forge runs.

## Smoke battery (22 checks, rejection-driven)

1. reset settles: finite, gate shut (q > −0.01), milk outside · 2. baseline score
≤ 0.02 · 3. vault xy + yaw randomization readback (8 seeds) · 4. carton poses vary ·
5. null policy: 300 idle steps, score ≈ 0 · 6. mass readback (`get_masses()`: vault
30, gate 0.12, milk 0.35, juice 0.45) · 7. roof denial: milk dropped from above the
crib lands ON the roof (z ≥ 0.540), no credit · 8. locked door: milk held lying and
pressed against the gate with 2 N for 2 s — gate holds (min q ≥ −0.02), carton never
passes the wall plane, score ≤ 0.02 (the anti-i275 probe) · 9. 0.8 N down on the gate:
holds shut (non-trivially preloaded) · 10. 3.0 N down: opens ≤ −0.15, released springs
back shut (non-vacuous both ways) · 11. juice laid on tray: sinks to the stop and
holds 300 hands-free steps, score 0.25 · 12. milk posed mid-doorway judged
immediately: not in crib, ≤ 0.26 · 13. incomplete end: milk delivered, juice still on
tray → success FALSE (gate clause), score 0.65 cap · 14. acceptance: juice stripped to
the floor → gate re-seals on its own, success TRUE, score ≥ 0.99 · 15. settle gate:
delivered milk kicked and judged immediately → rejected · 16. juice parked ON the
roof → FALSE (decoy clause) · 17. juice standing inside beside the crib → FALSE ·
18. removed → TRUE again · 19. wrong object: JUICE in the crib, milk outside → 0.00 ·
20. near miss: milk standing on the vault floor beside the crib → 0.00, no success ·
21. audit: success() never True at any judged point except the constructed acceptance
probes (14, 18) · 22. final no-NaN. Frames (600 × 960) → `frames.npz`.

## Verification record (forge, 2026-08-17)

- `solve --seed 0`: rc=0, `SIM_GEN_SOLVE: SUCCESS`, 18.2 s. Scores 0.000 → 0.250 →
  0.650 → 1.000 (non-decreasing). Vault (−0.010, +0.002) yaw +170°; ballasted
  q = −0.2250; delivered milk at vault-local (+0.027, −0.000, +0.296) lying; stripped →
  re-sealed q = −0.0000; zero persistence flickers; ladder rung 0.
- `solve --seed 1`: rc=0, SUCCESS, 18.1 s. Vault (+0.039, −0.047) yaw +62°; rung 0.
- `solve --seed 2`: rc=0, SUCCESS, 18.1 s. Vault (−0.007, +0.033) yaw −153°; milk
  spawn (+0.492, +0.375), juice (+0.550, −0.322); re-sealed q = −0.0000; rung 0.
- `smoke`: rc=0, `SIM_GEN_SMOKE: ALL PASS 22/22`, 50.7 s, 351 frames saved. Key
  readbacks: roof-drop rests at z = 602 mm ON the roof; 2 N cargo press moves the gate
  min q = −0.0000 (never budges); 0.8/3.0 N probe pair −0.0000 / −0.2083 then springs
  back shut; laid juice sinks the gate to q = −0.2250 and holds hands-free 2.5 s;
  randomization spreads Δvault = 69/77 mm, Δyaw 3.36 rad, Δmilk 1.28 m, Δjuice 1.34 m.
