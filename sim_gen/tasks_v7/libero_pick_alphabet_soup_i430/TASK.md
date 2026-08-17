# press_latch_vault — `libero_pick_alphabet_soup_i430`

## Provenance

Seed task: `libero/libero_pick_alphabet_soup` — pick the alphabet-soup can out of a
clutter of distractor groceries and put it in an open basket. What survives from the
seed: a RED soup can (r 3 cm, h 10 cm) is the goal object, distractor objects are
present (a GREEN decoy can, a decoy tool), and the goal is "the can ends up inside a
container".

## Strategic difference

- **vs the seed**: the seed is a pure grasp-carry-drop into an OPEN basket. Here the
  container is a fully ROOFED vault — top entry is denied by construction — and its
  only doorway is sealed by a spring-driven gate held shut by a **two-point
  simultaneity interlock**: two spring pins, 17 cm apart, must both be pressed down
  ~2 cm **at the same time** (each alone blocks the gate within millimetres; pressing
  them one-after-the-other achieves nothing). No single finger/object press works:
  the robot must **select the long blue bar** (a too-short yellow rod is the decoy),
  lay it across both white caps, and press its middle ~25 N. The gate then drives
  itself fully open and the released pins pop up between the gate's catch strips — an
  **irreversible ratchet** (the doorway can never be fully re-sealed). Delivery is a
  **push-through** along a porch, over the sill, through the doorway — never a lift.
- **vs `living_room_scene2_..._i426` (ballast-gate vault)**: that gate opens by
  PARKING WEIGHT on a tray — a persistent hands-free state — and success requires
  RE-SEALING. This gate **ignores parked weight by design** (7 N pin preload > any
  loose object's weight; clearing both pins takes ~20 N > the 14 N of ALL loose
  objects combined — both margins asserted in cfg): it opens only during an active,
  simultaneous, bridged press, and opening is irreversible. Opposite temporal
  structure (momentary event vs held state), plus tool selection and a ratchet.

## Scene (vault body frame; +x out the doorway; all procedural geometry)

- Roofed vault 26 × 26 × 19.2 cm, interior floor at z = 0.068, FULL roof at 0.180;
  doorway (|y| ≤ 0.052, z 0.068–0.180) in the +x wall is the only aperture.
- ORANGE gate plate on a prismatic +y slide (travel 0–0.168) with a linear drive
  whose target (0.30) sits beyond the open stop — it always wants to open. Two catch
  strips under the plate are blocked by the collars of two spring pins (prismatic z,
  preload 7 N, 10.4 N at full 24 mm press). Collar clears the strips at q ≤ −0.020.
- Pin press caps: 3 × 3 cm, tops at z 0.130, stations (x 0.162, y ∓0.085) — 17 cm
  apart. Blue press bar 26 × 3 × 2 cm (0.60 kg) bridges both; yellow rod (12 cm)
  cannot. At full gate travel the trailing strip stops short of the far pin, so the
  released pins pop up BETWEEN the strips: shoving the gate back re-closes it only to
  q ≈ 0.026 > the 0.020 "sealed" line (ratchet).
- Porch deck (x 0.144–0.344, top 0.072) in front of the doorway; deck → sill (0.070)
  → floor (0.068) step DOWN 2 mm each so a pushed can never catches a lip.
- RED can starts standing on the porch; GREEN decoy can, bar and rod on randomized
  arcs on the floor. Vault xy jitter ±5 cm + free 360° yaw; every object free-yaw.

## Rubric

- `success()`: RED can settled with center inside the interior window
  (|x| ≤ 0.085, |y| ≤ 0.085, z ∈ [0.080, 0.150]) — any orientation.
- `score()` (latched in `post_step` at sim rate, order-gated, non-decreasing):
  0.20 once both pins have been clear in the SAME substep; 0.50 once the gate has
  driven fully open AFTER that; 0.80 once the can enters after that; 1.0 iff
  `success()`. Null policy ~0. A teleported-open gate without a press earns 0.
- 33 `__post_init__` asserts audit the interlock honestly (seal, per-pin block,
  no snap-lock race, force budgets, bridge/decoy lengths, slide-before-tip push
  window, doorway-passes-can, window-strictly-interior).

## Solution phases (solve.py — teleports are TRANSPORT ONLY)

- P0: settle, layout readback, sealed-gate asserts (score 0).
- P1: bar teleported to hover over both caps; ~25 N pressed down on its middle via
  per-substep external force until the gate drives itself open (score 0.50).
- P2: bar teleported to a park spot clear of the porch lane; gate stays open.
- P3: red can pushed along the porch with a continuous kinetic force law
  (v_des 0.10 m/s, y-steering, cap 1.6 N < tip threshold) over the sill, through the
  doorway, to rest inside (score 0.80 → 1.0 on settle).
- P4 + persistence: streak-gated success, then 3.3 s hands-off hold.
- Verified on forge, seeds 0 and 1: `SIM_GEN_SOLVE: SUCCESS`, monotone
  `SIM_GEN_SCORE` 0 → 0.5 → 1.0.

## Franka feasibility (single arm, parallel jaw)

Base pose: ~0.55 m out along the vault's door axis (+x), facing the porch — caps,
porch and floor tools all inside a 0.85 m reach envelope (tools spawn at 0.42–0.55 m
from the vault on ±40–75° arcs).

- **Bar**: 3 × 2 cm cross-section — an easy parallel-jaw grasp at its middle; lay it
  across both caps (17 cm span vs 26 cm bar leaves ±4.5 cm placement slack), then
  press down on its middle with the closed gripper: ~25 N is well inside Franka's
  payload/force envelope, and the grippy caps stop the bar skating.
- **Can**: 6 cm diameter — pushable with the side of the closed gripper at CoM
  height (~0.74 N slides it, 1.77 N tips it: 2.4× window); the porch rails the push
  line toward the doorway. (Grasping the can is also possible but never needed.)
- **Rod / green can**: decoys — leave them.

## Ordering requirements (physically forced, not rubric-suggested)

1. The roof is full and the walls are closed: the ONLY way in is the doorway.
2. The doorway is sealed until the gate opens, and the gate opens ONLY during a
   simultaneous two-point press (each pin alone blocks it; sequential presses fail;
   parked weight fails; shoving the gate or the cargo fails) → unlock BEFORE
   delivery, with the bar, actively.
3. The bar must then leave the porch lane (it blocks the push line at the caps).
4. Score latches are order-gated in the same sequence, so partial credit cannot be
   collected out of order (open-gate-by-teleport earns 0).

## Checks (smoke.py — 23, verified on forge)

settle+seal/no-NaN ×2; randomization readback ×2 (vault pose; can/bar/rod/green);
null policy; authored-mass readback (7 bodies); roof denies the seed's top-drop;
pushed cargo cannot breach the closed door; single pin A; single pin B; sequential
(non-simultaneous) press fails; 5 N gate shove held; parked bar cannot unlock; short
rod falls between the stands; positive control (simultaneous press → self-open,
score 0.50, not success); ratchet refuses to re-seal then re-opens; doorway
near-miss rejected; acceptance construct (success TRUE); settle gate (kicked can
rejected); wrong object (green can inside) rejected; teleported-open gate earns no
credit; rejection audit; final no-NaN. Camera frames saved to `frames.npz`.
