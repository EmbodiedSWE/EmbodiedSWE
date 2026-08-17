# libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i242 — ballast the deadman stove, then seat the pan

**Env:** `simgen.ballast_stove` (scene `ballast_stove`, robot `null`).

## Provenance

Seed: `libero_90/libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it`
— grasp the stove knob, rotate its revolute joint past a threshold
(`joint_pos > 0.5`), then set the frypan down on a flat, always-available cook
plate.

## Strategic difference from the seed

The seed is *grasp-and-rotate a fixture joint* + *free placement onto a static
surface*. This task keeps the "turn on the stove, then put the pan on it" story
but replaces both mechanisms:

- **Nothing is grasped-and-rotated; no fixture joint is commanded.** The stove
  is a ROCKER — a 440 mm beam on a pedestal pivot with two joint-limit hard
  stops. The east arm carries the fenced burner **trivet** over the gas dish;
  the west arm carries an open-top **ballast tray**. The beam's authored CoM
  bias (0.9 kg at +85 mm) parks the empty rocker trivet-DOWN at +16° — the
  deadman gas valve is shut. "Turning it on" means **loading a counterweight**:
  dropping the 1.4 kg iron ingot into the tray out-torques the bias and pure
  gravity swings the beam onto its LEVEL stop (−0.5°), opening the valve.
- **The discrimination is by MASS, read only through the mechanism.** A pale
  wood decoy block of the *same 60×60×50 mm size* (90 g, 15× lighter) spawns
  beside the ingot (slot sides coin-swapped per episode). It can never level
  the rocker — τ margins are asserted in cfg (decoy ≤ 0.35× bias; ingot ≥
  1.3× (bias + pan)) and the inert decoy is proven live in smoke.
- **The "on" state is a LIVE deadman, not a latch.** `success()` requires the
  beam level AND the ingot *still in the tray* AND the pan seated — lift the
  ballast out and gravity re-tilts the trivet, and success collapses the same
  step (demonstrated in smoke check 11, with the pan riding the re-tilt swing).
- **The cook surface itself moves.** The pan is judged in the BEAM frame: the
  seed-naive plan (just put the pan "on the stove") leaves it seated on a
  tilted, unlit trivet — 0.15 partial credit, never success.

Distinct within the corpus: same-seed sibling **i150** is a two-body rail
shunt to a self-aligning end stop (prismatic joints, no counterweight, no
mass reasoning). Among the ballast-family tasks, the closest is **i5**
(`counterweight_shelf`: cube in a socket levels a tipping shelf, then a bowl
is placed) — but i5 has a single counterweight and no decoy, judges the bowl
only on the *level* shelf, and never demonstrates unloading; here the
identical-size mass decoy refutes the "any block works" plan, the placement
target rides the *opposite arm* of the same rocker (seat credit is available
at any beam pitch), and the deadman cut — success collapsing live when the
ballast leaves — is an explicit smoke check. The weighing tasks (i160, i175,
i189) identify or sum masses on a balance; nothing here is weighed or served,
and the mechanism is a means, not the goal readout.

## Scene

Fully procedural (native PhysX box colliders; the rocker rides a spawn-authored
per-env revolute joint about Y whose LIMITS are the two hard stops; the
kinematic bench is the joint anchor and never moves):

- **bench** (kinematic compound): 800×500×160 mm counter, pivot pedestal
  (82 mm — asserted to clear the tilting bar's swept underside), glowing gas
  dish (130 mm, collider ON) under the trivet arm, flame flicker (visual only,
  swept clear of the tilted trivet — asserted).
- **beam** (dynamic, on pivot at z = 0.27): bar 440×30×20 mm; east arm
  (+185 mm) = 150 mm trivet plate with a 16 mm rim fence; west arm = 130 mm
  tray with 35 mm walls. Authored mass 0.90 kg, CoM (+85, 0, +5) mm, diagonal
  inertia, ang damping 20 (gentle stop-slams). Empty rest: +16° tilt stop.
- **ingot** (dynamic, free): rust-red iron 60×60×50 mm, **1.40 kg**.
- **decoy** (dynamic, free): pale wood, same size, **0.09 kg**.
- **pan** (dynamic, free): 90 mm square dish + 140 mm stick handle, 0.30 kg,
  spawns upright on the south apron, handle south at any sampled yaw
  (asserted).

Randomization (readback-verified in smoke): ingot/decoy slot sides via a
`torch.rand` coin at ±0.26 m, ±30 mm xy jitter each; pan xy jitter
(±70/±30 mm) and free yaw (−90°±60°).

18 honesty asserts in `SceneCfg.__post_init__` pin the claims: the three
torque margins; fence/tray walls box their objects inside the judged xy
tolerances; lip/wall perches read outside the z bands; spawn zones clear the
rocker sweep, the counter edge, and each other; the seated pan's handle clears
the fence; the pedestal clears the bar sweep; the tilted trivet clears the
dish and flame.

## Rubric

Latched calm-gated partial credit, capped at 0.55 unless success holds live:

| credit | clause |
|---|---|
| 0.20 | ingot ever in the ballast tray, calm (latched, beam frame) |
| 0.20 | beam ever at its level stop, calm (latched) — \|pitch\| ≤ 3° |
| 0.15 | pan ever seated on the trivet, calm (latched, beam frame: xy ≤ 30 mm, z ≤ 10 mm, upright within 12° of the BEAM up-axis — so seat credit exists even on the tilted trivet) |
| 1.00 | `success()` live |

`success()` = beam level AND ingot in the tray AND pan seated on the trivet
AND everything settled (|v| < 0.05 m/s, beam |ω| < 0.10 rad/s) AND finite.
Because the ingot term is live, success is a deadman: it certifies the stove
is *being held on*, not that it once was.

## Solution outline (solve.py — teleport-transport contract)

All load-bearing interaction is gravity + contact; teleports transport only,
and every hover pose is asserted to satisfy nothing (ingot hover z-offset
55 mm > tray band hi 50 mm; pan hover above the seat z band):

- **P0** settle 60 steps; assert the beam rests inside 1.2° of the +16° tilt
  stop (pins the forge-verified joint sign convention), ingot and decoy in
  opposite slots, pan upright on the apron, score ≤ 0.03.
- **P1 BALLAST** — teleport the ingot to a hover over the tray (inside the
  wall funnel, above the judged band) and release: it drops in by contact and
  its gravity torque swings the beam ~17° onto the level stop; hands-off until
  level + settled, then 90 extra steps → tray + level latches → score ≥ 0.39.
- **P2 PAN** — teleport the pan (handle south) to 20 mm above the trivet seat
  and let it drop between the fence lips; 3-attempt re-drop retry → success →
  score 1.0.
- **P3** ≥ 3.3 s fully hands-off persistence (10 × 40 steps, success at every
  checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` prints non-decreasing at every phase boundary
(0.00 → 0.40 → 1.00). Verified on the forge for seeds 0 and 1
(`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

Base at ~(0, −0.55, 0) facing +y; all contacts lie 0.16–0.31 m high inside an
80×50 cm counter — comfortable reach.

- **Ingot (1.40 kg, 60 mm cube)**: top-down pinch across two opposite faces —
  fits the 80 mm jaw. Holding 1.4 kg needs ~mg/(2μ) ≈ 12 N grip at μ≈0.6,
  far inside the Franka gripper's 70 N continuous force. Carry ~25 cm and
  release ~10 mm above the tray mouth; the 114 mm tray window vs the 60 mm
  block leaves ±27 mm drop tolerance, and the walls funnel the drop. The
  robot then simply waits — the swing is the mechanism's job.
- **Pan (0.30 kg)**: the 140 mm stick handle (20 mm square section) is a
  canonical pinch grasp; carry and lower onto the trivet; the fence window
  gives ±22 mm placement slack and self-boxes the pan into the judged xy
  tolerance (asserted).
- **Decoy**: requires no interaction at all — the correct plan leaves it
  untouched.
- No contact under an overhang, through an aperture, or at sub-centimeter
  tolerance; the finest requirement is the ±22 mm pan drop.

## Execution order

No order is code-enforced, and physics permits both: the cfg torque assert
guarantees the ingot levels the beam even with the pan already on the trivet,
and smoke's deadman cut shows the pan riding a full re-tilt swing without
unseating. The demonstrated solve uses ballast-first (level the stove, then
seat the pan) as the natural reading of "turn it on, then cook". Declared
module order:

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the counterweight
   cascade and the monotone score trace 0.00 → 0.40 → 1.00.
2. `smoke.py` — rejection battery, 13 checks:
   1. settle/no-NaN (beam on the tilt stop, blocks in opposite slots, pan
      upright on the apron, score 0);
   2. randomization readback (ingot/decoy/pan xy and pan yaw differ across
      seeds);
   3. slot coin: the ingot appears in BOTH apron slots over 10 resets;
   4. null policy 240 steps (score ≤ 0.01, no creep off the tilt stop);
   5. SEED-NAIVE: pan seated on the TILTED unlit trivet → beam-frame seat
      credit only (positive control), 0.149 ≤ score ≤ 0.151, no success;
   6. DECOY: the wood block dropped in the tray cannot move the beam off its
      tilt stop; the ingot-specific tray clause gives nothing → score ≤ 0.01;
   7. SWAP (ingot on the trivet, pan in the tray): the beam stays tipped (the
      ingot reinforces the bias) → score ≤ 0.01;
   8. fence-perch: pan dropped 65 mm off-center rests half over the rim fence
      → xy/upright windows refuse → ≤ 0.401;
   9. settle gate: the exact success geometry written with 0.40 m/s velocity
      is refused while moving (dismantled before it can calm);
   10. MECHANISM: the ingot released ABOVE the judged tray band (hover
       asserted to satisfy nothing) rotates the beam > 14° onto the level
       stop by pure gravity + contact → score 0.40, no success;
   11. DEADMAN: legitimate success reached live (score 1.0), then the ingot
       is lifted out → success collapses immediately, the beam re-tilts onto
       its stop, score falls back to the latched 0.55 (float32-exact band);
   12. audit: success() was never True outside the sanctioned deadman window;
   13. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS 13/13`) and hard-exit via a watchdog Timer +
`os._exit`; a top-level try/except prints a FAIL verdict on any exception so
the forge watchdog never hangs.
