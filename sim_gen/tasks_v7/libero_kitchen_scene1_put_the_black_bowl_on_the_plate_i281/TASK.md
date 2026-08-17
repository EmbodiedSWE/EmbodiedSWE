# berry_pour — empty the black bowl onto the plate (i281)

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate` — a single
rigid pick-and-place: grasp the black bowl, set it down on the plate (`_terminated` =
bowl xy within 0.06 of the plate and 0 < dz < 0.03).

Kept from the seed: the two protagonist objects (a white dinner plate and the black
bowl) and the kitchen-tabletop setting. Everything else is new: the bowl now *holds
cargo* (3–6 small red berries, count sampled per seed), and the goal predicate is about
the berries, not the bowl.

## What the task is

Procedural scene (no external assets): a white rimmed plate (r = 0.115 m, 16-segment
raised rim, h = 0.014 m) rests on the floor around (0.38, 0.02) ± 0.030 m; the black
open bowl (inner r = 0.055 m, 8 mm wall, 60 mm tall) rests around (0.04, −0.04) ±
0.040 m with random yaw, holding every present berry (r = 11 mm, 8 g). Per-seed
randomization: plate xy, bowl xy + yaw, berry count k ∈ [3, 6] (all verified by
readback in smoke).

**Success** (all clauses, judged settled):

1. every present berry rests on the plate's flat top inside its rim
   (xy within `on_tol` = 0.100 of the plate center, resting z band, **and not inside
   the bowl** — computed in the bowl's body frame), and
2. the bowl stands upright (tilt ≤ 20°) on the **floor**, ≥ 0.20 m (xy) from the
   plate — fully off it, at its floor rest height, and
3. everything is settled (velocity gates above the GPU phantom-creep band).

**Score** (monotone, latched in `post_step`): 0.10 for ever lifting the bowl ≥ 5 cm +
0.55 × (max fraction of present berries simultaneously counted on the plate while
slow), clamped to 0.90; exactly 1.0 iff `success()`.

Rubric honesty notes:

- `on_tol` = 0.100 covers the rim-junction corner rest of the inner rim 16-gon
  ((apothem 0.105 − berry_r)/cos(π/16) ≈ 0.0958), so every genuine against-the-rim
  rest counts, while a berry perched ON the rim top (center xy ≥ apothem 0.105) does
  not. (With 12 segments the corner rest was 0.0973 > the old 0.095 tolerance — a berry
  creeping into a junction during the persistence window silently un-counted; found and
  fixed on the forge.)
- The in-bowl exclusion is load-bearing: it is what makes the seed's own end state
  (loaded bowl set down on the plate) score **zero** delivery credit.

## Why strategically different

- **vs the seed**: the bowl's role is inverted from *payload* to *tool*. The seed's
  entire goal (bowl on plate) is here a **rejected** state, refused independently by
  two clauses (in-bowl exclusion strips the cargo credit; bowl-clear requires the bowl
  on the floor ≥ 0.20 m away). The manipulation is granular content transfer — tilt
  past horizontal and pour — not a rigid transport, and the objects that matter (the
  berries) are never grasped at all.
- **vs i187 (beam balance)**: no hidden state, no measurement/inference; this is a
  dynamics-execution task (controlled pouring of free bodies).
- **vs i221 (cloche service)**: no enclosure/uncover–recover structure and no forced
  ordering beyond physics; the challenge is a continuous pour trajectory, not a
  blocker-reuse sequence.
- **vs pen_holder exemplar**: transfer *out of* a container by reorientation, the
  opposite of tip-up insertion *into* one; success counts a variable-size set of free
  bodies on an open surface, not captive contents.

## Solution outline (solve.py)

Teleport-hold plays the hand: the bowl is carried by gentle per-step
`write_root_state_to_sim` (~2–3 mm/step, zero velocity) so the free berries follow
**only through contact**; berries are never teleported toward the goal (the only berry
teleports are stray-returns back INTO the bowl, asserted un-scored at release).

1. **LIFT** — raise the bowl to z = 0.20, yaw to 0 (prints `SIM_GEN_SCORE 0.100`).
2. **CARRY** — to the pour stance beside the plate.
3. **POUR** — tilt 0 → 128° about the horizontal axis ẑ×u (u = bowl→plate), with the
   *lip* held fixed over the plate interior: root = lip_target − lip_offset(θ), lip
   height easing 0.14 → 0.065 m above the plate top so the lowest bowl point always
   clears the rim and any 2-berry pile.
4. **SHAKE** — ±6 mm at 3 Hz along the tilt axis until the bowl reads empty.
5. **UNTILT / HOME** — return upright, carry home, lower to rest + 3 mm, release.
6. **Retry** (≤ 3 cycles): any present berry neither counted nor in the bowl is
   returned into the bowl and re-poured.
7. Settle to success, print `SIM_GEN_SCORE 1.000`, keep simulating hands-off 420 steps
   (3.5 s), re-verify success, print `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seed 0 (4 berries, one pour cycle) and seed 1 (6 berries, one
pour cycle) both reach SUCCESS with monotone score prints and the 3.5 s persistence
window intact.

## Embodiment argument (Franka)

Base at ≈ (−0.15, 0, 0) facing +x: the bowl start (~0.06 m) and the pour stance beside
the plate (~0.40 m) both sit inside the ~0.75 m reach annulus at tabletop height.

- **Bowl**: the 8 mm open rim wall is a standard parallel-jaw pinch (jaw opening
  ≥ 20 mm ≫ wall; asserted in the config). Grasp the rim near the side facing the
  robot, lift vertically, then the 128° pour is a wrist-roll about the tool axis while
  the arm holds the lip over the plate — exactly the solve's lip-fixed trajectory. The
  6 mm / 3 Hz shake is a small Cartesian end-effector oscillation. Set-down is a
  plain place-and-release on the floor region between robot and plate.
- **Berries**: never require grasping — they leave the bowl under gravity and are
  retained on the plate by its rim. Strays (rare) are re-scooped by pushing them back
  over the bowl lip or re-poured; the solve's stray-return teleport stands in for a
  sweep-and-regather that a gripper can do with the bowl itself as a scoop.

## Execution-order declaration

No artificial ordering is imposed. The only order is physical necessity: the bowl must
be lifted and tilted for berries to leave it, and the final state requires the bowl
parked clear — a solver may pour first and park later, or in any interleaving that
ends in the success state. No gate latches an order-dependent credit.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 13/13` on the forge)

1. settle: finite states, all present berries in the bowl, bowl upright/clear, settled
2. settle: score 0 at reset, no success, no latches
3. mass readback: plate 1.2 / bowl 0.25 / berry 0.008 kg (custom spawners author MassAPI)
4. randomization readback (seeds 21–26): plate xy, bowl xy, bowl yaw move; berry count
   varies within [3, 6]
5. null policy: 240 idle steps → score ≤ 0.02, no success
6. deliver latch: two berries settled on the plate → exactly 0.55·2/k, no lift credit
7. no-evaporation: removing a delivered berry does not reduce the latched score
8. negative A (seed end state): loaded bowl ON the plate → 0 counted (in-bowl
   exclusion), bowl-clear fails, score ≤ 0.11
9. negative B: all berries on the plate but bowl resting on it too → bowl-clear rejects
10. negative C: berries on the plate, bowl lying on its side → upright rejects
11. negative D (near miss): one berry on the floor beside the plate → not all delivered
12. negative E (floor dump): berries tipped onto the floor by the bowl → 0 counted
13. rim-hug honesty: a berry resting against the rim's inner face still counts

frames.npz (254 × 600 × 960 × 3) recorded from the tabletop camera.
