# libero_pick_milk_i359 — Milk Shunt Yard

## Provenance

Seed task: `libero/libero_pick_milk` — pick the milk carton up among distractors and
place it in the basket (grasp → carry → release).

This task keeps the seed's cast (a milk container, a distractor object, a delivery
basket) and its end state (milk resting inside the basket) but replaces the entire
strategy with **push-only rail-yard routing**: an elevated walled deck where the milk
crate can only be slid along lanes, a blocker that must first be shunted into a siding
pocket, and a randomized binary GREEN/RED aperture choice where the wrong hole is a
terminal trap. Success requires an **aperture-transit credential** — the crate's CoM
crossing the goal aperture plane moving downward — so placing the crate into the basket
by any route other than a genuine fall through the green hole never counts.

## Strategic difference

- **vs the seed (grasp-carry-release):** nothing is ever grasped or carried. The milk
  crate (100 mm cube) exceeds the Franka jaw span; the deliverable motion is planar
  CoM pushing plus a gravity drop. The smoke battery's flagship negative hand-lowers
  the crate into the basket from the highest pose that fits under the deck: containment
  and `outcome_ok` are genuinely satisfied, yet the credential never fires and score
  stays 0.
- **vs `libero_pick_milk_i3` (milk carousel):** i3 is ride-delivery on a revolute
  carrier under a canopy gate. Here there are no joints at all — the mechanism is pure
  static geometry (lanes, a siding, floor apertures, enclosed cells) plus rigid-body
  routing with an irreversible branch choice.
- **vs `libero_pick_butter_i51` (pile driver):** no impact/momentum transfer, no
  striking tool; quasi-static regulated pushes only.
- **vs `libero_pick_butter_i282` (butter dispenser):** i282 is a stage-then-actuate
  machine (blade stroke through a magazine, exactly-one retention). Here nothing is
  actuated: the obstacle itself is task cargo that must be re-routed (shunted) out of
  the way, and the core decision is a spatial commit (green vs red side) whose wrong
  branch is physically unrecoverable, not an ordering of machine operations.
- **vs the pen_holder exemplar:** no insertion/containment-by-placement; the container
  is unreachable (sealed under-deck cell) and can only be reached ballistically
  through the deck aperture.

## Layout (yard-local frame; yard root at world (0.42, 0), jittered ±30 mm, yaw ±8°)

Elevated deck (top at z=0.200) carrying 55 mm walls that form:

- a south spawn **BAY** (x∈±0.080, y∈[−0.260, −0.100]) holding the white milk crate;
- an east–west **CROSS-LANE** (x∈±0.345, y∈[−0.100, 0.060]) whose junction with the
  bay is gated by the orange **BLOCKER** (76 mm cube);
- a narrow **SIDING** pocket straight north (94 mm wide, funnel-chamfered mouth) that
  admits the blocker but refuses the 100 mm crate;
- two floor **APERTURES** (150 × 160 mm) at the lane ends. Under one (marked by the
  GREEN tab beside it; side randomized per episode) an enclosed under-deck cell holds
  the delivery basket (170 mm interior); under the other (RED tab) is a bare walled
  trap cell. Both cells are sealed: walls to the deck, deck roof, the aperture as the
  only opening.

## Solution outline (solve.py — the legitimacy certificate)

Nothing is teleported after reset; every phase is a velocity-regulated planar CoM push
(the same external-wrench channel a fingertip on the proud crate tops exercises) with
lateral P-steering, plus gravity:

1. **SHUNT** — push the blocker straight north out of the junction into the siding
   pocket and park it past the shunt line (latched +0.15).
2. **ENTER** — push the crate north out of the bay into the cross-lane.
3. **COMMIT** — push the crate along the lane toward the green-tab side past the
   commit line (latched +0.20).
4. **DELIVER** — keep pushing to the aperture edge and cut the force the moment the
   crate leaves the deck; it tips through the green aperture (CoM crosses the aperture
   plane moving down → transit credential, latched +0.30) and free-falls into the
   basket. Real contact landing.
5. Success = credential ∧ crate settled inside the upright basket ∧ blocker not in the
   basket → score 1.0; verified to persist 3.3 simulated seconds hands-off.

Verified on the forge, seeds 0/1/2 (both goal sides, yard yaw up to +7.6°), score
trajectory [0.0, 0.15, 0.15, 0.35, 1.0, 1.0], ~29 s each.

## Embodiment argument (Franka, 80 mm parallel jaw)

Base at (0, 0), facing the yard centered at (0.42, 0).

- **Milk crate (100 mm cube, 0.35 kg):** exceeds the jaw span → ungraspable,
  push-only. Its top face stands ≥ 30 mm proud of the 55 mm lane walls, so a fingertip
  or closed-jaw knuckle can push it anywhere along its route.
- **Blocker (76 mm cube):** marginally graspable, but grasping it is never necessary
  (a straight push parks it in the siding) and never sufficient (moving the blocker
  does not move the crate; dropping it into the basket fails the task). Top ≥ 20 mm
  proud of the walls for the same fingertip push.
- **Basket:** unreachable — sealed inside the under-deck cell; it can only receive the
  crate through the aperture. Marker tabs are visual-only (no colliders).
- **Reach:** the farthest contact the solution requires (crate south face at spawn /
  aperture-edge push) is ≈ 0.71 m from the base, inside the 0.78 m envelope (asserted
  in `__post_init__` at worst-case jitter).

## Execution order

No strict order is imposed a priori — but physics forces shunt-before-enter (the
blocker fully gates the bay mouth; asserted no slip-past corridor), and two branches
are terminal: the crate through the RED aperture (enclosed trap, crate ungraspable →
unrecoverable, `_trapped` latched) and the blocker into the basket (occupies the goal
container → success permanently false). The rubric's stage latches are monotone;
success itself is a current-state conjunction gated by the latched credential.

## Checks (smoke.py — forge: `SIM_GEN_SMOKE: ALL PASS 11/11`)

1. settle/no-NaN + layout sanity (crate in bay, blocker at junction, basket under the
   green aperture), score 0;
2. randomization readback real (3-seed max-pairwise yard xy / yard yaw / crate yaw);
3. both goal sides drawn over 10 resets + real yaw spread;
4. null policy ≈ 0, no success;
5. **seed-strategy rejection:** hand-lowered drop into the basket is contained
   (`outcome_ok` true) but earns no credential → score 0, no success;
6. **trap terminality:** fall through the red aperture latches `_trapped`, no credit;
7. **blocker hazard:** blocker in basket first, then a genuinely credentialed crate
   delivery → success false, score capped at 0.50;
8. **siding refusal (physical, anti-vacuous):** the crate shoved north at 2× solve
   force engages the mouth (readback ≥ 15 mm travel) and jams far short of the shunt
   line;
9. near-miss gates: hover above the basket ceiling rejected; settled containment
   accepted; 25° tilted basket rejected;
10. latch persistence + regression beyond the latch (commit credit survives carrying
    the crate back to the bay);
11. frames.npz video captured (111 frames, 960×600).
