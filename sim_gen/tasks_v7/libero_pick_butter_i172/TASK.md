# libero_pick_butter_i172 — Butter Switchyard (ordered non-prehensile routing)

## Seed provenance

Derived from **`libero/libero_pick_butter`** (RoboVerse
`roboverse_pack/tasks/libero/libero_pick_butter.py`): *"Pick up the butter and
place it in the basket"* — a Franka picks the butter from among five grocery
distractors on a table, carries it through free space, and releases it over an
open basket; the checker is a relative-bbox containment test on the basket.

## What changed, and why it is strategically different

The seed's entire plan is **one prehensile transport**: identify → grasp → carry →
release, with the distractors as passive scenery and containment judged as pure
final-state geometry. This task keeps the cast (butter, basket, one distractor) and
inverts every load-bearing element of that plan:

- **No grasp and no carry exist anywhere in the task.** Both blocks are 90 mm
  square in footprint — wider than the 80 mm Franka jaw span on every horizontal
  axis (asserted in `SceneCfg.__post_init__`) — and travel captive inside an
  open-top, walled T-channel network sunk into a raised switchyard deck. The walls
  (65 mm, taller than the 50 mm blocks) flank them besides. The only manipulation
  that exists is **pushing along the channels**; delivery into the basket is by
  **gravity** off a drop mouth, never by release from a gripper.
- **The distractor is an active obstruction with its own goal state.** The grey
  brick sits between the butter and the junction in a **single-block-wide** trunk
  (asserted: `chan_w < 2*block_s`), so the butter physically cannot pass it. The
  execution order is forced by geometry, not by the rubric text: **first** dispose
  of the brick out the mouth AWAY from the basket (it must end resting on the bare
  ground), **then** route the butter through the junction and out the basket-side
  mouth.
- **A routing decision replaces free-space transport.** The crossbar ends in two
  identical drop mouths; which one has the basket under it is sampled per episode
  (`swap_sides`), so the agent must read the scene and steer each block to a
  *different* exit.
- **Irreversible contamination replaces "distractors are scenery".** Pushing the
  brick out the basket-side mouth drops it INTO the basket — latched as permanent
  failure (`dirty`, score capped at 0.10 forever, even if the brick is later
  removed).
- **Containment alone is worthless.** The seed's own strategy — drop the butter
  into the basket from above — is explicitly rejected: `delivered` only latches
  while the butter is *descending* into the basket interior with the junction
  transit credential (`junction`) already earned. Vertical drop-ins and
  over-the-wall crossbar entries reach geometric containment but never score.

Strategic difference from the sibling derivatives examined (i43 cellar tow, i46
pudding sift, i119 silo tip-pour, i104 rocker lock, i3 milk carousel, i51 pile
driver, i6 ledge catch, i129 ballast vault, i151 flat-pack crate): none of them is
an *ordered two-body routing* problem through a shared single-width network with a
forced disposal-before-delivery order and a contamination hazard at the goal.

## Scene (procedural, self-contained)

- **Deck** (kinematic compound spawner): pedestal column + slab (channel floor at
  0.17 m) + six walls forming a trunk (rear inner wall at x = −0.46, deck frame)
  into a T-junction with a crossbar whose two open mouths are the slab's side edges
  at y = ±0.33. The slab overhangs the column so the ground under each mouth is
  open.
- **Blocks**: butter (yellow) and brick (grey), 0.09 × 0.09 × 0.05 m, 0.15 kg,
  μ = 0.30 (defined, so push forces are calibratable). Block diagonal (0.127 m)
  clears the 0.14 m channel — no wedge lock (asserted).
- **Basket** (dynamic compound spawner): open-top 0.30 m square, 0.10 m walls,
  on the ground under the sampled mouth, centre 0.055 m outboard of the deck edge;
  its interior brackets the mouth so the falling butter lands inside (asserted).
- **Randomization (readback-verified in smoke):** deck yaw ±8° + xy ±0.02; basket
  side 50/50; brick slot x ∈ [−0.25, −0.20]; butter slot x ∈ [−0.39, −0.365]
  (slot bands can never interpenetrate at worst-case yaw — asserted); slot y ±0.01
  and block yaw ±8°; basket xy ±0.015 + free yaw.

## Rubric (latched; null policy scores exactly 0)

`0.10·prog + 0.20·cleared + 0.20·junction + 0.35·delivered`, capped at 0.85 for
non-success; **1.0 iff `success()`**:

- `prog` — running max of the brick's route progress, normalized from its own
  spawn slot (so idling scores 0.00x);
- `cleared` — brick ever at rest on the bare ground (off deck, not in basket);
- `junction` — butter ever inside the T-junction box on the deck (the only
  connection between trunk and crossbar);
- `delivered` — butter ever descending (vz < −0.25) into the upright basket's
  interior **with `junction` already latched** — the anti-shortcut gate;
- `dirty` — brick ever inside the basket: score capped at 0.10 forever;
- `success()` — butter settled inside the upright grounded basket with
  `delivered`, brick settled on bare ground with `cleared`, never `dirty`,
  everything still.

## Teleport-solution outline (solve.py — zero teleports)

All four legs are contact dynamics — velocity-regulated horizontal CoM forces
(≤ 2.0 N, under the 2.65 N tipping bound), aimed at deck-frame target points so
wall deflections self-correct; the force-frame encoding is probed at runtime
(some pods rotate `is_global=True` wrenches by rotation-since-reset):

1. **P1** push the brick along the trunk into the junction (walls guide it);
2. **P2** push it down the crossbar away from the basket; force cut the instant its
   CoM crosses the deck edge → gravity grounds it (`cleared`);
3. **P3** push the butter along the now-open trunk through the junction
   (`junction`);
4. **P4** push it down the crossbar toward the basket; force cut at the edge →
   free fall into the basket (`delivered` fires while descending) → settle;
5. **P5** hands-off ≥ 3.3 simulated seconds, then `SIM_GEN_SOLVE: SUCCESS` only if
   `success()` still holds. `SIM_GEN_SCORE` printed at every phase boundary,
   non-decreasing (verified by asserts).

Verified on the forge: seeds 0, 1 (side −1) and 2 (side +1 — mirrored disposal and
delivery) all reach SUCCESS in ~41 s.

## Embodiment argument (single Franka + OSC)

Intended strategy per object: reach a **fingertip into the open channel top** (the
channels are open-topped; walls are only 65 mm above the channel floor, which is at
0.17 m — well inside Franka's dexterous envelope) and push the block's rear face,
exactly what the scene-level CoM pushes stand in for. Push forces needed are ~0.5–2 N
against a 0.44 N friction load — trivial for the arm. A base pose at
(−0.35, 0, 0) facing +x puts the trunk rear at ~0.32 m and the far mouths at
~0.85 m reach along a raised, obstacle-free corridor; the whole channel top is
approachable from above at 45°. Nothing requires grasping (impossible by design:
90 mm > 80 mm jaw span, asserted), two-handed manipulation, or reaching under the
deck. The success criteria depend only on block/basket poses, never on robot state.

## Execution-order declaration

The order **brick-disposal → butter-delivery** is physically forced: the trunk is
single-block-wide and the brick spawns between the butter and the junction, so no
butter route to any mouth exists while the brick is in the trunk. The reverse
order is impossible, and disposing of the brick out the basket-side mouth
(`dirty`) is latched permanent failure.

## Checks

- `solve` (forge): `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1, 2 (both basket sides),
  scores monotonic 0.00 → 0.05 → 0.30 → 0.50 → 1.00.
- `smoke` (forge): 14 named checks — settle/no-NaN ×2, randomization readback ×2,
  null policy, seed-strategy drop-in rejected, crossbar-shortcut rejected,
  wrong-mouth exit worthless, contamination caps below earned credit,
  contamination irreversible, brick-clause isolation (legit delivery still
  rejected), latched credit survives regression, rejection audit (success never
  True in the battery), final no-NaN — `SIM_GEN_SMOKE: ALL PASS 14/14` with
  frames.npz recorded.
