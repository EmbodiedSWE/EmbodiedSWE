# libero_pick_butter_i282 — Butter Dispenser (stage the container, actuate the mechanism)

## Seed provenance

Derived from **`libero/libero_pick_butter`** (RoboVerse
`roboverse_pack/tasks/libero/libero_pick_butter.py`): *"Pick up the butter and
place it in the basket"* — a Franka picks the butter from among five grocery
distractors on a table, carries it through free space, and releases it over an
open basket; the checker is a relative-bbox containment test on the basket.

## What changed, and why it is strategically different

The seed's entire plan is one prehensile transport OF THE BUTTER into a passive
container, judged by final-state containment. This task keeps the cast (butter,
basket) and swaps every role:

- **The container is what gets carried; the butter is never grasped.** All butter
  blocks are 88 mm square in footprint — wider than the 80 mm Franka jaw span
  (asserted in `__post_init__`) — and live stacked inside an elevated gravity-fed
  **magazine tower**. The only way a block ever moves is through the tower's own
  mechanism. The basket, by contrast, has a thin graspable rim (9 mm, asserted)
  and must be **picked up, carried, and set down** on a green catch mat under the
  tower's outlet — the seed's transport skill applied to the *other* object.
- **A prismatic mechanism replaces the grasp.** The tower's pusher blade (driven
  via the red paddle at the back, on a bind-time USD prismatic rail) extrudes
  exactly the **bottom** block of the stack through a front outlet slot; the block
  free-falls off the elevated lip into whatever is (or is not) below. The
  front wall passes only the bottom block (slot 58 mm vs 50 mm block, and
  2·block_h exceeds the slot by 42 mm — retention asserted and physically tested
  under a 2× force / 3× speed shove in smoke).
- **Execution order is forced by irreversibility, not rubric fiat.** A block
  dispensed before the basket is staged lands on the bare ground; it cannot be
  grasped (too wide) and the magazine floor is 220 mm up, so it is forfeit. The
  rubric mirrors the physics: `delivered` only latches when an **outlet-transit
  credential** (block CoM crossing the slot volume with outward velocity) fires
  **while** the basket is currently staged.
- **Containment alone is worthless.** The seed's own strategy — drop the butter
  into the basket from above — reaches geometric containment and is rejected:
  no credential, no delivery (smoke check 5). The credential cannot be faked from
  above because the front wall fills the slot's xy for 140 mm above the slot
  window; a block dropped onto the lip tips outboard far outside the window
  before it descends into it (smoke check 9).
- **"Exactly one" replaces "the butter".** Success requires exactly one present
  block contained in the staged basket and every other present block still inside
  the tower; a pre-loaded/overfilled basket is rejected (smoke check 12).

Distinctness from the sibling butter derivative examined, **i172 Butter
Switchyard** (ordered non-prehensile routing of two captive blocks through a
single-width T-channel network, gravity delivery off a mouth, contamination
latch): i282 has no routing, no channel network, and no active obstruction —
its core is a **prehensile container-staging leg plus a mechanism-actuation
leg** (prismatic dispenser with forced dispense-order and exactly-one retention),
i.e. the manipulated object is a tool interface (red paddle), not the cargo.
Also distinct from i3 (rotary carousel interlock: align-then-lift of the cargo
itself), i43/i119/i104/i51/i6/i129/i151 (tow, tip-pour, rocker lock, pile
driver, ledge catch, ballast vault, flat-pack crate — none stages a container
under a fixed dispenser and none has a magazine-fed exactly-one mechanism).

## Scene (procedural, self-contained)

- **Tower** (kinematic compound spawner) at fixed (0.30, 0): recessed pedestal
  column + elevated chamber (floor top at 0.220 m, front edge = drop lip at
  local x −0.062) + magazine walls (top 0.430 m): front wall with a 58 mm outlet
  slot, back wall with a 48 mm blade slot, full-height sides.
- **Pusher blade** (dynamic 0.6 kg compound: blade 30×90×40 mm + rod through the
  back slot + tall RED paddle outside the tower), on a per-env bind-time USD
  prismatic joint (body0 = the never-moving kinematic tower; 100 mm stroke;
  collision-disabled pair; sleep/stabilization thresholds 0).
- **Butter blocks** (2–3 per episode, 88×88×50 mm, 0.15 kg, μ 0.30, restitution
  0): stacked in the magazine; absent blocks parked in a ground depot.
- **Basket** (dynamic 0.5 kg compound: 0.2 m square interior, 100 mm walls,
  9 mm rim) and a green **catch mat** (kinematic, 0.26 m) centered 85 mm outboard
  of the lip; the ejected block's landing window is inside the basket interior
  across the whole ±30 mm staging tolerance (asserted).
- **Randomization (readback-verified in smoke):** block count 2 or 3 (uniform
  comparisons after a burn-in draw — first-draw degeneracy), in-magazine jitter
  ±1.5 mm + yaw ±2°, basket side 50/50 at |y| ≈ 0.30, xy jitter ±40 mm, free yaw.

## Rubric (latched; null policy scores exactly 0)

`0.20·staged + 0.15·actuated + 0.35·delivered`, capped at 0.70 for non-success;
**1.0 iff `success()`**:

- `staged` — basket ever settled upright on the mat (±30 mm, ≤15°, on-mat z);
- `actuated` — blade ever past 60 % of its stroke;
- `delivered` — outlet-transit credential (CoM in the slot volume, outward
  velocity) fired **while** the basket was staged — the anti-shortcut gate;
- `success()` — current state: basket staged, **exactly one** present block
  contained in it, that block credentialed, `delivered` latched, every other
  present block still in the tower, everything settled.

## Teleport-solution outline (solve.py)

1. **STAGE** — one free-space teleport of the basket to 20 mm above the mat,
   upright; real gravity set-down must satisfy the staged gate (0.20);
2. **DISPENSE** — velocity-regulated bang-bang x-force on the blade (10 N cap,
   0.12–0.20 m/s), through 65 % (actuated, 0.35) then the full stroke: the bottom
   block is extruded under real contact (riders retained by the front wall) and
   free-falls off the lip into the basket (`delivered`, 0.70 → success 1.0);
3. **RETRACT** — same drive backward; the retained stack drops to the floor as
   the new bottom block;
4. persistence: ≥3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`
   only if `success()` still holds. `SIM_GEN_SCORE` printed at every phase
   boundary, non-decreasing (asserted).

Verified on the forge: seeds 0 (3 blocks, basket side −y), 1 (2 blocks) and
2 (2 blocks, side +y) all reach SUCCESS in ~36 s, trajectory
0.00 → 0.20 → 0.70 → 1.00.

## Embodiment argument (single Franka + OSC)

Two legs, one arm: (1) grasp the basket rim (9 mm wall < jaw span; the basket
spawns in the open at |y| ≈ 0.30, walls 100 mm tall, nothing above it), carry it
~0.35 m, and set it down on the mat — the mat sits in front of the tower under
open sky (the chamber overhang is 0.21 m up and the staged basket clears the
recessed pedestal column by ≥5 mm at every point of the tolerance, asserted);
(2) push the red paddle: it stands proud of the tower's back face at ~0.29 m
height, travel 100 mm ending 20 mm clear of the back wall, force needed ~2–4 N
against stack friction (the scene-level blade force is the stand-in for a
fingertip/knuckle press on the paddle). A base pose at (0.85, 0.0, 0) facing −x
puts the paddle at ~0.36 m reach and the mat center at ~0.70 m, both inside the
Franka envelope, with the whole approach corridor obstacle-free. Blocks never
need grasping (impossible by design, asserted); success criteria depend only on
object poses, never robot state.

## Execution-order declaration

**Stage before dispense** is enforced physically and in the rubric: a block
dispensed with no staged basket lands on the bare ground, is too wide to grasp,
and can never re-enter the elevated magazine — and `delivered` can only latch
during a staged transit, so a later staging plus hand repair keeps `success()`
False forever (smoke checks 6–7 prove both halves). The reverse order does not
exist: nothing about staging requires the blade, and dispensing first forfeits
the block.

## Checks

- `solve` (forge): `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1, 2 (2- and 3-block
  episodes, both basket sides), scores monotonic 0.00 → 0.20 → 0.70 → 1.00.
- `smoke` (forge): 13 named checks — settle/no-NaN, randomization readback,
  count+side+yaw coverage, null policy, seed-strategy hand drop-in rejected,
  unstaged dispense forfeits (order part 1), late-stage hand repair rejected
  (order part 2), exactly-one retention under an aggressive shove, over-wall
  drop cannot fake the credential, staged-gate near-misses (off-center / tilt /
  hover), staged credit survives regression, overfilled basket rejected, video —
  `SIM_GEN_SMOKE: ALL PASS 13/13` with frames.npz recorded.
