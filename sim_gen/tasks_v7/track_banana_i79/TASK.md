# track_banana_i79 — Clothespin Banana Harvest (`simgen.banana_line`)

## Seed provenance

Derived from **pick_place/track_banana**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_banana.py`): a Stage-3
trajectory-tracking task — the banana starts ALREADY RIGIDLY GRASPED in the Franka's
closed gripper, and reward is dense tracking of a prescribed free-space waypoint path
(position + orientation gates along the path). The seed's whole strategy is *transport
of a held object along given waypoints*; there is no grasp decision, no contact event,
no failure mode beyond deviating from the path.

## What changed, and why it is strategically different

Kept only the protagonist (a banana that must end up somewhere specific). Every element
of the seed's strategy is inverted:

- **The judged bananas can never be held or pulled.** Each hangs from a spring-loaded
  clothespin CLAMP by a stem knob that is form-closed in the pinch slot: the knob plate
  cannot pass the 14 mm closed gap, a roof plate blocks lifting it out, knob-level curbs
  + a rear stop cage it fore/aft, and touching tip nubs block the stem. Smoke check 6
  *constructs* the seed's strategy — 12 N pulls in ±x, ±y, +z (≈ 10× the banana's
  weight) — and shows visible disturbance (up to 31 mm) with re-capture every time.
- **The only way to free a banana is to actuate a mechanism**: squeezing the clamp's
  two tail paddles (a native parallel-jaw close) scissors the pads apart until the knob
  falls STRAIGHT DOWN by gravity. Nothing is tracked; the banana is airborne and
  uncontrolled exactly when the seed would be holding it tightest.
- **Transport happens to a different object**: the catch crate is staged under a clamp
  BEFORE opening it, then pushed along the floor to the depot — floor pushing with
  routing around the gantry base, not free-space waypoints.
- **Irreversible failure and a per-episode selection decision**: a banana that lands
  anywhere but inside the crate is BRUISED (latched, fatal), and the GREEN banana —
  whose station is randomized — must be left hanging. The seed has neither.
- **Versus the read corpus:** no existing task frees an object by actuating a
  spring-loaded release mechanism above a staged catcher. `flap_chute_pantry` (i33) has
  a passive one-way flap, `latch_canister` (i54) turns a latch on a container the robot
  then opens directly, the LIBERO drawer/microwave tasks articulate the container
  itself. The catch-before-release ordering (stage → squeeze → gravity catch), the
  spring-loaded scissor-pair plant, and the bruise-fatal drop rubric appear nowhere in
  tasks_v4–v7.

## Apparatus (fully procedural, no meshes)

Heavy DYNAMIC gantry (28 kg — never kinematic: spawn-authored joint anchors on this
stack stay world-fixed if body0 is a teleported kinematic; base slab, posts, beam, and
per station: bracket-plate roof, two axle pins, a rear stop, and two knob-level CURBS
above the lever sweep). Six red lever arms (0.60 kg each) on spawn-authored vertical
revolute joints; a `post_step` plant applies spring (5 N·m/rad) + damping + a 0.5 N·m
closing PRELOAD per lever (a pre-tensioned clothespin — an impulsive yank cannot
ratchet the slick pads open), plus the clamped `lever_drive` actuation buffer. Three
bananas (two yellow, one green) hang by 8 mm stems ending in flat 34 × 34 × 8 mm 8-gon
knob PLATES (flat-on-flat defeats snatch camming; a single lever at its 20° hard stop
opens 32.1 mm < 34 mm, asserted). Blue crate (210 × 150 mm interior, 85 mm walls),
magenta depot pad (no collider — judged by xy). Slot/pad/knob faces carry a slick
physics material: captivity is FORM closure, never friction. Config honesty is
asserted in `__post_init__` (knob passes the open slot at worst yaw but never the
closed one, roof/curb clearances, spring holds the declared yank with 1.4× margin
while the drive still reaches the release angle, tail span fits the Franka jaw, crate
passes under the hang line, depot admits the crate).

**Randomization (readback-verified):** gantry xy ±3 cm + yaw ±20°, green banana's
station (0/1/2) with yellows shuffled, per-banana hang yaw ±180° + stem-slot jitter,
crate xy ±5 cm + free yaw, depot xy ±5 cm.

## Rubric

- `success()` = both YELLOW bananas inside the crate (crate-frame containment, below
  rim) ∧ crate centred on the depot (≤ 5.5 cm) ∧ GREEN banana still HANGING at its
  clamp ∧ all six levers re-closed (< 4°) ∧ no banana ever bruised ∧ everything still.
- `score()` (monotonic, latched): 0.25 per yellow ever settled in the crate + 0.20 for
  the delivery configuration ever held, capped at 0.70; exactly 1.0 iff success().
  Bruising is fatal to success but latched credit never evaporates.

## Solution outline (solve.py — the legitimacy certificate)

Teleport = TRANSPORT ONLY: one pose write moves the EMPTY crate across open floor to a
hover under the first yellow clamp (satisfies no gate — asserted). Everything
load-bearing is contact/mechanism: each release is a ramped pair of opposed z-torques
(≤ 2.2 N·m) on that station's levers through `lever_drive` — the squeeze analog —
against the spring until the pads part and the banana falls by gravity into the crate;
crate moves are a velocity-servoed planar floor drag (≤ 8 N, gain-escalating on
friction stall, force-frame probe against wrench drag) with BFS routing around the
gantry slab. Phases: P0 settle → P1 stage (teleport, empty) → P2 squeeze A, catch
(0.25) → P3 drag under clamp B (contact) → P4 squeeze B, catch (0.50) → P5 drag to
depot (1.0) → P6 hands-off persistence 3.3 s → `SIM_GEN_SOLVE: SUCCESS`. Verified on
the forge for seeds **0** (39.5 s) and **1** (43.5 s), scores non-decreasing
0 → 0.25 → 0.50 → 1.0.

## Embodiment argument (Franka, one base pose)

Base at the origin facing +x/+y. Farthest work point is the outer station's tail pair
(~0.73 m at z 0.28 with worst jitter) — inside the 0.855 m reach envelope; crate and
depot work is at 0.2–0.5 m. Per-object contact strategy:

- **Clamp tails (the release):** the closed tail-paddle pair spans 64 mm — inside the
  80 mm Franka jaw stroke (asserted), so squeezing them is a NATIVE parallel-jaw
  close. Reaching the 17.5° release angle needs ≈ 2.0 N·m per lever = ≈ 45 N at the
  45 mm tail arm, within the Franka gripper's 70 N continuous grasp force. Tails
  stick out on the open −y side of the gantry, approach is horizontal from behind,
  and the 220 mm station pitch leaves the neighbours clear.
- **Crate:** pushed along the floor with the closed gripper against its 85 mm walls
  (0.6 kg, ≤ 8 N demonstrated); the crate passes under the hang line with 3 cm+
  clearance (asserted), so it can be staged fully before any release.
- **Bananas:** never touched while hanging (12 N pulls provably fail); once in the
  crate they ride it.
- **Gantry/depot:** furniture; no contact required.

## Execution-order declaration

The physics enforces the only order that matters: the crate MUST be under a clamp
before that clamp is opened (a banana dropped anywhere else is irreversibly bruised —
smoke 7/13), and delivery only latches with both yellows already aboard. The two
catches may happen in either order; solve.py picks station A then B.

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 16/16` on forge)

1. Settle/no-NaN: three bananas hanging, crate on the floor, levers at 0°, all still.
2. Score ~0 at reset, no success.
3. Randomization readback: gantry xy + yaw vary within the declared band.
4. Randomization readback: green station takes ≥ 2 values; crate/depot jitter real.
5. Null policy (240 steps): score ~0, no success.
6. **Seed-strategy analog (captivity):** 12 N pulls in 5 directions visibly disturb
   the hanging banana (up to 31 mm, lever telemetry printed) yet it re-settles
   HANGING every time; score ~0.
7. Release control (positive control of the mechanism): driving the tails opens
   16.8° ≥ 10°, the banana falls; on the open floor it latches a BRUISE; the spring
   re-closes the clamp; score stays ~0.
8. Empty delivery: crate at the depot without bananas — no delivery latch, score ~0.
9. Green harvested too: all three in the delivered crate — refused, capped 0.70.
10. Rim height: a banana at rim height over the crate is NOT contained; the same xy
    inside the volume IS (positive control).
11. Depot near-miss (10 cm): no delivery latch, score exactly the 0.50 of the two
    in-crate latches.
12. Latched credit survives removing both yellows from the crate.
13. Bruise is fatal: the otherwise-perfect final configuration reached after a bruise
    — every other conjunct verified True — success still False, score ≤ 0.70.
14. Settle gate: the perfect configuration sliding at 0.46 m/s is refused.
15. Rejection audit: success() never True at any judged point in the battery.
16. Final no-NaN; frames.npz saved.
