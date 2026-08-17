# track_banana_i240 — Banana Kiln Ram-Feed (`simgen.banana_kiln`)

## Seed provenance

Derived from **pick_place/track_banana**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_banana.py`): a Stage-3
trajectory-tracking task — the banana starts ALREADY RIGIDLY GRASPED in the Franka's
closed gripper, and reward is dense tracking of a prescribed free-space waypoint path
(position + orientation gates along the path). The seed's whole strategy is *carrying
a held object along given waypoints through open space*; the hand and the banana go
everywhere together, and the goal region is wherever the path ends.

## What changed, and why it is strategically different

Kept only the protagonist (a banana that must end up somewhere specific). The seed's
strategy is made *physically impossible* and replaced by an indirect, tool-mediated
one:

- **The goal region is unreachable by anything held.** The banana must end up on the
  floor of a fully roofed, fully walled kiln chamber whose ONLY opening is a low,
  deep feed tunnel (18 cm wide, 7.5 cm tall, 21.5 cm deep). No carried object and no
  gripper can pass it. Smoke check 6 *constructs* the seed's strategy — a banana
  pressed down onto the goal with 6 N (≈ 5× its weight) — and shows it riding the
  roof (support height measured), never reading inside.
- **Ballistic delivery is excluded too**: the channel floor carries a gritty
  high-friction material — a 2 m/s shuffleboard flick (launch verified by velocity
  readback) tumbles to a stop within centimetres (smoke 7). Only a *sustained push*
  crosses the tunnel.
- **The only way in is a TOOL that lives in the scene**: the channel's rammer — a
  wide pusher head (170 mm, nearly the full 180 mm channel: nothing slips past it)
  on a 42 cm handle ending in a graspable 22 mm knob. The robot works the knob,
  OUTSIDE the kiln, while the head bulldozes the banana down the channel, through
  the covered tunnel (working blind past the mouth), and out onto the chamber floor.
- **Mandatory un-doing**: success also requires WITHDRAWING the rammer clear of the
  tunnel afterwards — the seed has no analog of "your instrument must leave the
  goal". A banana perched on the parked head is rejected by a z gate (smoke 10),
  and delivery with the head still inside is refused (smoke 11).
- **Nothing shared with the seed's code shape**: no waypoints, no path tracking, no
  held object — the deliverable travels its last 40+ cm under pure contact (pusher
  face + floor friction + wall guidance) while the hand stays at the knob.
- **Versus the read corpus, especially sibling i79 (same seed):** i79 frees bananas
  from spring-loaded clothespin clamps (actuate-a-mechanism release, gravity catch
  into a staged crate, floor-drag delivery); here there is no mechanism, no joint
  anywhere in the task, no catch — the strategy axis is *reach extension through a
  scene tool into a space the arm cannot enter*, plus mandatory tool retraction.
  No task in tasks_v4–v7 delivers an object by pushing it with a scene-provided ram
  through a covered passage: `flap_chute_pantry` (i33) drops through a passive flap,
  `latch_canister` (i54) unlocks a container the robot then opens directly, the
  LIBERO drawer/microwave tasks articulate the container itself. The
  place → ram-in → withdraw ordering and the roofed-goal carry-exclusion appear
  nowhere in the corpus.

## Apparatus (fully procedural, no meshes, no joints)

Heavy DYNAMIC kiln (30 kg, one compound rigid body — teleport-safe for pose
randomization): floor slab, open-top dock channel (40 cm long, 5 cm side walls),
covered tunnel (interior 180 × 75 mm, 21.5 cm deep), roofed chamber (interior
240 mm wide × 165 mm deep × 120 mm tall) with jambs, header, far wall and roof.
Kiln-local frame: origin at the tunnel mouth on the ground, +x = feed. The channel
floor is GRIT (μ 0.90/0.85 — kills flicks); every wall the banana or head can brush
(dock, tunnel, jambs, header, chamber, far) is GLAZED slick (μ 0.04/0.03 — a
brushing banana slides instead of friction-locking; captivity of the goal is pure
geometry). Rammer (0.40 kg, CoM authored low at z −0.015 so pushing above the floor
contact cannot tip it): blue head 50 × 170 × 50 mm, thin handle, red knob
22 × 22 × 75 mm whose bottom clears the floor. Banana (0.12 kg): three yellow
angled segments (± 22°, 138 mm flat span — rocks flat, never rolls) + stem nub,
peel-friction material. Config honesty is asserted in `__post_init__`: the banana
passes every aperture flat at any yaw and can wedge nowhere; the head fills the
channel within 4–14 mm; at full insertion the pusher face reaches the sill while
the head's REAR is still inside the tunnel-wall guidance span (so it can always be
pulled straight back out — no yaw-hook self-lock) and the knob is still 180+ mm
outside the mouth; a head-perched banana provably fails the rest-z gate; the rammer
starts already withdrawn.

**Randomization (readback-verified):** kiln xy ±4 cm + yaw ±20°, banana staging
xy ±6 cm + free yaw on the open floor, rammer start depth ±2 cm.

## Rubric

- `success()` = banana ON the chamber floor past the sill line (kiln body frame,
  z gate rejects perching) ∧ rammer head withdrawn ≥ 18 cm outside the mouth
  (its reset pose already satisfies this, so retraction alone earns nothing) ∧
  banana and rammer both at rest.
- `score()` (monotonic, latched): 0.25 per stage ever reached — banana settled in
  the open feed channel / banana inside the covered tunnel / banana inside the
  chamber — capped at 0.75; exactly 1.0 iff success() holds live. Latched credit
  never evaporates (smoke 12).

## Solution outline (solve.py — the legitimacy certificate)

Teleport = TRANSPORT ONLY: one pose write moves the loose banana from its staging
spot across the open floor into the OPEN-TOP feed channel, broadside, ahead of the
pusher face (satisfies only the first 0.25 latch — asserted). Everything
load-bearing is contact: ram-in is a velocity-servoed horizontal force on the
rammer (≤ 12 N, gain-escalating on friction stall, force-frame probe against wrench
drag) driving the pusher face exactly to the sill — the banana's own broadside
half-depth carries its center past the sill line onto the chamber floor; then the
same servo pulls the rammer back out to its dock. Phases: P0 settle → P1 stage
banana (teleport, 0.25) → P2 ram-in (contact push through the tunnel, 0.75: tunnel
+ chamber latches) → P3 withdraw (contact pull) → P4 hands-off persistence 3.3 s →
`SIM_GEN_SOLVE: SUCCESS`. Verified on the forge for seeds **0** (25.7 s) and **1**
(24.3 s), scores non-decreasing 0 → 0.25 → 0.75 → 1.0.

## Embodiment argument (Franka, one base pose)

Base at the origin. The kiln mouth sits at (0.35, 0.18); ALL robot work happens on
the near side of the mouth. Farthest work point is the rammer knob at its reset
depth (kiln-local x ≈ −0.74): ≤ 0.57 m from the base under worst jitter + yaw —
well inside the 0.855 m reach envelope; the banana staging zone is 0.2–0.45 m out.
Per-object contact strategy:

- **Banana (the only pick):** grasped across one segment (32 × 34 mm cross-section
  ≪ 80 mm jaw stroke) from the open floor and laid into the OPEN-TOP channel — a
  10 cm vertical place over 5 cm walls. It is never touched again.
- **Rammer knob (the push/pull handle):** a 22 mm square post standing proud of the
  handle — a native parallel-jaw pinch. Ram-in and withdrawal are horizontal drags
  of ≤ 12 N (demonstrated cap) — far under the arm's capability — and the channel
  walls guide the head (170 vs 180 mm), so no fine lateral servoing is needed. The
  knob NEVER passes the mouth (asserted ≥ 18 cm outside at full insertion): the
  hand never has to enter or reach over the tunnel.
- **Kiln:** furniture; no contact required. The chamber interior is never entered
  by anything but the banana and the front 5 cm of the pusher head.

## Execution-order declaration

The geometry enforces the only order that matters: the banana must be laid into the
open channel BEFORE ramming (the head fills the channel — nothing can be slipped
past it, and the covered tunnel admits no hand), and the rammer can only be
withdrawn AFTER delivery if success is to hold (retraction is a success conjunct
and its reset state earns nothing, so "withdraw first" is a no-op). Within that,
the exact staging spot in the channel is free; solve.py places the banana just
ahead of the face.

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 15/15` on forge)

1. Settle/no-NaN: banana loose at its staging spot, rammer withdrawn, kiln settled,
   all still; authored masses read back (custom spawners apply no cfg schemas).
2. Reset worthless: score ~0, no success — and the rammer STARTS retracted, so the
   retraction conjunct alone can never earn credit.
3. Randomization readback: kiln xy + yaw vary within the declared band (8 seeds).
4. Randomization readback: banana staging spread > 2 cm, rammer depth spread > 8 mm.
5. Null policy (240 steps): score ~0, no success.
6. **Seed-strategy analog (carry exclusion):** a banana pressed down onto the
   chamber with 6 N rides the ROOF — support height measured, never inside.
7. Ballistic flick: a 2 m/s launch down the channel (velocity readback proves the
   probe fired) tumbles to a stop far short of the sill; chamber latch never sets.
8. Tool drivable (positive control): 10 N advances the rammer 240 mm up the
   channel — yet rammer motion alone earns NOTHING (score ~0).
9. Tunnel near-miss: banana settled 6 cm short of the sill — tunnel latch only
   (0.25), no success.
10. Perch/z-gate: a banana at rammer-head height over the chamber floor is never
    inside; the same xy ON the floor is (positive control, success stays blocked
    by the parked rammer).
11. No-retraction: banana genuinely inside + everything still, head still in the
    tunnel — every other conjunct True, success False.
12. Latched credit survives removing the banana from the chamber to the open floor.
13. Settle gate: the success configuration with the banana still sliding at
    0.46 m/s is refused at that instant.
14. Rejection audit: success() never True at any judged point in the battery.
15. Final no-NaN; frames.npz saved.
