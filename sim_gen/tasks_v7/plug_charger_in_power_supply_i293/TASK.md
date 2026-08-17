# plug_charger_in_power_supply_i293 — tote pour-out onto the charging mat

Scene `tote_pour_dock`, env id `simgen.tote_pour_dock`.

## Seed provenance

Seed: `rlbench/plug_charger_in_power_supply`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/plug_charger_in_power_supply.py`).
The seed is a terminal INSERTION alignment task: a free, directly graspable charger
sits on the table, a fixed power strip is always exposed, and the goal is one pose —
prongs aligned with the socket holes, charger pushed in. The charger is the grasp
target, the motion is grasp → align → insert, and the episode ends at the inserted
pose.

## What changed and why it is strategically different

The charger brick starts **at the bottom of a deep, narrow tote it can never be
grasped inside of**, and the goal is a flat *placement on an open mat* plus a tidy
workspace — the task is a **container-inversion pour**, not an insertion:

| | seed | this task |
|---|---|---|
| payload access | free on the table, direct grasp | at the floor of a 130 mm-deep tote whose 164×82 mm mouth is narrower than the gripper palm (88 mm) and deeper than the fingers (53 mm + brick top at 22 mm → 108 mm below the rim); in-situ grasp is geometrically impossible (asserted in `__post_init__`) |
| what is manipulated | the payload itself | the **container**: the tote must be lifted and rotated past vertical (slick walls, μ≈0.10; pair friction with the brick ≈0.275 → release tilt ≈105°) until the brick slides out and free-falls onto the mat |
| goal | one inserted pose (payload only) | brick lying FLAT on the mat centre patch (|xy| < 44 mm mat-frame, z in a 4–18 mm band that rejects edge/end stands, tilt < 20°) **AND** the emptied tote parked clear of the mat (mat-frame Chebyshev > 0.19 — footprint overlap is impossible) **AND** everything settled |
| receptacle | rigid socket, tight tolerance, physical stop | the mat is **visual-only (no collider)** — nothing self-aligns the brick; the placement band is met by pour dispersion + gentle ≤2.5 N floor nudges |
| naive strategy | grasp payload, align, push | the seed's whole strategy is unavailable: the payload cannot be grasped until *after* the pour has already happened; pouring and stopping (brick freed, tote still over/near the mat) caps at **0.55**; doing nothing scores ~0 |

Also different from every sibling task read this session:
- `plug_charger_in_power_supply_i21` (keyhole unplug): captive plug, unlock–EXTRACT–stow
  through a geometric latch. Here nothing is captive-jointed, there is no keyhole or
  latch, and the core motion is rotating a *container* past vertical so gravity
  extracts the payload.
- `plug_charger_i230` (shutter garage): mechanism-gated insertion with a re-close
  goal. Here there is no mechanism, no doorway, no insertion — the goal surface is
  an open mat and the "gate" is the geometry of grasp infeasibility.
- `peg_insertion_side_i1` (ram-rod ejection): a free TOOL is driven into a tube to
  eject a payload. Here there is no tool and no tube — the container itself is
  reoriented; extraction is by gravity through the mouth, not by pushing through.
- `peg_insertion_side_i2` (create-passage-then-fasten): builds a passage, then
  threads a pin. Nothing is built or fastened here.
- `robobench/suites/packing/scenes/pen_holder.py`: put objects INTO an open-top
  container. This is the exact inverse — get the object OUT of the container, and
  the container's final pose (parked away) is itself judged.

## Scene

Fully procedural (`UsdGeom.Cube` boxes, 1 mm contact offsets). Materials are
authored inside the custom spawn funcs (custom spawners apply no cfg schemas):

- **tote** — DYNAMIC compound, 0.40 kg: floor 180×98×8 mm + four 8 mm walls,
  interior 164×82×130 mm; **slick interior** (μ 0.10/0.08, restitution 0) so the
  brick slides out at ≈105° pair tilt. Spawned near (−0.05, 0.05) with ±40 mm xy
  jitter + free yaw.
- **bank** — DYNAMIC charger brick 90×55×22 mm, 0.20 kg, grippy (μ 0.45/0.40),
  visual stripe on top; spawned flat on the tote floor with randomized in-tote xy
  (5 mm wall clearance) + yaw.
- **mat** — KINEMATIC charging mat, 200×200 mm pad + centre patch, **no colliders**
  (visual-only goal marker — flush marker colliders would wall the nudge); placed at
  distance U(0.26, 0.32) from the tote at a random azimuth, within 0.36 m reach of
  the origin (ring-resampled with a deterministic toward-origin fallback).

Randomization verified by readback in smoke: tote xy + yaw, in-tote bank xy + yaw,
mat distance + azimuth (cos/sin spread) across 8 seeds.

`success()` iff, settled (< 0.06 m/s): bank flat on the mat patch (mat-frame
|x|,|y| < 44 mm, 4 mm < z < 18 mm, tilt < 20°) AND tote clear (mat-frame Chebyshev
of the tote centre > 0.19 m). `score()` = 0.20·tip latch (tilt fraction toward
100°, gated on the bank still being in the tote) + 0.35·freed latch (bank out of
the tote interior) + 0.30·placed latch (flat on the patch, settled), capped at
0.85; exactly 1.0 iff `success()`. Pour-and-stop (brick freed, tote not parked)
= 0.55; the seed's grasp-and-insert opening move is impossible.

## Solution outline (solve.py, the legitimacy certificate)

Transport-only teleportation: exactly TWO pose writes, both pure transport of
settled objects. Everything load-bearing — the pour, the slide-out, the landing,
the final centring — is contact dynamics via `set_external_force_and_torque`:

- **P1 carry (teleport #1)** — the tote+brick assembly is moved RIGIDLY (relative
  pose read back and preserved) from its random spawn to a pour stance beside the
  mat: tote mouth-up, long axis across the pour direction, base ~0.10 m short of
  the mat centre, held airborne. A rigid carry of a settled assembly is what a
  grasped container does for free.
- **P2 pour** — wrench-held PD (kp 20, kd 6, ≤ 18 N; kr 1.2, ≤ 0.6 N·m; gravity
  feedforward on the effective mass) rotates the tote about a virtual rim pivot at
  z = 0.21 (the 0.169 m body sweep radius clears the ground and anything lying on
  it) through 115° → 135°
  with a 2 Hz ± 12 mm shake stage for stragglers. The brick leaves when tote/brick
  contact physics says so (pair μ 0.275 → ≈105°), free-falls, and lands on the mat
  region; the hold continues until the brick has landed and slowed. FAIL exit if
  the brick is never freed — the pour cannot be faked.
- **P3 park (teleport #2)** — the now-EMPTY tote is transported to a parking spot
  0.32 m from the mat (toward the origin), released, settled. Transport of a free
  settled object only; the clearance predicate is then satisfied by where it rests.
- **P4 nudge** — the brick is centred onto the patch with velocity-regulated floor
  pushes (≤ 2.5 N, stall escalation), mat-frame error → world force direction;
  stops inside 10 mm with the flat gate held. No lifting, no pose writes.
- **P5 persistence** — ≥ 3.3 simulated seconds (10×41 steps) hands-off; SUCCESS
  only if `success()` holds throughout and the phase-boundary `SIM_GEN_SCORE`
  prints never decreased.

Execution order is physics-enforced: P4 placement cannot precede P2 (the brick is
unreachable inside the tote — the only way it gets out is the pour), and the tote
must leave the mat area after pouring because the pour stance itself violates the
clearance predicate. The scored subgoals are reachable only as tip → freed →
placed → clear.

## Embodiment argument (single Franka + parallel jaw, recorded not simulated)

- Base pose ≈ (0.0, −0.45) facing +y; tote spawn (±4 cm around (−0.05, 0.05)),
  mat (≤ 0.36 m from origin) and the parking ring are all within ~0.75 m reach;
  everything happens below ~35 cm height (pour pivot 0.19 m + rim sweep).
- Tote: the 8 mm wall tops are a natural rim pinch far inside the 80 mm jaw
  stroke; 0.60 kg total (tote + brick) is trivial payload; the pour is a wrist
  rotation about the grasped rim — exactly the virtual-pivot rotation P2 performs,
  and the ≤ 0.6 N·m steering torque is what a rigid grasp provides for free.
- Brick: never grasped inside the tote (impossible by assert: mouth 82 mm < palm
  88 mm, brick top 108 mm below the rim > 53 mm fingers); after the pour it is
  centred by fingertip floor pushes at ≤ 2.5 N — no grasp needed at all.
- Tolerances the hand must meet are coarse by design: a ±44 mm landing window on
  a 200 mm mat, a 20° flatness cone, a parking spot with centimetres of slack.
  The fine numbers (release angle, slide-out) are properties of the tote/brick
  materials, not of the hand.

## Checks (smoke.py — 16 checks, recorded to frames.npz)

1. settle/no-NaN: brick read back inside the tote, flat, settled; nothing NaN.
2. randomization readback: tote xy + yaw vary over 8 seeds.
3. randomization readback: in-tote bank xy + yaw and mat distance + azimuth vary.
4. null policy 240 steps → score ≤ 0.02, no success.
5. SEED-strategy end state (brick perfectly placed by construction, tote still at
   its spawn ≈ over/near the ring) → placed latch fires but NOT success.
6. in-situ retention: assembly wrench-held at 85° (below release) for 240 steps —
   tilt actually held (75°–95°), brick stays in the tote, freed latch 0,
   score ≤ 0.20.
7. edge-standing brick on the patch (z ≈ 27 mm) → placed latch 0, no success.
8. brick flat but off-patch (|x| > 44 mm) → placed latch 0, no success.
9. tote parked ON the mat with the brick perfectly placed (tote constructed on the
   mat FIRST so no transient success) → not clear, no success.
10. tote lying on its side, settled → sanity: readback consistent, no success.
11. brick placed but tilted > 20° → placed latch 0.
12. latched pour credit survives the tote being parked away afterwards.
13. tip credit is monotone in tilt (45° vs 85° airborne holds; deeper strictly
    larger).
14. empty-tote gate: tote tilted 90° with the brick teleported far away → tip
    latch stays 0 (no vacuous tip credit).
15. rejection audit: `success()` never True anywhere in the battery.
16. final no-NaN.

## Verification (forge server, RTX 4090)

- `solve --seed 0`: bank released at tilt 115.0°, landed at mat-local
  (+16, −13) mm, nudged to (+7, −6) mm; SIM_GEN_SCORE 0.0000 → 0.5500 → 1.0000 →
  1.0000 → 1.0000, `SIM_GEN_SOLVE: SUCCESS` (rc=0).
- `solve --seed 1`: released at 115.0°, landed (+18, +10) mm, nudged to
  (+8, +4) mm; SIM_GEN_SCORE 0.0000 → 0.8500 → 1.0000 → 1.0000 → 1.0000,
  `SIM_GEN_SOLVE: SUCCESS` (rc=0).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 16/16` (rc=0), frames.npz saved.
