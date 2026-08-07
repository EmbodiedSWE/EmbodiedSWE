# native_tasks_i37 — TrayPack: jointly tile three blocks flat into a walled tray

Scene `tray_pack`, env `simgen.tray_pack` (scene-level, `robot="null"`).

## Provenance

Seed: `maniskill/native_tasks` (`roboverse_pack/tasks/maniskill/native_tasks.py` — the
MetaSim-native ManiSkill single-primitive suite: PickCube / PushCube / PullCube /
StackCube / PokeCube / LiftPegUpright / PullCubeTool / PlaceSphere / RollBall). Every
seed task moves ONE free primitive to ONE independent goal pose in open space and is
judged by a point/pose distance. What is kept from the seed: the tabletop-scale free
rigid primitives (plain colored cuboids of ManiSkill cube size class), the
place-an-object-at-a-pose skill vocabulary, and the fully procedural asset style.

## Strategic difference

The seed's plan is *independent placement*: each object has its own goal site, no
placement affects any other, and any single success is a full success. Here the
manipulation model is replaced by a **joint feasibility (packing) constraint**:

- **No block has a goal pose.** Success is a property of the *set*: all three blocks
  simultaneously flat on the floor of a 135×135 mm walled tray, fully inside.
- **Placements couple through rigid non-overlap.** The footprints (120×40, 80×80,
  80×40 mm) tile 144 cm² of a 182 cm² floor: they only all fit in deliberate tilings
  (long bar flush along a wall, big slab square into a corner beside it, small brick
  into the one remaining pocket, up to rotation/mirror). A block placed thoughtlessly
  (e.g. mid-floor) makes the rest of the pack *physically impossible* — the smoke
  battery demonstrates this (check 11), so the solver must plan an arrangement and
  re-arrange, sliding blocks flush against walls and each other.
- **A blocker starts inside:** the blue slab begins IN the tray standing on its side
  (scores nothing there), at a random spot and heading — it must be laid flat and
  packed like the rest.

Versus the rest of the tasks_v7 corpus (39 tasks surveyed): nothing does
arrangement/tiling packing. Nearest neighbours and why this is different:
`empty_dishwasher_i23` (sliding-tile puzzle: ordered corridor moves of ONE tile class
through slots — here nothing is slotted; the constraint is areal tiling of free
blocks), `close_grill_i8` / `close_box_i26` (articulated-lid construction — no
articulation here), `setup_checkers_i2` (ordered insertion into a one-disc channel —
here there is no order and no channel), pick-and-place families (independent goal
sites — exactly the seed's model this task abandons).

## Solution outline (solve.py, demonstrated on the forge)

Teleports are TRANSPORT ONLY (always to free space, hovering 2.5–7 mm, then free
fall); every judged fact — seating, flushness, containment — is produced by gravity
and horizontal contact pushes at the CoM (frame-drag-probing force encoder, see
below). Phases (each ends with a monotone `SIM_GEN_SCORE` print):

- **P0** reset + settle; assert baseline (slab standing, nothing seated, score ≤ 0.01);
  capture per-block reference rotations for the force encoder.
- **P1** transport the standing slab OUT of the tray, flat onto open ground (score
  stays 0 — clears the floor for the bar's lane).
- **P2** transport the bar to a hover over the tray's south lane, drop, then PUSH it
  flush into the south wall and to lane centre (score 0.25).
- **P3** transport the slab back to a hover over the NW corner, drop, PUSH flush west
  and north against the wall corner (score 0.50).
- **P4** transport the brick (rotated 90°) over the NE pocket, drop, PUSH flush east
  and north into the pocket between bar, slab and walls (score 1.0 = success).
- **P5** hands-off persistence ≥ 3 s (10×40 steps), re-assert success, print
  `SIM_GEN_SOLVE: SUCCESS`.

Drops are deliberately coarse (up to 6 mm / several deg off) so every block *needs*
its pushes: on the forge the bar slid 17+7 mm, the slab 15+9 mm, the brick 10+10 mm
under pushed contact into flush position. Pushes start at 2 N and escalate ×1.5 (cap
8 N) only if measured progress stalls; a runtime probe toggles between raw world
force and the `R_ref·R_nowᵀ` pre-encoding if the block moves the wrong way (IsaacLab
`is_global` frame-drag quirk is pod-dependent). Passed clean on seeds 0 and 1
(`SIM_GEN_SCORE` staircase 0 → 0 → 0.25 → 0.50 → 1.0 → 1.0, rc=0).

## Rubric

`score()` in `[0,1]`, latched partial credit anchored in the demonstrated solve:
0.25 per block *ever* seated-and-settled (latched in `post_step`), capped at 0.75;
exactly 1.0 iff `success()` holds live: all three blocks **simultaneously** seated
(flat within 10°, centre within 10 mm of floor-top + half-height, all four bottom
corners inside the interior footprint +4 mm, in the live tray frame) and settled
(|v| < 0.05 m/s, |ω| < 0.5 rad/s), states finite. The z gate rejects
resting-on-another-block (40 mm too high), on-the-rim (73 mm too high) and
on-the-ground-outside (20 mm too low); the corner gate rejects overhang past the
walls; the flat gate rejects the standing/leaning slab. Null policy scores ~0 (the
standing slab is statically stable and scores nothing).

## Embodiment: single Franka with parallel-jaw gripper

Base pose: on the ground at the origin side, base ≈ (0, 0, 0), facing +x; the tray
centre (0.42 ± 0.05, 0.02 ± 0.05) and both scatter spots (0.28, −0.30) / (0.24, 0.30)
± 0.04 all lie 0.24–0.50 m from the base — comfortably inside Franka's ~0.85 m reach,
with the 50 mm rim posing no wrist clearance problem for top grasps.

Per object:

- **Blue slab (80×80×40, 0.32 kg), start = standing on its side inside the tray.** It
  pokes 68 mm above the rim; grasp the protruding part across its 40 mm thickness
  (Franka jaw opening 80 mm — comfortable), lift out, reorient the wrist 90°, lay
  flat on open ground. Later re-grasp across the 40 mm thickness with fingertips from
  above, place into the corner, release, and *nudge flush* with a fingertip against
  the block's side face — exactly the 2–8 N horizontal CoM-height pushes solve.py
  applies. 40 mm block height leaves the fingertips ~25 mm above the floor inside a
  50 mm wall — reachable with a vertical tool axis.
- **Red bar (120×40×40, 0.24 kg), start = flat on open ground.** Top grasp across the
  40 mm width at mid-length, place onto the wall lane, fingertip-nudge flush. The
  lane admits ±7 mm placement error before the pushes; the pushes themselves are
  fingertip drags along the floor.
- **Green brick (80×40×40, 0.16 kg), start = flat on open ground.** Same top grasp
  across 40 mm; the final pocket (55×80 mm after bar and slab, brick 40×80) admits
  ±7 mm aim, and two fingertip nudges seat it flush.

All grasps are 40 mm across — dead centre of the Franka jaw range; masses ≤ 0.32 kg;
no forces beyond ~8 N are ever needed (all pushes are horizontal floor slides of
light blocks). Tolerances are millimetric only *after* flush pushes, which a
fingertip drag supplies naturally.

## Order requirements

None required by the rubric — success is a joint end-state predicate and each 0.25
latch is per-block. The demonstrated order (slab out → bar lane → slab corner →
brick pocket) is chosen purely for drop clearance; any order that ends in a valid
tiling (including mirrored/rotated tilings — smoke check 12) succeeds. What IS
forced by physics: the slab cannot stay mid-floor (blocks the pack, smoke check 11),
so a thoughtless placement demands re-arrangement.

## Checks (smoke.py battery, all constructed-settled probes)

1. settle/no-NaN — reset settles finite: slab standing inside, nothing seated, score 0.
2. randomization readback — tray yaw / tray xy / slab tray-local spot / bar xy differ across seeds.
3. slab coverage — standing spot spans > 12 mm both axes, ≥ 4 heading bins over 10 resets.
4. null policy — 240 idle steps: score ~0, no success.
5. WALL INTERLOCK — seated bar shoved at 3× weight toward both walls, 1.5 s each: stays seated inside.
6. SEED strategy — three blocks flat at tidy separate ground poses: no success, score ~0.
7. wrong place — the exact tiling built on the ground beside the tray: rejected.
8. stacking cheat — brick flat ON TOP of the seated slab in the tray: z gate refuses, ≤ 0.30.
9. perched — brick flat on top of the STANDING slab (inside footprint, rim height): score ~0.
10. near-miss — bar+brick seated, slab still standing between them: no success, capped 0.50.
11. JOINT CONSTRAINT — slab seated mid-floor physically prevents the bar from ever seating.
12. mirror accepted — the mirrored tiling settles to success (rubric judges the class, not one layout).
13. frames.npz video saved.

Verification: `solve` rc=0 with `SIM_GEN_SOLVE: SUCCESS` on seeds 0 and 1;
`smoke` rc=0 with `SIM_GEN_SMOKE: ALL PASS 13/13` — all on the forge
(`forge_client.py run --task native_tasks_i37 --module solve|smoke`).
