# lift_peg_upright_i262 — Burrow Ram

Env id: `simgen.burrow_ram` · Scene: `burrow_ram` (`BurrowRamScene`) · Robot: `null`

## Provenance

Seed task: `maniskill/lift_peg_upright` (RoboVerse
`roboverse_pack/tasks/maniskill/lift_peg_upright.py`): a 24 cm peg lies flat on the
table; grasp it, pitch it 90°, and stand it upright — success is a pose readout on
the peg itself (height band + vertical long axis).

What is kept from the seed: the long square-section peg as the central manipulated
object, and horizontal-floor manipulation of it with a single arm. Everything the
seed *judges* is discarded.

## Strategic difference

- **vs the seed** (`maniskill/lift_peg_upright`): the seed is one terminal
  reorientation of the peg — the peg's own pose IS the goal. Here the peg (stretched
  to a 30 cm rod) is never stood upright and never judged: it is a **reach-extension
  tool**. The judged body is a blue cargo cube that starts captive deep inside a low
  stone through-tunnel (bore 80 × 60 mm — far smaller than the Franka hand; the cube
  at least ~68 mm in from either mouth — beyond fingertip reach). The only physical
  way to move the cube is to slide the rod in through one mouth and push the cube out
  the opposite mouth through contact, then pick up the freed cube and place it on a
  green pedestal. The seed's whole goal state (rod stood upright, free, settled,
  correct height) is constructed in smoke and scores ~0 with no success.
- **vs `lift_peg_upright_i116` (hood_prop)**, the sibling read this session: there
  the peg is a **static prop** stood into a socket inside a bistable lid mechanism;
  the judged body rests ON the peg, and ordering comes from the lid covering the
  socket. Here nothing is propped and nothing rests on the rod — the rod performs a
  guided **dynamic stroke** through a tunnel, is discarded afterwards, and the judged
  body ends far away on a pedestal.
- **vs `pen_holder` (packing exemplar)**: that is repeated tip-up insertion INTO a
  container. Here the goal is extraction — getting the cargo OUT of an enclosure —
  and the only "insertion" is the rod transiting an open through-tunnel.

## Execution order (declared, forced by geometry)

1. **Ram first**: the cube cannot be grasped, hooked, or lifted while captive — the
   aperture excludes the hand and the cube is beyond fingertip reach from either
   mouth (cfg-asserted: `min_face_depth ≥ finger_reach`). It must be pushed out with
   the rod.
2. **Place second**: only a freed cube can be picked up and set on the pedestal.

Any trajectory that ends with the cube at rest on the pedestal top physically passed
through the ram-out, so success() needs no explicit order latch.

## Rubric

- `score()`: latched stage credit — 0.25 once the cube has shifted ≥ 40 mm along the
  bore axis from its start, +0.35 once it has been fully outside the tunnel footprint
  (`|x_local| ≥ freed_x = 135 mm`), capped ~0.60; 1.0 iff success.
- `success()`: cube at rest ON the pedestal top — xy within 30 mm of the pedestal
  axis, bottom on the top face within 8 mm, settled.
- Null policy scores ~0 (the captive cube never moves on its own).

## Solution outline (solve.py — the legitimacy certificate)

- **P0**: settle, layout readback, assert captive, `SIM_GEN_SCORE 0.0000`.
- **P1 (ram)**: teleport the rod (TRANSPORT ONLY) to a rest pose on the ground,
  aligned with the bore, tip ~30 mm outside the entry mouth. Drive it with a per-step
  external force (velocity servo on a finite-difference position rate — the velocity
  readback is phantom under wrenches) with a stall-escalated **friction-bias
  feedforward** (the P-term alone tops out at K·v_des = 0.24 N, below sliding
  friction), a hard speed guard, and a velocity-triggered bias reset. The rod's tip
  pushes the cube ahead of it out of the far mouth until it is ~18 mm past `freed_x`;
  cut the force, settle. Score 0.60. No force is ever applied to the cube; the cube
  is never teleported while captive.
- **P2 (place)**: teleport the freed cube (at rest on the open floor) to 4 mm above
  the pedestal top and DROP it. Score 1.0.
- **Persistence**: ≥ 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS` only
  if success() still holds.

Verified on forge: seeds 0 and 1, both `SIM_GEN_SOLVE: SUCCESS`, monotone scores
0.0000 → 0.6000 → 1.0000, rc=0.

## Embodiment argument (single Franka, parallel jaw, OSC)

- The rod is a 50 mm square section — comfortably inside the ~80 mm parallel jaw —
  and 30 cm long. During the whole stroke the rod's butt end stays ≥ 60 mm proud of
  the entry mouth (stroke needed ≈ 250 mm of the 300 mm rod even in the worst case),
  so the gripper can hold the butt from above for the entire push: one straight
  horizontal ~0.25 m stroke along the bore axis, no regrasp inside the tunnel.
- The rod enters the bore with 10 mm vertical and 15 mm-per-side lateral play; the
  tunnel walls square it up during the stroke, so mm-precise alignment is not needed.
- The freed cube (44 mm) rests on open floor, trivially graspable from above; the
  pedestal top is at 40 mm with 30 mm xy tolerance.
- Everything lives within a ~0.6 m ring around the burrow; a base pose ~0.5–0.6 m
  from the burrow on the entry-mouth side reaches the rod grasp, the whole stroke,
  the freed cube, and the pedestal.

## Randomization (readback-verified in smoke)

Burrow xy jitter (±6 cm) + free yaw; cube depth (±2 cm along the bore) + lateral
jitter + free yaw; pedestal side (left/right of the tunnel axis, both occur) +
bearing + range (0.32–0.42 m); rod ring angle/radius/yaw (0.48–0.58 m, kept ≥ 35°
of bearing from the pedestal). Cfg `__post_init__` asserts captivity, aperture,
stroke length, footprint-clearing `freed_x`, pedestal support, and ring clearances.

## Checks

- `solve.py`: forge seeds 0 and 1 → `SIM_GEN_SOLVE: SUCCESS` (rc=0, ~17 s each).
- `smoke.py`: forge → `SIM_GEN_SMOKE: ALL PASS 13/13` (rc=0, frames.npz 225 frames):
  1. settle/no-NaN — cube starts captive, score ~0, no success;
  2. randomization readback — burrow pose / cube depth / pedestal / rod all vary;
  3. randomization — pedestal lands on both sides of the axis;
  4. null policy — 300 idle steps, score ~0, no success;
  5. seed strategy — rod stood upright free on the ground: score ~0, no success;
  6. near-miss freed — cube half out of the mouth: shift credit only (~0.25);
  7. freed-only — cube clear on the open floor: ~0.60, no success;
  8. near-miss placement — on the pedestal but 36 mm off-axis: rejected;
  9. wrong object — rod on the pedestal, cube captive: score ~0;
  10. stacking cheat — cube on the rod on the pedestal (wrong height): rejected;
  11. wrong place — cube on the burrow roof: score ~0;
  12. rejection audit — success() never True anywhere in the battery;
  13. final no-NaN.
