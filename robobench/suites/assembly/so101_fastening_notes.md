# SO101 joint-3 M2 fastening — hole documentation

All coordinates are in the **upper_arm LINK frame** (= the rigid-body frame of both the baked
proximal's `upper_arm` link and the free motor asset `sts3215_03a.usd` — they are the same URDF
link, so "motor seated in the pocket" is identity). Measured on the shell/motor visual meshes
(scripts in the session scratchpad: `hole_truth.py`, `farview.py`, `pinhole.py`, `countersunk.py`).

## THE M2 hole (what the benchmark drives)

The **countersunk hole at joint 3**, piercing the upper arm's outer wall directly over the
seated elbow servo. One per tab, two total:

| | hole A | hole B |
|---|---|---|
| axis (x, y) | (-0.1227, 0.0010) | (-0.1022, 0.0010) |
| axis direction | link ±Z (out of the hole = **-Z**, the outer face) | same |
| pilot bore | Ø1.9 through the wall (link z 0..3 mm) | same |
| countersink | mouth widens to r 2.86 at the outer face (z≈0) | same |
| beneath | the seated servo's registration **pin**, reaching up inside the bore to z = +2 | same |
| seated head-top | z = **-0.0015** (slim-head M2; rim meets the cone at z≈+0.45) | same |

Physics of the drop (all real collision, no scripting): the countersink cone funnels a dropped
screw upright; the Ø2 shaft enters the Ø1.9 pilot ~2 mm and perches on the motor's pin. Driving
it home (rule-based: gate → latched kinematic advance → pre-baked weld) fastens **screw→arm and
motor→arm** together — this screw is what locks the servo into the pocket.

The M2 asset (`assets/so101/screw/screw_m2.usd`) is baked with a **radially shrunken head
(Ø4.7 → Ø4.0, `head_shrink=0.85` in `build_drill_screw_assets.py`)** so the head settles into
the countersink instead of sitting on the face — the shell SDF (256 ≈ 0.43 mm voxels) eats some
of the real Ø5.7 mouth. Shaft (Ø2 × 6 mm) and head height are unchanged. The drill bit is baked
at `DRIVER_SCALE=0.45` (Ø4.0 shank) to match.

## Holes that look right but are NOT the M2 drive target

- **Motor tab through-holes** at (x, y) = (-0.1228, +0.0052) and (-0.1023, +0.0052), axis ±Z:
  Ø3.44 through the servo tabs, original screw heads at link z 4.3–6.9 (stripped in the bake).
  The real kit fastens these from INSIDE the clevis gap with the L-key — both outer faces are
  closed (the -Z side wall is solid self-tap material; the +Z side has only Ø3.2 pilot bores),
  so no straight-line driver access exists. Their far-tab twins (link z 33–37, same x,y) are
  likewise stripped in the bake.
- **Shoulder-horn countersunk ring** at (±0.0047, ±0.005) around link (0, 0): four Ø1.8
  self-tap pilots with wide countersinks on the shoulder-end face — real M2 positions for the
  shoulder horn, usable for a demo, but they fasten arm↔shoulder servo, not joint 3.

## Simulation settings that this geometry needs

- Shell + motor colliders: **SDF mesh** (convex decomposition seals/misses the small holes).
  Collision meshes are vertex-welded in the bake (STL soups cook pathologically otherwise).
- Screw collider: convex decomposition of its own mesh (a single hull is a fat cone that can't
  enter any hole).
- `dt = 1/240` (verified; 1/480 is the fallback if drop/perch contacts ever misbehave).
- Bit↔screw collision is filtered — that pair is rule-based (drive gate), and contact only lets
  the spinning bit bat the screw. Everything else collides normally.
