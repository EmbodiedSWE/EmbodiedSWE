# G1 Pick-Place (Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0) — empirical learnings

## Run command (headless, no video, ~1.5 min boot)
```
VIRTUAL_ENV=/home/yilang/research/IsaacLab-v6/env_isaaclab PYTHONUNBUFFERED=1 \
  /home/yilang/research/IsaacLab-v6/env_isaaclab/bin/python -u <script>.py --device cuda:0
```
Must run from repo root `/home/yilang/research/IsaacLab-v6`. Isaac Sim's hard exit drops block-buffered
stdout — always use `PYTHONUNBUFFERED=1` / `python -u` when redirecting to a file.

## Action (28-d, WORLD frame, num_envs=1 so world==env coords)
`[L_pos(3), L_quat_wxyz(4), R_pos(3), R_quat_wxyz(4), hand14]`. The term converts world→pelvis internally.
Hand-14 order = cfg order: [L_index0,L_mid0,L_thumb0, R_index0,R_mid0,R_thumb0, L_index1,L_mid1,L_thumb1,
R_index1,R_mid1,R_thumb1, L_thumb2, R_thumb2]. Open = zeros. IK silently STALLS on infeasible targets.

## Geometry
Pelvis fixed (0,0,0.75) rotated +90° about Z: robot left = world −X, right = +X, forward = +Y.
Object = steering wheel, AABB 0.286×0.286×0.084, starts hub at (−0.35,0.45,0.70) on table top z≈0.70
(top 0.78, mid-plane 0.74). Light, easily knocked.
Reach (wrist): LEFT reaches the object (x=−0.35) and maxes wrist-x≈0.26; RIGHT reaches wrist-x≈0.40–0.42
and y≈0.40 SOLO; cannot cross to the object. Forward reach (y) ≈0.40 solo, drops to ~0.30 when BOTH arms
are forward (shared waist null-space coupling). Orientation Q_DOWN=[0.7071,0,0.7071,0] → fingers point −Z.

## Success (task_done_pick_place), ALL simultaneously, world frame:
object x∈(0.40,0.85), y∈(0.35,0.60), z<1.10, |vel|<0.20 each axis, right_wrist_yaw_link x<0.26.
Object must never drop below z=0.5 (object_dropping termination). ~1000 control steps @ 50Hz.

## What WORKS
- LEFT deep grasp (descend cmd z=0.80 → wrist stalls ~0.90, fingertips ~0.72) at near-center (y≈0.40)
  then DRAG horizontally: reliably drags the wheel along the table from −0.35 to ~0.10, staying LOW (z=0.70).
- Approach must be a HIGH traverse: raise to z=1.18, move over the wheel at 1.18 (open fingers hang ~0.18
  below wrist; at z<1.05 they brush/knock the wheel), THEN descend vertically. Approaching in front
  (y=0.10) then over the grasp point avoids the +y brush.
- v14: RIGHT gripping the wheel's CORNER/rim (offset from hub) released CLEANLY (open→lift, no fling),
  leaving the wheel at rest on the table. This is the key clean-release observation.

## What FAILS (do not repeat)
- LIFTING a CENTER grip straight up FLINGS the wheel (fingers hook under the open hub/spokes; the wheel
  hangs from the fingers and is hurled to z~0.9+ and random x/y). Center grips hook; rim/corner grips don't.
- Sliding the open hand −x or −y at low z DRAGS the wheel ~1:1 (open fingers still grip enough).
- ROTATING the wrist (Q_DOWN→fingers-forward) while near the wheel SWEEPS the fingers through it and shoves
  it violently.
- BIMANUAL pin (left holds while right grasps): prevents the fling (wheel stays low!) BUT the shared-waist
  coupling drops the right's forward reach to y≈0.30, so it misses/slips on the wheel (hub rides ~0.10
  forward of the gripping hand). Needs the wheel kept at very low y AND a rim/corner grip the right can reach.
- Descending onto the near rim (y≈0.33) glancingly SHOVES the wheel +y (to ~0.65). Descend at y≈0.40
  (near-center) shoves only slightly (0.45→0.48).
- Shallow grasp (fingertips ~0.77 at the top) grips unstably and flings worse.

## Open problem
Get the wheel hub to x∈(0.40,0.85), y∈(0.35,0.60), at rest, with right wrist <0.26. Requires LEFT→RIGHT
relay (left can't reach the place, right can't reach the object). The crux is a CLEAN release that neither
flings nor drags the wheel, plus a right drag that keeps the wheel LOW (a lifted/hooked wheel becomes a
heavy hanging load that stalls the right's forward drag at wrist≈0.17).
