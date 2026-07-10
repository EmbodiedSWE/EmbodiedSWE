# SO101 elbow-horn M3 fastening — geometry + design notes

Attaching the **distal half (lower_arm..gripper)** to the assembled elbow: the forearm fork
clips over the seated STS3215 and is locked by **M3 screws on the four peripheral screw lines
around the elbow axis — four on each side of the fork** (Haoxiang's correction: the center bore
is only driver access, not a screw seat). Continues `so101_fastening_notes.md` (the four M2 tab
screws).

## The four M3 screw lines (v2 — the real fastening)

Axes measured from the horn's Ø2.46 hole ring; in the lower_arm frame: (±4.97, ±4.97) mm (use
the per-hole values in the scene cfg — they differ by ~0.05 mm).

- **NEAR (horn) side** (driven in the smoke, lying pose): Ø5.36 access channel through the
  fork's outer skin (z -12) → the M3 head seats on the fork's INNER PLATE (Ø2.96 pilot, face at
  z -3.5) → the shaft crosses the recess-floor pilot and threads into the horn's TWO metal
  layers (Ø2.46 holes at z 0..0.5 and 2.5..3.0). An M3x6 fits this stack exactly. Seat head-top
  z = -0.0066 (head bottom 0.14 proud of the plate — contact-free, the weld holds), quat
  (0,0,1,0).
- **FAR (case-back) side** (driven in the smoke after a weld-safe flip re-grab, like the far
  M2 tabs): Ø2.96 pilots through the far plate (z 36..36.5 and 39.5..40), threading into the
  case-back's two Ø2.46 layers (z 34..34.5 and 36..36.5). Seat head-top z = +0.0431, quat
  identity.
- The M3 asset is baked with head Ø7.05 -> Ø5.0 (`head_shrink=0.71`) so it passes the Ø5.36
  access channels (the real Ø5.5 pan head passes by print flex).
- Collision-only drills on all four axes (`MOTOR_M3_CUTS` / `LOWER_ARM_M3_CUTS`, r 2.0): the
  printed Ø2.96 pilots (self-tap interference vs the Ø3 shaft) and the metal Ø2.46 layers (the
  seated tip lives inside one) are both sub-voxel fits that would fight the welds.

## Frames (everything measured in LINK frames)

- **The elbow axis** in the motor/upper_arm frame: (x, y) = (-0.11257, -0.028), direction ±Z —
  from the URDF `elbow_flex` joint (`localPos0`, axis Z). It sits between the four tab-screw
  axes (x -0.1227/-0.1022), 29 mm off the tab line toward -y.
- **Seated lower_arm pose in the motor frame** = the joint transform at joint zero:
  pos (-0.11257, -0.028, 0), quat wxyz (0.7071068, 0, 0, 0.7071068) (= Rz+90; localPos1 = 0, so
  the lower_arm link ORIGIN lies on the elbow axis). This is `elbow_lower_arm_seat_pos/quat` in
  the scene cfg and the frame of the pre-authored `lower_arm_weld` (motor<->lower_arm).
- The **horn side is the NEAR side** (link -Z, same facing as the countersunk tab wall), so the
  lying working pose (base Ry(90)) puts the horn hole up — one pose serves tab + horn phases.

## The parts around the axis (probe profiles, link-frame mm)

Motor (visual): splined output shaft z -2.6..-1 (r<=2.6, center tap bore r~0.6), horn disc
z -0.5..+0.5 (OD r 10), spacer drum z +2..3 (r 4.5..6), case-face boss z +3.5..+4.5; back
bearing post at z +33..37.5 (ring r<=6, top boss r<=3.15). Collision (post-3%-shrink): spline
r<=2.85 spans z -1..0 only; disc OD r 9.94.

Lower_arm fork (`under_arm_so101_v1`): a CLOSED fork. Drive-side plate: outer skin z -12..-11
with the **Ø3.2 center bore** (the M3 clearance/access hole), hollow box, inner skin z -3.5..-3
(Ø8.6 hole), internal hub tube ID r 3.0 (z -1.5..0, washer face r 1.6..3.0 at z -1.4), recess
ring floor z 0..0.5 (r 4..10.9), rim ID r 10.1 (z 0.5..2). Far plate: z +36..43, hole r 3.0
riding the post top. The four (±5, ±5) holes match the horn's peripheral M2 pattern (not
driven — no straight-line access; same status as the tab-line inner screws).

**The fork ATTACH is a SLIDE-ON (v2 — replaces the v1 teleport-place):** the fork approaches
from the servo's FRONT — where the forearm extends into open space — and slides mouth-first
along the servo's long axis (28 mm travel, lifted 1.0 mm off the engagement bosses), then
presses down onto the horn. This is the real clevis motion; the kit's print-flex snap is
replaced by collision trims on the swept locating features (below). Paths that do NOT work:
axial (the far plate sweeps through the servo case), lateral across the axis (the shell wall
sits 2.2 mm from the fork's mouth edge — no runway), rigid tilt-hook (the plate corner digs
into the case back). The hand still holds the fork at the seat until the first screw bites.

Slide-enabling collision trims (visuals untouched; the weld owns all seated locating):
- fork cup RIM ring removed outright (it closes around the horn disc — blocks any entry);
- fork HUB TUBE below the cup removed (sweeps through the horn-disc plane);
- motor BACK-POST RING top trimmed 1.6 mm below the far-plate plane (coplanar with the
  sliding plate otherwise), with the four far M3 drive bores SPARED (r 2.4 around their axes
  — an early un-spared trim gashed the bore walls open and the poisoned SDF ejected the
  driven screw at its seat, +-0.7 mm bounce, drive timer never completing).

## The M3 (v1, superseded)

v1 drove ONE M3 into the center bore with the head on the outer skin — geometrically drivable
but wrong per the kit (Haoxiang's catch). The center-bore collision drill was dropped in the
from-source distal rebuild; everything else (fork clip cuts, hand-held attach, magnetic-bit
pickup-and-carry placement) carries over unchanged to the four real screw lines above.

## Collision-only cuts (build_so101_assets.py: MOTOR_HORN_CUTS / LOWER_ARM_CUTS)

Every seated fit measured SUB-VOXEL (SDF ~0.26 mm voxels + 0.15 contact offsets) — each would
phantom-contact-fight the enabled weld (the far-seat z=0.0450 lesson):

| fit | stock | cut | result |
|---|---|---|---|
| M3 shaft r1.5 vs skin bore r1.6 | 0.1 | fork bore -> r 2.0 | 0.5 |
| spline OD r2.85 vs fork hub ID r3.0 | 0.15 | spline stub cut r<2.95 z<0 AND fork hub -> r 3.4 | 0.45 |
| post top r3.15 vs plate hole r3.0 | -0.15 | post top removed above z 36.5 (stub r 3.3) | free |
| horn disc OD r9.94 vs fork rim ID r10.1 | 0.16 | fork rim -> r 10.65 | 0.7 |

Also: `tighten_distal_offsets` — the lower_arm colliders carried 0.5 mm contact offsets
(pre-dating the phantom-contact finding); now 0.15 like shell/motor. All cuts are collision
only, applied by `cut_link_cylinders` (link-frame zones, face-drop + shard projection,
idempotent); visuals byte-identical. The fork hub widening had to be on the FORK side because
projection can only push walls outward and motor verts hug r 2.95 right up to z 0.

## The assembled elbow is a DRIVEN JOINT (v3)

Fastening the horn screws no longer welds the forearm rigid — it closes the real elbow: a
pre-authored DISABLED RevoluteJoint (axis Z at the elbow transform, limits ±96.8°) with the
URDF elbow_flex servo drive (stiffness 5.286/deg, maxForce 10), enabled while any horn screw
is fastened. Command it via `scene.set_elbow_target(rad)` — the smokes' finale swings the
assembled elbow ±30° while lifted. Tuning that mattered:
- drive damping 0.025 (near-critical) — the URDF's 0.0021 is tuned for the implicit
  articulation solver and leaves a LOOSE joint ~10x underdamped (the forearm rang to 102° on
  a ±30° command);
- reconcile keeps the live joint ANGLE on small heals but resets it to ZERO on real snaps
  (a teleport's stale relative pose yields a garbage angle that can violate the limits at
  re-enable — this broke the far screws after the flip);
- the gate's fork-alignment check is angle-AGNOSTIC (position + off-axis residual only): the
  horn holes rotate with the joint;
- `drive_time` (the min-drive window before a weld) survives momentary gate breaks while the
  screw stays ON the magnetic bit — an uninterrupted-window requirement starved the last far
  screw whenever a seat-contact flicker broke the gate for single steps.
Known cosmetic limit: the horn VISUAL is baked into the motor mesh, so it does not spin with
the articulated forearm (it is hidden inside the fork's cup; visible only edge-on).

## Scene mechanics (so101_assembly.py)

Hole GROUPS: flat index order [4 elbow tabs, 1 horn]. `_pair_ok` restricts pairing to same
group. Per-hole alignment: tab holes need the servo seated (motor vs upper_arm); the horn hole
needs the lower_arm at `lower_arm_seat_w()` (vs the LIVE motor pose — free assembly order).
Per-screw welds: tab screws body0=upper_arm, horn screw body0=lower_arm. Part welds: motor weld
on while any tab screw fastened; `lower_arm_weld` on while any horn screw fastened.
`_reconcile_fastened` snaps along the expected CHAIN arm -> motor -> lower_arm -> screws
(set_state passes the snapshot motor pose since live data is stale after root writes).

## Side effect on the full smoke (fixed)

The motor cuts changed the collision mesh's AABB (spline tip z -1.0 and post top z +37.4 were
its extremes) → the SDF re-cooks on a shifted voxel grid → the near-hole PIN catch (knife-edge:
0.25 mm guidance vs 0.26 mm voxels) flipped for hole 0 and the coax-dropped M2 fell through.
Fix in `so101_smoke.py`: near screws are now PLACED nose-first at the cone mouth
(NEAR_DROP_H 0.010 -> 0.006, so the tip starts at the wall face with near-zero fall energy —
the far-hole probe-verified recipe — plus lateral 0.0015 onto the cone). Rests -0.68/-4.08 mm,
full smoke PASS again. Higher drops or the old 8-degree tilt (which relied on the removed
thumb) punch past the cone bake-dependently. Any future motor re-bake re-rolls SDF-marginal
catches — place gently, don't drop.

## Smoke

`so101_horn_smoke.py` — the standalone debug smoke: starts FULLY ELBOW-ASSEMBLED (hand-built
post-[G] state), hand places + holds the fork, drives all eight M3s (near four lying, weld-safe
flip, far four), stress, finale.

**MERGED into `so101_smoke.py`** (the full-assembly sequence): A wiggle -> B servo insert ->
C rotate -> 4 M2 tabs (all magnetic-bit carried; free drops retired — every free-release recipe
hinged on SDF-marginal catches that re-roll on each motor bake) -> H fork attach (hand press +
hold) -> 8 M3s (near lying, flip, far) -> stress (M2+M3 knock, servo+fork wrench) -> finale.
The drill parks at DRILL_PARK (-0.55, -0.35 — OUTSIDE the assembled robot's ~0.55 m reach and
the finale's 180-degree sweep) before every re-fixture.

The merge surfaced a LATENT one-word bug that had been in the committed smoke all along:
`arm_target_quat = q_hold.expand(n, 4).contiguous()` — at n=1 the expanded view already counts
as contiguous, so `.contiguous()` returns an ALIAS of q_hold, and grab()'s in-place clamp-target
write overwrote q_hold with the flip pose at the first far re-grab. Every later "lying" grab
then silently re-applied the far facing (invisible pre-merge: all post-flip phases were
facing-agnostic; fatal post-merge: the fork got held upside-down on the horn and the fasten
gate deadlocked). Fix: `.clone()`. Debug pattern that cracked it: phase bisect -> minimal probe
(clean = state exonerated) -> RAW quat dumps inside the failing variant (derived metrics
mislead) — plus an offscreen-camera render for the first real clue.
