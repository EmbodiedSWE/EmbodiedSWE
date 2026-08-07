# obstacle_i17 — Skittle Gallery (scene `skittle_gallery`)

Knock over the RED pin — and ONLY the red pin — inside a sealed, roofed two-cell
gallery whose interior can never be reached by hand: identify the red pin's lane by
looking through a letterbox slit, then bowl the 60 mm blue ball through that lane's
floor-level tunnel hard enough that it crosses the cell and topples the top-heavy pin
by impact, while the white pin in the other sealed cell stays standing. A 96 mm orange
ball is a decoy that fits nothing.

## Provenance

- **Seed:** `pick_place/obstacle`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/obstacle.py`) — pick a cube and
  carry it OVER a wall standing between it and the goal zone; the wall is a detour in
  an otherwise free transport, and the checker is the cube's final resting pose.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `skittle_gallery`,
  env `simgen.skittle_gallery`, robot `"null"`), `solve.py` (teleport solution),
  `smoke.py` (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's obstacle is soft — a wall you simply lift over; the
  judged object is the carried payload and its judged state is a RESTING POSE the
  gripper itself places. Here the obstacle is absolute: the gallery is roofed and
  sealed, so "go over" does not exist, and the judged bodies (the pins) can NEVER be
  touched by the agent — the only aperture is a floor tunnel narrower than any
  end-effector approach at depth. The judged state is not a placement but an EVENT
  OUTCOME (a toppling caused by momentum the agent imparted before contact), plus a
  fragile NEGATIVE invariant (the white pin must survive), plus a perception gate
  (which lane is red flips per episode and is only visible through the slits/tunnels).
  The seed's whole strategy — carry the payload to the goal and set it down — is
  physically impossible for the embodiment here, and its constructed end state (ball
  at rest beside the standing red pin) is explicitly rejected (smoke check 6).
- **vs `peg_insertion_side_i1` (ram a rod through a tube to eject a body into a
  basin), the nearest cousin read this session:** i1 is quasi-static TOOL USE — the
  rod stays effectively "in hand" the whole time and pushes the payload out by direct
  chained contact. Here the projectile is RELEASED: after launch, the shot is
  ballistic and unactuated through funnel, tunnel, open cell, and impact — success is
  decided by aim + imparted momentum, not by a maintained push. i1 also has no lane
  choice and no body that must be left standing.
- **vs `peg_insertion_side_i2`:** ordered two-body fastening (create a passage, then
  thread it). No linkage exists here; the passage is pre-existing but the payoff is a
  remote dynamic strike with a binary, perception-gated target selection.
- **vs the pick-and-place family** (`libero_i2/i3/i9/i11/i13/i14`,
  `pick_single_egad_i3/i4`, `approach_grasp_spoon_i12`, `setup_checkers_i2`): all
  judge poses of bodies the gripper handles directly (grasp quality, placement,
  arrangement). Here the judged bodies are untouchable by construction and the
  handled body (the ball) has NO goal pose of its own.
- **vs the articulated family** (`close_microwave_i4/i5`, `close_grill_i8`,
  `open_oven_i6`): those judge a joint coordinate of a mounted mechanism. There is no
  articulation here at all — one kinematic shell, four free rigid bodies, and the
  outcome is a rigid-body toppling event.
- **vs `pour_water_i7`:** granular transfer by container reorientation; nothing is
  poured here and the payload is a single projectile.
- **Execution order is REQUIRED and declared:** identify the red lane FIRST (it flips
  per episode), then bowl. Shooting the wrong lane fells the white pin, and the
  top-heavy pins make "down" absorbing — the mistake cannot be repaired, so the
  perception step cannot be skipped or gambled (a 50% coin-flip strategy fails half
  of all episodes unrecoverably).

## Randomization (per episode, verified by readback in smoke)

Gallery xy jitter ±40 mm + yaw ±12° (lane axes and mouths move — the solver must read
the gallery pose), RED/WHITE lane assignment coin-flipped, pin xy jitter inside each
cell, ball and decoy scattered on the front floor with batched keep-out resampling.
The approach baseline `d0` (ball spawn → red mouth) is captured per episode.

## Rubric

`success()` iff, settled (pins lin < 0.05 m/s and ang < 0.5 rad/s, ball < 0.30 m/s):

- RED pin DOWN inside its cell: tilted past 60° AND centre below 40 mm (lying, not
  leaning) AND inside the chamber (a pin knocked out of bounds or lying outside
  counts nothing);
- WHITE pin STANDING: within 25° of vertical at standing height (±20 mm).

`score()` (latched every physics substep in `post_step`): `0.10 ×` best ball approach
toward the red mouth (normalized by the episode's own spawn distance) `+ 0.35 ×` ball
entered the red cell `+ 0.20 ×` red pin tilted past 30° — gated on the ball having
actually entered the cell, so waving forces outside earns no tilt credit. Base capped
at 0.65; exactly 1.0 iff `success()`. Doing nothing scores ~0.

## Teleport solution (`solve.py`) — transport only, ONE write, ending in free space

- **P1 — transport:** the ball is carried from its scatter spawn to a STAGING point
  on the red lane's axis, 420 mm in front of the wall — open floor, outside the
  funnel splays, clear of the decoy. That is the only teleport.
- **P2 — the shot is CONTACT DYNAMICS end to end:** a floating-hand throw first LIFTS
  the ball a few cm (vertical PD — a held ball does not spin), then accelerates it
  along the lane axis to ~1.2 m/s with a lateral PD holding the lane line, and CUTS
  all forces 180 mm before the wall, before the funnel throat. From there the shot is
  ballistic: land short of the mouth, funnel in, cross the tunnel and the cell on
  momentum alone, topple the red pin by impact. (The airborne carry is also a
  physics-API necessity: this stack applies external wrenches in a frozen body frame,
  so forcing a ROLLING ball precesses the push — a lifted, non-spinning ball keeps
  the force world-true.) The ball is never teleported into the tunnel or cell; no
  force ever touches a pin; the white pin and the decoy are never touched at all.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing
  (0.000 → ~0.01 → 1.000 → 1.000), ≥3.3 simulated seconds hands-off persistence, then
  `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge on three seeds with provably
  distinct layouts by stdout readback** (seed 0: gallery (−0.008, +0.121) yaw +11.3°,
  d0 0.423, red LEFT; seed 1: (+0.031, +0.082) yaw +4.1°, d0 0.510, red LEFT; seed 2:
  red RIGHT — all three SUCCESS, covering both lane assignments).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0.0, −0.85), facing the gallery front: the ball scatter zone, the
staging line, and both slits are within ~0.6 m reach, and nothing the task requires is
deeper — the pins are intentionally OUT of reach (225 mm behind a 60 mm wall through
an 82 mm bore; no arm pose reaches them, which is the point). Per-object contact
strategy: peek — camera/wrist to a slit (100–135 mm height) or tunnel mouth to read
which head is red; the wide 42 mm colored head sits exactly at slit height. Bowl —
the 60 mm ball is a clean parallel-jaw sphere grasp (< 80 mm stroke), carried to the
lane axis and thrown/rolled with a horizontal arm sweep; ~1.2 m/s release speed is
comfortably within Franka end-effector velocity limits, and the funnel walls forgive
±~55 mm of lateral release error, far above arm repeatability. The 96 mm decoy
exceeds the gripper stroke — ungraspable as well as useless, a pure identity/size
control. The task is genuinely completable and its constraints genuinely bind.

## Checks (`smoke.py` — rejection battery, 14 named checks, ALL PASS on the forge)

1. settle: states finite; both pins upright at standing height, ball on the floor.
2. settle: score ~0 at reset, no success.
3. randomization readback: gallery xy + yaw vary AND red lane takes both sides.
4. randomization readback: ball xy, decoy xy, d0 vary.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: ball SET DOWN at rest beside the standing red pin (in-cell delivery
   verified by readback) → NOT success, score ≤ 0.55 — transport without the strike
   earns at most partial credit.
7. wrong lane: WHITE pin felled while red stands (the unrecoverable failure) → NOT
   success.
8. both pins down: red genuinely down (`red_down()` True) — the white-standing clause
   alone rejects success.
9. chamber clause: red pin lying flat and low but OUTSIDE the gallery → `red_down()`
   False, NOT success.
10. wrong object: orange decoy driven at the red mouth at 1.5 m/s reaches the wall
    but never enters the chamber; pin untouched → NOT success (size/identity control).
11. latched credit: removing the delivered ball leaves the latched approach + entry
    score unchanged, still no success.
12. monotonicity: closer ball placement latches strictly more approach credit.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` (149 × 600 × 960 × 3) recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest: the
ball clears tunnel/alley/funnel with real margin, the decoy fits NONE of them, the
slits pass nothing, the pins are beyond reach and below the roof, every pin can fall
FLAT in all four directions (no lean-on-wall limbo), a forward-falling pin fits
between the alley rails, the colored head fills the slit band, and the down/standing
bands are mutually exclusive.
