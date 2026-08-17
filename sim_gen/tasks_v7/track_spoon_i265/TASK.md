# track_spoon_i265 — Knife-Edge Spoon: balance the loaded spoon (`simgen.spoon_knife_edge`)

## Seed provenance

Derived from **pick_place/track_spoon**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_spoon.py`): a Stage-3
trajectory-tracking task — the spoon starts ALREADY RIGIDLY GRASPED in the Franka's
closed gripper, and reward is dense per-step position+rotation tracking of a
prescribed free-space waypoint path toward a basket. The seed's whole strategy is
*transport fidelity of a held payload along given waypoints*; where the spoon ends up
has no physical consequence, and no property of the spoon itself ever matters.

## What changed, and why it is strategically different

Kept only the protagonist (a spoon that must be carried somewhere). The judged skill
is inverted from transport fidelity to **composed centre-of-mass reasoning verified
by a static-equilibrium outcome**:

- **Nothing is prescribed and nothing is tracked.** No waypoints, no per-step reward.
  A dense 26 mm cube (~47.5 g — as heavy as the whole ~46.5 g spoon) must be seated
  in the spoon's bowl pocket, and the LOADED spoon laid across a 20 mm-wide
  knife-edge fin so it rests LEVEL with both ends in free air.
- **The placement is only the INPUT; the settled EQUILIBRIUM is what is judged.**
  The crest supports the spoon only if the COMBINED CoM of spoon+cube lies over the
  20 mm crest. Every naive anchor misses the window by 2.5–5x its half-width (all
  cfg-asserted): the spoon's geometric centre by ~48 mm, the EMPTY spoon's own CoM by
  ~28 mm, the bowl centre by ~27 mm. A solver must *compose* the masses
  (bal = (m_s·com_s + m_c·x_bowl)/(m_s+m_c) ≈ 48 mm bowl-ward of centre) — placing
  "where the object is" or "where the object balances empty" both tip decisively (a
  handle-side tip rests 24° down on the slab, 6x the 4° level band).
- **The load changes the answer.** The empty spoon balances at a point ~28 mm away
  from the loaded balance point — smoke perches the empty spoon at its own CoM as a
  positive control (it balances; success refused for the missing cube), proving the
  task cannot be solved by memorizing the object's unloaded behaviour.
- **Versus the read corpus:** `track_spoon_i160` (same seed) is hidden-information
  discovery through a JOINTED two-pan mechanism — here there are NO joints, nothing
  hidden, and the skill is quantitative statics prediction, not hypothesis testing.
  `pen_holder` and the drawer/microwave tasks judge containment/articulation
  geometry; no corpus task judges an unsupported equilibrium whose feasible
  placement set is defined by a computed composite CoM.

## Apparatus (fully procedural, boxes only — no meshes, no joints)

Heavy DYNAMIC stand (6 kg, teleported at reset): 0.20 x 0.16 x 0.020 m base slab +
orange fin plate (20 mm thick along stand x, 140 mm long along stand y, crest top
80 mm up). Spoon: one rigid compound with a FULLY FLAT underside (bowl floor and
handle bottom coplanar at local z=0 — it can rest flush on the crest anywhere along
its 198 mm length): rimmed square bowl pocket (40 mm inside, 12 mm rims) at the −x
end, 150 x 16 x 8 mm handle, and a raised 16 x 16 x 22 mm green tail KNOB for a
parallel-jaw pinch. Spoon mass (~46.5 g) and CoM (~20 mm bowl-ward, 7 mm up) are
DERIVED in cfg from part dimensions and density — a single source of truth shared by
the spawner (explicit MassAPI mass + CoM: custom spawn funcs ignore cfg mass_props,
and MassAPI-only mass leaves the CoM at the origin), the rubric, and the solver —
and read back in smoke check 2. Cube: standard spawner, aluminium density. Config
honesty is asserted in `__post_init__`: cube-slack CoM blur ≤ 0.4x the crest
half-width, every naive anchor ≥ 25 mm outside the window, real overhangs, ends-air
band discriminates crest from slab, tipped rest ≥ 3x the level band, cube fits the
pocket at any yaw and the Franka jaw, floor slots clear the stand across all
jitters.

**Randomization (readback-verified):** stand xy ±3 cm + yaw ±20°, spoon slot xy
±3 cm + free yaw, cube slot xy ±3 cm.

## Rubric

Judged on the settled state, in body frames (cube containment in the SPOON's frame —
valid at any pose; perch geometry relative to the LIVE stand):

- `success()` = cube seated in the bowl pocket ∧ spoon origin at crest height with
  its long axis within 35° of the fin normal, crossing the fin plane inside the fin
  span, both ends beyond the crest by ≥ 30 mm on OPPOSITE sides ∧ LEVEL (axis tilt
  ≤ 4°) ∧ not rolled (body-z ≤ 10° from vertical) ∧ both end points in FREE AIR
  (above 45 mm — slab rest at 20 mm fails) ∧ stand upright ∧ everything at rest
  ("at rest" = velocity thresholds set above the GPU phantom-velocity artifact band
  PLUS a pose-stillness window: every body's pose frozen over the last 0.25 s —
  falling, tipping and freshly-teleported states all break the window, so no
  transient state can judge settled).
- `score()` (monotonic, latched every substep): 0.20 · the cube ever seated in the
  pocket (quietly) + 0.30 · the seated assembly ever rested spanning the crest +
  0.20 · success() ever held, capped at 0.70; exactly 1.0 iff success() now. Null
  policy ~0; the seed's strategy (carry the spoon somewhere and set it down) ~0.

## Solution outline (solve.py — the legitimacy certificate)

Teleport = TRANSPORT ONLY, always ending in a non-contact hover; every load-bearing
interaction is gravity + contact:

- **P0 SETTLE** — reset scatters stand/spoon/cube; everything rests. Score ~0
  (asserted).
- **P1 LOAD** — the cube is hovered 25 mm above the floor-lying spoon's open pocket
  (yaw-aligned) and DROPPED; the rims retain it and it seats on the pocket floor —
  real containment by contact. Score latches 0.20 (asserted).
- **P2 PERCH** — the cube's TRUE seated pose is read back in the spoon frame, the
  loaded balance point recomputed from the authored masses, and the assembly
  (relative pose preserved) hovered 3 mm above the crest with the balance point
  deliberately offset 4 mm from the crest centre — inside the 10 mm window, proving
  the window is physically real — then RELEASED. Gravity and the contact patch
  settle it LEVEL → success, score 1.0.
- **P3 PERSISTENCE** — hands off ≥ 3.6 simulated seconds; verdict only if success
  still holds.

`SIM_GEN_SCORE` printed at each phase boundary, non-decreasing 0 → 0.20 → 1.0 → 1.0.
Verified on the forge for seeds **0** and **1** (rc=0) — different stand poses, spoon
yaws, and slot draws.

## Embodiment argument (Franka, one base pose)

Base on the floor at the world origin facing +x; the stand centre is at
(0.45 ± 0.03, ±0.03), the crest 80 mm up — everything within 0.28–0.55 m of the
base, inside the 0.855 m reach envelope at comfortable heights. Per-object contact
strategy:

- **Cube:** 26 mm — a standard parallel-jaw pinch (80 mm stroke); seating it is a
  hover-and-release into the open 40 mm pocket (7 mm slack per side), exactly the
  motion solve certifies.
- **Spoon:** pinched at the raised 16 x 16 x 22 mm tail knob from above (16 mm
  across the 80 mm jaw; no under-object clearance needed — the knob stands proud on
  the handle). Loaded weight ~94 g — trivial payload. The rimmed pocket keeps the
  cube seated during a slow level carry; placing on the crest is a low hover +
  release with a ±10 mm lateral window, comfortably inside closed-loop arm
  precision (and the window's reality is exactly what the task tests — the arm
  needs the right TARGET, which is the computed balance point, not fine motor
  skill).
- **The stand:** never needs to be touched.

## Execution-order declaration

NO required execution order. The rubric judges only the settled terminal state (plus
latched partial credit); in practice the cube must be seated before the perch
(placing a cube into the pocket of an already-balanced spoon would upset it — but if
a solver manages it, the rubric accepts it; nothing procedural is checked).

## Checks (smoke.py — rejection battery, recorded, `SIM_GEN_SMOKE: ALL PASS 14/14`)

1. Settle/no-NaN: spoon flat on the floor, cube on the floor, stand upright,
   score ~0, no success.
2. Authored-mass readback: spoon/cube match the cfg-derived masses via get_masses();
   stand 6 kg.
3. Randomization readback (8 seeds): stand xy + yaw, spoon xy + free yaw, cube xy
   all vary.
4. Null policy (240 steps): score ~0, no success.
5. Seed-strategy analog: the spoon carried and SET DOWN flat on the slab beside the
   fin — score ~0, no success.
6. Settle gate: the empty spoon released above the crest is refused mid-fall
   (velocity readback).
7. Empty-spoon balance (positive control): it settles LEVEL on the crest at its own
   CoM — level/perch readback True — but the pocket clause alone rejects; ~0.
8. Load only: cube dropped into the pocket on the floor — `loaded` latches 0.20 and
   nothing more.
9. Latched credit survives removing the cube from the pocket.
10. Naive midpoint perch: supporting the geometric centre (balance point ~48 mm
    away) TIPS; ≤ 0.55, no success.
11. Near-miss: the composed balance point placed 8 mm OUTSIDE the 10 mm window tips
    — the window is a real physical boundary.
12. Wrong place: cube on the HANDLE, assembly perched at its true composed balance —
    settles LEVEL but in_pocket rejects; ~0.
13. Rejection audit: success() never True at any judged point in the battery.
14. Final no-NaN; frames.npz saved.
