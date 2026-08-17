# place_sphere_i196 — `drain_plug`

Seat the seed's own 60 mm red sphere over the 55 mm drain gap of a tilted hopper
floor FIRST — the sphere is a VALVE, not a payload — then pour 2–4 white marbles
in; each rolls down the funnel and collects against the seated ball. Any marble
committed before the drain is plugged rolls down the same fall line, drops
through the gap into the sealed vault under the false floor, and is lost forever
— success becomes permanently impossible.

- **Env name:** `simgen.drain_plug` (robot `"null"`, scene-level task)
- **Package:** `scene.py` (scene + rubric), `solve.py` (legitimacy certificate),
  `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `maniskill/place_sphere`
(`RoboVerse/roboverse_pack/tasks/maniskill/place_sphere.py`): a Franka grasps a
30 mm-radius red sphere, carries it, and balances it ON TOP of a small shallow
bin; a `RelativeBboxDetector` box above the bin judges the placement. One grasp,
one carry, one precise release — placement precision IS the task.

## What changed and why it is strategically different

| Axis | Seed | This task |
|---|---|---|
| Sphere's role | payload — the judged object placed at the goal | **tool/valve** — the same 60 mm red sphere is the only body that can seal the 55 mm drain; the judged outcome is mostly about the OTHER objects (marbles) |
| Placement precision | the whole task (balance on a shallow bin) | **deliberately zero** — anything dropped anywhere on the 8°-tilted floor is funnelled by gravity to the drain corner; no precise pose exists anywhere in the plan |
| Strategic content | none (single atomic pick-place) | **irreversible ordering**: plug BEFORE pour. A marble sent in early drains into a sealed vault and is unrecoverable — the task has a permanent failure absorbing state |
| Object count | 1 sphere | 1 ball + K marbles, **K ∈ {2,3,4} varies per episode** (count them) — the goal itself must be read from the scene |
| Scene reading | fixed bin pose | **free hopper yaw (±180°)** + xy jitter: the drain corner's bearing must be read from the floor tilt every episode |
| Failure modes | drop / misplace (retryable) | **irreversible loss** through the drain; success requires every present marble contained |

The size relation does the sorting: marble Ø25 < gap 55 < ball Ø60. The seed's
literal strategy — set the red sphere down on a shallow bin — is reproduced in
the scene (the TRAY is that bin) and scores exactly 0 (smoke check 9).

## Why it is different from the rest of the corpus

Nearest neighbours checked and the discriminating feature:

- **stopper_vault (`i178`)**: also has a plug and a vault — but the order is
  INVERTED: there the stopper must be REMOVED to release contents; here the plug
  must be INSTALLED before anything is committed, and the plug is the seed's own
  judged sphere. No funnel, no per-episode K.
- **ledge_catch / hatch_shelf / checker_silo / ice_doser**: sorting or routing by
  aperture exists, but none has an irreversible pre-commit ordering where the
  blocking body must be EARNED into place by gravity rolling (not held), and none
  varies the success cardinality K per episode.
- **roll_ball_i81 (arch_quarry)**: ordering task (demolish then lift) — but the
  order is enforced by force-closure of an arch; here it is enforced by
  irreversibility (a sealed one-way drain), and nothing needs demolition.
- **basketball_in_hoop_i128 (crater_run)**: gravity as adversary up a ramp; here
  gravity is the ALLY that does all transport — the only decision is WHEN each
  body is allowed to fall.
- **hockey_i325 / bowling_i17**: aim/launch precision toward a goal; here there
  is nothing to aim — every trajectory ends at the drain corner by construction.

No corpus task has (a) the seed's judged object role-inverted into a valve,
(b) an irreversible order-before-anything gate through a sealed one-way drain,
and (c) per-episode variation of the success cardinality (K marbles).

## Scene summary

Kinematic **hopper** at ~(0.58, 0) ± 2.5 cm, yaw FREE ±180°: four walls (ground →
280 mm rim, so the under-floor volume is a sealed vault) around a 300×300 mm
inner square; a bone-white false floor of two coplanar plates tilted 8° about
the (1,−1) diagonal (top surface plane z = z_mid + s·(x+y), s ≈ 0.0994; low
corner 160 mm, high corner ~220 mm), leaving one 55×55 mm square DRAIN GAP at
the low corner (−h, −h). Kinematic **tray** (shallow open bin — the seed's own
goal geometry) on the ground at x ∈ (0.02, 0.12), |y| ∈ (0.34, 0.44), side
coin-flipped, yaw free, holding the red **ball** (r = 30 mm, 100 g) and K live
**marbles** (r = 12.5 mm, 20 g); absent marbles are parked in a walled depot far
outside the workspace. Spheres carry explicit damping (marble 0.40/0.90, ball
0.15/0.45) as the rolling-resistance stand-in PhysX lacks — without it a marble
oscillates in the corner pocket for tens of seconds.

Randomised per seed (readback-verified in smoke): hopper yaw ±180° + xy jitter,
tray band/side/yaw, K ∈ {2,3,4} (drawn via `torch.rand`, not the degenerate
first `randint`), per-object slot jitter.

Geometry proved in `__post_init__`: marble passes the gap with ≥6 mm slop, ball
cannot (≥4 mm interference); the seated ball's centre (z_seat ≈ 0.185, corner
distance r√2 ≈ 0.042) lies inside the seat window (z ∈ (0.150, 0.192), plan
< 0.075) while a ball resting ON the plates anywhere in that plan window sits
≥ 0.196 — the sunk/unsunk discrimination is the z-ceiling; the seated ball
SEALS (max leak sphere radius ~5.1 mm < marble 12.5 mm); a drained marble rests
~150 mm BELOW the floor plane; the parapet retains settled marbles.

## Rubric

- `success()`: ball inside the seat window LIVE + plug latch earned, AND for
  every present marble: inside the hopper resting on/above the floor plane LIVE
  + bead latch earned; everything slow. Settling is PROVED by the latches
  (strict 0.05 m/s held for 24 steps — a rolling transit through a window earns
  nothing, and a slow pocket oscillation cannot latch at its turnaround); the
  live gate uses a looser 0.15 m/s so the bounded PhysX sharp-edge phantom
  oscillation of the seated pile (measured peak ~0.09 m/s, self-resetting)
  cannot flip a settled outcome, while genuine motion (≥0.3 m/s for anything
  actually rolling in this funnel) is still rejected.
- `score()`: 0.35·plug + 0.50·(latched beads / K), latched and monotone, bead
  latches gated on the plug latch (the physical order), capped at 0.85; exactly
  1.0 iff `success()`. Null policy scores ~0 (everything starts in the tray).

## Teleport-solution outline (`solve.py`)

Teleports are zero-velocity HOVER RELEASES only — every seat and every
containment rest is earned by gravity + rolling contact; no pose write ever
lands a body in a scoring region (the freshly written states are airborne and
satisfy no clause):

1. **P0** settle + layout readback (hopper pose/yaw, drain-corner world position,
   K); assert baseline score ≈ 0.
2. **P1 plug**: the ball is released a few mm above the floor plane ~12 cm
   upslope of the corner; it rolls down the fall line and seats itself in the
   gap (readback z ≈ 0.186 vs derived 0.185). `SIM_GEN_SCORE 0.35`.
3. **P2 pour**: each live marble is released at its own upslope point (staggered
   perpendicular to the fall line); each rolls down and settles against the
   seated ball. Score climbs 0.35 + 0.5·j/K per marble.
4. **Deep settle** (≤5 s): wait for max body speed < 0.03 m/s for 30 consecutive
   steps, then **P3 persistence**: 3.5 s fully hands-off with per-block success
   telemetry; `SIM_GEN_SOLVE: SUCCESS` only if success() held.

The ORDER is load-bearing: `smoke.py` proves the converse (a marble released
before the plug drains into the vault and is lost — from either entry point).

## Embodiment argument (single Franka + parallel jaw 80 mm, OSC)

Plausible base pose: world origin, facing +x. Tray at ≤ 0.46 m, hopper rim at
0.43–0.73 m — both inside a 0.75 m reach envelope (drop points only need to be
over the NEAR side of the hopper: gravity routes from anywhere on the floor).

- **Red ball (Ø60 < 80 mm jaw)**: top pinch grasp out of the open tray (walls
  25 mm — no approach constraint), carry over the hopper rim (280 mm — an easy
  clearance for a wrist-down carry at ~0.4 m), open the jaw anywhere above the
  false floor. No placement precision required — the funnel does the rest.
- **Marbles (Ø25)**: same pinch-and-release, one at a time, in any order among
  themselves — only AFTER the ball is seen seated (the seat is visible from
  above through the open hopper top).
- **Fixtures (hopper/tray/depot, kinematic)**: never manipulated.
- Nothing requires two hands, regrasp, or in-hand manipulation; every contact is
  a vertical pinch + a release over an open top.

## Execution-order declaration

Plug-before-pour is *physically forced*, not convention: a marble committed to
the floor before the plug exists ends in the sealed vault (proved physically in
smoke checks 6–7, non-vacuity control in check 8), and the vault has no other
opening — the loss is permanent, so no reordering can reach success. The rubric
mirrors the physics: bead latches are gated on the plug latch.

## Execution order (how the package was built)

1. `scene.py` first — geometry derived and proved in `__post_init__`.
2. `solve.py` iterated on the forge until `SIM_GEN_SOLVE: SUCCESS` on seeds
   0/1/2 (iteration: sphere damping added as rolling-resistance stand-in after a
   persistence flicker from undamped pocket oscillation on seed 2).
3. `smoke.py` rejection battery on the forge, frames.npz recorded.
4. `TASK.md` + final clean runs.

## Check list (smoke, 19/19)

1. settle/no-NaN + layout sanity (ball + K marbles at rest in the tray, absent
   marbles parked in the depot, all still)
2. score ~0 at reset, no success
3. randomization readback (8 seeds): hopper yaw spans a wide arc (−111° … +139°
   observed), xy jitter real, tray side flips
4. randomization readback: K varies across seeds (2, 3 and 4 all observed)
5. null policy: 240 idle steps → score ~0
6. OPEN DRAIN (order violation, physical): marble released with no plug ends in
   the VAULT (fixture z = 0.013 ≈ 0.16 below the plane), containment never
   latches, score 0
7. funnel: a release near the HIGH corner also drains (every entry is routed)
8. control setup (non-vacuous): plug seated by genuine rolling (readback
   z = 0.186 vs derived 0.185)
9. control: the SAME release now stays contained and latches — the open drain,
   not the probe, caused the loss
10. seed strategy: ball placed at rest in the shallow TRAY (the seed's own
    success geometry) → 0
11. wrong place: ball on open ground + marble grounded outside → 0
12. unsunk near-miss: ball resting ON the plate INSIDE the seat plan window
    (state witnessed frame-by-frame) rejected by the z-ceiling alone
13. unsunk near-miss control: the same ball then rolls in and latches only once
    genuinely sunk + settled
14. lost-marble setup: plug seats by genuine rolling
15. lost marble: plug seated, one marble at rest in the vault, the rest
    genuinely contained → success False, score ≤ plug + (K−1)/K beads (0.725
    at K = 4)
16. latched-credit setup: plug seats by genuine rolling
17. latched credit survives yanking a contained marble to the depot
    (0.475 → 0.475)
18. rejection audit: success() never True at any judged point
19. final no-NaN; frames.npz saved (334 × 600 × 960 rgb)

## Validation evidence (forge, RTX-4090 pod)

- `solve --seed 0`: SUCCESS, 18.7 s — `SIM_GEN_SCORE` 0.00 → 0.35 → 0.52 →
  0.68 → 1.00 → 1.00 (K = 3).
- `solve --seed 1`: SUCCESS, 18.5 s (hopper yaw +140.5° vs −36.3° at seed 0 —
  the bearing must be read; both seat readbacks z = 0.186).
- `solve --seed 2`: SUCCESS, 18.3 s (K = 2 branch; persistence telemetry shows
  the bounded edge-contact oscillation peak 0.086 m/s < judge gate 0.15,
  success held through all 10 blocks).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 19/19`, 44.9 s, frames.npz saved.
