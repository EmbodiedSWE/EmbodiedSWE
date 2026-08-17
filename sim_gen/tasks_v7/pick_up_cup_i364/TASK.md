# pick_up_cup_i364 — Cooper's Curve

**Env:** `simgen.cooper_curve` · **Scene:** `cooper_curve` · **Robot slot:** `null`
(scene-level task; bodies driven through scene handles)

## Seed provenance

Derived from **rlbench/pick_up_cup**: one free cup and a distractor cup on a table;
the robot grasps the target cup and lifts it straight up; judged by a height
condition on the grasped cup.

Kept from the seed: a free "cup" object, a look-alike distractor, and the
identify-then-manipulate structure. Replaced: everything about the manipulation
model and the judgment.

## Strategic difference argument

The seed's whole skill is *grasp-and-raise*. Here **nothing is lifted to succeed
and grasp-and-raise is worth zero** — the roof of the delivery gallery (underside
115 mm) physically refuses a lifted/upright cup (standing height 185 mm) while
admitting only a body rolling on its side (rolling profile 90 mm). The replacement
skill is **taper-steered rolling**: the cup is a conical wheelset (base rim r=30 mm,
mouth rim r=45 mm, 110 mm apart) that *cannot roll straight* — rolling on its rims
it orbits the apex of its own cone, 220 mm beyond the base rim. The 150° annular
gallery between wall radii 155/420 mm has exactly that curvature, so the only way
through is to stage the cup at the gallery mouth with its base toward the arena
centre (apex on the arc centre) and launch it tangentially: **the object's own
geometry is the steering law**. The equal-rim decoy (r=37.5 mm both ends) rolls
dead straight and wedges on the outer wall at ~0.17 m of travel, 21 % of the way
to the bay — identifying the taper is load-bearing, not decoration (`__post_init__`
asserts the jam arithmetic; smoke check 7 demonstrates it live with the solve's own
launch push).

Against the corpus (~300 tasks): the rolling family is straight-line or
sphere-based — `roll_ball_i81` (arch-lock quarry, sphere through arches),
`roll_ball_i358` (windmill toll-gate, sphere timing), `cellar_roll`/`die_roll`
variants (straight chutes and tumbling), aimed-ballistic tasks (turret + gravity
chute). None makes the *cargo's own taper* define a curved path; none has a
curvature-matched corridor; none uses a straight-rolling twin as the physical
foil. `pick_up_cup_i76` (Bend Gallery) shares only the seed. The staging skill —
a *pose* (apex alignment within ±25°, apex-true radius ±45 mm), not a height —
and the launch-then-hands-off structure (all contact happens in the open pen; the
150° coast is untouched) have no corpus neighbour.

## Randomization (readback-verified)

Cup and decoy are permuted over two pen slots (azimuth −100°/−134°, radius
287.5 mm) each episode, with ±3° azimuth jitter, ±10 mm radius jitter and free yaw
per body; overlap-rejected sampling with a deterministic radial-yaw fallback.

## Teleport-solution outline (solve.py)

1. **P0** settle; readback layout + authored mass (0.25 kg) + score ~0.
2. **P1 TRANSPORT**: one root-state write carries the cup from its slot to the
   staging pose (lying on its rims, base toward centre, ρ = ρ_stage = 274.5 mm,
   φ = −89°…−78°, destination clearance-checked against the decoy's actual axis
   segment; if all candidates blocked the decoy is first carried to a clear park
   spot deeper in the pen). Gravity+contact settle produce the staged latch (0.10).
3. **P2 LAUNCH** (applied force, pen only): tangential velocity-servo wrench at the
   CoM (v_des 0.75 m/s, cap 1.6 N — under the μ·m·g = 1.96 N budget), CUT by an
   explicit zero-wrench write at azimuth −76.5°, before the roof at −75°.
4. **P3 COAST** (pure physics): the taper holds the cup on its ~274 mm circle
   through 180° of gallery (measured: ρ stays 0.269–0.274, v decays only
   0.44→0.40 m/s), through cp1/cp2/cp3 in order, into the bay; the restitution-0
   arrest wall stops it; it settles → success, score 1.0. If a coast stalls, the
   cup is transported back to staging and relaunched harder (escalating ladder,
   max 3.4 N) — retry, not cheat.
5. **P4** hands-off persistence 3.3 s → `SIM_GEN_SOLVE: SUCCESS`.

Measured on the forge: seed 0 first-try success in 18 s wall clock, single launch
attempt, no retries.

## Rubric (ordered latch chain, anchored in the demonstrated trajectory)

| credit | clause |
|---|---|
| 0.10 | `staged` — cup lying in the staging window, base toward centre, apex-true radius, calm |
| 0.20 | `cp1` — *then* crossed gallery azimuth −45° at floor level |
| 0.20 | `cp2` — *then* crossed 0° |
| 0.15 | `cp3` — *then* crossed +45° |
| 0.10 | `bay` — *then* entered the catch bay |
| 1.0 iff `success()` | full chain **and** cup at rest lying in the bay, decoy out of the bay, all finite; otherwise capped at 0.75 |

Execution order is **declared and enforced**: `staged → cp1 → cp2 → cp3 → bay`,
each latch requiring the previous (teleport-to-goal, out-of-order transit and
wrong-object transit all score ~0 — smoke checks 5, 6, 8).

## Embodiment argument (Franka)

- **Reach:** the start pen (azimuth −160°…−75°) is open-topped, 300 mm walls; a
  Franka based just outside the pen's outer wall (base ~55 cm from the arena
  centre at azimuth ≈ −117°, i.e. behind the pen) reaches over the wall to
  everything in the pen — both spawn slots and the staging window are within
  ~65 cm of that base. Nothing under the gallery roof (115 mm underside) is
  reachable, which is exactly the point: the task *forces* launch-and-coast.
- **Cup re-orientation/staging:** the cup masses 250 g; fingertip pushes on the
  mouth rim roll/pivot it across the slick pen floor into the staging pose (a
  ±25° alignment cone and ±45 mm radius band — coarse targets). The solve's
  transport teleport stands in for exactly this carry.
- **Launch:** a palm/fingertip push accelerating 250 g to ~0.5 m/s over ~5 cm
  needs ~1.6 N — trivially within Franka contact capability, and the push happens
  in the open staging window, fully reachable.
- **Restraint:** the decoy is simply left in the pen (or nudged aside within the
  pen), also fully reachable.

## Files & checks

- `scene.py` — procedural arena (static ring walls + roofed gallery + arrest
  wall), wheelset spawners with authored MassAPI/inertia/materials, 12
  geometry-honesty asserts in `__post_init__` (wall clearances, roof
  admits-rolling/refuses-carry, decoy jam arithmetic, slot/staging disjointness,
  weight sum).
- `solve.py` — teleport-transport + pen-only servo launch + hands-off coast;
  `SIM_GEN_SCORE` at phase boundaries (0.00 → 0.10 → 1.00 → 1.00), watchdog +
  hard exit.
- `smoke.py` — 12-check rejection battery: settle/no-NaN, 3-seed max-pairwise
  randomization readback, slot-permutation coverage, null policy, seed-analog
  (teleport-to-goal) refused, wrong-object course refused, decoy-cannot-turn
  (live launch, non-vacuous), out-of-order refused, near-miss (4° short) capped
  at 0.65, roof gate (6 N lift arrested, non-vacuous), restraint (decoy-in-bay
  refuses success at full 0.75 cap), frames.npz video.
