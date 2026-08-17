# robosuite_env_i175 — Counterweight Balance

**Env:** `simgen.counter_balance` (scene `counter_balance`, robot `"null"`)
**Seed:** `robosuite/robosuite_env` (RoboVerse `roboverse_pack/tasks/robosuite/robosuite_env.py`)

## What the task is

A beam balance stands on a table: a steel beam on a ±12° centre pivot between
two uprights, a pan HANGING from each end (36 cm apart), a heavy bob under the
pivot, and a red needle sweeping over tick marks on the base (white = level,
amber = the ±3° pass band). One pan holds a RED CARGO cylinder whose mass is
**sampled per episode** (2..6 units — readable from its height, 15 mm per unit);
**which pan** carries it is also sampled. The beam rests pinned against its stop,
tipped toward the cargo. On the table lie three BRASS counterweights of 1, 2 and
4 units (readable from diameter: 24 / 34 / 48 mm), scattered over permuted,
jittered slots.

Goal: place counterweights summing EXACTLY to the cargo's mass into the EMPTY
pan — and nothing anywhere else on the scale — so the beam settles level inside
the amber band, everything at rest, hands off. The cargo must stay seated in
its own pan; no counterweight may ride with the cargo or sit on the beam;
unused weights and spare cargos stay off the scale.

**Execution order: NONE required.** The chosen weights may go into the pan in
any order (the solve does largest-first purely as a convention); every order
ends in the same equilibrium.

## Seed provenance and why this is strategically different

The seed wraps the five robosuite oracle tasks (Lift, Stack, Door, PickPlaceCan,
NutAssemblySquare). Every one of them — and their oracle plans — has the same
shape: **the manipulated object's own pose IS the success predicate** ("move THE
object to ITS goal region": lift it above height h, stack A on B, put the can in
its bin, thread the nut on its peg, swing the door past an angle). This task
breaks that shape on every axis:

- **No placed object has a goal pose.** The judged observable is the
  *mechanism's* equilibrium angle — an emergent function of total mass aboard
  each pan, produced by the pivot + hanging-pan linkage after release. Weight
  position inside the pan is deliberately worthless: a hanging pan swings until
  its load acts through the hang point, so ANY resting spot in the pan gives
  exactly the same lever arm (that is why real balance scales hang their pans).
  Centimetres of placement slop cost nothing; one unit of *mass* anywhere is
  fatal.
- **The hard part is discrete reasoning, not motion.** The policy must *read*
  the cargo's mass off its height, decompose it over the {1, 2, 4} inventory
  (the binary decomposition is unique), and load exactly that subset. A
  memorized motion can never work: subset AND side change per episode.
- **Exactness is two-sided.** Undershoot and overshoot fail identically (one
  unit off rests the beam at ~7.7°, far outside the 3° band) — unlike every
  seed predicate, which is a one-sided threshold (higher, inside, past).
- **The seed's own strategy is a tested negative**: smoke check 4 carries the
  cargo (THE object) off the scale — the beam physically relevels, and it earns
  ~0, because success demands the cargo stay aboard and the counter-pan
  inventory match it.

It is also deliberately unlike the corpus tasks read while building it: no
setpoint regulation of a grasped DOF (press_switch_i161 servos dials to sampled
bearings — continuous control; here nothing is regulated, physics settles a
*chosen* discrete load-out), no containment/pick-place geometry (pen_holder),
no interlock/extraction (lamp_off_i101).

## Mechanics

Plain rigid bodies + authored USD D6 joints (the proven pattern): post→beam D6
frees exactly rotY within ±12° (the end stops the cargo pins the beam against);
beam→pan D6s free rotY within ±25° so each pan hangs plumb. The beam's authored
MassAPI puts 2.2 kg at 91 mm below the pivot — the bob IS the restoring spring
(K = M·|com|·g ≈ 1.96 N·m/rad), so one unit of imbalance (0.15 kg × g × 0.18 m
lever) rests at atan(0.265/1.96) ≈ 7.7° ≫ tol 3° (measured 7.0–8.0° on the
forge). Pans hang 120 mm below the hang points: crossbar-to-pan-floor clearance
109 mm clears the tallest cargo (90 mm) — at 70 mm the 6-unit cargo wedged
against the crossbar and drove a permanent depenetration limit cycle (found and
fixed on the forge). Reset writes the whole linkage at its PINNED equilibrium
(beam at the stop toward the cargo side, pans plumb under the tilted hang
points) so nothing slams; free props carry real angular damping so drop
transients die instead of ringing. Pure passive physics — no wrench interface,
no post_step drives; the only input is where bodies get placed.

## Rubric (score 0..1, anchored in the solve trajectory)

- **0.25 · A_latch** — some counterweight rests in the counter pan while the
  scale is quiet and the cargo is seated (the first load-bearing drop).
- **0.35 · B_latch** — best QUIET levelness progress `1 − |tilt|/12°`, gated on
  cargo seated + weights only in the counter pan + spares clear; latched only
  through a 30-substep quiet streak (swing apexes latch nothing — velocity
  passes zero at every turning point).
- **1.0 iff success()** — level (±3°) ∧ quiet ∧ cargo seated ∧ every weight in
  the counter pan or off the scale ∧ spares clear ∧ counter-pan mass matches
  the cargo to half a unit (integer units ⇒ only the exact subset passes; also
  closes the "prop the beam level by hand" exploit).

Null policy = 0.000 (beam rests pinned, no latch arms). One-unit near miss ≈
0.37–0.40. Latched credit survives revocation (smoke 10: score 0.600 after
success is revoked live).

## Solution outline (solve.py — teleports are TRANSPORT ONLY)

Each chosen weight is teleported to a hover point 10 mm above its seat in the
counter pan (computed from the pan's LIVE pose readback) and released with zero
velocity — the drop, the pan swing, and the beam's rotation to equilibrium all
flow through the passive D6 + gravity plant. Nothing ever places or touches the
beam. Read k off the scene → unique binary subset {2:[2], 3:[2,1], 4:[4],
5:[4,1], 6:[4,2]} → drop largest-first, waiting out the quiet streak between
drops → success matures → ≥3.5 s hands-off persistence → `SIM_GEN_SOLVE:
SUCCESS`. Verified on forge seeds 0/1/2/3 (k = 3, 6, 4, 6; both sides; final
tilts 0.15–0.34°; score exactly 1.000).

## Embodiment argument (single Franka, parallel jaw)

- **Base pose:** on the table at ~(0.0, −0.46, 0.40), facing the scale (+y).
  Reach: weights at y = −0.28 (0.20–0.35 m), pans at (±0.18, 0) (~0.49 m) —
  all inside a comfortable envelope, no obstruction between.
- **Grasps:** every manipulated object is an upright cylinder, dia 24–48 mm ≪
  the 80 mm jaw, 30 mm tall, standing free on the table with ≥ 60 mm of clear
  air (slots 150 mm apart, jitter ±30 mm). Plain top-down grasp.
- **The drop:** the pan mouth is a 78 × 158 mm opening whose rim top sits at
  z ≈ 0.53, with open sky above except the 30 mm-wide beam strip along y = 0 —
  approach over the pan's ±y half (the seats used by the solve are at pan-local
  y = ±40–48 mm) and release 10–20 mm above the rim. Required precision:
  land anywhere inside the pan — centimetres of slop, by design.
- **Perception:** cargo mass = height (90 vs 30 mm across the range — coarse),
  weight identity = diameter (24/34/48 mm); or close the loop on the needle:
  the beam always tips toward the heavier side, and the amber ticks mark the
  pass band.
- **Forces:** heaviest weight 0.6 kg ≪ payload; nothing is pushed or held —
  the scale does the measuring after release.

## Randomization (readback-verified)

Per episode: cargo mass k ~ U{2..6} (five distinct cargo cylinders; the active
one seated, spares parked in a ground depot), cargo side ~ U{L, R}, weight
scatter = permuted slots + uniform jitter. Sampling uses `torch.rand` only
(first-`randint` degeneracy on this stack). Smoke readback: cargo physically
seated in the sampled pan on every draw; k, side and slot permutation all vary.

## Checks (smoke.py — 11, frames.npz recorded)

1. **settle** — clean reset: finite, beam physically RESTS pinned toward the
   cargo side (~12°), cargo seated, authored beam mass took (get_masses
   readback = 2.200), score < 0.05, no success.
2. **random** — 8 seeds: ≥3 distinct k, both sides, ≥3 distinct slot
   permutations, cargo seated readback on every draw.
3. **null** — 2.5 s of nothing: beam stays pinned, score < 0.02, no success.
4. **negative (seed strategy)** — carrying the CARGO off the scale relevels the
   beam PHYSICALLY (|tilt| < 3°) and earns ~0.
5. **negative (under)** — subset for k−1: settles ~7° toward the cargo, no
   success, score 0.20–0.55.
6. **negative (over)** — subset for k+1: settles ~8° the OTHER way — overshoot
   fails like undershoot.
7. **negative (split pans)** — extra weight riding WITH the cargo + heavier
   subset opposite: beam levels PHYSICALLY, still fails (brass only in the
   counter pan).
8. **negative (wrong object)** — spare cargo cylinder as makeweight: levels
   PHYSICALLY, still fails (spares must stay off the scale).
9. **exactness** — the exact subset → success() and score == 1.0, still true
   1 s later hands-off.
10. **latch** — plucking one weight back off revokes success (live state);
    latched credit remains (score 0.600).
11. **frames** — ≥ 20 video frames saved to `frames.npz`.

## Files

- `scene.py` — cfg + scene + registration (`counter_balance`, `simgen.counter_balance`)
- `solve.py` — `python -u -m simgen_tasks.robosuite_env_i175.solve --headless [--seed N]`
- `smoke.py` — `python -u -m simgen_tasks.robosuite_env_i175.smoke --headless`
