# scene_c_i171 — beam balance: count the paint bands, weigh out the unique counter-subset

## Seed provenance

Seed task: `calvin/scene_C` — the CALVIN play-table C: a Franka at a desk with
four independent articulated primitives (`base__button`, `base__switch`,
`base__slide`, `base__drawer`) and three colored blocks (pink/blue/red) on the
open tabletop. Every CALVIN-C objective is either "actuate one single-DOF
primitive to its other end-stop" or "pick up / push / stack a block sitting in
free space". Carried over here: a desk-scale scene with one articulated
single-DOF mechanism (a revolute), a handful of colored free blocks among
which the *right ones* must be selected, and pick-and-place as the motor
vocabulary.

## Strategic difference

**vs. the seed.** CALVIN-C's DOF goals are binary end-stop pushes and its
block goals are color-keyed but *mass-blind* — any block can be picked and any
placement pose works. Here the single DOF (a balance beam on a free revolute)
is never the thing you actuate at all: you cannot usefully touch the beam,
because success is a **continuous equilibrium** of that DOF — the beam must
*settle level hands-off* — and the only legal way to move it is to change the
mass it carries. The block skill is inverted from color-matching into
**perceptual counting + subset-sum arithmetic**: a cargo slab with k painted
bands (k ∈ 1..5, its mass is k×100 g) pins the beam on one hard stop, and the
scene provides exactly three candidate cubes of 1u/2u/4u (binary weights, so
every k has a **unique** answer subset). The solver must count the bands,
decompose k in binary, and load *precisely* that subset into the opposite
pocket — one cube too few and the beam never leaves the stop (settles ≥ 0.39
rad, tolerance 0.262), one too many and it slams through to the *other* stop.
No CALVIN objective involves counting, arithmetic, choosing a set of objects,
or an equilibrium-valued goal.

**vs. the rest of the corpus.** `scene_d_i130` (read in full as the
structural template) is aperture-gated extraction — rotate a hidden payload
under a window and lift it out; its judge is positional (cube on pad). Here
nothing is hidden, nothing is extracted, and the judge is the *angle of a
mechanism no hand may hold* — an emergent physical quantity determined by the
mass arithmetic. `libero_..._i88` is tool-mediated (bayonet key → drawer);
here there is no tool and no container. `pen_holder` (robobench exemplar) is
free-space pick-and-insert. No other task read has a balance, a
counting/arithmetic selection problem, or a success predicate that is an
equilibrium of an unactuated DOF. There is no stored energy: the beam is a
plain damped revolute between hard stops, the cubes and slabs free bodies —
every intermediate outcome persists hands-off.

## Scene (`beam_balance`, env `simgen.beam_balance`, robot="null")

Procedural geometry only (compound spawners; a per-env USD revolute joint
authored once at bind time; joint collision filtering applies only to the
tower↔beam pair, so every cube/slab contact — the actual load path — stays
live):

- **Tower** (kinematic): base slab + two pivot plates (x ∈ ±0.035,
  y = ±(0.020..0.038), z up to 0.36) straddling the beam's hub.
- **Beam** (dynamic, revolute about y at z = 0.35, hard stops ±0.50 rad,
  ang. damping 8, mass 1.3 kg): 0.89 m arm, a keel dropping the CoM 5 cm
  below the pivot (restoring stiffness m·g·|com_z| ≈ 0.64 N·m/rad), and two
  open-top pockets at x = ±0.40 — floor 8 mm thick, inner half-width 36 mm,
  walls up to 15 mm above the arm top (mouth 72 mm vs the 50 mm cubes).
- **Cargo slabs** `cargo_1..5` (dynamic): 0.11 m square slabs, j painted
  bands (alternating maroon/tan) and exactly j×100 g of mass. One rides a
  beam pocket at reset; the other four wait in a **depot** 2+ m away.
- **Candidate cubes** (dynamic, 50 mm): `cand_1` ivory 100 g, `cand_2`
  orange 200 g, `cand_4` charcoal 400 g — scattered on the ground near the
  tower (shuffled slots + jitter + random yaw).

Per-episode randomization (readback-verified): k ∈ 1..5, cargo side s = ±1
(the beam is *written at the corresponding stop* with the slab seated in the
dropped pocket, so frame 0 is already the pinned equilibrium), and the three
candidate ground slots are permuted and jittered ±3 cm with random yaw.

Geometry contract (~25 asserts in `__post_init__`): the angle ladder
theta_tol 0.262 < lift_theta 0.28 < 3-stack-prop angle 0.311 < wrong-by-one
settle ≥ 0.362 (even at max load stiffening) < stop 0.50; placement play in
the pocket (11 mm) perturbs equilibrium ≤ 0.8·tol; the null policy stays
pinned (τ_cargo_min > k_beam·stop); ground prop stacks of ≤3 cubes top out
below the pocket floor at lift_theta (no propping); a two-cube ground stack
under the pocket topples rather than props; pocket walls overtop a full
3-cube stack's CoM (bracing); subset-sum uniqueness (binary weights);
depot ≥ 1.5 m from the tower; cube ≤ 75 mm jaw fit; partial-credit weights
sum to the cap.

**Success** (live state, no memory): cargo seated in its pocket ∧ counter
pocket loaded ∧ every candidate legal (in the counter pocket or on the
ground — no arm-draping, tower-perching, or cargo-pocket stuffing) ∧ all
four spare slabs still in the depot (no slab counterweights) ∧ |tilt| <
theta_tol ∧ settled ∧ finite. **Score**: latched partial credit — `loaded`
0.20 (a candidate rides the counter pocket while the cargo stays seated),
`lifted` 0.25 (loaded ∧ legal ∧ |tilt| < 0.28 ∧ slow, sustained 30 steps),
capped at 0.45; exactly 1.0 iff success. Latches clear on reset.
Note (honesty): an over-by-one drop can transiently sweep through the level
band and latch `lifted` (0.45 cap) before slamming into the far stop — the
settled gate blocks success throughout, and the audit in smoke check 13
verifies success stayed False; the score ladder stays monotone either way.
Also, for k ∈ {1, 2, 4} the single-cube answer can reach success in one
drop, so the 0.20 print and the 1.0 print can coincide in one phase.

## Solution (`solve.py`) — teleports for transport only, zero applied wrench

1. **P0 settle + perception**: 150 steps; read k and s back from the LIVE
   state (never hard-coded); mass readback via `get_masses()` on the beam,
   all five slabs, and all three candidates guards the density-mass trap;
   assert the beam rests ON its stop on the cargo side, score ≈ 0.
2. **P1 load** (the permitted transport): decompose k into the unique
   subset of {1u, 2u, 4u} (binary digits) and, heaviest first, teleport each
   chosen cube to a release pose *inside the counter pocket's open mouth* —
   orientation-matched to the live (tilted) beam, 6 mm above its seat, zero
   velocity — and DROP it. Landing contact, stacking, and the beam's revolute
   response are all real dynamics. First drop fires `loaded` (0.20).
3. **P2 hands off**: the beam swings up off its hard stop and settles level;
   `success()` turns True on the live state (score 1.0) and holds through a
   **3.3 s hands-off persistence** window before `SIM_GEN_SOLVE: SUCCESS`.

The whole episode repeats on a second seed (fresh reset; SIM_GEN_SCORE stream
non-decreasing). Forge result: both seeds OK, rc=0, ~21 s — **first
submission, no iteration**.

## Franka embodiment argument

Base pose: one fixed base on the ground at ≈ (0.0, −0.52), facing the tower —
both pocket mouths (x = ±0.40, z ≈ 0.36–0.42 across the tilt range), the
candidate scatter (|x|, |y| ≤ 0.6 on the ground), and the tower are inside a
0.35–0.70 m reach annulus; the depot (2+ m away) never needs to be reached.

- **Candidate cubes** (50 mm, ≥ 100 g): canonical parallel-jaw pinch
  (50 mm < 80 mm jaw) from free ground space — reset scatter keeps them
  clear of the tower and of each other.
- **Counter-pocket drop**: the pocket mouth is 72 mm square vs the 50 mm
  cube — 11 mm of finger clearance per side; with the beam pinned at the
  stop the counter pocket is the *high* end (mouth at z ≈ 0.42, tilted
  ≤ 29°), so the cube is released from just above the mouth and the pocket
  walls act as a chute onto the seat. No insertion below the mouth plane is
  needed; later cubes stack through the same mouth.
- **Perception**: the k bands are 22 mm tall, high-contrast maroon/tan
  stripes on a 0.11 m slab riding the beam at z ≈ 0.30–0.45 — face-on to a
  wrist or shoulder camera; the three candidates are color-coded
  ivory/orange/charcoal, never distinguished by reach.
- **No beam interaction**: the beam must never be held — the rubric's
  settled ∧ hands-off gates make any steadying grasp useless, so the arm
  only ever touches free cubes in open space.

## Execution order declaration

Scene was designed first with the minimal success predicate and the full
geometry contract; `solve.py` was then run on the forge and passed both
seeds on the **first submission** (rc=0, ~21 s), with telemetry matching the
paper margins (under-by-one bound 0.362 vs observed 0.388; null pinned at
exactly the 0.500 stop). Only after the solve was demonstrated was
`smoke.py` written. Its first forge run scored 14/15: the rejection audit
caught the over-by-one probe passing through a **genuine success** — dropping
the k+1 subset heaviest-first means the first cube alone totals exactly k for
k ∈ {2, 4} (the binary subsets nest), so the beam legitimately balanced
mid-probe. The fix changed the PROBE, not the rubric: that check now drops
lightest-first, whose prefixes only ever total 1u. No check was weakened at
any point to make a run pass; the rubric is byte-identical before and after.

## Smoke battery (`smoke.py`) — 15 checks, all rejection/health

1. **settle/no-NaN** — beam pinned on the cargo-side stop, cargo seated,
   counter pocket empty, spares in depot, score ≈ 0, no success.
2. **randomization A** — k and side vary across 10 seeded resets and the
   live state (beam tilt sign, seated slab identity) tracks the sample.
3. **randomization B** — candidate ground slots vary (std > 2 cm) and the
   live bodies track the sampled scatter.
4. **null policy** — 300 idle steps: beam stays pinned at the stop,
   score ≈ 0.
5. **under-by-one** — subset for k−1 dropped into the counter pocket: the
   beam lifts off the stop but settles ≥ tol+0.05 on the cargo side; loaded
   fires, `lifted` never does, score ≤ 0.20 band, no success.
6. **over-by-one** — subset for k+1 dropped (lightest first, so no
   intermediate stack totals exactly k): the beam slams through level to the
   *far* side (tilt sign flips), never settles level, no success.
7. **seed reflex (end-stop push)** — the whole beam+cargo assembly written
   at the OPPOSITE stop (the CALVIN move: shove the DOF to its other
   end-stop): physics returns it to the cargo-side stop, score ≈ 0.
8. **wrong pocket** — the correct subset dropped ONTO the cargo slab in the
   cargo pocket: beam stays pinned, counter pocket never loads, no success.
9. **spare-slab cheat** — a depot slab teleported into the counter pocket
   plus a deliberately light cube subset: the beam levels and `loaded` is
   genuine, but `spares_in_depot` fails — score capped at 0.45, no success.
10. **cargo removed** — the cargo slab teleported off to open ground: the
    empty beam self-levels, but `cargo_seated` fails — score ≈ 0, no
    success (no empty-beam leveling).
11. **near-miss angle** — full correct final assembly written at
    tilt = tol+0.03: every clause holds except `level` — judged without
    stepping, then removed; no success.
12. **level-but-moving** — the correct final assembly written level with
    beam ω = 0.5 rad/s: the settle gate refuses success (judged without
    stepping, then removed).
13. **rejection audit** — `success()` observed False at every step of the
    entire battery.
14. **final no-NaN.**
15. **video** — frames.npz (rgb frames, 960×600) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, ~92 s), 345-frame
video recorded. The solve was re-run clean after the final submission.
