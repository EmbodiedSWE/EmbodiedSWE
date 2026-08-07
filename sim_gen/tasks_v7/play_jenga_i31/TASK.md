# `play_jenga_i31` — Set the Compass Dials (scene `compass_dials`)

Env id: `simgen.compass_dials` · robot: `null` · files: `scene.py`, `solve.py`, `smoke.py`.

## Seed provenance

Seed task: **`rlbench/play_jenga`** (`roboverse_pack/tasks/rlbench/play_jenga.py`): a tower
of 14 identical jenga cuboids plus one `target_cuboid`; the goal is to pull the target
block out of the tower without toppling the rest — a free-body **extraction** under a
stability constraint, judged (implicitly, via the demo trajectory) on **positions**.

What is inherited: the jenga **bar** as the manipulated body (a flat, light, elongated
block, here reshaped into an arrow so its heading is well-defined), and the seed's core
**skill** — fine, disturbance-sensitive nudging of a light bar that must NOT be knocked
out of its place.

## What changed, and why it is strategically different

- **Extraction inverted into captivity.** The seed's whole plan is "slide the bar out and
  carry it somewhere". Here every bar is CAPTIVE on a fixed vertical pivot pin through a
  square hole in its body (a free pivot built purely from contact — no articulation), and
  taking a bar off its pin is exactly what the rubric rejects: `on_pin()` gates every
  credit term. The seed's strategy, executed perfectly (arrow laid down pointing exactly
  at its post), scores **zero** — smoke check 6 constructs precisely that end state.
- **Position goal replaced by a pure ORIENTATION goal.** Nothing is transported; the
  judged quantities are three azimuth headings (arrow tip vs. pin→post direction), each
  bound by COLOR to a randomized target post, plus a yellow decoy that matches no arrow.
  None of the 33 corpus tasks read during design judges headings of pivot-captive bodies:
  the corpus covers extraction, stacking, counterweights, insertions, twist-locks,
  pouring, herding, projectile toppling, sliding puzzles, mechanism releases, ramps,
  drop-order silos, hanging, die tipping (a discrete face-up goal reached by tipping over
  edges — not a continuous heading on a pivot), carousel feeding (rotation as a MEANS to
  transport — here rotation IS the goal), and plank bridging.
- **Different plan and code structure.** A solver must perceive three pin→post azimuths,
  swing each arrow about its pivot with tangential nudges working against pin contact and
  bench friction, stop inside an angular tolerance, and disturb nothing off its pin. The
  rubric is a per-dial heading/angle rubric with latches — not position-of-a-block checks.

## Scene (fully procedural, no external assets)

Kinematic bench slab (0.96 × 0.48 m, top at 0.10 m); three kinematic steel pivot pins
(18 mm dia, 55 mm tall) in a row; three dynamic ARROW bars (red/green/blue, 14 mm thick,
100 g, square 28 mm pivot hole → 5 mm radial play, arrowhead tip 110 mm from the pivot,
short blunt tail); three kinematic colored posts (30 mm dia, 120 mm) each 0.19 m from its
same-colored pin at a randomized azimuth, plus one yellow decoy post. Custom compound
spawner (boxes with per-prim transforms + collision APIs, contact offset 1 mm so the
5 mm pivot play is real).

**Randomization per episode** (verified by readback in smoke): pin xy jitter (±15 mm),
each post's azimuth in a mirrored window (±50–130° off the row) rejection-sampled so
from EVERY pin the own-post vs. any-other-post/decoy separation is ≥ 25°, the decoy's
azimuth, and each arrow's spawn heading (forced ≥ 25° from its target).

**Geometry honesty, asserted in `__post_init__`:** arrows spin with real play; posts
stand beyond the arrow sweep (an arrow can never touch a post); adjacent sweeps disjoint
under worst-case jitter; a window-edge post stays clear of the neighbouring sweep; spawn
separation ≫ tolerance (null policy ~0); wrong-post separation ≥ 2·tol + 5° (wrong-color
rejection is always well-defined); everything on the bench.

## Rubric

- `aligned()[k]`: tip heading within 10° of the pin→own-post azimuth AND on-pin (hole
  centre within 12 mm of the pin axis, resting in the on-bench height band, flat within
  12° of world-up).
- `success()`: all three dials aligned simultaneously AND everything still.
- `score()`: per dial, latched — 0.10 × best on-pin-gated fractional heading progress
  (normalized by the episode's own spawn error) + 0.20 × ever aligned-and-still; capped
  at 0.90; exactly 1.0 iff `success()`. Null policy ~0; off-pin, flipped (tail toward
  post), wrong-color and decoy aims all earn nothing for that dial.

**Execution order: NO required order** — the three dials are independent and may be set
in any order (solve.py happens to go red→green→blue; the rubric never checks order).

## Teleport solution (`solve.py`) — nothing is teleported

Every arrow already sits captive on its pin after reset; there is no transport phase at
all, so the solution uses **zero teleports** — the entire task runs through contact
dynamics:

- **P0** settle 60 steps, print layout readback (targets, spawn errors, pin/post xy) so
  distinct seeds are provable from stdout; baseline `SIM_GEN_SCORE`.
- **P1–P3** per dial: a floating-hand **torque about world +z** (the tangential nudge a
  fingertip on the shaft applies) spins the arrow about its pin — pin-hole contact
  constrains the swing, bench friction is what the controller works against and what
  finally holds the heading. The regime is friction-dominated, so the controller is
  **pulse-and-coast creep**: drive pulse only while the swing rate is below an
  error-scaled cap, coast otherwise (friction brakes almost instantly), escalate the
  pulse torque only if the arrow provably is not creeping (static friction unbroken),
  brake + cut the wrench inside the stop band, then 90 hands-off steps so real friction
  holds the set heading before it is judged. The +z torque axis coincides with the swing
  axis, so the actuator's rotate-with-the-body wrench frame drag cannot corrupt it.
  `SIM_GEN_SCORE` after each dial; asserted non-decreasing.
- **P4** persistence: ≥ 3.3 simulated seconds fully hands-off with `success()` required
  to hold throughout; `SIM_GEN_SOLVE: SUCCESS` only then.

Verified on the forge: **seed 0 and seed 1 both end `SIM_GEN_SOLVE: SUCCESS`** with all
dials at ~1.1° error and monotone scores 0.000 → 0.299 → 0.598 → 1.000.

## Embodiment argument (a real robot could do this)

- **Franka base at (0.0, −0.50, 0.0)** facing the bench: the farthest interaction point
  (blue dial tip region, ≈ (+0.41, +0.20, 0.11)) is ≈ 0.75 m away — within a Franka's
  ≈ 0.85 m reach envelope; all three dials sit in a 0.82 m row directly in front.
- **Bench height** raises all contact to z ≈ 0.10–0.12 m, so pushes are comfortable
  side-on fingertip contacts, not near-ground scraping.
- **Per-object contact strategy:** each ARROW is spun by pressing a closed fingertip
  laterally against its shaft/head/tail (14 mm thick, 24 mm wide — a natural fingertip
  target) and sweeping tangentially; the 100 g bar needs ~0.03–0.3 N·m about the pin,
  i.e. fingertip forces of a few newtons at an 8–11 cm lever arm. PINS and POSTS are
  never manipulated (kinematic scenery); the tolerance of 10° = ±19 mm of tip arc is far
  coarser than arm repeatability. Multiple approach directions exist for every dial at
  every heading, so no configuration is unreachable.

## Checks (smoke.py, forge: `SIM_GEN_SMOKE: ALL PASS 14/14`, frames.npz recorded)

1. settle/no-NaN: reset finite, every arrow seated on its pin, still.
2. score ~0 at reset, no success.
3. randomization readback: post azimuths + arrow spawn headings vary over 6 seeds.
4. randomization readback: pin xy jitter varies; invariants (posts on ring, spawn error
   ≥ init_sep) hold.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: all three arrows OFF their pins, laid pointing perfectly at their own
   posts → heading within tol yet zero credit, no success (off-pin gating). Probe
   geometry margins: the 45 mm lateral offset clears the arrow's own pin (eye outer half
   28 mm + pin radius 9 mm = 37 mm) and, being perpendicular to headings confined to the
   ±50–130° window, keeps adjacent displaced arrows' x-extents (≤ 0.12 m each) inside
   the ≥ 0.25 m worst-case pin spacing.
7. near-miss: all arrows on-pin at target + 14° (tol 10°) → nothing aligned, no
   alignment credit.
8. wrong color: each arrow aimed at the next dial's post → nothing aligned.
9. decoy: green arrow aimed at the yellow post → not aligned.
10. flipped 180°: tail toward the post → nothing aligned.
11. latched credit: align red, spin it away → latched score unchanged, success gone.
12. monotonicity: closer heading latches strictly more progress credit.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN.
