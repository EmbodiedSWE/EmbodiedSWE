# living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i423 — "inertia_derby"

## Seed provenance

Seed task: `libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray`
(a Franka picks the tomato-sauce bottle out of a clutter of look-alike grocery
items and sets it in a tray). The seed's whole skill is visual identification +
pick-and-place.

## The task

On a low wooden bench (top at 100 mm) stand three **visually identical sealed
red orbs** (∅64 mm), each nested in a curbed staging cradle. All three weigh
**exactly 300 g** — equal masses BY CONSTRUCTION, so a scale or hefting reveals
nothing. Exactly ONE (hidden `genuine_idx`, sampled per episode) is the genuine
**solid-filled** orb (inertia factor k = I/(mr²) = 0.40); the other two are
hollow counterfeit shells (k = 2/3). The per-orb inertia tensors are written
through the physx view every reset; masses are untouched.

The bench carries the measuring instrument: a **twin-lane race ramp** (12.5°,
350 mm long) with a free yellow **start bar** resting across both lanes in
slotted posts, its underside only 4 mm above the ramp (an orb cannot pass
under; the fences flanking the gate slit stop it going around). Below: a flat
runout and a slick finish wall. Rolling from rest,
a = g·sinθ/(1+k) → solid 1.52 m/s², hollow 1.27 m/s²; at the checkpoint the
analytic gap is **62 mm = 2.8× the 22 mm verdict threshold**, while two hollow
orbs finish in a dead heat (measured on forge: tie gap +0.0 mm, solid-vs-hollow
gap 61.4 mm).

Declared legality rules (in `describe()`):
1. at least TWO orbs must actually be raced down the ramp — delivery without
   the experiment does not count (`race_done` gates success and delivery);
2. a counterfeit that SETTLES inside the tray (25-step streak) permanently
   **contaminates** the shipment — success is impossible forever after and the
   score clamps ≤ 0.15 (kills brute-force "try them all in the tray").

**Goal**: the genuine orb settled inside the upright free wooden tray (130 mm
square, sampled x/yaw), no counterfeit in it, the race performed — all at rest.

## Strategic difference

**vs the seed**: the seed is solved by *looking*. Here the candidates are
pixel-identical and mass-identical; the discriminating signal exists ONLY in
rotational dynamics, so the agent must *perform a designed experiment* (stage
two orbs, release the start bar, read the finish order) and then apply a
two-branch *inference*: clear leader ⇒ leader is genuine; near-tie ⇒ both
racers are fakes and the un-raced third orb is genuine. The delivery itself is
trivial; the physics and logic before it are the task.

**vs the corpus kin** (spot-checked this session):
- `i253` (same seed, "sauce_balance"): hides MASS and reads it off a beam
  balance. Here masses are EQUALIZED by design — weighing is useless; the
  hidden state lives in the inertia tensor, and the instrument is a dynamic
  race, not a static balance. No corpus task discriminates by I/(mr²).
- `i166` ("sauce_chute") and other routing tasks: transport puzzles; no hidden
  state, no experiment, no inference branch.
- balance-scale / weigh-press / tare-lift family: all read *weight* with
  static equilibrium. A race reads a *dynamic* quantity that statics cannot
  see (equal weights, different acceleration).
- No other corpus scene is named `inertia_derby`; none has an
  equal-mass/different-inertia object set or a tie-implies-third inference.

## Teleport solution (solve.py) — transport-only teleports

Teleports only carry free bodies between rest poses (orb staging, bar parking,
orb delivery). Every load-bearing outcome is contact dynamics:

- **P0 settle + readback**: 120 steps; orb→cradle permutation, tray pose, and
  the per-orb k values read back (masses asserted equal; exactly one solid).
  The genuine IDENTITY is *not* used — it is inferred by the race and only
  ASSERTED against the hidden truth afterwards.
- **P1 STAGE** (carry): orbs #0 and #1 hover-dropped into the two lanes uphill
  of the bar; gravity rolls them down against the resting bar (both settle at
  the same x — verified).
- **P2 RACE** (the experiment): hand-proxy HOLDS (≤ 2.5 N per orb:
  feed-forward cancels the slope gravity, PD nulls drift) pin both racers
  while a force-limited PD lift (≤ 1.5 N on the 35 g bar — a two-finger pinch)
  raises the bar out of its pockets; the bar is parked (transport) and the
  racers settle to rest under the holds (release symmetry asserted ≤ 6 mm).
  BOTH holds are cut on the SAME step → both orbs roll from rest at the same
  instant; only inertia differentiates. At the leader's checkpoint crossing
  the gap is read and the two-branch verdict formed, then asserted against
  the hidden `genuine_idx`. (Why holds: a bare force-lift of the bar tilts it
  and opens one lane ~tens of ms early — measured 42 mm of spurious gap; a
  real GRASPED bar is rigid and level, so the robot needs no holds — see
  embodiment.)
- **P3 DELIVER** (carry): the *inferred* genuine orb hover-dropped over the
  tray centre; it lands and settles by gravity. No counterfeit ever touches
  the tray.
- **Persistence**: 10×40 hands-off steps (3.33 s at 120 Hz) with success()
  required throughout; the whole episode repeats on a second seed (silent)
  before `SIM_GEN_SOLVE: SUCCESS`. On forge, seed 0 raced two fakes (tie →
  third-orb branch) and seed 1 raced the genuine (leader branch): both
  inference branches are exercised.

`SIM_GEN_SCORE` printed at each phase boundary, asserted non-decreasing.

## Rubric

- 0.075 per orb that ROLLED THROUGH the runout at speed (latched; max 2 → 0.15)
- 0.45 delivered (latched, 10-step streak): genuine seated in the upright tray,
  no counterfeit inside, race done, no foul, settled
- base capped at 0.60; a contamination foul clamps the total ≤ 0.15 forever;
  exactly 1.0 iff **live** success(): genuine_in ∧ ¬fake_in ∧ race_done ∧
  ¬foul ∧ settled ∧ finite. Null ≈ 0.

## Franka embodiment (per object; one base pose)

Base at ≈ (−0.20, +0.45, 0) on the floor beside the bench (bench half-width
0.35), facing −y. Reach radii: cradles (y = 0.21) ≈ 0.26–0.36 m; start bar
mid-span (−0.31, 0) ≈ 0.46 m; runout/finish pickup (0.22, ±0.05) ≈ 0.62 m;
tray band (−0.23, −0.25) ≈ 0.70 m — all inside a Franka's 0.855 m envelope,
at bench heights 0.10–0.25 m.

- **Orbs**: ∅64 mm < 80 mm jaw — equatorial pinch from above. In the cradle
  the orb's top hemisphere stands proud (centre 29 mm above the 10 mm curbs);
  in the tray the orb top is 22 mm proud of the 50 mm walls and the inner
  clearance is 50 mm to the finger; on the runout it rests against the slick
  low wall under open sky.
- **Start bar**: 14×14 mm section, 276 mm long, 35 g — pinch at mid-span
  (posts are at |y| ≥ 115 mm, far from the grasp), lift vertically ~80 mm out
  of the pockets. A *grasped* bar is rigid and level, so both lanes open
  simultaneously — the robot gets the symmetric release for free and needs no
  third hand (the solve's holds exist only because its force-proxy lift
  cannot enforce levelness).
- **Tray**: never needs touching (it spawns upright on the bench).

## Execution order declaration

The rules are DECLARED in `describe()`/`instruction()` (race ≥ 2 orbs before
the delivery counts; never let a fake settle in the tray) and enforced by
latches: `race_done` gates delivery credit and success; the foul latch is
permanent. Physically, the bar gate forces "lift bar → race" ordering: 4 mm
under-bar gap and flanking fences make rolling through without lifting
impossible (verified in smoke check 4).

## smoke.py — 13 rejection-oriented checks

1. settle/no-NaN: orbs cradled per sampled permutation, bar seated, tray
   upright, EQUAL masses read back, hidden inertia read back (one k≈0.40,
   two k≈2/3), score ~0
2. randomization: genuine_idx varies, permutation varies (≥3 distinct), tray
   x span > 40 mm, yaw span > 6°; inertia readback tracks genuine_idx
3. null policy 240 steps: score ≤ 0.05
4. gate sanity: an orb rolls against the RESTING bar and stays blocked uphill
   of the gate — no raced latch (the race requires lifting the bar)
5. SEED strategy: genuine orb (truth peeked by smoke) dropped straight into
   the tray with NO race → race_done False, score ≤ 0.05, no success
6. single-race: only ONE orb raced then genuine delivered → ≥2-raced rule
   refuses; score ≤ half race credit
7. contamination FLAGSHIP (end-state-identical): two orbs raced, a fake
   settled in the tray → permanent foul (score ≤ 0.15); fake removed, genuine
   delivered cleanly → final state IDENTICAL to success but never success
8. near-miss: race done, genuine resting on the bench beside the tray → no
   delivery, no success
9. tipped tray: race done, tray on its side, genuine at its footprint →
   tray_ok False, never success
10. raced persistence: racers carried back to their cradles → 0.15 raced
    credit survives (latched), delivery off, no success
11. rejection audit (success() False at every step of the battery)
12. final no-NaN
13. video frames.npz written (> 10 frames)

## Files

- `scene.py` — InertiaDerbySceneCfg/InertiaDerbyScene,
  `SCENES.register("inertia_derby")`, `register_env("simgen", …, robot="null")`;
  ~35 honesty asserts in `__post_init__` (gate blocks/clears, lane and cradle
  fits, rolling-not-sliding ≥3× margin, ANALYTIC race gap ≥2.5× threshold,
  tray retention/clearances, park pose, k separation, rubric weights).
- `solve.py` — transport-only-teleport certificate, 2 seeds (both inference
  branches), watchdog, hard exit.
- `smoke.py` — the 13 checks above + frames.npz.
- `TASK.md` — this file.
