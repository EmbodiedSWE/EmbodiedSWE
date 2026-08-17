# press_switch_i161 — Dual-Dial Setpoint Console

**Env:** `simgen.dial_setpoints` (scene `dial_setpoints`, robot `"null"`)
**Seed:** `rlbench/press_switch` (RoboVerse `roboverse_pack/tasks/rlbench/press_switch.py`)

## What the task is

A charcoal console lies face-up on a table, carrying two identical free-spinning
dials 31 cm apart. Each dial is a gray knob (7 cm across) with a dark grip blade
standing across its top and a red pointer reaching toward a pale bezel with white
tick dots. On each dial's own rim stands one small ORANGE FLAG whose bearing is
**sampled fresh every episode** (uniform in ±85°); the knob itself **starts at a
sampled angle** at least 40° from its flag. The dials spin freely about their
vertical axes between physical end stops at ±92° — no detent anywhere, angular
damping simply parks the knob wherever it is released.

Goal: rotate EACH knob until its red pointer rests within ~6° of that dial's
orange flag, then let go, with BOTH pointers on their flags at the same time
(settled, hands off). Sweeping the pointer through the flag at speed earns
nothing — only where the pointer *rests* is judged. Pressing or shoving a knob
does nothing; the reading only changes by rotation.

**Execution order: NONE required.** Either dial first, either direction. The
solve does 0 then 1 purely as a convention.

## Seed provenance and why this is strategically different

The seed (`press_switch`) is a **single binary poke**: push the wall switch's
lever through its flip point, in one fixed direction, with no magnitude control
and no stopping problem — contact anywhere on the lever, outcome is a discrete
state bit. This task inverts every one of those properties:

- **Continuous, not binary.** The goal state is a sampled *angle* (sign AND
  magnitude vary per episode per dial), not a discrete on/off. A memorized
  motion can never work; the policy must *read* each flag's bearing off the rim
  and regulate to it.
- **A stopping problem, not a crossing problem.** The seed only needs to push
  *past* a point. Here there is no detent at the flag: overshoot fails exactly
  like undershoot, and the rubric's slow-gate latch makes a fast sweep through
  the band worthless. The hard part is decelerating and releasing inside ±6°.
- **Two simultaneous sub-goals, unordered**, vs. the seed's one; success is
  judged live on both settled knob orientations at once.
- **Grasp-and-turn (wrist roll about a vertical axis), not poke.** The seed's
  contact is a fingertip prod; here the demanded interaction is a sustained
  grasp on the grip blade with regulated rotation and a deliberate release.
- **The seed's own strategy is a tested negative**: smoke check 5 presses and
  shoves a knob at 8 N and asserts the pointer moves < 2° and earns ~0.

It is also deliberately unlike the other corpus tasks I read: no interlock /
extraction / decoy-alignment (lamp_off_i101's twist-lock-unplug), and no
pick-place / containment (pen_holder). Nothing is transported anywhere — the
entire task lives in two rotational degrees of freedom.

## Mechanics

Plain rigid bodies + authored USD D6 joints (the proven pattern): each knob is
one dynamic compound body (cylinder + collidable grip blade + visual pointer)
hung on a console→knob D6 that frees exactly rotZ within ±92° (locked limits
`low=1 > high=-1` on all other axes; joint pair collision-filtered). The spin
axis is vertical so the plant is gravity-neutral; angular damping (6.0) parks
the knob where released (coast from the release speed ≈ ω/damp ≪ tol). The
flags are separate kinematic posts re-posed at reset to the sampled bearings,
standing at r = 100 mm — outside the pointer's 79 mm sweep. `post_step` owns
both knobs' wrench slots: it applies the `drive_t`/`drive_f` buffers every
substep (sleep thresholds zeroed so external wrenches always act) and advances
the rubric's slow-gate latch.

## Rubric (score 0..1, anchored in the solve trajectory)

- **0.30 · mean(align_latch)** — per-dial best *quasi-static* progress
  `(err0 − err)/(err0 − tol)`, latched only after the knob has been slower than
  0.30 rad/s for 24 consecutive substeps (0.2 s). NaN-guarded; never decreases.
- **0.30 · mean(aligned_now)** — per-dial current `|err| < 6°` AND slow.
- **0.40 · success()** — both dials aligned_now and everything settled.

Null policy ≈ 0. One dial set ≈ 0.30. Both set and settled = 1.000 exactly.
Latched credit survives later regression (smoke check 11); the live 70% does
not — success is judged on the *current* physical state.

## Solution outline (solve.py — NO teleports at all)

There is no transport in this task, so nothing is teleported ever: both knobs
start at their sampled reset angles and every degree of progress flows through
the live D6 + damping plant. The solver writes only the scene's `drive_t`
buffer — the stand-in for the Franka's wrist roll on the grasped grip blade —
and post_step applies it as a torque about the dial's one free axis.

Per dial (0 then 1): a cascaded velocity servo — outer rate command
`w_des = clamp(4.0 · wrap(target − yaw), ±1.2 rad/s)`, inner
`τ = KW · (w_des − w)` with `|τ| ≤ 0.04 N·m` (a fingertip pair on the 3.2 cm
blade arm ≈ 1.3 N — comfortable jaw authority). Gains respect the one-substep
wrench delay (`KW·dt/I ≈ 0.3 < 1` at KW₀, ≤ 0.8 at the escalation ceiling); on
stalls the GAIN escalates, never the torque cap. The servo decelerates on the
proportional cone and releases only when slow AND inside 2.5° — the stopping
problem *is* the task. Phases print non-decreasing `SIM_GEN_SCORE`: reset (~0)
→ dial 0 set (~0.30) → dial 1 set + success (1.000) → ≥3.5 s hands-off
persistence with asserted-zero drives → `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, parallel jaw)

- **Base pose:** on the table at ~(0.45, 0.0, 0.40), facing the console (−x).
  Both dial axes are at x = 0, y = ∓0.155, knob top at z ≈ 0.50 — 33 cm apart
  diagonally, both inside a comfortable 0.47–0.55 m reach envelope; the console
  face points up so the approach is a plain top-down grasp.
- **Grip blade (per knob):** 64 × 14 × 26 mm bar standing across the knob top —
  a canonical parallel-jaw target (14 mm across the jaws, 26 mm of purchase
  height, free air on both sides; nearest obstruction is the flag at 100 mm
  radius, outside the 45 mm blade half-span sweep). Grasp top-down with the
  jaws across the blade's short axis.
- **The turn:** pure wrist roll (joint 7) about the vertical dial axis with the
  wrist aligned to it — no arm repositioning during the turn. Worst-case
  required rotation ≈ 173° (start 88° one side, flag 85° the other), within the
  Franka wrist's ±166° after choosing the initial blade-grasp orientation
  (blade symmetry gives two grasp yaws 180° apart, halving the worst case to
  ≤ ~90°). Torque demand ~0.04 N·m ≪ gripper/wrist authority.
- **The release:** open the jaws when the pointer rests in the band — the knob
  stays put (no detent needed; damping parks it). Then re-grasp the other
  blade. No forces beyond fingertip scale anywhere.

## Randomization (readback-verified)

Per episode, per dial: flag bearing t ~ U(±85°) (flag physically re-posed on
the rim) and knob start s ~ U(±88°) with |s − t| ≥ 40° (rejection sampling,
12 rounds, then a deterministic 64°-toward-the-roomier-side fallback). Sampling
uses `torch.rand` only (first-`randint` degeneracy on this stack).

## Checks (smoke.py — 12, one linear run, frames.npz recorded)

1. **settle** — clean reset: all-finite, knobs physically AT sampled starts
   (readback < 2.5°), score < 0.05, no success.
2. **readback** — flags physically stand at the sampled bearings (world pose vs
   formula < 2 mm), separations ≥ 40°.
3. **random** — 8 seeds: ≥5 distinct targets/starts, both signs occur,
   separation holds, knob-at-start readback on every draw.
4. **null** — 2 s of nothing: score < 0.05, no success.
5. **negative (seed strategy)** — 8 N axial press + 8 N lateral shove: pointer
   moves < 2°, knob height within 4 mm, score < 0.05.
6. **negative (end stops)** — servo commanded to 150°: parks at the ~92° stop,
   finite, no success.
7. **negative (fly-through)** — fast sweep THROUGH the flag (peak in-band speed
   > 0.6 rad/s), parked past it: latch < 0.95, no success, score < 0.55.
8. **negative (near miss)** — dial 0 exact, dial 1 parked 2.5×tol off: partial
   credit only (0.20–0.75), no success.
9. **negative (crossed flags)** — each pointer on the OTHER dial's bearing: no
   success, score < 0.55.
10. **exactness** — both dials rested on their flags → success() and
    score == 1.0, still true 1 s later.
11. **latch** — turning dial 1 away revokes success (live state); the latched
    share (0.38–0.55) remains.
12. **frames** — ≥ 20 video frames saved to `frames.npz`.

## Files

- `scene.py` — cfg + scene + registration (`dial_setpoints`, `simgen.dial_setpoints`)
- `solve.py` — `python -u -m simgen_tasks.press_switch_i161.solve --headless [--seed N]`
- `smoke.py` — `python -u -m simgen_tasks.press_switch_i161.smoke --headless`
