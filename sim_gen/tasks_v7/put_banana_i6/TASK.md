# mug_tipout_rehome — pour the fruit OUT of the mug into the green dish, park the mug on its pad

**Seed:** `embodiedgen/put_banana`
(`sim_gen/RoboVerse/roboverse_pack/tasks/embodiedgen/put_banana.py`)
**Tier:** easy-medium — **2 goals** (fruit in the green dish, mug upright on the blue pad).
**Execution order: NOT required** — either goal may be completed first; both must hold at
the end (parking the still-loaded mug on the pad is a recoverable non-success, tested as
smoke #6).
**Env name:** `simgen.mug_tipout_rehome` (scene `mug_tipout_rehome`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed asks for a free **pick-and-place into a container**: the banana lies in the open
on a cluttered table, is grasped, carried through the air, and dropped into a mug — a
bounding-box containment check relative to the mug, orientation ignored, clutter as
distractors.

Here the task relation is **inverted and the manipulandum is swapped**, not
re-parameterized:

- The seed's SUCCESS state is this task's START state: the fruit (an orange, a 32 mm
  ball) begins **inside the mug**. The end state requires the mug **empty**.
- The fruit is **ungraspable by construction**: the mug bore is 46 mm across flats and
  95 mm deep, leaving 7 mm of annular clearance around the 32 mm ball — no parallel-jaw
  finger pair fits into the bore around the fruit (each Franka finger is ~18 mm thick;
  jaw + ball needs ≳68 mm of bore). The seed's opening move — grasp the fruit — does not
  exist. The only feasible plan manipulates the **container**: grasp the mug body, carry
  it, **tip it until the fruit pours out** over the rim into the target, then set the
  empty mug down upright on its **home pad**. The solver's verb changes from pick-place
  to pour; the fruit is only ever touched by the mug's own walls and by gravity.
- The seed's distractor role is kept but **moved to the goal side**: two
  identical-geometry dishes (GREEN target, WHITE decoy) randomly swap slots per episode,
  so the pour target must be identified by **color**, not layout (smoke #9 pins the
  decoy as a failing outcome; smoke #4 proves the swap is real by readback).
- A second, seed-absent goal: the mug itself must finish **standing upright on the blue
  home pad** — the container is a first-class manipulandum with its own placement
  tolerances, not a static fixture.

A solver therefore needs a different plan (tip-and-pour of a held container plus a
container set-down — no fruit grasp exists) and different code structure (a pour
controller with tilt scheduling and a two-goal terminal check — not a
grasp/lift/lower-into-cup pipeline), not different numbers.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects; they never do the task. The core interaction (the pour) goes
through contact dynamics. Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing —
all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (mug, green/white dish slots, pad
  — seed provenance in stdout), assert the orange is seated in the bore, baseline ~0.
- **P1 TRANSPORT (teleport, consistent pair)**: one pose write carries the mug WITH its
  seated orange across free space to a hold pose 100 mm above the green dish floor
  (offset ~35 mm back along the pour direction). The orange is written at the exact
  bore-bottom seat of the new mug pose — the same relative state as at spawn, so the
  containment is neither created nor bypassed. Hovering satisfies no rubric clause:
  every credit term is gated on the orange LEAVING the bore. 30 hold steps re-seat it.
- **P2 POUR (contact dynamics — no teleport can produce this without bypassing the
  task)**: the mug is pose-HELD each step (the kinematic-hold emulation of a rigid
  grasp, gravity-compensated vz=+g·dt) and tilted at 40 deg/s about a horizontal axis.
  The orange is NEVER pose-written: it rides the real bore wall, exits over the rim
  under gravity (~100-120 deg), free-falls ~5 cm, impacts the dish (restitution 0) and
  settles through real contacts. The tilt freezes the moment the orange leaves the
  bore. Asserts the orange settled INSIDE the green dish.
- **P3 MUG SET-DOWN (teleport transport + contact dynamics)**: one pose write carries
  the empty mug to 25 mm ABOVE the pad top, upright — deliberately OUTSIDE the 10 mm
  base-height tolerance (asserted: the freshly-released state does NOT satisfy the home
  clause). Gravity drops it onto the pad; it beds down and settles for 1.5 s.
  success() first turns True here, judged on settled poses.
- **P4 persistence**: 3.3 more simulated seconds with no intervention; only if
  success() held prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(0.0, 0.0, 0.0)` facing +x. Working radii from this base: mug spawn
0.51–0.61 m, dish slots 0.47–0.58 m, home pad 0.44–0.55 m — all inside the proven
0.30–0.71 m ground-level comfort envelope, with the pour executed at a comfortable
0.10–0.15 m height.

**Mug (the only object the arm touches):** a canonical side grasp — the octagonal body
is ~54 mm across flats (58 mm corners) against an 80 mm jaw, 95 mm tall (plenty of
finger engagement above mid-height), 180 g loaded. Contact strategy: approach
horizontally, close across the body flats, lift. The **pour** is a wrist-roll of the
held mug (Franka q7 range ±2.9 rad covers the ~110 deg tilt with margin) hovering over
the green dish — the dish mouth is 140 mm across for a 32 mm ball falling ~5 cm, and the
solve measured the exit tilt and landing to be repeatable. The **set-down** tolerances
are wide: mug centre within 45 mm of the pad axis (pad is 120 mm across), tilt within
10 deg, base within 10 mm of pad height — far above closed-loop OSC noise for a flat
base dropped the last centimetre.

**Orange:** never directly touched — by design it CANNOT be (7 mm annular bore
clearance vs ~18 mm finger thickness). Every interaction it undergoes (bore wall
contact during the tilt, free fall, dish impact) is one the arm produces by moving the
mug. **Dishes, pad:** kinematic fixtures, never need to be touched.

No other objects exist; every required contact is one the arm can make.

## Success and rubric (physical outcomes only)

`success()` iff BOTH, live and settled:
- the orange rests INSIDE the green dish: centre within `dish_capture_r = 65 mm` of the
  dish axis (wall ring is at 70 mm), at floor-contact height (`ball_z_tol = 12 mm`),
  **not inside the mug** (the container-cheat clause — a loaded mug parked in the dish
  counts for nothing, smoke #7), |v| < 5 cm/s;
- the mug stands upright ON the blue pad: centre within `mug_home_xy_tol = 45 mm`,
  base at pad-top height ±10 mm, tilt < 10 deg, |v| < 5 cm/s.

`score()` ∈ [0,1], latched each physics substep:
`0.20·poured-out + 0.30·ball-approach-to-green-dish (gated on poured-out) +
0.20·in-green-dish + 0.15·mug-approach-to-pad (gated on poured-out)`, capped 0.85;
**1.0 iff success()**; ~0 for the null policy (every term is gated on the orange leaving
the bore, which requires acting). Latched credit never evaporates (smoke #12); the
solve's phase prints are monotone.

## Randomization

Per episode (verified by sim READBACK in the smoke): mug spawn xy ±4 cm + yaw; green vs
white dish slot assignment (Bernoulli swap) + per-dish xy jitter ±2.5 cm; home pad xy
±3 cm.

## Check list (smoke.py — rubric REJECTION battery, 14 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite, orange seated inside the mug bore, everything still
2. settle: score ~0 at reset, no success
3. randomization readback: mug spawn xy + home pad xy vary
4. randomization readback: green/white slot assignment flips AND per-slot dish jitter
   is real
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: the seed's terminal relation (fruit INSIDE the mug), mug even
   parked perfectly on the home pad → all credit gates locked, score ~0, no success
7. container cheat: the loaded mug parked standing IN the green dish → the in-mug
   clause rejects the containment, score ~0, no success
8. near-miss: orange settled on the ground just outside the green dish wall → NOT
   success, score < 0.9 (load-bearing containment ring)
9. wrong place: orange settled in the WHITE decoy dish → NOT success, score < 0.9
   (color identification is load-bearing)
10. mug off home: orange in the green dish, mug upright 12 cm from the pad (tol
    4.5 cm) → NOT success
11. mug tilted: orange in the green dish, mug lying on its side over the pad → upright
    + base-height gates reject, NOT success
12. latched credit survives regression: re-seating the orange back inside the mug
    leaves the latched score unchanged, still no success
13. rejection audit: success() never True at any judged point of this battery
14. final no-NaN

N/A notes: **out-of-order end state** — no execution order is declared (either goal may
complete first); the order-adjacent hazard ("park the loaded mug first") is exactly
smoke #6. **wrong object** — the scene has a single movable fruit; identity is carried
by the destination controls instead (#7/#8/#9). The seed's literal end state IS
expressible here and is smoke #6.

Video frames are recorded throughout and saved to `frames.npz` in the working directory.
