# letterbox_deposit — push the green parcel through the letterbox's one-way flap

**Seed:** `rlbench/put_item_in_drawer`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_item_in_drawer.py`)
**Tier:** medium — one core mechanism interaction (push-through a gravity-closed one-way
flap) + color identification + a decoy-exclusion clause.
**Execution order: NOT required** — there is a single deposit interaction; the flap
recloses by itself, so no separate "close" step exists to order against.
**Env name:** `simgen.letterbox_deposit` (scene `letterbox_deposit`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed is an OPEN-THEN-INSERT task: the drawer cabinet must first be opened (grasp
the drawer handle, pull the prismatic joint out ~20 cm), which exposes an open volume,
and the item is then picked and lowered into that volume from above. The receptacle has
a persistent open state; creating and using it is the whole plan.

Here the access mechanism is **inverted, not re-parameterized** — the receptacle is
never opened and has no open state:

- The letterbox is **sealed on every face**; its only entry is a mail slot covered from
  the inside by a **gravity-closed one-way swing flap** (hinged above the slot, joint
  limits allow only inward swing). There is **no handle and nothing on the box is ever
  grasped or pulled**; the seed's opening move does not exist. A drawer-puller arriving
  here finds nothing to pull.
- The **parcel itself is the key**: the only way in is to push the parcel horizontally
  through the slot so that the parcel's own nose displaces the flap against gravity.
  The barrier is displaced *by the transported object*, not by a prior mechanism
  action, and it re-closes behind the parcel on its own. The solver's verb chain
  changes from grasp-handle / pull / pick / lower-in / (push shut) to grasp-parcel /
  align-with-slot / push-through / release.
- The insertion axis is **horizontal, through an aperture**, with a tip-over-the-lip
  drop finishing the deposit — not a vertical drop into an open drawer. The deposit is
  **irreversible** (the parcel lands 12.5+ cm below the slot), where the seed's drawer
  contents remain retrievable.
- The seed's end state — item resting in/on an opened receptacle — is expressible here
  only as the parcel sitting **on top of the closed box** (the sole from-above
  placement a sealed box affords) and is an explicit smoke-tested failure (#6).
- Added identification + exclusion structure absent in the seed: a WHITE decoy parcel
  of identical shape swaps spawn slots with the GREEN target per episode; success
  requires the decoy left OUTSIDE (#9, #10).

A solver therefore needs a different plan (no opening action exists; one aligned
horizontal push through a self-closing barrier) and different code structure (slot
alignment + push-depth control against a compliant flap, not an
open/pick/lower/close pipeline), not different numbers.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects; they never do the task. The core interaction (flap
displacement + lip tip-over) goes through contact dynamics. Phases (each boundary
prints `SIM_GEN_SCORE`, non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (parcel/decoy slots, flap
  angle — seed provenance in stdout), assert flap shut + parcel outside, baseline
  score asserted ≤ 0.02.
- **P1 TRANSPORT (teleport)**: one pose write carries the green parcel to a hold pose
  nose-on, 20 mm outside the front face, centred on the slot, bottom 4 mm above the
  lip. Outside the box, flap untouched — asserted to satisfy no containment/flap
  clause (only the 0.15-weight approach term moves, as any real carry would).
- **P2 PUSH-THROUGH (contact dynamics — no teleport can produce this without
  bypassing the task)**: the parcel is pose-HELD each step (kinematic-hold emulation
  of a rigid grasp, gravity-compensated vz=+g·dt, write velocity consistent with the
  4 cm/s trajectory) and advanced along the slot axis. The FLAP is never written: the
  parcel's nose presses it open about its real hinge against gravity — max opening is
  measured and asserted ≥ the 25 deg latch threshold (~70 deg in practice) — while
  the parcel rides the slot's bottom lip. The hold stops with the CoM 20 mm past the
  lip's inner edge.
- **P3 RELEASE (contact dynamics)**: hands off. The freshly-released state is perched
  at slot height and asserted NOT contained (containment requires the parcel far
  below the slot — the release-outside-the-scoring-band rule). Gravity tips it over
  the lip; it falls ~15 cm to the cavity floor, settles through real impacts, and the
  flap swings shut behind it under its own gravity + viscous hinge damping. success()
  first turns True here, judged on settled poses.
- **P4 persistence**: 3.3 more simulated seconds with no intervention; only if
  success() held prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(0.0, 0.0, 0.0)` facing +x. Working radii: parcel spawns at
0.42–0.49 m; slot mouth at 0.49 m, insertion end at 0.52 m, all at 0.18–0.23 m height
— inside the proven 0.45–0.71 m comfort envelope (spawn pickups at ~0.45 m, from
above, are routine).

**Green parcel (the only object the arm touches):** a 70 × 60 × 45 mm box, 80 g. Grasp:
close the jaw across the 60 mm faces (80 mm jaw opening, 20 mm total margin), picked
from above at its spawn, re-oriented nose-first in free space. Insertion: the slot is
130 × 70 mm for a 60 × 45 mm cross-section — ±35 mm lateral and ±12.5 mm vertical
alignment tolerance, an order of magnitude above closed-loop OSC noise. Push: the flap
is a 50 g plate; displacing it to ride-over takes ~0.3 N at the parcel nose — trivial
for the arm. The fingers holding the parcel's rear half stay outside/at the slot mouth
until the CoM is past the lip (parcel is 70 mm deep, lip is 10 mm); if a final nudge
is wanted, the closed fingertip pair (~25 mm wide) fits the 35 mm-per-side lateral
margin easily. Release anywhere past the lip's inner edge and gravity finishes the
deposit — the drop is deliberately self-completing, so required precision ends at
"CoM past a 10 mm lip".

**Flap:** never touched directly — it is displaced by the parcel being pushed, which
is exactly the intended contact. **White decoy:** never needs to be touched.
**Letterbox:** kinematic fixture, never needs to be touched (brushing it is harmless).

No other objects exist; every required contact is one the arm can make.

## Success and rubric (physical outcomes only)

`success()` iff, live and settled:
- the GREEN parcel is inside the cavity BELOW the slot: origin within the interior
  footprint (±5 mm margin), z in (0.4 cm, 14 cm) above the cavity floor (the slot's
  bottom edge is at 17 cm — a parcel perched in the slot is not contained), |v| <
  5 cm/s;
- the flap is fully re-closed: |opening| ≤ 10 deg, flap angular velocity < 0.5 rad/s;
- the WHITE decoy is NOT inside the cavity.

`score()` ∈ [0,1], latched each physics substep:
`0.15·slot-mouth approach (vs the episode's own spawn distance — exactly 0 for the
null policy) + 0.20·flap-ever-pushed-past-25 deg + 0.35·transit (parcel ever inside)
+ 0.15·deposited (inside with the flap re-closed)`, capped 0.85; **1.0 iff
success()**. Latched credit never evaporates (smoke #12); the solve's phase prints
are monotone.

## Randomization

Per episode (verified by sim READBACK in the smoke): Bernoulli green/white spawn-slot
swap; per-parcel xy jitter ±3 cm + yaw ±30 deg; initial flap ajar angle in [0, 8] deg
(falls shut in the first settle — visual variety, no credit; 8 deg is below every
flap threshold). The box is a fixed fixture (its slot is the calibration target the
whole task aims at), as in the seed, whose cabinet is also fixed.

## Check list (smoke.py — rubric REJECTION battery, 15 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite, flap fallen shut, both parcels outside on the floor, still
2. settle: score ~0 at reset (≤ 0.02), no success
3. randomization readback: green/white spawn-slot assignment flips across seeds
4. randomization readback: per-slot xy jitter, spawn yaw and initial flap ajar all vary
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: green parcel settled ON TOP of the closed letterbox (the
   drop-from-above end state) → no success, score < 0.5
7. near-miss: green parcel settled on the ground against the front face just below
   the slot → no success, score ≤ 0.3
8. shallow insertion: parcel perched nose-on-lip with CoM outside the support tips
   back OUT to the ground under gravity (an under-pushed deposit physically fails) →
   no success
9. wrong object: WHITE decoy settled on the cavity floor, green outside → no success,
   score ≤ 0.3 (color identification is load-bearing)
10. both inside: green AND white on the cavity floor → decoy-exclusion clause rejects,
    no success, score ≤ 0.85
11. flap-open clause (transient probe): green parcel at rest on the cavity floor but
    the flap held open 70 deg → no success; the parcel is removed before the flap can
    fall shut, so the battery never constructs success
12. flap-open aftermath: the flap falls shut by itself, parcel back outside → still no
    success
13. latched credit survives regression: transit/deposit latches earned mid-air, parcel
    yanked back outside → score unchanged across 0.5 s, still no success
14. rejection audit: success() never True at any judged point of this battery
15. final no-NaN

N/A notes: **out-of-order end state** — no execution order is declared (single
interaction; the flap recloses autonomously). The order-adjacent hazard — leaving the
flap propped open — is exactly #11.

Video frames are recorded throughout and saved to `frames.npz` in the working
directory.
