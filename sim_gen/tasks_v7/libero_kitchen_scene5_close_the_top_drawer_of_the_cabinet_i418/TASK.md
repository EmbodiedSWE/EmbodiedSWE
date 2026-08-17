# libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i418 — "stowflat_cabinet"

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet`
(a Franka in the LIBERO kitchen pushes the cabinet's top drawer shut; success is
the drawer's prismatic joint reading closed). The seed's entire skill is a
single straight push on a free-sliding drawer.

## The task

A fixed kitchen cabinet has one top drawer on a real per-env USD prismatic
joint (limits [0, 175 mm] are the hard stops). The drawer starts sampled open
at q0 ∈ [125, 160] mm. Inside the drawer are two cargo items:

- a **tall blue carton** (44×44×130 mm, 0.12 kg) standing **upright** in the
  drawer's protruding front band, at a sampled x/y/yaw;
- a small **white companion bar** (45×30×20 mm, 0.05 kg) deeper inside, at a
  sampled y.

The cabinet mouth (slab underside minus drawer floor top) is **89 mm**. The
upright carton is **130 mm** tall — it protrudes 41 mm above the mouth.
**Goal**: the drawer seated (q ≤ 10 mm) with BOTH cargo items inside it —
which requires laying the carton flat first.

## Strategic difference

**vs the seed**: the seed's strategy — push the drawer — is *physically
impossible* as the scene stands. Pushing rams the too-tall carton against the
slab's front edge while the drawer's front panel (top at 394 mm, within 4 mm
of the mouth) pinches it from the other side; the pinch hard-locks at
q ≈ carton_w + panel_t = 62 mm (tilting only increases the carton's horizontal
diagonal, so the wedge binds — verified in smoke: an 8 N push stalls at
q ≥ 20 mm with score ~0). The new load-bearing skill is **in-place
reorientation of cargo**: tip the carton sideways *inside* the drawer so it
lies flat (44 mm ≪ 89 mm mouth), and only then does the seed's push work. The
ordering (flatten → close) is forced by geometry, not by declared rules.

**vs every corpus task read this session** (spot checks on the closest kin):
- `i163` (same seed, "bascule counterweight"): there the robot never touches
  the drawer — it removes a counterweight and gravity closes it. Here the
  robot *does* push the drawer; the novelty is the cargo-reorientation
  prerequisite. No masses/lever discrimination anywhere.
- removal-to-dock tasks (e.g. take the blocker OUT): here removal FAILS —
  ejecting either cargo item zeroes the closing credit and blocks success();
  the contents must end INSIDE the drawer. The blocker is not removed, it is
  *reshaped in place* (reoriented).
- free-drawer reinstall / bayonet / cam-gate drawer tasks: the drawer here
  stays on its joint the whole time; no locks, no keys, no re-installation.
- No other corpus scene is named `stowflat_cabinet` and none uses
  taller-than-mouth cargo whose *orientation* is the gate.

## Teleport solution (solve.py) — ZERO teleports

Every interaction is a force-limited contact proxy for a hand.

- **P0 settle + readback**: 120 steps; read q0, the carton's drawer-local pose
  and side (sign of local y), the companion pose — never hard-coded.
- **P1 TIP** (fingertip push): 0.45 N horizontal at a body point 45 mm above
  the CoM (upper third of the carton), directed toward the drawer centreline
  from the MEASURED side. That is 1.9× the tipping moment yet only 0.7× the
  sliding limit, so the carton pivots on its bottom edge instead of skidding.
  Force is REMOVED once past the diagonal (axis_z < 0.65); gravity lays it
  flat on the drawer floor. 180 hands-off settle steps. Drawer must not move.
- **P2 CLOSE** (the seed's push, now unblocked): force-limited PD on the
  drawer (≤ 1.2 N → ≤ 2.3 m/s² on drawer+cargo, far below the ~5 m/s²
  friction limit, so the flat cargo rides the floor without sliding) until
  q ≤ 4 mm; force released; success() arrives on the live settled state.
- **Persistence**: 10×40 hands-off steps (3.33 s at 120 Hz) with success()
  required throughout; then the whole episode repeats on a second seed
  (silent — the rubric restarts on reset) before `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at each phase boundary and asserted non-decreasing.

## Rubric (score.py semantics in scene.py)

- 0.25 — latched: carton has lain flat inside the drawer (`_fflat`).
- 0.45 × max closing fraction, GATED on (carton flat ∧ companion in ∧ finite)
  (`_fclose`) — the seed's blind push earns nothing (gate false while the
  carton is upright / jammed), and ejecting cargo earns nothing.
- Base capped at 0.70. Score 1.0 iff **live** `success()`:
  q ≤ 10 mm ∧ drawer in channel ∧ carton flat inside ∧ companion inside
  ∧ settled ∧ finite.

Null policy ≈ 0. Seed strategy: jams at q ≥ 20 mm, score ≈ 0. Eject carton
then close: ≈ 0. Eject companion, flatten + close: ≤ 0.30, no success.

## Franka embodiment (per object)

Plausible base pose: on the counter side, base at ≈ (+0.55, 0, 0) in scene
frame, facing −x toward the cabinet; drawer front and its handle are at
x ∈ [0, q0+14 mm], z ≈ 345–357 mm — squarely inside a Franka's sweet spot.

- **Carton tip** (P1): the carton's top third stands at z ≈ 380–430 mm in the
  drawer's OPEN front band — open sky above (the slab starts 200+ mm behind
  it), so a fingertip side-push at z ≈ 0.42 m is a trivial reach; 0.45 N is
  far below arm force limits. Alternatively the 44 mm carton fits easily in
  the 80 mm parallel jaw for a grasped re-orientation — the contact-proxy
  push is the *minimal* embodiment.
- **Drawer push** (P2): the 100×12 mm handle bar at z ≈ 0.35 m is the seed's
  own affordance; a palm push of ~1 N over 150 mm of straight horizontal
  travel.
- **Companion bar**: never needs touching.

## Execution order declaration

Geometry forces flatten-before-close (upright carton wedges at q ≈ 62 mm and
the wedge locks). Closing first is impossible; flattening requires no drawer
motion. No hidden order rules — the latch gate only encodes "closing counts
only while the cargo is stowed", which is the task goal itself.

## smoke.py — 13 rejection-oriented checks (ALL PASS 13/13 on forge)

1. settle/no-NaN + authored masses read back via `get_masses()` (drawer,
   carton, companion within 10%), carton upright, cargo in, baseline ≤ 0.05
2. randomization A: q0 span > 12 mm over 10 resets, drawer tracks it
3. randomization B: carton x span > 10 mm, both y sides seen, companion y
   span > 50 mm; cargo settles inside
4. null policy 240 steps: score ≤ 0.05
5. SEED strategy: 8 N PD push on the untouched scene HARD-JAMS at
   q_min ≥ 20 mm, carton stays in, score ≤ 0.05
6. upright-carton-in-closed-drawer construct: refused instantly (flat clause)
7. eject carton → push seated: score ≤ 0.05, no success
8. eject companion → carton laid flat → push seated: score ≤ 0.30, no success
9. near-miss flush (stops at q ≈ 35 mm): no success, score < 0.70
10. latch regression: pulling the drawer back open with a real force does not
    revoke earned `_fclose`; the live clause reads open again, no success
11. rejection audit (success() observed False at every step of the battery)
12. final no-NaN
13. video frames.npz written (> 10 frames)

## Files

- `scene.py` — StowflatSceneCfg/StowflatScene, `SCENES.register("stowflat_cabinet")`,
  `register_env("simgen", …, robot="null")`; 28 honesty asserts in
  `__post_init__` (jam/pass gate, pinch band, spawn clearances, tip swing and
  flat-landing clearances, judged-box consistency).
- `solve.py` — zero-teleport certificate, 2 seeds, watchdog, hard exit.
- `smoke.py` — the 12 checks above + frames.npz.
- `TASK.md` — this file.
