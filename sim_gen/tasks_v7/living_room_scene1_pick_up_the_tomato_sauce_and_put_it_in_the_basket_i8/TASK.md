# slidelid_hamper — open the captive sliding lid, stash the red can, slide it shut

**Seed:** `libero_90/living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.py`)
**Tier:** medium — **3 stages**. **Execution order: REQUIRED** — open → insert → close.
The first ordering (open before insert) is physically forced: the lid covers the
hamper's only opening, and a can released above the shut hamper just rests ON the lid
(smoke #6). The second (close after insert) is demanded by the rubric: success is
CLOSED containment, and closing an empty hamper earns nothing (smoke #12).
**Env name:** `simgen.slidelid_hamper` (scene `slidelid_hamper`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed is a one-stage free pick-and-place: the tomato-sauce can sits in the open, is
grasped, carried over the OPEN-topped basket, released, and a bounding-box containment
check ends the episode. No fixture is ever actuated; nothing has to happen after the
release.

Here the container itself is a mechanism, and the goal state is sealed:

- The hamper's only opening is covered by a **captive sliding lid** riding in a guide
  channel (retaining lips overhang its edges — it cannot be lifted off, only slid
  along one axis between two stops). The seed's plan — carry the can over the
  container and let go — is physically dead: the can lands on the lid and scores
  nothing (smoke #6 constructs exactly this).
- The solver must actuate the fixture **twice, in opposite directions, with the
  insertion in between**: slide the lid open ≥ the can-passing travel, drop/place the
  red can through the exposed aperture, then slide the lid back to within 12 mm of its
  closed stop. Success is **closed containment** — the achieved state is visually
  hidden at the end (the can is under the lid), judged physically in the hamper's
  body frame.
- A **beige distractor can** (same shape, different color) must be left out; only the
  red can's containment counts (smoke #7).

A solver therefore needs a different plan (mechanism actuation → insertion →
mechanism restoration, an ordered 3-stage program with a bidirectional fixture
interaction) and different code structure (a stage machine that revisits the same
handle twice with opposite goals), not different parameters. The seed's entire plan is
strictly contained in stage 2 — and executing only that stage fails.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects; they never do the task. Both lid slides and the insertion go
through contact dynamics. Phases (each boundary prints `SIM_GEN_SCORE`,
non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (hamper pose + yaw, both can
  positions, lid travel — seed provenance in stdout), baseline score ~0.
- **P1 OPEN (contact dynamics)**: a world-frame horizontal force at the lid's CoM,
  aligned with the hamper's slide axis (velocity-regulated bang-bang: 4 N while axial
  speed < 0.10 m/s, +2 N per 2 s stall up to 12 N), drags the captive lid along its
  rails until its travel exposes the full can-sized aperture (≥ 130 mm). Friction,
  rails, lips and stops act on it the whole way; no pose write ever touches the lid.
- **P2 TRANSPORT (teleport, the only pose write on the can)**: one root-state write
  carries the red can from the ground to a release pose with its base **25 mm above
  the lid plane**, centred over the exposed aperture — fully outside the containment
  volume, so the freshly-teleported state earns no insertion credit and cannot be
  success(). Both endpoints are in free air.
- **P3 INSERT (contact dynamics)**: gravity drops the can through the real aperture;
  it impacts the hamper floor, beds down and settles (1.25 s). The insertion latch
  first sets here.
- **P4 CLOSE (contact dynamics)**: the same force scheme drives the lid back until it
  seats against the closed stop within the 12 mm tolerance (plus a gentle trim nudge
  if it rebounds). `success()` first turns True here, judged on the settled state.
- **P5 persistence**: 3.3 more simulated seconds hands-off; only if `success()` held
  prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(-0.15, 0.0, 0.0)`, facing +x. Working radii from this base: cans
0.31–0.46 m, lid handle across its whole travel 0.59–0.67 m, release point over the
aperture ~0.61 m — all inside the proven 0.30–0.71 m ground-level comfort envelope.

**Lid (stages 1 and 3): pinch the handle block and drag.** The handle is a
28 × 64 × 32 mm block on top of the lid, top face at ~185 mm height, approached from
above with nothing overhead; the jaw closes across the 28 mm dimension (80 mm max
opening) with 19+ mm of clear space to the retaining lips on either side (lips stop at
±70 mm from the channel centreline; the handle spans ±32 mm). Opening and closing are
the same straight-line horizontal drag (~135 mm travel at constant height) in opposite
directions — the channel tolerates ±3 mm of lateral error, and the lid's own guides
absorb it, so the required precision is far above OSC control noise. A pure push on
the handle's 28 × 32 mm face works as a fallback in both directions (stops end the
travel, so overshoot is impossible).

**Red can (stage 2): side pinch, lift, release over the aperture.** A 60 mm-diameter
free-standing cylinder on open ground — a canonical Franka side pinch (60 mm across an
80 mm jaw, grasp height ~50 mm, approach from above/side unobstructed). Carry it above
the hamper (rail tops at 170 mm are the tallest obstruction) and release it centred
over the exposed 120 × 160 mm aperture — the can clears every edge by ≥ 30 mm, and the
gripper itself stays above the plane and never enters the cavity (release-above-band:
the drop is exactly what the teleport solution certifies). No precision finer than
~2 cm is needed anywhere.

**Distractor can, hamper:** never need to be touched (the hamper is a kinematic
fixture; the distractor must merely be left alone — it spawns ≥ 9 cm from the red can,
verified by readback).

Every contact the task requires is one the arm can make.

## Success and rubric (physical outcomes only)

All geometry is judged in the hamper's body frame (its yaw is randomized). `success()`
iff, simultaneously and settled (|v| < 5 cm/s on both bodies): the red can's centre is
inside the interior footprint (|x|,|y| < 7.5 cm) and genuinely below the lid plane
(centre z < 10.5 cm — a can on the lid sits at ~20 cm), AND the lid is seated in its
channel within `closed_tol = 12 mm` of fully shut (plus channel y/z sanity gates).
`score()` ∈ [0,1], latched each physics substep:
`0.30·open-progress(max travel / 135 mm) + 0.35·inserted + 0.20·re-close-progress
(gated on the insertion latch)`, capped 0.85; **1.0 iff success()**; ~0 for doing
nothing (the lid starts shut, so open progress starts at 0). Latched credit never
evaporates (smoke #12); closing an EMPTY hamper earns nothing (the close term is gated
on insertion — same check). The solve's phase prints are monotone.

## Randomization

Per episode (verified by sim READBACK in the smoke): hamper yaw 90° ± 10° + xy jitter
± 2 cm; red can and beige can in disjoint ground bands whose sides SWAP 50/50 (so
"the can nearer the deck" is not memorizable), x ∈ [0.16, 0.24], y bands ±[0.05,
0.22], guaranteed ≥ 9 cm apart. The lid always starts fully shut.

## Check list (smoke.py — rubric REJECTION battery, 14 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite — lid fully shut in its channel, red can upright on the
   ground, everything at rest
2. settle: score ~0 at reset, no success
3. randomization readback: hamper yaw + xy vary
4. randomization readback: both can positions vary (incl. band swaps); cans ≥ 9 cm
   apart at every seeded reset
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: can released above the SHUT hamper settles ON the lid —
   never inside (the aperture is physically blocked → open-before-insert is forced by
   geometry, not just unscored), no success, score ≤ 0.05
7. wrong object: beige distractor enclosed in the shut hamper, red can outside → no
   success, score ≤ 0.05
8. near-miss: red can inside but lid settled ~30 mm from shut (tol 12 mm) → NOT
   success, score < 0.9 (the closed tolerance is load-bearing)
9. wrong place: red can settled on the hamper's DECK (on the fixture, wrong region) →
   not inside, no success, score ≤ 0.05
10. incomplete: red can inside but lid left fully OPEN → NOT success, score ≤ 0.85
    (the re-close stage is load-bearing — this is the seed's own end state, "can in an
    open container", and it is rejected)
11. monotonicity: fully-open probe latches strictly more open credit (and score) than
    a half-open probe
12. latched credit + close gating: sliding the lid back shut with NO can inside leaves
    the latched score unchanged (open credit kept; close credit gated on insertion),
    still no success
13. rejection audit: success() never True at any judged point of this battery
14. final no-NaN

N/A notes: **out-of-order end states** — "insert before open" is not constructible as
a settled state (the shut lid blocks the aperture; #6 is exactly the attempt); "close
before insert" IS constructible and is #12 (it earns nothing beyond the open credit).
A can wedged between lid and cavity cannot rest (the lid underside at 146 mm clears
the tallest can pose at ~117 mm), so no separate wedge control is expressible.

Video frames are recorded throughout and saved to `frames.npz` in the working
directory.
