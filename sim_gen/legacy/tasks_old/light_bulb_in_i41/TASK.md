# light_bulb_in_i41 — `bulb_triage` (hidden-state diagnosis, then conditional routing)

**Env name:** `simgen.bulb_triage` (scene `bulb_triage`, robot `null` — scene-level;
robot bindings are a later stage)
**Tier: hard — 5-8 stages** depending on the sampled hidden state: k test cycles
(seat in the tester + hold the dwell + extract, k = 1..3 until the working bulb glows),
one verified install into the lamp socket, and two disposal placements into the bin.
Worst case (working bulb tested last): 3 tests + 1 install + 2 disposals = 8 pick-place
operations, two of them precision collar insertions (4 mm radial funnel).
**Execution order: PARTIALLY REQUIRED** — verification must precede installation (the
protocol gate makes the order test-then-install mandatory, and each tester cycle must
finish before the next bulb can be tested: the collar holds one bulb); the interleaving
of disposals with tests/install is free.

## Seed provenance

- Seed id: `rlbench/light_bulb_in`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/light_bulb_in.py`
- The seed is RLBench's light-bulb-in: two bulbs on stand-holders next to a lamp base;
  the task is to pick the (known) target bulb off its holder, carry it to the lamp and
  screw it into the socket. One object, one known goal pose — the whole plan is a
  single transport with a precision insertion at the end.

## What changed, and why it is strategically different

Kept from the seed's world: light bulbs, a lamp with an empty socket, and "put the
bulb in the lamp" as the terminal act. Changed: **the plan class — from transport to
information-gathering.**

1. **The goal-relevant property is HIDDEN STATE.** Three visually identical
   frosted-globe bulbs (stems color-coded only for reference); exactly one works,
   sampled per episode, unobservable at reset (the smoke asserts the three globes read
   back the identical displayColor). In the seed, *which* bulb belongs in the lamp is
   given; here it is unknowable without acting.
2. **A physical TEST is the only way to observe it.** Seating a bulb stem-down in the
   tester collar and holding it seated + settled for a real dwell (24 consecutive
   physics substeps) latches `revealed[i]` and recolors the globe — amber glow
   (working) or smoke-gray (dead). A teleport through the seat pose reveals nothing
   (tested). The tester is a one-bulb station, so tests are sequential cycles:
   insert, dwell, extract.
3. **The seed's own plan is a tested failing control.** Carrying a bulb straight to
   the lamp — even the actual working bulb, seated perfectly — is an UNVERIFIED
   install: capped at 0.05, never success (negative A). The lamp accepts only a bulb
   that has been seen to glow (`revealed[working]` is required by success and by the
   install credit) — the QC-protocol port of the seed's "the correct bulb", made
   physically checkable.
4. **The outcome of the test CONDITIONS the rest of the plan.** Each bulb is routed by
   what the tester showed: glowing bulb -> lamp socket; dead bulbs -> disposal bin
   (routing inverted = score 0, negative B). Once the working bulb has glowed, the
   remaining bulbs are dead by inference and may be binned untested — so both the
   required plan LENGTH (1-3 tests) and the routing differ every episode. A memorized
   fixed sequence cannot solve the family.

Strategy axes claimed (vs sibling generated tasks): **hidden state revealed by a
physical probing action**, **verify-before-commit protocol gating**, **conditional
routing / sort-by-discovered-property**. (Distinct from i28's buffer-forced move
sequencing, i14's identity permutation — visible there — i18/i21's dwell used as a
cooking hazard: here the dwell is an information-gathering act, and nothing about the
correct final arrangement is observable at reset.)

## Scene summary

Fully procedural: three dynamic compound bulbs (27 mm sphere globe + 11 mm colored
stem cylinder), a kinematic tester pedestal (recess floor 40 mm, cyan collar), a
kinematic lamp (bronze column, gold collar socket at 85 mm), a kinematic open bin
(190 mm interior, 70 mm walls). Collar apertures are octagonal, inradius 15 mm ->
4 mm radial funnel around the stem (measured in the smoke sweep). Honesty by
construction: max captive lean in a collar ~15 deg < the 22 deg upright gate; max
in-bin resting offset 68 mm < the 70 mm gate; a bulb stacked on another exceeds the
bin z gate.

Randomization per episode: hidden working index (3), bulb spawn-slot permutation (6),
xy jitter + free yaw per bulb, xy jitter for all three fixtures.

Rubric (score in [0, 1], 0 for doing nothing, latched reveals): +0.08 per bulb ever
revealed, +0.16 once the working bulb has glowed, +0.15 per dead bulb resting in the
bin, +0.30 verified install (0.05 if unverified), capped at 0.95 unless success;
exactly 1.0 iff success = verified working bulb seated + settled in the lamp AND both
dead bulbs binned + settled.

## Smoke battery (22 checks)

1. reset-sane (finite states, bulbs lying, score exactly 0, nothing revealed)
2. hidden state invisible at reset (globe displayColor readback identical)
3. randomization: working index + spawn permutation vary (readback)
4. randomization: fixture jitter real (readback spread)
5. null policy fails (2 s idle -> score 0, no reveals)
6-8. oracle x3 seeds (routing decisions read only the latched reveal): success, 1.0
9-14. rubric ladder, working forced last: 0.08 / 0.23 / 0.31 / 0.46 / 0.70 / 1.0,
      strictly increasing, with recolor readbacks (gray at 9, amber at 13)
15. negative A — the seed's carry-and-insert: working bulb seated perfectly in the
    lamp UNTESTED -> 0.05, no success
16. negative B — routing inverted (dead installed, working binned) -> exactly 0
17. negative C — near-miss: verified install but one dead bulb not disposed -> 0.77,
    no success
18. anti-teleport: drive-through seat (3 substeps < 24-substep dwell) reveals nothing
19. negative D — bulb lying across the collar mouth never reveals (upright/depth)
20. reset clears latches and restores frosted globes (readback)
21. calibration sweep: centred + 3 mm drops seat 3/3 (4 mm funnel is real)
22. calibration sweep: 35 mm (past the collar) never seats

Physics notes: reveal dwell counts consecutive seated+settled substeps in
`post_step` (buffers fresh per substep), so only genuine held-in-place physics
latches it; bulbs carry a 0.5 m/s depenetration cap and mild damping (top-heavy
rocking in the collar); collar contact offsets 1 mm so the 4 mm funnel is not eaten
by speculative contact.
