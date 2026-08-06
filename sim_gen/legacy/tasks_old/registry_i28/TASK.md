# registry_i28 — `musical_cans` (buffer-constrained rearrangement puzzle)

**Env name:** `simgen.musical_cans` (scene `musical_cans`, robot `null` — scene-level;
robot bindings are a later stage)
**Tier: medium — 3-4 stages** (one pick-transit-seat move per stage: 3 moves when two
cans are present, 4 moves when all three are). **Execution order: REQUIRED** — a move is
only physically possible into an *empty* cup, so the buffer-parking move must come first
and the cascade order after it is forced by the sampled permutation.

## Seed provenance

- Seed id: `simpler_env/registry`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/simpler_env/_metasim/registry.py`
- The seed file is the registry of the whole SimplerEnv suite: 25 tasks whose plans are
  each ONE free manipulation primitive — pick the coke can, move object A near object B,
  put the carrot on the plate / eggplant in the basket, open/close a drawer,
  open-then-place. In every one of them the commanded destination is **free**: the
  solver transports the object and releases it, and the world never resists the plan.

## What changed, and why it is strategically different

Kept from the seed's world: upright drink cans on a kitchen counter, and instructions of
the seed's very shape ("put the can at X", "move A near B"). Changed: **the plan class.**

1. **Every direct placement is impossible at reset.** The start arrangement is a sampled
   *derangement*: each of the color-coded pedestal cups that a can belongs in is
   occupied by a *wrong* can, and a cup physically seats only ONE can (60 mm can, 76 mm
   aperture — a second can ends up stacked 95 mm too high or toppled, never seated; the
   smoke's negative A executes the seed's own put-on plan and it fails). The seed's
   single-primitive plan is not merely insufficient — its first step cannot be completed.
2. **The seed's escape hatch is a tested failing strategy.** In every SimplerEnv task the
   table is a legal resting place; here the counter is forbidden (a present can that
   ever comes down to counter level latches an irreversible `dropped`, capping the score
   at 0.25 and killing success forever — negative B *completes a perfect arrangement*
   after one counter touch and still fails).
3. **The solution is a computed move SEQUENCE, not a placement.** The only legal
   waypoint is one spare gray cup. Solving is cycle decomposition through that buffer:
   park one can, cascade the freed cups home, retrieve the parked can —
   `n_present + 1` moves whose order is forced by occupancy. The smoke *asserts* the
   oracle needed exactly `n_present + 1` moves on every seed: the n-move "just put each
   can where it belongs" plan does not exist in this world.
4. **Randomization changes the plan, not the numbers.** Which derangement is sampled
   (either 3-cycle, or any of the three 2-can swaps with the third can absent) changes
   the *length and order* of the required move sequence every episode; pedestal jitter
   changes the metric targets on top.

So a solver needs a different plan (read the permutation → schedule moves through a
buffer under an irreversibility constraint), not different parameters of the seed's
plan. This is a state-space puzzle in the Tower-of-Hanoi family expressed with the
seed's own objects.

### Differentiation from sibling generated tasks (claimed-axes check)

No sibling claims occupancy-conflict / permutation-scheduling: i4/i9 claim clearance
confinement, i11/i17 stacking or demolition, i12 size-keyed matching (geometric keying —
here all cans are the same size and the constraint is *logistical occupancy*), i13
unload-before-close, i14 orientation-selective apertures, i15 fetch-restore through a
drawer, i18/i21 dwell timing, i20 interception (its irreversible-loss latch is reused
here only as a supporting fence — the claimed core axis is the buffer-forced move
scheduling, which i20 does not have).

## Scene summary

Kinematic counter slab; four kinematic compound pedestals (column + shallow octagonal
collar cup, custom `clone()` spawner): red/green/blue home cups + gray buffer, in a
jittered row; three dynamic cylinder cans color-matched to the home cups. Fully
procedural, no external assets. Physics honesty: the oracle places kinematically, but
`correct`/`success`/`score` judge only settled physical poses (seat height, cup-axis
xy, upright cone chosen above the max physical in-cup lean, velocity gates) plus latch
history written by real fall physics.

Rubric: 0 for doing nothing (a deranged start has zero correct cans by construction);
0.05 latched first lift + 0.10 latched buffer use + 0.55 × fraction of present cans
seated home; capped at 0.25 forever once `dropped`; **1.0 iff success** (all present
cans home, settled, nothing ever dropped). Max non-success ≈ 0.52.

## Smoke check list (22)

1. reset sane: finite states, genuine derangement readback, zero correct
2. reset: score exactly 0, no success
3. randomization: arrangement varies, both subset sizes occur (readback)
4. randomization: pedestal jitter moves (readback)
5. null policy (2 s idle): score 0, no latches, no success
6-8. oracle ×3 seeds: success(), score 1.0 (kinematic transit + real 15 mm seating drops)
9. move economy: every seed solved in exactly n_present+1 moves (buffer detour forced)
10-14. rubric ladder: 0.05 → 0.15 → 0.333 → 0.517 → 1.0, non-decreasing, partials < 1
15. negative A (the seed's own strategy): direct put-on with the home cup occupied —
    stacked/toppled, zero correct, no success
16. negative B: one counter touch latches `dropped`; a subsequent PERFECT arrangement is
    all-correct yet success-less, score pinned at the 0.25 cap
17. reset clears latches
18. negative C (near-miss): can lying across the cup mouth — xy and z inside tolerance,
    rejected by the upright clause alone (and no buffer latch)
19. negative D: seated in the WRONG (buffer) cup is not `correct`, no success
20. negative D recovery: the oracle finishes from that mid-game state → success
    (tolerances honest, state recoverable)
21. calibration sweep: centred and funnel-edge (5 mm) drops seat 3/3
22. calibration sweep: a 40 mm offset never seats (middle offsets published, not
    asserted — raw drop physics is stochastic, measured)

Result: `SIM_GEN_SMOKE: ALL PASS 22/22` on the forge (see run log).
