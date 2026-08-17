# insert_onto_square_peg_i148 — `hook_escape`

Env: `simgen.hook_escape` (robot="null") · Files: `scene.py`, `solve.py`, `smoke.py`

## Seed provenance

Seed task: **rlbench/insert_onto_square_peg** (`RoboVerse/roboverse_pack/tasks/rlbench/insert_onto_square_peg.py`) — pick up a free square ring and drop it DOWN over the correct one of three colored vertical pegs; one straight vertical insertion, judged by the ring encircling the chosen peg; the choice lives in the DESTINATION.

## What changed and why it is strategically different

The seed is **inverted and path-constrained**:

- **Insertion → constrained extraction.** The rings START threaded on a post — the seed's goal state (ring encircling a vertical peg) is this task's *reset* state and scores ~0 (smoke check 5 constructs it fresh and asserts rejection). The work is getting a ring OFF.
- **Straight vertical stroke → 3-segment escape path with a mid-path reorientation.** The rail is an inverted-L (post → 90° elbow → horizontal arm with an OPEN tip). The only topological exit is: slide UP the post, pitch the ring ~90° through the elbow, run OUT along the arm, off the tip. Captivity is physically real: smoke force-probes prove a quasi-static ~1.8×-weight straight-up pull jams the ring under the arm and a sideways yank cannot detach it.
- **Choice moved from destination to object.** One dish, two same-size rings (BLUE target, ORANGE decoy). On ~half the episodes the decoy spawns ABOVE the target and must be shepherded off the hook first — a conditional ordering the seed does not have.
- **New terminal predicate.** Success is a settled containment: blue ring flat on the dish floor inside the walls, fully off the rail, decoy NOT in/on the dish.

Differences vs the corpus tasks read this session are documented in the `scene.py` module docstring (pen_holder, peg_insertion_side_i1's ramrod eject, screw_nail_i59's latch vault, lift_peg_upright_i116's hood prop): no insertion-into-container, no tool/third-body ejection, no jointed latches (the "lock" is pure topology), no propped-statics goal.

## Teleport-solution phases (solve.py — the legitimacy certificate)

Teleports are TRANSPORT ONLY; the whole escape happens through contact dynamics under applied wrenches (a floating force/torque servo standing in for a gripper carry: F = Kp·(carrot−p) − Kd·v + m·g·ẑ, τ = Kq·(â×t̂) − Kw·ω, gains bounded by the one-substep wrench delay; pod frame-drag pre-encoded R_ref·R_now^T with a runtime progress probe that flips the mode then escalates gains).

- **P0 readback**: settle; read rack pose (the arm heading is randomized), dish position, stack order. Score ~0 asserted.
- **P1 (orange-on-top episodes)**: servo the ORANGE ring up the post, through the elbow, off the tip; once `rail_dist` certifies it free, teleport it (transport of a free object) to a park spot on the far side of the rack from the dish.
- **P2 climb+elbow**: servo the BLUE ring up the post and pitch its bore axis from world-up to along-arm through the elbow (score → ~0.29).
- **P3 arm run**: servo along the arm and off the OPEN tip; hover past it; `rail_dist > free_tol` asserted (score → ~0.65).
- **P4 place**: teleport the free ring flat 3 cm above the dish floor, release, settle (score → 1.0).
- **Persistence**: ≥3.3 simulated seconds hands-off; success must still hold.

Forge: `SIM_GEN_SOLVE: SUCCESS` on seeds 0 (ORANGE on top), 1 and 2 (BLUE on top) — both stack orders demonstrated. `SIM_GEN_SCORE` prints are non-decreasing (0.00 → [0.00] → 0.29 → 0.65 → 1.00), stage credit latched in `post_step`.

## Embodiment argument (single Franka arm + parallel jaw)

- **Base pose**: on the floor between rack and dish, ~0.45 m from the rack, facing it (e.g. rack at (0.45, 0) and dish within ±0.4 m in base frame). All work heights are ≤ 0.35 m (elbow) and ≥ floor level (dish), inside the Franka's ~0.85 m reach envelope from that stance.
- **Rings**: 14 mm-thick flat washers with a 16 mm radial band — a parallel jaw (80 mm max aperture) pinches the rim flats anywhere. The escape is a slow guided carry: grasp the rim, slide up the post (bore slack 72 mm vs 18 mm rail is 4×, no precision jam), rotate the wrist ~90° through the elbow (the solve's ≤0.08 N·m alignment torques and ≤2.5 N guide forces are trivially within Franka wrenches), run outward along the arm, and the ring is free in-hand — exactly the solve's wrench servo with a real grip. Regrasping at the hover is allowed but unnecessary.
- **Placement**: carry ~0.4 m to the dish, hold flat a few cm up, open the jaw — the solve's P4 drop is precisely what release-from-carry produces.
- **Decoy**: identical handling; park it by opening the jaw anywhere away from the dish.
- **Rack / dish**: kinematic scenery; never manipulated.

## Execution-order declaration

Ordering is **conditional and physically enforced**: when the orange ring lies above the blue one, the rail topology blocks the blue ring's exit until the decoy is off (declared in `describe()`/`instruction()`, exercised by solve seed 0). No other ordering is imposed; the rubric latches stage credit in any order but success requires the terminal containment state with the decoy clear.

## Rubric

`success()` = in_dish(blue: xy ≤ 5.3 cm of dish axis, center z in the flat-on-floor band 1.2–2.8 cm, plane ≤20° from horizontal) ∧ settled(blue) ∧ free_of_rail(blue, `rail_dist` > 12 cm) ∧ decoy_clear(orange outside the r=13 cm, z<8 cm exclusion cylinder).

`score()` = 0.25·climb (on-rail rise above a deadband set above both rest heights) + 0.25·arm travel (on-rail, at arm height) + 0.15·freed + 0.15·in-dish — all latched running-max for the BLUE ring only — capped at 0.80; exactly 1.0 iff success. Null policy ~0. Honesty bounds (`cfg.__post_init__` asserts): elbow passable by construction, climb deadband above both rest heights, dish tolerance admits every physically-inside flat ring (corner-reach bound), dish never under the arm sweep (a ring dropped off the tip cannot land in it).

## Smoke battery (rejection checks, `SIM_GEN_SMOKE: ALL PASS 14/14` on forge)

1. settle/no-NaN — rings threaded on-rail at rest, score ~0, no success
2. randomization readback — rack xy+yaw and dish position vary across 12 seeded resets
3. randomization readback — BOTH stack orders occur
4. null policy — 300 idle steps, score ~0
5. seed strategy — ring dropped DOWN over the post (the seed's goal), settled: score ~0
6. captivity (up) — quasi-static velocity-servoed pull (≤1.4 N ≈ 1.8× weight, drag-pre-encoded to stay world-up) climbs the ring (readback: it MOVED >10 cm) but jams under the arm, never leaves the rail (ends on-rail, free latch 0); partial credit only, ≤0.35
7. captivity (side) — 1.5 N yank displaces but cannot detach; score ~0
8. partial escape — ring left dangling on the arm: partial latched credit only (≤0.47), no success
9. near-miss — blue flat+settled on the floor beside the dish: free credit only (≤0.17)
10. near-miss — blue perched on/tipped off the dish wall: outside xy/z acceptance
11. wrong ring — orange laid flat in the dish: earns NOTHING, decoy_clear False
12. z-band + decoy — blue stacked flat ON TOP of in-dish orange: xy/flat pass but z-band rejects, decoy_clear False
13. audit — success() never True at any judged point
14. final no-NaN

frames.npz recorded via the viewport rgb annotator.
