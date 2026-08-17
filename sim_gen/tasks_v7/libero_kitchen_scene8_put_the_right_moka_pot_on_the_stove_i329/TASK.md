# libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i329 — rake the copper pot out of the roofed cubby, then stove it

## Provenance

Seed: `libero_90/libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove` —
pick "the right" moka pot (a discrimination between two pots standing in the
open) and set it on the stove: a direct pick-and-place in which the hand can
reach the goal object from the very first step.

## Strategic difference from the seed

The end state deliberately overlaps the seed's (the copper pot upright on the
burner) — what changes is **reachability** and the plan that restores it. The
two pots start ~180 mm deep inside a low roofed **CUBBY**: two lanes, each
85 mm wide and 150 mm tall, separated by a centre rib, open only at the front.
No hand or gripper fits down a lane and the roof kills every top-down grasp —
the goal object is unreachable by prehension. The counter carries the remedy:
a 400 mm **RAKE** (flat bar, red hook flange at the far end). The solver must

1. slide the rake nose-first into the COPPER pot's lane, riding *above* the
   pot;
2. lower it so the flange drops into the gap behind the pot's mushroom-head
   lid knob, hooking the head;
3. **drag** the pot down the lane and out of the mouth onto the slick apron —
   a sub-tipping friction pull (slick floor + smooth pot base, pair-averaged
   μ_eff asserted ≥ 2× below the tipping threshold at the catch height) that
   is the load-bearing mechanic;
4. only then place the freed pot upright on the burner, leaving the STEEL
   distractor untouched in its lane and parking the rake clear of the stove.

A solver needs a different plan (tool acquisition → guided insertion → hook →
extraction drag → place, where the placement is physically impossible before
the extraction) and different code structure (the core interaction is a
held-tool contact drag through a confined corridor, not a grasp of the goal
object). Distinct from sibling i167 (falsework prop: erect a strut that
carries a live gravity load): here nothing is held open and nothing stands
under load — the gate is REACH, and the tool leaves the scene when done (the
success clause demands the rake clear of the burner). No other task in this
corpus is a recessed-object tool-retrieval.

## Scene

Fully procedural (native PhysX box colliders), on a 900×750×24 mm kinematic
counter **deck**:

- **cubby** (kinematic): floor slab + back wall + side walls + centre rib +
  roof; two lanes 85 wide × 260 deep × 150 mm tall, mouth facing +x. The
  floor is SLICK (μ≈0.04) and extends 100 mm past the mouth as a pale apron.
- **stove** (kinematic): base plate + raised octagonal burner pad (96 mm
  across, grippy top μ≈0.95) on the open counter, front-right.
- **rake** (dynamic, 0.30 kg): 400×40×8 mm bar, 22 mm red hook flange at the
  +x end, grip block at the −x end.
- **pot_t / pot_w** (dynamic, 0.28 kg): identical octagonal moka pots (50 mm
  across-flats, 88 mm tall) with a mushroom-head lid knob (24 mm head — the
  rake's catch); COPPER = target, STEEL = distractor. No side handles: a pot
  fits its lane at any yaw (star-corner radius of the union-of-two-squares
  octagon is the asserted worst case).

Randomization (readback-verified in smoke): which LANE the copper pot spawns
in is a coin flip; both pots jitter in depth (±12 mm), cross-lane (±2 mm) and
free yaw; the rake jitters in xy (±15 mm), yaw (±6°) and a 50 % end flip.

Honesty-by-construction asserts in `SceneCfg.__post_init__` (12 groups): lane
fit at any yaw + jitter, insertion corridor over the knob and under the roof,
engaged-flange hook geometry, flange descent gap behind the deepest knob,
sub-tipping drag margin (2×), burner grip, rake reach with ≥10 cm of bar
outside the mouth, apron length, lane width vs bar/flange, spawn bands, exit
clearance under the roof edge, and zone separations on the deck.

## Rubric

Latched partial credit, capped at 0.75 unless success holds live:

| credit | clause |
|---|---|
| 0.20 | reached: copper pot dragged ≥ 40 mm mouth-ward while inside the cubby (latched) |
| 0.30 | extracted: copper pot fully outside the mouth, upright at rest on the apron (latched) |
| 0.25 | on_stove: copper pot upright at rest on the burner pad (latched) |
| 1.00 | `success()` live |

`success()` (all live) = copper pot upright on the pad AND the steel pot
still inside the cubby AND the rake bar (as an xy segment) ≥ 10 cm from the
burner centre AND rake + both pots at rest AND finite. The steel pot on the
burner, a tipped pot, a pot abandoned on the apron (caps at 0.50), the steel
pot disturbed out of its lane, or the rake left against the stove all refuse
success.

## Solution outline (solve.py — teleport-transport contract)

Teleports carry ONE object at a time through free space; the rake is wielded
by an external 6-DOF wrench (gravity feedforward + PD with a lag-clamped
moving reference); the extraction drag is pure contact dynamics:

- **P0** settle 120 steps; assert both pots deep in their lanes, score ≈ 0.
- **P1** teleport the rake to a hover aligned with the copper lane, wrench it
  nose-first into the lane riding above the pot, then lower until the flange
  sits behind the knob (score still 0 — insertion earns nothing).
- **P2** drag mouth-ward: the flange hooks the knob head and the pot slides
  (never tips) down the slick lane and out onto the apron; hold until calm →
  reach + out latches, 0.50. Stall detection with up to 3 lift-and-retry
  attempts.
- **P3** lift the rake free, teleport-park it at its spawn corner, drop the
  wrench, settle; assert `rake_clear`.
- **P4** teleport the copper pot to a hover over the burner; it drops and
  seats upright on the grippy pad; success must hold 240 consecutive substeps
  → 1.0.
- **P5** ≥ 3.3 s fully hands-off persistence (10 × 40 substeps, success at
  every checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified
on the forge for seeds 0 and 1 (`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

Every interaction is tabletop manipulation of hand-sized rigid objects on a
0.90×0.75 m counter. The rake's grip block is a canonical pinch feature; the
asserted geometry keeps ≥ 10 cm of bar outside the mouth even at the deepest
pot, so the hand never needs to enter the cubby — the arm holds the grip end
exactly as the wrench does and performs the same guarded insert / 20 mm lower
/ horizontal drag, all slow quasi-static motions of a 0.30 kg tool. The drag
force is bounded by μ_eff·m·g ≈ 0.25 N — trivial for the arm, and the
sub-tipping assert means no delicate force control is needed. The freed pot
(50 mm across, 0.28 kg, standing in the open on the apron) is a standard
top-down grasp; the placement is a vertical set-down onto a 96 mm pad with
±40 mm tolerance. With the base at the counter's front edge all targets
(cubby mouth ~0.37 m, apron 0.37–0.47 m, stove 0.66 m, rake 0.40–0.80 m) sit
inside the Franka's 0.855 m reach, and every manipulated feature is between
24 and 90 mm above the deck — no confined-space wrist poses required.

## Execution order

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the task and the
   monotone score trace.
2. `smoke.py` — rejection battery, 11 checks:
   1. settle/no-NaN (both pots upright deep in their lanes, burner empty,
      score ≈ 0);
   2. randomization readback (copper depth + yaw, steel yaw, rake xy; 3-seed
      max-pairwise deltas);
   3. copper-lane coin flip (both lanes seen over 10 resets);
   4. null policy 240 steps (score ≤ 0.05, no success);
   5. roof-jam (a 1.3×-weight lift wrench presses the pot up: verified risen,
      it jams under the roof and never leaves the cubby — the reach gate is
      physical, not scripted);
   6. wrong pot (STEEL seated on the burner, verified: score ≈ 0);
   7. copper pot TIPPED on its side on the burner (upright clause refuses);
   8. apron-only (dragged-path to the apron and abandoned: caps at 0.50);
   9. steel pot displaced out of the cubby FIRST, then the copper pot fully
      delivered: all latches fire (0.75) but success refused;
   10. rake parked inside the burner clear-radius FIRST (verified), then the
       copper pot fully delivered: 0.75, success refused;
   11. frames.npz video.
3. Probe constructs are ordered so no prefix of any check satisfies the goal.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS 11/11`) and hard-exit; a top-level try/except prints
a FAIL verdict on any exception so the forge watchdog never hangs.
