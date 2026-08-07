# moat_causeway — lay the channel plank across the moat, push the oversized cube over the bridge onto the pad

Env: `simgen.moat_causeway` · Scene: `moat_causeway` · Robot: `null` (rubric + solution harness)

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene5_put_the_black_bowl_on_the_plate`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_on_the_plate.py`).
The seed is a single prehensile transport on one kitchen counter: grasp the black bowl,
carry it through free space, set it down on the plate; its checker is xy proximity
(< 6 cm) plus a small height band between two co-planar objects.

## What changed

| | seed | this task |
|---|---|---|
| payload | graspable bowl | 110 mm RED cube — **wider than the 80 mm Franka jaw: push-only** |
| terrain | one flat counter | two elevated platforms separated by an open **moat** (randomized 14–18 cm gap, deeper than the cube) |
| core act | carry through free space | **build the road first**: lay the BLUE channel plank so both ends rest on the deck rims, then push the cube across it |
| goal check | xy proximity + height band | support-span predicate (plank geometry in the layout frame) + **crossing pathway latch** (cube observed supported over the void on a live bridge) + pad rectangle + settled |
| distractor | none relevant | WHITE decoy plank, **shorter than every gap** — cannot span at any pose |
| randomization | fixed scene | layout centre xy + FREE yaw, moat gap width, cube deck pose, plank/decoy side + ground pose + free yaw |

## Why strategically different

- **From the seed:** the seed's entire plan (grasp payload → carry → set down) is
  impossible here by construction — the payload cannot be grasped (wider than the jaw)
  and the goal cannot be reached without first creating the path (moat wider than the
  cube, deeper than the cube: a dropped cube is lost below deck level). The seed's
  strategy, executed literally (payload set down at the goal), is smoke-tested to score
  ~0 and never succeed (no crossing pathway).
- **From the corpus:** no other task uses *structure-as-road then non-prehensile
  traversal*. `close_grill_i8` builds a structure that IS the goal; `pour_water_i7`
  places a passive chock; `living_room_i18` uses a pre-existing runway to drain a
  hopper; `obstacle_i17`/`peg_i1` use tools ballistically or as ramrods. Here the
  placed object (bridge) is *infrastructure* whose correctness is judged by support
  geometry, and the payload then *rides* it under a bounded push — a two-act plan with
  a physically forced order (the push is impossible until the bridge exists).

## Rubric

`success()` = cube ON the yellow pad (target-frame rectangle, deck-height z band)
AND crossing pathway latch (cube seen supported over the open moat — deck-height band,
low |v_z| — while the plank was live-spanning) AND everything settled.
`score()` = 0.25·span_latch + 0.30·cross_latch + 0.20·arrive_latch (capped 0.75);
exactly 1.0 iff success. All latches update every physics substep and never evaporate.

Honesty asserts in `__post_init__`: cube > jaw span (push-only); every gap > cube
(cannot self-bridge) and moat deeper than cube (fall = unrecoverable); decoy < every
gap (cannot span); plank overhangs the widest gap with real support margin on both
rims; channel passes the cube with clearance; curb rail graspable; pad beyond the
plank landing zone; cube spawn clear of the landing zone.

## Solution outline (solve.py — the legitimacy certificate)

- P0 reset + settle; layout readback (centre, free yaw, gap, cube lane, plank side).
- P1 **transport only**: the plank teleports to a hover 30 mm above deck-top height
  over the moat centre, aligned with the crossing axis (read from the scene), and is
  RELEASED — gravity and real contact seat both ends on the rims (retry centred once).
- P2 **driven, not bypassed**: a force governor pushes the cube along the crossing
  axis (≈2–4.2 N: PD to 0.07 m/s plus a quasi-static stall boost; mounting the 12 mm
  channel-floor lip is a quarter-tumble driven at a constant just-above-tip-over
  ~2.9 N while the CoM rises, near-zero on the falling side), lateral PD centring on
  the plank lane, gentle yaw-keeping torque. The cube climbs onto the channel floor, crosses
  the void riding the bridge, descends onto the green deck, and is braked to rest on
  the pad. Fall below deck height at any point = hard FAIL.
- P3 hands-off persistence ≥ 3.3 sim-seconds, then `SIM_GEN_SOLVE: SUCCESS`.
`SIM_GEN_SCORE` printed at every phase boundary; asserted non-decreasing.

## Embodiment argument (single Franka, 80 mm parallel jaw, OSC)

- **BLUE plank (prehensile):** grasped by a curb rail — 8 mm thick, 30 mm tall,
  comfortably inside the jaw span; the rail's top edge is ~42 mm above the ground so
  fingers can straddle it at ground pickup. 0.80 kg is well inside Franka payload.
  Laying = hover over the moat, level, release — exactly solve.py's P1.
- **RED cube (non-prehensile):** 110 mm > 80 mm jaw, so the only contact strategy is
  pushing on a vertical face at mid-height; the required 2–4.2 N is trivial for the
  arm; the curb rails passively keep the cube in the lane so a straight-line push
  suffices.
- **One plausible base pose:** layout-frame (0, ±0.62) on the plank's spawn side,
  base facing the moat. From there: plank spawn |y| 0.36–0.46 → 0.16–0.30 m reach;
  moat centre 0.62 m; cube spawn ≤ 0.70 m; pad ≈ 0.66 m — all inside the Franka's
  ~0.85 m reach envelope at deck height 0.12 m.

## Execution order

**REQUIRED, physically forced:** the bridge must exist before the cube can cross
(the moat is wider than the cube and deeper than the cube; there is no other path).
Pushing first loses the cube irrecoverably (smoke check "moat drop"). Laying the
plank after the cube is at the rim wastes nothing but cannot be skipped.

## Checks (smoke.py — 15, all constructed-wrong-outcome rejections)

1. reset settles finite, cube on the start deck (readback height), planks on ground
2. reset score ≤ 0.02, no success
3. randomization readback: centre xy + free yaw + gap vary across 8 seeds
4. randomization readback: cube xy, plank/decoy ground poses vary; plank on BOTH sides
5. null policy: 240 idle steps → score ~0, no success
6. **SEED strategy**: cube written directly onto the pad, settled — on_pad verified
   True, crossing latch stays 0 → NOT success, score ~0
7. moat drop: cube dropped into the bare moat ends below deck level; falling through
   the crossing band earns nothing (|v_z| gate + no live bridge)
8. decoy cannot span: released over the moat centre it falls; span latch stays 0
9. bridge-only: honest lay → spanning verified, score ~0.25, no success
10. short lay: one end short of the far rim → tips, never spanning, score ~0
11. mid-bridge stop: cube dropped onto the spanning bridge — crossing latch fires but
    NOT success, score ≤ 0.56
12. arrived-not-on-pad: cube on the green deck outside the pad — arrival latch fires,
    NOT success, score ≤ 0.76
13. latched credit survives removing the cube (score unchanged, success gone)
14. rejection audit: success() never True anywhere in the battery
15. final no-NaN
