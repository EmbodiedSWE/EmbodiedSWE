# flat_pack_crate — assemble the shipping crate around the salad dressing

## Seed provenance

Seed: `libero/libero_pick_salad_dressing` — pick the amber salad-dressing bottle
out of a clutter of grocery distractors and place it into a basket that already
exists on the table.

## What changed and why it is strategically different

The seed (and the sibling variants examined: i43 cellar tow, i51 pile driver,
i104 rocker lock, i119 silo tip-pour, i18 wedge hopper, i46 pudding sift, i129
ballast vault, i52 ice doser) all share one structural assumption: **the
receptacle exists at reset** and the strategy is about how the object travels
into it (pour, tow, ratchet, meter, vault...).

This task inverts that assumption: **the receptacle does not exist at reset.**
A flat-pack shipping crate lies disassembled on the table — a heavy base pallet
with two fixed walls and two open sides, two loose side panels lying flat, a
lipped lid lying upside-nowhere, the amber dressing bottle, and a red decoy
bottle. The strategy is *construction*: slide each panel down its funnel
channel on the base until it seats between the retaining rails, stand the
dressing bottle upright on the pad inside the part-built crate, and finally
seat the lid — its underside lip must drop inside the wall/panel mouth so the
lid registers square. The decoy must be left outside. The manipulation is
about assembly order and insertion tolerances, not about transporting an
object into a pre-existing container. No lever, no hinge, no counterweight,
no impact mechanism — none of the sibling mechanisms reappear.

Execution-order structure (declared, enforced, and probed):

- The two panels and the bottle may go in **any order among themselves**.
- The **lid must be last**. Enforcement is physical + latched:
  - the seated lid covers the slot mouths and the interior, so late panel or
    bottle entries are physically blocked (smoke check 8 drops both against a
    seated lid and verifies neither can enter);
  - any slot/interior entry that first becomes true while the lid is seated
    trips an irreversible `_breach` latch (anti-teleport-cheat), which zeroes
    future credit and vetoes success (smoke check 9).

## Scene summary

- **Base pallet** (30 kg dynamic compound, free on the table): 0.30×0.30 pad,
  two fixed walls (+x and +y), and two funnel channels (−x "south", −y "west"):
  inner/outer rails 22 mm apart with 34 mm flared mouths and end posts.
- **Panels** (2, identical, 0.016×0.100×0.170, 0.12 kg): spawn lying flat at
  randomized stations; either panel fits either channel.
- **Lid** (0.30 kg compound): 0.20×0.20 plate with a 0.112×0.112 underside lip
  that must enter the crate mouth for the lid to register.
- **Bottles**: amber dressing (target) and red decoy, r=0.026 cylinders,
  spawn upright at randomized stations.
- **Randomization** (verified by readback in smoke): base yaw 180°±12° and
  ±30 mm translation; every loose part gets independent ±30 mm jitter and
  random yaw; bottle stations can swap.

## Teleport solution (solve.py)

Teleport = transport only; every load-bearing interaction is gravity +
contact through the funnel/rail/lip geometry.

1. **P0 settle** — 180 steps; assert score ≤0.03, no success.
2. **P1 panel A → south channel** — teleport to a hover pose above the funnel
   mouth (base-local, base-yaw-aligned), release; gravity slides it down the
   34→22 mm funnel until `_panel_seated`; settle. Retries at ±3 mm if needed
   (none needed on either seed). Score 0.12.
3. **P2 panel B → west channel** — same, rotated 90°. Score 0.24.
4. **P3 dressing bottle** — teleport upright to free air 10 mm above the pad
   centre, drop, settle. Score 0.40.
5. **P4 lid** — teleport level to free air 10 mm above the wall tops, drop;
   the lip self-registers into the mouth; settle. Live success → score 1.0.
6. **P5 persistence** — 400 further steps hands-off, score stays 1.0, then
   `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge with `--seed 0` and `--seed 1` (both rc=0, monotone score
trace 0.00 → 0.12 → 0.24 → 0.40 → 1.00).

## Rubric (written after the solution worked)

Latched credits, each gated on "finite ∧ no breach ∧ slow (<0.08 m/s)":
`0.12` south slot filled + `0.12` west slot filled + `0.16` dressing placed
inside + `0.20` lid seated, clamped at **0.60** for any non-success state.
Score = **1.0** iff live `success()`: both slots filled ∧ dressing inside ∧
decoy excluded ∧ lid seated (position, tilt <8°, yaw-registered) ∧ no breach ∧
settled (60-step still counter) ∧ all states finite.

## EMBODIMENT ARGUMENT (single Franka, one base pose)

Place the Franka base ~0.55 m from the crate centre on the −x/−y diagonal, so
its workspace covers both open channel sides, the scatter stations, and the
crate mouth (all parts live inside a 0.75 m disc around the base pallet).

- **Panels** (0.016 m thick, 0.12 kg): flat two-jaw pinch across the 16 mm
  thickness at the panel's top edge — well inside the Franka gripper's 80 mm
  stroke, and the 0.12 kg load is trivial. The funnel mouth gives ±9 mm
  lateral capture and the flare guides the final 4 mm alignment, so the
  required placement accuracy (~±8 mm, ±10° about the channel axis) is well
  within Franka repeatability. Insertion is a vertical guarded move; the rails
  take over compliance.
- **Bottle** (r=0.026, 0.15 kg): cylindrical wrap grasp around the barrel
  (52 mm diameter < 80 mm stroke); lift over the 0.17 m walls and lower to
  the pad — a plain top-down place with >40 mm of clearance to walls when
  centred.
- **Lid** (0.30 kg, 0.20×0.20 plate): rim pinch on the plate edge (10 mm
  thick) or a two-finger span across a corner; the 12 mm lip vs 130 mm mouth
  interior means ~±9 mm registration tolerance, and the lip chamfers the last
  alignment as it drops. Held moment (~0.3 N·m at plate edge) is far below
  the Franka wrist limit.
- **No step requires force closure beyond pinch**, no bimanual coordination,
  no regrasp is forced (each part is grasped once, placed once), and all
  drop heights match the solve's free-air release poses, which the physics
  demonstrably funnels to success.

## Files

- `scene.py` — `FlatPackCrateScene`, registered as `flat_pack_crate`,
  env `simgen.flat_pack_crate` (robot="null").
- `solve.py` — teleport solution, `python -m simgen_tasks.libero_pick_salad_dressing_i151.solve --headless [--seed N]`.
- `smoke.py` — 15-check rejection battery (below), records `frames.npz`.
- `TASK.md` — this file.

## Smoke checks (15) — forge result: `SIM_GEN_SMOKE: ALL PASS 15/15`

1. Settle/no-NaN: layout settles finite; slots empty, bottles upright on the
   open pad, lid off the crate; score ~0, no success.
2. Randomization-is-real: base yaw, base xy, panel and lid pose READBACK all
   differ across two seeds.
3. Bottle swap: the amber bottle occupies BOTH scatter stations over 10 resets.
4. Null policy: 240 idle steps, nothing seats, score ~0, no success.
5. Panel near-miss: a panel standing free on the pad (not in a channel) earns
   no slot credit.
6. Channel mechanism: a panel dropped in free air above the funnel is
   captured, guided and SEATED by gravity alone (probe verified to have
   fallen 80 mm), latching the 0.12 slot credit.
7. Negative (SEED strategy): bottle placed on the pad of the unbuilt crate →
   partial credit only (≤0.17), no walls, no lid, no success.
8. Lid blocks late entries: with the lid seated, a dropped bottle lands ON
   the plate (not inside) and a panel dropped over its slot mouth cannot
   enter (both probes verified fallen and rejected by contact).
9. Anti-teleport: a panel WRITTEN into its covered slot under a seated lid
   latches `_breach` — success permanently blocked, credit frozen (≤0.33).
10. Tipped bottle: the bottle lying on its side on the pad interior earns no
    placed credit.
11. Lid ajar: a 45°-yawed lid dropped onto the crate rests on top (probe
    landed at wall-top height) but does NOT register: no lid credit.
12. Legal assembly + no premature success: the aligned lid seats by gravity
    on the full crate → success; per-step assert that success never fires
    before the still window elapsed.
13. Live + cap: lid lifted off the succeeded crate → success OFF and score =
    latched cap 0.60 (float-safe bounds); re-seated by gravity → success
    returns.
14. Negative (wrong object): the RED decoy sealed inside instead of the amber
    bottle → decoy-exclusion clause fails, no success, credit ≤0.45.
15. Video: >10 frames captured and saved to frames.npz (248 frames).
