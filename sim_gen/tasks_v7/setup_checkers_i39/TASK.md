# checker_crypt — pry the crate open, retrieve only the red king

`sim_gen/tasks_v7/setup_checkers_i39` · scene id `checker_crypt` · env `simgen.checker_crypt`

## Seed provenance

Seed: `rlbench/setup_checkers`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/setup_checkers.py`).
The seed is a pure pick-and-place: checkers lie exposed on a board and must be
moved onto their starting squares. Its skills are locate → grasp → place,
repeated.

## What changed

| | seed | this task |
|---|---|---|
| Object access | pieces exposed from t=0 | pieces sealed inside a crate; **not graspable at reset** |
| Core skill | pick-and-place | **class-1 lever pry** through a notch to unseal, *then* selective retrieval |
| Tool use | none | pry bar must be found, inserted into a notch, and pressed down |
| Placement | many pieces to squares | ONE red king to a pad, with **restraint**: both whites must stay inside |
| Assets | RLBench meshes | fully procedural (compound USD spawners, no external files) |

The crate lid sits **flush** inside a raised rim ring: the lid top is level with
the rim top (no edge to pinch) and the rim blocks lateral sliding. The only way
to create a graspable edge is to slide the flat pry bar through the wall notch
(44 × 12 mm, directly under the lid edge; the rim above it is gapped) and press
the handle down. The bar pivots on the notch sill — a class-1 lever — and
levers the lid's near edge ~10–15 mm proud of the rim. Only then can the lid be
gripped, lifted out, and set aside; only then are the kings reachable.

## Why strategically different

**vs the seed:** the seed never asks *how to make an object graspable*. Here the
central skill is force amplification through a fulcrum to change the scene's
affordances — the manipulation target (lid edge) does not exist until the lever
creates it. Retrieval is then *selective* (one of three shuffled kings, colour
identifies the target, the other two are negative-constrained), not exhaustive.

**vs every task read in tasks_v7 (~40):** no other task uses a lever/fulcrum or
any force-amplification mechanism, and none makes "graspability must be
created" the core obstacle. Closest neighbours, and why this is not them:

- **i1 (silo ram-rod)**: a rod *pushes a puck through* a channel — force
  transmission, not amplification; no fulcrum, no pivot, and the target is
  already reachable by the tool. Here the bar rotates about a sill and the
  mechanical advantage (long handle, short tip) is what defeats the seal.
- **i25 (support insertion)**: a tool becomes a static *shelf/support*. The bar
  here is dynamic — its purpose is a transient torque, and it may end anywhere.
- **i11 (matchbox sleeve slide)**: opening by *translating* a sleeve along its
  free axis; the sleeve is directly pushable from reset. This lid has **no free
  axis** — flush + rim ring blocks both pinch and slide by construction.
- **i21 (keyhole unlock)**: shape-keyed insertion + rotation of a captive part.
  Here nothing rotates in a socket; the mechanism is a rigid-body lever with a
  contact fulcrum, and the freed part (lid) must then be transported away.
- **i30 (tool-carry)**: the tool extends *reach*; here it multiplies *force*.
- **i2 (silo latch, same seed family)**: gravity latch flips by direct push;
  no tool, no fulcrum, no selectivity among identical-shaped distractors.
- **i4-micro (gate release)**: releasing a pre-stressed gate; nothing must be
  pried, and no restraint clause.

The task couples three sub-skills rarely combined: (a) tool-mediated lever pry
with real contact mechanics, (b) transport of the freed lid to a clearance
constraint (≥ 24 cm), (c) colour-selective retrieval under a keep-in constraint
on the two distractors. Partial credit rewards exactly this order.

## Teleport solution (solve.py) — phases

Teleports are transport only; every load-bearing interaction is contact
dynamics or applied force.

- **P0 settle + flushness certificate**: verify by readback that the lid top is
  within 4 mm of the rim top, tilt < 1.5°, all three kings inside, score ≤ 0.03
  — i.e. the "sealed, ungraspable" premise physically holds this episode.
- **P1 pry (the load-bearing act)**: teleport the bar to the notch mouth (crate
  yaw read back per episode), blade resting on the sill — a placement a gripper
  holding the handle would make. Then apply a plain downward force at the bar's
  CoM (escalating 1.5→9 N; the CoM is handle-side of the fulcrum, so a down
  press is exactly what a hand does). The bar pivots on the sill and the lid's
  near edge rises past the rim: measured tilt 4.0° ≈ 12 mm proud.
  *Frame-drag robustness*: force is applied raw (mode 0) first; if measured
  tilt shows no progress, re-seat and retry with the rotation-compensated
  encoding M = R_ref·R_nowᵀ (mode 1) — the pod-dependent IsaacLab wrench quirk
  is probed, not assumed.
- **P2 lid hand-off + removal**: **only while the pry force is still holding
  the edge proud** (verified by tilt readback) is the lid teleported up and out
  — mirroring pinching the popped edge. It is then dropped from 5 cm above a
  ground spot chosen ≥ 0.30 m out with maximal clearance from pad and bar, and
  must settle by physics. Asserts: lid_clear, `_opened` latched, kings still
  in, success still false.
- **P3 selective retrieval**: the red king (slot shuffled per episode) is
  hopped out of the crate and dropped 6 cm above the pad; it must land and
  stand by physics. Asserts: red_on_pad (distance + height band + uprightness),
  whites still inside, success true.
- **P4 persistence**: ≥ 3.3 sim-seconds hands-off; success and score 1.0 must
  hold at every sample. Then `SIM_GEN_SOLVE: SUCCESS`.

Score trace on every seed: `0.0000 → 0.2000 → 0.5000 → 1.0000 → 1.0000`,
never decreasing. Passed on forge for seeds 0, 1, 2 and 3 (crate yaws
including +39.0° and −7.5°, distinct king permutations per seed).

## Rubric (anchored in the demonstrated solution)

- `success()` — all judged **live**: lid on the ground ≥ 0.24 m from the crate
  and low (z < 3 cm, so not leaning on anything tall), red king within 45 mm of
  the pad centre, in the standing height band, axis within 30° of vertical;
  both whites inside the crate interior volume; everything at rest and finite.
- `score()` — latched partial credit, cleared on reset, persisted through
  get/set_state: 0.20 pry (tilt > 3.0° while the lid is engaged in the pocket —
  the demonstrated pop is 4.0°), +0.30 lid off (clear radius + low), +0.25 red
  out of the crate; capped at 0.75; exactly 1.0 iff `success()`.
- **Interlock honest by construction**: the flush lid + rim ring means physics
  itself forbids the no-pry shortcut; smoke check 5 certifies that 3×-weight
  horizontal shoves neither move the lid past 3 cm nor tilt it past the pry
  gate, so the 0.20 pry latch cannot be farmed without the lever *and* the seal
  cannot be defeated without it.

## Embodiment argument (single Franka, parallel jaw, OSC, one base pose)

- One base pose ~0.45 m from the crate nominal reaches every contact: crate at
  (0.42, 0.10) ± 5 cm, pad (0.30, −0.30) ± 5 cm, bar (0.32, 0.30) ± 5 cm — all
  inside a ~0.75 m disc. Nominal crate yaw faces the notch toward the robot;
  ±50° jitter keeps it within the arm's lateral workspace.
- The bar handle is 22 × 6 mm — a comfortable parallel-jaw pinch; grasping the
  handle half leaves ≥ 100 mm of free blade for insertion.
- Notch insertion tolerance: 44 mm opening vs 22 mm blade (±11 mm lateral);
  vertically the sill *guides* the blade — sliding along the wall drops it into
  the 12 mm opening (±3 mm), a funnel-assisted insertion.
- The press is a pure downward end-effector force on the held handle — no
  regrasp, no wrist torque beyond holding a flat bar.
- The popped lid edge stands 10–15 mm proud through the 52 mm rim gap above the
  notch: pinchable by a parallel jaw from the side the robot already occupies.
- The lid (168 mm square, 0.18 kg) is carried flat; the kings (Ø36 × 20 mm,
  40 g) are top-grasped over 70 mm walls — standard clearance for a Franka
  hand descending vertically into a 150 mm square opening.

## Execution order (as mandated, and as actually done)

1. `scene.py` written with minimal `success()`; geometry lever-checked on paper
   (handle-grounding angle vs notch-top jam) before any run.
2. `solve.py` iterated **on the forge** until the goal was physically reached —
   the pop, the drop, the stand all by contact dynamics (seeds 0/1/2 pass).
3. Only then the final rubric: gates anchored in demonstrated values
   (pry gate 3.0° vs demonstrated 4.0°; clear radius 0.24 vs demonstrated
   0.30; pad radius 45 mm vs demonstrated ~5 mm error).
4. `smoke.py` rejection battery written last; both modules re-run clean.

## Check list (smoke.py, 12 checks)

1. settle/flush — layout settles finite; lid flush (< 2 mm proud, tilt < 1.5°),
   kings sealed, score 0.
2. randomization-is-real — crate yaw/xy, pad xy, bar xy differ across seeds by
   readback.
3. slot permutation — the red king occupies all three interior anchors across
   10 resets.
4. null-policy — 240 idle steps: score ~0, no success.
5. SEAL INTERLOCK — 3×-weight horizontal shoves on the lid (toward the rim gap,
   then sideways): lid moves < 3 cm, never tilts past the pry gate; `_pried`
   and `_opened` stay false. Certifies irreversibility of the seal AND honesty
   of the pry latch.
6. seed-strategy negative — red king standing on the pad but crate still
   sealed: no success, score ≤ 0.30.
7. near-miss (lid) — lid on the ground 0.19 m out (< 0.24 gate), red on pad:
   no success.
8. wrong object — a WHITE king on the pad, red still inside, lid clear:
   no success.
9. restraint — red on pad AND a white king also out of the crate: no success.
10. near-miss (pad) — red upright ~70 mm from the pad centre (> 45 mm): no
    success.
11. on-side — red lying on its side at the pad centre: uprightness refuses.
12. video — frames.npz recorded (> 10 frames).
