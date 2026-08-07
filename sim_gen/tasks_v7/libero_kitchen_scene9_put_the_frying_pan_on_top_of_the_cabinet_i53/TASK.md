# slab_easel — pivot-erect display slabs against tilted backrests

Registered env: `simgen.slab_easel` (robot `"null"`).
Task id: `libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i53`.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet`
— pick up a frying pan and place it on top of a cabinet. The seed's essence is
"relocate one graspable object into a target bounding box above a fixture."

## Strategic difference

The seed (and the classic LIBERO family) is solved by **grasp → transport →
release inside a bbox**: the final state is reached by controlling the object's
*position* while it hangs from the gripper, and the placement itself is
quasi-static. This task removes every one of those levers:

1. **No grasp exists.** Each slab is 0.24 × 0.20 × 0.03 m: both faces are far
   wider than an 80 mm parallel jaw, and the 30 mm edge is only reachable
   through 30 mm rail gaps — enough for a fingertip, not for a closed pinch on
   a load-bearing lift. The scene's `__post_init__` asserts ungraspability
   (`min(L, W) > 0.075 and T < 0.075`). The intended manipulation is
   **nonprehensile**: press a fingertip down on the head edge and hinge the
   slab about its grounded foot edge.
2. **The goal is an orientation + contact regime, not a bbox.** Success is a
   *ladder lean*: each colored slab must stand at ~80° tilt with its face
   flush (normal within 12°) against its **color-matched** 10°-back-tilted
   backrest panel, supported by foot-edge friction plus panel contact.
   Position tolerances only pin which bay it leans in.
3. **The physics is a dynamic tip-over, not a quasi-static place.** The balance
   angle over the foot edge is atan(L/T) ≈ 82.9°, *behind* the 80° rest lean;
   the slab must be driven past balance with enough momentum to fall onto the
   panel, then be caught by it. A pure pose-write cannot bank the rubric's
   rising-tilt accumulator (below), and lowering the slab quasi-statically
   into the lean from above is exactly the teleport cheat the rubric rejects.
4. **Three ordered-free subgoals + one negative constraint.** All three colored
   slabs must be erected (any order); the blank gray slab is a distractor that
   must remain flat — the seed has a single object and no distractor logic.

Versus siblings: **i11** (matchbox drawer) is a prismatic-joint containment
task; **i32** (gear train) is a meshing/rotation-transfer task. Nothing else in
the corpus scores a *pivot-erection against a leaning rest* or uses a
rising-tilt accumulator as its anti-teleport primitive.

## Scene

Kinematic "studio" deck (1.05 × 1.40 m, jittered xy ±0.06 m, free yaw) carrying:

- a raised work plane of 6 rails (30 mm gaps between rail tops — the fingertip
  access channels),
- 3 display bays at rack-local y = −0.36 / 0 / +0.36, each with a tinted bay
  plate and a 10°-back-tilted backrest panel (RED / GREEN / BLUE) whose foot
  meets the work plane at rack-local x = −0.24,
- 4 dynamic slabs (RED / GREEN / BLUE with color tiles on **both** faces, plus
  a blank gray one) spawned flat in the band x ∈ [0.04, 0.22] with a
  seed-driven `randperm` slot permutation over y ∈ {−0.48, −0.16, +0.16, +0.48},
  per-slab yaw, and rejection-resampled min separation.

Seed-dependent randomization (verified by readback in smoke): studio xy + yaw,
slab slot permutation, slab xy jitter + yaw.

## Rubric

`score()` per colored slab: 0.06 staged-latch + 0.12·clamp(erect_acc / 25°)
+ 0.10 seated-latch, capped at 0.85; 1.0 iff `success()`.

`success()` = all three colored slabs (seated **and** erect-accumulator ≥ 25°)
∧ blank slab flat on the work plane ∧ scene still (consecutive-substep pose
delta counter, ≥ 30 substeps — velocity readings lie for edge-resting bodies).

**Anti-teleport core — rising-tilt accumulator** (`post_step`, every physics
substep): `erect_acc[k] += Δtilt` only while the slab is inside its own bay
∧ tilt ∈ (30°, 70°) ∧ Δtilt ∈ (5e-4, 0.08) rad ∧ Δpos < 8 mm. A teleport to
the seated pose is one radians-scale jump (no accrual); an incremental
pose-write sweep done outside the bay is blocked by the in-bay gate; doing the
sweep *inside* the bay is exactly the physical hinge motion — i.e. the only
way to bank the credit is to actually erect the slab where it must stand.
Seated detection folds both faces (tilt from |body-z ·world-z|, normal cone
test against the panel's outward normal), so there is no hidden-color
orientation loophole.

## Teleport solution (solve.py) — phases

- **P0 settle**: 60 substep-batches, capture per-body reference rotations,
  print layout readback. `SIM_GEN_SCORE 0.000`.
- **Per colored slab k (RED, GREEN, BLUE)**:
  - `clear_obstructors(k)` — *transport only*: any loose slab lying within
    0.30 m of the staging point is teleported (free-space pose write) to a
    parking spot on the open deck; it then just lies there.
  - `stage(k)` — *transport only*: teleport the flat slab to the staging point
    in front of its bay (rack-local x = x_bp + 0.128, flat, work-plane
    height), settle.
  - `erect_attempt(k)` — **contact dynamics does all load-bearing work**: a
    world-frame torque about the foot-edge axis, governed by
    `τ = clip(12·(0.9 − ω), ±3.5)` N·m, hinges the slab up. The wrench
    encoding is **probed at runtime** (candidates: raw world vs
    R_ref·R_now^T pre-encode, ± sign) because
    `set_external_force_and_torque(is_global=True)` drags frames by
    rotation-since-reset on some pods; if tilt stalls below 6° by step 40 the
    encoding toggles. Torque cuts at 81°; momentum carries the slab over the
    82.9° balance; it falls ~17° onto the panel and is caught (angular
    damping 0.25, restitution 0). Retry with +1.5° cut if it fell back.
  - Verify `seated_now ∧ erect_ok`, print non-decreasing `SIM_GEN_SCORE`.
- **P4 hold**: arm stillness, assert `success()`, then ≥ 3.4 s completely
  hands-off (no writes, no wrenches) with success asserted throughout →
  `SIM_GEN_SOLVE: SUCCESS`.

Teleports move *unloaded* bodies through free space to poses where they rest
stably; every credit-bearing state change (the 25°+ of accumulated in-bay
rising tilt, the seat against the panel) is produced by applied torque +
contact + gravity. Verified end-to-end on the forge for **seeds 0 and 7**
(different studio yaws, slot permutations; both runs: all three slabs seated
on attempt 0, mode-0 encoding, score 0.000 → 0.280 → 0.560 → 1.000,
`SIM_GEN_SOLVE: SUCCESS`).

## Embodiment argument (Franka)

- **Base pose**: on the deck's front strip, rack-local ≈ (+0.40, 0.0), facing
  the bays. Spawn band (x ∈ [0.04, 0.22]) and all three staging points
  (x = −0.112, y ∈ {−0.36, 0, +0.36}) lie within a 0.30–0.75 m annulus —
  inside Franka's comfortable reach with margin for the ±0.06 m studio jitter.
- **Per-object contact strategy**: (a) *reposition flat slabs* — fingertip
  drag/push on the top face (μ ≈ 0.42–0.45) or edge-push through a rail gap;
  (b) *erect* — press one fingertip (or the closed fingertip pair) down-and-
  forward on the slab's head edge; the foot edge is chocked against the rail
  gap / bay plate lip, so the slab hinges. Peak required force ≈ m·g·(L/2)/L
  ≈ 11 N vertical at the head edge — well inside Franka's payload; the 30 mm
  rail gaps guarantee fingertip clearance under the head edge at low tilt;
  (c) *finish* — ride the head edge until ~81° and release; dynamics completes
  the fall onto the panel, exactly as the scripted torque governor does.
- The blank slab requires no interaction (negative constraint).

## Execution-order declaration

No ordering constraints beyond mechanism physics: the three colored slabs may
be erected in any order; parking of obstructing slabs is only required when a
loose slab happens to block a staging point (seed-dependent). The scripted
solution uses RED → GREEN → BLUE for determinism only.

## Checks (smoke.py — 14)

1. Settle: finite state, all slabs flat, at work-plane height, still.
2. Baseline score ≤ 0.02.
3. Studio xy/yaw randomization readback varies across seeds 21–28.
4. Slab randomization: xy/yaw spreads; RED occupies ≥ 2 distinct slots.
5. Null policy 240 steps → no success, score ≈ 0.
6. Seed-strategy end state: RED laid flat at its staging point (the "placed in
   the target region" analog) → not seated, no success, score ≤ 0.10.
7. Teleport-to-seated inject in own bay → `seated_now` True but accumulator
   < 3° → rejected, score ≤ 0.05.
8. Wrong bay: RED seated in GREEN's bay → rejected.
9. Lateral near-miss: seated pose offset +0.18 m in y → rejected.
10. Wrong object: BLANK slab seated in RED bay → `distractor_flat` False → no
    success.
11. Latched staging credit survives object removal (latch semantics).
12. Out-of-bay incremental pose-write tilt sweep (8 mrad/write to 80°) →
    accumulator gain < 2° (in-bay gate holds).
13. `ever_success` audit: success never latched during the battery.
14. Final state has no NaNs.

Records `frames.npz` (camera frames at each check boundary) in CWD.
