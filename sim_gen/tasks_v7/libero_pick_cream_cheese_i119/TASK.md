# silo_tip_pour — stage the basket under the silo, then pour the ungraspable box into it

`libero_pick_cream_cheese_i119`
Env: `simgen.silo_tip_pour` · robot slot: `null` (scene-level task) · procedural assets only.

## Seed provenance

Seed: `roboverse_pack/tasks/libero/libero_pick_cream_cheese.py` — grasp the cream-cheese box
among distractors, carry it through free air, release it over an open basket; judged by a
containment bounding box. One unordered pick-and-place; the only physics is release-and-rest.

## What changed, and why it is strategically different

The payload is **ungraspable** and the receptacle is the thing you carry. A heavy stand holds
two identical elevated **silo hutches** (roofed, walled, interior 108×90×78 mm) at 180 mm, each
caging one 60×60×55 mm box behind a spout that leaves only 15 mm per side and no headroom — no
grasp exists. Each silo pivots on a **revolute hinge at its spout lip**, rests tilted **back
12°** (box against the back wall), and carries a red **lever paddle**: pressing the paddle down
tips the silo forward (hard stop +43°); past ~12° forward the box slides out and falls; a
released silo **always self-returns** (CoM balance angle ~55° is beyond the stop). One silo
holds the CREAM target, the other a BROWN decoy — sides shuffled per episode, color through the
spout is the only cue. Success = cream box inside the upright grounded basket ∧ brown box not
inside ∧ no fail latch ∧ everything still (60-step counter).

Strategic distance, verified in physics on the forge:

- **The seed's entire plan is impossible.** There is no grasp: teleporting the box from the
  resting silo straight into the basket is an exit from an untipped silo → permanent **breach**
  fail, score ~0 (smoke #7); dragging it out of the open spout at 1.5× weight is the same
  breach (smoke #8). The seed's one skill — carry the payload through free air — cannot occur.
- **Inverted transport, new to the corpus:** the solver carries the *receptacle* to the
  payload's future landing point (basket → catch pad under the correct spout), i.e. it must
  reason about a **ballistic hand-off it sets up in advance**. Pour with the basket unstaged
  and the box grounds → permanent fail that even a constructed in-basket state cannot undo
  (smoke #6) — so **stage-first ordering is rubric-hard**.
- **Mechanism actuation replaces prehension:** the only legal extraction is pressing the lever
  paddle (a ~2.3 N fingertip press at rest, decaying as it tips) and releasing — the pour is a
  transient, not a holdable state; the silo swings back shut on its own (smoke #5).
- **Perception:** cream/brown silo identity is a per-episode shuffle (smoke #3); a flawless
  pour on the decoy side scores ~0 (smoke #10).
- Distinct from the corpus: i80 (same seed family) is static torque equilibrium on a beam
  balance; i33 is push-through a one-way flap into a sealed box. No tasks_v7 task stages a
  catcher under a self-returning tip hopper or judges *receiving a poured payload*.

## Teleport solution (solve.py — passes seeds 0 and 1)

Teleports are transport-only; the pour, flight, catch and all settling are contact dynamics on
the free hinge:

- **P0** reset + settle: silos at rest −12.5°, boxes caged, basket away; score 0.
- **P1** one pose write puts the basket 30 mm above the cream-side pad centre; it falls and
  seats by gravity → staged latch, score 0.15.
- **P2** hinge torque on the cream silo (gravity-feedforward PD: the keel's restoring torque
  decays as sin(55°−p), so a constant "hold" is a runaway push — the PD tracks 22° at
  ≤0.8 N·m). Box slides out at **+21.3°** (step 92), tilt held ~0.2 s to clear the lip, torque
  released at +24.7°. Hands off: ballistic drop into the staged basket, silo self-returns to
  −12.5° → success, score 1.0. The torque is applied along the hinge axis itself (invariant to
  the silo's rotation → immune to wrench frame-drag), with a runtime sign probe.
- **P3** persistence: 10 × 40 hands-off steps (3.33 s) with success asserted each block →
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` prints at every phase boundary: 0.0000 → 0.1500 → 1.0000 → 1.0000
(non-decreasing; credit is latched in `post_step`).

## Embodiment argument (Franka, base at (−0.40, 0, 0) facing +x)

Stand nominal (0.42, 0) yaw 180° — spouts and pads face the base. Everything the arm touches
lies at radius 0.37–0.65 m from the base, heights 0–0.27 m: comfortably inside the envelope.

- **Basket carry (the only transport):** 0.30 kg, 180 mm square, 8 mm walls — a rim pinch (jaw
  80 mm ≫ 8 mm) from its open-ground start (~0.40 m from base), set down on the pad at ~0.60 m.
  Target tolerance is generous: staged = within 60 mm of the pad centre, upright, on ground.
- **Paddle press:** the red paddle (35×46 mm) sits outboard at world ~(0.15, ∓0.25, 0.25),
  radius ~0.60 m. Fingertip press-down ≈ 2.3 N at rest (0.33 N·m holding torque over the
  0.14 m paddle arm), decaying with tilt; the fingertip rides a ~9 cm arc through the ~37°
  sweep. The paddle is outboard (inner edge |y| ≈ 0.222 stand-local) while the falling box
  stays within |y| ≤ 0.21 — the finger is never in the payload's path. Release anywhere; the
  silo returns by itself.
- **The box is never grasped, never touched** — that is the point of the task.
- The stand is 30 kg: incidental contact cannot move the goal frame, and every predicate is
  stand/basket-frame relative regardless.

## Execution-order declaration

`stage the basket on the cream-side pad → press that silo's paddle until the box slides out →
release, hands off`. Stage-first is REQUIRED and rubric-enforced by physics-side latches: a
pour with no basket beneath grounds the box (permanent fail, smoke #6), and the box cannot
leave the silo any other way (breach fail, smoke #7/#8). No other order constraints; either
silo's paddle may be pressed at any time (only the decoy's *outcome* is unrewarded, smoke #10).

## Rubric

- 0.15 — basket staged on the cream-side pad (latched in `post_step`, velocity-gated)
- 0.25 — cream box exits its silo while tipped ≥ 2° (a real pour; latched)
- 0.20 — poured box inside the upright grounded basket (latched)
- non-success cap 0.60; **1.0 iff** success(): cream in basket ∧ brown not in basket ∧ basket
  upright on ground ∧ no breach/grounded latch ∧ 60-step stillness.
- Permanent fails: **breach** (exit while pitch < 2°) and **grounded** (poured box at ground
  level, > 90 mm from the basket). Null policy: 0.000 (smoke #4).

Anti-flake measures baked in (memory-verified forge traps): explicit MassAPI CoM + diagonal
inertia for the silo (mass-only leaves CoM at the hinge — no restoring torque), heavy dynamic
stand as joint body0 (kinematic anchors stay world-fixed after reset teleports), torque along
the hinge axis (frame-drag invariant) + sign probe, gravity-feedforward PD instead of constant
hold torques (constant push is runaway past ~13° tilt), consecutive-still counter instead of
instantaneous velocity gates, exit-event (`prev_in & ~now_in`) breach detection instead of
position thresholds, containment gate sized for a wall-lean (box centre ≤ 0.060 vs wall inner
face 0.082 — a straddle or rim perch is ≥ 0.086 / z ≥ 0.11), float32 slack on score asserts.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 14/14` on the forge)

1. settle/no-NaN: finite, silos at rest tilt, boxes caged, basket upright away; score ~0
2. randomization readback: stand yaw Δ9.4°, stand xy Δ29 mm, basket Δ185 mm / Δ15.2° across seeds
3. side swap: cream box occupies BOTH silos over 10 resets
4. null policy: 240 idle steps → boxes caged, score ~0, no success
5. lever mechanism: hinge torque tips the silo to +24.5° (exit observed — the probe moved it);
   released, it swings back to −12.5° by its own CoM bias alone
6. out-of-order: pour with the basket unstaged → grounded permanent fail; even CONSTRUCTED
   into the basket afterwards, success stays False (score = latched 0.25)
7. **seed strategy**: box teleported from the resting silo into the basket → breach, score ~0
8. drag-out: 1.5×-weight pull extracts the box from the resting spout (it moved) → breach
9. staging near-miss: basket 100 mm off the pad centre → nothing; centred → 0.15
10. wrong silo: full tidy episode on the DECOY (staged + clean pour into the basket) → score ~0
11. legal pour → success 1.0; success never fired while the box was still moving fast
12. decoy contamination (live): brown dropped into the succeeded basket → success OFF; removed
    → success returns
13. no post-hoc: cream lifted out → grounded, capped at 0.60; re-constructed into the basket
    → STAYS failed
14. video: frames.npz (183 × 600 × 960) captured and saved

Verification: forge pod (Isaac Sim 5.1, RTX 4090). solve seeds 0 & 1 → `SIM_GEN_SOLVE: SUCCESS`
(rc=0, 20.4 s / 25.2 s); smoke → `SIM_GEN_SMOKE: ALL PASS 14/14` (rc=0, 57.9 s).
