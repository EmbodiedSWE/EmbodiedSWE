# close_microwave_i306 — Roll the millstone to close the vault

**Scene:** `simgen.stone_door_vault` (`StoneDoorVaultScene`, registered as `stone_door_vault`)
**Seed task:** `rlbench/close_microwave`
**Files:** `scene.py`, `solve.py`, `smoke.py`, `TASK.md` (this file). Procedural geometry only — no external assets.

## Seed provenance and what changed

The RLBench seed is *close_microwave*: a microwave with a hinged door stands on a table and the
robot pushes the door shut — a single unconstrained push on a 1-DoF revolute panel, success = door
angle near zero. This task keeps the seed's **abstract goal** — *close an open container by moving
its closure element into the sealed configuration* — and replaces everything about *how* closure is
achieved:

| Aspect | Seed (`close_microwave`) | This task |
| --- | --- | --- |
| Closure element | hinged door already attached (1 revolute DoF) | FREE rolling disc (a tomb-style millstone) standing in a guide channel along the wall front |
| Motion class | short push toward the frame, normal to the door plane | long-range controlled ROLL parallel to the wall (~15–25 cm), then gravity capture |
| Kinematic constraint | the hinge does all the guiding | no joint at all: two rails + a raised bed guide the roll; a recessed pocket in front of the doorway captures the stone as a pure contact gravity detent |
| Direction of the load-bearing motion | toward the wall | ALONG the wall — pushing toward the wall (the seed's motion) presses the inner rail and achieves nothing (smoke check 5) |
| Distractors | none | RED square block on the opposite side of the track; seating it earns nothing and physically blocks the seat |
| Success test | door angle threshold | stone SEATED in the pocket: vault-frame position, seated height band, in-channel, upright, settled — judged live on physical poses |
| Assets | RLBench microwave USD + trajectory | fully procedural compound rigid bodies |

## Strategic difference (vs seed and vs the corpus)

- **Vs the seed:** the seed's entire strategy is "push the panel toward the frame about its
  hinge". Here that motion is *provably useless*: the wall and inner rail block it (smoke
  check 5 pushes the stone toward the wall at ~0.6 mg for 1.5 s — it travels ~4 mm to rail
  contact and stops, earning nothing). Closing requires the orthogonal skill: discriminate the
  closure element by color/shape, then drive it a long distance PARALLEL to the wall with
  enough control that the pocket can capture it. The seed's strategy is inexpressible here and
  vice versa.
- **Vs sibling `close_microwave_i4` (same seed):** i4 is *mechanism-release* — remove contents,
  pull a prop post, gravity swings an articulated drop-gate shut. i306 has no articulation, no
  prop, no removal, no ordering: the door is a free body and the skill is transport-with-capture
  (a controlled roll into a detent). No shared mechanism or motion class.
- **Vs sibling `close_microwave_i5` (same seed):** i5 is *additive rotational locking* — carry a
  lid to a flange, key-align, insert, twist to a hard stop (wrist-roll skill, alignment window).
  i306 never rotates anything into a lock and has no key-fit: the load-bearing motion is a
  meter-scale guided ground roll and the capture is a passive gravity pocket. Different motion
  class (locomotion-like transport vs in-place wrist work), different capture physics
  (detent drop-in vs lug-under-flange).
- **Vs the rest of the corpus read this session:** bowl_decant (pouring), peg_insertion_side_i1
  (ramrod eject), pick_single_egad_i3 (gate-pin tunnel), setup_checkers_i2 (silo ordering) —
  none roll a disc along a guide track into a recessed detent. The rolling-transport-with-
  gravity-capture interaction is new to the corpus.

## The scene

A kinematic vault wall (240 mm tall, 30 mm thick) with an open 80 × 150 mm doorway at its
center. Along the wall's FRONT face runs a straight guide channel: two rails (45 mm tall)
flanking a raised bed strip (10 mm tall, 48 mm wide) that spans the full ±0.42 m track EXCEPT
for a 90 mm gap — the POCKET — directly in front of the doorway; the pocket floor is the ground,
10 mm below the bed top. End caps close the channel. Two free bodies stand in the channel, one
per side of the pocket: the BLUE STONE (upright disc, r 60 mm × 40 mm, 0.5 kg — the vault door)
and a RED DECOY block (70 mm square face, 0.2 kg) that fits the pocket but is not the door.

**Randomization (seed-driven, verified by readback in smoke checks 2/3):** vault yaw ±30° about
its nominal heading + xy jitter (±5 cm), WHICH SIDE of the pocket the stone starts on (the decoy
takes the other), and both objects' distances along the track (stone 0.14–0.26 m). The agent
must read the track direction and the stone's side from the scene every episode.

**Rubric (latched partial credit, updated in `post_step`, all measured on the BLUE stone):**
0.15 moved ≥70 mm along the track toward the pocket (the threshold sits above the ~45 mm
bounded free-stone wander band measured over 10 idle seconds on 5 seeds, so no credit accrues
force-free) · +0.25 ever within 75 mm of the pocket
center in-channel · +0.20 ever dropped into the pocket (|y| < 30 mm, center below bed-top
height) → capped at 0.60 unless `success()`; `success()` = `stone_seated()` (vault-frame
|y| < 20 mm ∧ in-channel ∧ center height 45–65 mm ∧ rolling plane upright) ∧ settled ∧ finite,
judged live. The tolerances are honest by construction: a stone physically resting on the pocket
floor is bounded by the bed edges to within ~12 mm of the pocket center and rests at ~60 mm
center height; a stone perched on the bed sits at ~70 mm and one lying flat cannot enter the
48 mm channel (smoke checks 9/10/11 probe each clause).

## Solution outline (`solve.py`)

Teleports are never used; the stone is driven from spawn to seat entirely by external wrenches
(fingertip-scale forces) through contact dynamics.

- **P0** settle 1.5 s, layout readback (vault yaw, stone side, start distance), assert score ≈ 0.
- **P1 guided roll:** a velocity servo (v_des 0.08 m/s, force-capped at 4 N ≈ 0.8 mg) drives the
  stone along the track toward the pocket, with a small yaw-squaring torque holding the rolling
  plane parallel to the rails. **Wrench-frame note (root cause of the only solve failure):**
  `is_global=True` wrenches on a ROLLING body are applied through a stale rotation reference, so
  a "world" force direction rotates with the disc's accumulated roll angle — the stone becomes a
  driven pendulum in roll-angle space and parks ~(π/2)·r past its start. All wrenches are
  therefore re-expressed in the stone's body frame with the CURRENT quat every step. At 90 mm
  from the pocket center the drive is cut and the stone COASTS in force-free (entry KE ~2.4 mJ,
  far below the ~49 mJ needed to climb out the far side): the pocket captures it as a pure
  gravity detent — feet drop 10 mm, the bed edges center it. A stall watchdog does a
  square-and-backoff, then mildly escalates the servo.
- **P2 persistence:** all wrenches cleared, ≥3.3 simulated seconds hands-off; success must hold
  every sample.

Prints `SIM_GEN_SCORE` at each phase boundary (non-decreasing: 0 → 0.40 → 1.0 → 1.0) and
`SIM_GEN_SOLVE: SUCCESS` only if success persists. Verified on the forge on seeds 0, 1, 2, 3.

## Embodiment argument (Franka, table-top base)

- **Base pose:** robot base at the origin; the vault stands at ~(0.42, 0) m facing the robot,
  track endpoints at ~(0.42, ±0.42) → the whole channel and both objects lie within a Franka's
  ~0.85 m reach envelope, all interaction below 0.15 m height.
- **Blue stone (the only object touched):** driven by pushing on its rim/face at fingertip
  force scale (the solve's servo peaks at 4 N ≈ 0.8 mg; a fingertip side-push suffices — no
  grasp needed, mirroring how one rolls a heavy disc). The roll is a slow (~0.08 m/s) guarded
  push parallel to the wall; the rails passively reject lateral error, so the required control
  is 1-D effort along the track — well within arm impedance control. The final seating needs no
  precision at all: the pocket captures any quasi-static arrival.
- **Vault:** kinematic; never manipulated.
- **Red decoy:** never touched (touching it earns nothing; the rubric reads only the stone).

## Ordering declaration

No inter-object ordering: exactly one object is manipulated. The roll-then-capture sequence is
intrinsic to the geometry (the stone cannot be seated except by arriving through the channel —
the rails and wall exclude every other approach), not an imposed rubric ordering.

## Checks (`smoke.py` — 12 checks)

1. Settle/no-NaN baseline: stone upright in-channel 0.10–0.28 m out, decoy opposite side, score 0.
2. Randomization readback: 3-seed max-pairwise deltas on vault yaw / vault xy / stone distance.
3. Side swap occurs in both directions over 10 resets.
4. Null policy 2 s → score ≤ 0.05; the decoy block does not wander (it cannot roll).
5. **Seed-strategy rejection:** 3 N push TOWARD the wall for 1.5 s → presses the inner rail
   (min-x readback proves contact), no track progress, no credit.
6. **Acceptance + retention:** constructed seat → success, score 1.0; then a 0.6 N steady track
   drag. For a *constant* force the extraction bound is energetic, not static: the force works
   over the ~12 mm slack + ~33 mm climb, so escape needs F·45 mm > mgh ≈ 49 mJ → F ≳ 1.1 N (a
   2 N pull DOES extract — measured on the forge). 0.6 N (~1.8× margin) rolls the stone to the
   bed edge (peak-|y| readback proves the drag bit) but it stays captive and successful.
7. Near-miss short: stone on the bed 100 mm out on its own side → on the bed, no near/in
   credit, no success, decoy undisturbed.
8. Wrong object: red decoy seated in the pocket → pocket readback confirms, score ≤ 0.05.
9. Wrong orientation: stone lying FLAT bridges the rails over the pocket (a flat disc cannot
   enter the 48 mm channel) → upright clause refuses.
10. Perch: stone on the bed edge at 55 mm on its own side (past the tipping point at ~43.5 mm)
    → stays perched above the seated band, no `in` credit.
11. Right pose, wrong place: stone upright on the GROUND at the pocket's track coordinate —
    y/z/upright clauses all pass, the in-channel clause alone rejects.
12. Video frames saved (`frames.npz`).
