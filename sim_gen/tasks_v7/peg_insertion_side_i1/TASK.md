# Task `peg_insertion_side_i1` — Ramrod Ball Eject

## Seed provenance

Derived from **`maniskill/peg_insertion_side`** (RoboVerse pack:
`roboverse_pack/tasks/maniskill/peg_insertion_side.py`). The seed task: pick a peg
lying on the ground and insert it sideways into the hole of a fixed block. Its checker
is pure peg-pose containment — `DetectedChecker` on the *stick* against a relative bbox
in the *box* frame (`relative_pos=[-0.05,0,0]`, half-extent x = 0.05): the task is DONE
the instant the peg's origin sits deep enough in the hole. Insertion is the goal;
nothing downstream of the peg matters.

## What changed, and why it is strategically different

Here **insertion is demoted from goal to instrument**, and the judged object is a body
the gripper can never touch:

- A gray horizontal **tube** (180 mm, square 32 mm bore, enclosed on all sides, open
  only at both ends) is mounted 60 mm above the table on a pedestal. A **yellow ball**
  (28 mm) is trapped 60–105 mm deep inside the bore — visible only through the open
  ends, geometrically unreachable by any parallel-jaw gripper (fingers cannot enter a
  32 mm enclosed bore 60+ mm deep).
- The far end of the tube (the **muzzle**) overhangs a walled blue **catch basin** on
  the table. The near end (the **entry**) faces open space.
- A **green rod** (Ø22 × 280 mm — longer than the tube) rests on a two-post rack. A
  same-size **red decoy ball** lies loose on the table.
- **success()** = the YELLOW ball settled inside the basin interior (position inside
  the walls, height on the basin floor, velocity below threshold). The rod's final pose
  is *irrelevant* — the seed's success predicate (peg pose in block frame) appears
  nowhere in the rubric.

So the strategic plan is inverted relative to the seed:

| | seed | this task |
|---|---|---|
| judged object | the peg itself | a third body (yellow ball) the peg must act *through* |
| insertion | terminal goal | instrumental sub-step (~5 mm insertion is worthless) |
| stroke | ~5 cm to containment | ~170 mm full-length ram THROUGH the tube |
| after insertion | done | ball must exit the far end, free-fall, land and settle in a container |
| distractor | none | red decoy ball (identity check) |

The seed's entire strategy — "get the stick's pose into the box's bbox" — scores at
most the latched insert credit here (cap 0.35 + 0.10 approach) and is **not** success;
this exact end-state is constructed and rejected in smoke check #6.

Code structure is also disjoint from the seed: custom compound USD spawners (kinematic
tube/pedestal/basin assembly; two-post notched rack), a per-substep latched progress
rubric in `post_step`, and per-episode baselines (`s0` ball depth, `d0` rod-to-entry
distance) captured at reset.

## Randomization (all verified by readback in smoke #3/#4)

- Assembly (tube+pedestal+basin, one rigid group): xy ± 3 cm, yaw ± 25° — the bore
  axis direction must be read from the scene, not hard-coded.
- Ball spawn depth `s0` ∈ (0.060, 0.105) m along the bore.
- Rack + rod: xy ± 5 cm, yaw ± 180°, rod axial jitter ± 2 cm in its notches.
- Decoy ball: xy ± 3 cm.

## Rubric

Latched every physics substep in `post_step` (credit never evaporates):

`score() = 0.10·approach + 0.25·insert + 0.45·push` (cap 0.80),
raised to ≥ 0.9 while the ball is inside the basin, exactly 1.0 iff `success()`.

- **approach**: best rod-tip proximity to the entry, normalized by the episode's own
  spawn distance `d0`.
- **insert**: best in-bore rod-tip depth / `eject_tip_x` (0.166 m), gated on the tip
  actually being inside the bore cross-section.
- **push**: best ball displacement `(ball_x − s0)/(tube_len − s0)`, gated in-bore;
  full when the ball is in the basin.
- **success**: `ball_in_basin() & settled()` — yellow ball only; the red decoy in the
  basin scores nothing (smoke #9).

Null policy scores ~0 (smoke #2/#5).

## Teleport solution (`solve.py`)

- **P0** — reset(seed), 60-step settle, layout readback printed (assembly xy/yaw, s0,
  rod spawn, d0).
- **P1 TRANSPORT (teleport, free space only)** — one pose write carries the rod from
  the rack to a hover pose on the bore axis, tip **15 mm OUTSIDE** the entry face, axis
  aligned. Exactly the pose an arm reaches after pick-and-align; no scoring gate is
  satisfied by it, the ball is untouched, the rod is entirely outside the tube.
- **P2 ENTRY + RAM (contact dynamics)** — a floating-hand force controller
  (gravity-compensating vertical PD → light 0.6·m·g ride once deep, assembly-frame
  lateral centring, axis-alignment torque, velocity-regulated 2 N axial push with
  stall-escalation) drives the rod INTO the bore. The rod tip meets the ball and pushes
  it ~80–120 mm along the bore floor under real contact until it loses support at the
  muzzle, free-falls, and lands in the basin. Forces cut, 2 s settle. Every millimetre
  of ball motion is contact transmission; the ball is never teleported, the rod is
  never teleported into the tube.
- **P3 PERSISTENCE** — ≥ 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS`
  only if `success()` still holds.

`SIM_GEN_SCORE` printed at each phase boundary; non-decrease asserted (P0 0.000 →
P1 ≈ 0.045 → P2 1.000 → P3 1.000 on seeds 0 and 1). Watchdog + `os._exit` guard the
known Kit-teardown hang.

## Execution-order declaration

No fixed sub-goal ordering is imposed by the rubric; ordering is enforced *physically*:
the ball cannot move (push credit) until the rod is inside the bore pressing on it, and
the rod cannot act on the ball without first approaching and entering. The rod may be
left inserted, withdrawn, or dropped after ejection — only the yellow ball's settled
containment is judged.

## Embodiment argument (single Franka arm, parallel jaw, OSC)

One plausible base pose serves the whole episode: **base at (0.0, −0.55, 0)** facing
+y toward the workspace (assembly origin drifts ± 3 cm around (0,0)). All working
points then lie 0.38–0.68 m from the base — inside Franka's comfortable reach annulus:

1. **Pick the rod**: it rests across a two-post rack at centre height 56 mm — an
   elevated, ground-clear side grasp (Ø22 mm ≪ 80 mm jaw span) anywhere along its free
   middle span. No table-scrape approach needed.
2. **Align + insert**: hold the rod near its rear third, bring the tip to the entry
   face at bore-centre height 76 mm, align to the bore axis (read from the tube's
   yaw). The bore leaves 5 mm radial clearance around the rod — generous for OSC with
   compliant lateral gains, and the enclosed bore itself funnels the tip.
3. **Ram**: a single straight-line ~170 mm push along the bore axis. The rod is 280 mm
   long, so at full ejection stroke the gripper is still ≥ 90 mm OUTSIDE the entry
   face — the hand never needs to approach the tube mouth, and wrist orientation stays
   constant. Push force required ≈ 2 N (measured in solve) — trivially within payload.
4. **Release anywhere**: success does not depend on the rod afterwards.

The ball's ~30–60 mm free-fall from the muzzle into a 160 mm-long basin directly below
needs no manipulation at all. No bimanual need, no regrasp requirement (one grasp
suffices for align+ram since the stroke is shorter than rod-length minus grip
allowance), no forces beyond a ~2–6 N axial push.

## Constructibility notes (smoke design)

- The **seed-strategy end state** (rod inserted, ball unmoved) is constructible only at
  shallow tip depth (tip_x = 0.045 < min s0 = 0.060 minus ball radius) — deeper and the
  bodies would overlap. Smoke #6 constructs it, settles it, and asserts NOT success
  with score ≤ 0.45.
- "Rod fully inserted with ball unmoved" is NOT constructible (they would
  interpenetrate) — physics itself forbids the seed's full-insertion goal without the
  ball moving; noted here instead of tested.

## Checks

`smoke.py` runs a 13-check rejection battery: (1) settle/finite + spawn readback
(ball in bore at depth s0, rod at rack height); (2) initial score ≤ 0.02, no success;
(3) rod xy+yaw randomization spread by readback over 6 seeds; (4) assembly xy+yaw and
ball-depth spread; (5) null policy 240 steps ≤ 0.02; (6) SEED-STRATEGY end state
(rod inserted, ball unmoved) settled → rejected, score ≤ 0.45; (7) near-miss: ball
12 mm short of the muzzle inside the bore, settled → NOT success, score < 0.9;
(8) wrong-place: yellow ball settled on the table outside the basin → score ≤ 0.20;
(9) identity: RED decoy settled in the basin centre (`_in_basin` geometrically true
for it) → score ≤ 0.05, NOT success; (10) partial-push monotonicity: shorter push
latches strictly less than the near-miss; (11) latched credit: ball regressed backwards
→ score does not drop; (12) `ever_success` audit False across all constructed
non-successes; (13) final no-NaN/no-drift. Records `frames.npz` (RGB) in CWD.
Prints `SIM_GEN_SMOKE: ALL PASS 13/13`.

## Run

```
python -m simgen_tasks.peg_insertion_side_i1.solve --headless [--seed N]
python -m simgen_tasks.peg_insertion_side_i1.smoke --headless
```
