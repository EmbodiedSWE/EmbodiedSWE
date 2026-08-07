# roll_ball_i81 — Arch-Lock Quarry

**Env name:** `simgen.arch_quarry` (robot `"null"`)
**Seed task:** `maniskill/roll_ball`
**Files:** `scene.py`, `solve.py`, `smoke.py`, `TASK.md`

## 1. Seed provenance and what changed

The ManiSkill seed `roll_ball` is: one dynamic ball on a flat open table, push/roll it
laterally across the tabletop into a flat circular goal region. One object, one contact,
open workspace, translation-only strategy, success = ball xy inside a painted disc.

`arch_quarry` keeps exactly one seed atom — *a ball must be rolled by pushing, not
carried* — and replaces the strategic content around it:

| Aspect            | Seed `roll_ball`                    | `arch_quarry` (this task)                              |
|-------------------|-------------------------------------|--------------------------------------------------------|
| Workspace         | flat open table                     | tilted quarry channel (20° ramp + back slope + walls)   |
| Objects           | 1 ball                              | 3 balls: 2 heavy 95 mm boulders + 1 light 50 mm red    |
| Blocking state    | none                                | boulders **wedge into a self-locking arch** across the channel, pinning the red ball in a V-groove underneath |
| Required strategy | lateral push to goal                | **demolition first**: expel one boulder uphill over the crest to break the arch, then lift the red ball vertically out through the open top, then place it in a raised nest ring |
| Direction of push | horizontal, toward goal             | horizontal push is *away* from the goal (up the ramp); the goal-ward direction is blocked by the wall and useless |
| Goal region       | flat painted disc                   | raised hex nest ring that admits 50 mm but rejects 95 mm |
| Success           | ball in disc                        | red in nest ring **and** both boulders ≥ 0.10 m away from the nest (exclusion clause) |

The direct seed strategy — grab/push the red ball horizontally toward the goal — is
physically impossible in the initial state: the red ball is pinned under ~23 N of
boulder weight in a V-groove between two walls (smoke check 4 proves a 6 N goal-ward
drag moves it < 3 cm). The only route to the goal runs through breaking the arch, and
breaking the arch requires pushing a *different* ball in the *opposite* direction from
the goal. This inversion (push the non-target object away from the goal first) is what
makes the task strategically distinct, not just re-parameterised.

## 2. Why it is different from the rest of the corpus

Nearest corpus neighbours checked this session and the discriminating feature:

- **hockey_i325 (gate goal)**: roll one puck through a gate — single object, no
  blocking mechanism, no demolition. Here the rolled object is a *blocker*, not the
  target, and the target leaves *vertically*.
- **bowling_i17 / tile shunt i23 / spring lane i329**: launch/slide an object toward a
  target. Here the load-bearing push is *anti-goal-ward* and the scored object is
  never pushed toward the goal at all (it is lifted, then placed).
- **trap crate i26 / flap pantry i33**: object imprisoned behind a movable barrier —
  but the barriers are hinged/sliding fixtures. Here the "barrier" is an *emergent
  two-body friction arch* of free rigid bodies; there is no joint anywhere in the
  scene, and the lock is purely a force-closure of sphere contacts.
- **lever hoist i16 / dipper i76 / hook rake i55**: tool-mediated transport. No tool
  here; the mechanism is destroyed, not used.
- **rings i64 / dock i58**: placement-precision tasks. Placement here is trivial
  (open ring from above); all the difficulty is in the unlock ordering.

No corpus task has (a) a self-locking arch of free bodies as the obstacle, (b) an
expel-the-wrong-ball-uphill-first ordering, or (c) an exclusion clause that forbids the
demolition debris from ending near the goal.

## 3. Scene summary

Kinematic **quarry** fixture at ~0.42 m from origin, yaw-randomised ±20°: a 20°
uphill ramp plate (crest at height 7.5 cm), a 20° back slope, two side walls 22.5 cm
apart, a tall back wall. The channel floor is slick (μ ≈ 0.12–0.15 < tan 20°, bound
material, combine mode "min") so the ramp is *restoring*: anything on it rolls back
into the channel unless pushed all the way over the crest. Two 95 mm, 1.1 kg
**boulders** drop in at reset and wedge shoulder-to-shoulder against the walls at
|v| ≈ 0.065 m — a friction/geometry arch. The 50 mm, 0.15 kg **red** ball sits in the
V-groove beneath them, pinned by their weight. A separate raised hex **nest ring**
(inner apothem 33 mm: admits 50 mm, rejects 95 mm) is placed at a randomised polar
offset (0.30–0.38 m, bearing 95–150° off the channel axis, side coin-flipped) — the
nest sector never intersects the +u expulsion lane, so expelled boulders cannot
accidentally violate the exclusion clause.

Randomised per seed: quarry xy (±3 cm) and yaw (±20°), nest distance/bearing/side/yaw,
boulder drop jitter (±8 mm). All verified by readback in smoke check 7.

**Rubric (monotone, latched):** 0.30 when a boulder is expelled from the channel,
+0.20 when the red ball is freed (rises above z_loc 0.12 or leaves the cell), +0.15·
approach shaping toward the nest (gated on freed), 1.0 only on full success. Success =
red ball inside the nest ring (xy < 2 cm, correct height band), still, AND both
boulders ≥ 0.10 m from the nest. Latched credit never decreases (smoke check 11).

## 4. Teleport-solution phase outline (solve.py)

Teleports are used **only** for transport of a free ball through open air; every
load-bearing interaction (arch demolition, extraction from the groove) is applied
contact force.

- **P0 settle** (1 s): drop boulders, verify the arch formed by readback (both
  boulders wedged at z_loc ∈ (0.040, 0.085), |v| ∈ (0.045, 0.085), red pinned below
  z_loc 0.05). `SIM_GEN_SCORE 0.0000`.
- **P1 expel** (forces only): horizontal CoM force on boulder A along the channel +u
  axis, velocity-regulated (0.20 m/s), 3→8 N escalation on stall, ±1 N lateral
  centering. The external-force frame mode (world vs reset-frame-dragged; pod
  dependent) is **probed at runtime**: 40-step progress windows, flip mode on
  regression or two consecutive stalls. Boulder crests the ramp and rolls out; arch
  broken. Hands-off 1.5 s. `SIM_GEN_SCORE 0.3000`.
- **P2 free** (forces only): vertical contact lift on the red ball (gravity + 1.2 →
  + 3.5 N, same probe logic, lateral +u nudge fallback if trapped under the survivor)
  until it clears z_loc 0.15. `SIM_GEN_SCORE ≈ 0.57`.
- **P3 deliver**: the red ball is now a free body in open air — one pose write to a
  hover 4.5 cm above the nest, then physics drops and settles it into the ring
  (30-step stillness confirmation). `SIM_GEN_SCORE 1.0000`.
- **P4 persistence**: 3.33 s (10 × 40 steps) fully hands-off, success re-checked each
  block, then `SIM_GEN_SOLVE: SUCCESS`. Watchdog `threading.Timer(1350) → os._exit`;
  any exception prints traceback + `SIM_GEN_SOLVE: FAIL` + hard exit.

**Execution-order declaration:** expel-before-extract is *physically forced*, not
convention: the red ball cannot leave the groove while the arch stands (23 N pin,
smoke checks 2 & 4), and the arch cannot be broken by acting on the red ball. P3
before P1/P2 is impossible because the red ball is not reachable. The rubric latches
mirror this order and are monotone.

## 5. Embodiment argument (Franka, single arm, parallel jaw ≤ 80 mm, OSC)

Plausible base pose: base at the world origin, facing +x; the quarry sits 0.42 m away
and the nest 0.30–0.38 m further at 95–150° bearing — everything inside a 0.75 m
reach envelope.

- **Boulder expulsion (P1)**: the boulders are 95 mm — *larger than the 80 mm jaw
  opening*, so they cannot be grasped, exactly as intended. But their crowns stand
  proud of the 95 mm side walls (crown ≈ 0.10–0.11 m vs wall top 0.095 m), so the arm
  reaches over the open top and pushes a boulder crown with closed fingertips, arm
  ~horizontal, force along the channel axis. Required force 4–7 N at ≤ 0.2 m/s — well
  inside Franka's ~30 N sustained end-effector capability. The open top and the
  22.5 cm channel width leave clearance for the wrist.
- **Red-ball extraction (P2)**: after demolition the 50 mm ball fits the 80 mm jaw
  with 30 mm margin. Top-down grasp inside the open channel (walls 95 mm tall, ball
  centre at ~27 mm; the wrist approaches vertically through the open top). Lift is
  2 N — trivial.
- **Placement (P3)**: nest ring is open from above, inner width 66 mm vs 50 mm ball —
  8 mm radial clearance for a vertical place-and-release. OSC vertical descent with
  force cutoff suffices.
- **Per-object contact summary:** boulder → fingertip *push* (non-prehensile, forced
  by jaw asymmetry); red ball → *pinch grasp* + transport; nest → *release* above.

## 6. Smoke battery (smoke.py) — 14 checks

1. Settle & no-NaN; arch forms (both boulders wedged, red pinned).
2. **Arch lock**: 8 N upward yank on the pinned red for 1.5 s → rises < 10 mm,
   score ≤ 0.02 (the pin is real).
3. **Probe control**: the identical 8 N lift on a *free* red ball rises > 8 cm
   (check 2 is not vacuous — actuator moves when unobstructed).
4. **Seed-strategy rejection**: 6 N horizontal drag of the red ball toward the nest
   (the literal `roll_ball` strategy) for 2.5 s → moves < 3 cm, no freed latch, no
   success.
5. **Restoring ramp**: boulder placed halfway up the ramp rolls back into the channel
   (u < 0.06), no expel latch — you cannot half-expel.
6. Null policy 2 s → score ≈ 0, never success.
7. Randomization readback, 8 seeds: quarry xy span > 8 mm, yaw span > 5°, nest span
   > 5 cm, both bearing signs observed.
8. Determinism: same torch seed twice → state Δ < 1e-4.
9. **Near-miss**: arch broken + red resting *against* the nest rim (7.5 cm off
   centre) → not success, score ≤ 0.65.
10. **Exclusion clause**: red correctly in the nest but a boulder parked 8.5 cm away
    → rejected, score ≤ 0.65.
11. **Latched credit**: after latches set, teleporting the red away does not reduce
    the latched floor.
12. **Wrong object**: boulder dropped onto the nest ring (95 mm cannot enter 66 mm
    opening) with red still imprisoned → score ≤ 0.35, no success.
13. Rejection audit: `ever_success` was never set by any rejection construction.
14. Final finite-state check; frames.npz written (persp camera, 960×600).

Exact final line: `SIM_GEN_SMOKE: ALL PASS 14/14`, then hard exit.

## 7. Validation evidence (forge, RTX-4090 pod)

- `solve --seed 0`: SUCCESS, 19.6 s, scores 0.00 → 0.30 → 0.55 → 1.00 → 1.00.
- `solve --seed 1`: SUCCESS, 23.9 s (required runtime force-frame mode flip → probe
  logic exercised both modes).
- `solve --seed 2`: SUCCESS, 24.1 s (mode toggled 0→1 mid-expulsion; recovered).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 14/14` (see run log), frames.npz saved.
