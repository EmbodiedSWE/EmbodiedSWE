# kerf_chop — sever the welded rod through the kerf slot (`put_knife_in_knife_block_i41`)

## Seed provenance

Seed: `rlbench/put_knife_in_knife_block` — pick a knife off a chopping board and drop it into a
slot in a knife block. The seed is a pure pick-and-insert: grasp a thin object, align it with a
narrow vertical slot, lower it in, let go. Success = knife resting inside the block.

## Strategic difference

This task keeps the seed's signature *thin-blade-through-narrow-slot* insertion but makes the
insertion the **means, not the goal**. The goal is a **force-mediated irreversible state change**:
a 240 mm rod bridging two anvils is made of three butted segments joined by two **breakable weld
joints** (PhysX `breakForce`/`breakTorque` fixed joints). The agent must

1. choose the **thin knife (4 mm blade)** over the decoy **cleaver (20 mm blade)** — only the
   knife fits the 13 mm kerf slot;
2. insert the blade through the slot in a fixed cover (the cover makes the rod unreachable by any
   gripper — tool use is *forced*, not optional);
3. **press down hard through the blade** (≥ ~8 N sustained) to shear both seam welds — merely
   resting the blade on the rod, or the null policy, never breaks them;
4. leave the red middle span to **fall into the well** while both beige outer segments stay seated.

Nothing in the corpus is like this. The corpus (~43 tasks) is dominated by pick-and-place
(bowls/cans/pans/trays), articulation (drawers/microwave/oven/grill/dishwasher), insertion-as-goal
(peg_insertion, plug_charger, and the sibling knife seeds `approach_grasp_knife_i25`, which is a
grasp task), pushing/tracking, and stacking (jenga, checkers, pyramid). No task has **breakable
joints**, none requires **sustained force application through a held tool**, and none has an
**irreversible physical outcome** as the success predicate. The insertion here is contact-free
transport; all the scoring physics is the press, the weld shear, and the drop — a different
strategy, different failure modes, different reward surface than the seed.

Tool choice is also load-bearing (seed has one knife, no decoy): the cleaver has the same handle
and mass but a 20 mm blade that physically cannot enter the 13 mm slot — smoke check 7 presses it
at 30 N and the seams hold.

## Scene

- **Chopping station** (25 kg dynamic body, jittered xy ±4 cm, yaw ±20°): two anvil towers
  (top 60 mm) flanking an open well, rod fences (2 mm rod clearance — the 4 mm blade cannot slip
  past the rod), end stops, and a fixed cover at 96–104 mm with a kerf slot 100 × 13 mm directly
  over the rod's red span.
- **Rod**: three 80 mm segments, 24 × 24 mm, 0.12 kg each — beige `RodA`/`RodB` on the anvils,
  red `RodMid` spanning the well — welded at two seams with `breakForce=60 N`,
  `breakTorque=0.16 N·m` authored at spawn. Rod x-offset jittered ±6 mm.
- **Tools**: knife (blade 50 × 4 × 40 mm) and cleaver (blade 20 mm thick) on the ground in front,
  side shuffled per episode, xy jitter ±3 cm, free yaw.

Randomization (verified by readback in smoke check 3–4): station yaw & xy, rod offset, tool side
swap, per-tool xy/yaw.

## Solution outline (solve.py)

Teleport is used **only for transport through free space**; every load-bearing interaction is
contact dynamics under applied wrenches (`set_external_force_and_torque`, force-frame drag
compensated by the 3-candidate probe).

1. **Fetch & hover** — teleport the knife from its ground slot to a hover pose above the kerf
   slot, blade down, aligned with the slot (attitude PID + clamped lateral spring hold it there).
2. **Engage** — teleport to a *contact-free* pose inside the slot aperture (blade in the 13 mm
   slot, tip 3 mm above the rod), then a gentle 0.3 N touch-down onto the red span.
3. **Chop** — press with a ramped downward force (8 N → +8 N every 60 steps, cap 40 N) through
   the blade; a flight guard re-stages if the knife ever escapes. If one seam snaps first, the
   next press targets the *dropped side* (±20 mm offset) where the moment arm is largest. In
   practice both seams shear together at the first 8 N stage.
4. **Retract & park** — teleport the knife out of the slot and to a park pose on the ground.
5. **Hands-off** — 720 free steps: the red span falls and settles in the well; then success is
   re-checked over 10 × 40 steps (≥ 3.3 simulated seconds of persistence).

`SIM_GEN_SCORE` printed at every phase boundary; trajectory 0.00 → 0.15 (engaged) → 1.00
(severed + success) → 1.00, non-decreasing via latches. Verified end-to-end on the forge on
**seeds 0, 1, 2** (rc=0, `SIM_GEN_SOLVE: SUCCESS` on all three).

## Rubric (score)

Latched partial credit, capped, full marks only on the live physical goal:

- `+0.15` engaged — knife blade tip below the cover, inside the slot footprint (latched)
- `+0.25` per seam severed (seam separation readback > 20 mm, latched)
- partial sum capped at 0.65
- `= 1.0` iff live `success()`: both seams severed **and** red segment settled in the well
  **and** both beige segments still seated **and** station upright **and** settled **and** finite.

The rubric was written after the solution worked and anchored to measured forge behavior
(resting ≈ 1.5 N holds indefinitely; deliberate 8 N press shears; `describe()` matches).

## Embodiment argument (Franka, base pose)

Place the Franka base ~0.35 m in front of the station (station nominal at (0.45, 0)). The tools
lie flat on open ground at (0.16, ±0.24) — a standard top-down pinch on the knife's 24 mm-tall
handle block, well within reach. The station cover top is at 104 mm, so the slot is approached
from above at ~0.10–0.25 m height, comfortably inside the Franka workspace. The press is a
downward Cartesian force along −z through a vertical blade — exactly the wrench a Franka applies
best (stiff in z, tool held in a power pinch; 8–40 N is well under its 70 N payload-direction
capability). The 100 mm slot length gives ±35 mm of x tolerance for blade placement; the lateral
spring in the solution mirrors what impedance control provides for free. The cleaver decoy is
graspable identically — the discrimination is cognitive (blade thickness vs. slot width), not a
dexterity gate.

## Execution order declaration (smoke.py)

Weld breaks are **irreversible across `env.reset()`** (reset restores poses, not welds), so the
battery is strictly ordered: all intact-weld checks run first (1–7), the single deliberate chop
runs once (8), and every post-break negative (9–13) is constructed by writing body states
directly — **no resets after the chop**.

## Smoke battery (14 checks)

1. Settle + no-NaN (120 steps, seed 100).
2. Geometry gate: knife blade + margin fits the slot; cleaver blade does not.
3. Randomization readback: yaw > 2°, station xy > 3 mm, rod offset > 1 mm, knife xy > 5 mm
   across seeds.
4. Tool-side swap occurs across 10 resets, both sides observed.
5. Null policy 240 steps: seams intact, score ≤ 0.05, no success.
6. **Seed-strategy end state**: knife written to rest tip-on-rod through the slot (the seed's
   "knife in block" outcome), 240 steps — seams hold, score = engaged credit only, **no success**.
7. **Wrong tool**: cleaver pressed at 30 N over the slot, 180 steps — blade never passes the
   cover, seams intact, score ≤ 0.05.
8. **The chop** (solve replica, recorded to video): both seams snap, red span in well, success,
   score = 1.0, ≥ previous score.
9. Red span parked on the ground outside the well — score ≤ 0.65, no success.
10. Red span resting on the cover — no success.
11. **Wrong piece in well**: beige segment placed in the well, red outside — no success.
12. Beige outer knocked off its anvil (red in well) — no success.
13. Restore the goal state — success, score = 1.0 (proves the negatives failed for the right
    reasons).
14. `frames.npz` video saved (> 10 frames).

Passes with `SIM_GEN_SMOKE: ALL PASS 14/14` on the forge.
