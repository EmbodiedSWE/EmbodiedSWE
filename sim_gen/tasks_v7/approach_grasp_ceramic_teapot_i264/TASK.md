# approach_grasp_ceramic_teapot_i264 — Teapot Bayonet Lock (scene `teapot_bayonet`)

Lock the teapot's lid with a BAYONET twist. A slate-blue kinematic teapot (round body,
blocky spout and handle) stands on the ground at a randomized pose; its free lid (cream
cap, dark-green knob, two RED LUGS on the plug flanks) lies upright on the ground
nearby. The pot's mouth is a keyed collar: a cream throat ring with two diametric NOTCH
gaps (44°, marked by red stripes on the outer wall), an annular chamber under the ring,
and an internal ledge. The lid enters ONLY with its lugs aligned to the notches; once
the plug seats on the ledge the lugs sit in the chamber below the ring, and twisting
the lid ≥ 36° (either way, internal stops at ~48°) slides the lugs under the solid
ring — the lid is then geometrically CAPTIVE. Success = lid seated at depth, on axis,
upright, twisted into the lock window, and at rest.

## Provenance

- **Seed:** `pick_place/approach_grasp_ceramic_teapot`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_ceramic_teapot.py`)
  — approach the known ceramic teapot in tabletop clutter, close the parallel jaw on
  it, hold (gripper-object distance < 2 cm sustained) with a small lift; the episode
  ends HOLDING the object.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `teapot_bayonet`,
  env `simgen.teapot_bayonet`, robot `"null"`), `solve.py` (teleport solution),
  `smoke.py` (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's judged outcome is a GRIPPER RELATION — be near/holding
  the object; the strategy is approach + grasp, and the episode ends mid-hold. Here no
  gripper quantity is ever read and holding the lid aloft is worth exactly the 0.15
  carry latch (smoke check 6 constructs the seed's end state — lid held above the
  mouth — and it scores ≤ 0.16, NOT success). The judged outcome is a MECHANISM STATE
  between two rigid bodies: a keyed insertion (perceive the red stripes → align the
  fold) followed by in-place ROTATION to geometric captivity. The load-bearing
  resources are relative-yaw perception, a keyed aperture, and torque about a seated
  axis — none of which exist in the seed.
- **vs siblings read this session:** `..._i109` (mug-rack hang) is prehensile placement
  into a suspended equilibrium on a peg — no keying, no rotation DOF, nothing captive;
  `..._i209` (tea-ball transfer) is unseal → transfer → gravity wedge-seat, judged on
  CONTAINMENT of a third body — its lid mechanics are a straight pull and a drop-in
  wedge, never a keyed rotation, and nothing ends captive; `close_jar_i54`
  (latch canister) also ends with a secured lid BUT the lock lives in fixture-mounted
  revolute TURN-TABS the solver flips over a passive lid — here the pot has ZERO
  moving parts and the PAYLOAD ITSELF is the rotating lock element (the verb is
  twist-the-held-object, not actuate-the-fixture); `approach_grasp_bowl_i133` (cliff
  sweep) is multi-body herding with a tool under a never-lift invariant — different in
  every axis. The packing exemplar `pen_holder` is repeated insert-into-aperture with
  no key and no lock state.
- **Execution order (declared, geometry-forced):** align → insert → twist. A rotated
  lid CANNOT enter (lugs land on the solid ring: smoke 7); an inserted-but-unrotated
  lid is NOT locked and pulls straight out (smoke 8 extracts it with a slow 0.15 m/s
  pull); rotation credit only accrues while seated (`rot_gate_xy`, z < seat band). No
  prefix of the sequence satisfies the goal.

## Randomization (per episode, verified by readback in smoke)

Pot centre = (0.10, 0) ± 45 mm uniform xy AND free yaw ∈ (−π, π] — the yaw is the
NOTCH PHASE, the quantity a solver must perceive (red stripes mark it). Lid spawns
upright on the ground on a free polar slot around the pot at radius U(0.28, 0.38) —
asserted in cfg to clear the pot's spout/handle envelope — with its own free yaw, so
the initial relative fold is uniform. All via `torch.rand` (no first-`randint` draw).

## Rubric

`success()` iff, judged live on physical poses: lid origin within 10 mm of the pot
axis, origin z in the seat band (0.188, 0.208) — analytic seat 0.198, a rim-resting
lid reads 0.222 and FAILS by 14 mm — lid axis within 8° of world-up, folded relative
yaw (mod 180°, the lug symmetry) in [36°, 55°] (full lug engagement ends at 33.3°,
stops end travel at ~48°), and the lid settled (|v| < 0.08 m/s, |ω| < 0.5 rad/s).

`score()` (latched in `post_step`): 0.15 carried (lid above the rim near the axis)
+ 0.25 entered (origin below the ring bottom while on-axis — reachable only through
the keyed throat) + 0.35 × latched max seated rotation / 36°, capped at 0.75; exactly
1.0 iff `success()`. Null policy ~0; the seed strategy tops out at the carry latch.

Cfg `__post_init__` asserts the honesty geometry: aligned lugs have ≥ 8° yaw margin
through the notches; the lock threshold sits above full engagement and below the
stops; captive vertical play is 3–10 mm (lugs really trapped under the ring); the
pilot nose cannot pass the ledge hole while giving ≥ 10 mm xy capture; the plug-throat
clearance is 5–10 mm; lugs overlap the ring ≥ 8 mm radially (real captivity); the rim
rest fails the seat band with ≥ 10 mm margin; the entered gate sits between seat band
and ring bottom; the knob fits the Franka jaw with 30 mm spare; the seated cap stays
above the rim (nothing fouls).

## Teleport solution (`solve.py`) — transport only, single free-space teleport

- **P0:** settle, layout readback (pot xy/yaw, lid pose, fold), baseline score.
- **P1 transport + keyed insertion:** ONE teleport of the lid to a free-space hover on
  the pot axis, nose bottom 18 mm above the rim, yaw matched to the pot readback
  (lugs over the notches), velocities zero. Then CONTACT: a velocity-servoed press
  (F = m·K·(v_des − v) + gravity feedforward, K·dt = 0.5, descent 0.08 m/s, xy
  centering, weak yaw-hold torque ≤ 0.05 N·m) lowers the lid through the notch gaps
  onto the ledge; wedges re-hover and retry (≤ 5).
- **P2 twist:** light continued seat press + angle-PD yaw torque (Kp 0.08 → ×1.7 on
  90-step stalls up to 0.9, damping 0.010·ω, clamp ± 0.15 N·m; ledge friction torque
  ≈ 0.03 N·m, so authority ≈ 5×) toward 43° — overshooting the 36° credit line
  (latch-lag margin); the symmetric direction is tried if one jams.
- **P3 release:** brake yaw rate, clear the wrench, 1.5 s hands-off settle.
- **P4 persistence:** ≥ 3.4 simulated seconds hands-off with `success()` held, then
  `SIM_GEN_SOLVE: SUCCESS`. `SIM_GEN_SCORE` at every phase boundary is asserted
  non-decreasing (0 → 0.40 → 0.75 → 1.0 → 1.0).
- **Verified on the forge: seeds 0 and 1, both SUCCESS**, provably distinct by
  stdout readback — seed 0: pot (+0.091, +0.043) yaw −173.0°, lid at (+0.004, +0.406)
  fold 24.9°; seed 1: pot (+0.135, +0.015) yaw −90.9°, lid at (+0.121, +0.320) fold
  31.4°. Both traces 0.00 → 0.40 (insert attempt 0, z = 0.204) → 1.00 (twist reached
  39.2° in 44 steps at Kp 0.08, no stall) → 1.00 → 1.00. Smoke battery:
  `SIM_GEN_SMOKE: ALL PASS 14/14`, frames.npz (168, 600, 960, 3).

## Embodiment sanity (single-arm Franka feasibility)

One ground-level base pose serves the family: base at (0.10 ± 0.05, −0.65) facing the
pot — every contact lives in a 0.28–0.43 m radius ring at z ≤ 0.30, inside the ~0.85 m
envelope. Per-object strategy: the LID is grasped by its dark-green KNOB (22 mm square
≪ 80 mm jaw, cfg-asserted, with 34 mm of finger height and the cap as a natural depth
stop); carried 0.3 m and lowered onto the mouth — the 12 mm pilot-nose capture and the
10.7° notch yaw margin absorb OSC placement error. The twist is ≤ 48° of pure wrist
rotation about a vertical axis — well inside `panda_joint7`'s ± 166° range with the
hand pointing down — under ≤ 4 N seat force and ≈ 0.03 N·m of ledge friction, trivial
for OSC torque limits. The POT is a fixed fixture; nothing else is ever touched. The
spout/handle sit at ± 90° from the notch line below the rim, so the approach cone
above the mouth is always clear.

## Checks (`smoke.py` — rejection battery, 14 named checks)

1. settle: states finite; lid upright at ground height inside its spawn ring, pot at
   its jittered pose (readback).
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: pot xy jitter AND free pot yaw (notch phase) vary across 8
   seeds, jitter within its band.
4. randomization readback: lid ring radius in band every seed; polar slot, lid yaw and
   relative fold all vary.
5. null policy: 300 idle steps → score ~0, no success, lid drift < 30 mm.
6. SEED strategy: lid grasped and HELD above the pot mouth (the seed's end state),
   then set down beside the pot → carry latch only (≤ 0.16), NOT success at either
   judged point. THE key check: the seed's verb earns almost nothing.
7. rim-rest at the lock angle: lid arrives already rotated 45° → lugs perch on the
   solid ring 24 mm above the seat band, NO entered credit, NOT success (right angle,
   wrong depth).
8. seated-unrotated: lid dropped through the notches seats on the ledge (seated()
   true, entered credit in, score 0.38–0.45) yet NOT locked, NOT success.
9. pull-out proof: a slow (0.15 m/s servo) upward pull EXTRACTS that unrotated lid
   above the rim — the unlocked state is physically not captive (actuator-moved
   readback, no vacuous probe).
10. under-rotated near-miss: seated at ~25° (< 36°) → NOT success, partial credit
    only (0.35–0.78; measured 0.493).
11. latched regression: teleporting that lid back to the ground leaves the latched
    credit unchanged (0.493 → 0.493), success stays gone.
12. inverted lid (knob-down): cap perches on the rim far above the seat band
    (z = 0.278), upright fails → NOT success.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` recorded and saved in CWD.

Note: the locked-lid captivity itself (pull-up on a LOCKED lid) is deliberately NOT a
smoke probe — constructing the locked state would violate the battery's rejection-only
audit; captivity is instead guaranteed by the cfg-asserted lug/ring overlap + vertical
play and demonstrated by the solve's hands-off persistence hold.
