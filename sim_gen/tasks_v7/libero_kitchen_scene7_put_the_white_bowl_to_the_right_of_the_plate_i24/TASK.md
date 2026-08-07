# mug_hook — hang the white mug on the BLUE peg by its handle

`libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i24` · env key `simgen.mug_hook`

## Seed provenance

Seed: `libero_90/libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate`
(RoboVerse `roboverse_pack/tasks/libero_90/...`). The seed's plan: grasp one rigid
object (white bowl), carry it through free space, **set it down** on the support
surface at a relative xy offset from a reference object (the plate); `_terminated`
is a bbox check on the transported object's own resting pose (x/y bands vs the
plate, z equal to the table).

## What changed, and why it is strategically different

The transported-white-vessel cast survives; the **goal mechanics are replaced
wholesale**:

- **Suspension replaces placement.** Nothing may end resting on a surface. The mug
  must end **hanging** — its rectangular handle loop threaded over a peg so the peg
  carries the whole weight through the loop while the mug dangles off the ground.
  The seed's own end state (mug set on the floor at the "right spot" under/beside
  the target peg) is expressible in this scene and scores ~0 (smoke check 4).
- **Topology replaces offset bands.** Success is not "center inside a box" but a
  topological relation — peg **through** the closed handle aperture — checked as
  aperture-center-to-axis-segment distance plus a live airborne + settled gate.
- **The target is indexed by COLOR, not position.** Three pegs (red/green/blue) are
  permuted over the three mounting slots every episode and each peg's height is
  resampled; a memorized pose fails; only the BLUE peg counts. The seed has no
  identity-binding requirement at all.
- **A handle-less distractor** (amber beaker) sits next to the mug — the "which
  object" binding is also load-bearing (it physically cannot be hung).

Distinct in PLAN from every corpus task read while building this: `i2`
(pour-contents-then-ordered-inverted-park), `i3` (bayonet twist-lock insertion),
`i19` (bridge/causeway push across a moat), `i9` (rotary airlock feeder). None of
them contains suspension-by-aperture-threading or color-permuted target binding.

## Scene (all procedural geometry)

- **rack** (kinematic): floor-standing panel 0.40×0.36 m facing the robot, on a base
  plate; xy jitter ±30 mm, yaw ±8°.
- **pegs** red/green/blue (kinematic): ⌀16 mm × 95 mm, root flange disc ⌀40 mm,
  mounted on the panel face **12° angled up** (a hung loop slides toward the panel
  and stays on); color→slot permutation and per-peg height 0.20–0.25 m resampled
  per episode.
- **mug** (dynamic, 120 g): white octagonal cup (~67 mm across, 70 mm tall) with a
  three-bar rectangular handle enclosing a **30×34 mm aperture**; floor start, xy
  jitter ±40 mm, free yaw.
- **beaker** (dynamic, 100 g): amber cup, no handle; xy jitter, free yaw.

## Rubric (latched partial credit anchored in the demonstrated solve)

| credit | clause |
|---|---|
| 0.15 | `lift` (latched): mug origin ever above 0.10 m |
| 0.30 | `thread` (latched): aperture center ever within 30 mm of the BLUE peg's axis segment `[root+10 mm, tip−25 mm]` **AND** the aperture normal (mug local +y) within ~60° of the peg axis |
| 1.00 iff `success()` (live) | threaded on blue AND airborne (origin > 0.10 m) AND settled (<0.05 m/s) AND finite; non-success caps at 0.45 |

Honesty by construction (smoke-verified): a real hang holds the aperture ~9 mm off
the axis (gate 30) with alignment |cos| ≈ 1 (its swing pivots about the peg axis);
a **mouth-hook** (peg inside the cup — the plausible near-miss) can settle with the
aperture center as close as ~29 mm, which is why distance alone is not the test —
its cup axis lies along the peg so the aperture normal stays ⊥ to the axis
(|cos| ≈ 0) under both tilt and roll, and the alignment clause rejects it; the
neighboring peg is ≥ 90 mm away; the 25 mm **tip margin** keeps an
aligned-but-unthreaded loop hovering off the tip outside the gate (a teleport to
the solve's staging pose cannot latch); no resting support in the scene reaches
the 0.10 m airborne gate (floor 0, on-beaker ~0.066, real hang ~0.15–0.19); an
unsupported mug at the right pose simply falls.

## Solution outline (solve.py, teleport = transport only)

1. **P0** settle 180 steps; read back the blue peg's root/axis (the permutation is
   random — never assume a slot); baseline score ~0 asserted.
2. **P1 TRANSPORT** (teleport-glide, ≤2 mm / ≤1.5° per step): lift, carry to a
   pre-staging point on the blue axis 110 mm off the tip, slide down the axis to
   STAGING — aperture aligned, centered **10 mm past the tip: NOT threaded**
   (asserted: outside the gate, latch unset). Score 0.15 (lift).
3. **P2 THREADING** (forces + contact): positional teleports stop; a rigid "wrist"
   re-writes orientation only (position copied back from physics each step) while
   translation is driven by CoM forces — gravity support + 0.4 mg axial push +
   radial centering, velocity-governed. The loop slides over the peg; the thread
   latch can only set here. Score 0.45.
4. **P3 HANG**: all forces cleared — hands off. Gravity seats the peg on the loop's
   top bar; the mug slides down the 12° tilt, swings, settles dangling (~40 mm
   below the axis, cup leaning toward the panel). success() live → score 1.0.
5. **P4** persistence ≥ 3.3 sim-seconds hands-off; success still holds →
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0, 1, 2 all SUCCESS (seed 2 with a different
color→slot permutation, blue in slot 0); `SIM_GEN_SCORE` prints
0.00 → 0.15 → 0.45 → 1.00 → 1.00, non-decreasing.

## Embodiment argument (single Franka + parallel-jaw, OSC)

- **Grasp the mug**: the cup body is ~63–67 mm across — inside the ~80 mm Franka jaw
  span — so the natural grasp is a side wrap of the cup below the rim with the
  handle pointing away from the palm; alternatively a rim pinch (5 mm wall). Mass
  120 g, trivial payload.
- **Threading**: with the cup held, the handle aperture (30×34 mm) must be slid over
  a ⌀16 mm peg — 7 mm lateral / 9 mm vertical clearance per side, a coarse-tolerance
  insertion well within OSC positioning. The peg tips point toward the robot and are
  angled 12° up: the approach is a straight wrist-forward slide, no regrasp and no
  wrist gymnastics (the solve's staging orientation keeps the mug upright).
- **Release**: open the gripper and retract; gravity does the rest (solve P3 is
  exactly this hands-off phase).
- **Reach / base pose**: robot base at the origin; mug and beaker start ~0.16 m
  ahead; peg tips are at x ≈ 0.31–0.32 m, heights 0.20–0.27 m — all well inside the
  Franka workspace envelope with the elbow up.
- **Perception binding**: colors are baked into the peg bodies; the policy binds
  "blue" visually — no privileged state needed.

## Execution-order declaration

No mandated action ordering. The only ordering is physical: the loop must be
threaded before release can produce a hang (an early release just drops the mug —
recoverable by re-grasping). Partial credits (`lift`, `thread`) are monotone
latches; success is judged live and must persist.

## Checks

- **solve.py** (forge, seeds 0/1/2): baseline ~0 assert; staged-outside-gate
  assert (no latch from teleports); thread-latch-set-during-force-phase assert;
  monotone `SIM_GEN_SCORE` at every phase boundary; ≥3.3 s hands-off persistence;
  watchdog + hard exit.
- **smoke.py** (forge): 10 checks — settle/no-NaN; randomization-by-readback
  (rack/mug/beaker xy move, blue slot permutes, blue height spans >5 mm);
  null-policy ~0; SEED-STRATEGY floor placement under the blue peg rejected;
  wrong-object (beaker on peg) rejected; WRONG-PEG real hang on red rejected
  (score caps at lift); LOOP INTERLOCK (3×-weight downward shove cannot tear the
  loop off — suspension is load-bearing); mouth-hook near-miss rejected by the
  alignment clause (aperture edge-on, no latch); off-tip aligned staging outside
  the gate + released mug falls; frames.npz saved.
