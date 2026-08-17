# put_mug_i173 — hang the mug by its handle on the red peg

**Env key:** `simgen.mug_hook` · **Robot slot:** `null` (scene-level task)
**Seed:** `roboverse_pack/tasks/embodiedgen/put_mug.py` (RoboVerse, EmbodiedGen pack)

## Seed provenance and what changed

The seed is a flat pick-and-place: a Franka lifts a mug off a table and releases it
inside a marked target region on the same table, judged by a `DetectedChecker` with a
`RelativeBboxDetector` (±0.1 m box around a marker, orientation ignored). Success is
*containment on a support surface*; any release inside the box counts.

Here the mug keeps its identity but the seed's goal topology is deleted entirely:

- **No valid support surface exists.** The only success state is the mug HANGING in
  mid-air from the upper (RED) peg of a free-standing mug stand, with the peg passing
  THROUGH the mug's closed handle window (a topological condition computed in the
  mug's body frame), suspended high off the ground, and swung to rest (a stillness
  streak with a teleport/pose-jump guard).
- **The seed's own strategy is a scored decoy.** A flat blue display pad sits on the
  ground; setting the mug down on it — the seed's entire plan — scores ~0. So do:
  parking the mug on the stand's base plate, hanging it on the lower GREY decoy peg,
  and hooking it on the red peg by its RIM (peg in the cup cavity instead of through
  the handle window).
- **New skills required, absent from the seed:** reorienting the mug so the handle
  window faces along the peg axis, threading a 34×50 mm window over a 14 mm peg
  (~10 mm lateral clearance), sliding the hung loop along the tilted peg, and a
  release that leaves the mug pendulum-swinging until it settles.

### Strategically different from every task read this session

- vs **`pen_holder`** (format exemplar): that is insert-many-into-a-container with
  subset sampling; no suspension, no threading topology, no swing-settle.
- vs **`put_bottle_in_fridge_i311`**: that is lay-align-in-trough + ride a sliding
  rack through a mouth (prismatic transport, containment end state). Here nothing
  slides on rails and nothing is contained — the end state is a body suspended from a
  hook by a topological link, reached through a dynamic catch and swing decay.

## Assets (fully procedural, custom compound spawners)

- **mug** (dynamic, 0.25 kg authored + readback-asserted): hollow cup (floor disc +
  8 wall boxes, Ø80 mm × 90 mm) with a closed rectangular handle loop on its +x side
  (two horizontal bars + outer vertical bar) enclosing an open ~34×50 mm window.
- **stand** (kinematic): square base plate, vertical post (0.64 m), two Ø14 mm pegs
  tilted 12° up — RED high (root 0.45 m, the target), GREY lower opposite (0.32 m,
  decoy). μ ≈ 0.5 ≫ tan 12°, so a hung mug cannot walk off the tip.
- **pad** (kinematic): flat blue disc on the ground (seed-strategy decoy).

Randomization (all readback-verified in smoke): stand xy jitter ±3 cm + yaw ±30°
about nominal 180°, mug xy in a ground band with free yaw ±180°, pad xy in the
opposite band, bands swap 50/50.

## Rubric

Latched partial credit (never evaporates), updated only in `post_step`:
`0.15·lifted + 0.20·approach + 0.35·threaded_ever`, capped at 0.85; exactly 1.0 iff
`success()` live. `success()` = threaded-now on RED ∧ suspended (origin > 0.22 m) ∧
settled (45 consecutive still post_steps, streak reset by any per-step pose jump
> 2 cm, plus a live no-jump re-check) — so a teleported-in "success pose" judges
False until it has genuinely hung there.

## Solution phases (solve.py, teleport = transport only)

- **P0** settle + baselines: mass readback, layout readback, not-success assert.
- **P1 TRANSPORT** (the only mug pose write): one root-state write to a contact-free
  hover off the RED peg's tip — upright, window plane perpendicular to the peg, peg
  axis already crossing the open window 20 mm from the tip (free space), peg line
  27 mm below the catch line. Asserted: threaded-now ∧ NOT success (streak/jump
  guard).
- **P2 CATCH** (contact dynamics): 27 mm free fall, handle top bar lands on the peg,
  pendulum swing on the hook.
- **P3 SLIDE HOME** (contact dynamics): velocity-regulated bang-bang horizontal force
  at the CoM (1.5→5 N escalation, 0.06 m/s cap, runtime frame-encode probe for the
  pod's wrench-rotation drag) drags the hanging loop along the tilted peg toward the
  root; release. Asserted: still threaded, ≥10 mm progress.
- **P4** hands-off swing decay until `success()`.
- **P5** ≥3.5 simulated seconds hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at each phase boundary, asserted non-decreasing.

## Embodiment argument (Franka, single arm + parallel jaw)

- **Grasp:** the mug body is Ø80 mm — exactly the Franka jaw limit, so the intended
  grasp is the RIM (wall thickness 6 mm, an easy pinch) or the handle bar (8×10 mm).
  A rim grasp from above leaves the handle window fully exposed for threading.
- **Reach/clearance:** with the base at ≈(−0.15, 0, 0) facing +x and the stand at
  (0.50±0.03, ±0.03), the red peg tip sits ≈0.37–0.42 m ahead at height 0.48 m —
  well inside the Franka workspace at comfortable elbow height. Threading approaches
  the peg tip from open air (no clutter within the 0.13 m peg sweep); the hand stays
  above the rim, clear of the peg.
- **Precision:** the window clears the peg by ~10 mm laterally and ~18 mm vertically
  (per side), comfortably above closed-loop arm repeatability; the 12° up-tilt means
  small release errors slide the loop *toward the root*, into the pocket.
- **Forces:** the slide-home drag is ≲5 N on a 0.25 kg mug — trivially within Franka
  payload; the mug can also simply be released deeper along the peg, so the slide is
  a robustness bonus, not a strength demand.

## Execution order

No required ordering beyond the physical one (thread before release); there are no
independent sub-goals — declared: any approach that ends with the mug genuinely
hanging by its handle window on the red peg, settled, succeeds.

## Checks

- `solve.py`: mass/layout readback asserts, threaded-hover-not-success assert, catch
  asserts, slide progress assert, non-decreasing `SIM_GEN_SCORE`, success + 3.5 s
  hands-off persistence, on ≥2 seeds.
- `smoke.py` (rejection battery, 14 checks): settle/no-NaN ×2, randomization
  readback ×2, null policy, seed-strategy pad set-down, stand-base park, settled
  GREY-peg hang, RED-peg rim/cavity hang, fly-through immediate judge, mid-catch
  yank-away (latch persists, success never), lift-latch survival, never-success
  audit, final no-NaN. Records `frames.npz`.
