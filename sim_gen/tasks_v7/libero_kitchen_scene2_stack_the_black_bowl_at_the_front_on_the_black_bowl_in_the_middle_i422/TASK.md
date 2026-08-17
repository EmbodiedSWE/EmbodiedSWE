# screed_patch — rebuild the deck: inter the keystone, pave to flush, screed the patch

`sim_gen` task `libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_i422`
· scene name `screed_patch` · env `simgen.screed_patch` · robot `null`

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle`
(RoboVerse pack, `roboverse_pack/tasks/libero_90/...stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.py`).
The seed is a plain pick-and-place stack: grasp one bowl, hover it over the other, release;
success is an xy/height proximity predicate on the two origins. Its whole difficulty lives
in grasping and carrying one payload to one target.

## What this task is

A **deck repair job**. A rectangular pit is sunk through a raised deck; its floor sits
exactly TWO or THREE slab-thicknesses (40 or 60 mm) below the deck plane — the depth is
randomized per episode and must be judged by looking into the pit. Four identical slabs
(one RED keystone, three gray pavers) lie racked on skid rails to one side, and a screed
bar is parked on the near side. The job, in forced order:

1. **INTER** the red keystone: lower it flat so it rests directly ON the pit floor
   (bottom course; a keystone resting on pavers is a full 20 mm course out of its 6 mm
   z band).
2. **PAVE** with exactly *depth − 1* gray pavers so the patch top is FLUSH with the deck
   plane (±4 mm). One paver short leaves the patch 20 mm low; one extra leaves it 20 mm
   proud; both fail — the flush band is 5× smaller than a one-slab miscount, and a
   7°-tilt fake can raise a slab centre at most ~8 mm, still short.
3. **SCREED**: only after the patch is complete, drag the bar by its handle straight
   across the patch — a genuine deck-level sliding pass — and park it clear on the far
   side.

Success = interred ∧ covered ∧ flush ∧ pass latched ∧ bar clear ∧ everything settled.

So the task is simultaneously **ordered** (keystone must go down first — the interred gate
reads the keystone's z against the pit floor), **counted against a hidden-in-plain-sight
observation** (the paver count is not fixed: it must be read off the pit depth, and the
flush gate arithmetically forces the right count), and **finished with a tool pass** (the
screed latch requires *sustained horizontal motion* of the bar over the completed patch —
5 consecutive steps of |vx| > 0.03 m/s with |vz| < 0.015 m/s at deck level).

## Why it is strategically different

- **vs the seed**: the seed stacks one object on another by proximity — every strategy is
  grasp–carry–release, and any two-bowl overlap scores. Here nothing is stacked *on a
  target object*: slabs are seated *into a cavity in the static world*, the number of
  placements is episode-dependent (count must be perceived, not memorized), the order is
  physically graded (bottom course identity), and the goal is completed by a **tool
  motion** (the screed pass), a verb the seed does not contain at all.
- **vs the corpus** (tasks examined): `.._i117` (same seed) is a waiter's-pull support
  extraction — payload never touched; here every slab is transported and seated, and the
  bar is dragged, the opposite manipulation profile. Fill/insertion tasks in the corpus
  (pen-holder-style counting, moat causeway bridging) either count a fixed quota or bridge
  a gap for transit; none makes *flush-with-a-plane* the arithmetic verifier of a
  perceived count, and none requires a post-construction tool pass over the built
  structure as a separate physics-gated phase.

## Scene (procedural, no external assets)

Kinematic bench (top z 0.12) carrying four kinematic deck plates (top z 0.20 = the deck
plane) that leave a 154 × 164 mm pit; a kinematic pit-floor block sits depth·20 mm below
the deck (depth ∈ {2,3}). Dynamic bodies:

- **keystone** (red, 0.20 kg) and **3 pavers** (gray, 0.18 kg): 130 × 140 × 20 mm slabs,
  racked flat on two 24 mm skid rails (under-edge clearance for a parallel jaw) to one
  side of the pit.
- **screed bar** (0.12 kg compound body): 50 × 240 × 15 mm blade plate + 24 mm square
  yellow handle post (the pinch point) riding on two 6 mm end **skid runners**. The
  240 mm span out-reaches the pit, so a dragged bar rides its runners on the flanking
  deck strips across the opening, the blade passing 6 mm above the deck — it clears a
  flush (±4 mm) patch but jams on a slab standing a course proud.

Per-seed randomization (all verified by readback in smoke): pit xy jitter ±30/±40 mm,
depth 2 or 3 (pit-floor block physically re-seated), depot side mirrored ±y, keystone
rack slot (4 slots), bar park x jitter, slab yaw jitter ±5°.

## Rubric

Milestone latches in `post_step`, additive, monotone; rest latches velocity-gated:

- **+0.15 interred** — keystone at rest flat on the pit floor (z band 6 mm, in-pit xy).
- **+0.35 capped** — patch complete at rest: interred ∧ covered (a paver ≥ 14 mm above
  the keystone, in pit) ∧ flush. Flush membership is by **pit XY footprint at any
  height** (not a z band): the max top of every slab over the footprint must sit within
  ±4 mm of the deck plane and each such slab must be flat — so an extra slab perched a
  course proud breaks flush even though its centre is above the deck plane.
- **+0.20 swept** — the screed pass: patch already complete ∧ bar over the pit at
  deck-level rest z (±8 mm) ∧ flat ∧ **sustained horizontal motion** — 5 consecutive
  steps of |vx| > 0.03 m/s AND |vz| < 0.015 m/s. A teleported bar has zero written
  velocity; a bar teleport-dropped just above the patch picks up a vertical impact
  transient (vz ≈ 0.1+ m/s inside the band) and is vetoed by the vz gate + streak.
- **1.00** iff `success()` = interred ∧ covered ∧ flush ∧ swept ∧ bar parked clear
  (≥ 130 mm past the pit centre, deck level, flat) ∧ settled.

## Execution order (declared)

1. Minimal goal predicate + scene brought up on the forge.
2. `solve.py` made to pass on the forge (this drove the skid-runner fix below).
3. Rubric finalized (sweep motion gate hardened to horizontal-streak) — `smoke.py`
   written against it.

Design iterations worth recording:

- **Rubric hole caught at smoke-design time**: the original sweep motion gate was a bare
  speed threshold; a bar teleported 1 mm above the finished patch gains ~0.14 m/s of
  *drop* speed while inside the sweep z band and would have latched. Fixed by splitting
  the gate into horizontal-only motion (|vx| floor, |vz| veto) plus a 5-consecutive-step
  streak counter. Smoke checks pin both the air-carry and the teleport-drop constructs.
- **Forge-caught geometry bug**: the first bar was a flat-bottomed plate skimming at
  exactly deck level. A legitimately flush patch can rest a fraction of a millimetre
  proud, and that step stopped the drag dead (and at higher force flipped the bar onto
  its leading edge). Fix: 6 mm skid runners under the plate ends — the blade now clears
  the whole ±4 mm flush band while a one-course (20 mm) proud slab still physically
  blocks the pass. The runner stance is asserted to stay on the deck strips flanking the
  pit.
- **Second rubric hole caught by the smoke battery on the forge**: the original flush()
  took membership from the in-pit z gate, so an OVERFILL slab perched a course proud
  (centre above the deck plane) silently dropped out of the computation and flush read
  true. Fixed to footprint membership at any height (rubric above); the overfill smoke
  construct was also rewritten as a burst hover-release of the whole depth+1 column so
  it never transits a legitimate complete-patch rest state (which would rightly latch).
- **Forge-caught drag dynamics**: an unsteered world-x CoM force lets uneven runner
  friction yaw the bar and walk it sideways (~0.18 m over a 0.4 m pull) — benign over a
  filled patch, but over the OPEN pit (smoke's premature-sweep probes) a ~30 mm drift
  drops a runner into the opening and tips the bar in. Fix: the small lateral P-D steer
  toward the pit centreline in every force-drag (solve and smoke alike) — exactly the
  correction an arm dragging the handle would apply.

## Teleport solution (solve.py) — phase outline

Teleports transport only; every load-bearing interaction is contact dynamics.

- **Phase 0** — reset(seed), settle, layout readback (pit xy, DEPTH via depth readback —
  the paver count is computed from the episode, a fixed count fails half the episodes;
  pit-floor z cross-checked against physical readback). `SIM_GEN_SCORE 0.000`.
- **Phase 1 — inter**: keystone teleported to a hover 30 mm above its pit-floor seat
  (outside the 6 mm interred band), released; it lands and seats by gravity. Retry ladder
  with small xy offsets. `SIM_GEN_SCORE 0.150`.
- **Phase 2 — pave**: depth−1 pavers hover-released onto the stack, one course at a
  time, each seat verified by readback; patch-top vs deck printout. `SIM_GEN_SCORE 0.500`.
- **Phase 3 — screed**: bang-bang horizontal external force on the bar (the body the arm
  would drag by its handle) under a 0.20 m/s speed cap until well past the pit, plus a
  small lateral P-D steer toward the pit centreline (clamped ±0.8 N — uneven runner
  friction walks an unsteered bar sideways), then an active reverse-force brake; force
  ladder 1.5/2.0/2.5 N across retries; failed attempts rebuild the phase START by a
  transport teleport (bar back to its park) and retry through physics.
  `SIM_GEN_SCORE 1.000`.
- **Phase 4 — persistence**: ≥ 3.3 simulated seconds hands-off, success must hold.
  `SIM_GEN_SCORE 1.000`, then `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (Franka, one plausible base pose)

Base plausibly at **(pit_x − 0.42, pit_y, 0.20)** on the deck, facing +x: the pit
(~0.42 m), the depot rack (~0.3–0.5 m), and the far-side bar park (~0.65 m) all sit
inside a Franka's ~0.85 m comfortable reach.

- **slabs**: 130 × 140 × 20 mm, 0.18–0.20 kg — racked on 24 mm rails so a parallel jaw
  can close on a slab *edge* (20 mm thickness ≪ 80 mm stroke) with under-edge clearance;
  the placement is a lower-and-release into a 12 mm/side clearance pit.
- **screed bar**: dragged by its 24 mm square handle post (comfortable pinch), a
  straight-line ~0.4 m pull at ≤ 0.2 m/s — slow, well inside joint limits; the 0.12 kg
  bar slides on its runners.
- **deck/pit/rails**: kinematic furniture; never manipulated.
- Nothing in the task needs a grasp wider than 80 mm or a force beyond a few newtons.

## Smoke battery (20 checks, frames.npz recorded)

1–2. Racked start settles finite; rubric clean at reset (score 0, bar not clear).
3. Determinism: same seed → identical layout readback.
4–6. Randomization by readback: depth takes both values and matches the physical
   pit-floor z; pit xy jitters; depot side/keystone slot/bar park vary.
7. Null policy: idle steps → score ~0, no success.
8. **Wrong order rejected**: pavers first, keystone on TOP — flush can be true but score
   stays 0 (no interred, no capped credit).
9. Underfill: keystone alone → 0.15 only.
10. Overfill: keystone + depth pavers burst-released as one column (never passing
    through a complete patch) → the proud slab is over the pit footprint → not flush
    → 0.15.
11. Flush-without-keystone (gray-only fill) → 0.
12. Genuine build → 0.50 latched.
13. Bar TELEPORTED clear (zero written velocity) → no sweep latch, no success at 0.50+.
14. Bar flown OVER the pit 60 mm up with written velocity → no latch (z band).
15. Bar teleport-DROPPED onto the patch (vertical impact transient) → no latch
    (vz veto + streak).
16–17. **Premature sweep**: bar genuinely force-dragged across the EMPTY pit (it crosses
    on its runners) → no latch, score 0; completing the patch afterwards still leaves
    success False (the pass must come after).
18. Tolerance twins: modestly off-centre keystone still interred; off-centre paver still
    covers.
19. Latch is a latch: 0.50 survives the top paver being knocked back off.
20. All states finite.

## Checks

- solve: forge `SIM_GEN_SOLVE: SUCCESS`, ≥ 2 seeds, score trace
  0.000 → 0.150 → 0.500 → 1.000 → 1.000, monotone.
- smoke: forge `SIM_GEN_SMOKE: ALL PASS 20/20`, rc 0, frames.npz written.
