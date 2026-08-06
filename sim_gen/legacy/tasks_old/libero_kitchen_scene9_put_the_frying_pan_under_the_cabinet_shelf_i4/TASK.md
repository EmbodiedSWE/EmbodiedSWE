# pan_lowshelf_slide — slide the frying pan under a low-clearance shelf

**Seed:** `libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf.py`)
**Tier:** easy — **1 stage**, single skill. **Execution order:** none required (one goal, no sequencing).
**Env name:** `sim_gen.pan_lowshelf_slide` (scene `pan_lowshelf_slide`, robot `null`).

## What changed vs the seed

The seed asks for a free **pick-and-place**: grasp the `chefmate_8_frypan`, carry it through
the air, and lower/insert it into the open bottom compartment of a two-layer shelf; success is
a bounding-box check on the pan's position in the shelf's `bottom_region` site. The compartment
has generous headroom, so the demonstrated plan is grasp → transport in air → place.

Here the goal region is a **low alcove** (roof + two side walls + back wall, open only at the
front) whose roof underside is only `h_gap = 48 mm` above the table — ~18 mm above the 30 mm
pan body (the margin is asserted in the scene config, `0.008 ≤ h_gap − pan_h ≤ 0.035`). All
geometry is procedural (compound spawners, no asset files); a white bowl distractor (the seed's
`white_bowl`) sits nearby and also *fits* under the shelf, so object identity — not fit — is
what rejects it.

## Why strategically different

The seed's plan is **prehensile transport**: lift the pan and lower it into the goal region.
That plan is **physically impossible** here — the goal footprint is fully covered by a roof, so
the pan cannot descend into it from above, and the 18 mm of headroom leaves no room to hold the
pan from the top once under. The only working plan is **non-prehensile planar manipulation**:
keep the pan flat on the table and push/slide it (naturally by its handle) through the front
opening until the whole disc is under the roof. A solver needs a different plan (push vs
pick-and-place), not different parameters. The seed's own strategy is expressible and is smoke
negative-control A: lowering the pan from above lands it ON TOP of the roof, success stays
false and score ≤ 0.2 (the z-gate excludes on-roof poses from all insertion credit).

## Success and rubric (physical outcomes only)

Judged in the shelf's body frame (its yaw is randomized). `success()` iff the pan **disc**
(handle excluded — it may stick out the front) is fully inside the alcove footprint with 8 mm
slack, the pan is flat (≤10° tilt, base within 12 mm of the table), UNDER the roof (z-gate),
and settled (<0.05 m/s). `score()` ∈ [0,1], latched each physics step: `0.2·best-approach +
0.6·best-insertion-fraction` (insertion credit only while under the roof, laterally inside, and
roughly flat), capped 0.8; 0.9 once fully-in + flat; **1.0 iff success**; ~0 for doing nothing
(approach is normalized by the episode's own spawn distance). The oracle only teleport-*slides*
along the table (never lifts), then releases and lets real physics settle before judging.

## Randomization

Per episode: shelf xy jitter ±3 cm + yaw ±10°, pan spawn xy jitter ±5 cm + free yaw (handle
direction random), bowl xy jitter ±3 cm. Verified by sim readback in the smoke.

## Smoke checks (13)

1. settle: reset is clean (finite, still, score ~0)
2. randomization is real (pan pose + shelf yaw vary across resets — readback)
3. null policy: score stays ~0
4. oracle succeeds (seed 0) — teleport slide-in, release, settle
5. oracle succeeds (seed 1)
6. oracle succeeds (seed 2)
7. rubric: score monotone with insertion depth (fresh reset per depth)
8. rubric: partial insertion scores strictly between 0 and 1
9. near-miss: 90%-inserted pan is NOT success (tolerance control)
10. calibration: measured success onset matches predicted fraction `1 − edge_tol/(2·pan_r)` ≈ 0.95
11. negative A (seed strategy): lift-and-lower from above is blocked by the roof (lands on top, ~0 credit)
12. negative B (wrong object): the bowl slid fully under the shelf scores nothing
13. latch: partial credit survives pulling the pan back out

Video frames are recorded throughout and saved to `frames.npz` in the working directory.
