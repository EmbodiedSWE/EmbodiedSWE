# pan_lowshelf_slide — slide the frying pan under a low-clearance shelf

**Seed:** `libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf.py`)
**Tier:** easy — **1 stage**, single skill. **Execution order:** none required (one goal,
no sequencing).
**Env name:** `simgen.pan_lowshelf_slide` (scene `pan_lowshelf_slide`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed asks for a free **pick-and-place**: grasp the `chefmate_8_frypan`, carry it
through the air, and lower/insert it into the open bottom compartment of a two-layer
shelf; success is a bounding-box check on the pan's position in the shelf's
`bottom_region` site. The compartment has generous headroom, so the demonstrated plan is
grasp → transport in air → place.

Here the goal region is a **low alcove** (roof + two side walls + back wall, open only at
the front) whose roof underside is only `h_gap = 48 mm` above the table — ~18 mm above
the 30 mm pan body (the margin is asserted in the scene config,
`0.008 ≤ h_gap − pan_h ≤ 0.035`). All geometry is procedural (compound spawners, no asset
files); a white bowl distractor (the seed's `white_bowl`) sits nearby and also *fits*
under the shelf, so object identity — not fit — is what rejects it.

## Why strategically different

The seed's plan is **prehensile transport**: lift the pan and lower it into the goal
region. That plan is **physically impossible** here — the goal footprint is fully covered
by a roof, so the pan cannot descend into it from above, and the 18 mm of headroom leaves
no room to hold the pan from the top once under. The only working plan is
**non-prehensile planar manipulation**: keep the pan flat on the table and push/slide it
(naturally by its handle) through the front opening until the whole disc is under the
roof. A solver needs a different plan (push vs pick-and-place) and different code
structure (a regulated planar push with a guarded approach line, not a grasp/lift/lower
pipeline), not different parameters. The seed's own strategy is expressible and is smoke
control #6: lowering the pan from above lands it ON TOP of the roof; success stays false
and score ≤ 0.25 (the z-gate excludes on-roof poses from all insertion credit).

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move the pan; they never do the task. Phases (each boundary prints
`SIM_GEN_SCORE`, non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, print baseline (~0) + layout readback (seed
  provenance in stdout).
- **P1 TRANSPORT (teleport)**: one pose write carries the pan across open table to a
  staging spot on the alcove's axis, disc leading edge ~60 mm in FRONT of the opening,
  flat on the table, handle pointing away from the shelf. The pan is still entirely
  outside the goal; no required interaction is bypassed.
- **P2 PUSH (contact dynamics)**: a horizontal external force at the pan's CoM
  (world-frame, velocity-regulated bang-bang: 5 N while axial speed < 0.08 m/s, +2 N
  per 2 s stall up to 12 N; ±1.2 N lateral centering until the side walls take over)
  slides the pan under the 48 mm roof through real friction/wall/roof contacts. The
  force is CUT the moment the disc centre crosses the alcove midline — well before the
  back wall — and the pan coasts out on friction.
- **P3** hands off, settle 1 s → `success()` (disc fully in, flat, settled).
- **P4 persistence**: 3.3 more simulated seconds with no intervention; only if
  `success()` held prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

- **Base pose:** `(0.0, -0.60, 0.0)`, facing +y (toward the alcove opening). Working
  radii from this base: pan spawn ~0.38 m, staging line 0.30–0.45 m, final handle-tail
  contact ~0.58 m — all inside the proven 0.30–0.71 m ground-level comfort envelope.
- **Pan (the one manipulated object):** non-prehensile PUSH; no grasp is required
  anywhere. Contact strategy: closed-fingertip push on the pan's trailing geometry —
  the disc rim (30 mm tall) while the disc is still outside, then the **handle** (24 mm
  wide, top at ~21 mm, protruding out the front for the entire final stroke because it
  points away from the back wall). The pusher never has to enter the alcove: when the
  disc is fully in, the handle still sticks out ~180 mm past the front plane, and the
  roof front edge is 48 mm high while the fingertip contact happens at 10–25 mm height,
  outside the roof footprint. Precision demanded: the depth success band is 56 mm wide
  (±28 mm) and the walls guarantee the lateral band by construction (max wall-hugging
  offset 30 mm < 38 mm tolerance) — far above arm control noise; pushing to back-wall
  contact is itself inside the band, so the stop is physical.
- **Alignment:** the approach line only needs the disc centred within ±3 cm at the
  95 mm-wide-to-disc doorway (220 mm opening vs 160 mm disc); a straight push from the
  staging pose achieves this and the walls funnel the rest.
- **Bowl (distractor):** never needs to be touched; it spawns 0.3 m off the slide
  corridor.

## Success and rubric (physical outcomes only)

Judged in the shelf's body frame (its yaw is randomized). `success()` iff the pan
**disc** (handle excluded — it may stick out the front) is fully inside the alcove
footprint with 8 mm slack, the pan is flat (≤10° tilt, base within 12 mm of the table),
UNDER the roof (z-gate), and settled (<0.05 m/s). `score()` ∈ [0,1], latched each physics
substep: `0.2·best-approach + 0.6·best-insertion-fraction` (insertion credit only while
under the roof, laterally inside, and roughly flat), capped 0.8; 0.9 once fully-in +
flat; **1.0 iff success**; ~0 for doing nothing (approach is normalized by the episode's
own spawn distance). Latched credit never evaporates (smoke #11); the solve's phase
prints are monotone.

## Randomization

Per episode: shelf xy jitter ±3 cm + yaw ±10°, pan spawn xy jitter ±5 cm + free yaw
(handle direction random), bowl xy jitter ±3 cm. Verified by sim READBACK in the smoke.

## Check list (smoke.py — rubric REJECTION battery, 13 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite, pan at rest flat on the table
2. settle: score ~0 at reset, no success
3. randomization readback: pan spawn xy + yaw vary
4. randomization readback: shelf xy + yaw vary
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: pan lowered from above the goal footprint lands ON the roof —
   no insertion credit, no success, score ≤ 0.25
7. wrong object: the bowl settled fully inside the alcove scores ~0, no success
8. near-miss: pan settled under the roof 17 mm short of the fully-in band → NOT success,
   score < 0.9 (load-bearing depth tolerance)
9. straddle: pan settled half in the doorway → partial credit strictly in (0, 0.9), NOT
   success
10. monotonicity: deeper probe latched strictly more insertion credit
11. latched credit survives pulling the pan back out (score unchanged, no success)
12. rejection audit: success() never True at any judged point of this battery
13. final no-NaN

Near-miss N/A notes: a settled-but-TILTED pan inside the alcove is not constructible —
the 48 mm gap physically excludes any pose beyond the 10° flat gate (a tilted disc is
taller than the roof clearance), so the flat-gate rejection is enforced by construction
and exercised indirectly by the on-roof control (#6). Execution-order controls: N/A —
single-stage task, no order declared.

Video frames are recorded throughout and saved to `frames.npz` in the working directory.
