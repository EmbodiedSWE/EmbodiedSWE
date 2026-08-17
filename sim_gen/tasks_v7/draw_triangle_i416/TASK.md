# draw_triangle_i416 — `tray_poise`

A slender pedestal carries an orange square cap only 5.6 cm across — the only
legal support in the scene. A serving tray loaded with two dark slugs of the
same material but different sizes lies on the floor. Balance the LOADED tray
level on the cap: the only point it stands on is directly under the ensemble
centre of mass, which the visible slug sizes shift 3.8–5.6 cm off the tray's
geometric centre. Then, with the balance live, set two golden cubes down on the
deck without capsizing what you built — and leave everything at rest.

- Scene id: `tray_poise` · Env id: `simgen.tray_poise` · Robot: `null`
- Files: `scene.py` (procedural geometry only), `solve.py`, `smoke.py`, this file.

## Seed provenance and why this is strategically different

Seed: **maniskill/draw_triangle** — the Franka holds a rigid stylus and traces a
prescribed triangle outline on a passive canvas; one long guarded tool-tip
sweep, judged on the coverage of the trace, and nothing in the scene ever
pushes back.

| | seed (draw_triangle) | this task (tray_poise) |
|---|---|---|
| goal | a prescribed curve, fully given | an **equilibrium the solver must compute**: the mount point is the ensemble CoM, derivable only by combining the visible slug sizes/pockets into a mass-weighted sum |
| scene response | passive canvas | **live statics** — a 28 mm cap half-width against a 38–56 mm CoM offset: every naive placement capsizes |
| plan structure | waypoint tracking | compute → place-at-a-point → then **keep the equilibrium alive** while adding 0.56 kg of new load symmetrically |
| error signal | path deviation | the tray itself: ≥ 8 mm asserted CoM overhang tips a centre mount off the pedestal |

Against the tasks_v7 corpus: i107 `beam_balance` is the nearest neighbour and
the contrast is deliberate — there a **hidden** mass is **measured** through an
articulated beam's response; here **nothing is hidden** and there is **no
articulation**: the solver must *compute* a free-body balance point from visible
geometry (size is the mass cue), and — unlike every containment / transport /
mechanism task in the corpus (i1 shape_sorter, i8 bar_triangle, i77
skyway_bridge, pen_holder...) — the achieved goal state stays **fragile during
the rest of the task**: serving the cubes is manipulation performed *on top of*
a live balance that a careless drop provably destroys (smoke check 12).

## Success criteria (all judged live, simultaneously)

- **A. MOUNTED** — tray within `level_tol_deg = 5°` of level, tray centre
  within `mount_z_tol = 6 mm` of `mount_z = 0.138`, and the cap axis inside the
  tray-frame footprint (|x| < 0.125, |y| < 0.075). Standing on the cap is the
  only way to satisfy the height band while level.
- **B. SLUGS HOME** — each slug seated in **its assigned pocket** (tray-frame,
  16 mm xy / 9 mm z). Rejects rearranging the slugs to make a more convenient
  balance (flagship smoke check 9: a *physically perfect* swapped-slug balance
  earns nothing).
- **C. SERVED ×2** — each golden cube resting on the deck: tray-frame footprint
  ∩ z ∈ [0.012, 0.10]. A cube on the ground under/beside the tray never
  qualifies (tray-frame z), and serve credit is **gated on the mounted state**
  (check 6: cubes on the grounded tray's deck score zero).
- **D. STILL** — all five movables < `settle_speed = 0.10 m/s`, tray angular
  speed < `settle_omega = 0.50 rad/s`.

**Score (latched, non-decreasing):** `0.35·mounted_ever + 0.15·each
served_ever`, all streak-latched (24 consecutive substeps = 0.2 s of
mounted∧home∧still — a tray sweeping through level, or a cube bouncing across
the deck, is fast at the moment it looks right and cannot latch), capped at
0.65; exactly 1.0 iff `success()` holds now. Null policy ≈ 0 (check 4).

## Per-seed randomization (readback-verifiable via `describe()`)

Slug loadout: heavy slug in one of the two OUTER pockets (`side = ±1` via
`torch.rand` comparison), light slug in the centre or the opposite outer pocket
— four loadouts, CoM offsets {±38.2, ±55.5} mm, both signs exercised. Pedestal
xy ±4 cm + free yaw; tray spawn xy ±3 cm + free yaw; cube staging xy ±3 cm.
All discrete draws use `torch.rand` comparisons (first-`randint` degeneracy on
this stack). Smoke check 3 verifies spreads and that the OBSERVED slug pockets
match the assignment over 8 seeds.

## Teleport-solution outline (`solve.py`)

Teleportation is transport only; the mount and both serves are free releases
settled by contact dynamics:

1. Settle; read the loadout from the OBSERVED slug tray-frame positions
   (nearest pocket), cross-check vs the assignment, and compute
   `x_com = (m_h·hx + m_l·lx) / (m_tray + m_h + m_l)`.
2. Mount: teleport the loaded tray (slugs seated, tray yaw ALIGNED to the
   pedestal yaw) to hover 10 mm above the cap with the cap axis under
   tray-frame `(x_com, 0)`; release; the drop and the stand are pure physics.
   On a failed mount, honestly recollect everything to the floor and retry
   (≤ 3 attempts).
3. Serve: teleport BOTH cubes simultaneously to 12 mm above the deck at
   tray-frame `(x_com, ±serve_y)` in the LIVE tray pose — symmetric about the
   cap axis, so the roll impulses cancel and the ensemble CoM stays on the cap.
4. Hold ≥ 3.5 simulated seconds fully hands-off, printing `SIM_GEN_SCORE` at
   each phase boundary (non-decreasing, latched) and `SIM_GEN_SOLVE: SUCCESS`
   only if success still holds at the end.

## Franka embodiment argument

- **Tray**: 180 g; the 6 mm skids lift the 12 mm slab so its long edges
  overhang free — a parallel-jaw pinch on the slab edge (12 mm < 80 mm jaw
  span). Loaded mass 770 g, far under payload. The mount is a coarse
  set-down: the tray stands as long as the ensemble CoM lands inside the 56 mm
  cap, a ±~20 mm placement budget around the computed point (the near-miss
  check shows 40 mm off fails; the task is precision-of-*reasoning*, not
  precision-of-motion).
- **Slugs**: never need moving (they spawn seated); if regrasped, 44/30 mm
  blocks under the jaw span.
- **Cubes**: 35 mm / 280 g pinch grasps from open floor slots; serving is a
  gentle drop into a lane whose clearance to the pocket walls and footprint
  edge is asserted in `__post_init__`. Serving one cube at a time is fine too —
  each single cube must go down near the cap axis (the far-end drop capsizes),
  a legible constraint from the visible geometry.
- Everything sits on open floor within a 0.44–0.60 m fan of a base at
  env-local **(−0.45, 0.0, 0), facing +x**: pedestal (0.08, 0.15)±4 cm, tray
  spawn (0.08, −0.18), cube slots (−0.16, ±0.34); mount height 0.144 m — all
  inside Franka's 0.85 m reach.

## Execution order

**No execution order is required by the rubric.** Mount-then-serve (the
demonstrated solve) and serve-then-mount (mount the fully-laden tray at the
combined CoM — the symmetric cube lanes keep it at `x_com`) are both legal;
only the final joint state is judged, plus latched partial credit along the
way. One order IS physically forced and deliberately so: the slugs must ride
the tray from the start — adding either single slug to an already-mounted tray
shifts the CoM off the cap with a ≥ 1.3× asserted margin (sequential loading
on the pedestal always capsizes).

## Verification

- `smoke.py`: **16 checks**, rejection-only battery — (1) settle premise
  (finite, tray flat on skids, slugs in DISTINCT pockets by readback, cubes on
  floor, still); (2) reset score ≈ 0; (3a/3b) randomization by readback over 8
  seeds (pedestal xy/yaw spreads; heavy slug on BOTH sides, ≥ 2 loadouts,
  observed pockets match assignment, tray/cube jitter); (4) null policy 240
  steps → 0; (5) state roundtrip restores poses + loadout + score; (6)
  ground-serve bypass: cubes on the grounded tray's deck read in-band but earn
  nothing (serve gated on mounted); (7) naive centre mount capsizes; (8)
  near-miss mount (cap 40 mm past the CoM) capsizes; (9) FLAGSHIP: slugs
  swapped and mounted at the swapped balance point — stands level, at height,
  still (readback proves a genuine equilibrium) yet earns nothing (assigned-
  pocket identity is load-bearing); (10) correct mount earns exactly 0.35, not
  success; (11) cube on the ground inside the mounted tray's footprint is not
  served; (12) far-end cube drop tips the live balance out of the level band
  (readback transient) and capsizes or sheds the cube — serve never latches,
  mount credit survives latched; (13) fresh reset clears to ≈ 0; (14) success
  never observed anywhere in the battery; (15) no NaN/Inf; frames.npz recorded.
  Prints `SIM_GEN_SMOKE: ALL PASS 16/16`.
- `solve.py` on the forge, seeds 0/1/2: `SIM_GEN_SOLVE: SUCCESS` on all three
  (x_com = −55.5, +38.2, +38.2 mm; both CoM signs and both magnitudes
  exercised; all first-attempt mounts), score trajectory 0 → 0.35 → 1.0,
  non-decreasing, ≥ 3.5 s hands-off persistence.

## Design notes (physics traps avoided)

- The tray root is the slab centre, so MassAPI-only mass puts the body CoM
  exactly at the geometric centre — the ensemble offset then comes *only* from
  the slugs, which is what the interlock arithmetic (`__post_init__`, all
  asserted with margins) assumes: every loadout's CoM overhangs the cap by
  ≥ 8 mm (centre mount tips), the extreme mount point keeps the cap inside the
  footprint, serve lanes clear the pocket walls, cap clears the skids at any
  relative yaw.
- Flat-on-flat cap contact sustains a PhysX contact-injected velocity limit
  cycle that body damping cannot kill (forge-measured ~0.074 m/s slug lin,
  ~0.27 rad/s tray ang after raising tray damping to 0.20/3.0): the stillness
  gates (0.10 / 0.50) sit above that artifact floor, while rejection stays
  sharp — a capsizing tray passes 0.5 rad/s within ~3 substeps and leaves the
  5° level band well inside the 24-substep streak, so fly-through cannot latch
  (i107 precedent).
- A tray mounted with its yaw MISALIGNED to the pedestal yaw rings a larger
  edge-contact limit cycle than the aligned square-on-square case; the solve
  and all smoke mount probes align tray yaw to the pedestal yaw.
- Stillness is streak-latched in `post_step`, never an instantaneous velocity
  test (turning-point trap).
- The far-end serve rejection is stated as the physics actually resolves it:
  the 0.28 kg cube against the 0.77 kg ensemble moves the CoM ~10 mm past the
  cap edge, so the tray tips and either capsizes or sheds the free cube over
  the nearby deck end and re-rights — either way no level served state exists
  there, and the smoke check asserts the tilt transient by readback so the
  probe can never pass vacuously.
