# pan_cubby_retrieval — slide the frying pan OUT of the cabinet, set it on the burner

**Seed:** `libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf.py`)
**Tier:** easy — **2 stages**. **Execution order: REQUIRED** and topologically forced
(extract before place — the pan cannot reach the burner without first leaving the cubby
through its one open face; back and sides are closed and the roof is too low to lift over).
**Env name:** `simgen.pan_cubby_retrieval` (scene `pan_cubby_retrieval`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed asks for a free **pick-and-place**: the `chefmate_8_frypan` sits in the open on
the counter, is grasped, carried through the air, and placed **onto** the top region of a
two-layer shelf (bounding-box check on the shelf's `top_region` site; the flat stove is a
distractor).

Here the spatial relationship and the required plan are **inverted**, not
re-parameterized:

- The pan starts **inside** a closed-back cabinet cubby whose roof leaves only ~29 mm of
  head room above the 36 mm pan. A vertical lift — the seed's opening move — is
  **geometrically impossible**: the pan physically jams on the roof (verified by the
  velocity-kick probe, smoke #7: a 1 m/s upward kick rises < 4 cm). The only feasible
  plan is to **slide the pan horizontally** out through the single open face, and only
  then lift it.
- The seed's goal fixture and its distractor **swap roles**: the shelf/cabinet is now the
  obstacle the pan must escape, and the stove burner (a distractor in the seed) is the
  placement target. Placing the pan **on top of the cabinet** — the seed's literal goal —
  is a tested negative control (smoke #6): it never counts as extracted (footprint test)
  and scores ≤ 0.3 with no success.
- Success is judged on physical outcome only: pan settled flat ON the burner pedestal
  (centre within 5 cm of the axis, base at the burner top ±2 cm, upright within 15°,
  |v| < 5 cm/s) and fully outside the cubby footprint.

A solver therefore needs a different plan (confined horizontal retrieval under a ceiling
constraint, then a precision placement on a raised pedestal) and different code structure
(a guarded drag/slide stage feeding a place stage — not a grasp/lift/lower pipeline), not
different numbers.

### Relation to sibling tasks from the same LIBERO scene

Sibling `..._put_the_frying_pan_under_the_cabinet_shelf_i4` (tasks_v4 scene
`pan_lowshelf_slide`) also uses a low-clearance alcove, but with the **opposite goal
topology and a different skill**: there the pan starts in the open and must be pushed
non-prehensilely INTO the confined space, which is the terminal state (1 stage,
insertion, nothing else is touched). Here the confinement is only the *starting*
obstacle: the pan must come OUT of it (retrieval), and the deliverable is a two-stage
retrieve-then-place onto a separate raised target with placement tolerances (centring,
height, uprightness, settling) — plus the goal/distractor role swap of the seed's
fixtures. The plans do not transfer: i4's push-in policy never leaves the cabinet region
and never touches a placement target; i9's policy must exit it and finish a precision
set-down elsewhere.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects; they never do the task. Both load-bearing interactions go
through contact dynamics. Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing —
all credit is latched):

- **P0** reset (seeded), settle 0.5 s, print baseline (~0) + layout readback (cubby pose
  + yaw, pan spawn, burner xy — seed provenance in stdout).
- **P1 EXTRACTION (contact dynamics — no teleport is possible here without bypassing the
  task)**: a horizontal external force at the pan's CoM (world frame, aligned with the
  cubby's mouth axis; velocity-regulated bang-bang: 5 N while axial speed < 0.10 m/s,
  +2 N per 2 s stall up to 12 N; ±1.2 N lateral centring while inside) drags the pan
  across the cubby floor and out the mouth through real friction/roof/wall contacts. The
  force is CUT only once the pan centre fully clears the cubby footprint (x_local >
  x_exit); the pan coasts to rest on friction. Asserts `extracted` before proceeding.
- **P2 TRANSPORT (teleport, the only pan pose write)**: one root-state write carries the
  now-free pan across open ground to a release pose with its base **28 mm above** the
  burner top — outside the 20 mm on-burner z-tolerance, so the freshly-teleported state
  does not satisfy `success()` — orientation kept exactly as it coasted to rest,
  velocities zeroed, never seated on the target.
- **P3 SET-DOWN (contact dynamics)**: gravity drops the pan the last 28 mm; it impacts
  the kinematic pedestal, beds down and settles for 1.25 s → `success()` first turns
  True here, judged on the settled pose.
- **P4 persistence**: 3.3 more simulated seconds with no intervention; only if
  `success()` held prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(0.10, -0.48, 0.0)`, facing +y (the cubby mouth region and the burner
band both in front). Working radii from this base: handle grasp region at the mouth
0.49–0.58 m, extraction end ~0.49 m, burner samples 0.34–0.68 m — all inside the proven
0.30–0.71 m ground-level comfort envelope.

**Pan — stage 1 (extraction): pinch the protruding handle and drag.** The spawn-depth
band (60–160 mm behind the mouth plane) plus the ±30° handle yaw keeps the **handle tip
between 4 mm behind and 120 mm proud of the mouth plane** for every sample, i.e. always
within fingertip reach from OUTSIDE the compartment (Franka fingers reach ~54 mm past
the palm face; the hand itself never has to enter the 65 mm-tall opening). Contact
strategy: hand horizontal, pinch the handle's **sides** (22 mm wide across an 80 mm jaw,
12 mm thick, top face at ~32 mm — 33 mm of free space below the roof edge at the mouth),
then drag outward along the mouth axis, pan sliding flat on its base exactly as the
solve's force does. Required precision: the mouth is 300 mm wide vs the 150 mm disc —
±7 cm of lateral slack; no alignment finer than ~2 cm is ever needed.

**Pan — stage 2 (placement): standard top pinch on the handle, lift, set down.** Once
the pan is in the open the handle is a canonical Franka grasp (22×12 mm bar at 20–32 mm
height, approach from above, jaw closes across the width). Set-down tolerances are wide:
5 cm xy on a 20 cm-diameter pedestal, 15° tilt, 2 cm height — far above closed-loop OSC
noise. Holding the handle keeps the hand ~10 cm outside the burner footprint, so there
is no clearance conflict at release; the burner top is at a comfortable 50 mm height.

**Cubby, burner:** kinematic fixtures, never need to be touched.

No other objects exist; every required contact is one the arm can make.

## Success and rubric (physical outcomes only)

Judged with the cubby's body frame for extraction (its yaw is randomized). `success()`
iff the pan is settled flat ON the burner: centre within `place_xy_tol = 5 cm` of the
burner axis, base at the burner top within ±2 cm, upright within 15°, |v| < 5 cm/s, and
the pan centre fully outside the cubby footprint. `score()` ∈ [0,1], latched each physics
substep: `0.35·extraction-progress + 0.15·extracted + 0.35·approach-to-burner (gated on
extracted)`, capped 0.85; **1.0 iff success()**; ~0 for doing nothing (extraction
progress is normalized by the episode's own spawn depth). A pan on the cabinet ROOF is
horizontally inside the footprint: it earns no extraction credit and keeps the burner
stages locked. Latched credit never evaporates (smoke #12); the solve's phase prints are
monotone.

## Randomization

Per episode (verified by sim READBACK in the smoke): cubby yaw ±10° + xy jitter ±3 cm;
pan spawn depth 60–160 mm behind the mouth plane + lateral ±4 cm + handle yaw ±30°;
burner x ±5 cm, y ±15 cm (a 30 cm band).

## Check list (smoke.py — rubric REJECTION battery, 14 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite, pan at rest on the cubby floor, inside the compartment
2. settle: score ~0 at reset, no success
3. randomization readback: pan spawn depth + lateral + handle yaw vary
4. randomization readback: cubby yaw + burner xy vary
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: pan settled ON TOP of the cabinet — not extracted, no success,
   score ≤ 0.3
7. roof-block probe: 1 m/s upward velocity kick inside the cubby rises < 4 cm — the
   lift-first plan is physically jammed, not just unscored
8. near-miss: pan settled flat ON the burner but 6.5 cm off its axis (tol 5 cm) → NOT
   success, score < 0.9 (load-bearing xy tolerance)
9. wrong place: pan settled upright on the GROUND beside the burner → NOT success
   (z gate — correct-looking pose on the wrong surface)
10. flipped: pan upside-down centred on the burner → upright gate rejects, score ≤ 0.86
    (the 0.85 non-success cap), NOT success
11. monotonicity: fully-out probe latches strictly more extraction credit (and score)
    than a mouth-straddle probe
12. latched credit survives teleporting the pan back inside (score unchanged, no success)
13. rejection audit: success() never True at any judged point of this battery
14. final no-NaN

N/A notes: **wrong object** — the scene has a single movable object (the pan), so a
wrong-object control is not expressible; identity is carried by the goal surface controls
(#8/#9/#10) instead. **Out-of-order end state** — the required order is topologically
forced (a pan on the burner that never left the cubby is not constructible: the burner is
outside the footprint, so being on it implies extraction happened); the order controls
are #6/#7, which pin the pan's only alternative exit routes shut. A settled-but-tilted
pan INSIDE the cubby is not constructible (29 mm head room < any ≥15° pose), so the
tilt gate is exercised by the flipped-on-burner control (#10).

Video frames are recorded throughout and saved to `frames.npz` in the working directory.
