# libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i429 — ratchet the portcullis open, slide the skillet INSIDE

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet`
— a Franka in the LIBERO kitchen grasps the frying pan, lifts it high, and sets
it down ON TOP of the cabinet. Same protagonist object (a flat skillet), same
furniture archetype (a closed kitchen cabinet) — and a deliberately inverted
goal geometry and skill.

## Strategic difference

**vs. the seed.** The seed's entire strategy is *pick up high and place on the
top surface*. Here the top surface is worth exactly nothing: the hutch roof is
sealed, already occupied by a bystander bowl that must not move, and a smoke
check constructs the pan settled ON the roof and asserts score ≈ 0. The goal is
*inside* the hutch, at floor level, on a stove pad reachable only through a low
doorway — and that doorway is barred by a PORTCULLIS: a gravity-closed gate on a
vertical slide carrying a rack of teeth that runs under a gravity pawl (a
one-way ratchet). So the seed's one grasp-lift-place is replaced by an ordered
two-mechanism plan: (1) ratchet the gate open by its red lift bar — each lift
stroke is *kept hands-off* by the pawl, which is the mechanism's point: it
solves the single-arm "third hand" problem, freeing the arm for the pan; (2)
slide the skillet flat along the floor, through the doorway (handle trailing —
the doorway is narrower than the pan is long), onto the pad. The pan is never
lifted at all; placing it on top of anything scores zero. Delivery is certified
by a physically-earned transit credential (the pan observed crossing the
doorway slab *while* the gate is raised), so the crossing itself — not just the
end pose — is judged.

**vs. the rest of the corpus.** No other `tasks_v7/` package uses a
rack-and-pawl ratchet or a portcullis-barred aperture (grep: no scene authors a
pawl or ratchet mechanism; matches elsewhere are prose coincidences). The
corpus's closest neighbours read during construction: `..._i87` (rocker
transmission — continuous, force-through-linkage, no retained state) and the
drawer/lid families (bodies that stay where friction leaves them). The ratchet
is different in kind: it is a *state-retaining one-way mechanism* — lifting is
free, lowering is forbidden, and the mechanism itself holds the robot's work
product hands-off (verified by a real 2×-weight pull-down probe in smoke). The
tasks that put objects on top of cabinets (`..._i308`, `..._i5`) are exactly
what this task's rubric rejects.

## Scene (`ratchet_portcullis`, env `simgen.ratchet_portcullis`, robot="null")

Procedural geometry only. A kinematic roofed HUTCH (front face x=0, +x toward
the robot; sides, back, lintel, piers, full roof sealed; the only opening a
20 cm × 13 cm floor-level doorway; a visual-only flush stove pad centred inside
at x=−0.24) plus three dynamic bodies:

- **Gate** (0.30 kg, prismatic z, limits [0, 0.175] m): panel sealing the
  doorway (hovers 5 mm off the ground), red lift bar at z 0.058–0.078, and a
  5-tooth rack (pitch 28 mm) on its front face, offset to the +y side.
- **Pawl** (0.05 kg, revolute y, limits [−3°, +75°]): a small flap whose hinge
  sits 17 mm heel-ward of its CoM (root at flap centre + MassAPI CoM-at-origin
  ⇒ gravity droops it onto its −3° jam stop). Teeth click past it rising; it
  jams a tooth falling. Mounted on a SIDE cheek plate offset +y with 1 mm
  clearance — the flap's whole swept disc must intersect nothing: a jointed
  pair that interpenetrates at spawn rigidly locks the GPU revolute despite
  the collision filter (that exact bug cost the first forge iteration).
- **Skillet** (0.40 kg, free): 14 cm disc, 2.2 cm tall, straight handle to
  18.5 cm. Spawns outside at seeded (x∈[0.26,0.34], y±0.08, yaw±0.30 —
  yaw bounded so the handle can trail through the doorway).
- **Bowl** (0.15 kg, free): bystander ON the roof, seeded xy.

Geometry contract (all asserted in `__post_init__`, pure math, verified locally
before any forge run): pawl-held hold ladder [0.113, 0.141, 0.169] m — every
hold clears both the `opened` threshold (0.080) and the pan's transit clearance
(0.060) with margin; release from the 0.175 stop falls a few mm onto a real
catch; at the stop the pawl sits inside a tooth gap ≥4 mm both sides; the pawl's
swept disc stays ≥2 mm in front of the hutch face plane; the doorway passes the
pan at every sampled yaw with ≥15 mm slack; the pan fits the interior at the
pad; rack/pawl carry a polished (μ≈0.1, min-combine) material so teeth click
instead of dragging.

**Success** (live state, no memory of how except the credentials):
`_transit ∧ on_pad ∧ settled ∧ finite`. `on_pad` = pan root within ±6 cm x /
±5 cm y of the pad, z in [−5, 25] mm, upright ≥ 0.90. `settled` = all four
movers' CoM speed < 0.05 m/s (the pawl judged by CoM linear velocity — its
loaded jam-stop rest carries a steady ~0.94 rad/s GPU resting-contact velocity
bias that never decays while nothing moves; an angular gate could never pass,
a genuinely swinging pawl still trips the linear gate at 2–5× threshold).
**Score**: latched `opened` 0.20 (lift ≥ 0.080) + `propped` 0.20 (60
consecutive substeps open & still, i.e. the pawl provably carries the gate
hands-off) + `transit` 0.25 (pan root observed in the doorway slab x∈[−0.09,0],
|y|<0.10, z<0.10 *while* lift ≥ 0.060), capped 0.65; exactly 1.0 iff success.
Latches reset on `reset()`.

## Solution (`solve.py`) — applied force only, zero teleports

Teleports are permitted for transport only; this solve needs none.

1. **P0 settle + perception**: 120 steps; read back the pan's sampled pose and
   the bowl's roof pose from the live state (never hard-coded); baseline ≈ 0.
2. **P1 ratchet the gate** (≤12 N clamped PD on the gate, the hand on the red
   bar): lift to the 0.175 stop — teeth click past the pawl, real contact —
   hold half a second (the last tooth flicks the pawl; let it re-seat), then
   *ease the gate down* onto a catch (dropping it from the stop is a race the
   falling gate wins, skipping catches — a real ratchet quirk; the eased
   set-down is exactly what a human hand does). Forces off 1.25 s: the pawl
   carries the gate at a hold ≥ 0.105, `opened`+`propped` latch. Score 0.40.
3. **P2 transit** (≤8 N then ≤6 N PD floor-drag on the pan, fingers on the
   rim): align to the doorway centreline, slide through under the raised gate
   — the transit credential latches on the real crossing. Score 0.65.
4. **P3 deliver**: drag onto the pad (≤5 N), forces off, everything settles;
   `success()` turns True on the live state (score 1.0), then holds a **3.3 s
   hands-off persistence** window — during which the bowl is asserted still on
   the roof, unmoved, and the pawl still holding the gate — before
   `SIM_GEN_SOLVE: SUCCESS`.

The whole episode is repeated on a second seed (fresh reset, no score prints —
the rubric restarts at 0 and the printed SIM_GEN_SCORE stream must stay
non-decreasing) to certify seed robustness. Forge result: both seeds OK, rc=0
(~29 s).

## Franka embodiment argument

Base pose: on the floor ~0.55 m in front of the hutch (face at x=0), facing −x.
Single arm + parallel jaw via OSC.

- **Gate lift**: the red bar (32 mm deep × 100 mm wide × 20 mm tall) protrudes
  in front of the panel at z 0.06–0.08 (+lift) — an easy pinch or under-hook at
  fingertip height. Total effort ≤ 12 N vertical, under the Franka's payload;
  and because the ratchet keeps every stroke, the arm may lift in short
  re-grip strokes — no second hand, no holding while reaching for the pan.
- **Skillet slide**: push the disc rim or pinch the handle (22 mm wide, 16 mm
  tall — inside the jaw span) and slide at ≤8 N along the floor. The final
  push leaves the handle tip only ~5 cm behind the face plane: the last
  increment is fingers-through-the-doorway (20 cm wide × 13 cm tall, ample for
  the ~4 cm-tall extended fingertips at floor level), or a nudge on the handle
  end from outside.
- **Bowl (bystander)**: on the roof at 0.26 m — reachable but the task is to
  leave it alone.

## Execution order declaration

Scene first with the minimal success predicate; `solve.py` then iterated on the
forge until the task was *physically* solved; rubric weights anchored to the
observed trajectory; `smoke.py` written last. Forge iterations fixed physics
and probe bugs, never weakened the rubric: (1) the overhead pawl bracket
interpenetrated the flap by 0.5 mm at spawn → GPU revolute rigidly locked →
re-authored as a y-offset side cheek (geometry fix + new cfg assert); (2) the
released gate outran the pawl's re-seat and skipped the top catch → the solve
eases the gate down instead (better physics, same rubric; the solve's own
"which catch" expectation was corrected from ≥0.155 to "any hold ≥0.105" — the
rubric's `opened` threshold 0.080 never moved); (3) the loaded pawl's
steady GPU velocity bias made the *as-written* settle gate unsatisfiable in
every success state → the pawl is now judged by CoM linear speed under the
same 0.05 m/s gate as all other movers (rubric bug fix, intent preserved).

## Smoke battery (`smoke.py`) — 15 checks, rejection/health only

1. **settle/no-NaN** — gate shut, pan tracks its sampled pose, bowl on roof.
2. **randomization A** — pan (x, y, yaw) spans across 8 seeded resets; the
   settled pan tracks the sampled values every time.
3. **randomization B** — bowl roof pose varies.
4. **null policy** — 240 idle steps: gate shut, score ≈ 0.
5. **SEED strategy rejected** — pan constructed settled ON TOP of the hutch
   beside the bowl (the seed's goal state): score ≈ 0, no success.
6. **teleport bypass** — gate stolen open + pan stolen onto the pad without
   crossing the doorway: `on_pad` and `settled` are TRUE, yet no success —
   the transit credential is unearnable by teleport.
7. **closed-gate push** — real 8 N drag at the doorway with the gate shut:
   the pan moves (probe non-vacuous) and is walled outside; order forcing.
8. **ratchet one-way** — real lift + set-down: pawl carries the gate hands-off
   (partial credit only); then a real 6 N (2× gate weight) downward drag
   cannot pull the ratcheted gate back down.
9. **near miss** — the real mechanism run end-to-end but the pan stopped short
   of the pad: transit latched, capped 0.65, no success.
10. **judged-state gates** — pan TILTED 30° on the pad refused (upright gate);
    pan flat on the pad but MOVING refused (settle gate; judged without
    stepping, then retracted — a flat still pan there would be genuine
    success).
11. **latch steal** — the pan stolen back outside after the real open+transit:
    latches survive (score stays 0.65), success does not.
12. **wrong object** — the bystander bowl shoved off the roof: score ≈ 0.
13. **rejection audit** — `success()` observed False at every step.
14. **final no-NaN.**
15. **video** — frames.npz (>10 rgb frames) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, ~98 s).
