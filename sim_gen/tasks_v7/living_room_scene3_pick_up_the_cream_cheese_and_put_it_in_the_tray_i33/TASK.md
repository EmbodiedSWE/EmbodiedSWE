# living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray_i33 — FlapChutePantryScene (`simgen.flap_chute_pantry`)

Get the CREAM-WHITE carton inside a SEALED pantry box whose only entrance is a
gravity-closing ONE-WAY FLAP DOOR: stage the carton on the apron platform, push it
through the flap (the flap yields inward and rides over it), let it tip over the sill
and slide down the internal chute, and let the flap fall shut behind it. The BROWN
decoy carton must stay outside. There is no open top — the seed's pick-and-place plan
is physically impossible.

## Seed provenance

- **Seed task**: `libero_90/living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray`
  — "pick up the cream cheese and put it in the tray": grasp one box among tabletop
  distractors, carry it through free air, lower it into an OPEN-TOP tray. One
  unordered pick-and-place; the only physics is release-and-rest; judged by
  in-container containment.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| goal container | open-top tray — insertion from above is the whole plan | SEALED pantry with a fully closed ROOF; the only entrance is a doorway covered by a one-way flap (smoke 8 constructs the seed's lower-from-above end state: the carton on the roof, score ~0) |
| insertion act | release the grasped box over the tray; it falls in | PUSH the carton through a yielding flap door: the flap swings inward over the carton, the carton tips over the sill, an internal low-friction chute carries it to the back, and the flap gravity-closes behind it |
| mechanism | none — the tray is passive geometry | hinged 40 g flap with a physical one-way stop (its margins are wider than the doorway and press against the wall from behind — smoke 6), gravity return (smoke 5), and a seal interlock (smoke 7: an interned carton shoved at 3× weight never gets out) |
| ordering | none | **physically forced stage-then-push**: the doorway sill is 50 mm up — a ground-level push jams the carton against the apron edge; the carton must first be lifted onto the apron runway |
| perception | fixed named objects | CREAM vs BROWN cartons of identical size; which start slot holds which is shuffled per episode — color is the only cue |
| restraint | n/a | the brown decoy must NOT end up inside (smoke 11/12) |
| assets | LIBERO USDs | 100 % procedural (compound dynamic pantry with sloped chute + apron; jointed flap; two cartons) |
| judging | containment bbox | live geometric success (carton inside AND settled beyond the flap's swing arc, flap re-closed, decoy outside, settled, finite) + latched stage credit |

Code structure shares nothing with the seed: `@SCENES.register` BaseScene, compound
spawner authoring a REVOLUTE joint at spawn (heavy DYNAMIC pantry as body0 — on this
stack a joint anchored to a teleported kinematic body0 stays world-fixed; and
joint-pair collision explicitly ENABLED, because the one-way stop IS flap-vs-wall
contact), pantry-frame predicates, latches in `post_step`, `register_env(...,
robot="null")`.

## Why strategically different

The seed's entire skill is *carry a box through free air and drop it into an open
top*. Here that plan is not merely unrewarded — it is geometrically impossible: the
pantry has no open top, and its end state (the carton resting on the closed roof,
smoke 8) scores ~0. What the solver must bring instead: (1) **perception** of a
shuffled color-only cream/brown identity; (2) a **mechanism model** — recognizing the
orange panel as a one-way inward door that yields to a push and re-closes by gravity,
rather than as a wall; (3) **push-through-contact manipulation** — driving the carton
along a runway INTO a yielding obstruction, with force modulated so the carton crosses
the sill instead of wedging under the flap (smoke 10 shows the wedge is a real, stable
failure state — the flap statically holds a straddling carton at +61°); (4) a **physically forced order** — lift onto the apron before
pushing, because the sill is 50 mm above the ground; and (5) **hands-off dynamics** —
after release the sill tip-over, the chute slide, and the flap's gravity re-close
finish the task with no contact. The one thing the seed does (drop into an opening
from above) is the one thing that cannot happen here.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; readback layout (pantry xy/yaw ±25°, carton slot swap +
   jitter + free yaw); assert flap closed, cartons on the ground; score 0.0.
2. **P1** (teleport = transport only, the seed-legal pick): carry the cream carton to
   35 mm above the apron centre, zero velocity; it FALLS and settles on the apron by
   contact → `staged` latch (score 0.15). Brown is never touched.
3. **P2** (all contact dynamics): a bang-bang horizontal force at the carton's CoM
   (pantry-local −x, 0.6 N base, stall-escalation to ≤3.0 N; frame-drag guard probes
   the force mode at runtime) pushes it along the apron into the flap. The flap yields
   and rides over the carton; release the instant the carton centre crosses
   x_loc < 0.100 (deep enough that its tipping torque, ~3× the 40 g flap's holding
   moment, defeats the doorway wedge). Hands-off: the carton tips over the sill, the
   chute carries it to the back wall (rest ≈ x_loc −0.096), the flap falls shut
   (≈ −1°) → success (score 1.0). A deeper re-push is armed for the wedge state but
   only fires if the carton straddles the sill after 2 s.
4. **P3** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.0000 → 0.1500 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(−0.35, 0.00, 0.00), facing +x** (nominal reach 0.855 m). The
pantry doorway faces the base (nominal yaw 180°); the cartons start ~0.30 m from the
base, the apron span is 0.45–0.65 m, the doorway plane ~0.62 m — all inside the
dexterous shell.

- **Pick**: the cream carton is a free-standing 60 mm cube (150 g) on open ground with
  ≥ 120 mm to the decoy — a top-down or side pinch (jaw max 80 mm ≫ 60 mm) with no
  clutter, easier than the seed's tabletop grasp.
- **Place**: the apron is a 200 × 200 mm platform at 50 mm height — generous open
  target for a set-down.
- **Push**: fingertip push on the carton's rear face at ~78 mm height (carton
  mid-height on the apron), driving it ~0.24 m along the apron into the flap. Push
  force demonstrated at 0.6–1.0 N — trivial for the arm. At the release point
  (carton centre x_loc 0.10) the carton's rear face sits at the front wall plane, so
  the fingertip needs to enter the 140 × 70 mm doorway aperture by at most a
  fingertip's depth — and the flap is riding on top of the carton, clear of the
  finger. Retract straight back; gravity and the chute finish the job.
- **Flap**: never grasped, never held — it is pushed only THROUGH the carton.
- **Decoy**: never touched — restraint, not manipulation.
- The pantry is a 25 kg dynamic body with damped, zero-sleep-threshold settings:
  incidental contact cannot meaningfully move the goal frame (and every predicate is
  pantry-frame relative regardless).

## Execution order (declared)

`pick cream carton → place on apron → push through the flap (hands off before the
sill) → flap gravity-closes`. The stage-then-push order is REQUIRED and physically
forced: the sill/apron top is 50 mm above the ground, so a ground-level push cannot
enter the doorway — the carton must first be lifted onto the runway. No other order
constraints. Once the carton is inside it is sealed in (smoke 7), so there is no
"undo and redo" path — but none is needed.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0`: SUCCESS, scores 0.0 / 0.15 / 1.0 / 1.0 (rest x_loc −0.096, flap −0.8°).
- `solve --seed 1`: SUCCESS (different pantry pose, slot arrangement).
- `solve --seed 2` (pantry (0.406, −0.037) yaw 176.3°, cream +y slot): SUCCESS —
  three seeds; the wedge fallback fired on none of them after the release deepening.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 13/13**, frames.npz (98 × 600 × 960) saved:
  1. settle/no-NaN; flap closed (−0.8°), cartons on the ground outside; score 0
  2. randomization readback differs (pantry yaw Δ19.9°, pantry xy Δ38 mm, cream
     carton Δ59 mm / Δ46.5° yaw)
  3. slot swap: the cream carton occupies both start slots over 10 resets
  4. null policy: 240 idle steps → flap closed, score 0, no success
  5. flap mechanism: 0.3 N inward push opens to +38.4°; gravity re-closes to −0.7°
  6. flap one-way: 0.6 N outward push for 1 s — most-negative angle −1.5° (stop is
     flap-margin-vs-wall contact, not fiat)
  7. SEAL INTERLOCK: interned carton shoved toward the doorway at 3× weight for
     1.5 s — never leaves the chamber (ends back at x_loc −96 mm)
  8. SEED STRATEGY: carton lowered-from-above ends on the closed ROOF → score 0
  9. apron-only: carton staged and abandoned → staged latch only (0.15), no success
  10. doorway wedge: carton stuck holding the flap +61.2° open → no success (0.25)
  11. wrong object: BROWN inside, cream outside → no success
  12. both inside: only the decoy clause fails → no success
  13. video frames.npz saved

## Files

- `scene.py` — FlapChutePantryScene + pantry/flap compound spawners (joint authored at
  spawn) + rubric; registers `simgen.flap_chute_pantry`.
- `solve.py` — teleport-transport + force push-through + gravity/chute/flap
  follow-through certificate (`--seed N`).
- `smoke.py` — 13-check rejection battery + video.
