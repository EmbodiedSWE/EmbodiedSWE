# trestle_service — build an elevated serving bridge from the two black bowls, serve the dessert on it (i7)

**Env name:** `simgen.trestle_service` (scene `trestle_service`, registered with robot `null`;
solve.py and smoke.py build the same scene-level env).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl.py`)
- Seed plan: ONE unconstrained grasp-carry-place — pick up one loose black bowl, set it down
  on top of another black bowl. Success = a single proximity relation between the two bowl
  origins (xy within 6 cm, z within 3 cm). One object moves; one primitive suffices.

## What changed, and why it is strategically different

The two-black-bowls premise is kept; everything that made the seed's one-step plan
sufficient is removed:

1. **The bowls are never stacked and never picked.** They sit UPSIDE-DOWN and their 100 mm
   rim is wider than the 80 mm Franka jaw span, so no grasp exists — each bowl must be
   **PUSHED** across the counter until centred on one of two green pad marks (a
   positioning-to-a-mark objective, 25 mm tolerance, and the bowls must stay foot-up).
   The seed's core primitive (free pick-and-place of a bowl) is physically unavailable.
2. **The goal is a three-stage CONSTRUCTION, not a two-body relation.** With both bowls
   seated, a wooden serving board must be laid ACROSS them so it rests LEVEL on both raised
   bowl feet, elevated clear of the counter — a genuine two-point support/balance outcome
   that the physics has to hold up. A board on the counter, a ramp with one end on the
   counter, a board balanced on a single bowl, and the same bridge built away from the pads
   are all constructed, settled, rejected states in the smoke.
3. **The only pick-and-place in the task lands on a structure the solver had to build
   first**: the pink dessert cube goes on the board's raised deck (the grip-bar top, the
   counter, and the bowl feet are all rejected rests).
4. **A solver needs a different plan and different code structure**: identify pads and
   assign bowls, execute two closed-loop planar PUSHES to marks, then an oriented placement
   spanning two supports, then a load placement on the built platform — vs. the seed's
   single place-on-target primitive. The seed's own end state (one black bowl stacked on
   the other) is an explicitly tested zero-score outcome (smoke check 8).

Claimed strategy axes: **push-to-mark positioning of ungraspable objects + multi-body
elevated-structure construction (two-point span/balance) + physics-enforced staged order.**
Nothing is stacked bowl-in-bowl, poured, threaded, flipped, or carried in a container.

## Embodiment argument (single Franka arm + parallel jaw, OSC)

Intended base pose: **(-0.42, 0.0, 0.20)** — mounted on the counter slab (the
plated-meal-verified on-counter mount). Worst-case action sites: pads out to ~(0.13, ±0.05)
→ 0.55 m; bowl parks (-0.05, ±0.22) → 0.43 m; board grip at park (0.26 ± 0.03, ±0.10) →
≤ 0.70 m; cube park (0.10, ±0.24) → 0.57 m — all inside the demonstrated 0.35–0.71 m
envelope, most inside the 0.35–0.65 m comfort band.

Per manipulated object:

- **Bowls (x2, must move):** cannot be grasped (100 mm rim > 80 mm jaw) — intended contact
  is a PUSH with closed fingertips on the 50 mm-tall side wall at mid-height, sliding the
  bowl on counter friction (μ≈0.5, ~1.2 N — trivial for the arm; the 25 mm seat tolerance
  is ~10x OSC noise, and solve.py's force-push lands at ~7 mm). The bowl's wide rim and low
  CoM make it slide, not tip (tip force ≈ 5.6 N ≫ 1.2 N slide force). Push direction is
  corrected closed-loop from the visible pad mark; approach from the bowl's own side keeps
  the wrist clear of the other objects.
- **Board (must move):** pinch the raised dark GRIP BAR (24 mm across < 80 mm jaw, top face
  46 mm above the counter — full finger clearance; the bar is the designed pinch feature).
  Carry level, lay across the two seated bowls. Longitudinal tolerance ±~45 mm (260 mm board
  over a 160 mm span), lateral ±~25 mm, yaw ±~10° — all generous for closed-loop placement,
  and the release is from a few cm up: solve.py shows a 30 mm free drop still lands and
  balances.
- **Cube (must move):** top pinch of the 30 mm cube, place on the deck over a trestle foot
  (no tipping moment; solve.py drops it there). Free deck on each side of the grip bar is
  ~85 mm long × 80 mm wide — a huge target 66 mm above the counter, approached from above
  with nothing overhead.

No required contact is near the ground (lowest is the bowl side-wall push at ~25 mm, an
open lateral approach), under an overhang, or through an aperture.

## Execution order

Required order **bowls → board → cube**, enforced by physics itself (declared, and the
rubric's latches are additive so any legal order is monotone): the board has nothing to
rest on before the bowls are seated (bridged() also requires pads_seated, so pre-laying the
board elsewhere earns nothing), and the deck the cube must rest on does not exist until the
board is bridged (topped() requires bridged()).

## Teleport-solution outline (solve.py — all load-bearing interactions through contact dynamics)

0. `reset(seed)`, settle, layout readback (pad centres, bridge yaw, park sides) → score 0.
1. **Bowl A**: teleport to a staging spot 90 mm to the SIDE of its pad (transport only —
   fully off the pad, resting on the counter), then **force-push** it onto the pad:
   velocity-regulated (0.08 m/s) 3.5 N horizontal force with closed-loop heading, force cut
   8 mm from the pad centre, coasts to rest on real friction → 0.15.
2. **Bowl B**: same for the other pad → 0.30.
3. **Board**: teleported level to 30 mm ABOVE the two bowl feet (far outside the 8 mm
   scoring z-band), released; it falls, lands, and must balance on BOTH feet under gravity
   and contact alone → 0.65.
4. **Cube**: released 25 mm above the deck over a trestle foot, settles on → success →
   1.000.
5. Persistence: ≥ 3.3 simulated seconds hands-off; success() must still hold →
   `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing.
**Verified on the forge: seeds 0, 1, 2 — all `SIM_GEN_SOLVE: SUCCESS`, first run** (~21 s
each; bowl pushes seat at ~7 mm error; board and cube land on first drop on all seeds).

## Rubric

- `success()`: both pads seated by distinct upside-down bowls (25 mm) ∧ board level (≤8°),
  in the elevated z-band (bowl height ± 8 mm), both feet inside its footprint ∧ cube flat
  on the deck (board frame, 8 mm z-band) ∧ everything settled (< 0.05 m/s).
- `score()` (latched in post_step, velocity-gated, additive): +0.15 first bowl ever seated,
  +0.15 both pads ever seated, +0.35 board ever bridged, +0.20 cube ever on the bridged
  deck; 1.0 iff success(). Null policy scores 0; latched credit never evaporates.

## Check list (smoke.py — rejection battery, forge: `SIM_GEN_SMOKE: ALL PASS 18/18`, first run)

1. settle/no-NaN + nothing scored at reset; board readback flat on the counter
2. rubric clean at reset (score 0, no success)
3. determinism: same seed → identical layout readback
4. randomization readback: bridge-axis yaw varies
5. randomization readback: bridge centre jitters
6. randomization readback: bowl-swap / board-park / cube-park signs all take both values
7. null policy: 240 idle steps → score 0, no success
8. **seed strategy**: one black bowl stacked on the other (stable) → score 0, no success
9. flat service: board on the COUNTER between the pads + cube on it → score 0
10. ramp: bowls seated, board one-end-on-foot one-end-on-counter → not bridged, score 0.30
11. seat tolerance twin: 15 mm off-centre seats; near miss: settled 45 mm does not
12. flipped bowl (opening-up, centred on the pad, stable) → not seated
13. complete bridge + cube built OFF the pads → score 0 (position-to-mark is load-bearing)
14. grip-bar cube: legit bridge, cube settled on the grip-bar top → not on_deck, score 0.65
15. cube on the counter beside the finished bridge → no cube credit
16. latched credit survives removing the board (score stays 0.65)
17. finite states at the end
(+ frames.npz video recorded, 107 frames)

Assets: fully procedural (3-tier cylinder-compound bowls, visual-only kinematic pad discs,
slab+grip-bar board compound, plain cube). No external asset files.
