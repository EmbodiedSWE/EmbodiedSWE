# Task: die_roll_match — roll the menu die to the displayed face, serve it on the crimson mat

**Env id:** `simgen.die_roll_match`
**Package:** `sim_gen/tasks_v7/libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i369/`

## Seed provenance

Derived from `libero_90/libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate`
(RoboVerse pack). The seed grasps a free white bowl and translates it to a pose
RELATIVE to a plate — a pure pick-and-place whose goal lives entirely in position
space (`_terminated` checks the bowl−plate offset box).

## What changed and why it is strategically different

The goal is moved from position space into the cube's ORIENTATION group, and
prehension is removed entirely:

- The manipulated object is an oversized six-color **menu die** (140 mm cube, 1 kg
  — far wider than a ~80 mm parallel jaw, so it cannot be grasped or lifted).
- Per episode a small kinematic **reference die** on a pedestal displays one face
  up; that color is the episode's target. The big die must end **flat on the
  crimson mat with the matching face UP**.
- Sliding or spinning the die can NEVER change which face is up. The only way to
  reach the goal orientation is to **tip the die over its bottom edges, one
  quarter-roll at a time** (rolling toward direction u raises the face that faced
  −u; a target that starts facing DOWN needs two rolls in the same direction).
  The physics margins are asserted in `cfg.__post_init__`: at the high push
  height the tipping force (0.583 mg) is well below the slide threshold
  (μ_s·mg = 0.9 mg), and at the low push height the die slides without tipping
  (h = 0.035 < w/2μ_s = 0.078).

So a solver needs a different **plan** (read the displayed face, plan a
quarter-roll sequence through the cube's rotation group, execute tip-rolls
without chain-rolling, then push-slide at a non-tipping height) and a different
**code structure** (SO(3) face bookkeeping + non-prehensile edge-roll execution
+ face-preserving slide — not grasp-transport-place). No prior corpus task uses
sequential edge-rolling of a cube to hit an orientation goal read off a
displayed reference object.

## Solution phases (solve.py — passes forge seeds 0, 1, 2; seed 2 is a two-roll target-DOWN episode)

- **P0** settle; READBACK: reference die's up face == sampled target; die rests on
  its sampled face; score ~0.
- **P1** quarter-roll loop: read the target face's world normal from the die quat;
  target horizontal → one roll toward −n̂; target down → two rolls the same way
  (bottom-edge direction chosen toward the mat). Each roll: horizontal push force
  at `roll_push_h` (CoM force + lever torque, world wrench rotated into the BODY
  frame every step), CUT ~5° past the balance point so gravity finishes the
  quarter-roll without chaining; verified by top-face readback with force
  escalation 0.70→0.86 mg on refused tips. Latches: first roll 0.15, target-up
  +0.25.
- **P2** low push-slide onto the mat: velocity-regulated horizontal force at
  `slide_push_h` with continuous kinetic-friction feedforward (a hard static-FF
  cutoff limit-cycles at the threshold speed) and counter-tip lever torque;
  +0.20 running-max approach credit.
- **P3** release, ring down to success (still 30+ steps, flat ≤10°, target up,
  Chebyshev ≤8 cm of mat center, counter-height z band).
- **P4** hands-off persistence 400 steps (3.3 s).

No transport teleports at all; every interaction is contact dynamics through
external wrenches, zeroed before judging.

## Embodiment argument (Franka, base at the counter edge near (0, 0))

- **Die (140 mm, 1 kg):** ungraspable by construction (jaw span 80 mm) — the
  intended contact is PUSHING with the closed-gripper knuckles/palm: high pushes
  (~120 mm, ~7–8 N) for tip-rolls, low pushes (~35 mm, ~8–12 N) for slides. Both
  heights and forces are comfortably inside Franka's reach and payload; the
  0.9/0.8 friction counter means push forces stay under ~12 N.
- **Reference die (55 mm, kinematic):** never needs touching — it is a display.
  Perception-only (top face color).
- **Mats:** flush visual markers, nothing to manipulate.
- All action happens in a 1.1 m × 0.8 m counter patch in front of the base;
  every push is a horizontal end-effector motion at 35–120 mm height, no
  regrasping, no lifting.

## Execution order

Any order of rolling and sliding is allowed (the rubric judges only the settled
end state, like the seed); the natural plan rolls first (rolls translate the die
~0.14 m per tip, so fine positioning last is easier), and the solve declares and
follows roll → slide.

## Checks (smoke.py — 16/16)

1. settle/no-NaN; 2. fresh reset score ~0; 3–4. randomization by READBACK (top
face + target vary, target never the initial top, reference displays the target,
xy jitter + orientation vary); 5. null policy ~0; 6. sub-threshold push (0.40 mg)
neither tips nor slides; 7. 0.70 mg push genuinely quarter-rolls (6+7 bracket the
threshold); 8. seed-strategy end state — pure transport to the mat with the top
face READBACK-preserved — rejected, progress-only score; 9. spin cheat: yaw never
changes the up face, no orientation credit; 10. wrong place: target-up settled on
the DECOY mat rejected; 11. place near-miss 3 cm outside tolerance rejected;
12. tilt near-miss 18° > 10° rejected by the flat clause alone (removed before
ring-down); 13. settle gate: perfect state judged while sliding rejected
(removed before ring-down); 14. wrong object: reference die on the mat earns
nothing; 15. success() never True at any judged point; 16. final no-NaN.
