# hooked_basket — load the red can into the basket, then hang the basket on the hook

**Seed:** `libero_90/living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.py`)
**Tier:** medium — **2 stages**. **Execution order: NOT required** — the rubric and the
scene text both allow "load the can, then hang the loaded basket" AND "hang the empty
basket, then drop the can in" (the mouth openings stay reachable in either support
state). The demonstrated order (solve.py) is load-then-hang.
**Env name:** `simgen.hooked_basket` (scene `hooked_basket`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed is a one-stage free pick-and-place: grasp the tomato-sauce can among six
grocery distractors, carry it over an open basket standing on the table, release, and a
bounding-box containment check ends the episode. The basket never moves; the world
never has to hold anything up.

Here the seed's ENTIRE goal state is demoted to a rejected intermediate, and the
critical manipuland becomes the **loaded container**:

- Success is a **suspension equilibrium**: the red can inside the basket AND the basket
  hanging from a hook stand's orange arm **by its blue handle bar** — bar seated on the
  arm's top face, arm threaded through the arch under the bar, basket upright, fully
  clear of the ground and of every other support, everything at rest, beige can out.
- The exact end state the seed rewards — red can settled inside the basket while the
  basket stands on the ground — is explicitly **NOT success** here and earns only the
  containment latch (smoke #6 constructs it and asserts rejection).
- The load-bearing interaction is a **compound carry + aperture-threading placement**:
  the ~0.7 kg loaded basket must be carried by its 18 x 16 mm handle bar (spilling the
  can voids the containment clause — smoke #11), the arm's capped tip must pass through
  the 166 x 95 mm arch under the bar, and the bar must be set down on the arm's
  24 x 18 mm crossing patch inside a 71 mm seat window (post side and end-plate side
  both excluded). Correctness of the seat is certified by physics: only a genuinely
  seated bar keeps `success()` true through 3+ hands-off seconds of pendulum ring-down.
- A **beige distractor can** (same shape, different color) must be left out of the
  basket (smoke #10 shows both-cans-in is rejected).

A solver therefore needs a different plan (object-into-container, then
container-onto-fixture — a hierarchical transport where the second stage carries the
result of the first) and different code structure (a carry that maintains an
equilibrium constraint, plus a guarded threading/seating move), not different
parameters. Versus the sibling tasks_v7 packages from the same seed family: no fixture
DOF is actuated (unlike the sliding-lid hamper, spring bay, or dosing valve), nothing
is balanced against a counterweight, and nothing rolls through a channel — the novel
mechanic is hanging a loaded container on a hook.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects; they never do the task. Insertion and seating both go through
gravity + contact dynamics. Phases (each boundary prints `SIM_GEN_SCORE`,
non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (stand pose + yaw, basket pose,
  both can positions, basket mass — seed provenance in stdout), baseline score ~0.
- **P1 INSERT (teleport to free air, then gravity)**: one pose write puts the red can
  6 cm above the rim, centred over a mouth opening beside the handle bar — basket-local
  z = 0.180, ABOVE `inside_z_max` = 0.115, so the freshly-teleported state earns no
  containment credit (asserted). Gravity drops it through the real 76 mm opening; it
  settles on the basket floor; only that settled state latches `_in` (score 0.35).
- **P2 TRANSPORT (composite teleport)**: basket and can are written together with the
  can's basket-relative pose preserved — exactly what a careful carry does — to a
  staging pose over the arm: bar perpendicular to the arm, centred over the seat window
  (stand-local x = `seat_x_mid`), bar underside 22 mm ABOVE the arm's top face. That is
  outside the seat z-window, so hang geometry is false at the teleport instant and
  success() is unreachable there (both asserted). Lift credit latches (score 0.50).
- **P3 SEAT (contact dynamics)**: gravity lowers the loaded basket; the arm threads the
  arch and the bar lands on the arm's top patch; the loaded basket becomes a pendulum
  (CoM 0.20 m below the seat — passively stable) and rings down. Because a single
  `success()` sample can land on a swing turning point, the solve requires success at
  SIX consecutive 1/3 s samples before calling the hang settled. Score 1.0.
- **P4 persistence**: 3.3 more simulated seconds hands-off; only if `success()` held
  prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (distinct layouts in the stdout readback: stand
yaw +173.9° vs +203.4°, different basket/can positions; both runs end
`SIM_GEN_SOLVE: SUCCESS` with monotone scores 0.00 → 0.35 → 0.50 → 1.0 → 1.0).
Measured seat readback sits mid-window: bar at stand-local x = 0.163 (window
0.128–0.199), z = 0.408–0.409 (window 0.402–0.418), up-axis 0.998 (limit 0.940).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(0.05, 0.05, 0.0)`, facing +x. Working radii from this base: can
spawns 0.29–0.55 m (ground-level side pinch), basket handle bar 0.21–0.35 m at 0.23 m
height, seat region on the arm 0.26–0.56 m at 0.41–0.44 m height — all inside the
Franka's reach envelope with comfortable margin, and the base footprint clears every
spawn region by ≥ 6 cm.

**Red can: side pinch, lift, release over an opening.** A free-standing 60 mm cylinder
on open ground (spawn bands are ≥ 10 cm from the other can, verified by readback) — a
canonical side pinch across an 80 mm jaw at ~50 mm grasp height, approach unobstructed.
Insertion is a release above the basket mouth: the two openings beside the handle bar
are 76 mm wide x 170 mm long vs the 60 mm can — ±8 mm lateral tolerance, far above OSC
noise, and the drop itself is exactly what P1 certifies physically. Works on the
grounded basket (rim at 12 cm) or the hanging basket (rim at ~31 cm) — either order.

**Loaded basket: pinch the handle bar, carry, thread, set down.** The blue bar is an
ideal jaw target: 18 mm wide x 16 mm tall, with ~75 mm of free span on either side of
centre between the struts and the open 95 mm-tall arch beneath giving full finger
clearance. Total load ~0.7 kg (basket 0.40 + can 0.30) — a fraction of the Franka
payload; the load hangs below the grasp, so the carry is passively stable, and the
rubric's containment is judged in the basket frame, so normal carry tilt cannot spill
credit. The hang move is two straight Cartesian segments: (1) thread — slide the
basket horizontally along the arm axis so the arm's capped tip passes through the
arch: 166 x 95 mm aperture vs the 70 x 70 mm end plate → ±48 mm lateral and a 25 mm
vertical window for the basket height, ~10 cm of travel; (2) set down — lower ~25 mm
until the bar rests on the arm anywhere in the 71 mm seat window (± 45 mm lateral
tolerance), then open the jaw and retreat upward through the open top. Every tolerance
is ≥ 8 mm; the gentle set-down that P2→P3 certifies (22 mm free drop) is strictly
harsher than a servoed release.

**Distractor can, stand:** never need to be touched (the stand is a kinematic fixture;
the beige can must merely be left alone).

Every contact the task requires is one the arm can make.

## Success and rubric (physical outcomes only)

Geometry is judged in body frames (both fixture and container poses are randomized):
containment in the BASKET frame (|x|,|y| < 7.5 cm, floor < z < 11.5 cm — so a carried
or hanging basket judges identically), the seat in the STAND frame (bar centre inside
x ∈ (0.128, 0.199) along the arm, |y| < 45 mm, z ∈ (peg_top + 2 mm, peg_top + 18 mm)).
`success()` iff, simultaneously and settled (basket |v| < 5 cm/s, |ω| < 0.4 rad/s, can
|v| < 5 cm/s): red can inside + bar seated + basket upright (≤ 20° from vertical) +
basket elevated (root > 10 cm) + beige can NOT inside. `score()` ∈ [0,1], latched each
physics substep: `0.35·in + 0.15·lift + 0.30·hang` — `in` = red can ever settled
inside; `lift` = basket ever clearly off the ground with the can inside; `hang` = hang
geometry ever achieved with the can inside — capped at 0.80; **1.0 iff success()**;
~0 for doing nothing. Every term gates on the RED can (hanging the empty basket or
loading the beige can earns nothing — smoke #7/#8); latched credit never evaporates
(smoke #11).

## Randomization

Per episode (verified by sim READBACK in the smoke): stand xy ± 5 cm and yaw
180° ± 30° (the seat's world position and approach direction change); basket xy ± 6 cm
with FREE yaw; both cans in disjoint ground bands, x ∈ [0.34, 0.44], with the two
y-bands SWAPPED 50/50 (so "the nearer can is the target" is not memorizable),
guaranteed ≥ 10 cm apart.

## Check list (smoke.py — rubric REJECTION battery, 13 checks; forge result
`SIM_GEN_SMOKE: ALL PASS 13/13`)

1. settle: reset finite — basket upright on the ground, red can upright on the ground,
   everything at rest
2. settle: score ~0 at reset, no success
3. randomization readback: stand xy + yaw and basket xy + yaw vary
4. randomization readback: can y-bands vary AND swap (both orders observed); cans
   ≥ 10 cm apart at every seeded reset
5. null policy: 240 idle steps → score ~0, no success
6. SEED END STATE: red can settled inside the GROUNDED basket — the seed task's whole
   goal — containment latched but NOT success, score ≤ 0.36
7. wrong object: beige can inside the properly HUNG basket (bar seated, verified),
   red can on the ground → no success, score ≤ 0.02
8. empty hang: EMPTY basket properly seated on the arm, elevated, at rest → no
   success, score ≤ 0.02 (hanging earns nothing without the can)
9. wrong support: loaded basket PERCHED on the post top — elevated + upright + can
   inside + at rest, but the bar is not on the arm → NOT success, score ≤ 0.51 (the
   stand-frame seat window is load-bearing, not just "elevated with can")
10. exclusion: BOTH cans inside the properly hung basket — hang latch True yet NOT
    success, score ≤ 0.80
11. spill + latch non-evaporation: a genuine success hang is constructed (sampled
    outside the audit), then the red can is teleported out to the ground → success
    turns False (live gating) while the latched score stays 0.80
12. rejection audit: success() never True at any judged REJECTION probe
13. final no-NaN

N/A notes: **out-of-order end states** need no extra control — both orders are
legitimately allowed, and each partial state is already covered (#6 = loaded-not-hung,
#8 = hung-not-loaded). A bar seated but tilted basket is not constructible as a settled
state (the hanging pendulum self-rights: CoM 0.20 m below the seat), and off-window
seats slide against the post or the end plate by construction of the window bounds.

Video frames are recorded throughout and saved to `frames.npz` in the working
directory.
