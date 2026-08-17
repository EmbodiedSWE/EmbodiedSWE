# sauce_chute (i166)

## Seed provenance

- Seed id: `libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.py`
- Seed strategy: PICK a freestanding sauce can off a table (disambiguated from four
  freestanding distractors by identity), CARRY it through free space, LOWER it into a
  STATIC wooden tray. Success = can centre inside the tray's contain-region bounding
  box. One grasp, one placement, the container never moves.

## What changed and WHY it is strategically different

Kept from the seed only the abstract goal ("the sauce can ends up inside the wooden
tray, same-shape distractor excluded") and inverted every element of the plan:

1. **The target object is ungraspable — metrically.** The red can sits in a detent
   cradle inside an elevated, roofed chute: the roof leaves 14 mm above it (no jaw
   fits) and the only other access, a 55 × 50 mm rear push port, is smaller than the
   can (115 mm long, 60 mm dia) in both critical dimensions. The seed's core skill —
   grasp the target — is physically impossible here.
2. **The container is what gets transported — twice.** In the seed the tray is static
   scenery. Here the TRAY is the only thing the solver carries: first EMPTY to a catch
   dock under the chute's spout, then LOADED (can riding inside) onto a green delivery
   pad whose position varies per episode. Object-to-container is replaced by
   container-to-object-to-goal.
3. **The object moves only by mechanism physics.** The can leaves the chute
   exclusively by ROLLING: a gentle fingertip nudge through the port (~3 N to pop the
   5 mm cradle ridge) hands it to the 2° ramp; gravity rolls it 0.24 m to the open
   spout, it free-falls 144 mm, and the docked tray must CATCH it. The hand never
   touches the can, and the catch is ballistic contact, not placement.
4. **A declared trajectory rule creates an irreversible failure.** The red can must
   never land outside the tray: touching open floor/pad latches a permanent `floored`
   failure (a spoiled canned good). This forces a strict physical ordering — dock the
   tray BEFORE pushing — with no analogue in the seed, where the object can be put
   down and re-picked at will.
5. **Same-shape color discrimination is kept but weaponized against the seed's plan**:
   the one can that IS loose and graspable (the white decoy) is exactly the one that
   must be left alone. The seed's strategy, executed here, acts on the wrong object.

A solver needs a different PLAN (dock the container → actuate the dispenser → catch →
deliver the loaded container) and a different CODE STRUCTURE (kinematic compound chute
with ramp/roof/port/cradle, capsule rolling dynamics, catch-dock and pad predicates,
a permanent-failure trajectory latch) — not different parameters of pick-and-place.

## Teleport-solution outline (solve.py)

- **P0** reset, settle; assert the can seated captive in the cradle, score ≈ 0.
- **P1 DOCK (transport teleport)**: one pose write carries the EMPTY tray from its
  jittered spawn zone to the catch dock on the ground under the spout; it settles
  through real ground contact; `tray_docked` asserted on the settled pose.
- **P2 PUSH (applied force + hands-off physics)**: a velocity-servo force
  F_y = clamp(15·(−0.22 − v_y), −6 N, 0) at the can's centre, world −y, active ONLY
  within the 30 mm stroke a fingertip poked through the port could reach (pod
  frame-drag corrected: F pre-encoded as quat_apply_inverse(q_now, F_world) each
  step; K·dt/m = 0.36; peak force asserted ≤ 6 N). The wrench is then zeroed and
  everything after is gravity + contact: ridge pop, 2° ramp roll, spout drop, tray
  catch. Asserted: exited ∧ in_tray ∧ never floored ∧ `caught` latched.
- **P3 DELIVER (transport teleport)**: tray AND can moved by ONE rigid delta
  (relative pose preserved — a loaded carry), tray bottom 2 mm above the pad top,
  velocities zeroed; settles onto the pad through contact; success() asserted.
- **P4** ≥ 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS` only if
  success() still holds. `SIM_GEN_SCORE` printed at every phase boundary;
  non-decreasing asserted. Passes on seeds 0 and 1 (see forge logs).

Teleports transport bodies across free space only. The single applied force is the
fingertip surrogate — magnitude-capped, stroke-limited to the port's reach, and the
load-bearing outcome (ridge pop, roll, drop, catch) is pure contact dynamics.

## Embodiment argument (single Franka, parallel jaw, OSC)

Plausible base pose: arm base at the world origin (0, 0, 0); the chute centreline at
x = 0.45, the port face at y ≈ 0.364, the pad near (0.16, −0.40), the tray spawn near
(0.20, 0.38), the decoy near (0.66, −0.24). Everything actionable lies within a
0.42–0.72 m radial band; the highest waypoint (the port, z ≈ 0.19) is low and frontal.

- **Tray (wooden open box, 216 × 210 mm, 70 mm tall, 0.45 kg)**: pinch any 8 mm wall
  top edge — the jaw closes over the 54 mm wall from above with the fingers outside
  and inside the box; 0.45 kg (+0.35 kg can on the second carry, 0.80 kg total) is a
  normal Franka payload. Set-down tolerances are ±35 mm (dock) and ±30 mm (pad) —
  far coarser than arm repeatability.
- **Red can (the push)**: close the jaw and poke the closed fingertip pair through
  the 55 × 50 mm port (a closed Franka fingertip cluster is ~20 × 35 mm); a ~3 cm
  forward stroke at z ≈ 0.19 with ~3 N of force pops the can over the ridge — the
  proven closed-fingertip push pattern. The hand never enters past the wrist; the
  roof and walls are never load-bearing for the arm.
- **Catch and delivery need no dexterity**: the tray is docked before the push; the
  can lands in it by gravity. The loaded carry keeps the tray level — walls 24 mm
  above the resting can's centre retain it against normal carry accelerations.
- **Nothing else must be touched.** The white decoy is a leave-alone distractor; the
  chute is a fixed fixture.

## Execution order

Physically forced order: DOCK must precede PUSH — once pushed, the can rolls out and
falls within ~0.6 s, and a can that lands anywhere but the tray latches the permanent
`floored` failure. DELIVER must follow the catch (the can can only get into the tray
via the chute drop). The rubric itself imposes no artificial sequencing beyond these
physics; describe() states the ordering rule and the gentle-push warning explicitly.

## Rubric

success(): red can settled INSIDE the upright tray (tray-frame cavity band, rim-perch
rejected) ∧ tray resting ON the pad (xy ± 30 mm, bottom at pad top ± 10 mm, upright
≤ 10°) ∧ can never floored ∧ white decoy not in the tray ∧ everything at rest.
score(): 0.15 docked + 0.15 roll progress (running MIN of can-y mapped cradle→spout,
exactly 0 for null) + 0.15 exited + 0.25 caught (unreachable once floored) + 0.15
delivered — all latched/rising-only, cap 0.85; 1.0 iff success().

## Check list (smoke.py — rejection battery, 15 checks)

1. settle/no-NaN: can seated captive in the cradle, tray/decoy on the floor, still.
2. reset score ≈ 0, no success.
3. randomization: tray / pad / decoy positions vary (readback over 8 seeds).
4. randomization: tray yaw + can cradle-seat x vary (readback).
5. null policy: 240 idle steps → score ≈ 0, no success, can never leaves the cradle
   (captivity real; no capsule creep).
6. SEED-STRATEGY end state / wrong object: the loose WHITE decoy hand-placed into the
   tray on the pad → rejected, score ≈ 0.
7. dock near-miss: tray settled 50 mm off the dock target (tol 35 mm) → never docked.
8. floored rule: red can landing on open floor latches permanent failure; the manual
   rescue (same can hand-placed into the tray on the pad) still rejected — `caught`
   can never set, score ≤ 0.31.
9. catch without delivery: tray docked, can settled inside → caught latched, but no
   success off the pad, score ≤ 0.71.
10. pad near-miss: loaded tray set down 45 mm off the pad centre (tol 30 mm) → no
    delivery, no success.
11. decoy clause at the cap: decoy INSIDE the tray, everything else perfect on the
    pad → every latch earned, score pinned at 0.85, success still False.
12. latched credit survives the red can being yanked back out; still no success.
13. tray not presented: tray INVERTED on the pad with the can on top → rejected.
14. rejection audit: success() never True anywhere in the battery.
15. final no-NaN.
