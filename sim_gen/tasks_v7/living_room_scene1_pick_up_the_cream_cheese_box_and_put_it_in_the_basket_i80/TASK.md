# counterweigh_basket — put the cheese box in the basket of a beam balance and make it settle level

`living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket_i80`
Env: `simgen.counterweigh_basket` · robot slot: `null` (scene-level task) · procedural assets only.

## Seed provenance

Seed: `roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.py` —
grasp the cream-cheese box among distractors (alphabet soup, tomato sauce, ketchup), carry it
over a passive open basket, release; `_terminated` is a bounding-box containment check against
`basket/contain_region`.

## What changed, and why it is strategically different

The receptacle is no longer passive. The basket is one end of a **free-swinging beam balance**:
a heavy portal stand carries a 60 cm beam on a revolute joint (hard stops ±20°) with a keel CoM
0.155 m below the pivot, so the *empty* beam self-levels. The basket hangs at one arm end; a
shallow red-rimmed **tray** hangs at the other. On the floor lie the blue cheese box (0.20 kg)
and two same-size 45 mm cubes distinguishable only by look: a **dark metal block (0.20 kg — the
box's match)** and a **white foam decoy (0.03 kg)**. Success = box **inside the basket** AND the
beam settled **level (|pitch| ≤ 8°)** AND everything still (60-consecutive-step counter).

Strategic distance, verified in physics on the forge:

- **The seed's entire plan fails here.** "Put the box in the basket" tips the beam to its loaded
  equilibrium, −14.8° (smoke #6): containment holds, the level clause rejects. The seed's task
  ends where this one *starts*.
- **New reasoning step, absent from the seed and from every task I read:** static torque
  equilibrium. The solver must infer that the tray needs a counterweight matching the box's
  weight, select the metal block *by material appearance* (the foam decoy leaves −13.9°, smoke
  #8), and put it on the **opposite** end (dumped in the basket with the box → pinned at the
  −20° stop, smoke #9).
- **Not a containment task in disguise:** a perfectly *balanced* beam with the loads mirrored
  (metal in basket, box on tray) settles level and still and is **rejected** with score ~0
  (smoke #10) — the goal is the conjunction, not either half.
- Distinct from the corpus: i292 (same scene family) is declutter + roll-in transport through a
  roofed garage; no other tasks_v7 task is a balance/counterweight mechanism (the closest
  mechanisms are flap/latch/labyrinth/pyramid — none judge a settled equilibrium angle).

## Teleport solution (solve.py — passes seeds 0 and 1)

Teleports are transport-only; every load-bearing interaction is contact dynamics on the free
revolute joint:

- **P0** reset + 60-step settle: empty beam self-levelled (|pitch| ≤ 3° asserted), score ~0.
- **P1** one pose write puts the **metal** block 3 cm above the tray, oriented with the beam;
  the drop, the catch, and the swing to the loaded equilibrium (+14.85° measured) are physics.
  Score 0.25 (counterweight latched).
- **P2** one pose write puts the box just above the basket floor of the now-raised end; the
  drop and the counterswing back to level (−0.59° measured) are physics. Score 1.0.
- **P3** 2 s extra hands-off ring-down, then **P4** strict persistence: 3.3 s with success
  asserted every 40 steps → `SIM_GEN_SOLVE: SUCCESS`. Nothing touches any body after P2's
  release; the level condition is torque equilibrium and cannot be faked by a pose write.

`SIM_GEN_SCORE` prints at every phase boundary: 0.0000 → 0.2500 → 1.0000 → 1.0000 → 1.0000
(non-decreasing; credit is latched in `post_step`).

## Embodiment argument (Franka, base at the origin)

All manipulands sit at radius 0.20–0.30 m; the tray/basket ends orbit (0.42, ±0.30) at height
~0.19 m — inside a Franka's comfortable envelope from a base at the origin facing +x.

- **Metal/foam cubes (45 mm, ≤ 0.20 kg):** direct top-down pinch — cube width is well inside
  the gripper stroke; place = release 2–3 cm above the tray, exactly what P1 emulates.
- **Cheese box (96×64×36 mm, 0.20 kg):** pinch across the 64 mm width (fits the ~80 mm stroke)
  or across the 36 mm height from the side; release above the basket opening (130×100 mm inner
  — several cm of clearance around the box), exactly what P2 emulates.
- **The tilted-target drop is easy for an arm:** after loading the tray the basket end rises to
  ~+15°, i.e. its opening *faces the robot slightly* — release height ~2 cm, walls 55 mm catch
  drift. No forceful interaction with the beam is ever needed; the mechanism does all the work.

## Execution-order declaration

Either order works physically (load tray first — as solve does — or basket first: the box-first
path tips the beam to −16°, the tray end rises, and the metal drop onto the raised tray restores
level; the stops guarantee nothing jams). The rubric is order-free: both partial credits are
independent latches, and success judges only the settled end state.

## Rubric

- 0.25 — metal block ever resting on the tray (latched in `post_step`, velocity-gated)
- 0.35 — box ever inside the basket volume (latched, velocity-gated)
- non-success cap 0.60; **1.0 iff** success(): box in basket ∧ |pitch| ≤ 8° ∧ 60-step stillness.
- Null policy: 0.000 (smoke #5).

Anti-flake measures baked in (memory-verified forge traps): heavy *dynamic* stand (kinematic
body0 anchors stay world-fixed after reset teleports), explicit MassAPI CoM for the keel
(mass-only leaves CoM at the origin — no restoring torque), strong angular damping instead of
jointFriction (inert outside articulations), consecutive-still counter instead of instantaneous
velocity gates (turning-point stillness — proven live in smoke #12), probe velocities on every
teleport + ≥60-step minimum before consulting stillness (teleport-vacuous predicates), float32
slack on score comparisons.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 14/14` on the forge)

1. settle: finite states, empty beam self-levelled, objects at slots, still
2. settle: score ~0, no success
3. randomization: metal/foam Bernoulli slot swap flips, always opposite slots (readback)
4. randomization: per-slot jitter, box jitter, stand yaw spread (readback)
5. null policy: 240 idle steps → score ~0, no success
6. **seed strategy**: box in basket, nothing else → beam −14.8°, NOT success, score ≤ 0.35
7. latched credit: box teleported back out → score unchanged, no success
8. wrong block: foam decoy on tray + box in basket → −13.9°, no cw credit, score ≤ 0.35
9. wrong place: metal stacked in the basket → pinned at −20°, score ≤ 0.35
10. mirrored: metal in basket + box on tray → **level and still, yet rejected**, score ~0
11. near-miss: box on the bar outside the basket window → score ≤ 0.25
12. mid-swing: correct outcome judged during the counterswing (max ω 0.84 rad/s crossing the
    level band) → success never fires; probe dismantled before settling
13. rejection audit: success() never True at any judged point
14. final no-NaN

Verification: forge pod (Isaac Sim 5.1, RTX 4090). solve seeds 0 & 1 → `SIM_GEN_SOLVE: SUCCESS`
(rc=0, 18.4 s each); smoke → `SIM_GEN_SMOKE: ALL PASS 14/14` (rc=0, 47.4 s, 377 frames recorded).
