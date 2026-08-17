# Find the heavy bowl — beam-balance weighing (`libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187`)

## Seed provenance

Derived from `libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate.py`).
The seed is a single blind pick-and-place: one known black bowl, one passive plate, success =
bowl xy within 6 cm of the plate at plate height. Two seed elements survive: a black bowl ends
up standing on a white plate, in a kitchen-table setting.

## What changed, and why it is strategically different

| | Seed | This task |
|---|---|---|
| Plate | passive, on the table | the **pan of a beam balance** — a rimmed plate on one end of a revolute beam whose other end carries a built-in counterweight bias (authored CoM offset, `beam_com_y = -0.086` on a 1.0 kg beam ⇒ 0.84 N·m ≈ a 0.43 kg bowl at the 0.20 m arm). Empty, the beam rests tilted pan-side-**up** at its +8° stop. |
| Bowl | one, known | **three visually identical** black bowls; exactly one is secretly **heavy** (0.60–0.75 kg — which one, and its mass, are randomized per episode); the other two are light (0.22–0.30 kg). |
| Goal | put the bowl on the plate | beam fully tipped **down** at its −8° stop, held there by **exactly one** upright bowl on the pan, the other two at rest on the ground off the balance. Only the heavy bowl can overcome the bias (light margin ≥ 0.26 N·m up, heavy margin ≥ 0.33 N·m down — asserted in `__post_init__`). |

The strategic core is **measurement, not placement**: an information-gathering plan. The solver
cannot see which bowl is correct — it must use the balance as an instrument (place a bowl,
*observe* whether the beam drops, remove it and try the next if not). The rubric never reads the
hidden masses; the beam reaching its down-stop **is** the mass check, executed by the simulator's
own contact dynamics. This is unlike the seed and unlike every other bowl/plate task in the
corpus (all placement-, stacking-, transport-, or mechanism-actuation-shaped): here the *same
action* (bowl on plate) is right or wrong depending on a hidden physical property, and the plan
must branch on an observation.

Cheats each die on a specific clause (matched 1:1 by smoke negatives):
- seed-style blind place of a light bowl → beam stays up, `beam_down()` false (and 1-in-3 luck at best);
- piling both light bowls on the pan to fake the weight → tips, but `on_pan.sum()==1` fails;
- parking a bowl on the counterweight arm or on the bar inboard of the pan → outside the pan
  xy/z band, `on_pan` false;
- an inverted bowl on the pan → `bowls_upright` clause fails;
- hovering/stacked bowls above the pan → `pan_z_lo/hi` band (0.028–0.075, rest ≈ 0.041) rejects;
- transient tip (e.g. a slammed light bowl) → beam relaxes back up; `settled()` + the ≥3 s
  hands-off persistence window in solve.py kill it.

Score (latched, never decreasing): 0.10 any bowl lifted, +0.15 any bowl weighed on the pan
(`_weigh_ever`), +0.35 the beam ever driven to its down-stop by exactly one panned bowl
(`_tip_ever`), capped at 0.90 unless full `success()` = 1.0.

## Teleport solution (solve.py)

Teleports are TRANSPORT ONLY; release points are provably outside every credit band
(release z = 0.086 beam-local > `pan_z_hi` = 0.075, asserted at runtime).

1. **Weigh a light bowl** — transport it to just above the pan, gravity drops it in; the beam
   *stays up*: the honest negative measurement. (Score 0.25.)
2. **Remove it** — transport back to its ground slot; latched credit persists.
3. **Weigh the heavy bowl** — same release; its real mass (verified by PhysX readback) drives
   the beam to the −8° stop. `success()` becomes true. (Score 1.0.)
4. **Persistence** — 3.5 simulated seconds hands-off; `success()` must still hold, then
   `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka arm)

A Franka based at ≈ (−0.18, 0.05, 0), facing +x, reaches all three ground bowls
(x ≈ 0.08–0.11, y ∈ [−0.21, 0.23]) and the pan (world ≈ (0.38, 0.20, 0.27) with the beam up).

- **Bowls**: rim pinch-grasp — the 10 mm wall thickness fits the parallel gripper; grasp the
  near rim from above, lift vertically (clears the 18 mm pan rim), transport at z ≈ 0.35.
- **The weighing loop** maps 1:1 onto the teleport phases: place a bowl gently on the pan
  (release ~1 cm above, exactly the solve's release), retract clear of the beam sweep, *watch*
  the beam (the ±8° swing moves the pan center ~5 cm vertically — visually and
  proprioceptively unambiguous), and if it stays up, re-grasp the bowl (pan rim leaves the
  full bowl rim exposed) and set it back on the ground.
- Nothing requires a second arm, regrasping in-air, or force control beyond gentle placing;
  the balance is the sensor, so no wrist F/T is needed.

## Execution order

No fixed order is required or checked. Any weighing order works; bowls may be probed in any
sequence and even skipped (a lucky first guess is a legal success). The only requirements are
end-state: exactly one (necessarily the heavy) bowl upright on the pan, beam at its down stop,
other bowls grounded, everything settled — plus the latched partial credits, which are
order-free.

## Checks (smoke.py — 16 printed checks, `SIM_GEN_SMOKE: ALL PASS 16/16` on the forge)

1. Reset settles, no NaNs, beam pan-up, all bowls grounded upright.
2. Hidden-mass readback: PhysX masses == sampled table; exactly one heavy, two light.
3. Score ≈ 0 at reset, no latches.
4. Randomization across seeds: heavy index varies, heavy mass spread, slot permutation moves,
   beam start angle varies within band.
5. Null policy 240 steps: score stays ≈ 0.
6.–8. Oracle on seeds 0/1/2: heavy bowl on pan → success, score 1.0, persists 240 steps.
9. Monotonic ladder: idle < weighed (0.25) == after-removal (latch) < success (1.0); partials < 1.
10. **Seed-strategy negative**: a *light* bowl placed on the pan settles, beam stays up — no
    success, score ≤ 0.255.
11. Both light bowls piled on the pan: beam tips, but exactly-one clause rejects.
12. Bowl parked on the counterweight arm (+ heavy panned): rejected.
13. Heavy bowl on the bar inboard of the pan: outside pan band, rejected; inverted heavy on the
    pan: `beam_down` but upright clause rejects; discrimination: re-resetting one seed and
    panning each bowl alone, *exactly* the heavy one tips.

(The numbered items above group into 16 individual printed PASS lines — e.g. item 1 is three
separate checks, item 13 is three. Verified on the forge: `SIM_GEN_SMOKE: ALL PASS 16/16`;
solve verified `SIM_GEN_SOLVE: SUCCESS` on seeds 0 and 1.)
