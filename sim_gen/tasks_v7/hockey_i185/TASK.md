# hockey_i185 — TipFeeder: load, tip and release the hopper to feed the ball through the elevated window

`simgen.tip_feeder` — single-arm tabletop-scale manipulation, NullRobot physics
package (scene + rubric + teleport solve + rejection smoke).

## Seed provenance

Derived from **rlbench/hockey**: *grasp the hockey stick and strike the ball into
the goal* — one ballistic tool swing at a passively available, open-mouthed,
floor-level receptacle.

## Strategic difference

The seed's plan is "acquire a striking tool, hit the ball toward an opening that is
already reachable". Every element of that plan is removed or inverted here:

- **The receptacle admits nothing passively.** The goal becomes a fully SEALED BOX
  (interior 16 x 18 cm, roof at 25 cm) whose only ball-sized opening is an
  ELEVATED WINDOW (sill 13 cm up, 8.5 cm tall, 13 cm wide) in the near wall. A
  ball shot, rolled or thrown along the floor bounces off the sealed lower wall
  (smoke check 6 fires the seed's shot and watches it bounce off). Nothing can be
  dropped in from above either: the roof seals the top and a HOOD (top + side
  plates) overhangs the window tunnel (smoke checks 7-8).
- **No tool, no strike.** The only path through the window is MECHANICAL: an
  open-top TIPPING HOPPER hinged between the cheek plates of a heavy stand, its
  low-lipped mouth facing the window at sill height, gravity-held 12 deg back on
  its seat. The plan is *load the payload into the mechanism* (place from above —
  the one thing the seed never does is place the ball), *actuate a revolute joint
  through a handle bar* (pull up and forward, hard stop at +42 deg), *let gravity
  deliver* (the ball runs down the tilted basin, hops the retaining lip, crosses
  the sill, drops through the window), then *release* (the hopper gravity-returns
  to its seat; success is judged on the settled, hands-off machine).
- **Different code structure:** hinge-angle readback + load/tip/release phase
  control replaces grasp-stick + swing; nothing is ever struck.

Within tasks_v7 there is no other tip-feeder: i325 (the other hockey derivative)
is gate-extraction + floor-level push into an open goal mouth — here the floor
route is exactly what the sealed wall forbids, and the delivering actor is a
hinged mechanism, not the ball-pusher.

## Scene

Fully procedural compound spawners (no external assets):

- **box** (kinematic): 4 walls + roof, window aperture in the -x wall, sill shelf
  and hood framing the window tunnel. Origin at the interior floor centre.
- **stand** (dynamic, 33 kg): base slab + two cheek plates. Heavy-dynamic, never
  kinematic — a kinematic joint body0 would leave the hinge anchor world-fixed
  after the randomization teleport.
- **hopper** (dynamic, 0.25 kg): basin floor, low front lip, side walls, tall back
  wall, strut and 16 mm yellow handle BAR; origin ON the hinge axis; mass, CoM
  (behind the hinge — real gravity return) and diagonal inertia authored via
  MassAPI. Spawn-authored USD revolute joint (body0 = stand, axis Y, limits
  -12/+42 deg); the joint pair is collision-filtered by PhysX default, geometry
  keeps 13 mm clearance. Every forward-sweeping part stays within |y| <= 58 mm so
  the hopper mouth passes THROUGH the 130 mm window aperture over the full hinge
  travel (a 136 mm-wide first cut jammed its side-wall corners on the hood side
  plates at +11.7 deg — swing radius 55 mm vs the 14 mm hinge-to-apron gap).
- **ball**: white sphere, 54 mm, 60 g.

**Randomization (readback-verified in smoke 3-4):** the whole apparatus (box +
stand + hopper — one coherent linkage write, per the teleport-vs-linkage rule)
gets a planar offset (±3/±5 cm) and yaw (±8 deg) about the hinge point; the ball
spawns on a Bernoulli LEFT/RIGHT side with xy jitter (torch.rand comparison, not
the degenerate first randint).

## Rubric

Latched partial credit (transient achievements keep credit), non-success cap 0.70:

| credit | clause |
|---|---|
| 0.20 | ball ever in the hopper basin (hopper-frame readback) |
| 0.15 | hinge ever past +20 deg (real actuation) |
| 0.35 | ball ever inside the box below the sill (box-frame, z < 0.10 — containment judged BELOW the aperture, so window-tunnel perches never count) |
| 1.00 | iff success(): ball at rest on the box floor inside AND hopper seated (< -6 deg) at rest |

## Teleport solution (solve.py)

Teleports are transport-only; every load-bearing interaction is contact dynamics:

- **P0** settle + layout readback (stand pose, yaw, seat angle, ball slot).
- **P1 LOAD** (the one teleport): ball staged just above the open basin top —
  the arm's place-from-above — falls in and rolls to the back wall by contact.
- **P2 TIP** (wrench-driven): body-frame torque about the hopper's own Y (= the
  hinge axis — invariant to the pod's wrench frame drag) under PD + gravity
  feedforward, hinge rate by finite difference, with a STALL PROBE that escalates
  the torque clamp (2 -> 8 N*m) as a guard against pod-side wrench
  under-application. The ball's delivery is entirely gravity + contact; if it
  delivers below +20 deg the full handle pull is finished with the empty hopper
  so the tilt credit reflects real actuation. Fallback fingertip nudge through
  the window only if the ball perches in the tunnel.
- **P3 RELEASE**: torque ramped down and CUT; gravity reseats the hopper;
  hands-off from the cut to the verdict.
- **P4** persistence: >= 3.3 simulated seconds hands-off after success(), then
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at each phase boundary, asserted non-decreasing.

## Embodiment argument (single Franka, parallel jaw, OSC, one base pose)

Base at the world origin facing +x; everything actionable lies 0.15-0.60 m out at
0-0.35 m height, inside the comfortable workspace:

- **Pick the ball:** 54 mm sphere < 80 mm jaw opening, on open floor with
  clearance on all sides.
- **Load:** the basin top is OPEN between the cheek plates (142 mm inner span,
  basin interior 100 mm wide vs the 54 mm ball); place-from-above at ~0.36 m
  reach, 0.2 m height.
- **Actuate:** the 16 mm handle bar (jaw-sized) sticks back over the back wall,
  clear of everything; its pull arc runs roughly from (0.24, 0, 0.18) to
  (0.33, 0, 0.33) world — a short up-and-forward wrist arc; the 2-8 N*m hinge
  torque at the 0.22 m handle lever arm is a 9-36 N hand pull.
- **Release:** open the jaw; gravity does the rest. No bimanual step, no regrasp
  under load, no reach outside the frontal workspace.

## Execution order is physics-forced, not rubric-timestamped

Load must precede tip (an empty tip delivers nothing; the ball cannot enter the
basin while tilted forward because the basin floor then faces the window, not the
sky). Tip must precede delivery (the sealed wall + hood reject every other route —
proven by smoke 6-8). Release must follow delivery (success requires BOTH ball
inside and hopper seated; holding the handle keeps the hopper off its seat —
smoke 12). No rubric clause consults time.

## Checks (smoke.py rejection battery, 15 checks)

1. settle/no-NaN + MASS READBACK (custom spawners ignore cfg mass_props);
2. reset score ~0; 3-4. randomization readback over 8 seeds (side flips, jitter,
yaw/offset spreads); 5. null policy ~0; 6. SEED-STRATEGY end state: ball fired
along the floor bounces off the sealed wall, ends outside, ~0; 7. roof drop rests
on the roof; 8. hood drop: the hood DEFLECTS the ball (observed: into the open
hopper basin below — the honest 0.20 load credit) and it never enters the box;
9. load-only = exactly the 0.20 band; 10. near-miss: ball resting on the sill
above the inside_z gate — no delivery credit (judged on a short contact-settle:
the perch is metastable and a long settle would creep through the window into the
goal); 11. wrong place: against the outside wall — box-frame math rejects;
12. held-tilted: ball inside but hopper servo-held at +30 deg — seat clause alone
rejects; 13. latched credit unchanged after the ball is removed (<= 0.70 cap), no
success; 14. never-success audit; 15. final no-NaN. Frames recorded to frames.npz.

## Files

- `scene.py` — TipFeederScene, `SCENES.register("tip_feeder")`, `register_env`
  (robot="null").
- `solve.py` — `python -m simgen_tasks.hockey_i185.solve --headless [--seed N]`.
- `smoke.py` — `python -m simgen_tasks.hockey_i185.smoke --headless`.
