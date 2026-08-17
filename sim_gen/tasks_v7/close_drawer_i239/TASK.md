# close_drawer_i239 — Drawer Re-Rail (re-seat a fallen drawer)

## Seed provenance

Derived from **`rlbench/close_drawer`** (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/close_drawer.py`):
an articulated cabinet whose drawer rides a prismatic joint; the whole skill is one
guided push on the drawer front until the joint reads closed.

## What changed and why it is strategically different

The seed's drawer is *captive*: the articulation does the aiming, the robot supplies a
single 1-DoF translation along a rail that already exists. Here **there is no joint and
no pushable drawer to begin with** — the drawer box has *fallen out* of the cabinet and
lies **on its side on the ground** in front of it. The skill inverts from "operate a
mechanism" to "**re-capture a free body into a fixture**":

1. **retrieve** the fallen drawer from the ground — a real 6-DoF reorientation (it
   starts on its side, random heading), not a yaw fix;
2. **discriminate**: the cabinet has TWO identical open bays stacked vertically; a green
   BEACON lamp mounted over one bay's mouth marks the drawer's home (which bay is
   marked is sampled per episode — the other is a decoy);
3. **insert** the drawer upright, handle out, through the marked bay's mouth (8 mm/side,
   34 mm headroom — a real aperture constraint) and **slide it home** along the bay
   floor until its front face sits flush at the cabinet front (the back wall is the
   hard stop, and the hard stop *is* the flush pose by construction).

Executing the seed's whole strategy for real — pushing the drawer along the ground
toward the cabinet — merely jams it against the plinth below the mouths and scores ~0
(smoke check 5 proves this with a real force-limited shove).

Contrast with the nearest corpus neighbours examined: `close_drawer_i58` ("Return
Dock") is a *never-touch-the-tray* interlock-removal task where gravity closes a
self-closing tray; here the judged body IS the drawer, it is manipulated directly, and
nothing is self-closing. `pen_holder`-style packing tasks fill a passive open container;
here the container-shaped object is itself the payload and must be threaded into a
fixture aperture and seated against a stop. No corpus task combines ground recovery +
6-DoF re-orientation + marker-based bay discrimination + aperture insertion to a hard
stop.

## Scene

Fully procedural (compound spawners, no external assets):

- **Cabinet** (kinematic; free yaw ±180° + xy jitter per episode): solid plinth
  (0.10 m tall), two side panels, back wall, mid shelf, top slab → two identical open
  bays, mouths 176 × 104 mm, floors at z = 0.100 / 0.234, depth 0.210 m.
- **Drawer** (dynamic, 0.18 kg): open-top box 200 × 160 × 70 mm, amber front wall,
  yellow handle bar standing 40 mm off the front face on a centre post.
- **Beacon** (kinematic): green emissive cube re-mounted over the *marked* bay's mouth
  at every reset.

Randomization (readback-verified by smoke): cabinet yaw + xy, WHICH bay is marked,
drawer spawn spot on the apron, WHICH side it lies on, its yaw.

Geometric honesty contract (asserted in `DrawerRerailSceneCfg.__post_init__`): the
drawer fits the mouth with ≥ 6 mm/side + ≥ 30 mm headroom; `seat_x` (hard-stop pose)
lies inside the flush window; a backwards (handle-first) drawer stands ≥ 22 mm proud of
the flush window; an upside-down drawer still *fits* (so the upright clause is
load-bearing, not vacuous); the two bays' credit bands and the ground band never
overlap.

## Rubric

Latched partial credit + live success (score exactly 1.0 iff `success()`):

| weight | clause |
|-------:|--------|
| 0.15 | **hold** (latched): drawer carried upright, clear of the ground, into the marked bay's approach window |
| 0.55 | **insertion** (latched max fraction): depth into the MARKED bay, gated to upright/facing/centred/bay-band poses |
| 1.00 | **success** (live): resting on the marked bay's floor, upright, handle out, front face flush (±12 mm), centred, settled, finite |

Non-success cap 0.70. Null policy ~0; seed strategy ~0; decoy bay ~0.

## Teleport solution (`solve.py`)

Teleports are transport only; every load-bearing interaction is applied force:

- **P0** settle + layout readback (cabinet pose, marked bay, beacon height, drawer
  fallen on its side); baseline score ≈ 0.
- **P1** transport: teleport to an upright hover fully OUTSIDE the mouth (front face
  35 mm proud), then a PD "grasp hold" (gravity feedforward, 6 N clamp) stabilises;
  hold latch fires. Score 0.15.
- **P2a** traversal (force): with weight support lowered to 15 % the drawer rides the
  bay floor under real friction while a gentle horizontal PD (kp 40) drives it through
  the mouth; it stalls where kp·err = μN, ~10 mm short of the back wall.
- **P2b** press home (force): from ~10 mm out a stiff PD (kp 120) aimed 4 mm past the
  wall presses through friction to the hard stop (a stiff gain from far away is wrong:
  the in-flight sag crashes the drawer into the plinth — found and fixed on the forge).
- **P3** release: forces zeroed, drawer settles seated; success → score 1.0; ≥ 3.3 s
  hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0, 1 (lower bay marked, different yaws/sides) and seed 2
(upper bay marked) all reach `SIM_GEN_SOLVE: SUCCESS` with non-decreasing scores
0.00 → 0.15 → 1.00 → 1.00.

## Embodiment argument (Franka, no required base motion)

Base pose: on the apron, ~0.55 m in front of the cabinet's front face, centred. All
interaction sites lie in a 0.10–0.45 m horizontal band at heights 0.03–0.30 m — inside
the Franka envelope from one base pose.

- **Drawer**: 0.18 kg — far below payload. Graspable by the 12 mm yellow handle bar
  (parallel-jaw across the bar), by the 8 mm-thick open-top walls (jaw across a wall
  rim, the natural grasp for the reorientation from the side), or by the 70 mm front
  wall face-pinch. Lying on its side the top rim faces sideways at z ≈ 0.08 m — an easy
  lateral rim pinch, then wrist rotation rights it in free air.
- **Insertion**: mouths at 0.10–0.20 m / 0.234–0.338 m; clearances (8 mm/side, 34 mm
  headroom) admit a held drawer with closed-loop tolerance; the final slide is a push
  on the handle/front wall to the hard stop — no precision release required (the bay
  walls cage the drawer once inside).
- **Beacon**: perception-only (bright green, emissive, on the front face) — never
  touched.

## Execution order

No required order beyond the inherent dependency: retrieve → orient → insert → seat.
There is no other object and no interlock; nothing forbids re-grasping.

## Checks

- `solve.py`: forge-verified `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1, 2 (both bays,
  both lying sides covered); scores non-decreasing across phase boundaries.
- `smoke.py`: forge-verified `SIM_GEN_SMOKE: ALL PASS 15/15` —
  1. settle/no-NaN (drawer on its side, score ~0);
  2. randomization A (cabinet yaw span > 90°, xy jitter);
  3. randomization B (marked bay varies, beacon tracks it every reset, lying side varies);
  4. null policy ~0;
  5. SEED strategy executed for real (ground shove) jams on the plinth, ~0;
  6. decoy bay: physically flush-seated in the unmarked bay → refused, ~0;
  7. upside-down seated in the marked bay (it fits) → upright clause refuses;
  8. backwards handle-first to the back wall → stands proud, facing clause refuses;
  9. near-miss 35 mm short of flush → no success, capped insertion credit (0.43);
  10. upright on the cabinet top → refused;
  11. upright on the ground in the approach lane → hold gates refuse;
  12. latched credit (0.58) survives theft to the ground, success does not;
  13. rejection audit (success never observed anywhere in the battery);
  14. final no-NaN;
  15. video: frames.npz (304 frames) recorded.
