# hutch_feed — open the hutch door, push the feed block in through the doorway, close the door behind it

**Seed:** `embodiedgen/put_banana`
(`sim_gen/RoboVerse/roboverse_pack/tasks/embodiedgen/put_banana.py`)
**Tier:** medium — **3 goals in a physically forced order** (doorway opened, block
inside, door re-seated).
**Execution order: REQUIRED — open → feed → close.** Open-before-feed is enforced by
physics (the seated door blocks the only way in; smoke #7 proves a bounded push against
the closed door achieves nothing). Close is necessarily last: success requires the door
re-seated WITH the block inside, and the close-credit term latches only in that
conjunction (closing an empty hutch earns nothing — smoke #12).
**Env name:** `simgen.hutch_feed` (scene `hutch_feed`, robot `null` — scene-level;
solve.py and smoke.py build this same env).

## What changed vs the seed

The seed asks for a free **aerial pick-and-place into an open-topped container**: grasp
the banana on a cluttered table, carry it through the air over the mug, release —
bounding-box containment, no obstruction anywhere on the path.

Here the container's affordance is inverted and a mechanism is put in the way:

- The container is a **roofed hutch — its top is sealed**. The seed's entire plan
  (approach from above, drop in) physically produces a FAILING state: the object lands
  on the roof and stays there (smoke #6 executes the seed strategy literally and pins
  it as a rejected outcome). "Into the container" stops being a free-space drop.
- The only way in is a **floor-level doorway** (90 mm wide, 85 mm tall) in one wall,
  and entry is a **ground-plane slide THROUGH an aperture** — the block is pushed along
  the ground under the lintel, never carried over anything. The transport verb changes
  from carry-and-drop to push-through-a-hole.
- At reset the doorway is **closed by a removable door**: a slab standing in an
  open-topped C-channel (wall behind, rail posts in front, stops at the sides). The
  scene has a **mechanism with a state**, absent from the seed: the door must be lifted
  straight OUT of the channel before anything can pass, and lowered back IN afterwards.
  This forces a three-phase plan with an irreducible order and makes the container
  opening itself a manipulandum with placement tolerances.
- The terminal state is **not "object in container"** but the conjunction "object in
  container AND container re-sealed" — the last action is on the door, not the object,
  so the seed's terminal relation alone is worth strictly less than half credit
  (smoke #10).

A solver therefore needs a different plan (a mechanism cycle around a ground push — no
aerial drop exists) and different code structure (door-channel extraction/re-seat plus a
through-aperture push controller — not a grasp/carry/release pipeline), not different
numbers.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects across free space; they never do the task. Every load-bearing
interaction goes through contact dynamics or a contact-respecting kinematic hold.
Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (hutch xy + doorway-heading yaw,
  door pose, block spawn, door mass from the physx view — custom-spawner MassAPI
  audit). Asserts door seated CLOSED, block outside, baseline score ~0.
- **P1 OPEN (kinematic lift + transport teleport)**: the door is pose-HELD each step
  (knob-grasp emulation, gravity-compensated vz=+g·dt) and raised straight up 10 cm out
  of the open-topped channel at 7 cm/s — a hold, not a jump, so any clash with the
  rails would fight real contacts. Then one transport teleport parks it lying flat at
  hutch-local (0.30, 0.30), dropped the last 2 cm, settled. Opening alone is worth 0.15
  latched credit; success still requires a CLOSED door at the end.
- **P2 FEED (contact dynamics — the core interaction)**: transport teleport stages the
  block on the doorway axis 26 cm out, OUTSIDE the hutch (a state a hand trivially sets
  up), square to the doorway. Then a velocity-regulated horizontal CoM push (≤ 0.9 N,
  0.12 m/s far / 0.07 m/s near, stiction floor 0.45 N) slides it along the ground
  through the doorway until fully inside; the passage under the lintel and between the
  jambs is pure friction + contact — the block is never pose-written past the aperture.
  A runtime progress probe toggles the force-frame encoding if the pod rotates applied
  wrenches by rotation-since-reset (not needed on this pod: mode 0 throughout).
- **P3 CLOSE (kinematic lower + gravity seat)**: transport teleport to a hover above
  the channel; pose-hold lowered INTO the channel at 6 cm/s to 18 mm above the seat —
  asserted NOT yet `door_closed_now` (outside the 12 mm height tolerance) — then
  RELEASED: gravity drops it the last stretch and it beds down between wall, rails and
  stops through real contact. success() first turns True here, judged on settled poses.
- **P4 persistence**: 3.3 more simulated seconds hands-off; only if success() held
  prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 (hutch yaw +174°) and 1 (yaw +202°): all phase scores
monotone 0.00 → 0.31 → 0.60 → 1.00, `SIM_GEN_SOLVE: SUCCESS` both.

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(0.0, 0.0, 0.0)` facing +x. The hutch centre sits at 0.52–0.58 m; its
doorway faces the base (heading π ± 28°), so the door channel is at ~0.42–0.48 m and
the block spawns at 0.17–0.35 m from the base — all inside the proven 0.30–0.71 m
ground-level comfort envelope (the block at worst spawns close-in, still reachable with
an elbow-up config).

**Door (grasped object):** the slab carries a dedicated **grasp knob on its top edge**
(36 mm long, 14 mm thick, 22 mm tall) at ~11–13 cm height — a canonical top pinch for
an 80 mm jaw (14 mm across the pads, full-depth engagement). The OPEN move is a pure
vertical lift of ~10 cm from a top grasp (no wrist reorientation), the SET-ASIDE a free
carry, and the CLOSE a lower-into-slot with generous funnels: 20 mm channel for an 8 mm
slab (~6 mm/side), ±25 mm lateral tolerance against 130 mm slab width, and the last
18 mm done by gravity after release — far above closed-loop OSC noise.

**Block (pushed object):** never grasped. It is pushed along the ground with fingertip
contact at ~2 cm height — the doorway is 85 mm tall, so the LAST stretch (under the
lintel) matters: with the block staged on the axis, the final push is a straight-line
20 cm stroke and the fingers never need to enter the aperture more than a fingertip's
depth — the block's trailing face is still outside (or in the mouth) when its centre
crosses the inside-margin line (centre 3.5 cm past the inner wall face ⇒ trailing face
at the wall plane, reachable through the 90 mm × 85 mm opening with a straight
horizontal wrist). 22.5 mm lateral slack per side forgives push-direction error.

**Hutch:** kinematic fixture, never needs to be touched. No other objects exist; every
required contact is one the arm can make.

## Success and rubric (physical outcomes only)

`success()` iff ALL, live and settled:
- block fully INSIDE the hutch interior: centre ≥ 3.5 cm past the inner wall faces on
  both axes, resting at ground height (±1.5 cm), |v| < 5 cm/s;
- door SEATED in its channel: centred on the doorway (x ±14 mm, y ±25 mm), at seat
  height (±12 mm), upright within 15°, |v| < 5 cm/s.

`score()` ∈ [0,1], latched each physics substep:
`0.15·opened (doorway ever cleared) + 0.20·block-approach-to-mouth (gated on opened) +
0.25·block-ever-inside + 0.25·door-closed-WHILE-block-inside`, capped 0.85;
**1.0 iff success()**; ~0 for the null policy (the door starts closed, so every term is
gated behind acting). Latched credit never evaporates (smoke #11); the solve's phase
prints are monotone.

## Randomization

Per episode (verified by sim READBACK in the smoke): hutch xy ±3 cm AND doorway-heading
yaw π ± 28° (the approach lane must be found, not memorized); the door pose follows the
hutch (spawns seated closed in every draw — smoke #4); block spawn distance
0.30–0.38 m out of the doorway, lateral ±7 cm, yaw ±180°.

## Check list (smoke.py — rubric REJECTION battery, 14 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite, door seated CLOSED in its channel, block outside, all still
2. settle: score ~0 at reset, no success
3. randomization readback: hutch xy jitter + doorway-heading yaw really move
4. randomization readback: block spawn distance + lateral vary AND the door spawns
   seated closed in every reset
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: the block dropped from above the hutch centre lands ON THE
   ROOF and stays there → not inside, score ~0, no success
7. out-of-order: the solve's own bounded push servo against the CLOSED door —
   demonstrably reaches the door (movement asserted, not vacuous) and is stopped;
   opened never latches, score ~0, no success
8. near-miss: door open, block settled just OUTSIDE the doorway mouth → NOT success,
   score < 0.9
9. straddle: door open, block settled IN the doorway (centre on the wall plane) → the
   3.5 cm inside margin rejects it, NOT success
10. open hutch: block fully INSIDE but the door left parked away → closed-door clause
    rejects, NOT success, score < 0.9
11. latched credit survives regression: dragging the block back OUT leaves the latched
    score unchanged, still no success
12. empty-hutch close: door lifted out and re-seated with the block still outside →
    `door_closed_now` holds but the close term stays unlatched, NOT success,
    score < 0.5
13. rejection audit: success() never True at any judged point of this battery
14. final no-NaN

N/A notes: **wrong object** — the scene has a single movable non-door object; identity
pressure is carried by the mechanism and aperture controls instead (#6/#7/#9). The
seed's literal strategy IS expressible here and is smoke #6; the seed's terminal
relation without the mechanism cycle is smoke #10.

Video frames are recorded throughout and saved to `frames.npz` in the working directory.
