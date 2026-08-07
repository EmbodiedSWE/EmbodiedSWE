# push_cube_i30 — Silo Scoop

Ladle the amber balls out of a deep fixed silo with a long-handled dustpan scoop,
dump them into the green-walled tray, then lay the scoop down clear of both
containers (scene `silo_scoop`, env `simgen.silo_scoop`).

## Seed provenance

Seed: `maniskill/push_cube` — a free 2 cm cube on an open table, the robot puts its
gripper behind it and pushes it a few centimetres along the tabletop into a marked
goal region; judged by planar proximity of the cube to the goal (`obj_to_goal_dist`
< radius). One object, one contact, one straight-line stroke on the support surface.

The seed's plan is expressible here — and is constructed and rejected: pushing the
reachable object (the scoop) along the floor to the goal container is smoke check 5
and scores ~0, because the payloads are at the bottom of a bolted-down 300 mm silo
that no planar push can reach into (the silo itself does not budge under 30 N,
proven in the same check).

## What the task is

Three procedural assets: a SILO — a kinematic slate-blue open-top bin, interior
240 × 115 mm and 300 mm deep (rim at 312 mm), with small 45° chamfers in the two
far interior corners; a TRAY — a kinematic green-walled basin (inner 160 × 160 mm,
50 mm walls); and the SCOOP — the only dynamic tool, a red dustpan (pan 48 × 92 mm,
28 mm sides, full-width ramp lip that scrapes the floor) with a vertical 400 mm
handle, 150 g. One or two amber 25 mm balls (15 g) start on the silo floor.

Per episode: silo xy + yaw, tray xy + yaw, scoop xy + free yaw (lying on its side),
the PRESENT ball count (1 or 2 — the judged subset), and each present ball's
silo-local position all randomize.

Goal: every present ball rests INSIDE the tray (tray-frame containment below the
wall top) and the scoop is STOWED — lying low on open floor, clear of both fixture
footprints — everything settled. The pan fits the silo mouth with 7.5 mm per side;
the handle keeps the grip above the rim at full dip depth (asserted).

## Strategic difference

- **vs. the seed**: the seed pushes the payload along the support toward the goal.
  Here the payload is UNREACHABLE by any planar push — it sits at the bottom of a
  deep, narrow, bolted-down container — and the goal is a different container across
  the floor. The solver must acquire a TOOL and execute a multi-phase plan: a
  constrained insertion (pan through the mouth, 7.5 mm per side), a shovel-
  against-the-wall capture (ramp wedges under the balls, pure contact), a nose-up
  heel-pivot tilt to trap them over the crest, an aerial CARRY of a loose payload
  riding in an open pan, a pour-out into the tray, and a tool stow. Different plan,
  different code (scoop-frame containment + 6-DoF tool control vs a planar push).
- **vs. corpus tasks read**: the pick-place family (libero bowls/pan, coke_task_i15,
  pick_and_lift_i16, living-room trays) transports grasped objects freely;
  close_box_i26 cages a block under an inverted crate and slides it;
  empty_dishwasher_i23 and pour_water_i7 pour/extract from containers the robot can
  tip or grasp directly. **No corpus task extracts payloads from a fixture that can
  be neither moved, tipped, nor reached into, via a carried tool whose pan must pass
  a constrained mouth** — tool-mediated extraction with a loose payload riding an
  open pan through free space.
- The mechanics are proven, not asserted: the shovel capture, the ramp-crest trap
  under tilt, the loose-cargo carry and the pour are each demonstrated by contact in
  solve.py, and the rubric's rejections (scoop-in-tray, stow near-miss, ball-in-pan,
  floor-drop, settle gate) are constructed in smoke.py.

## Solution outline (solve.py — the legitimacy certificate)

Teleports are TRANSPORT ONLY (free space to free space); every rubric-relevant fact
is produced by contact dynamics via external wrenches on the SCOOP only — no force
or teleport ever touches a ball:

1. **P0 settle + readback** — layout printed from readback; score ~0.
2. **P1 wrench-convention probe** — this stack's `set_external_force_and_torque
   (is_global=True)` applies the wrench rotated by the body's rotation since
   env.reset (pod-dependent). Single-step force impulses on the free-hovering scoop
   measure the applied-frame matrix M from dv responses; q_ref is recovered
   empirically and the encoding is verified closed-loop (hover + 60° pitch,
   drag-toggle fallback) before any real work.
3. **P2 trips** (repeat while a present ball remains): DIP — teleport to a hover
   over the guaranteed ball-free entry window, force-servo straight down through
   the mouth to the silo floor; SHOVEL — press down and stroke toward the far wall
   (two-speed, then a quasi-static PRESS with a COM-lever feedforward so the
   contact couple cannot pitch the scoop); TILT — 25° nose-up about the pan's rear
   bottom edge while still pressing (the wall blocks forward escape; the wedged
   ball climbs the tip), then DEEPENED in place toward 45° with a geometric
   wall-tracking x carrot (the origin advances by exactly the tip retraction
   nose_x·(1−cos), ~0.45 N bias, heel pinned at 2.5 N) so the tip edge rotates
   under the ball and it rolls back over the crest into the pan — a sustained
   far bias here instead braces the tip into the corner and the jammed scoop
   slides up the wall (observed); EXTRACT/CARRY — rise out of the
   mouth and glide to the tray, ball riding loose in the open pan; DUMP — staged
   nose-down pour with a wiggle fallback. Empty-pan and cargo-lost rechecks retry
   the trip.
4. **P3 stow** — teleport the empty hovering scoop to open floor (free space, 6 mm
   up), drop, settle. **P4 persistence** — ≥ 3.3 simulated seconds hands-off,
   success still holds, then `SIM_GEN_SOLVE: SUCCESS`.

Phases print `SIM_GEN_SCORE` at every boundary (non-decreasing, asserted).
**Verified on the forge: seed 0 (k=1) `SIM_GEN_SOLVE: SUCCESS` score 1.000 in
74 s; seed 1 (k=2, both balls captured in one trip and poured together) SUCCESS
score 1.000 in 92 s. The per-trip capture check retried empty trips exactly as
designed (2 empty trips before the successful one on each seed). smoke.py:
`SIM_GEN_SMOKE: ALL PASS 14/14` in 70 s.**

## Rubric

Latched partial credit (never evaporates), per PRESENT ball: 0.25 · ever OUT of the
silo volume + 0.45 · ever IN the tray; present-mean, cap 0.70; exactly 1.0 iff
`success()` = all present balls settled in the tray AND the scoop stowed (low, on
open floor, outside a clearance ring around both fixtures) AND everything settled.
Null policy ≈ 0 (the scoop's spawn is stow-legal but stow only pays through
success, which needs delivery).

## Embodiment argument (Franka, parallel-jaw)

- **Grasp**: the handle is a 14 mm cylinder (jaw span 80 mm) rising 0.43 m from the
  pan — a natural side pinch anywhere along it; the scoop is 150 g (payload 3 kg).
- **Dip + shovel**: gripping the handle ~0.35 m up, the pan reaches the silo floor
  while the wrist stays above the rim (asserted in cfg); the stroke is a ≤ 4.5 N
  horizontal press at ~6 cm/s and the hold-down is 2.5 N — far inside Franka's
  envelope. The ~0.6 N·m contact couple the grip must resist is what a rigid jaw
  purchase on a cylinder provides geometrically (TAU_MAX 2 N·m is honest).
- **Tilt/pour**: 25–60° wrist pitches; the wrist covers ±2.9 rad.
- Workspace: silo near (0.38, 0.26), tray near (0.38, −0.32), scoop spawn near
  (0.06, −0.03) (± jitter); everything within 0.65 m of a base at the origin, tool
  poses between the floor and 0.42 m.
- Execution order: trips may run in any order over any subset; only the final state
  is judged (stated in `describe()`).

## Checks (smoke.py — rejection battery, frames.npz recorded)

1. settle + no-NaN (present balls in the silo, scoop lying low, score ~0); 2.
randomization readback (silo/tray xy + yaw, scoop xy, ball positions differ across
seeds); 3. present-count coverage (k = 1 and k = 2 both sampled); 4. null policy
300 steps → score ~0; 5. SEED strategy: planar-push the reachable scoop to the tray
+ 30 N shove on the silo → balls untouched, silo unmoved, score ~0; 6. delivered
but scoop dropped in the tray → no success, score capped 0.70; 7. stow near-miss
(scoop against the silo, inside the clearance ring) → not stowed; 8. latched credit
survives the balls being teleported back into the silo; 9. floor-drop: balls out of
the silo but on open floor → escape credit only (≤ 0.30); 10. subset judging: a
k = 1 seed — the absent depot ball neither blocks nor inflates (score exactly
0.70); 11. ball resting in the PAN on open floor is not delivery; 12. settle gate:
a perfect stow pose still sliding at 0.45 m/s is refused; 13. rejection audit —
success() never True at ANY step of the battery; 14. frames.npz saved.

Run (forge):
`python -u -m simgen_tasks.push_cube_i30.solve --headless [--seed N]`
`python -u -m simgen_tasks.push_cube_i30.smoke --headless`
