# rocker_twin_drawers (i424) — load the drawer that starts shut, through the chest's anti-phase twin-drawer transmission

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/...py`). Seed strategy: grasp
the top drawer's HANDLE and pull it open, pick the chocolate pudding off the
table, drop it into the exposed cavity from above, then push the drawer shut.
End state: pudding inside, the whole cabinet closed.

## What changed / strategic difference

**vs. the seed.** Three seed pillars are removed or inverted:

1. **No pull, no handle — the goal drawer cannot be opened directly at all.**
   Both drawers of this chest are knobless with perfectly flush smooth fronts
   (front plates sit exactly in the chest's face plane, 4 mm perimeter gaps, no
   lip); the goal drawer starts SHUT on its inner joint stop. There is nothing
   to hook and nothing to pull. The ONLY way to expose its tray is to **push
   the OTHER (twin) drawer inward**: a hidden vertical-axis walking beam
   (rocker) inside the sealed carcass couples the two side-by-side drawers in
   strict anti-phase, so driving the open twin fully IN slides the shut goal
   drawer OUT through its aperture (~10.6 cm, tray exposed). The seed's
   opening move (pull the judged drawer by its handle) is physically
   impossible; its replacement is a push on a different object.
2. **A conservation law replaces free drawer state.** The captive pin-in-fork
   coupling enforces `ext_L + ext_R ≈ travel` at all times: *at most one drawer
   can be shut*. Consequently the seed's end state (everything closed) is
   unreachable — shutting the loaded goal drawer necessarily pops the empty
   twin back OUT through the beam. The episode exercises the transmission in
   BOTH directions (twin-in→goal-out, then goal-in→twin-out), and success is
   judged on the goal drawer + cargo only, with the twin standing open by
   design.
3. **The deposit window is mechanism-gated and closes again.** In the seed the
   drawer cavity is open-topped whenever the drawer is open and stays
   accessible. Here the tray is only exposed while the twin is held/parked in;
   after loading, the same anti-phase mechanism must be driven the other way
   (a push on the goal drawer's own front panel, cargo riding inside on
   nothing but friction) to reach the judged shut-and-loaded state.

**vs. every examined tasks_v7 neighbour.**

- `..._i87` (`rocker_cabinet`, seed `open_bottom_drawer`) is the near
  neighbour and the differences are deliberate and structural: i87 uses a
  **one-way disengaging pusher** (unilateral pad-on-plate contacts on a
  gravity-balanced horizontal-axle plank) between **stacked** drawers, driven
  in ONE direction once, with an **open-only displacement goal** on a drawer
  that is never touched, and no cargo anywhere. i424 uses a **captive
  bidirectional pin-in-fork inverter** (vertical-axis walking beam, pins
  trapped between fork plates — it pushes AND pulls) between **side-by-side**
  drawers, enforcing the `ext_L+ext_R≈travel` conservation constraint that i87
  fundamentally lacks (i87's drawers have free independent travel and its
  transmission disengages); the transmission is exercised in both directions
  within one episode; the goal is a full **deposit-and-close cycle** with a
  cargo object (open latch is only 0.15 of the rubric); and the judged drawer
  IS directly manipulated (pushed shut) in the final phase — the exact move
  that in i87 can never score.
- `..._i118` (`pudding_carousel`): rotating drum vault on a revolute axle —
  no drawers, no coupling, entirely different mechanism and goal topology.
- `..._i332` (`drawbridge_vault`): hinged falling-flap vault — rotary
  gravity mechanism, no prismatic pair, no anti-phase constraint.
- `libero_pick_ketchup_i104` (`rocker_lock`): gravity-biased see-saw ferry
  behind a letterbox window — the rocker there is a carrier/lock the cargo
  rides on, not a motion inverter between two judged-relevant drawers.
- No other task in `tasks_v7/` couples two prismatic bodies through a captive
  rotary link, and none has the "closing the goal necessarily opens its twin"
  invariant.

## Scene (`rocker_twin_drawers`, env `simgen.rocker_twin_drawers`, robot="null")

Procedural geometry only. A kinematic chest carcass (plinth, three front
pillars framing two apertures, lintel, side walls, rear wall, roof — sealed:
every gap is smaller than the 48 mm cargo cube) contains three dynamic
compound bodies, joined to the plinth by per-env spawn-authored joints
(collision-filtered only between joint partners):

- **Left / right drawers** (prismatic X, stroke −0.110→0 m each): tray floor
  160×130 mm, side + rear walls, flush knobless 155×113 mm front panel (teal L,
  orange R), and a two-plate fork on the tray roofline that captures a rocker
  pin with 4 mm backlash.
- **Rocker** (revolute Z at (0.455, 0, 0.143), limits ±33°): a walking beam
  with two downward pins at ±118.5 mm that ride in the drawer forks. Pin
  kinematics give `ext_L + ext_R = travel` (0.110 m) across the whole swing;
  MassAPI CoM at the pivot → no stored energy. Explicit viscous damping on all
  three joints (c·dt/I ≪ 1) parks the assembly wherever the push leaves it.
- **Cargo**: 48 mm dynamic "chocolate pudding" cube, spawns on the ground in
  front of the chest (seeded x, y, yaw).

Randomization (readback-verified): which side is the goal (shut) drawer, and
the cargo spawn pose. Geometry contract asserted at import time in
`smoke.audit_geometry`: panel flush with the face plane; aperture clearances
exceed summed contact offsets; pin throw = travel/2 with the pin inside the
fork slot across the full swing; carcass gaps all < cargo size (sealed); the
exposed open tray (86 mm) fits the cube with margin; the tray-band test volume
lies strictly inside the tray interior.

**Success** (live state, no memory): cargo inside the goal drawer's tray band
∧ goal drawer flush (`target_ext < 0.008 m`) ∧ everything settled ∧ finite.
**Score**: latched partial credit — open 0.15 (goal drawer driven ≥ 0.085 m
out via the transmission), load 0.35 (cargo settled in the goal tray), success
0.50; exactly 1.0 iff `success()`. Latches clear on `reset()`.

## Solution (`solve.py`) — one transport teleport, everything else contact dynamics

1. **P1 open (pure joint/contact dynamics)**: a velocity-servoed world-X force
   (cap 10 N, slide speed cap 0.10 m/s) on the open TWIN's front panel drives
   it fully in; the walking beam inverts the motion and slides the shut GOAL
   drawer out ~10.6 cm. Drive cut at the inner stop; damping parks the
   assembly. Score 0.15.
2. **P2 load (the only teleport — transport only)**: one pose write moves the
   cube from its floor spawn to a hover 4.9 cm above the exposed goal tray
   (free air, outside every collider, under the lintel with clearance).
   Gravity lands it; seating is live contact. Score 0.50.
3. **P3 shut (pure joint/contact dynamics)**: the same servo pushes the GOAL
   drawer's own panel back in, cargo riding on friction alone; the twin pops
   back out through the beam (by design). Drive cut at the stop. Score 1.0.
4. **P4 persistence**: all drive buffers asserted zero, ≥ 3.5 simulated
   seconds hands-off, `success()` re-checked on the live state, then
   `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary; monotonicity asserted.
Forge: seeds 0 and 1 both `SIM_GEN_SOLVE: SUCCESS`, rc=0 (~21 s each),
score stream 0.000 → 0.150 → 0.500 → 1.000.

## Franka embodiment argument

Base pose: on the floor at ≈ (−0.15, 0, 0), facing +x toward the chest face at
x = 0.306; all interaction points lie 0.20–0.42 m ahead at heights
0.03–0.17 m — the near-center of the Franka workspace.

- **Drawer pushes (P1, P3)**: each is a straight horizontal −x…+x push on a
  flat 155 × 113 mm front panel at ≤ 10 N against ~6 N·s/m viscous drag — a
  closed-gripper knuckle push, the easiest Franka primitive; no grasp, no
  wrist reorientation. The two panels are 210 mm apart in y, both trivially
  reachable from the single base pose.
- **Cargo pick-and-place (P2's real-world counterpart)**: a 48 mm cube in the
  open — a canonical top grasp for the parallel-jaw gripper (opening
  80 mm > 48 mm), lifted over the 86 mm-exposed open tray and released from a
  few cm up; the solve's hover-drop is exactly this release.
- **Nothing requires pulling, hooking, or two-handed coordination**; the
  knobless fronts are unopenable by design and the task never asks for it.

## Execution order declaration

The scene was written first with the minimal success predicate; `solve.py` was
then run on the forge and passed physically on seeds 0 and 1 (rc=0, SUCCESS);
only after the solve numbers were in hand were the rubric anchors confirmed
(open threshold 0.085 vs observed 0.102 m emergence; tray band vs observed
seated pose local (−0.046, 0.000, +0.028)) and `smoke.py` finalized. The one
smoke iteration fixed a *probe* bug, not the rubric: the smoke servo's approach
ramp targeted `cut_ext` itself, so the drawer asymptoted at the threshold and
the helper timed out even though every judged state was already correct;
restoring solve.py's ramp-to-zero law fixed all five affected probes. No check
was weakened at any point to make a run pass.

## Smoke battery (`smoke.py`) — rejection/health checks

0. **Import-time geometry audit** (fail-fast, before Kit): flushness, aperture
   clearances, pin/fork kinematics + backlash, sealed-carcass gaps < cargo,
   exposed-tray fit, tray band containment.
1. **settle/no-NaN** — goal shut on its stop, twin out on its stop, cargo in
   its spawn band, score < 0.05.
2. **randomization readback** — across seeds: both goal sides observed, cargo
   spawn poses differ (max pairwise > 3 cm), drawers posed consistently with
   the sampled side.
3. **null policy** — 240 idle steps: nothing latches, score < 0.05.
4. **roof drop rejected** — cube dropped from above the closed goal drawer:
   the sealed carcass keeps it out; no load latch, no success.
5. **shut-drawer push rejected** — 10 N push on the already-flush goal panel
   for 1.5 s: moves < 1 mm (inner stop), no latch — the seed's "just push it
   shut" reflex scores nothing without the full cycle.
6. **wrong-drawer deposit rejected** — cube genuinely loaded into the TWIN's
   tray and the twin pushed shut: load latch stays 0, score ≤ 0.16 (the shut
   twin re-opens the goal, so only the open latch can fire).
7. **near miss** — real transmission drive stopped early: goal drawer ajar
   (1–4 cm, not flush), cargo in tray → score pinned at 0.50, no success.
8. **belly smuggle rejected** — cube teleported under the drawer bodies inside
   the carcass, goal pushed shut over it: cargo ends bulldozed behind the
   tray (outside the band), load latch 0, score ≤ 0.16.
9. **exactness** — score = 1.0 iff success (verified at the one success-shaped
   construction and nowhere else).
10. **reversal** — from the shut-loaded pose, driving the twin back in
    re-opens the goal: success is live state, not a memory.
11. **video** — ≥ 20 rgb frames → `frames.npz` in CWD.

Forge: `SIM_GEN_SMOKE: ALL PASS 11/11`, rc=0 (~66 s), frames.npz
(280 × 500 × 800 × 3) saved.
