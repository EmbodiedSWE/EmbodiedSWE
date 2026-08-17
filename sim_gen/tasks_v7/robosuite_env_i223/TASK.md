# robosuite_env_i223 — Slider Gauntlet

**Env:** `simgen.slider_gauntlet` (scene `slider_gauntlet`, robot `"null"`)
**Seed:** `robosuite/robosuite_env` (RoboVerse `roboverse_pack/tasks/robosuite/robosuite_env.py`)

## What the task is

A gray puzzle board (position + heading sampled per episode) carries a straight
MAIN CORRIDOR (64 mm wide) sunk between walls and covered by an opaque ROOF with
a 22 mm slot along the centreline. The corridor runs from a closed back wall to
an open EXIT EDGE; past the edge a green CATCH TRAY sits 30 mm lower, on the
ground. The BLUE RUNNER — a 60×56×30 mm block whose only handle is a 16 mm mast
sticking up through the slot — starts near the back wall. It is CAPTIVE: the
slot is far narrower than the block, so it cannot be lifted out anywhere; it can
only be dragged. Two stations cross the corridor at right angles with
open-topped SIDE CORRIDORS (50 mm wide). At each station an ORANGE CROSS-SLIDER
(40×88×30 mm, mast-handled, also captive in its own crossing) seals the main
corridor. One arm of each side corridor is filled by a fixed RED PLUG — **which
arm is sampled per episode** and readable from above (the arms are open-topped);
the other arm is an empty pocket.

Goal: shunt each cross-slider fully into its OPEN pocket (pushing toward the
plug jams after ~6 mm), then drag the runner down the corridor, past both
stations, out over the exit edge so it drops and rests INSIDE the catch tray.

**Execution order: PARTIAL, physically enforced.** Each station's slider must be
cleared before the runner can pass THAT station; the two sliders may be cleared
in either order, and the runner may be advanced between clears.

## Seed provenance and why this is strategically different

The seed wraps the five robosuite oracle tasks (Lift, Stack, Door, PickPlaceCan,
NutAssemblySquare). Every one — and every oracle plan — has the same shape: an
exposed FREE object is grasped in open space and carried to its goal pose in a
single unobstructed transport; the manipulated object's own pose IS the
predicate. This task breaks that manipulation model wholesale:

- **Nothing can be picked up.** Every moving piece is captive under static
  geometry (the roof / its crossing); the only affordance is DRAGGING by a mast.
  The seed's one skill — grasp-and-carry through free space — is physically
  impossible here (smoke check 4 pulls the runner upward at 4× its weight; the
  roof holds it).
- **The goal object's path is obstructed, twice, by other movable objects.**
  The core skill is obstruction reasoning: notice the runner cannot pass, find
  the blocker, move the blocker somewhere legal, THEN move the goal object. No
  seed task has any obstruction chain at all (Door is a single articulated DOF
  standing alone).
- **A per-episode discrete perception bit gates each sub-move.** The open-arm
  side is sampled per station; pushing the memorized direction jams against the
  plug (smoke check 6). A memorized trajectory cannot work — 4 discrete layouts
  × continuous board pose × runner depth.
- **The order is a physical partial order**, not a rubric clause: the sealed
  station stops a hard runner shove (smoke check 5), and clearing it is the only
  way through.

Also deliberately unlike the other corpus tasks examined while building it:
sibling i175 (same seed) is discrete MASS decomposition read out by a balance
mechanism's equilibrium angle — here there is no emergent mechanism readout, no
mass reasoning, and the interlocks are kinematic-geometric; setup_checkers_i39
is tool use (a pry lever) — no tool exists here; pen_holder / close_box are
free-object containment/closure. None of them contains a movable-obstacle
gauntlet with sampled dead-ends.

## Mechanics

One KINEMATIC compound board (plate, wall grid, slotted roof, tray walls — ~30
authored box collision slabs, all procedural) plus three dynamic compound pieces
(body box + mast cylinder, authored MassAPI mass/CoM/inertia, vel iters 4,
authored friction material) and two kinematic red plugs. No joints, no drives:
every interlock is plain rigid-body containment. Clearance ledger: runner body
60 mm > crossing width 50 mm (cannot turn off the corridor); roof underside
70 mm sits 10 mm over every 60 mm-tall piece top (cannot climb or be lifted);
slot cross 22 mm ≪ 56 mm body (captive at junctions too); a blocker seals the
corridor until its centre is 76 mm off-axis (clear gate 80 mm, pocket admits to
~112 mm, plug jams at ~6 mm). The runner's start band keeps its rear face 5 mm
clear of the back wall (a flush start let depenetration latch phantom progress —
found on the forge and fixed).

## Rubric (score 0..1, anchored in the solve trajectory)

- **0.20 · clear_A + 0.20 · clear_B** — station cleared: the blocker no longer
  seals its crossing, latched only through a 20-substep quiet streak (the latch
  survives re-sealing — credit for having opened the way; smoke check 10).
- **0.25 · progress** — latched furthest runner advance from its sampled start
  toward the exit, gated to the corridor band (null policy: 0.000 measured).
- **1.0 iff success()** — the runner RESTS inside the tray: board-local x in the
  tray span, |y| inside the walls, body centre BELOW the corridor floor level
  (it physically dropped out of the maze), runner + blockers settled, finite.
  Non-success capped at 0.65; success is live (revoking it drops the score back
  to the latched 0.65 — smoke check 11).

## Solution outline (solve.py — no teleports at all)

Every action is a world-frame velocity-servo force at the piece's CoM — the
horizontal pull a gripper holding the mast would exert (kp·dt/m ≈ 0.5 for
stability against the one-substep wrench delay; stall force kp·v_des ≈ 1.4× the
drag friction, v_des escalates on stall; 2.5 N cap is far below every tipping
threshold). Read each station's open side (the bit a camera reads from the
open-topped arms), servo each blocker to ≥ 95 mm off-axis, release, wait out the
streak latch; then servo the runner along the corridor until it passes the lip
and cut the force — gravity and momentum drop it into the tray. Settle, verify
success, hold hands-off ≥ 3.3 s, print `SIM_GEN_SOLVE: SUCCESS`. Verified on
forge seeds 0/1/2 (yaws +83.9°/+113.4°/+85.5°, open sides [−,+]/[−,+]/[+,+]):
scores 0 → 0.20 → 0.40 → 1.000 on all three, ~34 s wall clock each; seed 0
re-verified after the start-band fix (x0 −0.096, same score staircase), and
smoke check 11 runs the same solve inline on the final geometry.

## Embodiment argument (single Franka, parallel jaw)

- **Base pose:** on the ground at the origin, facing the board centre 0.40 m
  away (+x). Every mast the arm must hold lives inside a 0.29–0.55 m radius
  annulus at z 0.06–0.15 — comfortably inside the ~0.85 m Franka envelope, with
  open sky above the whole board (tallest static geometry: roof at 78 mm).
- **Grasps:** every handle is a free-standing vertical 16 mm cylinder ≪ the
  80 mm jaw. The runner's mast is proud of the roof by 57 mm (grip band z
  0.078–0.135); the blockers' masts stand fully exposed in the open-topped
  crossings (grip band z 0.060–0.150). Top-down or shallow-angle side grasp,
  nothing overhead to clear.
- **The moves:** three planar drags at fixed height — exactly what a stiff
  grasp on a mast gives. Drag friction is 0.5–0.7 N ≪ payload; the piece's own
  corridor guides the path (±4–5 mm lateral slack), so centimetre-level
  compliance suffices; the final move ends by simply releasing after the lip —
  gravity finishes the task.
- **Perception:** the red plugs are visible from above in the open-topped arms
  (the one discrete bit per station); the mast positions are visible through /
  beside the roof slot; the tray is open.

## Randomization (readback-verified)

Per episode: board xy jitter ±40 mm, yaw 90°±30°; open-arm side per station
~ U{L, R} (4 layouts; torch.rand comparisons only — first-randint degeneracy on
this stack); plug seated in the closed arm (physically verified each draw);
runner start depth ~ U(−0.125, −0.085); blocker centring jitter ±10 mm.

## Checks (smoke.py — 12, frames.npz recorded)

1. **settle** — seeded reset settles finite, both stations sealed, authored
   masses took (get_masses readback), score ~0, no success.
2. **randomization** — 8 seeds: ≥3 of 4 open-side layouts, plug physically in
   the closed arm on every draw, runner depth / board yaw / board xy spread.
3. **null** — 2.5 s of nothing: score ≤ 0.02, no success, runner stays put.
4. **CAPTIVITY** — 4×-weight straight-up pull: the runner presses into the roof
   (z demonstrably rises — the pull acts) but cannot leave; released it drops
   back; no credit. The anti-lift claim is physics, not fiat.
5. **ORDER interlock** — hard forward shove with stations sealed: the runner
   MOVES ≥ 15 mm then stalls against blocker A short of the station, never
   climbs (roof); tiny approach credit only.
6. **PLUG interlock** — blocker A shoved toward its red plug: moves, then jams
   ~6 mm off-axis; station stays sealed, no clear credit ever latches.
7. **near-miss** — both stations legitimately cleared, runner dragged to the
   exit lip but NOT out: score capped at 0.65, no success.
8. **wrong object** — a BLOCKER constructed on the tray floor, runner still in
   the corridor: no success (the tray clause keys on the runner).
9. **tray gates** — the runner constructed on the ground BESIDE the tray
   (y gate) and BEYOND the end wall (x gate): both refused.
10. **clear latch** — blocker A cleared (0.20) then pushed back to re-seal: the
    live state reads obstructed again, the latched credit survives.
11. **success + revocation** — full inline solve: success() with score exactly
    1.0, still true 1 s later; plucking the runner back into the corridor
    revokes success LIVE and the score falls to the latched 0.65.
12. **frames** — ≥ 20 video frames saved to `frames.npz`.

## Files

- `scene.py` — cfg + scene + registration (`slider_gauntlet`, `simgen.slider_gauntlet`)
- `solve.py` — `python -u -m simgen_tasks.robosuite_env_i223.solve --headless [--seed N]`
- `smoke.py` — `python -u -m simgen_tasks.robosuite_env_i223.smoke --headless`
