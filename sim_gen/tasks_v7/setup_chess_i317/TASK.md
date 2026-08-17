# setup_chess_i317 — Promote, then Punch the Clock

Stand the **ivory queen** upright inside the **gold promotion square** on the pedestal
board, then **press the raised button** of the mechanical **chess clock** beside the board
so its gravity-bistable rocker beam tips fully onto the other side — under the real chess
rule that the move must be completed BEFORE the clock is pressed (an early press is an
**illegal punch** that forfeits the episode permanently). Env id: `simgen.move_clock`
(scene-level, `robot="null"`).

## Provenance

- **Seed task:** `rlbench/setup_chess` — "set up the chess board": 32 chess pieces are
  picked off the table one by one and placed on their marked board squares
  (`RoboVerse/roboverse_pack/tasks/rlbench/setup_chess.py`; USD board + free-standing
  pieces, pure pick-and-place, no ordering, no checker in the ported config).
- **Kept from the seed:** the chess dressing (a checkerboard playing surface, an ivory
  piece, a marked goal square) and the placement-tolerance flavor of the seat predicate.
- **Replaced:** the plan. The seed's plan is placement, 32 times over; here there is
  exactly ONE placement, and the task is only complete when the move is *punched in* on a
  real chess-clock mechanism, legally.

## Why it is strategically different

| axis | seed `setup_chess` | this task |
|---|---|---|
| the goal act | placement is the whole task | placement is only HALF the task — success requires a **mechanism state**: the clock's rocker beam tipped to the far stop |
| mechanism | none (free pieces on a board) | a **gravity-bistable rocker** on a spawn-authored revolute joint (CoM 25 mm above the pivot, ±15° stops): an over-center toggle that holds its side (9.5e-3 N·m holding torque) and must be flipped through the hinge by a real press — smoke-proved that a sub-holding press cannot move it |
| ordering | none — 32 independent moves | **temporal-legality order judged by an event latch**: "move before you press". The end state alone CANNOT distinguish a legal from an illegal punch (both end seated + flipped + settled); the rubric tracks the event history like a chess arbiter — a foul latched at press time forfeits success forever |
| failure modes | drop/misplace | illegal punch (permanent foul), stand outside the square, queen lying down in the square, beam never flipped, success lost live when the queen is knocked off |

Also differs from the siblings read while working: `setup_chess_i121` (captive-slide
channel routing, geometry-enforced order, beacon decoy) and `setup_chess_i302` (ejection
through an aperture + gravity chute, **occupancy**-enforced order). Nothing here is slid,
ejected, or occupancy-gated — the mechanics are a bistable hinge mechanism, and the order
is enforced by declared RULE + event latch (an arbiter's call), a third, distinct
order-enforcement mechanism.

## Scene (fully procedural)

- **Board** (kinematic compound): 0.40×0.40×0.10 m pedestal slab, top painted as a
  visual-only 6×6 checkerboard (no tile colliders — no proud edge anywhere on the playing
  surface). Yaw ±20° + xy jitter ±4 cm per episode.
- **Frame** (kinematic): gold square OUTLINE, interior 90×90 mm, walls 6 mm thick and only
  4 mm tall, NO floor — the promotion square; teleported to a randomized board spot each
  reset (x ∈ [−0.10, 0.10], |y| ∈ [0.03, 0.11], side drawn).
- **Queen** (dynamic compound): flanged base Ø44×14 mm, shaft Ø24×75 mm, collar, gold ball
  coronet; m = 0.15 kg, authored base-weighted CoM (z = 12 mm) + diagonal inertia. Starts
  standing on the half opposite the frame, random spot + yaw.
- **Chess clock** at a FIXED world pose beside the board (jointed assembly, never
  teleported — kinematic-body0 joint anchors stay world-fixed after teleports):
  **ClockBase** (kinematic wood housing: block + two cheek plates carrying the pivot at
  z = 0.12) and **Rocker** (dynamic red beam 0.20 m, ivory button cap on the +y end, black
  on the −y end; spawn-authored RevoluteJoint, axis x, ±15° limits, CoM 25 mm above the
  pivot → gravity-bistable). The start side (which button is raised) is drawn per episode
  by an in-DOF quaternion write about the pivot. The rocker's ±43 mm sweep stays >40 mm
  clear of the housing — the joint limits are the only stops.
- Friction materials bound per-collider; restitution 0; solver iters 16/4; dt = 1/120.

**Randomization (readback-verified):** board yaw + xy jitter, promotion-square spot,
queen spot + yaw, rocker start side.

## Rubric (latched partial credit; success judged live)

- **0.45** `l_seat` — the queen stood upright (≤12°) with its base centre within 3 cm of
  the square's centre, resting on the board (z-window), while slow.
- **0.30** `l_flip` — the beam reached the far side (past 12°) while `l_seat` was already
  latched and no foul was on record.
- **FOUL** `_foul` — the beam left its start side (below 8°) before `l_seat`: latches
  forever, blocks `l_flip` and success.
- **1.0 iff `success()`** (else capped at 0.75): queen seated AND beam fully tipped to the
  far side AND no foul, everything settled, states finite.

Tolerances are honest by construction: a queen physically standing inside the 90 mm frame
interior has its base centre within 45−22 = 23 mm < 30 mm of the centre (every real
in-square stand counts), while a queen outside the frame is ≥ 51 mm off (wall outer face +
base radius). The rocker rests only at the ±15° stops; `flip_deg` = 12° sits 3° inside the
far stop and `foul_deg` = 8° is 7° of margin from the held stop, far beyond solver wobble.

## Solution outline (`solve.py`, demonstrated on forge)

The punch — the load-bearing interaction — is pure joint/contact dynamics: a body-frame
torque about the hinge axis (0.030 N·m ≈ a 0.36 N fingertip press at the 83 mm button
lever, 3.2× the holding torque) with a bang-bang ~0.8 rad/s speed governor drives the beam
off its stop, through over-center, onto the far stop; no teleport ever touches the rocker.

P0 settle + layout/mass readback (score 0) → P1 the one transport teleport: park the queen
20 mm ABOVE the promotion-square centre, release; the seating is a real contact settle
(`l_seat`, 0.45) → P2 press the raised button through the hinge until the beam tips, coast
onto the stop, settle → success, 1.0 → P3 ≥3.4 s hands-off persistence →
`SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (Franka, one base pose)

Base at ≈ (0.08, −0.40, 0) world, facing +y: every board cell (board centre ±jitter at
(0, 0), cells within |x|,|y| ≤ 0.24, top z = 0.10) and the clock buttons (world
(0.44, ±0.083, 0.145)) lie in a 0.30–0.75 m reach disc at comfortable working heights.
Per-object contact strategy: the queen exposes a Ø24 mm shaft standing 75 mm above the
board — a parallel-jaw side-grasp, lift, carry, and lower over the square is exactly the
solve's transport-teleport-and-release; the raised clock button is a Ø36 mm cap facing UP
at z ≈ 0.145 on a fixed, never-randomized clock — a single fingertip press straight down
(~0.4 N) is exactly the solve's hinge wrench. The board, frame and clock housing are never
manipulated. All forces are far inside Franka payload; the two work sites are 0.44 m
apart, both in front of the base.

## Declared execution order

1. MOVE: stand the queen in the gold promotion square.
2. PUNCH: press the raised clock button so the beam tips fully to the far side.

The rule is declared in `describe()`/`instruction()` (chess legality: move before you
press) and judged by an event latch, exactly like an arbiter: smoke check 6 executes the
solve's own press BEFORE the move and then completes the move perfectly — the final state
is identical to the success state, yet the foul holds: no success, flip credit forfeit,
score pinned at 0.45.

## Validation evidence (all on the forge, Isaac Lab / RTX 4090)

- `solve` seeds 0 (black side starts raised) and 1 (ivory side starts raised):
  `SIM_GEN_SOLVE: SUCCESS`, scores monotone 0 → 0.45 → 1.0, ~18 s each; both rocker
  start sides and both frame halves exercised.
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 11/11`, frames.npz (63, 600, 960, 3) saved:
  1. settle/no-NaN (queen upright opposite the square, rocker held on its stop, score 0);
  2. randomization readback, 3 seeds max-pairwise (Δyaw 11.0°, Δboard-xy 56 mm,
     Δframe 181 mm, Δqueen 233 mm; frame body matches the frame_xy target to 0 mm);
  3. rocker start-side coverage over 10 resets, angle readback follows the draw every time;
  4. null policy ~0;
  5. **seed strategy rejected**: queen seated perfectly, clock untouched → 0.45, no
     success;
  6. **illegal punch**: the solve's own press executed BEFORE the move (beam verifiably
     flips), then a perfect late move — end state equals the success state, foul holds,
     score 0.45, no success;
  7. near-miss seat: upright stand 8.5 cm off the square's centre → score 0;
  8. near-miss pose: queen lying down at the square's centre → score 0;
  9. latches-vs-live: legal solve to success, then the queen knocked off — latches hold
     (score exactly 0.75) yet live success refuses;
  10. **toggle holds**: a 0.004 N·m press (~0.4× holding torque, same actuator path as
      check 6) for 2.5 s leaves the beam on its stop — the bistability is physics;
  11. video saved.
