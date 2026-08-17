# stack_pyramid_i255 — Die Tumble

Scene: `die_tumble` · Env: `simgen.die_tumble` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed: `maniskill/stack_pyramid` (`RoboVerse/roboverse_pack/tasks/maniskill/stack_pyramid.py`):
three 0.04 m cubes; "pick up the red cube, place it next to the green cube, stack the
blue cube on top of both". Success is a `DetectedChecker` with two relative-bbox
detectors — blue on top of red AND green. Every skill in the seed is free-space
pick-and-place: graspable objects, position-only goals, orientation irrelevant, and
the hand can always reach the goal pose directly from above.

## What changed and why it is strategically different

The seed's cubes become ONE cube the seed's strategy cannot touch: a **90 mm, 0.30 kg
die** — wider than the ~80 mm Franka jaw span, so *ungraspable* — whose six faces are
colored and whose goal is **orientation-complete**: the die must rest inside a walled
yellow tray **red face up** (10° cone). With grasping off the table, the only way to
reorient the die is the rolling-cube group: **tumbling it edge-over-edge**, where every
quarter-turn is coupled to exactly one die-width of translation. The solver must plan
in this discrete group — pick the tumble sequence whose composition parks red on
world −z, because the tray entry itself costs two more quarter-turns: a **rim vault**
over the 18 mm curb (red −z → −x, landing in a shallow lean on the inner wall) and an
**in-tray seat tumble** (red −x → up). Physics forbids cheating the entry: smoke's rim
certificate shows a capped 2.2 N push (above the 1.62 N slide threshold, below the
~5.1 N climb-from-outside threshold) slides the die 140+ mm along the floor and stalls
it flush against the outer wall — sliding can NEVER enter the tray; only a deliberate
vault can. The signature skills — *ungraspable-object nonprehensile reorientation*,
*planning in the rolling-cube group where rotation and translation are coupled*, and a
*dynamic vault over a step* — appear nowhere in the seed (grasp + position goals) and
are distinct from the corpus read: not transport-of-a-fragile-stack (i218), not
falsework (i71), not build-a-pyramid-of-blocks (i42), not insertion (pen_holder), no
levers, pouring, counterweights, or captive gates.

The seed's own strategy is a settled reject state: the die placed in the tray with its
spawn orientation kept (a "pick-and-place" that ignores orientation) latches only
arrival credit, score 0.250, no success (smoke check 5).

## Teleport-solution outline (solve.py)

All teleports are transport-only (position writes preserving the current quaternion);
every orientation change is produced by contact dynamics under applied wrenches.

1. **P0** settle + baseline asserts (score 0, no success, red not up by construction).
2. **P1 staging tumbles (contact)**: read the die's red axis, BFS (≤2 moves) in the
   rolling-cube group for the shortest tumble word whose composition maps red to
   world −z (special case red already −z: the [py, ny] round trip, which still latches
   `tumbled`). Each tumble is a torque-servo about the ground edge:
   τ = clamp(ff + kp·(ω_des − ω·a), −0.10, 0.30) N·m, released at 55° (past the 45°
   balance angle), then hands-off settle. Asserts: predicted red axis after every
   move, ≥ 60 mm advance per tumble. Latches `tumbled` (0.15).
3. **P2 approach (teleport = transport)**: die teleported to the outer wall standoff
   x_app = tray_x − (inner/2 + wall_t + die_s/2 + 2 mm), same quaternion; assert red
   still on −z.
4. **P3 entry (contact)**: rim vault — same torque servo with curb feedforward,
   release 69° (past the 58.8° rim balance angle), plus a 0.7 N press toward the tray;
   lands red on −x in a ~6° lean on the inner curb. Then the in-tray seat tumble
   (release ~55°) lands the die flat, red up, at tray-local x ≈ +0.013. Settle; assert
   in_tray, red_up, success, score ≥ 0.999. All wrenches are pre-encoded per step into
   the body frame (`quat_apply_inverse(root_quat_w, wrench)`) so the commanded WORLD
   wrench is exact at every orientation.
5. **P4** hands-off persistence 10×40 steps (3.3 s at 120 Hz); `SIM_GEN_SOLVE:
   SUCCESS` only if success still holds. `SIM_GEN_SCORE` printed at every phase
   boundary, non-decreasing (latched credit). Passes seeds 0–7 (all five spawn red
   directions and every plan branch exercised).

## Embodiment argument (Franka, single arm)

The die (90 mm) exceeds the ~80 mm jaw span — grasping is genuinely impossible, and
the intended contact strategy is fingertip/knuckle pushing, well within Franka
capability: staging tumbles need a ~0.15 N·m edge torque ≈ 1.7 N fingertip force at
the top edge (lever arm ~90 mm); the rim vault peaks near 0.16 N·m + 0.7 N press
≈ 2.5 N combined; the smoke slide certificate caps at 2.2 N. All forces are 1.5–5 N —
two orders of magnitude under Franka's payload. Workspace: the die spawns 0.28–0.36 m
from the tray on a 180±50° bearing; a base at ≈ (−0.20, −0.42) relative to the tray
reaches the spawn scatter, the staging line, and the tray rim within a 0.75 m radius,
all contacts from above/side at ≥ 45 mm height (die half-width) — no floor-level
pinching. Tumbling is done by pushing the top edge past balance and letting gravity
finish each quarter-turn, exactly what the torque-servo + release models. No bimanual
holds, no regrasping, no in-hand manipulation.

## Execution order declared

Built in this order: (1) minimal goal predicate + scene, (2) working solve iterated on
the forge (8/8 seeds), (3) final rubric (latched weights 0.15/0.25/0.25, cap 0.65,
success → 1.0), (4) smoke battery. No fixed move order is imposed on solvers; only the
settled end state is judged (the entry vault is necessarily last by physics, not by
rubric fiat).

## Checks (smoke.py)

10 checks: settle/baseline · randomization readback (tray xy, die bearing/distance
bands, seeds differ) · orientation coverage (10 resets: ≥ 4 distinct red directions,
never red-up at spawn) · null-policy 3 s (score stays 0) · SEED-strategy reject (die
set down in the tray, spawn orientation kept → arrival latch only, 0.250, no success)
· red-up-outside reject (oriented latch only, 0.400, no success) · near-miss flush
against the outer wall red-up (no success, not in_tray) · RIM certificate (2.2 N-capped
quasi-static push slides 140+ mm and stalls flush outside, never enters, never climbs
> 6 mm) · motion gate (die dropped spinning in the tray: success() refuses on every
moving step; probe state discarded by reset) · frames.npz video (70 frames).
