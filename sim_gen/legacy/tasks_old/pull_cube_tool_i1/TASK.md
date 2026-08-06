# tunnel_shuffle (`pull_cube_tool_i1`)

**Seed provenance:** `maniskill/pull_cube_tool`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/pull_cube_tool.py`) — an L-shaped tool
within reach must be used to PULL a cube that is out of reach back INTO a sphere around
the robot base (reach extension by tool, quasi-static dragging toward the base).

**This task:** a red cube starts close at hand near the open end of a long walled lane.
Partway down the lane a low tunnel (opening 80 mm — barely taller than the 50 mm cube,
too low for any gripper holding it) spans the full width; beyond it a green goal box is
painted flush on the lane floor. The solver must **send the cube AWAY** with a calibrated
shove so it slides on the surface UNDER the tunnel and comes to rest **fully inside** the
goal box. Both lane ends are open: dragging the cube toward the base drops it off the
near edge; shoving too hard sends it off the far edge. Gate position and goal distance
are randomized per episode, so no memorized impulse works.

## Why strategically different from the seed

A solver needs a different **plan**, not different numbers:

| | seed (pull_cube_tool) | tunnel_shuffle |
|---|---|---|
| direction | pull a distant object **toward** the base | send a near object **away** from the base |
| mechanism | tool-mediated reach extension, quasi-static drag | **impulse / momentum control**: shove speed → friction stopping distance (v = √(2µgd)); no tool exists |
| constraint | none (open table) | low tunnel + tall walls: the cube can only transit sliding on the surface; carrying/teleporting it over is rejected by a physical-transit latch |
| success | object enters a reach-radius sphere (anywhere inside counts) | object **settles fully inside** a distant band — overshoot fails (open far end), undershoot fails |

The seed's own strategy is expressible here and **fails**: negative control A executes
"drag the cube toward the base" and the cube slides off the near end of the lane (score
~0). Conversely a pulled-into-reach outcome scores nothing because all scoring is forward
progress away from the spawn.

## Difficulty tier: easy — 1 stage

Single skill: one calibrated shove (optionally re-pushed to converge). No required
execution order (there is only one stage; multiple corrective pushes are allowed and the
rubric is indifferent to how many).

## Scoring (graded, latched)

- `0.45 ×` current normalized forward progress (spawn → goal center, on-lane only,
  clamped — overshoot not rewarded, off-lane earns nothing; doing nothing = exactly 0);
- `+0.35` latched **physical tunnel transit** (set only by a continuous on-surface
  crossing of the gate plane between consecutive physics steps; a kinematic teleport
  across the gate, or an over-the-roof pass, does not latch);
- `= 1.0` iff `success()`: settled flat on the lane, fully inside the goal box, transit
  latched. Max non-success score 0.8.

## Check list (smoke.py, teleport-oracle battery)

1. settle/no-NaN + score ~0 at reset (2 checks)
2. randomization-is-real by READBACK (cube spawn, tunnel gate, goal box)
3. null-policy-fails (240 idle steps → score ~0)
4. oracle reaches success() + score 1.0 on seeds 0/1/2 (3 checks; oracle measures µ from
   its first slide and re-pushes — a calibration-probe behavior in itself)
5. rubric monotonicity: placement ladder down the lane → non-decreasing scores, all < 1.0
   (2 checks)
6. anti-cheat: teleporting the cube into the goal box does not latch the transit and is
   not success (2 checks)
7. negative control A — the seed's strategy (pull toward base) → off the near edge,
   score ~0
8. negative control B — cube parked on top of the tunnel roof (over, not under) → score 0
9. calibration probe: launch-speed sweep → stopping-distance table, monotone + sane µ
   band (2 checks)
10. near-miss/tolerance control: measured-µ shove stops short of the box → partial credit,
    no success; corrective push then succeeds (2 checks)

Video frames are recorded via the viewport RGB annotator and saved to `frames.npz` in the
working directory.
