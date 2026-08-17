# screw_nail_i321 — `ram_dispenser` (pump a sealed gravity-fed block dispenser)

Operate a candy-machine-style block dispenser: 1–3 cubes are sealed inside a
capped magazine tower and can ONLY leave through pump cycles of the machine's
ram — pull the red handle back to the rear stop so the bottom cube
gravity-feeds into the roofed channel, then push it forward to the front stop
so the ram nose shoves that cube down the channel and off the muzzle lip.
Before pumping, the free catch bin parked at one of four corner slots must be
staged on the floor at the catch point under the muzzle, or dispensed cubes
are lost. Success = every spawned cube resting inside the upright bin, all
settled. The robot never touches a cube: all cargo motion is machine-mediated.

Env: `simgen.ram_dispenser` (robot="null"), scene `ram_dispenser` in
`scene.py`.

## Provenance

Seed task: `rlbench/screw_nail`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/screw_nail.py`) — a Franka
picks up a screwdriver and drives a nail by pressing down and rotating:
tool-mediated rotary fastening of a single fastener into a fixed base.

## Strategic difference

The seed's strategy is *tool-mediated rotary insertion*: grasp one tool, mate
it to one fastener, convert wrist rotation + downward force into screw
advance; success is one body reaching a depth along one axis.

This task keeps the "drive a mechanism against a fixed base" flavour but the
strategy shares nothing with it:

- **No tool, no rotation, no insertion.** The one manipulated element is the
  machine's own ram, driven in pure horizontal translation between two hard
  stops. Smoke check 6 applies the seed family's exact move (press down +
  twist) to the ram and asserts it moves ~0 mm and dispenses nothing.
- **Indirect, cyclic cargo transport instead of a single fastening.** The
  judged bodies (the cubes) are never touched: they are sealed in the capped
  tower (smoke check 7 lifts the top cube 40 mm with a regulated force and
  shows it cannot leave the shaft). Each cube requires one FULL pump cycle —
  a retract-to-feed stroke and an advance-to-eject stroke — so the plan is
  1 staging move + k repetitions of a two-phase machine cycle, not one
  press-and-turn.
- **Receiver staging as a distinct, prerequisite sub-goal.** The seed has one
  goal body and no ordering. Here the catch bin must first be transported to
  the catch point; ejection without a staged bin strands cubes on the floor
  (smoke check 9: floor cube beside the bin = ejection credit only, never
  in-bin credit or success).

Distinct from the siblings read this session: `screw_nail_i309`
("leaning_chain") is free-body statics *construction* with no mechanism at
all; `screw_nail_i59` ("latch_vault") is an unlock→uncover→retrieve
*disassembly* chain on prismatic sliders where the robot carries the prize
directly. Here nothing is assembled or disassembled and the robot may never
carry the cargo — credit flows only through repeated operation of a fixed
machine. Also unlike container-insertion tasks (e.g. `pen_holder`): the cubes
enter the bin ballistically off the muzzle lip, not by placement.

## Solution outline (solve.py, teleport = transport only)

1. Read the randomized layout (cube count/subset, ram start mode, bin park
   slot + yaw) from state readbacks after settling. Assert score ~0.
2. STAGE (the only teleport): move the free, empty bin from its park slot to
   the catch point under the muzzle and settle — pure transport across open
   floor. `SIM_GEN_SCORE` rises to 0.10.
3. For each of the k cubes (lowest present, un-ejected first — the order the
   magazine physically enforces): a velocity-regulated horizontal force on
   the ram (F = clamp(gain·(v_des − v), ±cap), gain·dt/m = 0.42 < 1)
   - PULLS to the rear stop until the tail bottoms out; the bottom cube
     gravity-feeds onto the channel floor (readback-verified, up to 3 s);
   - PUSHES forward — 0.12 m/s cruise, then a 0.55 m/s final segment so the
     cube leaves the lip fast enough to carry over the bin's near wall —
     until the handle post bottoms out on the end plate; the cube flies off
     the muzzle and lands in the bin.
   Stalls/misfeeds retry ×4 with 1.3× servo GAIN escalation and a jiggle
   stroke. Score rises by 0.50/k + 0.30/k per completed cycle.
4. Park the ram mid-stroke, hands-off settle to success(), then ≥3.3 s
   persistence with flicker diagnostics; `SIM_GEN_SCORE` is non-decreasing
   throughout (latched credit) and `SIM_GEN_SOLVE: SUCCESS` only if success
   persists. Verified on the forge, seeds 0 (k=3, 25.3 s) and 1 (k=1,
   forward-parked ram start, 19.2 s).

## Embodiment argument (single Franka + parallel-jaw gripper)

Base the Franka at env (0.42, −0.55), facing the rig. The only two things the
robot ever touches are graspable by design:

- **The ram handle**: a red 14 mm-square post rising 80 mm above the steel
  bar, top at z ≈ 0.21 m, travelling env x ≈ 0.21–0.36 m during the stroke —
  a natural parallel-jaw grasp (14 mm ≪ 80 mm jaw span) from above, with the
  tower 0.10 m away in +x and nothing overhead. The pump is a 153 mm
  horizontal straight-line stroke at fixed height, repeated k ≤ 3 times —
  low-force (the servo caps at 10 N on a 0.6 kg ram) and comfortably inside
  the dexterous workspace.
- **The bin**: a 160 mm open box with 10 mm-thick walls, top edge at
  z = 0.075 m — a standard rim pinch grasp; it is carried once, empty
  (0.25 kg), across open floor from a park slot ≤ 0.75 m from the base to
  the catch point, all below z = 0.1 m.

The robot never needs to reach a cube (they are sealed until ejected and
land inside the bin), so no small-object grasping, no bimanual work, no
regrasping and no tool use is required.

## Execution order

Declared and physically forced: **stage the bin FIRST, then run the k pump
cycles**. A cube ejected before staging lands on open floor and the machine
cannot re-ingest it — the tower feeds only downward and the solve never
touches cubes — so pump-first strands cargo irrecoverably (the rubric's
in-bin fraction can then never reach 1). Within the pumping phase the
magazine mechanically serializes the cubes: the one-cube-high underpass
(36 mm gap vs 30 mm cube, asserted in `__post_init__`) admits exactly the
bottom cube per retraction, so the ejection order is the stack order,
bottom-up, and each cube needs its own full cycle.

## Checks (smoke.py — 15)

1. Reset settles finite; cubes sealed in the tower, ram in a start band, bin
   parked at a corner slot; layout sane.
2. Score ~0 at reset, no success.
3. Randomization readback (seeds 31–36): cube COUNT and cube SUBSET vary.
4. Ram tip position spreads and BOTH start modes (retracted/forward) occur;
   bin park slot and yaw vary across seeds.
5. Null policy (240 idle steps) → cubes stay sealed, score ~0, no success.
6. Seed-strategy analog: press-down + twist on the ram → ~0 mm motion,
   nothing dispensed, score ~0.
7. Sealed tower: a regulated lift raises the top cube 40 mm (probe verified
   real) but it cannot leave the capped shaft; reseats with no credit.
8. Mid-channel cube (fed but not ejected, parked under the channel roof) →
   no ejection credit, no success.
9. Missed catch: cube on the floor beside the staged bin → ejection credit
   only (score = 0.10 + 0.50/k), NOT in-bin, no success.
10. Wall-top perch: cube centered on the bin wall reads outside the
    containment window (bin-frame |x| = 0.075 > 0.062 tol) → rejected.
11. Tipped bin: overturned bin with a cube on its upturned floor → upright
    gate rejects.
12. Settle gate: the only cube inside the bin window but moving at 0.4 m/s
    → not success (velocity gates are real).
13. Latched credit: staging the bin then parking it away keeps the 0.10
    stage credit while `bin_staged_now()` drops.
14. Audit: success() never True at any judged point of the battery.
15. Final no-NaN over all task objects.

Forge: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, 35.6 s), frames.npz recorded.
