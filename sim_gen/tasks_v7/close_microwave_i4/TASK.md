# close_microwave_i4 — DropGateOvenScene (`simgen.drop_gate_oven`)

Unload the BLUE can from the oven chamber onto the green pad, leave the RED can
inside, then pull the yellow prop post out from under the raised orange drop-gate so
GRAVITY drops the gate in its slots and seals the doorway. The arm never pushes
anything shut — closure is a mechanism release, and the forced order (unload before
release) is encoded in the physics of the sealed chamber.

## Seed provenance

- **Seed task**: `rlbench/close_microwave` (RoboVerse
  `roboverse_pack/tasks/rlbench/close_microwave.py`) — "close microwave": a microwave
  USD with one revolute door, robot=franka. The plan is a single unordered act: one
  pushing contact on the open hinged door, rotating it about its hinge until flush.
  No perception demands, no other object, no ordering, judged by the door joint.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| closure mechanism | revolute hinged door | guillotine DROP-GATE riding in vertical slot guides, held raised by a yellow PROP POST standing under its bottom edge |
| closing act | the arm PUSHES the door through its swing — the seed's entire skill | the arm NEVER pushes the gate: extracting the loaded post releases it and **gravity performs the closing** (the arm's contribution is a strut pull, a release, not a push) |
| other objects | none — the door is the task | two cans inside the chamber (one blue, one red, arrangement shuffled), a green target pad outside, the prop post |
| ordering | none | **physically forced order**: the blue can must exit BEFORE the release — a seated gate seals the chamber (smoke 5: a 3x-weight shove cannot get a can out), and a single arm cannot hold the gate up and reach inside at once |
| perception | fixed asset, fixed door | which can is blue is shuffled per episode; oven yaw/xy, pad xy, and the post's lateral slot all vary (readback-verified) |
| restraint | n/a | the RED can must NOT be taken out — it must end sealed inside |
| assets | RLBench microwave USD | 100 % procedural (compound kinematic oven with slot guides; dynamic gate, post, cans; kinematic pad) |
| judging | door joint angle | live geometric success (gate seated in-frame by position AND orientation, blue upright on pad, red inside, settled) + latched stage credit |

Code structure shares nothing with the seed: `@SCENES.register` BaseScene with a
compound kinematic spawner, oven-frame predicates, latches in `post_step`,
`register_env(..., robot="null")`. (The seed is a 10-line RLBench task cfg pointing at
a USD + trajectory file.)

## Why strategically different

The seed's entire skill is *push the hinged panel until it shuts* — one contact, one
rotation, nothing else in the scene, any moment, no consequence for order. Here that
plan earns nothing: smoke check 6 CONSTRUCTS exactly the seed's end state (the gate
seated, nothing else done) and the rubric scores it ~0 with no success — because the
`sealed` credit requires the blue can already out, and the task is mostly NOT about
closing. What the solver must bring instead is (1) **perception** of a shuffled
blue/red arrangement inside the chamber, (2) **object extraction and placement** — a
reach-in grasp through a doorway, carry, and precise upright set-down on a pad, absent
from the seed, (3) **restraint** — the red can must be left inside, (4) a **strict,
physically forced execution order** (sealed chamber ⇒ unload first; smoke 5 proves the
seal is load-bearing), and (5) closure by **mechanism release under load** — pulling a
strut out from under a 150 g gate — rather than by pushing a panel. The one thing the
seed does (arm pushes the door shut) is the one thing that never happens here.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; readback layout (oven xy/yaw ±20°, pad xy, post slot y ±50 mm,
   blue slot ±y); baseline score 0.0, no success.
2. **P1** (teleport = transport only): carry the blue can through the OPEN doorway
   (aperture ~200 × 114 mm vs a 50 mm can — free space) to 30 mm above the pad, zero
   velocity; it FALLS and seats upright on the pad by contact → out+placed latches
   (score 0.45). The red can is never touched.
3. **P2** (release under load): a horizontal force at the post's CoM (oven-local +x,
   ramp 2→10 N; 2 N sufficed on every tested seed) drags the loaded post out from
   under the gate through friction contact — the post is never teleported. The moment
   it clears, the gate free-falls ~115 mm in its slots and seats on the apron between
   the guides — pure ballistics + contact → success (score 1.0). (An escalating
   downward gate tap is armed for slot wedging but **never fired on any tested seed**.)
4. **P3** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.0000 → 0.4500 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m). The
oven doorway faces the base (nominal yaw 180°), its plane ~0.34 m away; the cans stand
~0.42 m away; the pad ~0.36 m away — all inside a comfortable dexterous shell.

- **Blue can** (Ø50 × 60 mm, 100 g): horizontal SIDE PINCH through the open doorway
  (aperture 200 mm wide × ~114 mm tall — admits the hand with the wrist horizontal;
  jaw max 80 mm ≫ 50 mm). The can stands ~80 mm past the doorway plane with ≥ 35 mm
  lateral clearance to the walls and 120 mm to the other can. Lift a few cm, retract
  through the aperture, carry, and set down on the pad — pad tolerance 45 mm on a
  60 mm-radius pad is generous for closed-loop placement.
- **Prop post** (25 mm square, 114 mm tall, 50 g): stands proud in the open doorway,
  fully exposed from the front. Pinch it anywhere along its exposed length (25 mm ≪
  80 mm jaw) and pull straight toward the robot (~2 cm of loaded travel frees the
  gate; measured release force ~2 N), or simply hook and drag it out. No precision:
  any escape — slide or topple — releases the gate.
- **Gate**: NEVER touched — gravity closes it; the slots do all the alignment.
- **Red can**: never touched — restraint, not manipulation.
- Oven and pad are kinematic: incidental contact cannot move the goal frames.

## Execution order (declared)

`extract blue can → place on pad → pull the post (gate falls)`. REQUIRED, and enforced
by physics rather than rubric fiat: once the gate seats, the chamber is sealed
(smoke 5: 3x-weight shove, the can never leaves), and a single arm cannot hold the
135 mm gate raised and simultaneously reach through the doorway it is blocking. The
rubric mirrors this: the `sealed` latch only arms once the blue can is out, so
releasing the gate early forfeits the run.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (post_y −5 mm, blue −y): SUCCESS, scores 0.0/0.45/1.0/1.0.
- `solve --seed 1` (post_y −34 mm, blue −y, different oven/pad poses): SUCCESS.
- `solve --seed 2` (post_y +45 mm, blue +y): SUCCESS — three seeds, no gate wedging,
  the gate nudge never fired.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 12/12**, frames.npz (67 × 600 × 960) saved:
  1. settle/no-NaN; gate raised on the post, cans inside; score 0
  2. randomization readback differs (oven yaw Δ16.1°, oven xy Δ39 mm, pad Δ56 mm,
     post slot Δ55 mm)
  3. arrangement swap: blue can occupies both interior slots over 10 resets
  4. null policy: 240 idle steps, post keeps the gate raised, score 0, no success
  5. GATE INTERLOCK: 3x-weight shove toward the doorway — the can never leaves the
     sealed chamber, the gate stays seated
  6. SEED STRATEGY: gate seated, both cans still inside → score 0, no success
  7. wrong object: RED can on the pad, blue sealed inside → no success
  8. near-miss: blue upright on the ground 73 mm from the pad axis (tol 45) → fail
  9. near-miss: gate resting on the toppled post 25 mm above its seat (tol 12) → fail
  10. tipped: blue lying on its side ON the pad → upright clause refuses
  11. gate discarded: gate laid flat on the ground → not seated, no success
  12. video frames.npz saved

## Files

- `scene.py` — DropGateOvenScene + oven compound spawner + rubric; registers
  `simgen.drop_gate_oven`.
- `solve.py` — teleport-transport + force-release + gravity-closure certificate
  (`--seed N`).
- `smoke.py` — 12-check rejection battery + video.
