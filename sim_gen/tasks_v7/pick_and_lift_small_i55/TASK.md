# hook_den_retrieval (`pick_and_lift_small_i55`)

Rake the BLUE cube out of a low roofed den with an L-shaped hook, then set it on the
yellow pedestal. The RED decoy cube must stay off the pedestal.

## Seed provenance

- **Seed task**: `rlbench/pick_and_lift_small`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/pick_and_lift_small.py`) — "pick up
  the [small cube] and lift it up to the target": grasp the small cube among shape
  distractors (star, moon, prism, cylinder) and raise it to a floating success sphere.
- **What is kept**: the object of desire is a small cube among same-scale distractor
  objects; identity discrimination is load-bearing (grab the right one); the goal
  involves elevating the cube (the 120 mm pedestal echoes "lift it up to the target").
- **What is strategically different**: the seed's whole strategy is ONE prehensile
  grasp-and-raise through free air. Here that exact strategy is physically removed:
  the cube starts 172–217 mm deep inside a den whose fully closed roof is 60 mm above
  the ground with a single 160 x 60 mm mouth. No gripper reaches it (a Franka finger
  is ~55 mm long; the palm stops at the mouth) and the roof caps any lift — the smoke
  battery pulls the denned cube UP at 2.5x its weight for 1.5 s and it never leaves
  the den. The only route is TOOL USE: slide the 340 mm L-hook flat under the roof
  along the wall lane OPPOSITE the cube, shift sideways so the 50 mm toe lands behind
  the cube, and DRAG it out through the mouth — a sustained tool-mediated pulling
  contact that the seed has no analogue of. Only then does the seed's own skill
  (pick, place up high) come into play, plus a decoy restraint the seed lacks: the
  conveniently reachable RED cube is the wrong one.

## Strategic differences vs the corpus (tasks read this session)

- `pick_and_lift_i16` (same seed family): ballast-and-lever — the cube rides a seesaw
  and is raised by loading a counterweight. No tool held against the object, no
  confined retrieval; here nothing pivots, and the cube is dragged by a carried tool.
- `libero_..._i1` (ramrod): PUSH-eject an object AWAY through a bore, tool moving
  forward the whole stroke. Here the tool must first pass the object, capture it from
  BEHIND, and PULL it back TOWARD the actor out the same opening it entered — an
  insert–shift–retract cycle with a chirality choice (which wall lane, which toe
  roll), not a single forward stroke.
- `close_box_i18` (wedge jack): insert-once wedge that stays put and holds by normal
  force. The hook is retrieved WITH the payload and does its work by sliding contact.
- `..._i33` (one-way flap pantry): push the object THROUGH a yielding gate INTO a
  chamber. Exactly inverted here: nothing yields, and the object comes OUT of the
  chamber via a tool.
- Other corpus verbs this session (drop-gate prop removal, bayonet twist, carousel,
  pour, cap, drawer, tumble, tunnel slide, hang, chock, silo, projectile, moat
  causeway): none is a tool-mediated pull-retrieval from a roofed confinement, and
  none combines it with a wrong-object restraint and a regrasp-and-elevate finish.

## Scene

Fully procedural (no external assets):

| Body | Spawn | Key numbers |
| --- | --- | --- |
| den | compound spawner, heavy dynamic (25 kg, damped) | interior 260 x 160 x 60 mm, walls 12 mm, no floor, one open mouth |
| hook | compound spawner, dynamic 0.18 kg | handle 340 mm + toe 50 mm, 16 mm square section, CoM authored |
| pedestal | `CuboidCfg`, 15 kg | 110 x 110 x 120 mm, YELLOW |
| cargo | `CuboidCfg`, 90 g | 40 mm cube, BLUE — the target |
| decoy | `CuboidCfg`, 90 g | 40 mm cube, RED — must stay off the pedestal |

Per-episode randomization (all readback-verified in smoke): den yaw 180±20° and xy
±50 mm; cargo depth (den-x −75..−30 mm), side (±y, both occur), lateral 22–40 mm, free
yaw; hook position, yaw ±55°, and TOE ROLL (0 or π about the handle — both occur);
decoy spot + side; pedestal side + jitter.

## Rubric (latched partial credit, anchored in the demonstrated solve)

- 0.15 `engaged` — the toe midpoint deeper in the den than the cube (by 10 mm),
  laterally within 50 mm, while the cube is still inside (the raking position).
- 0.35 `extracted` — the cube out through the mouth: den-frame x > 160 mm, on the
  ground (z < 50 mm), slow.
- 1.0 iff `success()`: cargo settled ON TOP of the upright pedestal (axis distance
  < 32 mm, rest height ±10 mm), decoy NOT on the pedestal, everything settled/finite.
  Non-success capped at 0.50. Null policy scores ~0.

## Demonstrated solution (solve.py, the legitimacy certificate)

1. Teleport (transport only, free space): stage the hook in the OPEN at den-local
   (0.30, −s·0.064), s = cargo side, rolled π when the cube is on −y so the toe
   points at it.
2. Force-slide INSERT along the opposite-wall lane (CoM force, yaw-servo torque,
   lateral P-force; force-frame probe per memory quirk) until the toe is 40 mm deeper
   than the cube.
3. Force SHIFT sideways until the toe pockets behind the cube (`engaged` latches).
4. Force DRAG out +x: the toe pushes the cube through the mouth; inside, the side
   wall backstops slip-offs; release and let it coast (`extracted` latches). Retry
   loop (pull out, re-insert) if the pocket is missed.
5. Teleport the now-reachable cube (transport only) to 22 mm ABOVE its pedestal rest
   pose — the write itself is outside `on_ped_z_tol`, gravity seats it.
6. 3.3 s hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge pod (RTX 4090, Isaac Sim 5.1 / isaaclab 0.54.2): seeds 0, 1, 2
end `SIM_GEN_SOLVE: SUCCESS` with monotone `SIM_GEN_SCORE` 0.00 → 0.15/0.50 → 1.00.

## Franka embodiment argument

Base at ~(−0.35, 0, 0) facing +x sees the whole workspace (den mouth at world
x ≈ 0.28, staging lane, pedestal ≈ 0.15–0.30, all within a 0.85 m reach envelope).
Contact strategy per object:

- **Hook**: power-grasp the 16 mm square handle anywhere along its free 200+ mm; the
  drag force needed is ~1–3 N (measured in solve) at ~13 cm/s — trivially inside
  Franka's payload. During the deepest insertion the handle still protrudes ≥ 75 mm
  past the mouth plane, so the gripper NEVER has to enter the den: the staging pose,
  the insert stroke (planar slide), the lateral shift and the pull are all
  end-effector motions in free air above open ground.
- **Cargo cube**: after extraction it rests in the open ≥ 160 mm outside the mouth —
  a standard 40 mm top-down pinch grasp, then a 140 mm place onto the 110 mm-square
  pedestal top (32 mm tolerance is ~1.5 finger widths).
- **Decoy/pedestal**: never need to be touched.

The two teleports in solve.py stand in for exactly these two reach-and-carry motions;
every load-bearing interaction (insert, shift, drag, seating) is contact dynamics.

## Execution order

No required order beyond what physics forces: the cube cannot be placed until it is
out (the roof lift-block makes extraction-before-placement physically mandatory, not
rubric-mandated). The decoy clause is a standing restraint, not a step.

## Checks

- solve: `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1, 2 (forge; monotone scores, 3.3 s
  hands-off persistence).
- smoke: 13-check rejection battery `SIM_GEN_SMOKE: ALL PASS 13/13` (forge) — settle,
  randomization readback, side/roll swap, null policy, ROOF LIFT-BLOCK (2.5x-weight
  upward pull stays denned), engage geometry (toe-in-front earns nothing / toe-behind
  earns exactly 0.15), seed-strategy end state (cube on the roof) rejected, off-axis
  and beside-pedestal near-misses rejected, wrong-object and both-on-top rejected,
  frames.npz recorded.
