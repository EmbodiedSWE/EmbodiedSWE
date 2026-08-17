# boom_corral (`pull_cube_tool_i348`)

Herd a permanently out-of-reach cube into a permanently out-of-reach three-walled pen
by working an **anchored pivoting boom**: the robot only ever touches the boom's short
handle (always in reach), while the far blade pocket — the mirror-image of the handle
across a fixed pivot — sweeps the cube along an arc and pushes it through the pen
mouth. The pen lands on a **randomized side** of the cube each episode, so the required
sweep direction (and which way to push the handle) flips per seed.

- Env id: `simgen.boom_corral` (NullRobot scene-level env)
- Files: `scene.py` (scene + rubric + plant), `solve.py` (solution certificate —
  **zero teleports on the success path**), `smoke.py` (16-check rejection battery,
  recorded), this file.

## Seed provenance

| | seed | this task |
|---|---|---|
| source | `maniskill/pull_cube_tool` (RoboVerse `roboverse_pack/tasks/maniskill/pull_cube_tool.py`) | new scene, procedural geometry only |
| setup | free L-shaped tool within reach; cube beyond reach on open floor | NO portable tool; a **fixed-pivot boom** (revolute Z, ±70°) spans the reach boundary — near handle in reach, far blade out among the cargo; cube AND goal pen both permanently beyond reach |
| plan | grasp the tool, hook it behind the cube, **drag radially inward** under direct quasi-static control | push the handle **tangentially**: the blade on the other side of the pivot sweeps the **opposite** direction (mirror inversion) and carries the cube **along an arc at constant radius** through the pen mouth; the cube never comes closer to the robot |
| goal | proximity disc: cube within 0.6 m of the base | cube **inside a walled pen** far away (pen-frame depth window + lateral window + rest + settled), after a physically-latched sweep-and-entry chain |

## Why strategically different

- **vs the seed**: the seed is reach extension with a portable implement — pick the
  tool up, carry it, hook, drag the cube *toward yourself*; the goal is proximity to
  the base. Here nothing is ever picked up or carried, the manipulated interface is a
  **permanently anchored mechanism**, the transport is **tangential (constant-radius
  arc), not radial**, and the cube *ends* as far away as it started — the goal is
  containment in a far-field pen, the exact opposite of "bring it near". Control is
  also **mirror-inverted and displacement-amplified** (handle arm 0.42 m, blade arm
  0.56 m): pushing the handle left moves the cargo right, farther and faster.
- **vs sibling `pull_cube_tool_i1` (carousel_ferry)**: there the cube RIDES a rotating
  carrier (transport by carriage; the mechanism holds the cargo). Here the blade
  **pushes** free cargo across ground contact — the cube is never on the mechanism,
  and the terminal state is inside separate static geometry (the pen), not on the
  machine.
- **vs sibling `pull_cube_tool_i186` (shelf_drop_dispatch)**: there the far-field leg
  is a one-shot stored-energy **release** (gravity delivers, arm never touches the
  cargo far away) followed by an in-reach precision pick. Here the transport is
  **continuous, active, closed-loop herding through the mechanism** the whole way,
  there is no irreversible event, no gravity delivery, and no pick at all — the cube
  is never within reach at any time.

## Mechanics

Plain rigid bodies + authored USD joints per env: one `RevoluteJoint` (kinematic
pedestal → spar, axis Z, limits ±70°) and four `FixedJoint`s (spar → handle post,
blade face, inner lip, outer lip). The spar (1.10 m, local x [−0.44, 0.66] about the
pivot) flies at z 0.07–0.10 — **above** the 6 cm pen walls, so only the blade pocket
(face + two lips, z 0.008–0.083, radial span 0.4795–0.6405) interacts at cargo height.
A `post_step` plant consumes the `boom_drive` buffer (yaw torque, clamped to
±1.5 N·m ≈ 3.6 N at the 0.42 m handle — a one-hand push) and adds viscous friction
(τ_v = −0.12·ω) every substep (`enable_external_forces_every_iteration`). The cube
(4.5 cm, 50 g) rests on the blade circle; the pen is three 1.5 cm walls (back + two
sides), mouth facing the cube's arc.

## Rubric (latched, anchored in solve.py)

- `arc_done` (monotone accumulator): signed toward-pen azimuth progress of the CUBE,
  accumulated only while the cube is in the sweep annulus (r ∈ 0.42–0.72, z < 0.06)
  and per-substep jump ≤ 2° (teleports don't count); `arc_prog = arc_done/arc_req`
  clamped to [0, 1].
- `swept` (latch): arc_done ≥ arc_req − 6° — the cube really travelled the arc.
- `entered` (latch): `swept` AND pen-frame depth u > 0.010 with |w| < 0.115 at ground
  level — mouth transit **after** the sweep (order enforced, not declared).
- `success()`: `entered` AND currently inside the pen (u ∈ 0.020–0.170, |w| < 0.090,
  z at rest ± 1.2 cm, |v| < 0.04, |ω| < 0.6).
- `score()` = 0.3·arc_prog + 0.2·entered + 0.5·success — exactly 1.0 iff success;
  null ≈ 0; a knock-out after success falls back to exactly 0.50, never 0.

**Execution order is REQUIRED and physically enforced**: the cube and the pen both
spawn > 1.0 m from the base (beyond the 0.855 m Franka envelope), so no direct
placement is ever possible; and `entered` requires the `swept` latch, so a cube that
appears inside the pen by any means without the arc scores ~0 (smoke check 7).

## Randomization (verified by READBACK in smoke)

Cube spawn azimuth ± 14° about +x from the pivot, radius ∈ [0.535, 0.585], free yaw;
pen side ∈ {−1, +1} (both observed in the seed scan); pen arc offset ∈ [30°, 40°];
boom start backed off 13–18° on the anti-pen side. Cube xy, boom yaw, and pen pose all
differ across seeds by readback.

## Solution outline (solve.py — ZERO teleports on the success path)

0. reset(seed), settle — cube resting far-field on the blade circle, no latches,
   score ~0.
1. **HERD** (contact dynamics): read the pen side; closed-loop yaw-rate servo on a
   **finite-difference** boom rate (ω_des = side·0.25 rad/s, τ = 4·(ω_des − ω_fd) +
   escalating stiction bias, total clamped to ±1.5 N·m) sweeps the blade into the cube
   and pushes it along the arc through the pen mouth to depth u ≥ 0.070 → latches +
   score ≥ 0.5.
2. **PARK**: reverse the servo ~12° so the blade clears the mouth, zero the drive,
   settle — the cube rests inside the pen → score exactly 1.00.
3. **PERSIST**: ≥ 3.5 s pure simulation, success at every poll →
   `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (0 → 0.5+ → 1.0 → 1.0).

## Franka embodiment (base at (−0.10, 0), facing +x)

- **Handle post** (2.8 cm dia, z 0.10–0.26, atop the spar's near end): sweeps a 0.42 m
  arc about the pivot at (0.62, 0); over the full ±70° travel it stays 0.30–0.70 m
  from the base — inside the comfortable envelope (smoke asserts < 0.75 m for the
  whole drive). Wrap-grasp the vertical post (standard 2.8 cm pinch) or push it with
  closed fingers; the drive is a tangential push ≤ 3.6 N — fingertip scale — and works
  in **both** directions, as the randomized pen side requires.
- **Red cube** (4.5 cm) and **green pen**: both spawn > 1.0 m from the base and stay
  far-field for the entire task — deliberately unreachable; the robot never needs to
  (and cannot) touch either.
- Nothing requires reaching past 0.70 m, pushing above ~4 N, grasping anything but
  the post, or entering closed cavities.

## Checks

- **solve.py**: 4 phase gates (clean initial state in-band with no latches; herd
  completed AND both latches; success with score exactly 1.0 after park; persistence
  ≥ 3.5 s with success at every poll), non-decreasing `SIM_GEN_SCORE` ladder,
  watchdog hard-exit. Passes on forge seeds 0, 1, and 2 (seed 2 exercises the
  opposite pen side / mirrored sweep).
- **smoke.py**: 16 named checks — settle, randomization readback, side flip (both pen
  sides observed), out-of-reach spawn (cube + pen > 1.0 m, handle < 0.75 m), null
  policy ~0, seed-strategy end state (cube parked near the base) rejected,
  teleport-into-pen anti-cheat rejected (order enforcement), boom exercise away from
  the cube (27° real excursion) earns nothing, oracle herd with the solve's servo law
  (latches + success), exactness (score == 1.0), monotone score trace, anchored pivot
  (< 5 mm drift, handle < 0.75 m throughout), near-miss herd stopped ~10 cm short
  (partial credit only, no latch), 2 s persistence, knock-out latch (falls to exactly
  0.50), frames recorded (`frames.npz` in CWD). Verdict:
  `SIM_GEN_SMOKE: ALL PASS 16/16`.
