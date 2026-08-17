# native_libero_i365 — crank the Scotch-yoke ferry to deliver the red cube into a sealed under-floor vault

## Seed provenance

Seed task: `libero/native_libero` (RoboVerse `roboverse_pack/tasks/libero/native_libero.py`)
— the LIBERO pick-and-place family: a Franka grasps the goal object, carries it
through free space, and sets it down on/in an open goal region. The judged
outcome here is still "the goal object at rest in the goal region", but the
goal region is sealed against exactly that strategy.

## Strategic difference

**vs. the seed.** In the seed the goal region is open from above and the
solution is grasp → carry → place. Here the goal region is a VAULT under the
floor of a sealed, roofed tunnel: its only entry is a hole in the tunnel floor,
and the tunnel itself admits nothing from above (the roof's only opening is a
32 mm slit — narrower than the 46 mm cargo cube; a smoke probe drops the cube
right above the vault and it lands straddling the slit). Carrying the object to
the goal is impossible *by construction*. Instead the robot must operate a
machine: a crank disc drives a ferry cage through a Scotch yoke
(`cage_x = 0.240 + 0.145·cos θ`), converting rotation into linear travel down
the tunnel. The solution is (1) crank the cage to the loading window's dead
center (θ=±180°, where the yoke self-locks against rail forces), (2) push the
cube sideways across an apron through the wall window into the three-sided
cage, (3) crank onward so the cage plows the cube down the lane until it tips
through the floor hole into the vault. Plan-level contrasts: rotary→linear
motion conversion, an alignment-gated loading step, a hard ORDER requirement
(align before load — loading early strands the cube), and a terminal gravity
discharge. The manipulated bodies (crank handle, cube-on-apron) are never the
judged terminal state; the seed's grasp-and-carry never touches the goal.

**vs. the rest of the corpus (and i87).** i87 routes a linear push through a
rocker to another linear slide; no task in `tasks_v7/` uses a crank, a
Scotch-yoke (rotation→translation) transmission, a dead-center self-locking
alignment, or a vehicle that the cargo must ride. The ferry is not a carried
shuttle: it is joint-bound inside a sealed tunnel and can only be moved by
turning the crank. No stored energy anywhere: the disc's CoM is on its axle,
the cage hangs on a horizontal prismatic joint, all motion is robot-supplied
and friction-held (the mechanism stays wherever it is left).

## Scene (`crank_ferry`, env `simgen.crank_ferry`, robot="null")

Procedural geometry only. A kinematic STRUCTURE (sealed tunnel with loading
window + roof slit + floor hole over a sealed vault, apron table, crank gantry)
contains three dynamic assemblies plus two free cubes:

- **Ferry cage** (prismatic along x, stroke ±0.155 about crank_x=0.240): a
  three-sided open-bottom box (interior 90 mm, open toward the +y window;
  cargo rides on the tunnel floor), mast up through the roof slit, and an
  overhead crossbar with two rails forming the transverse yoke channel.
- **Crank disc** (revolute about z at (0.240, 0, 0.372), NO limits —
  continuous): disc r=0.17, downward drive pin at r=0.145 riding in the cage
  channel, and a red HANDLE peg (18 mm dia, top z=0.460) at the same radius —
  what the robot pushes.
- **Cargo** (red 46 mm cube, m=0.12 — the goal object) and **decoy** (blue
  56 mm cube, m=0.18) spawn at seeded poses/yaws on the apron.

Per-episode randomization (smoke-verified readback): crank angle θ0 (sign
random, |θ0|∈[55°,125°] — keeps the cage off both dead centers) and both cube
xy+yaw poses. Geometry contract (~25 `__post_init__` asserts): window ⊂
aligned-cage interior across the ±12 mm tolerance; both cubes pass the window,
neither passes the slit; cargo (any yaw) passes the hole; full stroke plows the
cargo past the hole edge; the near pocket swallows a stranded cube without
jamming; pin/channel slack, rail spans, sweep clearances, non-overlapping
spawn bands. Slick material (μ≈0.06, combine "min") on the yoke surfaces so
PhysX's default ~0.5 friction cannot fight the transmission.

**Success** (live state): cargo center inside the vault box (x∈[0.337,0.433],
|y|≤0.055, z≤0.070 — below lane level) ∧ settled ∧ finite. **Score**: latched
partial credit — align 0.10 (cage ever within 12 mm of the window dead
center), loaded 0.20 (cargo ever aboard), ferry 0.25 × running-max aboard
progress toward the hole, dropped 0.30 (cargo ever below floor level in the
vault), capped at 0.85; exactly 1.0 iff success. Latches clear on `reset()`.

## Solution (`solve.py`) — one transport teleport, all interactions by force

1. **P0 settle + perception**: 120 steps; read back θ0/cage_x and both cube
   poses from the live state (never hard-coded); baseline score ≈ 0.
2. **P1 align (torque)**: a velocity-servo torque about the crank axle (what a
   hand pushing the handle peg around its circle applies, |τ|≤0.8→2.4 N·m with
   stall escalation) turns the disc toward θ=±180° (direction = sign(θ0));
   active brake at the dead center. Align latch, score ≥ 0.10.
3. **P2 load (force)**: the cargo is teleported across the apron to the window
   mouth (TRANSPORT ONLY — the free carry a gripper would do), then a gentle
   force servo (≤2 N) pushes it through the window into the cage. The dead
   center holds the cage still (asserted: it moved <10 mm). Loaded latch,
   score ≥ 0.30.
4. **P3 deliver (torque)**: same crank servo continues in the same direction;
   the cage plows the cube down the sealed lane and it tips through the floor
   hole (dropped latch, score ≥ 0.85).
5. **P4 hands-off**: all wrenches off; everything settles; `success()` turns
   True on the live state (score 1.0) and holds through a **3.3 s hands-off
   persistence** window before `SIM_GEN_SOLVE: SUCCESS`.

The whole episode repeats on a second seed (fresh reset, no score prints — the
SIM_GEN_SCORE stream must stay non-decreasing) to certify seed robustness.

## Franka embodiment argument

Base pose: on the floor at ≈ (0.10, 0.45, 0), facing −y across the apron
toward the tunnel; everything it must touch lies within ~0.55 m reach and
below 0.46 m height.

- **Cube pushes (apron)**: the cubes sit on the apron at z≈0.15, 10–23 cm from
  the base. Loading is a planar fingertip push (≤2 N) across the apron through
  the 65×67 mm window (the transit surface steps down apron→sill→lane floor,
  so the slide never catches an edge) — no grasp needed (though the 46 mm cube is also a
  trivial parallel-jaw pinch). The easiest Franka primitive.
- **Crank (handle peg)**: the red peg (18 mm dia) stands proud of the disc,
  orbit r=0.145 around (0.24, 0, 0.37), top at z=0.460 — under the gantry arm
  (z≥0.470) but fully exposed from the side and above-the-disc. The robot
  pushes the peg tangentially around its circle in strokes (or pinches it and
  re-grasps); total rotation needed ≤ ~305°, torque ≤ 2.4 N·m at r=0.145 ⇒
  ≤ ~17 N tangential — comfortable. The disc is friction-held: it stays put
  between strokes, so intermittent contact suffices.
- **No reach-through cheat**: the floor hole is ≥22 cm down a 100 mm-wide,
  90 mm-tall roofed bore beyond the window, with the cage in the lane; no arm
  can thread it, and there are no tools in the scene.
- **Decoy**: reachable but must simply be left alone.

## Execution order declaration

Scene was designed first with the minimal success predicate and the geometry
contract asserted locally; `solve.py` was then run on the forge until the task
was *physically* solved on two seeds; only after that were the rubric weights
anchored to the observed trajectory and `smoke.py` finalized. No check was
ever weakened to make a run pass.

## Smoke battery (`smoke.py`) — 14 checks, all rejection/health

1. **settle/no-NaN** — cage rests at its sampled cage_x (±10 mm), cubes on the
   apron, score ≈ 0, no success.
2. **randomization A** — θ0 (via cage_x) varies across 8 seeded resets
   (span > 30 mm) and the settled cage tracks it every time (readback).
3. **randomization B** — cargo/decoy xy poses vary across resets.
4. **null policy** — 240 idle steps: the friction-held mechanism stays put,
   score ≈ 0.
5. **SEED strategy rejected** — the cargo released right above the vault for a
   REAL gravity drop (asserted it fell): it lands straddling the 32 mm roof
   slit — the carry-and-drop plan can never enter the sealed vault; score ≈ 0.
6. **wrong object** — the DECOY constructed settled on the vault floor
   (asserted really in the vault): identity matters, score ≈ 0.
7. **out-of-order** — a cube pushed through the window EARLY (constructed
   loose in the lane, yoke parked at the hole end), then the crank driven for
   REAL back to the window: the returning cage plows the cube into the
   near-end pocket (asserted it got shoved ≥3 cm); never aboard, ≤ align
   credit, no success.
8. **near miss** — cargo at rest in the lane just short of the hole edge
   (fully supported): no drop latch, no success.
9. **in-vault but moving** — cargo written inside the vault box with 0.35 m/s:
   the settle gate refuses success on the live moving state.
10. **latched credit** — the REAL solve prefix (torque-crank to dead center,
    force-push the cargo aboard; dead center asserted to hold), then the cargo
    stolen back to the apron: align+loaded latches survive (score in the 0.30
    band), success does not.
11. **empty ferry** — the crank spun >5 rad for real, the empty cage sweeping
    >25 cm: ≤ align credit — cranking without loading is worthless.
12. **rejection audit** — `success()` observed False at every step of the
    battery.
13. **final no-NaN.**
14. **video** — frames.npz (>10 rgb frames) written to CWD.
