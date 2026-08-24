# dough — elastoplastic dough manipulation (Newton implicit MPM)

Roll the dough out: flatten a dough ball into a thin, wide wrapper with a **rolling pin**. The
dough is a Newton implicit-MPM particle material with finite Young's modulus, von-Mises yield
stress, and full cohesion — it permanently keeps whatever shape it is pressed into. The first
(and only) scene is `dumpling` (the wrapper is a dumpling wrapper; the task is the roll-out).

## Scene: `dumpling`

- A ~4.4 cm dough ball rests at (0, 0) on the table (top at `surface_z = 0.2`; grippy:
  `table_friction = 0.8` — floured-board grip, so the sheet does not ride the sliding pin).
- A wooden **rolling pin** — ONE free rigid body: low-friction capsule barrel (axis +x, the
  dough-facing part; `pin_friction = 0.05` so a sliding pass SQUEEZES the sheet instead of
  plowing it) + a square grip stub on top + square END-CAPS over the barrel domes — rests in a
  static two-block cradle at `pin_stand = (-0.14, -0.20)`. The caps rest flat across the
  cradle slot edges and are load-bearing design: a bare barrel in the slot is an inverted
  pendulum (top-heavy stub), and MuJoCo's regularized contact friction only creeps, never
  sticks — the stub would end up lying sideways, unreachable for a top-down pinch. The
  flat-on-edges contact resists the roll by statics, not friction.
- Grasping runs the pouring suite's **auto-weld contract**: close the gripper on the stub
  within `auto_weld_dist` and the pin welds on at the measured pose; open past
  `auto_weld_release` to let go. No scripted attach calls. (The contract currently binds
  panda-handed arms — the weld row names the hand body.)

Substrate (`newton_sim.py::DoughSimCfg`): coupled MJWarp + implicit MPM via the suite-local
`coupled_manager.py` (a verbatim twin of the pouring suite's — suites must not import each
other) — `dt = 1/200`, 3 MuJoCo substeps per MPM tick, fixed
grid + CUDA graph, one-way rigid → dough (`liquid_feedback` off: the commanded pin height is
ground truth). The registered robot-less binding runs the pure-MPM manager (`coupled=False`)
with a KINEMATIC pin for material tuning.

## Registered envs

- `dough.dumpling` — robot-less physics-tuning env (kinematic pin, pure-MPM substrate).

The suite ships the TASK only. Robot bindings and solutions live outside the benchmark tree:
an experiment builds its own `EnvCfg(scene="dumpling", robot=..., ...)` on the coupled
substrate (see `experiments/`, gitignored).

Single-env only (`num_envs=1`): the MPM fixed grid spans the whole scene.

## Run

Requires the Newton venv (`env_newton`, see the root README). Build the env and read the task
from the scene itself:

```python
import robobench
from robobench.core import ENVS

robobench.discover()
env = ENVS.get("dough.dumpling")().build(num_envs=1, device="cuda:0")
print(env.scene.describe())   # the task, gates included
ok = env.scene.success()      # (N,) bool — the verdict
```

One simulator rule for any runner: NEVER record the coupled substrate live — live rendering
corrupts the coupled MPM physics. Dump states during the run and replay offline
(`scripts/replay_render.py`, local-only, last in-tree at `f8c101d`).

## Pass criteria

`scene.success()` gates the rolled sheet plus guards (sample it over a settle window and gate
medians; every threshold is a scene-cfg tunable):

| stage | metric | gate |
|---|---|---|
| rolled out | `flatten_height()` — q95 dough height above the table | <= 9 mm |
| | `spread_radius()` — q90 radius about the dough centroid | >= 38 mm |
| guards | `final_extent()` of the dough (q99.5-q0.5 span) | <= 140 mm |
| | `conservation()` (work-zone fraction) | >= 0.995 |
| | `min_particle_z()` (no table punch-through) | >= surface_z - 8 mm |

## Layout

```
robobench/suites/dough/
  __init__.py          suite registration entry (imports configs + scenes)
  newton_sim.py        DoughSimCfg — coupled MJWarp+MPM substrate cfg
  scenes/dumpling.py   DumplingSceneCfg + DumplingScene + the ball seeder + the pin spawner
  configs/envs.py      "dough.dumpling" (robot-less tuning binding)
```

## Physics findings (the load-bearing simulator facts)

1. **Push works; tear-and-carry does not.** The implicit-MPM collider contact moves dough
   robustly by NORMAL push (rolling, pressing, confining), but no pinch can tear material off
   the bulk and carry it (a pinched tongue sheds the instant the pads separate from the sheet —
   verified across pinch widths, friction, adhesion, contact bands, and pad geometries). A
   FREE blob does ride a gentle pinch, and material rests ON a supporting surface — but the
   roll-out task needs neither: it is pure pushing.
2. **Sliding beats rolling.** One-way coupling cannot spin a passive roller, so the pin
   SLIDES; with a low-friction barrel over a grippy table, a height-scheduled sliding pass
   squeezes the sheet flat exactly like a rolling pass would.
3. **Fat analytic primitives are the well-behaved colliders.** Imported thin collision shells
   barely register on the 2.5 mm MPM grid; the pin is a capsule + boxes by design. Keep every
   collider (mostly) inside the fixed MPM grid footprint — a collider fully outside corrupts
   the collider bake.
4. **Parked-tool stability is geometric.** Regularized contact friction never truly sticks —
   a top-heavy tool on line contacts creeps over. The pin's end-caps rest FLAT across the
   cradle edges (statics, not friction). Set a welded tool DOWN to verified contact before
   releasing (a strained release pops it).

## Material notes

The dough recipe derives from Newton's cohesive "mud" (`examples/mpm/example_mpm_multi_material.py`)
stiffened for shape retention: `E = 2e5 Pa`, `poisson = 0.45`, `yield_stress = 2 kPa` (self-weight
stress is ~54 Pa, so the wrapper holds at rest yet yields under the pin), `yield_pressure = 1e10`
(never crumbles), `tensile_yield_ratio = 1.0` (full cohesion), `viscosity = 20`, `friction = 0`.
Hardening/dilatancy stay 0 so the dough remains re-workable.

## Future

- Robot bindings as registered presets once the eval harness wants them back in-tree.
- Follow-on stages (filling, wrapping, molding) were prototyped and then cut for scope — the
  full dumpling pipeline (place a filling, fold/mold a wrap) lives on as an archived
  experiment with verified runs and videos.
