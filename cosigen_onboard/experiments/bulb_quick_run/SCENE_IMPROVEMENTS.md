# bulb scene improvements — findings from the Franka OSC screw-in solve

Audience: a Claude session tasked with improving `robobench/suites/assembly/scenes/bulb_assembly.py`
and/or the `assembly.bulb.franka` preset in `robobench/suites/assembly/configs/envs.py`, based on the
solve in `experiments/bulb_franka_osc_fable/` (Fable 5, 2026-07-13; re-solved at the natural mounted
layout 2026-07-14). Full evidence: `BUILD_LOG.md`, `SUMMARY.md`, `README.md`, `probe_bulb.py`.

Task: `assembly.bulb.franka.osc` — pick the loose bulb (lying on its side), stand it upright, set the
Edison cap into the fixed socket, screw it in until `scene.seated()`. Verified end-to-end:
`DONE | seated=1 | h=24.0mm | strokes=19`.

## GOAL OF THIS DOC: solve with ZERO scene-init-config modifications

Today `solve.py` builds its own `scene_cfg`/`robot_cfg` and overrides the bare registered preset. The
question this doc answers: **what must change so the registered `assembly.bulb.franka.*` preset is
solvable out of the box, with no `scene_cfg` override at all?**

Short answer: the friction fix (the hard, session-consuming one) is ALREADY upstreamed. Only two
PLACEMENT overrides remain, and both are pure preset defaults — no geometry/physics change needed.

## The override set — what solve.py changes vs. the bare preset

The registered preset is bare: `EnvCfg(scene="bulb", robot="franka", control_mode=mode, env_spacing=2)`
— no `scene_cfg`, no `robot_cfg`. `solve.py`'s deltas, by kind:

| Override | Kind | Value | Still needed for a solve? |
|---|---|---|---|
| `socket_slots` | **scene init config** | `((-0.09, 0.0),)` → socket world `(0.41, 0)` | **YES** |
| `bulb_init_xy` | **scene init config** | `((-0.24, 0.25),)` → bulb world `(0.26, +0.25)` | **YES** |
| `bulb_friction` | scene init config | (not overridden; default `0.01`) | **NO** — per-shape friction upstreamed |
| `gripper_stiffness` | robot_cfg | `8000.0` (default `2000`) | yes, but not "scene" |
| `nullspace_dof_pos` | robot_cfg | `()` (home-pose nullspace) | yes, but not "scene" |
| OSC `kp`/`kd`, `rot_scale=0.15`, `TORQUE_CONTROL_DT` | runtime (post-build) | see solve.py | yes — can't live in any cfg |

So the ONLY scene-init-config modifications the solve requires are the **two placements**. The whole
friction saga — the finding that consumed the original session — is already in the default scene.

## Already handled by prior upstreaming (no change needed)

- **Per-shape friction (the big win, done 2026-07-14):** `bulb.usd` binds per-shape materials — glass
  envelope μ=0.3 (grippable), cap/thread μ=0.01 (slick thread). Scene dials: `bulb_glass_friction`
  (default 0.3), `bulb_friction` (cap/thread, default 0.01), `socket_friction` (0.75). The default
  scene is now robot-solvable; `--bulb_friction` is NO LONGER needed. Under the old uniform μ=0.01 on
  the glass, every parallel-jaw pinch watermelon-seeded the bulb axially — close to disqualifying.
- **The thread mechanic is sound at dt=1/240** — no tunneling (unlike nut_thread's M16 at 1/120). Free-
  body press sweep 0.3–5 N all thread pitch-perfect at 2.51 mm/turn (M20 pitch) to full seat. No dt
  change needed.
- **`bulb_init_z` / `BULB_FREE_END` / dt comments** were corrected in place.
- **Render fix:** glass is now opaque frosted-white OmniPBR (depth-tests correctly; the earlier
  "glass penetrates the arm" was a translucency-sorting render artifact, not physics), with
  `emissive_intensity` pre-wired (authored 0) and live-settable for a seat-driven glow.

## Why the two placements are NECESSARY (not cosmetic)

The bare preset inherits the `nut_thread.franka` "reach the socket at +x" row convention, never
re-tuned for the bulb's pick → reorient → insert → thread reach envelope. With the fixed-base Franka
at the origin and the `lab_table` at `workbench_pos=(0.5, 0)`:

1. **Bulb default world `(0.63, 0)` = 0.63 m ahead — out of reach for a table-level top-down pick.**
   A first try at ~0.69 m ended `REORIENT_STUCK` (descend stalled 60 mm high). The bulb must be pulled
   to the verified **~0.43 m pick radius** → the `bulb_init_xy` shift.
2. **Default bulb on the centerline (y=0) parks wrist q7 near its stop during the wind strokes**
   (q7 rests at −2.11, ~0.8 rad from the stop it turns toward). Mirroring the bulb to **+y** rests q7
   at +0.46 with maximal margin.
3. **Socket pulled ~9 cm closer** keeps the insert/thread poses inside that same reach band.

## The recommended fix — one preset edit (ikea-analogous)

Bake the two solve-verified placements into the `assembly.bulb.franka` preset via `scene_cfg` (the same
pattern used for `assembly.ikea_table.bimanual_franka`):

```python
scene_cfg=BulbAssemblySceneCfg(
    socket_slots=((-0.09, 0.0),),
    bulb_init_xy=((-0.24, 0.25),),
),
```

This makes `assembly.bulb.franka.{osc,impedance,joint}` solvable with **zero scene-init-config
overrides**. Optionally also bake `robot_cfg=FrankaRobotCfg(gripper_stiffness=8000.0,
nullspace_dof_pos=())` — not "scene" config, but a solver needs it (default grip stiffness 2000 won't
hold the pinch). Keep the cfg field names (`socket_slots`, `bulb_init_xy`, `bulb_friction`,
`bulb_glass_friction`) unchanged so any existing override keeps working.

**What CANNOT be baked into a preset:** the OSC gains (`kp=[150,150,150,600,600,600]`,
`kd=2√kp`), `rot_scale=0.15`, and the per-step torque latch (`FrankaRobot.TORQUE_CONTROL_DT`) are
runtime mutations on controller objects, not cfg fields — a solver must still set these, exactly as in
the ikea case.

## Task-specific tolerance notes (informational; scene is fine as-is)

- **A bulb set loosely in the socket mouth is only metastable:** released with the cap resting on the
  thread it stays up within ~9–10° of vertical (12° falls); a 4° lean creeps (3.8→6.1° over ~3 s). The
  place-and-release step has a real but workable tolerance budget. No scene change — just be aware.
- **Franka geometric trap (embodiment, not scene):** the palm underside is 66 mm below `panda_hand`,
  pad-band centre ~99 mm; on an upright bulb the palm bottoms on the dome (origin+83 mm) before the
  pads reach below the glass equator (origin+44 mm). The reachable grip band is always ≥~2 mm ABOVE
  the widest ring, so pinches push the bulb UP unless friction self-locks — hence the μ=0.3 glass is
  load-bearing for any Franka policy. Grip near/above the equator and rely on friction + a top press.
- **The lying bulb settles tilted ~25° with the cap end drooping** (origin ~5.3 mm above the table, not
  the nominal 24 mm) — read the live axis, don't assume horizontal.
