# change_channel_i249 — ChannelConsoleScene

Env: `simgen.channel_console` · robot: `null` (scene-level task) · assets: fully
procedural (boxes only, custom compound spawners).

## Seed provenance

Seed task: `rlbench/change_channel`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/change_channel.py`): a Franka picks
the hand-held **TV remote** off the table, aims it at the TV frame, and **presses one
channel button**. Strategy family: one grasp of a free-standing device, one
transport/aim, one discrete press on that device.

## What changed, and why it is strategically different

Same theme (select a TV channel), different *plan* and different *code structure*:

- **Nothing is hand-held and nothing is pressed.** The "channel control" is a fixed
  kinematic **console** with a **captive selector**: an orange-tabbed slider rides
  inside a slotted C-channel (overhanging lips leave only a 16 mm slot the tab
  protrudes through — a force probe in smoke proves the slider cannot be lifted
  out). Channels are **positions along a continuous track**, not buttons: the
  selector must be left at rest with its center inside a ±16 mm band.
- **A physically-enforced execution order.** A **lock pin** (square shaft, wider
  square head) seats 36 mm deep in a socket mid-track; while seated, its shaft
  fills the channel cross-section, and the target band is *always sampled on the
  far side of the pin* from the selector's start. So the pin must be extracted
  (a guided ~75 mm vertical pull through socket, floor hole, and lip slot) **before**
  the slide can succeed — smoke proves a pushed slider stalls at the pin with a
  regulated 3 N probe, far short of every far-side band.
- **A symbolic goal.** Four colored tick plates (red/green/blue/yellow) mark the
  bands; a single colored **indicator cube** on the rear pedestal names the target
  per episode (the matching cube is swapped onto the pedestal at reset; spares are
  parked in a floor depot). The agent must read a color and map it to a position.

A solver therefore needs: read a symbolic cue → contact-guided vertical extraction of
an interlock → closed-loop continuous positioning within a tolerance band. The seed
needs: grasp pose → aim → discrete button press. No overlap in plan or in code
structure. Also distinct from the corpus packages read while building this task
(`basketball_in_hoop_i128`: push-only ball conveyance up a switchback ramp;
`pen_holder` exemplar: multi-object insertion into a cup) — neither has an interlock,
a captive prismatic mechanism, or a color-matched symbolic goal.

## Execution-order declaration

1. Read the indicator cube's color → target band `x_target` (readback in scripts).
2. **Extract the pin** straight up until its shaft bottom clears the lip slot
   (~75 mm of guided travel); set it aside anywhere out of the track.
3. **Slide the selector** along the track, across the now-empty interlock column,
   into the color-matched band; release and let it settle.

Order is load-bearing: step 3 is physically impossible before step 2 (asserted in
`ChannelConsoleSceneCfg.__post_init__` — worst-case pin tilt still keeps every
far-side band unreachable — and proved by the smoke interlock probe).

## Teleport-solution outline (`solve.py`)

Teleports = **transport only**; every load-bearing interaction is contact dynamics:

- **P1 — pin extraction (dynamics):** vertical velocity-servo force
  (`F = m·g + kv·(v_des − v_z)`, capped 4 N, gain — not cap — escalation on stall)
  lifts the pin out of the socket, up through the floor hole and the lip slot, a
  guided contact extraction the whole way. Only after readback confirms the shaft
  bottom cleared the slot is the free pin teleported to the ground depot (pure
  transport across free space).
- **P2 — slide (dynamics):** horizontal velocity-servo force in the console's
  canonical frame pushes the captive slider across the interlock column and the
  floor-hole seam into the band, then releases; friction/track finish the job. The
  success state is never spawned.
- Non-decreasing `SIM_GEN_SCORE` at each phase boundary; ≥ 3.5 simulated seconds
  hands-off persistence after `success()` first holds; `SIM_GEN_SOLVE: SUCCESS`.
- All targets (console yaw/xy, start side, target band) come from per-episode scene
  readbacks, so one script serves every seed.

## Embodiment argument (single Franka + parallel jaw, OSC)

Plausible base pose: at the console's front-left, facing the track (console is
0.60 × 0.34 m, track at ~0.25–0.28 m height — comfortably inside a Franka's
workspace from the floor next to it).

- **Pin:** the 20 mm square head sits proud of the track top (~0.33 m height) with
  clear grasp access from above; 20 mm ≤ 80 mm jaw span (asserted in cfg). Grip the
  head, pull straight up ~75 mm, place aside. Exactly the guided vertical extraction
  P1 emulates with a vertical servo force.
- **Selector:** the orange tab protrudes ~48 mm through the slot — a jaw pinch on
  the tab (10 mm thick) or a fingertip push on its side both work; the slide is a
  horizontal guided push along a straight track, exactly P2's horizontal servo. No
  regrasp gymnastics; both interactions are single-handed, top-down.
- The indicator cube is read visually (color), never touched.

## Rubric

- `success()`: slider **in-track** (canonical |y| ≤ 10 mm, |z − rest| ≤ 7 mm — an
  imposter on the lip tops reads ~38 mm high and is rejected) **and** center inside
  the target band (±16 mm) **and** settled (|v| < 0.04 m/s, |ω| < 1 rad/s).
- `score()`: latched partial credit — 0.20 pin-clear + 0.15 crossed-the-column +
  0.25·best-approach (normalized by start distance), cap 0.60; 1.0 iff success.
  Null policy ≈ 0 (start ≥ 60 mm from any band edge, pin seated).

## Check list (smoke.py — 18 checks)

1. settle/no-NaN + full layout sanity (console/pin/slider/cubes readback)
2. score ~0 at reset, no success
3. randomization readback: console yaw/xy jitter varies, sane on 8 seeds
4. randomization readback: both sides, ≥ 2 target channels, start varies, pedestal
   cube matches the target every seed
5. null policy (240 steps) → score ~0
6. seed-strategy transplant (slider at the target COLOR on the open deck) rejected;
   grasp+press N/A by constructed premise
7. wrong channel (in-track, settled, non-target band) rejected
8. band near-miss (8 mm outside band_tol) rejected
9. interlock force probe: ≤ 3 N quasi-static push moves the slider ≥ 25 mm then
   stalls at the seated pin; crossed never fires; pin stays seated
10. captivity force probe: ≤ 4 N upward pull lifts the slider only 2–14 mm (lips);
    re-settles in-track
11. on-lips imposter at the target x rejected by the z window
12. settle gate: in-band but moving ≠ success
13. part-way cap: all three latches constructed → score == 0.60 cap, not success
14. latched credit invariant under removing the slider
15. wrong object: extracted pin laid across the track at the band ≠ success
16. rejection audit: success() never True at any judged point
17. final no-NaN
18. camera ≥ 20 frames → frames.npz

## Verification (forge, final package)

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1 (side=+1 → green) and 2
  (side=−1 → yellow, the opposite branch); non-decreasing `SIM_GEN_SCORE`
  0 → 0.20 → 1.0 → 1.0 with 3.5 s hands-off persistence on every seed.
- `smoke.py`: `SIM_GEN_SMOKE: ALL PASS 18/18` (randomization readback hit all
  four target channels and both sides across the 8 audit seeds; 299 rgb frames
  → frames.npz).
