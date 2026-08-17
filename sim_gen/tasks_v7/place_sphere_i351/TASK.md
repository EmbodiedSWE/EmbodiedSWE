# place_sphere_i351 — `ball_pump`

The seed's red sphere starts LODGED INSIDE a bolted-down pump machine's internal
single-file conduit — unreachable by any gripper. The machine has no buttons,
levers or joints: the only actuator is CONSUMABLE MEDIA. Drop white supply balls
into the intake funnel on top; each adds its weight to the column standing in the
vertical shaft, and the column quasi-statically drives the single-file chain
around a 45° bend, along a roofed flat passage and up a roofed 30° incline until
the red ball tips over the crest and drops into the machine's open EXIT TRAY.
How many feeds that takes is not told — it varies with the red ball's random
start depth x0 — so the exit must be WATCHED, not counted to. Then pick the red
ball out of the exit tray and place it in the free-standing green GOAL BIN.

- **Env name:** `simgen.ball_pump` (robot `"null"`, scene-level task)
- **Package:** `scene.py` (scene + rubric), `solve.py` (legitimacy certificate),
  `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `maniskill/place_sphere`
(`RoboVerse/roboverse_pack/tasks/maniskill/place_sphere.py`): a Franka grasps a
red sphere, carries it, and balances it ON TOP of a small shallow bin; a
`RelativeBboxDetector` box above the bin judges the placement. The sphere is
directly reachable and placement precision IS the task.

## What changed and why it is strategically different

| Axis | Seed | This task |
|---|---|---|
| Sphere reachability | on the table, graspable from step one | **CAPTIVE** — lodged inside a roofed internal conduit; no gripper can touch it until the machine has ejected it |
| Actuation | direct grasp of the judged object | **indirect, via consumable media**: the 9 white balls ARE the actuator; their committed weight is what moves the red ball (the machine has zero moving parts) |
| Placement precision | the whole task | **deliberately zero** — the funnel catches any drop, the conduit is single-file by construction, the goal bin is a large open box |
| Plan length / structure | one atomic pick-place | **repeat-until-observed**: feed, watch the exit, feed again … then a final pick-place; the required feed count varies per episode with x0 |
| Scene reading | fixed goal pose | free machine yaw ±180° + jitter (funnel, crest bearing and exit tray all move), tray/bin side-swapped bands, hidden x0 |
| Failure modes | drop / misplace | wasting media is harmless (whites may land anywhere); the only wrong answers are shortcuts, all gated by the latch chain |

The seed's literal strategy — carry the red sphere and set it down at the goal —
is reproduced verbatim in the scene (drop the red ball straight into the goal
bin) and scores ~0 (smoke check 9): the deliver latch is gated on having come
out THROUGH the machine.

## Why it is different from the rest of the corpus

Nearest neighbours checked and the discriminating feature:

- **drain_plug (`i196`, same seed)**: there the sphere is a TOOL (a valve seated
  before pouring); here it is the captive GOAL object, extracted indirectly. No
  irreversible ordering here — instead an observation-terminated repeat loop.
- **ram_dispenser / bottom_dispense magazine recipes**: dispensing exists, but
  driven by a mechanical plunger/joint the robot presses; here there are NO
  joints — the drive force is the accumulated weight of committed free bodies.
- **gravity_ram chute**: one ball delivers one impact; here the mechanism is
  quasi-static displacement pumping — a standing column drives a chain, and the
  media stays committed inside the machine.
- **marble_router / checker_silo / sieve sorters**: routing/sorting of the
  dropped bodies themselves; here the dropped bodies are never judged at all —
  they are fuel, and the judged object is a DIFFERENT ball they push.
- **captive-ring rail traversal**: captivity there constrains HOW the object
  moves under direct manipulation; here captivity removes direct manipulation
  entirely until the machine releases the object.

No corpus task has (a) the judged object starting unreachable inside a sealed
fixture, (b) consumable free bodies as the sole actuator of a joint-less
machine, and (c) a per-episode-variable repeat count that must be terminated by
observing the exit rather than by a known count.

## Scene summary

Kinematic **pump** at ~(0.52, 0.05) ± 2.5 cm, yaw FREE ±180°. Local frame:
origin on the ground, +x = flow. Interior channel 47 mm square (balls Ø40:
single file, no climbing headroom, proved). Vertical shaft (interior
x ∈ [−0.1035, −0.0565], mouth z = 0.240) with a 45° four-plate intake funnel
above (~130 mm square opening); parallel-wall 45° elbow at the bottom (chamfer
floor + hood plate 45.3 mm away along the normal — a plain wall-corner bend has
a ~33 mm throat and wedges the ball, proved by assert); roofed flat passage
(floor top 0.006, roof interior 0.053); roofed 30° incline to a crest at
(0.0901, 0.058); past the crest a walled open-top EXIT TRAY (x ∈ [0.075,
0.232]). All conduit surfaces carry an authored SLICK material (μ 0.06/0.05 —
pair-averaged-friction trap) and the exit tray a GRIP material. Dynamic **red**
sphere r = 20 mm starts lodged in the flat passage at random depth
x0 ∈ (−0.042, −0.016); 9 dynamic **white** spheres (same size, mass 60 g,
damping 0.12/0.30 as rolling-resistance stand-in) start in a 3×3 grid in the
kinematic supply **tray** (band-sampled side, free yaw); the kinematic green
goal **bin** stands on the OPPOSITE side (band + free yaw).

`__post_init__` PROVES the mechanism: single-file/no-climb channel, elbow
throat ∈ (2r+4 mm, 4r), sealed flat→incline roof transition (3.7 mm gap) with
the centre path clearing both corners, quasi-static drive margin (worst case:
shaft column ≥ 2× incline resistance), chain capacity for all 10 balls, crest
drop ≥ 30 mm, eject-window z-ceiling excludes any in-conduit rest (centre
z ≥ 0.078 > 0.060), red start band resting clear of the chamfer, deliver window
covers every bin rest pose, fixture bands never overlap the pump footprint.

Randomised per seed (readback-verified in smoke): pump yaw ±180° + xy jitter,
x0, tray band/side/yaw, bin band/yaw opposite side, per-ball slot jitter (the
first post-seed draw is burned — degenerate-first-draw trap).

## Rubric

- Latch chain (updated every `post_step`): **progress** = latched max of the red
  ball's normalized advance (x − x0)/(0.086 − x0), counted ONLY while provably
  inside the conduit — over the roofed run the membership z-bound follows the
  roof interior, so a ball perched on the machine's roof EXTERIOR earns nothing
  (smoke check 6); **eject latch** = still-streak (0.05 m/s × 24 steps) in the
  exit-tray window, gated on progress > 0.999 (a ball teleported into the exit
  tray latches nothing — check 8); **deliver latch** = still-streak in the goal
  bin, gated on the eject latch (a ball dropped straight into the bin can never
  succeed — check 9).
- `success()`: red ball inside the goal bin LIVE + slow (0.15 m/s judge gate,
  above the PhysX sharp-edge phantom band) + deliver latch.
- `score()`: 0.55·progress + 0.15·eject, latched and monotone, capped at 0.70;
  exactly 1.0 iff `success()`. Null policy scores ~0 (nothing moves).

## Teleport-solution outline (`solve.py`)

Teleports are zero-velocity HOVER RELEASES only — every millimetre of the red
ball's conduit progress is earned by the pushing chain (pure contact dynamics);
no pose write ever touches the red ball while it is inside the machine, and no
write ever lands a body in a scoring region:

1. **P0** settle + layout readback (pump pose/yaw, x0, tray, bin); assert
   baseline score ≈ 0.
2. **P1 feed loop**: each white in turn is written to a zero-velocity hover
   above the funnel throat and released (a grasp-carry-release); gravity drops
   it through the funnel into the shaft, and the column drives the chain around
   the elbow, along the flat and up the incline. The solver only WATCHES the
   red ball's readback for the crest drop (6 feeds on seeds 0–2; the count is
   x0-dependent). Score climbs monotonically with each feed.
3. **P2 deliver**: once the red ball rests in the OPEN exit tray it is
   reachable; one hover release above the goal bin; gravity lands it.
   `SIM_GEN_SCORE 1.0000`.
4. **Deep settle** (≤5 s to max body speed < 0.03 m/s for 30 steps), then
   **P3 persistence**: 3.5 s fully hands-off with per-block telemetry;
   `SIM_GEN_SOLVE: SUCCESS` only if success() held throughout.

## Embodiment argument (single Franka + parallel jaw 80 mm, OSC)

Plausible base pose: world origin, facing +x. Supply tray at ≤ 0.46 m, goal bin
at ≤ 0.47 m, pump funnel centre at ≤ 0.63 m horizontal / 0.28 m high, exit-tray
far reaches ≤ 0.78 m — all inside a 0.85 m Franka reach envelope.

- **White balls (Ø40 < 80 mm jaw)**: top pinch grasp out of the open supply
  tray (walls 30 mm — no approach constraint), carry to above the funnel
  (opening ~130 mm square at 0.24–0.27 m — an easy wrist-down release), open
  the jaw anywhere over it. Repeat while watching the exit tray (open top,
  fully visible from above). No precision at any step.
- **Red ball**: same top pinch out of the open exit tray (walls 45 mm), release
  anywhere over the green bin (110 mm inner opening, walls 50 mm).
- **Fixtures (pump/tray/bin, kinematic)**: never manipulated; the machine has
  no joints to operate.
- Nothing requires two hands, regrasp, in-hand manipulation, or force control;
  every contact is a vertical pinch + a release over an open top.

## Execution-order declaration

Feed-before-retrieve is *physically forced*, not convention: the red ball is
unreachable inside the roofed conduit until the machine ejects it (there is no
opening a gripper fits through), so no reordering can reach the goal. The
rubric mirrors the physics — deliver latch gated on eject latch gated on full
in-conduit progress — and smoke checks 8/9 prove both shortcut teleports
(straight to the exit tray / straight to the bin) score ~0 with success False.
The feed COUNT is deliberately undeclared: it varies with x0, so the policy
must observe the exit rather than execute a fixed script.

## Execution order (how the package was built)

1. `scene.py` first — geometry derived and proved in `__post_init__`.
2. `solve.py` iterated on the forge until `SIM_GEN_SOLVE: SUCCESS` on seeds
   0/1/2 (iteration: the naive wall-corner elbow wedged the chain — telemetry
   showed the whites stacking vertically in the shaft; fixed by rebuilding the
   bend as a parallel-wall elbow with a hood plate and adding the throat
   assert).
3. Rubric hole found while designing smoke (roof-exterior perch would have
   earned progress) — conduit membership z-bound tightened to follow the roof
   interior; solve re-validated on all three seeds.
4. `smoke.py` rejection battery on the forge, frames.npz recorded.
5. `TASK.md` + final clean runs.

## Check list (smoke, 14/14)

1. settle/no-NaN + layout sanity (red at rest at its sampled x0 in the flat
   passage, all 9 whites at rest in the supply tray, all still)
2. score ~0 at reset, no success
3. randomization readback (8 seeds): pump yaw spans a wide arc (−43° … +150°
   observed), xy jitter real, tray side flips
4. randomization readback: x0 varies (span > 5 mm; −0.041 … −0.020 observed)
5. null policy: 240 idle steps → score ~0
6. roof perch: red at rest ON TOP of the machine roof, inside the conduit xy
   footprint (z = 0.085) → conduit membership False, progress stays 0, score 0
7. wrong object: a WHITE ball settled in the goal bin and another in the exit
   tray → nothing latches, score 0
8. exit-tray shortcut: red constructed at rest IN the exit tray without conduit
   passage → eject_now True but the progress gate keeps the latch off, score 0
9. bin shortcut (the seed's own strategy): red dropped straight into the goal
   bin → deliver_now True but the eject gate keeps success False, score 0
10. feed non-vacuity: 2 whites genuinely fed → chain advances, score 0.166 > 0.05
    (the zeros of 8/9 are gates, not a dead rubric)
11. latched credit: the mid-conduit red then yanked to open ground → latched
    progress score holds (0.166 → 0.166), no success
12. eject control: fresh episode fed to full genuine ejection (6 feeds) → eject
    latch earned, score == 0.70 cap, success still False (never delivered — the
    battery never constructs success)
13. rejection audit: success() never True at any judged point
14. final no-NaN; frames.npz saved (314 × 600 × 960 rgb)

## Validation evidence (forge, RTX-4090 pod)

- `solve --seed 0`: SUCCESS, 20.0 s — 6 feeds; `SIM_GEN_SCORE` monotone
  0.00 → … → 0.32 → 0.41 → 0.44 → 0.70 (eject latch) → 1.00 (delivered) → 1.00
  (persistence); deep settle 0.007 m/s; success held all 10 persistence blocks.
- `solve --seed 1`: SUCCESS, 20.2 s (pump yaw +61.9° vs +170.0° at seed 0,
  x0 = −0.0356 vs −0.0177 — the bearing and the depth must be read).
- `solve --seed 2`: SUCCESS, 20.3 s (yaw −153.4°, x0 = −0.0380, tray/bin sides
  flipped: tray on −y).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 14/14`, 42.9 s, frames.npz saved.
