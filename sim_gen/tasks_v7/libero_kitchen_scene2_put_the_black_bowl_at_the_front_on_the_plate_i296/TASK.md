# bayonet_dock — lock the front canister into the keyed collar (i296)

## Seed provenance

`libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate`:
three identical akita black bowls in a row on a kitchen counter, a plate, a cabinet;
the goal is to put the bowl **at the front** on the plate. The rubric is one
unordered pick-and-place: `_terminated` checks the front bowl's xy distance to the
plate (< 0.06) and a height band (0 < dz < 0.03). Nothing about orientation, order,
or mechanism.

## What changed and why it is strategically different

Kept: the cast — three **identical black vessels in a row**, of which the **front**
one (position-defined identity, latched at reset exactly like the seed's `bowl_1`)
is the target, and a single goal fixture on the same counter.

Replaced: the entire goal semantics. The plate became a **bayonet dock** — a white
base disc with a segmented collar, an inward **catch flange** near the rim broken by
two 40°-wide **slots** on opposite sides, and four red **stop pegs** on the collar
floor. The canisters carry two side **lugs** and a knob handle. Success requires:

1. **Keyed insertion** — the canister only enters when its lugs are yaw-aligned
   (within ~9°) with the two slot gaps; set down at any other heading it perches on
   the flange 26 mm above the floor (forge readback) and is rejected.
2. **Relative twist to a hard stop** — once seated, the canister must be rotated
   **counterclockwise ≥ 45° relative to the dock** so the lugs travel under the
   flange to the stop pegs (physical stop at ~+56°, measured +52.7–52.8° at the
   torque cut). Clockwise is physically blocked by back-stop pegs at ~−8°
   (measured −7.1°), so the [45°, 90°) window is CCW-only and the folded-angle
   rubric can never be gamed by twisting the wrong way.
3. The dock's **heading is randomized over the full circle**, so the required world
   yaw differs every episode — the twist must be sensed and controlled relative to
   the dock, not memorized.

Strategy shift: the seed judges a **final xy/z window** (transport + set-down).
Here the load-bearing skills are **relative-orientation alignment** and a
**two-phase in-place manipulation** (insert through a key, then rotate under a
kinematic constraint) with a physically real "locked" outcome: the smoke battery
shows a 2 N pull (1.7× weight) extracts an unlocked canister 57 mm but moves a
locked one only 6 mm before the lugs jam on the flange. No sibling scene2 task uses
a keyed-insertion + twist-lock mechanism (they cover balance sensing, sliding-tile
vaults, trajectory-ordered stacking, dowel rings, rotary ferries, roofed pushes,
chute drops, crank lifts, slat yanks, counterweight gates, bridge building,
tilt-decanting, and hinged screens).

## Solution outline (proven on forge, seeds 0 and 1)

- **P0** settle + readback: masses (3 × 0.12 kg), dock pose and yaw from quat
  readback, front body by position readback (cross-checked against the latched
  target). Score 0.
- **P1** teleport (transport only) the front canister to a hover above the dock —
  latches the 0.15 lift stage; the canister is airborne, nothing else satisfied.
- **P2** teleport to the **aligned entry pose** (centred on the dock axis, yaw =
  dock yaw, bottom 28 mm above the collar floor, lugs fully above the flange), then
  a hands-off **gravity drop**: the lugs pass through the two slot gaps and the
  canister lands seated on the collar floor. Score 0.45.
- **P3** bang-bang **body-z torque** (0.03 N·m ≈ 2× the friction estimate,
  ×1.6 escalation on a 90-step stall, ω capped at 1.2 rad/s), cut at 52° of
  relative twist; the ~56° hard stop catches the coast; torque zeroed; settle.
  Success → score 1.0.
- **P4** hands-off persistence 3.33 s (400 steps at 120 Hz), success holds →
  `SIM_GEN_SOLVE: SUCCESS`. Both seeds: twist settles at +52.7°, d_xy 0.0 mm,
  dz +0.0 mm.

Teleports carry the canister across free space only; insertion is pure gravity +
contact, the twist is an applied torque resolved by contact against flange, floor,
wall, and pegs.

## Embodiment argument (Franka)

Base on the front counter edge at ~(0.68, 0.00, counter top), facing −x: the front
canister at (0.20, −0.25) is 0.54 m away, the dock at (0.08, 0.20) is 0.63 m —
both inside the 0.855 m reach with elbow-up top-down grasps; the other two
canisters never need to be touched. Per-object contact strategy: the knob cap
(34 mm diameter flare on a 22 mm stem, at 68–76 mm height) is a purpose-built
parallel-jaw handle — grip under the cap flare, lift vertically. Alignment and
insertion are a vertical approach with wrist-roll yaw control; the ≥45° twist is a
single wrist-roll rotation (joint 7 range ±166°), well within one grasp with no
regrasp. Required forces: 1.2 N weight, ~0.02 N·m twisting torque — trivial for
the arm; the 6 mm lug/wall clearances are far above Franka repeatability.

## Execution order (declared)

1. Wrote `scene.py` with the minimal goal predicate + staged latched score.
2. Iterated `solve.py` on the forge until `SIM_GEN_SOLVE: SUCCESS` — passed on the
   first submission, seeds 0 and 1 (17.8 s each).
3. Rubric finalized unchanged (thresholds verified against forge readbacks: perch
   dz +26 mm vs 12 mm insert gate; stops at +52.7°/−7.1° vs the 45° gate).
4. Wrote `smoke.py`; `SIM_GEN_SMOKE: ALL PASS 11/11` on the first run.

## Smoke checks (11/11 on forge)

1. settle/no-NaN + mass authoring readback, score 0, no success;
2. randomization real: 8 seeded resets → 8 distinct dock yaws, all 3 target
   bodies, jitter readback differs;
3. null policy: 240 idle steps, no latches, score ~0;
4. **seed strategy** (set-down with no yaw control): perches on the flange at
   dz +26 mm — not inserted, score caps at 0.15;
5. inserted-but-untwisted (solve P2 alone): success False, score exactly 0.45;
6. under-twist to ~28° (past the 25° partial latch): score caps at 0.75, no
   success;
7. wrong direction: CW drive blocked by back stops at −7° — no twist credit;
8. **retention contrast**: identical 2 N pull — unlocked rises 57 mm (extracted),
   locked rises 6 mm and success survives;
9. wrong (middle) canister perfectly locked: success False, score ~0;
10. set down beside the dock: not inserted, no insert credit;
11. frames.npz saved (65 × 600 × 960 × 3).
