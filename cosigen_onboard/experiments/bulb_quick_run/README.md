# bulb_franka_osc_fable

**Task:** `assembly.bulb.franka.osc` — Franka picks up the loose light bulb (lying on its side),
stands it upright, sets its Edison cap into the fixed lamp socket, and screws it in until
`scene.seated()`. Agent: Claude Code (Fable 5), session 2026-07-13. Predecessor lineage:
`../nut_thread_franka_osc_fable` (same OSC scaffolding + paced-stroke threading).

**Status: SOLVED (run r5, 2026-07-13).**
`SOLVE[fable] | DONE | seated=1 | h=24.0mm lat=0.0mm tilt=0.0deg | strokes=21` — full task
end-to-end: pick the lying bulb at the neck, stand it upright, insert, release (it stands), regrasp
from above, thread −1185° (3.3 revolutions, 32→24 mm, pitch-consistent, every stroke 100% coupled),
release — the bulb keeps its seat on its own. **Caveat: requires the scene's `bulb_friction` dial at
0.3** (run with `--bulb_friction 0.3`); at the shipped 0.01 the task is most likely unsolvable for a
parallel-jaw gripper (see scene issue 4/5 below + BUILD_LOG final verdict). Run-by-run record in
`BUILD_LOG.md`.

Watch it live:
```bash
.venv/bin/python experiments/bulb_franka_osc_fable/solve.py --livestream 2 --bulb_friction 0.3
```

## Files

- `SUMMARY.md` — one-page digest of the whole session (strategy, run table, bugs, artifacts).
- `NOTES.md` — pre-registered plan (written before any run) + the two task-specific risks
  (90° reorientation of a lying part; friction-only torque on round glass).
- `probe_bulb.py` — no-robot go/no-go probe: geometry, release-in-socket stability, press window.
- `solve.py` — the Franka solver (state machine, OSC, per-step target latch).
- `BUILD_LOG.md` — what actually happened, run by run, including env bugs.
- `logs/` — raw stdout per run; `videos/` — recorded mp4 of a full solve.
- `session_transcript_6337eeb2.jsonl` — the Claude Code session transcript (turn-by-turn JSONL,
  for cross-model review like the nut_thread experiments; canonical live copy under
  `~/.claude/projects/-home-haoxiang-Documents-Research-CoSiGen/`).

## Strategy (see NOTES.md for rationale)

grasp the lying bulb at the NECK WAIST (self-centering — any pinch on the sloped glass barrel
watermelon-seeds the bulb out of flat jaws) → closed-loop pitch about the finger axis until the bulb
axis reads vertical (the lying bulb settles tilted ~25°, so the angle is measured, not assumed 90°)
→ servo the bulb origin onto the socket axis → touch-detect lower → release + horizontal backout
(the bulb stands in the bore if ≤ ~9° tilt) → regrasp from straight above at palm-graze depth (the
palm on the dome is the hard floor; the pad band lands 3–7 mm above the glass equator, needs μ≥~0.3
to self-lock) → friction-coupled wind strokes (120°, feedback-paced, z-follow with 3 mm lean, no hex
logic — reclose at any angle) until seated; open and retreat — thread friction holds the bulb.

## Scene bugs / possible issues found (for upstreaming into robobench)

Verified findings from probes + runs (details in `BUILD_LOG.md`):

1. **The thread mechanic itself is sound at the committed dt=1/240** — no tunneling: free-body press
   sweep 0.3–5 N all thread pitch-perfect at 2.51 mm/turn (= M20 pitch) to full seat (h=21.8 mm,
   `seated=1`). Wider press window than the M16 nut scene had at the same dt. Not a bug — worth
   knowing this env does NOT have nut_thread's dt bug.
2. **`bulb_init_z=0.024` ("~glass radius") does not match how the bulb actually rests.** The lying
   bulb settles with its Ø20 cap end drooping to the table: origin ends 5.3 mm above the surface and
   the screw axis sits ~25° off horizontal. Any policy assuming a horizontal axis / origin at 24 mm
   will mis-grasp. (Solver reads the live axis instead.)
3. **A bulb set loosely in the socket mouth is only metastable.** Released with the cap resting on
   the thread it stays up if within ~9–10° of vertical (12° falls), and a 4° lean slowly creeps
   (3.8→6.1° over ~3 s). Solvable, but the place-and-release step has a real tolerance budget.
4. **The stock `bulb_friction=0.01` applies to EVERY bulb collider, including the GLASS** — not just
   the metal cap the slickness was meant for. For a parallel-jaw robot this is close to disqualifying:
   torque on the round glass is friction-only, and at μ=0.01 the pads cannot self-lock anywhere on
   the barrel — every pinch watermelon-seeds the bulb axially (runs r1–r3). If the intent is "slick
   thread, grippable glass", the scene should use per-shape friction (cap SDF slick, glass grippy).
   Workaround used here: the official `bulb_friction` dial (whole bulb) raised to 0.3.
5. **Franka-specific geometric trap:** the gripper's palm underside is 66 mm below `panda_hand` and
   the pad band centre ~99 mm — so on an upright bulb the palm bottoms on the dome (origin+83 mm)
   before the pads can reach below the glass equator (origin+44 mm). The reachable grip band is
   always ≥~2 mm ABOVE the widest ring ⇒ pinches push the bulb UP unless friction self-locks
   (see 4). Any Franka policy must grip near/above the equator and rely on friction + a top press.
6. Cosmetic: spawn emits `Could not perform 'modify_articulation_root_properties'` on `Bulb_0`
   (articulation disable on an already-plain rigid body?) — benign so far. The smoke's
   `BULB_FREE_END=0.004` comment ("thread free end above origin") disagrees with the measured bbox
   (origin IS the lowest point); staging still works because the 4 mm just adds slack.

## Reproduce

```bash
.venv/bin/python experiments/bulb_franka_osc_fable/probe_bulb.py --headless
.venv/bin/python experiments/bulb_franka_osc_fable/solve.py --headless          # or --livestream 2
```

## Re-solved at the natural mounted layout (2026-07-14, follow-up session)

The original session solved with the scene's `surface_z=0.20` reach trick, which sinks the robot
20 cm into the table on camera. Per Haoxiang's review, `solve.py` now uses the natural layout —
robot base IN the table's robot-mount cutout at `base_pos=(-0.09, 0, 0)`, table top at the default
z=0, socket 0.50 m ahead of the base, bulb beside it at the verified 0.43 m pick radius (mirrored to
+y for wrist q7 margin). Two solver changes made it work: hover/descend timers 1.7/1.5 → 3.0/3.0 s
(bigger posture swing from home at table level; r6 stalled 60 mm high on the old timers), and the
+y bulb side. **Re-verified: `DONE | seated=1 | h=24.0mm | strokes=19`.** Fresh video:
`videos/solve_bulb_lit.mp4` (frosted glass + glow ramp + mounted layout).

## Upstreamed (2026-07-14, follow-up session)

Findings 2, 4 and 6 were fixed in robobench after this session:
- **finding 4 (the big one):** bulb.usd now binds per-shape physics materials (glass μ=0.3, cap/thread
  μ=0.01) and the scene grew a `bulb_glass_friction` tunable (default 0.3) alongside `bulb_friction`
  (cap/thread, still 0.01) — the default scene is now robot-solvable, `--bulb_friction` no longer needed.
- **finding 2/6:** the `bulb_init_z` / `BULB_FREE_END` / dt comments were corrected in place.
- Also from Haoxiang's video review: the "glass looks like a weird collision blob / penetrates the arm"
  impression was diagnosed as a RENDER issue, not physics — colliders are invisible, the glass convex
  hull tracks the visual within ~0.6 mm, and the panda hand collider is a convex hull ⊇ its visual, so
  no real interpenetration beyond the deliberate palm-press compliance. Two rendering defects stacked:
  the OmniGlass MDL rendered near-black over the dark table, AND real-time translucency sorting drew the
  glass ON TOP of the robot regardless of depth (the fake "penetration"). Fix: the glass is now OPAQUE
  frosted-white OmniPBR (depth-tests correctly; scene translucency turned off), with
  `emissive_intensity` pre-wired (authored 0) and verified live-settable — ramp it at runtime to make
  the bulb light up as it seats.
