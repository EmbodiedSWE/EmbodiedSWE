# bulb_franka_osc_fable — design notes (pre-registered, written BEFORE any run)

**Agent:** Claude Code (Fable 5), 2026-07-13. **Task:** `assembly.bulb.franka.osc` — Franka picks up
the loose light bulb (lying on its side on the table), stands it up, sets its Edison cap into the
fixed lamp socket, and screws it clockwise until `scene.seated()` (bulb origin ≤ 27 mm above the
socket origin, ≤ 15 mm lateral, ≤ 12° tilt).

**IMPORTANT CAVEAT (per Haoxiang):** nobody has verified this scene is solvable *by a Franka*. The
scene's own smoke only proves a *free body* threads under an applied wrench (press −2.5 N + capped
twist −0.15 N·m at dt=1/240, staged upright by teleport). Everything robot-shaped — the pick, the
reorientation, the release stability, the friction-only torque coupling — is unproven. This file is
the pre-registration; `BUILD_LOG.md` records what actually happened, including env bugs.

**Session transcript (for the cross-model comparison):** the literal turn-by-turn Claude Code JSONL
of this session is
`~/.claude/projects/-home-haoxiang-Documents-Research-CoSiGen/6337eeb2-7b5b-48ee-a693-cccac48c002a.jsonl`
(canonical, keeps growing while the session lives); a snapshot is copied here as
`session_transcript_6337eeb2.jsonl`. Human-readable digest: `SUMMARY.md`.

---

## 1. What I start from (no runs yet)

Predecessor: `../nut_thread_franka_osc_fable` (same agent lineage) solved
`assembly.nut_thread.franka.osc` with a light-pinch **wrench** strategy: soft pinch for helix
engagement (compliance), firm pinch for wind strokes (torque), 120° hex-symmetric strokes, z
closed-loop follow, feedback-paced wind, width-feedback aligning reclose. Final:
`seated=1 | dz=12.0mm | strokes=35`, −6 revolutions of real threading.

Scene knowledge from the bulb-scene build session (memory, to re-verify in-probe):
- Socket = fixed-base articulation (M20 female thread inside a shell; bore mouth 38.5 mm above
  socket origin). Bulb = free rigid body, cap-down at identity; baked mass 0.05 kg, COM z=44 mm.
- Staged upright with the thread free end 1 mm over the bore, the bulb origin sits ~35.5 mm above
  the socket origin; screws down to ~21.8 mm at full seat. `seat_z=27 mm` ⇒ ~8.5 mm of travel ≈
  3.4 turns at the M20 2.5 mm pitch (probe must confirm pitch).
- dt=1/240 is the committed scene step and was verified to thread the *driven free bulb* cleanly
  (1/120 tunnels). The nut session's press window at 1/240 was 0.3–2.5 N (5 N cross-jams) for M16;
  probe the bulb's own window.
- Frictions (scene dials): bulb (ALL shapes incl. glass) μ=0.01, socket 0.75.
- Glass Ø48 mm, bulb ~87 mm long, glass belly (widest) at ~44 mm above the cap-end origin.
- Bulb spawns lying: quat = +90° about x ⇒ local +z (cap→dome axis) maps to world −y: dome points
  −y, cap points +y. Row default x0=+0.13 (≈0.68 m world — too far; I'll move it to −0.12 like the
  nut solve, i.e. 0.43 m) and I'll use surface_z=0.20 (the Franka reach band the nut sessions
  established).

## 2. What's genuinely NEW vs nut_thread (the two risks)

1. **Reorientation.** The nut lay flat with its thread axis already vertical; the bulb lies with
   its screw axis horizontal. The gripper must pitch the bulb 90°. A rigidly-held 90° flip ends
   with the *hand horizontal* — and you cannot thread with a horizontal hand (screwing = rotation
   about the vertical bulb axis = the wrist would have to orbit the arm around the socket). So the
   plan is **place-then-regrasp**: insert the cap into the bore with the horizontal hand, release,
   retreat, come back from straight above, regrasp the glass, thread with wrist-roll strokes like
   nut_thread. This hinges on the **released bulb standing upright in the socket mouth** (cap Ø20
   in bore Ø26, COM 44 mm up — plausibly stable, completely unverified → probe it FIRST).
2. **Round glass = friction-only torque.** The nut's hex gave geometric wrench coupling at any
   pinch. Flat pads on a Ø48 cylinder transmit torque ≈ 2·μ_eff·F·r (r≈24 mm). With glass μ=0.01
   and PhysX "average" combine vs a ~1.0 pad, μ_eff≈0.5 ⇒ ~0.6 N·m at a 25 N pinch — plenty vs the
   ~0.15 N·m the free-body drive needs. But if the combine mode multiplies (μ_eff≈0.01) the jaws
   can transmit ~nothing and can't even lift the bulb. **Measure empirically at first grasp.**
   Fallback (official dial only): raise `bulb_friction` (whole bulb, e.g. 0.3) and re-probe that
   the thread still works at dt=1/240 — the nut session showed M16 threads at μ=0.4 at fine dt.
   Upside of round: NO hex alignment logic — reclose at any angle, no corner cam-back, no
   corner-bind. Slip is graceful (a slipping clutch still transmits its max friction torque).

## 3. Pre-registered plan

1. **Probe (no robot), `probe_bulb.py`:**
   a. Geometry from BBoxCache: bulb bbox lying (grasp math), socket bbox, bore mouth height,
      resting-on-socket height (calibrates touch-detect + stage expectations).
   b. **Release-stability**: teleport the bulb upright, cap resting in the bore, zero velocity, NO
      forces, settle 2 s → does it stay upright? (tilt < a few deg). This is the go/no-go for the
      place-then-regrasp plan.
   c. **Press window**: per-env press ∈ {−0.3, −0.6, −1.2, −2.5, −5} N + capped twist (−0.15 N·m,
      cap −3 rad/s) at the scene's dt=1/240; check descent/turn ≈ pitch (real threading, not
      tunneling — the nut session caught the committed nut scene tunneling this way).
   *Predictions:* (b) stays upright (bore guides the cap, COM inside the support footprint);
   (c) 0.3–2.5 N thread pitch-consistently, 5 N jams.
2. **Solver `solve.py`** state machine (reusing the nut solve's OSC scaffolding: per-step target
   latch, kp_rot 600 / rot_scale 0.15, servo clamp, watchdog):
   - settle → hover over the lying bulb's **belly** (grasp near COM) → descend (hand down, fingers
     closing along world x, ⊥ the bulb axis) → grasp (measure grip width + hand→bulb offsets) →
     lift → **reorient**: ramp the goal quat by −90° about world x (the finger-close axis, so the
     pinch never fights the turn) so the cap swings down; hand ends horizontal → carry over the
     socket (bulb-origin xy servoed onto the socket axis using live bulb feedback) → **lower with
     touch-detect** (bulb z quiets while the target keeps sinking) → release + retreat sideways/up
     → verify upright → **regrasp from above** (hand straight down, fingers straddle the glass,
     descend so the pad band is at the belly equator, pinch) → **thread loop**: firm pinch →
     wind −120° (feedback-paced) with z-follow (hand_z = bulb_z + grip_off − lean 3 mm) → open to
     clear Ø48 → rewind +120° → reclose (any angle — round) → repeat until
     `seated()` (+ margin: stop at bulb-above-socket ≈ 24 mm) → open, retreat, settle, verdict.
   - Instrumentation: bulb height above socket to 0.1 mm, unwrapped bulb yaw vs EE yaw (slip %),
     grip width, tilt, per-phase tags.
3. **Iterate from logs.** Expected failure modes + planned responses:
   - bulb slips from jaws at lift (μ_eff too low) → raise `bulb_friction` dial + re-probe thread;
   - released bulb tips over → don't fully release: lower it until the cap is deeper (press a mm
     or two through pad slip) before opening; or catch-and-retry; worst case raise `bulb_friction`
     (glass-vs-bore contact also gets grippier);
   - jaws spin on the glass during wind (torque slip) → firmer pinch (40 N), slower wind, more
     lean (press adds thread normal force but also resistance — tune);
   - bulb tilts/jams while threading → the nut solve's bind-escape (end stroke, recycle grip);
   - regrasp knock-over → descend wide-open, centered on the *socket* axis, slow.

## 4. Falsifiable expectations

- If the release-stability probe fails AND deep-insert doesn't stabilize it, the place-then-regrasp
  plan dies; documented fallback = hold-and-thread with a tilted (not fully horizontal) hand grip
  on the upper dome — messy, and I'd report the scene as robot-hostile as designed.
- If friction coupling can't turn the bulb at any sane pinch with stock μ, the task as shipped
  (bulb_friction=0.01) is likely unsolvable for a parallel-jaw gripper; the honest report is "needs
  the bulb_friction dial ≥ ~0.2", with probe numbers.
