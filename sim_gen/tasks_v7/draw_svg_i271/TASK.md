# draw_svg_i271 — Domino Relay to the Ball Pocket

**Scene:** `domino_relay` · **Env:** `simgen.domino_relay` · **Robot:** `null` (procedural geometry only)

Stand six navy dominoes in a line from the green start pad to the ball pedestal
(pad→pedestal span, bearing, pad position, and staging depot are randomized per
episode), topple the pad-end domino with **one push**, and let the travelling
cascade knock the red ball off its lipped pedestal — over its single open rear
edge — into the walled catch pocket behind it. Judged entirely on **latched
events**: every domino stood-then-fell, one bounded collapse window (≤ 3 s) that
**started at the pad** (first faller within 90 mm), and the ball's **first**
pocket entry occurring after the cascade began and within 2.5 s of the last fall,
then everything settled hands-off.

## Seed provenance

Derived from **maniskill/draw_svg**: a Franka drags a red marker cube across a
canvas so its contact trace reproduces a prescribed SVG path. Kept from the seed:
a tabletop, a red object whose motion across the surface is the deliverable, and
the "make the red thing traverse a prescribed course" framing. Everything else is
rebuilt.

## Strategic difference

The seed is **one long, continuous, contact-maintained tracing motion**: the
robot holds the marker throughout, and the judged quantity is the position trace
the arm itself generated — the arm's trajectory *is* the product. Here the
product is a **physical event the robot never touches while it happens**: the
solver first *constructs* an energy-transmission line (six dominoes upright,
spaced within topple reach across a randomized span), then injects **one**
localized trigger push and watches contact dynamics do the task — the topple
wave propagates domino-to-domino and the last one bats the ball off its pedestal
into the pocket. This demands a different **plan** (derive a feasible spacing
`s = (L − standoff)/5` from the episode's span readback so every gap stays inside
the topple-reach band, place six discrete objects on a randomized bearing, then a
single timed trigger in a narrow force window — 0.065 N tips the 50 g domino,
0.30 N slides it) and a different **code structure** (discrete placement + one
impulse + hands-off observation, instead of waypoint tracking with sustained
contact). The rubric is also structurally different: latched *events* (per-domino
stood-then-fell, collapse window, wave origin, ball arrival order/deadline), not
a trace of the robot's own motion. Against the rest of tasks_v7 (surveyed before
design): pendulum arrest removes energy, the shape sorter is aperture perception,
bar-triangle/stacking/jenga build structures that must *stay up*, beam balance
measures a hidden scalar — no existing task builds a structure whose purpose is
to **fall in a propagating sequence**, and none judges a chain reaction.

## Teleport solution (solve.py)

Transport-only teleports; all load-bearing physics is contact dynamics; one
applied trigger force, removed at 12° tilt; final state fully hands-off.

- **P0** — settle, read back the episode (pad xy, bearing, span, depot side);
  assert ball seated, no latches, score ≈ 0.
- **P1 build** — unit vector `u = (term − pad)/L`, spacing `s = (L − 0.065)/5`;
  teleport each domino upright onto the line at the terminal yaw (12 settle
  steps apiece); wait for all six STOOD latches (score = 6 × 0.04 = 0.24).
- **trigger** — one body-frame force (0.12 N) on the pad-end domino, cut when it
  passes 12° (past the 7.6° tipping angle); force zeroed. Everything after is
  hands-off.
- **P2 cascade** — the topple wave runs pad→pedestal (~0.4 s collapse window,
  first faller ~25 mm from the pad); the last domino strikes the ball over the
  pedestal's open rear edge into the pocket (~0.4 s after the last fall);
  score = 0.66. Up to 3 attempts with adjusted standoff/force on a fresh
  `reset(seed)` if a run misfires.
- **P3 success** — all down + ball settled in pocket (60-step still streak) →
  success(), score 1.0.
- **P4 persistence** — ≥ 3.3 simulated seconds hands-off; success() must hold
  throughout; scores printed at every boundary are non-decreasing (latched).

Verified on the forge: `SIM_GEN_SOLVE: SUCCESS` on **seed 0** (window 0.37 s,
ball entry +0.38 s, first faller 23 mm) and **seed 1** (0.36 s, +0.39 s, 24 mm)
with provably different layouts (bearing −17.3° vs −9.1°, pad (−0.272, +0.067)
vs (−0.290, −0.066), depot side flipped).

## Embodiment argument (Franka)

Base the Franka at the bench edge mid-way along the run line. Every manipulation
is standard tabletop pick-and-place plus one fingertip push: (1) pick each 90 ×
40 × 12 mm domino from the flat depot with a top-down pinch on its 12 mm faces
(well inside the 80 mm gripper stroke), rotate the wrist 90°, and set it upright
on the run line — placements are on an open bench with ≥ 20 mm clearance between
neighbours (spacing ~48–55 mm, domino 12 mm thick) and nothing overhead; (2) tip
the pad-end domino with one fingertip push at its top edge — 0.065–0.30 N force
window, orders of magnitude below Franka capability, and precision demands are
loose (any push past 7.6° at the top face works). The 0.26–0.34 m run plus the
depot row fits inside a comfortable reach annulus; the terminal fixture is
approached only by the cascade, never by the arm. No regrasp under constraint,
no bimanual coordination, no force control beyond a light poke.

## Execution order

**No order is required beyond causality.** The constraints (each domino stands
before it falls, ONE bounded collapse that starts at the pad, ball delivered by
the collapse within its deadline) define *what* must happen, not the sequence of
construction: the dominoes may be stood up in any order, and the scene never
checks placement order. The order-sensitive parts — cascade origin, collapse
window, ball-after-cascade — are physical properties of a chain reaction, i.e.
the task itself, not an imposed script.

## Smoke battery (smoke.py) — 15 checks, rejection-only

1. **premise** — reset readback: six dominoes verifiably flat in the depot, ball
   verifiably seated on the pedestal, states finite, score ≈ 0.
2. **randomization (run geometry)** — 10 seeds, body-pose readback: pad centre
   moves in x AND y and appears on both bench sides; bearing varies in magnitude
   and sign; span sweeps its band.
3. **randomization (depot)** — depot row swaps bench sides, staged yaws vary;
   invariants hold on every seed (span in band, ball seated, flat, score ≈ 0).
4. **null policy** — 400 idle steps: no STOOD latch, the three-sided lip keeps
   the ball seated, score ≈ 0.
5. **dunk-only** — ball teleported straight into the pocket, settles there
   (bin_streak ≥ 60): no cascade ever happened → ball_ok False, score ≈ 0.
6. **held-in-air** — a domino held verifiably vertical (up_z readback) 6 cm
   above the bench for 2×up_steps: the on-bench z-band keeps STOOD at streak 0.
7. **fell-without-stood** — a never-stood domino laid flat near the pad: 91°
   tilt, yet no fall event latches (only stood-then-fell counts).
8. **piecemeal** — chain built (legitimate 0.24 stood credit), then knocked
   down one per second: all six fall events latch but the 5 s window breaks
   window_s = 3 s → relay False, score ≤ 0.485.
9. **wrong start** — a real cascade triggered at the *pedestal* end fells all
   six in 0.3 s, but the first faller is 216 mm from the pad (> 90 mm) →
   relay False.
10. **relay-without-ball** — a short chain (last domino out of pedestal reach)
    gives a fully valid relay, but the ball never leaves its pedestal →
    ball_ok False, score ≤ 0.60.
11. **late delivery** — ball hand-dunked 3.4 s after the last fall
    (> ball_window_s = 2.5 s): settled in pocket with a valid relay, ball_ok
    still False.
12. **ball-before-cascade** — ball pre-dunked, then a full valid pad-started
    relay: everything down, ball settled in pocket, relay True — delivery
    *order* wrong → ball_ok False, NOT success.
13. **latched credit** — teleporting all dominoes back flat to the depot leaves
    the latched score unchanged (0.58 → 0.58).
14. **rejection audit** — success() was never True at any judged point in the
    battery.
15. **finite** — all task-object states finite; frames.npz saved.

Forge: `SIM_GEN_SMOKE: ALL PASS 15/15`.

## Physics notes

- Tipping angle atan(t/h) = 7.6°; topple-reach band keeps spacing < 0.65 h;
  fallen shingle rest ≈ 72°.
- The cascade arrives **quasi-statically** (lean-push, not a sharp knock), so
  the pedestal lip is three-sided with the **rear edge open** toward the pocket:
  an untouched ball stays seated (null check 4), but the last domino's strike
  (50 g domino vs 12 g hollow ball, m_eff = I/ℓ²) carries the ball off the open
  edge into the pocket.
- `fallen_deg = 30°`: the last domino can wedge against the pedestal at ~33°;
  past the 7.6° tipping angle and ~26° neighbour contact the fall is
  irreversible, so 30° is a sound "fallen" threshold, not a weakened one.
- STOOD requires upright **and** base-on-bench (z-band ±20 mm) for 12
  consecutive steps — held-in-air and mid-flight states never latch.
