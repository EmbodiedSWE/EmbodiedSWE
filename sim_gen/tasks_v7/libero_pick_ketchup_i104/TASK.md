# rocker_lock — operate the rocker-gate ferry to seal the ketchup inside the pantry bin

Env id: `simgen.rocker_lock` · Scene: `rocker_lock` · Robot: `null` (scripted physics solution)

## Seed provenance

Derived from `libero/libero_pick_ketchup` — *"pick up the ketchup and place it in the
basket"*. What the seed tests: identify the named bottle among grocery clutter, prehensile
pick, carry over an OPEN basket, release; a bounding-box containment check ends the episode.

What is kept from the seed:

- A target **ketchup bottle** that must be identified against a same-shape distractor
  (RED ketchup vs. YELLOW mustard, on randomly swapped floor spots).
- A **containment goal**: the episode ends with the ketchup resting inside a box-shaped
  receptacle, distractor excluded.

## What changed, and why it is strategically different

The seed's receptacle is open-topped: the entire task is grasp selection plus a carry, and
"place in the basket" is one release. Here the receptacle is a **fully sealed pantry bin** —
roof, four walls, and a single letterbox window that is **plugged by the tip of a rocker
beam at rest**. The plug is not decoration: every static gap around the closed gate is
smaller than the bottle diameter, *asserted in `__post_init__`* (3.8 cm top aperture, 1.5 cm
beam-to-wall standoff, skirt overlapping the sill band, rails overtopping the lintel, beam
face masking the window sideways vs. a 5.5 cm bottle). No direct placement — from any side,
at any speed, in any orientation — can put a bottle inside.

The only way in is to **operate the machine**:

1. **Load** — lay the ketchup sideways into the walled tray channel at the rocker's low
   loading end (it rests against the end wall).
2. **Press and hold** — push the green side paddle down (~25 N). The see-saw pitches on its
   pure-contact journal onto the press stop; its tip sinks to sill level, un-plugging the
   window; the tray becomes a ramp; **gravity ferries the bottle** down the channel, over
   the tip and through the window into the bin. The hand never touches the bottle again
   after loading — every centimetre of its travel is produced by the machine.
3. **Release** — the beam's authored center-of-mass bias re-parks it against the rest stop,
   re-plugging the window and **sealing the bottle inside**.

Strategy vs. the corpus tasks read for this construction: `pen_holder` is multi-insert into
an open cup; `cellar_tow` (i43) builds and unmakes a peg-in-eye tool coupling and tows;
`roll_in_garage` (i292) relocates a blocking post and then manually push-rolls the target
through a floor doorway. Here the solver **never pushes or carries the target to the goal at
all** — the plan is load-then-actuate-then-release on a gated ferry whose gate is the
machine's own moving part, and the end state (gate closed, bottle captive behind it) is one
that direct manipulation can never produce. The seed's entire plan — carry the bottle over
the receptacle and drop — lands it on the bin **roof** and scores nothing (constructed and
rejected in smoke).

## Scene (procedural geometry only)

- **Rig** (one 40 kg dynamic fixture, teleported coherently at reset): two journal stands
  (capped U-notches, low-friction), rest stop, press stop, and the sealed blue bin — floor,
  sill band / mullions / lintel forming a 16 × 10 cm letterbox window (sill 14 cm, lintel
  24 cm), side walls, far wall, roof at 25.4 cm.
- **Beam** (0.6 kg, yellow): steel axle + end flanges riding the journal (pure contact, no
  joints anywhere), 46 cm walled tray channel, load-end stop wall, tip skirt (the gate
  plug), steel ballast plate, and a green press paddle cantilevered off the +y rail.
  Authored CoM bias (`CreateCenterOfMassAttr`) parks it tip-up at +8.0°; full press is
  −6.5°. Rest aperture above the tip: 3.8 cm < 5.5 cm bottle; pressed aperture: 7.8 cm.
- **Ketchup** (red) / **mustard** (yellow): identical 5.5 × 14 cm cylinders.
- Per-seed randomization (verified by state readback in smoke): rig xy jitter ±3 cm and yaw
  ±12°, the two bottles **slot-swapped** between their floor spawns (`torch.rand`
  comparison, not the degenerate first-randint draw) plus ±5 cm xy jitter and free yaw.

## Teleport solution (solve.py) — teleport is transport only

- **P0** settle + layout readback; assert gate closed (+7.6°), axle seated, bottles on the
  floor, score ≈ 0.
- **P1** teleport-carry the ketchup from its floor spawn to the tray's open loading bay
  (lying across the channel, zero velocity, open sky above — a plain pick-and-carry);
  gravity seats it against the end wall. Loaded latch fires (0.15).
- **P2** press: an emulated hand force at the paddle — world-frame downforce ramped
  4 → ~27 N at 12 N/s plus the paddle-lever torque τ = (R_beam·r_paddle) × F — through
  `encode_force` with a runtime force-frame probe (mode 1 then rollback to mode 0; the beam
  only rotates ~15°, so the probe is belt-and-braces). The beam pitches onto its press
  stop; the hold continues while gravity rolls the bottle down the ramp, over the tip and
  through the window (opened 0.35, delivered 0.75). Forces held ~0.5 s after delivery.
- **P3** release: forces cleared; the CoM bias re-parks the beam against the rest stop
  (+7.6°), sealing the window with the ketchup inside → success, score 1.0. Hands-off
  persistence 400 steps (3.3 s) with success held, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds 0, 1, 2 — score trace 0.00 → 0.15 → 0.75 → 1.00, monotone,
success persists ≥ 3.3 s hands-off.

## Rubric (anchored in the demonstrated solve)

Latched milestones (updated only in `score()`, non-decreasing, serialized in
`get_state`/`set_state`): loaded in tray (0.15) → gate opened while loaded (+0.20) →
delivered into the bin volume (+0.40) → 1.0 iff `success()`. `success()` requires ALL of:
ketchup resting inside the bin interior below the sill band and settled; gate CLOSED (beam
re-parked within 3° of rest tilt and settled) with the axle still seated in its journal
(machine intact); mustard outside the (margin-grown) bin volume; all states finite. The
partial states — bottle still in the tray, or delivered but the paddle never released —
score at most 0.75 and are not success.

## Embodiment argument (Franka, 80 mm parallel jaw)

- The bottles are 5.5 cm across — a comfortable side or top pinch within the 80 mm jaw
  span; the loading bay is at the beam's LOW end under open sky, reachable from above.
- The press paddle is a 5 × 6 cm plate, 1.2 cm thick — a natural pinch or open-jaw push
  target; the measured press is ~25–27 N straight down, well inside Franka's quasi-static
  push capability, applied at ~19 cm height.
- Loading bay, paddle and window span ~75 cm of workspace at ≤ 25 cm height; a base posted
  on the paddle side (−y) of the rig reaches the bottle spawns, the bay and the paddle
  without exceeding ~85 cm reach. Press-and-hold is single-arm: after loading, only the
  paddle needs contact.

## Execution order is physically forced

The interlock is asserted in `__post_init__`: every closed-gate aperture undercuts the
bottle diameter while the open gate passes it with ≥ 2 cm margin — so entry REQUIRES the
press, and the press must be held while the bottle transits (releasing early re-plugs the
window). Success further requires the gate CLOSED again and the machine intact, so "press
and walk away" fails the gate clause, wrecking the beam fails the seat clause, and the
seed's carry-and-drop strands the bottle on the roof. Each of these is constructed and
asserted rejected in smoke.py; the seal itself is probed dynamically (a 1.6 m/s slam up the
ramp reaches the gate and does not pass).

## Checks (smoke.py, 21)

1 settle/no-NaN + layout · 2 baseline score ≈ 0 · 3–5 randomization real via readback (rig
pose spread; slot swap occurs both ways and always opposite; per-bottle jitter) · 6 null
policy 300 steps ≈ 0 · 7 press mechanism real (32 N paddle wrench opens to −6.7°, released
bias re-parks, non-vacuous both ways) · 8 closed gate blocks a 1.6 m/s ramp slam (reaches
the tip zone by readback, never crosses the window plane) · 9 seed strategy (carry + drop
over the receptacle) lands on the roof, rejected · 10 wrong object (mustard inside by fiat)
rejected · 11 gate-left-open (delivered, beam kinematically held pressed) rejected ·
12 acceptance construct (hold released, bias re-parks with cargo sealed) accepted ·
13/14 mustard added inside flips success off / set_state restore flips back on ·
15/16 beam dumped off its journal flips off / restore flips on · 17 settle gate (sealed
bottle kicked, judged moving → rejected; resettled → accepted) · 18 near miss: loaded in
tray only (0.15, no success) · 19 near miss: standing against the bin's outer wall ·
20 rejection audit (success never True except 12/14/16/17b) · 21 final finite-state audit.
Frames recorded to `frames.npz`.
