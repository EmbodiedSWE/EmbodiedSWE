# Task `close_drawer_i58` — Return Dock

Scene: `return_dock` (env `simgen.return_dock`, robot `"null"`).

## Provenance

Seed task: `rlbench/close_drawer` — "close the drawer": the hand pushes the drawer
front until its prismatic joint reads closed. One guided translation OF the judged
part; the drawer's position IS the goal.

Kept from the seed: a fixture with a sliding drawer-like part (an open-top tray in
a channel under a roofed enclosure), and the identical judged OUTCOME — the slide's
front face flush with its doorway, "drawer closed", judged on the slide's terminal
position. The tray here is a free rigid body captured between guide rails on a
polished runway (contact physics, not an articulation), reproducing the seed's
drawer by mechanism rather than by joint.

## Strategic difference

**vs the seed:** in the seed the robot ACTUATES the judged part — the whole skill
is a push on the drawer front. Here the ROBOT NEVER TOUCHES THE TRAY. The station
stands on a wedge plinth, visibly pitched nose-up 8°, so the tray glides shut BY
ITSELF the moment its path is clear; what holds it open is a pair of obstructions,
and clearing them is the entire task: (1) a red stop pin standing in one of three
runway sockets, loaded by the tray's own downhill press — extract it VERTICALLY
and park it clear; (2) a blue bottle standing in the open-top tray, taller than
the doorway — even pinless, the tray stalls when the bottle meets the roof header;
lift it out and dock it UPRIGHT inside the green rimmed pad on the enclosure's
roof. The seed's plan is one push; this plan is REMOVAL (extract a loaded pin;
relocate a container's payload to a rooftop dock), with the judged closing motion
delegated to the scene's own mechanism. The seed's skill executed for real is
provably useless — smoke pushes the tray with a real 8 N PD force for 3 s and the
socketed pin arrests it after millimetres, score ~0.

**vs the corpus surveyed this session:**
- `gumball_meter_i34` (nearest slide cousin — force-driven prismatic shuttle):
  there the robot DRIVES the slide, repeatedly, and the objective is a count
  (K cycles + restraint). Here the slide is never driven by the robot at all —
  the new axis is obstruction-clearing + a self-acting mechanism, not metering.
- `pen_holder` (robobench packing exemplar): pick-and-insert into a passive
  container; nothing moves by itself and nothing blocks anything. Here both
  manipulated objects are INTERLOCKS whose removal releases a stored-energy
  mechanism, and one of them (the pin) is under live mechanical load.
- The gumball corpus survey (`matchbox_drawer_i11`, `carousel_airlock_i9`,
  `chute_switch_i14`, `ram_eject_i1`, `tunnel_shuttle_i3`, `silo_scoop`,
  `cam_press_i35`): every one has the robot actuate the mechanism (pull, rotate,
  push, press) to reach a terminal state. None makes "do NOT actuate the judged
  part — clear its path and let gravity act" the objective.

## The machine (honesty by construction)

`scene.py` asserts the interlock contract in `__post_init__`: the 8° incline beats
the polished runway friction 2× (the mechanism is really alive — without the bound
mu 0.06/0.05 min-combine material, PhysX's default ~0.5 friction would kill it,
tan 8° ≈ 0.14); the standing bottle really fouls the doorway header (+22 mm) while
the empty tray, its grab bar, and a knocked-over bottle really pass under it — the
fallen-bottle clearance is asserted against the WORST resting pose (bridged across
both wall tops, wall_top + a full shaft diameter), not the friendly lean; the pin
stands proud of the tray walls by >30 mm (a clean
top grasp exists), drops freely into its 21 mm well, and the well is deep enough
to hold it upright under the tray's press; the tray at every socket sits clear of
the end stop; the rear-wall hard stop IS the closed pose (gap < closed_tol); the
docked bottle is statically stable on the pitched pad (tip angle 17.8° vs 8°+3°
margin) and the grippy pad override (0.90/0.80, "average") pins it there. The
header edge itself stays SLICK so a stalled bottle is pushed purely horizontally —
moment analysis: ~0.7 N of tray drive at 4.9 cm above the bottle's centre against a
0.055 N·m gravity restoring moment (1.13 N tipping threshold, 1.66× margin) — the
stall is a stable stall, not a launcher.

## Teleport solution (`solve.py`) — teleports are transport only

- P0 — settle; read back dock pose/yaw, WHICH socket, the opening x0; baseline ~0.
- P1 — BOTTLE (applied force): PD carry + gravity feedforward (≤8 N, a firm
  grasp) lifts the bottle straight up along the dock's local up, out through the
  tray's open top — `bottle_out` is judged DURING the force lift; then the held
  bottle is teleported over the pad and RELEASED 2 cm up; gravity seats it on the
  grippy pad. Score 0.40.
- P2 — PIN (applied force): vertical PD extraction from the socket against the
  well walls and the tray's live press; once in free air it is teleported to open
  ground beside the station and released upright. Score ≥0.60.
- P3 — HANDS OFF: the tray glides ~0.15–0.22 m down the runway at ~0.11 m/s
  terminal (linear damping 8), through the doorway, onto the rear-wall stop —
  `success()` in ~140–300 hands-off steps. Score 1.0.
- Persistence — ≥3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge on all nine seeds 0–8, covering all three sockets (0: seeds 1,3;
1: seeds 0,2,4,5,8; 2: seeds 6,7 — the longest glide), yaws spanning −132° to
+164°, distinct dock xy; self-close in 76–189 hands-off steps. `SIM_GEN_SCORE`
printed at each boundary, non-decreasing (0.00 → 0.40 → 0.71–0.77 → 1.00 → 1.00).

## Embodiment argument (Franka, single arm)

Base ~0.5 m from the station on the apron (+x, uphill) side; everything touched
lives at world z 0.05–0.30 within a 0.35 m disc around the dock — comfortably
inside Franka's workspace at any episode yaw (the ±5 cm xy jitter and free yaw
only rotate the approach direction; heights never change).

- RED PIN: 18 mm square section, 140 mm tall, standing with its top ~118 mm above
  the runway floor and >30 mm proud of the tray walls (asserted) — a top-down
  grasp with the parallel jaw (18 mm ≪ 80 mm max jaw) on the exposed upper
  shaft, then a straight vertical pull. The extraction load is the tray's
  downhill press (~0.7 N) times polished-contact friction — well under 1 N over
  a 140 mm stroke; set-down is an unconstrained place on open ground.
- BLUE BOTTLE: 45 mm cylinder (≪ 80 mm jaw), standing in the open-top tray with
  ~28 mm side clearance to the walls and its upper half fully above the wall top
  — side grasp from above, straight 0.2 m lift (2.5 N), transport, and a place
  inside the 72 mm rimmed pad (13 mm xy slack vs the 30 mm tolerance). The pad
  sits on the enclosure roof at z ≈ 0.19 world — an easy downward place.
- The TRAY is never touched in the nominal plan. Its yellow grab bar (60 mm wide,
  protruding 30 mm from the front face) exists so the TRAP state (a knocked-over
  bottle riding into the enclosure) is recoverable by a hand: pull the bar to
  re-open, fish the bottle out. Honesty: the enclosure is roofed but its doorway
  and open runway keep both obstructions reachable from above at all times in the
  nominal flow.

## Ordering

Either obstruction may be cleared first — the rubric imposes no order. The only
ordering in the task is physically inherent: closure cannot precede clearing
(a blocked path cannot yield the seated pose — the pin socket and the header
stall are real arrests, both demonstrated by smoke under real force), and the
final state is judged LIVE (tray closed & pin clear & bottle docked & settled).
The three partial-credit latches (`_out`, `_dock_l`, `_pin_l`) and the max-closure
fraction latch only anchor demonstrated progress; success itself is latch-free.

## Rubric

- 0.20 — bottle ever fully out of the tray box (latched).
- 0.20 — bottle ever settled upright inside the roof pad (latched).
- 0.20 — pin ever fully clear of the runway corridor (latched).
- 0.30 × latched max closure fraction of the initial opening.
- Cap 0.90 without success; 1.0 iff `success()`: tray seated closed in its
  channel (face < 8 mm from the doorway plane, centred, at floor height), pin
  clear, bottle docked upright (xy < 30 mm, on the plate, tilt < 10°), all
  settled and finite. Null policy ~0; the seed's strategy (push the tray) ~0.

## Checks (`smoke.py`, rejection-only, 15 checks)

settle/no-NaN (tray held open at its socket's opening); randomization A (yaw span
>90°, xy jitter); randomization B (socket varies, settled opening tracks it every
reset); null policy ~0; SEED strategy (real 8 N push — pin arrests, ~0, no
success); header interlock (pin removed, standing bottle stalls the tray open);
wrong spot (bottle on the ground, tray closes → docked clause refuses, ≤0.72);
lying on pad (horizontal bottle across the rim → upright clause refuses); pin
left standing (bottle docked, tray still held → refused); pin rides inside
(pin dropped into the tray, tray closes with it → pin-clear refuses); knocked-over
trap (fallen bottle passes UNDER the header and rides in → out/docked refuse);
latched credit survives theft (no success); rejection audit (success never
observed anywhere in the battery); final no-NaN; frames.npz video recorded.
