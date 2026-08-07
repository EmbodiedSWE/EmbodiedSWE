# libero_kitchen_scene1_open_bottom_drawer_i87 — open the untouchable bottom drawer through the cabinet's rocker transmission

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene1_open_bottom_drawer` — a Franka in the
LIBERO kitchen reaches the cabinet, grasps the bottom drawer's handle, and pulls
it open. The judged outcome is the same state — *the cabinet's bottom drawer
standing open* — but the physical route to it is deliberately inverted.

## Strategic difference

**vs. the seed.** In the seed the judged drawer is the manipulated object: you
grab its handle and pull. Here the judged bottom drawer is *untouchable by
design*: its front plate is recessed 4 mm behind the cabinet face, perfectly
smooth (no handle, no lip, no ledge), with only a 2 mm perimeter gap — there is
nothing to hook, pinch, or suction, and it rests on its inner joint stop so it
cannot be nudged inward into reach either. The ONLY way to open it is to push
the *other* drawer — the protruding TOP drawer, which spawns 12.5–13.4 cm open
with a bright red lip — **inward, flush into the cabinet**. Inside the sealed
carcass a gravity-balanced rocker (a vertical plank pivoted mid-height on the
cabinet's side walls, contact pads at both arm tips) converts the top drawer's
in-stroke into an out-stroke of the bottom drawer: the top drawer's rear pusher
plate presses the upper pad, the rocker tips, and the lower pad drives the
bottom drawer's push plate outward ~11+ cm, past the 8 cm success threshold.
So the seed's motion (pull the judged drawer toward you) is replaced by its
opposite (push a different drawer away from you), and the seed's strategy
(handle grasp on the goal object) is replaced by force transmission through a
hidden mechanism — pulling the top drawer, the seed's reflex, actively
*disengages* the transmission and can never score (verified by a smoke probe).

**vs. the rest of the corpus.** `close_drawer_i58` and `close_microwave_i4`
judge a *closing* motion on the pushed object itself — here the pushed object is
NOT judged, and the judged object *opens*. No other task in `tasks_v7/` uses an
antagonistic two-drawer cabinet or a rocker (seesaw) transmission: the
transmission tasks in the corpus are direct pushes, pulls-with-tools
(`pull_cube_tool_i1`), buttons (`push_button_i35`), or carried shuttles (i34);
none route the load through a pivoting intermediate link, and none make the
goal object physically ungraspable. There is no stored energy anywhere: the
rocker's CoM sits at its pivot (gravity-balanced), all joints are plain
prismatic/revolute with hard limits, so the mechanism is purely a motion
converter — the robot supplies every joule.

## Scene (`rocker_cabinet`, env `simgen.rocker_cabinet`, robot="null")

Procedural geometry only. A kinematic cabinet carcass (side walls, back, base,
mid shelf, top) contains three dynamic bodies, each authored in place and bound
to the carcass by per-env spawn-authored joints (collision-filtered only
between joint partners, so transmission contacts stay live):

- **Top drawer** (prismatic, stroke 0→0.145 m): red protruding lip, tray, side
  rails, and a rear pusher plate at x∈(−0.395,−0.38). Spawns at a *seeded*
  opening d0∈[0.125, 0.134].
- **Bottom drawer** (prismatic, stroke 0→0.160 m — the judged body): recessed
  smooth front, box, and a rear push plate at x∈(−0.272,−0.257). Spawns closed.
- **Rocker** (revolute about y at (−0.30, 0.36), limits [−34°, +3°]): plank
  with pads at both tips; root at the pivot and MassAPI CoM at the pivot →
  gravity-balanced, no stored energy, holds any pose hands-off.

Geometry contract (all asserted in `__post_init__`): engagement travel
d_eng = 0.12 m < d0 (5–14 mm free travel before first pad contact); ideal
transmission output ≈ 0.117 m ≥ goal + margin; rocker swing clears the back
wall; pads stay on their plates across the full arc; the recessed front offers
no purchase; distractor spawn bands don't overlap. Distractors: a bowl and a
plate on the cabinet top, seeded xy bands + jitter. Slick physics material
(μ≈0.06, combine "min") on all movers so PhysX's default ~0.5 friction cannot
jam the pad-on-plate transmission.

**Success** (live state, no memory): `bot_open ≥ 0.080 m` ∧ settled (all
bodies' velocities under gates) ∧ finite. **Score**: latched partial credit —
engage 0.15 (top drawer driven ≥3 cm inward from d0), crack 0.25 (bottom
≥3 cm), ajar 0.25 (bottom ≥6 cm), capped at 0.65; exactly 1.0 iff success.
Latches reset on `reset()`.

## Solution (`solve.py`) — applied force only, zero teleports

Teleports are permitted for transport only; this solve needs none.

1. **P0 settle + perception**: 120 steps; read back d0 and distractor poses
   from the live state (never hard-coded); assert baseline score ≈ 0.
2. **P1 engage**: PD force servo (kp 400, kd 40, clamp 35 N — a fingertip
   pressing the red lip, force-limited) drives the top drawer inward along its
   prismatic axis until the pusher plate meets the rocker's upper pad (engage
   latch, score ≥ 0.15).
3. **P2 crack**: same push continues; the rocker transmits and the bottom
   drawer cracks ≥3 cm (score ≥ 0.40). The judged drawer is never forced,
   never teleported, never touched.
4. **P3 flush**: push to top_open < 4 mm; the bottom drawer is driven out
   (ajar latch; momentum legitimately coasts it past the transmission maximum
   to its 0.16 m outer stop, where it disengages from the rocker — irreversibly
   open, since every contact is unilateral). Score ≥ 0.65.
5. **P4 hands-off**: forces off; the friction-held mechanism settles;
   `success()` turns True on the live state (score 1.0), then holds through a
   **3.3 s hands-off persistence** window before `SIM_GEN_SOLVE: SUCCESS`.

The whole episode is then repeated on a second seed (fresh reset, no score
prints — the rubric restarts at 0 and the SIM_GEN_SCORE stream must stay
non-decreasing) to certify seed robustness. Forge result: both seeds OK,
rc=0 (~18 s).

## Franka embodiment argument

Base pose: on the floor ~0.6 m in front of the cabinet face (cabinet face at
x=0, so base at ≈ (+0.60, 0.0, 0)), facing −x.

- **Top drawer (the manipulated object)**: its red lip protrudes 12.5–13.4 cm
  from the cabinet face at height ≈0.58 m, spanning ±0.148 m in y — squarely
  inside the Franka's workspace from that base pose. The action is a single
  straight horizontal push: closed fingertips (or knuckles) on the lip face,
  translating −x by ~13 cm at ≤35 N. No grasp, no re-orientation, no wrist
  singularities; force-limited pushing is the easiest Franka primitive.
- **Bottom drawer (the judged object)**: never contacted by design; nothing to
  argue — the embodiment requirement is precisely that no end-effector can
  acquire it (smooth recessed front, 2 mm gap, resting on its inner stop).
- **Bowl / plate (distractors)**: sit on the cabinet top at 0.70 m — reachable
  but irrelevant; the task requires not touching them.

## Execution order declaration

Scene was designed first with the minimal success predicate; `solve.py` was
then iterated on the forge until the task was *physically* solved on two seeds
(rc=0, SUCCESS on seeds 0 and 1); only after that were the rubric weights
anchored to the observed trajectory and `smoke.py` written. No check was ever
weakened to make a run pass — the two smoke iterations fixed *probe* bugs
(momentum coast after the partial-transmission push; the swung rocker
depenetrating the stolen-shut drawer back open), not the rubric.

## Smoke battery (`smoke.py`) — 14 checks, all rejection/health

1. **settle/no-NaN** — top drawer rests at its sampled d0 (±6 mm), bottom
   sealed, score ≈ 0, no success.
2. **randomization A** — d0 varies across 8 seeded resets (span > 3 mm) and
   the *settled* drawer tracks the sampled value every time (readback).
3. **randomization B** — bowl/plate xy poses vary across resets.
4. **null policy** — 240 idle steps: bottom stays sealed, score ≈ 0.
5. **SEED strategy rejected** — a real 20 N pull (the seed's move) drags the
   top drawer OUT to its stop (probe asserts it moved — non-vacuous): the
   rocker disengages, bottom stays sealed, score ≈ 0.
6. **bottom push probe** — a real 15 N push on the judged drawer's front:
   first proves the force pathway is live (drives a constructed-2-cm-open
   drawer back to its stop), then proves pushing cannot open it. (Pulling it
   open is N/A for an external-force probe: a wrench applied to the drawer
   body would bypass the very no-purchase geometry being claimed; the
   impossibility is geometric — smooth recessed front, 2 mm gap, inner stop —
   and asserted in the cfg contract.)
7. **timid push** — a real push stopping ~2 cm in (asserted moved) stays below
   the engage latch: score ≈ 0.
8. **near miss** — bottom constructed settled at 7 cm (past both partial
   latches, short of the 8 cm goal): capped partial credit ≤ 0.65, no success.
9. **open but moving** — bottom written at 10 cm with 0.3 m/s outward
   velocity: the settle gate refuses success on the moving state.
10. **wrong object** — bowl and plate shoved to the ground: score ≈ 0.
11. **latched credit** — the REAL transmission driven until the bottom cracks
    (engage+crack latch from genuine physics), state frozen, bottom stolen
    shut: latches survive (score = 0.40 band), success does not.
12. **rejection audit** — `success()` observed False at every step of the
    entire battery.
13. **final no-NaN.**
14. **video** — frames.npz (>10 rgb frames) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 14/14` (rc=0, ~66 s).
