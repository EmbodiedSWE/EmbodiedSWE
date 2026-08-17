# libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i420 — stow the bowl in a sprung drawer held shut by a gravity drop-bolt

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet`
— a Franka opens the cabinet's bottom drawer, picks up the black bowl, puts it
inside. The judged outcome here is deliberately *stricter than* the seed's: the
bowl must end up **sealed inside the SHUT drawer**, and the drawer actively
fights being shut.

## Strategic difference

**vs. the seed.** The seed's drawer is a passive damped slider: open it, drop
the bowl, done — closing is optional and free. Here the drawer is **spring-
loaded OPEN** (a live linear drive, 6.6 N at shut, still 4.2 N at the out-stop)
and is held shut only by a **gravity drop-bolt latch**: a heavy (0.5 kg)
vertical bolt riding a prismatic-Z joint whose dangling tip blocks the drawer's
tall front panel with 12 mm of engagement. The seed's reflex — grab the front
and pull — is physically refused: a sustained 25 N pull cannot open the latched
drawer (smoke-probed). The only route is a three-step, order-forced cycle:
**lift the bolt's T-handle** (the spring then throws the drawer open on its
own), **load the bowl**, and **push the drawer shut against the live spring**
until the bolt's 45° tip wedge cams it up and over the panel and it re-seats
behind it. Success is judged on the re-sealed state, so the seed's "open and
drop" trajectory earns at most capped partial credit (0.45) — the hard part of
this task is everything the seed never asks for.

**vs. the rest of the corpus.** The sibling drawer/cabinet tasks route force
through a rocker transmission (`open_bottom_drawer_i87`), pour beads through an
open drawer (i134), or post a bowl through a letterbox slot (i212). None of
them combine **stored energy** (a spring that actively re-opens the goal
container) with a **releasable gravity latch** and a **self-re-engaging cam**;
none make *keeping the container shut* the judged difficulty. The latch is not
scripted state — release, throw, cam-over, and re-seat are all plain PhysX
contacts and joints.

## Scene (`boltlatch_stow`, env `simgen.boltlatch_stow`, robot="null")

Procedural geometry only; cabinet frame: x=0 is the front face, +x the opening
direction, z=0 the ground.

- **Cabinet** (kinematic compound): plinth, side walls, back, roof (seals the
  bay from above, underside 0.33), front apron, and an overhead **guide
  collar** (y-fingers ±(14–34) mm and front/rear plates at z 0.402–0.442,
  1.5–3 mm x-clearance) that takes latch shear on the bolt shaft
  geometrically.
- **Drawer** (0.5 kg, prismatic-X, limits = hard stops, stroke 0.20 m):
  open box (cavity floor 0.150, walls to 0.240) + tall front panel to
  z 0.325 whose top 12 mm is the latch engagement band. A **DriveAPI linear
  spring** (k 12 N/m, target 0.55 m) pushes it OPEN across the whole stroke.
- **Bolt** (0.5 kg, prismatic-Z, travel 0.05 m): shaft (x 0.028–0.046, tip
  z 0.313 → 12 mm engagement behind the panel), a 45°-rotated **tip wedge**
  (reach x 0.071) whose lower-outer face is the re-latch cam, and a T-handle
  crossbar at z 0.505. Slick material (μ 0.06/0.05, combine "min") on the
  latch faces. The bolt is deliberately heavy: a 0.06 kg bolt let the GPU
  contact solver *creep* the panel through it at only ~9 N quasi-static (the
  chain static→contact→light-body→joint is ill-conditioned); at 0.5 kg the
  latch holds a 30 N ramp with 0.08 mm total creep.
- **Bench + items**: bench top 0.12 beyond the drawer's sweep; the black
  **bowl** (r 0.035, h 0.045, 0.11 kg) and a distractor green **bottle**
  (r 0.030, h 0.19, 0.30 kg — cannot stand in the cavity) spawn at seeded
  x∈(0.34, 0.42), |y|∈(0.06, 0.20), bowl on a sampled side, bottle opposite.

Geometry contract asserted in `__post_init__`: engagement exactly 12 mm and
bolt travel clears it; latched rest gap (8 mm) strictly inside `shut_tol`
(14 mm); shut cavity is sealed (front slit ≤ 8 mm, roofed); walls and the
stowed bowl pass under the dangling bolt; collar clearances and the lifted
wedge's sweep; cavity fits the bowl, drop zone clears wedge and open panel;
spring 4–10 N at shut and ≥ 3 N at the stop; bench beyond the sweep; rubric
weights sum to the 0.60 cap.

**Success** (live state, no memory): bowl inside the cavity ∧ opening ≤ 14 mm
∧ all movers settled ∧ finite. Only the re-seated bolt can hold the shut band
against the live spring, so persistence is load-bearing. **Score**: latched
partial credit — release 0.15 (bolt lift ≥ 14 mm), open 0.20 (opening
≥ 0.12), load 0.25 (bowl in cavity while opening ≥ 0.10), capped at 0.60;
exactly 1.0 iff success. Latches reset on `reset()`.

## Solution (`solve.py`) — teleport for transport only

1. **P0 settle + perception**: 120 steps; read back the latched rest
   (0.0081 m), seated bolt, and both items' bench poses; score ≈ 0.
2. **P1 release**: force-limited T-handle lift servo on the bolt (ff 4.9 N ≈
   its weight, clamp 9 N) to 35 mm; the spring throws the drawer to its
   out-stop by itself; bolt released → it falls and dangles. Score ≥ 0.35.
3. **P2 load** (the only teleport): the bowl is teleported to a hover over
   the exposed drop zone (x 0.125, z 0.26) and **dropped** — gravity and
   contact seat it in the cavity. Score ≥ 0.60.
4. **P3 re-latch**: velocity-limited palm push on the panel (v_des 0.18 m/s,
   clamp 14 N) against the live spring; the panel edge rides the 45° wedge,
   cams the 0.5 kg bolt up ~10 mm, passes under, and the bolt free-falls and
   re-seats behind the panel; a brief −8 N seat hold, then forces off.
5. **P4 hands-off**: success on the live state (score 1.0), then a **3.3 s
   hands-off persistence** window — the spring is still pressing the drawer
   into the bolt the whole time — before `SIM_GEN_SOLVE: SUCCESS`.

The full episode runs on two seeds (second with no score prints; the
SIM_GEN_SCORE stream stays non-decreasing). Forge result: both seeds OK,
rc=0 (~25 s).

## Franka embodiment argument

Base pose: on the open +y side, ≈ (0.35, 0.38, 0), facing the cabinet/bench
corner — bench items at x 0.30–0.46, the T-handle at (≈0.04, 0, 0.51), and the
panel's full sweep (x 0.02–0.22) all inside a 0.85 m reach envelope.

- **Bolt T-handle**: a 72 mm-wide crossbar at 0.51 m height — hook two
  fingers under it (no closing force needed) and lift ≤ 9 N (bolt weighs
  4.9 N); Franka payload ≈ 29 N. Release is just letting go.
- **Drawer panel**: closing is a straight −x palm push at ≤ 14 N against the
  ≤ 7 N spring, at 0.2–0.3 m height. Force-limited pushing is the easiest
  Franka primitive; the cam does the latching.
- **Black bowl**: a 70 mm-diameter, 0.11 kg cylinder — inside the Franka
  parallel-jaw's ~80 mm max opening for a rim pinch; the drop zone accepts a
  release from 0.26 m (gravity seats it), so no precision insertion.
- **Bottle (distractor)**: same bench, opposite side; graspable but wrong —
  it cannot stand in the 90 mm-deep cavity and stowing it earns nothing.

## Execution order declaration

Scene first (minimal success predicate + cfg contract), then `solve.py`
iterated on the forge until the task was physically solved on two seeds, then
the rubric anchored to the observed trajectory, then `smoke.py`. No check was
ever weakened: when the 25 N seed-pull probe *defeated the latch* (13/14), the
fix was in the **scene** — a quasi-static ramp diagnostic showed the 0.06 kg
bolt creeping through GPU contact at 8.7 N, so the bolt became honest 0.5 kg
hardware — and the probe stayed at 25 N (it now holds with a 0.08 mm budge).

## Smoke battery (`smoke.py`) — 14 checks, all rejection/health

1. **settle/no-NaN** — drawer at its bolt-latched rest (8.1 mm), bolt seated,
   score ≈ 0, no success.
2. **randomization A** — bowl bench xy varies across seeded resets
   (std 0.076 m, readback from live state).
3. **randomization B** — bowl and bottle always on opposite sides; the bowl's
   side flips across resets.
4. **null policy** — 240 idle steps: still latched shut, score ≈ 0.
5. **SEED strategy rejected** — a real, sustained 25 N PD pull on the drawer
   (the spring helping) cannot open the latched drawer: peak opening a few
   mm, bolt stays seated, score ≈ 0.
6. **seed end state** — the seed's terminal state (bowl settled inside the
   OPEN drawer) constructed and held 240 settled steps: capped partial credit
   0.45, never success.
7. **spring reopen** — drawer constructed near-shut (~9 cm) but unlatched
   with the bowl inside: the spring throws it back out; the shut band is
   never entered, no success.
8. **wrong object** — the full REAL cycle with the BOTTLE (bolt lifted,
   spring opened, bottle laid in on its side, drawer pushed shut through the
   real cam, bolt re-seats): latch holds hands-off, success never fires,
   score stays at the release+open credit (0.35).
9. **give-up push** — a real close that stops ~3 cm short of the cam and
   lets go: the spring reopens the drawer to its stop; shut never entered.
10. **settle gate** — bowl written inside the shut cavity WITH 0.4 m/s
    velocity and judged without stepping: the moving state is refused.
11. **latched credit** — real release earned 0.35 (latches survive), then the
    drawer stolen shut while empty: a shut empty drawer is not success.
12. **rejection audit** — `success()` observed False at every step of the
    battery.
13. **final no-NaN.**
14. **video** — frames.npz (362 rgb frames) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 14/14` (rc=0, ~79 s).
