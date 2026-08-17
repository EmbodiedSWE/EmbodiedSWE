# bowl_airlock — feed the black bowl into the sealed display case through its gate airlock

**Task id:** `libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297`
**Env:** `simgen.bowl_airlock` (scene-level, `robot="null"`)
**Files:** `scene.py` (scene + rubric), `solve.py` (teleport-transport + force solution),
`smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet.py`).
The seed is a single pick-and-place: grasp the akita black bowl off the table and set it
inside a bbox above a wooden cabinet; the white plate is a distractor; success is a
static pose test on the bowl alone.

Kept from the seed: the black bowl as the cargo, the white plate as a distractor, a
wooden-cabinet setting, and the goal predicate "bowl at rest in a designated region on
top of the cabinet."

## What the task is

The cabinet's top surface is now a **sealed display case**: a deck (22 cm high) enclosed
by wooden walls, a back wall, side walls, and a roof — at no time does any opening
connect the case to free space. The **only** way in is the cabinet's built-in
**airlock**:

- a **loading sill** (30 cm high) in front of a **front window** W1 (14 cm × 10 cm);
- behind W1, a **slick 20°-sloped chamber floor** (directly bound low-friction material;
  pairwise μ with the bowl ≈ 0.19 « tan 20° = 0.36) with centering curbs, descending to
  an **inner window** W2 that opens onto the case deck;
- one rigid **gate carriage** (free dynamic body, NO joints — guided purely by fixture
  geometry: roof slot, guide rail, anti-lift lip, end posts) rides on the roof and
  hangs a **front panel** over W1 and an **inner panel** through the roof slot over W2,
  with a red handle post on top. The two panel cutouts are offset by the full 20 cm
  travel, so the windows are **mutually exclusive by construction**: at the LOAD end W1
  is open and W2 walled; at DISPENSE, W2 is open and W1 walled; at any intermediate
  position the two part-open slits sum to 2w − L = 8 cm on **opposite lateral sides** —
  the 9.8 cm bowl can never pass either while the other is open at all. The carriage
  randomizes to **either end** at reset.

The goal:

> "Put the black bowl into the sealed display case on top of the cabinet using the
> airlock: stage it on the sill, push it through the front window with the gate at
> LOAD, then slide the gate to DISPENSE so the bowl glides through the inner window
> and settles upright on the case deck."

`success()` = bowl upright (≤15°) at rest in the placed band on the case deck
(fixture-frame test) ∧ everything settled ∧ the delivered latch fired.

## Strategic difference — vs the seed and vs every corpus neighbor

- **vs the seed:** the seed is one grasp-and-place through free space onto an open top.
  Here the destination is **sealed at every instant** — the seed's plan is
  geometrically impossible (smoke check 10 drops the bowl from above and it lands on
  the roof). The bowl must be routed *through the fixture's interior* by
  reconfiguring the fixture itself, in a physically forced order: open W1 → insert →
  gravity feed → swap the gate → gravity dispense. The final approach to the goal is
  performed by **gravity through a one-way slope**, not by the agent carrying the
  object.
- **vs i61 cart_ferry (same seed):** there the destination rides a vehicle that is
  positioned and then driven home with the cargo aboard; every transport is agent-
  driven. Here nothing carries the bowl: the moving body (the gate) never bears or
  transports the cargo — it only *reconfigures the topology* (which window exists),
  and the cargo moves solely by being pushed on a surface or by gravity.
- **vs i306 hatch_shelf / i5 counterweight_shelf:** those gate access to an open shelf
  and the object is still *placed by transport* through the opened aperture. Here no
  aperture to the goal ever opens to the outside — the interlock guarantees the agent
  can never reach or see the deck through an open path; the last leg is autonomous.
- **vs i3 tunnel_shuttle / i14 chute_switch / i48 cask_weight_sort:** no shuttle to
  push through, no routing choice among outcomes; the difficulty is a *mutual-
  exclusion mechanism* (an airlock) that forces a strict open-insert-close-open cycle.
- **vs i57 hanoi_rings / i27 bell-herd / robobench (balance_scale, combination_safe,
  syringe, pen_holder):** no stacking order, no multi-agent herding, no articulated
  dial/insertion mechanism — the interlock is emergent from one rigid compound sliding
  between two walls.

No task in the read corpus has a goal region that is *never* directly reachable, nor a
two-panel mutual-exclusion gate whose cycling is the load-bearing mechanism.

## Rubric (latched credit, `score()`)

| latch | condition (fixture frame, post_step every substep) | weight |
|---|---|---|
| `_staged_ever` | bowl upright + slow on the loading sill band | 0.15 |
| `_chambered_ever` | bowl inside the transfer-chamber band | 0.25 |
| `_delivered_ever` | bowl inside the case volume (any pose) | 0.30 |
| success | placed band ∧ upright ∧ settled ∧ `_delivered_ever` | = 1.0 |

`score = Σ weights` clamped at 0.95 unless `success()`, which returns exactly 1.0.
Latches are evaluated every physics substep with a 2-substep reset grace re-pin, so the
printed `SIM_GEN_SCORE` sequence is non-decreasing by construction. Null policy ≈ 0.

## Solution (`solve.py`) — teleport = transport only

1. **GATE→LOAD** — pulsed 8 N push on the carriage along the fixture −y (force on only
   below 0.10 m/s) until it seats at the LOAD end post (no-op when it randomized
   there — both branches were exercised: seeds 0/1 start at DISPENSE, seed 2 at LOAD).
2. **STAGE** — the single teleport: bowl to 2 cm above the sill center (pure transport;
   a robot does this leg by rim-pinch grasp + carry), drop, settle. → 0.150
3. **LOAD** — pulsed 2.2 N push on the bowl along +x (cap 0.08 m/s): across the grippy
   sill, through the open W1, onto the slick slope; released past the wall plane; the
   slope gravity-feeds it down against the closed inner panel. → 0.400
4. **DISPENSE** — pulsed 8 N push on the carriage along +y to the DISPENSE end. **No
   force ever touches the bowl again**: W2 opens, the bowl glides through, drops onto
   the deck, and settles upright. → 1.000
5. **PERSISTENCE** — ≥ 3.5 simulated seconds fully hands-off; `success()` must still
   hold before `SIM_GEN_SOLVE: SUCCESS` is printed.

`drive()` probes the pod's wrench-frame encoding at runtime (raw-world vs body-frame)
from measured progress, as in the sibling tasks.

Verified on the forge: seeds 0, 1, 2 all print 0.000 → 0.150 → 0.400 → 1.000 →
`SIM_GEN_SOLVE: SUCCESS`, rc = 0 (final package re-confirmed after the last edit).

## Embodiment argument (single Franka, 8 cm parallel jaw, OSC)

Every stage is a standard manipulation primitive, and no stage admits a shortcut:

- **Stage** = pick-and-place onto the sill: the bowl's outer diameter (9.8 cm) exceeds
  the jaw span, but it is an open cup with a 0.9 cm wall — a **rim pinch** (one finger
  inside, one outside, anywhere on the rim) is a textbook grasp; the sill top is at
  0.30 m in front of the fixture, fully exposed.
- **Gate slides** = power-grasp the red handle post (r 1.2 cm, top at ~0.48 m,
  vertical — graspable from any yaw) or flat-palm the bridge edge; 20 cm lateral
  stroke terminated by end posts, so no precise servoing is needed (≈3 N of friction).
- **Load push** = planar drag/push of the staged bowl 10 cm along the sill through W1
  (a fingertip behind the far wall of the bowl, ≈2 N); W1 is 14 cm wide × 10 cm tall,
  and the panel at LOAD is out of the hand's way (the open cutout is exactly there).
- **No stage requires reaching inside** the chamber or the case — which is the point:
  after W1 seals, gravity finishes the job. A hand *cannot* shortcut the interlock
  (smoke checks 10-12: over-the-roof placement and both part-open configurations are
  physically rejected).
- **One plausible base pose:** base at fixture ≈ (−0.45 m, −0.55 m), facing the sill.
  From there: pedestal (x −0.55, reach ≈ 0.45 m), sill and W1 (≈ 0.5 m), and the
  handle at both travel ends (x −0.12, y ±0.10, z 0.48; ≈ 0.75 m) are all inside a
  0.855 m Franka reach envelope, with the fixture body never between base and targets.

## Execution order (declared)

Strictly sequential; each stage physically enables the next (not just in the rubric):

1. Gate to LOAD (else the front window is sealed — smoke check 11 proves the stage
   push stalls on the panel).
2. Stage + push the bowl through W1 (only possible while W1 is open; the chamber
   cannot be entered any other way — check 10).
3. Gate to DISPENSE (only now does W2 exist; mid-travel both slits are sub-bowl —
   check 12), which simultaneously seals W1 behind the bowl.
4. Gravity dispenses; hands-off persistence.

## Checks (`smoke.py`) — 15

1. reset settles finite (bowl on pedestal, carriage at an end); 2. score ~0, no
latches at reset; 3. randomization READBACK across 8 seeds (fixture xy/yaw, bowl
xy/yaw spreads; carriage start takes BOTH ends); 4. null policy ≈ 0; 5-7. oracle
passes on seeds 0/1/2 via the full force pipeline (score 1.0, persists 240 steps);
8. monotone ladder 0 < 0.15 < 0.40 < 0.70 (tipped in case: delivered, NOT placed, no
success) < 1.0; 9. partials < 1.0; 10. seed-analog negative: bowl dropped from above
lands on the roof, zero credit; 11. interlock front: with the gate at DISPENSE the
oracle's own push moves the bowl ≥3 cm then stalls it at the front panel, chamber
latch never fires (non-vacuous actuator check); 12. interlock mid-travel: sustained
push cannot dispense a chambered bowl through the part-open slits; 13. wrong object:
the plate settled on the deck earns nothing; 14. wrong place: chambered with the gate
never cycled = 0.25, no success; 15. placed-band calibration drop table
(upright-in ×2, tipped-out, roof-out). Frames recorded to `frames.npz` (500 × 960×600).

Forge result: `SIM_GEN_SMOKE: ALL PASS 15/15`, rc = 0.
