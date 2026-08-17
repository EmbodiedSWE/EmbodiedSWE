# put_toilet_roll_on_stand_i304 — load two rolls into a roofed gravity-feed magazine

Scene `roll_magazine`, env `simgen.roll_magazine` (robot="null").

## Seed provenance

Derived from **rlbench/put_toilet_roll_on_stand**: a toilet roll and a wall-mounted
stand with a fixed horizontal peg (mesh visuals, franka, proximity checker); the
implied strategy is transport + axial slide — carry the roll to the stand, align its
hollow core with the fixed cantilever peg, and slide it on. The goal pose is directly
reachable by the gripper, and the peg does all the holding.

## Strategic difference

There is NO peg and the roll's core plays no role. The "stand" is a gravity-feed
MAGAZINE — a level entry APRON, then a covered RAMP descending at 18 degrees under an
80 mm roof to a red STOP WALL — and the goal region (the storage BAY, the last
stretch of ramp before the wall) is DEEP INSIDE a sealed tunnel, under a roof whose
headroom above a parked roll is less than a roll radius. The mouth under the roof
edge is the only opening, and it passes a roll only while it is ON the surface.
Loading each of TWO rolls is a two-stage rolling delivery the seed never needs:

- **Stage**: set the roll down on the open apron with its axis ACROSS the channel
  (only an axis-across roll can roll down the chute; an axis-along roll slides and
  jams by friction — kinetic mu 0.5 > tan 18 deg).
- **Feed**: push it along the apron and under the roof edge; once its center crosses
  the crest, gravity takes over — the roll self-delivers down the covered ramp, out
  of reach, impacts the stop wall (or the queued first roll) and parks. The solver
  never touches the object again after the crest; the goal state forms by rolling
  contact dynamics alone, twice, with the second roll queuing against the first.
- The rubric judges settled poses in the magazine frame: resting ON the ramp surface
  (a z-band above the local surface — the anti-seed clause: a roll set down ON the
  magazine perches on the roof ~130 mm above the surface and reads nothing), past
  the bay line, axis across the channel, sustained stillness. The seed's outcome
  ("roll placed on the stand") scores ~0 here.

A solver needs a different plan (stage + irreversible push handoff to gravity, twice,
with queuing — not carry-to-goal-pose) and different code (channel-frame surface-band
containment, cross-axis alignment clause, a crest gate, per-roll staged/loaded
latches — no threading or insertion geometry anywhere).

## Solution outline (solve.py — teleports for transport only)

- **P0** settle + layout readback; assert score ~0 (rubric-leak guard).
- **P1 STAGE A (transport + gravity)**: teleport roll A to a hover 20 mm above the
  open apron, axis across the channel, then release — gravity sets it down. Score 0.10.
- **P2 FEED A (contact dynamics)**: regulated CoM push (velocity-capped force along
  the channel, K·dt/m = 0.25 inside the wrench-delay bound, + weak lateral centering
  PD), cut the instant the center crosses the crest; the roll self-delivers down the
  covered ramp and parks against the stop wall at local x = +0.200 (= X_WALL −
  OUT_CORNER, the wall-rest anchor). Score 0.30.
- **P3 STAGE B**: same staging for roll B. Score 0.40.
- **P4 FEED B + hands-off settle**: same push + cut; roll B impacts the parked A and
  queues at x = +0.133 (= wall rest − 2·OUT_FLAT·cos θ). All wrenches are already
  cut; wait for the full success gate + 2 s extra margin. Score 1.0.
- **P5 persistence**: 400 more substeps (3.3 s) hands-off with success() checked
  every substep, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (0 → 0.10 → 0.30 →
0.40 → 1.0 → 1.0). Passes on seeds 0, 1, 2 on the forge (queue positions
reproducible to ±1 mm across seeds).

## Rubric

`success()` = both rolls (`parked` ∧ `aligned`) ∧ `settled`:

- **inside**: center past the crest by 20 mm, |y| < 45 mm (walls at 52 mm), and z
  within (23, 50) mm ABOVE THE LOCAL RAMP SURFACE — a true rest reads ~37 mm; a roll
  on the roof reads ~130 mm, a roll stacked on another ~105 mm, both rejected.
- **parked**: inside AND past the bay line x > 90 mm (wall at 235 mm; the two-roll
  queue rests at 200/133 mm with >40 mm margin).
- **aligned**: tube axis within 30 degrees of the channel cross-axis (a jammed
  axis-along roll reads ~90 degrees off).
- **settled**: per-substep pose deltas of BOTH rolls below 0.25 mm SUSTAINED for 30
  consecutive substeps (counter in post_step — immune to GPU rolling-contact phantom
  velocity, and any teleport resets it by construction).

`score()`: latched per roll — 0.10 ever staged on the apron + 0.20 ever inside on
the ramp surface (cap 0.60); 1.0 iff success(). Null policy ~0 (asserted: the rolls
spawn on the open floor; the staged band sits 128 mm above floor rest height, so no
floor pose can latch anything).

## Embodiment argument (Franka)

Both manipulated objects are 70 mm diameter x 90 mm rolls (0.10 kg) — a Franka power
grasp across the tube (70 < 80 mm gripper stroke) or across the width (90 mm face
pinch is not needed). From a base at the origin facing the magazine (nominal
footprint center (0.42, 0.00) ± 50 mm jitter, ±50 degrees heading; roll spawns at
garage-local (−0.34, ±0.16) → world radius 0.15–0.35 m; the apron center sits at
world radius ~0.25 m, apron top at 140 mm height — all well inside Franka's
envelope), the arm: picks a roll from the open floor, sets it down on the open-top
apron lying across the channel (the 30-degree alignment tolerance and the low
40 mm side rails forgive placement error), then pushes it 10 cm along the level apron
— a planar push at roll mid-height, reaction taken by the apron, force well under
1 N — until it passes under the roof edge and the crest, where gravity takes over
and the arm retracts. Nothing is ever manipulated inside the tunnel: the delivery
and queuing are autonomous. The hard parts are perception (magazine heading, channel
axis) and the commitment structure of the push, not dexterity, payload, or reach.

## Execution order

1. `scene.py` written first (geometry + rubric).
2. `solve.py` iterated on the forge: run 1 exposed that a 24-facet roll STALLS
   mid-tunnel on a 13-degree ramp (facet-impact loss ≈ 1.7 m/s² of effective drag —
   the gravity feed was not self-sustaining); fixed in the SCENE (48 facets +
   18-degree ramp + apron raised so the ramp foot clears the floor + tunnel walls
   raised past the roof line), not by pushing harder or weakening the rubric.
   SUCCESS on seeds 0, 1, 2.
3. Rubric anchored against the demonstrated deliveries (wall rest 200 mm, queue
   133 mm, dz 37 mm mid-band across seeds).
4. `smoke.py` rejection battery: **ALL PASS 15/15** on the forge first run,
   frames.npz saved (172 frames).

## Smoke checks (15)

1. settle/no-NaN + layout sanity; 2. score ~0 at reset; 3. magazine pose
randomization readback (x/y/yaw spreads over 6 seeds); 4. roll spawn + yaw readback;
5. null policy ~0 after 240 idle steps; 6. seed-strategy analog — the roll set down
ON the magazine rolls down the sloped roof and perches against the stop wall
directly over the bay (plan-view x = 200 mm > bay line) at 130 mm above the surface
— never inside, score ~0; 7. a roll staged on the apron and abandoned latches only
0.10 (staged is not loaded); 8. crooked feed — an axis-along roll pushed with the
solve's own servo (moved 134 mm, non-vacuous) jams on the upper ramp at x = −56 mm:
inside but never parked, aligned() False; 9. a roll stacked ON a staged roll reads
105 mm above the apron — outside the band, not staged; 10. no back door — a roll
driven at the stop wall from outside (moved 60 mm) is blocked at x = 282 mm, never
inside; 11. one roll alone parked+aligned in the bay is not success (score 0.20);
12. settle gate — the exact goal geometry with an injected 0.31 m/s uphill launch
on roll B is not success while moving; 13. latched credit survives teleport-away
(score keeps 0.40, parked drops, no success); 14. rejection audit (success never
True in the battery); 15. final no-NaN.
