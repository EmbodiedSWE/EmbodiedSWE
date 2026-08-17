# bascule_keep — unload the counterweights, let the drawbridge fall, push the ketchup into the keep

Env id: `simgen.bascule_keep` · Scene: `bascule_keep` · Robot: `null` (scripted physics solution)

## Seed provenance

Derived from `libero/libero_pick_ketchup` — *"pick up the ketchup and place it in the
basket"*. What the seed tests: identify the named bottle among grocery clutter, prehensile
pick, carry over an OPEN basket, release; a bounding-box containment check ends the episode.

What is kept from the seed:

- A target **ketchup bottle** that must be identified against a same-shape distractor
  (RED ketchup vs. YELLOW mustard, on randomly swapped ground spawns).
- A **containment goal**: the episode ends with the ketchup resting inside a box-shaped
  receptacle, distractor excluded.

## What changed, and why it is strategically different

The seed's receptacle is open-topped: the whole task is grasp selection plus one carry and
one release. Here the receptacle is a **raised, fully ROOFED keep** (floor at 14.8 cm,
walls, lintel, roof) whose only opening is an 8 cm letterbox doorway high in its front
wall — and the only path to that doorway is a **counterweighted bascule bridge** that
spawns RAISED at +34.9°, parked by two 0.8 kg steel ballast blocks sitting in a walled
tray on its tail. The interlock is asserted in `__post_init__` with pre-computed statics
margins:

- Raised, the tongue-tip-to-lintel aperture is **3.8 cm < 5.5 cm** bottle diameter — no
  entry past the raised machine, at any speed (probed dynamically in smoke with a 2.5 m/s
  slam up the raised channel).
- The doorway is 8 cm wide and the sill is 13.5 cm high with **no exterior landing** —
  a 14 cm bottle cannot be posted through from the ground side, and a carry-and-drop
  lands it on the keep **roof** (the seed's entire plan, constructed and rejected).
- **Either block alone holds the bridge raised** (single-block restoring moment ≥ 1.8×
  the tip-heavy toppling moment; both blocks ≥ 3×, asserted). The leaf's authored CoM
  (`CreateCenterOfMassAttr`, 5 cm toward the tip) makes the **empty** leaf fall level on
  its own and rest on the doorway sill porch.

So the actuation is **subtractive**: the hand never presses, holds, or steers the bridge.

1. **Unload** — pick each counterweight block (5 × 5 × 3 cm, jaw-sized) out of the open
   tray and set it on the ground. After the first block: nothing moves (asserted and
   probed — one block still holds). After the second: **gravity deploys the bridge**,
   the leaf swinging on its pure trunnion hinge until the tongue rests on the sill.
2. **Cross** — set the ketchup on the deck tail and push it along the walled channel,
   across the tongue, through the doorway (transit steps DOWN 2 mm onto the interior
   floor), into the keep.
3. Done — the bridge *stays* deployed (it is now its stable state), sealing nothing but
   proving the machine was operated: success requires the tray EMPTY and the leaf level
   and intact, so kinematically holding the ballasted bridge down scores **zero** (deploy
   credit is gated on tray-empty; constructed and rejected in smoke).

Strategy vs. the corpus tasks read for this construction: sibling `rocker_lock` (i104,
same seed) is **additive** press-and-hold actuation — push a paddle, hold ~25 N while
gravity ferries the bottle, release to re-seal; here there is **no hold and no re-seal**
— the machine is operated by *removing mass*, the state change is one-way-stable, and the
cargo is moved by plain pushing after the machine has finished moving. `plank` moat tasks
(i192/i19) build a bridge by *adding* a loose spanning part; here the bridge is a jointed
machine already present, and the moves are removals. `drawbridge_vault` (i332) winches
its bridge with a held tool. The failure the rubric centers on — pressing the bridge down
by force instead of unloading it — has no analogue in the seed at all.

## Scene (procedural geometry only)

- **Rig** (static): ground heel rest, two block depots, keep — front wall with an 8 cm ×
  17.8 cm doorway (sill top 13.5 cm, lintel bottom 31.3 cm), pedestal, raised interior
  floor (top 14.8 cm), side walls, back wall, roof (underside 32.5 cm).
- **Pillar** (dynamic, 40 kg): trunnion stands flanking the leaf at hinge height 14.3 cm.
- **Leaf** (1.0 kg, yellow): 42.5 cm main deck slab + 7 cm-wide tongue (fits the doorway),
  side curbs forming a 12 cm push channel, walled ballast tray on the tail, riding a
  cross-body **RevoluteJoint** (axis y, ±45° limits) authored to the pillar. Authored
  tip-heavy CoM; two **steel blocks** (0.8 kg each) in the tray park it at +34.9°.
- **Ketchup** (red) / **mustard** (yellow): identical 5.5 × 14 cm cylinders.
- Per-seed randomization (verified by state readback in smoke): whole machine coherent
  xy jitter ±3 cm + yaw ±12° (blocks re-seated in the tray in leaf frame), the two
  bottles **slot-swapped** between ground spawns (`torch.rand` comparison, not the
  degenerate first-randint draw) plus ±5 cm xy jitter and free yaw.

## Teleport solution (solve.py) — teleport is transport only

- **P0** settle + layout readback; assert bridge raised at ≈ +34.9°, both blocks in the
  tray, bottles on the ground, score ≈ 0.
- **P1** teleport-carry block A from the tray to its ground depot (open-sky pick from an
  open tray, zero velocity — a plain pick-and-carry); settle; **assert the bridge is
  still raised** (single-block hold is real, measured +34.88°). Score 0.10.
- **P2** teleport-carry block B to the other depot, then **hands off**: gravity swings
  the leaf down until the tongue rests on the sill (measured −0.26° level rest). Assert
  deployed + hinge intact. Score 0.45.
- **P3** teleport-carry the ketchup from its ground spawn onto the open deck tail
  (standing in the channel, zero velocity — a plain pick-and-place).
- **P4** push: an emulated low pushing hand — CoM force from a rig-frame velocity servo
  (0.08 m/s target, capped **1.5 N** < the ~2 N tip limit) plus the push-height torque
  τ = r×F with r = 5 cm below center, world-encoded through `encode_force` with a runtime
  force-frame probe (mode 1, snapshot rollback to mode 0) and a y-centering term. The
  bottle slides the channel, crosses the tongue, steps down through the doorway
  (up-vector stays +1.000, no tipping). Score 1.0 on arrival.
- **P5** hands off, settle; assert in-keep + bridge deployed + tray empty; persistence
  400 steps (3.3 s) with success held, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds 0, 1, 2 — score trace 0.00 → 0.10 → 0.45 → 1.00, monotone,
success persists ≥ 3.3 s hands-off.

## Rubric (anchored in the demonstrated solve)

Latched milestones (updated only in `score()`, non-decreasing, serialized in
`get_state`/`set_state`): first block out of the tray (0.10) → tray empty (+0.10) →
bridge deployed **gated on tray empty** (+0.25) → ketchup past the doorway plane, gated
on deployed (+0.25) → 1.0 iff `success()`. `success()` requires ALL of: ketchup resting
inside the keep interior and settled; bridge DEPLOYED (leaf level within tolerance,
settled, hinge intact — axle still on its trunnion); ballast tray EMPTY; mustard outside
the keep; all states finite. Pressing the ballasted bridge down and slipping the bottle
in scores **0.000** — no milestone fires (constructed with a 150-step kinematic hold in
smoke and rejected).

## Embodiment argument (Franka, 80 mm parallel jaw)

- The counterweight blocks are 5 × 5 × 3 cm — a comfortable top pinch within the 80 mm
  jaw (asserted `BLOCK < 0.08`); the tray is open-topped with 4.5 cm walls, tail end of
  the raised leaf at ~28 cm height, reachable from above.
- The bottles are 5.5 cm across (asserted < 8 cm) — a side or top pinch; the deck tail
  sits low and open when deployed; the push is a quasi-static ~1.5 N slide at ≤ 22 cm
  height, single-arm, no hold anywhere (blocks are set down and abandoned).
- Tray, depots, bottle spawns and channel span ~90 cm of workspace at ≤ 30 cm height;
  a base posted beside the tail reaches the tray, the depots and the whole channel
  without exceeding ~85 cm reach.

## Execution order is physically forced

The interlock is asserted in `__post_init__`: raised, every entry aperture undercuts the
5.5 cm bottle (3.8 cm tongue-to-lintel; sill higher than the bottle diameter with no
exterior landing; doorway narrower than the lying bottle length) while the deployed
bridge passes it with ≥ 3 cm channel margin — so entry REQUIRES deployment, and
deployment REQUIRES removing both blocks (each alone holds ≥ 1.8× margin). Success
further requires the tray empty and the machine intact and level, so press-and-hold with
ballast aboard scores zero, wrecking the bridge off its trunnion fails the intact clause,
and the seed's carry-and-drop strands the bottle on the roof. Each of these is
constructed and asserted rejected in smoke.py; the raised seal is probed dynamically
(a 2.5 m/s slam up the raised channel reaches the tip zone and never passes).

## Checks (smoke.py, 23)

1 settle/no-NaN + layout · 2 baseline score ≈ 0 · 3–5 randomization real via readback
(machine pose spread; slot swap both ways and always opposite; per-bottle jitter) ·
6 null policy 300 steps ≈ 0 · 7 raised interlock real (ketchup laid along the raised
channel, slammed 2.5 m/s up the slope, reaches the tip zone, never crosses the doorway
plane) · 8 one-block hold (block A removed → bridge stays ≥ +25°) · 9 self-deploy
(block B removed → leaf falls level, 0.45 latched, no success) · 10 seed strategy
(carry + drop over the receptacle) lands on the roof, rejected · 11 wrong object
(mustard inside by fiat) rejected · 12 ballast-hold cheat (leaf kinematically pressed
level 150 steps with blocks aboard, ketchup slipped in) → not success AND score 0.000 ·
13 released ballasted leaf re-raises (+34.5°) · 14 acceptance construct (unload → self-
deploy → cargo in) accepted, 1.0 · 15/16 mustard added inside flips success off /
set_state restore flips back on · 17/18 bridge dragged off its trunnion flips off /
restore flips on · 19 settle gate (delivered bottle kicked, judged moving → rejected;
resettled → accepted) · 20 near miss: on the deck only (0.45, no success) · 21 near
miss: standing against the keep's outer wall · 22 rejection audit (success never True
except the accept states) · 23 final finite-state audit. Frames recorded to `frames.npz`.
