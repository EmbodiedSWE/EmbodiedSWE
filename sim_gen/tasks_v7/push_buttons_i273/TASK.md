# push_buttons_i273 — press each button, then trap it under the sliding shutter

## Seed provenance

Seed task: `rlbench/push_buttons` — press 3 colored buttons in a prescribed order. The
seed is a pure sequencing task: each press is instantaneous and self-latching, and the
only difficulty is remembering the order.

## The task

A kinematic console carries three spring-return plungers (red, green, blue heads, 60 mm
pitch) and one captive sliding shutter plate (yellow tab) that rides a low corridor above
the deck. The corridor's roof lips leave only a 2 mm vertical clearance, so the plate can
slide along the console x axis but can never be lifted out; end stops keep it captive.
The plate spawns at a *randomized* corridor end (either side), and the console pose
(xy ± 10 cm, free yaw) is randomized every reset.

Pressed buttons do **not** latch: a plunger pressed to its 24 mm hard stop creeps back up
at ~4 mm/s the moment it is released (bounded-force creeping-command spring plant, so the
return is robust, not a spring-constant explosion). The only way to keep a button down is
to press it and then **chase it with the shutter plate** before it rises: with the head
below deck level the plate can slide over the well; when the head rises ~12 mm it meets
the plate's underside and stays trapped under it. Success = all three heads trapped at
depth with the plate covering all three wells (5 mm judged margin), sustained for
`hold_steps` while the mechanism is intact and still.

## Why this is strategically different

- **vs the seed (`rlbench/push_buttons`)**: the seed's entire plan — press the three
  buttons in some order and be done — scores only the 0.05 press band here and ends with
  every button popped back up (smoke check 4 constructs exactly this and asserts
  rejection). Here pressing is only the *precondition*; the goal state is maintained by a
  second body (the shutter) that must be interleaved with the presses under a ~3 s
  spring-return deadline. The order is not prescribed by an instruction but **forced by
  geometry**: the plate enters from one corridor end, so the only feasible order is
  nearest-that-end-first — and which order that is, is randomized per reset.
- **vs `push_button_i217` (recoil button)**: i217 is a single button whose difficulty is
  the press dynamics themselves (recoil). Here the presses are easy; the difficulty is
  the press-chase interleave, the captive-shutter transport, and the geometry-forced
  ordering of three buttons.
- **vs `weighbridge_i6`, `balance subset-sum i40`**: those are statics/equilibrium
  puzzles; no timing, no captive slide.
- **vs `cam_staircase_i35`, `dial_setpoints_i161`, `oven_dials`**: those are rotary
  mechanism-setpoint tasks; this is a prismatic press-and-cover race with an
  irreversibility twist (the plate covering well k also seals the retreat past well k —
  pry-back live-reverts success, smoke check 10).
- **vs `pen_holder` exemplar**: that is a place-into-receptacle task; nothing is placed
  here — both manipulated bodies are captive parts of the mechanism.

## Solution outline (solve.py — no teleports at all)

Scene-level env (`robot="null"`); every action is a world-frame force through the scene's
probe-force plant (frame-encoded plant-side, applied every substep).

1. Reset, settle, read which corridor end the plate spawned at; order the buttons
   nearest-shutter-first (geometry-forced).
2. Per button (×3): velocity-servo press straight down to the 24 mm hard stop
   (gain audit 20·dt/m = 0.83 < 1), short hold; force off; velocity-servo **chase** the
   plate to a staging target that covers the just-pressed head with the judged 5 mm
   margin while stopping ≥ 9 mm short of the next raised head (gain 40·dt/m = 0.83 < 1,
   0.5 N feedforward to beat ~0.55 N rail static friction); brake, settle, verify the
   head is trapped at 12 mm by readback. Each chase takes < 1.5 s of the ~3 s rise
   window.
3. Forces off, wait for the sustained-success counter, then ≥ 3.5 s hands-off
   persistence; `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Scores are monotone across phases: 0.000 → 0.050 (press latch) → 0.300 → 0.550 (traps)
→ 1.000. Verified on the forge for seeds 0, 1 (side +1) and 2 (side −1, mirrored order).

## Embodiment argument (single Franka + parallel jaw, OSC)

Base ≈ 0.55 m from the console center, perpendicular to the corridor. All interaction
points sit at 6–12 cm height in a 72 × 12 cm footprint — comfortably inside a Franka's
dexterous workspace. Presses: closed-gripper fingertip push straight down on the 40 mm
square button heads (the same primitive as the seed task). Shutter transport: the plate
carries a 24 × 20 mm tab standing 30 mm proud of its top; a closed-gripper lateral push
on the tab slides the plate along its corridor (planar push, no grasp needed; the
corridor bears all reaction forces). No regrasps, no bimanual needs, forces ≤ ~4 N.

## Execution order (declared)

Forced by geometry, randomized per reset: the plate spawns at corridor end s ∈ {+x, −x};
the only feasible order is nearest-that-end-first (s > 0 → blue, green, red; s < 0 →
red, green, blue), because the plate cannot pass a raised head and cannot leave the
corridor. The solve reads s from the settled state and derives the order.

## Checks (smoke.py — 12)

1. settle/no-NaN; 2. randomization readback across seeds (console xy/yaw spreads, both
corridor ends seen, start gap in band, plungers assembled); 3. null policy (600 steps);
4. **seed-strategy rejection** (press all three, no chase → all pop, 0.05 only);
5. held press banks nothing (2 s at the stop, released → pops); 6. shove at a raised
head stalls the plate (probe-real: moved then held still under 6 N); 7. 12 N lift-out
blocked by the roof lips (probe-real: took up the 2 mm clearance); 8. late chase after
the pop is blocked; 9. partial coverage: plate physically overhangs all three pressed
heads but sits outside the judged all-covered window → 0.55, no success; 10. pry-back
from a constructed success live-reverts success (trap latches keep 0.80); 11. no
accidental success outside the sanctioned segment; 12. final no-NaN. Records frames.npz.

## Files

- `scene.py` — geometry class `G`, compound procedural spawners (console / plunger /
  plate), `SceneCfg` (tunables + info), `ShutterTrapScene`, `SCENES.register
  ("shutter_trap")`, env `simgen.shutter_trap` (robot="null").
- `solve.py` — force-only solution (above).
- `smoke.py` — 12-check rejection battery (above).
