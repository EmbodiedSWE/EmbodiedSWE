# libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_i90 — Hang the Bottle by its Lip (scene `lip_hang_rack`)

A floor-standing RACK TOWER carries two cantilevered HANGING SLOTS side by side —
each a pair of parallel horizontal rails (14 cm long, 34 mm gap) at 32 cm height,
OPEN at the front end, closed at the back by the tower. A GREEN and a RED tag above
the slots mark them, and the colors SWAP SIDES per episode. One wine bottle (55 mm
body, 20 mm neck, 50 mm LIP disk at the top) stands on the floor. Success = the
bottle HANGING by its lip in the GREEN-tagged slot, slid all the way back to the
tower stop, body dangling in free air, settled. The lip is wider than the rail gap
and the neck narrower: the only route is a horizontal THREADING of the neck into the
open front end of the correct slot, an 11 cm slide backward with the lip riding the
rail tops, and a release into a stable suspended hang.

## Provenance

- **Seed:** `libero_90/libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack.py`)
  — pick THE wine bottle and rest it ON TOP of THE rack; success is a bbox test over
  the rack's top region. Support from below, goal visually given, pure prehensile
  pick-and-place.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `lip_hang_rack`,
  env `simgen.lip_hang_rack`, robot `"null"`), `solve.py` (teleport solution),
  `smoke.py` (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the SUPPORT RELATION is inverted — the seed ends with the bottle
  resting ON a surface (support from below); here the bottle must end SUSPENDED
  (support from ABOVE through a lip-on-rail catch), a relation the seed's plan
  cannot produce by construction. The seed's strategy transplanted verbatim ("put
  the bottle on the rack") is exactly what smoke checks 6-7 physically construct —
  laid across the rail tops, stood on the tower top — and the rubric rejects both
  (score ≤ 0.20, never success). The motion plan is also different in kind: instead
  of lower-onto-a-surface, the solver must exploit the lip/gap/neck metric triple
  (50 > 34 > 20 mm) — reorient to upright at height, thread the neck between the
  rails from the slot's one open end, slide backward to a blind stop, release into
  a pendulum hang. And the goal is not visually unique: two identical slots differ
  only by their color tags, which swap sides per episode.
- **vs the corpus tasks read this session:** `stack_wine_i48` (same seed family)
  is an epistemic task — hidden mass probing with trivially easy placement into
  open V-cradles; here everything is visible and the challenge is the threading
  kinematics + support-relation inversion, with color-binding. The bistable tilt-bin
  task (i237) is mechanism actuation (open/close a counterweighted drawer); this
  rack has NO moving parts. `hockey_i325` is captive-gate extraction plus floor
  conveyance of a ball INTO a chamber (containment from the side, at floor level);
  here nothing is captive and the goal relation is suspension at height. No task
  read this session ends with an object HANGING from an overhead support it was
  threaded into.
- **Execution order (declared):** thread-then-slide is geometrically forced (the
  slot is closed everywhere but the front), but the rubric hard-codes no order —
  any trajectory that ends settled, hanging in the green slot within the stop band,
  wins. There is no hidden second ordering constraint.

## Randomization (per episode, verified by READBACK in smoke)

GREEN side Bernoulli — the tag bodies are physically re-posed to their episode
sides (readback of tag rack-frame positions matches the internal `_green_v` on
every seeded reset); rack xy jitter ±3 cm + yaw ±10°; bottle spawn side Bernoulli
± xy jitter ±3 cm + free yaw.

## Rubric

`success()` iff, judged live on physical settled poses in the RACK frame:

- origin height within (−9, +12) mm of the nominal hanging height (lip underside on
  the rail-top plane — rejects floor stand (12 cm below), resting across the rail
  tops (14 cm above), tower top (30 cm above), and aloft);
- laterally within 12 mm of the GREEN slot centre (rejects the red slot at 15 cm);
- depth within [8, 45] mm of the tower face (rejects hangs short of the stop; the
  mouth is at 110 mm);
- upright within 15°; settled (|v| < 0.05 m/s, |ω| < 0.60 rad/s).

`score()` (latched every physics substep in `post_step`): `0.15 ×` lift (origin
ever above 0.17 m — a null policy never) `+ 0.15 ×` mouth (neck ever threaded into
the GREEN slot, loose bands) `+ 0.30 ×` running max of slide-to-stop progress while
threaded, capped at 0.60; exactly 1.0 iff `success()`. Doing nothing scores ~0; a
perfect seed-strategy placement (resting anywhere ON the rack) is capped at 0.15.

## Teleport solution (`solve.py`) — transport only, ending in free space

- **P1 — transport + catch:** ONE pose write stages the bottle upright at the open
  front end of the green slot, lip underside 3 mm ABOVE the rail-top plane, zero
  velocity — nothing threaded-to-depth yet (depth progress is 0 at the mouth by
  construction, success 65 mm of slide away). Released, it drops 3 mm and the lip
  CATCHES across the two rails — the hang itself is earned through contact.
- **P2 — slide to the stop (pure contact dynamics):** a velocity-limited world
  force (1.2 N, v ≤ 0.10 m/s, stall escalation never needed) pushes the hanging
  bottle backward along the slot, lip riding the rail tops, plus a lateral
  centering term and the compensating torque (r × F, r = 10 cm up the neck) that
  re-expresses the CoM force as the arm's push just under the lip so the bottle
  does not pendulum-tilt. Force CUT at 40 mm depth readback; the bottle coasts,
  settles, and hangs — success judged hands-off.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing (0.00 → 0.30 →
  1.00), ≥ 3.3 simulated seconds hands-off persistence, then `SIM_GEN_SOLVE:
  SUCCESS`. **Verified on the forge: seeds 0, 1 and 2, all SUCCESS, provably
  distinct by stdout readback** — seed 0: rack (0.648, 0.017) yaw 178.0°, green
  side +, bottle at −y; seed 1: rack (0.630, −0.016) yaw 187.8°, green side +;
  seed 2: rack (0.594, −0.006) yaw 178.5°, green side − (solved into the MIRRORED
  slot), bottle at +y.

## Embodiment sanity (single-arm Franka feasibility)

Base at the origin: bottle spawn at ~0.36 m radius, the slot mouths at ~0.48 m and
the tower stop at ~0.62 m, all at working heights ≤ 0.35 m — inside a Franka reach
envelope. Per-object contact strategy: a side grasp of the 55 mm vertical body with
the 80 mm jaw (25 mm margin, bottle 0.45 kg), lift to bring the 50 mm lip just
above the rail-top plane (0.32 m), thread the 20 mm neck down between the rails of
the green slot from the OPEN FRONT END (±7 mm lateral play), then slide 11 cm
horizontally backward — the hand stays BELOW and OUTSIDE the slot the whole way
(the rails are cantilevered with nothing above them, the tower is behind the stop),
so the only clearance needed is for the fingers around the body in free air.
Release is a plain jaw-open: the lip is already resting on the rail tops during the
slide, so letting go simply transfers support to the rack. The rack is kinematic
and cannot be disturbed; the tags are visual markers without colliders. Identifying
the green slot is a color observation, not a manipulation.

## Checks (`smoke.py` — rejection battery, 15 named checks, ALL PASS on the forge)

1. settle: states finite; bottle standing on the floor; tags at OPPOSITE slots with
   the green tag at the green slot (readback); all still.
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization A: GREEN side flips across 8 seeded resets, tag-body readback
   matches `_green_v` every time, bottle spawn side flips.
4. randomization B: rack yaw spread > 2° and rack xy jitter > 4 mm (readback).
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy A: bottle laid HORIZONTALLY across the rack's rail tops (body
   bridging the two inner rails), resting level and still ON the rack → NOT
   success, score ≤ 0.20 (the hang, not the rest, is the task).
7. SEED strategy B: bottle standing upright ON the tower top → NOT success, ≤ 0.20.
8. under the slot: bottle standing on the FLOOR directly below the green slot →
   score ~0, no success (right (x, y), wrong relation).
9. near-miss depth: GENUINE lip-hang in the green slot (built by a real 3 mm drop)
   but at mid-rail depth, short of the stop → NOT success, score ≤ 0.601.
10. latched credit: teleporting the bottle from that hang back to the floor leaves
    the latched score unchanged, still no success.
11. wrong slot: GENUINE lip-hang at STOP depth but in the RED-tagged slot → NOT
    success, score ≤ 0.20 (threading credit is green-gated).
12. false middle slot: bottle released upright BETWEEN the slots falls STRAIGHT
    THROUGH (76 mm middle span > 50 mm lip) to the floor → NOT success — the
    "any pair of rails" exploit is physically impossible.
13. transient motion: bottle in the success pose but MOVING (0.6 m/s) → the settle
    gate rejects at the judged instant; probe removed unsettled.
14. rejection audit: success() never True at any judged point in the battery.
15. final no-NaN. Plus `frames.npz` recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest:
the neck threads the gap with play, the lip cannot pass the gap, the middle span
cannot hold the lip, a hanging bottle clears the floor by > 5 cm, a bottle standing
under the slot cannot reach the rails, the body fits the Franka jaw, and the
success depth window is reachable past the lip radius.
