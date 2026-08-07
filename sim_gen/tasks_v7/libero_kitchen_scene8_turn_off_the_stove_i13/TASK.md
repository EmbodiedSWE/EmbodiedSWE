# libero_kitchen_scene8_turn_off_the_stove_i13 — Smother the Burner (scene `burner_snuff`)

Put a stove out WITHOUT any joint: a steel KETTLE stands on the lit burner's post —
lift it off and set it upright on the GREEN TRIVET (not the white plate; the two
pads swap sides per episode, so the target must be identified by COLOR), then take
the copper SNUFFER CUP lying mouth-UP on the deck, flip it MOUTH-DOWN, and seat it
over the burner post so its rim rests on the hob plate, fully enclosing post +
flame. The order is physically inherent: while the kettle occupies the post the
cup cannot seat (the 98 mm kettle body is wider than the 94 mm cup interior).

## Provenance

- **Seed:** `libero_90/libero_kitchen_scene8_turn_off_the_stove`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene8_turn_off_the_stove.py`)
  — "turn off the stove": a flat stove USD with a knob joint, two untouched
  moka-pot distractors, robot=franka; success = the knob's JOINT ANGLE below a
  threshold. The whole plan is one rotating contact on a fixture DoF — no
  transport, no reorientation, no ordering, no other object judged.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `burner_snuff`,
  env `simgen.burner_snuff`, robot `"null"`), `solve.py` (teleport solution),
  `smoke.py` (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's entire skill is actuating a fixture's internal DoF
  (servo a knob past a joint threshold). Here there is NO JOINT ANYWHERE in the
  scene — nothing rotates on a fixture, and the stove is extinguished by
  SMOTHERING: two free bodies each transported to a judged outcome, one through a
  180° in-hand REORIENTATION and an enclosure-over-a-post placement, with a
  physically inherent occupancy precondition (clear before cap) and a
  color-identity binding (green trivet vs white plate, sides shuffled). The
  seed's naive translation — "do something at the burner" — is constructed in
  smoke checks 8/10/11/12 (cover without flipping, seat beside, cap while
  occupied, pile everything on the burner) and all are refused.
- **vs `close_grill_i8` (read this session):** that task BUILDS a free-standing
  multi-body structure (log-cabin crib) where every placement rests on prior
  placements. Here nothing is stacked into a structure: the two placements are
  independent seatings on fixtures (trivet, hob plate), the cup placement is an
  ENCLOSURE (a body lowered AROUND a post), and the coupling between the two
  moves is occupancy (must vacate before capping), not support.
- **vs `light_bulb_out_i5` (read this session):** that is a counterweighted
  lever/ballast mechanism released by extraction — a fixture with internal DoF
  does the work. Here there is no articulation and no stored mechanical energy;
  both outcomes are direct placements.
- **vs `pen_holder` (robobench exemplar, read this session):** that inserts a
  small object INTO a container. Here the container itself is the manipulandum,
  inverted over a fixed post — the enclosure is upside-down and the insertion
  direction is reversed (fixture enters the moving body).
- **vs `close_microwave_i4/i5`, `setup_checkers_i2`, `peg_insertion_side_i1/i2`,
  `pick_single_egad_i3/i4`, `pour_water_i7`, bowl/pan pick-place tasks (known
  secondhand):** no drop-gate or twist-lock mechanism, no ordered silo insertion,
  no pin fastening or ramrod, no pouring, and not a bare pick-and-place — the cup
  move alone (flip 180° + enclose a post within a 22 mm annulus, rim seated at
  hob level) has no analog in any of them, and the color-vs-position pad binding
  adds an identity decision none of them require.
- **Execution order is REQUIRED and physically inherent:** the cup cannot reach
  the rim band while the kettle stands on the post (98 mm body vs 94 mm interior,
  asserted in cfg `__post_init__` and proven by smoke check 11).

## Randomization (per episode, verified by readback in smoke)

Burner xy jitter (the cap target moves), trivet/plate SIDE SWAP + per-pad xy
jitter (color identity, not position, marks the target), cup xy jitter + free
yaw, kettle free yaw (handle direction) + tiny xy jitter on the post.

## Rubric

`success()` iff, settled (kettle and cup |v| < 0.05 m/s) and finite:

- KETTLE: upright (≤12° off vertical), centred on the GREEN trivet (|x|,|y| <
  35 mm in the trivet frame), bottom at the trivet top ±12 mm;
- CAP: cup mouth-down (axis ≤15° off straight down), rim in the band −6..+12 mm
  about the hob-plate top, cup axis within 15 mm of the burner-post axis (with
  the rim seated and the axis in tolerance, the post is fully inside the interior
  BY CONSTRUCTION — cfg assert).

`score()` (latched in `post_step`; kettle/cap latches calm-gated): `0.15 ×` clear
(kettle ever clear of the burner) `+ 0.25 ×` kettle ever at rest on the trivet
`+ 0.15 ×` cup ever mouth-down `+ 0.20 ×` cap ever seated. Capped at 0.75;
exactly 1.0 iff `success()`. Doing nothing scores ~0.

Cfg `__post_init__` asserts the honesty geometry: kettle wider than the cup
interior (cannot be hidden under the cup, and blocks the cap while on the post);
cap tolerance ⇒ post fully inside; seated cup swallows post + flame; a cup
resting on the post top reads far above the rim band; ≥18 mm annular clearance
(arm-feasible); kettle rests stably centred on the post.

## Teleport solution (`solve.py`) — transport only, three writes, all ending in free space

Each write carries a body to a free-space HOVER with zero velocity (the carry a
gripper performs): (1) kettle straight up off the post (the physical fact "bottom
far above the post top" fires the clear latch); (2) kettle to 12 mm above the
green trivet — target xy read back from the trivet body itself (color binding);
(3) cup, reoriented mouth-down, to a hover with its rim 12 mm ABOVE the post top
— `capped()` is asserted False at the hover. Every seating is GRAVITY + CONTACT:
the kettle falls onto the real trivet; the cup falls, the post enters the
interior, and the rim lands on the real hob plate. `SIM_GEN_SCORE` at every
phase boundary is non-decreasing (0.000 → 0.150 → 0.400 → 0.550 → 1.000 →
1.000), ≥3.3 simulated seconds hands-off persistence, then
`SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1, both SUCCESS,
provably distinct layouts by stdout readback** (burner (+0.326,+0.014) vs
(+0.346,+0.005); kettle yaw +76.9° vs −19.7°; cup (+0.125,−0.020) yaw +32.2°
vs (+0.119,+0.028) yaw −101.2°).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (−0.12, 0), facing the deck: burner ~0.45 m, pads ~0.45–0.48 m,
cup spawn ~0.24 m — all within a comfortable dexterous shell, tallest judged
point (kettle top on the post, ~0.17 m) trivially low. Per-object contact
strategy: the kettle (250 g) is pinched TOP-DOWN across its 20 mm stick handle
(≪ 80 mm jaw stroke), lifted vertically off the post, carried, and set down on
the trivet — ±35 mm centring and ±12° upright tolerance are far beyond arm
repeatability, and the set-down is a release from a few mm. The cup (150 g) is
pinched by its 8 mm wall rim, rotated 180° in the wrist (a single joint-7/6
motion with the plate hanging clear), centred over the post, and lowered — the
22 mm annular clearance self-guides the final seating and the 15 mm axis
tolerance is compliant. Both grasp approaches are clear verticals; nothing is
reached under or through anything.

## Checks (`smoke.py` — rejection battery, 13 named checks, ALL PASS on the forge)

1. settle: states finite, kettle standing on the lit post, cup mouth-up; score 0.
2. randomization readback: burner xy Δ16.1 mm, cup xy Δ14.7 mm, cup yaw Δ95.5°,
   kettle yaw Δ66.6° across seeded resets.
3. side permutation: the green trivet appears on BOTH sides over 10 resets
   (sides [1,1,−1,1,−1,1,1,1,−1,−1]; side flag verified against the trivet
   pose readback).
4. null policy: 240 idle steps → score ~0, no success.
5. kettle-only partial: kettle correctly on the trivet, cup untouched → no
   success, score ≤ 0.45.
6. WRONG PAD (color identity is load-bearing): kettle on the WHITE PLATE +
   burner correctly capped → trivet clause refuses, no success.
7. bare deck: kettle set down on the bare deck + burner correctly capped → no
   success.
8. unflipped cover: cup rested mouth-UP on the post top ("cover" without the
   flip) → rim far above the band, not capped, no success.
9. cap near-miss: cup GENUINELY seated (rim_dz −0.0 mm, post enclosed) but
   19.0 mm off the post axis (physically possible: clearance 22 mm; tolerance
   15 mm) → capped refuses, no success.
10. beside-burner: cup mouth-down seated flat on the deck — its rim height reads
    INSIDE the rim band → the axis clause alone refuses, no success.
11. BLOCKED CAP (the order is physics): cup dropped over the OCCUPIED burner
    lands high on the kettle (98 vs 94 mm) — rim settles +139 mm above the hob
    (band −6..+12 mm) → not capped, no success.
12. kettle-perch: burner correctly capped, kettle rested on TOP of the seated
    cup (flame covered, both bodies "at the burner") → kettle not on the trivet,
    no success.
13. frames.npz (71 × 600 × 960 × 3) recorded and saved in CWD.
