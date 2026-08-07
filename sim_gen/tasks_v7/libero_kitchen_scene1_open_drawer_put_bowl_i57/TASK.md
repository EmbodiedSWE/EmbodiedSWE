# libero_kitchen_scene1_open_drawer_put_bowl_i57 — HanoiRingsScene (`simgen.hanoi_rings`)

Rebuild a tower of three graduated rings on the GREEN post under the Tower of Hanoi
rule: move one ring at a time, every move is a thread-over-post insertion, and a
LARGER ring must never come to rest on a SMALLER one — the scene watches every
settled configuration and one violation permanently voids the episode. With one
spare post this forces the classic 7-move recursion.

## Seed provenance

- **Seed task**: `libero_90/libero_kitchen_scene1_open_drawer_put_bowl` (RoboVerse
  `roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_drawer_put_bowl.py`) —
  "open the top drawer of the cabinet and put the bowl in it." An articulated
  wooden cabinet (prismatic drawer) plus one bowl; the plan is two independent
  subgoals — actuate the drawer open, then a single pick-and-place into the revealed
  containment region — judged by one final bbox-containment test on the bowl.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| fixture | articulated cabinet (prismatic joint to actuate) | three KINEMATIC posts on a board — nothing articulated, nothing to open; the state that matters lives entirely in the ring configuration |
| objects | one bowl (one placement) | three graduated rings (LARGE red 70 mm, MID yellow 57 mm, SMALL white 44 mm across flats, common 28 mm hole) |
| plan depth | 2 fixed subgoals, no interaction between them | **7-move computed recursion** through a buffer post (small→T, mid→B, small→B, large→T, small→S, mid→T, small→T) — each move's legality depends on the full current state |
| placement act | set the bowl down inside an open box region | **thread a ring over a post tip**: release above the cone apex, gravity + cone + shaft do the seating; a ring can only enter or leave a post vertically over the tip (physics, smoke 5) |
| ordering | trivial (open before insert) | **rule-constrained total order**: the ONLY legal 7-move plan; any larger-on-smaller rest state latches a permanent violation (smoke 8) that even a subsequently perfect tower cannot undo (smoke 9) |
| reversibility | drawer/bowl freely re-doable | mistakes are latched forever — the rubric encodes irreversibility even though every single move is physically reversible |
| perception | fixed layout | board xy ±40 mm, yaw ±30°, and the three posts are PERMUTED over the slots — which post is the green target must be perceived every episode |
| assets | LIBERO USDs | 100 % procedural (compound spawners): kinematic board + 3 kinematic posts (pad+shaft+cone), 3 dynamic octagonal-annulus rings (8 box colliders each) |
| judging | one final bbox containment | latched first-move / large-home / mid-home credit (0.15/0.35/0.30) + live success: all threaded on green, strict L<M<S order, seated, contiguous, at rest, violation-free; violation caps at 0.15 |

Code structure shares nothing with the seed: `@SCENES.register` BaseScene, two
compound spawners, post-frame threaded/ordering predicates, a consecutive-step
violation watchdog latched in `post_step`, `register_env(..., robot="null")`.

## Why strategically different

The seed's whole skill is *actuate one articulated container, then one free
placement into it* — two independent subgoals, no ordering pressure, judged by a
single containment box, every step re-doable. Here that plan gains nothing: there is
no container and nothing articulated to open, and delivering the objects to the
target — even in the correct order — scores 0 unless each one was threaded over a
post tip (smoke 6 constructs the correct L/M/S tower standing on the board beside
the green post: score 0). What the solver must bring instead is (1) **multi-step
look-ahead planning** — the 7-move Hanoi recursion, where the right move depends on
the whole configuration and the buffer post must be used twice, with intermediate
states (small parked on mid on the spare post) that superficially move AWAY from the
goal; (2) **rule-keeping under a permanent watchdog** — a single settled
larger-on-smaller configuration voids the episode forever (smoke 8/9), so greedy
shortcuts (drop the large ring last, stack out of order and fix it) are latched
failures, not recoverable states; (3) **perception of a randomized post permutation
and board pose** — the target is identified only by its green pad. Every move is a
funnel-guided vertical insertion (release over a cone tip, 8 mm annular clearance),
not the seed's tolerance-loose set-down.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; readback layout (board xy/yaw, post→slot permutation,
   green slot); all three rings threaded on the source post; baseline score 0.
2. Seven moves, each: **teleport = transport only** — one root-state write to the
   free-space release point 129 mm above the destination post's pad (over the cone
   apex), ring flat with the post's yaw, zero velocity — then hands-off: gravity
   drops the ring, the cone funnels the hole onto the shaft, it lands seated.
   An escalating downward CoM nudge (0.5×–2.5× ring weight) is armed for hang-ups.
   - move 1 small→green: first-move latch, score 0.15 (P1)
   - moves 2–3 mid→spare, small→spare: buffer stack, score holds 0.15 (P2)
   - move 4 large→green: large-home latch, score 0.50 (P3)
   - moves 5–6 small→source, mid→green: mid-home latch, score 0.80 (P4)
   - move 7 small→green: success, score 1.0 (P5)
3. **P6** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.00 → 0.15 → 0.15 → 0.50 → 0.80 → 1.0 → 1.0.
Every insertion is pure gravity + contact; the score trace staying monotone to 1.0
is itself the certificate that no move ever tripped the violation watchdog.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m).
Board centre at 0.36 m, slots at 0.21/0.36/0.51 m (±jitter/yaw): all release points
(≈0.17 m height) sit deep inside the dexterous shell.

- **Grasp**: rings are octagonal annuli lying flat — top-down parallel-jaw pinch
  across opposite OUTER FLATS (70/57/44 mm across flats, all < 80 mm max opening;
  16 mm tall faces). Picking the top ring of a stack: fingers straddle only that
  ring; the next ring down is ≥ 13 mm smaller in radius, so a ~60 mm aperture
  clears it. Ring masses 60–100 g — trivial payload.
- **Move**: lift straight up ≥ 110 mm (clear of the 105 mm post tip) — the hole
  around the shaft enforces exactly the vertical extraction the rule describes —
  translate, hover over the destination tip.
- **Release**: open the gripper at ~130 mm above the pad; the cone tip (20 mm tall)
  funnels the 28 mm hole onto the 12 mm shaft (8 mm radial clearance per side) and
  gravity seats the ring. No insertion force, no precision placement: the release
  tolerance is the ±8 mm the cone recovers, demonstrated by every solve drop.
- **Perception**: the green pad, the ring colors/sizes, and the slot permutation
  are all visually explicit (distinct display colors, 13 mm size steps).
- Board and posts are kinematic: incidental contact cannot move the goal frames.

## Execution order (declared)

`small→green, mid→spare, small→spare, large→green, small→source, mid→green,
small→green` — the unique legal 7-move plan. The rule is enforced by a latched
watchdog (12 consecutive substeps of a settled larger-on-smaller state), and the
thread-over-tip constraint is physics: smoke 5 shoves the threaded small ring
sideways at 3× its weight for 1.5 s and its centre never leaves the shaft.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (board yaw −6.1°, green in the middle slot, perm [1,2,0]):
  SUCCESS in 23 s, monotone scores 0.00/0.15/0.15/0.50/0.80/1.0/1.0.
- `solve --seed 1` (board yaw +23.4°, perm [1,0,2]): SUCCESS — two seeds, every
  drop threaded first try, the wedge nudge **never fired on any tested seed**.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 11/11**, frames.npz (67 × 600 × 960) saved:
  1. settle/no-NaN; three rings threaded on the source post, none on green; score 0
  2. randomization readback differs (board yaw Δ24.2°, board xy Δ39 mm)
  3. slot permutation: all 6 post→slot permutations seen over 10 resets, green
     post in all 3 slots
  4. null policy: 240 idle steps, score 0, no success, no violation
  5. POST INTERLOCK: 3×-weight sideways shove for 1.5 s, ring centre max 8.1 mm
     off-axis (hole apothem 14 mm) — never leaves the shaft, re-seats threaded
  6. SEED STRATEGY: correct L/M/S tower standing BESIDE the green post → score 0
  7. wrong post: correct tower threaded on the spare gray post → score 0.15
  8. HANOI VIOLATION: S/M/L inverted stack on green → watchdog latches, score 0.15
  9. VIOLATION PERMANENT: correct L/M/S tower rebuilt after the latch — lg/md
     latches set, yet score stays 0.15 and success stays False
  10. near-miss: L/M correct, small ring flat on the board centimetres away →
      score 0.65, no success
  11. video frames.npz saved

## Files

- `scene.py` — HanoiRingsScene + post/ring spawners + rubric; registers
  `simgen.hanoi_rings`.
- `solve.py` — 7-move teleport-transport + gravity-threading certificate (`--seed N`).
- `smoke.py` — 11-check rejection battery + video.
