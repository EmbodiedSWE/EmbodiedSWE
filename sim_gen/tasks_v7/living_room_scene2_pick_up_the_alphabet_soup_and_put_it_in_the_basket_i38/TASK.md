# spring_bay — press the blue can against the spring plunger and seat it captive behind the lip

`living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i38` · scene `spring_bay` · env `simgen.spring_bay`

## Seed provenance

Seed task: `roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.py`
(RoboVerse / LIBERO-90). The seed is a pick-and-place: lift the alphabet-soup can off a
table crowded with six other grocery items and release it inside an open basket;
`_terminated` is a bounding-box containment readout on the basket's `contain_region`.

Kept from the seed: the protagonist (a soup-can-sized cylinder, 66 mm x 105 mm), the
"one target among identified distractor clutter" framing (reduced to a single RED foil
can), and the surface story of loading a can into a container.

Changed: the open basket became a **spring-loaded dispenser bay** whose cavity is
SHORTER than the can; the drop-in strategy became a **compress-then-seat** mechanism;
the containment readout became a **mechanism readback** (plunger compression) plus
settled-geometry windows; the tabletop scene became a floor scene with a randomized
free-yaw fixture.

## Strategic difference argument

- **vs the seed:** the seed's entire strategy is one gravity drop — transport the can
  over the basket and release; any release inside the rim succeeds. Here that exact
  move is constructed in smoke check 7 and REJECTED: the rest gap between the lip and
  the plunger is 18 mm shorter than the can, so a released can settles PROPPED on the
  lip at ~45° with the spring disengaged (~4 mm gravity compression, below the 8 mm
  engage gate and the 10 mm success gate). Success requires deliberately loading an
  elastic element and releasing so its stored energy finishes the seating — a strategy
  with no counterpart in the seed.
- **vs the corpus (35 tasks surveyed; read in full: pull_cube_i20 beam scale,
  close_box_i26 trap crate, living_room_scene3_i33 flap pantry, pen_holder exemplar):**
  every existing corpus mechanism is kinematic (hinges, slides, bayonet twists),
  gravity-driven (counterweights, drop gates, tilt-drains) or pure rigid contact
  (bridges, wedges, pile-ups). **No corpus task has an elastic element.** The
  load-bearing interaction here — press through a force that GROWS with progress
  (Hooke's law, ~3->9 N across the stroke), hold it while a second DOF resolves (the
  propped front end pivots down behind the lip), then release deliberately so the
  preloaded spring completes the task — exists nowhere else. Nearest neighbour is
  i33's pantry flap (one moving part on the container); its flap is a passive one-way
  gate that gravity re-closes, nothing is ever loaded against it, and its end state is
  unstressed, whereas this end state is a PRELOADED mechanism whose standing ~18 mm
  compression is what the rubric reads.

## The mechanism (numbers)

- Channel: floor at 0.100 m, width 96 mm (< can length 105 mm — a can can NEVER lie
  crosswise), orange front lip 45 mm tall (above the lying can's 33 mm axis — a seated
  can is captive), spring plunger at the back (prismatic joint, travel 32 mm, linear
  drive k = 400 N/m, c = 25).
- Rest gap lip-to-pad: 87 mm = 105 − 18. Seated compression 18 mm (preload ~7.2 N).
- Gravity alone on a propped can compresses ~4 mm; `comp_engage` = 8 mm,
  `comp_min` = 10 mm — three physically separated regimes, asserted in
  `SpringBaySceneCfg.__post_init__`.
- The bay is a heavy DYNAMIC body (25 kg, zero sleep/stabilization thresholds), never
  kinematic, so the joint anchor follows the reset teleport; the joint pair stays
  collision-filtered (travel is bounded by the joint limits).

## Rubric

`success()`: blue can flat + centered + aligned (bay-frame x/y/z windows, axis within
20°) AND plunger compression >= 10 mm (readback of the joint's actual travel) AND can
and plunger settled AND the red decoy nowhere on/in the bay.

`score()` (stateless, monotone along the solution): 0.10 blue near the bay + 0.15
inside the loose channel volume + 0.30 in-channel with the spring engaged (>= 8 mm),
capped at 0.55; 1.0 iff success. Null policy ~0 (cans spawn beyond the near radius).

## Solution (solve.py) — teleport for transport only

- P0 settle + layout readback; assert plunger at rest, score ~0.
- P1 TRANSPORT: teleport the blue can once, to a hover pose above the channel
  (horizontal, axis along the channel, zero velocity) — what a pick-and-carry
  delivers. Score 0.10.
- P2 hands-off drop: gravity lands it PROPPED on the lip (the naive end state;
  asserted NOT success). Score 0.25.
- P3 press by applied wrench: bay −x force ramp (5->16 N) plus the pitch-down torque
  that moves the line of action to the can's back-bottom contact (a bare CoM push
  rears the can upright — observed on the forge and now aborted + retried). The force
  is world-encoded via `encode_force` with a runtime force-frame probe (mode locked
  from measured spring progress; pods differ). The spring compresses, the front end
  pivots down the lip's inner face, the can ends flat at >= 21 mm. Score 0.55.
- P4 release: forces cleared; the STORED SPRING ENERGY shoves the can forward and
  seats it against the back of the lip (~18 mm standing compression). success() True,
  then >= 3.3 s hands-off persistence before `SIM_GEN_SOLVE: SUCCESS`.

Scores printed at phase boundaries are non-decreasing: 0 -> 0.10 -> 0.25 -> 0.55 -> 1.0.

## Embodiment argument (single Franka, parallel jaw)

- Grasp: the can is 66 mm across — inside the ~80 mm jaw span; it spawns upright on
  open floor with clearance on all sides (arc radius 0.29–0.35 m from the bay).
- Place: the channel is 96 mm wide vs the 66 mm can — >= 12 mm finger slots per side
  (asserted), so the gripper can lower the can into the channel while still holding it.
- Press: the loaded press peaks < 13 N along the stroke — comfortably inside Franka's
  continuous end-effector force envelope; the natural motion is to keep the grasp and
  push axially, low on the can (exactly the wrench solve.py applies), then open the
  gripper and retract upward past the 45 mm lip.
- Base pose: bay jitter ±5 cm and free yaw keep everything within a ~0.4 m disc;
  a Franka based ~0.45 m from the bay center reaches both can spawn arcs and the
  channel at comfortable elbow-up configurations. All interactions are from above /
  the open top; nothing requires reaching under an overhang.

## Execution order

1. `scene.py` (registers `spring_bay` + `simgen.spring_bay`)
2. `python -m simgen_tasks.<task>.solve --headless [--seed N]` — prints
   `SIM_GEN_SCORE` at phase boundaries and `SIM_GEN_SOLVE: SUCCESS`.
3. `python -m simgen_tasks.<task>.smoke --headless` — rejection battery, saves
   `frames.npz`, prints `SIM_GEN_SMOKE: ALL PASS <n>/<n>`.

## Smoke battery (smoke.py)

1. settle/no-NaN: plunger at rest, cans upright on the floor
2. baseline score ~0, no success
3. randomization readback: bay xy + free yaw vary across seeds
4. randomization readback: both can positions vary
5. null policy: ~0 after 300 idle steps
6. carried-not-placed: can posed at the hover release pose — near credit only
7. SEED STRATEGY: drop into the container -> PROPPED on the lip, spring below the
   engage gate, partial credit only, no success
8. upright-in-channel near-miss: stood in the gap -> rejected (windows + axis)
9. perched across the wall tops: no stable resting state up there — the can rolls
   off the bay on its own, never engages anything -> rejected
10. mechanism real: plunger posed at 25 mm with no can returns to rest by itself
11. acceptance construct: plunger held retracted, can laid in the opened gap,
    RELEASED — stored spring energy alone seats it -> success TRUE
12. decoy clause: red can laid on the bay while blue stays seated -> success flips
    FALSE
13. decoy removed -> success returns TRUE
14. settle gate: the seated can kicked and judged immediately -> NOT success
15. wrong object: RED can seated by the same construct -> rejected twice over
16. rejection audit: success never True outside the two acceptance probes
17. final no-NaN

## Verification record

- solve: forge (RTX-4090 pod), seeds 0 / 1 / 2 — all `SIM_GEN_SOLVE: SUCCESS`, rc=0.
  Scores non-decreasing 0 -> 0.10 -> 0.25 -> 0.55 -> 1.0; press exit 21.5–21.6 mm;
  released standing compression 17.9 mm (theory 18); >= 3.3 s hands-off persistence.
- smoke: forge — `SIM_GEN_SMOKE: ALL PASS 17/17`, rc=0, 310 frames saved to
  `frames.npz`. Seed-strategy drop lands PROPPED at 4.3–4.5 mm (< 8 mm engage gate);
  acceptance construct seats at 17.9 mm by stored spring energy alone; decoy clause
  flips success both ways.
