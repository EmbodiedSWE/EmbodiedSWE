# Task `libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i375` — FlameKeeper: swap the pot onto the burner without killing the flame

## Provenance

Seed task: `libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove` — pick the
moka pot off the counter and put it on the stove.

## The task

A kitchen stove has ONE burner: a dark spring-loaded burner PLATE (26 x 15 cm) riding a
vertical prismatic joint (20 mm travel) inside a walled well. A pilot spring (4.7 N,
re-applied every physics step as a live external force) pushes the plate UP; the plate
stays PRESSED only while enough weight actually rests on it through contact. The burner
is a dead-man valve: once the foul is armed (the plate has rested pressed), if the plate
rises past 70 % of its travel for a sustained streak, the flame goes out PERMANENTLY —
an irreversible foul that forces the score to 0 for the rest of the episode.

At reset a frying pan sits on the plate (random side, random yaw), keeping it pressed,
and the moka pot stands on the counter (randomized xy and yaw). The goal: the moka pot
ALONE upright on the pressed plate, the pan parked on the counter at least 25 cm clear
of the stove, flame still lit, everything at rest.

## Strategic difference from the seed (and from every task examined)

- The seed is a free pick-and-place onto an empty stove. Here the destination is
  OCCUPIED, and the naive order — "remove the pan first, then place the pot" — is a
  PERMANENT, PHYSICAL failure: the instant the plate is unloaded the pilot spring pops
  it past the foul band (~17 steps) and the flame dies forever. The only winning plan
  is pot-FIRST: install the pot on the plate's free half while the pan still presses it
  (the two payloads share the plate transiently), and only then remove the pan.
- The order is enforced by a live force balance (spring vs. resting weight through
  contact), not by a scripted rule: nothing about the plate is ever written after
  reset. The end state of the illegal order is geometrically IDENTICAL to the winning
  arrangement — only the event history distinguishes them, and the rubric reads that
  history through the physical foul latch.
- Unlike the corpus's mechanism tasks (balances, gates, chutes, dispensers), the
  mechanism here is a *constraint on when the hand may leave*, not a tool the solver
  drives: the plate is never the transported object and never the goal — it is the
  clock the solver must keep pressed.

## Solution outline (solve.py, teleport embodiment)

1. P0 — settle 120 steps; read back the pan's random side; assert pressed/armed/alive.
2. P1 — teleport the pot to a hover 10 mm above the plate's FREE half (handle pointing
   away from the pan), release; contact dynamics seat it beside the pan. The plate
   never unloads; the flame never flickers. (retry on bounce)
3. P2 — teleport the pan off to the counter park spot (33 cm from the stove); the pot
   alone keeps the plate pressed. Swap latch fires; score 0.60.
4. P3/P4 — success() holds continuously 2 s, then 3.3 s hands-off persistence; 1.0.

Teleports are transport-only: single root-state writes carrying ONE object at a time
through free space to a hover above the destination; every seating, the plate's height,
the foul, and the final rest are 100 % contact dynamics. Forge-proven on seeds 0 and 1
(monotone 0.00 -> 0.35 -> 0.60 -> 1.00, `SIM_GEN_SOLVE: SUCCESS`).

## Embodiment argument (Franka)

Base at ~(0.9, 0.0) facing the counter: every waypoint (pot start ~(0.62,-0.20), plate
at (0.30, 0.10), pan park (0.62, 0.18)) is within a 0.85 m reach disc. The moka pot is
a 6.3 cm octagonal body with a side handle — a standard top-down or handle pinch grasp;
the pan has a bar handle overhanging the well ring (the ring top sits 27 mm below the
pan's rim), a natural side pinch. The plate well is open from above; the pot slot and
the pan spot are 12+ mm apart at worst case, so the pot can be lowered vertically beside
the pan without contact. No step needs more than one object in hand, and the critical
skill — set the pot down BEFORE lifting the pan — is pure task ordering, not dexterity.

## Execution order (declared)

1. Place the pot on the burner plate's free half (pan still on the plate).
2. Move the pan from the plate to the counter, ≥ 25 cm clear of the stove.
Reverse order = permanent flame-out foul; score is forced to 0.

## Rubric

Latched, monotone, flame-gated: near 0.10 (pot within 12 cm) + on-plate 0.25 (pot
upright at rest inside the plate's judge box) + swap 0.25 (pot on the pressed plate,
pan clear) — base clamped to 0.60, forced to 0 the moment the foul latches, 1.0 only
while success() holds live. success() = flame alive AND pot alone upright on the
pressed plate AND pan clear on the counter AND everything settled.

## smoke.py battery (10 checks, all forge-passed)

1. settle/no-NaN — pan pressing, foul armed, flame alive, score 0, no success.
2. randomization readback — 3-seed max-pairwise pot xy + pan yaw; both plate halves
   occur over 6 seeds.
3. null policy — 300 idle steps, score ~0.
4. FLAGSHIP illegal order — the seed's plan (pan off first): the unloaded plate
   physically pops past the foul band (q_max readback), flame dies; the pot is then
   installed anyway — the end state passes EVERY geometric success clause yet success
   is refused and score == 0; 240 further steps: the foul never clears.
5. grace period — a 5-step pan lift-and-return never reaches the foul band; the flame
   survives (the foul is not hair-triggered).
6. rim perch — pot stood on the well ring beside the plate: near credit only (≤ 0.105).
7. swap incomplete — pot on the free half, pan still on the plate: 0.35, no swap latch.
8. tipped pot — pot on its side on the free half: upright/z clauses refuse the latch.
9. pan too close — legal order but the pan inside the 25 cm clearance: no swap latch.
10. frames.npz — video captured and saved.
