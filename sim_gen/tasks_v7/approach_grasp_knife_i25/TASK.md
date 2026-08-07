# approach_grasp_knife_i25 — `underpin_swap`

Swap the support column under a raised deck WITHOUT ever letting the deck drop, tip,
or be lifted: insert the replacement BEFORE removing the loaded support.

## Seed provenance

- **Seed task**: `pick_place/approach_grasp_knife`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_knife.py`)
- **Seed strategy**: pure approach–grasp–lift. Servo the Franka gripper to a knife on
  a table, close the jaw, lift via a joint offset; the checker is gripper–object
  distance `< 0.02` held for a few frames. The manipulated object IS the judged
  object, and picking it up IS the task.

## What changed and why it is strategically different

The plan is inverted, not re-parameterized:

1. **The judged object must never be grasped or lifted.** The grey DECK is the judged
   object, and touching it the seed's way — hoisting it — fires an **irreversible
   latched spoil flag** (deck free-end underside below 46 mm, deck centre above
   100 mm, or tilt beyond 12° — monitored every physics substep). The seed's end
   state is expressible here (deck hoisted) and is a smoke control: latched failure
   even after the deck is put back perfectly.
2. **The solver plan is insert-before-remove under load** (underpinning), not
   approach-grasp: slide the BLUE replacement column in under the deck's blue-banded
   end FIRST (through a real 3 mm clearance), only then drag the loaded RED column
   out sideways by its handle — a friction-loaded extraction with the deck's weight
   transferring onto the blue block purely through contact — then park the red column
   on the green pad. Ordering is enforced by construction: extraction credit is
   **counted only while the blue column is seated**, and removing red first makes the
   deck physically collapse (spoil latch, no rebuild possible).
3. **Code structure**: a support-transfer sequence with an always-on collapse monitor
   and monotone progress latches — not a gripper-distance servo loop.
4. **Distinct from the rest of the corpus**: no other task in tasks_v4–v7 is a
   support swap / underpinning under load with an irreversible negative invariant
   (nearest neighbours are the beam-balance and wedge-jack families, which load or
   lift a beam; none replace a loaded support while forbidding any deck motion).

A YELLOW decoy column shares the handle but is 18 mm too short: seating it and
removing red drops the deck end ~27 mm — past the spoil threshold — so the decoy is
rejected by **physics**, not just by the rubric (asserted in `__post_init__` lever
arithmetic and exercised in smoke).

## Teleport-solution phases (solve.py)

Teleport = TRANSPORT ONLY; every load-bearing interaction is contact dynamics:

- **P0** settle, layout readback, `SIM_GEN_SCORE` (~0.00)
- **P1** *transport*: one pose write carries the blue column to a staging pose beside
  the deck (slot x-station, block 160 mm outside the deck footprint, handle
  trailing). Nothing is under the deck; only latched approach credit (~0.05).
- **P2** *contact*: floating-hand force controller (velocity-regulated push along
  fixture −y, lateral PD on the slot x-station, yaw-hold torque) slides the blue
  column under the deck through its 3 mm clearance until seated under the band
  (~0.35). Ground friction real; stall-escalation on the push force.
- **P3** *contact, load-bearing*: same controller drags the loaded red column out
  sideways against ground friction plus the deck's weight on its block; the deck
  settles 3 mm onto the blue block purely under gravity/contact (~0.85).
- **P4** *transport*: carry the free-standing red column to the green pad; 2 mm drop
  onto the pad under gravity → success (1.00).
- **P5** persistence: ≥ 3.3 sim-seconds hands-off, success still holds →
  `SIM_GEN_SOLVE: SUCCESS`.

Forge results: SUCCESS on seeds 0 and 3 (fixture yaw +42.5° / −20.9°), score sequence
0.00 → 0.05 → 0.35 → 0.85 → 1.00, monotone, ~19 s each.

## Embodiment argument (Franka, one base pose)

Place the Franka base at world ≈ (0.25, 0.55) facing −y (behind the blue column's
spawn zone, on the +y side of the fixture): every manipulated object stays inside a
~0.75 m reach envelope and the deck itself is never in the grasp path.

- **Blue column** (grasped once): its 20 mm-square handle post rises to ~112 mm and
  fits the 80 mm parallel jaw with 60 mm to spare. The handle trails on the +y side
  while sliding in, and the config asserts the post always stays **outside the deck
  footprint** when the block is seated — the jaw never has to enter under the deck.
  Sliding in from +y is a horizontal push of a ground-supported object: the arm
  holds the post and translates, exactly the P2 controller's wrench.
- **Red column**: its handle points the **opposite** way (−y), so pulling it out
  sideways keeps the hand on the open side, moving AWAY from the fixture; blue and
  red handles can never collide during the swap. The extraction force (≲ 9 N
  horizontal) is well inside Franka payload, and the 3 mm support drop happens under
  the deck, not at the hand.
- **Parking**: the red column is carried by the same post (column mass 0.15 kg) and
  set down on the 120 mm pad with a ±45 mm tolerance; its ground-level foot bar
  keeps it free-standing on the pad (support polygon spans block-to-post).
- Tolerances are gripper-realistic: ±25/30 mm slot, ±45 mm pad, 15° uprightness.

## Execution order declared

1. `scene.py` written with a **minimal** goal predicate first; geometry honesty
   encoded as `__post_init__` assertions (rail fit, 3 mm blue clearance, blue-borne
   end inside the band and ≥ 6 mm above the spoil line, decoy-borne end ≤ spoil
   − 6 mm, post-vs-jaw fit, post outside the deck footprint).
2. `solve.py` iterated on the forge until physically solved — passed seed 0 and
   seed 3 on the first run (contact margins were designed conservatively).
3. Final rubric (latches + spoil cap) already in place; `smoke.py` rejection battery
   written and iterated on the forge: 13/14 → fixed a probe-construction bug (blue
   and deck poses must be written in the same frame after the collapse) and a
   float32 `0.1` comparison epsilon → **14/14 ALL PASS**.

## Smoke checks (14, all rejection/audit)

1. settle: states finite, deck level at ride height, red at its station (readback)
2. settle: score ≤ 0.02, no success, no spoil
3. randomization readback: fixture xy + yaw vary (deck/red/slot/band ride the frame)
4. randomization readback: blue / decoy / pad xy vary
5. null policy 360 steps: score ~0, no success, no spoil
6. SEED strategy (grasp + hoist the deck): spoil latch fires, score capped ≤ 0.10
7. seed strategy: restoring the deck perfectly does NOT un-fail (irreversible latch)
8. order violation (red first): deck collapses by physics, spoiled, extraction
   credit ~0 (ordering gate), score ≤ 0.10
9. rebuilt perfect end state after the collapse: every geometric predicate True,
   still NOT success, score capped (failure is permanent)
10. wrong object (short YELLOW decoy seated): deck sags past the spoil threshold —
    physics rejects the decoy; no seat credit
11. near-miss (blue under deck, 45 mm off the slot): safely supported, no spoil,
    NOT seated, NOT success, score ≤ 0.25
12. wrong park (red on open ground, not the pad): NOT success, score 0.85
13. rejection audit: success() never True anywhere in the battery
14. final no-NaN on all task objects

Recorded `frames.npz`: 198 frames @ 960×600.
