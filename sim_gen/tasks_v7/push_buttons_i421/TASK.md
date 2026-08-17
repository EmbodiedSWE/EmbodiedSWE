# push_buttons_i421 — read the slits, then drop the marbles in order (`drop_sequencer`)

## Seed provenance

Seed task: `rlbench/push_buttons`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/push_buttons.py`) — a Franka presses 3
colored buttons in a prescribed color order. In the seed, the button IS the goal object
(its color is painted on the thing you press), every press is self-latching, and a wrong
press merely wastes time — the episode is always still winnable.

## The task

A dark-grey MARBLE VAULT: a 25 cm sealed collection channel (3° floor slope down toward
an end wall painted white — the STOP) with three sealed chimney towers standing over its
raised end. Each chimney holds one colored marble (24 mm; crimson / amber / azure) parked
on an internal perch, visible only through two narrow front slits; **which marble sits in
which chimney is a fresh uniform permutation every episode**. Between the slits sits that
chimney's colorless square plunger cap: pressing it horizontally 20 mm to its hard stop
(light spring, < 2 N) shoves that chimney's marble off its perch; the marble falls the
sealed shaft into the channel and rolls down the slope to queue against the stop behind
any marbles already there. The channel is roofed, 44 mm wide (< 2 marbles: single file,
no passing), and every outer opening is smaller than a marble — a drop is **irreversible**
and queue position is purely drop chronology. Goal: the settled queue reads crimson,
amber, azure from the stop, with all plungers returned home (spring return) and
everything still, sustained. Vault pose (xy ± 10 cm, free yaw) randomized per reset.

## Why this is strategically different

- **vs the seed (`rlbench/push_buttons`)**: three inversions of the seed's plan
  structure. (1) The buttons are *colorless*; color lives on hidden payload marbles, so
  the required order must be **perceived** (read through the slits) per episode, not
  parroted from the instruction. (2) A press is not the goal but a *trigger* of a
  gravity-fed transport whose consequence must be awaited (marble must settle before the
  next drop is judged in sequence). (3) One wrong press makes the episode **permanently
  unwinnable** (smoke checks 5–6 construct exactly the seed's press-them-all plan and a
  mid-way slip, and assert the score caps at 0.10 / 0.35 forever) — the seed has no
  irreversibility at all.
- **vs `push_buttons_i273` (shutter trap)**: i273 keeps *non-latching* buttons down by
  interleaving a captive shutter under a spring-return deadline — a press-and-maintain
  race whose order is forced by geometry. Here presses are momentary and unhurried, no
  second body is transported, and the difficulty is **information (read the permutation)
  + irreversible sequencing of a FIFO queue** the agent never touches directly.
- **vs `push_buttons_i40` (counterweight scale)**: i40 is a counting / subset-sum statics
  puzzle judged at beam equilibrium; nothing here is weighed or placed — the goal state
  is an ordering inside a sealed channel produced only through the vault's mechanism.
- **vs FIFO/order siblings (`fifo_rack`, tile shuffle, marble-router recipes)**: those
  transport the payloads directly (push/carry into a rack or route junctions live); here
  the payloads are **untouchable** (sealed behind < 24 mm openings from spawn to goal) —
  the only interface to the goal state is three identical plungers plus perception.

## Solution outline (solve.py — NO teleports at all)

Scene-level env (`robot="null"`); every action is a world-frame force through the scene's
probe-force plant (frame-encoded plant-side each substep). Nothing needs transporting —
gravity does the transport.

1. Reset, settle; read the per-episode marble→chimney permutation from `scene.chim` (the
   readback the real agent gets by looking through the slits); press order =
   crimson's chimney, amber's, azure's.
2. Per marble (×3): capped velocity-servo press on that chimney's plunger cap along the
   vault press axis (gain audit 6·dt/m = 0.83 < 1, feedforward = spring load, cap
   4.5 N) to the 20 mm hard stop; hold until the marble leaves its perch (readback);
   force off (spring returns the plunger); hands off while the marble falls and rolls to
   the stop; verify the settled correct-prefix latch advanced (a wrong drop can never fix
   itself, so this is the whole ballgame).
3. Forces off, wait for the sustained-success counter, then ≥ 3.5 s hands-off
   persistence; `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Scores are monotone across phases: 0.000 → 0.350 → 0.600 → 0.850 → 1.000. Verified on
the forge for seeds 0 (permutation [1,2,0]) and 3 (permutation [0,2,1]); plungers return
to q = 0.0 and the queue settles at x ≈ 12 / 24.5 / 37 mm (staggered single file).

Rubric (latched): 0.10 any marble ever dropped; 0.35 / 0.60 / 0.85 for a settled correct
queue prefix of 1 / 2 / 3 ever achieved (a wrong marble ahead permanently blocks the
prefix); 1.0 iff success (full correct settled queue + plungers home, sustained, live).

## Embodiment argument (single Franka + parallel jaw, OSC)

Base ≈ 0.55 m in front of the vault's front face (the −y side, where all three plunger
caps and all slits face). Perception: the marble colors are visible through 8 × 28 mm
slits at 13–16 cm height from the front — a wrist-camera glance at each chimney.
Interaction: one primitive only — a closed-gripper fingertip push, horizontally, on a
26 × 26 mm plunger cap at ~14 cm height, 20 mm of travel against < 2 N of spring —
repeated three times at 6 cm lateral pitch inside a 12 × 20 cm frontal workspace. No
grasps, no regrasps, no transport, forces ≤ ~4.5 N; between presses the arm simply
retracts and waits ~1.5 s for the marble to settle (audible/visible through the low
channel viewing slot).

## Execution order (declared)

Prescribed by the goal but bound per-episode by perception: press the chimney holding
the CRIMSON marble first, then AMBER's, then AZURE's — and which chimney is which is a
fresh permutation each reset, readable only through the slits. Each drop must settle
before the next matters (the rubric's prefix latch is settle-gated). A single
out-of-order press permanently caps the episode.

## Checks (smoke.py — 13, ALL PASS on forge)

1. settle/no-NaN (perched, plungers home, score 0); 2. randomization readback (vault
xy/yaw spreads, ≥ 3 distinct permutations in 6 seeds, marble positions match the `chim`
readback every reset); 3. null policy (600 steps → nothing drops); 4. weak press
(ramped 0.5 N, probe-real ≥ 1.5 mm travel, below the ~6 mm drop threshold → no drop,
score 0); 5. **seed-strategy / wrong-first rejection** (all three chimneys really
pressed, azure's first — everything settled and home, score capped at 0.10 forever);
6. mid-way wrong drop (crimson banks 0.35, then azure out of order → latched at 0.35);
7. missing third (constructed 2-prefix → 0.60, no success); 8. correct order OUTSIDE the
vault counts nothing (0.10); 9. plunger-home gate (correct queue with one plunger held
at its stop → hold counter pinned 0 for 6 s; sanctioned release → success, 1.0);
10. roof captivity (0.5 N lift probe-real rise, marble cannot leave the channel);
11. no reordering (whole train driven 33 mm up-slope — success live-reverts, marbles
never pass (44 mm < 2·24 mm makes x-crossover geometrically impossible), released →
re-queues in the same order, success returns); 12. no accidental success; 13. final
no-NaN. Records frames.npz.

## Files

- `scene.py` — geometry class `G` (single source of truth + audited `__post_init__`
  asserts: sealed openings, FIFO width, perch wedge, plunger guide/captivity/re-entry
  clearances, spring-plant stability), compound procedural spawners (vault / plunger),
  `DropSequencerSceneCfg`, `DropSequencerScene`, `SCENES.register("drop_sequencer")`,
  env `simgen.drop_sequencer` (robot="null").
- `solve.py` — force-only solution (above), seeds 0 and 3 pass on the forge.
- `smoke.py` — 13-check rejection battery (above), 13/13 on the forge.
