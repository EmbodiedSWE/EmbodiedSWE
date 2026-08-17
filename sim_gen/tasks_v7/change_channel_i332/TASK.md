# change_channel_i332 — ChannelFlipboardScene

Env: `simgen.channel_flipboard` · robot: `null` (scene-level task) · assets: fully
procedural (boxes + authored USD revolute joints, visual-only decorations).

## Seed provenance

Seed task: `rlbench/change_channel`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/change_channel.py`): a Franka picks
the hand-held **TV remote** off the table, aims it at the TV frame, and **presses one
channel button**. Strategy family: one grasp of a free-standing device, one
transport/aim, one discrete press on that device.

## What changed, and why it is strategically different

Same theme (select a TV channel), different *plan* and different *code structure*:

- **Nothing is hand-held and nothing is pressed.** The channel indicator IS the
  mechanism: a mechanical **flip-board** on the TV cabinet — four rigid colored cards
  (red/green/blue/yellow = channels 1–4), each hanging from its **own horizontal axle**
  on end pins with an open radial air band (`r_gap`), the axles stacked vertically
  (card 1 highest = outermost). The cards nest like a card tent over a ridge; the
  channel "playing" is the color of the **front slope's outermost card**. A guide tile
  on the cabinet top names the per-episode target color (symbolic cue, never touched).
- **A physically-forced execution order, both ways.** Two real interlocks, both proved
  by force probes in smoke:
  - *outer-first drag lock*: lifting an inner card presses its slab out-and-up into
    every card resting outside it and drags them — at the solve's fingertip torque cap
    it stalls far short of the apex. Only the current outermost card of a slope can
    cross.
  - *rider lock*: a card that a later-landing card has **propped on top of it** can
    never be flipped back — the lifter's root corner stays inside the rider's swept
    annulus band from ~99° up to the apex (`cos φ > −axis_dz/(2·r_gap)`), so the rider
    is carried geometrically, at any force (probed at 2.5× the solve cap).
- **A plan-level one-way board.** The rider lock makes overshoot *irreversible* once
  anything lands on the overshot card: reachable targets are exactly `t > s` (peel the
  front slope backward, outer-first) or `t = 1` (clear the whole back slope forward).
  `reset()` samples only reachable pairs; the solver must read the tile FIRST and flip
  exactly the required cards — a memorized sequence or trial-and-error overshooting
  fails.
- **The interaction is a gravity-assisted over-center flip**, repeated in a forced
  order: lift the free lower edge over the apex, release past the top, gravity drops
  the card onto the far slope (onto its joint stop, or propped on an earlier card —
  both categorical). Nothing discrete is pressed; nothing is transported.

A solver therefore needs: read a symbolic cue → plan a flip count and direction on a
one-way mechanism → a sequence of ordered over-center flips through live hinge
dynamics. The seed needs: grasp pose → aim → discrete button press. No overlap in plan
or code structure. Also distinct from the corpus packages read while building this
task: `change_channel_i249` (captive slider on a track + lock-pin interlock + color
bands — prismatic positioning, no hinges, no ordering chain), `open_oven_i7` (rotary
detent knobs), the `pen_holder` exemplar (multi-object insertion into a cup). None has
stacked hinged plates, over-center flips, or an irreversibility-driven plan.

## Execution-order declaration

1. Read the guide tile's color → target channel `t` (readback in scripts).
2. If `t > s`: flip cards `s..t−1` front → back, **ascending index** — each is the
   front slope's outermost card exactly when its turn comes.
   If `t = 1`: flip cards `1..s−1` back → front, **ascending index** — same invariant
   on the back slope.
3. Leave every card resting categorically on its slope (settled).

Order is load-bearing twice over: an inner card cannot cross at fingertip force (drag
lock, smoke check 9), and any overshoot becomes permanent the moment a later card
lands on the overshot one (rider lock, smoke check 10) — which is why `reset()` only
samples reachable `(s, t)` pairs (asserted in the solve, audited over 10 seeds in
smoke).

## Teleport-solution outline (`solve.py`)

This task has **no transport component, so the teleport budget goes unused**: no root
pose of any task object is ever written by the solution. Every load-bearing
interaction runs through the live plant (the scene's `card_drive` hinge-torque buffer,
clamped to `TAU_MAX = 0.12 N·m` — a 0.67 N fingertip push at the card's free edge,
~2× the 0.059 N·m single-card gravity peak, NOT enough to bulldoze an inner card plus
its dragged stack):

- Per required card (always the current outermost of its slope): **DRIVE** — gravity
  feed-forward + rate servo to a deliberate ~2.5 rad/s slew up the slope; **RELEASE**
  — drive zeroed 10° past the apex on the goal side, gravity + hinge viscosity drop
  the card onto the far slope; **SETTLE** — hands off until the card reads
  categorically fallen (> 60° on the dest side) and its hinge rate has died.
- Flip schedule read back from the scene per episode (`t > s`: cards `s..t−1`
  front→back ascending; `t = 1`: cards `0..s−1` back→front ascending); the sampler
  contract is asserted before any flip.
- Non-decreasing `SIM_GEN_SCORE` at each phase boundary; ≥ 3.5 simulated seconds
  hands-off persistence after `success()` first holds (live state — a knocked-off
  board would revert it); then `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka + parallel jaw, OSC)

Plausible base pose: beside the TV cabinet, facing the flip-board (cabinet top at
0.20 m, card free edges sweep 0.26–0.53 m — comfortably inside a Franka's workspace).

- **Flip:** fingertip under the outermost card's free lower edge (≥ 5 cm of clear air
  under every resting edge, audited in cfg), lift it up-and-over along its hinge arc —
  a one-finger guided push, exactly the tangential drive the solve's hinge torque
  emulates — and release past the top; gravity finishes. Repeat per card. No grasp,
  no regrasp, single-handed throughout.
- The card slabs are 10 × 12 cm rigid plates on end-pin axles between two cheek
  plates; the sweep corridor is open from the front and above.
- The guide tile is read visually (color), never touched.

## Rubric

- `success()`: every card rests categorically on its side of the target partition
  (cards left of the target channel on the back, the rest on the front, all beyond
  `side_min_deg = 30°`) **and** everything is settled (|ω| < 0.6 rad/s,
  |v| < 0.08 m/s).
- `score()`: latched flip credit — 0.55 · (required movers observed past 35° on their
  goal side / required movers) + 0.45 · success; ~0 for the null policy, exactly 1.0
  iff success, 0.55 retained if a finished board is later knocked off.

## Check list (smoke.py — 17 checks)

1. settle/no-NaN + full layout sanity (stops partition, tile in frame, spares hidden)
2. score ~0 at reset, no success
3. randomization readback: start jitter physically posed and varies, 10 seeds sane
4. randomization readback: sampler reachability contract (t > s or t = 1) on all 10
   seeds, both flip directions occur, starts/targets vary, tile matches target
5. null policy (240 steps) → score ~0
6. seed-strategy transplant: grasp+press N/A by premise; a fingertip press on the
   front card rocks it ~10° and gravity returns it — channel unchanged, no credit
7. wrong channel (all cards front = channel 1, settled, not the target) rejected
8. near-miss: the single mover 5° short of side_min on its goal side rejected
9. ordering interlock probe at the solve's own cap: inner card lifts ≥ 5° then
   stalls; the dragged outer card never leaves the front
10. rider-lock probe: constructed landed pile holds; card 1 driven back at 2.5× the
    solve cap — rider disturbed, clean extraction NEVER occurs
11. settle gate: goal partition with one mover spinning ≠ success
12. part-way cap: all movers latched, board not showing target → score == 0.55 cap
13. latched credit invariant under knocking a second mover back
14. wrong object: card 4 (never a mover) flipped back earns nothing
15. rejection audit: success() never True at any judged point
16. final no-NaN
17. camera ≥ 20 frames → frames.npz

## Verification (forge, final package)

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on seeds 0 (ch 1→2, one flip), 1 and 2
  (ch 4→1, three flips back→front — the opposite branch) and 3 (ch 2→4, two flips);
  non-decreasing `SIM_GEN_SCORE` 0 → … → 1.0 with 3.5 s hands-off persistence on
  every seed.
- `smoke.py`: `SIM_GEN_SMOKE: ALL PASS 17/17` (randomization readback honored the
  reachability contract on all 10 audit seeds with both flip directions; frames →
  frames.npz).
