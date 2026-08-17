# pry_lid_vault — lever-pry a captive lid, then vault the ketchup

**Task dir:** `living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i414`
**Scene name:** `pry_lid_vault`
**Seed:** `libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray`

## Seed provenance and what changed

The seed is a plain tabletop pick-and-place: grasp the ketchup bottle, carry it, drop it
into an open tray. Here the "tray" has become a **chest with a captive, recessed lid**:
the lid sits flush inside a rabbeted rim (upstands on all four sides overhang the lid on
every edge), so it cannot be slid, dragged, or lifted by its faces — there is no purchase.
The only way in is a **first-class lever**: a thin steel pry bar must be inserted
horizontally through a slot in the front wall (a notch in the front rim gives the bar
angular clearance), levered over the sill fulcrum so its buried tip lifts the lid's front
edge proud of the rim, and only that pried, proud edge is graspable to remove the lid.
Then the bar must be withdrawn (a bar left in the cavity blocks success), and only then
can the ketchup be placed inside.

## Why strategically different

- **vs the seed:** the seed's receptacle is open; here the receptacle is sealed by a
  captive lid and the core skill is *tool-mediated prying*, not pick-and-place. The
  pick-and-place at the end is the trivial coda, not the task.
- **vs i158 (captive shuttle drawer):** i158 is a roofed shuttle + chimney transport
  puzzle — no tool, no lever, no lid. Here the lid is removed by a lever tool.
- **vs i207 (gravity-feed queue):** a dispensing/ordering task; no sealed receptacle,
  no prying.
- **vs i33 (flap chute) / i139 (beam balance):** chute routing and mass sensing; neither
  involves tool insertion or lever mechanics.
- **vs the memory corpus (~120 recipes):** there are crowbar-free lid tasks
  (chest-swap sliding lid, groove-lid thread, two-flap weave) and there are tool tasks
  (rake, scoop, hook-thread), but **no lever-pry**: no task where a bar through a slot
  must be levered over a fulcrum to break a lid free. `SCENES.register` grep across
  tasks_v7 confirms no pry-style scene name; `pry_lid_vault` is free.

## Scene

Procedural geometry only (compound cuboid chest, cuboid lid, cuboid bar, box bottles).

- **Chest** (~24 kg static-ish heavy body, 200×200 mm footprint): floor, four upstand
  walls forming a 160×160×?? mm cavity, rabbet shelf at z 0.060 on which the
  177×177×12 mm, 1.2 kg **lid** rests recessed; rim top at z 0.072 overhangs the lid
  edge on all four sides (lid top is flush ~6 mm below the rim top — no grasp purchase).
  Front wall carries a 24×10 mm **slot** at bar height with a notch in the front rim
  above it (angular clearance for the pry).
- **Pry bar**: 180×18×5 mm, 0.12 kg, spawns flat on the table near the chest with free yaw.
- **Bottles**: ketchup + 2 distractors (mustard, relish) on three shuffled shelf slots.
- **Randomization** (verified by readback in smoke): chest yaw ±25° and xy jitter ±4 cm,
  bottle slot permutation + per-bottle jitter/yaw, bar jitter/yaw.

## Rubric (order-gated latches)

- `_pried` (w 0.10): lid front edge proud of the rim by ≥6 mm **while a bar end is in
  the tip box under the lid** (bar-mediated — a barless tilt never fires it).
- `_lid_off` (w 0.14): lid fully off the chest footprint, flat, low, slow — gated on
  `_pried`.
- `_in_cav` (w 0.16): ketchup inside the cavity box, slow — gated on `_lid_off`.
- `success()`: all three latches AND live end state (lid off now, ketchup in cavity,
  **bar clear of the cavity**, no distractor in the cavity, everything settled/finite).
- Score capped at 0.40 unless success (then exactly 1.0).

**Ordering declaration:** pry → lid off → ketchup in. Enforced two ways: physically
(the recessed flush lid has no graspable purchase until pried; the closed lid seals the
cavity) and by the latch chain (a lid teleported off without a genuine pry earns nothing —
flagship smoke check 7 proves an end-state-identical no-pry run scores ≈0).

## Solution outline (solve.py)

- **P0** settle, mass readback asserts, lid-seated assert, score ≈ 0.
- **P1** teleport bar to the slot mouth (transport only — outside, not pried), then
  **force-drive** it horizontally through the slot with a PD-held carry + bang-bang
  forward push until the tip is deep under the lid (bar center chest-x < 0.120).
  Frame-drag probe toggles world/body force encoding; 3 attempts with re-square.
- **P2** **pry**: ramped torque about the chest +y axis (0.15→3.0 N·m) plus a small
  hold force presses the tail down; buried tip lifts the lid edge proud; `_pried`
  latches. Wrong-way and ejection guards; 3 attempts with re-insert.
- **P3** grasp-teleport the pried lid to the table (flat, clear of the footprint) —
  justified: the pried proud edge is the only purchase, and the pry is held while
  grasping. Assert `_lid_off`, not success yet.
- **P4** force-withdraw the bar back out (+x drive) until clear; assert `bar_clear`.
- **P5** teleport ketchup to a hover **above the rim** (asserted outside the in-cavity
  gate), drop; contact dynamics seat it; success fires.
- **P6** hands-off persistence 10×40 steps, `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport-only: insertion, pry, and withdrawal are force/torque-driven;
the drop-in is a real free fall; the lid grasp happens only after the physical pry earned
the credential and is the standard proxy for a gripper pinch on the proud edge.

## Embodiment argument (single Franka, parallel jaw)

Base in front of the chest (~0.4 m reach to all waypoints). (1) Grasp the flat bar
across its 18 mm width, insert horizontally at 52 mm height through the slot —
straight-line cartesian push, 3 mm/5 mm clearances. (2) Press the exposed tail down with
a fingertip: ~5 N at the tail (lever ratio ≈ 4.4 from the 8 N tip force needed for the
1.2 kg lid) — trivially within Franka payload; the rim notch gives the ~15° swing room.
(3) Pinch the pried lid edge: 12 mm thickness across an 80 mm jaw span, lift and place
aside. (4) Pull the bar back out. (5) Pick the 150 mm ketchup box and lower it into the
160 mm cavity. All five interactions are reachable from the one base pose.

## Smoke battery (14 checks)

1. settle / no-NaN / lid flush / score ≈ 0
2. randomization readback (chest yaw+xy, bar, ketchup deltas across seeds)
3. bottle slot shuffle across 10 resets
4. null policy 240 steps: lid immobile, never proud, score ≤ 0.01
5. **seed strategy fails**: bottle lowered onto the closed chest ends on the lid, not in
   the cavity, score ≤ 0.01
6. lid is captive: 10 N × 150 steps lateral pushes move it < 8 mm and never proud
   (+ positive control: 3 N moves the free bar ≥ 30 mm)
7. **flagship no-pry cheat**: lid teleported off + ketchup dropped in — live end state
   identical to success, but `_pried` never earned → not success, score ≤ 0.01
8. positive control: real ramped lever pry earns `_pried`, full pipeline → success,
   score ≥ 0.999 (snapshot taken)
9. wrong object (mustard in cavity) revokes success
10. bar left inside the cavity revokes success
11. lid balanced back on top revokes success (lid_off_now false)
12. ketchup removed → score falls to the 0.40 cap band
13. **barless tilt**: lid held proud by direct writes never fires `pried_now`/`_pried`;
    released lid reseats
14. frames.npz saved with > 10 frames

## Checks

- [x] scene.py registers `pry_lid_vault` + `register_env(..., robot="null")`
- [x] randomization real and readback-verified
- [x] rubric latched, order-gated, capped at 0.40, exactly 1.0 iff success
- [x] solve prints monotone `SIM_GEN_SCORE` lines and `SIM_GEN_SOLVE: SUCCESS` after ≥3 s hands-off
- [x] smoke prints `SIM_GEN_SMOKE: ALL PASS 14/14` and saves frames.npz
- [x] no teleport past a required interaction (insert/pry/withdraw/drop all force-driven)
