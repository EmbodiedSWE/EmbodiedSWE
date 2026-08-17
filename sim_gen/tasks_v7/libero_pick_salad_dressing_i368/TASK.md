# dressing_ferry_depot — extract the tote cart, load the dressing, ferry it home

**Env id:** `simgen.dressing_ferry_depot` · **Robot:** `null` (NullRobot; solve drives
scene-owned external-force channels + transport teleports)

## Seed provenance

Derived from **`libero/libero_pick_salad_dressing`** (RoboVerse
`roboverse_pack/tasks/libero/libero_pick_salad_dressing.py`): a Franka picks the
salad-dressing bottle standing among distractors and places it in a stationary
basket — one grasp-carry-release into an open, always-reachable receptacle, checked
by a bbox detector.

## What changed, and why it is strategically different

The seed's receptacle is passive, stationary, and open from above at all times: the
whole task is one prehensile transport of the *object*. Here the receptacle itself is
the thing that must be logistically managed:

- The delivery receptacle is a square **tote well built into a sliding CART** that
  starts **docked inside a roofed depot garage** (kinematic furniture).
- **Roof-block interlock:** inserting a cylinder (d = 50 mm, L = 165 mm) into the
  snug well (w = 58 mm) needs overhead clearance of at least `L·d/w = 142 mm` above
  the rim at *every* feasible tilt (footprint `d/cosθ ≤ w` forces
  `overhead ≥ L·cosθ ≥ L·d/w`). The roof grants only 130 mm — loading while docked is
  **geometrically impossible**, asserted in `__post_init__` and probed physically in
  smoke. At reset the goal literally cannot be worked on.
- The agent must therefore **(1) extract** the cart by its protruding handle, riding
  guide rails down the apron until the well clears the roof, **(2) load** the amber
  bottle upright into the flare-mouthed well in the open, and **(3) ferry** the loaded
  cart back through the doorway (the seated bottle clears the roof by 20 mm) until it
  hits the back stop.
- A **red decoy bottle** of identical shape (floor stations swapped per episode)
  punishes loading the wrong object; the well refuses two bottles (w < 2d).

versus the sibling variants: i151 assembles a crate *around* the bottle, i286 hangs
the bottle on overhead rails (no containment), i359-style tasks push-route an object
through static geometry. None of them move the receptacle; none have a
geometry-enforced *extract-before-load-before-return* order. The seed's strategy
(carry the bottle to the receptacle where it stands and put it down) is a smoke
negative here: set down on the docked cart's slab, it earns nothing, and the well
itself is roof-blocked.

## Scene

Procedural only. Kinematic **depot**: deck, two garage side walls, back stop, low
roof (underside 205 mm above the deck), and a rail apron (inner width 170 mm) running
out of the open doorway. Dynamic **cart** (260×160 mm slab, 0.45 kg): snug square
well (58 mm inner, 55 mm walls) toward its rear with a 45° flare mouth (capture
±18 mm), handle post + red knob at its front (protrudes 25 mm outside the doorway
when docked). Two dynamic **bottles** (body ∅50×125 mm + neck, 0.30 kg): amber
target / red decoy on floor stations either side of the apron.

Randomization (verified by smoke readback): depot xy ±30 mm + yaw ±8°, station swap
(fair draw), bottle xy ±20 mm + free yaw, cart y jitter. First post-seed draws burned.

## Rubric (latched, monotone; score in [0, 1])

| credit | stage |
|---|---|
| 0.15 | latched: cart extracted — well (incl. flare) fully clear of the roof edge, in the rail channel |
| +0.35 | latched: amber bottle seated upright in the well (cart-frame check, settled) |
| +0.35 | running-max: loaded-carry fraction of depot-x from extraction line to dock line, credited only while the bottle rides seated |
| cap 0.85 | pre-success cap |
| 1.00 | iff `success()`: amber seated + cart docked against the back stop + cart and bottle settled + decoy **not** in the well |

`success()` is pure current-state geometry; the execution order is enforced by the
roof interference, not by a latch. Null policy scores 0 (cart starts docked, latches
start False, nothing moves by itself).

## Teleport-solution phases (solve.py)

Teleport = **transport only** (one hover placement of the bottle above the flare
mouth in free air). All load-bearing interactions are contact dynamics:

1. **EXTRACT** — velocity-regulated depot-local −x CoM pull on the cart (4 N,
   ≤0.1 m/s, lateral P-steer between the rails) to the pull park; extraction latch
   fires. `SIM_GEN_SCORE 0.15`
2. **LOAD** — bottle teleported to hover above the extracted well's flare, released;
   it falls, the flare funnels it, the well squares it upright (fallback gentle 2 N
   press if perched — contact, not teleport). `SIM_GEN_SCORE 0.50`
3. **FERRY** — +x push (6 N, ≤0.06 m/s so the bottle rides, not sloshes) until the
   cart collides with the back stop = docked; settle. `SIM_GEN_SCORE 1.00`
4. **PERSIST** — 400 steps (3.3 s) hands-off; success must still hold.
   `SIM_GEN_SOLVE: SUCCESS`

Verified on the forge: seeds 0 and 1, score trajectory `[0, 0.15, 0.5, 1.0, 1.0]`,
both `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, OSC, 80 mm parallel jaw)

Plausible base at **(−0.22, 0.0)** facing the depot (farthest required contact
≈0.65 m, asserted < 0.78 m):

- **Cart extract/ferry:** the 34 mm knob on the 16 mm handle post protrudes ≥15 mm
  outside the doorway even at full dock (asserted) — a top or side pinch on the knob,
  then a straight ~0.24 m pull/push along the apron at deck height; the rails absorb
  lateral error (5 mm/side, ≤2° yaw). The solve's CoM force channel is the
  sanctioned proxy for exactly this contact.
- **Bottle load:** body ∅50 mm (or neck ∅24 mm) fits the jaw; the insertion happens
  in the open air above the extracted cart with a ±18 mm flare capture, so a
  top-down place-and-release suffices — the solve's hover-drop is the same motion
  minus the gripper.
- No contact is required under the roof at any point: loading is done outside, and
  docking is a push on the outside knob.

## Execution order (declared)

`extract → load → ferry-home`. Any other order is dead on arrival: load-first is
roof-blocked (physically probed), ferry-first is a no-op (cart starts docked), and
returning the cart empty earns only the latched 0.15 (smoke check 10).

## Checks

- **solve.py** — forge, seeds 0 and 1: `SIM_GEN_SOLVE: SUCCESS`, monotone scores.
- **smoke.py** — forge: `SIM_GEN_SMOKE: ALL PASS 11/11`:
  1. settle/no-NaN + layout readback (docked cart, stations, swap, score 0);
  2. randomization real (3-seed max-pairwise depot xy/yaw + target xy);
  3. swap coverage over 10 resets + depot-yaw spread;
  4. null policy 240 steps → score ~0;
  5. seed-strategy negative: bottle set down settled on the docked cart's slab → 0;
  6. roof-block physical probe: drop + 4 N press over the docked well rests ON the
     roof (anti-vacuity readback), never seats;
  7. undelivered load (seated, cart parked out) → 0.50, no success;
  8. dock near-miss (30 mm short) → below cap, no success;
  9. wrong object (decoy seated, docked) → score ~0, no success;
  10. latch persistence: real pull (0.15) then empty return → still 0.15;
  11. frames.npz saved (67 frames).
- **scene.py `__post_init__`** — 18 geometric/embodiment asserts (roof-block bound,
  doorway pass headroom, one-bottle well, flare capture, rail clearances, extraction
  and dock margins, handle protrusion, station-vs-apron sweep, reach).
