# cart_ferry — dock the transfer cart, slide the bowl aboard, ferry it home

**Task id:** `libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61`
**Env:** `simgen.cart_ferry` (scene-level, `robot="null"`)
**Files:** `scene.py` (scene + rubric), `solve.py` (zero-teleport force solution), `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet.py`).
The seed is a single pick-and-place: grasp the akita black bowl off the table and set it inside a
bbox above a wooden cabinet. Success is a static pose test on the bowl alone; the cabinet never
moves; the plate is a distractor.

What is kept from the seed: the black bowl as the cargo, a kitchen-fixture setting (a plinth/hutch
standing over a work deck), and the goal of relocating the bowl onto a designated elevated
destination surface.

## What the task is

A kinematic **fixture** stands on the floor: a work deck with a raised **plinth** at the +x end.
Set into the plinth is a roofed **alcove** — a low ledge (the "hutch shelf") holding the black bowl,
guarded by side rails, a back wall, and a roof that leaves only **1.4 cm** above the bowl's rim. The
alcove's only open face looks down the **lane**: a curbed track running toward −x, interrupted by a
full-width **slot** (an open pit through the deck) directly in front of the plinth, and ending at a
**HOME band** at the far end.

A free rigid **cart** (4 kg, with a cantilevered counter top and a handlebar) starts parked mid-lane.
Its counter top cantilevers toward +x. Only when the cart is pushed all the way up the lane and
seated against the **dock stop** does its counter top bridge the slot and meet the alcove ledge
(2.5 cm gap, 2 mm step down). The goal:

> "Dock the cart under the hutch, slide the black bowl out of the alcove onto the cart's counter
> top, then pull the cart home with the bowl riding on it."

`success()` = cart in the HOME band ∧ bowl at rest **on the cart's counter top** (cart-frame test)
∧ everything settled ∧ the transfer latch fired while docked ∧ the bowl never entered the slot.

## Strategic difference — vs the seed and vs every corpus neighbor

- **vs the seed:** the seed is one grasp-and-place of the bowl through free space. Here the bowl is
  deliberately *impossible to grasp or lift* (see embodiment), and free space over the lane is cut
  by the slot: the only route is a three-stage mechanism — reconfigure the environment (dock the
  cart) to create a temporary bridge, transfer the cargo across it by sliding, then transport the
  cargo *by moving the fixture it rests on*. The bowl itself is never carried.
- **vs i306 hatch_shelf / i5 counterweight_shelf:** those gate access to a shelf (hatch, counterweight);
  the destination is static and the object is placed by transport. Here the destination itself is a
  *vehicle* that must first be positioned and then driven; nothing is ever placed by carrying.
- **vs i53 slab_easel / i14 chute_switch / i48 cask_weight_sort / i17 skittle_gallery:** those are
  orientation/routing/sorting puzzles about *which* state to produce; this is a sequenced logistics
  chain where each stage physically enables the next (undocked cart ⇒ the bowl falls in the slot —
  smoke check B proves it).
- **vs i3 tunnel_shuttle:** the shuttle is a passive slider the object is pushed through; the cart
  here is the *goal-bearing* body — success is a conjunction over cart pose AND cargo-on-cart in the
  cart's own frame, and the final phase moves cart+cargo together on friction alone.
- **vs i57 hanoi_rings / i27 bell-herd:** no stacking-order or multi-agent herding component.
- **vs robobench suite (balance_scale, combination_safe, syringe, pen_holder):** no articulated
  mechanism, dial, or insertion; the "mechanism" is emergent from rigid-body contact (bridging,
  friction ride-along).

No task in the read corpus transports cargo by riding a repositioned fixture, and none has a
permanent void hazard that the docking stage exists to bridge.

## Rubric (latched credit, `score()`)

| latch | condition (fixture/cart frame) | weight |
|---|---|---|
| `_docked_ever` | cart seated in dock band, upright | 0.20 |
| `_transferred_ever` | bowl at rest on counter top while cart docked/forward | 0.30 |
| `_ride_ever` | bowl aboard while cart crosses mid-lane (x < ride_x) | 0.15 |
| `_home_ever` | bowl aboard ∧ cart in HOME band | 0.35 |
| `_voided_ever` | bowl ever inside the slot volume | caps score at 0.20, kills success permanently |

`score = Σ weights`, capped at 0.95 unless `success()`, which returns exactly 1.0. Latches are
evaluated in `post_step` every physics substep, so credit cannot be gamed by transient poses, and
the printed `SIM_GEN_SCORE` sequence is non-decreasing by construction.

## Solution (`solve.py`) — ZERO teleports

All interaction is pulsed external force + contact dynamics (the travel and slide ARE the task, so
teleporting would skip the mechanism):

1. **DOCK** — pulsed 25 N push on the cart along the lane's +x (force on only below 0.15 m/s), so it
   creeps up the lane and the dock stop takes a soft seating impact. Release, settle. → 0.200
2. **TRANSFER** — pulsed 1.2 N push on the bowl along −x (cap 0.08 m/s): it slides under the roof,
   out the open face, across the 2.5 cm gap and 2 mm step onto the docked counter top, driven until
   it sits ~11 cm inboard of the plate's front edge. Release, settle. → 0.500
3. **FERRY** — pulsed 20 N pull on the cart along −x (cap 0.08 m/s; net accel ≈ 2.1 m/s² ≪ μg ≈ 3.7,
   so the bowl rides on friction alone, cart-frame position unchanged) until the cart passes the
   home line. Release, settle. → 1.000
4. **PERSISTENCE** — ≥ 3.5 simulated seconds fully hands-off; `success()` must still hold before
   `SIM_GEN_SOLVE: SUCCESS` is printed.

Because the bowl spawns with random yaw and the pod's external-force API may interpret wrenches in
the body's current frame, `drive()` probes force encoding at runtime from measured progress and
toggles between raw-world and `quat_apply_inverse(q_now, f)` if the body stalls.

Verified on the forge: seeds 0, 1, 2 all print 0.000 → 0.200 → 0.500 → 1.000 →
`SIM_GEN_SOLVE: SUCCESS`, rc = 0.

## Embodiment argument (single Franka, 8 cm parallel jaw, OSC)

The same plan executes with one arm, and **no stage admits a shortcut by grasping the bowl**:

- **Bowl cannot be grasped or lifted.** Outer diameter 10.4 cm > 8 cm jaw span, so no side grasp
  anywhere. Inside the alcove the roof leaves 1.4 cm above the rim — too little for a fingertip to
  hook the rim from above, and the side rails + back wall block wrap-around approaches. The bowl can
  only be *slid* along the ledge, exactly what the solution does: reach the hand in level along the
  open face, place one fingertip behind the bowl's far wall, and drag it out and across onto the
  counter top (a planar drag, ≈1–2 N, well within fingertip force).
- **Dock stage** = palm-push: flat hand against the cart base's −x face, push down the lane; the
  dock stop terminates the motion, so no precise servoing is needed.
- **Ferry stage** = pull the handlebar: the Y-axis cylinder bar (r 1.2 cm, at z ≈ 0.55 m) is exactly
  a power-grasp feature; pull gently toward home — the friction budget (accel < 3.7 m/s²) is
  generous at arm speeds.
- **One plausible base pose:** base at ≈ (fixture x = −0.35 m, y = −0.55 m), facing the lane. From
  there the arm reaches the cart faces mid-lane (±0.3 m), the handlebar, and the alcove's open face
  at x ≈ 0.05 m, z ≈ 0.42 m — all within ~0.8 m reach, and the curbs keep the lane clear of the base.

## Execution order (declared)

Strictly sequential; each stage gates the next physically, not just in the rubric:

1. Dock the cart (without it, the transfer slide drops the bowl into the slot — permanent failure,
   proven by smoke negative B).
2. Transfer the bowl (the `_transferred_ever` latch only fires while the cart is docked/forward;
   dock-band credit cannot be earned retroactively by placing the bowl first).
3. Ferry home (home credit requires the bowl aboard; driving the empty cart home scores 0).
4. Hands-off persistence.

## Checks (`smoke.py`) — 16

1. reset settles, no NaN; 2. no latch fires at reset; 3. randomization READBACK (fixture xy/yaw,
cart x, bowl x/yaw spreads across seeds); 4. null policy ≈ 0; 5–7. oracle passes on seeds 0/1/2
(score 1.0, persists); 8. monotone phase ladder 0 < 0.20 < 0.50 < 0.65 < 1.0; 9. partials < 1.0
without success; 10. void negative (bowl in slot ⇒ score ≤ 0.20, no success); 11. void permanence
(recovering the bowl aboard afterwards still fails); 12. undocked transfer is physically impossible
(bowl falls in the slot); 13. overhanging bowl on the plate edge does not count aboard; 14. inverted
bowl on the plate does not count; 15. dock+transfer but never ferried = exactly 0.50, no success;
16. aboard-band calibration drop sweep. Frames recorded to `frames.npz`.
