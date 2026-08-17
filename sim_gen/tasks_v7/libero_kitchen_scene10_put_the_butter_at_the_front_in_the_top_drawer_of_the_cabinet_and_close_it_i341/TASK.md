# shuttle_vault (i341)

**Seed:** `libero_90/libero_kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it`
(open the top drawer → put the butter at the front inside it → close the drawer).

**Env id:** `simgen.shuttle_vault` — scene-level task, `robot="null"`; all assets
procedural (raw-pxr compound spawners, no external files).

## The task

A slate-gray VAULT block is fixed to the floor: a low walled cavity whose fixed
ceiling has a hidden rectangular HATCH, covered from above by a raised HOOD (a roofed
box over the hatch, open only at its front mouth). Through the hood's mouth runs a
captive SHUTTLE — a thin sliding plate on a front-back prismatic rail, with a tall
green drag TOWER standing on its exposed front end. When the shuttle is closed the
plate seals the hatch; the plate's exposed front end is an open-sky PORCH shelf.
Inside the hood, an orange STOP BAR hangs from the roof directly over the hatch's
rear edge. The shuttle starts slightly ajar (randomized 15–35 mm). Two identical
90 mm sticks lie on the floor, shuffled per episode — a YELLOW butter and a RED clay
brick.

Goal: lay the butter flat on the porch (the exposed plate end), then drag the green
tower rearward so the plate carries the butter by friction under the hood until the
stop bar arrests the butter over the hatch — the plate keeps receding underneath it,
the support is withdrawn, and the butter gravity-drops through the hatch into the
cavity ("at the front" of the vault interior). Then drive the tower forward again so
the same plate reseals the hatch. The brick stays outside. Success = butter settled
in the cavity + shuttle fully closed + brick out + everything at rest (live physical
readback, no latched shortcut).

## Strategic difference

Vs the **seed** (pull drawer open → lower the butter in from above → push it shut):
nothing is ever opened for a carried placement — the deposit is by **SUPPORT
WITHDRAWAL**: the payload never moves toward the goal; its *floor leaves from under
it* while a fixed bar holds it in place, and gravity does the insertion through a
ceiling hatch that is roofed over by the hood, so top-down placement is impossible
(smoke check 7). And the opening/closing member IS the depositor: one prismatic
plate both delivers the butter (by receding) and seals the vault (by returning) — in
the seed the drawer is a passive receptacle and the hand does the placing.

Vs the corpus read: **i62** (rammer_gallery) drives a bladed ram that SWEEPS the
payload down a channel — here the payload is never pushed anywhere; it rides
passively and is deposited by removing its support (the moving member travels *away*
from the goal at the moment of deposit). **i307** (butter_cellar) is a vertical
gravity-press with a one-way pawl — no press, no ratchet here. **i332** (drawbridge
vault) hinges a wall down to create access — nothing rotates here and no access is
created; the hatch is permanently there, just covered. **i9** (carousel airlock)
rotates a turntable — no rotary indexing. **i3** (stopper vault) extracts a blocking
plug — here the sliding member is the goal-state closure and the delivery vehicle,
not an obstacle. **i34** (gumball meter) meters balls down a column — no gravity
feed magazine, no counting. **i237** (tilt_bin_stow) reorients the receptacle — the
vault is rigid and fixed. **i45** (domino relay) is a chain reaction — the single
mechanism here is deliberately cycled (back, then forth) with the order forced by
geometry: a closed plate covers the hatch, so nothing can enter a sealed vault
(smoke check 14), and a butter dropped while the plate is anywhere forward of the
arrest point simply rides the plate. The color-decoy brick makes identification
load-bearing (smoke check 11).

## Teleport-solution outline (solve.py; passes seeds 0 and 1)

Teleport is used ONLY to transport the free butter through open space onto the
porch; every load-bearing interaction is contact/friction/gravity physics on the
scene's own prismatic joint.

- **P0** settle + layout readback asserts (shuttle ajar in its randomized range, not
  closed, blocks outside on the floor, score ≤ 0.02).
- **P1** pose-write the butter 20 mm above the porch point (crosswise yaw) and let
  it FALL and settle flat on the plate — a contact landing, endpoint verified
  outside all geometry; boarded latch + approach ≈ 0.90 asserted.
- **P2** draw: velocity-servo the shuttle rearward (+x) by external force on its own
  prismatic rail (`F = gain·(v_des − v)` + escalating stiction floor, capped 4 N;
  0.08 m/s cruise, 0.04 m/s near arrest). The butter rides by friction, hits the
  stop bar at bx ≈ 0.19, the plate recedes underneath, and the butter drops through
  the hatch. Clear forces, settle, assert `in_vault`.
- **P3** reseal: same servo forward (−0.08 m/s, capped 3 N) until opening ≤ 4 mm;
  assert `closed & success`, score = 1.0.
- **P4** hands-off persistence 3.3 s (10 × 40 steps), success re-asserted every
  window, then `SIM_GEN_SOLVE: SUCCESS`.

(The pod's external-force frame-drag quirk is moot: the shuttle rides a prismatic
joint and never rotates, so world +x/−x stay themselves.)

Forge traces — seed 0: score 0.000 → 0.246 (porch) → 0.346 (conveyed) → 0.696
(vaulted) → 1.000 (sealed), SUCCESS in 21.2 s. Seed 1: same shape, SUCCESS in
21.4 s.

## Embodiment argument (single Franka, parallel jaw, OSC)

One plausible base pose: **(−0.10, −0.50, 0)**, facing +y toward the vault's front.
All contact points lie within ~0.25–0.50 m reach, approached from the open −y/+z
half-space, at heights 0.02–0.31 m:

- **Butter (and brick avoidance):** 90 × 45 × 45 mm stick on open floor at the two
  spawn slots (~0.30–0.45 m from base) — pinch across the 45 mm faces (fits the
  parallel jaw), lift, lay flat on the open-sky PORCH shelf at z ≈ 0.195; the porch
  is the exposed front end of the plate, outside the hood, unobstructed from above.
- **Shuttle:** hook the gripper behind the 40 mm-square green TOWER (top z ≈ 0.30,
  always outside the hood, riding open air over the vault's front block) and drag
  +x ~0.21 m at constant height — a planar pull, required force ~O(1–3 N); then
  push the same tower back −x to reseal. No regrasp needed: hook for the pull, open
  jaws and push with closed fingertips for the return.

No step needs a second arm, regrasp-in-flight, or reach under the hood: the arm
only ever touches the butter (on open floor / open porch) and the green tower.
A benign alternate route exists (fingertip-nudge the butter deeper along the plate
through the hood mouth) but it still requires the full shuttle draw-and-reseal
cycle, so no clause is bypassed.

## Execution order

`scene.py` (minimal scene, registered) → `solve.py` iterated on the forge to
SUCCESS (both seeds passed on the first submission) → final rubric (latched
partials, 0.85 non-success cap, 1.0 iff live success) → `smoke.py` rejection
battery (ALL PASS 16/16 on the first run).

## Smoke battery (16 checks)

1. settle/no-NaN: shuttle parked ajar in its randomized range, blocks outside, still
2. score ≤ 0.02, no success at reset
3. butter/brick slot assignment varies (readback, 8 seeded resets)
4. per-slot xy jitter and spawn yaw vary (readback)
5. shuttle initial opening op0 varies (readback)
6. null policy: 240 idle steps → score ≤ 0.02, no success
7. seed strategy (drop from above over the vault + push shut): butter lands ON THE
   HOOD ROOF (z ≈ 0.275), never inside → rejected ≤ 0.15
8. porch camp (laid on the porch, never drawn) → boarded credit only, ≤ 0.27
9. under-hood camp (parked on the plate at the arrest position, hatch still
   covered) → conveyed but not in the cavity, ≤ 0.36
10. vault left open (butter in cavity, shuttle at full open) → rejected ≤ 0.60
11. brick smuggled inside with everything else right → rejected at the 0.85 cap
12. shuttle ajar 25 mm with butter in the cavity → rejected ≤ 0.60
13. latched credit survives the butter being yanked back out (score holds ≥ 0.35)
14. sealed plate admits nothing: butter pressed down at 2× its weight over the
    CLOSED hatch stays at plate height (z tracked DURING the press), the plate
    holds its stop → no entry, ≤ 0.40
15. rejection audit: success() never True at any judged point
16. final no-NaN
