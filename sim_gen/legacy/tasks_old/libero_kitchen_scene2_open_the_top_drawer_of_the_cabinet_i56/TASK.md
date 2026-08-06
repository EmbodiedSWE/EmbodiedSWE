# skating_cabinet — brace the free-standing cabinet, then pull its stiff drawer open (i56)

**Registered as:** `SCENES["skating_cabinet"]`, env `simgen.skating_cabinet` (robot="null",
scene-level). **Tier: easy — 1 composite manipulation stage** (open the drawer), with the
difficulty on the *interaction-topology* axis: the single stage needs **two simultaneous,
opposing contacts** (anchor the fixture + pull the drawer). **Execution order: N/A** — there
is no stage sequence to order; what is required is *simultaneity* (the brace must be active
while the pull happens; physics enforces it, not a code interlock).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet.py`)
- Seed plan: the wooden cabinet is an articulated asset with `fix_base_link=True` — the
  world holds the carcass for free. The solver grasps the top-drawer handle and pulls
  along the prismatic axis; a joint-position checker (`top_level <= -0.1 rad`) fires.
  One hand, one pull, nothing pushes back.

## What changed, and why it is strategically different

The goal *sounds* identical — "open the drawer" — but the seed's plan is physically
self-defeating here, because the one assumption the seed's world grants for free
(**the cabinet is anchored**) is removed:

1. **Nothing is bolted down.** The cabinet is a light (1.2 kg), free-standing dynamic
   hutch resting on slick glide feet (feet↔ground friction ~0.05, `combine="min"` so no
   ground default can rescue it). There is no articulation at all — the drawer is a
   jointless tray riding a grippy deck inside the hutch (tray↔deck friction ~1.15,
   `combine="max"`).
2. **Newton's third law is the adversary.** The stiff glides couple drawer to carcass
   with ~10.4 N of friction; the floor offers the feet only ~1.0 N. Pulling the handle
   the seed's way — at any careful speed — therefore drags the WHOLE cabinet across the
   floor: the drawer never extends *relative to its housing*. This is negative control
   A, measured on the forge: 150 mm of careful pulling yields < 40 mm of extension
   (dry-predicted ~1 mm) while the hutch is dragged > 60 mm off home.
3. **The required plan is force closure on the fixture:** brace/anchor the hutch (hold
   the carcass) with one contact while pulling the drawer with another — two opposing
   simultaneous interactions instead of the seed's single pull. The oracle does exactly
   this (per-substep kinematic pin of the hutch at home + 0.08 m/s glide of the tray).
4. **The goal is judged in the fixture's live body frame, conditioned on the fixture
   staying put:** success = drawer settled in a 100–130 mm extension band (housing
   frame) AND the hutch still at its home pose (±30 mm, ±12° yaw), upright, tray still
   seated on its deck. Opening the drawer "somewhere else" (cabinet displaced) is not
   opening the drawer.
5. **The physically real manner-cheat is measured and closed.** A violent snatch CAN
   beat the coupling by inertia (the hutch cannot accelerate fast enough): the
   dry-computed cliff is ~1.25 m/s and negative control B measures a 1.8 m/s yank
   physically reaching > 50 mm of extension. It is rejected on outcomes: extension
   banks only under a per-substep quasi-static gate (tray ≤ 0.15 m/s, hutch step ≤ 2 mm,
   hutch at home, tray seated), the un-banked crossing blocks `in_band_now`/success, and
   the yank also knocks the hutch off home. A one-write teleport trips the permanent
   `warped` latch (> 30 mm in one substep).

A different PLAN, not different parameters: the seed needs *reach → grasp → pull*; this
task needs *establish a second, opposing contact and maintain it while actuating*. The
solver must reason about reaction forces, not just the handle trajectory.

### Differentiation from sibling drawer/cabinet tasks (claimed-axis check)

- **i15 drawer_fetch_restore** (multi-stage open→fetch→deliver→restore on a kinematic
  housing): here the housing is *dynamic* — no fetch, no restore, no roof-blocks-lift
  trick as the point; the single opening act itself is the challenge.
- **i19 barred_drawer** (remove a lock bar, then open/stow/shut): nothing is locked
  here; the drawer is free to slide from step one — the *cabinet* is what's free.
- **i22 drawer_stash / i33 unjam_drawer / i35 bayonet_switch / i6 pressure_plate**
  (spring/stored-energy mechanisms): fully passive here — no springs, no applied
  forces, no self-closing, no let-the-mechanism-act.
- **i50 stemware_glide** (care/speed manner constraint with fragile cargo riding the
  closing drawer): no cargo and no fragility here; the constraint is reaction-force
  management. The quasi-static credit gate exists only to close the measured inertial
  yank hole, and the headline failure mode (cabinet drags along) happens at *any*
  speed — slower makes it worse, the opposite of a gentleness task.
- **i39 fridge_clearway** (clear objects out of a swept volume): nothing obstructs the
  sweep; **i34 compass_crate** (anti-transport stray latch on the manipulated object):
  here the home requirement is on the *fixture* and emerges from the reaction force,
  and the manipulated object is *supposed* to translate.

## Scene / physics honesty

- All procedural primitives, two compound dynamic bodies (hutch: feet slab + deck slab
  + walls + fascia + roof; tray: floor + walls + face panel + handle bar). No joints,
  no springs, no post_step forces — post_step only banks judging state.
- The coupling asymmetry is dry-computed in the cfg (`coupling_force`, `ground_resist`,
  `yank_cliff`) and *measured* by the smoke's unbraced drag-speed sweep {0.08, 0.5,
  1.8 m/s}: extension grows monotonically with speed, quasi-static pulls stay < 30 mm,
  only the yank exceeds 50 mm.
- Two physics materials on the hutch, bound per collider AFTER authoring (i33 stack
  lesson): slick feet (`min`) vs grippy deck (`max`); the tray wears the matching
  grippy material. Sleep thresholds zeroed on both bodies (kinematic writes raise no
  wake events).
- The oracle is a teleport-oracle only in HOW it acts (kinematic brace + kinematic
  glide); success()/score() judge settled physical outcomes: real extension in the live
  housing frame, real hutch pose, tray really seated, everything below settle speed,
  after all handles are released.
- Tip-out cliff (tray CoM passes the deck sill) sits at ~144 mm extension, safely
  beyond the 130 mm band top; the deck sill runs 20 mm past the fascia so an in-band
  tray is fully supported.
- Randomization: hutch xy (±35 mm) + yaw (−30° ± 50°) — pull direction, brace point and
  goal frame all move together; verified by readback. The drawer always starts
  flush-closed (the seed's start state).

## Rubric

- `score() = 0.6 · banked-extension fraction (8 mm deadband → open_min) + 0.25 ·
  in_band_now + (1.0 iff success)`; `warped` → 0.03 flat. Null policy: exactly 0.
- Banked extension latches transient achievement (a solver that opened properly and
  then over-pulled keeps its 0.6·fraction, but not the band/success credit).
- Measured milestones (oracle): 45 mm → 0.24, 85 mm → 0.50, success → 1.0 (strictly
  increasing on all 3 oracle seeds).

## Smoke checklist (14 checks)

1. reset settles finite, drawer flush-closed, hutch at home, score exactly 0
2. randomization is real (hutch xy + yaw by readback, 4 seeds)
3. null policy: no creep on slick feet, score exactly 0
4-6. oracle (brace + 0.08 m/s glide to 115 mm) reaches success() on seeds 0/1/2
7. milestone scores land in bands (0.24 / 0.50 / 1.0)
8. rubric strictly increasing on all oracle seeds
9. negative A — the seed's own strategy (careful unbraced pull): cabinet drags > 60 mm,
   extension < 40 mm, score ≤ 0.10, no success
10. negative B — 1.8 m/s inertial yank: physically extends > 50 mm but banks nothing,
    hutch displaced, score ≤ 0.10, no success
11. negative C — one-write teleport to a perfect open pose: `warped` latched, refused
12. near-miss — braced pull released 15 mm short of the band: mid score, no success
13. tolerance probe (authored poses, no stepping): 111 mm in-band vs 90 mm short vs
    145 mm over-pulled
14. calibration — unbraced drag-speed sweep across the ~1.25 m/s inertial cliff:
    extension monotone in speed, extremes asserted
