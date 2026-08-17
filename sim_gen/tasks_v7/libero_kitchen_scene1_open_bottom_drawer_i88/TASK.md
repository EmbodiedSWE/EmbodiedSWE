# libero_kitchen_scene1_open_bottom_drawer_i88 — open the handle-less bottom drawer with a bayonet twist-lock T-key

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene1_open_bottom_drawer` — a Franka in the
LIBERO kitchen reaches the cabinet, grasps the bottom drawer's handle, and pulls
it open (JointPosChecker on the drawer joint). The judged outcome here is the
same state — *the cabinet's bottom drawer standing open past 8 cm* — but the
handle is gone and the route to the state runs through a tool and a lock.

## Strategic difference

**vs. the seed.** In the seed the drawer front carries a handle: reach, grasp,
pull. Here the drawer front is a smooth, flush, slick 30 mm slab with a 4 mm
perimeter gap — nothing to grasp, hook, pinch, or suction (a ~23 mm fingertip
cannot enter the 16 mm-tall slot, and the slab offers no ledge). The only way
in is a **bayonet twist-lock**: fetch the free T-key from the kitchen floor,
insert its flat 46×10 mm crossbar *flat* through the drawer's keyed slot
(56×16 mm), twist ~90° so the crossbar stands vertical behind the slab
(46 mm > 16 mm — it can no longer come back out), and pull: the crossbar bears
on the slab's inner face and drags the drawer open. So the seed's single
primitive (handle pull) is replaced by a four-stage tool protocol
(fetch → insert → twist → pull), and each stage is individually necessary:
pulling an *untwisted* key just extracts it (smoke check 6), a *pre-twisted*
key cannot enter the slot (check 7), and the visually identical **decoy slot**
in the fixed panel above the drawer swallows the same protocol and yields
nothing (check 8).

**vs. the rest of the corpus.** The same-seed sibling `i87` opens this drawer
by pushing the *other* drawer through a rocker transmission — here there is one
drawer, no transmission, and the pushed-object trick is replaced by tool-use
with an orientation-gated interlock. `pull_cube_tool_i1` also pulls with a
tool, but its hook engages by pure translation; here engagement is impossible
without a *rotation about the insertion axis*, and the corpus's other keyed
task (`keyhole_unplug` i21) removes an obstruction rather than transmitting a
pull, and has no decoy. No other task in `tasks_v7/` has an orientation-gated
captive coupling or a decoy affordance. There is no stored energy: the drawer
joint is a plain prismatic with hard stops, the key is a free rigid body, and
the robot supplies every joule.

## Scene (`bayonet_drawer`, env `simgen.bayonet_drawer`, robot="null")

Procedural geometry only. A kinematic cabinet carcass (side walls, back, base,
mid shelf, top, and a **fixed decoy panel** spanning the upper cavity with an
identical 56×16 mm slot at z = 0.475) contains one dynamic drawer bound by a
per-env spawn-authored prismatic joint (axis x, limits 0→0.14 m = hard stops;
collision filtered only between the joint partners, so key↔drawer and
key↔cabinet contacts stay live):

- **Bottom drawer** (the judged body): slotted front slab (30 mm thick,
  ±0.146 m wide → 4 mm side gaps, slot center z = 0.175, 56×16 mm), floor,
  rear wall, side walls; mass 1.5 kg, slick material.
- **T-key** (free body on the floor, mass 0.10 kg): flat crossbar
  46×10 mm(y×z) at the tip, 9 mm square shaft (spins freely in the slot), and
  a 28 mm knob that can never pass the slot. Spawns at a *seeded* floor pose:
  x ∈ [0.32, 0.52], y ∈ ±0.35, uniform yaw.
- **Distractors**: a bowl and a plate on the cabinet top (z = 0.70), seeded
  x-bands + y jitter.

Geometry contract (all asserted in `__post_init__`, ~25 checks): the flat
crossbar clears the slot with ≥3 mm on each axis; the twisted crossbar is
captive with ≥10 mm of bearing overlap on the slab and stays captive down to
~15° while an untwisted key passes below ~4°; the knob never fits; the
fingertip-vs-gap no-purchase claim (23 mm > 16 mm slot, 4 mm gaps); the decoy
slot is z-discriminated from the real slot by the keyed z-window; the pull
output (key root at `key_pull_x` → drawer ≈ 0.093 m) exceeds goal + 10 mm
inside the 0.14 m stroke; twist swing clearances; distractor bands don't
overlap. Slick physics material (μ≈0.06, combine "min") on all movers so
PhysX's default ~0.5 friction cannot jam the crossbar in the slot.

**Success** (live state, no memory): `open ≥ 0.080 m` ∧ settled (drawer and
key velocities under gates) ∧ finite. **Score**: latched partial credit —
keyed 0.15 (crossbar inserted behind the slab through the *real* slot),
locked 0.20 (keyed ∧ crossbar within 45° of vertical), crack 0.25 (drawer
≥3 cm), capped at 0.60; exactly 1.0 iff success. Latches reset on `reset()`.

## Solution (`solve.py`) — one transport teleport, then applied wrench only

1. **P0 settle + perception**: 120 steps; read back the key's seeded floor
   pose and the distractor poses from the live state (never hard-coded);
   assert the key settled where sampled, drawer sealed, score ≈ 0.
2. **P1 transport** (the single permitted teleport, transport only): the key
   is moved from the floor to a free-space staging pose 9 cm in front of the
   slot, crossbar flat, touching nothing; 60-step hold proves the pose is
   stable and worth 0 (not keyed, score ≈ 0).
3. **P2 insert**: a PD wrench servo (kp 150 N/m, kd 6, clamp 20 N, gravity
   feedforward; kp-only orientation servo, clamp 0.06 N·m — a wrist holding
   the knob) drives the key −x through the slot to `key_ins_x`; the keyed
   latch fires (score ≥ 0.15) while the drawer stays sealed.
4. **P3 twist**: the orientation target rolls to 90° about the world x-axis;
   the crossbar stands vertical behind the slab (locked latch, tilt ≈ 1.00,
   score ≥ 0.35). The drawer has still not moved.
5. **P4 pull**: the position target retreats to `key_pull_x`; the captive
   crossbar bears on the slab's inner face and drags the drawer past crack
   (0.60 cap) and past the 8 cm goal (observed open ≈ +0.091 live).
6. **P5 hands-off**: forces off; the friction-held drawer settles at
   ≈ 0.095 m; `success()` turns True on the live state (score 1.0), then holds
   through a **3.3 s hands-off persistence** window before
   `SIM_GEN_SOLVE: SUCCESS`.

The whole episode is then repeated on a second seed (fresh reset, no score
prints — the SIM_GEN_SCORE stream stays non-decreasing: 0 → 0.15 → 0.35 → 1.0)
to certify seed robustness. Forge result: both seeds OK, rc=0 (~43 s).

## Franka embodiment argument

Base pose: on the floor ~0.6 m in front of the cabinet face (cabinet face at
x = 0, so base at ≈ (+0.60, 0.0, 0)), facing −x.

- **T-key (the manipulated object)**: spawns on the open floor at
  x ∈ [0.32, 0.52], y ∈ ±0.35 — between the robot base and the cabinet, prime
  tabletop-grasp territory. The 28 mm knob is a canonical parallel-jaw
  cylinder grasp. The protocol is then three single-arm primitives from one
  grasp: a straight −x insertion at slot height 0.175 m (well inside the
  workspace), a ~90° wrist roll about the tool axis (joint 7 alone can supply
  it), and a straight +x pull of ~10 cm at ≤20 N. No regrasp, no bimanual
  coordination, no wrist singularities.
- **Bottom drawer (the judged object)**: never touched directly by design —
  the embodiment requirement is precisely that no end-effector can acquire it
  (smooth slick recessed-gap front, 16 mm slot vs ~23 mm fingertip, no ledge).
- **Decoy slot**: same height band, equally reachable — reachability is not
  the discriminator; reading which slot sits in the *drawer* slab is.
- **Bowl / plate (distractors)**: on the cabinet top at 0.70 m — reachable but
  irrelevant; the task requires not touching them.

## Execution order declaration

Scene was designed first with the minimal success predicate; `solve.py` was
then run on the forge until the task was *physically* solved on two seeds
(rc=0, SUCCESS on seeds 0 and 1 — first submission); only after that were the
rubric weights anchored to the observed trajectory and `smoke.py` written. No
check was ever weakened to make a run pass — the only post-forge edits were
two `describe()` unit-formatting fixes (cm/mm multipliers), not physics or
rubric.

## Smoke battery (`smoke.py`) — 15 checks, all rejection/health

1. **settle/no-NaN** — key rests within 3 cm of its sampled floor pose,
   drawer sealed, score ≈ 0, no success.
2. **randomization A** — key xy/yaw vary across 8 seeded resets
   (xy std 0.110, yaw span 5.29 rad) and the *settled* key tracks the sampled
   pose every time (readback).
3. **randomization B** — bowl/plate xy poses vary across resets.
4. **null policy** — 240 idle steps: drawer stays sealed, score ≈ 0.
5. **front push probe** — first proves the force pathway is live (a real 15 N
   push returns a constructed-2-cm-open drawer to its stop), then proves
   pushing cannot open it. (Pulling the drawer open is N/A for an
   external-force probe: a wrench applied to the drawer body would bypass the
   very no-purchase geometry being claimed; the impossibility is geometric —
   16 mm slot vs 23 mm fingertip, 4 mm gaps, slick flush slab — and asserted
   in the cfg contract.)
6. **SEED strategy / untwisted pull rejected** — the key constructed fully
   inserted but FLAT, then a real servo pull: the crossbar passes straight
   back out (key extracted to x ≥ 0.15 — probe asserts it moved), drawer
   sealed, score ≤ 0.15 (keyed latch only). Pulling without the twist — the
   closest analogue of the seed's handle-pull — can never open the drawer.
7. **pre-twisted key rejected** — key constructed outside already rolled 90°,
   real servo push: it jams on the slab (moved ≥ 1 cm, then stopped with the
   crossbar still outside), keyed never fires, score ≈ 0. Insertion *must*
   precede the twist.
8. **decoy misuse** — the full real protocol (insert, twist, pull) executed
   on the decoy slot: `decoy_keyed` asserted hit (non-vacuous), the twisted
   crossbar is captive in the *fixed* panel (pull cannot extract it), drawer
   sealed, score ≈ 0.
9. **near miss** — drawer constructed settled at 7 cm (past crack, short of
   the 8 cm goal): capped partial credit ≤ 0.60, no success.
10. **open but moving** — drawer written at 10 cm with 0.3 m/s outward
    velocity: the settle gate refuses success on the moving state.
11. **wrong object** — bowl and plate shoved to the ground: score ≈ 0.
12. **latched credit** — the REAL servo performs insert + twist (keyed and
    locked latches from genuine physics), then the key is stolen to the
    floor: latches survive (score = 0.35 band), success does not.
13. **rejection audit** — `success()` observed False at every step of the
    entire battery.
14. **final no-NaN.**
15. **video** — frames.npz (481 rgb frames, 960×600) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, ~211 s).
