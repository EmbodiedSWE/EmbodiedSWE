# fragile_pack — build the cushion first, then set the fragile orb down onto it

**Seed:** `embodiedgen/put_banana`
(`sim_gen/RoboVerse/roboverse_pack/tasks/embodiedgen/put_banana.py`)
**Tier:** medium — **2 goals in a physically forced order** (sprung cradle seated in
the crate, THEN the intact orb resting on its tray), guarded by an **irreversible
fragility foul**: the orb carries an internal core held by a breakable weld
(break_force 1.5 N), and the seed's own strategy — release the payload above the
container and let it fall in — SNAPS the weld permanently (smoke #10).
**Execution order: REQUIRED — cushion before payload.** The orb must end ON the tray
of the seated cradle, so the cradle physically has to be installed first (there is no
tray to land on otherwise), and the payload leg must be a *gentle set-down*, not a
drop: free-fall over the wall breaks the core loose and every credit latch freezes.
**Env name:** `simgen.fragile_pack` (scene `fragile_pack`, robot `null` — scene-level;
solve.py and smoke.py build this same env).

## What changed vs the seed

The seed asks for a free **aerial pick-and-place**: grasp the banana on a cluttered
table, carry it over the mug, release — bounding-box containment, and the drop itself
is harmless (the payload is inert, any landing counts).

Here the payload is made **fragile with a physical crash sensor**, and the container
starts **unfit to receive it**:

- The orb (r 22 mm shell, 50 g) contains a concentric **core (r 10 mm, 20 g) welded to
  the shell by a PhysX breakable fixed joint** (1.5 N). A hard landing stops the shell
  in ~one solver step and the core's impulse tears the weld; the core then visibly
  sags out of concentricity (readback: 0 mm → 32 mm shell-frame offset). The break is
  **permanent for the episode** — success requires `intact` (core within 8 mm of
  centre AND never broken), and every score latch is gated on the unbroken state, so a
  break freezes credit at whatever was earned before it (~0 for a straight drop —
  smoke #10, #12).
- The crate (0.20 × 0.20 m interior, 80 mm walls) has a **bare hard floor**. The safe
  receiving surface must be **built**: a separate shock cradle (base slab + grasp fin)
  carries a light tray on a real **prismatic spring suspension** (k = 130 N/m,
  c = 3 N·s/m, 30 mm stroke, authored as a USD joint with a linear drive). Smoke #6
  proves the mechanism does its job: an orb dropped 4 cm onto the tray is caught
  INTACT with 14.5 mm of visible spring compression.
- The terminal state is **not "payload in container"** but the conjunction "cradle
  seated in the crate AND intact orb settled on the tray". An intact orb parked on the
  crate floor beside the tray is rejected (smoke #8); a *broken* orb sitting perfectly
  on the tray of a perfectly seated cradle — the end-state-identical flagship — is
  rejected purely on the break history (smoke #11).

A solver therefore needs a different plan (install the cushioning fixture first, then
a controlled low-height set-down — no free drop exists) and different code structure
(a two-object assembly with a compliance-aware release, plus permanent-foul awareness
— not a single grasp/carry/release), not different numbers.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects across free space; they never do the task. Every load-bearing
interaction goes through contact dynamics or a contact-respecting kinematic hold.
Jointed pairs (cradle+tray, orb+core) are always written together, same step. Phases
(each boundary prints `SIM_GEN_SCORE`, non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (crate xy+yaw, cradle pose, orb
  spawn) + custom-spawner MassAPI audit (0.18 / 0.02 / 0.05 / 0.02 kg asserted).
  Asserts orb intact, nothing seated, baseline score ~0.
- **P1 INSTALL (kinematic lift + transport teleport + gravity seat)**: the cradle is
  pose-HELD each step (fin-grasp emulation, gravity-compensated vz = +g·dt) and lifted
  15 cm at 7 cm/s — the tray rides its own suspension the whole way, never written.
  One transport teleport parks the pair hovering 12 cm over the crate centre; a
  hold-lower at 6 cm/s brings the slab to 5 mm, then RELEASE — gravity beds it on the
  crate floor. Asserts `cradle_seated_now`.
- **P2 PACK (compliant catch — the core interaction)**: transport teleport stages
  orb+core hovering 5 cm above the tray (a state a hand trivially sets up); hold-lower
  BOTH at 6 cm/s to 15 mm above ride height — asserted NOT yet `orb_on_tray_now`
  (outside the 12 mm z-tolerance) and still intact — then RELEASED with zero velocity:
  the last stretch is a free fall onto the sprung tray and the suspension absorbs it
  through real contact (the same drop onto the bare floor from wall height breaks the
  weld — smoke #10). Asserts intact + on-tray + success().
- **P3 persistence**: 3.3 more simulated seconds hands-off; only if success() held
  prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1: phase scores monotone 0.00 → 0.34 → 0.45 →
1.00, core offset 0.0 mm throughout, `SIM_GEN_SOLVE: SUCCESS` both.

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(0.0, 0.0, 0.0)` facing +x. The crate centre sits at 0.54 m ± 3 cm,
the cradle spawns at ~0.46 m ± 3 cm, the orb at ~0.32 m ± 4 cm — all inside the
proven 0.30–0.71 m ground-level comfort envelope.

**Cradle (grasped fixture):** carries a dedicated **grasp fin** (15 mm thick, 50 mm
wide, top at 120 mm) — a canonical top pinch for an 80 mm jaw (15 mm across the pads,
full-depth engagement). The install is lift-carry-lower from a single top grasp with
no wrist reorientation; the fin top rides 4 cm above the 80 mm crate walls while the
slab is lowered in, so the fingers never enter the crate. Seating tolerance is ±35 mm
on a 0.20 m floor with the last 5 mm done by gravity after release — far above
closed-loop OSC noise. The slab (0.12 m square, diagonal 0.17 m) fits the interior at
any yaw.

**Orb (gently placed payload):** a 44 mm sphere — a direct pinch for the 80 mm jaw.
The critical skill is the release height: let go ≤ 1.5 cm above the tray (the solve's
release) and the spring catches it (4 cm is still survivable — smoke #6); drop it
over the wall and it breaks (smoke #10). Tray placement tolerance ±28 mm against an
80 mm plate with 12 mm curbs to check roll-off. The tray sits at ~53 mm, well below
the wall top, and the orb is released from above the wall plane with a straight
vertical wrist — the hand never needs to reach under anything.

**Crate:** kinematic fixture, never needs to be touched. The core is internal
(collision-filtered to its shell) and is never interacted with directly. No other
objects exist; every required contact is one the arm can make.

## Success and rubric (physical outcomes only)

`success()` iff ALL, live and settled (|v| < 5 cm/s):
- cradle SEATED in the crate: centre within 35 mm of the crate centre (xy), at ground
  rest height (±8 mm), upright within 15°;
- orb ON THE TRAY: centre within 28 mm of the tray centre (tray frame), at
  plate-top + orb-radius height (±12 mm, tray frame);
- orb INTACT: core within 8 mm of shell centre (shell frame) AND the weld never broke.

`score()` ∈ [0,1], latched each step, **every latch update gated on the unbroken
orb**: `0.25·seated (ever) + 0.20·orb-approach-to-tray (gated on seated) +
0.40·packed (seated AND on-tray ever, intact)`, capped 0.85; **1.0 iff success()**;
~0 for the null policy AND for the seed's drop-it-in plan (the break freezes all
credit). Latched credit never evaporates on regression (smoke #9).

## Randomization

Per episode (verified by sim READBACK in the smoke): crate xy ±3 cm AND yaw ±180°;
cradle spawn xy ±3 cm AND yaw ±180° (on open ground, outside the crate); orb spawn
xy ±4 cm. The orb spawns intact in every draw (readback of the core offset).

## Check list (smoke.py — rubric REJECTION battery, 14 checks; no probe reaches
success(), enforced by the audit check; intact-dependent probes strictly BEFORE the
irreversible break probes)

1. settle: reset finite, orb INTACT (core concentric), cradle + orb loose on open
   ground, nothing seated, everything still
2. settle: score ~0 at reset, no success
3. randomization readback: crate xy jitter + crate yaw really move
4. randomization readback: cradle xy + yaw and orb xy really vary AND the orb is
   intact in every draw
5. null policy: 240 idle steps → score ~0, no success
6. MECHANISM control: orb dropped 4 cm onto the sprung tray (cradle on open ground)
   is caught INTACT with ≥ 8 mm visible spring compression (14.5 mm measured) — but
   nothing installed: score ~0, no success
7. no-cushion delivery: orb set down GENTLY on the bare crate floor stays intact but
   the cradle is outside → score ~0, no success
8. near-miss: cradle seated + intact orb settled on the crate floor BESIDE the tray →
   seated credit only, NOT success, score < 0.9
9. latched credit survives regression: yanking the cradle back out of the crate
   leaves the latched score unchanged, still no success
10. SEED-STRATEGY control: the orb released above the crate free-falls onto the bare
    floor and the core weld SNAPS (0 mm → 32 mm separation asserted) → permanently
    broken, score ~0, no success
11. end-state-identical (flagship): the BROKEN orb placed on the tray of the seated
    cradle — the geometric terminal conjunction HOLDS (asserted) yet NOT success:
    only the break history distinguishes this from the goal
12. foul freeze: nothing latches after the break — score ~0 even in the fully
    assembled state
13. rejection audit: success() never True at any judged point of this battery
14. final no-NaN

N/A notes: **wrong object** — the scene has one payload and one fixture with disjoint
roles; identity pressure is carried by the fragility sensor and the assembly
conjunction instead (#7/#8/#11). The seed's literal strategy IS expressible here and
is smoke #10; the seed's terminal relation without the built cushion is smoke #7.

Video frames are recorded throughout and saved to `frames.npz` in the working directory.
