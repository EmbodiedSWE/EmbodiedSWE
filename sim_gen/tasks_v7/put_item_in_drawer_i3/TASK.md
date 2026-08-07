# stopper_vault — push the cube into the always-open vault, then SEAL it with the blue stopper

**Seed:** `rlbench/put_item_in_drawer`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_item_in_drawer.py`)
**Tier:** medium-hard — two ordered contact interactions (ground-level push-through +
a fitted stopper seating), a physically forced execution order, and a wrong-part
decoy clause.
**Execution order: REQUIRED — and physically forced.** The cube must be deposited
BEFORE the stopper is seated: an accepted seated stopper leaves side gaps of 10 mm
and a header gap of 14 mm around the doorway — the 50 mm cube can never pass it
(demonstrated by smoke check #9, which force-pushes the cube at a sealed doorway for
2.5 s and it stays outside). The rubric's seal term is additionally credit-gated on
the cube being inside, so seating first also earns nothing.
**Env name:** `simgen.stopper_vault` (scene `stopper_vault`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed is an OPEN-THEN-INSERT task: grasp the drawer handle, pull the prismatic
joint out, then pick the item and lower it into the opened drawer from above. The
receptacle starts closed, is opened by an articulation, and the deposit is a vertical
drop into an exposed volume.

Here the access model is **inverted and the closing move is promoted to a second,
harder insertion of a separate part**:

- The vault is **jointless and its doorway is ALWAYS open** — there is no handle, no
  articulation, and no opening move at all. A drawer-puller arriving here finds
  nothing to pull. The interior is fully roofed, so the seed's lower-from-above
  deposit is impossible: the only deposit is a horizontal, ground-level PUSH-THROUGH
  of the cube along the doorway tunnel (the ground is the sill).
- What replaces the seed's (trivial, optional) push-the-drawer-shut is a **fitted
  closure with a loose second body**: the doorway must be sealed by fetching the wide
  BLUE stopper, orienting it (upright, slab square to the doorway, grip boss
  outward), and sliding it into the doorway pocket until two inner stop ribs arrest
  it flush. The closure is itself an insertion with its own tolerance and its own
  failure modes (shy of the ribs, yawed slab, tipped slab — all smoke-tested).
- **Wrong-part identification** absent in the seed: a RED decoy stopper of the same
  shape family but visibly too narrow to cover the doorway spawns next to the blue
  one (slot order randomized); success requires it left clear of the doorway and
  interior.
- **Execution order is physically forced by the mechanism** (seat first → the cube
  can never enter), where the seed's order (open before insert) is merely procedural.

A solver therefore needs a different plan (no opening action exists; push-through a
fixed aperture, then a part-selection + fitted-seating closure, in a forced order)
and different code structure (ground-level push control through a tunnel + a
peg-in-pocket seating against ribs, not an open/pick/lower/close pipeline), not
different numbers.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects across free space; they never do the task. Both load-bearing
interactions go through contact dynamics. Phases (each boundary prints
`SIM_GEN_SCORE`, non-decreasing — all credit is latched):

- **P0** seeded reset, settle 0.5 s, layout readback (vault pose/yaw, all three
  bodies, normalizers — randomization provable from stdout); asserts score ≤ 0.02,
  no success.
- **P1 TRANSPORT (teleport, cube)**: one pose write carries the cube to the doorway
  axis ~10 cm OUTSIDE the mouth, on the ground, square to the vault — asserted not
  inside; only the 0.15-weight approach term moves, as any real carry would.
- **P2 PUSH-THROUGH (contact dynamics)**: a horizontal world-frame force at the
  cube's CoM (velocity-regulated bang-bang at 10 cm/s, stall escalation 3→8 N, small
  lateral centring force) drives the cube along the ground through the 104 mm
  doorway and past the 80 mm rib exit until its centre passes vault-x 0.068 — the
  depth a Franka fingertip reaching ~42 mm through the aperture leaves it at, below
  the 0.075 containment threshold. No pose write crosses the doorway; the aperture,
  tunnel walls and ground do the constraining.
- **P3 TRANSPORT (teleport, blue stopper)**: one pose write stages the stopper
  upright on the doorway axis, slab facing the tunnel, inner face ~5 mm outside the
  mouth — asserted outside the seat band.
- **P4 SEAT (contact dynamics)**: the same force scheme (6 cm/s, 2.5→6 N) slides the
  stopper into the pocket until the stop ribs ARREST it — the final pose is decided
  by rib contact, not by any write. Asserts seated; success() first turns True here.
- **P5 persistence**: ≥ 3.3 more simulated seconds hands-off; only if success()
  still holds prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge: seeds 0 and 1, both SUCCESS, monotone scores
(0 → 0.10 → 0.45 → 0.72 → 1.0).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(-0.20, 0.0, 0.0)` facing +x (vault interior centre at world
(0.50, 0)). Working radii from that base: cube spawn 0.39–0.54 m, stopper spawns
0.36–0.56 m, doorway mouth 0.565 m, deepest required fingertip contact 0.61 m, boss
face of the seated stopper 0.61 m — all inside the proven 0.45–0.71 m comfort
envelope (the near bound only brushes floor-level pickups, which are routine).

**Green cube (deposit):** 50 mm, 120 g. Grasp: top-down, jaw across two faces (80 mm
opening vs 50 mm — 15 mm per side). Carry to the doorway axis, set down at the
mouth, then PUSH: the aperture is 104 × 70 mm; the closed fingertip pair (~25 mm
wide, ~20 mm tall) follows the cube's rear face through it. Required insertion
depth: cube rear face 40 mm past the mouth (fingertip depth ≈ 42 mm < the ~54 mm
finger length, hand staying outside the 70 mm-tall aperture). Lateral tolerance for
the cube in the 104 mm doorway: 27 mm per side; the 80 mm rib exit still leaves
15 mm per side — an order of magnitude above OSC noise.

**Blue stopper (closure):** slab 84 × 40 × 56 mm + grip boss 32 × 30 × 24 mm, 180 g.
Grasp: jaw across the boss's 30 mm width (or across the 40 mm slab depth), top-down
at its spawn. Carry upright to the mouth, set down square, then push on the BOSS
FACE — which stays ≥ 48 mm OUTSIDE the mouth even fully seated, so the hand never
enters the aperture and the stopper is never captive (pull the boss to reopen).
Alignment budget: 10 mm per side lateral, ~±7° yaw to enter; the tunnel walls guide
the final 24 mm and the ribs decide the stop. Seat tolerances (12 mm depth, 15 mm
lateral, 12° tilt, 20° yaw) are all looser than the physical channel, so the rubric
cannot demand more precision than the geometry itself enforces.

**Red decoy:** never needs to be touched. **Vault:** kinematic fixture, never needs
to be touched (brushing it is harmless).

Every contact the task requires — two top-down grasps in the open, one fingertip
push through a 104 × 70 mm aperture to 42 mm depth, one boss push that stays outside
the vault — is one the arm can make.

## Success and rubric (physical outcomes only)

`success()` iff, live and settled (cube and stopper |v| < 5 cm/s):
- cube centre in the vault frame: x < 0.075 (rear face fully past the front wall's
  inner face), |y| < 0.078, 0.005 < z < 0.120 (inside, on the floor, under the roof);
- BLUE stopper seated: slab centre within 12 mm of the rib-arrest depth, |y| <
  15 mm, standing on the sill (z within 10 mm), upright within ~12°, slab square to
  the doorway within ~20°;
- RED decoy clear of the doorway, tunnel and interior.

`score()` ∈ [0,1], latched every physics substep:
`0.15·approach (cube progress toward the mouth, normalized by the episode's own
spawn distance — exactly 0 for the null policy) + 0.30·deposited (cube ever fully
inside) + 0.30·seal (stopper progress toward the seat, normalized by its own spawn
distance, counted ONLY while the cube is currently inside — sealing an empty vault
or seating first earns nothing)`, capped 0.85; **1.0 iff success()**. Latched credit
never evaporates (smoke #14); the solve's phase prints are monotone.

## Randomization

Per episode (verified by sim READBACK in the smoke): vault yaw ±8° + xy ±2 cm; the
cube band and the stopper band SWAP SIDES of the doorway axis 50/50; blue/red swap
their two slots 50/50; every body gets band-uniform xy and free yaw.

## Check list (smoke.py — rubric REJECTION battery, 16 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite, cube + both stoppers standing outside, everything still
2. settle: score ~0 at reset (≤ 0.02), no success
3. randomization readback: side swap flips AND blue/red slot order flips across seeds
4. randomization readback: vault yaw + xy and cube xy + yaw all vary
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: cube dropped from above settles ON THE ROOF (the
   lower-into-receptacle end state) → not inside, no success, score < 0.3
7. near-miss deposit: cube settled in the doorway, mostly through but not fully past
   the front wall → no success, score ≤ 0.25
8. out-of-order / empty seal: BLUE seated (verified) with the cube outside → no
   success, order-gated seal term ≈ 0 (score ≤ 0.05)
9. order physically forced: cube force-pushed at the sealed doorway for 2.5 s stays
   OUTSIDE → no success (the interlock is real, not just declared)
10. wrong stopper: RED decoy standing in the doorway → no success, score ≤ 0.2
11. decoy-inside clause: cube inside + BLUE seated + RED inside → the red-clear
    clause alone rejects, no success, score ≤ 0.85
12. near-miss seat: cube inside, BLUE 20 mm shy of the seat depth → no success
13. misaligned seat: cube inside, BLUE at seat depth yawed 90° (doorway not
    covered) → axis clause rejects, no success
14. latched credit: deposit credit earned, cube yanked back outside → score holds,
    no success
15. rejection audit: success() never True at any judged point of this battery
16. final no-NaN

N/A notes: the seed's exact end state (item resting in an opened drawer) is not
expressible in a scene whose receptacle has no openable state; its nearest analogue —
the from-above placement a roofed box affords — is check #6.

Video frames are recorded throughout and saved to `frames.npz` in the working
directory.
