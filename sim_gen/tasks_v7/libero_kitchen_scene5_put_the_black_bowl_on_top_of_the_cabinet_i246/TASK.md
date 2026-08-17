# crown_socket — evict the red plug from the rooftop socket, then seat the black bowl in it

**Task id:** `libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i246`
**Scene:** `crown_socket` (`register_env("simgen", ...)` → `simgen.crown_socket`, robot="null")

## Seed provenance

`libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet.py`).
Seed strategy: pick the akita black bowl off the table, place it on the white
cabinet's FIXED flat top; success = bowl inside a bbox above the cabinet. One
pick-and-place onto an empty, always-available surface.

## What changed, and why it is strategically different

Two independent inversions of the seed, with NO joints and NO mechanism anywhere —
the interlock is pure rigid-body occupancy and shedding geometry:

1. **The top no longer accepts placement.** The cabinet's whole top is a slick
   pitched GABLE ROOF (22°, μ≈0.05 sheet; pair-averaged friction angle ~10° ≪ pitch).
   The seed's entire plan — carry the bowl up and set it down on top — is a live
   trap: anything released on the roof slides over the eave onto the floor (smoke
   checks 6-7 construct exactly this, on the slope and astride the ridge). The only
   sanctioned rest is a sunken 10×10 cm SOCKET well at the roof's crown, enclosed by
   a raised rim tower.
2. **The destination starts FULL.** At reset a red cylindrical PLUG (9.2 cm body,
   4 mm/side well clearance, parallel-jaw knob on top) is seated in the socket.
   While it is there, the bowl volumetrically cannot enter the well — it can only
   stack on the plug's knob, which the seated z band rejects. The winning plan is
   EVICTION + SUBSTITUTION: extract the plug vertically by its knob (a contact
   interaction the seed never asks for), discard it, and only then drop the bowl
   through the rim aperture onto the well floor.

Versus the sibling tasks from this seed: i176 is a counterweight/torque interlock on
a see-saw platform, i306 is over-centre hinged-lid actuation + decoy discrimination —
both actuate a jointed mechanism to build/tilt a surface. Here there is no joint at
all: the blocker is a free rigid body that must be REMOVED, the "surface" always
exists but is occupied, and ordering is enforced by volume exclusion. Different code
structure too: per-reset re-posed kinematic cabinet frame (8 boxes, xy+yaw), a
custom compound dynamic spawner (plug), and a state-based (non-latched) occupancy
rubric that honestly revokes credit if the socket is re-occupied.

## Teleport-solution outline (solve.py)

- P0: reset (seed via `env.reset(seed=...)` after build), settle; assert plug seated
  (socket occupied), bowl standing on the floor, score ~0.
- P1 (contact dynamics — the core interaction): regulated vertical pull at the plug
  (gravity feedforward + velocity servo at 0.25 m/s, ≤ 12 N) slides the plug up
  through the well's 4 mm/side clearance and past the rim entirely under force
  (asserted: rose ≥ 5 cm under the pull, socket no longer occupied, BEFORE any pose
  write). Then ONE pose write transports the plug — hovering in free air above the
  rim at that instant — to an empty patch of floor (discard; transport only).
- P2 (transport ONLY): one pose write moves the bowl across free space to a hover
  centred over the vacated socket, bowl bottom 1 cm ABOVE the rim — above the seated
  z band, satisfying nothing (asserted NOT seated, NOT success).
- P3 (contact dynamics): the bowl free-falls through the rim aperture and settles
  upright on the well floor; admission is physically possible only because the plug
  is out.
- P4: hands-off persistence ≥ 3.3 simulated s, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing
(vacated and seated are persistent physical states; nothing in the solve re-occupies
the socket). Passes seeds 0 and 1 on the forge (see run logs).

## Embodiment argument (single Franka, parallel jaw, OSC; base at the origin)

- Plausible base pose: (0, 0, 0), facing +x. Cabinet frame centre at x ≈ 0.55 m
  (±2.5 cm, ±10° yaw); every required contact lies in x 0.43-0.62 m, z 0.0-0.52 m —
  inside a comfortable Franka envelope for tabletop-height work.
- Plug (must move): knob pinch. The grasp knob is a 3.2 cm dia × 3.5 cm cylinder
  (well inside the 8 cm jaw span) standing proud at 48.5-52 cm height, in open air
  above the rim — a standard top-down pinch. Extraction is a straight vertical lift
  of 7-8 cm (the well guides the 4 mm/side clearance; no lateral precision beyond
  the ±2 cm knob-centring needed to close the jaw around a 3.2 cm post). Discard:
  release anywhere over open floor.
- Bowl (must move): body pinch. A 6.4 cm dia × 5 cm squat cylinder standing free on
  the open floor at (≈0.21, ±0.27) — top-down or side pinch, full hand clearance.
  Seat target: hold the bowl centred over the 10×10 cm well (bowl needs ±1.8 cm
  centring against a 2.2 cm rubric tolerance — the well walls themselves funnel the
  last few mm during the 5 cm drop) and release above the rim; gravity finishes.
- Cabinet: never touched; all its parts are kinematic scenery.
- Clearances: both grasps happen in open air (the knob is the highest point of the
  assembly; the bowl stands ≥ 8 mm clear of the carcass by construction). The
  22° roof falls away from the rim tower, so the wrist approaches the socket from
  straight above without any overhang nearby.

## Execution order

Ordering is FORCED BY PHYSICS, not declared: while the plug fills the well there is
physically no room for the bowl (it can only stack on the plug's knob, which scores
nothing and is unstable), and the roof sheds any bowl parked "for later" on top.
Only extract-then-seat reaches the goal. No additional artificial ordering.

## Check list (smoke.py — rejection only; solve.py is the acceptance evidence)

1. settle/no-NaN: plug seated on the well floor, bowl standing on the floor, all
   states finite, everything still.
2. score ~0 at reset, no success.
3. randomization readback: cabinet frame really moves — seated-plug world xy spread
   \> 1.5 cm and rim-wall yaw spread > 5° across 8 seeded resets (sim-state readback).
4. randomization readback: bowl spawn side swaps and xy jitter is real.
5. null policy: 240 idle steps → score ≤ 0.02, no success.
6. SEED STRATEGY: bowl set down on top of the cabinet (roof slope) is SHED over the
   eave onto the floor → no success, score ≤ 0.02.
7. ridge perch: bowl balanced astride the ridge line sheds down a slope → the roof
   holds nothing anywhere.
8. occupied socket admits nothing: bowl dropped dead-centre over the SEATED plug
   stacks on the knob or tumbles off → plug still occupies, nothing seats, score ≤ 0.02.
9. partial extraction: plug raised 2 cm but still inside the well still OCCUPIES →
   vacated stays False, score ≤ 0.02.
10. vacated grading: plug discarded to the floor → score = w_vacated (in [0.35,
    0.45]) only, no success.
11. re-occupied socket: plug dropped back into the well re-seats → vacated credit
    honestly REVOKED (state-based, not latched), score ≤ 0.02.
12. wrong orientation: bowl lying on its SIDE in the vacated well — inside the xy
    tol and the seated z band but not upright → rejected, no seated credit.
13. rejection audit: success() never True anywhere in the battery.
14. final no-NaN.

## Rubric

score = 0.40·socket-vacated (plug outside the occupancy band — state-based, revocable)
      + 0.30·bowl-seated (root within ±2.2 cm of the socket axis, z in the seated
        band 0.450-0.478 — rejects hovers, rim rests, and stacking on the plug —
        and upright)
      exactly 1.0 iff success() holds live.
success = bowl seated in the well AND at rest. Null policy = 0 (the plug occupies
the socket from t=0 and the bowl starts on the floor). Credit persists under correct
behavior because the graded states persist; it honestly disappears if the socket is
re-occupied or the bowl leaves the well.
