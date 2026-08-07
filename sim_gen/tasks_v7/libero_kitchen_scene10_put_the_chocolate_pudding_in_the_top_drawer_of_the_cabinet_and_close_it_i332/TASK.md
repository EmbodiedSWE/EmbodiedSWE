# drawbridge_vault (i332)

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/...py`).
- Seed strategy: pick the chocolate pudding off the table, pull the top PRISMATIC
  drawer open (or use it already-open), DROP the pudding into the exposed cavity FROM
  ABOVE, then push the drawer shut. Success = pudding inside the drawer region AND
  drawer joint near zero. Distractors: a bowl and two butters.

## What changed, and why it is strategically different

The "store an item and close the container" goal survives; every element of the
seed's PLAN is invalidated and replaced:

1. **No top access, ever.** The receptacle is a raised, fully ROOFED chamber (a
   "keep") on a 10 cm pedestal. The seed's core move — hover above the cavity and
   release — lands the box on the roof and earns nothing (smoke check 6). The only
   aperture is a side doorway.
2. **Entry is a tangential push up an incline, not a vertical drop.** The doorway
   sits at the top of a drawbridge that starts LOWERED as a ~20 deg ramp to the
   table. The box must be slid/pushed up the ramp, over the sill, across the porch,
   through the doorway, and DEEP onto the chamber floor — a sustained contact-rich
   push with a defined 0.75 friction budget, where the seed needs one grasp-and-
   release.
3. **Closure is a gravity-bistable ROTATION, not a prismatic slide.** The bridge must
   be rotated ~118 deg about its sill hinge, PAST vertical, where gravity pins it
   against its +98 deg joint stop. There is no stable "almost closed": an
   under-rotated bridge falls back open (smoke check 10). The seed's drawer stays
   wherever it is left.
4. **Execution order is forced by physics** (see below); the seed's two goal clauses
   are order-free in principle (an already-closed drawer can be reopened).
5. **Discrimination is color-only on identical shapes**: a WHITE decoy cube of the
   same size must stay out; the seed's distractors are visually distinct objects.

A solver therefore needs a different plan (approach the ramp, push uphill through an
aperture, then re-grip and rotate a door past vertical) and different code structure
(slope push control + hinge rotation control, no drawer-joint pulling, no
place-from-above).

## Required execution order

**Yes: enter first, close second.** The shut bridge seals the only entry (plate
15 cm wide in a 16 cm doorway; remaining gaps are far smaller than the 6 cm box).
Closing with the box left straddling the doorway does not produce success either:
the swing stalls on the box or merely wedges it near the doorway plane, never deep
inside (smoke check 11 constructs exactly this with the solve's own capped torque
and asserts no success). Closing an EMPTY vault earns zero credit (close credit is
gated on the inside latch) and locks the solver out.

## Teleport-solution outline (solve.py)

- P0: reset, settle; assert bridge rests lowered (~-20 deg), boxes on table, score ~0.
- P1 TRANSPORT (the only teleport of the target): one pose write moves the BROWN box
  from its table slot to a hover 8 mm above the lower-middle of the lowered ramp —
  free space, ~25 cm outside the deep-inside band. Gravity lands it; friction holds
  it on the slope. The decoy is never touched.
- P2 ENTRY (contact dynamics): a velocity-servoed force (0.22 m/s target, capped at
  2.0 N) pushes the box up the ramp, over the sill, through the doorway; the force is
  CUT the moment the centre crosses the deep-inside threshold and the box slides to
  rest under friction alone.
- P3 CLOSURE (joint/contact dynamics): a velocity-servoed torque about the hinge
  (capped 0.8 N*m, ~3x the static gravity torque) raises the bridge from its table
  rest; torque is CUT at 93 deg, just past vertical — gravity alone carries it onto
  the +98 deg stop and pins it shut, with the box already inside.
- P4 PERSISTENCE: >= 3.3 simulated seconds hands-off; success() must still hold.
- `SIM_GEN_SCORE` printed at every phase boundary (latched rubric — non-decreasing),
  then `SIM_GEN_SOLVE: SUCCESS`. Verified on seeds 0 and 1.

## Embodiment argument (single Franka, parallel jaw, OSC)

Plausible base pose: arm base at the world origin, facing +x; everything task-
relevant lies at x 0.22-0.55, |y| <= 0.25, z 0-0.30 — inside the 0.45-0.71 m
comfortable envelope measured in prior tasks of this family.

- **Brown pudding box (6 cm cube, 0.12 kg)**: top or side pinch grasp anywhere on the
  open table (jaw opens 8 cm > 6 cm), free approach from above; place it on the lower
  ramp, then push it up the slope with closed fingertips against the box's downhill
  face — the push line is 10-16 cm above the table on a 20 deg slope, no overhang
  above it while the bridge is lowered. Push force needed ≈ mg(sin20 + 0.75 cos20)
  ≈ 1.2 N, far under arm capability. The doorway (16 x 11 cm) clears the 6 cm box
  with >= 2.5 cm on every side; the fingers only ever need to reach the doorway
  plane, not deep inside (final centimetres are a shove; deep-inside band starts
  5.5 cm past the doorway and the box can be released at the threshold with residual
  velocity or nudged with an extended-finger push through the 11 cm-tall opening —
  fingertip depth ~6 cm past the plane at most, hand outside).
- **Drawbridge (0.20 kg plate, brass ridge at the free edge)**: the 1.8 cm ridge is a
  pinchable bar (jaw span fits over it with the palm above the plate); lift-rotate by
  the ridge from table rest through ~90 deg — required grip force is small (plate
  weight 2 N, lever mostly carried by the hinge), and past ~93 deg the plate falls
  onto its stop by itself, so the hand releases before the plate leans into the
  chamber wall region. Alternative contact: hook fingertips under the free edge
  (1 cm plate thickness off the table at the ridge) or push the ridge with closed
  fingers along the rising arc.
- **White decoy**: never needs to be touched.
- Precision budget: deep-inside band is 10.5 cm long x 11 cm wide for a 6 cm box;
  closed band is >= 92 deg with the stop at 98 and gravity finishing the last 5 deg —
  both are far coarser than OSC control noise.

## Check list (smoke.py — rejection battery; solve.py is the acceptance evidence)

1. settle/no-NaN: bridge rests lowered (-24..-15 deg), boxes standing, all still.
2. score ~0 at reset, no success.
3. randomization readback: brown/white slot assignment flips across seeds.
4. randomization readback: per-box xy jitter + free yaw vary.
5. null policy: 240 idle steps -> score <= 0.02, no success.
6. seed strategy (drop from above): lands ON THE ROOF -> rejected.
7. doorway near-miss: through the doorway but short of the deep band -> rejected.
8. inside-but-open: deep inside, bridge lowered -> inside credit only, no success.
9. empty-close gating: bridge shut on its stop, nothing inside -> close_max stays 0,
   score <= 0.05, no success.
10. bistable near-miss: bridge released at 85 deg falls back OPEN -> no closed
    credit, no success (a settled almost-closed state cannot exist).
11. order violation: closing torque with the box straddling the doorway -> no
    success (stall, or box trapped short of deep-inside).
12. wrong object: WHITE decoy inside + bridge shut -> rejected.
13. latched credit: inside credit survives the box being yanked back out.
14. rejection audit: success() never True anywhere in the battery.
15. final no-NaN.

Seed-strategy end state expressible? YES — check 6 constructs the seed's
place-from-above move and asserts rejection; check 8 additionally covers the seed's
"deposited but never closed" partial end state.
