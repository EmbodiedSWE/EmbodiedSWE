# tool_hangup — sort screwdrivers onto a size-keyed hanging rack

**Env name:** `sim_gen.tool_hangup` (scene `tool_hangup`, robot `null`)
**Tier:** medium — **3 stages of breadth** (one keyed hang per present tool; each hang is
its own sub-sequence: reorient flat→vertical, align over the matching eyelet, thread the
shaft through, release to a catch). **Execution order:** NOT required across tools (any
order); within a tool the thread-then-release micro-order is forced by geometry.

## Seed provenance

- Seed: `pick_place/track_screwdriver`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_screwdriver.py`)
- Seed semantics: stage-3 **trajectory tracking**. The screwdriver starts *already
  grasped* (states loaded from a pkl of grasp states), five XFORM waypoint markers show
  the exact free-space path, randomization is explicitly zeroed, the gripper is forced
  closed every step, and the episode *terminates if the object leaves the gripper*.
  Reward = waypoint approach/progress + rotation tracking. The entire skill is following
  a prescribed aerial curve with a held object, never touching the environment.

## What changed, and why it is strategically different

Every load-bearing pillar of the seed is inverted — a solver needs a different **plan**,
not different numbers:

1. **No prescribed path, no pre-grasp.** Nothing starts in hand: three screwdrivers of
   graded sizes (6/11/16 mm shafts, 32/42/56 mm handles, color-coded) lie flat on the
   floor. There are no waypoint markers; the solver derives its own motion.
2. **The goal is contact-rich suspension, not free-space motion.** Each tool must end
   *physically hanging* from an elevated rack: shaft threaded down through an eyelet
   ring, fat handle caught on the rim, tool dangling settled under gravity. The seed's
   one hard constraint (never lose the object / never touch anything) is replaced by its
   opposite — success **requires releasing** the object into a supported contact state.
3. **The correspondence must be reasoned, and it is enforced by honest physics.** The
   three ring apertures (24/34/46 mm) are size-keyed: only the matching tool satisfies
   `shaft < aperture < handle`. At the extremes the keying is physical, not just
   rubric-side: the small tool's handle passes the large ring (it falls straight through
   to the floor — smoke control 6c) and the large shaft cannot enter the small ring
   (control 6d). The stand order along the rack is **permuted per episode** and the
   present-tool subset is sampled (2–3), so no fixed trajectory can be memorized —
   the exact opposite of the seed's deliberately zeroed randomization.
4. **The seed's own strategy is expressible and fails** (control 6a): kinematically
   carrying a tool along a five-waypoint aerial path and holding at the end — the seed's
   entire task — earns only the latched 0.10/P "raised" credit (score ≤ 0.05), and
   releasing there just drops the tool on the floor. It can never reach success.

Sibling-collision note: no other batch task claims size-keyed matching / suspension
(hanging) as its axis (i10 = extraction+convex perch+bounciness, i8 = multi-piece
construction, i9 = slide-out-of-cubby, i4 = push-under-clearance, i5 = pour,
i1 = tunnel tool-use, i3 = stand-up reorientation, i6 = weight-activated plate).

## Scene facts

- Fully procedural compound spawners (no asset files): 3 KINEMATIC eyelet stands
  (foot + post + arm + 8-box octagonal ring, ring band top at 226 mm), re-posed per
  reset (row jitter ±5 cm, row yaw ±25°, slot permutation — verified by readback);
  3 dynamic screwdrivers (shaft + handle cylinders, one rigid body, explicit low CoM at
  0.35·shaft_l so a hung tool is a stable pendulum), scattered lying flat with free yaw.
  Absent tools park off-camera at (1.1, 1.1) (fine for the null smoke's num_envs=1;
  robot bindings should bump env_spacing ≥ 3 like the pen_holder depot).
- Rubric (graded, [0,1], transients latched in `post_step`): 0.10/P per present tool
  ever RAISED to rack height (latched), 0.80/P per present tool CURRENTLY hanging in its
  matching ring (axis ≤ 35° from vertical, handle bottom within (−12, +15) mm of the rim
  top, tip dangling ≥ 42 mm below the band, clear of the floor, |v| < 0.05 m/s), and
  1.0 iff `success()` = every present tool counted. Success is a current-state physical
  predicate; a tool that fell through or tipped off keeps only its raise credit.
- Physics honesty: the oracle threads kinematically but always **releases 15 mm above
  the rim** — the catch-and-hang (and every judged outcome) is real drop physics.

## Smoke check list (21 checks)

1. settle/no-NaN + score 0 at reset (tools lying flat)
2. randomization-is-real: stand/tool readback deltas across seeded resets
3. randomization: keyed ring order permutes across episodes (≥2 signatures / 6 resets)
4. subset-is-real: present count varies (2 and 3 both occur over 8 resets)
5. null-policy-fails: 240 idle steps → score ≤ 0.02, no success
6–8. oracle reaches success() on 3 seeds (subset-sampled layouts)
9. rubric: raise latch gives small nonzero credit before any hang
10. rubric: first hang lands in the graded middle (0.2 < s < 0.95)
11. rubric monotonicity: 0 → raised → hangs → 1.0, nondecreasing
12. negative (seed strategy, held): 5-waypoint aerial tracking pins score ≤ 0.05
13. negative (seed strategy, released): tool ends on the floor, still no hang credit
14. negative (across-ring, static): tool on the rim is not hung
15. negative (across-ring, settled): still not counted
16. negative (wrong ring, fall-through): small tool drops through the large ring
17. negative (wrong ring, blocked): large shaft cannot enter the small ring
18. negative (near-miss): threaded but 45 mm above the rim not hung (z-band)
19. tolerance boundary real: releasing that near-miss settles into a counted hang
20. calibration: centered free drop threads ≥ 2/3 (funnel = 6 mm, published sweep)
21. calibration: 30 mm off-funnel drop threads ≤ 1/3

Calibration probe publishes thread-rate vs lateral offset (0–30 mm, 3 seeds each) for
the mid tool — the measured aperture funnel. Video (`frames.npz`, CWD) records the
reset, oracle seed 0, and the fall-through keying control.
