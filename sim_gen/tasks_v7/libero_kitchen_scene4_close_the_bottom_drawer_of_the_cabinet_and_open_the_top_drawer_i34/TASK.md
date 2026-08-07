# Task `libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_i34` — Gumball Meter

Scene: `gumball_meter` (env `simgen.gumball_meter`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer`
— "close the bottom drawer of the cabinet and open the top drawer": operate TWO prismatic
slides on one fixture by their handles, judged purely by the slides' joint positions
(bottom fully in, top fully out).

Kept from the seed: a ground-standing fixture whose moving part is a **horizontal
prismatic slide operated by a protruding handle**, the seed's two motor verbs — *pull the
slide out to its stop* and *push the slide back in to its stop* — and a success clause on
the slide's terminal position (the shuttle must finish fully IN). The slide here is a free
rigid body captured in a channel (guided by walls/floor/roof, not a joint), so the seed's
drawer skill is reproduced by contact physics rather than an articulation.

## Strategic difference

**vs the seed:** in the seed the slide positions ARE the goal — one close + one open is
terminal, no objects exist, and both strokes together carry all the credit. Here the
identical pull/push primitive is demoted to the *stroke of an escapement*: the shuttle is
a single-ball airlock between a sealed silo and a sealed basin, so each pull+push cycle
dispenses exactly one ball. The task is **exact-count dispensing**: read the order quota K
(the number of green marker posts, resampled every episode), execute exactly K cycles, and
**stop** — the (K+1)-th cycle overfills the sealed basin, which is latched irreversible
(score clamped to 0.20, success permanently refused). The seed has no perception, no
counting, no repetition, and no irreversible failure mode; here the slide's end position
is just one clause of success. A policy that only replays the seed's skill (one pull + one
push, no counting) earns at most 0.06 + 0.64/K.

**vs the corpus surveyed this session:**
- `matchbox_drawer_i11` (nearest motor cousin — prismatic tray pull/push by a knob): the
  tray is a *container* that is opened, hand-LOADED with a free cube through the exposed
  aperture, and re-closed once. Here nothing is ever placed by hand — the silo, channel
  and basin are sealed, balls move only *through* the mechanism, and credit comes from
  doing the cycle exactly K times and self-terminating.
- `carousel_airlock_i9` (nearest mechanism cousin — airlock feeder): a rotary rotor
  sweeps ONE manually loaded ball to the basin; a single delivery is success. Here the
  airlock is prismatic and pre-loaded, and the objective is a *count* against an
  episode-varying quota with an overfill spoiler — the new skill axis is counting +
  restraint, not transport.
- `chute_switch_i14` (ball machine): routes colored balls to matching bins by setting a
  two-state switch — an identity/routing task with no metering; extra balls are harmless
  there, fatal here.
- `ram_eject_i1`, `tunnel_shuttle_i3`, `silo_scoop` (`push_cube_i30`), `cam_press_i35`:
  eject/shuttle/scoop/press mechanisms, all "reach the terminal state" tasks. None of the
  ~34 corpus tasks makes **how many times** a primitive is repeated — and stopping early
  under an irreversible penalty — the objective.

## The machine (honesty by construction)

`scene.py` asserts the airlock contract in `__post_init__`: every opening a ball is not
meant to pass is sub-ball (roof gap, shuttle side gaps, tail opening, viewing slit); the
flared silo throat is sub-2-ball (single file); at OUT the pocket clears the throat and
the shuttle's solid rear seals it; the pocket at OUT aligns with the drop hole; the basin
is sealed and its internal ramp rolls dispensed balls clear of the hole. Anti-jam reliefs
(45° pocket-lip chamfers, 45° bore countersink) are asserted to stay sub-ball and to keep
the seal intact. All slide surfaces carry a bound polished-steel physics material
(mu 0.12/0.10, min combine) — without it, PhysX's default friction (~0.5) lets a two-ball
chain force-close against the moving shuttle faces, which no real gumball slide allows.

## Teleport solution (`solve.py`) — zero teleports

- P0 — settle, read the layout back (tower pose, quota K, stock), baseline score ~0.
- P1..PK — one airlock cycle per quota ball, all through applied force on the shuttle
  (PD force along the tail axis, ≤10 N, exactly what a hand on the T-knob does; balls are
  never forced): pull past the silo wipe, creep to the OUT stop (front wall arrests it),
  hands-off while gravity drops the pocket ball through the hole onto the basin ramp,
  push back with a gentle creep into the IN stop (rear wall arrests it), gravity reloads
  one ball. `SIM_GEN_SCORE` printed at each boundary, asserted non-decreasing.
- Restraint — exactly K cycles; asserts the overfill latch never fired.
- Persistence — ≥3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge on seeds 0, 1, 3 (K=3) and 5 (K=2), covering both quota values and
distinct tower poses/yaws.

## Embodiment argument (Franka, single arm)

- The only thing the task requires touching is the yellow T-knob: a 16 mm square-section
  crossbar, 56 mm long, at z ≈ 0.11, protruding 68–154 mm beyond the channel's front wall
  (stroke 70 mm). Side-on grasp with the parallel gripper (16 mm << 80 mm max jaw), or
  hook the closed fingers behind the T for the pull and press the bar for the push.
- The strokes are 70 mm straight horizontal translations at fixed height, needing ≤10 N —
  trivially inside Franka payload and workspace with the base ~0.45 m from the tower on
  the tail (+x) side; the ±5 cm xy jitter and free yaw keep the knob within reach (planar
  yaw only, knob height never changes).
- Perception: count green posts (25 mm square pillars, 10 cm tall, standing beside the
  tower) and watch the basin through the slit window — visual only.
- The sealed silo/basin are honest for an arm: nothing requires (or permits) reaching
  inside; the roof, cap and walls make the knob the only lever on the system, which is
  the point of the mechanism.

## Ordering

Pull → drop → push → reload is physically inherent (a ball can only leave when the pocket
is over the hole, and the pocket can only refill under the bore), and cycles are
inherently serialized — the count in the basin can only grow one ball per full cycle
(smoke check 6 verifies holding the shuttle OUT does not stream balls). The rubric adds no
arbitrary order: `_maxcnt` latches the settled basin count (each increment requires a full
cycle), `_pull` latches mechanism engagement, `_over` latches overfill; success is
otherwise judged live.

## Rubric

- 0.06 — the shuttle was ever pulled past half travel (mechanism engaged).
- 0.64 × min(maxcount, K)/K — latched settled basin count, one increment per cycle.
- Cap 0.70 without success; overfill clamps the base to ≤0.20 forever.
- 1.0 iff `success()`: live basin count == K, every active amber in the basin or stowed
  in the silo column, decoy outside, shuttle parked fully IN, never overfilled, settled,
  finite.

## Checks (`smoke.py`, rejection-only, 16 checks)

settle/no-NaN; randomization A (yaw span >90°, xy jitter); randomization B (K and stock
vary, standing markers == K every reset); null policy ~0; SEED-strategy (one real
uncounted cycle → one cycle's credit, no success); one-per-cycle (shuttle held OUT ~5 s
gains no extra ball); overfill spoiled (K+1 constructed → spoil cap); spoil irreversible
(ball removed, count == K again → still refused); under-count (K−1 → partial only);
exact-but-OUT (parked-IN clause refuses); red decoy in basin (refused); stray amber on
the floor (refused); latched credit survives removal; rejection audit (success never
observed anywhere in the battery); final no-NaN; frames.npz video recorded.
