# Task `libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i276` — Ram-Chute Cabinet

Scene: `ram_chute_cabinet` (env `simgen.ram_chute_cabinet`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet` —
"close the top drawer of the cabinet": the Franka pushes the drawer front along its
prismatic travel until the cabinet joint reads closed. One guided translation OF the
judged part; the drawer's position IS the goal.

Kept from the seed: a cabinet with a sliding drawer and the identical judged OUTCOME —
the drawer's front face flush with the cabinet face, "drawer closed", judged on the
drawer's terminal position. The drawer here is a free rigid body captured in a channel
(floor slab, side walls, rear hard stop, roof — contact physics, not an articulation),
reproducing the seed's drawer by mechanism rather than by joint.

## Strategic difference

**vs the seed:** in the seed the robot actuates the judged part directly and
CONTINUOUSLY — a guided hand-push on the drawer front through its whole travel. Here
the robot NEVER touches the drawer and never sustains ANY actuation: the drawer's
protruding front lives inside a steel GUARD TUNNEL (roofed, 100 mm interior — the
drawer and the 60 mm ball pass, a parallel-jaw hand does not), and the only way to
close it is a GRAVITY RAM: pick the heavy steel BALL (60 mm, 1.4 kg) out of its dock
tray and drop it into the upper mouth of the green 15° CHUTE. The ball accelerates
down the incline, shoots through the tunnel, strikes the drawer's front face, and its
MOMENTUM drives the drawer to its rear stop — an impulsive, ballistic delivery, the
opposite of the seed's quasi-static guided push. The robot's entire contribution is
transport-and-release of a passive projectile BEFORE anything judged happens; from
release onward the episode is hands-off physics. Success demands BOTH terminal
states: drawer seated (< 12 mm) AND the ram ball parked at rest against the closed
drawer face — so the seed's own end state (drawer closed, no ram delivered) can
never succeed.

**vs the corpus surveyed this session:**
- `close_the_top_drawer_i98` (same-seed sibling, read in full): there the robot
  continuously torque-servos a hinged GATE through a ~70° arc and a sliding cam
  converts the regulated rotation into the drawer's translation — sustained guided
  actuation of a transmission's input link, quasi-static throughout. Here there is
  no transmission and no sustained actuation of anything: the input is one discrete
  release, the energy source is gravity (not the hand), and the coupling is a single
  ballistic impact, not a maintained cam contact.
- `close_drawer_i58` (read in full): the drawer closes itself by stored gravity on
  an inclined runway once obstructions are removed — the robot subtracts constraints.
  Here the runway is FLAT, nothing is stored, nothing obstructs: the robot must ADD
  the energy carrier (fetch and release the ball); doing nothing leaves the drawer
  open forever.
- `pen_holder` (robobench packing exemplar): pick-and-insert into a passive
  container; the placed object is the judged object. Here the transported object is
  a PROJECTILE whose delivered momentum — not its placement — produces the judged
  outcome on a different body.

## The machine (honesty by construction)

`scene.py` asserts the ram contract in `__post_init__`: the drawer's rear hard stop
IS the seated pose (q_stop = +3 mm, inside the 12 mm tolerance); the ball fits the
chute corridor, the tunnel, and the aperture with stated margins; the ball's centre
at rest height meets the drawer's front face band; the widest sampled drawer leaves
the chute foot clear (no spawn overlap); a parked ball fits wholly inside the tunnel;
the jittered ball always lands inside its dock tray; the chute foot stands 1 mm
PROUD of the runway (a drop-off, never an upward lip that could park the ball). The
ENERGY AUDIT is asserted at BOTH ends of the q0 range: released at the drop station,
the ball arrives at the drawer with ≥ 2.5× the work needed to seat it (incline
acceleration with slick friction, flat-runway loss, inelastic momentum transfer,
friction + damping work over the travel). All sliding surfaces share a slick
material (μs 0.10 / μd 0.08, min combine, restitution 0) — the audit and the
no-bounce parking both depend on it. The ball carries
solver_velocity_iteration_count=4 (kills GPU phantom creep during the hands-off
persistence window).

## Teleport solution (`solve.py`) — ONE transport teleport, ZERO forces

The solve applies no force and no wrench, ever. Its single teleport is pure
transport — the ball lifted from the dock tray and released (zero velocity) at the
scene's own `drop_point()` above the chute's upper mouth, mapped through the
MEASURED station pose:

- P0 — settle; read back station pose/yaw, q0 (drawer readback matches), ball dock
  position; baseline score ~0.
- P1 — TRANSPORT: the one teleport. Ball at the drop point, velocity zero. Drawer
  asserted unmoved (< 5 mm); score 0.30 (delivery credit only).
- P2 — GRAVITY RAM, hands off: the ball falls ~12 mm onto the chute, rolls down,
  crosses the runway, strikes the drawer front (~84 steps after release on seed 0)
  and drives it to the rear stop; restitution 0 parks the ball against the closed
  face. Settle; success() live; score 1.0.
- Persistence — ≥ 3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge on seeds 0–4 (yaws spanning the free ±180° range, q0 across its
band, distinct xy and dock jitter). `SIM_GEN_SCORE` printed at each boundary,
non-decreasing (0.00 → 0.30 → 1.00 → 1.00). The drawer is never wrenched and never
teleported: its closing translation is delivered entirely by the ballistic ram.

## Embodiment argument (Franka, single arm)

Base ~0.55 m out on the dock/chute (+x, +y) side of the station; everything touched
lives at world z 0.36–0.47 (the station stands on a 0.30 m plinth) within a
~0.5 m disc — inside Franka's workspace at any episode yaw (the ±5 cm jitter and
free yaw only rotate the approach; heights never change).

- STEEL BALL: the only thing the robot must touch. A 60 mm sphere (≪ 80 mm jaw
  span), 1.4 kg (well under Franka's 3 kg payload), resting in an open-topped dock
  tray whose rims stop 20 mm up its equator — a standard top grasp with the
  parallel jaw, unobstructed from above. Carry ~20 cm to the chute's upper mouth
  (open air above the corridor, 30 mm lateral clearance walls) and open the
  gripper. Releasing from a few cm above the mouth is fine — the walls funnel the
  ball onto the slope. No precision beyond "over the corridor, near the top".
- The DRAWER is never touched and CANNOT be touched: its protruding front is
  roofed by the guard tunnel (interior 134 mm wide × 80 mm above the slab under a
  fixed steel hood) — no gripper or forearm fits, so the seed's direct push is
  physically denied, not merely un-rewarded. Smoke demonstrates the seed's skill
  with an oracle force anyway: the drawer seats, no success (~0.55).

## Ordering

The rubric imposes no order — it judges two live terminal states (drawer seated,
ram parked against it). The task's only ordering is physically inherent: the ball
must be delivered before it can end up parked against a closed drawer, and the
drawer only closes when the ball arrives. There is no alternate honest route: the
drawer cannot be reached by hand (guard tunnel), and a ball laid gently anywhere
does not close it (no kinetic energy — smoke's dead-ram check). The two
partial-credit latches (`_fdeliver`, `_fdrawer`) only anchor demonstrated
progress; success itself is latch-free and judged live on the settled state.

## Rubric

- 0.30 × latched delivery — the ball entered the chute corridor / guard tunnel.
- 0.55 × latched max closure fraction of the drawer's initial opening
  (channel-guarded: a drawer stolen out of the cabinet earns nothing).
- Base capped at 0.85; exactly 1.0 iff `success()`: drawer seated (front face
  < 12 mm from the cabinet face, riding centred in its channel) AND the ram ball
  parked at rest against it inside the tunnel, all settled and finite, live.
- Null policy ~0; the seed's end state (drawer closed, no ram) ~0.55, never
  success.

## Checks (`smoke.py`, rejection-only, 12 checks)

settle/no-NaN (drawer at q0 in channel, ball docked, score ~0); randomization A
(yaw span > 90°, xy jitter); randomization B (q0 span AND dock jitter vary, settled
readback tracks both every reset); null policy ~0; SEED strategy (oracle 6 N push
seats the drawer for real — no ram delivered, ~0.55, no success); dead ram (ball
laid at rest against the open drawer's face: no kinetic energy, drawer stays put,
~0.30, no success); park near miss (drawer seated + ball 20 mm outside the park
gap: parked-clause refuses, credit ≤ 0.85 cap); channel guard (drawer stolen to the
ground + ball in the tunnel → drawer latch refuses, delivery credit only); latch
survival (ball delivered then returned to its dock → delivery latch survives, live
clauses read False, no success); rejection audit (success never observed anywhere
in the battery); final no-NaN; frames.npz video recorded.
