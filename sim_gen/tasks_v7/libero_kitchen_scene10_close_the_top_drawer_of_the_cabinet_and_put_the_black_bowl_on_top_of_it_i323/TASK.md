# Task `libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_i323` — Weigh-Press Cabinet

Scene: `weigh_press_cabinet` (env `simgen.weigh_press_cabinet`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it`
— a CONJUNCTION of two independent manual subtasks: (1) push the cabinet's prismatic
top drawer shut by hand, (2) pick the black bowl and place it on top of the cabinet.
In the seed the two clauses share nothing but the cabinet: either can be done first,
each is satisfied by its own guided motion.

Kept from the seed: a cabinet with a sliding drawer, a heavy BLACK BOWL, and the
identical judged OUTCOME — drawer front flush with the cabinet face AND the black
bowl sitting on top of the cabinet. The drawer is a free rigid body captured in an
inclined channel (contact physics, not an articulation), reproducing the seed's
drawer by mechanism rather than by joint; the "cabinet top" is the cabinet's roof,
whose working surface is a raised weigh tray.

## Strategic difference

**vs the seed:** the seed's two clauses are fused into ONE causal machine. The
cabinet top carries a silver WEIGH TRAY — the head of a free vertical PLUNGER whose
stem runs down a guided shaft through the roof to a 45° PRESS BLADE resting flush on
a matching 45° RAMP FIN on the drawer's rear tail. The drawer rides a 15° inclined
slideway (rear higher): gravity parks it OPEN on front stops, and holds it there —
a dead-man return. Setting the 1.6 kg black bowl into the tray is simultaneously
(a) the seed's second clause, verbatim ("bowl on top of the cabinet"), and (b) the
POWER STROKE of the first: the bowl's weight drives the plunger down, the 45/45
wedge converts the descent into up-slope travel, and the drawer runs shut to its
rear stop — then the weight KEEPS holding it shut. The robot performs exactly one
pick-and-place of the goal object and never touches the drawer; the judged closed
state is a standing LIVE force balance, not a snapshot: remove the bowl and the
drawer re-opens on its own (smoke proves it). The seed's own strategy — push the
drawer shut by hand, park the bowl on the bare roof — produces a state where BOTH
seed clauses look satisfied, yet it is refused twice over: judged live WHILE the
oracle hand still presses, it is not success (the bowl is not in the tray), and the
moment the hand lets go the dead-man drawer re-opens.

**vs the corpus (all same-seed / same-strategy neighbours read in full):**
- `close_the_top_drawer_i276` (gravity-ram chute): the drawer is closed by the
  MOMENTUM of a heavy ball released down an incline — impulsive, ballistic, the
  carrier ends parked as debris. Here nothing is thrown and no kinetic energy is
  delivered: the transmission is quasi-static WEIGHT through a prismatic wedge, the
  held state is a live force balance (the ram's outcome persists inertially; this
  one collapses the instant the load leaves), and the pressing mass is not an
  instrument but the second judged clause itself.
- `close_the_top_drawer_i319` (ballast-press lever): the closest relative — also a
  weight-powered press, but a HINGED counterweighted rocker fed with generic steel
  ballast blocks (mass redistribution across a tipping threshold, two blocks vs
  one). Here there is no lever, no pivot, no counterweight and no threshold
  counting: the transmission is purely PRISMATIC (vertical plunger, matched-incline
  45/45 wedge, drawer on its slideway), the return force is the drawer's own weight
  on the incline, and the load is the task's unique goal object — the black bowl,
  whose placement upright in the tray IS the other half of success, with butter
  distractors below the drive threshold and an inverted bowl refused by the upright
  clause.
- `close_the_top_drawer_i98` (cam gate): the robot continuously torque-servos a
  hinged gate whose cam face pushes the drawer — the hand is the energy source and
  sustains the whole travel. Here the robot supplies no closing energy at all: it
  only positions mass, and gravity does the work.
- The `put_the_black_bowl_on_top_of_the_cabinet` family (scenes 1/2/4/5) and other
  put-bowl siblings: pure transport/placement tasks — the placed bowl's support is
  the whole judged outcome. Here the placement is judged AND is the power source of
  a second, mechanically-coupled judged outcome on a different body.

## The machine (honesty by construction)

`scene.py` asserts the press contract in `__post_init__` with real trig — geometry:
the drawer's rear hard stop IS the seated pose (q_stop = +3 mm, inside the 12 mm
tolerance); the true OPEN rest q_open is derived from the pitched front face meeting
the stops' top edge (readback matches to 0.1 mm); the blade footprint stays on the
fin face at BOTH travel ends and can NEVER land on the tail's flat top plate
(tail_h below the fin's low end, corner clearance asserted at q_stop); the fin
sweep clears the roof; the stem clears its shaft; the tray clears the collar at
both stroke ends; the bowl fits the tray interior; butter fits the tray. The FORCE
AUDIT is asserted four ways with a friction knockdown on the 45/45 wedge
(slick μs 0.08 / μd 0.06, min combine everywhere): bowl-in-tray DRIVES the drawer
shut with ≥ 1.7× margin (measured 2.09×); the empty plunger and a butter-loaded
tray both hold LESS than 0.5× of the force needed to move the drawer; the unloaded
drawer back-drives the wedge and RE-OPENS with ≥ 1.3× margin (measured 1.66×) — the
dead-man both ways. Settling is judged by velocity gates set ABOVE the GPU
phantom-velocity band plus a pose-stillness streak (every judged body moved
< 0.4 mm/step for 24 consecutive steps — teleport-safe, artifact-immune).

## Teleport solution (`solve.py`) — ONE transport teleport, ZERO forces

The solve applies no force and no wrench, ever. Its single teleport is pure
transport — the bowl lifted off its dealt plinth slot and released (zero velocity)
just above the weigh-tray floor, mapped through the MEASURED station and plunger
poses:

- P0 — settle; read back station pose/yaw, the drawer's open rest (matches the
  derived q_open), the plunger's flush rest on the fin (matches z_press), the
  bowl's dealt slot; baseline score ~0.
- P1 — TRANSPORT: the one teleport. Bowl upright just above the tray floor,
  velocity zero; the drawer is asserted still open. Score ≈ 0.39 (load credit +
  the first slither of travel as the press begins).
- P2 — WEIGH PRESS, hands off: the bowl's weight drives the plunger down, the
  blade presses the fin, the drawer runs shut UP the incline to its rear stop
  (~8 steps) and the weight keeps holding it there. Settle; success() live;
  score 1.0.
- Persistence — ≥ 3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge on seeds 0, 1 and 2 (distinct yaws, slots, jitters). `SIM_GEN_SCORE`
printed at each boundary, non-decreasing (0.00 → 0.39 → 1.00 → 1.00). The drawer
and plunger are never wrenched and never teleported: the closing translation is
delivered entirely by the machine under the goal object's weight.

## Embodiment argument (Franka, single arm)

Base ~0.55 m out on the +x (plinth slot) side of the station; everything touched
lives at world z 0.30–0.62 (the station stands on a 0.30 m plinth) within a ~0.6 m
disc — inside Franka's workspace at any episode yaw (the ±5 cm jitter and free yaw
only rotate the approach; heights never change).

- BLACK BOWL: the only thing the robot must touch. An open 110 mm square dish,
  55 mm tall, 1.6 kg (well under Franka's 3 kg payload), dealt onto one of three
  open plinth slots — rim grasp with the parallel jaw (9 mm walls ≪ 80 mm span),
  unobstructed from above. Carry ~35 cm to the cabinet top and set it down into
  the 170 mm square tray opening — 30 mm of xy slack all round for a 110 mm bowl,
  open air above (the collar guides the stem, not the tray), release from a few
  cm up is fine. No precision beyond "upright, inside the rims".
- The DRAWER never needs to be touched, and touching it is useless rather than
  merely forbidden: it can be pushed shut by hand, but the dead-man slideway
  re-opens it on release — the honest route is the machine. Smoke demonstrates
  the seed's skill with an oracle force anyway: held shut + bowl on the bare
  roof, still not success; released, it re-opens.

## Ordering

The rubric imposes no order — success is the settled live conjunction (drawer
seated in its channel AND the bowl upright in the tray). Physically the task is a
SINGLE action: the placement that satisfies the bowl clause is the same event that
powers the drawer clause; there is no second thing to sequence. No alternate
honest route exists: hand-pushing the drawer cannot persist (dead-man), butter is
below the drive threshold, an inverted bowl fails the upright clause, and a bowl
anywhere but the tray moves nothing. The two partial-credit latches (`_fload`,
`_fclose`) only anchor demonstrated progress; success itself is latch-free and
judged live on the settled state.

## Rubric

- 0.30 × latched load — the black bowl has ridden the weigh tray (upright, plunger
  in its shaft).
- 0.45 × latched max closure fraction of the drawer's open rest (channel-guarded:
  a drawer stolen off its slideway earns nothing).
- Base capped at 0.75; exactly 1.0 iff `success()`: drawer seated (front face
  < 12 mm from the cabinet face, centred in its channel) AND the black bowl
  upright in the tray, all settled (pose-stillness streak) and finite, live.
- Null policy ~0 (gravity parks the drawer open); the seed's end state (drawer
  pushed shut + bowl on the bare roof) ≤ ~0.45 while held and never success.

## Checks (`smoke.py`, rejection-only, 13 checks)

settle/no-NaN (drawer parked at the derived q_open, blade flush on the fin, bowl
on its dealt slot, score ~0); randomization A (yaw span > 90°, xy jitter);
randomization B (bowl slot permutes ≥ 2 values with real in-slot jitter, drawer
rest tracks q_open, plunger rests flush — every reset, readback); null policy ~0;
SEED strategy (bowl on the bare roof + oracle PD hand-push seats the drawer for
real — judged WHILE HELD it is still not success, released it re-opens, ~0.45);
light load (both butters in the tray: below the drive threshold, drawer unmoved,
~0); inverted bowl (its weight DOES close the drawer — machine works — but the
upright clause refuses an end state differing from success only by orientation,
~0.45); hand-slam reopen (drawer + plunger written seated, latch fires, wedge
back-drives and re-opens: no settled closed state exists without the load); latch
yank (bowl into the tray — load latch fires — then yanked to the ground: latch
survives, live clauses read False, drawer re-opens, no success); shaft guard
(plunger stolen to the ground + drawer slammed + bowl on the roof: bowl-on-tray
requires the plunger in its shaft — refused, drawer re-opens); rejection audit
(success never observed at any step of the battery); final no-NaN; frames.npz
video recorded.
