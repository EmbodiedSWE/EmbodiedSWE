# SUMMARY — bulb_franka_osc_fable (Claude Fable 5, 2026-07-13)

**Task:** `assembly.bulb.franka.osc` — Franka picks up a light bulb lying on the table, stands it
up, sets its Edison cap into a fixed lamp socket, and screws it in until `scene.seated()`
(≤27 mm above socket origin, ≤15 mm lateral, ≤12° tilt). The scene had never been verified
robot-solvable before this session.

> **2026-07-14 follow-up:** the μ caveat below is fixed upstream — bulb.usd/scene now default to
> per-shape friction (glass 0.3, cap/thread 0.01) and the glass render was fixed (white frosted, was
> near-black); see README "Upstreamed".

**Result: SOLVED in 5 solver iterations + 2 probes (~1 h wall).**
`SOLVE[fable] | DONE | seated=1 | h=24.0mm lat=0.0mm tilt=0.0deg | strokes=21` — pick → stand →
insert → release (bulb stands alone) → regrasp → thread −1185° (3.3 revolutions, 32→24 mm,
pitch-consistent 2.5 mm/turn, every stroke 100% torque-coupled) → release; the bulb keeps its seat.
**Caveat: needs the scene's official `bulb_friction` dial at 0.3** (run `--bulb_friction 0.3`);
at the shipped 0.01 the task is most likely unsolvable for any parallel-jaw gripper (see "the
physics wall" below). μ=0.3 was probe-verified physics-neutral for the thread itself.

## The strategy (what's new vs the nut_thread solve)

The bulb adds two problems the M16 nut never had: it spawns **lying on its side** (screw axis
horizontal — must be reoriented 90°), and it is **round** (no hex flats — torque is pad friction
only). The solve:

1. **Pick at the neck waist**, not the glass: flat pads on the sloped Ø48 barrel watermelon-seed
   the bulb axially out of the jaws at ANY force (failed twice); the concave neck is
   self-centering and carries rigidly.
2. **Closed-loop reorient**: the lying bulb actually rests tilted ~25° (cap end droops to the
   table), so the wrist pitches about the finger axis *until the measured bulb axis reads
   vertical* (lands at −78…−93°), not a scripted 90°.
3. **Place-then-regrasp**: you cannot screw with a horizontal hand, so insert the cap into the
   bore (touch-detect), release (probe showed the bulb stands if ≤~9° of vertical), back the jaws
   out horizontally, and regrasp from straight above.
4. **Palm-graze regrasp + friction wrench**: the palm bottoms on the glass dome 66 mm below
   `panda_hand`, so the pads can only grip 3–7 mm ABOVE the glass equator — at stock μ=0.01 every
   pinch there ratchets the bulb up and out (the r3 failure); at μ=0.3 it self-locks. Then
   nut_thread-style feedback-paced 120° wind strokes with z-follow and a 3 mm lean (the palm
   pressing the dome doubles as the engagement press). Round glass means no hex-alignment logic
   at all — reclose at any angle; slip is a graceful clutch.

## Run-by-run (details in BUILD_LOG.md)

| run | outcome | lesson |
|-----|---------|--------|
| probe_v1 | all green | thread real at dt=1/240 (2.51 mm/turn, press 0.3–5 N); released bulb stands ≤9° tilt; lying bulb rests TILTED ~25° |
| r1 | fell on release | approach chain works; fixed-close grasp seeded the glass out of the jaws onto the neck (masked by closed-loop); release opened vs a stale width → 0.1 mm clearance dragged it over |
| r2 | grasp failed | force-calibrated close only *kissed* a Ø38 sloped section (~0 N) — pad band never wraps the equator of a tilted barrel |
| r3 | fell while threading | neck pick + insert + release all clean; threading 100% coupled BUT each pinch ratcheted the bulb UP ~0.4 mm (band above equator + μ=0.01) until it climbed out |
| r4 | fell on redescend | aiming the band below the equator just ploughs the PALM into the dome (palm offset = 66 mm, measured from r3's grip_off=149.0) |
| probe_v2 | green | μ=0.3: thread + stability identical to stock |
| r5 | **SOLVED** | μ=0.3 + palm-graze regrasp: 21 strokes, 100% coupling, seated with the jaws open |

## Scene bugs / issues found (full list in README.md)

1. dt=1/240 is genuinely sound here (no nut_thread-style tunneling) — positive finding.
2. `bulb_init_z` comment ("~glass radius") doesn't match the real resting pose (tilted, origin at 5.3 mm).
3. A bulb set loosely in the socket is only metastable (9–10° tipping budget; slow lean creep).
4. **Stock `bulb_friction=0.01` covers ALL bulb colliders including the glass** → likely
   robot-unsolvable as shipped; recommend per-shape friction in bulb.usd (slick cap, grippy glass).
5. Franka-specific: palm-on-dome limit means no below-equator grip exists on a standing bulb.
6. Benign `modify_articulation_root_properties` warning at spawn; smoke's `BULB_FREE_END=0.004`
   comment disagrees with the measured bbox.

## Artifacts

- `NOTES.md` — pre-registered plan (written before any run)
- `BUILD_LOG.md` — run-by-run record incl. probes and the final verdict
- `probe_bulb.py`, `solve.py` — the probe and the solver (both re-runnable)
- `logs/` — raw stdout of every run
- `videos/solve_bulb.mp4` — headless recording of a full solve (fresh rollout)
- `session_transcript_6337eeb2.jsonl` — snapshot of the Claude Code session transcript (canonical,
  complete version: `~/.claude/projects/-home-haoxiang-Documents-Research-CoSiGen/6337eeb2-7b5b-48ee-a693-cccac48c002a.jsonl`)

Reproduce: `.venv/bin/python experiments/bulb_franka_osc_fable/solve.py --headless --bulb_friction 0.3`
(or `--livestream 2` to watch).

## Post-solve session log (conversation with Haoxiang, ongoing)

- **Q: open-loop or closed-loop?** Answer: a closed-loop *scripted* policy on privileged sim state —
  all phases servo on live poses (grasp point from the live bulb axis, reorient until the measured
  axis is vertical, bulb-origin insert servo, touch-detect, width-calibrated grips, z-follow,
  EE-yaw-paced winds, seated()/height termination). Open-loop parts: the phase sequence, a few
  settle timers, and the fixed stroke cycle. No vision/F-T — a perception version would need to
  re-derive poses and contact events.
- **Video recorded** (`videos/solve_bulb.mp4`, 170 s, 1280×720@30, real-time speed, camera from
  front-right): a FRESH rollout, which solved independently —
  `seated=1 | h=24.0mm | strokes=22` — a second success / repeatability data point (r5 was 21
  strokes). Recording ~10 min wall via `scripts/record_video.py` (1 frame / 8 steps).
- **Haoxiang's review of the video:** solution looks good overall; flagged a ~40 s-in long idle
  followed by a threading restart. **Diagnosis (from the recorded run's log, steps 10.5k–11.9k):**
  stroke 7's wind sat in a near-deadlock — the open-jaw REWIND overswings the +45° landing by up to
  ~80° (per-step latch, no load), the reclose pinches with the wrist ~90° out of position, and the
  wind pacing gate (advance while lag < 20°; bind-escape needs lag ≥ 20° for 2.5 s straight) sat
  exactly on its boundary: the crawling wrist kept resetting the bind timer, so the escape that
  should fire in 2.5 s took ~60 s (command trickled +35°→+1°, h frozen at 29.80). Escape then fired
  (`stroke 7 BOUND at wound 1 — recycling`), grip recycled, threading resumed. Same stall hit
  strokes 13/19 but was caught fast. Harmless (grip stays coupled, zero regression), just slow.
- **Decision (Haoxiang): keep the current solution as-is** — the known fix (hold the rewind until
  the EE settles at the landing before reclosing + an absolute ~6 s per-stroke timeout, est. ~30%
  faster threading) is documented but intentionally NOT applied.
- Docs/transcript housekeeping: session JSONL snapshotted into the folder (refreshed after each
  milestone); this section is the running conversation record.
- **Q: would an RL environment + training improve these tasks?** Assessment given: RL helps
  robustness-across-jitter and threading speed (threading is a near-ideal RL sub-task: dense
  descent reward, repetitive, Factory/IndustReal precedent; it would optimize away the stroke-7
  stall and reclose drag). RL would NOT have found the two real blockers — the μ=0.01 solvability
  wall and the palm-on-dome grip limit are environment properties, and probe-first found them in
  minutes. End-to-end RL is a bad deal (~25k-step episodes, multi-stage, irreversible failure);
  recommended path if pursued: gym-style task wrapper over `assembly.bulb` (obs/reward/termination
  — robobench has no task layer yet), phase-local policies staged by the scripted solver (train
  threading from "bulb standing" start states), and residual-RL/BC-init from this solver's
  demonstrations. Script-first → probe physics → RL on the contact-rich phase is the defended
  ordering. (No action taken — offered to scope the wrapper as a next experiment.)
- **Follow-up: iteration or final performance?** Final performance. Iteration was diagnosis-bound
  (probes answer causal questions in minutes; an RL curve can't say WHY reward is flat, and at the
  μ=0.01 wall it would burn GPU-days to say "unlearnable" — and produces no scene-bug list). Final
  performance is where RL pays: no stalls, learned timing, and statistical robustness over hundreds
  of randomized resets vs the scripted solver's 2-for-2 existence proof. Division of labor:
  scripted solve + probes establish solvability & env bugs; RL distilled from the script hardens
  and speeds the policy.
- **Follow-up: assuming a fixed scene — RL before or after scripts, and which algorithm?**
  Position: script-first even in a healthy scene (the solver IS the RL infra: start-state
  generator, demo generator, eval harness, curriculum). RL only on the THREAD phase (pick/
  reorient/insert stay scripted/BC — sparse signal + catastrophic exploration). Algorithm ranking:
  (1) residual PPO over the scripted loop as the cheap first probe (safe exploration, answers
  "what does learning buy" in a day; ceiling = can't restructure the cadence); (2) the real
  result hinges on parallel-env count in this contact-heavy scene (SDF + dt=1/240 + 192 solver
  iters): ≥512 envs → PPO + demo warm-start (Factory/IndustReal-proven for nut-on-bolt);
  ~64–256 envs → RLPD (SAC + 50/50 demo buffer) — the actual bet for this scene given physics
  cost; (3) vanilla PPO from scratch: skip (engagement is needle-in-haystack exploration that
  demos solve free). Implementation notes: action space = OSC task level (Δpose + grip force,
  30–60 Hz decimation, never raw torques); train per-stroke/30 s segments with early termination,
  not full 100 s episodes.
- **Follow-up: is the idle fixable by re-iteration (no RL)? Was it previously unnoticed?**
  (a) Yes — deterministic, fully-diagnosed defect ⇒ two surgical script edits (settle EE at the
  rewind landing before reclose; absolute ~6 s stroke timeout), one verification run. RL is for
  un-enumerable defects (robustness tails), not named ones. (b) Honest split: the BOUND-recycle
  phenomenon WAS caught and documented during r5 (mechanism logged, fix deferred as cosmetic);
  the ~60 s worst-case magnitude was MISSED — the iteration loop measured per-stroke net progress
  (h, seated) but not dwell time between milestones. A "sim-seconds per stroke" metric would have
  flagged it; instrumentation lesson recorded, and it validates the user's visual-review-first
  rule (3 s of video caught what the milestone logs summarized away). Decision unchanged: keep
  as-is.
- **Follow-up: is RL unnecessary for Franka in general? Where would it help (dexhand / transfer /
  multi-stage)? Critical view on data-at-scale?** Position: for Franka parallel-jaw quasi-static
  assembly, RL is structurally optional (1-DOF gripper = enumerable contact states, OSC task
  space, stable equilibria to servo). RL value ranked: (1) dexterous hands — strongest case,
  evidenced in-repo by the G1 sessions (aperture/ratchet/IK-bias walls; contact-mode combinatorics
  beat state machines); (2) sim-to-real perception policies (scripts act on privileged sub-mm
  signals a camera can't see); (3) dynamic non-quasi-static tasks; (4) multi-stage: scripted
  orchestration over RL-hardened fragile skills, never end-to-end (compounding 95%^10≈60%);
  (5) transfer across similar scenes: scripts + LLM-agent porting is cheaper today (bulb reused
  nut_thread's scaffolding) — flips only if the deliverable is one generalist policy. Data-at-scale
  critique: scripted experts are great engines BUT (a) narrow support/one homotopy, no recovery
  behaviors → cloned brittleness; (b) information asymmetry — expert conditions on privileged
  signals, unimitable from images at contact-critical moments; (c) success-only filtering deletes
  near-failure states the student needs; (d) sim artifacts (μ dial, dt behavior) get baked in.
  Standard fixes (DAgger, RLPD-style fine-tuning) mean RL re-enters as the LAST pipeline stage
  anyway. Net: scripts to solve + seed data; RL for robustness, dexterity, and leaving the sim.
- **Follow-up: for sim-data-gen at scale — "RL overkill for jaw, good for dexhand"? (+G1 has
  bugs.)** Story confirmed with two corrections: (1) for dexhand, RL isn't better data-gen, it's
  the ONLY way to obtain an expert for skills no script can express (in-hand reorient, gaiting) —
  and RL experts carry their own data pathologies (jittery, physics-exploiting, non-naturalistic;
  shape for smoothness or distill; prefer hybrid scripted-arm + learned-fingers). (2) Embodiment
  verification is the prerequisite: a script hits an embodiment bug and STOPS WITH A DIAGNOSIS
  (the G1 sessions' ratchet/bias findings); RL either silently reward-hacks the bug into the
  dataset at scale or silently fails, indistinguishably. G1 bugs → scripted probes first, RL
  after. Cost warning: RL multiplies per skill (GPU-days each), scripts amortize (LLM-hours per
  scene) — so even for dexhand, train a SMALL reusable skill library (grasp/pinch/rotate/guarded
  insert) and compose with scripted orchestration per scene, not per-task RL.
- **Follow-up: pretrained RL skill library — fine-tune per task, or compose frozen?** Middle
  answer: compose FROZEN by default, fine-tune only as targeted repair, and put the adaptation
  burden on the ORCHESTRATOR. Rationale: contact skills are valid only inside their training
  distribution, violated mostly at skill boundaries (handoffs) — mirroring how the scripted solve
  staged each phase's preconditions. Pure per-task fine-tuning defeats library amortization and
  forks/forgets the shared skill. Structure: (1) pretrain goal-conditioned, object-parameterized,
  geometry-conditioned skills under wide DR (contact-generic verbs amortize; "thread-a-bulb"
  doesn't); (2) frozen composition where the scripted orchestrator provides entry FUNNELS (steer
  the world into each skill's distribution) + pre/postcondition VERIFICATION with retry (checkable
  contracts like seated(), tilt ≤ 9°) — robustness at zero training cost; (3) fine-tune only on
  measured failure, via residual/adapter heads on the frozen backbone with a small budget. Same
  philosophy as the scripted session: fix distribution problems at orchestration; touch the skill
  only when a probe proves the skill is what's broken.
- **Follow-up: how do scene get_state/set_state help here?** They're the enabling primitive for
  the whole factory: (1) phase-local RL resets — snapshot "bulb standing", set_state thousands of
  threading episodes (25k→7k steps, no prefix replay); (2) boundary-distribution banks — empirical
  skill-entry distributions from jittered scripted rollouts, for training and for the frozen-skill
  "measured failure" gate; (3) recovery/near-failure data — bank states just before failures,
  replay with varied actions (fixes success-only bias); (4) branching data-gen — K diverse
  suffixes per expensive prefix (fixes one-homotopy narrowness); (5) orchestrator retry = restore
  to last verified postcondition; (6) deterministic eval seeds. Caveats: scene state ≠ full
  checkpoint (robot joints/controller targets needed; NB the scripted solver carries history —
  grip_off, wound, unwrap — so it can't resume from bare sim state, while a Markovian RL policy
  can: a genuine architectural point FOR learned skills); PhysX contact/warm-start caches aren't
  captured — probe restore fidelity (restore→zero action→step→assert no pop) before trusting;
  completeness is per-scene work (weld/gate scenes need extra state).
