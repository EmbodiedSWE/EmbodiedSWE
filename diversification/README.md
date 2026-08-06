# Trajectory diversification for `assembly.nut_thread.franka.osc`

Proposal 1 (Lift-and-Sample), implemented against the real scene and the real reference solve.

The premise is that `examples/solve_nut_and_bolt.py` is not a trajectory — it is a closed-loop
policy. Its z target follows the nut's measured height, its wind strokes are paced by the
measured end-effector yaw, its reclose is gated on measured finger width. A trajectory is one
sample from that program, so the artifact worth producing is not more trajectories but a
*sampler*: the same program with its constants lifted to distributions, filtered by the scene's
own success predicate.

## The three steps, and where each one lives

| Step | What it is | Here |
|---|---|---|
| 1. Tolerance analysis | Classify every constant as slack / strategy-critical / frozen, with the reason | `params_nut_thread.yaml` |
| 2. Sample against the verifier | Draw theta, run headless, keep what `scene.success()` passes | `sample_theta.py`, `run_episode.py`, `launch_pool.py` |
| 3. Repair | Read per-parameter marginals + failure telemetry, then narrow a band or patch the policy | `collect.py` output |

Every band in `params_nut_thread.yaml` is justified by something the solve script itself
records — its argparse help strings, its inline comments, or its nine-round repair log. The
`frozen:` section carries the reason each constant is load-bearing (`dt: 1/480 — the scene's
1/120 TUNNELS a pressed M16`). This is the step only a coding agent can do, and the knowledge
was already written down in the script; a blind AST fuzzer would break exactly these.

## Files

- `params_nut_thread.yaml` — the schema: `env`, `policy` (continuous bands), `strategy`
  (discrete choices), `frozen` (with reasons).
- `sample_theta.py` — theta from an episode index alone, so pod *k* runs episode *k* and a re-run
  reproduces it. Continuous bands come from a low-discrepancy sequence (even coverage at ~100
  points, no dependency, identical on every machine); strategy choices are cycled so all 18
  combinations appear 5–7 times. **Episode 0 is the nominal** — the parameters the script was
  solved at — and serves as the control.
- `nut_thread_policy.py` — the reference phase machine, unchanged, reading its lifted constants
  from theta. Frozen constants stay literals, each with the reason inline.
- `recorder.py` — records `(state, action)` per control step (state captured *before* the step, so
  pairs are not shifted) plus every scene object's 13-dim root state, and writes the video of that
  same rollout.
- `run_episode.py` — one episode: theta, scene, policy, verdict, artifacts, HDFS upload.
- `launch_pool.py` — N L20 pods, one episode each. The job spec is
  `scripts/launch_cosigen_render_pool.py`'s, unchanged where it matters.
- `collect.py` — pulls a batch back and reports the success rate plus per-parameter marginals.
- `probe_capture.py` — one-boot diagnostic for the capture pipeline.

## Running it

```bash
# smoke first (episode 0 is the nominal control)
python launch_pool.py --batch nut_b1 --n 2 --total 100 --upload

# the batch: 100 pods, one episode each, in parallel
python launch_pool.py --batch nut_b1 --n 100 --total 100 --upload

# results: success rate, per-parameter marginals, videos
python collect.py --batch nut_b1 --n 100
```

Artifacts land in `hdfs://haruna/tmp/zeyu.shen/cosigen_div/<batch>/` as `epNNNN.mp4` (the
rollout), `epNNNN.npz` (the trajectory) and `epNNNN.json` (the verdict), and `collect.py`
mirrors them to `eval_result/diversification/<batch>/`.

## pen_holder (batch `pen_b2`, 100 episodes)

12/100 succeeded, scores {0: 23, 10: 23, 25: 30, 40: 12, 100: 12}. Restricted to `all_pens=False`
(the regime the baseline ran in) that is 11/51 = 22% against the baseline's 17%, i.e. randomizing
world + parameters + structure together cost no yield.

Marginals worth acting on:
- `all_pens`: 11/51 with a sampled subset vs 1/49 with all four pens. Four pens in a 44 mm inradius
  holder is near-infeasible for this policy — a task-difficulty fact, not a bad band.
- `grasp_pick`: best 4/34, second 5/32, random_feasible 3/34. No yield difference, so which of the
  enumerated jaw candidates gets taken is free structural diversity.
- `order_rule`: far_first 4/20, thin_first 3/20, fat_first 2/21, shuffled 2/20, near_first 1/19. The
  solution's own documented order (`fat_first`, "the last pen threads the leftover gap") is not
  better than the alternatives — its recorded precedence reasoning is not load-bearing.


Phase-0 baseline, `examples/solve_pen_holder.py` unmodified, 12 seeds: 2 successes (seeds 0 and 3),
mean score 27.9. Scores are deterministic per seed, so the baseline doubles as a fidelity check.

The port passed that check only after one fix: `build_env` wrote the scene CLASS defaults over the
registered franka preset (`holder_pos (0.12,0.18)`, `pens_center (-0.10,0.15)`, `spawn_radii
(0.14,)`), which moved the holder 15 cm and put a pen at d_base 0.22 m — the exact radius where
`solve_pen` gives up. That scored 0/12 at nominal. With world dims applied as DELTAS to the preset,
the port reproduces the reference's score on 10 of 10 reported seeds, both successes included.

Invalidated by that fix: `pen_b1` (1/100) and `pen_v_world` / `pen_v_policy` / `pen_v_structure`
(0/30 each) were all run against the wrong layout.

Structural dims sampled here, which nut_thread does not have: pick order (`fat_first` — the
solution's own, with its reason recorded — plus `thin_first`, `near_first`, `far_first`,
`shuffled`), grasp choice among the enumerated jaw candidates (`best` / `second` /
`random_feasible`), the release spread pattern, and pen count.

## Results (batch `nut_b2`, 100 episodes)

26/100 succeeded; 45% within the `nut_center=True` arm against 6% without it, which is why
`nut_center` is now frozen. Successes thread from 24.5 mm to the 5 mm stop in 25–57 strokes, and
stroke count tracks `sweep_deg` (60° → 44 strokes mean, 120° → 34, 180° → 27; r = −0.60) more
strongly than any environment dimension (|r| ≤ 0.35).

Fixed-world control (`fixenv1`): with the world pinned to the nominal — same spawn, same +/-10 mm
jitter draw, same friction, same bolt slot — 5 of 12 previously-successful theta still succeeded
(ep0013/0018/0021/0067/0096, dz ~5.0 mm, 28-44 strokes, sweeps of 60 and 120 deg). So the successes
are a volume in theta, not isolated (world, theta) points. The nominal theta itself fails in that
world, so parameters matter within a fixed scene.

Worlds and parameters interact strongly (`fixenv13`, complete: 3/12 in ep0013's world, positive
control reproduced at 28 strokes). Successes barely overlap across worlds — of the 12 theta tested
in both, only ep0013's succeeds in both; ep0018/0067/0096 solve the nominal world but not ep0013's,
ep0006/0031 the reverse. ep0013's world also shows a second failure mode: theta that keep the nut
on the bolt for 63-75 strokes at dz ~22 mm without ever descending.

Read rates only over COMPLETED episodes. Lost-nut failures abort in 1-4 strokes while successes run
~50 min, so any early sample is failure-biased: this batch looked like 0/62 at one point and
0/6 in `fixenv1`, against true rates of 26/100 and 5/12.

## Two things worth knowing before reading the numbers

**Video and trajectory come from the same rollout.** GPU PhysX in this scene (192 solver
iterations, SDF collision) is not run-to-run identical, so a video re-rendered from a replay
would not be footage of the run the verifier passed. Recording inline costs one capture per 16
control steps and does not touch physics.

**The reference solve does not currently reproduce its claimed solve.** Run unmodified with its
own defaults, on the pod recipe the render pool uses, it loses the nut during the first
grip-recycle: `dz` freezes at 20.00 mm, lateral offset jumps to ~30 mm, the nut's angular velocity
goes to zero and the jaws wind air. That is its own documented r6 failure. The scene jitters the
nut spawn by +/-10 mm on every reset (`reset_pos_jitter = 0.01`, which the solve never disables),
so there is no fixed "solved" draw — the r9 run was one sample of a marginal policy. This was
established with a controlled experiment (the unmodified script and this port, same recipe, same
failure), so it is a property of the policy, not of this harness. Widening the basin is Step 3's
job, and the batch's per-parameter marginals are what drives it.
