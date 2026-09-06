# vla — from generated episodes to a trained, evaluated policy

Everything downstream of data generation: bake raw episodes into a LeRobot
dataset, train a policy on it, drive the same sim closed-loop with the trained
policy. lerobot is vendored as a submodule so the whole chain lives in one
checkout at one pinned revision.

    convert/    raw episodes -> LeRobotDataset (one canonical Episode, many label conventions)
    eval/       the sim behind a socket (Isaac side) + the lerobot eval plugin (client side)
    lerobot/    huggingface/lerobot, submodule, pinned (see below)

## Two venvs, one contract

Isaac Sim is pinned to Python 3.11 / torch 2.7; lerobot needs Python >= 3.12 and its
own torch. They never share an interpreter — they talk over the socket protocol in
`eval/protocol.py` (stdlib + numpy only). So the repo has two environments:

| venv | python | built by | runs |
|---|---|---|---|
| `.venv` | 3.11 | `scripts/bootstrap_isaaclab_5_1.sh` | robobench, data generation, `eval/serve.py`, `eval/replay_actions.py` |
| `.venv-lerobot` | 3.12 | `scripts/bootstrap_lerobot.sh` | `convert/convert.py`, `lerobot-train`, `lerobot-eval --env.type=cosigen` |

Setup, from the repo root:

```bash
git submodule update --init vla/lerobot        # once (or clone with --recurse-submodules)
./scripts/bootstrap_lerobot.sh                 # creates .venv-lerobot, installs lerobot[training,pi,smolvla,diffusion] + the eval plugin
```

The bootstrap installs the submodule **editable**, so an edit under `vla/lerobot/src`
is live in the venv without a reinstall. Extras are a knob
(`COSIGEN_LEROBOT_EXTRAS=training,pi` for a smaller install); the defaults cover
every policy family we have trained on CoSiGen data (ACT, Diffusion, pi0/pi0.5,
SmolVLA).

Video decode: lerobot's fast path is torchcodec, which dlopens FFmpeg's shared
libraries at import time and **silently falls back to pyav** (several times slower
dataloading) when it cannot. The bootstrap's verification step prints which one you
got. If it says torchcodec is not loadable, put an FFmpeg 5–7 `lib/` on
`LD_LIBRARY_PATH` (conda-forge `ffmpeg`, or your cluster's FFmpeg module) before
training.

## Pinned revision

`vla/lerobot` is pinned to upstream `huggingface/lerobot` @ `6adf5151` (v0.6.1 + 24,
lerobot 0.6.2 dev). This is the exact tree every CoSiGen policy result so far was
produced with; `convert/` and `eval/lerobot_env_cosigen/` are written against its
dataset v3.0 format, processor pipeline and `lerobot-eval` plugin discovery. Bumping
the pin is a deliberate change: re-run a bake (`convert/README.md`), a `check_load`,
and one closed-loop eval before committing a new SHA.

## Walkthrough: fine-tune pi0.5 on a task and evaluate it closed-loop

The bulb task (`assembly.bulb.franka.pink_ik`) as the worked example; every other task is
the same five steps with a different preset. Paths in `<…>` are yours. Isaac steps use
`.venv`, lerobot steps use `.venv-lerobot`; a step never mixes the two.

### 1. Generate episodes (Isaac venv)

`data_engine/scripts/generate.py` runs the solved strategy for a task, records full state +
commands every tick, and renders the declared training cameras. One call = one batch of
`--num_envs` episodes; a campaign is many batches under one `<gen_root>`:

```bash
.venv/bin/python data_engine/scripts/generate.py --headless <gen_root> \
    --batch b0 --num_envs 8 --seed 0 --nominal --render
```

Each episode lands in `<gen_root>/data/<batch>/ep_XXXX/` (`traj.npz`, `meta.json`, and
`imgs/<view>.mp4` + `imgs/render_<view>.json` per camera). See `data_engine/` and the scene's
`CAMERAS` declaration for which views are rendered.

### 2. Bake a LeRobot dataset (lerobot venv)

```bash
.venv-lerobot/bin/python vla/convert/convert.py <gen_root> \
    --repo-id <org>/<task>_<convention> --control_freq 15 --workers auto
```

Defaults: `--control_space joint_target` (the commanded joint targets; presses/squeezes survive
as target offsets), successful episodes only, all rendered views. The dataset lands at
`<gen_root>/datasets/<repo_id>/` with **`meta/bake.json`**, the provenance stamp the eval loads
the sim from in step 4. Reasons for these two choices, and the label caveats, are in
`convert/README.md`. The 15 Hz `joint_target` bake is what produced every bulb result;
achieved-joint labels (`joint_pos`) at 60 Hz gave 0 % for pi0.5.

Push to the Hub if training runs elsewhere (`lerobot-train` also accepts a local dir):

```bash
.venv-lerobot/bin/hf upload <org>/<repo_id> <gen_root>/datasets/<repo_id> --repo-type dataset
# then tag it, or lerobot-train refuses to load it ("revision not found"):
.venv-lerobot/bin/python -c "from huggingface_hub import HfApi; HfApi().create_tag('<org>/<repo_id>', tag='v3.0', repo_type='dataset')"
```

### 3. Fine-tune pi0.5 (lerobot venv, 1 GPU)

`lerobot/pi05_base` is built on PaliGemma, a gated checkpoint: accept the license on the Hub
once and log in (`hf auth login`). Then:

```bash
.venv-lerobot/bin/lerobot-train \
    --dataset.repo_id=<org>/<repo_id> \
    --dataset.root=<gen_root>/datasets/<repo_id> \        # omit to pull from the Hub
    --dataset.video_backend=torchcodec --tolerance_s=0.005 \
    --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --policy.n_action_steps=10 \
    --policy.empty_cameras=1 \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.device=cuda \
    --policy.push_to_hub=false \
    --steps=30000 --policy.scheduler_decay_steps=30000 --save_freq=5000 \
    --batch_size=64 --num_workers=4 --seed=1000 \
    --output_dir=<out>/pi05_<task> --job_name=pi05_<task> \
    --wandb.enable=false
```

What the non-obvious flags do:

| flag | why |
|---|---|
| `--tolerance_s=0.005` | lerobot's default 1e-4 is below float32 timestamp resolution deep into a long v3 video file; without this, training dies with `FrameTimestampError` ~50 min in. A short smoke run cannot catch it. |
| `--policy.empty_cameras=1` | pi0.5 expects 3 image slots; our bakes have 2 cameras (front + wrist), so one slot is padded. Set to `3 - <your camera count>`. |
| `normalization_mapping` | MEAN_STD on state/action (the pi0.5 recipe); IDENTITY on images since the model normalizes internally. |
| `--policy.n_action_steps=10` | the re-plan horizon at inference; a training-time default that eval may override (below). |
| `scheduler_decay_steps = steps` | decay the LR to the end. Across every model we trained, only the fully decayed final checkpoint produced successes. |
| `freeze_vision_encoder=false`, `train_expert_only=false` | full fine-tune. Fits one 80–96 GB GPU at batch 64 with gradient checkpointing + bf16. |

Host memory: budget ~240 GB RAM for the pi0.5 dataloader over a long run (RSS creeps).
Resume after preemption with `lerobot-train --config_path=<out>/checkpoints/last/pretrained_model/train_config.json --resume=true`.
A 20-step smoke (`--steps=20 --save_freq=10`) checks the config and that the checkpoint loads
before you commit a GPU-day.

### 4. Evaluate closed-loop (two terminals, one GPU node with RT cores)

The sim is pinned to the dataset's `bake.json`, so it is, by construction, the world the policy
was trained on: same preset, control space + rate, controller gains, cameras, task sentence.

```bash
# terminal 1 — Isaac venv. Boots the scene once and stays warm across runs.
.venv/bin/python vla/eval/serve.py <gen_root>/datasets/<repo_id>/meta/bake.json \
    --headless --port 5555 --record-dir <eval_out>/rollouts

# terminal 2 — lerobot venv. Declares nothing about the sim; it binds cameras/dims/fps
# from the server handshake.
.venv-lerobot/bin/lerobot-eval \
    --policy.path=<out>/pi05_<task>/checkpoints/030000/pretrained_model \
    --policy.device=cuda --policy.n_action_steps=10 \
    --env.type=cosigen --env.port=5555 --env.max_episode_seconds=240 \
    --eval.n_episodes=32 --eval.batch_size=1 --eval.use_async_envs=false \
    --seed=1000 --output_dir=<eval_out>
```

Knobs that matter:

- `--env.max_episode_seconds` is the per-episode cap in **sim seconds** (rate-independent).
  Pick it from the demos: bulb successes all end by ~90 s, so 240 s is generous and anything
  longer only burns GPU on failures.
- `--policy.n_action_steps` is a legitimate eval axis (how many chunk steps execute before
  re-planning). Sim time freezes while the policy thinks, so inference latency is not modeled.
- `--seed` selects the scene randomization per episode; shards of one eval use disjoint seeds.
  `serve.py --init dataset --init-batch <gen_root>/data/<batch>` starts from recorded episode
  states instead (episode = seed % n), the generalization-eval mode.
- `--record-dir` on the server writes per-episode state/action rollouts next to lerobot's videos.
- For long contact tasks, run 1 episode per job with a wall-clock watchdog and count a killed
  episode as a failure. On the bulb, ~20–40 % of failed episodes wedge the bulb in the socket rim
  and PhysX drops to seconds per step; re-rolling those biases success upward.

### 5. Read the result

`<eval_out>/eval_info.json`:

- `overall.pc_success` = success rate. Success is the suite grader's verdict
  (`robobench/suites/<suite>/grader/`), not a scene heuristic.
- `per_task[0].metrics.max_rewards` = per-episode **peak grader progress** on the same 0..1
  rubric generation reports as `score` (bulb: picked 0.2 + engaged 0.2 + threaded 0.6, so a
  run stuck at 0.40 grasped and seated the bulb but never threaded it). `sum_rewards` is the
  area under the progress curve.
- `<eval_out>/videos/` has one mp4 per episode. Watch the failures before trusting a number.

Reference, bulb / `joint_target` 15 Hz / `pi05_base` / 30k steps / nas 10: 25–38 % success
depending on checkpoint and seeds; `pi05_droid` init: 0 % (all episodes plateau at 0.40).

## Where to read next

- `convert/README.md` — the bake CLI, the two decisions (`--control_freq`,
  `--control_space`), the label caveats.
- `eval/README.md` — `load_sim`, open-loop replay certification, and the
  two-terminal closed-loop `lerobot-eval` recipe.
