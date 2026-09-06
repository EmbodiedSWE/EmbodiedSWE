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

## Where to read next

- `convert/README.md` — the bake CLI, the two decisions (`--control_freq`,
  `--control_space`), the label caveats.
- `eval/README.md` — `load_sim`, open-loop replay certification, and the
  two-terminal closed-loop `lerobot-eval` recipe.
