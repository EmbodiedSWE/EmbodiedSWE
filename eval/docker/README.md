# robobench eval — docker

Run coding agents (e.g. Claude Code, Codex) against robobench inside isolated
containers. The benchmark code is **never baked into an image**: each
experiment mounts a minimal extracted tree — built and boot-validated by
`eval/scripts/build_env.py` — read-only at `/bench`, so what the agent sees
is decided per-experiment. Architecture diagrams: `eval/README.html`.

```
rb-l0-isaaclab:5.1.0    Isaac Sim 5.1 + Isaac Lab 2.3.2 baked venv    ~35 GB, build once
└─ rb-l1-agent:<ccver>  + node & pinned agent CLIs + agent user       small layer on top
rb-test                 plumbing/dev image (no Isaac)                 ~2 GB
```

## 0. One-time host setup

Follow `INSTALL.md` (docker + nvidia-container-toolkit + **CDI spec** — CDI is
mandatory: legacy `--gpus` injection breaks Vulkan/RTX). Then:

```bash
make test-image && make test-gpu && make test-claude
```

## 1. Build the real images

```bash
make l0                                        # 30–60 min, once
make warm                                      # boots the baked venv, warms shader cache
make l1 CC_VERSION=$(npm view @anthropic-ai/claude-code version)   # minutes
```

## 2. Run an experiment — the container contract

This folder fixes the container contract; anything honoring it can launch runs
(`eval/scripts/run_agent.py` is the reference single-run launcher —
higher-level sequencing/orchestration stays user-side). A run is: start
`rb-l1-agent:<tag>` with entrypoint `/opt/entrypoints/agent-entry.sh` and:

| you provide | as |
|---|---|
| benchmark code | minimal tree built by `eval/scripts/build_env.py`, mounted ro at `/bench` (`PYTHONPATH=/bench` is baked into the image). "Minimal" = no other suites, no `smokes/` (they demonstrate scene-tampering), only the assets the scene/robots need. Ablation patches (e.g. disabling `set_states`) are applied at build time — the mount is read-only. Asset detection is a heuristic, so every bundle is proven by a headless boot of its registered preset before it ships |
| task instruction | `/task/task_prompt.md` (mount a dir read-only at `/task`) |
| workspace | host dir mounted rw at `/workspace` — the agent's ONLY writable dir; `solution/` and `.agent/{transcript,stderr}` land here live |
| GPU | `--device nvidia.com/gpu=N` (CDI) + `--shm-size 2g` |
| shader cache | `-v rb-ovcache:/ovcache` (warmed by `make warm`) |
| agent choice | env `AGENT=claude\|codex`, optional `MODEL=`, credential env (§3) |
| budget | enforced from outside: `docker stop -t 30` at your deadline |
| isolation | network topology from `compose.agent.yaml` (or your equivalent). The allowlist admits only the Anthropic API + PyPI; everything else — GitHub above all — is denied, because the public repo's history cites solve-calibrated configs |
| provenance | keep the extractor's MANIFEST (commit + tree hash) with each run so every result is attributable to exact code |

`compose.agent.yaml` wires all of this plus the egress allowlist:

```bash
RUN_DIR=… TASK_DIR=… BENCH_DIR=… AGENT_TAG=<ccver> docker compose -f compose.agent.yaml up
```

The builder behind `/bench` is `eval/scripts/build_env.py` (library:
`eval/envbuild/`): resolve CLI intent against robobench's registries (only
registered presets build — the feasibility gate) → extract the minimal tree →
apply arm patches → **boot-validate** the registered preset from the tree (no
bundle ships unbooted; `scene.describe()` is harvested for the prompt) →
render `/task` → write `resolved.json`, a re-runnable provenance receipt.
Planned on top (not built): a declarative per-experiment spec layer (e.g.
hydra composition for agent/base configs) and generated hint-doc tiers.

## 2b. Transitional — scheduled for DELETION

- **`make dev`** — mounts the HOST `.venv` + uv python into `rb-test` (tmpfs
  holes for Kit's in-venv writes). Pre-bake bootstrap; still handy for
  live-editing robobench with instant container feedback; dies when a
  baked-image dev workflow replaces it. Don't build on it.

## 3. Switching the agent (Claude Code ↔ Codex)

One env var; the entrypoint has an adapter per agent:

| | Claude Code (default) | Codex |
|---|---|---|
| select | `AGENT=claude` | `AGENT=codex` |
| credential (host env) | `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) or `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| model override | `MODEL=<id>` | `MODEL=<id>` |
| in rb-l1-agent | always (pin `CC_VERSION=`) | opt-in: `make l1 … CODEX_VERSION=<pin>` |
| smoke test | `make test-claude` | `make test-codex` |

Note: a subscription OAuth token draws on your personal usage limits and
reports no per-run dollar cost — use an API key for measured eval campaigns.

## 4. Switching the sim backend (Isaac ↔ Newton, e.g. the latte-art sim)

The agent image's base is a build parameter:

```bash
make l1 CC_VERSION=<pin> L0_IMAGE=rb-l0-newton:<tag>
```

**TODO (not built yet):** `Dockerfile.l0-newton` — mirror the l0-isaaclab
pattern with the `env_newton` stack (python 3.12 / isaacsim 6 / Newton-VBD).
Blocked on pinning that venv's package recipe. Needed before folding or
latte-art tasks run containerized.

## 5. Maintenance rules

- **After every NVIDIA driver upgrade**: `sudo nvidia-ctk cdi generate
  --output=/etc/cdi/nvidia.yaml` (the spec pins driver-file versions).
- Rebuilds are layer-cached: deps change → `make l0` (slow); agent-CLI pin
  changes → `make l1` (minutes). Benchmark changes need NO rebuild — re-run
  the extractor.
- Credentials only ever enter via `docker run -e` / compose env — never baked
  into images, never written into task files.

## TODO (docker-scoped; grading/eval strategy is out of scope here)

- [ ] Higher-level runner/sequencing scripts (user-side) against the §2
      contract — real experiments must use the compose network path.
- [ ] Delete `make dev` once a baked-image dev workflow replaces it (§2b).
- [ ] `Dockerfile.l0-newton` (§4) for Newton-backend suites (e.g. folding).
- [ ] Codex adapter has no live smoke yet (`make test-codex` needs an
      `OPENAI_API_KEY`).
- [ ] Confirm the RTX shader cache persists to `/ovcache` across runs (if
      first-render cost recurs, point `__GL_SHADER_DISK_CACHE_PATH` there).
