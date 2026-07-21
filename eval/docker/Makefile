# robobench eval docker. Usage: README.md · host setup: INSTALL.md
GPU           ?= 0
CC_VERSION    ?=                       # pin: npm view @anthropic-ai/claude-code version
CODEX_VERSION ?=                       # optional: also bake codex into L1
L0_IMAGE      ?= rb-l0-isaaclab:5.1.0  # sim backend base; rb-l0-newton:<tag> once it exists

REPO  := $(abspath ../..)
UV_PY := $(HOME)/.local/share/uv
KIT   := $(REPO)/.venv/lib/python3.11/site-packages/isaacsim/kit

.PHONY: test-image test-gpu test-claude test-codex dev l0 warm l1

# ---- plumbing tests (small image, minutes) --------------------------------------
test-image:
	docker build -f Dockerfile.test -t rb-test .

test-gpu:
	docker run --rm --device nvidia.com/gpu=$(GPU) rb-test nvidia-smi

test-claude:
	docker run --rm -e ANTHROPIC_API_KEY -e CLAUDE_CODE_OAUTH_TOKEN rb-test \
	  claude -p "Reply with exactly: CONTAINER_OK" --output-format text

test-codex:
	docker run --rm -e OPENAI_API_KEY rb-test \
	  codex exec --sandbox danger-full-access --skip-git-repo-check \
	  "Reply with exactly: CONTAINER_OK"

# ---- dev shell: LIVE repo + host .venv in a container (no bake needed) ----------
# TRANSITIONAL (README §2b) — pre-bake bootstrap, scheduled for deletion once
# workflows move onto the baked images. Don't build on it.
# Repo + uv python mounted at their HOST paths (venv symlinks / editable install
# keep working); runs/ is the only rw dir; tmpfs holes because Kit writes inside
# its own install (kit/{cache,data,logs}) and the venv mount is ro.
dev:
	mkdir -p $(REPO)/runs
	docker volume create rb-ovcache
	docker run --rm -it --device nvidia.com/gpu=$(GPU) --shm-size 2g \
	  -v $(REPO):$(REPO):ro \
	  -v $(REPO)/runs:$(REPO)/runs \
	  -v $(UV_PY):$(UV_PY):ro \
	  -v rb-ovcache:/ovcache \
	  --tmpfs $(KIT)/cache:rw,size=6g,uid=1000,gid=1000 \
	  --tmpfs $(KIT)/data:rw,size=1g,uid=1000,gid=1000 \
	  --tmpfs $(KIT)/logs:rw,size=1g,uid=1000,gid=1000 \
	  -e XDG_CACHE_HOME=/ovcache \
	  -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
	  -w $(REPO) \
	  rb-test bash

# ---- the real images (big, build once, layer-cached) ----------------------------
l0:
	docker build -f Dockerfile.l0-isaaclab -t rb-l0-isaaclab:5.1.0 .

# One-time boot of the baked venv: verifies L0 + warms the shared cache volume.
warm:
	docker volume create rb-ovcache
	docker run --rm --device nvidia.com/gpu=$(GPU) --shm-size 2g \
	  -v rb-ovcache:/ovcache rb-l0-isaaclab:5.1.0 \
	  python -c "\
	from isaaclab.app import AppLauncher; \
	app = AppLauncher(headless=True).app; \
	import isaaclab.sim as sim_utils; \
	print('warm ok'); app.close()"

# Agent image: CLIs on the Isaac base, NO benchmark inside (mounted per-run).
l1:
	docker build -f Dockerfile.l1-agent \
	  --build-arg L0_IMAGE=$(L0_IMAGE) \
	  --build-arg CLAUDE_CODE_VERSION=$(CC_VERSION) \
	  --build-arg CODEX_VERSION=$(CODEX_VERSION) \
	  -t rb-l1-agent:$(CC_VERSION) .
