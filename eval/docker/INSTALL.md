# Host setup — docker + NVIDIA container toolkit (Ubuntu 26.04, checked 2026-07-21)

Run these yourself (they need your sudo password). In a Claude Code session,
prefix each with `!` to run it interactively.

## 1. Docker engine (Ubuntu archive build — most reliable on a brand-new release)

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
```

The group change needs a NEW login shell: log out/in, or use `newgrp docker` in
the current shell (or prefix docker commands with `sudo` until then).

## 2. NVIDIA container toolkit (GPU passthrough)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. Sanity checks (no sudo once the group is active)

```bash
docker run --rm hello-world                       # engine works
docker run --rm --gpus all ubuntu nvidia-smi      # GPU visible in a container
```

`nvidia-smi` showing the RTX 5090 from inside the container is the green light.

## 2b. CDI spec (REQUIRED for rendering — Vulkan/RTX inside containers)

Legacy `--gpus` injection misses the graphics pieces (measured 2026-07-21:
Vulkan dies with ERROR_INCOMPATIBLE_DRIVER). Generate the CDI spec once:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

All our run configs use `--device nvidia.com/gpu=N` (CDI). **Re-run this
command after every NVIDIA driver upgrade** — the spec pins exact driver-file
versions and goes stale otherwise.

## 4. Then, from eval/docker/:

```bash
make test-image     # build the tiny test image (~2-4 min)
make test-gpu       # nvidia-smi inside our image
make test-claude    # Claude Code answers a prompt headless inside the container
                    #   (needs ANTHROPIC_API_KEY in your host env)
```

After all four pass, the real L0 Isaac bake is `make l0` (30–60 min) → `make warm`.
