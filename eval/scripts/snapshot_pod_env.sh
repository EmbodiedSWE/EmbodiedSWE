#!/usr/bin/env bash
# Capture the template pod's environment as the campaign snapshot on the network volume.
# Run ON the template pod (as root) after the environment is complete:
#   bash /opt/cosigen/CoSiGen/eval/scripts/snapshot_pod_env.sh [/snapshot/cosigen_env2.tar.zst]
#
# What the snapshot IS: the L0+L1 image contents, captured once — the Isaac venv, the repo,
# node + the pinned agent CLIs, the relay venv, and the warmed omniverse/kit caches (incl.
# the PhysX SDF cooks, which take 10+ min cold for the pc scenes). What it is NOT: any run
# state — agent homes, workspaces, relay logs and built experiments are excluded, because a
# run pod gets those per-run from the launcher.
OUT="${1:-/snapshot/cosigen_env2.tar.zst}"
command -v zstd >/dev/null || { apt-get update -qq && apt-get install -y -qq zstd; }

KIT=opt/cosigen/.venv/lib/python3.11/site-packages/isaacsim/kit
tar -I "zstd -T0 -8" -cf "$OUT.partial" -C / \
  --exclude="opt/cosigen/CoSiGen/experiments" \
  --exclude="opt/cosigen/CoSiGen/sim_gen/super_relay/logs" \
  --exclude="$KIT/data/*" --exclude="$KIT/logs/*" \
  opt/cosigen opt/node-v22.11.0-linux-x64 opt/npm opt/relay-venv \
  root/.cache root/.nv root/.nvidia-omniverse \
  || { echo "SNAPSHOT_FAILED"; rm -f "$OUT.partial"; exit 1; }

# Prove the archive reads end to end before it becomes the snapshot (a truncated archive
# that restores "successfully" would hand every run a broken interpreter).
tar -I zstd -tf "$OUT.partial" > /dev/null || { echo "SNAPSHOT_UNREADABLE"; exit 1; }
mv "$OUT.partial" "$OUT"
ls -la "$OUT"
echo "SNAPSHOT_OK"
