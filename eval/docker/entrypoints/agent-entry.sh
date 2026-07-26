#!/usr/bin/env bash
# Agent-phase entry. Prepares /workspace as root, then drops to the non-root
# `agent` user and launches the chosen CLI. The agent never has root.
# NOTE: ablation-arm patches (e.g. disabling set_states) are applied at build
# time in the mounted /bench tree (eval/envbuild) — the mount is read-only,
# so nothing is patched here. /task holds the per-run prompt folder
# (instructions + task + rules + hints); instructions.md is the entry point.
set -euo pipefail

mkdir -p /workspace/solution /workspace/.agent
chown -R agent:agent /workspace

PROMPT_FILE=/task/instructions.md
[ -f "$PROMPT_FILE" ] || { echo "missing $PROMPT_FILE" >&2; exit 64; }

# Transcript lands in the bind-mounted workspace → survives kill.
# Wall-clock budget is enforced OUTSIDE (harness: docker compose stop -t 30).
# AGENT selects the adapter (claude | codex); MODEL optionally overrides.
AGENT=${AGENT:-claude}
case "$AGENT" in
  claude)
    exec runuser -u agent -- env \
        HOME=/home/agent \
        XDG_CACHE_HOME=/ovcache \
      claude -p "$(cat "$PROMPT_FILE")" \
        --output-format stream-json --verbose \
        ${MODEL:+--model "$MODEL"} \
        --allowedTools Bash Edit Write Read Glob Grep LS \
        ${MAX_TURNS:+--max-turns "$MAX_TURNS"} \
        > /workspace/.agent/transcript.jsonl \
        2> /workspace/.agent/stderr.log
    ;;
  codex)
    exec runuser -u agent -- env \
        HOME=/home/agent \
        XDG_CACHE_HOME=/ovcache \
      codex exec --sandbox danger-full-access --skip-git-repo-check \
        ${MODEL:+-m "$MODEL"} \
        "$(cat "$PROMPT_FILE")" \
        > /workspace/.agent/transcript.txt \
        2> /workspace/.agent/stderr.log
    ;;
  *) echo "unknown AGENT=$AGENT (claude|codex)" >&2; exit 64 ;;
esac
