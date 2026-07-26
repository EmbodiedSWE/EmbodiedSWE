#!/usr/bin/env bash
# Interactive-session entry: the agent CLI runs in a detached tmux session so
# the container needs no TTY of its own. A human attaches/detaches at will:
#     docker exec -it <container> runuser -u agent -- tmux attach -t agent
# The container stays up until the tmux session ends (agent CLI exits) or the
# container is stopped. Permissions are skipped: the sandbox IS the guardrail.
set -euo pipefail

mkdir -p /workspace/solution /workspace/.agent
chown -R agent:agent /workspace

PROMPT_FILE=/task/instructions.md
[ -f "$PROMPT_FILE" ] || { echo "missing $PROMPT_FILE" >&2; exit 64; }

cat > /opt/entrypoints/_interactive_start.sh <<'INNER'
#!/usr/bin/env bash
cd /workspace
exec claude --dangerously-skip-permissions ${MODEL:+--model "$MODEL"} "$(cat /task/instructions.md)"
INNER
chmod 755 /opt/entrypoints/_interactive_start.sh

runuser -u agent -- env \
    HOME=/home/agent \
    XDG_CACHE_HOME=/ovcache \
    MODEL="${MODEL:-}" \
  tmux new-session -d -s agent -x 220 -y 50 "bash /opt/entrypoints/_interactive_start.sh"

echo "INTERACTIVE_READY — attach: docker exec -it <container> runuser -u agent -- tmux attach -t agent"
# keep the container alive while the tmux session exists
while runuser -u agent -- tmux has-session -t agent 2>/dev/null; do sleep 5; done
echo "tmux session ended — exiting"
