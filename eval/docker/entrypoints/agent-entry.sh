#!/usr/bin/env bash
# Agent-phase entry. Prepares /workspace as root, then drops to the non-root
# `agent` user and launches the chosen CLI. The agent never has root.
# NOTE: ablation-arm patches (e.g. disabling set_states) are applied at build
# time in the mounted /bench tree (eval/envbuild) — the mount is read-only,
# so nothing is patched here. /task holds the per-run prompt folder
# (instructions + task + rules + skills + concise router + full tool reference);
# instructions.md is the entry point.
set -euo pipefail

mkdir -p /workspace/solution /workspace/.agent
chown -R agent:agent /workspace

PROMPT_FILE=/task/instructions.md
[ -f "$PROMPT_FILE" ] || { echo "missing $PROMPT_FILE" >&2; exit 64; }

# Transcript lands in the bind-mounted workspace → survives kill.
# Wall-clock budget is enforced OUTSIDE (harness: docker compose stop -t 30).
# AGENT selects the adapter (claude | cosigen | codex); MODEL optionally overrides.
AGENT=${AGENT:-claude}
# KEEP_GOING=1: when the CLI returns before the harness stops the container, continue
# the same conversation instead of ending the run. `claude -p` exits as soon as the
# agent decides it is done, so without this the effort floor is only the prompt line
# telling it that stopping is final — most of a long budget goes unused. The transcript
# is APPENDED to across legs, so nothing recorded is lost, and legs.log marks the seams.
NUDGE='You stopped, but this session is still running and the task is not finished.
Nothing has been lost — your workspace, your files and the environment are as you left
them. Pick up where you were: check the current state of your work, then carry on. Keep
the best version you have in /workspace/solution/ and run `submit` whenever it improves.
Keep working without stopping until the task is solved.'

# Granted tools are detected from the workspace itself (file-driven: this script knows nothing
# about conditions — a tool is granted iff its module is in /task/tools). Reminders preserve
# agent autonomy: availability is never enough; each reminder also checks its eligibility gate.
assessment_requests_parameter_search() {
  [ -f /workspace/.assessments/reviews.jsonl ] || return 1
  grep -Eq '"constants_plan"[[:space:]]*:[[:space:]]*"searching"' \
    /workspace/.assessments/reviews.jsonl 2>/dev/null
}

parameter_search_artifact_exists() {
  local psearch_dir
  if grep -rlq --include='*.py' --exclude-dir=tmp --exclude-dir=.checkpoints \
      'parameter_search' /workspace 2>/dev/null; then
    return 0
  fi
  psearch_dir="${PSEARCH_DIR:-/workspace/.psearch}"
  compgen -G "$psearch_dir/*/result.json" >/dev/null
}

# The stage protocol is REQUIRED in the tools arm (2026-09-05): plan recorded in the tree, one
# boundary node per completed stage, later stages developed from those boundaries. The nudge
# reads the tree's manifest and says exactly which part of the protocol is missing.
stage_protocol_nudge() {
  [ -f /task/tools/checkpoint_tree.py ] || return 0
  local manifest=/workspace/.checkpoints/tree.json
  if [ ! -f "$manifest" ]; then
    printf '\n%s' 'Stage protocol: no checkpoint tree exists yet. Record your stage plan in the tool (tree.plan([...]); a single stage is a valid plan), write each stage as solution/stages/stage_<k>.py with run(env) and check(env), and develop each stage with tree.run_stage(k) so every completed stage gets a boundary node you can continue from. Every script still starts fresh unless it restores explicitly.'
    return 0
  fi
  python3 - "$manifest" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
plan = d.get("plan") or []
nodes = d.get("nodes") or {}
if not plan:
    print("\nStage protocol: the checkpoint tree has no recorded plan. Call tree.plan([...]) with your stages (one stage is a valid plan), then develop each stage with tree.run_stage(k) so its boundary is saved when your check passes.", end="")
    sys.exit()
done = {n.get("stage") for n in nodes.values() if n.get("stage") and n.get("reusable", True)}
missing = [f"{i} ({s})" for i, s in enumerate(plan, 1) if i not in done]
if missing:
    print(f"\nStage protocol: plan has {len(plan)} stage(s); boundaries exist for {sorted(done) or 'none'}; still without a boundary: {', '.join(missing)}. Develop the next stage with tree.run_stage(k) from the previous boundary (goto_stage), and save its boundary when your check passes -- do not re-derive earlier stages from scratch.", end="")
else:
    print("\nStage protocol: every planned stage has a boundary. Compose the stages into solution/solve.py, run the integrated program from a fresh reset, and submit; if the verifier fails, go back to the boundary of the failing stage rather than starting over.", end="")
EOF
}

tool_nudges() {
  [ -d /task/tools ] || return 0
  stage_protocol_nudge
  # A parameter-search reminder is warranted only after the agent explicitly records that a
  # viable maneuver's constants are ready for search, and only while no script/result exists.
  if [ -f /task/tools/parameter_search.py ] && \
     assessment_requests_parameter_search && ! parameter_search_artifact_exists; then
    printf '\n%s' 'Your assessment explicitly set constants_plan="searching", but no parameter-search script or result exists. If the maneuver is still a stable viable seed, launch the promised numeric tuning; if that gate no longer holds, update the assessment instead. Treat the winner as provisional until integrated replay and a 2–3-instance holdout.'
  fi
  if [ -f /task/tools/assessment.py ] && [ ! -f /workspace/.assessments/reviews.jsonl ]; then
    printf '\n%s' 'You have recorded no run reviews. After each run that moved the world, record successes and failures (from assessment import assess) — history() is how a later script recalls what already failed, so you do not attempt it twice.'
  fi
}
# Report whether verification would collide with live GPU work. nvidia-smi's compute-app query
# names only processes that own a GPU compute context, so relay/CLI processes that merely exist
# are not blockers. The parameter-search registry separately catches a detached search while it
# is booting, before it acquires a GPU context. Status uncertainty is reported and deferred
# rather than gambling two Isaac processes against one another.
verification_blockers() {
  local gpu_output gpu_rc gpu_active registry_path registry_output registry_rc
  gpu_rc=0
  gpu_output="$(nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>&1)" || gpu_rc=$?
  gpu_active=0
  if [ "$gpu_rc" -eq 0 ] && \
     printf '%s\n' "$gpu_output" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*,'; then
    gpu_active=1
  fi

  registry_path="${PSEARCH_DIR:-/workspace/.psearch}/registry.json"
  registry_rc=0
  registry_output="$(python3 - "$registry_path" 2>&1 <<'PY'
import json
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    entries = json.loads(path.read_text())
except Exception as exc:
    print(f"could not read {path}: {exc!r}")
    raise SystemExit(2)
if not isinstance(entries, list):
    print(f"registry is not a list: {path}")
    raise SystemExit(2)

active = []
invalid = []
for entry in entries:
    if not isinstance(entry, dict):
        invalid.append({"entry": entry, "error": "entry is not an object"})
        continue
    try:
        pid = int(entry["pid"])
    except (KeyError, TypeError, ValueError) as exc:
        invalid.append({"entry": entry, "error": f"invalid pid: {exc!r}"})
        continue

    state = ""
    command = ""
    argv = []
    proc = Path(f"/proc/{pid}")
    try:
        stat = (proc / "stat").read_text()
        state = stat.rsplit(")", 1)[-1].split()[0]
        argv = [
            arg.decode(errors="replace")
            for arg in (proc / "cmdline").read_bytes().split(b"\0")
            if arg
        ]
        command = " ".join(argv)
    except OSError:
        # Portable fallback makes the coordination check testable off Linux as well.
        seen = subprocess.run(
            ["ps", "-o", "stat=", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        row = seen.stdout.strip()
        if row:
            fields = row.split(None, 1)
            state = fields[0]
            command = fields[1] if len(fields) > 1 else ""
            argv = command.split()

    if not state or state.startswith("Z"):
        continue
    script = str(entry.get("script") or "")
    if script and script not in argv:
        # The PID was reused after this registry entry stopped; it is not this search.
        continue
    active.append({**entry, "process_state": state, "process_command": command})

if active:
    print(json.dumps(active, indent=2, ensure_ascii=False))
    raise SystemExit(0)
if invalid:
    print(json.dumps({"invalid_registry_entries": invalid}, indent=2, ensure_ascii=False))
    raise SystemExit(2)
raise SystemExit(1)
PY
)" || registry_rc=$?

  if [ "$gpu_rc" -eq 0 ] && [ "$gpu_active" -eq 0 ] && [ "$registry_rc" -eq 1 ]; then
    return 1
  fi

  printf '%s\n' 'verification blocker diagnostic'
  printf 'timestamp: %s\n' "$(date -Is)"
  printf 'nvidia-smi query rc=%s active_compute=%s\n' "$gpu_rc" "$gpu_active"
  if [ -n "$gpu_output" ]; then
    printf '%s\n' "$gpu_output"
  else
    printf '%s\n' '(no compute-app rows)'
  fi
  printf 'parameter-search registry: %s\n' "$registry_path"
  printf 'registry inspection rc=%s (0=active, 1=none, other=unreadable)\n' "$registry_rc"
  if [ -n "$registry_output" ]; then
    printf '%s\n' "$registry_output"
  else
    printf '%s\n' '(no active registered parameter-search PID)'
  fi
  return 0
}

# SUCCESS_CHECK: a command that answers "is the delivered solution done?" — exit 0 solved,
# anything else not. Before invoking it, defer while GPU compute or a registered detached search
# is active; no process is killed or cancelled. Set SUCCESS_CHECK and the loop becomes
# keep-working-until-success; leave it unset and the loop runs to budget as before. The verifier
# command itself remains adapter-independent and unchanged.
solved() {
  local blockers
  [ -n "${SUCCESS_CHECK:-}" ] || return 1
  if blockers="$(verification_blockers)"; then
    {
      printf '%s\n' \
        'The official fresh-reset success check was DEFERRED; live or unknown compute must clear first.'
      printf '%s\n' "$blockers"
    } > /workspace/.agent/last_check.log
    cat /workspace/.agent/last_check.log >> /workspace/.agent/verify.log
    {
      printf '%s\n' '----- verification deferred -----'
      cat /workspace/.agent/last_check.log
      printf '%s\n' '----- end verification deferral -----'
    } >> /workspace/.agent/legs.log
    : > /workspace/.agent/last_check.deferred
    return 1
  fi
  rm -f /workspace/.agent/last_check.deferred
  # Each check's output also lands in its own file: the between-legs message quotes it, so
  # the agent learns what the graded fresh-reset run of its delivered solution actually did
  # (previously this measurement was computed and then buried in verify.log unannounced).
  sh -c "$SUCCESS_CHECK" > /workspace/.agent/last_check.log 2>&1
  rc=$?
  cat /workspace/.agent/last_check.log >> /workspace/.agent/verify.log
  echo "success check rc=$rc $(date -Is)" >> /workspace/.agent/legs.log
  [ "$rc" -eq 0 ]
}

# The deferred-or-failed check's verdict for the next leg. Deferral diagnostics are passed in
# full. A failed verifier keeps the existing 20-line payload tail; its complete output remains
# appended to verify.log.
check_feedback() {
  [ -s /workspace/.agent/last_check.log ] || return 0
  if [ -f /workspace/.agent/last_check.deferred ]; then
    printf '%s\n%s\n(full diagnostic: /workspace/.agent/last_check.log; accumulated log: /workspace/.agent/verify.log)' \
      'The official fresh-reset success check is queued but was deferred. Do not kill or cancel the listed work; continue this leg and let it finish:' \
      "$(cat /workspace/.agent/last_check.log)"
    return 0
  fi
  printf '%s\n%s\n(full output: /workspace/.agent/verify.log)' \
    'The official success check just ran your delivered solution from a fresh reset and it did not pass. Its final lines:' \
    "$(tail -20 /workspace/.agent/last_check.log)"
}

# Each adapter defines run_leg "$payload" "$resume_flag"; the leg loop below is shared, so
# two harnesses differ only in the agent they start — not in when the run stops, what judges
# it, or what it leaves behind.
case "$AGENT" in
  claude)
    # The concise grant-aware router + binding rules ride the SYSTEM prompt on every Claude leg:
    # compaction preserves this block structurally while the initial user message can be
    # summarized. GPT receives the identical router in instructions.md. Full tools.md stays
    # on-demand for both adapters instead of consuming every Claude system prompt.
    system_extras() {
      local rule
      [ ! -f /task/tool_router.md ] || cat /task/tool_router.md
      for rule in /task/rules/*.md; do
        [ -f "$rule" ] || continue
        cat "$rule"
      done
      return 0
    }
    run_leg() {  # $1 = -p payload, $2 = extra flag (empty or --continue)
      local sys_extra
      sys_extra="$(system_extras)"
      runuser -u agent -- env \
          HOME=/home/agent \
          XDG_CACHE_HOME=/ovcache \
        claude ${2:+$2} -p "$1" \
          ${sys_extra:+--append-system-prompt "$sys_extra"} \
          --output-format stream-json --verbose \
          ${MODEL:+--model "$MODEL"} \
          --allowedTools Bash Edit Write Read Glob Grep LS \
          ${MAX_TURNS:+--max-turns "$MAX_TURNS"} \
          >> /workspace/.agent/transcript.jsonl \
          2>> /workspace/.agent/stderr.log
    }
    ;;
  codex)
    # run_leg, not `exec` (changed 2026-08-14): exec-ing the CLI made codex a single-leg
    # harness — KEEP_GOING, SUCCESS_CHECK and RESUME below never applied, so a codex run
    # ended the moment the CLI first returned while a claude run worked its whole budget.
    # The same shared leg loop now drives both: continuation legs resume the CLI's own last
    # session (`codex exec resume --last`), codex's equivalent of claude's --continue.
    run_leg() {  # $1 = prompt payload, $2 = extra flag (empty or --continue)
      runuser -u agent -- env \
          HOME=/home/agent \
          XDG_CACHE_HOME=/ovcache \
        codex exec --sandbox danger-full-access --skip-git-repo-check \
          ${MODEL:+-m "$MODEL"} \
          ${2:+resume --last} \
          "$1" \
          >> /workspace/.agent/transcript.txt \
          2>> /workspace/.agent/stderr.log
    }
    ;;
  *) echo "unknown AGENT=$AGENT (claude|cosigen|codex)" >&2; exit 64 ;;
esac

# RESUME=1: this container was handed a previous run's workspace AND that run's CLI state, so
# leg 1 continues that conversation instead of opening a new one on top of files the agent has
# no memory of writing. The payload is the keep-going nudge for the same reason it is used
# between legs — re-sending the full instructions would re-read as a new task.
# `|| rc=$?`, not a bare call: under `set -e` a non-zero exit from the agent would end this
# script here — no legs.log, no keep-going loop, the rest of the budget unused.
rc=0
if [ "${RESUME:-0}" = "1" ] && [ "${RESUME_NEW_TASK:-0}" = "1" ]; then
  # Fork-onto-a-new-task (added 2026-08-15): the restored conversation continues, but the
  # first leg delivers the NEW task's instructions (carryover variant) instead of the
  # keep-going nudge — the nudge would tell the agent to keep working the PREVIOUS task.
  echo "resuming the previous session onto a NEW task $(date -Is)" >> /workspace/.agent/legs.log
  run_leg "$(cat "$PROMPT_FILE")" --continue || rc=$?
elif [ "${RESUME:-0}" = "1" ]; then
  echo "resuming the previous session $(date -Is)" >> /workspace/.agent/legs.log
  run_leg "$NUDGE$(tool_nudges)" --continue || rc=$?
else
  run_leg "$(cat "$PROMPT_FILE")" "" || rc=$?
fi
echo "leg 1 rc=$rc $(date -Is)" >> /workspace/.agent/legs.log
# A run ends on exactly two conditions: the success check passes, or the wall-clock budget
# runs out. Nothing here infers that the agent has given up. An agent that returns
# immediately is simply resumed again — if it will not work, that is its own behaviour and
# belongs in the results; a rule that ended the run instead would decide run length on
# grounds unrelated to the task, so the same task would stop at different points for
# different agents and the comparison would be worthless.
if [ "${KEEP_GOING:-0}" = "1" ]; then
  leg=1
  fastfail=0
  while true; do
    if solved; then
      echo "the delivered solution solves the task — stopping" >> /workspace/.agent/legs.log
      break
    fi
    leg=$((leg + 1))
    t0=$SECONDS
    # ${fb:+...}: a blank line between the check feedback and the nudge, only when there is
    # feedback (command substitution strips trailing newlines, so the separator lives here).
    fb="$(check_feedback)"
    rc=0; run_leg "${fb:+$fb

}$NUDGE$(tool_nudges)" --continue || rc=$?
    echo "leg $leg rc=$rc seconds=$((SECONDS - t0)) resumed=1 $(date -Is)" \
      >> /workspace/.agent/legs.log
    # A leg that FAILS in under 10 s is the CLI not starting (bad config, dead relay),
    # not the agent deciding anything — unbraked, that spun 7500 legs in 6 minutes
    # (measured 2026-08-15). Pause between such legs, and after 20 in a row stop LOUDLY:
    # a run whose agent cannot start is a setup failure the operator must see, not a
    # budget quietly burned on a crash loop.
    if [ "$rc" -ne 0 ] && [ $((SECONDS - t0)) -lt 10 ]; then
      fastfail=$((fastfail + 1))
      if [ "$fastfail" -ge 20 ]; then
        echo "20 consecutive instant CLI failures — the agent cannot start; ending the run" \
          >> /workspace/.agent/legs.log
        exit 65
      fi
      sleep 15
    else
      fastfail=0
    fi
  done
fi
