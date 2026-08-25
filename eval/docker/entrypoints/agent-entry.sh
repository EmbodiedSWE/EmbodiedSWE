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

# Granted-but-unused tools, detected from the workspace itself (file-driven: this script
# knows nothing about conditions — a tool is granted iff its module is in /task/tools).
# The reminder rides the keep-going nudge, once per leg, only while the tool goes unused.
tool_nudges() {
  [ -d /task/tools ] || return 0
  if [ -f /task/tools/checkpoint_tree.py ] && [ ! -d /workspace/.checkpoints ]; then
    printf '\n%s' 'You have the checkpoint_tree tool but have never saved a state: every stage you reach is one crash away from being re-derived from scratch. Save reached stages (from checkpoint_tree import CheckpointTree; tree = CheckpointTree(env); tree.save("stage")) and restore them in later scripts with tree.goto(id) instead of replaying your way back.'
  fi
  # The inverse reminder, for agents that ARE using checkpoints (usage-quality, not
  # adoption): iterating from restored states is exploration, but the graded artifact runs
  # from a cold reset — so from-reset validation must happen throughout, not at the end.
  if [ -f /task/tools/checkpoint_tree.py ] && [ -d /workspace/.checkpoints ]; then
    printf '\n%s' 'Reminder: before storing a state, check its quality and think through whether continuing from it is really helpful — avoid storing low-quality states or states with clear issues. A checkpoint may be suboptimal or broken in itself — when attempts from one keep failing, check whether it is really where you want to start from. And you will ultimately be graded by a full end-to-end program running from a fresh reset, so aside from building off checkpoints, actively validate your full end-to-end program as you go.'
  fi
  if [ -f /task/tools/parameter_search.py ] && \
     ! grep -rlq --include='*.py' --exclude-dir=tmp --exclude-dir=.checkpoints \
         'parameter_search' /workspace 2>/dev/null; then
    printf '\n%s' 'No script of yours uses the parameter_search tool yet. You should actively use parameter search if some part depends on hand-picked constants (offsets, depths, angles, timings) — the search runs async in parallel on the spare GPU and will not affect your other work, and the result is never worse than what you pass as seed_values.'
  fi
  if [ -f /task/tools/assessment.py ] && [ ! -f /workspace/.assessments/reviews.jsonl ]; then
    printf '\n%s' 'You have recorded no run reviews. After each run that moved the world, write down what happened (from assessment import assess) — history() is how a later script recalls what already failed, so you do not attempt it twice.'
  fi
}
# SUCCESS_CHECK: a command that answers "is the delivered solution done?" — exit 0
# solved, anything else not. Set it and the loop becomes keep-working-until-success;
# leave it unset and the loop runs to budget as before. ONLY exit 0 stops the run, so
# a check that cannot evaluate the scene (or that crashes) keeps the agent working.
# Adapter-independent on purpose: every harness is judged by the same predicate.
solved() {
  [ -n "${SUCCESS_CHECK:-}" ] || return 1
  # Each check's output also lands in its own file: the between-legs message quotes it, so
  # the agent learns what the graded fresh-reset run of its delivered solution actually did
  # (previously this measurement was computed and then buried in verify.log unannounced).
  sh -c "$SUCCESS_CHECK" > /workspace/.agent/last_check.log 2>&1
  rc=$?
  cat /workspace/.agent/last_check.log >> /workspace/.agent/verify.log
  echo "success check rc=$rc $(date -Is)" >> /workspace/.agent/legs.log
  [ "$rc" -eq 0 ]
}

# The just-failed check's verdict and output tail, for the next leg's payload. Only
# meaningful right after solved() returned non-zero (the keep-going loop below).
check_feedback() {
  [ -s /workspace/.agent/last_check.log ] || return 0
  printf '%s\n%s\n(full output: /workspace/.agent/verify.log)' \
    'The official success check just ran your delivered solution from a fresh reset and it did not pass. Its final lines:' \
    "$(tail -20 /workspace/.agent/last_check.log)"
}

# Each adapter defines run_leg "$payload" "$resume_flag"; the leg loop below is shared, so
# two harnesses differ only in the agent they start — not in when the run stops, what judges
# it, or what it leaves behind.
case "$AGENT" in
  claude)
    # Granted tools + binding rules ride the SYSTEM prompt on every leg (user directive
    # 2026-07-25, restored 2026-07-31): compaction preserves the system block structurally,
    # while a first user message can be summarized away hours into a long run. File-driven:
    # exactly what the condition installed under /task is what gets appended, re-read at
    # each leg so a mid-run update to /task takes effect on the next leg.
    system_extras() {
      cat /task/tools.md /task/rules/*.md 2>/dev/null
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
