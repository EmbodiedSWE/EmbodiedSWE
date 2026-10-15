#!/usr/bin/env bash

# Build-and-step every given task binding with the generic smoke launcher, one Kit at a time, and
# write a results table. The branch's stand-in for CI: a quick "does each registered task still
# build and step on this Isaac Sim / Isaac Lab stack" answer.
#
#   scripts/smoke_matrix.sh [--steps N] [--timeout SEC] [--out DIR] [--envs-file FILE] [ENV ...]
#   scripts/smoke_matrix.sh --modules-file FILE [--timeout SEC] [--out DIR]
#
# ENV names are registered env configs (`python -m robobench.scripts.smoke --list`); --envs-file is
# one name per line (# comments allowed). --modules-file lists suite CAPABILITY smokes instead
# (e.g. robobench.suites.assembly.smokes.bulb_assembly_smoke), run as `python -m MODULE --headless`;
# their verdict line decides: PASS = exit 0 and no `FAIL` verdict. Results: <out>/results.tsv
# (name, status, seconds, note) + <out>/results.md, logs in <out>/logs/<name>.log. Status: PASS
# (built + stepped, exit 0), FAIL (traceback / non-zero exit / FAIL verdict), TIMEOUT. The
# interpreter is $PYTHON or ./.venv/bin/python.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
STEPS=300
TIMEOUT=600
OUT="$REPO_ROOT/experiments/smoke_matrix/$(date +%Y%m%d-%H%M%S)"
ENVS=()
MODE=env

while (($#)); do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --envs-file) while IFS= read -r line; do line="${line%%#*}"; line="${line// /}"; [[ -n "$line" ]] && ENVS+=("$line"); done < "$2"; shift 2 ;;
        --modules-file) MODE=module; while IFS= read -r line; do line="${line%%#*}"; line="${line// /}"; [[ -n "$line" ]] && ENVS+=("$line"); done < "$2"; shift 2 ;;
        -h|--help) sed -n 3,14p "$0"; exit 0 ;;
        *) ENVS+=("$1"); shift ;;
    esac
done
(( ${#ENVS[@]} )) || { echo "no env names given (see --help)" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "python not found at $PYTHON (set PYTHON=...)" >&2; exit 2; }

mkdir -p "$OUT/logs"
TSV="$OUT/results.tsv"
printf 'env\tstatus\tseconds\tnote\n' > "$TSV"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$REPO_ROOT"

pass=0; fail=0; tmo=0
for env in "${ENVS[@]}"; do
    log="$OUT/logs/$env.log"
    start=$(date +%s)
    printf '[matrix] %s ... ' "$env"
    if [[ "$MODE" == module ]]; then
        timeout --signal=KILL "$TIMEOUT" "$PYTHON" -m "$env" --headless > "$log" 2>&1
    else
        timeout --signal=KILL "$TIMEOUT" "$PYTHON" -m robobench.scripts.smoke --env "$env" --steps "$STEPS" > "$log" 2>&1
    fi
    rc=$?
    secs=$(( $(date +%s) - start ))
    verdict="$(grep -E '\b(PASS|FAIL)\b' "$log" | grep -v 'SMOKE FAILED' | tail -1 | cut -c1-120)"
    if (( rc == 137 )) && (( secs >= TIMEOUT - 1 )); then
        status=TIMEOUT; note="killed after ${TIMEOUT}s"; tmo=$((tmo + 1))
    elif [[ "$MODE" == module ]] && (( rc == 0 )) && ! grep -qE '\bFAIL\b|SMOKE FAILED' "$log"; then
        status=PASS; note="$verdict"; pass=$((pass + 1))
    elif [[ "$MODE" == env ]] && (( rc == 0 )) && grep -q "^BUILT " "$log"; then
        status=PASS; note="$(grep -oE '[0-9.]+ fps/env' "$log" | tail -1)"; pass=$((pass + 1))
    else
        status=FAIL; fail=$((fail + 1))
        note="$(grep -E '^[A-Za-z_.]*(Error|Exception)[:]|^SystemExit|Segmentation fault|CUDA' "$log" | tail -1 | cut -c1-160)"
        [[ -n "$note" ]] || note="${verdict:-exit $rc (see log)}"
    fi
    printf '%s (%ss) %s\n' "$status" "$secs" "$note"
    printf '%s\t%s\t%s\t%s\n' "$env" "$status" "$secs" "$note" >> "$TSV"
done

{
    echo "# smoke matrix — $(date -Iseconds)"
    echo
    echo "steps=$STEPS timeout=${TIMEOUT}s python=$PYTHON"
    echo
    echo "| env | status | s | note |"
    echo "|---|---|---:|---|"
    tail -n +2 "$TSV" | awk -F'\t' '{printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4}'
    echo
    echo "PASS $pass  FAIL $fail  TIMEOUT $tmo  (of ${#ENVS[@]})"
} > "$OUT/results.md"
echo
echo "[matrix] PASS $pass  FAIL $fail  TIMEOUT $tmo  (of ${#ENVS[@]}) -> $OUT/results.md"
(( fail == 0 && tmo == 0 ))
