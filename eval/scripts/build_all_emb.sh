#!/usr/bin/env bash
# Build the 12 cross-embodiment no_tools worlds (6 tasks x {gen3n7_panda, xarm7+panda-hand},
# osc control), sequentially — each build boot-validates its preset on the GPU.
# Mimics build_all_notools.sh verbatim; run ON a GPU box from the repo root with PY set to
# the Isaac venv python (defaults to .venv/bin/python like the original).
cd "$(dirname "$0")/../.." || exit 1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
PY="${PY:-.venv/bin/python}"

STAGES=(
  "allen_bolt_gen3n7:allen_bolt:gen3n7_panda:osc"
  "allen_bolt_xarm7:allen_bolt:xarm7:osc"
  "bulb_gen3n7:bulb:gen3n7_panda:osc"
  "bulb_xarm7:bulb:xarm7:osc"
  "pc_motherboard_gen3n7:pc_motherboard:gen3n7_panda:osc"
  "pc_motherboard_xarm7:pc_motherboard:xarm7:osc"
  "pc_ram_gen3n7:pc_ram:gen3n7_panda:osc"
  "pc_ram_xarm7:pc_ram:xarm7:osc"
  "tool_packing_gen3n7:packing.tool_packing:gen3n7_panda:osc"
  "tool_packing_xarm7:packing.tool_packing:xarm7:osc"
  "spatula_gen3n7:puzzle.spatula:gen3n7_panda:osc"
  "spatula_xarm7:puzzle.spatula:xarm7:osc"
)

fail=0
for entry in "${STAGES[@]}"; do
  name="${entry%%:*}"
  stage="${entry#*:}"
  if [ -f "experiments/${name}_notools/resolved.json" ]; then
    echo "=== ${name}_notools already built, skipping ==="
    continue
  fi
  echo "=== building ${name}_notools  (stage ${stage}) ==="
  "$PY" eval/scripts/build_env.py \
      --name "${name}_notools" --stage "${stage}" --config no_tools \
      || { echo "BUILD_FAILED: ${name}"; fail=1; }
done
if [ "$fail" -eq 0 ]; then echo "ALL_BUILDS_OK"; else echo "SOME_BUILDS_FAILED"; fi
