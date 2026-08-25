#!/usr/bin/env bash
# Build the 13-task no_tools experiment set (one world per task, franka.osc except the
# two bimanual tasks), sequentially — each build boot-validates its preset on the GPU.
# Run ON a GPU box from the repo root:  bash eval/scripts/build_all_notools.sh
cd "$(dirname "$0")/../.." || exit 1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

STAGES=(
  "allen_bolt:allen_bolt:franka"
  "bulb:bulb:franka"
  "ikea_table:ikea_table:bimanual_franka"
  "nut_thread:nut_thread:franka"
  "pc_gpu:pc_gpu:franka"
  "pc_gpu_ram:pc_gpu_ram:franka"
  "pc_motherboard:pc_motherboard:franka"
  "pc_ram:pc_ram:franka"
  "pen_holder:packing.pen_holder:franka"
  "tool_packing:packing.tool_packing:franka"
  "spatula:puzzle.spatula:franka"
  "syringe:puzzle.syringe:bimanual_franka"
  "coffee:puzzle.coffee:franka"
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
  .venv/bin/python eval/scripts/build_env.py \
      --name "${name}_notools" --stage "${stage}" --config no_tools \
      || { echo "BUILD_FAILED: ${name}"; fail=1; }
done
if [ "$fail" -eq 0 ]; then echo "ALL_BUILDS_OK"; else echo "SOME_BUILDS_FAILED"; fi
