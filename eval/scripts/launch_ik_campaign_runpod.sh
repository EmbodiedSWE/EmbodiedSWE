# IK campaign: 13 tasks x gpt-5.6-sol x tools (config default), diff_ik experiments.
# EXACT mimic of the proven rtools1 launch command; only exp dir + run name differ.
cd /Users/bytedance/Desktop/CoSiGen
for task in allen_bolt bulb ikea_table nut_thread pc_gpu pc_gpu_ram pc_motherboard pc_ram pen_holder tool_packing coffee spatula syringe; do
  nohup caffeinate -dims python3 -u eval/scripts/run_agent_runpod.py experiments/${task}_ik \
    --config default --agent codex --model gpt-5.6-sol --force-model gpt-5.6-sol \
    --budget-min 240 --keep-going --auto-submit-min 30 \
    --run gpt_5_6_sol_ik1 --gpu-count 2 \
    --ssh-key ~/.ssh/cosigen_campaign --gpt-via-bridge \
    > /tmp/ik1_${task}.log 2>&1 &
  echo "launched ${task} pid $!"
  sleep 20   # stagger pod creation to avoid RunPod rate limits
done
