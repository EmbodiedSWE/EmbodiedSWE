# IK campaign: 13 tasks x gpt-5.6-sol x tools (config default), diff_ik experiments.
# 13 pre-reserved secure-cloud 2xA40 pods in EU-SE-1. The agent command is the exact
# proven rtools1 command; only the experiment, run label, GPU substrate, and attached pod differ.
cd /Users/bytedance/Desktop/CoSiGen

export RB_GPU_TYPE="NVIDIA A40"
export RB_DATACENTER="EU-SE-1"
export RB_NETWORK_VOLUME_ID=""
export RB_SNAPSHOT_SOURCE_HOST="213.173.110.226"
export RB_SNAPSHOT_SOURCE_PORT="23073"
export RB_SNAPSHOT_TRANSFER_KEY="/tmp/cosigen_snapshot_xfer"
export RB_SNAPSHOT_SHA256="edf73c34796024583a7724d593297be3b513fe403a47e505dbf810321c7c105e"

pod_for() {
  case "$1" in
    allen_bolt)     echo ogcg5g5h8mp30t ;;
    bulb)           echo vkc5ay2a0v4i5q ;;
    ikea_table)     echo 9nttbvmvlnhg9i ;;
    nut_thread)     echo dz6odwsi8khuka ;;
    pc_gpu)         echo 9bbzhvuj3264gp ;;
    pc_gpu_ram)     echo 6h3igpldz35uod ;;
    pc_motherboard) echo qdggmu6ohwncsf ;;
    pc_ram)         echo mc549qe0du2rg2 ;;
    pen_holder)     echo 6lfh44sk5ud1uk ;;
    tool_packing)   echo urxbiuosrjdxay ;;
    coffee)         echo k4lh1qpr70q3rw ;;
    spatula)        echo zjty8fbvx8p88i ;;
    syringe)        echo kmfofydn1dqvpe ;;
  esac
}

for task in allen_bolt bulb ikea_table nut_thread pc_gpu pc_gpu_ram pc_motherboard pc_ram pen_holder tool_packing coffee spatula syringe; do
  pod=$(pod_for "$task")
  nohup caffeinate -dims python3 -u eval/scripts/run_agent_runpod.py experiments/${task}_ik \
    --config default --agent codex --model gpt-5.6-sol --force-model gpt-5.6-sol \
    --budget-min 240 --keep-going --auto-submit-min 30 \
    --run gpt_5_6_sol_ik_a40_s1 --gpu-count 2 \
    --pod "$pod" --terminate-attached \
    --ssh-key ~/.ssh/cosigen_campaign --gpt-via-bridge \
    > /tmp/ik_a40_${task}.log 2>&1 &
  echo "launched ${task} on ${pod}, pid $!"
  sleep 5
done
