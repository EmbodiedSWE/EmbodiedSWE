# Relaunch IK-campaign launchers that died WITHOUT ever acquiring a pod
# (RunPod capacity 500s). A task whose log ever shows "up at" is never touched:
# its launcher either runs or completed a real run.
cd /Users/bytedance/Desktop/CoSiGen
TASKS="allen_bolt bulb ikea_table nut_thread pc_gpu pc_gpu_ram pc_motherboard pc_ram pen_holder tool_packing coffee spatula syringe"
while true; do
  pending=0
  for t in $TASKS; do
    log=/tmp/ik1_${t}.log
    if grep -aq "up at" "$log" 2>/dev/null; then continue; fi
    alive=$(pgrep -f "run_agent_runpod.py experiments/${t}_ik" | wc -l | tr -d ' ')
    pending=$((pending+1))
    if [ "$alive" = "0" ]; then
      # a launcher that died before getting a pod leaves a stale pre-assembled run dir;
      # the next launch dies instantly on "refusing to overwrite" unless it is removed.
      # Only wipe when it holds no pod artifacts (never got a pod: no run.json).
      d=experiments/${t}_ik/runs/gpt_5_6_sol_ik1
      if [ -d "$d" ] && [ ! -f "$d/run.json" ] && [ ! -f "$d/agent_home.tgz" ]; then
        rm -rf "$d"
      fi
      echo "$(date +%H:%M) relaunching $t"
      nohup caffeinate -dims python3 -u eval/scripts/run_agent_runpod.py experiments/${t}_ik \
        --config default --agent codex --model gpt-5.6-sol --force-model gpt-5.6-sol \
        --budget-min 240 --keep-going --auto-submit-min 30 \
        --run gpt_5_6_sol_ik1 --gpu-count 2 \
        --ssh-key ~/.ssh/cosigen_campaign --gpt-via-bridge \
        >> "$log" 2>&1 &
      sleep 30
    fi
  done
  if [ "$pending" = "0" ]; then echo "ALL_PODS_ACQUIRED"; break; fi
  echo "$(date +%H:%M) pending pods: $pending"
  sleep 300
done
