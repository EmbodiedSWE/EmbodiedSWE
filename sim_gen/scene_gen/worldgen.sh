#!/bin/bash
# HY-World 2.0 worldgen driver: panorama -> 3D Gaussian splats + TSDF collider mesh.
#
#   SCENE_DIR=/path/scene RESULT_DIR=/path/results ./worldgen.sh
#
# Requires (see README.md "3D stage setup"):
#   HYWORLD2_ROOT   HY-World-2.0 checkout (github.com/Tencent-Hunyuan/HY-World-2.0)
#   HYWORLD2_PY     python of the worldgen env (torch cu128, pytorch3d, gsplat fork, ...)
#   VLLM_BIN        vllm executable (its own env is fine)
#   SCENE_DIR       directory containing panorama.png (1920x960 equirect)
#   RESULT_DIR      output directory (3DGS ply + fuse_*.ply mesh land in ply/)
# GPU-count adaptive; one node, 1-8 GPUs. Stage-5 step table follows the upstream README.
set -euo pipefail
: "${HYWORLD2_ROOT:?}" "${HYWORLD2_PY:?}" "${VLLM_BIN:?}" "${SCENE_DIR:?}" "${RESULT_DIR:?}"
LLM_NAME=${LLM_NAME:-Qwen/Qwen3-VL-8B-Instruct}
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
test -f "$SCENE_DIR/panorama.png" || { echo "missing $SCENE_DIR/panorama.png"; exit 1; }

VENV_BIN="$(dirname "$HYWORLD2_PY")"
# The pipeline shells out to a bare `torchrun` (WorldMirror subprocess): the worldgen
# env's bin must precede any other python env on PATH.
export PATH="$VENV_BIN:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$HYWORLD2_ROOT/hyworld2/worldgen"

NGPU=$(nvidia-smi -L | wc -l)
ALL=$(seq -s, 0 $((NGPU-1))); PA_N=$(( NGPU > 1 ? NGPU-1 : 1 )); PA=$(seq -s, 0 $((PA_N-1)))
case $NGPU in 8|7|6|5) STEPS=1500;; 4|3) STEPS=2000;; 2) STEPS=4000;; *) STEPS=8000;; esac
RS=$((150*STEPS/1500)); RE=$((750*STEPS/1500)); RV=$((100*STEPS/1500))
echo "NGPU=$NGPU phaseA=$PA all=$ALL stage5_steps=$STEPS"

# Phase A: VLM server on the last GPU; trajectory planning + rendering on the rest.
CUDA_VISIBLE_DEVICES=$((NGPU-1)) "$VLLM_BIN" serve "$LLM_NAME" \
    --served-model-name "$LLM_NAME" --port 8000 --host 127.0.0.1 \
    --max-model-len 32768 --trust-remote-code --gpu-memory-utilization 0.85 \
    > "$RESULT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!; trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
mkdir -p "$RESULT_DIR"
for _ in $(seq 90); do curl -sf 127.0.0.1:8000/health >/dev/null && break
    kill -0 $VLLM_PID || { echo "vLLM died (see $RESULT_DIR/vllm.log)"; exit 1; }; sleep 10; done
curl -sf 127.0.0.1:8000/health >/dev/null || { echo "vLLM never came up"; exit 1; }

CUDA_VISIBLE_DEVICES=0 "$HYWORLD2_PY" traj_generate.py --target_path "$SCENE_DIR" \
    --llm_addr 127.0.0.1 --llm_port 8000 --llm_name "$LLM_NAME" \
    --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm --skip_exist
CUDA_VISIBLE_DEVICES=$PA "$VENV_BIN/torchrun" --nproc_per_node $PA_N traj_render.py \
    --target_path "$SCENE_DIR" --llm_addr 127.0.0.1 --llm_port 8000 --llm_name "$LLM_NAME"
kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true; sleep 5

# Phase B: world expansion, GS data, 3DGS training (+ TSDF mesh export).
CUDA_VISIBLE_DEVICES=$ALL "$VENV_BIN/torchrun" --nproc_per_node $NGPU video_gen.py \
    --target_path "$SCENE_DIR" --fsdp --skip_exist
CUDA_VISIBLE_DEVICES=$ALL "$VENV_BIN/torchrun" --nproc_per_node $NGPU gen_gs_data.py \
    --root_path "$SCENE_DIR" --save_normal --split_sky
CUDA_VISIBLE_DEVICES=$ALL "$HYWORLD2_PY" -m world_gs_trainer default \
    --data_dir "$SCENE_DIR/gs_data" --result_dir "$RESULT_DIR" \
    --max_steps $STEPS --save_steps $STEPS --eval_steps $STEPS --ply_steps $STEPS \
    --save_ply --disable_video --use_scale_regularization --antialiased \
    --depth_loss --normal_loss --sky_depth_from_pcd \
    --use_mask_gaussian --mask_export_stochastic \
    --no-mask-export-anchor-protection --use_anchor_protection --export_mesh \
    --strategy.refine-start-iter $RS --strategy.refine-stop-iter $RE \
    --strategy.refine-every $RV --strategy.refine-scale2d-stop-iter $RE \
    --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1

echo "== outputs =="
find "$RESULT_DIR/ply" -name "*.ply" -exec ls -la {} \;
