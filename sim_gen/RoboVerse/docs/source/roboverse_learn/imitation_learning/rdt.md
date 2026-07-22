# RDT
Robotics Diffusion Transformer (RDT) is a 1B-parameter Diffusion Transformer trained with imitation learning. To finetune RDT with RoboVerse data, you need to follow the steps below.

## Installation
First, you need to clone the RoboVerse [version](https://github.com/wbjsamuel/RoboVerse-RDT) of RDT and install the environment as required. We provide an official transform and configuration of RoboVerse fine-tuning setup there.
```bash
# Clone this repo
git clone https://github.com/wbjsamuel/RoboVerse-RDT
cd RoboticsDiffusionTransformer

# Create a Conda environment
conda create -n rdt python=3.10.0
conda activate rdt

# Install pytorch
# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121

# Install packaging
pip install packaging==24.0

# Install flash-attn
pip install flash-attn --no-build-isolation

# Install other prequisites
pip install -r requirements.txt
```
Then, download the multi-modal encoders from huggingface and create a symbolic link to the `google` directory. Here is what you may refer to:
```bash
huggingface-cli download google/t5-v1_1-xxl --cache-dir YOUR_CACHE_DIR
huggingface-cli download google/siglip-so400m-patch14-384 --cache-dir YOUR_CACHE_DIR

# Under the root directory of this repo
mkdir -p google

# Link the downloaded encoders to this repo
ln -s /YOUR_CACHE_DIR/t5-v1_1-xxl google/t5-v1_1-xxl
ln -s /YOUR_CACHE_DIR/siglip-so400m-patch14-384 google/siglip-so400m-patch14-384
```
Lastly, add your buffer path to the `configs/base.yaml`. We'll directly use pre-training pipeline for fine-tuning, so the buffer path is necessary.
```bash
# ...

dataset:
# ...
# ADD YOUR buf_path: the path to the buffer (at least 400GB)
   buf_path: PATH_TO_YOUR_BUFFER
# ...
```
## Data Preparataion
For RDT, we still employ the RLDS-format data for fine-tuning. Please make sure your data have been successfully converted to RLDS format. If not, please refer to the [OpenVLA](https://roboverse.wiki/roboverse_learn/openvla) documentation for related data conversion part. Then, all you need to do is just create a symbolic link of the converted RLDS data to the required path.
```bash
ln -s PATH_TO_RLDS_DATA data/datasets/openx_embod/YOUR_TASK_NAME
```
## Finetune RDT
After the above steps, you can start finetuning RDT with the following command. We have modified the `finetune.sh` script for you and you can directly use it.:
```bash
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=1

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-finetune-1b-pickcube"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
# export CUTLASS_PATH="/path/to/cutlass"

export WANDB_PROJECT="rdt-finetune-1b-pickcube"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in multi nodes/machines
# deepspeed --hostfile=hostfile.txt main.py \
#    --deepspeed="./configs/zero2.json" \
#     ...

accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --sample_batch_size=32 \
    --max_train_steps=200000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="pretrain" \
    --state_noise_snr=40 \
    --report_to=wandb

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \
```
