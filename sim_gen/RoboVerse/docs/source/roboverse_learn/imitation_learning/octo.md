# Octo

## Installation
Please clone the RoboVerse [version](https://github.com/wbjsamuel/RoboVerse-Octo.git) of Octo and install the environment as required. We provide an official transform and configuration of RoboVerse fine-tuning setup there.
```bash
git clone https://github.com/wbjsamuel/RoboVerse-Octo.git

conda create -n octo python=3.10
conda activate octo
pip install -e .
pip install -r requirements.txt
```
Then, install jax based on your hardware settings. Here is an example for installing jax on a GPU machine with nvcc version 12.4:
```bash
pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install -c nvidia cudnn=8.9
```
## Data Preparation
For Octo, we still employ the RLDS-format data for fine-tuning. Please make sure your data have been successfully converted to RLDS format. If not, please refer to the [OpenVLA](https://roboverse.wiki/roboverse_learn/openvla) documentation for related data conversion part. Then, all you need to do is just create a symbolic link of the converted RLDS data to the required path.
```bash
mkdir data
ln -s PATH_TO_RLDS_DATA data/YOUR_TASK_NAME
```
## Finetune Octo
Launch your finetuning with `finetune.sh`:
```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=1

python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-base
```
