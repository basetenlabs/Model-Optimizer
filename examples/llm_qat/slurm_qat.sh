#!/bin/bash
#SBATCH --job-name=qwen3-235b-qat
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

###############################################################################
# Configuration — edit these for your run
###############################################################################
MODEL="parsed/gamma-qwen3-235b-instruct-matt-merged"
DATASET="baseten/gamma-paste-text-train-v3"
HF_TOKEN="$(cat /secrets/harrypartridge_hf_token_whetstone)"
QUANT_CFG="NVFP4_DEFAULT_CFG"
OUTPUT_DIR="qwen3-235b-qat"

NUM_EPOCHS=2
LR="1e-5"
TRAIN_BS=1
EVAL_BS=1
ACCUM_STEPS=1
MAX_SEQ_LENGTH=32768
CALIB_SIZE=512
EVAL_SIZE=50
FSDP_LAYER="Qwen3MoeDecoderLayer"

###############################################################################
# Repo setup — each node clones the repo + installs deps (no shared filesystem)
###############################################################################
GH_TOKEN="${GH_TOKEN:?Set GH_TOKEN env var with a GitHub PAT for cloning}"
REPO_URL="https://${GH_TOKEN}@github.com/basetenlabs/Model-Optimizer.git"
REPO_BRANCH="qwen3-235b-qat"
WORK_DIR="/tmp/Model-Optimizer"

###############################################################################
# Cluster setup — derived from Slurm environment
###############################################################################
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
NUM_PROCESSES=$((NNODES * GPUS_PER_NODE))

# Get master addr/port from Slurm — use IP since hostnames may not resolve across nodes
MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{print $1}' | head -1)
MASTER_ADDR=${MASTER_ADDR:-$(hostname -i)}
MASTER_PORT=${MASTER_PORT:-29500}

SAVE_STEPS=$((192 / NUM_PROCESSES))
[ "$SAVE_STEPS" -lt 1 ] && SAVE_STEPS=1

###############################################################################
# Launch — srun runs once per node; each node bootstraps then launches training
###############################################################################
srun bash -c '
  set -eo pipefail

  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export HF_TOKEN="'"${HF_TOKEN}"'"
  export NCCL_DEBUG=INFO

  # Force PyTorch c10d to use IP instead of unresolvable hostname
  export MASTER_ADDR="'"${MASTER_ADDR}"'"
  export MASTER_PORT="'"${MASTER_PORT}"'"

  WORK_DIR="'"${WORK_DIR}"'"

  # Install uv if not available
  if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  # Clone repo on this node (fresh clone each run to ensure latest code)
  rm -rf "${WORK_DIR}"
  git clone --branch "'"${REPO_BRANCH}"'" --single-branch "'"${REPO_URL}"'" "${WORK_DIR}"

  cd "${WORK_DIR}"

  # Install modelopt + HF deps into a local venv via uv
  VENV_DIR="${WORK_DIR}/.venv"
  if [ ! -d "${VENV_DIR}" ]; then
    uv venv "${VENV_DIR}"
    uv pip install --python "${VENV_DIR}/bin/python" -e ".[hf]"
  fi

  export PATH="${VENV_DIR}/bin:${PATH}"

  cd "${WORK_DIR}/examples/llm_qat"

  accelerate launch \
    --config-file accelerate_config/fsdp1_multinode.yaml \
    --num_machines '"${NNODES}"' \
    --num_processes '"${NUM_PROCESSES}"' \
    --machine_rank ${SLURM_NODEID} \
    --main_process_ip '"${MASTER_ADDR}"' \
    --main_process_port '"${MASTER_PORT}"' \
    --fsdp_transformer_layer_cls_to_wrap '"${FSDP_LAYER}"' \
    main.py \
    --model_name_or_path '"${MODEL}"' \
    --model_max_length '"${MAX_SEQ_LENGTH}"' \
    --dataloader_drop_last True \
    --do_train True \
    --do_eval True \
    --output_dir '"${OUTPUT_DIR}"' \
    --dataset '"${DATASET}"' \
    --hf_token '"${HF_TOKEN}"' \
    --eval_size '"${EVAL_SIZE}"' \
    --num_train_epochs '"${NUM_EPOCHS}"' \
    --per_device_train_batch_size '"${TRAIN_BS}"' \
    --per_device_eval_batch_size '"${EVAL_BS}"' \
    --gradient_accumulation_steps '"${ACCUM_STEPS}"' \
    --eval_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps '"${SAVE_STEPS}"' \
    --eval_strategy steps \
    --eval_steps '"${SAVE_STEPS}"' \
    --load_best_model_at_end True \
    --save_total_limit 2 \
    --learning_rate '"${LR}"' \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --report_to tensorboard \
    --lora False \
    --compress False \
    --quant_cfg '"${QUANT_CFG}"' \
    --calib_size '"${CALIB_SIZE}"'
'
