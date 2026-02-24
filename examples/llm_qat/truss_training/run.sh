#!/bin/bash
set -eux

###############################################################################
# Configuration
###############################################################################
MODEL="parsed/gamma-qwen3-235b-instruct-matt-merged"
DATASET="baseten/gamma-paste-text-train-v3"
QUANT_CFG="NVFP4_DEFAULT_CFG"
OUTPUT_DIR="${BT_CHECKPOINT_DIR:-qwen3-235b-qat}"

NUM_EPOCHS=2
LR="1e-5"
TRAIN_BS=1
EVAL_BS=1
ACCUM_STEPS=1
MAX_SEQ_LENGTH=32768
CALIB_SIZE=512
EVAL_SIZE=50
FSDP_LAYER="Qwen3MoeDecoderLayer"

# GH_TOKEN is set via Baseten secret in config.py
REPO_URL="https://${GH_TOKEN}@github.com/basetenlabs/Model-Optimizer.git"
REPO_BRANCH="qwen3-235b-qat"
WORK_DIR="/root/Model-Optimizer"

###############################################################################
# Cluster — derived from Baseten Training env vars
###############################################################################
NNODES="${BT_GROUP_SIZE}"
GPUS_PER_NODE="${BT_NUM_GPUS}"
NUM_PROCESSES=$((NNODES * GPUS_PER_NODE))

RDZV_PORT="29400"
RDZV_TIMEOUT="1800"

SAVE_STEPS=$((192 / NUM_PROCESSES))
[ "$SAVE_STEPS" -lt 1 ] && SAVE_STEPS=1

###############################################################################
# System deps
###############################################################################
apt-get update
apt-get install --no-install-recommends -y git netcat-openbsd curl

###############################################################################
# NCCL config — IB disabled due to sriov-network-config-daemon not running
# (label mismatch: DaemonSet wants nvidia.com/gpu=present but nodes have
#  nvidia.com/gpu.present=true). TCP fallback confirmed working on 2-node 4B.
###############################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800000
export NCCL_DEBUG=INFO

###############################################################################
# Install uv
###############################################################################
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

###############################################################################
# Clone repo + install deps
###############################################################################
if [ -d "${WORK_DIR}/.git" ]; then
    cd "${WORK_DIR}"
    git fetch origin "${REPO_BRANCH}"
    git checkout "${REPO_BRANCH}"
    git reset --hard "origin/${REPO_BRANCH}"
else
    git clone --branch "${REPO_BRANCH}" --single-branch "${REPO_URL}" "${WORK_DIR}"
    cd "${WORK_DIR}"
fi

uv pip install --system -e ".[hf]" tensorboard
# Uninstall deepspeed — we use FSDP, and deepspeed 0.18.x has circular import issues
uv pip uninstall --system deepspeed

###############################################################################
# Generate accelerate config
###############################################################################
cd "${WORK_DIR}/examples/llm_qat"

cat > /tmp/accelerate_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_reshard_after_forward: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: ${FSDP_LAYER}
  fsdp_use_orig_params: true
  fsdp_version: 1
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: ${NNODES}
num_processes: ${NUM_PROCESSES}
rdzv_backend: c10d
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

###############################################################################
# Wait for leader (follower nodes only)
###############################################################################
if [[ "${BT_NODE_RANK}" != "0" ]]; then
    echo "Waiting for leader ${BT_LEADER_ADDR}:${RDZV_PORT}..."
    timeout 20m bash -c 'until nc -z "$0" "$1"; do sleep 5; done' "${BT_LEADER_ADDR}" "${RDZV_PORT}" || {
        echo "Timed out waiting for leader rendezvous port" >&2
        exit 1
    }
fi

###############################################################################
# Launch training
###############################################################################
accelerate launch \
    --config_file /tmp/accelerate_config.yaml \
    --num_machines "${NNODES}" \
    --num_processes "${NUM_PROCESSES}" \
    --machine_rank "${BT_NODE_RANK}" \
    --main_process_ip "${BT_LEADER_ADDR}" \
    --main_process_port "${RDZV_PORT}" \
    --fsdp_transformer_layer_cls_to_wrap "${FSDP_LAYER}" \
    main.py \
    --model_name_or_path "${MODEL}" \
    --model_max_length "${MAX_SEQ_LENGTH}" \
    --dataloader_drop_last True \
    --do_train True \
    --do_eval True \
    --output_dir "${OUTPUT_DIR}" \
    --dataset "${DATASET}" \
    --hf_token "${DATASET_HF_TOKEN}" \
    --eval_size "${EVAL_SIZE}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${TRAIN_BS}" \
    --per_device_eval_batch_size "${EVAL_BS}" \
    --gradient_accumulation_steps "${ACCUM_STEPS}" \
    --eval_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --eval_strategy steps \
    --eval_steps "${SAVE_STEPS}" \
    --load_best_model_at_end True \
    --save_total_limit 2 \
    --learning_rate "${LR}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --report_to tensorboard \
    --lora False \
    --compress False \
    --quant_cfg "${QUANT_CFG}" \
    --calib_size "${CALIB_SIZE}"
