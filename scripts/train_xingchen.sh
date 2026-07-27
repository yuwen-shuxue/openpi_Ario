#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OPENPI_DIR}"

# Activate venv
source "${OPENPI_DIR}/.venv/bin/activate"

# OSS credentials (passed via environment, e.g. sslaunch -e)
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY env var}"

# Install transformers_replace patches
cp -r "${OPENPI_DIR}/src/openpi/models_pytorch/transformers_replace/"* \
    "${OPENPI_DIR}/.venv/lib/python3.12/site-packages/transformers/"

# Setup PYTHONPATH
export PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

# GPU count
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# Distributed training env vars (set by PyTorchJob / sslaunch)
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
MASTER_PORT=${MASTER_PORT:-'1239'}
NNODES=${WORLD_SIZE:-'1'}
NODE_RANK=${RANK:-'0'}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

DIST_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

CONFIG_NAME="${1:-pi05_xingchen}"
shift || true

# Default weight path (under home directory for write access)
WEIGHT_PATH="${OPENPI_DIR}/checkpoints/pi05_base_pytorch"

# --- Preparation steps (rank 0 only) ---
if [ "$NODE_RANK" = "0" ]; then
    # 1. Convert JAX weights to PyTorch if not already done
    if [ ! -f "${WEIGHT_PATH}/model.safetensors" ]; then
        echo "PyTorch weights not found at ${WEIGHT_PATH}, converting..."
        python "${OPENPI_DIR}/examples/convert_jax_model_to_pytorch.py" \
            --checkpoint-dir "gs://openpi-assets/checkpoints/pi05_base" \
            --config-name "$CONFIG_NAME" \
            --output-path "$WEIGHT_PATH" \
            --precision bfloat16 || {
            echo "ERROR: Weight conversion failed."
            exit 1
        }
    fi

    # 2. Compute norm stats if not already present
    if [ ! -f "${OPENPI_DIR}/assets/${CONFIG_NAME}/xingchen/new_blocks/norm_stats.json" ]; then
        echo "Computing norm stats for ${CONFIG_NAME}..."
        python "${OPENPI_DIR}/scripts/compute_norm_stats.py" --config-name "$CONFIG_NAME" --max-frames 5000
        echo "Norm stats computed."
    fi
fi

# Sync: wait for rank 0 to finish preparation before other nodes start training
if [ "$NNODES" -gt "1" ] && [ "$NODE_RANK" != "0" ]; then
    echo "Waiting for rank 0 to finish preparation..."
    while [ ! -f "${WEIGHT_PATH}/model.safetensors" ]; do
        sleep 5
    done
fi

torchrun "${DIST_ARGS[@]}" "${OPENPI_DIR}/scripts/train_pytorch.py" "$CONFIG_NAME" "$@"
