#!/bin/bash

# 优化版训练脚本
# 提供更完善的环境设置和错误处理

set -e  # 遇到错误立即退出

echo "=== Optimized Generative Recommendation Training ==="
echo "Current script directory: ${RUNTIME_SCRIPT_DIR:-$(pwd)}"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# 设置默认环境变量
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-./logs}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-./tb_logs}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./data}"

# 创建必要的目录
mkdir -p "${TRAIN_LOG_PATH}"
mkdir -p "${TRAIN_TF_EVENTS_PATH}"

echo "Log path: ${TRAIN_LOG_PATH}"
echo "TensorBoard path: ${TRAIN_TF_EVENTS_PATH}"
echo "Data path: ${TRAIN_DATA_PATH}"

# 进入脚本目录
if [ -n "${RUNTIME_SCRIPT_DIR}" ]; then
    cd "${RUNTIME_SCRIPT_DIR}"
fi

# 检查Python环境
echo "Checking Python environment..."
python -c "import torch; print(f'PyTorch {torch.__version__} available')" || {
    echo "Error: PyTorch not found. Please install required dependencies:"
    echo "pip install -r requirements.txt"
    exit 1
}

# 检查数据目录
if [ ! -d "${TRAIN_DATA_PATH}" ]; then
    echo "Warning: Data directory ${TRAIN_DATA_PATH} not found"
    echo "Creating dummy data directory for testing..."
    mkdir -p "${TRAIN_DATA_PATH}"
fi

# 训练参数配置
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-0.001}
NUM_EPOCHS=${NUM_EPOCHS:-10}
HIDDEN_UNITS=${HIDDEN_UNITS:-64}
NUM_BLOCKS=${NUM_BLOCKS:-2}
NUM_HEADS=${NUM_HEADS:-2}
DEVICE=${DEVICE:-cuda}

echo "Training parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Hidden units: ${HIDDEN_UNITS}"
echo "  Transformer blocks: ${NUM_BLOCKS}"
echo "  Attention heads: ${NUM_HEADS}"
echo "  Device: ${DEVICE}"

echo ""
echo "Starting training..."
echo "===================="

# 执行训练
python -u main.py \\
    --batch_size "${BATCH_SIZE}" \\
    --lr "${LEARNING_RATE}" \\
    --num_epochs "${NUM_EPOCHS}" \\
    --hidden_units "${HIDDEN_UNITS}" \\
    --num_blocks "${NUM_BLOCKS}" \\
    --num_heads "${NUM_HEADS}" \\
    --device "${DEVICE}" \\
    --norm_first \\
    --optimizer adam \\
    --lr_scheduler cosine \\
    --weight_decay 0.01 \\
    --dropout_rate 0.2 \\
    --maxlen 101 \\
    --mm_emb_id 81 82 \\
    --save_freq 5 \\
    2>&1 | tee "${TRAIN_LOG_PATH}/training_output.log"

TRAIN_EXIT_CODE=$?

echo ""
echo "===================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Logs saved to: ${TRAIN_LOG_PATH}"
    echo "TensorBoard logs: ${TRAIN_TF_EVENTS_PATH}"
    echo ""
    echo "To view training progress:"
    echo "tensorboard --logdir=${TRAIN_TF_EVENTS_PATH}"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check logs in: ${TRAIN_LOG_PATH}"
    exit $TRAIN_EXIT_CODE
fi
