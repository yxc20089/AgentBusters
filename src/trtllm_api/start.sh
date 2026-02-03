#!/bin/bash
# TRT-LLM OpenAI API 启动脚本
# 在 TensorRT-LLM 容器内运行

set -e

# 默认配置
MODEL="${MODEL:-nvidia/DeepSeek-V3.2-NVFP4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PORT="${PORT:-8000}"

echo "============================================"
echo "TRT-LLM OpenAI Compatible API Server"
echo "============================================"
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Port: $PORT"
echo "============================================"

# 安装依赖
pip install --quiet fastapi uvicorn pydantic

# 启动 API 服务
python /app/trtllm_openai_api.py \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --port "$PORT"
