# AgentBusters Benchmark 运行指南

本文档详细说明如何使用实验室 GPU 运行开源 LLM 进行 benchmark 测试，以及如何使用 AgentBusters-Leaderboard 收集不同配置下的测试结果。

## 目录

1. [环境准备](#环境准备)
2. [LLM 配置选项](#llm-配置选项)
3. [评测数据配置](#评测数据配置)
4. [运行 Benchmark](#运行-benchmark)
5. [多配置实验管理](#多配置实验管理)
6. [结果收集与分析](#结果收集与分析)

---

## 环境准备

### 1. 基础安装

```bash
# 克隆仓库
cd /path/to/your/workspace/AgentBusters

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS
source venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

# Windows CMD
# .\.venv\Scripts\activate.bat

# 安装依赖
pip install -e ".[dev]"

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

### 2. 配置文件设置

复制 `.env.example` 到 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

---

## LLM 配置选项

### Purple Agent LLM 配置（在 `.env` 中定义）

Purple Agent 是被评测的金融分析 Agent，其 LLM 配置在 `.env` 文件中设置：

```dotenv
# ============================================
# 推荐配置: 本地 vLLM 部署（GPU 服务器）
# ============================================
OPENAI_API_KEY=dummy                          # vLLM 不需要真实 API key
OPENAI_API_BASE=http://localhost:8000/v1      # vLLM 服务地址
OPENAI_BASE_URL=http://localhost:8000/v1      # 别名

# --- Qwen3-32B（推荐，平衡性能与资源）---
LLM_MODEL=Qwen/Qwen3-32B

# --- DeepSeek-V3.2（最强性能）---
# LLM_MODEL=deepseek-ai/DeepSeek-V3

# --- Qwen3-14B（轻量级）---
# LLM_MODEL=Qwen/Qwen3-14B

# ============================================
# 备选: OpenRouter API（无需本地 GPU）
# ============================================
# OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxx
# OPENAI_API_BASE=https://openrouter.ai/api/v1
# LLM_MODEL=qwen/qwen3-32b              # Qwen3-32B via OpenRouter
# LLM_MODEL=deepseek/deepseek-chat      # DeepSeek via OpenRouter

# ============================================
# 商业 API（用于基准对比）
# ============================================
# OPENAI_API_KEY=sk-...
# LLM_MODEL=gpt-4o

# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL=claude-sonnet-4-20250514

# ============================================
# 通用配置
# ============================================
PURPLE_LLM_TEMPERATURE=0.0  # 设为 0.0 以获得可重复的结果
```

### Green Agent LLM 配置（在 eval_config.yaml 中定义）

Green Agent 是评测器，使用 LLM-as-judge 进行评分：

```yaml
# config/eval_config.yaml
llm_eval:
  enabled: true
  model: gpt-4o-mini       # 评判模型
  temperature: 0.0         # 固定为 0 以保证可重复性
```

### 启动本地 vLLM 服务（GPU 服务器）

```bash
# 重要：版本兼容性说明
# vLLM 0.15.0 会自动安装 PyTorch 2.9.x，这是推荐版本
# ⚠️ 注意：不同 CUDA 索引会安装不同 PyTorch 版本 (cu126→2.10, cu124→2.4)
# 推荐方法：让 vLLM 自动处理 PyTorch 版本管理

# 方法 1：让 vLLM 自动处理 PyTorch (推荐)
pip install vllm
# vLLM 会自动安装兼容的 PyTorch 2.9.x + CUDA 支持

# 方法 2：手动指定版本 (如果需要完全控制)
pip install torch==2.9.1 torchvision torchaudio  # 不使用索引 URL
pip install vllm

# 1. 检查当前 PyTorch 是否支持 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 如果显示 CUDA available: False 或版本包含 '+cpu'，需要重新安装
# 卸载 CPU 版本的 PyTorch
pip uninstall torch torchvision torchaudio -y

# 重新安装 vLLM (会自动安装正确的 PyTorch 版本)
pip install vllm --no-cache-dir

# 修复 NumPy 兼容性问题 (如果出现 NumPy 2.x 警告)
pip install "numpy<2.0"

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import vllm; print('vLLM installed successfully')"
```

#### 推荐模型部署

**🚀 GH200 超级计算 (单卡 480GB HBM3e) - 终极单卡**

**1. DeepSeek-V3.2-671B (单卡 GH200-480GB) - 单卡跑 671B 全参数！**
```bash
# 1x GH200 480GB - 单卡运行 671B MoE 全精度
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3

# 快速部署命令
python scripts/deploy_vllm.py --model deepseek-v3-gh200
```

**2. Qwen3-235B-A22B (单卡 GH200-480GB) - 超大 Context**
```bash
# 1x GH200 480GB - 单卡运行 235B MoE
vllm serve Qwen/Qwen3-235B-A22B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

# 注意: Qwen3-235B-A22B 的原生上下文长度为 40,960 tokens
# GH200-480GB 有足够内存，但模型架构限制了上下文长度

# ⚠️ 单卡 H100 80GB 无法运行此模型！需要至少 3 张 H100 80GB
# 8x H100 80GB 配置 (推荐)
vllm serve Qwen/Qwen3-235B-A22B \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

# 3x H100 80GB 配置 (最小配置)
vllm serve Qwen/Qwen3-235B-A22B \
    --port 8000 \
    --tensor-parallel-size 3 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

# 快速部署命令
python scripts/deploy_vllm.py --model qwen3-235b-gh200
```

---

**🔋 GH200 标准版 (单卡 96GB HBM3e) - 高效单卡**

**⚠️ 重要提示: Qwen3-235B-A22B 实际需要超过 96GB 内存，无法在 GH200-96GB 上运行** 

**推荐替代方案:**
```bash
# 1. Qwen3-32B (最佳选择) - 单卡运行，性能优秀
vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

# 2. DeepSeek-V3 量化版本 (实验性)
# 注意: 即使量化也可能超出 96GB 限制
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --quantization gptq \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3
```

---

**⚡ B200 顶配 (单卡 192GB HBM3e) - 最强单卡**

**1. Qwen3-235B-A22B (单卡 B200) - 单卡跑 235B！**
```bash
# 1x B200 192GB - 单卡运行 235B MoE
vllm serve Qwen/Qwen3-235B-A22B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

# 快速部署命令
python scripts/deploy_vllm.py --model qwen3-235b-b200
```

**2. DeepSeek-V3.2-671B (3x B200) - 全精度 671B**
```bash
# 3x B200 192GB - BF16 全精度运行 671B MoE
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 3 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3

# 快速部署命令
python scripts/deploy_vllm.py --model deepseek-v3-b200
```

---

**🚀 H100 顶配 (8x 80GB) - ⚠️ DeepSeek-V3 需要 Hopper GPU**

> **⚠️ 重要硬件要求**: DeepSeek-V3 使用 MLA (Multi-head Latent Attention) 架构，**必须使用 Hopper 架构 GPU (H100/H200)**。
> A100 (compute capability 8.0) **无法运行** DeepSeek-V3，即使有 8 张卡也不行！
> 
> 如果您使用 A100，请使用 Qwen3-32B、Llama-3.1-70B 或 Mixtral-8x22B 等替代模型。

**1. DeepSeek-V3.2-671B (FP8 量化) - 仅限 H100/H200**
```bash
# 8x H100 80GB, FP8 量化 - 推荐配置 (16K context，稳定运行)
# ⚠️ 此命令仅适用于 H100/H200 GPU！A100 无法运行！
vllm serve deepseek-ai/DeepSeek-V3.2 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.8 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3

# 8x H100 80GB, FP8 量化 - 较大 context (24K，降低内存利用率)
# 如果 16K 不够，可以尝试此配置
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3

# ⚠️ 注意: 32K context + 0.95 内存利用率会 OOM！
# 如果需要 32K context，请使用 0.85 内存利用率或更多 GPU

# 如果不使用 FP8 量化 (BF16，需要更多显存)
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3
```

---

**� TensorRT-LLM + NVFP4 (推荐: 最高吞吐)**

> **推荐路线**: 对于 DeepSeek-V3.2，TRT-LLM + NVFP4 是 NVIDIA 官方推荐的部署方式，比 vLLM 有更高吞吐量。
> NVFP4 是预量化模型，不需要自己 build engine，开箱即用。

**1. DeepSeek-V3.2-NVFP4 (8x H100 80GB) - 最优方案**
```bash
# 使用部署脚本 (推荐)
python scripts/deploy_trtllm.py --model deepseek-v3-nvfp4 --port 8000

# 或手动部署:
# Step 1: 启动 TRT-LLM 容器
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v /mnt/models:/models \
  -v $(pwd)/src/trtllm_api:/app \
  -w /app \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc1

# Step 2: 在容器内启动 API
pip install fastapi uvicorn pydantic
python trtllm_openai_api.py \
    --model nvidia/DeepSeek-V3.2-NVFP4 \
    --tensor-parallel-size 8 \
    --port 8000
```

**TRT-LLM vs vLLM 对比:**

| 对比项 | vLLM | TRT-LLM + NVFP4 |
|--------|------|-----------------|
| DeepSeek-V3.2 支持 | ✅ | ✅ |
| FP8/FP4 MoE | ❌ | ✅ |
| 8×80GB 稳定性 | 一般 | **稳定** |
| 吞吐量 | 高 | **更高** |
| 工程复杂度 | 低 | 中 |
| 生产可控性 | 中 | **高** |

---

**�💎 A100 顶配 (8x 80GB) - 推荐配置**

> A100 用户推荐使用以下模型 (不支持 DeepSeek-V3)

**1. Qwen3-32B (推荐) - 性能优秀，资源友好**
```bash
# 8x A100 80GB - 可以运行多实例或使用更大 context
vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

**2. Llama-3.1-70B-Instruct - 强大的通用模型**
```bash
# 8x A100 80GB
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

**3. Mixtral-8x22B-Instruct - MoE 架构 (无 MLA 限制)**
```bash
# 8x A100 80GB
vllm serve mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral
```

**2. Qwen3-235B-A22B (MoE) - 顶级 MoE**
```bash
# 8x H100 80GB, 235B 参数 (22B 激活)
vllm serve Qwen/Qwen3-235B-A22B \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

---

**📌 主要目标模型**

**1. Qwen3-32B（推荐，平衡性能与资源）**
```bash
# 单 GPU (A100 80GB / H100 80GB)
vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

2 x A6000
CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-32B --port 8100 --tensor-parallel-size 2 --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser qwen3_xml

# 双 GPU (2x A100 40GB / 2x RTX 4090)
vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

**2. DeepSeek-V3.2（高性能，需要多 GPU）**
```bash
# 4x A100 80GB 或 8x A100 40GB
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3

# 8x GPU 配置（更大 context）
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3
```

**3. Qwen3-14B（轻量级，适合单 GPU）**
```bash
# 单 GPU (RTX 4090 24GB / A100 40GB)
vllm serve Qwen/Qwen3-14B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

# RTX 3090 24GB（减少 context 长度）
vllm serve Qwen/Qwen3-14B \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

#### GPU 内存需求参考

| 模型 | 参数量 | 最低 GPU | 推荐配置 | Context 长度 |
|------|--------|----------|----------|-------------|
| Qwen3-14B | 14B | 1x RTX 4090 (24GB) | 1x A100 40GB | 32K |
| Qwen3-32B | 32B | 2x RTX 4090 | 1x A100 80GB | 32K |
| DeepSeek-V3.2 | 671B MoE | 4x A100 80GB | 8x H100 80GB | 32K |
| Qwen3-235B-A22B | 235B MoE (22B激活) | 1x GH200 480GB or 2x H100 80GB | 1x GH200-480GB | 40K |
| DeepSeek-V3 FP8 | 671B MoE | 8x H100 80GB | 8x H100 80GB | 32K |
| **⚡ Qwen3-235B (B200)** | **235B MoE** | **1x B200 192GB** | **1x B200** | **40K** |
| **⚡ DeepSeek-V3 (B200)** | **671B MoE** | **3x B200 192GB** | **3x B200** | **65K** |
| **🚀 Qwen3-32B (GH200-96GB)** | **32B** | **1x GH200 96GB** | **1x GH200 96GB** | **32K** |
| **🚀 Qwen3-235B (GH200-480GB)** | **235B MoE** | **1x GH200 480GB** | **1x GH200** | **40K** |
| **🚀 DeepSeek-V3 (GH200-480GB)** | **671B MoE** | **1x GH200 480GB** | **1x GH200** | **131K** |

#### 其他模型（备选）
```bash
# Llama 3.1 70B
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json

# Mixtral 8x22B
vllm serve mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral
```

---

## 评测数据配置

### 评测配置文件结构

评测配置在 `config/` 目录下的 YAML 文件中定义。以下是不同规模的配置示例：

### 快速测试配置（~10 tasks）

创建 `config/eval_quick_test.yaml`:

```yaml
name: "Quick Test (10 tasks)"
version: "1.0"

datasets:
  - type: bizfinbench
    task_types: [event_logic_reasoning]
    languages: [en]
    limit_per_task: 3
    shuffle: false
    weight: 1.0

  - type: synthetic
    path: data/synthetic_questions/questions.json
    limit: 4
    shuffle: false
    weight: 1.0

  - type: options
    path: data/options/questions.json
    limit: 3
    shuffle: false
    weight: 1.0

sampling:
  strategy: stratified
  total_limit: 10
  seed: 42

llm_eval:
  enabled: true
  model: gpt-4o-mini
  temperature: 0.0

timeout_seconds: 300
```

### 中等规模配置（~100 tasks）

创建 `config/eval_medium.yaml`:

```yaml
name: "Medium Scale Evaluation (100 tasks)"
version: "1.0"

datasets:
  - type: bizfinbench
    task_types:
      - event_logic_reasoning
      - financial_quantitative_computation
      - anomaly_information_tracing
    languages: [en, cn]
    limit_per_task: 8
    shuffle: false
    weight: 1.0

  - type: prbench
    splits: [finance, legal]
    limit: 20
    shuffle: false
    weight: 1.0

  - type: synthetic
    path: data/synthetic_questions/questions.json
    limit: 20
    shuffle: false
    weight: 1.0

  - type: options
    path: data/options/questions.json
    limit: 20
    shuffle: false
    weight: 1.0

  - type: crypto
    path: ../agentbusters-eval-data/crypto/eval_hidden  # 使用 eval-data 中的数据
    download_on_missing: false
    limit: 6
    shuffle: false
    weight: 1.0
    stride: 1
    max_steps: 100
    evaluation:
      initial_balance: 10000.0
      max_leverage: 3.0
      trading_fee: 0.0004
      price_noise_level: 0.001
      slippage_range: [0.0002, 0.0010]
      adversarial_injection_rate: 0.05
      decision_interval: 5
      funding_interval_hours: 8.0
      score_weights:
        baseline: 0.40
        noisy: 0.30
        adversarial: 0.20
        meta: 0.10
      metric_weights:
        sharpe: 0.50
        total_return: 0.25
        max_drawdown: 0.15
        win_rate: 0.10

  - type: gdpval
    hf_dataset: "openai/gdpval"
    limit: 10
    shuffle: false
    weight: 1.0

sampling:
  strategy: stratified
  total_limit: 100
  seed: 42

llm_eval:
  enabled: true
  model: gpt-4o-mini
  temperature: 0.0

timeout_seconds: 600
```

### 大规模配置（~1000 tasks）

创建 `config/eval_large.yaml`:

```yaml
name: "Large Scale Evaluation (1000 tasks)"
version: "1.0"

datasets:
  - type: bizfinbench
    task_types:
      - anomaly_information_tracing
      - event_logic_reasoning
      - financial_data_description
      - financial_quantitative_computation
      - user_sentiment_analysis
      - stock_price_predict
      - financial_multi_turn_perception
    languages: [en, cn]
    limit_per_task: 50   # 7 types × 2 languages × 50 = 700
    shuffle: true
    weight: 1.0

  - type: prbench
    splits: [finance, legal, finance_hard, legal_hard]
    limit: 100
    shuffle: true
    weight: 1.0

  - type: synthetic
    path: data/synthetic_questions/questions.json
    limit: 50
    shuffle: true
    weight: 1.0

  - type: options
    path: data/options/questions.json
    limit: 50
    shuffle: true
    weight: 1.0

  - type: crypto
    path: ../agentbusters-eval-data/crypto/eval_hidden
    download_on_missing: false
    limit: 12  # 全部 12 个 scenarios
    shuffle: false
    weight: 1.0
    stride: 1
    max_steps: 200
    evaluation:
      initial_balance: 10000.0
      max_leverage: 3.0
      trading_fee: 0.0004
      price_noise_level: 0.001
      slippage_range: [0.0002, 0.0010]
      adversarial_injection_rate: 0.05
      decision_interval: 1
      funding_interval_hours: 8.0
      score_weights:
        baseline: 0.40
        noisy: 0.30
        adversarial: 0.20
        meta: 0.10
      metric_weights:
        sharpe: 0.50
        total_return: 0.25
        max_drawdown: 0.15
        win_rate: 0.10
      meta_transforms:
        - identity
        - scale_1_1
        - invert_returns

  - type: gdpval
    hf_dataset: "openai/gdpval"
    limit: 50
    shuffle: true
    weight: 1.0
    include_reference_files: true

sampling:
  strategy: stratified
  total_limit: 1000
  seed: 42

llm_eval:
  enabled: true
  model: gpt-4o-mini
  temperature: 0.0

timeout_seconds: 900
```

### 使用 agentbusters-eval-data 中的 Crypto 数据

确保 crypto 数据路径正确指向 `agentbusters-eval-data`:

```yaml
- type: crypto
  path: ../agentbusters-eval-data/crypto/eval_hidden  # 相对路径
  # 或使用绝对路径
  # path: d:/code/finbenchmark/agentbusters-eval-data/crypto/eval_hidden
```

可用的 crypto scenarios (共 12 个):
- `scenario_520d87ed7569f147` (BTCUSDT)
- `scenario_b8aba67d7bfcc3b4` (BTCUSDT)
- `scenario_0a9c24d037aaa15c` (BTCUSDT)
- `scenario_9a1f49ebc9fcc664` (ETHUSDT)
- `scenario_a9d7b02930d276f2` (ETHUSDT)
- ... 等

---

## 运行 Benchmark

### 方法 1: 本地运行（推荐用于开发和调试）

```bash
# 终端 1: 启动 Green Agent (评测器)
python src/cio_agent/a2a_server.py \
    --host 0.0.0.0 \
    --port 9109 \
    --eval-config config/eval_medium.yaml \
    --store-predicted \
    --predicted-max-chars 200

python src/cio_agent/a2a_server.py \
    --host 0.0.0.0 \
    --port 9109 \
    --eval-config config/eval_quick_test.yaml \
    --store-predicted \
    --predicted-max-chars 200

# 终端 2: 启动 Purple Agent (被评测的 Agent)
purple-agent serve --host 0.0.0.0 --port 9110 --card-url http://127.0.0.1:9110

# 终端 3: 运行评测
python scripts/run_a2a_eval.py \
    --green-url http://127.0.0.1:9109 \
    --purple-url http://127.0.0.1:9110 \
    --num-tasks 100 \
    --timeout 1800 \
    -v \
    -o results/eval_medium_$(date +%Y%m%d_%H%M%S).json
```

### 方法 2: Docker 运行

```bash
# 构建镜像
docker build -t agentbusters-green -f Dockerfile.green .
docker build -t agentbusters-purple -f Dockerfile.purple .

# 运行
docker-compose up
```

### 方法 3: 使用 Leaderboard 框架

参见下一节 [多配置实验管理](#多配置实验管理)。

---

## 多配置实验管理

使用 AgentBusters-Leaderboard 框架来系统地管理不同配置下的实验结果。

### 实验配置模板

创建 `experiments/experiment_configs.yaml`:

```yaml
# 实验配置定义
experiments:
  # 实验 1: 不同模型对比
  - name: "model_comparison"
    description: "Compare different LLM models"
    configs:
      - id: "llama3.1-70b"
        llm_model: "meta-llama/llama-3.1-70b-instruct"
        eval_config: "config/eval_medium.yaml"
        num_tasks: 100
        
      - id: "qwen2.5-72b"
        llm_model: "qwen/qwen-2.5-72b-instruct"
        eval_config: "config/eval_medium.yaml"
        num_tasks: 100
        
      - id: "deepseek-chat"
        llm_model: "deepseek/deepseek-chat"
        eval_config: "config/eval_medium.yaml"
        num_tasks: 100
        
      - id: "mixtral-8x22b"
        llm_model: "mistralai/mixtral-8x22b-instruct"
        eval_config: "config/eval_medium.yaml"
        num_tasks: 100

  # 实验 2: 不同任务数量对比
  - name: "scale_comparison"
    description: "Compare evaluation at different scales"
    configs:
      - id: "scale-10"
        llm_model: "meta-llama/llama-3.1-70b-instruct"
        eval_config: "config/eval_quick_test.yaml"
        num_tasks: 10
        
      - id: "scale-100"
        llm_model: "meta-llama/llama-3.1-70b-instruct"
        eval_config: "config/eval_medium.yaml"
        num_tasks: 100
        
      - id: "scale-500"
        llm_model: "meta-llama/llama-3.1-70b-instruct"
        eval_config: "config/eval_large.yaml"
        num_tasks: 500

  # 实验 3: 抽样策略对比
  - name: "sampling_comparison"
    description: "Compare different sampling strategies"
    configs:
      - id: "stratified"
        sampling_strategy: "stratified"
        num_tasks: 100
        
      - id: "random"
        sampling_strategy: "random"
        num_tasks: 100
        
      - id: "sequential"
        sampling_strategy: "sequential"
        num_tasks: 100
```

### 批量运行脚本

创建 `scripts/run_experiments.py`:

```python
#!/usr/bin/env python3
"""
批量运行多配置实验

Usage:
    python scripts/run_experiments.py --experiment model_comparison
    python scripts/run_experiments.py --all
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_experiment_configs(config_path: str) -> dict:
    """加载实验配置"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_experiment(
    config_id: str,
    llm_model: str,
    eval_config: str,
    num_tasks: int,
    output_dir: str,
    green_url: str = "http://localhost:9109",
    purple_url: str = "http://localhost:9110",
    timeout: int = 1800,
) -> dict:
    """运行单个实验配置"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"{config_id}_{timestamp}.json"
    
    # 设置环境变量
    env = os.environ.copy()
    env["LLM_MODEL"] = llm_model
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/run_a2a_eval.py",
        "--green-url", green_url,
        "--purple-url", purple_url,
        "--num-tasks", str(num_tasks),
        "--timeout", str(timeout),
        "-v",
        "-o", str(output_file),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_id}")
    print(f"  Model: {llm_model}")
    print(f"  Tasks: {num_tasks}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, env=env, capture_output=False)
    
    return {
        "config_id": config_id,
        "llm_model": llm_model,
        "num_tasks": num_tasks,
        "output_file": str(output_file),
        "success": result.returncode == 0,
        "timestamp": timestamp,
    }


def run_experiment_suite(
    experiment_name: str,
    configs: list,
    output_dir: str,
) -> list:
    """运行一组实验"""
    
    results = []
    for config in configs:
        result = run_single_experiment(
            config_id=config["id"],
            llm_model=config.get("llm_model", os.getenv("LLM_MODEL", "gpt-4o")),
            eval_config=config.get("eval_config", "config/eval_config.yaml"),
            num_tasks=config.get("num_tasks", 100),
            output_dir=output_dir,
        )
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run multiple experiment configurations")
    parser.add_argument("--config", default="experiments/experiment_configs.yaml")
    parser.add_argument("--experiment", help="Specific experiment to run")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--output-dir", default="results/experiments")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = load_experiment_configs(args.config)
    
    all_results = []
    
    for experiment in config["experiments"]:
        if args.all or args.experiment == experiment["name"]:
            print(f"\n{'#'*60}")
            print(f"# Experiment: {experiment['name']}")
            print(f"# {experiment['description']}")
            print(f"{'#'*60}")
            
            results = run_experiment_suite(
                experiment["name"],
                experiment["configs"],
                args.output_dir,
            )
            all_results.extend(results)
    
    # 保存汇总结果
    summary_file = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Experiment summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
```

### Leaderboard 结果收集

修改 `AgentBusters-Leaderboard/scenario.toml` 来定义不同的实验:

```toml
# scenario.toml - 多配置实验示例

[green_agent]
agentbeats_id = "019bc421-99d0-7ee3-ae27-658145eff474"
env = { 
    OPENAI_API_KEY = "${OPENAI_API_KEY}", 
    EVAL_CONFIG = "config/eval_medium.yaml",  # 可修改为不同的配置
    EVAL_DATA_REPO = "${EVAL_DATA_REPO}", 
    EVAL_DATA_PAT = "${EVAL_DATA_PAT}" 
}

[[participants]]
agentbeats_id = "019c16a8-a0b2-77c3-aae3-8e0c23ca5de1"
name = "purple_agent"
env = { 
    OPENAI_API_KEY = "${OPENAI_API_KEY}",
    OPENAI_API_BASE = "https://openrouter.ai/api/v1",  # 或本地 vLLM
    LLM_MODEL = "meta-llama/llama-3.1-70b-instruct"   # 要测试的模型
}

[config]
num_tasks = 100              # 任务数量
conduct_debate = false
timeout_seconds = 600
datasets = ["bizfinbench", "synthetic", "options", "crypto", "gdpval"]
sampling_strategy = "stratified"

# 数据集限制
bizfinbench_limit = 30
synthetic_limit = 20
options_limit = 20
crypto_limit = 12
gdpval_limit = 18
```

---

## 结果收集与分析

### 结果文件格式

每次运行会生成 JSON 格式的结果文件：

```json
{
  "timestamp": "2026-02-02T10:30:00Z",
  "config": {
    "llm_model": "meta-llama/llama-3.1-70b-instruct",
    "num_tasks": 100,
    "eval_config": "config/eval_medium.yaml"
  },
  "results": {
    "overall_score": 65.4,
    "section_scores": {
      "knowledge": 70.2,
      "analysis": 62.5,
      "options": 58.3,
      "crypto": 71.8
    },
    "dataset_scores": {
      "bizfinbench": 0.72,
      "synthetic": 0.58,
      "options": 0.55,
      "crypto": 0.68,
      "gdpval": 0.61
    }
  }
}
```

### 结果汇总脚本

创建 `scripts/aggregate_results.py`:

```python
#!/usr/bin/env python3
"""汇总多次实验结果"""

import json
import sys
from pathlib import Path
import pandas as pd


def load_results(results_dir: str) -> list:
    """加载所有结果文件"""
    results = []
    for file in Path(results_dir).glob("*.json"):
        if file.name == "experiment_summary.json":
            continue
        with open(file) as f:
            data = json.load(f)
            data["filename"] = file.name
            results.append(data)
    return results


def create_summary_table(results: list) -> pd.DataFrame:
    """创建汇总表格"""
    rows = []
    for r in results:
        config = r.get("config", {})
        scores = r.get("results", {})
        rows.append({
            "Model": config.get("llm_model", "unknown"),
            "Tasks": config.get("num_tasks", 0),
            "Overall": scores.get("overall_score", 0),
            "Knowledge": scores.get("section_scores", {}).get("knowledge", 0),
            "Analysis": scores.get("section_scores", {}).get("analysis", 0),
            "Options": scores.get("section_scores", {}).get("options", 0),
            "Crypto": scores.get("section_scores", {}).get("crypto", 0),
            "File": r.get("filename", ""),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Overall", ascending=False)
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_results.py <results_dir>")
        sys.exit(1)
    
    results = load_results(sys.argv[1])
    df = create_summary_table(results)
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # 保存为 CSV
    output_file = Path(sys.argv[1]) / "leaderboard.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")


if __name__ == "__main__":
    main()
```

---

## 时间估算

| 任务规模 | 估计时间 (本地 vLLM) | 估计时间 (API) |
|---------|---------------------|---------------|
| 10 tasks | 5-10 分钟 | 3-5 分钟 |
| 100 tasks | 1-2 小时 | 30-60 分钟 |
| 500 tasks | 5-10 小时 | 3-5 小时 |
| 1000 tasks | 10-20 小时 | 6-10 小时 |

**注意**: 
- Crypto trading scenarios 比较耗时（每个 scenario 有多轮交互）
- 使用 `decision_interval: 5` 可以减少 crypto 评测时间（每 5 步决策一次）
- GDPVal 需要下载 HuggingFace 数据集，首次运行较慢

---

## 推荐的抽样策略

对于代表性评测，建议：

1. **快速验证** (10-20 tasks): 每个 dataset 2-3 个样本
2. **标准评测** (100 tasks): stratified 抽样，确保覆盖所有 task types
3. **完整评测** (500+ tasks): 包含全部 crypto scenarios 和较大的 BizFinBench 样本

```yaml
# 推荐的代表性抽样配置
sampling:
  strategy: stratified  # 分层抽样确保各类任务均衡
  total_limit: 100
  seed: 42              # 固定随机种子保证可重复性
```

---

## 常见问题

### Q: 如何切换不同的 LLM 模型？

修改 `.env` 文件中的 `LLM_MODEL` 和相关 API 配置，然后重启 Purple Agent。

### Q: Crypto 数据放在哪里？

使用 `agentbusters-eval-data/crypto/eval_hidden` 目录，在 eval_config.yaml 中配置路径。

### Q: 如何保证结果可重复？

1. 设置 `PURPLE_LLM_TEMPERATURE=0.0`
2. 在 eval_config.yaml 中设置 `llm_eval.temperature: 0.0`
3. 使用固定的 `sampling.seed`
4. 设置 `shuffle: false`

### Q: 如何并行运行多个实验？

不建议在同一机器上并行运行，因为资源竞争可能导致结果不稳定。建议顺序运行或使用多台机器。

### Q: vLLM 报错 "ImportError: libtorch_cuda.so: cannot open shared object file"？

这是因为安装了 CPU 版本的 PyTorch。**最佳解决方法是让 vLLM 自动管理 PyTorch：**

```bash
# 推荐方法：让 vLLM 自动处理
pip uninstall torch torchvision torchaudio vllm -y
pip install vllm  # vLLM 会自动安装正确的 PyTorch 2.9.x + CUDA
pip install "numpy<2.0"  # 修复 NumPy 兼容性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**如果上述方法不行，手动指定版本：**

```bash
# 手动方法 (避免使用 CUDA 索引 URL，它们会安装错误版本)
pip uninstall torch torchvision torchaudio vllm -y
pip install torch==2.9.1 torchvision torchaudio  # 不用索引 URL
pip install vllm
pip install "numpy<2.0"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

⚠️ **注意**: 不要使用 `--index-url` 因为不同索引会安装错误的 PyTorch 版本 (cu126→2.10, cu124→2.4)。

### Q: vLLM 报错 "undefined symbol: _ZN3c104cuda..." 或类似 C++ 符号错误？

这是因为 vLLM 和 PyTorch 的 CUDA 版本不匹配。**推荐使用 vLLM 自动管理方法：**

```bash
# 最佳方法：让 vLLM 重新管理 PyTorch 版本
pip uninstall vllm torch torchvision torchaudio -y
pip install vllm --no-cache-dir  # 会安装正确的 PyTorch 2.9.x
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import vllm; print('vLLM working')"
```

### Q: 为什么不同的 CUDA 索引安装不同的 PyTorch 版本？

PyTorch 的 CUDA 索引会安装特定版本：
- `cu126` → PyTorch 2.10.x (可能与 vLLM 0.15.0 不兼容)
- `cu124` → PyTorch 2.4.x (太老了)
- `cu121` → PyTorch 2.1.x (太老了)

**解决方法：**
- 方法1（推荐）：`pip install vllm` 让 vLLM 自动选择 PyTorch 2.9.x
- 方法2：手动指定 `pip install torch==2.9.1` 不使用索引URL

### Q: vLLM 报错 "auto tool choice requires --enable-auto-tool-choice"？

这是因为 Purple Agent 使用了自动工具选择功能，但 vLLM 服务没有启用相关参数：

```bash
# 错误的启动命令 (缺少工具调用支持)
vllm serve Qwen/Qwen3-32B --port 8100 --tensor-parallel-size 2

# 正确的启动命令 (添加工具调用支持)
vllm serve Qwen/Qwen3-32B --port 8100 --tensor-parallel-size 2 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

**不同模型的工具解析器 (vLLM 0.15.0+):**
- Qwen3 系列：`--tool-call-parser qwen3_xml`  
- DeepSeek-V3：`--tool-call-parser deepseek_v3`
- Llama3/4 系列：`--tool-call-parser llama3_json` 或 `llama4_json`
- Mistral 系列：`--tool-call-parser mistral`

查看所有可用解析器：`vllm serve --help | grep tool-call-parser`

### Q: 出现 NumPy 兼容性警告？

如果看到 "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6" 警告：

```bash
# 降级 NumPy 到 1.x 版本
pip install "numpy<2.0"

# 验证修复
python -c "import torch; import vllm; print('All packages working')"
```

### Q: vLLM 报错 "CUDA out of memory" 如何解决？

**原因**: 模型太大，超出 GPU 内存容量。

**解决方案 (按优先级排序):**

```bash
# 方案 1: 降低 GPU 内存使用率
vllm serve Qwen/Qwen3-32B \
    --gpu-memory-utilization 0.80  # 从 0.90 降到 0.80

# 方案 2: 使用更小的模型
vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90

# 方案 3: 使用量化 (如果支持)
vllm serve model_name \
    --quantization gptq  # 或 awq, fp8

# 方案 4: 多 GPU 并行 (如果有多张 GPU)
vllm serve large_model \
    --tensor-parallel-size 2  # 使用 2 张 GPU

# 方案 5: 减少上下文长度
vllm serve model_name \
    --max-model-len 16384  # 从 40960 减少到 16384
```

**GH200-96GB 推荐配置:**
- ✅ Qwen3-32B: 最佳平衡
- ✅ Qwen3-14B: 轻量级选择
- ❌ Qwen3-235B-A22B: 需要 480GB 或多 GPU
- ❌ DeepSeek-V3: 需要多 GPU

### Q: DeepSeek-V3 报错 "No valid attention backend found" / "FlashMLA Dense is only supported on Hopper devices"？

**完整错误信息:**
```
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(...use_mla=True...)
Reasons: {
  FLASHMLA: [compute capability not supported, FlashMLA Dense is only supported on Hopper devices.], 
  TRITON_MLA: [kv_cache_dtype not supported],
  ...
}
```

**根本原因**: DeepSeek-V3 使用 **MLA (Multi-head Latent Attention)** 架构，这是一种新型注意力机制，**只能在 Hopper 架构 GPU (H100/H200) 上运行**。

| GPU | Compute Capability | 支持 DeepSeek-V3? |
|-----|-------------------|------------------|
| A100 | 8.0 | ❌ 不支持 |
| A6000 | 8.6 | ❌ 不支持 |
| RTX 4090 | 8.9 | ❌ 不支持 |
| **H100** | **9.0** | ✅ 支持 |
| **H200** | **9.0** | ✅ 支持 |
| **GH200** | **9.0** | ✅ 支持 |

**解决方案:**

1. **尝试移除 FP8 KV cache** (可能让 TRITON_MLA 工作):
```bash
# 移除 --kv-cache-dtype fp8_e4m3 参数
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --quantization fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v3
```

2. **使用替代模型** (如果方案1失败 - A100 推荐):
```bash
# Qwen3-32B - 推荐
vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

# Llama-3.1-70B
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json

# Mixtral-8x22B (MoE 但无 MLA 限制)
vllm serve mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral
```

3. **使用 API 服务** (无需本地 GPU):
```bash
# 使用 OpenRouter API
export OPENAI_API_BASE=https://openrouter.ai/api/v1
export LLM_MODEL=deepseek/deepseek-chat
```

### Q: 如何检查 GPU 内存使用情况？

```bash
# 查看 GPU 状态
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi

# 在 Python 中检查
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

### Q: 评测时出现 "max_tokens is too large" 或上下文长度超限错误？

**错误示例:**
```
llm_bizfinbench_failed: Error code: 400 - {'error': {'message': "'max_tokens' is too large: 800. 
This model's maximum context length is 32768 tokens and your request has 32180 input tokens..."}}
```

**原因**: 评测使用的 LLM-as-judge 模型上下文长度不足以容纳长输入 + 评分输出。

**解决方案:**

1. **使用更大上下文的评测模型** (推荐):
```yaml
# config/eval_config.yaml
llm_eval:
  enabled: true
  model: gpt-4o-mini  # 支持 128K context
  temperature: 0.0
```

2. **增加 vLLM 上下文长度** (如果使用本地模型评测):
```bash
vllm serve Qwen/Qwen3-32B \
    --max-model-len 65536  # 增加到 64K
```

3. **系统已自动优化**: 评测器会自动截断过长输入并动态调整 max_tokens

### Q: 评测时出现 "LLM returned invalid JSON for PRBench evaluation"？

**原因**: LLM 没有返回有效的 JSON 格式响应，可能因为：
- 输出被截断
- 模型不遵循 JSON 格式指令
- 上下文溢出导致响应异常

**解决方案:**

1. **使用遵循指令能力更强的模型**:
```yaml
llm_eval:
  model: gpt-4o-mini  # 或 claude-3-haiku 等
```

2. **检查 vLLM 日志** 确认模型正常响应

3. **系统已自动处理**: 评测器会自动重试并使用简化提示

**注意**: 这些错误不会导致评测完全失败，只是该任务会使用备用评分策略（规则匹配）。
