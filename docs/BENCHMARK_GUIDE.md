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
cd d:\code\finbenchmark\AgentBusters

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装依赖
pip install -e ".[dev]"
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
# 安装 vLLM
pip install vllm
```

#### 推荐模型部署

**⚡ B200 顶配 (单卡 192GB HBM3e) - 最强单卡**

**1. Qwen3-235B-A22B (单卡 B200) - 单卡跑 235B！**
```bash
# 1x B200 192GB - 单卡运行 235B MoE
vllm serve Qwen/Qwen3-235B-A22B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code

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
    --trust-remote-code

# 快速部署命令
python scripts/deploy_vllm.py --model deepseek-v3-b200
```

---

**🚀 H100 顶配 (8x 80GB)**

**1. DeepSeek-V3.2-671B (FP8 原生) - H100 最强性能**
```bash
# 8x H100 80GB, FP8 原生精度
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype float8_e4m3fn \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3
```

**2. Qwen3-235B-A22B (MoE) - 顶级 MoE**
```bash
# 8x H100 80GB, 235B 参数 (22B 激活)
vllm serve Qwen/Qwen3-235B-A22B \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code
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
    --gpu-memory-utilization 0.9

# 双 GPU (2x A100 40GB / 2x RTX 4090)
vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 16384
```

**2. DeepSeek-V3.2（高性能，需要多 GPU）**
```bash
# 4x A100 80GB 或 8x A100 40GB
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code

# 8x GPU 配置（更大 context）
vllm serve deepseek-ai/DeepSeek-V3 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --trust-remote-code
```

**3. Qwen3-14B（轻量级，适合单 GPU）**
```bash
# 单 GPU (RTX 4090 24GB / A100 40GB)
vllm serve Qwen/Qwen3-14B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9

# RTX 3090 24GB（减少 context 长度）
vllm serve Qwen/Qwen3-14B \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
```

#### GPU 内存需求参考

| 模型 | 参数量 | 最低 GPU | 推荐配置 | Context 长度 |
|------|--------|----------|----------|-------------|
| Qwen3-14B | 14B | 1x RTX 4090 (24GB) | 1x A100 40GB | 32K |
| Qwen3-32B | 32B | 2x RTX 4090 | 1x A100 80GB | 32K |
| DeepSeek-V3.2 | 671B MoE | 4x A100 80GB | 8x H100 80GB | 32K |
| Qwen3-235B-A22B | 235B MoE (22B激活) | 8x H100 80GB | 8x H100 80GB | 32K |
| DeepSeek-V3 FP8 | 671B MoE | 8x H100 80GB | 8x H100 80GB | 32K |
| **⚡ Qwen3-235B (B200)** | **235B MoE** | **1x B200 192GB** | **1x B200** | **65K** |
| **⚡ DeepSeek-V3 (B200)** | **671B MoE** | **3x B200 192GB** | **3x B200** | **65K** |

#### 其他模型（备选）
```bash
# Llama 3.1 70B
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2

# Mixtral 8x22B
vllm serve mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --port 8000 \
    --tensor-parallel-size 2
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
