#!/usr/bin/env python3
"""
vLLM 模型部署脚本

用于快速部署 Qwen3-32B、DeepSeek-V3.2、Qwen3-14B 等模型。

Usage:
    # 列出所有可用模型配置
    python scripts/deploy_vllm.py --list
    
    # 部署 Qwen3-32B
    python scripts/deploy_vllm.py --model qwen3-32b
    
    # 部署 DeepSeek-V3
    python scripts/deploy_vllm.py --model deepseek-v3 --gpus 4
    
    # 部署 Qwen3-14B (轻量级)
    python scripts/deploy_vllm.py --model qwen3-14b
    
    # 自定义端口
    python scripts/deploy_vllm.py --model qwen3-32b --port 8001
    
    # 生成部署命令但不执行
    python scripts/deploy_vllm.py --model qwen3-32b --dry-run

Examples:
    # 快速开始：使用默认配置部署 Qwen3-32B
    python scripts/deploy_vllm.py --model qwen3-32b
    
    # 然后更新 .env 文件
    python scripts/deploy_vllm.py --model qwen3-32b --update-env
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# 模型配置定义
MODEL_CONFIGS = {
    # ========== B200 顶配 (单卡 192GB HBM3e) ==========
    "qwen3-235b-b200": {
        "name": "Qwen3-235B-A22B (B200 单卡)",
        "hf_model": "Qwen/Qwen3-235B-A22B",
        "description": "B200 顶配：单卡 192GB 运行 235B MoE",
        "min_gpus": 1,
        "recommended_gpus": 1,
        "gpu_memory": "1x192GB (B200)",
        "default_max_len": 65536,  # B200 大显存支持更长 context
        "extra_args": [
            "--trust-remote-code",
        ],
        "env_model_name": "Qwen/Qwen3-235B-A22B",
    },
    "deepseek-v3-b200": {
        "name": "DeepSeek-V3.2-671B (3x B200)",
        "hf_model": "deepseek-ai/DeepSeek-V3",
        "description": "B200 顶配：3卡运行 671B MoE，BF16 全精度",
        "min_gpus": 3,
        "recommended_gpus": 3,
        "gpu_memory": "3x192GB (B200)",
        "default_max_len": 65536,  # B200 大显存支持更长 context
        "extra_args": [
            "--trust-remote-code",
        ],
        "env_model_name": "deepseek-ai/DeepSeek-V3",
    },
    # ========== H100 顶配 (8x 80GB) ==========
    "deepseek-v3-fp8": {
        "name": "DeepSeek-V3.2-671B (FP8)",
        "hf_model": "deepseek-ai/DeepSeek-V3",
        "description": "H100 顶配：671B MoE，FP8 原生精度，8x H100 80GB",
        "min_gpus": 8,
        "recommended_gpus": 8,
        "gpu_memory": "8x80GB",
        "default_max_len": 32768,
        "extra_args": [
            "--trust-remote-code",
            "--dtype", "float8_e4m3fn",  # FP8 原生
            "--quantization", "fp8",
            "--kv-cache-dtype", "fp8_e4m3",
        ],
        "env_model_name": "deepseek-ai/DeepSeek-V3",
    },
    "qwen3-235b": {
        "name": "Qwen3-235B-A22B (MoE)",
        "hf_model": "Qwen/Qwen3-235B-A22B",
        "description": "H100 顶配：235B MoE (22B 激活)，8x H100 80GB",
        "min_gpus": 8,
        "recommended_gpus": 8,
        "gpu_memory": "8x80GB",
        "default_max_len": 32768,
        "extra_args": [
            "--trust-remote-code",
        ],
        "env_model_name": "Qwen/Qwen3-235B-A22B",
    },
    # ========== 主要目标模型 ==========
    "qwen3-32b": {
        "name": "Qwen3-32B",
        "hf_model": "Qwen/Qwen3-32B",
        "description": "推荐模型，平衡性能与资源消耗",
        "min_gpus": 1,
        "recommended_gpus": 1,
        "gpu_memory": "80GB",
        "default_max_len": 32768,
        "extra_args": [],
        "env_model_name": "Qwen/Qwen3-32B",
    },
    "deepseek-v3": {
        "name": "DeepSeek-V3.2",
        "hf_model": "deepseek-ai/DeepSeek-V3",
        "description": "671B MoE 模型，需要多 GPU (FP16/BF16)",
        "min_gpus": 4,
        "recommended_gpus": 8,
        "gpu_memory": "4x80GB+",
        "default_max_len": 16384,
        "extra_args": ["--trust-remote-code"],
        "env_model_name": "deepseek-ai/DeepSeek-V3",
    },
    "qwen3-14b": {
        "name": "Qwen3-14B",
        "hf_model": "Qwen/Qwen3-14B",
        "description": "轻量级模型，适合单 GPU 部署",
        "min_gpus": 1,
        "recommended_gpus": 1,
        "gpu_memory": "24GB+",
        "default_max_len": 32768,
        "extra_args": [],
        "env_model_name": "Qwen/Qwen3-14B",
    },
    # ========== 其他备选模型 ==========
    "deepseek-r1": {
        "name": "DeepSeek-R1",
        "hf_model": "deepseek-ai/DeepSeek-R1",
        "description": "DeepSeek 推理模型",
        "min_gpus": 4,
        "recommended_gpus": 8,
        "gpu_memory": "4x80GB+",
        "default_max_len": 16384,
        "extra_args": ["--trust-remote-code"],
        "env_model_name": "deepseek-ai/DeepSeek-R1",
    },
    "llama3.1-70b": {
        "name": "Llama-3.1-70B",
        "hf_model": "meta-llama/Llama-3.1-70B-Instruct",
        "description": "Meta Llama 3.1 70B 指令模型",
        "min_gpus": 2,
        "recommended_gpus": 2,
        "gpu_memory": "2x40GB+",
        "default_max_len": 8192,
        "extra_args": [],
        "env_model_name": "meta-llama/Llama-3.1-70B-Instruct",
    },
    "llama3.1-8b": {
        "name": "Llama-3.1-8B",
        "hf_model": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "轻量级 Llama 模型",
        "min_gpus": 1,
        "recommended_gpus": 1,
        "gpu_memory": "16GB+",
        "default_max_len": 8192,
        "extra_args": [],
        "env_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "mixtral-8x22b": {
        "name": "Mixtral-8x22B",
        "hf_model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "description": "Mistral MoE 模型",
        "min_gpus": 2,
        "recommended_gpus": 4,
        "gpu_memory": "2x80GB+",
        "default_max_len": 8192,
        "extra_args": [],
        "env_model_name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    },
}


def list_models():
    """列出所有可用的模型配置"""
    print("\n" + "=" * 70)
    print("可用模型配置")
    print("=" * 70)
    
    # 顶配模型 (8x H100)
    print("\n🚀 顶配模型 (8x H100 80GB):")
    print("-" * 70)
    for key in ["deepseek-v3-fp8", "qwen3-235b"]:
        cfg = MODEL_CONFIGS[key]
        print(f"\n  {key}")
        print(f"    模型: {cfg['name']}")
        print(f"    描述: {cfg['description']}")
        print(f"    GPU: {cfg['min_gpus']}x ({cfg['gpu_memory']})")
        print(f"    HuggingFace: {cfg['hf_model']}")
    
    # 主要目标模型
    print("\n\n📌 主要目标模型:")
    print("-" * 70)
    for key in ["qwen3-32b", "deepseek-v3", "qwen3-14b"]:
        cfg = MODEL_CONFIGS[key]
        print(f"\n  {key}")
        print(f"    模型: {cfg['name']}")
        print(f"    描述: {cfg['description']}")
        print(f"    最低 GPU: {cfg['min_gpus']} ({cfg['gpu_memory']})")
        print(f"    HuggingFace: {cfg['hf_model']}")
    
    # 其他模型
    print("\n\n📦 其他备选模型:")
    print("-" * 70)
    for key, cfg in MODEL_CONFIGS.items():
        if key not in ["qwen3-32b", "deepseek-v3", "qwen3-14b", "deepseek-v3-fp8", "qwen3-235b"]:
            print(f"\n  {key}")
            print(f"    模型: {cfg['name']} - {cfg['description']}")
            print(f"    GPU: {cfg['min_gpus']}+ ({cfg['gpu_memory']})")


def build_vllm_command(
    model_key: str,
    port: int = 8000,
    gpus: Optional[int] = None,
    max_len: Optional[int] = None,
    gpu_util: float = 0.9,
) -> list:
    """构建 vLLM 启动命令"""
    
    if model_key not in MODEL_CONFIGS:
        print(f"Error: 未知模型 '{model_key}'")
        print(f"可用模型: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    cfg = MODEL_CONFIGS[model_key]
    
    # 确定 GPU 数量
    num_gpus = gpus if gpus else cfg["recommended_gpus"]
    if num_gpus < cfg["min_gpus"]:
        print(f"Warning: {cfg['name']} 最少需要 {cfg['min_gpus']} 个 GPU")
        num_gpus = cfg["min_gpus"]
    
    # 确定 context 长度
    context_len = max_len if max_len else cfg["default_max_len"]
    
    # 构建命令
    cmd = [
        "vllm", "serve", cfg["hf_model"],
        "--port", str(port),
        "--tensor-parallel-size", str(num_gpus),
        "--max-model-len", str(context_len),
        "--gpu-memory-utilization", str(gpu_util),
    ]
    
    # 添加额外参数
    cmd.extend(cfg["extra_args"])
    
    return cmd


def update_env_file(model_key: str, port: int = 8000):
    """更新 .env 文件中的模型配置"""
    
    cfg = MODEL_CONFIGS[model_key]
    env_path = Path(__file__).parent.parent / ".env"
    
    if not env_path.exists():
        # 从 .env.example 复制
        example_path = env_path.parent / ".env.example"
        if example_path.exists():
            env_path.write_text(example_path.read_text())
            print(f"✅ 已从 .env.example 创建 .env")
        else:
            print("Error: .env.example 不存在")
            return False
    
    # 读取当前内容
    content = env_path.read_text()
    lines = content.split("\n")
    new_lines = []
    
    # 更新配置
    updated_keys = set()
    for line in lines:
        stripped = line.strip()
        
        # 跳过注释
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        
        # 更新 LLM_MODEL
        if stripped.startswith("LLM_MODEL=") or stripped.startswith("# LLM_MODEL="):
            if "LLM_MODEL" not in updated_keys:
                new_lines.append(f"LLM_MODEL={cfg['env_model_name']}")
                updated_keys.add("LLM_MODEL")
            continue
        
        # 更新 OPENAI_API_BASE
        if stripped.startswith("OPENAI_API_BASE=") or stripped.startswith("# OPENAI_API_BASE="):
            if "OPENAI_API_BASE" not in updated_keys:
                new_lines.append(f"OPENAI_API_BASE=http://localhost:{port}/v1")
                updated_keys.add("OPENAI_API_BASE")
            continue
        
        # 更新 OPENAI_BASE_URL
        if stripped.startswith("OPENAI_BASE_URL=") or stripped.startswith("# OPENAI_BASE_URL="):
            if "OPENAI_BASE_URL" not in updated_keys:
                new_lines.append(f"OPENAI_BASE_URL=http://localhost:{port}/v1")
                updated_keys.add("OPENAI_BASE_URL")
            continue
        
        # 更新 OPENAI_API_KEY (for vLLM)
        if stripped.startswith("OPENAI_API_KEY=") and "dummy" not in stripped:
            if "OPENAI_API_KEY" not in updated_keys:
                new_lines.append("OPENAI_API_KEY=dummy")
                updated_keys.add("OPENAI_API_KEY")
            continue
        
        new_lines.append(line)
    
    # 写回文件
    env_path.write_text("\n".join(new_lines))
    
    print(f"✅ 已更新 .env 文件:")
    print(f"   LLM_MODEL={cfg['env_model_name']}")
    print(f"   OPENAI_API_BASE=http://localhost:{port}/v1")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="vLLM 模型部署脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODEL_CONFIGS.keys()),
        help="要部署的模型",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用模型",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="vLLM 服务端口 (default: 8000)",
    )
    parser.add_argument(
        "--gpus", "-g",
        type=int,
        help="GPU 数量 (tensor parallel size)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        help="最大 context 长度",
    )
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=0.9,
        help="GPU 内存利用率 (default: 0.9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不执行",
    )
    parser.add_argument(
        "--update-env",
        action="store_true",
        help="同时更新 .env 文件",
    )
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        list_models()
        return
    
    # 检查是否指定了模型
    if not args.model:
        print("Error: 请指定要部署的模型 (--model)")
        print("使用 --list 查看可用模型")
        sys.exit(1)
    
    # 构建命令
    cmd = build_vllm_command(
        model_key=args.model,
        port=args.port,
        gpus=args.gpus,
        max_len=args.max_len,
        gpu_util=args.gpu_util,
    )
    
    cfg = MODEL_CONFIGS[args.model]
    
    print("\n" + "=" * 70)
    print(f"部署模型: {cfg['name']}")
    print("=" * 70)
    print(f"\n描述: {cfg['description']}")
    print(f"HuggingFace: {cfg['hf_model']}")
    print(f"端口: {args.port}")
    print(f"\n命令:")
    print(f"  {' '.join(cmd)}")
    
    # 更新 .env
    if args.update_env:
        print("\n")
        update_env_file(args.model, args.port)
    
    # 执行
    if args.dry_run:
        print("\n[Dry run - 不执行命令]")
    else:
        print("\n启动 vLLM 服务...")
        print("按 Ctrl+C 停止服务\n")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n服务已停止")
        except FileNotFoundError:
            print("\nError: vLLM 未安装")
            print("请先安装: pip install vllm")
            sys.exit(1)


if __name__ == "__main__":
    main()
