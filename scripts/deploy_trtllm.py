#!/usr/bin/env python3
"""
TensorRT-LLM 模型部署脚本

用于部署 DeepSeek-V3.2-NVFP4 等 NVIDIA 预量化模型。
TRT-LLM + NVFP4 是 NVIDIA 官方推荐的 DeepSeek-V3 部署路线。

Usage:
    # 列出所有可用模型配置
    python scripts/deploy_trtllm.py --list
    
    # 部署 DeepSeek-V3.2-NVFP4 (8x H100)
    python scripts/deploy_trtllm.py --model deepseek-v3-nvfp4
    
    # 自定义端口
    python scripts/deploy_trtllm.py --model deepseek-v3-nvfp4 --port 8001
    
    # 生成 Docker 命令但不执行
    python scripts/deploy_trtllm.py --model deepseek-v3-nvfp4 --dry-run

Why TRT-LLM + NVFP4?
    - NVFP4 是预量化模型，不需要 build engine
    - 比 vLLM 更高吞吐量
    - 8x H100 80GB 稳定运行 671B MoE
    - NVIDIA 官方支持的 DeepSeek-V3 路线
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# TRT-LLM 模型配置
TRTLLM_MODEL_CONFIGS = {
    # ========== DeepSeek-V3.2 NVFP4 (推荐) ==========
    "deepseek-v3-nvfp4": {
        "name": "DeepSeek-V3.2-NVFP4 (TRT-LLM)",
        "hf_model": "nvidia/DeepSeek-V3.2-NVFP4",
        "description": "NVIDIA 官方预量化 FP4，8x H100 80GB，最高吞吐",
        "min_gpus": 8,
        "recommended_gpus": 8,
        "gpu_memory": "8x80GB (H100/A100)",
        "default_max_len": 32768,
        "tensor_parallel_size": 8,
        "container_image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc1",
        "trust_remote_code": True,
        "env_model_name": "deepseek-v3.2-nvfp4",
    },
    "deepseek-v3-nvfp4-4gpu": {
        "name": "DeepSeek-V3.2-NVFP4 (4x GPU)",
        "hf_model": "nvidia/DeepSeek-V3.2-NVFP4",
        "description": "NVIDIA 预量化 FP4，4x A100 80GB (实验性)",
        "min_gpus": 4,
        "recommended_gpus": 4,
        "gpu_memory": "4x80GB",
        "default_max_len": 16384,
        "tensor_parallel_size": 4,
        "container_image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc1",
        "trust_remote_code": True,
        "env_model_name": "deepseek-v3.2-nvfp4",
    },
}


def list_models():
    """列出所有可用的 TRT-LLM 模型配置"""
    print("\n" + "=" * 70)
    print("TensorRT-LLM 可用模型配置")
    print("=" * 70)
    
    print("\n🚀 NVIDIA 预量化模型 (推荐):")
    print("-" * 70)
    
    for key, cfg in TRTLLM_MODEL_CONFIGS.items():
        print(f"\n  {key}")
        print(f"    模型: {cfg['name']}")
        print(f"    描述: {cfg['description']}")
        print(f"    GPU: {cfg['min_gpus']}x ({cfg['gpu_memory']})")
        print(f"    HuggingFace: {cfg['hf_model']}")
        print(f"    容器: {cfg['container_image']}")
    
    print("\n" + "=" * 70)
    print("部署架构:")
    print("-" * 70)
    print("""
    Client (OpenAI SDK / curl)
            |
            v
    ┌─────────────────────────┐
    │  OpenAI-compatible API  │  ← FastAPI (trtllm_openai_api.py)
    │  (adapter / gateway)    │
    └──────────┬──────────────┘
               |
               v
    ┌─────────────────────────┐
    │   TensorRT-LLM Runtime  │
    │   (DeepSeek-V3.2-NVFP4) │
    └─────────────────────────┘
               |
               v
           8×H100 / A100
    """)
    print("=" * 70)


def build_docker_command(
    model_key: str,
    port: int = 8000,
    model_path: str = "/mnt/models",
) -> list:
    """构建 Docker 启动命令"""
    
    if model_key not in TRTLLM_MODEL_CONFIGS:
        print(f"Error: 未知模型 '{model_key}'")
        print(f"可用模型: {', '.join(TRTLLM_MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    cfg = TRTLLM_MODEL_CONFIGS[model_key]
    
    cmd = [
        "docker", "run", "--rm", "-it",
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-p", f"{port}:8000",
        "-v", f"{model_path}:/models",
        "-v", f"{Path(__file__).parent.parent / 'src' / 'trtllm_api'}:/app",
        "-w", "/app",
        cfg["container_image"],
        "python", "trtllm_openai_api.py",
        "--model", cfg["hf_model"],
        "--tensor-parallel-size", str(cfg["tensor_parallel_size"]),
    ]
    
    return cmd


def generate_api_start_script(model_key: str, port: int = 8000) -> str:
    """生成容器内运行的启动脚本"""
    
    cfg = TRTLLM_MODEL_CONFIGS[model_key]
    
    script = f'''#!/bin/bash
# TRT-LLM OpenAI API 启动脚本
# 在 TRT-LLM 容器内运行

cd /app
pip install fastapi uvicorn pydantic

python trtllm_openai_api.py \\
    --model {cfg["hf_model"]} \\
    --tensor-parallel-size {cfg["tensor_parallel_size"]} \\
    --port {port}
'''
    return script


def update_env_file(model_key: str, port: int = 8000):
    """更新 .env 文件中的模型配置"""
    
    cfg = TRTLLM_MODEL_CONFIGS[model_key]
    env_path = Path(__file__).parent.parent / ".env"
    
    if not env_path.exists():
        print("Error: .env 文件不存在")
        return False
    
    content = env_path.read_text()
    lines = content.split("\n")
    new_lines = []
    updated = False
    
    for line in lines:
        stripped = line.strip()
        
        # 更新 OPENAI_API_BASE
        if stripped.startswith("OPENAI_API_BASE=") and not stripped.startswith("#"):
            new_lines.append(f"OPENAI_API_BASE=http://localhost:{port}/v1")
            updated = True
            continue
        
        # 更新 LLM_MODEL
        if stripped.startswith("LLM_MODEL=") and not stripped.startswith("#"):
            new_lines.append(f"LLM_MODEL={cfg['env_model_name']}")
            continue
        
        # 更新 OPENAI_API_KEY (for local TRT-LLM)
        if stripped.startswith("OPENAI_API_KEY=") and not stripped.startswith("#"):
            new_lines.append("OPENAI_API_KEY=dummy")
            continue
        
        new_lines.append(line)
    
    env_path.write_text("\n".join(new_lines))
    
    if updated:
        print(f"✅ 已更新 .env 文件:")
        print(f"   OPENAI_API_BASE=http://localhost:{port}/v1")
        print(f"   LLM_MODEL={cfg['env_model_name']}")
        print(f"   OPENAI_API_KEY=dummy")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM 模型部署脚本 (DeepSeek-V3.2-NVFP4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(TRTLLM_MODEL_CONFIGS.keys()),
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
        help="API 服务端口 (default: 8000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/models",
        help="模型存储路径 (default: /mnt/models)",
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
    parser.add_argument(
        "--generate-script",
        action="store_true",
        help="生成容器内启动脚本",
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
    
    cfg = TRTLLM_MODEL_CONFIGS[args.model]
    
    print("\n" + "=" * 70)
    print(f"部署模型: {cfg['name']}")
    print("=" * 70)
    print(f"\n描述: {cfg['description']}")
    print(f"HuggingFace: {cfg['hf_model']}")
    print(f"端口: {args.port}")
    print(f"容器: {cfg['container_image']}")
    
    # 生成启动脚本
    if args.generate_script:
        script = generate_api_start_script(args.model, args.port)
        script_path = Path(__file__).parent.parent / "src" / "trtllm_api" / "start.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        print(f"\n✅ 已生成启动脚本: {script_path}")
        return
    
    # 构建 Docker 命令
    cmd = build_docker_command(
        model_key=args.model,
        port=args.port,
        model_path=args.model_path,
    )
    
    print(f"\nDocker 命令:")
    print(f"  {' '.join(cmd)}")
    
    # 更新 .env
    if args.update_env:
        print("\n")
        update_env_file(args.model, args.port)
    
    # 执行
    if args.dry_run:
        print("\n[Dry run - 不执行命令]")
        print("\n📌 手动启动步骤:")
        print("1. 拉取 TRT-LLM 容器:")
        print(f"   docker pull {cfg['container_image']}")
        print("\n2. 启动容器:")
        print(f"   {' '.join(cmd)}")
        print("\n3. 在容器内启动 API:")
        print("   cd /app && python trtllm_openai_api.py")
    else:
        print("\n启动 TRT-LLM 服务...")
        print("按 Ctrl+C 停止服务\n")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n服务已停止")
        except FileNotFoundError:
            print("\nError: Docker 未安装或未运行")
            sys.exit(1)


if __name__ == "__main__":
    main()
