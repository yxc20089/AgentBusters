#!/usr/bin/env python3
"""
AgentBusters Benchmark 一键运行脚本

完整流程：启动 agents -> 运行评测 -> 收集结果

Usage:
    # 快速测试 (10 tasks)
    python scripts/run_benchmark.py --quick
    
    # 中等规模评测 (100 tasks)
    python scripts/run_benchmark.py --model qwen3-32b --tasks 100
    
    # 大规模评测 (500 tasks)
    python scripts/run_benchmark.py --model qwen3-32b --config eval_large.yaml --tasks 500
    
    # 对比三个目标模型
    python scripts/run_benchmark.py --compare-models
    
    # 只启动 agents (不运行评测)
    python scripts/run_benchmark.py --model qwen3-32b --start-only

Examples:
    # 使用本地 vLLM (假设已启动)
    python scripts/run_benchmark.py --model qwen3-32b --tasks 100 --vllm-url http://localhost:8000/v1
    
    # 使用 OpenRouter API
    python scripts/run_benchmark.py --model qwen3-32b --tasks 100 --api openrouter

注意：运行前请确保：
1. vLLM 服务已启动 (或使用 --api openrouter)
2. .env 文件已正确配置
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 模型配置
MODELS = {
    # B200 顶配 (192GB HBM3e)
    "qwen3-235b-b200": "Qwen/Qwen3-235B-A22B",   # 1x B200 192GB
    "deepseek-v3-b200": "deepseek-ai/DeepSeek-V3", # 3x B200 192GB
    # H100 顶配 (8x 80GB)
    "deepseek-v3-fp8": "deepseek-ai/DeepSeek-V3",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B",
    # 主要目标模型
    "qwen3-32b": "Qwen/Qwen3-32B",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "qwen3-14b": "Qwen/Qwen3-14B",
    # 其他
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
}

# 评测配置
EVAL_CONFIGS = {
    "quick": ("config/eval_quick_test.yaml", 10),
    "medium": ("config/eval_medium.yaml", 100),
    "large": ("config/eval_large.yaml", 500),
}


def check_vllm_health(url: str, timeout: int = 5) -> bool:
    """检查 vLLM 服务是否可用"""
    import httpx
    try:
        response = httpx.get(f"{url}/models", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def start_green_agent(
    eval_config: str,
    port: int = 9109,
    host: str = "0.0.0.0",
) -> subprocess.Popen:
    """启动 Green Agent"""
    cmd = [
        sys.executable,
        "src/cio_agent/a2a_server.py",
        "--host", host,
        "--port", str(port),
        "--eval-config", eval_config,
        "--store-predicted",
        "--predicted-max-chars", "200",
    ]
    
    print(f"🟢 启动 Green Agent (port {port})...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def start_purple_agent(
    port: int = 9110,
    host: str = "0.0.0.0",
) -> subprocess.Popen:
    """启动 Purple Agent"""
    cmd = [
        "purple-agent", "serve",
        "--host", host,
        "--port", str(port),
        "--card-url", f"http://127.0.0.1:{port}",
    ]
    
    print(f"🟣 启动 Purple Agent (port {port})...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def wait_for_agent(url: str, name: str, timeout: int = 60) -> bool:
    """等待 agent 启动"""
    import httpx
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(f"{url}/.well-known/agent.json", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {name} 已就绪")
                return True
        except:
            pass
        time.sleep(2)
        print(f"  ⏳ 等待 {name}...")
    
    print(f"  ❌ {name} 启动超时")
    return False


def run_evaluation(
    green_url: str,
    purple_url: str,
    num_tasks: int,
    output_file: str,
    timeout: int = 3600,
    conduct_debate: bool = False,
) -> dict:
    """运行评测"""
    cmd = [
        sys.executable,
        "scripts/run_a2a_eval.py",
        "--green-url", green_url,
        "--purple-url", purple_url,
        "--num-tasks", str(num_tasks),
        "--timeout", str(timeout),
        "-v",
        "-o", output_file,
    ]
    
    if conduct_debate:
        cmd.append("--conduct-debate")
    
    print(f"\n📊 开始评测 ({num_tasks} tasks)...")
    result = subprocess.run(cmd, capture_output=False)
    
    return {"success": result.returncode == 0, "output_file": output_file}


def update_env_for_model(model_key: str, api_base: str):
    """临时更新环境变量"""
    if model_key in MODELS:
        os.environ["LLM_MODEL"] = MODELS[model_key]
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_BASE_URL"] = api_base
    if "localhost" in api_base or "127.0.0.1" in api_base:
        os.environ["OPENAI_API_KEY"] = "dummy"


def run_single_benchmark(
    model_key: str,
    eval_config: str,
    num_tasks: int,
    vllm_url: str,
    output_dir: str,
    green_port: int = 9109,
    purple_port: int = 9110,
    timeout: int = 3600,
    start_only: bool = False,
) -> dict:
    """运行单个模型的 benchmark"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"{model_key}_{timestamp}.json"
    
    print("\n" + "=" * 70)
    print(f"📋 Benchmark: {model_key}")
    print("=" * 70)
    print(f"  模型: {MODELS.get(model_key, model_key)}")
    print(f"  配置: {eval_config}")
    print(f"  任务数: {num_tasks}")
    print(f"  vLLM: {vllm_url}")
    
    # 更新环境变量
    update_env_for_model(model_key, vllm_url)
    
    # 检查 vLLM
    if not check_vllm_health(vllm_url):
        print(f"\n⚠️  vLLM 服务不可用: {vllm_url}")
        print("请先启动 vLLM 服务:")
        print(f"  python scripts/deploy_vllm.py --model {model_key}")
        return {"success": False, "error": "vLLM not available"}
    
    print(f"  ✅ vLLM 服务可用")
    
    # 启动 agents
    processes = []
    try:
        green_proc = start_green_agent(eval_config, green_port)
        processes.append(green_proc)
        
        purple_proc = start_purple_agent(purple_port)
        processes.append(purple_proc)
        
        # 等待 agents 启动
        green_url = f"http://127.0.0.1:{green_port}"
        purple_url = f"http://127.0.0.1:{purple_port}"
        
        if not wait_for_agent(green_url, "Green Agent"):
            return {"success": False, "error": "Green Agent failed to start"}
        
        if not wait_for_agent(purple_url, "Purple Agent"):
            return {"success": False, "error": "Purple Agent failed to start"}
        
        if start_only:
            print("\n✅ Agents 已启动，按 Ctrl+C 停止")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n停止 agents...")
            return {"success": True, "mode": "start-only"}
        
        # 运行评测
        result = run_evaluation(
            green_url=green_url,
            purple_url=purple_url,
            num_tasks=num_tasks,
            output_file=str(output_file),
            timeout=timeout,
        )
        
        return result
        
    finally:
        # 清理进程
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except:
                    proc.kill()


def compare_models(
    models: list,
    eval_config: str,
    num_tasks: int,
    vllm_url: str,
    output_dir: str,
):
    """对比多个模型"""
    
    results = []
    
    for model_key in models:
        print(f"\n\n{'#' * 70}")
        print(f"# 模型 {len(results) + 1}/{len(models)}: {model_key}")
        print(f"{'#' * 70}")
        
        result = run_single_benchmark(
            model_key=model_key,
            eval_config=eval_config,
            num_tasks=num_tasks,
            vllm_url=vllm_url,
            output_dir=output_dir,
        )
        
        result["model"] = model_key
        results.append(result)
        
        # 模型之间的间隔
        if model_key != models[-1]:
            print("\n⏳ 切换到下一个模型...")
            print("   请在另一个终端中重启 vLLM 服务:")
            next_idx = models.index(model_key) + 1
            print(f"   python scripts/deploy_vllm.py --model {models[next_idx]}")
            input("   准备好后按 Enter 继续...")
    
    # 保存汇总
    summary_file = Path(output_dir) / "comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "eval_config": eval_config,
            "num_tasks": num_tasks,
            "results": results,
        }, f, indent=2)
    
    print(f"\n✅ 对比结果已保存: {summary_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="AgentBusters Benchmark 一键运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # 模型选择
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default="qwen3-32b",
        help="要评测的模型 (default: qwen3-32b)",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="对比三个目标模型 (qwen3-32b, deepseek-v3, qwen3-14b)",
    )
    parser.add_argument(
        "--compare-flagship",
        action="store_true",
        help="对比顶配模型 (deepseek-v3-fp8, qwen3-235b) - 需要 8x H100",
    )
    
    # 评测规模
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速测试模式 (10 tasks)",
    )
    parser.add_argument(
        "--tasks", "-n",
        type=int,
        default=100,
        help="任务数量 (default: 100)",
    )
    parser.add_argument(
        "--config", "-c",
        help="评测配置文件 (default: config/eval_medium.yaml)",
    )
    
    # 服务配置
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM 服务地址",
    )
    parser.add_argument(
        "--green-port",
        type=int,
        default=9109,
        help="Green Agent 端口",
    )
    parser.add_argument(
        "--purple-port",
        type=int,
        default=9110,
        help="Purple Agent 端口",
    )
    
    # 输出
    parser.add_argument(
        "--output-dir", "-o",
        default="results/benchmarks",
        help="结果输出目录",
    )
    
    # 其他选项
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="评测超时时间 (秒)",
    )
    parser.add_argument(
        "--start-only",
        action="store_true",
        help="只启动 agents，不运行评测",
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 确定评测配置
    if args.quick:
        eval_config, num_tasks = EVAL_CONFIGS["quick"]
    elif args.config:
        eval_config = args.config
        num_tasks = args.tasks
    else:
        eval_config = "config/eval_medium.yaml"
        num_tasks = args.tasks
    
    print("\n" + "=" * 70)
    print("🚀 AgentBusters Benchmark Runner")
    print("=" * 70)
    
    # 运行模式
    if args.compare_flagship:
        compare_models(
            models=["deepseek-v3-fp8", "qwen3-235b"],
            eval_config="config/eval_large.yaml",
            num_tasks=500,
            vllm_url=args.vllm_url,
            output_dir=args.output_dir,
        )
    elif args.compare_models:
        compare_models(
            models=["qwen3-32b", "deepseek-v3", "qwen3-14b"],
            eval_config=eval_config,
            num_tasks=num_tasks,
            vllm_url=args.vllm_url,
            output_dir=args.output_dir,
        )
    else:
        result = run_single_benchmark(
            model_key=args.model,
            eval_config=eval_config,
            num_tasks=num_tasks,
            vllm_url=args.vllm_url,
            output_dir=args.output_dir,
            green_port=args.green_port,
            purple_port=args.purple_port,
            timeout=args.timeout,
            start_only=args.start_only,
        )
        
        if result.get("success"):
            print(f"\n✅ 评测完成!")
            if "output_file" in result:
                print(f"   结果文件: {result['output_file']}")
        else:
            print(f"\n❌ 评测失败: {result.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
