"""
TRT-LLM OpenAI Compatible API

提供 TensorRT-LLM 的 OpenAI 兼容 API 封装。
"""

from .trtllm_openai_api import app, init_trtllm

__all__ = ["app", "init_trtllm"]
