#!/usr/bin/env python3
"""
TensorRT-LLM OpenAI-Compatible API Server

为 DeepSeek-V3.2-NVFP4 提供 OpenAI 兼容的 API 接口。
支持 /v1/chat/completions、/v1/models 等标准端点。

Usage:
    # 在 TRT-LLM 容器内运行
    python trtllm_openai_api.py --model nvidia/DeepSeek-V3.2-NVFP4 --tensor-parallel-size 8
    
    # 指定端口
    python trtllm_openai_api.py --model nvidia/DeepSeek-V3.2-NVFP4 --port 8001

API Endpoints:
    POST /v1/chat/completions  - Chat completions (OpenAI compatible)
    GET  /v1/models            - List available models
    GET  /health               - Health check

Streaming:
    设置 stream=True 支持流式输出

Requirements:
    - TensorRT-LLM >= 1.3.0
    - FastAPI
    - uvicorn
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("trtllm_api")

# 全局 LLM 实例
llm = None
model_name = "deepseek-v3.2-nvfp4"


# ============================================
# Pydantic Models (OpenAI Compatible)
# ============================================

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    n: int = Field(default=1, ge=1, le=1)  # 暂时只支持 n=1
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "nvidia"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================
# LLM Backend (TensorRT-LLM)
# ============================================

def init_trtllm(
    model: str,
    tensor_parallel_size: int = 8,
    trust_remote_code: bool = True,
):
    """初始化 TensorRT-LLM 模型"""
    global llm, model_name
    
    logger.info(f"Loading TRT-LLM model: {model}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    
    try:
        from tensorrt_llm import LLM
        
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )
        
        # 从模型路径提取模型名
        model_name = model.split("/")[-1].lower()
        
        logger.info(f"✅ Model loaded successfully: {model_name}")
        return True
        
    except ImportError:
        logger.error("TensorRT-LLM not installed. Please use the TRT-LLM container.")
        raise RuntimeError("TensorRT-LLM not available")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def format_messages_to_prompt(messages: List[Message]) -> str:
    """将 OpenAI 格式的消息转换为 prompt"""
    
    # DeepSeek 的对话格式
    prompt_parts = []
    
    for msg in messages:
        role = msg.role.lower()
        content = msg.content
        
        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{content}\n")
        elif role == "tool":
            prompt_parts.append(f"<|tool|>\n{content}\n")
    
    # 添加 assistant 前缀
    prompt_parts.append("<|assistant|>\n")
    
    return "".join(prompt_parts)


def generate_completion(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    stop: Optional[List[str]] = None,
) -> str:
    """使用 TRT-LLM 生成文本"""
    global llm
    
    if llm is None:
        raise RuntimeError("Model not initialized")
    
    try:
        from tensorrt_llm import SamplingParams
        
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        if stop:
            params.stop = stop
        
        outputs = llm.generate([prompt], params)
        return outputs[0].outputs[0].text
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise


async def generate_completion_stream(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    stop: Optional[List[str]] = None,
) -> AsyncGenerator[str, None]:
    """流式生成文本"""
    global llm
    
    if llm is None:
        raise RuntimeError("Model not initialized")
    
    try:
        from tensorrt_llm import SamplingParams
        
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            streaming=True,
        )
        
        if stop:
            params.stop = stop
        
        # TRT-LLM streaming generation
        stream = llm.generate([prompt], params, streaming=True)
        
        async for output in stream:
            if output.outputs:
                token = output.outputs[0].text
                yield token
                
    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        raise


# ============================================
# FastAPI Application
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info("TRT-LLM OpenAI API starting...")
    yield
    # Shutdown
    logger.info("TRT-LLM OpenAI API shutting down...")


app = FastAPI(
    title="TRT-LLM OpenAI Compatible API",
    description="OpenAI-compatible API for DeepSeek-V3.2-NVFP4 via TensorRT-LLM",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model": model_name,
        "backend": "tensorrt-llm",
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=model_name,
                created=int(time.time()),
                owned_by="nvidia",
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions 端点 (OpenAI 兼容)"""
    
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    
    # 转换消息格式
    prompt = format_messages_to_prompt(request.messages)
    
    # 处理 stop 参数
    stop = request.stop
    if isinstance(stop, str):
        stop = [stop]
    
    logger.info(f"Request {request_id}: {len(request.messages)} messages, max_tokens={request.max_tokens}")
    
    try:
        if request.stream:
            # 流式响应
            async def stream_response():
                async for token in generate_completion_stream(
                    prompt=prompt,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stop=stop,
                ):
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": token},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                
                # 发送结束标记
                final_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={},
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )
        
        else:
            # 非流式响应
            start_time = time.time()
            
            text = await asyncio.to_thread(
                generate_completion,
                prompt=prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=stop,
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Request {request_id} completed in {elapsed:.2f}s, {len(text)} chars")
            
            # 简单的 token 计数 (近似)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(text) // 4
            
            return ChatCompletionResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=text),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions 端点 (简化支持)"""
    data = await request.json()
    
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 1024)
    temperature = data.get("temperature", 0.7)
    
    text = await asyncio.to_thread(
        generate_completion,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "text": text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(text) // 4,
            "total_tokens": (len(prompt) + len(text)) // 4,
        },
    }


# ============================================
# Main Entry Point
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="TRT-LLM OpenAI Compatible API Server"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="nvidia/DeepSeek-V3.2-NVFP4",
        help="HuggingFace 模型路径",
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=8,
        help="Tensor parallel size (GPU 数量)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="API 服务端口",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API 服务地址",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Uvicorn workers (建议保持为 1)",
    )
    
    args = parser.parse_args()
    
    # 初始化模型
    init_trtllm(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # 启动 FastAPI
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
