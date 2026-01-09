"""
A2A Server for Purple Agent

Implements the A2A protocol server using FastAPI to handle
incoming requests from Green Agents (evaluators).
"""

import os
import asyncio
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from a2a.types import AgentCard

from purple_agent.card import get_agent_card
from purple_agent.executor import FinanceAgentExecutor


def create_app(
    host: str = "localhost",
    port: int = 8101,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    model: str | None = None,
    simulation_date: datetime | None = None,
) -> FastAPI:
    """
    Create the FastAPI application with A2A protocol support.

    Args:
        host: Hostname for the server
        port: Port number
        openai_api_key: OpenAI API key for LLM calls
        anthropic_api_key: Anthropic API key for LLM calls
        model: Model identifier
        simulation_date: Optional date for temporal locking

    Returns:
        FastAPI application instance
    """
    # Initialize LLM client
    llm_client = None
    default_model = model

    if openai_api_key:
        try:
            from openai import OpenAI
            import os
            base_url = os.environ.get("OPENAI_API_BASE")
            llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=base_url  # Supports local vLLM
            )
            default_model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        except ImportError:
            pass
    elif anthropic_api_key:
        try:
            from anthropic import Anthropic
            llm_client = Anthropic(api_key=anthropic_api_key)
            default_model = model or "claude-sonnet-4-20250514"
        except ImportError:
            pass

    # Create agent components
    agent_card = get_agent_card(host, port)
    executor = FinanceAgentExecutor(
        llm_client=llm_client,
        model=default_model or "gpt-4o",
        simulation_date=simulation_date,
    )

    # Create A2A infrastructure
    task_store = InMemoryTaskStore()
    queue_manager = InMemoryQueueManager()

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        queue_manager=queue_manager,
    )

    # Create FastAPI app
    app = FastAPI(
        title="Purple Finance Agent",
        description="A2A-compliant Finance Analysis Agent for AgentBeats",
        version="1.0.0",
    )

    # Create A2A FastAPI application
    a2a_app = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Mount A2A routes
    a2a_app.add_routes_to_app(app)

    # Add custom health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "agent": agent_card.name,
            "version": agent_card.version,
        }

    # Add agent card endpoint (standard A2A discovery)
    @app.get("/.well-known/agent.json")
    async def get_agent_card_endpoint():
        """Return the Agent Card for discovery."""
        return agent_card.model_dump(exclude_none=True)

    # Add direct analysis endpoint (non-A2A convenience)
    @app.post("/analyze")
    async def analyze_direct(request: Request):
        """
        Direct analysis endpoint (non-A2A).

        This provides a simpler interface for testing without
        the full A2A protocol overhead.
        """
        try:
            body = await request.json()
            question = body.get("question", "")
            ticker = body.get("ticker")

            if not question:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Question is required"},
                )

            # Parse and analyze
            task_info = executor._parse_task(question)
            if ticker and ticker not in task_info["tickers"]:
                task_info["tickers"] = [ticker] + task_info["tickers"]

            financial_data = await executor._gather_data(task_info)
            analysis = await executor._generate_analysis(
                user_input=question,
                task_info=task_info,
                financial_data=financial_data,
            )

            return {
                "analysis": analysis,
                "task_info": task_info,
                "tickers_analyzed": list(financial_data.get("tickers", {}).keys()),
            }

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8101,
    reload: bool = False,
):
    """
    Run the A2A server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn

    # Get configuration from environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("LLM_MODEL")
    sim_date_str = os.environ.get("SIMULATION_DATE")

    simulation_date = None
    if sim_date_str:
        try:
            simulation_date = datetime.fromisoformat(sim_date_str)
        except ValueError:
            pass

    app = create_app(
        host=host,
        port=port,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        model=model,
        simulation_date=simulation_date,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
