"""
A2A Server for Green Agent (CIO-Agent FAB++ Evaluator)

This is the main entry point for the Green Agent A2A server.
It exposes the FAB++ evaluation capabilities via the A2A protocol,
allowing the AgentBeats platform to run assessments.

Usage:
    uv run src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9009
    
    # Or with Docker:
    docker run -p 9009:9009 ghcr.io/your-org/cio-agent-green:latest --host 0.0.0.0

The server accepts:
    --host: Host address to bind to (default: 127.0.0.1)
    --port: Port to listen on (default: 9009)
    --card-url: URL to advertise in the agent card (optional)
"""

import argparse
import os
import uvicorn

from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from cio_agent.green_executor import GreenAgentExecutor

# Load environment variables from .env file
load_dotenv()


def main():
    """Main entry point for the Green Agent A2A server."""
    parser = argparse.ArgumentParser(description="Run the CIO-Agent Green Agent A2A server.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9009,
        help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url",
        type=str,
        help="URL to advertise in the agent card"
    )
    args = parser.parse_args()

    # Define agent skills
    skill = AgentSkill(
        id="fab-plus-plus-evaluation",
        name="FAB++ Finance Agent Benchmark",
        description=(
            "Evaluates finance agents using the FAB++ (Finance Agent Benchmark Plus Plus) "
            "framework. Assesses agents on macro analysis, fundamental accuracy, execution "
            "quality, and adversarial robustness. Provides comprehensive Alpha Scores."
        ),
        tags=[
            "finance",
            "evaluation",
            "benchmark",
            "agent-assessment",
            "adversarial-debate",
        ],
        examples=[
            "Evaluate a finance agent's ability to analyze NVIDIA Q3 earnings",
            "Test agent robustness with adversarial counter-arguments",
            "Assess fundamental data accuracy and macro thesis quality",
        ]
    )

    # Determine the advertised URL
    # Note: 0.0.0.0 is a bind address, not connectable. Use 127.0.0.1 for local testing.
    advertised_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    card_url = args.card_url or f"http://{advertised_host}:{args.port}/"

    # Create agent card
    agent_card = AgentCard(
        name="CIO-Agent FAB++ Evaluator",
        description=(
            "A Green Agent for the AgentBeats platform that evaluates finance agents "
            "using the FAB++ (Finance Agent Benchmark Plus Plus) methodology. "
            "Tests agents on earnings analysis, SEC filing interpretation, numerical "
            "reasoning, and investment recommendations with adversarial robustness testing."
        ),
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    # Create request handler with executor
    request_handler = DefaultRequestHandler(
        agent_executor=GreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A application
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"Starting CIO-Agent Green Agent A2A server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Agent Card URL: {agent_card.url}")
    print(f"  Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")
    
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
