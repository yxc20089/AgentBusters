"""
Baseline Purple Agent HTTP service.

Implements a simple A2A-compatible endpoint that can respond to
`task_assignment` messages with `task_response` messages and to
`challenge` messages with `rebuttal` messages using heuristic logic.
"""

import asyncio
from datetime import datetime
from typing import Optional

from aiohttp import web
import structlog

from cio_agent.models import (
    A2AMessage,
    A2AMessageType,
    AgentResponse,
    DebateRebuttal,
    GroundTruth,
    FinancialData,
    Task,
    TaskCategory,
    TaskRubric,
    TaskDifficulty,
)
from cio_agent.orchestrator import MockAgentClient

logger = structlog.get_logger()


def _parse_task_from_message(message: A2AMessage) -> Task:
    """Create a Task stub from an incoming task_assignment message payload."""
    payload = message.payload
    simulation_date = datetime.fromisoformat(payload.get("simulation_date"))

    # Fallbacks keep the baseline agent robust even if optional fields are missing.
    ticker = payload.get("ticker", "UNKNOWN")
    fiscal_year = int(payload.get("fiscal_year", simulation_date.year))

    return Task(
        question_id=payload["task_id"],
        category=TaskCategory(payload["category"]),
        question=payload["question"],
        ticker=ticker,
        fiscal_year=fiscal_year,
        simulation_date=simulation_date,
        ground_truth=GroundTruth(
            macro_thesis="Baseline purple agent ground truth placeholder",
            financials=FinancialData(),
        ),
        difficulty=TaskDifficulty(payload.get("difficulty", TaskDifficulty.MEDIUM.value)),
        rubric=TaskRubric(),
        available_tools=payload.get("available_tools", []),
        deadline_seconds=int(payload.get("deadline_seconds", 1800)),
        requires_code_execution=payload.get("requires_code_execution", False),
    )


class BaselinePurpleAgent:
    """Minimal A2A-compatible purple agent using MockAgentClient heuristics."""

    def __init__(
        self,
        agent_id: str = "purple-baseline",
        model: str = "gpt-4o-mini",
    ):
        self.agent_id = agent_id
        self.client = MockAgentClient(agent_id=agent_id, model=model)
        self._latest_responses: dict[str, AgentResponse] = {}

    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Route incoming A2A messages to the appropriate handler."""
        if message.message_type == A2AMessageType.TASK_ASSIGNMENT:
            return await self._handle_task_assignment(message)

        if message.message_type == A2AMessageType.CHALLENGE:
            return await self._handle_challenge(message)

        logger.warning("unsupported_message_type", type=message.message_type.value)
        return None

    async def _handle_task_assignment(self, message: A2AMessage) -> A2AMessage:
        """Generate a task response to a task_assignment message."""
        task = _parse_task_from_message(message)
        response = await self.client.process_task(task)
        self._latest_responses[task.question_id] = response

        return A2AMessage.task_response(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            response=response,
        )

    async def _handle_challenge(self, message: A2AMessage) -> A2AMessage:
        """Generate a rebuttal to a challenge message."""
        payload = message.payload
        task_id = payload["task_id"]
        counter_argument = payload.get("challenge", "")
        original_response = self._latest_responses.get(
            task_id,
            AgentResponse(
                agent_id=self.agent_id,
                task_id=task_id,
                analysis="Placeholder analysis",
                recommendation="HOLD",
            ),
        )

        rebuttal: DebateRebuttal = await self.client.process_challenge(
            task_id=task_id,
            challenge=counter_argument,
            original_response=original_response,
        )

        return A2AMessage.rebuttal(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            rebuttal=rebuttal,
        )


async def create_app(agent_id: str, model: str) -> web.Application:
    """Create the aiohttp application hosting the baseline purple agent."""
    agent = BaselinePurpleAgent(agent_id=agent_id, model=model)
    routes = web.RouteTableDef()

    @routes.get("/health")
    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "agent_id": agent_id})

    @routes.post("/a2a")
    async def a2a_handler(request: web.Request) -> web.Response:
        payload = await request.json()
        message = A2AMessage(**payload)
        logger.info(
            "a2a_received",
            type=message.message_type.value,
            sender=message.sender_id,
            receiver=message.receiver_id,
            task_id=message.payload.get("task_id"),
        )
        reply = await agent.handle_message(message)

        if reply is None:
            return web.json_response({"error": "unsupported_message"}, status=400)

        # Use json mode to ensure datetimes are serialized as ISO strings.
        response = web.json_response(reply.model_dump(mode="json"))
        logger.info(
            "a2a_replied",
            type=reply.message_type.value,
            sender=reply.sender_id,
            receiver=reply.receiver_id,
            task_id=reply.payload.get("task_id"),
        )
        return response

    app = web.Application()
    app.add_routes(routes)
    return app


async def run_server(host: str = "0.0.0.0", port: int = 8090, agent_id: str = "purple-baseline", model: str = "gpt-4o-mini") -> None:
    """Run the aiohttp web server."""
    app = await create_app(agent_id=agent_id, model=model)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    logger.info("purple_agent_starting", host=host, port=port, agent_id=agent_id, model=model)
    await site.start()

    # Keep running forever.
    while True:
        await asyncio.sleep(3600)
