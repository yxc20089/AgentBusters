"""
A2A HTTP client for communicating with external agents (e.g., purple baseline).

Provides an interface compatible with MockAgentClient: process_task and
process_challenge, using the A2A message schema over HTTP.
"""

from typing import Optional

import httpx
import structlog

from cio_agent.models import (
    A2AMessage,
    AgentResponse,
    DebateRebuttal,
    Task,
)

logger = structlog.get_logger()


class A2AHTTPAgentClient:
    """HTTP client that sends A2A messages to a remote agent endpoint."""

    def __init__(
        self,
        endpoint: str,
        agent_id: str = "purple-baseline",
        sender_id: str = "cio-agent-green",
        model: str = "purple-a2a",
        timeout_seconds: int = 60,
    ):
        self.endpoint = endpoint
        self.agent_id = agent_id
        self.sender_id = sender_id
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def _send(self, message: A2AMessage) -> A2AMessage:
        """Send an A2A message over HTTP and parse the reply."""
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.endpoint,
                json=message.model_dump(mode="json"),
            )
            response.raise_for_status()
            data = response.json()
            return A2AMessage(**data)

    async def process_task(self, task: Task) -> AgentResponse:
        """Send a task_assignment and return the agent's task_response."""
        message = A2AMessage.task_assignment(
            sender_id=self.sender_id,
            receiver_id=self.agent_id,
            task=task,
        )
        reply = await self._send(message)
        if reply.payload is None:
            raise ValueError("Empty payload in task_response")
        return AgentResponse.model_validate(reply.payload)

    async def process_challenge(
        self,
        task_id: str,
        challenge: str,
        original_response: Optional[AgentResponse] = None,
    ) -> DebateRebuttal:
        """Send a challenge and return the agent's rebuttal."""
        message = A2AMessage.challenge(
            sender_id=self.sender_id,
            receiver_id=self.agent_id,
            task_id=task_id,
            counter_argument=challenge,
        )
        reply = await self._send(message)
        if reply.payload is None:
            raise ValueError("Empty payload in rebuttal response")
        return DebateRebuttal.model_validate(reply.payload)
