import asyncio
import json
from datetime import datetime

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cio_agent.models import (
    A2AMessage,
    AgentResponse,
    DebateRebuttal,
    FinancialData,
    GroundTruth,
    Task,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)
from purple_agent.service import create_app


def _sample_task() -> Task:
    return Task(
        question_id="demo_task_001",
        category=TaskCategory.BEAT_OR_MISS,
        question="Did NVIDIA beat analyst EPS estimates in Q3 FY2026?",
        ticker="NVDA",
        fiscal_year=2026,
        simulation_date=datetime(2025, 11, 20),
        ground_truth=GroundTruth(
            macro_thesis="placeholder",
            financials=FinancialData(),
        ),
        difficulty=TaskDifficulty.MEDIUM,
        rubric=TaskRubric(),
    )


def test_a2a_message_task_response_serializes_without_datetime_error():
    """Ensure task_response is JSON-serializable (datetimes converted)."""
    agent_response = AgentResponse(
        agent_id="purple-baseline",
        task_id="demo_task_001",
        analysis="analysis",
        recommendation="hold",
        extracted_financials=FinancialData(),
    )
    msg = A2AMessage.task_response(
        sender_id="purple-baseline",
        receiver_id="cio-agent-green",
        response=agent_response,
    )

    # Should not raise when dumping to JSON
    dumped = json.dumps(msg.model_dump(mode="json"))
    assert "demo_task_001" in dumped
    assert msg.message_type.value == "task_response"


@pytest.mark.asyncio
async def test_purple_service_handles_assignment_and_challenge():
    """Purple service should accept task_assignment then challenge and reply."""
    app = await create_app(agent_id="purple-baseline", model="gpt-4o-mini")
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        task = _sample_task()
        assignment = A2AMessage.task_assignment(
            sender_id="cio-agent-green",
            receiver_id="purple-baseline",
            task=task,
        )
        resp = await client.post("/a2a", json=assignment.model_dump(mode="json"))
        assert resp.status == 200
        assignment_reply = await resp.json()
        assert assignment_reply["message_type"] == "task_response"
        assert assignment_reply["payload"]["task_id"] == task.question_id

        challenge = A2AMessage.challenge(
            sender_id="cio-agent-green",
            receiver_id="purple-baseline",
            task_id=task.question_id,
            counter_argument="Valuation looks stretched; justify your BUY.",
        )
        resp2 = await client.post("/a2a", json=challenge.model_dump(mode="json"))
        assert resp2.status == 200
        challenge_reply = await resp2.json()
        assert challenge_reply["message_type"] == "rebuttal"
        assert challenge_reply["payload"]["task_id"] == task.question_id
    finally:
        await client.close()
